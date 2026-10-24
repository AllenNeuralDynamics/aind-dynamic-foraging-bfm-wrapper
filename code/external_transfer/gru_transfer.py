"""Run one pinned source-GRU transfer cell against one external dataset.

Moved from aind-dynamic-foraging-bfm-studies `studies/09-gru-cross-species-transfer/run_gru_transfer.py` (dispatcher #179; identical to dispatcher `818cc6e` apart from reading no study
files at runtime). Run as ``python -m external_transfer.gru_transfer`` from the wrapper's ``code/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DATASETS = (
    "grossman",
    "chen",
    "zid",
    "lebedeva",
    "beron",
    "kwak",
    "miller",
    "findling",
    "alsio",
    "eckstein",
    "costa",
    "lopez_mouse",
    "hattori",
)
DATASET_IDS = {
    "grossman": "grossman-bari-cohen-2021",
    "chen": "chen-et-al-2021",
    "zid": "zid-et-al-2026-experiment-1",
    "lebedeva": "lebedeva-et-al-2026",
    "beron": "beron-et-al-2022",
    "kwak": "kwak-jung-2019-tab",
    "miller": "miller-et-al-2022-tab",
    "findling": "findling-et-al-volnoise",
    "alsio": "alsio-et-al-2019-vpvd-tsvr",
    "eckstein": "eckstein-et-al-2022",
    "costa": "costa-averbeck-2016-stochastic",
    "lopez_mouse": "lopez-yepez-et-al-2021-mouse",
    "hattori": "hattori-et-al-2023-imaging-mature",
}
PYARROW_VERSION = "21.0.0"
PYARROW_WHEEL_SHA256 = (
    "b7ae0bbdc8c6674259b25bef5d2a1d6af5d39d7200c819cf99e07f7dfef1c51e"
)


def _ensure_pyarrow(deps_root: Path = Path("/deps")) -> None:
    try:
        import pyarrow
    except ModuleNotFoundError:
        pass
    else:
        if pyarrow.__version__ != PYARROW_VERSION:
            raise RuntimeError(
                "Installed pyarrow version does not match the pinned runtime: "
                f"expected={PYARROW_VERSION} actual={pyarrow.__version__}"
            )
        return
    wheels = list(deps_root.glob(f"pyarrow-{PYARROW_VERSION}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected one pinned pyarrow wheel in {deps_root}; "
            f"found {len(wheels)}: {[wheel.name for wheel in wheels]}"
        )
    wheel = wheels[0]
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != PYARROW_WHEEL_SHA256:
        raise RuntimeError(
            f"Pinned pyarrow wheel digest mismatch: expected={PYARROW_WHEEL_SHA256} "
            f"actual={digest}"
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", str(wheel)],
        check=True,
    )


def _load_source_manifest(args: argparse.Namespace) -> tuple[bytes, str, dict]:
    """The source manifest's exact bytes, its study file name, and its parsed content.

    Inline text (``--source-manifest-json``) is how jobs receive it; the bytes are hashed
    as they are, so ``manifest_sha256`` matches the study's file when the sweep generator
    inlined that file verbatim.
    """
    if args.source_manifest_json is not None:
        manifest_bytes = args.source_manifest_json.encode()
        manifest_name = args.source_manifest_name
    else:
        manifest_bytes = args.source_manifest.read_bytes()
        manifest_name = args.source_manifest.name
    return manifest_bytes, manifest_name, json.loads(manifest_bytes)


def _budget(value: str) -> int:
    k = int(value)
    if k < 0:
        raise argparse.ArgumentTypeError("the adaptation budget must be >= 0")
    return k


def _condition(k: int | None) -> str:
    """``matched_half`` for the full adaptation half, else ``k<K>``."""
    return "matched_half" if k is None else f"k{k}"


def _run_meta() -> dict:
    """Launch provenance for the run config, from the env the launchers and SLURM scripts set."""
    return {
        "study": os.environ.get("BFM_META_STUDY"),
        "variant": os.environ.get("BFM_META_VARIANT"),
        "launch_id": os.environ.get("BFM_META_LAUNCH_ID"),
        # the studies repo commit that held the sweep (the launchers pass it since the split)
        "studies_commit": os.environ.get("BFM_META_STUDIES_COMMIT"),
        "dispatcher_commit": os.environ.get("DISPATCHER_COMMIT"),
        "wrapper_commit": os.environ.get("WRAPPER_COMMIT"),
        "foraging_models_commit": os.environ.get("FORAGING_MODELS_COMMIT"),
        "beaker_experiment_id": os.environ.get("BEAKER_EXPERIMENT_ID"),
        "beaker_job_id": os.environ.get("BEAKER_JOB_ID"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--source-key", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-manifest-json",
        help="the source manifest's exact file text, inlined by the study's sweep generator "
        "so the job reads no study files; its sha256 is recorded as manifest_sha256",
    )
    source.add_argument(
        "--source-manifest", type=Path, help="path to a source manifest file (local use)"
    )
    parser.add_argument(
        "--source-manifest-name",
        default="source_runs.json",
        help="the manifest's file name in the study, recorded as source.manifest",
    )
    parser.add_argument(
        "--adapt-sessions-per-subject",
        type=_budget,
        default=None,
        help="few-shot budget K: adapt each subject's embedding on its first K adaptation "
        "sessions (0 = zero-shot); omit for the full matched adaptation half",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/external"))
    parser.add_argument("--output-root", type=Path, default=Path("/results"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    _ensure_pyarrow()

    import wandb
    from post_training_analysis.heldout_finetuning import (
        run_heldout_subject_finetuning_from_config,
    )
    from post_training_analysis.wandb_model_dir import hydrate_model_dir

    manifest_bytes, manifest_name, source_manifest = _load_source_manifest(args)
    k = args.adapt_sessions_per_subject
    condition = _condition(k)
    cell = f"{args.dataset}-{args.source_key}" + ("" if k is None else f"-{condition}")
    try:
        source = source_manifest["runs"][args.source_key]
    except KeyError as exc:
        raise SystemExit(f"Unknown source key: {args.source_key}") from exc

    source_ref = (
        f"{source_manifest['entity']}/{source_manifest['project']}/"
        f"gru-output-{source['run_id']}:latest"
    )
    source_artifact = wandb.Api().artifact(source_ref, type="training-output")
    if source_artifact.digest != source["artifact_digest"]:
        raise RuntimeError(
            "Pinned source artifact digest mismatch: "
            f"expected={source['artifact_digest']} actual={source_artifact.digest}"
        )

    model_dir = hydrate_model_dir(
        source["run_id"],
        project=source_manifest["project"],
        entity=source_manifest["entity"],
        dest=args.output_root / "source-model",
        version=source_artifact.version,   # the version whose digest was just checked
    )
    audit = json.loads((args.data_root / f"{args.dataset}.audit.json").read_text())
    finetuning = source_manifest["embedding_finetuning"]
    run_config = {
        "source": {
            **source,
            "key": args.source_key,
            "manifest": manifest_name,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "wandb_project": source_manifest["project"],
            "wandb_group": source_manifest["group"],
            "checkpoint_policy": source_manifest["checkpoint_policy"],
        },
        "target": {
            "dataset": args.dataset,
            "audit": audit,
            "condition": condition,
            "adapt_sessions_per_subject": k,
        },
        "embedding_finetuning": finetuning,
        "meta": _run_meta(),
    }
    run = wandb.init(
        entity="AIND-disRNN",
        project="gru_cross_species_transfer",
        group=os.environ.get("WANDB_RUN_GROUP"),
        id=os.environ.get("WANDB_RUN_ID"),
        resume=os.environ.get("WANDB_RESUME", "allow"),
        name=f"gru-{cell}",
        dir=str(args.output_root / "wandb"),
        config=run_config,
        tags=["external-transfer", condition.replace("_", "-"), "gru"],
    )
    config = {
        "source_run": {
            "model_dir": str(model_dir),
            "checkpoint_policy": source_manifest["checkpoint_policy"],
        },
        "target_data": {
            "_target_": "data_loaders.external_bandit.ExternalBanditDatasetLoader",
            "file_path": str(args.data_root / f"{args.dataset}.parquet"),
            "split_manifest_path": str(args.data_root / f"{args.dataset}.split.json"),
            "dataset_id": DATASET_IDS[args.dataset],
            "batch_size": None,
            "batch_mode": "single",
            "adapt_sessions_per_subject": k,
            "seed": int(source["seed"]),
        },
        "heldout_finetuning": {
            "n_steps": int(finetuning["n_steps"]),
            "lr": float(finetuning["lr"]),
            "checkpoint_every_n_steps": 100,
            "batch_size": None,
            "batch_mode": "single",
            "checkpoint_plot_split_examples_every_n": 0,
            "checkpoint_save_output_df_every_n": 0,
            "train_example_sessions_per_subject": 0,
            "eval_example_sessions_per_subject": 0,
            "example_max_subjects": 0,
            "skip_subjects_with_insufficient_sessions": False,
            "keep_media_files": False,
            "selection_policy": finetuning["selection_policy"],
            "adapt_sessions_per_subject": k,
        },
        "output": {
            "output_root": str(args.output_root / "transfer"),
            "run_name_suffix": cell,
        },
        "seed": int(source["seed"]),
    }
    try:
        result = run_heldout_subject_finetuning_from_config(
            config,
            wandb_run=run,
            wandb_key_prefix="target",
        )
        metrics = json.loads(Path(result["test_metrics_path"]).read_text())
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                run.summary[f"target/test/{key}"] = value
        output_artifact = wandb.Artifact(
            f"external-gru-output-{run.id}",
            type="evaluation-output",
            metadata={
                "dataset": args.dataset,
                "source_key": args.source_key,
                "adapt_sessions_per_subject": k,
                "source_artifact_digest": source["artifact_digest"],
            },
        )
        output_artifact.add_dir(result["outputs_dir"])
        run.log_artifact(output_artifact)
        print(json.dumps(result, indent=2))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
