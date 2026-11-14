"""Re-score held-out mice of a finished multisubject GRU with an explicit fit/score table.

Wrapper #126 (studies #148, study 17). This extends the re-score-only path of
``resume_heldout_beaker.py``: the same source run, the same held-out fine-tune
(``heldout_finetuning.run_heldout_subject_finetuning_from_config``, only the new subject
embeddings are tuned) and the same fine-tune settings, read from the source run's
``auto_heldout_finetune`` block. Three things differ:

* the held-out mice are an explicit list, not the source run's reserved set;
* an explicit per-session role table (``subject_id, session_id, role`` with role ``fit`` or
  ``score``) replaces the ``eval_every_n`` alternation. Pass A tunes the embedding on the
  ``fit`` sessions and scores the ``score`` sessions; ``--swap`` (pass B) inverts them.
  Sessions the loader has but the table lacks, and the reverse, are counted and listed in
  the manifest and are never used;
* W&B logging, if asked for, goes to a NEW run, never into the source run.

Outputs in ``--output-dir``: ``trial_predictions.csv.gz`` (p_left, p_right, p_ignore per
trial; p_ignore is NaN for a 2-way model), ``session_scores.csv`` and ``mouse_scores.csv``
(summed log-likelihoods and trial counts, left/right on engaged trials and full 3-way),
``subject_embeddings.csv`` (the fine-tuned embedding per mouse) and ``manifest.json``.

``--session-lambda`` sets the session_curriculum_lambda used in BOTH the fine-tune and the
scoring forward pass: ``checkpoint`` (default) uses the selected checkpoint's own lambda, so
a checkpoint chosen at lambda = 0 is scored with the session delta off, as it was selected;
``1`` is the auto held-out fine-tune's behaviour (studies 01/02). The manifest and the
per-trial table record the checkpoint's lambda and the lambda used.

Run from the wrapper's ``code/``::

    python -m post_training_analysis.heldout_rescore \\
        --run-id <wandb_run_id> --entity AIND-disRNN --project mice_ignore_scaling \\
        --subjects heldout_mice.txt --roles sessions.csv [--swap] \\
        --output-dir /results/rescore
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROLES = ("fit", "score")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_subject_list(spec: str) -> list[str]:
    """Mouse ids from a file (one per line, ``#`` comments) or a comma-separated list."""
    path = Path(spec).expanduser()
    if path.is_file():
        tokens = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    else:
        tokens = [token.strip() for token in spec.split(",")]
    subjects = [token for token in tokens if token]
    if not subjects:
        raise ValueError(f"No subject ids in {spec!r}.")
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Duplicate subject ids in {spec!r}.")
    return subjects


def read_role_table(path: str | Path) -> pd.DataFrame:
    """Read ``subject_id, session_id, role`` from a CSV or parquet file."""
    path = Path(path).expanduser()
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=str)
    missing = {"subject_id", "session_id", "role"} - set(table.columns)
    if missing:
        raise ValueError(f"Role table {path} lacks columns {sorted(missing)}.")
    table = table[["subject_id", "session_id", "role"]].astype(str)
    bad_roles = sorted(set(table["role"]) - set(ROLES))
    if bad_roles:
        raise ValueError(f"Role table {path} has roles {bad_roles}; allowed: {list(ROLES)}.")
    duplicated = table.loc[table["session_id"].duplicated(), "session_id"].tolist()
    if duplicated:
        raise ValueError(f"Role table {path} lists sessions more than once: {duplicated[:10]}.")
    return table


def reconcile_session_roles(
    loader_sessions: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    subjects: Sequence[str],
    swap: bool,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    """Map each mouse's sessions to fine-tune (adapt) and scored (test) lists.

    ``loader_sessions`` holds the ``subject_id, session_id`` pairs the data loader keeps.
    Only sessions present in both the loader and the role table are used; the others are
    reported. Pass A tunes on ``fit`` and scores ``score``; ``swap`` inverts that.
    """
    subjects = [str(subject) for subject in subjects]
    roles = roles[roles["subject_id"].isin(subjects)]
    loader = loader_sessions[["subject_id", "session_id"]].astype(str).drop_duplicates()
    merged = loader.merge(roles, on=["subject_id", "session_id"], how="outer", indicator=True)
    loader_only = merged[merged["_merge"] == "left_only"]
    table_only = merged[merged["_merge"] == "right_only"]
    used = merged[merged["_merge"] == "both"]

    tune_role, score_role = ("score", "fit") if swap else ("fit", "score")
    split: dict[str, dict[str, list[str]]] = {}
    incomplete: list[str] = []
    for subject in subjects:
        rows = used[used["subject_id"] == subject]
        adapt = sorted(rows.loc[rows["role"] == tune_role, "session_id"])
        test = sorted(rows.loc[rows["role"] == score_role, "session_id"])
        if not adapt or not test:
            incomplete.append(subject)
            continue
        split[subject] = {"adapt_session_ids": adapt, "test_session_ids": test}
    if incomplete:
        raise ValueError(
            "These mice lack a fine-tune or a scored session (or any session) after "
            f"reconciling the role table with the loader: {incomplete}."
        )

    def _listing(frame: pd.DataFrame) -> dict[str, Any]:
        return {"count": int(len(frame)), "session_ids": sorted(frame["session_id"].tolist())}

    report = {
        "n_subjects_requested": len(subjects),
        "n_subjects_scored": len(split),
        "subjects_without_loader_sessions": sorted(set(subjects) - set(loader["subject_id"])),
        "subjects_without_role_rows": sorted(set(subjects) - set(roles["subject_id"])),
        "sessions_in_loader_not_in_table": _listing(loader_only),
        "sessions_in_table_not_in_loader": _listing(table_only),
        "n_sessions_used": int(len(used)),
        "n_sessions_tuned": int(sum(len(v["adapt_session_ids"]) for v in split.values())),
        "n_sessions_scored": int(sum(len(v["test_session_ids"]) for v in split.values())),
    }
    for key in ("sessions_in_loader_not_in_table", "sessions_in_table_not_in_loader"):
        if report[key]["count"]:
            logger.warning("%s: %d session(s), not used.", key, report[key]["count"])
    return split, report


def build_rescore_bundle(
    *,
    source_run: Any,
    heldout_selector: Mapping[str, Any],
    fine_tune_cfg: Mapping[str, Any],
    roles: pd.DataFrame,
    swap: bool,
) -> tuple[Any, dict[str, Any]]:
    """Load the listed mice like the held-out fine-tune does, then apply the role table."""
    from data_loaders import mice as mice_loader
    from post_training_analysis import heldout_finetuning as hf
    from utils.multisubject import load_subject_index_map

    # Only mice new to the source run get a tuned embedding; a training mouse would keep its
    # frozen embedding and be "scored" on data it was trained on.
    source_subjects, _ = load_subject_index_map(source_run.subject_index_map_path)
    seen = sorted({str(s) for s in heldout_selector.get("test_subject_ids") or []} & {str(s) for s in source_subjects})
    if seen:
        raise ValueError(f"These mice are in the source run's training set, not held out: {seen}.")

    source_data_cfg = dict(source_run.run_config.get("data") or {})
    ignore_policy = str(source_data_cfg.get("ignore_policy", "exclude"))
    heldout_df, subject_ids = hf._load_heldout_snapshot_selection(heldout_selector=heldout_selector)
    heldout_df["subject_id"] = heldout_df["subject_id"].astype(str)
    heldout_df, timing_feature_map = hf._attach_heldout_timing_features(
        heldout_df, heldout_selector=heldout_selector, source_data_cfg=source_data_cfg
    )
    if timing_feature_map is not None:
        source_data_cfg = {**source_data_cfg, "features": timing_feature_map}

    kept = mice_loader._align_raw_df_with_valid_sessions(heldout_df, ignore_policy=ignore_policy)
    loader_sessions = kept[["subject_id", "ses_idx"]].rename(columns={"ses_idx": "session_id"})
    split, report = reconcile_session_roles(
        loader_sessions, roles, subjects=[str(s) for s in subject_ids], swap=swap
    )
    # Every loader session stays in the packed sequence, so each session keeps the
    # session index (and per-mouse maximum) it has in training; sessions without a role
    # are only left out of the fine-tune and the scoring.
    bundle = hf._build_local_heldout_bundle(
        heldout_df=heldout_df,
        heldout_subject_ids=[str(s) for s in subject_ids if str(s) in split],
        source_data_cfg=source_data_cfg,
        fine_tune_cfg=fine_tune_cfg,
        session_split_by_subject=split,
        allow_unassigned_sessions=True,
    )
    bundle.metadata["dataset_id"] = f"rescore_pass_{'B' if swap else 'A'}"
    return bundle, report


def summarize_scores(table: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Summed log-likelihoods and trial counts per group."""
    grouped = table.groupby(list(keys), sort=True)
    return grouped.agg(
        ll_lr_sum=("ll_lr", "sum"),
        n_engaged=("ll_lr", "count"),
        ll_3way_sum=("ll_3way", lambda v: v.sum(min_count=1)),
        n_trials_3way=("ll_3way", "count"),
    ).reset_index()


def checkpoint_lambda(model_dir: str | Path, step: int | None, *, final: bool = False) -> float | None:
    """The session_curriculum_lambda recorded for one checkpoint, if any.

    Looks up the step in ``checkpoints/index.json``, then in ``output_summary.json``'s
    checkpoint list; for the final params it falls back to the summary's
    ``session_curriculum.final_lambda``.
    """
    outputs = Path(model_dir) / "outputs"
    index_path = outputs / "checkpoints" / "index.json"
    summary_path = outputs / "output_summary.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if step is not None:
        for record in [*(index.get("checkpoints") or []), *(summary.get("checkpoints") or [])]:
            if record.get("step") is not None and int(record["step"]) == int(step):
                if record.get("session_curriculum_lambda") is not None:
                    return float(record["session_curriculum_lambda"])
    final_lambda = (summary.get("session_curriculum") or {}).get("final_lambda")
    return float(final_lambda) if final and final_lambda is not None else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="Finished W&B run whose gru-output artifact to score.")
    source.add_argument("--model-dir", help="Local run dir (inputs.yaml + outputs/).")
    parser.add_argument("--entity", default="AIND-disRNN", help="Entity of --run-id.")
    parser.add_argument("--project", default="mice_ignore_scaling", help="Project of --run-id.")
    parser.add_argument("--subjects", required=True, help="Held-out mice: file or comma list.")
    parser.add_argument("--roles", required=True, help="subject_id,session_id,role CSV/parquet.")
    parser.add_argument("--swap", action="store_true", help="Pass B: tune on score, score fit.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="best_eval (the source run's default), final, or step_<N>.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Default: the source run's.")
    parser.add_argument("--n-steps", type=int, default=None, help="Default: the source run's.")
    parser.add_argument("--lr", type=float, default=None, help="Default: the source run's.")
    parser.add_argument(
        "--session-lambda",
        choices=("checkpoint", "1"),
        default="checkpoint",
        help="session_curriculum_lambda for the fine-tune and scoring: the selected "
        "checkpoint's own (default) or 1 (the auto held-out fine-tune, studies 01/02).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default=None, help="Log to a NEW run here.")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )
    import yaml

    from post_training_analysis import heldout_finetuning as hf
    from post_training_analysis.generative_analysis import resolve_model_run
    from resume_heldout_beaker import _build_finetune_config
    from utils.multisubject import extract_subject_embeddings_from_params, load_subject_index_map
    from utils.run_helpers import _code_versions

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_label = "B" if args.swap else "A"

    if args.run_id:
        from post_training_analysis.wandb_model_dir import hydrate_model_dir

        model_dir = hydrate_model_dir(
            args.run_id,
            project=args.project,
            entity=args.entity,
            # One directory per source run, so a reused --output-dir never mixes checkpoints.
            dest=output_dir / "source" / f"{args.entity}__{args.project}__{args.run_id}",
        )
    else:
        model_dir = Path(args.model_dir).expanduser().resolve()
    source_cfg = yaml.safe_load((Path(model_dir) / "inputs.yaml").read_text())

    subjects = read_subject_list(args.subjects)
    roles_path = Path(args.roles).expanduser()
    roles = read_role_table(roles_path)

    config = _build_finetune_config(source_cfg, Path(model_dir))
    if args.checkpoint is not None:
        config["source_run"]["checkpoint_policy"] = args.checkpoint
    if args.seed is not None:
        config["seed"] = args.seed
    if args.n_steps is not None:
        config["heldout_finetuning"]["n_steps"] = args.n_steps
    if args.lr is not None:
        config["heldout_finetuning"]["lr"] = args.lr
    config["heldout_subjects"]["test_subject_ids"] = list(subjects)
    config["heldout_finetuning"]["export_trial_predictions"] = True
    config["output"]["output_root"] = str(output_dir / "finetune")

    source_run = resolve_model_run(
        model_dir, split="train", checkpoint_policy=config["source_run"]["checkpoint_policy"]
    )
    if source_run.model_type != "gru":
        raise ValueError("heldout_rescore supports GRU source runs only.")
    selected_lambda = checkpoint_lambda(
        model_dir, source_run.checkpoint_step, final=source_run.checkpoint_label == "final"
    )
    if args.session_lambda == "checkpoint":
        if selected_lambda is None:
            raise ValueError(
                f"--session-lambda checkpoint: no session_curriculum_lambda recorded for "
                f"checkpoint {source_run.checkpoint_label}; pass --session-lambda 1 explicitly."
            )
        session_lambda = selected_lambda
    else:
        session_lambda = 1.0
    config["heldout_finetuning"]["session_curriculum_lambda"] = session_lambda
    resolved = hf._resolve_runtime_config(config, source_run=source_run, output_root_override=None)
    bundle, reconciliation = build_rescore_bundle(
        source_run=source_run,
        heldout_selector=resolved["heldout_subjects"],
        fine_tune_cfg=resolved["heldout_finetuning"],
        roles=roles,
        swap=args.swap,
    )

    architecture = dict(source_run.run_config["model"].get("architecture") or {})
    source_data_cfg = dict(source_run.run_config.get("data") or {})
    checkpoint = {
        "policy": source_run.checkpoint_policy,
        "step": source_run.checkpoint_step,
        "label": source_run.checkpoint_label,
        "params_path": source_run.params_path,
        "params_sha256": _sha256_bytes(Path(source_run.params_path).read_bytes()),
        "session_curriculum_lambda": selected_lambda,
        "session_conditioning": str(architecture.get("session_encoding_type", "none")),
        # The lambda actually used in the fine-tune and the scoring forward pass.
        "session_lambda_mode": args.session_lambda,
        "finetune_session_curriculum_lambda": session_lambda,
    }
    manifest: dict[str, Any] = {
        "tool": "post_training_analysis.heldout_rescore",
        "wrapper_commit": _code_versions().get("wrapper_commit"),
        "source": {
            "run_id": args.run_id,
            "entity": args.entity if args.run_id else None,
            "project": args.project if args.run_id else None,
            "model_dir": str(model_dir),
            "snapshot": source_data_cfg.get("snapshot"),
            "ignore_policy": source_data_cfg.get("ignore_policy"),
        },
        "checkpoint": checkpoint,
        "pass": pass_label,
        "swap": bool(args.swap),
        "subjects": {
            "spec": args.subjects,
            "n": len(subjects),
            "sha256": _sha256_bytes("\n".join(sorted(subjects)).encode()),
        },
        "roles": {
            "path": str(roles_path),
            "sha256": _sha256_bytes(roles_path.read_bytes()),
        },
        "finetune": {
            "seed": int(resolved["seed"]),
            "n_steps": int(resolved["heldout_finetuning"]["n_steps"]),
            "lr": float(resolved["heldout_finetuning"]["lr"]),
            "batch_mode": resolved["heldout_finetuning"]["batch_mode"],
        },
        "reconciliation": reconciliation,
    }
    logger.info(
        "Pass %s: %d mice, checkpoint %s (lambda=%s, scored at lambda=%s)",
        pass_label,
        reconciliation["n_subjects_scored"],
        checkpoint["label"],
        checkpoint["session_curriculum_lambda"],
        session_lambda,
    )

    wandb_run = None
    if args.wandb_project:
        from utils.run_helpers import _init_wandb_with_fallback

        wandb_run = _init_wandb_with_fallback(
            {
                "entity": args.wandb_entity,
                "project": args.wandb_project,
                "group": args.wandb_group,
                "name": args.wandb_name or f"rescore-{args.run_id or Path(model_dir).name}-{pass_label}",
                "job_type": "heldout_rescore",
                "config": manifest,
                "dir": str(output_dir),
            }
        )
    try:
        result = hf.run_heldout_subject_finetuning_from_config(
            config,
            target_bundle=bundle,
            wandb_run=wandb_run,
            wandb_key_prefix="heldout" if wandb_run is not None else None,
            wandb_step_offset=0 if wandb_run is not None else None,
        )

        role_by_session = roles.set_index("session_id")["role"]
        trials = pd.read_csv(result["trial_predictions_path"], dtype={"subject_id": str})
        trials = trials.rename(columns={"animal_response": "response"})
        trials["session_id"] = trials.pop("source_ses_idx").astype(str)
        trials = trials.drop(columns=["ses_idx"])
        # Sessions without a role are in the model's sequence (for their session index)
        # but neither tuned nor scored: leave them out of every output table.
        trials = trials[trials["split"] != "none"].copy()
        trials["role"] = trials["session_id"].map(role_by_session)
        trials["pass"] = pass_label
        trials["use"] = trials.pop("split").map({"train": "tuned", "eval": "scored"})
        trials["source_run_id"] = args.run_id
        trials["checkpoint_step"] = checkpoint["step"]
        trials["checkpoint_lambda"] = checkpoint["session_curriculum_lambda"]
        trials["session_lambda"] = session_lambda
        # ll_observed is the log-probability over the model's classes: 3-way only for a
        # 3-way model. ll_lr (engaged trials, renormalised over left/right) comes as is.
        three_way = bool(trials["p_ignore"].notna().all())
        trials["ll_3way"] = trials.pop("ll_observed") if three_way else np.nan
        columns = [
            "subject_id", "session_id", "trial", "response", "p_left", "p_right", "p_ignore",
            "role", "pass", "use", "ll_lr", "ll_3way",
            "source_run_id", "checkpoint_step", "checkpoint_lambda", "session_lambda",
        ]
        trials = trials[columns]
        session_scores = summarize_scores(trials, ["subject_id", "session_id", "role", "pass", "use"])
        mouse_scores = summarize_scores(trials, ["subject_id", "pass", "use"])

        params = hf._load_params(result["params_path"])
        embeddings = extract_subject_embeddings_from_params(params)
        subject_to_index, _ = load_subject_index_map(result["subject_index_map_path"])
        scored_subjects = sorted(trials["subject_id"].unique())
        embedding_table = pd.DataFrame(
            [embeddings[subject_to_index[subject]] for subject in scored_subjects],
            columns=[f"emb_{i}" for i in range(embeddings.shape[1])],
        )
        embedding_table.insert(0, "pass", pass_label)
        embedding_table.insert(0, "subject_id", scored_subjects)

        outputs = {
            "trial_predictions": output_dir / "trial_predictions.csv.gz",
            "session_scores": output_dir / "session_scores.csv",
            "mouse_scores": output_dir / "mouse_scores.csv",
            "subject_embeddings": output_dir / "subject_embeddings.csv",
        }
        trials.to_csv(outputs["trial_predictions"], index=False)
        session_scores.to_csv(outputs["session_scores"], index=False)
        mouse_scores.to_csv(outputs["mouse_scores"], index=False)
        embedding_table.to_csv(outputs["subject_embeddings"], index=False)
        manifest["finetune"]["output_dir"] = result["output_dir"]
        manifest["outputs"] = {key: str(path) for key, path in outputs.items()}
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

        if wandb_run is not None:
            import wandb

            wandb_run.log({"rescore/mouse_scores": wandb.Table(dataframe=mouse_scores)})
            artifact = wandb.Artifact(f"heldout-rescore-{wandb_run.id}", type="heldout-rescore")
            for path in [*outputs.values(), manifest_path]:
                artifact.add_file(str(path))
            wandb_run.log_artifact(artifact)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    logger.info("Held-out re-score done: %s", output_dir)
    return manifest


if __name__ == "__main__":
    main()
