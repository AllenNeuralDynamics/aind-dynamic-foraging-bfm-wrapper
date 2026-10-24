"""Fit the matched subject-level Q-learning baseline for one external dataset.

Moved from aind-dynamic-foraging-bfm-studies `studies/09-gru-cross-species-transfer/run_q_baseline.py` (dispatcher #179; identical to dispatcher `818cc6e` apart from reading no study
files at runtime). Run as ``python -m external_transfer.q_baseline`` from the wrapper's ``code/``.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject-workers", type=int, default=6)
    parser.add_argument(
        "--subject-id",
        help="Optional one-subject execution used only for preflight smoke tests.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()

    import wandb
    from data_loaders.external_bandit import ExternalBanditDatasetLoader
    from model_trainers.baseline_rl_trainer import BaselineRLTrainer

    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.data_root / f"{args.dataset}.audit.json").read_text())
    config = {
        "model": {
            "agent_class": "ForagerQLearning",
            "number_of_learning_rate": 1,
            "number_of_forget_rate": 1,
            "choice_kernel": "one_step",
            "action_selection": "softmax",
            "optimizer": "differential_evolution",
            "polish": True,
        },
        "target": {
            "dataset": args.dataset,
            "audit": audit,
            "condition": "matched_half",
            "subject_id": args.subject_id,
        },
        "seed": 0,
        "meta": {
            "study": "09-gru-cross-species-transfer",
            "variant": os.environ.get("BFM_META_VARIANT", "q-matched-half"),
            "dispatcher_commit": os.environ.get("DISPATCHER_COMMIT"),
            "wrapper_commit": os.environ.get("WRAPPER_COMMIT"),
            "foraging_models_commit": os.environ.get("FORAGING_MODELS_COMMIT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    run = wandb.init(
        entity="AIND-disRNN",
        project="gru_cross_species_transfer",
        group=os.environ.get("WANDB_RUN_GROUP"),
        name=f"q-{args.dataset}-matched-half",
        dir=str(output_dir),
        config=config,
        tags=["external-transfer", "matched-half", "q-learning"],
    )
    loader = ExternalBanditDatasetLoader(
        file_path=args.data_root / f"{args.dataset}.parquet",
        split_manifest_path=args.data_root / f"{args.dataset}.split.json",
        dataset_id=DATASET_IDS[args.dataset],
        subject_ids=[args.subject_id] if args.subject_id else None,
        batch_size=None,
        batch_mode="single",
        adapt_sessions_per_subject=None,
        seed=0,
        train_example_sessions_per_subject=0,
        eval_example_sessions_per_subject=0,
        heldout_example_sessions_per_subject=0,
    )
    trainer = BaselineRLTrainer(
        agent_class="ForagerQLearning",
        architecture={"multisubject": True},
        agent_kwargs={
            "number_of_learning_rate": 1,
            "number_of_forget_rate": 1,
            "choice_kernel": "one_step",
            "action_selection": "softmax",
        },
        fit_bounds_override={},
        clamp_params={},
        DE_kwargs={"polish": True},
        multisubject_subject_workers=args.subject_workers,
        output_dir=str(output_dir / "outputs"),
        seed=0,
    )
    try:
        result = trainer.fit(loader.load(), loggers={"wandb": run})
        for key, value in result["test_metrics"].items():
            if isinstance(value, (int, float)):
                run.summary[f"target/test/{key}"] = value
        print(json.dumps(result["test_metrics"], indent=2))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
