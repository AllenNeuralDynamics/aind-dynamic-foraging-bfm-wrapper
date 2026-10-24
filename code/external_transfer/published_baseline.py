"""Fit an author-aligned behavioral baseline on its matched target split.

Moved from aind-dynamic-foraging-bfm-studies `studies/09-gru-cross-species-transfer/run_published_baseline.py` (dispatcher #179; identical to dispatcher `818cc6e` apart from reading no study
files at runtime). Run as ``python -m external_transfer.published_baseline`` from the wrapper's ``code/``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BASELINES = {
    "grossman-meta-learning": {
        "dataset": "grossman",
        "dataset_id": "grossman-bari-cohen-2021",
        "agent_class": "ForagerGrossmanMetaLearning",
        "citation": "Grossman et al., Current Biology (2022)",
    },
    "chen-rlck": {
        "dataset": "chen",
        "dataset_id": "chen-et-al-2021",
        "agent_class": "ForagerRLCK",
        "citation": "Chen et al., eLife (2021)",
    },
    "zid-traditional-rlck": {
        "dataset": "zid",
        "dataset_id": "zid-et-al-2026-experiment-1",
        "agent_class": "ForagerRLCK",
        "citation": "Zid et al., Nature Communications (2026), Eq. 19",
    },
    "zid-history-kernel-foraging": {
        "dataset": "zid",
        "dataset_id": "zid-et-al-2026-experiment-1",
        "agent_class": "ForagerZidHistoryKernel",
        "citation": "Zid et al., Nature Communications (2026), Eq. 22",
    },
    "lebedeva-pr": {
        "dataset": "lebedeva",
        "dataset_id": "lebedeva-et-al-2026",
        "agent_class": "ForagerLebedevaPR",
        "citation": "Lebedeva et al., Nature Communications (2026), Eqs. 9-11",
    },
    "beron-rflr": {
        "dataset": "beron",
        "dataset_id": "beron-et-al-2022",
        "agent_class": "ForagerBeronRFLR",
        "citation": "Beron et al., PNAS (2022), Eqs. 3-5",
    },
    "miller-rhg": {
        "dataset": "miller",
        "dataset_id": "miller-et-al-2022-tab",
        "agent_class": "ForagerMillerRHG",
        "citation": "Miller et al., bioRxiv (2021), RHG model",
    },
    "eckstein-rl": {
        "dataset": "eckstein",
        "dataset_id": "eckstein-et-al-2022",
        "agent_class": "ForagerEcksteinRL",
        "citation": "Eckstein et al., Developmental Cognitive Neuroscience (2022), RL",
    },
    "eckstein-bi": {
        "dataset": "eckstein",
        "dataset_id": "eckstein-et-al-2022",
        "agent_class": "ForagerEcksteinBI",
        "citation": "Eckstein et al., Developmental Cognitive Neuroscience (2022), BI",
    },
    "hattori-q-learning": {
        "dataset": "hattori",
        "dataset_id": "hattori-et-al-2023-imaging-mature",
        "agent_class": "ForagerQLearning",
        "agent_kwargs": {
            "number_of_learning_rate": 2,
            "number_of_forget_rate": 1,
            "choice_kernel": "none",
            "action_selection": "softmax",
        },
        "citation": "Hattori et al., Nature Neuroscience (2023), Eqs. 17-19",
    },
    "lopez-double-trace": {
        "dataset": "lopez_mouse",
        "dataset_id": "lopez-yepez-et-al-2021-mouse",
        "agent_class": "ForagerLopezDoubleTrace",
        "citation": "López-Yépez et al., Frontiers in Behavioral Neuroscience (2021), double-trace RL",
    },
    "costa-feedback-dependent": {
        "dataset": "costa",
        "dataset_id": "costa-averbeck-2016-stochastic",
        "agent_class": "ForagerFeedbackDependentRL",
        "citation": "Costa et al., Neuron (2016), feedback-dependent RL",
    },
    "costa-feedback-dependent-bias": {
        "dataset": "costa",
        "dataset_id": "costa-averbeck-2016-stochastic",
        "agent_class": "ForagerFeedbackDependentRLBias",
        "citation": (
            "Costa et al., Neuron (2016), feedback-dependent RL, "
            "augmented with a fitted choice bias"
        ),
    },
    "costa-feedback-dependent-bias-ck1": {
        "dataset": "costa",
        "dataset_id": "costa-averbeck-2016-stochastic",
        "agent_class": "ForagerFeedbackDependentRLBiasCK1",
        "citation": (
            "Costa et al., Neuron (2016), feedback-dependent RL, augmented "
            "with fitted choice bias and Bari2019 one-step choice kernel"
        ),
        "comparison_role": "mechanism sensitivity: Costa dual-rate + bias + Bari CK1",
    },
    "alsio-dual-rate-sticky": {
        "dataset": "alsio",
        "dataset_id": "alsio-et-al-2019-vpvd-tsvr",
        "agent_class": "ForagerAlsioRL",
        "citation": "Alsiö et al., Neuropsychopharmacology (2019), dual-rate RL plus side stickiness",
        "comparison_role": "author-model sensitivity on a different released cohort",
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject-workers", type=int, default=12)
    return parser


def main() -> None:
    args = _parser().parse_args()
    specification = BASELINES[args.baseline]
    dataset = specification["dataset"]

    import wandb
    from data_loaders.external_bandit import ExternalBanditDatasetLoader
    from model_trainers.baseline_rl_trainer import BaselineRLTrainer

    output_dir = args.output_root / args.baseline
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.data_root / f"{dataset}.audit.json").read_text())
    config = {
        "model": {
            "agent_class": specification["agent_class"],
            "optimizer": "differential_evolution",
            "polish": True,
            "citation": specification["citation"],
            "comparison_role": specification.get(
                "comparison_role", "author-selected model"
            ),
        },
        "target": {
            "dataset": dataset,
            "audit": audit,
            "condition": "matched_half",
        },
        "seed": 0,
        "meta": {
            "study": "09-gru-cross-species-transfer",
            "variant": os.environ.get("BFM_META_VARIANT"),
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
        name=args.baseline,
        dir=str(output_dir),
        config=config,
        tags=["external-transfer", "matched-half", "author-baseline"],
    )
    loader = ExternalBanditDatasetLoader(
        file_path=args.data_root / f"{dataset}.parquet",
        split_manifest_path=args.data_root / f"{dataset}.split.json",
        dataset_id=specification["dataset_id"],
        batch_size=None,
        batch_mode="single",
        adapt_sessions_per_subject=None,
        seed=0,
        train_example_sessions_per_subject=0,
        eval_example_sessions_per_subject=0,
        heldout_example_sessions_per_subject=0,
    )
    trainer = BaselineRLTrainer(
        agent_class=specification["agent_class"],
        architecture={"multisubject": True},
        agent_kwargs=specification.get("agent_kwargs", {}),
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
