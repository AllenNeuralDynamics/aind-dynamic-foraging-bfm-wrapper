"""Fit Costa's feedback-dependent RL model with blockwise value resets.

Moved from aind-dynamic-foraging-bfm-studies `studies/09-gru-cross-species-transfer/run_costa_author_baseline.py` (dispatcher #179; identical to dispatcher `818cc6e` apart from reading no study
files at runtime). Run as ``python -m external_transfer.costa_author_baseline`` from the wrapper's ``code/``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject-workers", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _ordered_blocks(subject_rows, session_ids: set[str]) -> list:
    """Return real Costa stimulus blocks from the requested sessions."""
    blocks = []
    for (session_id, _), rows in subject_rows.groupby(
        ["ses_idx", "source_block"], sort=False
    ):
        if str(session_id) in session_ids:
            blocks.append(rows.sort_values("trial"))
    return blocks


def _fit_subject(payload: dict) -> dict:
    from aind_dynamic_foraging_models.generative_model import (
        ForagerFeedbackDependentRL,
    )

    train_blocks = _ordered_blocks(payload["rows"], payload["train_session_ids"])
    eval_blocks = _ordered_blocks(payload["rows"], payload["eval_session_ids"])
    if not train_blocks or not eval_blocks:
        raise AssertionError(
            f"Costa subject {payload['subject_id']!r} has an empty split"
        )
    train_choices = [
        block["animal_response"].to_numpy(dtype=int) for block in train_blocks
    ]
    train_rewards = [
        block["earned_reward"].to_numpy(dtype=float) for block in train_blocks
    ]
    agent = ForagerFeedbackDependentRL(seed=payload["seed"])
    fitting_result, _ = agent.fit(
        fit_choice_history=train_choices,
        fit_reward_history=train_rewards,
        fit_bounds_override={},
        clamp_params={},
        DE_kwargs={"polish": True, "workers": 1},
    )
    params = {
        name: float(value) for name, value in fitting_result.params.items()
    }
    agent.set_params(**params)
    eval_choices = [
        block["animal_response"].to_numpy(dtype=int) for block in eval_blocks
    ]
    eval_rewards = [
        block["earned_reward"].to_numpy(dtype=float) for block in eval_blocks
    ]
    probability_sessions = agent.perform_closed_loop_multi_session(
        eval_choices, eval_rewards
    )
    predictions = []
    for block, probabilities in zip(eval_blocks, probability_sessions):
        probability_right = np.asarray(probabilities, dtype=float)[1]
        predictions.extend(
            (int(row_index), float(probability))
            for row_index, probability in zip(block.index, probability_right)
        )
    return {
        "subject_id": payload["subject_id"],
        "params": params,
        "fit": {
            "log_likelihood": float(fitting_result.log_likelihood),
            "normalized_likelihood": float(fitting_result.LPT),
            "AIC": float(fitting_result.AIC),
            "BIC": float(fitting_result.BIC),
            "n_train_blocks": len(train_blocks),
            "n_train_trials": int(sum(map(len, train_choices))),
        },
        "n_eval_blocks": len(eval_blocks),
        "n_eval_trials": len(predictions),
        "predictions": predictions,
    }


def _subject_payloads(bundle, seed: int) -> list[dict]:
    metadata = bundle.metadata
    if metadata.get("split_strategy") != "explicit_manifest":
        raise ValueError("Costa (macaque) requires its audited schema-v1 split")
    train_session_ids = {str(value) for value in metadata["train_session_ids"]}
    eval_session_ids = {str(value) for value in metadata["eval_session_ids"]}
    subject_seeds = np.random.SeedSequence(seed).spawn(
        int(metadata["num_subjects"])
    )
    payloads = []
    for subject_index, (subject_id, rows) in enumerate(
        bundle.raw.groupby("subject_id", sort=False)
    ):
        payloads.append(
            {
                "subject_id": (
                    subject_id.item()
                    if isinstance(subject_id, np.generic)
                    else subject_id
                ),
                "rows": rows,
                "train_session_ids": train_session_ids,
                "eval_session_ids": eval_session_ids,
                "seed": int(subject_seeds[subject_index].generate_state(1)[0]),
            }
        )
    return payloads


def main() -> None:
    args = _parser().parse_args()

    import wandb
    from data_loaders.external_bandit import ExternalBanditDatasetLoader
    from evaluation.target_transfer import (
        build_binary_trial_predictions,
        summarize_binary_trial_predictions,
    )

    output_dir = args.output_root / "costa-feedback-dependent"
    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.data_root / "costa.audit.json").read_text())
    config = {
        "model": {
            "agent_class": "CostaBlockResetFeedbackDependentRL",
            "base_agent_class": "ForagerFeedbackDependentRL",
            "optimizer": "differential_evolution",
            "value_reset": "source_block",
            "initial_value": 0.5,
            "citation": "Costa et al., Neuron (2016), Eqs. 7-9",
        },
        "target": {
            "dataset": "costa",
            "audit": audit,
            "condition": "matched_half",
        },
        "seed": args.seed,
        "meta": {
            "study": "09-gru-cross-species-transfer",
            "variant": os.environ.get("BFM_META_VARIANT"),
            "dispatcher_commit": os.environ.get("DISPATCHER_COMMIT"),
            "wrapper_commit": os.environ.get("WRAPPER_COMMIT"),
            "foraging_models_commit": os.environ.get("FORAGING_MODELS_COMMIT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    run = wandb.init(
        entity="AIND-disRNN",
        project="gru_cross_species_transfer",
        group=os.environ.get("WANDB_RUN_GROUP"),
        name="costa-feedback-dependent",
        dir=str(output_dir),
        config=config,
        tags=["external-transfer", "matched-half", "author-baseline"],
    )
    loader = ExternalBanditDatasetLoader(
        file_path=args.data_root / "costa.parquet",
        split_manifest_path=args.data_root / "costa.split.json",
        dataset_id="costa-averbeck-2016-stochastic",
        batch_size=None,
        batch_mode="single",
        adapt_sessions_per_subject=None,
        seed=args.seed,
        train_example_sessions_per_subject=0,
        eval_example_sessions_per_subject=0,
        heldout_example_sessions_per_subject=0,
    )
    try:
        bundle = loader.load()
        payloads = _subject_payloads(bundle, args.seed)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.subject_workers, len(payloads))
        ) as executor:
            subject_results = list(executor.map(_fit_subject, payloads))

        probability_right = np.full(len(bundle.raw), np.nan, dtype=float)
        for result in subject_results:
            for row_index, probability in result.pop("predictions"):
                probability_right[row_index] = probability
        predictions = build_binary_trial_predictions(
            bundle.raw,
            bundle.metadata,
            probability_choice_1=probability_right,
            model="costa_block_reset_feedback_dependent_rl",
        )
        metrics = summarize_binary_trial_predictions(predictions)
        predictions_path = outputs_dir / "test_trial_predictions.csv"
        metrics_path = outputs_dir / "test_metrics.json"
        summaries_path = outputs_dir / "subject_fit_summaries.json"
        predictions.to_csv(predictions_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        summaries_path.write_text(
            json.dumps(
                {"num_subjects": len(subject_results), "subjects": subject_results},
                indent=2,
            )
            + "\n"
        )
        result = {
            "multisubject": True,
            "fit_strategy": "per_subject_block_reset",
            "agent_class": "CostaBlockResetFeedbackDependentRL",
            "num_subjects": len(subject_results),
            "test_metrics": metrics,
            "subject_artifacts": {
                "subject_fit_summaries_json": str(summaries_path),
                "test_trial_predictions_csv": str(predictions_path),
                "test_metrics_json": str(metrics_path),
            },
        }
        (outputs_dir / "baseline_rl_results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                run.summary[f"target/test/{key}"] = value
        artifact = wandb.Artifact(
            f"baseline-rl-output-{run.id}", type="training-output"
        )
        artifact.add_dir(str(outputs_dir))
        run.log_artifact(artifact)
        print(json.dumps(metrics, indent=2))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
