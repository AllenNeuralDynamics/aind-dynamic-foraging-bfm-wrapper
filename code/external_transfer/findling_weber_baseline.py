"""Fit Findling's Weber-imprecision model on the matched adaptation sessions.

Moved from aind-dynamic-foraging-bfm-studies `studies/09-gru-cross-species-transfer/run_findling_weber_baseline.py` (dispatcher #179; identical to dispatcher `818cc6e` apart from reading no study
files at runtime). Run as ``python -m external_transfer.findling_weber_baseline`` from the wrapper's ``code/``.
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
    parser.add_argument("--subject-workers", type=int, default=12)
    parser.add_argument("--fit-particles", type=int, default=2)
    parser.add_argument("--evaluation-particles", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subject-id", action="append", default=None)
    return parser


def _fit_subject(payload: dict) -> dict:
    from aind_dynamic_foraging_models.generative_model import (
        filter_findling_weber_session,
        findling_weber_parameter_grid,
        fit_findling_weber_map,
    )

    grid = findling_weber_parameter_grid()
    fitted, train_log_likelihood, map_index = fit_findling_weber_map(
        payload["train_choices"],
        payload["train_rewards"],
        parameters=grid,
        n_particles=payload["fit_particles"],
        seed=payload["seed"],
    )
    session_seeds = np.random.SeedSequence([payload["seed"], 1]).spawn(
        len(payload["eval_choices"])
    )
    predictions = []
    eval_log_likelihood = 0.0
    for choices, rewards, row_indices, session_seed in zip(
        payload["eval_choices"],
        payload["eval_rewards"],
        payload["eval_row_indices"],
        session_seeds,
    ):
        result = filter_findling_weber_session(
            choices,
            rewards,
            fitted[None, :],
            n_particles=payload["evaluation_particles"],
            seed=int(session_seed.generate_state(1)[0]),
        )
        probability_right = result.probability_right[0]
        chosen_probability = np.where(
            np.asarray(choices, dtype=int) == 1,
            probability_right,
            1.0 - probability_right,
        )
        eval_log_likelihood += float(
            np.log(np.clip(chosen_probability, 1e-10, 1.0)).sum()
        )
        predictions.extend(
            (int(row_index), float(probability))
            for row_index, probability in zip(row_indices, probability_right)
        )
    return {
        "subject_id": payload["subject_id"],
        "temperature": float(fitted[0]),
        "weber_slope": float(fitted[1]),
        "map_grid_index": int(map_index),
        "train_total_log_likelihood": float(train_log_likelihood[map_index]),
        "eval_total_log_likelihood": eval_log_likelihood,
        "n_train_trials": int(sum(map(len, payload["train_choices"]))),
        "n_eval_trials": int(sum(map(len, payload["eval_choices"]))),
        "predictions": predictions,
    }


def _subject_payloads(bundle, args: argparse.Namespace) -> list[dict]:
    raw = bundle.raw
    metadata = bundle.metadata
    if metadata.get("split_strategy") != "explicit_manifest":
        raise ValueError(
            "Findling (human) requires the audited schema-v1 session split."
        )
    train_session_ids = {str(value) for value in metadata["train_session_ids"]}
    eval_session_ids = {str(value) for value in metadata["eval_session_ids"]}
    subject_seeds = np.random.SeedSequence(args.seed).spawn(
        int(metadata["num_subjects"])
    )
    payloads = []
    for subject_index, (subject_id, subject_rows) in enumerate(
        raw.groupby("subject_id", sort=False)
    ):
        train_choices, train_rewards = [], []
        eval_choices, eval_rewards, eval_row_indices = [], [], []
        for session_id, session_rows in subject_rows.groupby("ses_idx", sort=False):
            session_rows = session_rows.sort_values("trial")
            choices = session_rows["animal_response"].to_numpy(dtype=int)
            rewards = session_rows["earned_reward"].to_numpy(dtype=int)
            normalized_session_id = str(session_id)
            if normalized_session_id in train_session_ids:
                train_choices.append(choices)
                train_rewards.append(rewards)
            elif normalized_session_id in eval_session_ids:
                eval_choices.append(choices)
                eval_rewards.append(rewards)
                eval_row_indices.append(session_rows.index.to_numpy(dtype=int))
            else:
                raise AssertionError(
                    f"Session {session_id!r} is not assigned to a split."
                )
        if not train_choices or not eval_choices:
            raise AssertionError(f"Subject {subject_id!r} has an empty matched split.")
        payloads.append(
            {
                "subject_id": (
                    subject_id.item()
                    if isinstance(subject_id, np.generic)
                    else subject_id
                ),
                "train_choices": train_choices,
                "train_rewards": train_rewards,
                "eval_choices": eval_choices,
                "eval_rewards": eval_rewards,
                "eval_row_indices": eval_row_indices,
                "fit_particles": args.fit_particles,
                "evaluation_particles": args.evaluation_particles,
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

    output_dir = args.output_root / "findling-weber-imprecision"
    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.data_root / "findling.audit.json").read_text())
    config = {
        "model": {
            "agent_class": "FindlingWeberImprecision",
            "optimizer": "released_1000_point_sobol_map",
            "fit_particles": args.fit_particles,
            "evaluation_particles": args.evaluation_particles,
            "citation": "Findling et al., Nature Communications (2025)",
        },
        "target": {
            "dataset": "findling",
            "audit": audit,
            "condition": "matched_half",
            "subject_ids": args.subject_id,
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
        name="findling-weber-imprecision",
        dir=str(output_dir),
        config=config,
        tags=["external-transfer", "matched-half", "author-baseline"],
    )
    loader = ExternalBanditDatasetLoader(
        file_path=args.data_root / "findling.parquet",
        split_manifest_path=args.data_root / "findling.split.json",
        dataset_id="findling-et-al-volnoise",
        subject_ids=args.subject_id,
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
        payloads = _subject_payloads(bundle, args)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.subject_workers, len(payloads))
        ) as executor:
            subject_results = list(executor.map(_fit_subject, payloads))

        probability_choice_1 = np.full(len(bundle.raw), np.nan, dtype=float)
        for result in subject_results:
            for row_index, probability in result.pop("predictions"):
                probability_choice_1[row_index] = probability
        predictions = build_binary_trial_predictions(
            bundle.raw,
            bundle.metadata,
            probability_choice_1=probability_choice_1,
            model="findling_weber_imprecision",
        )
        metrics = summarize_binary_trial_predictions(predictions)
        predictions_path = outputs_dir / "test_trial_predictions.csv"
        metrics_path = outputs_dir / "test_metrics.json"
        summaries_path = outputs_dir / "subject_fit_summaries.json"
        predictions.to_csv(predictions_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        summaries_path.write_text(
            json.dumps(
                {
                    "num_subjects": len(subject_results),
                    "fit_particles": args.fit_particles,
                    "evaluation_particles": args.evaluation_particles,
                    "subjects": subject_results,
                },
                indent=2,
            )
            + "\n"
        )
        output = {
            "multisubject": True,
            "fit_strategy": "per_subject_released_sobol_map",
            "agent_class": "FindlingWeberImprecision",
            "num_subjects": len(subject_results),
            "test_metrics": metrics,
            "subject_artifacts": {
                "subject_fit_summaries_json": str(summaries_path),
                "test_trial_predictions_csv": str(predictions_path),
                "test_metrics_json": str(metrics_path),
            },
        }
        (outputs_dir / "baseline_rl_results.json").write_text(
            json.dumps(output, indent=2) + "\n"
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
