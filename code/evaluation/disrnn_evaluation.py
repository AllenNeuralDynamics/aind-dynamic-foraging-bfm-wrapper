"""Held-out test-subject evaluation helpers for disRNN runs."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import data_loaders.disrnn_dataset as dl
from disentangled_rnns.library import disrnn, rnn_utils

from evaluation.plotting import (
    plot_latents_in_space,
    plot_latents_over_trials,
    save_figure,
)
from utils.load_mice_database import load_mice_from_database
from utils.split_digest import heldout_eval_digests, record_split_digests

from evaluation.heldout_eval_config import (
    HeldoutEvalConfig,
    _cfg_get,
    _resolve_heldout_eval_config,
    should_run_heldout_eval,
)
from evaluation.common import (
    _aligned_action_probabilities_from_output_df,
    _compose_log_prefix,
    _iter_subject_session_groups,
    _load_saved_params,
    _normalize_identifier,
    _prob_from_logits,
    _probs_from_logits_2d,
    _resolve_output_dir,
    _safe_filename_component,
)

logger = logging.getLogger(__name__)

OPEN_LATENT_THRESHOLD = 0.03


def _build_network_configs(model_cfg: Any, dataset: Any, ignore_policy: str) -> tuple[Any, Any]:
    architecture = model_cfg.architecture
    penalties = model_cfg.penalties
    if bool(getattr(architecture, "multisubject", False)):
        raise ValueError(
            "Held-out subject evaluation is not supported for multisubject disRNN. "
            "v1 only supports seen-subject personalization."
        )

    output_size = 2 if ignore_policy == "exclude" else 3
    disrnn_config = disrnn.DisRnnConfig(
        obs_size=dataset._xs.shape[2],
        output_size=output_size,
        x_names=dataset.x_names,
        y_names=dataset.y_names,
        latent_size=architecture.latent_size,
        update_net_n_units_per_layer=architecture.update_net_n_units_per_layer,
        update_net_n_layers=architecture.update_net_n_layers,
        choice_net_n_units_per_layer=architecture.choice_net_n_units_per_layer,
        choice_net_n_layers=architecture.choice_net_n_layers,
        activation=architecture.activation,
        noiseless_mode=False,
        latent_penalty=penalties.latent_penalty,
        choice_net_latent_penalty=penalties.choice_net_latent_penalty,
        update_net_obs_penalty=penalties.update_net_obs_penalty,
        update_net_latent_penalty=penalties.update_net_latent_penalty,
    )

    noiseless_network = copy.deepcopy(disrnn_config)
    noiseless_network.latent_penalty = 0
    noiseless_network.choice_net_latent_penalty = 0
    noiseless_network.update_net_obs_penalty = 0
    noiseless_network.update_net_latent_penalty = 0
    noiseless_network.l2_scale = 0
    noiseless_network.noiseless_mode = True
    return disrnn_config, noiseless_network


def _get_open_latents_from_params(
    params: Any,
    *,
    latent_size: int,
    threshold: float,
) -> list[int]:
    """Return latent indices with bottleneck value > threshold.

    Bottleneck is defined as ``1 - sigma`` where ``sigma`` is derived from
    ``latent_sigma_params``.
    """
    if latent_size <= 0:
        return []

    params_disrnn = None
    if isinstance(params, dict):
        if "hk_disentangled_rnn" in params:
            params_disrnn = params["hk_disentangled_rnn"]
        elif "multisubject_dis_rnn" in params:
            params_disrnn = params["multisubject_dis_rnn"]

    if not isinstance(params_disrnn, dict) or "latent_sigma_params" not in params_disrnn:
        logger.warning(
            "Could not locate latent_sigma_params in params; defaulting open latents to all."
        )
        return list(range(latent_size))

    latent_sigmas = np.asarray(
        disrnn.reparameterize_sigma(params_disrnn["latent_sigma_params"])
    ).reshape(-1)
    bottlenecks = 1.0 - latent_sigmas
    n = min(latent_size, bottlenecks.shape[0])
    return [i for i in range(n) if float(bottlenecks[i]) > threshold]


def plot_disrnn_examples_for_split(
    *,
    split_name: str,
    output_dir: Path,
    output_df: Any,
    network_states: np.ndarray,
    yhat_logits: np.ndarray,
    params: Any,
    sessions_per_subject: int,
    max_subjects_to_plot: int = 6,
    n_action_logits: int | None = None,
    open_latent_threshold: float = OPEN_LATENT_THRESHOLD,
    wandb_run: Any | None = None,
    log_scope: str | None = None,
) -> dict[str, Any]:
    """Generate example plots for a split and return a summary dictionary."""
    if sessions_per_subject < 0:
        raise ValueError("sessions_per_subject must be >= 0.")
    if max_subjects_to_plot < 0:
        raise ValueError("max_subjects_to_plot must be >= 0.")

    output_df = output_df.copy()
    states = np.asarray(network_states)
    logits = np.asarray(yhat_logits)
    if states.ndim != 3:
        raise ValueError(f"Expected network_states 3D, got shape={states.shape}")
    if logits.ndim != 3:
        raise ValueError(f"Expected yhat_logits 3D, got shape={logits.shape}")

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = _compose_log_prefix(log_scope, split_name)

    if sessions_per_subject == 0:
        summary = {
            "split": split_name,
            "num_sessions": int(output_df["ses_idx"].nunique()) if "ses_idx" in output_df.columns else 0,
            "num_trials": int(len(output_df)),
            "plotting_skipped": True,
            "example_sessions_per_subject": 0,
            "example_max_subjects": max_subjects_to_plot,
            "plots": {
                "latents_over_trials_examples": [],
                "latents_in_space_examples": [],
            },
        }
        summary_path = split_dir / "split_eval_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        return summary

    try:
        open_latents = _get_open_latents_from_params(
            params,
            latent_size=int(states.shape[2]),
            threshold=open_latent_threshold,
        )
        logger.info(
            "%s open latents by bottleneck > %.4f: %s",
            log_prefix,
            open_latent_threshold,
            open_latents,
        )

        subject_groups = _iter_subject_session_groups(output_df)
        if not subject_groups:
            raise ValueError(f"No sessions available for split plotting: {split_name}")
        if len(subject_groups) > max_subjects_to_plot:
            logger.info(
                "%s: Limiting example plotting to first %d subjects (of %d total)",
                log_prefix,
                max_subjects_to_plot,
                len(subject_groups),
            )
        subject_groups = subject_groups[:max_subjects_to_plot]

        session_order = list(dict.fromkeys(output_df["ses_idx"].tolist()))
        session_index_by_id = {session_id: index for index, session_id in enumerate(session_order)}
        if len(session_index_by_id) != states.shape[1]:
            raise ValueError(
                "Split plotting requires output_df session order to match network states. "
                f"Found sessions={len(session_index_by_id)} states={states.shape[1]}"
            )

        if n_action_logits is None or n_action_logits <= 0:
            n_action_logits = int(logits.shape[2] - 1)
        if n_action_logits <= 0:
            raise ValueError(f"Invalid number of action logits inferred for split '{split_name}'")

        probs_for_coloring = _prob_from_logits(
            logits[:, :, :n_action_logits],
            class_index=1 if n_action_logits > 1 else 0,
        )
        color_label = "P(right)" if n_action_logits == 2 else "P(class_1)"
        selected_latents = [int(i) for i in open_latents if 0 <= int(i) < states.shape[2]][:4]
        if len(selected_latents) < 2:
            selected_latents = list(range(min(4, states.shape[2])))
        run_len = min(30, states.shape[0])

        latent_cols = sorted([c for c in output_df.columns if c.startswith("latent_")])
        if not latent_cols:
            raise ValueError(f"No latent columns found in model outputs for split: {split_name}")

        selected_examples: list[dict[str, Any]] = []
        trial_plot_paths: list[Path] = []
        space_plot_paths: list[Path] = []

        for subject_id, session_ids in subject_groups:
            selected_session_ids = session_ids[:sessions_per_subject]
            session_indices = [
                session_index_by_id[s_id]
                for s_id in session_ids
                if s_id in session_index_by_id
            ]
            if not session_indices:
                continue

            subject_states = states[:, session_indices, :]
            subject_colors = probs_for_coloring[:, session_indices]
            example_run = subject_states[:run_len, 0, :][:, selected_latents]
            subject_states_points = subject_states.reshape(-1, subject_states.shape[2])
            subject_color_points = subject_colors.reshape(-1)

            fig_space = plot_latents_in_space(
                latent_states=subject_states_points,
                color_values=subject_color_points,
                color_label=color_label,
                selected_latents=selected_latents,
                example_run=example_run,
            )
            fig_space.suptitle(str(_normalize_identifier(session_ids[0])), fontsize=14)
            fig_space.subplots_adjust(top=0.93)
            subject_space_plot_path = save_figure(
                fig_space,
                split_dir / f"latents_in_space_subject_{_safe_filename_component(subject_id)}.png",
            )
            space_plot_paths.append(subject_space_plot_path)

            for session_id in selected_session_ids:
                session_df = output_df[output_df["ses_idx"] == session_id].sort_values("trial")
                latents = session_df[latent_cols].to_numpy()
                choices = session_df["animal_response"].to_numpy()
                rewards = session_df["earned_reward"].astype(int).to_numpy()
                trial_numbers = (
                    session_df["trial"].to_numpy() if "trial" in session_df.columns else None
                )
                action_probabilities = _aligned_action_probabilities_from_output_df(
                    session_df,
                    n_action_logits=n_action_logits,
                )
                if latents.shape[0] == 0:
                    continue

                fig_trials = plot_latents_over_trials(
                    choices=choices,
                    rewards=rewards,
                    latents=latents,
                    open_latents=open_latents,
                    action_probabilities=action_probabilities,
                    trial_numbers=trial_numbers,
                )
                fig_trials.suptitle(f"Session {_normalize_identifier(session_id)}", fontsize=14)
                fig_trials.subplots_adjust(top=0.92)
                trials_plot_path = save_figure(
                    fig_trials,
                    split_dir
                    / (
                        f"latents_over_trials_subject_{_safe_filename_component(subject_id)}"
                        f"_session_{_safe_filename_component(session_id)}.png"
                    ),
                )
                trial_plot_paths.append(trials_plot_path)
                selected_examples.append(
                    {
                        "subject_id": _normalize_identifier(subject_id),
                        "session_id": _normalize_identifier(session_id),
                        "latents_over_trials": str(trials_plot_path),
                    }
                )

        if not selected_examples:
            raise ValueError(f"Could not select any sessions for example plotting: {split_name}")
        if not space_plot_paths:
            raise ValueError(f"Could not generate latent-space plots for split: {split_name}")

        summary = {
            "split": split_name,
            "num_sessions": int(len(session_order)),
            "num_trials": int(len(output_df)),
            "example_sessions_per_subject": sessions_per_subject,
            "example_max_subjects": max_subjects_to_plot,
            "open_latent_threshold": open_latent_threshold,
            "open_latents": open_latents,
            "example_session": selected_examples[0]["session_id"],
            "example_subject": selected_examples[0]["subject_id"],
            "example_sessions": selected_examples,
            "plots": {
                "latents_over_trials_examples": [str(p) for p in trial_plot_paths],
                "latents_in_space_examples": [str(p) for p in space_plot_paths],
            },
        }
    except Exception as exc:
        logger.warning("Split plotting failed for %s: %s", split_name, exc)
        summary = {
            "split": split_name,
            "num_sessions": int(output_df["ses_idx"].nunique()) if "ses_idx" in output_df.columns else 0,
            "num_trials": int(len(output_df)),
            "plotting_failed": True,
            "error": str(exc),
            "example_sessions_per_subject": sessions_per_subject,
            "plots": {
                "latents_over_trials_examples": [],
                "latents_in_space_examples": [],
            },
        }

    summary_path = split_dir / "split_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None and not summary.get("plotting_failed", False):
        import wandb

        wandb_trial_images = [
            wandb.Image(str(path))
            for path in summary["plots"]["latents_over_trials_examples"]
        ]
        wandb_space_images = [
            wandb.Image(str(path))
            for path in summary["plots"]["latents_in_space_examples"]
        ]
        wandb_run.log(
            {
                f"{split_name}/latents_over_trials_examples": wandb_trial_images,
                f"{split_name}/latents_in_space_examples": wandb_space_images,
            }
        )

    return summary


def evaluate_disrnn_on_heldout_subjects(
    hydra_config: Any,
    *,
    wandb_run: Any | None = None,
    params_path: Path | None = None,
    output_subdir: str = "heldout_test",
    log_to_wandb: bool = True,
    heldout_data: dict[str, Any] | None = None,
    log_scope: str | None = None,
) -> dict[str, Any] | None:
    """Evaluate trained disRNN model on held-out test subjects.

    This loads test subjects from snapshot data according to the test selection
    fields in ``hydra_config.data`` and evaluates the saved model parameters.
    """
    heldout_cfg = _resolve_heldout_eval_config(hydra_config)
    model_cfg = hydra_config.model

    if not heldout_cfg.enabled:
        logger.info("Held-out evaluation disabled (no test subject selectors configured).")
        return None
    if bool(getattr(model_cfg.architecture, "multisubject", False)):
        raise ValueError(
            "Held-out subject evaluation is not supported for multisubject disRNN. "
            "v1 only supports seen-subject personalization."
        )
    heldout_cfg.validate()

    output_dir = _resolve_output_dir(model_cfg)
    resolved_params_path = params_path or (output_dir / "params.json")
    if not resolved_params_path.exists():
        raise FileNotFoundError(f"Could not find trained params at {resolved_params_path}")

    if heldout_data is None:
        heldout_data = load_disrnn_heldout_subject_data(hydra_config)
        # loaded here, after the runner's digest point: record what is scored (wrapper #131)
        # provenance goes to the run even when metric logging is off (the runner passes False)
        record_split_digests(wandb_run, lambda: heldout_eval_digests(
            heldout_data, ignore_policy=str(heldout_cfg.ignore_policy)
        ))

    df_test = heldout_data["df_test"]
    test_subject_ids = heldout_data["test_subject_ids"]
    dataset_test = heldout_data["dataset_test"]

    _, noiseless_network = _build_network_configs(
        model_cfg,
        dataset_test,
        ignore_policy=heldout_cfg.ignore_policy,
    )
    params = _load_saved_params(resolved_params_path)

    xs_test = heldout_data["xs_test"]
    ys_test = heldout_data["ys_test"]
    yhat_test, network_states_test = rnn_utils.eval_network(
        lambda: disrnn.HkDisentangledRNN(noiseless_network),
        params,
        xs_test,
    )

    n_action_logits = int(getattr(dataset_test, "n_classes", 0))
    if n_action_logits <= 0:
        n_action_logits = int(np.asarray(yhat_test).shape[2] - 1)
    if n_action_logits <= 0:
        raise ValueError(
            f"Invalid number of action logits inferred for held-out eval: {n_action_logits}"
        )

    test_likelihood = rnn_utils.normalized_likelihood(
        ys_test,
        yhat_test[:, :, :n_action_logits],
    )
    test_likelihood = float(test_likelihood)
    heldout_log_prefix = _compose_log_prefix(log_scope, "held-out test")
    logger.info("%s likelihood: %.4f", heldout_log_prefix, test_likelihood)

    plot_dir = output_dir / output_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    output_df = dl.add_model_results(
        df_test.copy(),
        np.asarray(network_states_test),
        np.asarray(yhat_test),
        ignore_policy=heldout_cfg.ignore_policy,
    )

    open_latent_threshold = OPEN_LATENT_THRESHOLD
    open_latents = _get_open_latents_from_params(
        params,
        latent_size=int(np.asarray(network_states_test).shape[2]),
        threshold=open_latent_threshold,
    )
    logger.info(
        "%s open latents by bottleneck > %.4f: %s",
        heldout_log_prefix,
        open_latent_threshold,
        open_latents,
    )

    session_ids = output_df["ses_idx"].unique()
    if len(session_ids) == 0:
        raise ValueError("No sessions found after held-out model evaluation.")

    if "subject_id" not in output_df.columns:
        raise ValueError("Held-out model outputs do not include required column: subject_id")

    sessions_per_subject = heldout_cfg.heldout_example_sessions_per_subject
    max_subjects_to_plot = heldout_cfg.example_max_subjects

    if sessions_per_subject == 0:
        summary = {
            "enabled": True,
            "params_path": str(resolved_params_path),
            "test_subject_ids": [_normalize_identifier(s) for s in test_subject_ids],
            "num_test_trials": int(len(df_test)),
            "num_test_sessions": int(len(session_ids)),
            "test_likelihood": test_likelihood,
            "heldout_example_sessions_per_subject": 0,
            "plotting_skipped": True,
            "plots": {
                "latents_over_trials_examples": [],
                "latents_in_space_examples": [],
            },
        }

        summary_path = plot_dir / "heldout_eval_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        if wandb_run is not None and log_to_wandb:
            wandb_run.summary["heldout/test_likelihood"] = test_likelihood
            wandb_run.summary["heldout/num_test_sessions"] = int(len(session_ids))
            wandb_run.summary["heldout/num_test_trials"] = int(len(df_test))
            wandb_run.summary["heldout/example_sessions_per_subject"] = 0

        return summary

    try:
        subject_session_rows = (
            output_df[["subject_id", "ses_idx"]]
            .drop_duplicates()
            .sort_values(["subject_id", "ses_idx"])
        )
        subject_groups = list(subject_session_rows.groupby("subject_id", sort=False))
        if len(subject_groups) > max_subjects_to_plot:
            logger.info(
                "%s: Limiting example plotting to first %d subjects (of %d total)",
                heldout_log_prefix,
                max_subjects_to_plot,
                len(subject_groups),
            )
        selected_subject_groups = subject_groups[:max_subjects_to_plot]

        selected_examples: list[dict[str, Any]] = []
        trial_plot_paths: list[Path] = []
        latent_cols = sorted([c for c in output_df.columns if c.startswith("latent_")])
        if not latent_cols:
            raise ValueError("No latent columns found in held-out model outputs.")

        for subject_id, subject_rows in selected_subject_groups:
            subject_sessions = subject_rows["ses_idx"].tolist()
            for session_id in subject_sessions[:sessions_per_subject]:
                session_df = output_df[output_df["ses_idx"] == session_id].sort_values("trial")
                latents = session_df[latent_cols].to_numpy()
                choices = session_df["animal_response"].to_numpy()
                rewards = session_df["earned_reward"].astype(int).to_numpy()
                trial_numbers = (
                    session_df["trial"].to_numpy() if "trial" in session_df.columns else None
                )
                action_probabilities = _aligned_action_probabilities_from_output_df(
                    session_df,
                    n_action_logits=n_action_logits,
                )
                if latents.shape[0] == 0:
                    continue

                fig_trials = plot_latents_over_trials(
                    choices=choices,
                    rewards=rewards,
                    latents=latents,
                    open_latents=open_latents,
                    action_probabilities=action_probabilities,
                    trial_numbers=trial_numbers,
                )
                fig_trials.suptitle(
                    f"Session {_normalize_identifier(session_id)}",
                    fontsize=14,
                )
                fig_trials.subplots_adjust(top=0.92)
                trials_plot_path = save_figure(
                    fig_trials,
                    plot_dir
                    / (
                        f"latents_over_trials_subject_{_safe_filename_component(subject_id)}"
                        f"_session_{_safe_filename_component(session_id)}.png"
                    ),
                )
                trial_plot_paths.append(trials_plot_path)
                selected_examples.append(
                    {
                        "subject_id": _normalize_identifier(subject_id),
                        "session_id": _normalize_identifier(session_id),
                        "latents_over_trials": str(trials_plot_path),
                    }
                )

        if not selected_examples:
            raise ValueError("Could not select any held-out sessions for example plotting.")

        states = np.asarray(network_states_test)
        session_order = list(dict.fromkeys(df_test["ses_idx"].tolist()))
        session_index_by_id = {session_id: index for index, session_id in enumerate(session_order)}
        prob_class_index = 1 if n_action_logits > 1 else 0
        probs_for_coloring = _prob_from_logits(
            np.asarray(yhat_test)[:, :, :n_action_logits],
            class_index=prob_class_index,
        )
        color_label = "P(right)" if n_action_logits == 2 else f"P(class_{prob_class_index})"
        selected_latents = [int(i) for i in open_latents if 0 <= int(i) < states.shape[2]][:4]
        if len(selected_latents) < 2:
            selected_latents = list(range(min(4, states.shape[2])))
        run_len = min(30, states.shape[0])

        space_plot_paths: list[Path] = []
        for subject_id, subject_rows in selected_subject_groups:
            subject_sessions = subject_rows["ses_idx"].tolist()
            session_indices = [
                session_index_by_id[s_id]
                for s_id in subject_sessions
                if s_id in session_index_by_id
            ]
            if not session_indices:
                continue

            subject_states = states[:, session_indices, :]
            subject_colors = probs_for_coloring[:, session_indices]
            example_session_id = subject_sessions[0]
            example_run = subject_states[:run_len, 0, :][:, selected_latents]
            subject_states_points = subject_states.reshape(-1, subject_states.shape[2])
            subject_color_points = subject_colors.reshape(-1)

            fig_space = plot_latents_in_space(
                latent_states=subject_states_points,
                color_values=subject_color_points,
                color_label=color_label,
                selected_latents=selected_latents,
                example_run=example_run,
            )
            fig_space.suptitle(
                str(_normalize_identifier(example_session_id)),
                fontsize=14,
            )
            fig_space.subplots_adjust(top=0.93)
            subject_space_plot_path = save_figure(
                fig_space,
                plot_dir
                / f"latents_in_space_subject_{_safe_filename_component(subject_id)}.png",
            )
            space_plot_paths.append(subject_space_plot_path)

        if not space_plot_paths:
            raise ValueError("Could not generate held-out latent-space plots for any subject.")

        summary = {
            "enabled": True,
            "params_path": str(resolved_params_path),
            "test_subject_ids": [_normalize_identifier(s) for s in test_subject_ids],
            "num_test_trials": int(len(df_test)),
            "num_test_sessions": int(len(session_ids)),
            "test_likelihood": test_likelihood,
            "heldout_example_sessions_per_subject": sessions_per_subject,
            "example_max_subjects": max_subjects_to_plot,
            "heldout_open_latent_threshold": open_latent_threshold,
            "open_latents": open_latents,
            "example_session": selected_examples[0]["session_id"],
            "example_subject": selected_examples[0]["subject_id"],
            "example_sessions": selected_examples,
            "plots": {
                "latents_over_trials_examples": [str(p) for p in trial_plot_paths],
                "latents_in_space_examples": [str(p) for p in space_plot_paths],
            },
        }
    except Exception as exc:
        logger.warning("Held-out plotting failed and will be skipped: %s", exc)
        summary = {
            "enabled": True,
            "params_path": str(resolved_params_path),
            "test_subject_ids": [_normalize_identifier(s) for s in test_subject_ids],
            "num_test_trials": int(len(df_test)),
            "num_test_sessions": int(len(session_ids)),
            "test_likelihood": test_likelihood,
            "heldout_example_sessions_per_subject": sessions_per_subject,
            "example_max_subjects": max_subjects_to_plot,
            "heldout_open_latent_threshold": open_latent_threshold,
            "open_latents": open_latents,
            "plotting_failed": True,
            "error": str(exc),
            "plots": {
                "latents_over_trials_examples": [],
                "latents_in_space_examples": [],
            },
        }

    summary_path = plot_dir / "heldout_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None and log_to_wandb:
        import wandb

        wandb_run.summary["heldout/test_likelihood"] = test_likelihood
        wandb_run.summary["heldout/num_test_sessions"] = int(len(session_ids))
        wandb_run.summary["heldout/num_test_trials"] = int(len(df_test))
        wandb_run.summary["heldout/example_sessions_per_subject"] = sessions_per_subject
        if not summary.get("plotting_failed", False):
            wandb_trial_images = [
                wandb.Image(str(path))
                for path in summary["plots"]["latents_over_trials_examples"]
            ]
            wandb_space_images = [
                wandb.Image(str(path))
                for path in summary["plots"]["latents_in_space_examples"]
            ]
            wandb_run.log(
                {
                    "heldout/latents_over_trials_examples": wandb_trial_images,
                    "heldout/latents_in_space_examples": wandb_space_images,
                }
            )

    return summary


def load_disrnn_heldout_subject_data(config_source: Any) -> dict[str, Any]:
    """Load and construct held-out disRNN data once for reuse across eval calls."""
    heldout_cfg = _resolve_heldout_eval_config(config_source)

    if not heldout_cfg.enabled:
        raise ValueError("Held-out evaluation disabled (data.heldout_eval is false).")
    heldout_cfg.validate()

    logger.info(
        "Loading held-out test subjects (split=heldout): ids=%s, curricula=%s, "
        "min_sessions=%s, heldout_every_n=%s",
        heldout_cfg.test_subject_ids,
        heldout_cfg.curricula,
        heldout_cfg.min_sessions,
        heldout_cfg.heldout_every_n,
    )

    df_test, test_subject_ids = load_mice_from_database(
        split="heldout",
        subject_ids=heldout_cfg.test_subject_ids,
        curricula=heldout_cfg.curricula,
        min_sessions=heldout_cfg.min_sessions,
        heldout_every_n=heldout_cfg.heldout_every_n,
        mature_only=heldout_cfg.mature_only,
        cols_to_retain=heldout_cfg.cols_to_retain,
        snapshot=heldout_cfg.snapshot,
    )
    if len(df_test) == 0:
        raise ValueError("Held-out test selection resulted in an empty dataset.")

    dataset_test = dl.create_disrnn_dataset(
        df_test,
        ignore_policy=heldout_cfg.ignore_policy,
        features=heldout_cfg.features,
        batch_size=heldout_cfg.batch_size,
        batch_mode=heldout_cfg.batch_mode,
    )
    # Keep held-out raw data aligned with sessions that survive dataset construction.
    # create_disrnn_dataset can silently drop sessions whose every trial is ignored
    # when ignore_policy == "exclude", which would otherwise break add_model_results.
    if heldout_cfg.ignore_policy == "exclude" and "animal_response" in df_test.columns:
        valid_sessions = df_test[df_test["animal_response"] != 2]["ses_idx"].unique()
        df_test = df_test[df_test["ses_idx"].isin(valid_sessions)].copy()
    _all = dataset_test.get_all()
    xs_test, ys_test = _all["xs"], _all["ys"]

    return {
        "df_test": df_test,
        "test_subject_ids": test_subject_ids,
        "dataset_test": dataset_test,
        "xs_test": xs_test,
        "ys_test": ys_test,
    }
