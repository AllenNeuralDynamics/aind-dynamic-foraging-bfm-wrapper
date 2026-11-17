"""Held-out evaluation helpers for GRU runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from disentangled_rnns.library import rnn_utils

from models.gru_network import make_gru_network
from evaluation.heldout_eval_config import HeldoutEvalConfig, _resolve_heldout_eval_config
from evaluation.common import (
    _aligned_action_probabilities_from_output_df,
    _iter_subject_session_groups,
    _load_saved_params,
    _normalize_identifier,
    _prob_from_logits,
    _resolve_output_dir,
    _safe_filename_component,
)
# load_disrnn_heldout_subject_data builds a disRNN-format dataset that GRU eval reuses;
# it is the one appropriately model-specific helper still sourced from disrnn_evaluation.
from evaluation.disrnn_evaluation import load_disrnn_heldout_subject_data
from utils.split_digest import heldout_eval_digests, record_split_digests
from evaluation.plotting import (
    plot_latents_in_space,
    plot_latents_over_trials,
    save_figure,
)

logger = logging.getLogger(__name__)

def add_gru_model_results(
    df_trials: Any,
    network_states: np.ndarray,
    yhat: np.ndarray,
    *,
    ignore_policy: str,
) -> Any:
    """Attach GRU hidden states and output logits/probabilities to the trial dataframe.

    The upstream ``add_model_results`` helper is disRNN-oriented and expects a
    particular output layout. GRU runs only need hidden-state trajectories plus
    action logits, so we align those tensors back onto the filtered trial rows
    directly.
    """
    output_df = df_trials.copy()
    if ignore_policy == "exclude" and "animal_response" in output_df.columns:
        output_df = output_df[output_df["animal_response"] != 2].copy()

    if "ses_idx" not in output_df.columns or "trial" not in output_df.columns:
        raise ValueError("GRU output dataframe requires 'ses_idx' and 'trial' columns.")

    session_order = list(dict.fromkeys(output_df["ses_idx"].tolist()))
    output_df["_session_order"] = output_df["ses_idx"].map(
        {session_id: index for index, session_id in enumerate(session_order)}
    )
    sort_columns = ["_session_order"]
    if "trial" in output_df.columns:
        sort_columns.append("trial")
    # Keep ``_session_order`` (dropped just before returning); the vectorized
    # alignment below reuses it as the dense session index and to derive the
    # per-row timestep index.
    output_df = output_df.sort_values(sort_columns).copy()

    states = np.asarray(network_states)
    logits = np.asarray(yhat)
    if states.ndim != 3:
        raise ValueError(f"Expected GRU network_states to be 3D, got shape={states.shape}")
    if logits.ndim != 3:
        raise ValueError(f"Expected GRU yhat to be 3D, got shape={logits.shape}")
    if states.shape[:2] != logits.shape[:2]:
        raise ValueError(
            "GRU states/logits shape mismatch: "
            f"states={states.shape} logits={logits.shape}"
        )

    if len(session_order) != states.shape[1]:
        raise ValueError(
            "Session mismatch between dataframe and GRU outputs: "
            f"df_sessions={len(session_order)} model_sessions={states.shape[1]}"
        )

    latent_cols = [f"latent_{idx}" for idx in range(states.shape[2])]
    logit_cols = [f"choice_logit_{idx}" for idx in range(logits.shape[2])]

    logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits_stable)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    prob_cols = [f"choice_prob_{idx}" for idx in range(probs.shape[2])]

    # Align the per-(timestep, session) GRU tensors back onto the trial rows with a
    # single vectorized gather. Rows are already sorted by (_session_order, trial),
    # so each session's rows are contiguous and ordered by trial: the within-session
    # positional index IS the timestep index. This replaces the previous
    # O(n_sessions * n_rows) per-session boolean-mask loop (a full df scan + masked
    # ``.loc`` write per session) with one O(n_rows) gather.
    session_index_per_row = output_df["_session_order"].to_numpy()
    timestep_index_per_row = (
        output_df.groupby("_session_order", sort=False).cumcount().to_numpy()
    )

    session_lengths = np.bincount(session_index_per_row, minlength=states.shape[1])
    overflow = np.flatnonzero(session_lengths > states.shape[0])
    if overflow.size:
        bad_index = int(overflow[0])
        raise ValueError(
            "More dataframe rows than GRU timepoints for session "
            f"{session_order[bad_index]}: rows={int(session_lengths[bad_index])} "
            f"timepoints={states.shape[0]}"
        )

    # Attach the latent/logit/prob columns in a single block. Assigning each
    # group separately calls frame.insert once per column (latent_cols alone is
    # hidden_size-wide), which fragments the frame; build one DataFrame and
    # concat once instead. Concat on a positional index so a non-unique trial
    # index can't trigger a cartesian align, then restore the original index.
    aligned = np.concatenate(
        [
            states[timestep_index_per_row, session_index_per_row, :],
            logits[timestep_index_per_row, session_index_per_row, :],
            probs[timestep_index_per_row, session_index_per_row, :],
        ],
        axis=1,
    )
    new_cols = pd.DataFrame(aligned, columns=latent_cols + logit_cols + prob_cols)
    trial_index = output_df.index
    output_df = pd.concat([output_df.reset_index(drop=True), new_cols], axis=1)
    output_df.index = trial_index

    return output_df.drop(columns=["_session_order"])


def _require_n_action_logits(dataset: Any, yhat: np.ndarray, *, context: str) -> int:
    n_action_logits = int(getattr(dataset, "n_classes", 0))
    if n_action_logits <= 0:
        raise ValueError(
            f"GRU {context} requires dataset.n_classes to be set to a positive integer."
        )
    actual_output_size = int(np.asarray(yhat).shape[2])
    if actual_output_size != n_action_logits:
        raise ValueError(
            f"GRU {context} logits shape mismatch: dataset.n_classes={n_action_logits} "
            f"but yhat.shape[2]={actual_output_size}"
        )
    return n_action_logits


def _project_hidden_states_to_pcs(
    states: np.ndarray,
    *,
    max_components: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Project hidden states onto the top principal components."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 3:
        raise ValueError(f"Expected hidden states to be 3D, got shape={states.shape}")

    n_hidden = int(states.shape[2])
    n_components = max(1, min(int(max_components), n_hidden))

    flat_states = states.reshape(-1, n_hidden)
    centered = flat_states - flat_states.mean(axis=0, keepdims=True)

    if n_hidden == 1:
        projected = centered
        explained = np.array([1.0], dtype=float)
    else:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh[:n_components]
        projected = centered @ components.T
        variances = singular_values**2
        total_variance = float(np.sum(variances))
        if total_variance <= 0:
            explained = np.zeros(n_components, dtype=float)
        else:
            explained = (variances[:n_components] / total_variance).astype(float)

    return projected.reshape(states.shape[0], states.shape[1], n_components), explained


def _compose_log_prefix(log_scope: str | None, label: str) -> str:
    if not log_scope:
        return label
    if log_scope.lower().startswith("checkpoint step"):
        return f"{log_scope}, {label}"
    return f"{log_scope} {label}"


def plot_gru_examples_for_split(
    *,
    split_name: str,
    output_dir: Path,
    output_df: Any,
    network_states: np.ndarray,
    yhat_logits: np.ndarray,
    sessions_per_subject: int,
    max_subjects_to_plot: int = 6,
    n_action_logits: int,
    wandb_run: Any | None = None,
    log_scope: str | None = None,
) -> dict[str, Any]:
    """Generate example plots for a GRU split using hidden states directly."""
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
        hidden_units = list(range(int(states.shape[2])))
        plotted_hidden_units = hidden_units[:5]
        projected_states, explained_variance_ratio = _project_hidden_states_to_pcs(states)

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

        if n_action_logits <= 0:
            raise ValueError(f"Invalid number of action logits for split '{split_name}'")

        probs_for_coloring = _prob_from_logits(
            logits[:, :, :n_action_logits],
            class_index=1 if n_action_logits > 1 else 0,
        )
        color_label = "P(right)" if n_action_logits == 2 else "P(right)"
        # For the 3-way (ignore-included) head, also color the SAME latent space
        # by P(ignore) so the engage/disengage axis is visible alongside the L/R
        # axis. Class index 2 = ignore (0=left, 1=right, 2=ignore).
        engage_probs_for_coloring = None
        if n_action_logits >= 3:
            engage_probs_for_coloring = _prob_from_logits(
                logits[:, :, :n_action_logits],
                class_index=2,
            )
        selected_latents = list(range(projected_states.shape[2]))
        run_len = min(30, states.shape[0])

        latent_cols = sorted([c for c in output_df.columns if c.startswith("latent_")])
        if not latent_cols:
            raise ValueError(f"No latent columns found in model outputs for split: {split_name}")

        selected_examples: list[dict[str, Any]] = []
        trial_plot_paths: list[Path] = []
        space_plot_paths: list[Path] = []
        engage_space_plot_paths: list[Path] = []

        for subject_id, session_ids in subject_groups:
            selected_session_ids = session_ids[:sessions_per_subject]
            session_indices = [
                session_index_by_id[s_id]
                for s_id in session_ids
                if s_id in session_index_by_id
            ]
            if not session_indices:
                continue

            subject_states = projected_states[:, session_indices, :]
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
                axis_label_prefix="PC",
            )
            fig_space.suptitle(str(_normalize_identifier(session_ids[0])), fontsize=14)
            subject_space_plot_path = save_figure(
                fig_space,
                split_dir / f"latents_in_space_subject_{_safe_filename_component(subject_id)}.png",
            )
            space_plot_paths.append(subject_space_plot_path)

            # Engagement (P(ignore)) view of the same latent space, 3-way only.
            if engage_probs_for_coloring is not None:
                subject_engage_colors = engage_probs_for_coloring[:, session_indices]
                subject_engage_color_points = subject_engage_colors.reshape(-1)
                fig_engage = plot_latents_in_space(
                    latent_states=subject_states_points,
                    color_values=subject_engage_color_points,
                    color_label="P(ignore)",
                    selected_latents=selected_latents,
                    example_run=example_run,
                    axis_label_prefix="PC",
                )
                fig_engage.suptitle(str(_normalize_identifier(session_ids[0])), fontsize=14)
                subject_engage_plot_path = save_figure(
                    fig_engage,
                    split_dir
                    / f"latents_in_space_engage_subject_{_safe_filename_component(subject_id)}.png",
                )
                engage_space_plot_paths.append(subject_engage_plot_path)

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
                    open_latents=plotted_hidden_units,
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
            "hidden_units": hidden_units,
            "plotted_hidden_units": plotted_hidden_units,
            "state_space_basis": "pca",
            "pca_explained_variance_ratio": explained_variance_ratio.tolist(),
            "example_session": selected_examples[0]["session_id"],
            "example_subject": selected_examples[0]["subject_id"],
            "example_sessions": selected_examples,
            "plots": {
                "latents_over_trials_examples": [str(p) for p in trial_plot_paths],
                "latents_in_space_examples": [str(p) for p in space_plot_paths],
                "latents_in_space_engage_examples": [
                    str(p) for p in engage_space_plot_paths
                ],
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
        wandb_engage_images = [
            wandb.Image(str(path))
            for path in summary["plots"].get("latents_in_space_engage_examples", [])
        ]
        space_log = {
            f"{split_name}/latents_over_trials_examples": wandb_trial_images,
            f"{split_name}/latents_in_space_examples": wandb_space_images,
        }
        if wandb_engage_images:
            space_log[f"{split_name}/latents_in_space_engage_examples"] = (
                wandb_engage_images
            )
        wandb_run.log(space_log)

    return summary


def load_gru_heldout_subject_data(config_source: Any) -> dict[str, Any]:
    """Load held-out GRU data once for reuse across evaluation calls."""
    return load_disrnn_heldout_subject_data(config_source)


def evaluate_gru_on_heldout_subjects(
    hydra_config: Any,
    *,
    wandb_run: Any | None = None,
    params_path: Path | None = None,
    output_subdir: str = "heldout_test",
    log_to_wandb: bool = True,
    heldout_data: dict[str, Any] | None = None,
    log_scope: str | None = None,
) -> dict[str, Any] | None:
    """Evaluate a trained GRU model on held-out test subjects."""
    heldout_cfg = _resolve_heldout_eval_config(hydra_config)
    model_cfg = hydra_config.model
    architecture = model_cfg.architecture

    if bool(getattr(architecture, "multisubject", False)):
        raise ValueError(
            "Held-out subject evaluation is not supported for multisubject GRU. "
            "v1 only supports seen-subject personalization."
        )

    if not heldout_cfg.enabled:
        logger.info("Held-out evaluation disabled (no test subject selectors configured).")
        return None
    heldout_cfg.validate()

    output_dir = _resolve_output_dir(model_cfg)
    resolved_params_path = params_path or (output_dir / "params.json")
    if not resolved_params_path.exists():
        raise FileNotFoundError(f"Could not find trained params at {resolved_params_path}")

    if heldout_data is None:
        heldout_data = load_gru_heldout_subject_data(hydra_config)
        # loaded here, after the runner's digest point: record what is scored (wrapper #131)
        # provenance goes to the run even when metric logging is off (the runner passes False)
        record_split_digests(wandb_run, lambda: heldout_eval_digests(
            heldout_data, ignore_policy=str(heldout_cfg.ignore_policy)
        ))

    df_test = heldout_data["df_test"]
    test_subject_ids = heldout_data["test_subject_ids"]
    dataset_test = heldout_data["dataset_test"]
    xs_test = heldout_data["xs_test"]
    ys_test = heldout_data["ys_test"]

    ignore_policy = heldout_cfg.ignore_policy
    output_size = 2 if ignore_policy == "exclude" else 3
    network_output_size = int(getattr(architecture, "output_size", output_size))
    if network_output_size != output_size:
        raise ValueError(
            "Configured GRU output size does not match dataset ignore_policy: "
            f"configured={network_output_size} expected={output_size}"
        )

    make_network = make_gru_network(
        hidden_size=int(architecture.hidden_size),
        output_size=network_output_size,
    )
    params = _load_saved_params(resolved_params_path)

    yhat_test, network_states_test = rnn_utils.eval_network(
        make_network,
        params,
        xs_test,
    )

    n_action_logits = _require_n_action_logits(
        dataset_test,
        np.asarray(yhat_test),
        context="held-out eval",
    )

    test_likelihood = rnn_utils.normalized_likelihood(
        ys_test,
        np.asarray(yhat_test)[:, :, :n_action_logits],
    )
    test_likelihood = float(test_likelihood)
    heldout_log_prefix = _compose_log_prefix(log_scope, "held-out test")
    logger.info("%s likelihood: %.4f", heldout_log_prefix, test_likelihood)

    plot_dir = output_dir / output_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    output_df = add_gru_model_results(
        df_test.copy(),
        np.asarray(network_states_test),
        np.asarray(yhat_test),
        ignore_policy=ignore_policy,
    )

    states = np.asarray(network_states_test)
    open_hidden_units = list(range(int(states.shape[2])))
    plotted_hidden_units = open_hidden_units[:5]
    projected_states, explained_variance_ratio = _project_hidden_states_to_pcs(states)

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
                    open_latents=plotted_hidden_units,
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

        session_order = list(dict.fromkeys(df_test["ses_idx"].tolist()))
        session_index_by_id = {session_id: index for index, session_id in enumerate(session_order)}
        prob_class_index = 1 if n_action_logits > 1 else 0
        probs_for_coloring = _prob_from_logits(
            np.asarray(yhat_test)[:, :, :n_action_logits],
            class_index=prob_class_index,
        )
        color_label = "P(right)"
        # Engagement (P(ignore)) coloring for the 3-way head (2 = ignore).
        engage_probs_for_coloring = None
        if n_action_logits >= 3:
            engage_probs_for_coloring = _prob_from_logits(
                np.asarray(yhat_test)[:, :, :n_action_logits],
                class_index=2,
            )
        selected_latents = list(range(projected_states.shape[2]))
        run_len = min(30, projected_states.shape[0])

        space_plot_paths: list[Path] = []
        engage_space_plot_paths: list[Path] = []
        for subject_id, subject_rows in selected_subject_groups:
            subject_sessions = subject_rows["ses_idx"].tolist()
            session_indices = [
                session_index_by_id[s_id]
                for s_id in subject_sessions
                if s_id in session_index_by_id
            ]
            if not session_indices:
                continue

            subject_states = projected_states[:, session_indices, :]
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
                axis_label_prefix="PC",
            )
            fig_space.suptitle(
                str(_normalize_identifier(example_session_id)),
                fontsize=14,
            )
            subject_space_plot_path = save_figure(
                fig_space,
                plot_dir
                / f"latents_in_space_subject_{_safe_filename_component(subject_id)}.png",
            )
            space_plot_paths.append(subject_space_plot_path)

            if engage_probs_for_coloring is not None:
                subject_engage_colors = engage_probs_for_coloring[:, session_indices]
                subject_engage_color_points = subject_engage_colors.reshape(-1)
                fig_engage = plot_latents_in_space(
                    latent_states=subject_states_points,
                    color_values=subject_engage_color_points,
                    color_label="P(ignore)",
                    selected_latents=selected_latents,
                    example_run=example_run,
                    axis_label_prefix="PC",
                )
                fig_engage.suptitle(
                    str(_normalize_identifier(example_session_id)),
                    fontsize=14,
                )
                subject_engage_plot_path = save_figure(
                    fig_engage,
                    plot_dir
                    / f"latents_in_space_engage_subject_{_safe_filename_component(subject_id)}.png",
                )
                engage_space_plot_paths.append(subject_engage_plot_path)

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
            "open_hidden_units": open_hidden_units,
            "plotted_hidden_units": plotted_hidden_units,
            "state_space_basis": "pca",
            "pca_explained_variance_ratio": explained_variance_ratio.tolist(),
            "example_session": selected_examples[0]["session_id"],
            "example_subject": selected_examples[0]["subject_id"],
            "example_sessions": selected_examples,
            "plots": {
                "latents_over_trials_examples": [str(p) for p in trial_plot_paths],
                "latents_in_space_examples": [str(p) for p in space_plot_paths],
                "latents_in_space_engage_examples": [
                    str(p) for p in engage_space_plot_paths
                ],
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
            "open_hidden_units": open_hidden_units,
            "plotted_hidden_units": plotted_hidden_units,
            "state_space_basis": "pca",
            "pca_explained_variance_ratio": explained_variance_ratio.tolist(),
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
            wandb_engage_images = [
                wandb.Image(str(path))
                for path in summary["plots"].get(
                    "latents_in_space_engage_examples", []
                )
            ]
            heldout_space_log = {
                "heldout/latents_over_trials_examples": wandb_trial_images,
                "heldout/latents_in_space_examples": wandb_space_images,
            }
            if wandb_engage_images:
                heldout_space_log["heldout/latents_in_space_engage_examples"] = (
                    wandb_engage_images
                )
            wandb_run.log(heldout_space_log)

    return summary
