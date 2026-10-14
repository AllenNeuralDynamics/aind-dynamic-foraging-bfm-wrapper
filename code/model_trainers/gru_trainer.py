from __future__ import annotations

import itertools
import json
import logging
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import wandb
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf

from disentangled_rnns.library import rnn_utils

from base.interfaces import ModelTrainer
from base.types import DatasetBundle
from model_trainers.base_multisubject_trainer import BaseMultisubjectTrainer
from model_trainers.checkpoint_resume import (
    find_latest_resumable_state,
    save_train_state,
)
from models.gru_network import make_gru_network
from models.session_conditioning import (
    compute_session_curriculum_lambda,
    resolve_session_conditioning_from_architecture,
    resolve_session_curriculum_steps,
)
from utils.disrnn_evaluation import HeldoutEvalConfig
from utils.gru_evaluation import (
    add_gru_model_results,
    evaluate_gru_on_heldout_subjects,
    load_gru_heldout_subject_data,
    plot_gru_examples_for_split,
)
from utils.multisubject import (
    compute_session_conditioned_context_dataframe,
    extract_subject_embeddings_from_params,
    normalize_subject_id,
    ordered_session_context_rows,
    prepend_session_index_to_multisubject_split_datasets,
    resolve_session_context_plot_subject_indices,
    save_multisubject_metadata,
    save_subject_index_map,
    save_session_context_map,
    session_regularization_index_arrays_from_session_context,
    subject_embeddings_to_dataframe,
)
from utils.session_regularized_training import (
    build_zero_mean_session_delta_regularization_apply,
    train_network_with_session_regularization,
)

logger = logging.getLogger(__name__)


def _build_frozen_gru_core_update_mask(params: Any) -> tuple[Any, dict[str, int]]:
    """Train only multisubject embeddings and the final GRU readout."""
    found_subject_embeddings = False
    found_readout = False
    found_gru = False
    trainable_parameter_count = 0
    frozen_parameter_count = 0

    def _mask_parameter(module_name: str, parameter_name: str, value: Any) -> Any:
        nonlocal found_subject_embeddings
        nonlocal found_readout
        nonlocal found_gru
        nonlocal trainable_parameter_count
        nonlocal frozen_parameter_count

        is_subject_embedding = parameter_name == "subject_embeddings"
        is_readout = module_name.endswith("/~/readout")
        is_gru = module_name.endswith("/~/gru")
        found_subject_embeddings = found_subject_embeddings or is_subject_embedding
        found_readout = found_readout or is_readout
        found_gru = found_gru or is_gru
        parameter_count = int(np.size(value))
        if is_subject_embedding or is_readout:
            trainable_parameter_count += parameter_count
            return np.ones_like(value)
        frozen_parameter_count += parameter_count
        return np.zeros_like(value)

    update_mask = hk.data_structures.map(_mask_parameter, params)
    missing = []
    if not found_subject_embeddings:
        missing.append("subject_embeddings")
    if not found_readout:
        missing.append("readout")
    if not found_gru:
        missing.append("gru")
    if missing:
        raise ValueError(
            "freeze_gru_core could not identify required parameter groups: "
            + ", ".join(missing)
        )
    return update_mask, {
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
    }


def _to_dict(config: Mapping[str, Any] | DictConfig) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


def _is_multisubject_mode(
    architecture: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bool:
    return bool(architecture.get("multisubject", False) or metadata.get("multisubject", False))


def _resolve_multisubject_subject_count(metadata: Mapping[str, Any]) -> int:
    num_subjects = metadata.get("num_subjects")
    if num_subjects is not None:
        resolved = int(num_subjects)
        if resolved > 0:
            return resolved

    subject_ids = metadata.get("subject_ids")
    if isinstance(subject_ids, list) and subject_ids:
        return int(len(subject_ids))

    raise ValueError(
        "Multisubject GRU requires metadata.num_subjects or metadata.subject_ids."
    )


def _validate_multisubject_dataset_inputs(
    dataset: Any,
    *,
    max_n_subjects: int,
    context: str,
    session_conditioning_enabled: bool = False,
    session_max_index_by_subject_index: Sequence[int] | None = None,
) -> None:
    xs = dataset.get_all()["xs"]
    xs = np.asarray(xs)
    if xs.ndim != 3:
        raise ValueError(
            f"Multisubject GRU {context} expects dataset inputs to be 3D, got shape={xs.shape}."
        )
    min_feature_count = 3 if session_conditioning_enabled else 2
    if xs.shape[2] < min_feature_count:
        raise ValueError(
            f"Multisubject GRU {context} requires {min_feature_count} input features "
            "including prepended subject/session indices."
        )

    subject_ids = np.asarray(xs[..., 0], dtype=float)
    if not np.all(np.isfinite(subject_ids)):
        raise ValueError(
            f"Multisubject GRU {context} encountered non-finite subject ids."
        )

    rounded_subject_ids = np.rint(subject_ids)
    integer_like_mask = np.isclose(subject_ids, rounded_subject_ids)
    if not np.all(integer_like_mask):
        bad_values = np.unique(subject_ids[~integer_like_mask]).tolist()
        raise ValueError(
            "Multisubject GRU "
            f"{context} expected integer-valued subject ids, got {bad_values}."
        )

    subject_ids_int = rounded_subject_ids.astype(int)
    invalid_mask = (subject_ids_int < -1) | (subject_ids_int >= int(max_n_subjects))
    if np.any(invalid_mask):
        bad_values = np.unique(subject_ids_int[invalid_mask]).tolist()
        raise ValueError(
            "Multisubject GRU "
            f"{context} encountered out-of-range subject ids {bad_values}. "
            f"Expected padding sentinel -1 or values in [0, {int(max_n_subjects) - 1}]."
        )

    if not session_conditioning_enabled:
        return

    if session_max_index_by_subject_index is None:
        raise ValueError(
            "Multisubject GRU session conditioning validation requires "
            "session_max_index_by_subject_index."
        )

    session_max = np.asarray(session_max_index_by_subject_index, dtype=int)
    if session_max.ndim != 1 or int(session_max.shape[0]) != int(max_n_subjects):
        raise ValueError(
            "Multisubject GRU session conditioning validation requires "
            f"session_max_index_by_subject_index length {max_n_subjects}."
        )

    session_ids = np.asarray(xs[..., 1], dtype=float)
    rounded_session_ids = np.rint(session_ids)
    integer_like_mask = np.logical_or(session_ids == -1, np.isclose(session_ids, rounded_session_ids))
    if not np.all(integer_like_mask):
        bad_values = np.unique(session_ids[~integer_like_mask]).tolist()
        raise ValueError(
            "Multisubject GRU "
            f"{context} expected integer-valued session ids, got {bad_values}."
        )

    session_ids_int = rounded_session_ids.astype(int)
    valid_rows = subject_ids_int >= 0
    if np.any(valid_rows):
        valid_subject_ids_int = subject_ids_int[valid_rows]
        valid_session_ids_int = session_ids_int[valid_rows]
        subject_specific_max = session_max[valid_subject_ids_int]
        invalid_session_mask = np.logical_or(
            valid_session_ids_int < 1,
            valid_session_ids_int > subject_specific_max,
        )
        if np.any(invalid_session_mask):
            bad_values = np.unique(valid_session_ids_int[invalid_session_mask]).tolist()
            raise ValueError(
                "Multisubject GRU "
                f"{context} encountered out-of-range session ids {bad_values} for the "
                "resolved subject-specific session ranges."
            )

    invalid_padded_session_mask = np.logical_and(subject_ids_int < 0, session_ids_int != -1)
    if np.any(invalid_padded_session_mask):
        bad_values = np.unique(session_ids_int[invalid_padded_session_mask]).tolist()
        raise ValueError(
            "Multisubject GRU "
            f"{context} expected padded rows to use session id -1, got {bad_values}."
        )


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


# Class-index convention for the 3-way (ignore-included) head: 0=left, 1=right,
# 2=ignore. Matches the data loader (animal_response) and the L/R prob columns
# in gru_evaluation. For the 2-way head only classes 0/1 exist.
_IGNORE_CLASS_INDEX = 2


def _threeway_likelihood_decomposition(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    n_action_logits: int,
) -> dict[str, float]:
    """Decompose a 3-way (L/R/ignore) eval into engagement-relevant metrics.

    The raw 3-way normalized likelihood lives on a 1/3-chance scale and is NOT
    comparable to the 2-way study's L/R numbers (1/2-chance, engaged trials
    only). This returns two extra scalars that ARE interpretable:

    - ``eval_likelihood_LR_engaged``: geometric-mean likelihood of the correct
      side (left/right) on ENGAGED trials only, with the softmax renormalized
      over the {left, right} logits. Directly comparable to the 2-way L/R
      likelihood. Implemented by masking non-engaged labels to -1 (the
      convention ``normalized_likelihood`` uses for padding) and slicing the
      logits to the first two classes, so log_softmax renormalizes over L/R.
    - ``eval_likelihood_engage``: geometric-mean likelihood of the correct
      engage-vs-ignore decision, over ALL trials, by collapsing the head to a
      binary {engaged, ignore} problem (engaged logit = logsumexp of L,R).

    Returns an empty dict for a 2-way head (nothing to decompose).
    """
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    if n_action_logits < 3 or logits.shape[2] < 3:
        return {}

    lab = labels[..., 0].astype(np.int64)  # (T, E)
    valid = lab >= 0  # padding mask used by normalized_likelihood

    out: dict[str, float] = {}

    # --- Conditional L/R on engaged trials (comparable to the 2-way number) ---
    engaged = valid & (lab != _IGNORE_CLASS_INDEX)
    if bool(engaged.any()):
        lr_labels = np.where(engaged, lab, -1)[..., None]  # mask ignore -> -1
        lr_logits = logits[:, :, :2]  # renormalize softmax over {L, R}
        out["eval_likelihood_LR_engaged"] = float(
            rnn_utils.normalized_likelihood(lr_labels, lr_logits)
        )

    # --- Engage vs ignore (binary calibration of the ignore head) ---
    if bool(valid.any()):
        # Binary label: 0 = engaged (was L or R), 1 = ignore.
        engage_label = np.where(lab == _IGNORE_CLASS_INDEX, 1, 0)
        engage_label = np.where(valid, engage_label, -1)[..., None]
        # Binary logits: [engaged = logsumexp(L, R), ignore].
        lr_logits = logits[:, :, :2]
        engaged_logit = _logsumexp(lr_logits, axis=-1)  # (T, E)
        ignore_logit = logits[:, :, _IGNORE_CLASS_INDEX]  # (T, E)
        binary_logits = np.stack([engaged_logit, ignore_logit], axis=-1)
        out["eval_likelihood_engage"] = float(
            rnn_utils.normalized_likelihood(engage_label, binary_logits)
        )

        # --- Ignore-class classification metrics (rare positive: ~5% of trials) ---
        # Likelihood alone is dominated by the ~95% engaged majority and cannot
        # tell genuine ignore-detection from base-rate hedging. Report class-
        # resolved precision/recall/F1 (at p=0.5) and threshold-free average
        # precision (PR-AUC) with ignore as the positive class.
        vmask = valid.reshape(-1)
        y_true = (lab.reshape(-1)[vmask] == _IGNORE_CLASS_INDEX).astype(np.int64)
        # P(ignore) from the binary head: softmax over [engaged, ignore].
        bl = binary_logits.reshape(-1, 2)[vmask]
        bl_max = np.max(bl, axis=-1, keepdims=True)
        bexp = np.exp(bl - bl_max)
        p_ignore = (bexp[:, 1] / bexp.sum(axis=-1))
        n_pos = int(y_true.sum())
        n_tot = int(y_true.size)
        out["engage_ignore_base_rate"] = float(n_pos / n_tot) if n_tot else 0.0
        if n_pos > 0:
            pred = (p_ignore >= 0.5).astype(np.int64)
            tp = float(np.sum((pred == 1) & (y_true == 1)))
            fp = float(np.sum((pred == 1) & (y_true == 0)))
            fn = float(np.sum((pred == 0) & (y_true == 1)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            out["engage_ignore_precision"] = float(prec)
            out["engage_ignore_recall"] = float(rec)
            out["engage_ignore_f1"] = float(f1)
            # Average precision (area under PR curve), threshold-free. No-skill
            # baseline == base rate, so lift over base_rate is what matters.
            order = np.argsort(-p_ignore)
            yt = y_true[order]
            tps = np.cumsum(yt)
            fps = np.cumsum(1 - yt)
            precisions = tps / np.maximum(tps + fps, 1)
            recalls = tps / n_pos
            rec_prev = np.concatenate([[0.0], recalls[:-1]])
            out["engage_ignore_pr_auc"] = float(
                np.sum((recalls - rec_prev) * precisions)
            )

    return out


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return np.squeeze(out, axis=axis)


def _bind_session_curriculum_lambda(
    make_network: Any,
    *,
    session_curriculum_lambda: float,
) -> Any:
    return lambda: make_network(session_curriculum_lambda)


def _maybe_prepend_session_indices_to_datasets(
    *,
    dataset: Any,
    dataset_train: Any,
    dataset_eval: Any,
    session_conditioning_enabled: bool,
    metadata: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    if not session_conditioning_enabled:
        return dataset, dataset_train, dataset_eval
    return prepend_session_index_to_multisubject_split_datasets(
        dataset=dataset,
        dataset_train=dataset_train,
        dataset_eval=dataset_eval,
        metadata=metadata,
    )


def _log_heldout_dataset_details(heldout_data: dict[str, Any] | None) -> None:
    if not heldout_data:
        return

    xs_test = heldout_data.get("xs_test")
    ys_test = heldout_data.get("ys_test")
    if xs_test is None or ys_test is None:
        dataset_test = heldout_data.get("dataset_test")
        if dataset_test is not None:
            xs_test = getattr(dataset_test, "_xs", None)
            ys_test = getattr(dataset_test, "_ys", None)

    if xs_test is None or ys_test is None:
        return

    logger.info(
        "Held-out test shapes: input %s, output %s",
        np.asarray(xs_test).shape,
        np.asarray(ys_test).shape,
    )


def _log_dataset_split_details(
    *,
    dataset: Any,
    dataset_train: Any,
    dataset_eval: Any,
) -> None:
    logger.info(
        "Full dataset shapes: input %s, output %s",
        dataset._xs.shape,
        dataset._ys.shape,
    )
    logger.info(
        "Training split shapes: input %s, output %s",
        dataset_train._xs.shape,
        dataset_train._ys.shape,
    )
    logger.info(
        "Eval split shapes: input %s, output %s",
        dataset_eval._xs.shape,
        dataset_eval._ys.shape,
    )


class GruTrainer(BaseMultisubjectTrainer):
    """Trainer that mirrors the disRNN pipeline for GRU models."""

    _MODEL_LABEL = "GRU"
    _TRAINER_CONTEXT_NAME = "GruTrainer"

    def __init__(
        self,
        architecture: Mapping[str, Any] | DictConfig,
        training: Mapping[str, Any] | DictConfig,
        heldout_data: dict[str, Any] | None = None,
        output_dir: str = "/results/outputs",
        seed: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__(
            architecture=architecture,
            training=training,
            heldout_data=heldout_data,
            output_dir=output_dir,
            seed=seed,
        )












    # Cap eval_network episodes per call so a wide GRU (e.g. hidden_size=256)
    # over a large multisubject cohort (~18k sessions) doesn't OOM the GPU; the
    # full-cohort forward pass allocated ~39.5 GB and exceeded a 48 GB L40s.
    # See BaseMultisubjectTrainer._eval_network_full.
    _eval_max_episodes = 4096

    def _resolve_n_action_logits(self, dataset, yhat, *, context):
        return _require_n_action_logits(dataset, yhat, context=context)

    def _add_model_results(self, raw_df, network_states, yhat, *, ignore_policy):
        return add_gru_model_results(
            raw_df, network_states, np.asarray(yhat), ignore_policy=ignore_policy
        )

    def _evaluate_heldout_subjects(
        self, *, heldout_eval_cfg, wandb_run, params_path, output_subdir, heldout_data, log_scope
    ):
        return evaluate_gru_on_heldout_subjects(
            heldout_eval_cfg,
            wandb_run=wandb_run,
            params_path=params_path,
            output_subdir=output_subdir,
            log_to_wandb=False,
            heldout_data=heldout_data,
            log_scope=log_scope,
        )

    def _plot_model_specific_diagnostics(
        self,
        *,
        params: Any,
        disrnn_config: Any,
        stage_dir: Path,
        stage_name: str,
        is_multisubject: bool,
        metadata: dict[str, Any],
        raw_df: Any,
        session_curriculum_lambda: float,
        wandb_run: Any | None,
        wandb_step: int | None,
        keep_media_files: bool,
    ) -> dict[str, Any]:
        plot_paths: dict[str, Any] = {
            "subject_embedding_state_space": None,
            "subject_session_context_state_space": None,
        }
        if is_multisubject:
            try:
                subject_embedding_fig = self._plot_subject_embedding_state_space(
                    params=params,
                    raw_df=raw_df,
                    metadata=metadata,
                )
                if subject_embedding_fig is not None:
                    subject_embedding_path = stage_dir / "subject_embedding_state_space.png"
                    subject_embedding_fig.savefig(subject_embedding_path)
                    plt.close(subject_embedding_fig)
                    plot_paths["subject_embedding_state_space"] = str(subject_embedding_path)
                subject_session_context_fig = (
                    self._plot_subject_session_context_state_space(
                        params=params,
                        raw_df=raw_df,
                        metadata=metadata,
                        session_curriculum_lambda=session_curriculum_lambda,
                    )
                )
                if subject_session_context_fig is not None:
                    subject_session_context_path = (
                        stage_dir / "subject_session_context_state_space.png"
                    )
                    subject_session_context_fig.savefig(subject_session_context_path)
                    plt.close(subject_session_context_fig)
                    plot_paths["subject_session_context_state_space"] = str(
                        subject_session_context_path
                    )
            except Exception as exc:
                logger.warning(
                    "Initialization subject-embedding plotting failed for %s: %s",
                    stage_name,
                    exc,
                )

        if wandb_run is not None and (
            plot_paths["subject_embedding_state_space"]
            or plot_paths["subject_session_context_state_space"]
        ):
            plot_payload = {}
            if plot_paths["subject_embedding_state_space"]:
                plot_payload["checkpoint/fig/subject_embedding_state_space"] = wandb.Image(
                    str(plot_paths["subject_embedding_state_space"])
                )
            if plot_paths["subject_session_context_state_space"]:
                plot_payload["checkpoint/fig/subject_session_context_state_space"] = (
                    wandb.Image(str(plot_paths["subject_session_context_state_space"]))
                )
            if wandb_step is None:
                wandb_run.log(plot_payload)
            else:
                wandb_run.log(plot_payload, step=wandb_step)
            if not keep_media_files:
                self._remove_media_files(
                    [
                        plot_paths["subject_embedding_state_space"],
                        plot_paths["subject_session_context_state_space"],
                    ]
                )
                plot_paths["subject_embedding_state_space"] = None
                plot_paths["subject_session_context_state_space"] = None
        return plot_paths

    def fit(
        self,
        bundle: DatasetBundle,
        loggers: dict[str, Any] | None = None,
    ):
        metadata = dict(bundle.metadata)
        ignore_policy = metadata.get("ignore_policy", "exclude")
        is_multisubject = _is_multisubject_mode(self.architecture, metadata)

        wandb_run = None
        if loggers and "wandb" in loggers:
            wandb_run = loggers["wandb"]

        dataset = bundle.extras.get("dataset") if bundle.extras else None
        if dataset is None:
            raise ValueError("Dataset bundle must include the constructed RNN dataset.")

        dataset_train = bundle.train_set
        dataset_eval = bundle.eval_set
        if dataset_train is None or dataset_eval is None:
            raise ValueError("Dataset bundle must include train and eval splits.")

        output = {
            "num_trials": metadata.get("num_trials"),
            "num_sessions": metadata.get("num_sessions"),
        }
        output["multisubject"] = bool(is_multisubject)

        key = jax.random.PRNGKey(self.seed)
        output["random_key"] = [int(x) for x in np.asarray(key).reshape(-1)]

        max_n_subjects: int | None = None
        subject_embedding_size: int | None = None
        subject_embedding_init = str(self.architecture.get("subject_embedding_init", "zeros"))
        session_conditioning_cfg = {
            "enabled": False,
            "session_encoding_type": "none",
            "session_integration_type": str(
                self.architecture.get("session_integration_type", "direct")
            ),
            "session_fourier_k": int(self.architecture.get("session_fourier_k", 4)),
            "session_delta_n_layers": int(self.architecture.get("session_delta_n_layers", 3)),
            "session_delta_hidden_size": int(
                self.architecture.get("session_delta_hidden_size", 16)
            ),
            "session_max_index_by_subject_index": (),
        }
        if is_multisubject:
            subject_embedding_size = self.architecture.get("subject_embedding_size")
            if subject_embedding_size is None or int(subject_embedding_size) <= 0:
                raise ValueError(
                    "Multisubject GRU requires architecture.subject_embedding_size > 0."
                )
            max_n_subjects = _resolve_multisubject_subject_count(metadata)
            if not isinstance(metadata.get("subject_id_to_index"), dict) or not isinstance(
                metadata.get("index_to_subject_id"),
                dict,
            ):
                raise ValueError(
                    "Multisubject GRU requires subject_id_to_index and "
                    "index_to_subject_id in bundle metadata."
                )
            session_conditioning_cfg = resolve_session_conditioning_from_architecture(
                architecture=self.architecture,
                metadata=metadata,
                multisubject=is_multisubject,
                max_n_subjects=max_n_subjects,
                subject_embedding_size=int(subject_embedding_size),
                context="GruTrainer",
            )
            dataset, dataset_train, dataset_eval = _maybe_prepend_session_indices_to_datasets(
                dataset=dataset,
                dataset_train=dataset_train,
                dataset_eval=dataset_eval,
                session_conditioning_enabled=bool(session_conditioning_cfg["enabled"]),
                metadata=metadata,
            )
            _validate_multisubject_dataset_inputs(
                dataset,
                max_n_subjects=max_n_subjects,
                context="full dataset",
                session_conditioning_enabled=bool(session_conditioning_cfg["enabled"]),
                session_max_index_by_subject_index=session_conditioning_cfg[
                    "session_max_index_by_subject_index"
                ],
            )
            _validate_multisubject_dataset_inputs(
                dataset_train,
                max_n_subjects=max_n_subjects,
                context="training split",
                session_conditioning_enabled=bool(session_conditioning_cfg["enabled"]),
                session_max_index_by_subject_index=session_conditioning_cfg[
                    "session_max_index_by_subject_index"
                ],
            )
            _validate_multisubject_dataset_inputs(
                dataset_eval,
                max_n_subjects=max_n_subjects,
                context="eval split",
                session_conditioning_enabled=bool(session_conditioning_cfg["enabled"]),
                session_max_index_by_subject_index=session_conditioning_cfg[
                    "session_max_index_by_subject_index"
                ],
            )
        else:
            session_conditioning_cfg = resolve_session_conditioning_from_architecture(
                architecture=self.architecture,
                metadata=metadata,
                multisubject=is_multisubject,
                max_n_subjects=max_n_subjects,
                subject_embedding_size=None,
                context="GruTrainer",
            )

        total_session_curriculum_steps = int(self.training.get("n_steps", 0))
        session_curriculum_cfg = resolve_session_curriculum_steps(
            total_training_steps=total_session_curriculum_steps,
            session_n_pretrain_steps=self.architecture.get("session_n_pretrain_steps"),
            session_n_warmup_steps=self.architecture.get("session_n_warmup_steps"),
            context="GruTrainer",
        )
        self.architecture["session_n_pretrain_steps"] = int(
            session_curriculum_cfg["session_n_pretrain_steps"]
        )
        self.architecture["session_n_warmup_steps"] = int(
            session_curriculum_cfg["session_n_warmup_steps"]
        )
        output["session_curriculum"] = {
            **session_curriculum_cfg,
            "total_training_steps": int(total_session_curriculum_steps),
        }

        _log_dataset_split_details(
            dataset=dataset,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
        )
        _log_heldout_dataset_details(self.heldout_data)
        if is_multisubject:
            self._log_session_conditioning_details(
                metadata=metadata,
                dataset=dataset,
                session_conditioning_cfg=session_conditioning_cfg,
                session_curriculum_cfg=session_curriculum_cfg,
                total_training_steps=total_session_curriculum_steps,
            )
            if bool(session_conditioning_cfg["enabled"]) and bool(
                self.training.get("plot_session_context_state_space", True)
            ):
                self._resolve_session_context_plot_subject_indices(metadata=metadata)

        expected_output_size = 2 if ignore_policy == "exclude" else 3
        args = types.SimpleNamespace(
            hidden_size=int(self.architecture["hidden_size"]),
            num_layers=int(self.architecture.get("num_layers", 1)),
            output_size=int(self.architecture.get("output_size", expected_output_size)),
            max_grad_norm=self.training["max_grad_norm"],
            n_steps=int(self.training["n_steps"]),
            learning_rate=self.training["lr"],
            loss=self.training["loss"],
            loss_param=self.training["loss_param"],
            checkpoint_every_n_steps=int(self.training.get("checkpoint_every_n_steps", 0)),
            auto_resume=bool(self.training.get("auto_resume", True)),
            checkpoint_eval_on_eval_split=bool(
                self.training.get("checkpoint_eval_on_eval_split", True)
            ),
            checkpoint_eval_on_train_split=bool(
                self.training.get("checkpoint_eval_on_train_split", True)
            ),
            checkpoint_log_eval_to_wandb=bool(
                self.training.get("checkpoint_log_eval_to_wandb", True)
            ),
            checkpoint_log_train_to_wandb=bool(
                self.training.get("checkpoint_log_train_to_wandb", True)
            ),
            checkpoint_plot_split_examples_every_n=int(
                self.training.get("checkpoint_plot_split_examples_every_n", 5)
            ),
            checkpoint_save_output_df_every_n=int(
                self.training.get("checkpoint_save_output_df_every_n", 0)
            ),
            checkpoint_log_split_examples_to_wandb=bool(
                self.training.get("checkpoint_log_split_examples_to_wandb", True)
            ),
            checkpoint_keep_media_files=bool(
                self.training.get("checkpoint_keep_media_files", True)
            ),
            checkpoint_run_heldout_eval=bool(
                self.training.get("checkpoint_run_heldout_eval", True)
            ),
            checkpoint_include_final_in_heldout=bool(
                self.training.get("checkpoint_include_final_in_heldout", False)
            ),
            initialization_eval_before_training=bool(
                self.training.get("initialization_eval_before_training", True)
            ),
            save_output_df=bool(self.training.get("save_output_df", False)),
        )

        if args.num_layers != 1:
            raise NotImplementedError(
                "GRU integration currently supports architecture.num_layers == 1 only."
            )
        if args.output_size != expected_output_size:
            raise ValueError(
                "Configured GRU output_size does not match dataset ignore_policy: "
                f"configured={args.output_size} expected={expected_output_size}"
            )
        if args.n_steps < 0:
            raise ValueError("training.n_steps must be >= 0")
        if args.checkpoint_every_n_steps < 0:
            raise ValueError("training.checkpoint_every_n_steps must be >= 0")
        if args.checkpoint_plot_split_examples_every_n < 0:
            raise ValueError("training.checkpoint_plot_split_examples_every_n must be >= 0")
        if args.checkpoint_save_output_df_every_n < 0:
            raise ValueError("training.checkpoint_save_output_df_every_n must be >= 0")

        freeze_gru_core = bool(self.training.get("freeze_gru_core", False))
        if freeze_gru_core and not is_multisubject:
            raise ValueError(
                "training.freeze_gru_core requires a multisubject GRU so subject "
                "embeddings remain trainable."
            )

        logger.info("max_grad_norm = %s", args.max_grad_norm)

        make_network = make_gru_network(
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            multisubject=is_multisubject,
            max_n_subjects=max_n_subjects,
            subject_embedding_size=(
                int(subject_embedding_size) if subject_embedding_size is not None else None
            ),
            subject_embedding_init=subject_embedding_init,
            session_encoding_type=str(session_conditioning_cfg["session_encoding_type"]),
            session_integration_type=str(session_conditioning_cfg["session_integration_type"]),
            session_fourier_k=int(session_conditioning_cfg["session_fourier_k"]),
            session_delta_n_layers=int(session_conditioning_cfg["session_delta_n_layers"]),
            session_delta_hidden_size=int(
                session_conditioning_cfg["session_delta_hidden_size"]
            ),
            session_max_index_by_subject_index=tuple(
                int(value)
                for value in session_conditioning_cfg["session_max_index_by_subject_index"]
            ),
        )

        def _session_curriculum_lambda_for_step(current_step: int) -> float:
            if not bool(session_conditioning_cfg["enabled"]):
                return 1.0
            return compute_session_curriculum_lambda(
                current_step=current_step,
                session_n_pretrain_steps=int(
                    session_curriculum_cfg["session_n_pretrain_steps"]
                ),
                session_n_warmup_steps=int(
                    session_curriculum_cfg["session_n_warmup_steps"]
                ),
            )

        def _make_eval_network_for_step(current_step: int) -> Any:
            return _bind_session_curriculum_lambda(
                make_network,
                session_curriculum_lambda=_session_curriculum_lambda_for_step(
                    current_step
                ),
            )

        lambda_reg_session = float(self.training.get("lambda_reg_session", 0.0))
        if lambda_reg_session < 0:
            raise ValueError("training.lambda_reg_session must be >= 0.")
        if lambda_reg_session > 0 and not bool(session_conditioning_cfg["enabled"]):
            logger.warning(
                "training.lambda_reg_session=%s but session conditioning is disabled "
                "(session_encoding_type=none); skipping session-delta regularization.",
                lambda_reg_session,
            )
            lambda_reg_session = 0.0
        session_regularization_apply = None
        if lambda_reg_session > 0:
            session_context = metadata.get("session_context")
            if not isinstance(session_context, Mapping):
                raise ValueError(
                    "training.lambda_reg_session requires metadata.session_context."
                )
            if max_n_subjects is None:
                raise ValueError(
                    "training.lambda_reg_session requires multisubject metadata.num_subjects."
                )
            reg_subject_indices, reg_session_indices = (
                session_regularization_index_arrays_from_session_context(session_context)
            )
            session_regularization_apply = build_zero_mean_session_delta_regularization_apply(
                make_network=make_network,
                subject_indices=reg_subject_indices,
                session_indices=reg_session_indices,
                max_n_subjects=int(max_n_subjects),
            )
            logger.info(
                "Zero-mean session-delta regularization enabled: lambda_reg_session=%s "
                "over %d subject-session pairs.",
                lambda_reg_session,
                int(reg_subject_indices.shape[0]),
            )

        parameter_update_mask = None

        def _train_network_with_optional_session_regularization(
            *,
            params: Any | None = None,
            opt_state: Any | None = None,
            n_steps: int,
            random_key: Any,
            optimizer: Any,
            wandb_step_offset: int = 0,
            total_step_offset: int = 0,
        ):
            return train_network_with_session_regularization(
                make_network,
                dataset_train,
                dataset_eval,
                loss=args.loss,
                loss_param=args.loss_param,
                n_action_logits=int(args.output_size),
                session_regularization_apply=session_regularization_apply,
                session_regularization_scale=lambda_reg_session,
                session_curriculum_lambda_schedule=(
                    lambda local_step: _session_curriculum_lambda_for_step(
                        total_step_offset + local_step
                    )
                ),
                params=params,
                opt_state=opt_state,
                opt=optimizer,
                parameter_update_mask=parameter_update_mask,
                n_steps=n_steps,
                max_grad_norm=args.max_grad_norm,
                random_key=random_key,
                report_progress_by="wandb",
                wandb_run=wandb_run,
                wandb_step_offset=wandb_step_offset,
            )

        heldout_eval_cfg = None
        heldout_test_data = self.heldout_data
        runtime_heldout_cfg = HeldoutEvalConfig.from_data_cfg(metadata)
        if runtime_heldout_cfg.enabled and is_multisubject:
            logger.info(
                "Skipping PER-CHECKPOINT held-out eval (checkpoint_run_heldout_eval) "
                "for multisubject GRU; v1 supports seen-subject personalization only. "
                "NOTE: this does NOT disable the end-of-training held-out fine-tune "
                "(auto_heldout_finetune) — that runs separately and logs heldout/* metrics."
            )
        elif runtime_heldout_cfg.enabled:
            heldout_eval_cfg = OmegaConf.create(
                {
                    "data": asdict(runtime_heldout_cfg),
                    "model": {
                        "architecture": self.architecture,
                        "output_dir": str(self.output_dir),
                    },
                }
            )
            if heldout_test_data is None:
                try:
                    heldout_test_data = load_gru_heldout_subject_data(runtime_heldout_cfg)
                    self.heldout_data = heldout_test_data
                    _log_heldout_dataset_details(heldout_test_data)
                except Exception as exc:
                    logger.warning(
                        "Preloading held-out test data failed; evaluation will fall back to lazy loading: %s",
                        exc,
                    )

        start = time.time()
        checkpoint_records: list[dict[str, Any]] = []
        checkpoint_heldout_summaries: list[dict[str, Any]] = []

        params, init_opt_state, _ = _train_network_with_optional_session_regularization(
            n_steps=0,
            random_key=key,
            optimizer=optax.adam(args.learning_rate),
            total_step_offset=0,
        )
        if freeze_gru_core:
            parameter_update_mask, mask_summary = _build_frozen_gru_core_update_mask(params)
            output["parameter_training"] = {
                "freeze_gru_core": True,
                **mask_summary,
            }
            logger.info(
                "Frozen random GRU reservoir enabled: %d trainable parameters "
                "(subject embeddings + readout), %d frozen parameters.",
                mask_summary["trainable_parameter_count"],
                mask_summary["frozen_parameter_count"],
            )

        # --- Resume from latest full-state checkpoint (preemption recovery) ---
        # Only the chunked checkpoint path writes resumable state, so resume is
        # scoped to it. When a checkpoint is found we override the fresh init
        # below and skip the one-time initialization eval.
        resume_state = None
        if args.auto_resume and args.checkpoint_every_n_steps > 0:
            resume_state = find_latest_resumable_state(self.output_dir / "checkpoints")
            if resume_state is not None:
                logger.info(
                    "Resuming GRU training from step %s", resume_state.steps_completed
                )

        if args.initialization_eval_before_training and resume_state is None:
            output["initial_evaluations"] = {
                "before_training": self._evaluate_initialization_snapshot(
                    stage_name="before_training",
                    params=params,
                    bundle=bundle,
                    metadata=metadata,
                    dataset=dataset,
                    dataset_train=dataset_train,
                    dataset_eval=dataset_eval,
                    ignore_policy=ignore_policy,
                    make_eval_network=_make_eval_network_for_step(0),
                    is_multisubject=is_multisubject,
                    heldout_eval_cfg=heldout_eval_cfg,
                    heldout_data=heldout_test_data,
                    wandb_run=wandb_run,
                    wandb_step=0,
                    keep_media_files=args.checkpoint_keep_media_files,
                    session_curriculum_lambda=_session_curriculum_lambda_for_step(0),
                )
            }

        if args.checkpoint_every_n_steps == 0:
            params, opt_state, losses = _train_network_with_optional_session_regularization(
                params=params,
                opt_state=init_opt_state,
                n_steps=args.n_steps,
                random_key=key,
                optimizer=optax.adam(args.learning_rate),
                total_step_offset=0,
            )
        else:
            optimizer = optax.adam(args.learning_rate)
            checkpoint_root = self.output_dir / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)

            if resume_state is not None:
                params = resume_state.params
                opt_state = resume_state.opt_state
                random_key = resume_state.random_key
                steps_completed = resume_state.steps_completed
                all_training_losses = list(resume_state.training_losses)
                all_validation_losses = list(resume_state.validation_losses)
            else:
                opt_state = init_opt_state
                random_key = key
                steps_completed = 0
                all_training_losses = []
                all_validation_losses = []
            _train_all = dataset_train.get_all()
            xs_train_all, ys_train_all = _train_all["xs"], _train_all["ys"]
            _eval_all = dataset_eval.get_all()
            xs_eval_all, ys_eval_all = _eval_all["xs"], _eval_all["ys"]
            xs_full_for_checkpoint = dataset.get_all()["xs"]
            df_for_checkpoint = bundle.raw

            # --- Length-bucketed batching (opt-in via training.length_bucketing) ---
            # Trim the per-batch unroll to the batch's session length instead of
            # the global T_max, cutting padding compute. Set on the train dataset
            # that session-regularized training samples from.
            if bool(self.training.get("length_bucketing", False)):
                dataset_train.length_bucketing = True
                dataset_train.length_bucket_grid = int(
                    self.training.get("length_bucket_grid", 128)
                )
                logger.info(
                    "Length-bucketed batching enabled (grid=%s)",
                    dataset_train.length_bucket_grid,
                )

            # --- Early stopping (opt-in via training.early_stopping) ---
            # On trigger we `break` the loop, so finalization (gru_config.json,
            # subject_index_map, checkpoints/index.json, auto held-out fine-tune)
            # still runs and the best_eval checkpoint is used downstream.
            _es_cfg = self.training.get("early_stopping", {}) or {}
            _es_enabled = bool(_es_cfg.get("enabled", False))
            _es_metric = str(_es_cfg.get("metric", "eval_likelihood"))
            _es_min_delta = float(_es_cfg.get("min_delta", 0.0))
            _es_patience = int(_es_cfg.get("patience", 2))
            _es_guard = _es_cfg.get("overfit_guard", None)
            _es_guard = float(_es_guard) if _es_guard is not None else None
            # Only arm the check once step >= start_after_step (default 0 = no gating).
            # Used to let the session-conditioning curriculum reach lambda=1 and train a
            # while before early stopping can fire (otherwise it stops in pretrain/warm-up).
            _es_start_after = int(_es_cfg.get("start_after_step", 0) or 0)
            _es_best = float("-inf")
            _es_stale = 0
            if _es_enabled:
                logger.info(
                    "Early stopping enabled: metric=%s min_delta=%s patience=%s "
                    "overfit_guard=%s start_after_step=%s",
                    _es_metric, _es_min_delta, _es_patience, _es_guard, _es_start_after,
                )

            while steps_completed < args.n_steps:
                chunk_steps = min(
                    args.checkpoint_every_n_steps,
                    args.n_steps - steps_completed,
                )
                params, opt_state, chunk_losses = _train_network_with_optional_session_regularization(
                    params=params,
                    opt_state=opt_state,
                    n_steps=chunk_steps,
                    random_key=random_key,
                    optimizer=optimizer,
                    wandb_step_offset=steps_completed,
                    total_step_offset=steps_completed,
                )

                all_training_losses.extend(np.asarray(chunk_losses["training_loss"]).tolist())
                all_validation_losses.extend(np.asarray(chunk_losses["validation_loss"]).tolist())

                random_key = jax.random.split(random_key, 2)[0]
                steps_completed += chunk_steps
                current_session_curriculum_lambda = _session_curriculum_lambda_for_step(
                    steps_completed
                )
                current_eval_network = _make_eval_network_for_step(steps_completed)

                checkpoint_dir = checkpoint_root / f"step_{steps_completed}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_params_path = checkpoint_dir / "params.json"
                with checkpoint_params_path.open("w") as f:
                    f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))

                # Persist the full training state (params + optimizer + PRNG key +
                # step + loss history) so a preempted job can resume from here.
                # random_key/steps_completed are already advanced for the *next*
                # chunk, so the restored state continues seamlessly.
                save_train_state(
                    checkpoint_dir,
                    steps_completed=steps_completed,
                    params=params,
                    opt_state=opt_state,
                    random_key=random_key,
                    training_losses=all_training_losses,
                    validation_losses=all_validation_losses,
                )

                train_likelihood_ckpt: float | None = None
                if args.checkpoint_eval_on_train_split:
                    yhat_train_ckpt, _ = self._eval_network_full(
                        current_eval_network,
                        params,
                        xs_train_all,
                    )
                    n_action_logits_train_ckpt = _require_n_action_logits(
                        dataset_train,
                        np.asarray(yhat_train_ckpt),
                        context="checkpoint train",
                    )
                    train_likelihood_ckpt = float(
                        rnn_utils.normalized_likelihood(
                            ys_train_all,
                            np.asarray(yhat_train_ckpt)[:, :, :n_action_logits_train_ckpt],
                        )
                    )

                eval_likelihood_ckpt: float | None = None
                eval_likelihood_decomp_ckpt: dict[str, float] = {}
                if args.checkpoint_eval_on_eval_split:
                    yhat_eval_ckpt, _ = self._eval_network_full(
                        current_eval_network,
                        params,
                        xs_eval_all,
                    )
                    n_action_logits_ckpt = _require_n_action_logits(
                        dataset_eval,
                        np.asarray(yhat_eval_ckpt),
                        context="checkpoint eval",
                    )
                    eval_likelihood_ckpt = float(
                        rnn_utils.normalized_likelihood(
                            ys_eval_all,
                            np.asarray(yhat_eval_ckpt)[:, :, :n_action_logits_ckpt],
                        )
                    )
                    # For the 3-way (ignore-included) head, also record the
                    # engagement-conditional L/R likelihood (comparable to the
                    # 2-way study) and the engage-vs-ignore calibration.
                    eval_likelihood_decomp_ckpt = _threeway_likelihood_decomposition(
                        ys_eval_all,
                        np.asarray(yhat_eval_ckpt)[:, :, :n_action_logits_ckpt],
                        n_action_logits=n_action_logits_ckpt,
                    )

                checkpoint_record = {
                    "step": int(steps_completed),
                    "params_path": str(checkpoint_params_path),
                    "session_curriculum_lambda": float(current_session_curriculum_lambda),
                }
                if train_likelihood_ckpt is not None:
                    checkpoint_record["train_likelihood"] = train_likelihood_ckpt
                    logger.info(
                        "Checkpoint step %s, training likelihood: %.4f",
                        steps_completed,
                        train_likelihood_ckpt,
                    )
                if eval_likelihood_ckpt is not None:
                    checkpoint_record["eval_likelihood"] = eval_likelihood_ckpt
                    logger.info(
                        "Checkpoint step %s, eval likelihood: %.4f",
                        steps_completed,
                        eval_likelihood_ckpt,
                    )
                for _decomp_key, _decomp_val in eval_likelihood_decomp_ckpt.items():
                    checkpoint_record[_decomp_key] = _decomp_val
                    logger.info(
                        "Checkpoint step %s, %s: %.4f",
                        steps_completed,
                        _decomp_key,
                        _decomp_val,
                    )

                checkpoint_plot_paths: dict[str, Any] = {
                    "subject_embedding_state_space": None,
                    "subject_session_context_state_space": None,
                }
                if is_multisubject:
                    try:
                        subject_embedding_fig_ckpt = self._plot_subject_embedding_state_space(
                            params=params,
                            raw_df=bundle.raw,
                            metadata=metadata,
                        )
                        if subject_embedding_fig_ckpt is not None:
                            subject_embedding_path_ckpt = (
                                checkpoint_dir / "subject_embedding_state_space.png"
                            )
                            subject_embedding_fig_ckpt.savefig(subject_embedding_path_ckpt)
                            plt.close(subject_embedding_fig_ckpt)
                            checkpoint_plot_paths["subject_embedding_state_space"] = str(
                                subject_embedding_path_ckpt
                            )
                        subject_session_context_fig_ckpt = (
                            self._plot_subject_session_context_state_space(
                                params=params,
                                raw_df=bundle.raw,
                                metadata=metadata,
                                session_curriculum_lambda=current_session_curriculum_lambda,
                            )
                        )
                        if subject_session_context_fig_ckpt is not None:
                            subject_session_context_path_ckpt = (
                                checkpoint_dir / "subject_session_context_state_space.png"
                            )
                            subject_session_context_fig_ckpt.savefig(
                                subject_session_context_path_ckpt
                            )
                            plt.close(subject_session_context_fig_ckpt)
                            checkpoint_plot_paths["subject_session_context_state_space"] = str(
                                subject_session_context_path_ckpt
                            )
                    except Exception as exc:
                        logger.warning(
                            "Checkpoint subject-embedding plotting failed for step=%s: %s",
                            steps_completed,
                            exc,
                        )

                checkpoint_record["plot_paths"] = checkpoint_plot_paths

                should_plot_split_examples_ckpt = (
                    args.checkpoint_plot_split_examples_every_n > 0
                    and (
                        steps_completed % args.checkpoint_plot_split_examples_every_n == 0
                        or steps_completed == args.n_steps
                    )
                )
                should_save_output_df_ckpt = (
                    args.checkpoint_save_output_df_every_n > 0
                    and (
                        steps_completed % args.checkpoint_save_output_df_every_n == 0
                        or steps_completed == args.n_steps
                    )
                )
                if should_plot_split_examples_ckpt or should_save_output_df_ckpt:
                    yhat_full_ckpt, network_states_full_ckpt = self._eval_network_full(
                        current_eval_network,
                        params,
                        xs_full_for_checkpoint,
                    )
                    # Only build the whole-cohort frame when it is persisted;
                    # plotting builds per-subject frames on demand below.
                    output_df_ckpt = None
                    if should_save_output_df_ckpt:
                        output_df_ckpt = add_gru_model_results(
                            df_for_checkpoint.copy(),
                            np.asarray(network_states_full_ckpt),
                            np.asarray(yhat_full_ckpt),
                            ignore_policy=ignore_policy,
                        )
                        output_df_ckpt_path = checkpoint_dir / "output_df.csv"
                        output_df_ckpt.to_csv(output_df_ckpt_path, index=False)
                        checkpoint_record["output_df_path"] = str(output_df_ckpt_path)

                    if should_plot_split_examples_ckpt:
                        n_action_logits_ckpt_full = _require_n_action_logits(
                            dataset,
                            np.asarray(yhat_full_ckpt),
                            context="checkpoint full-dataset plotting",
                        )
                        split_summaries_ckpt = self._generate_split_examples(
                            output_dir=checkpoint_dir,
                            output_df=output_df_ckpt,
                            raw_df=df_for_checkpoint,
                            ignore_policy=ignore_policy,
                            network_states_full=np.asarray(network_states_full_ckpt),
                            yhat_full=np.asarray(yhat_full_ckpt),
                            params=params,
                            metadata=metadata,
                            n_action_logits=n_action_logits_ckpt_full,
                            wandb_run=(
                                wandb_run
                                if args.checkpoint_log_split_examples_to_wandb
                                else None
                            ),
                            log_scope=f"Checkpoint step {steps_completed}",
                            wandb_step=int(steps_completed),
                            wandb_key_prefix="checkpoint",
                        )
                        checkpoint_record["split_examples"] = split_summaries_ckpt

                checkpoint_records.append(checkpoint_record)

                if (
                    wandb_run is not None
                    and args.checkpoint_log_train_to_wandb
                    and train_likelihood_ckpt is not None
                ):
                    wandb_run.log(
                        {
                            "checkpoint/train_likelihood": train_likelihood_ckpt,
                            "checkpoint/step": int(steps_completed),
                        },
                        step=int(steps_completed),
                    )

                if (
                    wandb_run is not None
                    and eval_likelihood_ckpt is not None
                    and args.checkpoint_log_eval_to_wandb
                ):
                    wandb_run.log(
                        {
                            "checkpoint/eval_likelihood": eval_likelihood_ckpt,
                            "checkpoint/step": int(steps_completed),
                            **{
                                f"checkpoint/{_k}": _v
                                for _k, _v in eval_likelihood_decomp_ckpt.items()
                            },
                        },
                        step=int(steps_completed),
                    )

                if (
                    wandb_run is not None
                    and args.checkpoint_log_eval_to_wandb
                    and (
                        checkpoint_plot_paths["subject_embedding_state_space"]
                        or checkpoint_plot_paths["subject_session_context_state_space"]
                    )
                ):
                    checkpoint_plot_payload = {}
                    if checkpoint_plot_paths["subject_embedding_state_space"]:
                        checkpoint_plot_payload[
                            "checkpoint/fig/subject_embedding_state_space"
                        ] = wandb.Image(
                            str(checkpoint_plot_paths["subject_embedding_state_space"])
                        )
                    if checkpoint_plot_paths["subject_session_context_state_space"]:
                        checkpoint_plot_payload[
                            "checkpoint/fig/subject_session_context_state_space"
                        ] = wandb.Image(
                            str(
                                checkpoint_plot_paths[
                                    "subject_session_context_state_space"
                                ]
                            )
                        )
                    wandb_run.log(
                        checkpoint_plot_payload,
                        step=int(steps_completed),
                    )
                    if not args.checkpoint_keep_media_files:
                        self._remove_media_files(
                            [
                                checkpoint_plot_paths["subject_embedding_state_space"],
                                checkpoint_plot_paths[
                                    "subject_session_context_state_space"
                                ],
                            ]
                        )
                        checkpoint_plot_paths["subject_embedding_state_space"] = None
                        checkpoint_plot_paths["subject_session_context_state_space"] = None

                if (
                    wandb_run is not None
                    and args.checkpoint_log_split_examples_to_wandb
                    and not args.checkpoint_keep_media_files
                    and "split_examples" in checkpoint_record
                ):
                    split_examples = checkpoint_record.get("split_examples", {})
                    for split_summary in split_examples.values():
                        if not isinstance(split_summary, dict):
                            continue
                        plots = split_summary.get("plots", {})
                        if not isinstance(plots, dict):
                            continue
                        for key_name in (
                            "latents_over_trials_examples",
                            "latents_in_space_examples",
                        ):
                            raw_paths = plots.get(key_name, [])
                            if not isinstance(raw_paths, list):
                                continue
                            for raw_path in raw_paths:
                                try:
                                    Path(str(raw_path)).unlink(missing_ok=True)
                                except Exception as exc:
                                    logger.warning(
                                        "Failed to remove checkpoint split-example media file %s: %s",
                                        raw_path,
                                        exc,
                                    )
                            plots[key_name] = []

                should_run_checkpoint_heldout = (
                    heldout_eval_cfg is not None
                    and args.checkpoint_run_heldout_eval
                    and (
                        int(steps_completed) < int(args.n_steps)
                        or (
                            args.checkpoint_include_final_in_heldout
                            and int(steps_completed) == int(args.n_steps)
                        )
                    )
                )
                if should_run_checkpoint_heldout:
                    try:
                        ckpt_summary = evaluate_gru_on_heldout_subjects(
                            heldout_eval_cfg,
                            wandb_run=wandb_run,
                            params_path=checkpoint_params_path,
                            output_subdir=f"heldout_test/checkpoints/step_{steps_completed}",
                            log_to_wandb=False,
                            heldout_data=heldout_test_data,
                            log_scope=f"Checkpoint step {steps_completed}",
                        )
                        if ckpt_summary is not None:
                            checkpoint_heldout_summaries.append(
                                {
                                    "step": int(steps_completed),
                                    "params_path": str(checkpoint_params_path),
                                    "heldout_test_likelihood": float(
                                        ckpt_summary["test_likelihood"]
                                    ),
                                }
                            )
                            if wandb_run is not None:
                                wandb_step = int(steps_completed)
                                wandb_run.log(
                                    {
                                        "checkpoint/heldout_test_likelihood": float(
                                            ckpt_summary["test_likelihood"]
                                        ),
                                        "checkpoint/step": int(steps_completed),
                                    },
                                    step=wandb_step,
                                )
                                try:
                                    trial_plot_paths = ckpt_summary.get("plots", {}).get(
                                        "latents_over_trials_examples", []
                                    )
                                    space_plot_paths = ckpt_summary.get("plots", {}).get(
                                        "latents_in_space_examples", []
                                    )
                                    checkpoint_plot_payload = {}
                                    if trial_plot_paths:
                                        checkpoint_plot_payload[
                                            "checkpoint/heldout/latents_over_trials_examples"
                                        ] = [
                                            wandb.Image(str(path)) for path in trial_plot_paths
                                        ]
                                    if space_plot_paths:
                                        checkpoint_plot_payload[
                                            "checkpoint/heldout/latents_in_space_examples"
                                        ] = [
                                            wandb.Image(str(path)) for path in space_plot_paths
                                        ]
                                    if checkpoint_plot_payload:
                                        wandb_run.log(checkpoint_plot_payload, step=wandb_step)
                                        if not args.checkpoint_keep_media_files:
                                            for path in trial_plot_paths + space_plot_paths:
                                                try:
                                                    Path(str(path)).unlink(missing_ok=True)
                                                except Exception as path_exc:
                                                    logger.warning(
                                                        "Failed to remove checkpoint held-out media file %s: %s",
                                                        path,
                                                        path_exc,
                                                    )
                                            plots = ckpt_summary.get("plots", {})
                                            if isinstance(plots, dict):
                                                plots["latents_over_trials_examples"] = []
                                                plots["latents_in_space_examples"] = []
                                except Exception as exc:
                                    logger.warning(
                                        "Checkpoint held-out image logging failed for step=%s: %s",
                                        steps_completed,
                                        exc,
                                    )
                    except Exception as exc:
                        logger.warning(
                            "Held-out evaluation failed for checkpoint step=%s: %s",
                            steps_completed,
                            exc,
                        )

                # --- Early-stopping check (opt-in; checkpoint already saved above) ---
                # Gated: only arm once steps_completed >= start_after_step, so the
                # session-conditioning curriculum reaches lambda=1 (+ buffer) first.
                if _es_enabled and steps_completed >= _es_start_after:
                    _es_val = (
                        train_likelihood_ckpt
                        if _es_metric == "train_likelihood"
                        else eval_likelihood_ckpt
                    )
                    if _es_val is not None:
                        if _es_val > _es_best + _es_min_delta:
                            _es_best = _es_val
                            _es_stale = 0
                        else:
                            _es_stale += 1
                        _es_overfit = (
                            _es_guard is not None and (_es_best - _es_val) > _es_guard
                        )
                        if _es_stale >= _es_patience or _es_overfit:
                            logger.info(
                                "Early stopping at step %s: %s=%.4f best=%.4f stale=%d/%d overfit=%s",
                                steps_completed, _es_metric, _es_val, _es_best,
                                _es_stale, _es_patience, _es_overfit,
                            )
                            break

            losses = {
                "training_loss": np.asarray(all_training_losses, dtype=float),
                "validation_loss": np.asarray(all_validation_losses, dtype=float),
            }

        training_time = time.time() - start
        actual_training_steps = int(args.n_steps)
        if args.checkpoint_every_n_steps > 0:
            actual_training_steps = int(steps_completed)
        output["training_time"] = training_time
        output["training_steps_completed"] = actual_training_steps
        output["training_steps_requested"] = int(args.n_steps)
        output["checkpoint_every_n_steps"] = int(args.checkpoint_every_n_steps)
        output["session_curriculum"]["final_lambda"] = float(
            _session_curriculum_lambda_for_step(actual_training_steps)
        )
        if checkpoint_records:
            output["checkpoints"] = checkpoint_records
        if checkpoint_heldout_summaries:
            output["heldout_test_checkpoints"] = checkpoint_heldout_summaries

        losses_path = self._plot_losses(
            losses,
            title="Loss over Training",
            output_name="validation.png",
        )
        if wandb_run is not None:
            wandb_run.log({"fig/validation_loss_curve": wandb.Image(str(losses_path))})

        final_session_curriculum_lambda = _session_curriculum_lambda_for_step(
            actual_training_steps
        )
        final_eval_network = _make_eval_network_for_step(actual_training_steps)

        if is_multisubject:
            subject_embedding_fig = self._plot_subject_embedding_state_space(
                params=params,
                raw_df=bundle.raw,
                metadata=metadata,
            )
            if subject_embedding_fig is not None:
                subject_embedding_path = self._save_figure(
                    subject_embedding_fig,
                    "subject_embedding_state_space.png",
                )
                output["subject_embedding_state_space_path"] = str(subject_embedding_path)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "fig/subject_embedding_state_space": wandb.Image(
                                str(subject_embedding_path)
                            )
                        }
                    )
            subject_session_context_fig = self._plot_subject_session_context_state_space(
                params=params,
                raw_df=bundle.raw,
                metadata=metadata,
                session_curriculum_lambda=final_session_curriculum_lambda,
            )
            if subject_session_context_fig is not None:
                subject_session_context_path = self._save_figure(
                    subject_session_context_fig,
                    "subject_session_context_state_space.png",
                )
                output["subject_session_context_state_space_path"] = str(
                    subject_session_context_path
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "fig/subject_session_context_state_space": wandb.Image(
                                str(subject_session_context_path)
                            )
                        }
                    )

        xs_full = dataset.get_all()["xs"]
        yhat_full, network_states_full = self._eval_network_full(
            final_eval_network,
            params,
            xs_full,
        )

        df = bundle.raw
        # Only materialize the whole-cohort per-trial frame when it is persisted.
        # Split-example plotting builds per-subject frames on demand from the
        # tensors (see _generate_split_examples), so we avoid the eval OOM on
        # large multisubject cohorts. Likelihoods below use the yhat tensors
        # directly and never need this frame.
        output_df = None
        if args.save_output_df:
            output_df = add_gru_model_results(
                df,
                np.asarray(network_states_full),
                np.asarray(yhat_full),
                ignore_policy=ignore_policy,
            )
            output_df.to_csv(self.output_dir / "output_df.csv", index=False)

        params_path = self.output_dir / "params.json"
        with params_path.open("w") as f:
            f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))
        if is_multisubject:
            output["subject_artifacts"] = self._save_multisubject_artifacts(
                params=params,
                metadata=metadata,
                raw_df=df,
            )

        _train_data = dataset_train.get_all()
        xs_train, ys_train = _train_data["xs"], _train_data["ys"]
        yhat_train, _ = self._eval_network_full(final_eval_network, params, xs_train)
        n_action_logits_train = _require_n_action_logits(
            dataset_train,
            np.asarray(yhat_train),
            context="final train",
        )
        likelihood_train = rnn_utils.normalized_likelihood(
            ys_train,
            np.asarray(yhat_train)[:, :, :n_action_logits_train],
        )
        output["likelihood_train"] = float(likelihood_train)
        logger.info("Final training likelihood: %.4f", float(likelihood_train))

        _eval_data = dataset_eval.get_all()
        xs_eval, ys_eval = _eval_data["xs"], _eval_data["ys"]
        yhat_eval, _ = self._eval_network_full(final_eval_network, params, xs_eval)
        n_action_logits = _require_n_action_logits(
            dataset_eval,
            np.asarray(yhat_eval),
            context="final eval",
        )

        likelihood = rnn_utils.normalized_likelihood(
            ys_eval,
            np.asarray(yhat_eval)[:, :, :n_action_logits],
        )
        output["likelihood"] = float(likelihood)
        logger.info("Final eval likelihood: %.4f", float(likelihood))

        final_output_dir = self.output_dir
        if args.checkpoint_every_n_steps > 0:
            final_output_dir = (
                self.output_dir / "checkpoints" / f"step_{actual_training_steps}"
            )
            final_output_dir.mkdir(parents=True, exist_ok=True)

        split_summaries: dict[str, Any] | None = None
        if checkpoint_records:
            last_checkpoint = checkpoint_records[-1]
            if (
                int(last_checkpoint.get("step", -1)) == int(actual_training_steps)
                and "split_examples" in last_checkpoint
            ):
                split_summaries = last_checkpoint["split_examples"]

        if split_summaries is None:
            split_summaries = self._generate_split_examples(
                output_dir=final_output_dir,
                output_df=output_df,
                raw_df=df,
                ignore_policy=ignore_policy,
                network_states_full=np.asarray(network_states_full),
                yhat_full=np.asarray(yhat_full),
                params=params,
                metadata=metadata,
                n_action_logits=n_action_logits,
                wandb_run=wandb_run if args.checkpoint_every_n_steps == 0 else None,
                log_scope="Final",
            )
        output["split_examples"] = split_summaries

        gt_likelihood = metadata.get("avg_eval_likelihood_groundtruth")
        if gt_likelihood is not None:
            output["groundtruth_likelihood"] = float(gt_likelihood)
            output["likelihood_relative_to_groundtruth"] = float(likelihood) / float(
                gt_likelihood
            )

        with open(self.output_dir / "output_summary.json", "w") as f:
            json.dump(output, f, indent=4)

        if checkpoint_records:
            checkpoint_metrics_path = self.output_dir / "checkpoint_metrics.json"
            with checkpoint_metrics_path.open("w") as f:
                json.dump(checkpoint_records, f, indent=2)

            checkpoint_root = self.output_dir / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            checkpoint_index_path = checkpoint_root / "index.json"
            checkpoint_index = {
                "checkpoint_every_n_steps": int(args.checkpoint_every_n_steps),
                "n_steps": int(actual_training_steps),
                "requested_n_steps": int(args.n_steps),
                "completed_steps": int(actual_training_steps),
                "early_stopped": bool(actual_training_steps < int(args.n_steps)),
                "count": int(len(checkpoint_records)),
                "checkpoints": checkpoint_records,
            }
            with checkpoint_index_path.open("w") as f:
                json.dump(checkpoint_index, f, indent=2)

        gru_config = {
            "architecture": dict(self.architecture),
            "training": dict(self.training),
            "output_size": int(args.output_size),
        }
        with open(self.output_dir / "gru_config.json", "w") as f:
            json.dump(gru_config, f, indent=4)

        if wandb_run is not None:
            if len(losses["validation_loss"]) > 0:
                wandb_run.summary["final/val_loss"] = float(losses["validation_loss"][-1])
            if len(losses["training_loss"]) > 0:
                wandb_run.summary["final/train_loss"] = float(losses["training_loss"][-1])
            wandb_run.summary["likelihood"] = float(likelihood)
            wandb_run.summary["likelihood_train"] = float(likelihood_train)

            if gt_likelihood is not None:
                wandb_run.summary["groundtruth_likelihood"] = float(gt_likelihood)
                wandb_run.summary["likelihood_relative_to_groundtruth"] = float(
                    likelihood
                ) / float(gt_likelihood)

            wandb_run.summary["elapsed_seconds"] = float(training_time)

            artifact_name = f"gru-output-{getattr(wandb_run, 'id', None) or 'latest'}"
            artifact = wandb.Artifact(artifact_name, type="training-output")
            artifact.add_dir(str(self.output_dir))
            wandb_run.log_artifact(artifact)

        return output


    def _plot_examples_for_split(
        self,
        *,
        split_name: str,
        output_dir: Path,
        output_df: pd.DataFrame,
        network_states: np.ndarray,
        yhat_logits: np.ndarray,
        params: Any,
        sessions_per_subject: int,
        max_subjects_to_plot: int,
        n_action_logits: int,
        wandb_run: Any | None,
        log_scope: str | None,
    ) -> dict[str, Any]:
        # GRU example plotting does not use the model params.
        return plot_gru_examples_for_split(
            split_name=split_name,
            output_dir=output_dir,
            output_df=output_df,
            network_states=network_states,
            yhat_logits=yhat_logits,
            sessions_per_subject=sessions_per_subject,
            max_subjects_to_plot=max_subjects_to_plot,
            n_action_logits=n_action_logits,
            wandb_run=wandb_run,
            log_scope=log_scope,
        )
