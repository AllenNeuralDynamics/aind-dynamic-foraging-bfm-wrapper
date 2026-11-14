"""Standalone held-out subject embedding fine-tuning for multisubject runs.

DOCUMENTED training-adjacent exception: unlike the rest of post_training_analysis, this
module genuinely *re-trains* (fine-tunes) held-out subject embeddings, so it legitimately
imports and uses ``DisrnnTrainer``/``GruTrainer`` and the training utilities. It is invoked
explicitly (e.g. ``run_analysis.py finetune``); importing the post_training_analysis package
does not import it (the lazy gateway defers it until called).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import data_loaders.disrnn_dataset as dl
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from base.types import DatasetBundle
from data_loaders import mice as mice_loader
from data_loaders.external_bandit import apply_external_adaptation_budget
from disentangled_rnns.library import rnn_utils
from evaluation.target_transfer import (
    build_binary_trial_predictions,
    summarize_binary_trial_predictions,
    target_test_mask,
)
from model_trainers.disrnn_trainer import DisrnnTrainer
from model_trainers.gru_trainer import GruTrainer, _threeway_likelihood_decomposition
from models.gru_network import make_gru_network
from models.session_conditioning import resolve_session_conditioning_from_architecture
from post_training_analysis.generative_analysis import resolve_model_run
from utils.gru_evaluation import add_gru_model_results
from utils.multisubject import (
    append_subjects_to_index_maps,
    build_subject_embedding_update_mask,
    expand_local_multisubject_params,
    extend_session_context,
    extract_subject_embeddings_from_params,
    load_session_context_map,
    load_subject_index_map,
)
from utils.session_regularized_training import train_network_with_session_regularization

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_ROOT = "/results/heldout_subject_finetuning"
_GRU_CONFIG_NAME = "gru_config.json"
_DISRNN_CONFIG_NAME = "disrnn_config.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_plain_dict(config_source: Any) -> dict[str, Any]:
    if isinstance(config_source, DictConfig):
        resolved = OmegaConf.to_container(config_source, resolve=True)
        if not isinstance(resolved, dict):
            raise ValueError("Expected mapping-style OmegaConf config.")
        return resolved
    if isinstance(config_source, Mapping):
        return dict(config_source)
    raise TypeError(
        "Held-out fine-tuning config must be a mapping, OmegaConf DictConfig, or path."
    )


def _load_config(config_source: str | Path | Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config_source, (str, Path)):
        cfg = OmegaConf.load(Path(config_source).expanduser())
        return _to_plain_dict(cfg)
    return _to_plain_dict(config_source)


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    raise ValueError(f"Expected a mapping, got {type(value).__name__}.")


def _safe_slug(value: Any) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    safe = safe.strip("_")
    return safe or "value"


def _next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = Path(f"{path}__rerun_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_optional_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value]


def _selector_fields_present(selector_cfg: Mapping[str, Any]) -> bool:
    return any(
        selector_cfg.get(key) is not None
        for key in ("test_subject_ids", "curricula", "min_sessions", "heldout_every_n")
    )


def _heldout_selector_from_data_config(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    # The heldout set is derived from the same pipeline used for training (the
    # reserved every-Nth subjects per curriculum), so it is always available — no
    # explicit selector keys are required in the source config.
    return {
        "test_subject_ids": _normalize_optional_list(data_cfg.get("test_subject_ids")),
        "curricula": list(data_cfg.get("curricula") or []) or None,
        "min_sessions": _coerce_optional_int(data_cfg.get("min_sessions")) or 10,
        "heldout_every_n": _coerce_optional_int(data_cfg.get("heldout_every_n")) or 5,
        "mature_only": bool(data_cfg.get("mature_only", True)),
        "cols_to_retain": data_cfg.get("cols_to_retain"),
        "snapshot": data_cfg.get("snapshot"),
        # Optional timing inputs (previous-trial reaction time / lick counts).
        # MUST be carried: the held-out bundle is built by an INDEPENDENT loader
        # path, so if this is dropped the held-out tensor is narrower than the
        # trained model's input and restore fails with a shape mismatch on
        # 'multisubject_gru/~/gru/w_i' — losing the held-out metric for the whole
        # arm while training itself looks perfectly healthy.
        "timing_features": data_cfg.get("timing_features"),
    }


def _resolve_heldout_selector(
    *,
    config: Mapping[str, Any],
    source_data_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    override_cfg = _normalize_mapping(config.get("heldout_subjects"))
    if not override_cfg or not _selector_fields_present(override_cfg):
        selector = _heldout_selector_from_data_config(source_data_cfg)
        logger.info(
            "Using held-out subject selectors from the source training run: "
            "test_subject_ids=%s curricula=%s min_sessions=%s heldout_every_n=%s",
            selector["test_subject_ids"],
            selector["curricula"],
            selector["min_sessions"],
            selector["heldout_every_n"],
        )
        return selector

    def _pick(key: str, default: Any) -> Any:
        if key in override_cfg and override_cfg[key] is not None:
            return override_cfg[key]
        return source_data_cfg.get(key, default)

    selector = {
        "test_subject_ids": _normalize_optional_list(override_cfg.get("test_subject_ids")),
        "curricula": list(_pick("curricula", []) or []) or None,
        "min_sessions": _coerce_optional_int(_pick("min_sessions", 10)) or 10,
        "heldout_every_n": _coerce_optional_int(_pick("heldout_every_n", 5)) or 5,
        "mature_only": bool(_pick("mature_only", True)),
        "cols_to_retain": _pick("cols_to_retain", None),
        # Carried for the same reason as in _heldout_selector_from_data_config:
        # dropping either of these makes the held-out bundle disagree with the
        # trained model's input width / data snapshot. This branch previously
        # omitted BOTH, so an explicit heldout_subjects override silently lost
        # them even when the source config set them.
        "snapshot": _pick("snapshot", None),
        "timing_features": _pick("timing_features", None),
    }
    logger.info(
        "Using held-out subject selectors from the fine-tuning config: "
        "test_subject_ids=%s curricula=%s min_sessions=%s heldout_every_n=%s",
        selector["test_subject_ids"],
        selector["curricula"],
        selector["min_sessions"],
        selector["heldout_every_n"],
    )
    return selector


def _heldout_selector_slug(selector: Mapping[str, Any]) -> str:
    if selector.get("dataset_id") is not None:
        return f"target_{_safe_slug(selector['dataset_id'])}"
    if selector.get("test_subject_ids") is not None:
        subject_ids = selector["test_subject_ids"]
        if not isinstance(subject_ids, list):
            subject_ids = list(subject_ids)
        joined = "_".join(_safe_slug(subject_id) for subject_id in subject_ids[:8])
        if len(subject_ids) > 8:
            joined = f"{joined}_plus_{len(subject_ids) - 8}"
        return f"ids_{joined}"
    curricula = selector.get("curricula") or []
    curricula_slug = "_".join(_safe_slug(curriculum) for curriculum in curricula[:4]) or "all"
    return f"heldout_{curricula_slug}_every{selector.get('heldout_every_n', 5)}"


def _resolved_output_root(
    config: Mapping[str, Any],
    *,
    output_root_override: str | Path | None,
) -> Path:
    output_cfg = _normalize_mapping(config.get("output"))
    output_root = output_root_override or output_cfg.get("output_root") or _DEFAULT_OUTPUT_ROOT
    return Path(str(output_root)).expanduser().resolve()


def _maybe_start_wandb_run(
    *,
    resolved_config: Mapping[str, Any],
    run_dir: Path,
    source_run: Any,
) -> Any | None:
    wandb_cfg = _normalize_mapping(resolved_config.get("wandb"))
    if not wandb_cfg:
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        logger.warning(
            "Skipping W&B logging because wandb is not installed in the active runtime."
        )
        return None

    init_kwargs = dict(wandb_cfg)
    tags = list(init_kwargs.pop("tags", []) or [])
    tags.extend(["heldout_subject_finetuning", str(source_run.model_type)])
    init_kwargs.setdefault("dir", str(run_dir))
    source_wandb_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("wandb"))
    default_name = source_wandb_cfg.get("name") or run_dir.name
    init_kwargs.setdefault("name", str(default_name))
    from utils.run_helpers import _init_wandb_with_fallback

    return _init_wandb_with_fallback(
        {
            **init_kwargs,
            "config": resolved_config,
            "tags": tags,
        }
    )


def _heldout_subject_run_name_suffix(heldout_subject_ids: Sequence[Any]) -> str:
    if not heldout_subject_ids:
        return ""
    encoded_ids = "\0".join(str(subject_id) for subject_id in heldout_subject_ids)
    digest = hashlib.sha1(encoded_ids.encode("utf-8")).hexdigest()[:8]
    return f"heldout-n{len(heldout_subject_ids)}-{digest}"


def _update_wandb_run_name_for_heldout_subjects(
    *,
    wandb_run: Any,
    resolved_config: Mapping[str, Any],
    source_run: Any,
    heldout_subject_ids: Sequence[Any],
) -> None:
    configured_name = _normalize_mapping(resolved_config.get("wandb")).get("name")
    source_wandb_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("wandb"))
    base_name = configured_name or source_wandb_cfg.get("name") or wandb_run.name or "heldout_finetuning"
    suffix = _heldout_subject_run_name_suffix(heldout_subject_ids)
    new_name = f"{base_name}_{suffix}" if suffix else str(base_name)
    wandb_run.name = new_name
    try:
        wandb_run.config.update({"resolved_heldout_subject_ids": list(heldout_subject_ids)})
    except Exception:
        logger.warning("Failed to update W&B config with resolved held-out subject IDs.")
    logger.info("Updated W&B run name to: %s", new_name)


def _resolve_runtime_config(
    config: Mapping[str, Any],
    *,
    source_run: Any,
    output_root_override: str | Path | None,
) -> dict[str, Any]:
    source_cfg = _normalize_mapping(config.get("source_run"))
    if "model_dir" not in source_cfg:
        raise ValueError("Config must set source_run.model_dir.")

    fine_tune_cfg = _normalize_mapping(config.get("heldout_finetuning"))
    source_data_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("data"))
    source_model_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("model"))
    heldout_selector = _resolve_heldout_selector(
        config=config,
        source_data_cfg=source_data_cfg,
    )
    target_data_cfg = config.get("target_data")
    if target_data_cfg is not None and not isinstance(target_data_cfg, Mapping):
        raise ValueError("target_data must be a Hydra-instantiable mapping or null.")

    if "n_steps" not in fine_tune_cfg:
        raise ValueError("Config must set heldout_finetuning.n_steps.")
    if "lr" not in fine_tune_cfg:
        raise ValueError("Config must set heldout_finetuning.lr.")

    checkpoint_every_n_steps = int(
        fine_tune_cfg.get("checkpoint_every_n_steps", fine_tune_cfg["n_steps"])
    )
    if checkpoint_every_n_steps < 0:
        raise ValueError("heldout_finetuning.checkpoint_every_n_steps must be >= 0.")

    seed = config.get("seed", source_run.seed)
    resolved = copy.deepcopy(config)
    resolved["seed"] = int(seed if seed is not None else 0)
    resolved.setdefault("source_run", {})
    resolved["source_run"]["checkpoint_policy"] = source_cfg.get(
        "checkpoint_policy",
        "best_eval",
    )
    resolved.setdefault("output", {})
    resolved["output"]["output_root"] = str(
        _resolved_output_root(config, output_root_override=output_root_override)
    )
    resolved["heldout_subjects"] = heldout_selector
    if target_data_cfg is not None:
        resolved["target_data"] = copy.deepcopy(dict(target_data_cfg))
    selection_policy = str(fine_tune_cfg.get("selection_policy", "fixed_final"))
    if selection_policy != "fixed_final":
        raise ValueError(
            "heldout_finetuning.selection_policy must be 'fixed_final'; target-test "
            "performance cannot select the fine-tuning step."
        )
    resolved["heldout_finetuning"] = {
        "n_steps": int(fine_tune_cfg["n_steps"]),
        "lr": float(fine_tune_cfg["lr"]),
        "checkpoint_every_n_steps": checkpoint_every_n_steps,
        "batch_size": fine_tune_cfg.get("batch_size", source_data_cfg.get("batch_size")),
        "batch_mode": str(fine_tune_cfg.get("batch_mode", "single")),
        "checkpoint_plot_split_examples_every_n": int(
            fine_tune_cfg.get("checkpoint_plot_split_examples_every_n", 0)
        ),
        "checkpoint_save_output_df_every_n": int(
            fine_tune_cfg.get("checkpoint_save_output_df_every_n", 0)
        ),
        "train_example_sessions_per_subject": int(
            fine_tune_cfg.get(
                "train_example_sessions_per_subject",
                source_data_cfg.get("train_example_sessions_per_subject", 1),
            )
        ),
        "eval_example_sessions_per_subject": int(
            fine_tune_cfg.get(
                "eval_example_sessions_per_subject",
                source_data_cfg.get("eval_example_sessions_per_subject", 1),
            )
        ),
        "example_max_subjects": int(
            fine_tune_cfg.get(
                "example_max_subjects",
                source_data_cfg.get("example_max_subjects", 6),
            )
        ),
        "keep_media_files": bool(fine_tune_cfg.get("keep_media_files", True)),
        "max_grad_norm": float(
            fine_tune_cfg.get(
                "max_grad_norm",
                _normalize_mapping(source_model_cfg.get("training")).get("max_grad_norm", 1.0),
            )
        ),
        # Few-shot adaptation budget: cap each held-out subject's ADAPT (train) sessions
        # to the first K (deterministic by session order). None/absent = use all adapt
        # sessions (current behavior). K=0 => no adaptation (zero-shot init eval only).
        # The EVAL (test) split is never touched, so held-out likelihood stays comparable
        # across K.
        "adapt_sessions_per_subject": (
            None
            if fine_tune_cfg.get("adapt_sessions_per_subject") is None
            else int(fine_tune_cfg["adapt_sessions_per_subject"])
        ),
        "selection_policy": selection_policy,
        "skip_subjects_with_insufficient_sessions": bool(
            fine_tune_cfg.get("skip_subjects_with_insufficient_sessions", True)
        ),
        # Opt-in per-trial export (GRU): p(left), p(right), p(ignore) for every
        # held-out trial, fine-tuned and scored sessions alike. Off by default.
        "export_trial_predictions": bool(fine_tune_cfg.get("export_trial_predictions", False)),
    }
    if (
        resolved["heldout_finetuning"]["adapt_sessions_per_subject"] is not None
        and resolved["heldout_finetuning"]["adapt_sessions_per_subject"] < 0
    ):
        raise ValueError(
            "heldout_finetuning.adapt_sessions_per_subject must be >= 0 or null."
        )
    return resolved


def _load_configured_target_bundle(target_data_cfg: Mapping[str, Any]) -> DatasetBundle:
    """Instantiate an external target loader without importing Hydra at module import."""
    from hydra.utils import instantiate

    loader = instantiate(OmegaConf.create(dict(target_data_cfg)))
    if isinstance(loader, DatasetBundle):
        bundle = loader
    elif hasattr(loader, "load"):
        bundle = loader.load()
    elif hasattr(loader, "load_data"):
        bundle = loader.load_data()
    else:
        bundle = loader
    if not isinstance(bundle, DatasetBundle):
        raise TypeError(
            "target_data must instantiate a DatasetLoader or DatasetBundle; got "
            f"{type(bundle).__name__}."
        )
    return bundle


def _resolve_output_run_dir(
    *,
    resolved_config: Mapping[str, Any],
    source_run: Any,
    heldout_selector: Mapping[str, Any],
) -> Path:
    output_root = Path(resolved_config["output"]["output_root"]).expanduser().resolve()
    # The source model_dir follows the W&B layout ``.../run-<id>/files``, so
    # ``Path(model_dir).name`` is the constant string "files" (likewise "results"/
    # "outputs" for other layouts) and the unique run id lives in the PARENT dir.
    # Using only .name made every concurrent grid cell share one held-out output
    # path on a shared filesystem (HPC scratch), so a second cell's rebuild could
    # read a first cell's config -> architecture/shape mismatch. Fold in the parent
    # dir name so the slug is unique per source run. (Beaker isolates /results per
    # task, so it never collided there.)
    _model_dir_path = Path(source_run.model_dir)
    _run_name = _model_dir_path.name
    if _run_name in {"files", "results", "outputs", ""} and _model_dir_path.parent.name:
        _run_name = f"{_model_dir_path.parent.name}_{_run_name}"
    source_run_slug = _safe_slug(_run_name)
    checkpoint_component = (
        f"{_safe_slug(source_run.checkpoint_policy)}_{_safe_slug(source_run.checkpoint_label)}"
    )
    selector_component = _heldout_selector_slug(heldout_selector)
    run_name_suffix = resolved_config.get("output", {}).get("run_name_suffix")
    if run_name_suffix:
        selector_component = f"{selector_component}_{_safe_slug(run_name_suffix)}"
    return _next_available_dir(
        output_root / source_run_slug / checkpoint_component / selector_component
    )


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _save_params(path: Path, params: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))


def _load_params(params_path: str | Path) -> Any:
    with Path(params_path).open("r") as f:
        return rnn_utils.to_np(json.load(f))


def _load_heldout_snapshot_selection(
    *,
    heldout_selector: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[Any]]:
    from utils.load_mice_database import load_mice_from_database

    df, subject_ids = load_mice_from_database(
        split="heldout",
        subject_ids=heldout_selector.get("test_subject_ids"),
        curricula=heldout_selector.get("curricula"),
        min_sessions=int(heldout_selector.get("min_sessions", 10)),
        heldout_every_n=int(heldout_selector.get("heldout_every_n", 5)),
        mature_only=bool(heldout_selector.get("mature_only", True)),
        cols_to_retain=heldout_selector.get("cols_to_retain"),
        snapshot=heldout_selector.get("snapshot"),
    )
    if len(df) == 0:
        raise ValueError("Held-out subject selection resolved to an empty snapshot.")
    return df, list(subject_ids)


def _attach_heldout_timing_features(
    df: pd.DataFrame,
    *,
    heldout_selector: Mapping[str, Any],
    source_data_cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, str] | None]:
    """Derive the optional timing inputs on the HELD-OUT dataframe.

    The held-out bundle is built by this module, not by
    ``data_loaders.mice.MiceSnapshotDatasetLoader``, so the timing derivation the
    training loader performs does NOT happen here for free. Without this the
    held-out tensor is narrower than the trained model's input and restoring the
    checkpoint fails with e.g.::

        'multisubject_gru/~/gru/w_i' with retrieved shape (9, 384)
        does not match shape=[6, 384]

    which loses the held-out metric for the entire arm while training itself
    reports success — the exact failure observed in study-07 `d100-bridge`
    (2026-08-22). Returns the augmented frame plus the feature map to use, or
    ``(df, None)`` when timing inputs are disabled so the base path is untouched.
    """
    from utils.trial_timing_features import resolve_timing_config

    raw_timing_cfg = (
        heldout_selector.get("timing_features")
        if heldout_selector.get("timing_features") is not None
        else source_data_cfg.get("timing_features")
    )
    # Resolve only to test the enabled flag; pass the RAW config downstream --
    # _augment_features_with_timing resolves it itself and rejects an
    # already-resolved TimingConfig.
    if not resolve_timing_config(raw_timing_cfg).enabled:
        return df, None

    from data_loaders.mice import _augment_features_with_timing

    # The shuffled control arm derives its permutation from the RUN seed when
    # `shuffle_seed` is left at the sentinel. The held-out loader is instantiated
    # independently of the training loader, so if the run seed were not passed
    # here the held-out frame would be permuted with seed 0 while training used
    # the run's own seed — a silent train/eval mismatch that is invisible in the
    # logs (correct tensor width, plausible numbers, wrong permutation). Same
    # class of bug as the missing timing columns fixed in 35d6a19.
    run_seed = source_data_cfg.get("seed")
    if run_seed is None:
        run_seed = heldout_selector.get("seed")
    df, feature_map = _augment_features_with_timing(
        df,
        base_features=source_data_cfg.get("features"),
        timing_features_cfg=raw_timing_cfg,
        snapshot=heldout_selector.get("snapshot") or source_data_cfg.get("snapshot"),
        run_seed=int(run_seed) if run_seed is not None else None,
    )
    logger.info(
        "Held-out bundle: attached timing features -> %d input feature(s): %s",
        len(feature_map),
        list(feature_map.values()),
    )
    return df, feature_map


def _build_local_heldout_bundle(
    *,
    heldout_df: pd.DataFrame,
    heldout_subject_ids: Sequence[Any],
    source_data_cfg: Mapping[str, Any],
    fine_tune_cfg: Mapping[str, Any],
    session_split_by_subject: Mapping[str, Mapping[str, Any]] | None = None,
    allow_unassigned_sessions: bool = False,
) -> DatasetBundle:
    """Pack the held-out trials; an explicit per-subject session split replaces eval_every_n."""
    skip_subjects_with_insufficient_sessions = bool(
        fine_tune_cfg.get("skip_subjects_with_insufficient_sessions", True)
    )
    metadata = {
        "subject_ids": list(heldout_subject_ids),
        "mature_only": bool(source_data_cfg.get("mature_only", True)),
        "curricula": list(source_data_cfg.get("curricula") or []),
        "ignore_policy": str(source_data_cfg.get("ignore_policy", "exclude")),
        "features": source_data_cfg.get("features"),
        "eval_every_n": int(source_data_cfg.get("eval_every_n", 2)),
        "heldout_example_sessions_per_subject": int(
            max(
                fine_tune_cfg.get("train_example_sessions_per_subject", 1),
                fine_tune_cfg.get("eval_example_sessions_per_subject", 1),
            )
        ),
        "train_example_sessions_per_subject": int(
            fine_tune_cfg.get("train_example_sessions_per_subject", 1)
        ),
        "eval_example_sessions_per_subject": int(
            fine_tune_cfg.get("eval_example_sessions_per_subject", 1)
        ),
        "example_max_subjects": int(fine_tune_cfg.get("example_max_subjects", 6)),
    }
    return mice_loader._build_multisubject_bundle(
        df=heldout_df,
        resolved_subject_ids=heldout_subject_ids,
        ignore_policy=str(source_data_cfg.get("ignore_policy", "exclude")),
        features=source_data_cfg.get("features"),
        eval_every_n=int(source_data_cfg.get("eval_every_n", 2)),
        batch_size=fine_tune_cfg.get("batch_size"),
        batch_mode=str(fine_tune_cfg.get("batch_mode", "single")),
        metadata=metadata,
        skip_subjects_with_insufficient_sessions=skip_subjects_with_insufficient_sessions,
        session_split_by_subject=session_split_by_subject,
        allow_unassigned_sessions=allow_unassigned_sessions,
    )


def _remap_dataset_subject_indices(
    dataset: Any,
    *,
    local_to_global_subject_index: Mapping[int, int],
) -> Any:
    _all = dataset.get_all()
    xs, ys = _all["xs"], _all["ys"]
    remapped_xs = np.asarray(xs).copy()
    subject_values = np.asarray(np.rint(remapped_xs[..., 0]), dtype=int)
    valid_mask = subject_values >= 0
    remapped_subject_values = subject_values.copy()
    for local_index, global_index in local_to_global_subject_index.items():
        remapped_subject_values[subject_values == int(local_index)] = int(global_index)
    remapped_xs[..., 0] = np.where(valid_mask, remapped_subject_values, subject_values).astype(
        remapped_xs.dtype,
        copy=False,
    )
    dataset_class = dataset.__class__
    return dataset_class(
        remapped_xs,
        ys,
        y_type=getattr(dataset, "y_type", "categorical"),
        n_classes=getattr(dataset, "n_classes", None),
        x_names=list(getattr(dataset, "x_names", [])),
        y_names=list(getattr(dataset, "y_names", [])),
        batch_size=getattr(dataset, "batch_size", None),
        batch_mode=getattr(dataset, "batch_mode", "random"),
    )


def _build_extended_session_metadata(
    *,
    source_run: Any,
    total_subject_count: int,
    local_bundle: DatasetBundle,
    local_to_global_subject_index: Mapping[int, int],
) -> tuple[dict[str, Any] | None, list[int]]:
    source_session_context = None
    session_max_index_by_subject_index = [0] * int(total_subject_count)
    if source_run.session_context_map_path:
        source_session_context = load_session_context_map(source_run.session_context_map_path)
        for row in source_session_context.get("per_subject", []):
            subject_index = int(row["subject_index"])
            session_max_index_by_subject_index[subject_index] = int(
                len(row.get("ordered_session_ids") or [])
            )

    local_session_context = _normalize_mapping(local_bundle.metadata.get("session_context"))
    appended_rows: list[dict[str, Any]] = []
    for row in local_session_context.get("per_subject", []):
        local_subject_index = int(row["subject_index"])
        global_subject_index = int(local_to_global_subject_index[local_subject_index])
        ordered_session_ids = [str(value) for value in row.get("ordered_session_ids") or []]
        appended_rows.append(
            {
                "subject_id": row.get("subject_id"),
                "subject_index": global_subject_index,
                "ordered_session_ids": ordered_session_ids,
                "ordered_source_session_ids": [
                    str(value)
                    for value in (
                        row.get("ordered_source_session_ids")
                        or row.get("ordered_session_ids")
                        or []
                    )
                ],
            }
        )
        session_max_index_by_subject_index[global_subject_index] = int(len(ordered_session_ids))

    if source_session_context is None:
        if appended_rows:
            source_session_context = {
                "indexing": "1_based",
                "per_subject": [],
            }
        else:
            return None, session_max_index_by_subject_index
    return extend_session_context(
        source_session_context,
        appended_rows=appended_rows,
    ), session_max_index_by_subject_index


def _maybe_prepend_session_indices(
    *,
    dataset_bundle: DatasetBundle,
    architecture: Mapping[str, Any],
) -> DatasetBundle:
    from utils.multisubject import prepend_session_index_to_multisubject_split_datasets

    metadata = dict(dataset_bundle.metadata)
    max_n_subjects = int(metadata.get("num_subjects", 0))
    subject_embedding_size = architecture.get("subject_embedding_size")
    session_cfg = resolve_session_conditioning_from_architecture(
        architecture=architecture,
        metadata=metadata,
        multisubject=True,
        max_n_subjects=max_n_subjects,
        subject_embedding_size=(
            None if subject_embedding_size is None else int(subject_embedding_size)
        ),
        context="Held-out fine-tuning",
    )
    if not bool(session_cfg["enabled"]):
        return dataset_bundle

    dataset_full, dataset_train, dataset_eval = prepend_session_index_to_multisubject_split_datasets(
        dataset=dataset_bundle.extras["dataset"],
        dataset_train=dataset_bundle.train_set,
        dataset_eval=dataset_bundle.eval_set,
        metadata=metadata,
    )
    return DatasetBundle(
        raw=dataset_bundle.raw,
        train_set=dataset_train,
        eval_set=dataset_eval,
        metadata=metadata,
        extras={"dataset": dataset_full},
    )


def _cap_adapt_sessions_per_subject(
    *,
    dataset_bundle: DatasetBundle,
    adapt_sessions_per_subject: int,
) -> DatasetBundle:
    """Cap each held-out subject's ADAPT (train) sessions to the first K columns.

    The merged train dataset stacks each subject's adapt sessions along axis 1 in the
    same order as ``metadata["train_session_ids"]`` (both built in one pass over the
    per-subject split in ``mice._build_multisubject_bundle``), so column ``j`` of the
    train set is ``train_session_ids[j]`` whose owning subject is recovered from
    ``metadata["session_context"]`` (same mapping used by the per-subject eval
    decomposition). Keeping the first K columns per subject is deterministic by session
    order. ``dataset_eval`` and ``dataset_full`` are left UNCHANGED so held-out
    likelihood stays comparable across K. K=0 yields an empty train set (no adaptation).
    """
    K = int(adapt_sessions_per_subject)
    metadata = dict(dataset_bundle.metadata)
    train_session_ids = [str(s) for s in (metadata.get("train_session_ids") or [])]

    train_set = dataset_bundle.train_set
    n_columns = int(np.asarray(train_set.get_all()["xs"]).shape[1])
    if len(train_session_ids) != n_columns:
        raise ValueError(
            "adapt-session capping: train session/column mismatch: "
            f"len(train_session_ids)={len(train_session_ids)} but train array has "
            f"{n_columns} columns."
        )

    session_context = _normalize_mapping(metadata.get("session_context"))
    subject_id_by_session: dict[str, str] = {}
    for row in session_context.get("per_subject", []):
        subject_id = str(row.get("subject_id"))
        for session_id in row.get("ordered_session_ids") or []:
            subject_id_by_session[str(session_id)] = subject_id

    kept_columns: list[int] = []
    kept_session_ids: list[str] = []
    seen_per_subject: dict[str, int] = {}
    capped_per_subject: dict[str, int] = {}
    for column_index, session_id in enumerate(train_session_ids):
        subject_id = subject_id_by_session.get(session_id, "unknown")
        seen = seen_per_subject.get(subject_id, 0)
        if seen < K:
            kept_columns.append(column_index)
            kept_session_ids.append(session_id)
            capped_per_subject[subject_id] = capped_per_subject.get(subject_id, 0) + 1
        seen_per_subject[subject_id] = seen + 1

    logger.info(
        "Few-shot adaptation budget adapt_sessions_per_subject=%d: capped adapt sessions "
        "per held-out subject to %s (was %s); total adapt columns %d -> %d",
        K,
        capped_per_subject,
        seen_per_subject,
        n_columns,
        len(kept_columns),
    )

    metadata["adapt_sessions_per_subject"] = K
    metadata["adapt_sessions_per_subject_counts"] = capped_per_subject

    if not kept_columns:
        # K=0 (zero-shot): no adapt columns. The DatasetRNN constructor rejects empty
        # arrays, so we leave the train_set untouched here; the caller forces 0 gradient
        # steps off ``metadata["adapt_sessions_per_subject"] == 0`` so the embedding stays
        # at its init and only the step-0 eval + per_subject logging run. The diagnostic
        # train metrics at step 0 then reflect the un-adapted model on the full train
        # split, which is harmless (eval is the comparable metric).
        return DatasetBundle(
            raw=dataset_bundle.raw,
            train_set=dataset_bundle.train_set,
            eval_set=dataset_bundle.eval_set,
            metadata=metadata,
            extras=dataset_bundle.extras,
        )

    _all = train_set.get_all()
    xs = np.asarray(_all["xs"])
    ys = np.asarray(_all["ys"])
    kept = np.asarray(kept_columns, dtype=int)
    capped_xs = xs[:, kept, :]
    capped_ys = ys[:, kept, :]
    dataset_class = train_set.__class__
    capped_train = dataset_class(
        capped_xs,
        capped_ys,
        y_type=getattr(train_set, "y_type", "categorical"),
        n_classes=getattr(train_set, "n_classes", None),
        x_names=list(getattr(train_set, "x_names", [])),
        y_names=list(getattr(train_set, "y_names", [])),
        batch_size=getattr(train_set, "batch_size", None),
        batch_mode=getattr(train_set, "batch_mode", "random"),
    )
    metadata["train_session_ids"] = kept_session_ids
    return DatasetBundle(
        raw=dataset_bundle.raw,
        train_set=capped_train,
        eval_set=dataset_bundle.eval_set,
        metadata=metadata,
        extras=dataset_bundle.extras,
    )


def _build_global_heldout_bundle(
    *,
    source_run: Any,
    heldout_selector: Mapping[str, Any],
    fine_tune_cfg: Mapping[str, Any],
    architecture: Mapping[str, Any],
    target_bundle: DatasetBundle | None = None,
) -> tuple[DatasetBundle, list[Any], list[int], dict[Any, int], dict[int, Any]]:
    if source_run.subject_index_map_path is None:
        raise FileNotFoundError(
            "Held-out fine-tuning requires outputs/subject_index_map.json in the source run."
        )
    if target_bundle is None:
        source_data_cfg = _normalize_mapping(
            _normalize_mapping(source_run.run_config).get("data")
        )
        heldout_df, heldout_subject_ids = _load_heldout_snapshot_selection(
            heldout_selector=heldout_selector
        )
        # Derive optional timing inputs before packing so the held-out tensor has
        # the same input width as the trained model.
        heldout_df, timing_feature_map = _attach_heldout_timing_features(
            heldout_df,
            heldout_selector=heldout_selector,
            source_data_cfg=source_data_cfg,
        )
        if timing_feature_map is not None:
            source_data_cfg = {**source_data_cfg, "features": timing_feature_map}
        local_bundle = _build_local_heldout_bundle(
            heldout_df=heldout_df,
            heldout_subject_ids=heldout_subject_ids,
            source_data_cfg=source_data_cfg,
            fine_tune_cfg=fine_tune_cfg,
        )
    else:
        local_bundle = target_bundle
        if not bool(local_bundle.metadata.get("multisubject")):
            raise ValueError("External target_data must produce a multisubject bundle.")
        if local_bundle.raw is None or not hasattr(local_bundle.raw, "columns"):
            raise ValueError("External target_data must retain its canonical raw table.")
    retained_heldout_subject_ids = list(local_bundle.metadata.get("subject_ids") or [])
    skipped_subject_rows = list(
        local_bundle.metadata.get("skipped_subjects_with_insufficient_sessions") or []
    )
    if skipped_subject_rows:
        logger.warning(
            "Skipped %d held-out subject(s) during fine-tuning bundle construction due to "
            "insufficient sessions for the inherited eval_every_n: %s",
            len(skipped_subject_rows),
            [row.get("subject_id") for row in skipped_subject_rows],
        )
    if not retained_heldout_subject_ids:
        raise ValueError(
            "No held-out subjects remain after filtering and per-subject split validation."
        )
    source_subject_id_to_index, source_index_to_subject_id = load_subject_index_map(
        source_run.subject_index_map_path
    )
    updated_subject_id_to_index, updated_index_to_subject_id = append_subjects_to_index_maps(
        source_subject_id_to_index,
        source_index_to_subject_id,
        retained_heldout_subject_ids,
    )
    added_subject_ids = [
        subject_id
        for subject_id in retained_heldout_subject_ids
        if subject_id not in source_subject_id_to_index
    ]
    if not added_subject_ids:
        raise ValueError(
            "Held-out subject selection did not add any new subjects beyond the source run."
        )

    local_subject_id_to_index = dict(local_bundle.metadata["subject_id_to_index"])
    local_to_global_subject_index = {
        int(local_index): int(updated_subject_id_to_index[subject_id])
        for subject_id, local_index in local_subject_id_to_index.items()
    }
    appended_subject_indices = [
        int(updated_subject_id_to_index[subject_id]) for subject_id in added_subject_ids
    ]
    remapped_full = _remap_dataset_subject_indices(
        local_bundle.extras["dataset"],
        local_to_global_subject_index=local_to_global_subject_index,
    )
    remapped_train = _remap_dataset_subject_indices(
        local_bundle.train_set,
        local_to_global_subject_index=local_to_global_subject_index,
    )
    remapped_eval = _remap_dataset_subject_indices(
        local_bundle.eval_set,
        local_to_global_subject_index=local_to_global_subject_index,
    )
    raw_df = local_bundle.raw.copy()
    raw_df["subject_index"] = raw_df["subject_id"].map(updated_subject_id_to_index)

    session_context, session_max_index_by_subject_index = _build_extended_session_metadata(
        source_run=source_run,
        total_subject_count=len(updated_index_to_subject_id),
        local_bundle=local_bundle,
        local_to_global_subject_index=local_to_global_subject_index,
    )
    metadata = dict(local_bundle.metadata)
    ordered_subject_ids = [
        updated_index_to_subject_id[index] for index in sorted(updated_index_to_subject_id)
    ]
    metadata.update(
        {
            "subject_ids": ordered_subject_ids,
            "subject_id_to_index": updated_subject_id_to_index,
            "index_to_subject_id": updated_index_to_subject_id,
            "num_subjects": int(len(updated_index_to_subject_id)),
            "session_max_index_by_subject_index": session_max_index_by_subject_index,
            "train_session_ids": list(local_bundle.metadata.get("train_session_ids") or []),
            "eval_session_ids": list(local_bundle.metadata.get("eval_session_ids") or []),
            "heldout_subject_ids": list(retained_heldout_subject_ids),
            "heldout_subject_indices": [
                int(updated_subject_id_to_index[subject_id])
                for subject_id in retained_heldout_subject_ids
            ],
            "plot_subject_indices": [
                int(updated_subject_id_to_index[subject_id])
                for subject_id in retained_heldout_subject_ids
            ],
            "multisubject": True,
        }
    )
    if session_context is not None:
        metadata["session_context"] = session_context

    global_bundle = DatasetBundle(
        raw=raw_df,
        train_set=remapped_train,
        eval_set=remapped_eval,
        metadata=metadata,
        extras={"dataset": remapped_full},
    )
    global_bundle = _maybe_prepend_session_indices(
        dataset_bundle=global_bundle,
        architecture=architecture,
    )
    adapt_sessions_per_subject = fine_tune_cfg.get("adapt_sessions_per_subject")
    if target_bundle is not None:
        global_bundle = apply_external_adaptation_budget(
            global_bundle,
            adapt_sessions_per_subject=adapt_sessions_per_subject,
        )
    elif adapt_sessions_per_subject is not None:
        global_bundle = _cap_adapt_sessions_per_subject(
            dataset_bundle=global_bundle,
            adapt_sessions_per_subject=int(adapt_sessions_per_subject),
        )
    logger.info(
        "Held-out fine-tuning dataset shapes after session packing: full input %s, train %s, "
        "eval %s, x_names=%s",
        tuple(np.asarray(global_bundle.extras["dataset"].get_all()["xs"]).shape),
        tuple(np.asarray(global_bundle.train_set.get_all()["xs"]).shape),
        tuple(np.asarray(global_bundle.eval_set.get_all()["xs"]).shape),
        list(getattr(global_bundle.extras["dataset"], "x_names", [])),
    )
    return (
        global_bundle,
        retained_heldout_subject_ids,
        appended_subject_indices,
        updated_subject_id_to_index,
        updated_index_to_subject_id,
    )


def _infer_n_action_logits(
    dataset: Any,
    yhat: np.ndarray | None = None,
    *,
    ignore_policy: str,
) -> int:
    n_action_logits = int(getattr(dataset, "n_classes", 0))
    if n_action_logits > 0:
        return n_action_logits
    if yhat is not None:
        inferred = int(np.asarray(yhat).shape[2])
        if inferred > 2 and str(ignore_policy) == "exclude":
            return inferred - 1
        return inferred
    return 2 if str(ignore_policy) == "exclude" else 3


def _compute_supervised_loss_from_outputs(
    yhat: np.ndarray,
    ys: np.ndarray,
    *,
    loss_name: str,
    loss_param: Any,
    n_action_logits: int,
) -> float:
    logits = np.asarray(yhat)[:, :, : int(n_action_logits)]
    targets = np.asarray(ys)
    if targets.ndim != 3 or targets.shape[-1] != 1:
        raise ValueError(
            "Held-out fine-tuning expects targets shaped [timesteps, sessions, 1]."
        )
    targets = np.squeeze(targets, axis=-1)
    valid_mask = np.isfinite(targets) & (targets != -1)
    if not np.any(valid_mask):
        return float("nan")

    safe_targets = np.where(valid_mask, targets, 0).astype(int)
    logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = logits_stable - np.log(np.sum(np.exp(logits_stable), axis=-1, keepdims=True))
    selected_log_probs = log_probs[valid_mask]
    selected_targets = safe_targets[valid_mask]
    supervised_loss = -np.mean(
        selected_log_probs[np.arange(selected_log_probs.shape[0]), selected_targets]
    )

    resolved_loss_name = str(loss_name).strip().lower()
    if resolved_loss_name != "penalized_categorical":
        return float(supervised_loss)

    if isinstance(loss_param, Mapping):
        if "penalty_scale" in loss_param:
            penalty_scale = float(loss_param["penalty_scale"])
        elif "value" in loss_param:
            penalty_scale = float(loss_param["value"])
        else:
            penalty_scale = 1.0
    else:
        penalty_scale = float(loss_param)
    penalty_values = np.asarray(yhat)[..., -1]
    penalty = float(np.mean(penalty_values[valid_mask]))
    return float(supervised_loss + penalty_scale * penalty)


def _make_gru_runtime(
    *,
    source_run: Any,
    output_dir: Path,
) -> tuple[GruTrainer, Any, Any, dict[str, Any], Mapping[str, Any]]:
    source_model_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("model"))
    architecture = _normalize_mapping(source_model_cfg.get("architecture"))
    training_cfg = _normalize_mapping(source_model_cfg.get("training"))
    trainer = GruTrainer(
        architecture=architecture,
        training=training_cfg,
        output_dir=str(output_dir),
        seed=source_run.seed,
    )
    return trainer, architecture, training_cfg, copy.deepcopy(source_run.model_config), source_model_cfg


def _make_disrnn_runtime(
    *,
    source_run: Any,
    output_dir: Path,
) -> tuple[DisrnnTrainer, Mapping[str, Any], Mapping[str, Any], dict[str, Any], Mapping[str, Any]]:
    source_model_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("model"))
    architecture = _normalize_mapping(source_model_cfg.get("architecture"))
    penalties = _normalize_mapping(source_model_cfg.get("penalties"))
    training_cfg = _normalize_mapping(source_model_cfg.get("training"))
    distillation_cfg = source_model_cfg.get("distillation")
    trainer = DisrnnTrainer(
        architecture=architecture,
        penalties=penalties,
        training=training_cfg,
        distillation=distillation_cfg,
        output_dir=str(output_dir),
        seed=source_run.seed,
    )
    return trainer, architecture, training_cfg, copy.deepcopy(source_run.model_config), source_model_cfg


def _save_state_space_figures(
    *,
    trainer: Any,
    params: Any,
    bundle: DatasetBundle,
    checkpoint_dir: Path,
) -> dict[str, str | None]:
    plot_paths = {
        "subject_embedding_state_space": None,
        "subject_session_context_state_space": None,
    }
    subject_embedding_fig = trainer._plot_subject_embedding_state_space(
        params=params,
        raw_df=bundle.raw,
        metadata=bundle.metadata,
    )
    if subject_embedding_fig is not None:
        path = checkpoint_dir / "subject_embedding_state_space.png"
        subject_embedding_fig.savefig(path)
        plt.close(subject_embedding_fig)
        plot_paths["subject_embedding_state_space"] = str(path)

    subject_session_context_fig = trainer._plot_subject_session_context_state_space(
        params=params,
        raw_df=bundle.raw,
        metadata=bundle.metadata,
        session_curriculum_lambda=1.0,
    )
    if subject_session_context_fig is not None:
        path = checkpoint_dir / "subject_session_context_state_space.png"
        subject_session_context_fig.savefig(path)
        plt.close(subject_session_context_fig)
        plot_paths["subject_session_context_state_space"] = str(path)
    return plot_paths


def _log_checkpoint_plot_paths_to_wandb(
    *,
    wandb_run: Any,
    plot_paths: Mapping[str, str | None],
    step: int,
    key_prefix: str = "checkpoint",
) -> None:
    try:
        import wandb
    except ModuleNotFoundError:
        logger.warning("wandb is unavailable; skipping checkpoint image logging.")
        return

    payload: dict[str, Any] = {}
    subject_embedding_path = plot_paths.get("subject_embedding_state_space")
    if subject_embedding_path:
        payload[f"{key_prefix}/fig/subject_embedding_state_space"] = wandb.Image(
            str(subject_embedding_path)
        )
    subject_session_context_path = plot_paths.get("subject_session_context_state_space")
    if subject_session_context_path:
        payload[f"{key_prefix}/fig/subject_session_context_state_space"] = wandb.Image(
            str(subject_session_context_path)
        )
    if payload:
        wandb_run.log(payload, step=int(step))


def _compute_per_subject_eval_likelihood(
    *,
    ys_eval: np.ndarray,
    yhat_eval: np.ndarray,
    eval_likelihood: float,
    metadata: Mapping[str, Any],
    rtol: float = 1e-4,
) -> dict[str, Any] | None:
    """Decompose the aggregate eval normalized likelihood per held-out subject/session.

    The merged eval dataset stacks each held-out subject's eval sessions along axis 1
    (episodes) in the same order they appear in ``metadata["eval_session_ids"]`` (both
    are built in a single pass over the same per-subject split in
    ``mice._build_multisubject_bundle``). Column ``j`` of ``ys_eval``/``yhat_eval`` is
    therefore eval session ``eval_session_ids[j]``, whose owning subject is recovered from
    ``metadata["session_context"]``.

    Normalized likelihood is ``exp(-total_nll / n_unmasked)`` (see
    ``rnn_utils.normalized_likelihood``), so per-session and per-subject values are the
    same quantity recomputed over the relevant columns; the per-subject aggregate is the
    trial-count-weighted combination of its sessions' log-likelihoods (i.e. recomputed
    over the subject's eval trials), not a plain mean of per-session values.

    Returns the JSON-ready wrapper dict, or ``None`` if the decomposition fails.
    """
    try:
        ys_eval = np.asarray(ys_eval)
        yhat_eval = np.asarray(yhat_eval)
        eval_session_ids = [str(s) for s in (metadata.get("eval_session_ids") or [])]
        n_columns = int(ys_eval.shape[1])
        if len(eval_session_ids) != n_columns:
            raise ValueError(
                "eval session/column mismatch: "
                f"len(eval_session_ids)={len(eval_session_ids)} but eval array has "
                f"{n_columns} columns."
            )

        # ses_idx -> subject_id from the session context (the same unique ses_idx form
        # used for the merged dataset columns).
        session_context = _normalize_mapping(metadata.get("session_context"))
        subject_id_by_session: dict[str, str] = {}
        for row in session_context.get("per_subject", []):
            subject_id = str(row.get("subject_id"))
            for session_id in row.get("ordered_session_ids") or []:
                subject_id_by_session[str(session_id)] = subject_id

        # Per-column total nll and unmasked-sample count.
        per_subject_records: list[dict[str, Any]] = []
        subject_order: list[str] = []
        subject_nll: dict[str, float] = {}
        subject_n: dict[str, int] = {}
        total_nll = 0.0
        total_n = 0
        for column_index, session_id in enumerate(eval_session_ids):
            subject_id = subject_id_by_session.get(session_id, "unknown")
            column_ys = ys_eval[:, column_index : column_index + 1, :]
            column_yhat = yhat_eval[:, column_index : column_index + 1, :]
            nll, n_unmasked = rnn_utils.categorical_neg_log_likelihood(
                column_ys, column_yhat
            )
            nll = float(nll)
            n_unmasked = int(n_unmasked)
            session_likelihood = (
                float(np.exp(-nll / n_unmasked)) if n_unmasked > 0 else float("nan")
            )
            per_subject_records.append(
                {
                    "heldout_subject_id": subject_id,
                    "ses_idx": session_id,
                    "split": "eval",
                    "n_trials": n_unmasked,
                    "likelihood": session_likelihood,
                }
            )
            if subject_id not in subject_nll:
                subject_order.append(subject_id)
                subject_nll[subject_id] = 0.0
                subject_n[subject_id] = 0
            subject_nll[subject_id] += nll
            subject_n[subject_id] += n_unmasked
            total_nll += nll
            total_n += n_unmasked

        for subject_id in subject_order:
            n_subject = subject_n[subject_id]
            subject_likelihood = (
                float(np.exp(-subject_nll[subject_id] / n_subject))
                if n_subject > 0
                else float("nan")
            )
            per_subject_records.append(
                {
                    "heldout_subject_id": subject_id,
                    "ses_idx": None,
                    "split": "eval",
                    "n_trials": int(n_subject),
                    "likelihood": subject_likelihood,
                }
            )

        # Correctness guard: the trial-count-weighted aggregate over ALL eval trials must
        # reproduce the existing aggregate eval_likelihood scalar.
        aggregate_likelihood = (
            float(np.exp(-total_nll / total_n)) if total_n > 0 else float("nan")
        )
        if total_n > 0 and not np.isclose(
            aggregate_likelihood, float(eval_likelihood), rtol=0.0, atol=rtol
        ):
            logger.warning(
                "Per-subject eval-likelihood decomposition does not reproduce the "
                "aggregate eval_likelihood (decomposed=%.6f vs aggregate=%.6f, atol=%g); "
                "the per-subject breakdown may not be faithful.",
                aggregate_likelihood,
                float(eval_likelihood),
                rtol,
            )

        return {
            "eval_likelihood_aggregate": float(eval_likelihood),
            "n_heldout_subjects": int(len(subject_order)),
            "records": per_subject_records,
        }
    except Exception as exc:  # never crash the run or the aggregate metric
        logger.warning("Per-subject eval-likelihood decomposition failed: %s", exc)
        return None


def _log_per_subject_likelihood_to_wandb(
    *,
    wandb_run: Any,
    per_subject_likelihood: Mapping[str, Any],
) -> None:
    """Log a compact per-subject eval-likelihood table to the held-out W&B run."""
    try:
        import wandb
    except ModuleNotFoundError:
        logger.warning("wandb is unavailable; skipping per-subject likelihood table.")
        return
    try:
        subject_rows = [
            row
            for row in per_subject_likelihood.get("records", [])
            if row.get("ses_idx") is None
        ]
        table = wandb.Table(columns=["heldout_subject_id", "n_trials", "eval_likelihood"])
        for row in subject_rows:
            table.add_data(
                str(row.get("heldout_subject_id")),
                int(row.get("n_trials", 0)),
                float(row.get("likelihood")),
            )
        wandb_run.log({"heldout/per_subject_likelihood": table})
    except Exception as exc:
        logger.warning("Failed to log per-subject likelihood table to W&B: %s", exc)


def _evaluate_checkpoint(
    *,
    model_type: str,
    trainer: Any,
    params: Any,
    make_train_network: Any,
    make_eval_network: Any,
    bundle: DatasetBundle,
    source_run: Any,
    fine_tune_cfg: Mapping[str, Any],
    checkpoint_dir: Path,
    step: int,
    loss_name: str,
    loss_param: Any,
    wandb_run: Any | None = None,
    wandb_key_prefix: str = "checkpoint",
    wandb_step: int | None = None,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    effective_wandb_step = int(wandb_step) if wandb_step is not None else int(step)
    _save_params(checkpoint_dir / "params.json", params)

    dataset_full = bundle.extras["dataset"]
    dataset_train = bundle.train_set
    dataset_eval = bundle.eval_set
    ignore_policy = str(bundle.metadata.get("ignore_policy", "exclude"))

    _train = dataset_train.get_all()
    xs_train, ys_train = _train["xs"], _train["ys"]
    _eval = dataset_eval.get_all()
    xs_eval, ys_eval = _eval["xs"], _eval["ys"]
    xs_full = dataset_full.get_all()["xs"]

    yhat_train_for_loss, _ = rnn_utils.eval_network(make_train_network, params, xs_train)
    yhat_eval_for_loss, _ = rnn_utils.eval_network(make_train_network, params, xs_eval)
    train_loss = _compute_supervised_loss_from_outputs(
        np.asarray(yhat_train_for_loss),
        ys_train,
        loss_name=loss_name,
        loss_param=loss_param,
        n_action_logits=_infer_n_action_logits(
            dataset_train,
            np.asarray(yhat_train_for_loss),
            ignore_policy=ignore_policy,
        ),
    )
    eval_loss = _compute_supervised_loss_from_outputs(
        np.asarray(yhat_eval_for_loss),
        ys_eval,
        loss_name=loss_name,
        loss_param=loss_param,
        n_action_logits=_infer_n_action_logits(
            dataset_eval,
            np.asarray(yhat_eval_for_loss),
            ignore_policy=ignore_policy,
        ),
    )

    yhat_train_eval, _ = rnn_utils.eval_network(make_eval_network, params, xs_train)
    yhat_eval_eval, _ = rnn_utils.eval_network(make_eval_network, params, xs_eval)
    train_n_action_logits = _infer_n_action_logits(
        dataset_train,
        np.asarray(yhat_train_eval),
        ignore_policy=ignore_policy,
    )
    eval_n_action_logits = _infer_n_action_logits(
        dataset_eval,
        np.asarray(yhat_eval_eval),
        ignore_policy=ignore_policy,
    )
    train_likelihood = float(
        rnn_utils.normalized_likelihood(
            ys_train,
            np.asarray(yhat_train_eval)[:, :, :train_n_action_logits],
        )
    )
    eval_likelihood = float(
        rnn_utils.normalized_likelihood(
            ys_eval,
            np.asarray(yhat_eval_eval)[:, :, :eval_n_action_logits],
        )
    )

    yhat_full_eval, network_states_full = rnn_utils.eval_network(
        make_eval_network,
        params,
        xs_full,
    )
    if model_type == "gru":
        output_df = add_gru_model_results(
            bundle.raw.copy(),
            np.asarray(network_states_full),
            np.asarray(yhat_full_eval),
            ignore_policy=ignore_policy,
        )
    else:
        output_df = dl.add_model_results(
            bundle.raw.copy(),
            np.asarray(network_states_full),
            yhat_full_eval,
            ignore_policy=ignore_policy,
        )

    record: dict[str, Any] = {
        "step": int(step),
        "params_path": str(checkpoint_dir / "params.json"),
        "train_loss": float(train_loss),
        "eval_loss": float(eval_loss),
        "train_likelihood": float(train_likelihood),
        "eval_likelihood": float(eval_likelihood),
    }
    # For the 3-way (ignore-included) head, also record the engagement-conditional
    # L/R held-out likelihood (comparable to the 2-way study's held-out number) and
    # the engage-vs-ignore held-out calibration. No-op for a 2-way head.
    eval_likelihood_decomp = _threeway_likelihood_decomposition(
        ys_eval,
        np.asarray(yhat_eval_eval)[:, :, :eval_n_action_logits],
        n_action_logits=eval_n_action_logits,
    )
    for _decomp_key, _decomp_val in eval_likelihood_decomp.items():
        record[_decomp_key] = float(_decomp_val)
    record["per_subject_likelihood"] = _compute_per_subject_eval_likelihood(
        ys_eval=ys_eval,
        yhat_eval=np.asarray(yhat_eval_eval)[:, :, :eval_n_action_logits],
        eval_likelihood=eval_likelihood,
        metadata=bundle.metadata,
    )
    record["plot_paths"] = _save_state_space_figures(
        trainer=trainer,
        params=params,
        bundle=bundle,
        checkpoint_dir=checkpoint_dir,
    )
    if wandb_run is not None:
        _log_checkpoint_plot_paths_to_wandb(
            wandb_run=wandb_run,
            plot_paths=record["plot_paths"],
            step=effective_wandb_step,
            key_prefix=wandb_key_prefix,
        )

    plot_examples_every_n = int(fine_tune_cfg.get("checkpoint_plot_split_examples_every_n", 0))
    save_output_df_every_n = int(fine_tune_cfg.get("checkpoint_save_output_df_every_n", 0))
    should_plot_examples = (
        plot_examples_every_n > 0
        and (step == 0 or step % plot_examples_every_n == 0)
    )
    should_save_output_df = (
        save_output_df_every_n > 0
        and (step == 0 or step % save_output_df_every_n == 0)
    )
    if should_save_output_df:
        output_df_path = checkpoint_dir / "output_df.csv"
        output_df.to_csv(output_df_path, index=False)
        record["output_df_path"] = str(output_df_path)
    if should_plot_examples:
        n_action_logits_full = _infer_n_action_logits(
            dataset_full,
            np.asarray(yhat_full_eval),
            ignore_policy=ignore_policy,
        )
        split_summaries = trainer._generate_split_examples(
            output_dir=checkpoint_dir,
            output_df=output_df,
            network_states_full=np.asarray(network_states_full),
            yhat_full=np.asarray(yhat_full_eval),
            params=params,
            metadata=dict(bundle.metadata),
            n_action_logits=n_action_logits_full,
            wandb_run=wandb_run,
            log_scope=f"Checkpoint step {step}",
            wandb_step=effective_wandb_step,
            wandb_key_prefix=wandb_key_prefix,
        )
        if not bool(fine_tune_cfg.get("keep_media_files", True)):
            trainer._cleanup_split_example_media(split_summaries)
        record["split_examples"] = split_summaries
    return record


def _build_final_target_predictions(
    *,
    model_type: str,
    params: Any,
    make_eval_network: Any,
    bundle: DatasetBundle,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Replay the full target sequence and score only immutable test trials."""
    ignore_policy = str(bundle.metadata.get("ignore_policy", "exclude"))
    if ignore_policy != "exclude":
        logger.info(
            "Skipping binary target-transfer trial export for ignore_policy=%s.",
            ignore_policy,
        )
        return None, None
    dataset_full = bundle.extras["dataset"]
    xs_full = dataset_full.get_all()["xs"]
    yhat_full, network_states_full = rnn_utils.eval_network(
        make_eval_network,
        params,
        xs_full,
    )
    if model_type == "gru":
        output_df = add_gru_model_results(
            bundle.raw.copy(),
            np.asarray(network_states_full),
            np.asarray(yhat_full),
            ignore_policy=ignore_policy,
        )
        probability_choice_1 = output_df["choice_prob_1"].to_numpy(dtype=float)
    else:
        output_df = dl.add_model_results(
            bundle.raw.copy(),
            np.asarray(network_states_full),
            yhat_full,
            ignore_policy=ignore_policy,
        )
        logits = output_df[["logit(left)", "logit(right)"]].to_numpy(dtype=float)
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probability_choice_1 = probabilities[:, 1]

    predictions = build_binary_trial_predictions(
        output_df,
        bundle.metadata,
        probability_choice_1=probability_choice_1,
        model=model_type,
    )
    return predictions, summarize_binary_trial_predictions(predictions)


def _build_trial_probability_table(
    *,
    params: Any,
    make_eval_network: Any,
    bundle: DatasetBundle,
) -> pd.DataFrame:
    """Per-trial GRU choice probabilities for every held-out trial the model sees.

    One row per modelled trial (under ``exclude`` the ignored trials are not in the
    model's sequence, so they have no row and ``p_ignore`` is NaN). ``split`` is
    ``eval`` for scored trials (``target_test_mask``) and ``train`` otherwise.
    """
    ignore_policy = str(bundle.metadata.get("ignore_policy", "exclude"))
    xs_full = bundle.extras["dataset"].get_all()["xs"]
    yhat_full, network_states_full = rnn_utils.eval_network(make_eval_network, params, xs_full)
    n_action_logits = _infer_n_action_logits(
        bundle.extras["dataset"], np.asarray(yhat_full), ignore_policy=ignore_policy
    )
    output_df = add_gru_model_results(
        bundle.raw.copy(),
        np.asarray(network_states_full),
        np.asarray(yhat_full)[:, :, :n_action_logits],
        ignore_policy=ignore_policy,
    )
    # Scored (eval) membership uses the target-test rule: eval sessions, or the test suffix
    # of a within-session prefix/suffix split. Train = sessions the fine-tune actually used
    # (none at K=0, where the optimisation is skipped). Anything else is "none".
    eval_mask = target_test_mask(output_df, bundle.metadata)
    tuned_ids = (
        set()
        if int(bundle.metadata.get("adapt_sessions_per_subject", -1)) == 0
        else {str(value) for value in bundle.metadata.get("train_session_ids") or []}
    )
    tuned_mask = output_df["ses_idx"].astype(str).isin(tuned_ids).to_numpy()
    columns = ["subject_id", "ses_idx", "trial", "animal_response"]
    if "source_ses_idx" in output_df.columns:
        columns.insert(2, "source_ses_idx")
    table = output_df[columns].reset_index(drop=True)
    logits = output_df[[f"choice_logit_{i}" for i in range(n_action_logits)]].to_numpy(
        dtype=np.float64
    )
    log_probabilities, ll_observed, ll_lr = _observed_log_likelihoods(
        logits, output_df["animal_response"].to_numpy(dtype=int)
    )
    probabilities = np.exp(log_probabilities)
    table["p_left"] = probabilities[:, 0]
    table["p_right"] = probabilities[:, 1]
    table["p_ignore"] = probabilities[:, 2] if n_action_logits >= 3 else np.nan
    table["ll_observed"] = ll_observed
    table["ll_lr"] = ll_lr
    table["split"] = np.where(eval_mask, "eval", np.where(tuned_mask, "train", "none"))
    return table


def _observed_log_likelihoods(
    logits: np.ndarray, responses: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable float64 log-softmax per trial.

    Returns all classes' log-probabilities, the log-probability of the observed response
    over the model's classes, and the left/right log-probability renormalised over
    {left, right} (NaN on ignored trials) -- the engagement-conditional quantity of
    ``_threeway_likelihood_decomposition``. Working in logs keeps a tiny probability from
    underflowing to zero.
    """
    logits = np.asarray(logits, dtype=np.float64)
    responses = np.asarray(responses, dtype=int)
    rows = np.arange(len(responses))

    def _log_softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    log_probabilities = _log_softmax(logits)
    ll_observed = log_probabilities[rows, responses]
    engaged = responses != 2
    log_lr = _log_softmax(logits[:, :2])
    ll_lr = np.where(engaged, log_lr[rows, np.where(engaged, responses, 0)], np.nan)
    return log_probabilities, ll_observed, ll_lr


def _trial_table_normalized_likelihood(table: pd.DataFrame, *, split: str) -> float:
    """exp(mean log p(observed response)) over one split of a trial probability table."""
    return float(np.exp(np.mean(table.loc[table["split"] == split, "ll_observed"])))


def _plot_checkpoint_metric_curve(
    *,
    checkpoint_records: Sequence[Mapping[str, Any]],
    metric_key_train: str,
    metric_key_eval: str,
    title: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> Path:
    if not checkpoint_records:
        raise ValueError("checkpoint_records must not be empty.")
    steps = np.asarray([int(record["step"]) for record in checkpoint_records], dtype=int)
    train_values = np.asarray(
        [float(record[metric_key_train]) for record in checkpoint_records],
        dtype=float,
    )
    eval_values = np.asarray(
        [float(record[metric_key_eval]) for record in checkpoint_records],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if log_scale:
        ax.semilogy(steps, train_values, marker="o", color="black", label="Train")
        ax.semilogy(steps, eval_values, marker="o", color="tab:red", label="Eval")
    else:
        ax.plot(steps, train_values, marker="o", color="black", label="Train")
        ax.plot(steps, eval_values, marker="o", color="tab:red", label="Eval")
    ax.set_xlabel("Fine-tuning Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_model_config(
    *,
    model_type: str,
    model_config: dict[str, Any],
    output_dir: Path,
    metadata: Mapping[str, Any],
) -> Path:
    saved_config = copy.deepcopy(model_config)
    if model_type == "disrnn":
        saved_config["max_n_subjects"] = int(metadata["num_subjects"])
        if metadata.get("session_max_index_by_subject_index") is not None:
            saved_config["session_max_index_by_subject_index"] = list(
                metadata["session_max_index_by_subject_index"]
            )
        path = output_dir / _DISRNN_CONFIG_NAME
    else:
        path = output_dir / _GRU_CONFIG_NAME
    _save_json(path, saved_config)
    return path


def _final_metrics_summary(checkpoint_records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not checkpoint_records:
        return {}
    final_record = checkpoint_records[-1]
    summary = {
        "train_loss": float(final_record["train_loss"]),
        "eval_loss": float(final_record["eval_loss"]),
        "train_likelihood": float(final_record["train_likelihood"]),
        "eval_likelihood": float(final_record["eval_likelihood"]),
    }
    # Propagate the 3-way decomposition (engagement-conditional L/R, engage-vs-
    # ignore) plus the ignore-class classification metrics (precision/recall/F1/
    # PR-AUC at the ~5% ignore base rate) to heldout/final/* if present. No-op
    # for a 2-way head.
    _decomp_keys = (
        "eval_likelihood_LR_engaged",
        "eval_likelihood_engage",
        "engage_ignore_base_rate",
        "engage_ignore_precision",
        "engage_ignore_recall",
        "engage_ignore_f1",
        "engage_ignore_pr_auc",
    )
    for _k in _decomp_keys:
        if _k in final_record:
            summary[_k] = float(final_record[_k])
    return summary


def run_heldout_subject_finetuning_from_config(
    config_source: str | Path | Mapping[str, Any] | DictConfig,
    *,
    output_root: str | Path | None = None,
    target_bundle: DatasetBundle | None = None,
    wandb_run: Any | None = None,
    wandb_key_prefix: str | None = None,
    wandb_step_offset: int | None = None,
) -> dict[str, Any]:
    """Fine-tune target subject embeddings for a trained multisubject GRU/disRNN.

    The target may be supplied directly as ``target_bundle`` or instantiated from
    ``target_data`` in the config. If neither is present, the legacy AIND held-out
    subject selection is used.

    When ``wandb_run`` is provided (e.g. the still-open training run in
    ``run_capsule``), held-out metrics are logged into that run instead of a new
    one: keys are namespaced with ``wandb_key_prefix`` (default ``"checkpoint"``)
    and steps are shifted by ``wandb_step_offset`` so they don't collide with the
    training run's step axis. The injected run is never renamed or finished here.
    With all three left ``None`` the behavior is identical to the standalone tool.
    """
    config = _load_config(config_source)
    source_cfg = _normalize_mapping(config.get("source_run"))
    source_model_dir = source_cfg.get("model_dir")
    if source_model_dir is None:
        raise ValueError("Config must define source_run.model_dir.")

    resolved_source_run = resolve_model_run(
        source_model_dir,
        split="train",
        checkpoint_policy=str(source_cfg.get("checkpoint_policy", "best_eval")),
    )
    if not bool(resolved_source_run.multisubject):
        raise ValueError("Held-out fine-tuning currently supports multisubject runs only.")
    if str(resolved_source_run.model_type).lower() not in {"gru", "disrnn"}:
        raise ValueError(
            "Held-out fine-tuning currently supports only GRU and disRNN source runs."
        )

    resolved_config = _resolve_runtime_config(
        config,
        source_run=resolved_source_run,
        output_root_override=output_root,
    )
    heldout_selector = _normalize_mapping(resolved_config.get("heldout_subjects"))
    target_data_cfg = resolved_config.get("target_data")
    if target_bundle is not None and target_data_cfg is not None:
        raise ValueError("Supply target_bundle or config target_data, not both.")
    if target_data_cfg is not None:
        target_bundle = _load_configured_target_bundle(
            _normalize_mapping(target_data_cfg)
        )
    run_selector = heldout_selector
    if target_bundle is not None:
        run_selector = {
            "dataset_id": target_bundle.metadata.get("dataset_id") or "external_target"
        }
    run_dir = _resolve_output_run_dir(
        resolved_config=resolved_config,
        source_run=resolved_source_run,
        heldout_selector=run_selector,
    )
    outputs_dir = run_dir / "outputs"
    checkpoints_root = outputs_dir / "checkpoints"
    figures_dir = outputs_dir / "figures"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(
        config=OmegaConf.create(resolved_config),
        f=str(run_dir / "resolved_config.yaml"),
        resolve=True,
    )
    _save_json(run_dir / "resolved_source_run.json", resolved_source_run.to_dict())
    external_wandb_run = wandb_run is not None
    if not external_wandb_run:
        wandb_run = _maybe_start_wandb_run(
            resolved_config=resolved_config,
            run_dir=run_dir,
            source_run=resolved_source_run,
        )
    wandb_log_key_prefix = wandb_key_prefix or "checkpoint"
    wandb_log_step_offset = int(wandb_step_offset) if wandb_step_offset is not None else 0
    logger.info(
        "Starting held-out subject fine-tuning: source_model_dir=%s model_type=%s checkpoint=%s "
        "heldout_ids=%s curricula=%s heldout_every_n=%s output_dir=%s",
        resolved_source_run.model_dir,
        resolved_source_run.model_type,
        resolved_source_run.checkpoint_label,
        heldout_selector.get("test_subject_ids"),
        heldout_selector.get("curricula"),
        heldout_selector.get("heldout_every_n"),
        run_dir,
    )

    try:
        if resolved_source_run.model_type == "gru":
            trainer, architecture, source_training_cfg, model_config, source_model_cfg = (
                _make_gru_runtime(
                    source_run=resolved_source_run,
                    output_dir=outputs_dir,
                )
            )
        else:
            trainer, architecture, source_training_cfg, model_config, source_model_cfg = (
                _make_disrnn_runtime(
                    source_run=resolved_source_run,
                    output_dir=outputs_dir,
                )
            )

        bundle, heldout_subject_ids, appended_subject_indices, _, _ = _build_global_heldout_bundle(
            source_run=resolved_source_run,
            heldout_selector=heldout_selector,
            fine_tune_cfg=resolved_config["heldout_finetuning"],
            architecture=architecture,
            target_bundle=target_bundle,
        )
        if wandb_run is not None and not external_wandb_run:
            _update_wandb_run_name_for_heldout_subjects(
                wandb_run=wandb_run,
                resolved_config=resolved_config,
                source_run=resolved_source_run,
                heldout_subject_ids=heldout_subject_ids,
            )
        source_params = _load_params(resolved_source_run.params_path)
        params = expand_local_multisubject_params(
            source_params,
            n_new_subjects=len(appended_subject_indices),
            init="mean",
        )
        parameter_update_mask = build_subject_embedding_update_mask(
            params,
            trainable_subject_indices=appended_subject_indices,
        )

        ignore_policy = str(bundle.metadata.get("ignore_policy", "exclude"))
        if resolved_source_run.model_type == "gru":
            output_size = int(
                _normalize_mapping(model_config).get(
                    "output_size",
                    2 if ignore_policy == "exclude" else 3,
                )
            )
            session_cfg = resolve_session_conditioning_from_architecture(
                architecture=architecture,
                metadata=bundle.metadata,
                multisubject=True,
                max_n_subjects=int(bundle.metadata["num_subjects"]),
                subject_embedding_size=int(architecture["subject_embedding_size"]),
                context="Held-out fine-tuning GRU",
            )
            gru_make_network = make_gru_network(
                hidden_size=int(architecture["hidden_size"]),
                output_size=output_size,
                multisubject=True,
                max_n_subjects=int(bundle.metadata["num_subjects"]),
                subject_embedding_size=int(architecture["subject_embedding_size"]),
                subject_embedding_init=str(architecture.get("subject_embedding_init", "zeros")),
                session_encoding_type=str(session_cfg["session_encoding_type"]),
                session_integration_type=str(session_cfg["session_integration_type"]),
                session_fourier_k=int(session_cfg["session_fourier_k"]),
                session_delta_n_layers=int(session_cfg["session_delta_n_layers"]),
                session_delta_hidden_size=int(session_cfg["session_delta_hidden_size"]),
                session_max_index_by_subject_index=list(
                    bundle.metadata.get("session_max_index_by_subject_index") or []
                ),
            )
            make_train_network = gru_make_network
            make_eval_network = gru_make_network
            runtime_model_config = model_config
        else:
            disrnn_config, noiseless_config = trainer._build_network_configs(
                dataset=bundle.extras["dataset"],
                ignore_policy=ignore_policy,
                metadata=bundle.metadata,
            )
            make_train_network = trainer._make_network_factory(
                disrnn_config, multisubject=True
            )
            make_eval_network = trainer._make_network_factory(
                noiseless_config, multisubject=True
            )
            runtime_model_config = asdict(disrnn_config)

        loss_name = str(source_training_cfg.get("loss", "categorical"))
        loss_param = source_training_cfg.get("loss_param", 1)
        n_action_logits = _infer_n_action_logits(
            bundle.train_set,
            ignore_policy=ignore_policy,
        )
        optimizer = optax.adam(float(resolved_config["heldout_finetuning"]["lr"]))
        opt_state = None
        rng = jax.random.PRNGKey(int(resolved_config["seed"]))
        checkpoint_every_n_steps = int(
            resolved_config["heldout_finetuning"]["checkpoint_every_n_steps"]
        )
        total_steps = int(resolved_config["heldout_finetuning"]["n_steps"])
        # Zero-shot (K=0): no adapt sessions, so skip all gradient steps and run only the
        # step-0 init eval + per_subject logging (embedding stays at its init).
        if int(bundle.metadata.get("adapt_sessions_per_subject", -1)) == 0:
            logger.info(
                "No selected adaptation observations; skipping gradient steps and "
                "running zero-shot initialization evaluation only."
            )
            total_steps = 0
        evaluation_steps = [0]
        if checkpoint_every_n_steps <= 0:
            if total_steps > 0:
                evaluation_steps.append(total_steps)
        else:
            next_step = checkpoint_every_n_steps
            while next_step < total_steps:
                evaluation_steps.append(next_step)
                next_step += checkpoint_every_n_steps
            if total_steps not in evaluation_steps:
                evaluation_steps.append(total_steps)

        checkpoint_records: list[dict[str, Any]] = []
        optimization_history = {
            "training_loss": [],
            "validation_loss": [],
        }
        previous_step = 0
        for step in evaluation_steps:
            if step > previous_step:
                chunk_key, rng = jax.random.split(rng)
                params, opt_state, chunk_losses = train_network_with_session_regularization(
                    make_train_network,
                    bundle.train_set,
                    bundle.eval_set,
                    loss=loss_name,
                    loss_param=loss_param,
                    n_action_logits=int(n_action_logits),
                    session_regularization_apply=None,
                    session_regularization_scale=0.0,
                    opt=optimizer,
                    parameter_update_mask=parameter_update_mask,
                    params=params,
                    opt_state=opt_state,
                    n_steps=step - previous_step,
                    max_grad_norm=float(resolved_config["heldout_finetuning"]["max_grad_norm"]),
                    random_key=chunk_key,
                    report_progress_by="log",
                    log_losses_every=10,  # mirror training's cadence (was 1 -> per-step spam)
                )
                optimization_history["training_loss"].extend(
                    np.asarray(chunk_losses["training_loss"], dtype=float).tolist()
                )
                optimization_history["validation_loss"].extend(
                    np.asarray(chunk_losses["validation_loss"], dtype=float).tolist()
                )
            checkpoint_wandb_step = wandb_log_step_offset + int(step)
            checkpoint_record = _evaluate_checkpoint(
                model_type=resolved_source_run.model_type,
                trainer=trainer,
                params=params,
                make_train_network=make_train_network,
                make_eval_network=make_eval_network,
                bundle=bundle,
                source_run=resolved_source_run,
                fine_tune_cfg=resolved_config["heldout_finetuning"],
                checkpoint_dir=checkpoints_root / f"step_{step}",
                step=step,
                loss_name=loss_name,
                loss_param=loss_param,
                wandb_run=wandb_run,
                wandb_key_prefix=wandb_log_key_prefix,
                wandb_step=checkpoint_wandb_step,
            )
            checkpoint_records.append(checkpoint_record)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"{wandb_log_key_prefix}/train_loss": float(
                            checkpoint_record["train_loss"]
                        ),
                        f"{wandb_log_key_prefix}/eval_loss": float(
                            checkpoint_record["eval_loss"]
                        ),
                        f"{wandb_log_key_prefix}/train_likelihood": float(
                            checkpoint_record["train_likelihood"]
                        ),
                        f"{wandb_log_key_prefix}/eval_likelihood": float(
                            checkpoint_record["eval_likelihood"]
                        ),
                        f"{wandb_log_key_prefix}/step": int(step),
                        **{
                            f"{wandb_log_key_prefix}/{_k}": float(checkpoint_record[_k])
                            for _k in (
                                "eval_likelihood_LR_engaged",
                                "eval_likelihood_engage",
                                "engage_ignore_base_rate",
                                "engage_ignore_precision",
                                "engage_ignore_recall",
                                "engage_ignore_f1",
                                "engage_ignore_pr_auc",
                            )
                            if _k in checkpoint_record
                        },
                    },
                    step=checkpoint_wandb_step,
                )
            previous_step = step

        _save_params(outputs_dir / "params.json", params)
        _save_json(outputs_dir / "checkpoint_metrics.json", checkpoint_records)
        test_trial_predictions, test_metrics = _build_final_target_predictions(
            model_type=resolved_source_run.model_type,
            params=params,
            make_eval_network=make_eval_network,
            bundle=bundle,
        )
        test_trial_predictions_path = None
        test_metrics_path = None
        if test_trial_predictions is not None and test_metrics is not None:
            final_eval_likelihood = float(checkpoint_records[-1]["eval_likelihood"])
            if not np.isclose(
                float(test_metrics["normalized_likelihood"]),
                final_eval_likelihood,
                rtol=1e-4,
                atol=1e-6,
            ):
                raise AssertionError(
                    "Canonical target-test likelihood does not match the final "
                    "checkpoint evaluation likelihood: "
                    f"table={test_metrics['normalized_likelihood']}, "
                    f"checkpoint={final_eval_likelihood}."
                )
            test_trial_predictions_path = outputs_dir / "test_trial_predictions.csv"
            test_metrics_path = outputs_dir / "test_metrics.json"
            test_trial_predictions.to_csv(test_trial_predictions_path, index=False)
            _save_json(test_metrics_path, test_metrics)

        trial_predictions_path = None
        if bool(resolved_config["heldout_finetuning"]["export_trial_predictions"]):
            if resolved_source_run.model_type != "gru":
                raise ValueError("export_trial_predictions supports GRU source runs only.")
            trial_table = _build_trial_probability_table(
                params=params,
                make_eval_network=make_eval_network,
                bundle=bundle,
            )
            final_eval_likelihood = float(checkpoint_records[-1]["eval_likelihood"])
            table_eval_likelihood = _trial_table_normalized_likelihood(trial_table, split="eval")
            if not np.isclose(table_eval_likelihood, final_eval_likelihood, rtol=1e-4, atol=1e-6):
                raise AssertionError(
                    "Per-trial export does not reproduce the final checkpoint evaluation "
                    f"likelihood: table={table_eval_likelihood}, checkpoint={final_eval_likelihood}."
                )
            trial_predictions_path = outputs_dir / "trial_predictions.csv.gz"
            trial_table.to_csv(trial_predictions_path, index=False)

        # Per-held-out-subject / per-session eval likelihood decomposition (from the
        # final checkpoint). Tidy artifact + compact W&B table for downstream paired
        # statistics across data-scaling conditions. Add-only; never gates the run.
        final_per_subject_likelihood = (
            checkpoint_records[-1].get("per_subject_likelihood")
            if checkpoint_records
            else None
        )
        if final_per_subject_likelihood is not None:
            _save_json(
                outputs_dir / "per_subject_likelihood.json",
                final_per_subject_likelihood,
            )
            if wandb_run is not None:
                _log_per_subject_likelihood_to_wandb(
                    wandb_run=wandb_run,
                    per_subject_likelihood=final_per_subject_likelihood,
                )
        _save_json(
            checkpoints_root / "index.json",
            {
                "n_steps": int(total_steps),
                "checkpoint_every_n_steps": int(checkpoint_every_n_steps),
                "count": int(len(checkpoint_records)),
                "checkpoints": checkpoint_records,
            },
        )
        _save_json(outputs_dir / "optimization_trace.json", optimization_history)

        model_config_path = _save_model_config(
            model_type=resolved_source_run.model_type,
            model_config=runtime_model_config,
            output_dir=outputs_dir,
            metadata=bundle.metadata,
        )
        subject_artifacts = trainer._save_multisubject_artifacts(
            params=params,
            metadata=bundle.metadata,
            raw_df=bundle.raw,
        )

        loss_curve_path = _plot_checkpoint_metric_curve(
            checkpoint_records=checkpoint_records,
            metric_key_train="train_loss",
            metric_key_eval="eval_loss",
            title="Held-out Subject Fine-tuning Loss",
            ylabel="Loss",
            output_path=figures_dir / "loss_across_checkpoints.png",
            log_scale=True,
        )
        likelihood_curve_path = _plot_checkpoint_metric_curve(
            checkpoint_records=checkpoint_records,
            metric_key_train="train_likelihood",
            metric_key_eval="eval_likelihood",
            title="Held-out Subject Fine-tuning Likelihood",
            ylabel="Normalized Likelihood",
            output_path=figures_dir / "likelihood_across_checkpoints.png",
            log_scale=False,
        )

        summary = {
            "output_dir": str(run_dir),
            "outputs_dir": str(outputs_dir),
            "source_model_dir": str(resolved_source_run.model_dir),
            "source_checkpoint_policy": str(resolved_source_run.checkpoint_policy),
            "source_checkpoint_step": resolved_source_run.checkpoint_step,
            "source_checkpoint_label": str(resolved_source_run.checkpoint_label),
            "source_params_path": str(resolved_source_run.params_path),
            "model_type": str(resolved_source_run.model_type),
            "multisubject": True,
            "seed": int(resolved_config["seed"]),
            "heldout_subject_ids": list(heldout_subject_ids),
            "trainable_subject_indices": [int(index) for index in appended_subject_indices],
            "n_steps": int(total_steps),
            "lr": float(resolved_config["heldout_finetuning"]["lr"]),
            "checkpoint_every_n_steps": int(checkpoint_every_n_steps),
            "selection_policy": str(
                resolved_config["heldout_finetuning"]["selection_policy"]
            ),
            "target_test_used_for_selection": False,
            "metrics": _final_metrics_summary(checkpoint_records),
            "test_metrics": test_metrics,
            "artifacts": {
                "resolved_config": str(run_dir / "resolved_config.yaml"),
                "resolved_source_run": str(run_dir / "resolved_source_run.json"),
                "params": str(outputs_dir / "params.json"),
                "checkpoint_metrics": str(outputs_dir / "checkpoint_metrics.json"),
                "test_trial_predictions": (
                    str(test_trial_predictions_path)
                    if test_trial_predictions_path is not None
                    else None
                ),
                "test_metrics": (
                    str(test_metrics_path) if test_metrics_path is not None else None
                ),
                "trial_predictions": (
                    str(trial_predictions_path) if trial_predictions_path is not None else None
                ),
                "optimization_trace": str(outputs_dir / "optimization_trace.json"),
                "model_config": str(model_config_path),
                "subject_index_map": subject_artifacts["subject_index_map"],
                "subject_embeddings": subject_artifacts["subject_embeddings"],
                "multisubject_metadata": subject_artifacts["multisubject_metadata"],
                "loss_curve": str(loss_curve_path),
                "likelihood_curve": str(likelihood_curve_path),
            },
        }
        if "session_context_map" in subject_artifacts:
            summary["artifacts"]["session_context_map"] = subject_artifacts[
                "session_context_map"
            ]
        _save_json(outputs_dir / "output_summary.json", summary)
        if wandb_run is not None:
            summary_prefix = (
                f"{wandb_log_key_prefix}/final" if external_wandb_run else "final"
            )
            for key, value in summary["metrics"].items():
                wandb_run.summary[f"{summary_prefix}/{key}"] = float(value)
            if not external_wandb_run:
                wandb_run.summary["output_dir"] = str(run_dir)
            # Log the loss/likelihood-over-checkpoints curves as images, mirroring
            # the training process's fig/validation_loss_curve.
            try:
                import wandb

                wandb_run.log(
                    {
                        f"{wandb_log_key_prefix}/fig/loss_curve": wandb.Image(
                            str(loss_curve_path)
                        ),
                        f"{wandb_log_key_prefix}/fig/likelihood_curve": wandb.Image(
                            str(likelihood_curve_path)
                        ),
                    }
                )
            except Exception as exc:
                logger.warning("Held-out curve image logging failed: %s", exc)
        logger.info(
            "Completed held-out subject fine-tuning: output_dir=%s summary=%s "
            "checkpoint_metrics=%s final_train_likelihood=%s final_eval_likelihood=%s",
            run_dir,
            outputs_dir / "output_summary.json",
            outputs_dir / "checkpoint_metrics.json",
            summary["metrics"].get("train_likelihood"),
            summary["metrics"].get("eval_likelihood"),
        )

        return {
            "output_dir": str(run_dir),
            "outputs_dir": str(outputs_dir),
            "summary_path": str(outputs_dir / "output_summary.json"),
            "checkpoint_metrics_path": str(outputs_dir / "checkpoint_metrics.json"),
            "test_trial_predictions_path": (
                str(test_trial_predictions_path)
                if test_trial_predictions_path is not None
                else None
            ),
            "test_metrics_path": (
                str(test_metrics_path) if test_metrics_path is not None else None
            ),
            "trial_predictions_path": (
                str(trial_predictions_path) if trial_predictions_path is not None else None
            ),
            "params_path": str(outputs_dir / "params.json"),
            "subject_index_map_path": subject_artifacts["subject_index_map"],
            "subject_embeddings_path": subject_artifacts["subject_embeddings"],
            "model_config_path": str(model_config_path),
            "loss_curve_path": str(loss_curve_path),
            "likelihood_curve_path": str(likelihood_curve_path),
        }
    finally:
        if wandb_run is not None and not external_wandb_run:
            wandb_run.finish()
