"""Cross-model next-trial likelihood comparison helpers.

This module evaluates trained GRU, disRNN, and baseline RL runs directly on the
real behavioral datasets used for training. Unlike the generative post-training
analysis pipeline, it does not simulate rollouts; it scores next-trial choice
likelihood on the train, eval, combined, and optionally held-out datasets.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from post_training_analysis.generative_analysis import (
    _load_structured_file,
    resolve_model_run,
)

logger = logging.getLogger(__name__)

_SUPPORTED_MODEL_TYPES = {"baseline_rl", "gru", "disrnn"}
_SPLIT_ORDER = ("train", "eval", "combined", "heldout_test")
_SESSION_METRIC_COLUMNS = [
    "model_index",
    "model_label",
    "model_dir",
    "model_type",
    "multisubject",
    "split",
    "subject_id",
    "session_id",
    "curriculum_name",
    "total_log_likelihood",
    "total_trials",
    "likelihood",
]
_SUBJECT_METRIC_COLUMNS = [
    "model_index",
    "model_label",
    "model_dir",
    "model_type",
    "multisubject",
    "split",
    "subject_id",
    "subject_index",
    "curriculum_name",
    "num_sessions",
    "total_log_likelihood",
    "total_trials",
    "likelihood",
]
_POOLED_METRIC_COLUMNS = [
    "model_index",
    "model_label",
    "model_dir",
    "model_type",
    "multisubject",
    "split",
    "status",
    "skip_reason",
    "num_sessions",
    "num_subjects",
    "total_log_likelihood",
    "total_trials",
    "pooled_trial_likelihood",
    "session_sem",
]
_NON_NUMERIC_METRIC_COLUMNS = {
    "model_label",
    "model_dir",
    "model_type",
    "split",
    "curriculum_name",
    "subject_id",
    "session_id",
    "status",
    "skip_reason",
}
_DEFAULT_CURRICULUM_COLORS = {
    "Mixed": "#7f7f7f",
    "Unknown": "#bdbdbd",
}
_MODEL_TYPE_COLORS = {
    "baseline_rl": "#59a14f",
    "gru": "#4e79a7",
    "disrnn": "#f28e2b",
}
_BAR_SPLIT_TITLES = {
    "train": "Train",
    "eval": "Eval",
    "combined": "Combined",
    "heldout_test": "Held-out Test",
}


@dataclass(frozen=True)
class ResolvedLikelihoodRun:
    """Resolved artifacts and metadata for predictive-likelihood comparison."""

    model_dir: str
    inputs_path: str
    outputs_dir: str
    model_type: str
    model_label: str
    model_index: int
    multisubject: bool
    seed: int | None
    checkpoint_policy: str | None
    checkpoint_step: int | None
    checkpoint_label: str | None
    params_path: str | None
    baseline_output_path: str | None
    artifact_selection_reason: str | None
    session_conditioning_enabled: bool = False
    session_conditioning_encoding_type: str = "none"
    run_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(asdict(self))


def run_prediction_likelihood_comparison(
    model_dirs: Sequence[str | Path],
    *,
    checkpoint_policy: str = "best_eval",
    output_dir: str | Path | None = None,
    model_labels: Sequence[str] | None = None,
    include_heldout: bool = True,
    precomputed_session_metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare next-trial prediction likelihood across trained model runs."""

    if not model_dirs:
        raise ValueError("model_dirs must contain at least one model directory.")

    resolved_model_dirs = [
        str(Path(model_dir).expanduser().resolve()) for model_dir in model_dirs
    ]
    resolved_labels = _deduplicate_model_labels(
        resolved_model_dirs,
        model_labels=model_labels,
    )

    resolved_runs = [
        _resolve_likelihood_run(
            model_dir=model_dir,
            model_label=model_label,
            model_index=model_index,
            checkpoint_policy=checkpoint_policy,
        )
        for model_index, (model_dir, model_label) in enumerate(
            zip(resolved_model_dirs, resolved_labels)
        )
    ]

    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(resolved_model_dirs[0]).resolve() / "outputs" / "likelihood_comparison"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    precomputed_session_metrics_df = _load_precomputed_session_metrics(
        precomputed_session_metrics_path
    )

    pooled_rows: list[dict[str, Any]] = []
    session_df_chunks: list[Any] = []
    subject_df_chunks: list[Any] = []
    model_summaries: list[dict[str, Any]] = []

    for run in resolved_runs:
        reused_session_metrics_df = _reuse_precomputed_session_metrics_for_run(
            run,
            precomputed_session_metrics_df=precomputed_session_metrics_df,
            include_heldout=include_heldout,
        )
        if reused_session_metrics_df is not None:
            split_results = _split_results_from_session_metrics(
                run,
                session_metrics_df=reused_session_metrics_df,
            )
            result_source = "precomputed_session_metrics"
        else:
            hydra_config, bundle = _load_training_bundle_for_run(run)
            split_results = _evaluate_resolved_run_splits(
                run,
                bundle,
                hydra_config=hydra_config,
                include_heldout=include_heldout,
            )
            result_source = "evaluated"

        model_summary = {
            "model_dir": run.model_dir,
            "model_label": run.model_label,
            "model_type": run.model_type,
            "multisubject": run.multisubject,
            "session_conditioning_enabled": bool(run.session_conditioning_enabled),
            "session_conditioning_encoding_type": run.session_conditioning_encoding_type,
            "result_source": result_source,
            "splits": {},
        }

        for split_name in _SPLIT_ORDER:
            split_result = split_results.get(split_name)
            if split_result is None:
                continue
            pooled_rows.append(dict(split_result["pooled_metric"]))
            session_metrics_df = split_result["session_metrics"]
            if not session_metrics_df.empty:
                session_df_chunks.append(session_metrics_df.copy())
            subject_metrics_df = split_result["subject_metrics"]
            if not subject_metrics_df.empty:
                subject_df_chunks.append(subject_metrics_df.copy())
            model_summary["splits"][split_name] = {
                "status": split_result["pooled_metric"]["status"],
                "skip_reason": split_result["pooled_metric"]["skip_reason"],
                "total_trials": split_result["pooled_metric"]["total_trials"],
                "pooled_trial_likelihood": split_result["pooled_metric"][
                    "pooled_trial_likelihood"
                ],
            }

        model_summaries.append(model_summary)

    pooled_metrics_df = _make_dataframe(pooled_rows, _POOLED_METRIC_COLUMNS)
    session_metrics_df = _concat_metric_frames(session_df_chunks, _SESSION_METRIC_COLUMNS)
    subject_metrics_df = _concat_metric_frames(subject_df_chunks, _SUBJECT_METRIC_COLUMNS)

    pooled_metrics_df = _sort_metrics_dataframe(
        pooled_metrics_df,
        split_column="split",
        sort_columns=("model_index", "split"),
    )
    session_metrics_df = _sort_metrics_dataframe(
        session_metrics_df,
        split_column="split",
        sort_columns=("model_index", "split", "subject_id", "session_id"),
    )
    subject_metrics_df = _sort_metrics_dataframe(
        subject_metrics_df,
        split_column="split",
        sort_columns=("model_index", "split", "subject_index"),
    )

    resolved_runs_payload = [run.to_dict() for run in resolved_runs]
    resolved_runs_path = resolved_output_dir / "resolved_runs.json"
    resolved_runs_path.write_text(
        json.dumps(resolved_runs_payload, indent=2, default=_json_default)
    )

    pooled_metrics_json_path = resolved_output_dir / "pooled_metrics.json"
    pooled_metrics_csv_path = resolved_output_dir / "pooled_metrics.csv"
    pooled_metrics_json_path.write_text(
        json.dumps(
            pooled_metrics_df.to_dict(orient="records"),
            indent=2,
            default=_json_default,
        )
    )
    pooled_metrics_df.to_csv(pooled_metrics_csv_path, index=False)

    session_metrics_csv_path = resolved_output_dir / "session_metrics.csv"
    session_metrics_pickle_path = resolved_output_dir / "session_metrics.pkl"
    session_metrics_df.to_csv(session_metrics_csv_path, index=False)
    session_metrics_df.to_pickle(session_metrics_pickle_path)

    subject_metrics_csv_path = resolved_output_dir / "subject_metrics.csv"
    subject_metrics_pickle_path = resolved_output_dir / "subject_metrics.pkl"
    subject_metrics_df.to_csv(subject_metrics_csv_path, index=False)
    subject_metrics_df.to_pickle(subject_metrics_pickle_path)

    splits_to_plot = _resolve_splits_to_plot(pooled_metrics_df)
    reference_lines, reference_line_omissions = _resolve_reference_lines(
        pooled_metrics_df,
        model_label=resolved_runs[0].model_label,
        splits_to_plot=splits_to_plot,
    )
    curriculum_palette = _build_curriculum_palette(subject_metrics_df)
    plot_title_session_counts = _build_plot_title_session_counts(
        pooled_metrics_df,
        model_order=resolved_labels,
    )

    plots_dir = resolved_output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    bar_plot_path = plots_dir / "prediction_likelihood_barplots.png"
    violin_plot_path = plots_dir / "prediction_likelihood_violins.png"
    subject_comparison_plot_path = (
        plots_dir / "prediction_likelihood_subject_comparison.png"
    )
    paired_t_tests_csv_path = resolved_output_dir / "paired_t_tests.csv"
    paired_t_tests_json_path = resolved_output_dir / "paired_t_tests.json"
    paired_t_tests_df = _compute_paired_t_test_results(
        subject_metrics_df=subject_metrics_df,
        model_order=resolved_labels,
        splits_to_plot=splits_to_plot,
    )
    paired_t_tests_df.to_csv(paired_t_tests_csv_path, index=False)
    paired_t_tests_json_path.write_text(
        json.dumps(
            paired_t_tests_df.to_dict(orient="records"),
            indent=2,
            default=_json_default,
        )
    )

    _plot_pooled_likelihood_bars(
        pooled_metrics_df=pooled_metrics_df,
        model_order=resolved_labels,
        splits_to_plot=splits_to_plot,
        reference_lines=reference_lines,
        figure_title_session_counts=plot_title_session_counts,
        output_path=bar_plot_path,
    )
    _plot_subject_likelihood_violins(
        pooled_metrics_df=pooled_metrics_df,
        subject_metrics_df=subject_metrics_df,
        model_order=resolved_labels,
        splits_to_plot=splits_to_plot,
        reference_lines=reference_lines,
        curriculum_palette=curriculum_palette,
        figure_title_session_counts=plot_title_session_counts,
        paired_t_tests_df=paired_t_tests_df,
        output_path=violin_plot_path,
    )
    _plot_subject_comparison_scatter(
        session_metrics_df=session_metrics_df,
        subject_metrics_df=subject_metrics_df,
        model_order=resolved_labels,
        splits_to_plot=splits_to_plot,
        curriculum_palette=curriculum_palette,
        figure_title_session_counts=plot_title_session_counts,
        output_path=subject_comparison_plot_path,
    )

    summary_payload = {
        "output_dir": str(resolved_output_dir),
        "checkpoint_policy": checkpoint_policy,
        "include_heldout": bool(include_heldout),
        "precomputed_session_metrics_path": (
            str(Path(precomputed_session_metrics_path).expanduser().resolve())
            if precomputed_session_metrics_path is not None
            else None
        ),
        "model_dirs": resolved_model_dirs,
        "model_labels": resolved_labels,
        "splits_plotted": list(splits_to_plot),
        "reference_lines": reference_lines,
        "reference_line_omissions": reference_line_omissions,
        "models": model_summaries,
        "artifacts": {
            "resolved_runs": str(resolved_runs_path),
            "pooled_metrics_json": str(pooled_metrics_json_path),
            "pooled_metrics_csv": str(pooled_metrics_csv_path),
            "session_metrics_csv": str(session_metrics_csv_path),
            "session_metrics_pickle": str(session_metrics_pickle_path),
            "subject_metrics_csv": str(subject_metrics_csv_path),
            "subject_metrics_pickle": str(subject_metrics_pickle_path),
            "bar_plot": str(bar_plot_path),
            "violin_plot": str(violin_plot_path),
            "subject_comparison_plot": str(subject_comparison_plot_path),
            "paired_t_tests_csv": str(paired_t_tests_csv_path),
            "paired_t_tests_json": str(paired_t_tests_json_path),
        },
    }
    summary_path = resolved_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=_json_default))

    return {
        "output_dir": str(resolved_output_dir),
        "summary": str(summary_path),
        "resolved_runs": str(resolved_runs_path),
        "pooled_metrics_csv": str(pooled_metrics_csv_path),
        "pooled_metrics_json": str(pooled_metrics_json_path),
        "session_metrics_csv": str(session_metrics_csv_path),
        "session_metrics_pickle": str(session_metrics_pickle_path),
        "subject_metrics_csv": str(subject_metrics_csv_path),
        "subject_metrics_pickle": str(subject_metrics_pickle_path),
        "bar_plot": str(bar_plot_path),
        "violin_plot": str(violin_plot_path),
        "subject_comparison_plot": str(subject_comparison_plot_path),
        "paired_t_tests_csv": str(paired_t_tests_csv_path),
        "paired_t_tests_json": str(paired_t_tests_json_path),
    }


def _resolve_likelihood_run(
    *,
    model_dir: str | Path,
    model_label: str,
    model_index: int,
    checkpoint_policy: str,
) -> ResolvedLikelihoodRun:
    model_dir_path = Path(model_dir).expanduser().resolve()
    inputs_path = model_dir_path / "inputs.yaml"
    outputs_dir = model_dir_path / "outputs"

    if not inputs_path.exists():
        raise FileNotFoundError(f"Could not find run inputs at {inputs_path}")
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Could not find run outputs at {outputs_dir}")

    run_config = _load_structured_file(inputs_path)
    if not isinstance(run_config, Mapping):
        raise ValueError(f"Expected mapping-style config in {inputs_path}")

    data_cfg = _as_dict(run_config.get("data", {}))
    model_cfg = _as_dict(run_config.get("model", {}))
    architecture_cfg = _as_dict(model_cfg.get("architecture", {}))
    model_type = str(model_cfg.get("type", "")).strip().lower()
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model.type={model_type!r} in {inputs_path}. "
            f"Supported types: {sorted(_SUPPORTED_MODEL_TYPES)}"
        )

    multisubject = bool(
        architecture_cfg.get("multisubject", False)
        or data_cfg.get("multisubject", False)
    )
    seed = _coerce_optional_int(run_config.get("seed", model_cfg.get("seed")))

    if model_type in {"gru", "disrnn"}:
        resolved_run = resolve_model_run(
            model_dir=model_dir_path,
            split="train",
            checkpoint_policy=checkpoint_policy,
        )
        return ResolvedLikelihoodRun(
            model_dir=str(model_dir_path),
            inputs_path=str(inputs_path),
            outputs_dir=str(outputs_dir),
            model_type=model_type,
            model_label=model_label,
            model_index=int(model_index),
            multisubject=multisubject,
            seed=seed,
            checkpoint_policy=resolved_run.checkpoint_policy,
            checkpoint_step=resolved_run.checkpoint_step,
            checkpoint_label=resolved_run.checkpoint_label,
            params_path=resolved_run.params_path,
            baseline_output_path=None,
            artifact_selection_reason=(
                resolved_run.checkpoint_selection_reason or resolved_run.fallback_reason
            ),
            session_conditioning_enabled=bool(
                resolved_run.session_conditioning_enabled
            ),
            session_conditioning_encoding_type=str(
                resolved_run.session_conditioning_encoding_type
            ),
            run_config=dict(resolved_run.run_config),
            model_config=dict(resolved_run.model_config),
        )

    baseline_output_path = outputs_dir / "baseline_rl_output.json"
    if not baseline_output_path.exists():
        raise FileNotFoundError(
            f"Could not find baseline RL output at {baseline_output_path}"
        )
    baseline_output = _load_structured_file(baseline_output_path)
    if not isinstance(baseline_output, Mapping):
        raise ValueError(
            f"Expected mapping-style baseline output in {baseline_output_path}"
        )

    return ResolvedLikelihoodRun(
        model_dir=str(model_dir_path),
        inputs_path=str(inputs_path),
        outputs_dir=str(outputs_dir),
        model_type=model_type,
        model_label=model_label,
        model_index=int(model_index),
        multisubject=multisubject,
        seed=seed,
        checkpoint_policy="final_fit",
        checkpoint_step=None,
        checkpoint_label="final_fit",
        params_path=None,
        baseline_output_path=str(baseline_output_path),
        artifact_selection_reason=(
            "Loaded final fitted parameters from outputs/baseline_rl_output.json."
        ),
        session_conditioning_enabled=False,
        session_conditioning_encoding_type="none",
        run_config=_to_serializable(dict(run_config)),
        model_config=_to_serializable(dict(baseline_output)),
    )


def _load_training_bundle_for_run(
    run: ResolvedLikelihoodRun,
    *,
    cols_to_retain: Sequence[str] | None = None,
) -> tuple[Any, Any]:
    """Rebuild the run's training data bundle from its saved config.

    ``cols_to_retain`` overrides which raw trial columns the loader keeps (e.g. to carry
    reward probabilities for analysis). It only affects ``bundle.raw``; model inputs are
    built from ``data.features`` and are unchanged.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from models.session_conditioning import resolve_session_conditioning_from_architecture
    from utils.multisubject import prepend_session_index_to_multisubject_split_datasets

    hydra_config = OmegaConf.load(run.inputs_path)
    if cols_to_retain is not None:
        hydra_config.data.cols_to_retain = list(cols_to_retain)
    dataset_loader = instantiate(hydra_config.data)
    dataset_bundle = dataset_loader.load()
    if run.model_type in {"gru", "disrnn"} and bool(run.multisubject):
        model_cfg = _as_dict(_as_dict(run.run_config).get("model", {}))
        architecture = _as_dict(model_cfg.get("architecture", {}))
        metadata = dict(getattr(dataset_bundle, "metadata", {}) or {})
        max_n_subjects = metadata.get("num_subjects")
        if max_n_subjects is None:
            subject_ids = metadata.get("subject_ids")
            if isinstance(subject_ids, list):
                max_n_subjects = len(subject_ids)
        session_cfg = resolve_session_conditioning_from_architecture(
            architecture=architecture,
            metadata=metadata,
            multisubject=True,
            max_n_subjects=(int(max_n_subjects) if max_n_subjects is not None else None),
            context="Likelihood comparison",
        )
        if bool(session_cfg["enabled"]):
            _require_session_conditioning_bundle_metadata(
                metadata,
                context=f"{run.model_label} ({run.model_type}) likelihood comparison",
            )
            dataset, dataset_train, dataset_eval = prepend_session_index_to_multisubject_split_datasets(
                dataset=dataset_bundle.extras["dataset"],
                dataset_train=dataset_bundle.train_set,
                dataset_eval=dataset_bundle.eval_set,
                metadata=metadata,
            )
            dataset_bundle = dataset_bundle.__class__(
                raw=dataset_bundle.raw,
                train_set=dataset_train,
                eval_set=dataset_eval,
                metadata=metadata,
                extras={**dict(dataset_bundle.extras or {}), "dataset": dataset},
            )
    return hydra_config, dataset_bundle


def _require_session_conditioning_bundle_metadata(
    metadata: Mapping[str, Any],
    *,
    context: str,
) -> None:
    missing_fields: list[str] = []
    if not isinstance(metadata.get("session_context"), Mapping):
        missing_fields.append("session_context")
    if not isinstance(metadata.get("train_session_ids"), list):
        missing_fields.append("train_session_ids")
    if not isinstance(metadata.get("eval_session_ids"), list):
        missing_fields.append("eval_session_ids")
    if not missing_fields:
        return

    raise ValueError(
        f"{context} requires dataset bundle metadata fields {missing_fields} for "
        "multisubject session-conditioned runs."
    )


def _evaluate_resolved_run_splits(
    run: ResolvedLikelihoodRun,
    bundle: Any,
    *,
    hydra_config: Any,
    include_heldout: bool,
) -> dict[str, dict[str, Any]]:
    split_results: dict[str, dict[str, Any]] = {}

    split_payloads = _build_training_split_payloads(bundle)
    for split_name, split_payload in split_payloads.items():
        if run.model_type in {"gru", "disrnn"}:
            split_results[split_name] = _evaluate_rnn_split(
                run,
                split_name=split_name,
                dataset=split_payload["dataset"],
                raw_df=split_payload["raw_df"],
                bundle=bundle,
                score_partition=split_payload.get("score_partition"),
            )
        else:
            split_results[split_name] = _evaluate_baseline_training_split(
                run,
                split_name=split_name,
                raw_df=split_payload["raw_df"],
                score_partition=split_payload.get("score_partition"),
            )

    if not include_heldout:
        return split_results

    heldout_enabled, heldout_reason = _heldout_selector_status(run.run_config)
    if not heldout_enabled:
        split_results["heldout_test"] = _make_skipped_split_result(
            run,
            split_name="heldout_test",
            status="omitted",
            reason=heldout_reason or "Held-out selectors are not configured for this run.",
        )
        return split_results

    if run.multisubject:
        split_results["heldout_test"] = _make_skipped_split_result(
            run,
            split_name="heldout_test",
            status="skipped",
            reason=(
                "Held-out evaluation is not supported for multisubject "
                f"{run.model_type} runs."
            ),
        )
        return split_results

    if run.model_type in {"gru", "disrnn"}:
        split_results["heldout_test"] = _evaluate_rnn_heldout_split(
            run,
            hydra_config=hydra_config,
        )
    else:
        split_results["heldout_test"] = _evaluate_baseline_heldout_split(
            run,
        )

    return split_results


def _load_precomputed_session_metrics(precomputed_session_metrics_path: str | Path | None) -> Any:
    if precomputed_session_metrics_path is None:
        return _make_dataframe([], _SESSION_METRIC_COLUMNS)

    pd = _import_pandas()

    resolved_path = Path(precomputed_session_metrics_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Precomputed session metrics file was not found: {resolved_path}"
        )

    loaded_df = pd.read_pickle(resolved_path)
    precomputed_session_metrics_df = _make_dataframe(
        pd.DataFrame(loaded_df).to_dict(orient="records"),
        _SESSION_METRIC_COLUMNS,
    )
    if "model_dir" not in precomputed_session_metrics_df.columns:
        raise ValueError(
            "Precomputed session metrics must include a 'model_dir' column."
        )
    if not precomputed_session_metrics_df.empty:
        precomputed_session_metrics_df["model_dir"] = precomputed_session_metrics_df[
            "model_dir"
        ].map(lambda value: str(Path(str(value)).expanduser().resolve()))
    return precomputed_session_metrics_df


def _reuse_precomputed_session_metrics_for_run(
    run: ResolvedLikelihoodRun,
    *,
    precomputed_session_metrics_df: Any,
    include_heldout: bool,
) -> Any | None:
    if precomputed_session_metrics_df.empty:
        return None

    cached_rows = precomputed_session_metrics_df[
        precomputed_session_metrics_df["model_dir"] == run.model_dir
    ].copy()
    if cached_rows.empty:
        return None

    cached_rows = _make_dataframe(
        cached_rows.to_dict(orient="records"),
        _SESSION_METRIC_COLUMNS,
    )
    cached_rows["model_index"] = int(run.model_index)
    cached_rows["model_label"] = run.model_label
    cached_rows["model_dir"] = run.model_dir
    cached_rows["model_type"] = run.model_type
    cached_rows["multisubject"] = bool(run.multisubject)
    cached_rows["subject_id"] = cached_rows["subject_id"].map(_normalize_identifier)
    cached_rows["session_id"] = cached_rows["session_id"].map(
        lambda value: str(_normalize_identifier(value))
    )
    cached_rows["split"] = cached_rows["split"].astype(str)
    if not include_heldout:
        cached_rows = cached_rows[cached_rows["split"] != "heldout_test"].copy()
    return _sort_metrics_dataframe(
        cached_rows,
        split_column="split",
        sort_columns=("model_index", "split", "subject_id", "session_id"),
    )


def _split_results_from_session_metrics(
    run: ResolvedLikelihoodRun,
    *,
    session_metrics_df: Any,
) -> dict[str, dict[str, Any]]:
    split_results: dict[str, dict[str, Any]] = {}
    for split_name in _SPLIT_ORDER:
        split_session_metrics_df = session_metrics_df[
            session_metrics_df["split"] == split_name
        ].copy()
        if split_session_metrics_df.empty:
            continue
        subject_metrics_df = _aggregate_subject_metrics(
            split_session_metrics_df,
            run=run,
            split_name=split_name,
        )
        split_results[split_name] = _make_completed_split_result(
            run,
            split_name=split_name,
            session_metrics_df=split_session_metrics_df,
            subject_metrics_df=subject_metrics_df,
        )
    return split_results


def _build_training_split_payloads(bundle: Any) -> dict[str, dict[str, Any]]:
    pd = _import_pandas()

    if getattr(bundle, "raw", None) is None:
        raise ValueError("Dataset bundle must include raw trial data.")

    raw_df = _normalize_raw_dataframe(pd.DataFrame(bundle.raw).copy())
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    train_session_ids = metadata.get("train_session_ids")
    eval_session_ids = metadata.get("eval_session_ids")
    if not isinstance(train_session_ids, list) or not isinstance(eval_session_ids, list):
        raise ValueError(
            "Dataset bundle metadata must include train_session_ids and eval_session_ids."
        )

    full_session_ids = list(dict.fromkeys(raw_df["ses_idx"].tolist()))
    trial_partition_column = metadata.get("trial_partition_column")
    if trial_partition_column is not None:
        if not isinstance(trial_partition_column, str) or not trial_partition_column:
            raise ValueError("trial_partition_column metadata must be a non-empty string.")
        if trial_partition_column not in raw_df.columns:
            raise ValueError(
                f"Trial-split raw data is missing {trial_partition_column!r}."
            )
        adapt_partition = str(metadata.get("adapt_trial_partition", "adapt"))
        test_partition = str(metadata.get("test_trial_partition", "test"))
        train_raw_df = raw_df[
            raw_df[trial_partition_column].astype(str) == adapt_partition
        ].copy()
        test_rows = raw_df[trial_partition_column].astype(str) == test_partition
        if train_raw_df.empty or not bool(test_rows.any()):
            raise ValueError(
                "Trial-split raw data must contain non-empty adaptation and test partitions."
            )
        return {
            "train": {
                "dataset": bundle.train_set,
                "raw_df": train_raw_df,
            },
            # The eval dataset includes the prefix so the recurrent state reaches
            # the correct boundary condition. Keep all rows aligned for forward
            # evaluation, then score only the declared suffix.
            "eval": {
                "dataset": bundle.eval_set,
                "raw_df": raw_df.copy(),
                "score_partition": (trial_partition_column, test_partition),
            },
            "combined": {
                "dataset": _resolve_full_dataset(bundle),
                "raw_df": _subset_raw_df_by_session_ids(raw_df, full_session_ids),
            },
        }
    return {
        "train": {
            "dataset": bundle.train_set,
            "raw_df": _subset_raw_df_by_session_ids(raw_df, train_session_ids),
        },
        "eval": {
            "dataset": bundle.eval_set,
            "raw_df": _subset_raw_df_by_session_ids(raw_df, eval_session_ids),
        },
        "combined": {
            "dataset": _resolve_full_dataset(bundle),
            "raw_df": _subset_raw_df_by_session_ids(raw_df, full_session_ids),
        },
    }


def _resolve_full_dataset(bundle: Any) -> Any:
    extras = getattr(bundle, "extras", {}) or {}
    dataset = extras.get("dataset")
    if dataset is None:
        raise ValueError("Dataset bundle extras must include the full constructed dataset.")
    return dataset


def _evaluate_rnn_split(
    run: ResolvedLikelihoodRun,
    *,
    split_name: str,
    dataset: Any,
    raw_df: Any,
    bundle: Any,
    score_partition: tuple[str, str] | None = None,
) -> dict[str, Any]:
    if dataset is None:
        raise ValueError(f"Bundle is missing dataset for split={split_name!r}.")

    if run.model_type == "gru":
        output_df, n_action_logits = _evaluate_gru_dataset(
            run,
            dataset=dataset,
            raw_df=raw_df,
            metadata=dict(getattr(bundle, "metadata", {}) or {}),
        )
    elif run.model_type == "disrnn":
        output_df, n_action_logits = _evaluate_disrnn_dataset(
            run,
            dataset=dataset,
            raw_df=raw_df,
            metadata=dict(getattr(bundle, "metadata", {}) or {}),
        )
    else:
        raise ValueError(f"Unexpected RNN model_type={run.model_type!r}")

    metric_raw_df = raw_df
    if score_partition is not None:
        partition_column, partition_value = score_partition
        if partition_column not in output_df.columns:
            raise ValueError(
                f"Model output is missing trial partition column {partition_column!r}."
            )
        output_df = output_df[
            output_df[partition_column].astype(str) == partition_value
        ].copy()
        metric_raw_df = raw_df[
            raw_df[partition_column].astype(str) == partition_value
        ].copy()
        if output_df.empty:
            raise ValueError(
                f"No model outputs remain for trial partition {partition_value!r}."
            )

    session_metrics_df = _session_metrics_from_output_df(
        run,
        split_name=split_name,
        output_df=output_df,
        raw_df=metric_raw_df,
        metadata=dict(getattr(bundle, "metadata", {}) or {}),
        n_action_logits=n_action_logits,
    )
    subject_metrics_df = _aggregate_subject_metrics(
        session_metrics_df,
        run=run,
        split_name=split_name,
    )
    return _make_completed_split_result(
        run,
        split_name=split_name,
        session_metrics_df=session_metrics_df,
        subject_metrics_df=subject_metrics_df,
    )


def _evaluate_rnn_heldout_split(
    run: ResolvedLikelihoodRun,
    *,
    hydra_config: Any,
) -> dict[str, Any]:
    if run.model_type == "gru":
        from utils.gru_evaluation import load_gru_heldout_subject_data

        heldout_data = load_gru_heldout_subject_data(hydra_config)
    elif run.model_type == "disrnn":
        from utils.disrnn_evaluation import load_disrnn_heldout_subject_data

        heldout_data = load_disrnn_heldout_subject_data(hydra_config)
    else:
        raise ValueError(f"Unexpected RNN model_type={run.model_type!r}")

    raw_df = _normalize_raw_dataframe(heldout_data["df_test"])
    metadata = _metadata_for_heldout_eval(
        run.run_config,
        raw_df=raw_df,
    )
    if run.model_type == "gru":
        output_df, n_action_logits = _evaluate_gru_dataset(
            run,
            dataset=heldout_data["dataset_test"],
            raw_df=raw_df,
            metadata=metadata,
        )
    else:
        output_df, n_action_logits = _evaluate_disrnn_dataset(
            run,
            dataset=heldout_data["dataset_test"],
            raw_df=raw_df,
            metadata=metadata,
        )

    session_metrics_df = _session_metrics_from_output_df(
        run,
        split_name="heldout_test",
        output_df=output_df,
        raw_df=raw_df,
        metadata=metadata,
        n_action_logits=n_action_logits,
    )
    subject_metrics_df = _aggregate_subject_metrics(
        session_metrics_df,
        run=run,
        split_name="heldout_test",
    )
    return _make_completed_split_result(
        run,
        split_name="heldout_test",
        session_metrics_df=session_metrics_df,
        subject_metrics_df=subject_metrics_df,
    )


def _evaluate_disrnn_dataset(
    run: ResolvedLikelihoodRun,
    *,
    dataset: Any,
    raw_df: Any,
    metadata: Mapping[str, Any],
) -> tuple[Any, int]:
    import numpy as np
    import data_loaders.disrnn_dataset as dl
    from disentangled_rnns.library import rnn_utils

    # DOCUMENTED training-adjacent coupling (lazy import, never at module load):
    # the MULTISUBJECT disRNN case borrows the trainer's network-construction helpers.
    # Single-subject eval uses evaluation.disrnn_evaluation's own _build_network_configs.
    # Clean fix (deferred to avoid touching training): extract disRNN construction into
    # models/disrnn_network.py (symmetric with models/gru_network.py) and call it from both.
    from model_trainers.disrnn_trainer import DisrnnTrainer
    from evaluation.common import _load_saved_params

    if run.params_path is None:
        raise ValueError("disRNN evaluation requires params_path to be resolved.")

    model_cfg = _as_dict(_as_dict(run.run_config).get("model", {}))
    trainer = DisrnnTrainer(
        architecture=_as_dict(model_cfg.get("architecture", {})),
        penalties=_as_dict(model_cfg.get("penalties", {})),
        training=_as_dict(model_cfg.get("training", {})),
        output_dir=run.outputs_dir,
        seed=run.seed,
    )
    ignore_policy = str(metadata.get("ignore_policy", "exclude"))
    _, eval_network_config = trainer._build_network_configs(
        dataset=dataset,
        ignore_policy=ignore_policy,
        metadata=dict(metadata),
    )
    make_eval_network = trainer._make_network_factory(
        eval_network_config,
        multisubject=bool(run.multisubject),
    )

    xs = dataset.get_all()["xs"]
    params = _load_saved_params(Path(run.params_path))
    yhat, network_states = rnn_utils.eval_network(make_eval_network, params, xs)

    n_action_logits = int(getattr(dataset, "n_classes", 0))
    if n_action_logits <= 0:
        n_action_logits = int(np.asarray(yhat).shape[2] - 1)
    if n_action_logits <= 0:
        raise ValueError("Could not infer a positive number of action logits for disRNN.")

    output_df = dl.add_model_results(
        _normalize_raw_dataframe(raw_df),
        np.asarray(network_states),
        np.asarray(yhat),
        ignore_policy=ignore_policy,
    )
    output_df = _normalize_raw_dataframe(output_df)
    return output_df, n_action_logits


def _build_gru_network(
    run: ResolvedLikelihoodRun,
    *,
    metadata: Mapping[str, Any],
) -> tuple[Any, Any, str]:
    """The run's GRU network factory, its saved params and the data ignore_policy.

    Shared by ``_evaluate_gru_dataset`` and analyses that run the network on edited inputs.
    """
    from models.gru_network import make_gru_network
    from models.session_conditioning import resolve_session_conditioning_from_architecture
    from utils.disrnn_evaluation import _load_saved_params

    if run.params_path is None:
        raise ValueError("GRU evaluation requires params_path to be resolved.")

    ignore_policy = str(metadata.get("ignore_policy", "exclude"))
    model_cfg = _as_dict(_as_dict(run.run_config).get("model", {}))
    architecture = _as_dict(model_cfg.get("architecture", {}))
    expected_output_size = 2 if ignore_policy == "exclude" else 3
    output_size = int(architecture.get("output_size", expected_output_size))
    if output_size != expected_output_size:
        raise ValueError(
            "Configured GRU output size does not match ignore_policy: "
            f"configured={output_size} expected={expected_output_size}"
        )

    max_n_subjects = None
    subject_embedding_size = None
    session_conditioning_cfg = {
        "enabled": False,
        "session_encoding_type": "none",
        "session_integration_type": str(architecture.get("session_integration_type", "direct")),
        "session_fourier_k": int(architecture.get("session_fourier_k", 4)),
        "session_delta_n_layers": int(architecture.get("session_delta_n_layers", 3)),
        "session_delta_hidden_size": int(architecture.get("session_delta_hidden_size", 16)),
        "session_max_index_by_subject_index": (),
    }
    if run.multisubject:
        max_n_subjects = metadata.get("num_subjects")
        if max_n_subjects is None:
            subject_ids = metadata.get("subject_ids")
            if isinstance(subject_ids, list):
                max_n_subjects = len(subject_ids)
        if max_n_subjects is None or int(max_n_subjects) <= 0:
            raise ValueError(
                "Multisubject GRU evaluation requires metadata.num_subjects or subject_ids."
            )
        subject_embedding_size = architecture.get("subject_embedding_size")
        if subject_embedding_size is None or int(subject_embedding_size) <= 0:
            raise ValueError(
                "Multisubject GRU evaluation requires architecture.subject_embedding_size > 0."
            )
        session_conditioning_cfg = resolve_session_conditioning_from_architecture(
            architecture=architecture,
            metadata=metadata,
            multisubject=bool(run.multisubject),
            max_n_subjects=int(max_n_subjects),
            subject_embedding_size=(
                int(subject_embedding_size) if subject_embedding_size is not None else None
            ),
            use_legacy_delta_defaults_when_missing=True,
            context="GRU likelihood comparison",
        )
    else:
        session_conditioning_cfg = resolve_session_conditioning_from_architecture(
            architecture=architecture,
            metadata=metadata,
            multisubject=bool(run.multisubject),
            max_n_subjects=max_n_subjects,
            subject_embedding_size=None,
            use_legacy_delta_defaults_when_missing=True,
            context="GRU likelihood comparison",
        )

    make_network = make_gru_network(
        hidden_size=int(architecture["hidden_size"]),
        output_size=output_size,
        multisubject=bool(run.multisubject),
        max_n_subjects=int(max_n_subjects) if max_n_subjects is not None else None,
        subject_embedding_size=(
            int(subject_embedding_size) if subject_embedding_size is not None else None
        ),
        subject_embedding_init=str(architecture.get("subject_embedding_init", "zeros")),
        session_encoding_type=str(session_conditioning_cfg["session_encoding_type"]),
        session_integration_type=str(session_conditioning_cfg["session_integration_type"]),
        session_fourier_k=int(session_conditioning_cfg["session_fourier_k"]),
        session_delta_n_layers=int(session_conditioning_cfg["session_delta_n_layers"]),
        session_delta_hidden_size=int(session_conditioning_cfg["session_delta_hidden_size"]),
        session_max_index_by_subject_index=tuple(
            int(value)
            for value in session_conditioning_cfg["session_max_index_by_subject_index"]
        ),
    )

    params = _load_saved_params(Path(run.params_path))
    return make_network, params, ignore_policy


def _evaluate_gru_dataset(
    run: ResolvedLikelihoodRun,
    *,
    dataset: Any,
    raw_df: Any,
    metadata: Mapping[str, Any],
) -> tuple[Any, int]:
    import numpy as np
    from disentangled_rnns.library import rnn_utils

    from utils.gru_evaluation import add_gru_model_results, _require_n_action_logits

    make_network, params, ignore_policy = _build_gru_network(run, metadata=metadata)
    xs = dataset.get_all()["xs"]
    yhat, network_states = rnn_utils.eval_network(make_network, params, xs)
    n_action_logits = _require_n_action_logits(
        dataset,
        np.asarray(yhat),
        context=f"{run.model_label} {run.model_type} {run.model_dir}",
    )
    output_df = add_gru_model_results(
        _normalize_raw_dataframe(raw_df),
        np.asarray(network_states),
        np.asarray(yhat),
        ignore_policy=ignore_policy,
    )
    output_df = _normalize_raw_dataframe(output_df)
    return output_df, n_action_logits


def _session_metrics_from_output_df(
    run: ResolvedLikelihoodRun,
    *,
    split_name: str,
    output_df: Any,
    raw_df: Any,
    metadata: Mapping[str, Any],
    n_action_logits: int,
) -> Any:
    import numpy as np
    import pandas as pd

    from utils.disrnn_evaluation import _aligned_action_probabilities_from_output_df

    if "ses_idx" not in output_df.columns:
        raise ValueError("Model outputs do not include required column: ses_idx")
    if "animal_response" not in output_df.columns:
        raise ValueError("Model outputs do not include required column: animal_response")

    subject_curriculum_map = _build_subject_curriculum_map(
        raw_df,
        metadata=metadata,
    )

    rows: list[dict[str, Any]] = []
    session_order = list(dict.fromkeys(output_df["ses_idx"].tolist()))
    for session_id in session_order:
        session_df = output_df[output_df["ses_idx"] == session_id].sort_values("trial")
        if session_df.empty:
            continue
        action_probabilities = _aligned_action_probabilities_from_output_df(
            session_df,
            n_action_logits=n_action_logits,
        )
        if action_probabilities.ndim != 2 or action_probabilities.shape[0] != len(session_df):
            raise ValueError(
                "Aligned action probabilities must be 2D and row-aligned to the session dataframe."
            )

        choices = session_df["animal_response"].to_numpy(dtype=int)
        trial_indices = np.arange(len(session_df), dtype=int)
        valid_choice = (choices >= 0) & (choices < int(n_action_logits))
        chosen_probabilities = np.full(len(session_df), np.nan, dtype=float)
        if np.any(valid_choice):
            chosen_probabilities[valid_choice] = action_probabilities[
                trial_indices[valid_choice],
                choices[valid_choice],
            ]
        finite_mask = valid_choice & np.isfinite(chosen_probabilities)
        if not np.any(finite_mask):
            continue
        probs = np.clip(chosen_probabilities[finite_mask], 1e-10, 1.0 - 1e-10)
        total_log_likelihood = float(np.sum(np.log(probs)))
        total_trials = int(np.sum(finite_mask))
        subject_id = (
            _normalize_identifier(session_df["subject_id"].iloc[0])
            if "subject_id" in session_df.columns
            else "unknown"
        )
        rows.append(
            {
                "model_index": int(run.model_index),
                "model_label": run.model_label,
                "model_dir": run.model_dir,
                "model_type": run.model_type,
                "multisubject": bool(run.multisubject),
                "split": split_name,
                "subject_id": subject_id,
                "session_id": str(session_id),
                "curriculum_name": _resolve_curriculum_name(
                    session_df,
                    subject_id=subject_id,
                    subject_curriculum_map=subject_curriculum_map,
                ),
                "total_log_likelihood": total_log_likelihood,
                "total_trials": total_trials,
                "likelihood": _optional_normalized_likelihood_from_log_stats(
                    total_log_likelihood,
                    total_trials,
                ),
            }
        )

    return _make_dataframe(rows, _SESSION_METRIC_COLUMNS)


def _evaluate_baseline_training_split(
    run: ResolvedLikelihoodRun,
    *,
    split_name: str,
    raw_df: Any,
    score_partition: tuple[str, str] | None = None,
) -> dict[str, Any]:
    baseline_output = _load_baseline_output_for_run(run)
    if _baseline_output_is_multisubject_per_subject(baseline_output):
        session_metrics_df = _evaluate_baseline_multisubject_sessions(
            run,
            split_name=split_name,
            raw_df=raw_df,
            baseline_output=baseline_output,
            score_partition=score_partition,
        )
    else:
        session_metrics_df = _evaluate_baseline_global_sessions(
            run,
            split_name=split_name,
            raw_df=raw_df,
            baseline_output=baseline_output,
            score_partition=score_partition,
        )

    subject_metrics_df = _aggregate_subject_metrics(
        session_metrics_df,
        run=run,
        split_name=split_name,
    )
    return _make_completed_split_result(
        run,
        split_name=split_name,
        session_metrics_df=session_metrics_df,
        subject_metrics_df=subject_metrics_df,
    )


def _evaluate_baseline_heldout_split(
    run: ResolvedLikelihoodRun,
) -> dict[str, Any]:
    baseline_output = _load_baseline_output_for_run(run)
    if _baseline_output_is_multisubject_per_subject(baseline_output):
        return _make_skipped_split_result(
            run,
            split_name="heldout_test",
            status="skipped",
            reason=(
                "Held-out evaluation is not supported for multisubject baseline RL "
                "per-subject fits."
            ),
        )

    from utils.load_mice_database import load_mice_from_database

    data_cfg = _as_dict(_as_dict(run.run_config).get("data", {}))
    df_test, _ = load_mice_from_database(
        split="heldout",
        subject_ids=data_cfg.get("test_subject_ids"),
        curricula=data_cfg.get("curricula"),
        min_sessions=int(data_cfg.get("min_sessions", 10)),
        heldout_every_n=int(data_cfg.get("heldout_every_n", 5)),
        mature_only=bool(data_cfg.get("mature_only", True)),
        cols_to_retain=_heldout_cols_to_retain(data_cfg),
        snapshot=data_cfg.get("snapshot"),
    )
    raw_df = _normalize_raw_dataframe(df_test)
    session_metrics_df = _evaluate_baseline_global_sessions(
        run,
        split_name="heldout_test",
        raw_df=raw_df,
        baseline_output=baseline_output,
    )
    subject_metrics_df = _aggregate_subject_metrics(
        session_metrics_df,
        run=run,
        split_name="heldout_test",
    )
    return _make_completed_split_result(
        run,
        split_name="heldout_test",
        session_metrics_df=session_metrics_df,
        subject_metrics_df=subject_metrics_df,
    )


def _evaluate_baseline_global_sessions(
    run: ResolvedLikelihoodRun,
    *,
    split_name: str,
    raw_df: Any,
    baseline_output: Mapping[str, Any],
    score_partition: tuple[str, str] | None = None,
) -> Any:
    fitted_params = baseline_output.get("fitted_params")
    if not isinstance(fitted_params, Mapping) or not fitted_params:
        raise ValueError("baseline_rl_output.json is missing fitted_params.")

    agent_class_name, agent_kwargs = _resolve_baseline_agent_spec(run, baseline_output)
    choice_sessions, reward_sessions, session_ids, session_subject_ids = _extract_sessions_from_df(
        raw_df
    )
    if not choice_sessions:
        return _make_dataframe([], _SESSION_METRIC_COLUMNS)

    choice_prob_sessions = _perform_baseline_agent_rollout(
        agent_class_name=agent_class_name,
        agent_kwargs=agent_kwargs,
        fitted_params=fitted_params,
        choice_sessions=choice_sessions,
        reward_sessions=reward_sessions,
        seed=run.seed,
    )
    score_masks = _extract_session_score_masks(raw_df, score_partition)
    subject_curriculum_map = _build_subject_curriculum_map(raw_df, metadata={})
    return _baseline_session_metrics_from_probabilities(
        run,
        split_name=split_name,
        session_ids=session_ids,
        session_subject_ids=session_subject_ids,
        choice_sessions=choice_sessions,
        choice_prob_sessions=choice_prob_sessions,
        score_masks=score_masks,
        subject_curriculum_map=subject_curriculum_map,
    )


def _evaluate_baseline_multisubject_sessions(
    run: ResolvedLikelihoodRun,
    *,
    split_name: str,
    raw_df: Any,
    baseline_output: Mapping[str, Any],
    score_partition: tuple[str, str] | None = None,
) -> Any:
    import pandas as pd

    agent_class_name, agent_kwargs = _resolve_baseline_agent_spec(run, baseline_output)
    fitted_params_per_subject = baseline_output.get("fitted_params_per_subject")
    if not isinstance(fitted_params_per_subject, Mapping) or not fitted_params_per_subject:
        raise ValueError("baseline_rl_output.json is missing fitted_params_per_subject.")

    normalized_raw_df = _normalize_raw_dataframe(pd.DataFrame(raw_df).copy())
    if "subject_id" not in normalized_raw_df.columns:
        raise ValueError(
            "Multisubject baseline RL comparison requires raw_df to include subject_id."
        )

    subject_curriculum_map = _build_subject_curriculum_map(normalized_raw_df, metadata={})
    session_df_chunks: list[Any] = []
    ordered_subject_ids = list(dict.fromkeys(normalized_raw_df["subject_id"].tolist()))
    for subject_id in ordered_subject_ids:
        subject_df = normalized_raw_df[normalized_raw_df["subject_id"] == subject_id].copy()
        if subject_df.empty:
            continue
        subject_fit_summary = fitted_params_per_subject.get(str(subject_id))
        if subject_fit_summary is None:
            raise ValueError(
                f"Missing fitted_params_per_subject entry for subject_id={subject_id!r}"
            )
        fitted_params = _extract_subject_fitted_params(subject_fit_summary)
        choice_sessions, reward_sessions, session_ids, session_subject_ids = _extract_sessions_from_df(
            subject_df
        )
        if not choice_sessions:
            continue
        choice_prob_sessions = _perform_baseline_agent_rollout(
            agent_class_name=agent_class_name,
            agent_kwargs=agent_kwargs,
            fitted_params=fitted_params,
            choice_sessions=choice_sessions,
            reward_sessions=reward_sessions,
            seed=run.seed,
        )
        score_masks = _extract_session_score_masks(subject_df, score_partition)
        session_df_chunks.append(
            _baseline_session_metrics_from_probabilities(
                run,
                split_name=split_name,
                session_ids=session_ids,
                session_subject_ids=session_subject_ids,
                choice_sessions=choice_sessions,
                choice_prob_sessions=choice_prob_sessions,
                score_masks=score_masks,
                subject_curriculum_map=subject_curriculum_map,
            )
        )

    return _concat_metric_frames(session_df_chunks, _SESSION_METRIC_COLUMNS)


def _baseline_session_metrics_from_probabilities(
    run: ResolvedLikelihoodRun,
    *,
    split_name: str,
    session_ids: Sequence[Any],
    session_subject_ids: Sequence[Any],
    choice_sessions: Sequence[Any],
    choice_prob_sessions: Sequence[Any],
    subject_curriculum_map: Mapping[Any, str],
    score_masks: Sequence[Any] | None = None,
) -> Any:
    rows: list[dict[str, Any]] = []
    if score_masks is None:
        score_masks = [None] * len(choice_sessions)
    if len(score_masks) != len(choice_sessions):
        raise ValueError("score_masks must align one-to-one with choice sessions.")
    for session_id, subject_id, choices, choice_prob, score_mask in zip(
        session_ids,
        session_subject_ids,
        choice_sessions,
        choice_prob_sessions,
        score_masks,
    ):
        total_log_likelihood, total_trials = _session_log_likelihood_from_choice_prob(
            choices,
            choice_prob,
            score_mask=score_mask,
        )
        if total_trials <= 0:
            continue
        normalized_subject_id = _normalize_identifier(subject_id)
        rows.append(
            {
                "model_index": int(run.model_index),
                "model_label": run.model_label,
                "model_dir": run.model_dir,
                "model_type": run.model_type,
                "multisubject": bool(run.multisubject),
                "split": split_name,
                "subject_id": normalized_subject_id,
                "session_id": str(_normalize_identifier(session_id)),
                "curriculum_name": str(
                    subject_curriculum_map.get(normalized_subject_id, "Unknown")
                ),
                "total_log_likelihood": total_log_likelihood,
                "total_trials": total_trials,
                "likelihood": _optional_normalized_likelihood_from_log_stats(
                    total_log_likelihood,
                    total_trials,
                ),
            }
        )
    return _make_dataframe(rows, _SESSION_METRIC_COLUMNS)


def _perform_baseline_agent_rollout(
    *,
    agent_class_name: str,
    agent_kwargs: Mapping[str, Any],
    fitted_params: Mapping[str, Any],
    choice_sessions: Sequence[Any],
    reward_sessions: Sequence[Any],
    seed: int | None,
) -> list[Any]:
    from aind_dynamic_foraging_models import generative_model

    agent_class_obj = getattr(generative_model, agent_class_name, None)
    if agent_class_obj is None:
        raise ValueError(
            f"Agent class {agent_class_name!r} was not found in generative_model."
        )

    agent = agent_class_obj(
        **dict(agent_kwargs),
        seed=seed,
    )
    agent.set_params(
        **{
            str(param_name): float(param_value)
            for param_name, param_value in dict(fitted_params).items()
        }
    )
    return list(
        agent.perform_closed_loop_multi_session(
            list(choice_sessions),
            list(reward_sessions),
        )
    )


def _aggregate_subject_metrics(
    session_metrics_df: Any,
    *,
    run: ResolvedLikelihoodRun,
    split_name: str,
) -> Any:
    pd = _import_pandas()

    if session_metrics_df.empty:
        return _make_dataframe([], _SUBJECT_METRIC_COLUMNS)

    rows: list[dict[str, Any]] = []
    ordered_subject_ids = list(dict.fromkeys(session_metrics_df["subject_id"].tolist()))
    for subject_index, subject_id in enumerate(ordered_subject_ids):
        subject_rows = session_metrics_df[session_metrics_df["subject_id"] == subject_id].copy()
        if subject_rows.empty:
            continue
        curricula = list(
            dict.fromkeys(
                str(value)
                for value in subject_rows["curriculum_name"].fillna("Unknown").tolist()
            )
        )
        curriculum_name = "Unknown"
        if len(curricula) == 1:
            curriculum_name = curricula[0]
        elif len(curricula) > 1:
            curriculum_name = "Mixed"

        total_log_likelihood = float(subject_rows["total_log_likelihood"].sum())
        total_trials = int(subject_rows["total_trials"].sum())
        rows.append(
            {
                "model_index": int(run.model_index),
                "model_label": run.model_label,
                "model_dir": run.model_dir,
                "model_type": run.model_type,
                "multisubject": bool(run.multisubject),
                "split": split_name,
                "subject_id": subject_id,
                "subject_index": int(subject_index),
                "curriculum_name": curriculum_name,
                "num_sessions": int(len(subject_rows)),
                "total_log_likelihood": total_log_likelihood,
                "total_trials": total_trials,
                "likelihood": _optional_normalized_likelihood_from_log_stats(
                    total_log_likelihood,
                    total_trials,
                ),
            }
        )

    return _make_dataframe(rows, _SUBJECT_METRIC_COLUMNS)


def _make_completed_split_result(
    run: ResolvedLikelihoodRun,
    *,
    split_name: str,
    session_metrics_df: Any,
    subject_metrics_df: Any,
) -> dict[str, Any]:
    pooled_metric = _pooled_metric_from_session_metrics(
        run,
        split_name=split_name,
        session_metrics_df=session_metrics_df,
        subject_metrics_df=subject_metrics_df,
    )
    return {
        "pooled_metric": pooled_metric,
        "session_metrics": session_metrics_df,
        "subject_metrics": subject_metrics_df,
    }


def _make_skipped_split_result(
    run: ResolvedLikelihoodRun,
    *,
    split_name: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "pooled_metric": {
            "model_index": int(run.model_index),
            "model_label": run.model_label,
            "model_dir": run.model_dir,
            "model_type": run.model_type,
            "multisubject": bool(run.multisubject),
            "split": split_name,
            "status": status,
            "skip_reason": reason,
            "num_sessions": 0,
            "num_subjects": 0,
            "total_log_likelihood": None,
            "total_trials": 0,
            "pooled_trial_likelihood": None,
            "session_sem": None,
        },
        "session_metrics": _make_dataframe([], _SESSION_METRIC_COLUMNS),
        "subject_metrics": _make_dataframe([], _SUBJECT_METRIC_COLUMNS),
    }


def _pooled_metric_from_session_metrics(
    run: ResolvedLikelihoodRun,
    *,
    split_name: str,
    session_metrics_df: Any,
    subject_metrics_df: Any,
) -> dict[str, Any]:
    if session_metrics_df.empty:
        return {
            "model_index": int(run.model_index),
            "model_label": run.model_label,
            "model_dir": run.model_dir,
            "model_type": run.model_type,
            "multisubject": bool(run.multisubject),
            "split": split_name,
            "status": "completed",
            "skip_reason": None,
            "num_sessions": 0,
            "num_subjects": 0,
            "total_log_likelihood": 0.0,
            "total_trials": 0,
            "pooled_trial_likelihood": None,
            "session_sem": None,
        }

    total_log_likelihood = float(session_metrics_df["total_log_likelihood"].sum())
    total_trials = int(session_metrics_df["total_trials"].sum())
    session_likelihoods = session_metrics_df["likelihood"].dropna().astype(float).to_numpy()
    return {
        "model_index": int(run.model_index),
        "model_label": run.model_label,
        "model_dir": run.model_dir,
        "model_type": run.model_type,
        "multisubject": bool(run.multisubject),
        "split": split_name,
        "status": "completed",
        "skip_reason": None,
        "num_sessions": int(len(session_metrics_df)),
        "num_subjects": int(len(subject_metrics_df)),
        "total_log_likelihood": total_log_likelihood,
        "total_trials": total_trials,
        "pooled_trial_likelihood": _optional_normalized_likelihood_from_log_stats(
            total_log_likelihood,
            total_trials,
        ),
        "session_sem": _session_sem(session_likelihoods),
    }


def _resolve_splits_to_plot(pooled_metrics_df: Any) -> list[str]:
    if pooled_metrics_df.empty:
        return []
    plotted_splits = []
    for split_name in _SPLIT_ORDER:
        if split_name == "combined":
            continue
        split_rows = pooled_metrics_df[pooled_metrics_df["split"] == split_name]
        if split_rows.empty:
            continue
        completed_rows = split_rows[
            (split_rows["status"] == "completed")
            & split_rows["pooled_trial_likelihood"].notna()
        ]
        if not completed_rows.empty:
            plotted_splits.append(split_name)
    return plotted_splits


def _resolve_reference_lines(
    pooled_metrics_df: Any,
    *,
    model_label: str,
    splits_to_plot: Sequence[str],
) -> tuple[dict[str, float], dict[str, str]]:
    reference_lines: dict[str, float] = {}
    omissions: dict[str, str] = {}
    for split_name in splits_to_plot:
        split_rows = pooled_metrics_df[
            (pooled_metrics_df["split"] == split_name)
            & (pooled_metrics_df["model_label"] == model_label)
        ]
        if split_rows.empty:
            omissions[split_name] = (
                f"The first model ({model_label}) does not have a {split_name} row."
            )
            continue
        first_row = split_rows.iloc[0]
        pooled_value = first_row.get("pooled_trial_likelihood")
        if first_row.get("status") != "completed" or pooled_value is None or _is_nan(pooled_value):
            omissions[split_name] = (
                f"The first model ({model_label}) does not have a supported {split_name} likelihood."
            )
            continue
        reference_lines[split_name] = float(pooled_value)
    return reference_lines, omissions


def _plot_pooled_likelihood_bars(
    *,
    pooled_metrics_df: Any,
    model_order: Sequence[str],
    splits_to_plot: Sequence[str],
    reference_lines: Mapping[str, float],
    figure_title_session_counts: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np

    n_panels = max(1, len(splits_to_plot))
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(6.0, 4.4 * n_panels), 5.4),
        squeeze=False,
        sharey=True,
    )
    x_positions = np.arange(len(model_order), dtype=float)
    model_type_by_label = _resolve_model_type_by_label(
        pooled_metrics_df,
        model_order=model_order,
    )
    figure_title = "Prediction Likelihood Across Models"
    if figure_title_session_counts:
        figure_title = f"{figure_title}\n{figure_title_session_counts}"
    fig.suptitle(figure_title, y=0.93)
    for axis_index, split_name in enumerate(splits_to_plot):
        ax = axes[0, axis_index]
        split_rows = pooled_metrics_df[pooled_metrics_df["split"] == split_name].copy()
        value_by_label = {
            str(row["model_label"]): row
            for row in split_rows.to_dict(orient="records")
            if str(row.get("status")) == "completed"
        }
        heights = []
        errors = []
        bar_colors = []
        for model_label in model_order:
            row = value_by_label.get(model_label)
            model_type = model_type_by_label.get(model_label)
            bar_colors.append(
                _MODEL_TYPE_COLORS.get(model_type, _DEFAULT_CURRICULUM_COLORS["Unknown"])
            )
            if row is None:
                heights.append(np.nan)
                errors.append(0.0)
                continue
            value = row.get("pooled_trial_likelihood")
            heights.append(float(value) if value is not None else np.nan)
            sem = row.get("session_sem")
            errors.append(float(sem) if sem is not None and not _is_nan(sem) else 0.0)

        ax.bar(
            x_positions,
            heights,
            yerr=errors,
            capsize=4,
            color=bar_colors,
            edgecolor="white",
            linewidth=1.0,
        )
        max_height = 0.0
        for x_position, height, error in zip(x_positions, heights, errors):
            if _is_nan(height):
                continue
            max_height = max(max_height, float(height) + float(error))
        if split_name in reference_lines:
            max_height = max(max_height, float(reference_lines[split_name]))
        label_offset = max(0.015, max_height * 0.04) if max_height > 0 else 0.03
        for x_position, height, error in zip(x_positions, heights, errors):
            if _is_nan(height):
                continue
            ax.text(
                float(x_position),
                min(0.895, float(height) + float(error) + label_offset),
                f"{float(height):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if split_name in reference_lines:
            ax.axhline(
                float(reference_lines[split_name]),
                color="dimgray",
                linestyle="--",
                linewidth=1.5,
                label="First-model reference",
            )
        ax.set_title(_BAR_SPLIT_TITLES.get(split_name, split_name.replace("_", " ").title()))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(list(model_order), rotation=35, ha="right")
        ax.set_ylabel("Prediction Likelihood")
        ax.grid(axis="y", color="0.9", linewidth=1.0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0.6, top=0.9)
        ax.tick_params(axis="y", labelleft=True)

    handles = []
    labels = []
    present_model_types = list(
        dict.fromkeys(
            model_type
            for model_type in (model_type_by_label.get(label) for label in model_order)
            if model_type in _MODEL_TYPE_COLORS
        )
    )
    handles.extend(
        [
            Patch(
                facecolor=_MODEL_TYPE_COLORS[model_type],
                edgecolor="none",
            )
            for model_type in present_model_types
        ]
    )
    labels.extend(present_model_types)
    if reference_lines:
        handles.append(
            Line2D([0], [0], color="dimgray", linestyle="--", linewidth=1.5)
        )
        labels.append("First-model reference")
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 4),
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.945))
    else:
        fig.tight_layout(rect=(0, 0.04, 1, 0.945))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_likelihood_violins(
    *,
    pooled_metrics_df: Any,
    subject_metrics_df: Any,
    model_order: Sequence[str],
    splits_to_plot: Sequence[str],
    reference_lines: Mapping[str, float],
    curriculum_palette: Mapping[str, str],
    figure_title_session_counts: str,
    paired_t_tests_df: Any,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    n_panels = max(1, len(splits_to_plot))
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(6.0, 4.6 * n_panels), 5.6),
        squeeze=False,
        sharey=True,
    )
    x_positions = np.arange(len(model_order), dtype=float)
    rng = np.random.default_rng(0)
    figure_title = "Subject Prediction Likelihood Across Models (paired t-test)"
    if figure_title_session_counts:
        figure_title = f"{figure_title}\n{figure_title_session_counts}"
    fig.suptitle(figure_title, y=0.93)

    for axis_index, split_name in enumerate(splits_to_plot):
        ax = axes[0, axis_index]
        split_df = subject_metrics_df[subject_metrics_df["split"] == split_name].copy()
        split_values_for_annotations: list[float] = []
        if split_df.empty:
            ax.text(
                0.5,
                0.5,
                "No subject data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(_BAR_SPLIT_TITLES.get(split_name, split_name.replace("_", " ").title()))
            ax.set_xticks(x_positions)
            ax.set_xticklabels(list(model_order), rotation=35, ha="right")
            ax.tick_params(axis="y", labelleft=True)
            continue

        for position, model_label in enumerate(model_order):
            model_df = split_df[split_df["model_label"] == model_label].copy()
            values = model_df["likelihood"].dropna().astype(float).to_numpy()
            if len(values) == 0:
                continue
            split_values_for_annotations.extend(values.tolist())

            violin = ax.violinplot(
                dataset=[values],
                positions=[position],
                widths=0.8,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor("#9ecae1")
                body.set_edgecolor("#4c78a8")
                body.set_alpha(0.45)

            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            ax.vlines(position, q1, q3, color="#111111", linewidth=2.8, zorder=6)
            ax.hlines(
                [q1, median, q3],
                position - 0.12,
                position + 0.12,
                color="#111111",
                linewidth=2.2,
                zorder=6,
            )

            jitter = rng.uniform(-0.12, 0.12, size=len(model_df))
            dot_x = np.full(len(model_df), float(position)) + jitter
            dot_y = model_df["likelihood"].astype(float).to_numpy()
            dot_colors = [
                curriculum_palette.get(str(curriculum), _DEFAULT_CURRICULUM_COLORS["Unknown"])
                for curriculum in model_df["curriculum_name"].fillna("Unknown").astype(str).tolist()
            ]
            ax.scatter(
                dot_x,
                dot_y,
                s=42,
                color=dot_colors,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.45,
                zorder=4,
            )

        if split_name in reference_lines:
            ax.axhline(
                float(reference_lines[split_name]),
                color="dimgray",
                linestyle="--",
                linewidth=1.5,
            )
        ax.set_title(_BAR_SPLIT_TITLES.get(split_name, split_name.replace("_", " ").title()))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(list(model_order), rotation=35, ha="right")
        ax.set_ylabel("Prediction Likelihood")
        ax.grid(axis="y", color="0.9", linewidth=1.0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0.5, top=0.9)
        ax.tick_params(axis="y", labelleft=True)
        _annotate_paired_t_tests_on_violin_plot(
            ax,
            split_name=split_name,
            model_order=model_order,
            paired_t_tests_df=paired_t_tests_df,
            annotation_positions_by_label=_resolve_uniform_violin_annotation_positions(
                model_order=model_order,
                split_values=split_values_for_annotations,
            ),
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            label=curriculum_name,
        )
        for curriculum_name, color in curriculum_palette.items()
    ]
    if reference_lines:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="dimgray",
                linestyle="--",
                linewidth=1.5,
                label="First-model reference",
            )
        )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(legend_handles), 5),
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.945))
    else:
        fig.tight_layout(rect=(0, 0.04, 1, 0.945))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_model_type_by_label(
    pooled_metrics_df: Any,
    *,
    model_order: Sequence[str],
) -> dict[str, str]:
    model_type_by_label: dict[str, str] = {}
    for model_label in model_order:
        model_rows = pooled_metrics_df[pooled_metrics_df["model_label"] == model_label]
        if model_rows.empty:
            continue
        model_type_by_label[model_label] = str(model_rows.iloc[0]["model_type"])
    return model_type_by_label


def _build_plot_title_session_counts(
    pooled_metrics_df: Any,
    *,
    model_order: Sequence[str],
) -> str:
    split_labels = {
        "train": "Train sessions",
        "eval": "Eval sessions",
    }
    title_parts: list[str] = []
    for split_name, title_label in split_labels.items():
        split_rows = pooled_metrics_df[pooled_metrics_df["split"] == split_name].copy()
        if split_rows.empty:
            continue

        count_by_label: dict[str, int | None] = {}
        for model_label in model_order:
            label_rows = split_rows[split_rows["model_label"] == model_label]
            if label_rows.empty:
                count_by_label[model_label] = None
                continue
            first_row = label_rows.iloc[0]
            if first_row.get("status") != "completed":
                count_by_label[model_label] = None
                continue
            count_value = first_row.get("num_sessions")
            count_by_label[model_label] = (
                None if count_value is None or _is_nan(count_value) else int(count_value)
            )

        supported_counts = [
            count_value for count_value in count_by_label.values() if count_value is not None
        ]
        if not supported_counts:
            continue
        title_parts.append(f"{title_label}:")
        if len(supported_counts) == len(model_order) and len(set(supported_counts)) == 1:
            title_parts.append(f"all models={supported_counts[0]}")
            continue
        formatted_entries = [
            f"{model_label}="
            f"{count_by_label[model_label] if count_by_label[model_label] is not None else 'n/a'}"
            for model_label in model_order
        ]
        for start_index in range(0, len(formatted_entries), 3):
            title_parts.append(
                ", ".join(formatted_entries[start_index : start_index + 3])
            )
    return "\n".join(title_parts)


def _set_matching_axis_ticks(
    ax,
    *,
    axis_min: float,
    axis_max: float,
    n_ticks: int = 4,
    tick_values: Sequence[float] | None = None,
) -> None:
    import numpy as np

    if tick_values is not None:
        resolved_tick_values = [float(value) for value in tick_values]
    elif axis_max <= axis_min:
        resolved_tick_values = np.array([axis_min], dtype=float).tolist()
    else:
        resolved_tick_values = np.linspace(axis_min, axis_max, num=max(2, int(n_ticks)))
        resolved_tick_values = np.unique(np.round(resolved_tick_values, 3)).tolist()
    ax.set_xticks(resolved_tick_values)
    ax.set_yticks(resolved_tick_values)


def _plot_subject_comparison_scatter(
    *,
    session_metrics_df: Any,
    subject_metrics_df: Any,
    model_order: Sequence[str],
    splits_to_plot: Sequence[str],
    curriculum_palette: Mapping[str, str],
    figure_title_session_counts: str,
    output_path: Path,
    n_bootstrap: int = 1000,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    first_model_label = str(model_order[0]) if model_order else "First model"
    comparison_labels = [str(model_label) for model_label in model_order[1:]]

    figure_title = f"Subject Comparison vs {first_model_label}"
    if figure_title_session_counts:
        figure_title = f"{figure_title}\n{figure_title_session_counts}"

    if not comparison_labels or not splits_to_plot:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle(figure_title, y=0.93)
        ax.text(
            0.5,
            0.5,
            "Need at least two models with supported splits.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n_rows = max(1, len(splits_to_plot))
    n_cols = max(1, len(comparison_labels))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(6.0, 4.8 * n_cols), max(4.8, 4.8 * n_rows)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    fig.suptitle(figure_title, y=0.93)

    for row_index, split_name in enumerate(splits_to_plot):
        for col_index, comparison_model_label in enumerate(comparison_labels):
            ax = axes[row_index, col_index]
            points_df = _build_subject_comparison_points(
                session_metrics_df=session_metrics_df,
                subject_metrics_df=subject_metrics_df,
                split_name=split_name,
                first_model_label=first_model_label,
                comparison_model_label=comparison_model_label,
                n_bootstrap=n_bootstrap,
            )
            if points_df.empty:
                comparison_superior_count = 0
                first_model_superior_count = 0
                ax.text(
                    0.5,
                    0.5,
                    "No overlapping subject data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(
                    f"{_BAR_SPLIT_TITLES.get(split_name, split_name.replace('_', ' ').title())}\n"
                    f"{comparison_model_label} ({comparison_superior_count}) vs "
                    f"{first_model_label} ({first_model_superior_count})"
                )
                ax.set_xlabel(first_model_label)
                ax.set_ylabel(comparison_model_label)
                ax.grid(color="0.9", linewidth=1.0)
                ax.set_axisbelow(True)
                ax.tick_params(axis="both", labelbottom=True, labelleft=True)
                continue

            axis_min = 0.56
            axis_max = 0.90

            comparison_superior_count, first_model_superior_count = _subject_superiority_counts(
                points_df
            )
            for point in points_df.to_dict(orient="records"):
                color = curriculum_palette.get(
                    str(point["curriculum_name"]),
                    _DEFAULT_CURRICULUM_COLORS["Unknown"],
                )
                ax.errorbar(
                    float(point["x"]),
                    float(point["y"]),
                    xerr=[
                        [max(0.0, float(point["x"]) - float(point["x_low"]))],
                        [max(0.0, float(point["x_high"]) - float(point["x"]))],
                    ],
                    yerr=[
                        [max(0.0, float(point["y"]) - float(point["y_low"]))],
                        [max(0.0, float(point["y_high"]) - float(point["y"]))],
                    ],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.9,
                    capsize=2,
                    alpha=0.35,
                    zorder=2,
                )
                ax.scatter(
                    float(point["x"]),
                    float(point["y"]),
                    s=64 if float(point["y"]) > float(point["x"]) else 26,
                    marker="*" if float(point["y"]) > float(point["x"]) else "o",
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.8,
                    zorder=3,
                )

            ax.plot(
                [axis_min, axis_max],
                [axis_min, axis_max],
                color="dimgray",
                linestyle="--",
                linewidth=1.4,
                zorder=1,
            )
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.set_aspect("equal", adjustable="box")
            _set_matching_axis_ticks(
                ax,
                axis_min=axis_min,
                axis_max=axis_max,
                tick_values=(0.6, 0.7, 0.8, 0.9),
            )
            ax.set_title(
                f"{_BAR_SPLIT_TITLES.get(split_name, split_name.replace('_', ' ').title())}\n"
                f"{comparison_model_label} ({comparison_superior_count}) vs "
                f"{first_model_label} ({first_model_superior_count})"
            )
            ax.set_xlabel(first_model_label)
            ax.set_ylabel(comparison_model_label)
            ax.grid(color="0.9", linewidth=1.0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelbottom=True, labelleft=True)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            label=curriculum_name,
        )
        for curriculum_name, color in curriculum_palette.items()
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="dimgray",
            linestyle="--",
            linewidth=1.5,
            label="Diagonal reference",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(legend_handles), 5),
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compute_paired_t_test_results(
    *,
    subject_metrics_df: Any,
    model_order: Sequence[str],
    splits_to_plot: Sequence[str],
) -> Any:
    pd = _import_pandas()

    if subject_metrics_df.empty or len(model_order) <= 1 or not splits_to_plot:
        return pd.DataFrame(
            columns=[
                "split",
                "reference_model_label",
                "comparison_model_label",
                "n_subjects",
                "statistic",
                "p_value",
                "significance_label",
                "mean_reference",
                "mean_comparison",
                "mean_difference",
                "status",
                "method",
                "note",
            ]
        )

    first_model_label = str(model_order[0])
    rows: list[dict[str, Any]] = []
    for split_name in splits_to_plot:
        reference_df = subject_metrics_df[
            (subject_metrics_df["split"] == split_name)
            & (subject_metrics_df["model_label"] == first_model_label)
        ].copy()
        reference_by_subject = {
            _normalize_identifier(row["subject_id"]): row
            for row in reference_df.to_dict(orient="records")
            if row.get("likelihood") is not None and not _is_nan(row.get("likelihood"))
        }

        for comparison_model_label in model_order[1:]:
            comparison_df = subject_metrics_df[
                (subject_metrics_df["split"] == split_name)
                & (subject_metrics_df["model_label"] == comparison_model_label)
            ].copy()
            paired_reference_values: list[float] = []
            paired_comparison_values: list[float] = []
            for comparison_row in comparison_df.to_dict(orient="records"):
                subject_id = _normalize_identifier(comparison_row["subject_id"])
                reference_row = reference_by_subject.get(subject_id)
                if reference_row is None:
                    continue
                reference_value = reference_row.get("likelihood")
                comparison_value = comparison_row.get("likelihood")
                if (
                    reference_value is None
                    or comparison_value is None
                    or _is_nan(reference_value)
                    or _is_nan(comparison_value)
                ):
                    continue
                paired_reference_values.append(float(reference_value))
                paired_comparison_values.append(float(comparison_value))

            t_test_result = _paired_t_test(
                paired_reference_values,
                paired_comparison_values,
            )
            rows.append(
                {
                    "split": split_name,
                    "reference_model_label": first_model_label,
                    "comparison_model_label": str(comparison_model_label),
                    "n_subjects": int(len(paired_reference_values)),
                    "statistic": t_test_result["statistic"],
                    "p_value": t_test_result["p_value"],
                    "significance_label": _format_significance_label(
                        t_test_result["p_value"]
                    ),
                    "mean_reference": (
                        None
                        if not paired_reference_values
                        else float(sum(paired_reference_values) / len(paired_reference_values))
                    ),
                    "mean_comparison": (
                        None
                        if not paired_comparison_values
                        else float(
                            sum(paired_comparison_values) / len(paired_comparison_values)
                        )
                    ),
                    "mean_difference": (
                        None
                        if not paired_reference_values
                        else float(
                            sum(
                                comparison_value - reference_value
                                for reference_value, comparison_value in zip(
                                    paired_reference_values,
                                    paired_comparison_values,
                                )
                            )
                            / len(paired_reference_values)
                        )
                    ),
                    "status": t_test_result["status"],
                    "method": t_test_result["method"],
                    "note": t_test_result["note"],
                }
            )

    return pd.DataFrame(rows)


def _paired_t_test(
    reference_values: Sequence[float],
    comparison_values: Sequence[float],
) -> dict[str, Any]:
    if len(reference_values) != len(comparison_values):
        raise ValueError("Paired t-test requires aligned reference/comparison values.")
    if len(reference_values) < 2:
        return {
            "statistic": None,
            "p_value": None,
            "status": "skipped",
            "method": None,
            "note": "At least two paired subjects are required.",
        }

    import importlib

    try:
        scipy_stats = importlib.import_module("scipy.stats")
        result = scipy_stats.ttest_rel(
            comparison_values,
            reference_values,
            alternative="two-sided",
            nan_policy="omit",
        )
        statistic = None if _is_nan(result.statistic) else float(result.statistic)
        p_value = None if _is_nan(result.pvalue) else float(result.pvalue)
        return {
            "statistic": statistic,
            "p_value": p_value,
            "status": "completed",
            "method": "scipy",
            "note": None,
        }
    except ModuleNotFoundError:
        return {
            "statistic": None,
            "p_value": None,
            "status": "unavailable",
            "method": None,
            "note": "scipy is not installed.",
        }


def _annotate_paired_t_tests_on_violin_plot(
    ax,
    *,
    split_name: str,
    model_order: Sequence[str],
    paired_t_tests_df: Any,
    annotation_positions_by_label: Mapping[str, tuple[float, float]],
) -> None:
    if paired_t_tests_df.empty or len(model_order) <= 1:
        return

    split_tests_df = paired_t_tests_df[
        paired_t_tests_df["split"] == split_name
    ].copy()
    if split_tests_df.empty:
        return

    for position, comparison_model_label in enumerate(model_order[1:], start=1):
        model_tests_df = split_tests_df[
            split_tests_df["comparison_model_label"] == comparison_model_label
        ]
        if model_tests_df.empty:
            continue
        test_row = model_tests_df.iloc[0]
        significance_label = str(test_row.get("significance_label") or "")
        p_value_text = _format_p_value_for_annotation(test_row.get("p_value"))
        n_subjects = int(test_row.get("n_subjects") or 0)
        symbol_y, text_y = annotation_positions_by_label.get(
            str(comparison_model_label),
            (0.885, 0.845),
        )
        if significance_label:
            ax.text(
                float(position),
                float(symbol_y),
                significance_label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                clip_on=True,
            )
        p_line = p_value_text if p_value_text.startswith("p") else f"p={p_value_text}"
        ax.text(
            float(position),
            float(text_y),
            f"{p_line}\nn={n_subjects}",
            ha="center",
            va="center",
            fontsize=7,
            clip_on=True,
        )


def _resolve_uniform_violin_annotation_positions(
    *,
    model_order: Sequence[str],
    split_values: Sequence[float],
) -> dict[str, tuple[float, float]]:
    import numpy as np

    finite_values = np.asarray(list(split_values), dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        symbol_y = 0.875
    else:
        symbol_y = min(0.885, float(finite_values.max()) + 0.015)
    text_y = 0.535
    return {
        str(model_label): (float(symbol_y), float(text_y))
        for model_label in model_order
    }


def _subject_superiority_counts(points_df: Any) -> tuple[int, int]:
    if points_df.empty:
        return (0, 0)

    comparison_superior_count = 0
    first_model_superior_count = 0
    for point in points_df.to_dict(orient="records"):
        x_value = point.get("x")
        y_value = point.get("y")
        if x_value is None or y_value is None or _is_nan(x_value) or _is_nan(y_value):
            continue
        if float(y_value) > float(x_value):
            comparison_superior_count += 1
        elif float(x_value) > float(y_value):
            first_model_superior_count += 1
    return (comparison_superior_count, first_model_superior_count)


def _format_significance_label(p_value: float | None) -> str:
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _format_p_value_for_annotation(p_value: float | None) -> str:
    if p_value is None:
        return "n/a"
    if p_value < 0.001:
        return "p<=0.001"
    if p_value < 0.01:
        return f"{float(p_value):.4f}"
    return f"{float(p_value):.3f}"


def _build_subject_comparison_points(
    *,
    session_metrics_df: Any,
    subject_metrics_df: Any,
    split_name: str,
    first_model_label: str,
    comparison_model_label: str,
    n_bootstrap: int,
) -> Any:
    pd = _import_pandas()

    first_subject_df = subject_metrics_df[
        (subject_metrics_df["split"] == split_name)
        & (subject_metrics_df["model_label"] == first_model_label)
    ].copy()
    comparison_subject_df = subject_metrics_df[
        (subject_metrics_df["split"] == split_name)
        & (subject_metrics_df["model_label"] == comparison_model_label)
    ].copy()
    if first_subject_df.empty or comparison_subject_df.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "curriculum_name",
                "x",
                "y",
                "x_low",
                "x_high",
                "y_low",
                "y_high",
            ]
        )

    comparison_by_subject = {
        _normalize_identifier(row["subject_id"]): row
        for row in comparison_subject_df.to_dict(orient="records")
        if row.get("likelihood") is not None and not _is_nan(row.get("likelihood"))
    }
    rows: list[dict[str, Any]] = []
    for first_row in first_subject_df.to_dict(orient="records"):
        subject_id = _normalize_identifier(first_row["subject_id"])
        comparison_row = comparison_by_subject.get(subject_id)
        if comparison_row is None:
            continue
        x_value = first_row.get("likelihood")
        y_value = comparison_row.get("likelihood")
        if x_value is None or y_value is None or _is_nan(x_value) or _is_nan(y_value):
            continue

        first_session_df = session_metrics_df[
            (session_metrics_df["split"] == split_name)
            & (session_metrics_df["model_label"] == first_model_label)
            & (session_metrics_df["subject_id"] == subject_id)
        ].copy()
        comparison_session_df = session_metrics_df[
            (session_metrics_df["split"] == split_name)
            & (session_metrics_df["model_label"] == comparison_model_label)
            & (session_metrics_df["subject_id"] == subject_id)
        ].copy()

        x_low, x_high = _bootstrap_subject_likelihood_interval(
            first_session_df,
            n_bootstrap=n_bootstrap,
            random_seed=_stable_bootstrap_seed(first_model_label, split_name, subject_id),
        )
        y_low, y_high = _bootstrap_subject_likelihood_interval(
            comparison_session_df,
            n_bootstrap=n_bootstrap,
            random_seed=_stable_bootstrap_seed(
                comparison_model_label,
                split_name,
                subject_id,
            ),
        )
        if _is_nan(x_low) or _is_nan(x_high):
            x_low = float(x_value)
            x_high = float(x_value)
        if _is_nan(y_low) or _is_nan(y_high):
            y_low = float(y_value)
            y_high = float(y_value)
        rows.append(
            {
                "subject_id": subject_id,
                "curriculum_name": _resolve_comparison_curriculum_name(
                    first_row.get("curriculum_name"),
                    comparison_row.get("curriculum_name"),
                ),
                "x": float(x_value),
                "y": float(y_value),
                "x_low": float(x_low),
                "x_high": float(x_high),
                "y_low": float(y_low),
                "y_high": float(y_high),
            }
        )

    return pd.DataFrame(rows)


def _bootstrap_subject_likelihood_interval(
    session_metrics_df: Any,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[float, float]:
    import numpy as np

    if session_metrics_df.empty:
        return math.nan, math.nan

    log_likelihoods = session_metrics_df["total_log_likelihood"].astype(float).to_numpy()
    total_trials = session_metrics_df["total_trials"].astype(int).to_numpy()
    observed_likelihood = _optional_normalized_likelihood_from_log_stats(
        float(log_likelihoods.sum()),
        int(total_trials.sum()),
    )
    if observed_likelihood is None or _is_nan(observed_likelihood):
        return math.nan, math.nan
    if len(log_likelihoods) <= 1:
        return float(observed_likelihood), float(observed_likelihood)

    rng = np.random.default_rng(random_seed)
    sample_indices = rng.integers(
        0,
        len(log_likelihoods),
        size=(max(1, int(n_bootstrap)), len(log_likelihoods)),
    )
    sampled_log_likelihoods = log_likelihoods[sample_indices].sum(axis=1)
    sampled_total_trials = total_trials[sample_indices].sum(axis=1)
    valid_mask = sampled_total_trials > 0
    if not np.any(valid_mask):
        return float(observed_likelihood), float(observed_likelihood)
    bootstrap_likelihoods = np.exp(
        sampled_log_likelihoods[valid_mask] / sampled_total_trials[valid_mask]
    )
    low, high = np.quantile(bootstrap_likelihoods, [0.025, 0.975])
    return float(low), float(high)


def _stable_bootstrap_seed(*parts: Any) -> int:
    seed = 17
    modulus = 2**32 - 5
    for part in parts:
        for character in str(part):
            seed = (seed * 131 + ord(character)) % modulus
    return int(seed)


def _resolve_comparison_curriculum_name(*curriculum_values: Any) -> str:
    normalized_values = []
    for curriculum_value in curriculum_values:
        if curriculum_value is None or _is_nan(curriculum_value):
            continue
        curriculum_name = str(curriculum_value).strip()
        if not curriculum_name or curriculum_name == "Unknown":
            continue
        normalized_values.append(curriculum_name)

    if not normalized_values:
        return "Unknown"
    unique_values = list(dict.fromkeys(normalized_values))
    if len(unique_values) == 1:
        return unique_values[0]
    return "Mixed"


def _build_curriculum_palette(subject_metrics_df: Any) -> dict[str, str]:
    if subject_metrics_df.empty:
        return dict(_DEFAULT_CURRICULUM_COLORS)

    import matplotlib.pyplot as plt

    curriculum_values = list(
        dict.fromkeys(
            str(value)
            for value in subject_metrics_df["curriculum_name"].fillna("Unknown").tolist()
        )
    )
    ordered_curricula = [
        curriculum_name
        for curriculum_name in curriculum_values
        if curriculum_name not in _DEFAULT_CURRICULUM_COLORS
    ]
    cmap = plt.get_cmap("tab10")
    palette = {
        curriculum_name: cmap(index % max(1, cmap.N))
        for index, curriculum_name in enumerate(ordered_curricula)
    }
    palette.update(_DEFAULT_CURRICULUM_COLORS)
    return {str(key): _to_hex_color(value) for key, value in palette.items()}


def _metadata_for_heldout_eval(
    run_config: Mapping[str, Any],
    *,
    raw_df: Any,
) -> dict[str, Any]:
    data_cfg = _as_dict(run_config.get("data", {}))
    metadata = {
        "ignore_policy": str(data_cfg.get("ignore_policy", "exclude")),
        "features": data_cfg.get("features"),
    }
    subject_curricula = _build_subject_curriculum_map(raw_df, metadata={})
    if subject_curricula:
        metadata["subject_curricula"] = subject_curricula
    return metadata


def _heldout_selector_status(run_config: Mapping[str, Any]) -> tuple[bool, str | None]:
    from utils.load_mice_database import MICE_DATABASE_DATA_TYPES

    data_cfg = _as_dict(run_config.get("data", {}))
    heldout_eval = data_cfg.get("heldout_eval")
    if heldout_eval is not None:
        enabled = bool(heldout_eval)
    else:
        # Auto-enable for database runs (where a heldout set is derived); read
        # ``type`` or, for round-tripped configs, ``data_type``.
        data_type = data_cfg.get("type", data_cfg.get("data_type"))
        enabled = data_type in MICE_DATABASE_DATA_TYPES
    if not enabled:
        return False, (
            "Held-out evaluation is not enabled for this run "
            "(data.type is not a database run; set data.heldout_eval=true to force)."
        )
    return True, None


def _heldout_cols_to_retain(data_cfg: Mapping[str, Any]) -> list[str] | None:
    cols_to_retain = data_cfg.get("cols_to_retain")
    if cols_to_retain is None:
        return None
    resolved_cols = list(cols_to_retain)
    for required_column in (
        "trial",
        "subject_id",
        "ses_idx",
        "animal_response",
        "earned_reward",
        "curriculum_name",
    ):
        if required_column not in resolved_cols:
            resolved_cols.append(required_column)
    return resolved_cols


def _load_baseline_output_for_run(run: ResolvedLikelihoodRun) -> dict[str, Any]:
    if run.baseline_output_path is None:
        raise ValueError("baseline RL evaluation requires baseline_output_path.")
    baseline_output = _load_structured_file(Path(run.baseline_output_path))
    if not isinstance(baseline_output, Mapping):
        raise ValueError(
            f"Expected mapping-style baseline output in {run.baseline_output_path}"
        )
    return dict(baseline_output)


def _baseline_output_is_multisubject_per_subject(
    baseline_output: Mapping[str, Any],
) -> bool:
    return bool(baseline_output.get("multisubject")) or str(
        baseline_output.get("fit_strategy", "")
    ) == "per_subject"


def _resolve_baseline_agent_spec(
    run: ResolvedLikelihoodRun,
    baseline_output: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    model_cfg = _as_dict(_as_dict(run.run_config).get("model", {}))
    agent_class_name = str(
        model_cfg.get("agent_class", baseline_output.get("agent_class", ""))
    ).strip()
    if not agent_class_name:
        raise ValueError("Could not resolve baseline RL agent_class.")
    agent_kwargs = baseline_output.get("agent_kwargs", model_cfg.get("agent_kwargs", {}))
    if not isinstance(agent_kwargs, Mapping):
        agent_kwargs = {}
    return agent_class_name, dict(agent_kwargs)


def _extract_subject_fitted_params(subject_fit_summary: Any) -> dict[str, float]:
    if isinstance(subject_fit_summary, Mapping) and isinstance(
        subject_fit_summary.get("fitted_params"), Mapping
    ):
        params = subject_fit_summary["fitted_params"]
    elif isinstance(subject_fit_summary, Mapping):
        params = subject_fit_summary
    else:
        raise ValueError("Subject fit summary must be a mapping.")
    return {
        str(param_name): float(param_value)
        for param_name, param_value in dict(params).items()
    }


def _extract_sessions_from_df(df: Any) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    import pandas as pd

    normalized_df = _normalize_raw_dataframe(pd.DataFrame(df).copy())
    required_cols = {"ses_idx", "trial", "animal_response", "earned_reward"}
    missing_cols = [column for column in required_cols if column not in normalized_df.columns]
    if missing_cols:
        raise ValueError(f"Dataframe missing required baseline RL columns: {missing_cols}")

    choice_sessions: list[Any] = []
    reward_sessions: list[Any] = []
    session_ids: list[Any] = []
    session_subject_ids: list[Any] = []

    ordered_session_ids = list(dict.fromkeys(normalized_df["ses_idx"].tolist()))
    for session_id in ordered_session_ids:
        session_df = normalized_df[normalized_df["ses_idx"] == session_id].sort_values("trial")
        choice_arr = session_df["animal_response"].to_numpy(dtype=int)
        valid_choice = (choice_arr == 0) | (choice_arr == 1)
        choice_arr = choice_arr[valid_choice]
        if len(choice_arr) == 0:
            continue
        reward_arr = session_df["earned_reward"].to_numpy(dtype=float)[valid_choice]
        choice_sessions.append(choice_arr.astype(int))
        reward_sessions.append(reward_arr.astype(float))
        session_ids.append(str(session_id))
        if "subject_id" in session_df.columns:
            session_subject_ids.append(_normalize_identifier(session_df["subject_id"].iloc[0]))
        else:
            session_subject_ids.append("unknown")
    return choice_sessions, reward_sessions, session_ids, session_subject_ids


def _extract_session_score_masks(
    df: Any,
    score_partition: tuple[str, str] | None,
) -> list[Any] | None:
    """Build per-session score masks aligned to valid binary-choice trials."""
    if score_partition is None:
        return None

    import pandas as pd

    normalized_df = _normalize_raw_dataframe(pd.DataFrame(df).copy())
    partition_column, partition_value = score_partition
    if partition_column not in normalized_df.columns:
        raise ValueError(
            f"Baseline raw data is missing trial partition column {partition_column!r}."
        )

    score_masks: list[Any] = []
    ordered_session_ids = list(dict.fromkeys(normalized_df["ses_idx"].tolist()))
    for session_id in ordered_session_ids:
        session_df = normalized_df[normalized_df["ses_idx"] == session_id].sort_values("trial")
        choice_arr = session_df["animal_response"].to_numpy(dtype=int)
        valid_choice = (choice_arr == 0) | (choice_arr == 1)
        if not valid_choice.any():
            continue
        score_masks.append(
            (
                session_df[partition_column].astype(str).to_numpy()[valid_choice]
                == str(partition_value)
            )
        )
    if not score_masks or not any(bool(mask.any()) for mask in score_masks):
        raise ValueError(
            f"No valid baseline trials remain for partition {partition_value!r}."
        )
    return score_masks


def _build_subject_curriculum_map(
    raw_df: Any,
    *,
    metadata: Mapping[str, Any],
) -> dict[Any, str]:
    import pandas as pd

    subject_curricula = metadata.get("subject_curricula")
    if isinstance(subject_curricula, Mapping) and subject_curricula:
        return {
            _normalize_identifier(subject_id): str(curriculum_name)
            for subject_id, curriculum_name in dict(subject_curricula).items()
        }

    normalized_df = _normalize_raw_dataframe(pd.DataFrame(raw_df).copy())
    if "subject_id" not in normalized_df.columns or "curriculum_name" not in normalized_df.columns:
        return {}

    subject_curriculum_map: dict[Any, str] = {}
    for subject_id, subject_rows in normalized_df.groupby("subject_id", sort=False):
        curricula = [
            str(value)
            for value in subject_rows["curriculum_name"].dropna().unique().tolist()
        ]
        normalized_subject_id = _normalize_identifier(subject_id)
        if not curricula:
            subject_curriculum_map[normalized_subject_id] = "Unknown"
        elif len(curricula) == 1:
            subject_curriculum_map[normalized_subject_id] = curricula[0]
        else:
            subject_curriculum_map[normalized_subject_id] = "Mixed"
    return subject_curriculum_map


def _resolve_curriculum_name(
    session_df: Any,
    *,
    subject_id: Any,
    subject_curriculum_map: Mapping[Any, str],
) -> str:
    if subject_id in subject_curriculum_map:
        return str(subject_curriculum_map[subject_id])
    if "curriculum_name" not in session_df.columns:
        return "Unknown"
    curricula = [
        str(value)
        for value in session_df["curriculum_name"].dropna().unique().tolist()
    ]
    if not curricula:
        return "Unknown"
    if len(curricula) == 1:
        return curricula[0]
    return "Mixed"


def _subset_raw_df_by_session_ids(raw_df: Any, session_ids: Sequence[Any]) -> Any:
    pd = _import_pandas()

    normalized_df = _normalize_raw_dataframe(pd.DataFrame(raw_df).copy())
    ordered_session_ids = [str(session_id) for session_id in session_ids]
    subset_df = normalized_df[normalized_df["ses_idx"].isin(ordered_session_ids)].copy()
    subset_df["ses_idx"] = pd.Categorical(
        subset_df["ses_idx"],
        categories=ordered_session_ids,
        ordered=True,
    )
    sort_columns = ["ses_idx"]
    if "trial" in subset_df.columns:
        sort_columns.append("trial")
    subset_df = subset_df.sort_values(sort_columns).reset_index(drop=True)
    subset_df["ses_idx"] = subset_df["ses_idx"].astype(str)
    return subset_df


def _normalize_raw_dataframe(raw_df: Any) -> Any:
    pd = _import_pandas()

    df = pd.DataFrame(raw_df).copy()
    if "ses_idx" in df.columns:
        df["ses_idx"] = df["ses_idx"].map(str)
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].map(_normalize_identifier)
    return df


def _deduplicate_model_labels(
    model_dirs: Sequence[str],
    *,
    model_labels: Sequence[str] | None,
) -> list[str]:
    if model_labels is not None and len(model_labels) != len(model_dirs):
        raise ValueError("model_labels must have the same length as model_dirs.")

    base_labels = [
        str(model_labels[index]).strip()
        if model_labels is not None
        else Path(model_dir).name
        for index, model_dir in enumerate(model_dirs)
    ]
    deduplicated: list[str] = []
    label_counts: dict[str, int] = {}
    for index, base_label in enumerate(base_labels):
        label = base_label or f"model_{index}"
        suffix_count = label_counts.get(label, 0)
        label_counts[label] = suffix_count + 1
        deduplicated.append(label if suffix_count == 0 else f"{label}_{suffix_count + 1}")
    return deduplicated


def _concat_metric_frames(metric_frames: Sequence[Any], columns: Sequence[str]) -> Any:
    pd = _import_pandas()
    import numpy as np

    if not metric_frames:
        return _make_dataframe([], columns)
    non_empty_frames = [pd.DataFrame(frame).copy() for frame in metric_frames if not frame.empty]
    if not non_empty_frames:
        return _make_dataframe([], columns)
    concatenated = pd.concat(non_empty_frames, ignore_index=True)
    for column in columns:
        if column not in concatenated.columns:
            concatenated[column] = (
                None if column in _NON_NUMERIC_METRIC_COLUMNS else np.nan
            )
    return concatenated.loc[:, list(columns)]


def _make_dataframe(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Any:
    pd = _import_pandas()

    if not rows:
        return pd.DataFrame(columns=list(columns))
    df = pd.DataFrame(list(rows))
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df.loc[:, list(columns)]


def _sort_metrics_dataframe(
    df: Any,
    *,
    split_column: str,
    sort_columns: Sequence[str],
) -> Any:
    pd = _import_pandas()

    if df.empty or split_column not in df.columns:
        return pd.DataFrame(df).copy()
    sorted_df = pd.DataFrame(df).copy()
    sorted_df[split_column] = pd.Categorical(
        sorted_df[split_column],
        categories=list(_SPLIT_ORDER),
        ordered=True,
    )
    sorted_df = sorted_df.sort_values(list(sort_columns)).reset_index(drop=True)
    sorted_df[split_column] = sorted_df[split_column].astype(str)
    return sorted_df


def _session_log_likelihood_from_choice_prob(
    choices: Any,
    choice_prob: Any,
    *,
    score_mask: Any | None = None,
) -> tuple[float, int]:
    import numpy as np

    choice_prob_array = np.asarray(choice_prob, dtype=float)
    if choice_prob_array.ndim != 2:
        raise ValueError(
            f"choice_prob must be 2D, received shape={choice_prob_array.shape}"
        )
    if choice_prob_array.shape[0] == 2:
        aligned_prob = choice_prob_array
    elif choice_prob_array.shape[1] == 2:
        aligned_prob = choice_prob_array.T
    else:
        raise ValueError(
            "choice_prob must have one axis of length 2 for binary choices; "
            f"received shape={choice_prob_array.shape}"
        )

    choices_array = np.asarray(choices, dtype=int)
    n_trials = min(len(choices_array), int(aligned_prob.shape[1]))
    if n_trials <= 0:
        return 0.0, 0

    choices_array = choices_array[:n_trials]
    valid_choice = (choices_array == 0) | (choices_array == 1)
    if score_mask is not None:
        score_mask_array = np.asarray(score_mask, dtype=bool)
        if len(score_mask_array) != len(choices_array):
            raise ValueError(
                "score_mask length must match the aligned baseline choice sequence."
            )
        valid_choice &= score_mask_array
    if not np.any(valid_choice):
        return 0.0, 0

    trial_index = np.arange(n_trials)[valid_choice]
    probs = aligned_prob[choices_array[valid_choice], trial_index]
    probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
    total_log_likelihood = float(np.sum(np.log(probs)))
    total_trials = int(np.sum(valid_choice))
    return total_log_likelihood, total_trials


def _optional_normalized_likelihood_from_log_stats(
    total_log_likelihood: float,
    total_trials: int,
) -> float | None:
    if total_trials <= 0:
        return None
    return float(math.exp(float(total_log_likelihood) / float(total_trials)))


def _session_sem(values: Sequence[float]) -> float:
    import numpy as np

    values_array = np.asarray(list(values), dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if len(values_array) < 2:
        return 0.0
    return float(np.std(values_array, ddof=1) / math.sqrt(len(values_array)))


def _normalize_identifier(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except ModuleNotFoundError:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return dict(value)


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ModuleNotFoundError:
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ModuleNotFoundError:
        pass
    return value


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _to_hex_color(value: Any) -> str:
    import matplotlib.colors as mcolors

    return str(mcolors.to_hex(value))


def _import_pandas():
    import pandas as pd

    return pd
