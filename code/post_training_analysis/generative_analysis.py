"""Standalone post-training generative analysis helpers.

This module is intentionally light on import-time dependencies so artifact
resolution and switch-statistics helpers remain usable in minimal Python
environments. Runtime-heavy dependencies such as pandas, NumPy, JAX, Haiku,
and the AIND packages are imported lazily inside the functions that need them.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import logging
import math
import os
import pickle
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from utils.env_config import get_env

logger = logging.getLogger(__name__)

_TRAIN_SPLIT = "train"
_HELDOUT_SPLIT = "heldout"
_CHECKPOINT_POLICY_BEST_EVAL = "best_eval"
_CHECKPOINT_POLICY_BEST_HELDOUT = "best_heldout"
_CHECKPOINT_POLICY_FINAL = "final"
_ROLLOUT_MODE_CURRICULUM_MATCHED = "curriculum_matched"
# Action-class index of the "ignore"/no-response action in 3-way
# (ignore_policy="include") models. Matches the data loader's animal_response
# encoding (0=left, 1=right, 2=ignore) and gru_trainer's _IGNORE_CLASS_INDEX.
_IGNORE_ACTION = 2
_DEFAULT_WINDOW_SIZE = 10
_DEFAULT_HISTORY_MAX_TRIALS_BACK = 3
_DEFAULT_HISTORY_AGGREGATE_MIN_TRIALS = 10
_DEFAULT_HISTORY_SUBJECT_MIN_TRIALS = 5
_SUBJECT_LEVEL_MIN_ANIMAL_N = 5
_DEFAULT_SUBJECT_BOOTSTRAP_RESAMPLES = int(
    get_env("BFM_SUBJECT_BOOTSTRAP_RESAMPLES", "250")
)  # 250 gives fine subject-level CIs at a fraction of the 1000-resample cost
_DEFAULT_SUBJECT_BOOTSTRAP_SEED = 0
_REWARD_CONDITIONS = ("rewarded", "unrewarded")
_RUN_LENGTH_CONDITIONS = ("run_length_1", "run_length_gt1")
_HISTORY_PATTERN_TYPES = ("detailed", "abstract")
_SESSION_PARTITION_TRAIN = "train"
_SESSION_PARTITION_EVAL = "eval"
_SESSION_PARTITION_COMBINED = "combined"


@dataclass
class ResolvedModelRun:
    """Resolved metadata and artifact paths for a trained model run."""

    model_dir: str
    inputs_path: str
    outputs_dir: str
    model_type: str
    split: str
    checkpoint_policy: str
    checkpoint_step: int | None
    checkpoint_label: str
    params_path: str
    config_path: str
    seed: int | None
    multisubject: bool
    mature_only: bool
    ignore_policy: str
    curricula: list[str]
    features: dict[str, Any] | None
    selection: dict[str, Any]
    baseline_output_path: str | None = None
    output_summary_path: str | None = None
    checkpoint_index_path: str | None = None
    subject_index_map_path: str | None = None
    session_context_map_path: str | None = None
    checkpoint_selection_reason: str | None = None
    fallback_reason: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    run_config: dict[str, Any] = field(default_factory=dict)
    trained_subject_ids: list[Any] | None = None
    resolved_subject_ids: list[Any] | None = None
    resolved_session_ids: list[str] | None = None
    session_split_manifest: dict[str, Any] | None = None
    session_conditioning_enabled: bool = False
    session_conditioning_encoding_type: str = "none"
    required_session_conditioning_artifacts: list[str] = field(default_factory=list)
    missing_session_conditioning_artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _to_serializable(asdict(self))


def resolve_model_run(
    model_dir: str | Path,
    split: str = _TRAIN_SPLIT,
    checkpoint_policy: str = _CHECKPOINT_POLICY_BEST_EVAL,
) -> ResolvedModelRun:
    """Resolve the trained-model artifacts needed for post-training analysis."""

    started_at = time.perf_counter()
    normalized_split = _normalize_split(split)
    normalized_policy = _normalize_checkpoint_policy(checkpoint_policy)
    model_dir_path = Path(model_dir).expanduser().resolve()
    inputs_path = model_dir_path / "inputs.yaml"
    outputs_dir = model_dir_path / "outputs"

    if not inputs_path.exists():
        raise FileNotFoundError(f"Could not find run inputs at {inputs_path}")
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Could not find run outputs at {outputs_dir}")

    run_config = _load_structured_file(inputs_path)
    if not isinstance(run_config, dict):
        raise ValueError(f"Expected mapping-style config in {inputs_path}")

    data_cfg = _as_dict(run_config.get("data", {}))
    model_cfg = _as_dict(run_config.get("model", {}))
    architecture_cfg = _as_dict(model_cfg.get("architecture", {}))
    model_type = str(model_cfg.get("type", "")).strip().lower()
    if model_type not in {"baseline_rl", "disrnn", "gru"}:
        raise ValueError(
            f"Unsupported model.type='{model_type}' in {inputs_path}. "
            "Only baseline_rl, GRU, and disRNN runs are supported."
        )

    selection = _resolve_split_selection(data_cfg, normalized_split)
    output_summary_path = outputs_dir / "output_summary.json"
    checkpoint_index_path = outputs_dir / "checkpoints" / "index.json"
    output_summary = (
        _load_structured_file(output_summary_path)
        if output_summary_path.exists()
        else {}
    )
    checkpoint_index = (
        _load_structured_file(checkpoint_index_path)
        if checkpoint_index_path.exists()
        else {}
    )
    if not isinstance(output_summary, dict):
        output_summary = {}
    if not isinstance(checkpoint_index, dict):
        checkpoint_index = {}

    baseline_output_path = None
    if model_type == "baseline_rl":
        baseline_output_path = outputs_dir / "baseline_rl_output.json"
        if not baseline_output_path.exists():
            raise FileNotFoundError(
                f"Could not find baseline RL output at {baseline_output_path}"
            )
        model_config = _load_structured_file(baseline_output_path)
        if not isinstance(model_config, Mapping):
            raise ValueError(
                f"Expected mapping-style baseline RL output in {baseline_output_path}"
            )
        config_path = baseline_output_path
        params_path = baseline_output_path
        checkpoint_step = None
        checkpoint_label = "final_fit"
        reason = (
            "Loaded final fitted parameters from outputs/baseline_rl_output.json."
        )
        if normalized_policy != _CHECKPOINT_POLICY_FINAL:
            reason = (
                "Baseline RL runs do not use checkpointed params; "
                "using final fitted parameters from outputs/baseline_rl_output.json."
            )
        fallback_reason = None
    else:
        config_name = "disrnn_config.json" if model_type == "disrnn" else "gru_config.json"
        config_path = outputs_dir / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find saved model config at {config_path}")
        model_config = _load_structured_file(config_path)
        if not isinstance(model_config, dict):
            raise ValueError(f"Expected mapping-style model config in {config_path}")
        params_path, checkpoint_step, checkpoint_label, reason, fallback_reason = (
            _resolve_checkpoint_artifact(
                model_dir=model_dir_path,
                outputs_dir=outputs_dir,
                checkpoint_policy=normalized_policy,
                checkpoint_index=checkpoint_index,
                output_summary=output_summary,
            )
        )

    is_multisubject = bool(
        architecture_cfg.get("multisubject", False)
        or data_cfg.get("multisubject", False)
        or (
            model_type == "baseline_rl"
            and _baseline_output_is_multisubject_per_subject(_as_dict(model_config))
        )
    )

    subject_index_map_path = None
    session_context_map_path = None
    trained_subject_ids = None
    if is_multisubject:
        if normalized_split != _TRAIN_SPLIT:
            raise NotImplementedError(
                "Held-out post-training analysis is not supported for multisubject "
                "runs. V1 supports seen-subject personalization only."
            )
        subject_index_map_path = outputs_dir / "subject_index_map.json"
        if not subject_index_map_path.exists():
            raise FileNotFoundError(
                f"Could not find saved subject index map at {subject_index_map_path}"
            )
        trained_subject_ids = _load_trained_subject_ids_from_subject_index_map(
            subject_index_map_path
        )
        if not trained_subject_ids:
            raise ValueError(
                "Resolved multisubject run does not contain any trained subjects in "
                f"{subject_index_map_path}."
            )
        resolved_session_context_path = outputs_dir / "session_context_map.json"
        if resolved_session_context_path.exists():
            session_context_map_path = resolved_session_context_path

    resolved = ResolvedModelRun(
        model_dir=str(model_dir_path),
        inputs_path=str(inputs_path),
        outputs_dir=str(outputs_dir),
        model_type=model_type,
        split=normalized_split,
        checkpoint_policy=normalized_policy,
        checkpoint_step=checkpoint_step,
        checkpoint_label=checkpoint_label,
        params_path=str(params_path),
        config_path=str(config_path),
        baseline_output_path=(
            str(baseline_output_path) if baseline_output_path is not None else None
        ),
        seed=_coerce_optional_int(run_config.get("seed", model_cfg.get("seed"))),
        multisubject=is_multisubject,
        mature_only=bool(data_cfg.get("mature_only", True)),
        ignore_policy=str(data_cfg.get("ignore_policy", "exclude")),
        curricula=[str(v) for v in data_cfg.get("curricula") or []],
        features=_coerce_optional_dict(data_cfg.get("features")),
        selection=selection,
        output_summary_path=str(output_summary_path) if output_summary_path.exists() else None,
        checkpoint_index_path=str(checkpoint_index_path)
        if checkpoint_index_path.exists()
        else None,
        subject_index_map_path=(
            str(subject_index_map_path) if subject_index_map_path is not None else None
        ),
        session_context_map_path=(
            str(session_context_map_path) if session_context_map_path is not None else None
        ),
        checkpoint_selection_reason=reason,
        fallback_reason=fallback_reason,
        model_config=_to_serializable(model_config),
        run_config=_to_serializable(run_config),
        trained_subject_ids=trained_subject_ids,
    )
    resolved = _refresh_resolved_run_session_conditioning_metadata(resolved)
    logger.info(
        "Resolved model run in %.2fs: model_type=%s split=%s checkpoint=%s step=%s "
        "session_conditioning=%s encoding=%s",
        time.perf_counter() - started_at,
        resolved.model_type,
        resolved.split,
        resolved.checkpoint_label,
        resolved.checkpoint_step,
        resolved.session_conditioning_enabled,
        resolved.session_conditioning_encoding_type,
    )
    return resolved


def _baseline_output_is_multisubject_per_subject(
    baseline_output: Mapping[str, Any],
) -> bool:
    return bool(baseline_output.get("multisubject")) or str(
        baseline_output.get("fit_strategy", "")
    ).strip().lower() == "per_subject"


def _load_baseline_output_from_resolved_run(
    resolved_run: ResolvedModelRun,
) -> dict[str, Any]:
    baseline_output = _as_dict(resolved_run.model_config)
    if baseline_output and (
        "agent_class" in baseline_output
        or "fitted_params" in baseline_output
        or "fitted_params_per_subject" in baseline_output
        or "fit_strategy" in baseline_output
    ):
        return baseline_output

    if resolved_run.baseline_output_path is None:
        raise ValueError(
            "Resolved baseline RL run does not define baseline_output_path."
        )

    loaded_output = _load_structured_file(
        Path(resolved_run.baseline_output_path).expanduser().resolve()
    )
    if not isinstance(loaded_output, Mapping):
        raise ValueError(
            "Expected mapping-style baseline RL output in "
            f"{resolved_run.baseline_output_path}."
        )
    return dict(loaded_output)


def _resolve_baseline_agent_spec(
    resolved_run: ResolvedModelRun,
    baseline_output: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    model_cfg = _as_dict(_as_dict(resolved_run.run_config).get("model", {}))
    agent_class_name = str(
        model_cfg.get("agent_class", baseline_output.get("agent_class", ""))
    ).strip()
    if not agent_class_name:
        raise ValueError("Could not resolve baseline RL agent_class.")
    agent_kwargs = baseline_output.get("agent_kwargs", model_cfg.get("agent_kwargs", {}))
    if not isinstance(agent_kwargs, Mapping):
        agent_kwargs = {}
    return agent_class_name, dict(agent_kwargs)


def _extract_baseline_subject_fitted_params(
    subject_fit_summary: Any,
) -> dict[str, float]:
    if isinstance(subject_fit_summary, Mapping) and isinstance(
        subject_fit_summary.get("fitted_params"), Mapping
    ):
        params = subject_fit_summary["fitted_params"]
    elif isinstance(subject_fit_summary, Mapping):
        params = subject_fit_summary
    else:
        raise ValueError("Subject fit summary must be a mapping.")

    if not params:
        raise ValueError("Subject fit summary is missing fitted_params.")

    return {
        str(param_name): float(param_value)
        for param_name, param_value in dict(params).items()
    }


def _resolve_baseline_fitted_params_for_subject(
    resolved_run: ResolvedModelRun,
    baseline_output: Mapping[str, Any],
    *,
    subject_id: Any,
) -> dict[str, float]:
    if not _baseline_output_is_multisubject_per_subject(baseline_output):
        fitted_params = baseline_output.get("fitted_params")
        if not isinstance(fitted_params, Mapping) or not fitted_params:
            raise ValueError("baseline_rl_output.json is missing fitted_params.")
        return {
            str(param_name): float(param_value)
            for param_name, param_value in dict(fitted_params).items()
        }

    fitted_params_per_subject = baseline_output.get("fitted_params_per_subject")
    if not isinstance(fitted_params_per_subject, Mapping) or not fitted_params_per_subject:
        raise ValueError("baseline_rl_output.json is missing fitted_params_per_subject.")

    subject_key = str(subject_id)
    subject_fit_summary = fitted_params_per_subject.get(subject_key)
    if subject_fit_summary is None:
        raise ValueError(
            "Missing fitted_params_per_subject entry for subject_id="
            f"{subject_key!r} in {resolved_run.baseline_output_path or resolved_run.model_dir}."
        )
    return _extract_baseline_subject_fitted_params(subject_fit_summary)


def _extract_baseline_history_from_agent(
    agent: Any,
    *,
    getter_name: str,
    expected_length: int,
    label: str,
) -> list[float]:
    getter = getattr(agent, getter_name, None)
    if getter is None:
        raise AttributeError(
            f"Simulated baseline RL agent is missing {getter_name}()."
        )

    values = list(getter())
    if len(values) != int(expected_length):
        raise ValueError(
            f"Expected {label} length {expected_length}, got {len(values)}."
        )
    return [float(value) for value in values]


def load_animal_session_history(
    model_dir: str | Path | ResolvedModelRun,
    split: str = _TRAIN_SPLIT,
):
    """Load the animal sessions that correspond to a model run split.

    V1 now reconstructs animal trial histories directly from the behavioral
    snapshot. That keeps post-training analysis independent from the raw NWB
    loading path and avoids re-querying/rehydrating the original session
    assets when the snapshot already contains the trial-level choice and reward
    histories we need for switch statistics.
    """

    started_at = time.perf_counter()
    resolved_run = _coerce_resolved_run(model_dir, split=split)
    _validate_multisubject_analysis_split(resolved_run)

    load_mice_database_mod = importlib.import_module("utils.load_mice_database")

    snapshot_cols = [
        "trial",
        "subject_id",
        "ses_idx",
        "animal_response",
        "earned_reward",
        "curriculum_name",
        "task",  # needed to rebuild the pseudo-curriculum for off-curriculum mice
        "current_stage_actual",
    ]
    logger.info(
        "Loading animal session history from snapshot for model_dir=%s split=%s",
        resolved_run.model_dir,
        resolved_run.split,
    )
    snapshot_started_at = time.perf_counter()
    selection_subject_ids = (
        resolved_run.trained_subject_ids
        if resolved_run.multisubject
        else resolved_run.selection.get("subject_ids")
    )
    snapshot_df, selected_subject_ids = load_mice_database_mod.load_mice_from_database(
        split=resolved_run.split,
        subject_ids=selection_subject_ids,
        curricula=resolved_run.curricula or None,
        subject_ratio=resolved_run.selection.get("subject_ratio"),
        min_sessions=int(resolved_run.selection.get("min_sessions", 10)),
        heldout_every_n=int(resolved_run.selection.get("heldout_every_n", 5)),
        seed=resolved_run.selection.get("subject_sample_seed"),
        mature_only=resolved_run.mature_only,
        cols_to_retain=snapshot_cols,
        snapshot=resolved_run.selection.get("snapshot"),
    )
    logger.info(
        "Loaded snapshot selection in %.2fs: %d trial rows across %d selected subject%s",
        time.perf_counter() - snapshot_started_at,
        len(snapshot_df),
        len(selected_subject_ids),
        "" if len(selected_subject_ids) == 1 else "s",
    )

    if len(snapshot_df) == 0:
        raise ValueError(
            "Snapshot selection resolved to an empty dataset for "
            f"{resolved_run.model_dir} split={resolved_run.split}."
        )

    snapshot_df = _fill_offcurriculum_curriculum_name(snapshot_df)

    snapshot_df = _align_snapshot_df_with_ignore_policy(
        snapshot_df,
        ignore_policy=resolved_run.ignore_policy,
    )
    if len(snapshot_df) == 0:
        raise ValueError(
            "Snapshot selection resolved to an empty dataset after applying "
            f"ignore_policy={resolved_run.ignore_policy!r} for {resolved_run.model_dir} "
            f"split={resolved_run.split}."
        )

    snapshot_df = snapshot_df.copy()
    snapshot_df["subject_id"] = snapshot_df["subject_id"].map(_normalize_identifier)
    snapshot_df["ses_idx"] = snapshot_df["ses_idx"].map(str)

    # The database selection already contains trial-level choice/reward history
    # and the session annotations added by load_mice_from_database(). Building the
    # canonical session dataframe from this result avoids the heavier raw-data/NWB
    # path.
    build_started_at = time.perf_counter()
    session_history = _build_session_history_dataframe(snapshot_df)
    if resolved_run.multisubject and resolved_run.session_context_map_path:
        session_history = _align_multisubject_session_history_to_training_ids(
            session_history,
            resolved_run=resolved_run,
        )
    logger.info(
        "Built session history dataframe in %.2fs: %d session%s",
        time.perf_counter() - build_started_at,
        len(session_history),
        "" if len(session_history) == 1 else "s",
    )
    session_subject_ids = _series_like_to_list(session_history, "subject_id")
    session_ids = _series_like_to_list(session_history, "ses_idx")
    if session_subject_ids is not None:
        resolved_run.resolved_subject_ids = _unique_preserve_order(session_subject_ids)
    else:
        selected_subject_ids = _unique_preserve_order(selected_subject_ids)
        resolved_run.resolved_subject_ids = [
            _normalize_identifier(value) for value in selected_subject_ids
        ]
    if session_ids is not None:
        resolved_run.resolved_session_ids = sorted(str(value) for value in session_ids)
    logger.info(
        "Finished loading animal session history in %.2fs",
        time.perf_counter() - started_at,
    )
    return session_history


def _run_batched_rollout(runner, lanes, *, n_actions: int):
    """Roll out every lane in ``lanes`` in lockstep, batching the model step
    across lanes.

    Each ``lane`` is a dict with keys ``subject_id``, ``model_ses_idx``,
    ``source_ses_idx``, ``curriculum_name``, ``n_trials`` and ``seed``. Lanes
    are mutually independent, so we advance them together over trial index and
    call ``runner.step_batch`` once per trial for the whole batch — instead of
    the old per-lane loop that called the model at batch size 1, once per
    (lane, trial). Every lane keeps its own seeded RNG and curriculum-matched
    task and is sampled only while ``t < n_trials``, so its (choice, reward)
    history is identical to a sequential per-lane rollout; only the model step
    is batched. Returns ``(choice_histories, reward_histories)`` aligned with
    ``lanes``.
    """
    np = _import_dependency("numpy")
    n_lanes = len(lanes)
    if n_lanes == 0:
        return [], []
    max_trials = max(int(lane["n_trials"]) for lane in lanes)

    tasks: list[Any] = []
    rngs: list[Any] = []
    prefixes: list[list[float]] = []
    for lane in lanes:
        task = _build_curriculum_matched_task(
            curriculum_name=lane["curriculum_name"],
            n_trials=int(lane["n_trials"]),
            seed=int(lane["seed"]),
        )
        if hasattr(task, "reset"):
            task.reset()
        tasks.append(task)
        rngs.append(np.random.default_rng(int(lane["seed"])))
        # encode_inputs returns [*static_prefix, prev_choice, prev_reward]; the
        # prefix (subject/session indices) is constant across trials, so build
        # it once and only mutate the trailing [prev_choice, prev_reward].
        if hasattr(runner, "encode_inputs"):
            encoded = runner.encode_inputs(
                lane["subject_id"],
                lane["model_ses_idx"],
                [0.0, 0.0],
                source_session_id=lane["source_ses_idx"],
            )
        else:
            encoded = [0.0, 0.0]
        prefixes.append([float(value) for value in encoded[:-2]])

    prefix_width = len(prefixes[0])
    inputs_np = np.full((n_lanes, prefix_width + 2), -1.0, dtype=np.float32)
    for lane_index in range(n_lanes):
        if prefix_width:
            inputs_np[lane_index, :prefix_width] = prefixes[lane_index]

    choice_histories: list[list[float]] = [[] for _ in range(n_lanes)]
    reward_histories: list[list[float]] = [[] for _ in range(n_lanes)]
    state = runner.initial_state_batch(n_lanes)

    # 3-way (ignore_policy="include") models emit an extra "ignore" action at
    # class index _IGNORE_ACTION (=2), matching the data loader's animal_response
    # convention (0=left, 1=right, 2=ignore/no-response). An ignore is a
    # NON-ACTION: the mouse made no port choice, so we do NOT step the foraging
    # task (no poke registered, block/bait state does not advance), earn no
    # reward, and feed (prev_choice=2, prev_reward=0) back to the model — exactly
    # how a real ignore trial appears in the training inputs. For 2-way models
    # (n_actions==2) this branch never fires and behaviour is unchanged.
    for trial_index in range(max_trials):
        logits_batch, state = runner.step_batch(inputs_np, state)
        for lane_index in range(n_lanes):
            if trial_index >= int(lanes[lane_index]["n_trials"]):
                continue  # finished lane: leave its RNG/history untouched
            probs = _softmax(logits_batch[lane_index])
            choice = int(rngs[lane_index].choice(n_actions, p=probs))
            if choice == _IGNORE_ACTION:
                reward = 0.0  # no port chosen -> no reward, task not stepped
            else:
                reward = _step_task_reward(tasks[lane_index], choice)
            choice_histories[lane_index].append(float(choice))
            reward_histories[lane_index].append(float(reward))
            inputs_np[lane_index, -2] = float(choice)
            inputs_np[lane_index, -1] = float(reward)

    return choice_histories, reward_histories


def simulate_model_sessions(
    resolved_run: ResolvedModelRun | Mapping[str, Any],
    animal_sessions,
    rollout_mode: str = _ROLLOUT_MODE_CURRICULUM_MATCHED,
    n_rollouts_per_session: int = 1,
):
    """Simulate one or more model rollouts per source animal session."""

    normalized_rollout_mode = str(rollout_mode).strip().lower()
    if normalized_rollout_mode != _ROLLOUT_MODE_CURRICULUM_MATCHED:
        raise NotImplementedError(
            "V1 only supports rollout_mode='curriculum_matched'. "
            f"Received {rollout_mode!r}."
        )
    if int(n_rollouts_per_session) <= 0:
        raise ValueError("n_rollouts_per_session must be >= 1.")

    run = _coerce_resolved_run(resolved_run)
    _validate_multisubject_analysis_split(run)
    if run.ignore_policy not in ("exclude", "include"):
        raise NotImplementedError(
            "Post-training simulation supports ignore_policy in "
            "{'exclude','include'}. "
            f"Received {run.ignore_policy!r}."
        )
    # ignore_policy="include" -> 3-way (L/R/ignore) model: the rollout samples an
    # extra ignore action (class 2) and treats it as a non-action (see
    # _run_batched_rollout). baseline_rl remains 2-way only.
    if run.ignore_policy == "include" and run.model_type == "baseline_rl":
        raise NotImplementedError(
            "ignore_policy='include' (3-way) simulation is not supported for "
            "baseline_rl models (2-action generative model only)."
        )

    if run.model_type == "baseline_rl":
        return _simulate_baseline_model_sessions(
            resolved_run=run,
            animal_sessions=animal_sessions,
            rollout_mode=normalized_rollout_mode,
            n_rollouts_per_session=int(n_rollouts_per_session),
        )

    np = _import_dependency("numpy")
    pd = _import_dependency("pandas")
    runner = _restore_model_runner(run)
    animal_rows = list(_iter_session_records(animal_sessions))
    if hasattr(runner, "validate_subject_ids"):
        runner.validate_subject_ids([row.get("subject_id") for row in animal_rows])
    total_source_sessions = len(animal_rows)
    total_requested_rollouts = total_source_sessions * int(n_rollouts_per_session)

    logger.info(
        "Starting model simulation for %d source sessions (%d rollout%s) from %s "
        "[checkpoint=%s, mode=%s]",
        total_source_sessions,
        total_requested_rollouts,
        "" if total_requested_rollouts == 1 else "s",
        run.model_dir,
        run.checkpoint_label,
        normalized_rollout_mode,
    )

    # Sessions (rollouts) are mutually independent. The previous implementation
    # looped per session and called ``runner.step`` at batch size 1, once per
    # (session, trial) — ~n_sessions * n_trials sequential model dispatches,
    # dominated by per-call host/device-sync overhead rather than compute.
    # Instead we advance every lane (= one requested rollout) in lockstep and
    # batch the model step across lanes (one dispatch per trial for the whole
    # batch). Each lane keeps its own seeded RNG and curriculum-matched task, so
    # the produced histories match the per-session loop; only the step is
    # batched. Lanes are chunked to bound memory at large D. The chunk size is
    # overridable via BFM_GENERATIVE_ROLLOUT_CHUNK (chunk=1 reproduces the
    # old per-session batch-1 path bit-for-bit; larger trades reproducibility
    # against speed, since batched matmul differs from batch-1 at the last bit).
    BATCHED_ROLLOUT_CHUNK = int(get_env("BFM_GENERATIVE_ROLLOUT_CHUNK", "4096"))

    lanes = []
    for session_index, animal_row in enumerate(animal_rows, start=1):
        model_ses_idx = str(animal_row.get("ses_idx"))
        source_ses_idx = _coalesce_session_identifier(
            animal_row.get("source_ses_idx"),
            fallback=model_ses_idx,
        )
        n_trials = int(animal_row.get("n_trials", 0))
        if n_trials <= 0:
            logger.info(
                "Skipping session %d/%d: subject=%s ses_idx=%s has n_trials=%s",
                session_index,
                total_source_sessions,
                _normalize_identifier(animal_row.get("subject_id")),
                model_ses_idx,
                animal_row.get("n_trials"),
            )
            continue
        for rollout_index in range(int(n_rollouts_per_session)):
            lanes.append(
                {
                    "subject_id": _normalize_identifier(animal_row.get("subject_id")),
                    "model_ses_idx": model_ses_idx,
                    "source_ses_idx": source_ses_idx,
                    "session_date": str(animal_row.get("session_date")),
                    "curriculum_name": animal_row.get("curriculum_name"),
                    "current_stage_actual": animal_row.get("current_stage_actual"),
                    "nwb_suffix": animal_row.get("nwb_suffix"),
                    "nwb_name": animal_row.get("nwb_name"),
                    "n_trials": n_trials,
                    "rollout_index": rollout_index,
                    "seed": _derive_session_seed(
                        run.seed, source_ses_idx, rollout_index=rollout_index
                    ),
                }
            )

    records = []
    completed_rollouts = 0
    for chunk_start in range(0, len(lanes), BATCHED_ROLLOUT_CHUNK):
        chunk = lanes[chunk_start : chunk_start + BATCHED_ROLLOUT_CHUNK]
        max_trials = max(lane["n_trials"] for lane in chunk)
        logger.info(
            "Simulating lanes %d-%d/%d (batched rollout): max_trials=%d",
            chunk_start + 1,
            chunk_start + len(chunk),
            len(lanes),
            max_trials,
        )
        choice_histories, reward_histories = _run_batched_rollout(
            runner, chunk, n_actions=runner.n_actions
        )
        for lane_index, lane in enumerate(chunk):
            if int(n_rollouts_per_session) == 1:
                simulated_ses_idx = lane["source_ses_idx"]
            else:
                simulated_ses_idx = (
                    f"{lane['source_ses_idx']}__rollout_{lane['rollout_index']}"
                )
            records.append(
                {
                    "subject_id": lane["subject_id"],
                    "ses_idx": simulated_ses_idx,
                    "source_ses_idx": lane["source_ses_idx"],
                    "session_date": lane["session_date"],
                    "curriculum_name": lane["curriculum_name"],
                    "current_stage_actual": lane["current_stage_actual"],
                    "n_trials": lane["n_trials"],
                    "choice_history": choice_histories[lane_index],
                    "reward_history": reward_histories[lane_index],
                    "nwb_suffix": lane["nwb_suffix"],
                    "nwb_name": lane["nwb_name"],
                    "random_seed": int(lane["seed"]),
                    "model_dir": run.model_dir,
                    "checkpoint_step": run.checkpoint_step,
                    "rollout_mode": normalized_rollout_mode,
                }
            )
            completed_rollouts += 1
        logger.info(
            "Completed %d/%d rollouts (batched).",
            completed_rollouts,
            total_requested_rollouts,
        )

    logger.info(
        "Finished model simulation: generated %d simulated session%s from %d source session%s.",
        len(records),
        "" if len(records) == 1 else "s",
        total_source_sessions,
        "" if total_source_sessions == 1 else "s",
    )
    return pd.DataFrame.from_records(records)


def _simulate_baseline_model_sessions(
    *,
    resolved_run: ResolvedModelRun,
    animal_sessions,
    rollout_mode: str,
    n_rollouts_per_session: int,
):
    pd = _import_dependency("pandas")
    generative_model_mod = _import_dependency(
        "aind_dynamic_foraging_models.generative_model"
    )
    baseline_output = _load_baseline_output_from_resolved_run(resolved_run)
    agent_class_name, agent_kwargs = _resolve_baseline_agent_spec(
        resolved_run,
        baseline_output,
    )
    agent_class_obj = getattr(generative_model_mod, agent_class_name, None)
    if agent_class_obj is None:
        raise ValueError(
            f"Agent class {agent_class_name!r} was not found in generative_model."
        )

    animal_rows = list(_iter_session_records(animal_sessions))
    total_source_sessions = len(animal_rows)
    total_requested_rollouts = total_source_sessions * int(n_rollouts_per_session)
    logger.info(
        "Starting baseline RL simulation for %d source sessions (%d rollout%s) from %s "
        "[checkpoint=%s, mode=%s]",
        total_source_sessions,
        total_requested_rollouts,
        "" if total_requested_rollouts == 1 else "s",
        resolved_run.model_dir,
        resolved_run.checkpoint_label,
        rollout_mode,
    )

    records = []
    completed_rollouts = 0
    for session_index, animal_row in enumerate(animal_rows, start=1):
        model_ses_idx = str(animal_row.get("ses_idx"))
        source_ses_idx = _coalesce_session_identifier(
            animal_row.get("source_ses_idx"),
            fallback=model_ses_idx,
        )
        subject_id = _normalize_identifier(animal_row.get("subject_id"))
        session_date = str(animal_row.get("session_date"))
        curriculum_name = animal_row.get("curriculum_name")
        current_stage_actual = animal_row.get("current_stage_actual")
        nwb_suffix = animal_row.get("nwb_suffix")
        nwb_name = animal_row.get("nwb_name")
        n_trials = int(animal_row.get("n_trials", 0))
        if n_trials <= 0:
            logger.info(
                "Skipping baseline RL session %d/%d: subject=%s ses_idx=%s has n_trials=%s",
                session_index,
                total_source_sessions,
                subject_id,
                model_ses_idx,
                animal_row.get("n_trials"),
            )
            continue

        fitted_params = _resolve_baseline_fitted_params_for_subject(
            resolved_run,
            baseline_output,
            subject_id=subject_id,
        )
        logger.info(
            "Simulating baseline RL session %d/%d: subject=%s ses_idx=%s source_ses_idx=%s curriculum=%s n_trials=%d "
            "(%d rollout%s)",
            session_index,
            total_source_sessions,
            subject_id,
            model_ses_idx,
            source_ses_idx,
            curriculum_name,
            n_trials,
            int(n_rollouts_per_session),
            "" if int(n_rollouts_per_session) == 1 else "s",
        )
        for rollout_index in range(int(n_rollouts_per_session)):
            random_seed = _derive_session_seed(
                resolved_run.seed,
                source_ses_idx,
                rollout_index=rollout_index,
            )
            task = _build_curriculum_matched_task(
                curriculum_name=curriculum_name,
                n_trials=n_trials,
                seed=random_seed,
            )
            if hasattr(task, "reset"):
                task.reset()

            agent = agent_class_obj(
                **dict(agent_kwargs),
                seed=int(random_seed),
            )
            agent.set_params(**fitted_params)
            agent.perform(task)
            choice_history = _extract_baseline_history_from_agent(
                agent,
                getter_name="get_choice_history",
                expected_length=n_trials,
                label="choice history",
            )
            reward_history = _extract_baseline_history_from_agent(
                agent,
                getter_name="get_reward_history",
                expected_length=n_trials,
                label="reward history",
            )

            if int(n_rollouts_per_session) == 1:
                simulated_ses_idx = source_ses_idx
            else:
                simulated_ses_idx = f"{source_ses_idx}__rollout_{rollout_index}"

            records.append(
                {
                    "subject_id": subject_id,
                    "ses_idx": simulated_ses_idx,
                    "source_ses_idx": source_ses_idx,
                    "session_date": session_date,
                    "curriculum_name": curriculum_name,
                    "current_stage_actual": current_stage_actual,
                    "n_trials": n_trials,
                    "choice_history": choice_history,
                    "reward_history": reward_history,
                    "nwb_suffix": nwb_suffix,
                    "nwb_name": nwb_name,
                    "random_seed": int(random_seed),
                    "model_dir": resolved_run.model_dir,
                    "checkpoint_step": resolved_run.checkpoint_step,
                    "rollout_mode": rollout_mode,
                }
            )
            completed_rollouts += 1
            logger.info(
                "Completed baseline RL rollout %d/%d for ses_idx=%s seed=%d",
                completed_rollouts,
                total_requested_rollouts,
                simulated_ses_idx,
                int(random_seed),
            )

    logger.info(
        "Finished baseline RL simulation: generated %d simulated session%s from %d source session%s.",
        len(records),
        "" if len(records) == 1 else "s",
        total_source_sessions,
        "" if total_source_sessions == 1 else "s",
    )
    return pd.DataFrame.from_records(records)


def compute_switch_stats(
    animal_sessions,
    simulated_sessions,
    window_size: int = _DEFAULT_WINDOW_SIZE,
) -> dict[str, Any]:
    """Compute switch-based statistics for real and simulated sessions."""

    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    animal_with_switches = insert_switch_info(animal_sessions)
    simulated_with_switches = insert_switch_info(simulated_sessions)

    animal_summary = _compute_switch_summary(animal_with_switches, window_size)
    simulated_summary = _compute_switch_summary(simulated_with_switches, window_size)
    subject_level = _compute_subject_level_comparison(
        animal_with_switches,
        simulated_with_switches,
    )
    session_level = _compute_session_level_comparison(
        animal_with_switches,
        simulated_with_switches,
    )
    subject_aggregate = _build_switch_subject_aggregate(subject_level)
    session_aggregate = _build_switch_session_aggregate(session_level)
    delta_significance_summary = _build_switch_delta_significance_summary(subject_level)
    quantitative_summary = _build_switch_quantitative_summary(
        animal_summary=animal_summary,
        simulated_summary=simulated_summary,
        subject_aggregate=subject_aggregate,
        session_aggregate=session_aggregate,
    )

    return {
        "window_size": window_size,
        "animal": animal_summary,
        "simulated": simulated_summary,
        "comparison": _compare_switch_summaries(animal_summary, simulated_summary),
        "subject_level": subject_level,
        "subject_aggregate": subject_aggregate,
        "session_level": session_level,
        "session_aggregate": session_aggregate,
        "delta_significance_summary": delta_significance_summary,
        "quantitative_summary": quantitative_summary,
    }


def compute_history_dependent_switch_stats(
    animal_sessions,
    simulated_sessions,
    *,
    max_trials_back: int = _DEFAULT_HISTORY_MAX_TRIALS_BACK,
    aggregate_min_trials: int = _DEFAULT_HISTORY_AGGREGATE_MIN_TRIALS,
    subject_min_trials: int = _DEFAULT_HISTORY_SUBJECT_MIN_TRIALS,
    default_pattern_type: str = "abstract",
) -> dict[str, Any]:
    """Compute history-dependent switch probabilities for animal and simulated sessions."""

    max_trials_back = int(max_trials_back)
    aggregate_min_trials = int(aggregate_min_trials)
    subject_min_trials = int(subject_min_trials)
    if max_trials_back <= 0:
        raise ValueError("max_trials_back must be a positive integer.")
    if aggregate_min_trials < 0:
        raise ValueError("aggregate_min_trials must be >= 0.")
    if subject_min_trials < 0:
        raise ValueError("subject_min_trials must be >= 0.")

    normalized_pattern_type = _normalize_history_pattern_type(default_pattern_type)
    subject_curriculum_map = _resolve_subject_curriculum_map(
        animal_sessions,
        fallback_rows=simulated_sessions,
    )
    animal_records = _build_history_pattern_count_records(
        animal_sessions,
        max_trials_back=max_trials_back,
    )
    simulated_records = _build_history_pattern_count_records(
        simulated_sessions,
        max_trials_back=max_trials_back,
    )

    animal_summary = _finalize_history_pattern_count_tree(
        _aggregate_history_pattern_count_records(
            animal_records,
            max_trials_back=max_trials_back,
            average_rollouts_by_source=False,
        ),
    )
    simulated_summary = _finalize_history_pattern_count_tree(
        _aggregate_history_pattern_count_records(
            simulated_records,
            max_trials_back=max_trials_back,
            average_rollouts_by_source=True,
        ),
    )
    animal_subject_stats = _build_subject_history_pattern_summary(
        animal_records,
        max_trials_back=max_trials_back,
        average_rollouts_by_source=False,
        subject_curriculum_map=subject_curriculum_map,
    )
    simulated_subject_stats = _build_subject_history_pattern_summary(
        simulated_records,
        max_trials_back=max_trials_back,
        average_rollouts_by_source=True,
        subject_curriculum_map=subject_curriculum_map,
    )
    comparison = _build_history_pattern_comparison(
        animal_summary,
        simulated_summary,
        max_trials_back=max_trials_back,
        aggregate_min_trials=aggregate_min_trials,
    )
    subject_level = _build_subject_history_pattern_comparison(
        animal_subject_stats,
        simulated_subject_stats,
        max_trials_back=max_trials_back,
        subject_min_trials=subject_min_trials,
    )
    animal_session_stats = _build_session_history_pattern_summary(
        animal_records,
        max_trials_back=max_trials_back,
        average_rollouts_by_source=False,
    )
    simulated_session_stats = _build_session_history_pattern_summary(
        simulated_records,
        max_trials_back=max_trials_back,
        average_rollouts_by_source=True,
    )
    session_level = _build_session_history_pattern_comparison(
        animal_session_stats,
        simulated_session_stats,
        max_trials_back=max_trials_back,
        session_min_trials=subject_min_trials,
    )
    subject_aggregate = _build_history_subject_aggregate(
        subject_level,
        max_trials_back=max_trials_back,
        subject_min_trials=subject_min_trials,
    )
    session_aggregate = _build_history_session_aggregate(
        session_level,
        max_trials_back=max_trials_back,
        session_min_trials=subject_min_trials,
    )
    delta_significance_summary = _build_history_delta_significance_summary(
        subject_level,
        max_trials_back=max_trials_back,
        subject_min_trials=subject_min_trials,
    )
    quantitative_summary = _build_history_quantitative_summary(
        comparison=comparison,
        subject_aggregate=subject_aggregate,
        session_aggregate=session_aggregate,
        max_trials_back=max_trials_back,
    )

    return {
        "config": {
            "max_trials_back": max_trials_back,
            "aggregate_min_trials": aggregate_min_trials,
            "subject_min_trials": subject_min_trials,
            "default_pattern_type": normalized_pattern_type,
        },
        "animal": animal_summary,
        "simulated": simulated_summary,
        "comparison": comparison,
        "subject_level": subject_level,
        "subject_aggregate": subject_aggregate,
        "session_level": session_level,
        "session_aggregate": session_aggregate,
        "delta_significance_summary": delta_significance_summary,
        "quantitative_summary": quantitative_summary,
    }


def run_post_training_analysis(
    model_dir: str | Path,
    *,
    split: str = _TRAIN_SPLIT,
    checkpoint_policy: str = _CHECKPOINT_POLICY_BEST_EVAL,
    rollout_mode: str = _ROLLOUT_MODE_CURRICULUM_MATCHED,
    n_rollouts_per_session: int = 1,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    output_dir: str | Path | None = None,
    save_animal_session_history: bool = False,
    session_partitions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the end-to-end standalone post-training generative analysis.

    Parameters
    ----------
    output_dir:
        Optional override for where analysis outputs are written. If omitted,
        outputs are saved under ``<model_dir>/outputs/post_training_analysis/``
        with a run-specific subdirectory.
    save_animal_session_history:
        If ``True``, also save ``animal_session_history.pkl``. This defaults to
        ``False`` because the animal history is reproducible from the resolved
        run configuration and snapshot data.
    """

    started_at = time.perf_counter()
    logger.info(
        "Starting post-training analysis: model_dir=%s split=%s checkpoint_policy=%s "
        "rollout_mode=%s n_rollouts_per_session=%d window_size=%d "
        "save_animal_session_history=%s output_dir=%s",
        model_dir,
        split,
        checkpoint_policy,
        rollout_mode,
        int(n_rollouts_per_session),
        int(window_size),
        save_animal_session_history,
        output_dir,
    )

    stage_started_at = time.perf_counter()
    resolved_run = resolve_model_run(
        model_dir=model_dir,
        split=split,
        checkpoint_policy=checkpoint_policy,
    )
    logger.info(
        "Completed resolve_model_run in %.2fs",
        time.perf_counter() - stage_started_at,
    )

    stage_started_at = time.perf_counter()
    animal_sessions = load_animal_session_history(resolved_run, split=split)
    logger.info(
        "Completed load_animal_session_history in %.2fs",
        time.perf_counter() - stage_started_at,
    )

    stage_started_at = time.perf_counter()
    simulated_sessions = simulate_model_sessions(
        resolved_run=resolved_run,
        animal_sessions=animal_sessions,
        rollout_mode=rollout_mode,
        n_rollouts_per_session=n_rollouts_per_session,
    )
    logger.info(
        "Completed simulate_model_sessions in %.2fs",
        time.perf_counter() - stage_started_at,
    )

    normalized_session_partitions = _normalize_session_partitions(session_partitions)
    if resolved_run.split == _TRAIN_SPLIT:
        requires_partition_manifest = (
            len(normalized_session_partitions) > 1
            or normalized_session_partitions[0] != _SESSION_PARTITION_COMBINED
        )
        stage_started_at = time.perf_counter()
        try:
            resolved_run = _ensure_session_split_manifest(resolved_run)
            logger.info(
                "Completed session split manifest reconstruction in %.2fs",
                time.perf_counter() - stage_started_at,
            )
        except Exception as exc:
            if requires_partition_manifest:
                raise
            logger.warning(
                "Skipping session split manifest persistence for %s because reconstruction "
                "failed in the current environment: %s",
                resolved_run.model_dir,
                exc,
            )

    result = run_post_training_analysis_from_histories(
        animal_sessions=animal_sessions,
        simulated_sessions=simulated_sessions,
        output_dir=output_dir,
        resolved_run=resolved_run,
        window_size=window_size,
        save_animal_session_history=save_animal_session_history,
        session_partitions=normalized_session_partitions,
    )
    logger.info(
        "Finished post-training analysis in %.2fs",
        time.perf_counter() - started_at,
    )
    return result


def run_post_training_analysis_from_histories(
    animal_sessions,
    simulated_sessions,
    *,
    output_dir: str | Path | None = None,
    resolved_run: ResolvedModelRun | Mapping[str, Any] | None = None,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    save_animal_session_history: bool = False,
    session_partitions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute and save post-training analysis outputs from in-memory histories."""

    resolved_run_obj = (
        None if resolved_run is None else _coerce_resolved_run(resolved_run)
    )
    normalized_session_partitions = _normalize_session_partitions(session_partitions)
    if output_dir is None:
        if resolved_run_obj is None:
            raise ValueError(
                "output_dir is required when resolved_run is not provided."
            )
        resolved_output_dir = _resolve_analysis_output_dir(
            resolved_run=resolved_run_obj,
            output_dir=None,
        )
    else:
        resolved_output_dir = Path(output_dir).expanduser().resolve()

    if (
        len(normalized_session_partitions) > 1
        or normalized_session_partitions[0] != _SESSION_PARTITION_COMBINED
    ):
        if resolved_run_obj is None:
            raise ValueError(
                "resolved_run is required when requesting session-partitioned analysis."
            )
        resolved_run_obj = _ensure_session_split_manifest(resolved_run_obj)
        return _compute_and_save_partitioned_post_training_outputs(
            animal_sessions=animal_sessions,
            simulated_sessions=simulated_sessions,
            output_dir=resolved_output_dir,
            resolved_run=resolved_run_obj,
            window_size=window_size,
            save_animal_session_history=save_animal_session_history,
            session_partitions=normalized_session_partitions,
        )

    return _compute_and_save_post_training_outputs(
        animal_sessions=animal_sessions,
        simulated_sessions=simulated_sessions,
        output_dir=resolved_output_dir,
        resolved_run=resolved_run_obj,
        window_size=window_size,
        save_animal_session_history=save_animal_session_history,
    )


def run_post_training_analysis_from_saved_histories(
    simulated_session_history_path: str | Path,
    *,
    animal_session_history_path: str | Path | None = None,
    resolved_run_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    save_animal_session_history: bool = False,
    session_partitions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Recompute analysis outputs from saved simulated session histories."""

    simulated_history_path = Path(simulated_session_history_path).expanduser().resolve()
    with simulated_history_path.open("rb") as f:
        simulated_sessions = pickle.load(f)

    resolved_run_obj = None
    if resolved_run_path is not None:
        resolved_run_payload = _load_structured_file(
            Path(resolved_run_path).expanduser().resolve()
        )
        if isinstance(resolved_run_payload, ResolvedModelRun):
            resolved_run_obj = resolved_run_payload
        elif isinstance(resolved_run_payload, Mapping):
            resolved_run_obj = ResolvedModelRun(**dict(resolved_run_payload))
        else:
            raise ValueError(
                f"Expected mapping-style resolved run payload in {resolved_run_path!s}."
            )

    if animal_session_history_path is not None:
        animal_history_path = Path(animal_session_history_path).expanduser().resolve()
        with animal_history_path.open("rb") as f:
            animal_sessions = pickle.load(f)
    else:
        if resolved_run_obj is None:
            raise ValueError(
                "Either animal_session_history_path or resolved_run_path must be provided."
            )
        animal_sessions = load_animal_session_history(
            resolved_run_obj,
            split=resolved_run_obj.split,
        )

    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else simulated_history_path.parent / "reanalysis"
    )
    return run_post_training_analysis_from_histories(
        animal_sessions=animal_sessions,
        simulated_sessions=simulated_sessions,
        output_dir=resolved_output_dir,
        resolved_run=resolved_run_obj,
        window_size=window_size,
        save_animal_session_history=save_animal_session_history,
        session_partitions=session_partitions,
    )


def _normalize_session_partition_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"train", "training"}:
        return _SESSION_PARTITION_TRAIN
    if normalized == "eval":
        return _SESSION_PARTITION_EVAL
    if normalized in {"combined", "all", "together"}:
        return _SESSION_PARTITION_COMBINED
    raise ValueError(
        f"Unsupported session partition {value!r}. Use 'train', 'eval', or 'combined'."
    )


def _normalize_session_partitions(
    session_partitions: Sequence[str] | None,
) -> tuple[str, ...]:
    if session_partitions is None:
        return (_SESSION_PARTITION_COMBINED,)
    if isinstance(session_partitions, str):
        requested = [session_partitions]
    else:
        requested = list(session_partitions)
    if not requested:
        raise ValueError("session_partitions must contain at least one partition.")

    normalized_partitions: list[str] = []
    seen: set[str] = set()
    for partition in requested:
        normalized = _normalize_session_partition_name(partition)
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_partitions.append(normalized)
    return tuple(normalized_partitions)


def _normalize_session_split_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    per_subject_rows = []
    for row in list(manifest.get("per_subject") or []):
        if not isinstance(row, Mapping):
            continue
        per_subject_rows.append(
            {
                "subject_id": _normalize_identifier(row.get("subject_id")),
                "full_session_ids": [str(session_id) for session_id in row.get("full_session_ids") or []],
                "train_session_ids": [
                    str(session_id) for session_id in row.get("train_session_ids") or []
                ],
                "eval_session_ids": [
                    str(session_id) for session_id in row.get("eval_session_ids") or []
                ],
            }
        )

    return {
        "source": str(manifest.get("source", "unknown")),
        "multisubject": bool(manifest.get("multisubject", False)),
        "eval_every_n": int(manifest.get("eval_every_n", 2)),
        "selected_subject_ids": [
            _normalize_identifier(subject_id)
            for subject_id in manifest.get("selected_subject_ids") or []
        ],
        "full_session_ids": [
            str(session_id) for session_id in manifest.get("full_session_ids") or []
        ],
        "train_session_ids": [
            str(session_id) for session_id in manifest.get("train_session_ids") or []
        ],
        "eval_session_ids": [
            str(session_id) for session_id in manifest.get("eval_session_ids") or []
        ],
        "per_subject": per_subject_rows,
    }


def _ensure_session_split_manifest(
    resolved_run: ResolvedModelRun,
) -> ResolvedModelRun:
    if resolved_run.session_split_manifest is not None:
        resolved_run.session_split_manifest = _normalize_session_split_manifest(
            resolved_run.session_split_manifest
        )
        return resolved_run

    if resolved_run.split != _TRAIN_SPLIT:
        raise NotImplementedError(
            "Session-partitioned post-training analysis is only supported for "
            "resolved_run.split='train'."
        )

    if resolved_run.model_type == "baseline_rl":
        split_manifest = _resolve_baseline_session_split_manifest(resolved_run)
        if split_manifest is not None:
            resolved_run.session_split_manifest = _normalize_session_split_manifest(
                split_manifest
            )
            return resolved_run

    data_cfg = _as_dict(resolved_run.run_config.get("data", {}))
    data_type = str(data_cfg.get("type", "mice_snapshot")).strip().lower()
    data_target = str(data_cfg.get("_target_", "")).strip().lower()
    if data_type and data_type != "mice_snapshot" and "micesnapshotdatasetloader" not in data_target:
        raise ValueError(
            "Session split reconstruction currently supports mice_snapshot-backed "
            f"runs only. Observed data.type={data_type!r}."
        )

    eval_every_n = int(data_cfg.get("eval_every_n", 2))
    selection_subject_ids = (
        list(resolved_run.trained_subject_ids)
        if resolved_run.multisubject and resolved_run.trained_subject_ids is not None
        else resolved_run.selection.get("subject_ids")
    )
    mice_loader_mod = importlib.import_module("data_loaders.mice")
    split_manifest = mice_loader_mod.resolve_mice_snapshot_session_split_manifest(
        split=resolved_run.split,
        subject_ids=selection_subject_ids,
        curricula=list(resolved_run.curricula or []),
        subject_ratio=resolved_run.selection.get("subject_ratio"),
        min_sessions=int(resolved_run.selection.get("min_sessions", 10)),
        heldout_every_n=int(resolved_run.selection.get("heldout_every_n", 5)),
        subject_sample_seed=resolved_run.selection.get("subject_sample_seed"),
        ignore_policy=resolved_run.ignore_policy,
        eval_every_n=eval_every_n,
        multisubject=bool(resolved_run.multisubject),
        mature_only=bool(resolved_run.mature_only),
        cols_to_retain=[
            "trial",
            "subject_id",
            "ses_idx",
            "animal_response",
            "earned_reward",
            "curriculum_name",
        ],
        snapshot=resolved_run.selection.get("snapshot"),
    )
    resolved_run.session_split_manifest = _normalize_session_split_manifest(split_manifest)
    return resolved_run


def _resolve_baseline_session_split_manifest(
    resolved_run: ResolvedModelRun,
) -> dict[str, Any] | None:
    baseline_output = _load_baseline_output_from_resolved_run(resolved_run)
    if not _baseline_output_is_multisubject_per_subject(baseline_output):
        return None

    fitted_params_per_subject = baseline_output.get("fitted_params_per_subject")
    if not isinstance(fitted_params_per_subject, Mapping) or not fitted_params_per_subject:
        raise ValueError("baseline_rl_output.json is missing fitted_params_per_subject.")

    data_cfg = _as_dict(resolved_run.run_config.get("data", {}))
    eval_every_n = int(data_cfg.get("eval_every_n", 2))
    selected_subject_ids: list[Any] = []
    full_session_ids: list[str] = []
    train_session_ids: list[str] = []
    eval_session_ids: list[str] = []
    per_subject_rows: list[dict[str, Any]] = []

    for default_subject_id, raw_subject_summary in fitted_params_per_subject.items():
        subject_summary = _as_dict(raw_subject_summary)
        subject_id = _normalize_identifier(
            subject_summary.get("subject_id", default_subject_id)
        )
        subject_train_session_ids = [
            str(session_id)
            for session_id in subject_summary.get("train_session_ids") or []
        ]
        subject_eval_session_ids = [
            str(session_id)
            for session_id in subject_summary.get("eval_session_ids") or []
        ]
        if not subject_train_session_ids and not subject_eval_session_ids:
            continue

        subject_full_session_ids = [
            str(session_id)
            for session_id in _unique_preserve_order(
                [*subject_train_session_ids, *subject_eval_session_ids]
            )
        ]
        selected_subject_ids.append(subject_id)
        full_session_ids.extend(subject_full_session_ids)
        train_session_ids.extend(subject_train_session_ids)
        eval_session_ids.extend(subject_eval_session_ids)
        per_subject_rows.append(
            {
                "subject_id": subject_id,
                "full_session_ids": subject_full_session_ids,
                "train_session_ids": subject_train_session_ids,
                "eval_session_ids": subject_eval_session_ids,
            }
        )

    if not per_subject_rows:
        return None

    return {
        "source": "baseline_rl_output",
        "multisubject": bool(resolved_run.multisubject),
        "eval_every_n": eval_every_n,
        "selected_subject_ids": _unique_preserve_order(selected_subject_ids),
        "full_session_ids": _unique_preserve_order(full_session_ids),
        "train_session_ids": _unique_preserve_order(train_session_ids),
        "eval_session_ids": _unique_preserve_order(eval_session_ids),
        "per_subject": per_subject_rows,
    }


def _filter_session_rows_by_session_ids(
    session_rows,
    *,
    allowed_session_ids: Sequence[str],
    prefer_source_session_ids: bool,
):
    allowed_session_id_set = {str(session_id) for session_id in allowed_session_ids}
    filtered_records = []
    for row in _iter_session_records(session_rows):
        session_id = None
        if prefer_source_session_ids:
            session_id = row.get("source_ses_idx")
        if _is_missing_session_identifier(session_id):
            session_id = row.get("ses_idx")
        if str(session_id) not in allowed_session_id_set:
            continue
        filtered_records.append(dict(row))
    return _restore_input_container(session_rows, filtered_records)


def _count_session_rows(session_rows) -> int:
    return sum(1 for _ in _iter_session_records(session_rows))


def _parallel_partition_worker(task):
    """Spawn-pool worker: run the unchanged per-partition pipeline for one
    partition. Returns ``(partition, result)`` — bit-identical to serial."""
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless figure rendering in workers
    partition, partition_animal_sessions, partition_simulated_sessions, output_dir, common = task
    result = _compute_and_save_post_training_outputs(
        animal_sessions=partition_animal_sessions,
        simulated_sessions=partition_simulated_sessions,
        output_dir=output_dir,
        **common,
    )
    return partition, result


def _compute_and_save_partitioned_post_training_outputs(
    *,
    animal_sessions,
    simulated_sessions,
    output_dir: str | Path,
    resolved_run: ResolvedModelRun,
    window_size: int,
    save_animal_session_history: bool,
    session_partitions: Sequence[str],
) -> dict[str, Any]:
    resolved_run = _ensure_session_split_manifest(resolved_run)
    manifest = _normalize_session_split_manifest(resolved_run.session_split_manifest or {})
    analysis_output_dir = Path(output_dir).expanduser().resolve()
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_path = analysis_output_dir / "resolved_run.json"
    resolved_run_path.write_text(json.dumps(resolved_run.to_dict(), indent=2))

    # Slice each partition's sessions up front (cheap; per-session frames). The
    # split manifest is in the SOURCE session namespace (e.g.
    # "632105_2022-07-16_162930"); multisubject animal rows are aligned to the
    # MERGED namespace ("632105__632105_..."), so match animal on source_ses_idx
    # (like simulated) — matching on ses_idx silently dropped all train/eval
    # animal sessions.
    partition_slices: dict[str, tuple[Any, Any]] = {}
    for partition in session_partitions:
        if partition == _SESSION_PARTITION_COMBINED:
            partition_slices[partition] = (animal_sessions, simulated_sessions)
        else:
            allowed_session_ids = manifest[f"{partition}_session_ids"]
            partition_slices[partition] = (
                _filter_session_rows_by_session_ids(
                    animal_sessions,
                    allowed_session_ids=allowed_session_ids,
                    prefer_source_session_ids=True,
                ),
                _filter_session_rows_by_session_ids(
                    simulated_sessions,
                    allowed_session_ids=allowed_session_ids,
                    prefer_source_session_ids=True,
                ),
            )

    # Partitions are independent whole computations, so compute them
    # concurrently — each worker runs the unchanged per-partition pipeline, so
    # outputs are bit-identical to serial; only scheduling is parallel. Use
    # 'spawn' (NOT fork): this runs right after the JAX/GPU rollout and forking a
    # CUDA-initialised process can hang. BFM_GENERATIVE_STATS_WORKERS=1 forces
    # serial (used by the in-process-mock unit tests); the launcher sets it >1.
    common = dict(
        resolved_run=resolved_run,
        window_size=window_size,
        save_animal_session_history=save_animal_session_history,
    )
    max_workers = int(get_env("BFM_GENERATIVE_STATS_WORKERS", "1"))
    partition_results: dict[str, Any] = {}
    if max_workers > 1 and len(session_partitions) > 1:
        import concurrent.futures
        import multiprocessing

        tasks = [
            (
                partition,
                partition_slices[partition][0],
                partition_slices[partition][1],
                str(analysis_output_dir / partition),
                common,
            )
            for partition in session_partitions
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(max_workers, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for partition, partition_result in executor.map(
                _parallel_partition_worker, tasks
            ):
                partition_results[partition] = partition_result
    else:
        for partition in session_partitions:
            partition_animal_sessions, partition_simulated_sessions = partition_slices[partition]
            partition_results[partition] = _compute_and_save_post_training_outputs(
                animal_sessions=partition_animal_sessions,
                simulated_sessions=partition_simulated_sessions,
                output_dir=analysis_output_dir / partition,
                **common,
            )

    partition_summary: dict[str, Any] = {}
    for partition in session_partitions:
        partition_result = partition_results[partition]
        partition_result["session_partition"] = partition
        partition_animal_sessions, partition_simulated_sessions = partition_slices[partition]
        partition_summary[partition] = {
            "output_dir": partition_result["output_dir"],
            "num_animal_sessions": _count_session_rows(partition_animal_sessions),
            "num_simulated_sessions": _count_session_rows(partition_simulated_sessions),
        }

    summary_path = analysis_output_dir / "session_partition_summary.json"
    summary_payload = {
        "output_dir": str(analysis_output_dir),
        "resolved_run": str(resolved_run_path),
        "session_partitions": list(session_partitions),
        "partition_summary": partition_summary,
    }
    summary_path.write_text(json.dumps(_to_serializable(summary_payload), indent=2))

    return {
        "output_dir": str(analysis_output_dir),
        "resolved_run": str(resolved_run_path),
        "session_partition_summary": str(summary_path),
        "partition_results": partition_results,
    }


def _compute_and_save_post_training_outputs(
    *,
    animal_sessions,
    simulated_sessions,
    output_dir: str | Path,
    resolved_run: ResolvedModelRun | None,
    window_size: int,
    save_animal_session_history: bool,
) -> dict[str, Any]:
    stage_started_at = time.perf_counter()
    switch_stats = compute_switch_stats(
        animal_sessions=animal_sessions,
        simulated_sessions=simulated_sessions,
        window_size=window_size,
    )
    logger.info(
        "Completed compute_switch_stats in %.2fs",
        time.perf_counter() - stage_started_at,
    )

    stage_started_at = time.perf_counter()
    history_dependent_switch_stats = compute_history_dependent_switch_stats(
        animal_sessions=animal_sessions,
        simulated_sessions=simulated_sessions,
    )
    logger.info(
        "Completed compute_history_dependent_switch_stats in %.2fs",
        time.perf_counter() - stage_started_at,
    )

    stage_started_at = time.perf_counter()
    analysis_output_dir = Path(output_dir).expanduser().resolve()
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_path = analysis_output_dir / "resolved_run.json"
    animal_history_path = analysis_output_dir / "animal_session_history.pkl"
    simulated_history_path = analysis_output_dir / "simulated_session_history.pkl"
    switch_stats_path = analysis_output_dir / "switch_stats.json"
    history_dependent_switch_stats_path = (
        analysis_output_dir / "history_dependent_switch_stats.json"
    )
    quantitative_summary_path = (
        analysis_output_dir / "model_vs_animal_quantitative_summary.json"
    )

    if resolved_run is not None:
        resolved_run_path.write_text(json.dumps(resolved_run.to_dict(), indent=2))
    if save_animal_session_history:
        with animal_history_path.open("wb") as f:
            pickle.dump(animal_sessions, f)
    with simulated_history_path.open("wb") as f:
        pickle.dump(simulated_sessions, f)

    figure_paths = _save_switch_figures(
        switch_stats=switch_stats,
        output_dir=analysis_output_dir / "figures",
    )
    switch_stats_with_figures = dict(switch_stats)
    switch_stats_with_figures["figure_paths"] = {
        name: str(path) for name, path in figure_paths.items()
    }
    switch_stats_path.write_text(
        json.dumps(_to_serializable(switch_stats_with_figures), indent=2)
    )

    history_figure_paths = _save_history_dependent_switch_figures(
        history_stats=history_dependent_switch_stats,
        output_dir=analysis_output_dir / "figures",
    )
    history_stats_with_figures = dict(history_dependent_switch_stats)
    history_stats_with_figures["figure_paths"] = {
        name: str(path) for name, path in history_figure_paths.items()
    }
    history_dependent_switch_stats_path.write_text(
        json.dumps(_to_serializable(history_stats_with_figures), indent=2)
    )

    combined_quantitative_summary = {
        "switch_triggered": {
            "quantitative_summary": _as_dict(switch_stats.get("quantitative_summary", {})),
            "delta_significance_summary": _as_dict(
                switch_stats.get("delta_significance_summary", {})
            ),
        },
        "history_dependent": {
            "quantitative_summary": _as_dict(
                history_dependent_switch_stats.get("quantitative_summary", {})
            ),
            "delta_significance_summary": _as_dict(
                history_dependent_switch_stats.get("delta_significance_summary", {})
            ),
        },
    }
    quantitative_summary_path.write_text(
        json.dumps(_to_serializable(combined_quantitative_summary), indent=2)
    )

    logger.info(
        "Saved analysis outputs in %.2fs to %s",
        time.perf_counter() - stage_started_at,
        analysis_output_dir,
    )

    result = {
        "simulated_session_history": str(simulated_history_path),
        "switch_stats": str(switch_stats_path),
        "history_dependent_switch_stats": str(history_dependent_switch_stats_path),
        "model_vs_animal_quantitative_summary": str(quantitative_summary_path),
        "figure_paths": {
            name: str(path)
            for name, path in {**figure_paths, **history_figure_paths}.items()
        },
        "output_dir": str(analysis_output_dir),
    }
    if resolved_run is not None:
        result["resolved_run"] = str(resolved_run_path)
    if save_animal_session_history:
        result["animal_session_history"] = str(animal_history_path)
    return result


def extract_switches_and_run_lengths(choices: Sequence[Any]) -> tuple[list[int], list[int]]:
    """Return switch indices and their preceding run lengths."""

    normalized_choices = _clean_choice_history(choices)
    if len(normalized_choices) < 2:
        return [], []

    switch_indices: list[int] = []
    for idx in range(1, len(normalized_choices)):
        if normalized_choices[idx] != normalized_choices[idx - 1]:
            switch_indices.append(idx)

    if not switch_indices:
        return [], []

    previous_switch = 0
    run_lengths = []
    for switch_idx in switch_indices:
        run_lengths.append(switch_idx - previous_switch)
        previous_switch = switch_idx
    return switch_indices, run_lengths


def insert_switch_info(session_rows):
    """Insert switch indices and run lengths into session-history rows."""

    updated_records = []
    for row in _iter_session_records(session_rows):
        updated = dict(row)
        switch_indices, run_lengths = extract_switches_and_run_lengths(
            row.get("choice_history", [])
        )
        updated["switch_indices"] = switch_indices
        updated["run_lengths"] = run_lengths
        updated_records.append(updated)
    return _restore_input_container(session_rows, updated_records)


def compute_switch_probabilities_around_switches(
    session_rows,
    window_size: int,
) -> list[dict[str, Any]]:
    """Compute switch-event windows around every switch."""

    window_size = int(window_size)
    events: list[dict[str, Any]] = []
    for row in _iter_session_records(session_rows):
        choices = _clean_choice_history(row.get("choice_history", []))
        switch_indices = [int(v) for v in row.get("switch_indices", [])]
        if not choices or not switch_indices:
            continue

        for switch_idx in switch_indices:
            if switch_idx <= 0:
                continue

            switch_events = [math.nan] * (2 * window_size + 1)
            start_idx = max(0, switch_idx - window_size)
            end_idx = min(len(choices), switch_idx + window_size + 1)
            for trial_idx in range(start_idx, end_idx):
                if trial_idx == 0:
                    continue
                rel_pos = trial_idx - switch_idx
                arr_idx = rel_pos + window_size
                if 0 <= arr_idx < len(switch_events):
                    switch_events[arr_idx] = float(choices[trial_idx] != choices[trial_idx - 1])

            events.append(
                {
                    "session_idx": row.get("ses_idx"),
                    "subject_id": row.get("subject_id"),
                    "session_date": row.get("session_date"),
                    "switch_trial": switch_idx,
                    "switch_events": switch_events,
                    "relative_positions": list(range(-window_size, window_size + 1)),
                    "window_size": window_size,
                }
            )

    return events


def create_pooled_switch_analysis(
    switch_analysis_data: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pool switch-event windows into switch probabilities per relative position."""

    if not switch_analysis_data:
        return []

    relative_positions = list(switch_analysis_data[0].get("relative_positions", []))
    pooled: list[dict[str, Any]] = []
    for idx, rel_pos in enumerate(relative_positions):
        values = []
        for event in switch_analysis_data:
            switch_events = list(event.get("switch_events", []))
            if idx >= len(switch_events):
                continue
            value = switch_events[idx]
            if _is_nan(value):
                continue
            values.append(float(value))

        if not values:
            continue

        pooled.append(
            {
                "relative_position": int(rel_pos),
                "switch_probability": _mean(values),
                "switch_probability_sem": _sem(values),
                "n_events": len(values),
                "switch_probability_std": _std(values),
            }
        )

    return pooled


def compute_conditional_switch_probabilities_by_reward(
    session_rows,
    window_size: int,
) -> dict[str, list[dict[str, Any]]]:
    """Compute switch-event windows conditioned on reward outcome."""

    conditioned = {"rewarded": [], "unrewarded": []}
    for row in _iter_session_records(session_rows):
        choices, rewards = _aligned_choice_reward_histories(
            row.get("choice_history", []),
            row.get("reward_history", []),
        )
        switch_indices = [int(v) for v in row.get("switch_indices", [])]
        if not choices or not rewards or not switch_indices:
            continue

        for switch_idx in switch_indices:
            if switch_idx <= 0 or switch_idx >= len(rewards):
                continue
            condition = "rewarded" if rewards[switch_idx] > 0 else "unrewarded"
            switch_events = _build_switch_window(choices, switch_idx, int(window_size))
            conditioned[condition].append(
                {
                    "session_idx": row.get("ses_idx"),
                    "subject_id": row.get("subject_id"),
                    "session_date": row.get("session_date"),
                    "switch_trial": switch_idx,
                    "switch_rewarded": rewards[switch_idx] > 0,
                    "switch_events": switch_events,
                    "relative_positions": list(range(-int(window_size), int(window_size) + 1)),
                    "window_size": int(window_size),
                }
            )

    return conditioned


def analyze_post_switch_probability_by_reward(
    conditional_analysis_data: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Compute p_switch(t+1) for rewarded and unrewarded switch trials."""

    return {
        condition: _summarize_post_switch_probability(events)
        for condition, events in conditional_analysis_data.items()
    }


def compute_conditional_switch_probabilities_by_reward_and_run_length(
    session_rows,
    window_size: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Compute switch-event windows conditioned on reward and run length."""

    conditioned = {
        "rewarded": {"run_length_1": [], "run_length_gt1": []},
        "unrewarded": {"run_length_1": [], "run_length_gt1": []},
    }

    for row in _iter_session_records(session_rows):
        choices, rewards = _aligned_choice_reward_histories(
            row.get("choice_history", []),
            row.get("reward_history", []),
        )
        switch_indices = [int(v) for v in row.get("switch_indices", [])]
        run_lengths = [int(v) for v in row.get("run_lengths", [])]
        if not choices or not rewards or not switch_indices or not run_lengths:
            continue

        for pos, switch_idx in enumerate(switch_indices):
            if switch_idx <= 0 or switch_idx >= len(rewards) or pos >= len(run_lengths):
                continue

            reward_condition = "rewarded" if rewards[switch_idx] > 0 else "unrewarded"
            run_condition = "run_length_1" if run_lengths[pos] == 1 else "run_length_gt1"
            switch_events = _build_switch_window(choices, switch_idx, int(window_size))
            conditioned[reward_condition][run_condition].append(
                {
                    "session_idx": row.get("ses_idx"),
                    "subject_id": row.get("subject_id"),
                    "session_date": row.get("session_date"),
                    "switch_trial": switch_idx,
                    "switch_rewarded": rewards[switch_idx] > 0,
                    "preceding_run_length": run_lengths[pos],
                    "switch_events": switch_events,
                    "relative_positions": list(range(-int(window_size), int(window_size) + 1)),
                    "window_size": int(window_size),
                }
            )

    return conditioned


def analyze_post_switch_probability_by_reward_and_run_length(
    conditional_analysis_data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute p_switch(t+1) for reward x run-length conditions."""

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for reward_condition, run_groups in conditional_analysis_data.items():
        results[reward_condition] = {}
        for run_condition, events in run_groups.items():
            results[reward_condition][run_condition] = _summarize_post_switch_probability(
                events
            )
    return results


def _normalize_history_pattern_type(pattern_type: str) -> str:
    normalized = str(pattern_type).strip().lower()
    if normalized not in _HISTORY_PATTERN_TYPES:
        raise ValueError(
            f"Unsupported default_pattern_type={pattern_type!r}. "
            "Use 'detailed' or 'abstract'."
        )
    return normalized


def _build_history_pattern_count_records(
    session_rows,
    *,
    max_trials_back: int,
) -> list[dict[str, Any]]:
    return [
        _extract_history_pattern_count_record(row, max_trials_back=max_trials_back)
        for row in _iter_session_records(session_rows)
    ]


def _extract_history_pattern_count_record(
    row: Mapping[str, Any],
    *,
    max_trials_back: int,
) -> dict[str, Any]:
    counts = _new_history_pattern_count_tree(max_trials_back)
    choices, rewards = _aligned_choice_reward_histories_for_history_patterns(
        row.get("choice_history", []),
        row.get("reward_history", []),
    )

    if len(choices) > 1:
        encoded_trials = [
            _encode_history_trial(choice, reward)
            for choice, reward in zip(choices, rewards)
        ]
        for n_back in range(1, int(max_trials_back) + 1):
            if len(encoded_trials) <= n_back:
                continue
            for trial_idx in range(n_back, len(encoded_trials)):
                history_pattern = "".join(encoded_trials[trial_idx - n_back : trial_idx])
                is_switch = choices[trial_idx] != choices[trial_idx - 1]
                _increment_history_pattern_counts(
                    counts["detailed"][n_back],
                    history_pattern,
                    is_switch=is_switch,
                )
                _increment_history_pattern_counts(
                    counts["abstract"][n_back],
                    _history_pattern_to_abstract(history_pattern),
                    is_switch=is_switch,
                )

    return {
        "subject_id": _normalize_identifier(row.get("subject_id")),
        "ses_idx": str(row.get("ses_idx")),
        "source_ses_idx": row.get("source_ses_idx"),
        "session_date": row.get("session_date"),
        "curriculum_name": row.get("curriculum_name"),
        "counts": counts,
    }


def _aligned_choice_reward_histories_for_history_patterns(
    choices: Sequence[Any],
    rewards: Sequence[Any],
) -> tuple[list[int], list[float]]:
    cleaned_choices: list[int] = []
    cleaned_rewards: list[float] = []
    max_len = min(len(choices), len(rewards))
    for idx in range(max_len):
        choice = _normalize_choice_value(choices[idx])
        if _is_nan(choice):
            continue

        raw_reward = rewards[idx]
        if raw_reward is None or _is_nan(raw_reward):
            continue

        cleaned_choices.append(int(choice))
        cleaned_rewards.append(float(_normalize_reward_value(raw_reward)))
    return cleaned_choices, cleaned_rewards


def _encode_history_trial(choice: int, reward: float) -> str:
    if int(choice) == 0:
        return "L" if float(reward) > 0 else "l"
    return "R" if float(reward) > 0 else "r"


def _history_pattern_to_abstract(pattern: str) -> str:
    if not pattern:
        return ""
    first_side = pattern[0].upper()
    if first_side == "L":
        mapping = {"L": "A", "l": "a", "R": "B", "r": "b"}
    else:
        mapping = {"L": "B", "l": "b", "R": "A", "r": "a"}
    return "".join(mapping[char] for char in pattern)


def _new_history_pattern_count_tree(
    max_trials_back: int,
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    return {
        pattern_type: {
            n_back: {}
            for n_back in range(1, int(max_trials_back) + 1)
        }
        for pattern_type in _HISTORY_PATTERN_TYPES
    }


def _increment_history_pattern_counts(
    target: dict[str, dict[str, float]],
    pattern: str,
    *,
    is_switch: bool,
) -> None:
    leaf = target.setdefault(
        pattern,
        {"switches": 0.0, "stays": 0.0, "total": 0.0},
    )
    leaf["total"] += 1.0
    if is_switch:
        leaf["switches"] += 1.0
    else:
        leaf["stays"] += 1.0


def _aggregate_history_pattern_count_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
    average_rollouts_by_source: bool,
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    aggregated = _new_history_pattern_count_tree(max_trials_back)
    if average_rollouts_by_source and all(
        not _is_missing_session_identifier(record.get("source_ses_idx"))
        for record in records
    ):
        averaged_records, _ = _average_history_pattern_records_by_source(
            records,
            max_trials_back=max_trials_back,
        )
        for record in averaged_records:
            _accumulate_history_pattern_count_tree(
                aggregated,
                _as_dict(record.get("counts", {})),
                max_trials_back=max_trials_back,
            )
        return aggregated

    for record in records:
        _accumulate_history_pattern_count_tree(
            aggregated,
            _as_dict(record.get("counts", {})),
            max_trials_back=max_trials_back,
        )
    return aggregated


def _average_history_pattern_records_by_source(
    records: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
) -> tuple[list[dict[str, Any]], dict[Any, int]]:
    grouped: dict[tuple[Any, str], list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            (
                record.get("subject_id"),
                _coalesce_session_identifier(
                    record.get("source_ses_idx"),
                    fallback=record.get("ses_idx"),
                ),
            ),
            [],
        ).append(record)

    averaged_records: list[dict[str, Any]] = []
    subject_session_counts: dict[Any, int] = {}
    for (subject_id, source_ses_idx), group_records in grouped.items():
        first_record = group_records[0]
        averaged_records.append(
            {
                "subject_id": subject_id,
                "ses_idx": source_ses_idx,
                "source_ses_idx": source_ses_idx,
                "session_date": first_record.get("session_date"),
                "curriculum_name": first_record.get("curriculum_name"),
                "counts": _average_history_pattern_count_trees(
                    [
                        _as_dict(record.get("counts", {}))
                        for record in group_records
                    ],
                    max_trials_back=max_trials_back,
                ),
            }
        )
        subject_session_counts[subject_id] = subject_session_counts.get(subject_id, 0) + 1

    return averaged_records, subject_session_counts


def _average_history_pattern_count_trees(
    trees: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    averaged = _new_history_pattern_count_tree(max_trials_back)
    if not trees:
        return averaged

    n_trees = float(len(trees))
    for pattern_type in _HISTORY_PATTERN_TYPES:
        for n_back in range(1, int(max_trials_back) + 1):
            all_patterns = set()
            for tree in trees:
                all_patterns.update(
                    _as_dict(_as_dict(tree.get(pattern_type, {})).get(n_back, {})).keys()
                )
            for pattern in all_patterns:
                leaf = averaged[pattern_type][n_back].setdefault(
                    pattern,
                    {"switches": 0.0, "stays": 0.0, "total": 0.0},
                )
                leaf["switches"] = (
                    sum(
                        float(
                            _as_dict(
                                _as_dict(_as_dict(tree.get(pattern_type, {})).get(n_back, {})).get(
                                    pattern,
                                    {},
                                )
                            ).get("switches", 0.0)
                        )
                        for tree in trees
                    )
                    / n_trees
                )
                leaf["stays"] = (
                    sum(
                        float(
                            _as_dict(
                                _as_dict(_as_dict(tree.get(pattern_type, {})).get(n_back, {})).get(
                                    pattern,
                                    {},
                                )
                            ).get("stays", 0.0)
                        )
                        for tree in trees
                    )
                    / n_trees
                )
                leaf["total"] = (
                    sum(
                        float(
                            _as_dict(
                                _as_dict(_as_dict(tree.get(pattern_type, {})).get(n_back, {})).get(
                                    pattern,
                                    {},
                                )
                            ).get("total", 0.0)
                        )
                        for tree in trees
                    )
                    / n_trees
                )
    return averaged


def _accumulate_history_pattern_count_tree(
    target: dict[str, dict[int, dict[str, dict[str, float]]]],
    source: Mapping[str, Any],
    *,
    max_trials_back: int,
) -> None:
    for pattern_type in _HISTORY_PATTERN_TYPES:
        source_pattern_group = _as_dict(source.get(pattern_type, {}))
        for n_back in range(1, int(max_trials_back) + 1):
            source_bucket = _as_dict(source_pattern_group.get(n_back, {}))
            for pattern, counts in source_bucket.items():
                leaf = target[pattern_type][n_back].setdefault(
                    pattern,
                    {"switches": 0.0, "stays": 0.0, "total": 0.0},
                )
                count_leaf = _as_dict(counts)
                leaf["switches"] += float(count_leaf.get("switches", 0.0))
                leaf["stays"] += float(count_leaf.get("stays", 0.0))
                leaf["total"] += float(count_leaf.get("total", 0.0))


def _copy_history_pattern_count_tree(
    counts: Mapping[str, Any],
    *,
    max_trials_back: int,
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    copied = _new_history_pattern_count_tree(max_trials_back)
    _accumulate_history_pattern_count_tree(
        copied,
        counts,
        max_trials_back=max_trials_back,
    )
    return copied


def _finalize_history_pattern_count_tree(
    raw_tree: Mapping[str, Any],
    *,
    intervals: Mapping[str, Any] | None = None,
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    finalized: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for pattern_type in _HISTORY_PATTERN_TYPES:
        finalized[pattern_type] = {}
        pattern_group = _as_dict(raw_tree.get(pattern_type, {}))
        interval_group = _as_dict(_as_dict(intervals).get(pattern_type, {}))
        for n_back, bucket in pattern_group.items():
            finalized[pattern_type][int(n_back)] = {}
            interval_bucket = _as_dict(interval_group.get(int(n_back), {}))
            for pattern, counts in sorted(_as_dict(bucket).items()):
                finalized[pattern_type][int(n_back)][str(pattern)] = _finalize_history_pattern_leaf(
                    _as_dict(counts),
                    intervals=_as_dict(interval_bucket.get(str(pattern), {})),
                )
    return finalized


def _finalize_history_pattern_leaf(
    counts: Mapping[str, Any],
    *,
    intervals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    probability = _history_probability_from_counts(counts)
    total = float(counts.get("total", 0.0))
    return {
        "switches": _normalize_count_output(counts.get("switches", 0.0)),
        "stays": _normalize_count_output(counts.get("stays", 0.0)),
        "total": _normalize_count_output(total),
        "switch_probability": probability,
        "switch_probability_sem": _history_switch_probability_sem(probability, total),
        "ci_low": _coerce_probability(_as_dict(intervals).get("ci_low")),
        "ci_high": _coerce_probability(_as_dict(intervals).get("ci_high")),
    }


def _history_probability_from_counts(counts: Mapping[str, Any]) -> float:
    total = float(counts.get("total", 0.0))
    if total <= 0:
        return math.nan
    return float(float(counts.get("switches", 0.0)) / total)


def _history_switch_probability_sem(probability: float, total: float) -> float:
    if total <= 0 or _is_nan(probability):
        return math.nan
    return float(math.sqrt(probability * (1.0 - probability) / total))


def _build_history_pattern_comparison(
    animal_summary: Mapping[str, Any],
    simulated_summary: Mapping[str, Any],
    *,
    max_trials_back: int,
    aggregate_min_trials: int,
) -> dict[str, dict[int, dict[str, Any]]]:
    comparison: dict[str, dict[int, dict[str, Any]]] = {}
    for pattern_type in _HISTORY_PATTERN_TYPES:
        comparison[pattern_type] = {}
        animal_group = _as_dict(animal_summary.get(pattern_type, {}))
        simulated_group = _as_dict(simulated_summary.get(pattern_type, {}))
        for n_back in range(1, int(max_trials_back) + 1):
            animal_bucket = _as_dict(animal_group.get(n_back, {}))
            simulated_bucket = _as_dict(simulated_group.get(n_back, {}))
            rows = []
            for pattern in sorted(set(animal_bucket) & set(simulated_bucket)):
                animal_leaf = _as_dict(animal_bucket.get(pattern, {}))
                simulated_leaf = _as_dict(simulated_bucket.get(pattern, {}))
                rows.append(
                    {
                        "pattern": str(pattern),
                        "animal_probability": _coerce_probability(
                            animal_leaf.get("switch_probability")
                        ),
                        "animal_sem": _coerce_probability(
                            animal_leaf.get("switch_probability_sem")
                        ),
                        "animal_total": _normalize_count_output(
                            animal_leaf.get("total", 0.0)
                        ),
                        "simulated_probability": _coerce_probability(
                            simulated_leaf.get("switch_probability")
                        ),
                        "simulated_sem": _coerce_probability(
                            simulated_leaf.get("switch_probability_sem")
                        ),
                        "simulated_total_effective": _normalize_count_output(
                            simulated_leaf.get("total", 0.0)
                        ),
                        "delta_probability": _finite_difference(
                            _coerce_probability(simulated_leaf.get("switch_probability")),
                            _coerce_probability(animal_leaf.get("switch_probability")),
                        ),
                    }
                )
            comparison[pattern_type][n_back] = {
                "rows": rows,
                "summary": _summarize_history_pattern_comparison_rows(
                    rows,
                    min_trials=aggregate_min_trials,
                ),
            }
    return comparison


def _summarize_history_pattern_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_trials: int,
) -> dict[str, Any]:
    valid_rows = _select_valid_history_pattern_rows(rows, min_trials=min_trials)
    xs = [float(row["animal_probability"]) for row in valid_rows]
    ys = [float(row["simulated_probability"]) for row in valid_rows]
    return {
        "n_patterns": len(valid_rows),
        "correlation": _pearson_correlation(xs, ys),
        "rmse": _rmse(xs, ys),
    }


def _select_valid_history_pattern_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_trials: int,
) -> list[dict[str, Any]]:
    valid_rows = []
    for row in rows:
        animal_probability = _coerce_probability(row.get("animal_probability"))
        simulated_probability = _coerce_probability(row.get("simulated_probability"))
        animal_total = row.get("animal_total")
        simulated_total = row.get("simulated_total_effective")
        if animal_probability is None or simulated_probability is None:
            continue
        if animal_total is None or simulated_total is None:
            continue
        if float(animal_total) < float(min_trials) or float(simulated_total) < float(min_trials):
            continue
        valid_rows.append(dict(row))
    return valid_rows


def _build_subject_history_pattern_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
    average_rollouts_by_source: bool,
    subject_curriculum_map: Mapping[Any, str] | None = None,
) -> dict[Any, dict[str, Any]]:
    if not records:
        return {}

    subject_order = _unique_preserve_order([record.get("subject_id") for record in records])
    if average_rollouts_by_source and all(
        not _is_missing_session_identifier(record.get("source_ses_idx"))
        for record in records
    ):
        per_subject_records, subject_session_counts = _average_history_records_for_subjects(
            records,
            max_trials_back=max_trials_back,
        )
    else:
        per_subject_records, subject_session_counts = _collect_direct_history_records_for_subjects(
            records,
            max_trials_back=max_trials_back,
        )
    curriculum_lookup = (
        dict(subject_curriculum_map) if isinstance(subject_curriculum_map, Mapping) else {}
    )

    summaries: dict[Any, dict[str, Any]] = {}
    for subject_id in subject_order:
        subject_records = per_subject_records.get(subject_id, [])
        if not subject_records:
            continue

        aggregated = _new_history_pattern_count_tree(max_trials_back)
        for count_tree in subject_records:
            _accumulate_history_pattern_count_tree(
                aggregated,
                count_tree,
                max_trials_back=max_trials_back,
            )

        bootstrap_intervals = _bootstrap_history_pattern_intervals(
            subject_records,
            max_trials_back=max_trials_back,
            seed=_seed_from_parts(
                _DEFAULT_SUBJECT_BOOTSTRAP_SEED,
                "history",
                subject_id,
            ),
            n_bootstrap=_DEFAULT_SUBJECT_BOOTSTRAP_RESAMPLES,
        )
        summaries[subject_id] = {
            "patterns": _finalize_history_pattern_count_tree(
                aggregated,
                intervals=bootstrap_intervals,
            ),
            "n_source_sessions": int(subject_session_counts.get(subject_id, 0)),
            "curriculum_name": str(curriculum_lookup.get(subject_id, "Unknown")),
        }

    return summaries


def _build_session_history_pattern_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
    average_rollouts_by_source: bool,
) -> dict[tuple[Any, str], dict[str, Any]]:
    if not records:
        return {}

    if average_rollouts_by_source and all(
        not _is_missing_session_identifier(record.get("source_ses_idx"))
        for record in records
    ):
        grouped_records = _average_history_records_for_sessions(
            records,
            max_trials_back=max_trials_back,
        )
    else:
        grouped_records = _collect_direct_history_records_for_sessions(
            records,
            max_trials_back=max_trials_back,
        )

    summaries: dict[tuple[Any, str], dict[str, Any]] = {}
    for record in grouped_records:
        session_key = _canonical_session_key(record)
        summaries[session_key] = {
            "subject_id": session_key[0],
            "session_id": session_key[1],
            "source_ses_idx": session_key[1],
            "session_date": record.get("session_date"),
            "curriculum_name": str(record.get("curriculum_name") or "Unknown"),
            "patterns": _finalize_history_pattern_count_tree(
                _copy_history_pattern_count_tree(
                    _as_dict(record.get("counts", {})),
                    max_trials_back=max_trials_back,
                )
            ),
        }
    return summaries


def _average_history_records_for_subjects(
    records: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
) -> tuple[dict[Any, list[dict[str, Any]]], dict[Any, int]]:
    averaged_records, subject_session_counts = _average_history_pattern_records_by_source(
        records,
        max_trials_back=max_trials_back,
    )
    per_subject_records: dict[Any, list[dict[str, Any]]] = {}
    for record in averaged_records:
        per_subject_records.setdefault(record.get("subject_id"), []).append(
            _copy_history_pattern_count_tree(
                _as_dict(record.get("counts", {})),
                max_trials_back=max_trials_back,
            )
        )
    return per_subject_records, subject_session_counts


def _average_history_records_for_sessions(
    records: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
) -> list[dict[str, Any]]:
    averaged_records, _ = _average_history_pattern_records_by_source(
        records,
        max_trials_back=max_trials_back,
    )
    return averaged_records


def _collect_direct_history_records_for_subjects(
    records: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
) -> tuple[dict[Any, list[dict[str, Any]]], dict[Any, int]]:
    per_subject_records: dict[Any, list[dict[str, Any]]] = {}
    session_keys_by_subject: dict[Any, list[str]] = {}
    for record in records:
        subject_id = record.get("subject_id")
        per_subject_records.setdefault(subject_id, []).append(
            _copy_history_pattern_count_tree(
                _as_dict(record.get("counts", {})),
                max_trials_back=max_trials_back,
            )
        )
        session_key = _coalesce_session_identifier(
            record.get("source_ses_idx"),
            fallback=record.get("ses_idx"),
        )
        session_keys_by_subject.setdefault(subject_id, []).append(str(session_key))
    return (
        per_subject_records,
        {
            subject_id: len(_unique_preserve_order(session_keys))
            for subject_id, session_keys in session_keys_by_subject.items()
        },
    )


def _collect_direct_history_records_for_sessions(
    records: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
) -> list[dict[str, Any]]:
    direct_records: list[dict[str, Any]] = []
    for record in records:
        session_key = _canonical_session_key(record)
        direct_records.append(
            {
                "subject_id": session_key[0],
                "ses_idx": session_key[1],
                "source_ses_idx": session_key[1],
                "session_date": record.get("session_date"),
                "curriculum_name": record.get("curriculum_name"),
                "counts": _copy_history_pattern_count_tree(
                    _as_dict(record.get("counts", {})),
                    max_trials_back=max_trials_back,
                ),
            }
        )
    return direct_records


def _build_subject_history_pattern_comparison(
    animal_subject_stats: Mapping[Any, Mapping[str, Any]],
    simulated_subject_stats: Mapping[Any, Mapping[str, Any]],
    *,
    max_trials_back: int,
    subject_min_trials: int,
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    subject_level: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        pattern_type: {
            n_back: {}
            for n_back in range(1, int(max_trials_back) + 1)
        }
        for pattern_type in _HISTORY_PATTERN_TYPES
    }
    matched_subject_ids = [
        subject_id
        for subject_id in _unique_preserve_order(list(animal_subject_stats.keys()))
        if subject_id in simulated_subject_stats
    ]

    for pattern_type in _HISTORY_PATTERN_TYPES:
        for n_back in range(1, int(max_trials_back) + 1):
            patterns = _collect_subject_history_patterns(
                animal_subject_stats,
                simulated_subject_stats,
                matched_subject_ids=matched_subject_ids,
                pattern_type=pattern_type,
                n_back=n_back,
            )
            for pattern in patterns:
                points = _build_subject_history_pattern_points(
                    animal_subject_stats,
                    simulated_subject_stats,
                    matched_subject_ids=matched_subject_ids,
                    pattern_type=pattern_type,
                    n_back=n_back,
                    pattern=pattern,
                )
                subject_level[pattern_type][n_back][pattern] = {
                    "points": points,
                    "summary": _summarize_subject_history_pattern_points(
                        points,
                        min_trials=subject_min_trials,
                    ),
                }
    return subject_level


def _build_session_history_pattern_comparison(
    animal_session_stats: Mapping[tuple[Any, str], Mapping[str, Any]],
    simulated_session_stats: Mapping[tuple[Any, str], Mapping[str, Any]],
    *,
    max_trials_back: int,
    session_min_trials: int,
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    session_level: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        pattern_type: {
            n_back: {}
            for n_back in range(1, int(max_trials_back) + 1)
        }
        for pattern_type in _HISTORY_PATTERN_TYPES
    }
    matched_session_keys = [
        session_key
        for session_key in animal_session_stats.keys()
        if session_key in simulated_session_stats
    ]

    for pattern_type in _HISTORY_PATTERN_TYPES:
        for n_back in range(1, int(max_trials_back) + 1):
            patterns = _collect_session_history_patterns(
                animal_session_stats,
                simulated_session_stats,
                matched_session_keys=matched_session_keys,
                pattern_type=pattern_type,
                n_back=n_back,
            )
            for pattern in patterns:
                points = _build_session_history_pattern_points(
                    animal_session_stats,
                    simulated_session_stats,
                    matched_session_keys=matched_session_keys,
                    pattern_type=pattern_type,
                    n_back=n_back,
                    pattern=pattern,
                )
                session_level[pattern_type][n_back][pattern] = {
                    "points": points,
                    "summary": _summarize_session_history_pattern_points(
                        points,
                        min_trials=session_min_trials,
                    ),
                }
    return session_level


def _collect_subject_history_patterns(
    animal_subject_stats: Mapping[Any, Mapping[str, Any]],
    simulated_subject_stats: Mapping[Any, Mapping[str, Any]],
    *,
    matched_subject_ids: Sequence[Any],
    pattern_type: str,
    n_back: int,
) -> list[str]:
    patterns = set()
    for subject_id in matched_subject_ids:
        animal_bucket = _as_dict(
            _as_dict(
                _as_dict(animal_subject_stats.get(subject_id, {})).get("patterns", {})
            ).get(pattern_type, {})
        )
        simulated_bucket = _as_dict(
            _as_dict(
                _as_dict(simulated_subject_stats.get(subject_id, {})).get("patterns", {})
            ).get(pattern_type, {})
        )
        patterns.update(_as_dict(animal_bucket.get(n_back, {})).keys())
        patterns.update(_as_dict(simulated_bucket.get(n_back, {})).keys())
    return sorted(str(pattern) for pattern in patterns)


def _collect_session_history_patterns(
    animal_session_stats: Mapping[tuple[Any, str], Mapping[str, Any]],
    simulated_session_stats: Mapping[tuple[Any, str], Mapping[str, Any]],
    *,
    matched_session_keys: Sequence[tuple[Any, str]],
    pattern_type: str,
    n_back: int,
) -> list[str]:
    patterns = set()
    for session_key in matched_session_keys:
        animal_bucket = _as_dict(
            _as_dict(
                _as_dict(animal_session_stats.get(session_key, {})).get("patterns", {})
            ).get(pattern_type, {})
        )
        simulated_bucket = _as_dict(
            _as_dict(
                _as_dict(simulated_session_stats.get(session_key, {})).get(
                    "patterns",
                    {},
                )
            ).get(pattern_type, {})
        )
        patterns.update(_as_dict(animal_bucket.get(n_back, {})).keys())
        patterns.update(_as_dict(simulated_bucket.get(n_back, {})).keys())
    return sorted(str(pattern) for pattern in patterns)


def _build_subject_history_pattern_points(
    animal_subject_stats: Mapping[Any, Mapping[str, Any]],
    simulated_subject_stats: Mapping[Any, Mapping[str, Any]],
    *,
    matched_subject_ids: Sequence[Any],
    pattern_type: str,
    n_back: int,
    pattern: str,
) -> list[dict[str, Any]]:
    points = []
    for subject_id in matched_subject_ids:
        animal_leaf = _as_dict(
            _as_dict(
                _as_dict(
                    _as_dict(
                        _as_dict(animal_subject_stats.get(subject_id, {})).get("patterns", {})
                    ).get(pattern_type, {})
                ).get(n_back, {})
            ).get(pattern, {})
        )
        simulated_leaf = _as_dict(
            _as_dict(
                _as_dict(
                    _as_dict(
                        _as_dict(simulated_subject_stats.get(subject_id, {})).get(
                            "patterns",
                            {},
                        )
                    ).get(pattern_type, {})
                ).get(n_back, {})
            ).get(pattern, {})
        )
        points.append(
            {
                "subject_id": subject_id,
                "animal_probability": _coerce_probability(
                    animal_leaf.get("switch_probability")
                ),
                "animal_total": _normalize_count_output(animal_leaf.get("total", 0.0)),
                "animal_n_sessions": int(
                    _as_dict(animal_subject_stats.get(subject_id, {})).get(
                        "n_source_sessions",
                        0,
                    )
                ),
                "animal_n_source_sessions": int(
                    _as_dict(animal_subject_stats.get(subject_id, {})).get(
                        "n_source_sessions",
                        0,
                    )
                ),
                "animal_ci_low": _coerce_probability(animal_leaf.get("ci_low")),
                "animal_ci_high": _coerce_probability(animal_leaf.get("ci_high")),
                "simulated_probability": _coerce_probability(
                    simulated_leaf.get("switch_probability")
                ),
                "simulated_total_effective": _normalize_count_output(
                    simulated_leaf.get("total", 0.0)
                ),
                "simulated_n_source_sessions": int(
                    _as_dict(simulated_subject_stats.get(subject_id, {})).get(
                        "n_source_sessions",
                        0,
                    )
                ),
                "simulated_ci_low": _coerce_probability(simulated_leaf.get("ci_low")),
                "simulated_ci_high": _coerce_probability(simulated_leaf.get("ci_high")),
                "curriculum_name": str(
                    _as_dict(animal_subject_stats.get(subject_id, {})).get(
                        "curriculum_name",
                        _as_dict(simulated_subject_stats.get(subject_id, {})).get(
                            "curriculum_name",
                            "Unknown",
                        ),
                    )
                ),
                "delta_probability": _finite_difference(
                    _coerce_probability(simulated_leaf.get("switch_probability")),
                    _coerce_probability(animal_leaf.get("switch_probability")),
                ),
            }
        )
    return points


def _build_session_history_pattern_points(
    animal_session_stats: Mapping[tuple[Any, str], Mapping[str, Any]],
    simulated_session_stats: Mapping[tuple[Any, str], Mapping[str, Any]],
    *,
    matched_session_keys: Sequence[tuple[Any, str]],
    pattern_type: str,
    n_back: int,
    pattern: str,
) -> list[dict[str, Any]]:
    points = []
    for session_key in matched_session_keys:
        animal_metrics = _as_dict(animal_session_stats.get(session_key, {}))
        simulated_metrics = _as_dict(simulated_session_stats.get(session_key, {}))
        animal_leaf = _as_dict(
            _as_dict(
                _as_dict(
                    _as_dict(animal_metrics.get("patterns", {})).get(pattern_type, {})
                ).get(n_back, {})
            ).get(pattern, {})
        )
        simulated_leaf = _as_dict(
            _as_dict(
                _as_dict(
                    _as_dict(simulated_metrics.get("patterns", {})).get(pattern_type, {})
                ).get(n_back, {})
            ).get(pattern, {})
        )
        points.append(
            {
                "subject_id": animal_metrics.get("subject_id", session_key[0]),
                "session_id": animal_metrics.get("session_id", session_key[1]),
                "source_ses_idx": animal_metrics.get("source_ses_idx", session_key[1]),
                "session_date": animal_metrics.get(
                    "session_date",
                    simulated_metrics.get("session_date"),
                ),
                "curriculum_name": str(
                    animal_metrics.get("curriculum_name")
                    or simulated_metrics.get("curriculum_name")
                    or "Unknown"
                ),
                "animal_probability": _coerce_probability(
                    animal_leaf.get("switch_probability")
                ),
                "animal_total": _normalize_count_output(animal_leaf.get("total", 0.0)),
                "animal_ci_low": _coerce_probability(animal_leaf.get("ci_low")),
                "animal_ci_high": _coerce_probability(animal_leaf.get("ci_high")),
                "simulated_probability": _coerce_probability(
                    simulated_leaf.get("switch_probability")
                ),
                "simulated_total_effective": _normalize_count_output(
                    simulated_leaf.get("total", 0.0)
                ),
                "simulated_ci_low": _coerce_probability(simulated_leaf.get("ci_low")),
                "simulated_ci_high": _coerce_probability(simulated_leaf.get("ci_high")),
                "delta_probability": _finite_difference(
                    _coerce_probability(simulated_leaf.get("switch_probability")),
                    _coerce_probability(animal_leaf.get("switch_probability")),
                ),
            }
        )
    return points


def _summarize_subject_history_pattern_points(
    points: Sequence[Mapping[str, Any]],
    *,
    min_trials: int,
) -> dict[str, Any]:
    return _summarize_group_history_pattern_points(
        points,
        selector=lambda rows: _select_valid_subject_history_pattern_points(
            rows,
            min_trials=min_trials,
        ),
        count_label="n_subjects",
    )


def _select_valid_subject_history_pattern_points(
    points: Sequence[Mapping[str, Any]],
    *,
    min_trials: int,
) -> list[dict[str, Any]]:
    valid_points = []
    for point in points:
        animal_probability = _coerce_probability(point.get("animal_probability"))
        simulated_probability = _coerce_probability(point.get("simulated_probability"))
        animal_total = point.get("animal_total")
        simulated_total = point.get("simulated_total_effective")
        if animal_probability is None or simulated_probability is None:
            continue
        if animal_total is None or simulated_total is None:
            continue
        if float(animal_total) < float(min_trials) or float(simulated_total) < float(min_trials):
            continue
        valid_points.append(dict(point))
    return valid_points


def _select_valid_session_history_pattern_points(
    points: Sequence[Mapping[str, Any]],
    *,
    min_trials: int,
) -> list[dict[str, Any]]:
    valid_points = []
    for point in points:
        animal_probability = _coerce_probability(point.get("animal_probability"))
        simulated_probability = _coerce_probability(point.get("simulated_probability"))
        animal_total = point.get("animal_total")
        simulated_total = point.get("simulated_total_effective")
        if animal_probability is None or simulated_probability is None:
            continue
        if animal_total is None or simulated_total is None:
            continue
        if float(animal_total) < float(min_trials) or float(simulated_total) < float(min_trials):
            continue
        valid_points.append(dict(point))
    return valid_points


def _summarize_session_history_pattern_points(
    points: Sequence[Mapping[str, Any]],
    *,
    min_trials: int,
) -> dict[str, Any]:
    return _summarize_group_history_pattern_points(
        points,
        selector=lambda rows: _select_valid_session_history_pattern_points(
            rows,
            min_trials=min_trials,
        ),
        count_label="n_sessions",
    )


def _summarize_group_history_pattern_points(
    points: Sequence[Mapping[str, Any]],
    *,
    selector,
    count_label: str,
) -> dict[str, Any]:
    valid_points = list(selector(points))
    xs = [float(point["animal_probability"]) for point in valid_points]
    ys = [float(point["simulated_probability"]) for point in valid_points]
    return {
        count_label: len(valid_points),
        "correlation": _pearson_correlation(xs, ys),
        "rmse": _rmse(xs, ys),
        "bias": _mean_difference(xs, ys),
    }


def _compute_subject_level_comparison(
    animal_sessions,
    simulated_sessions,
) -> dict[str, Any]:
    subject_curriculum_map = _resolve_subject_curriculum_map(
        animal_sessions,
        fallback_rows=simulated_sessions,
    )
    animal_subject_stats = _build_subject_level_metric_summary(
        animal_sessions,
        average_rollouts_by_source=False,
        subject_curriculum_map=subject_curriculum_map,
    )
    simulated_subject_stats = _build_subject_level_metric_summary(
        simulated_sessions,
        average_rollouts_by_source=True,
        subject_curriculum_map=subject_curriculum_map,
    )

    comparison = {
        "post_switch_by_reward": {},
        "post_switch_by_reward_and_run_length": {},
    }

    for reward_condition in _REWARD_CONDITIONS:
        points = _build_subject_comparison_points(
            animal_subject_stats,
            simulated_subject_stats,
            reward_condition=reward_condition,
        )
        comparison["post_switch_by_reward"][reward_condition] = {
            "points": points,
            "summary": _summarize_subject_comparison_points(points),
        }

    for reward_condition in _REWARD_CONDITIONS:
        comparison["post_switch_by_reward_and_run_length"][reward_condition] = {}
        for run_condition in _RUN_LENGTH_CONDITIONS:
            points = _build_subject_comparison_points(
                animal_subject_stats,
                simulated_subject_stats,
                reward_condition=reward_condition,
                run_condition=run_condition,
            )
            comparison["post_switch_by_reward_and_run_length"][reward_condition][
                run_condition
            ] = {
                "points": points,
                "summary": _summarize_subject_comparison_points(points),
            }

    return comparison


def _prepare_subject_metric_records(
    session_rows,
    *,
    average_rollouts_by_source: bool,
) -> tuple[dict[Any, list[dict[str, Any]]], dict[Any, int]]:
    session_counts = [
        _extract_subject_metric_counts(row)
        for row in _iter_session_records(session_rows)
    ]
    if not session_counts:
        return {}, {}

    if average_rollouts_by_source and all(
        not _is_missing_session_identifier(record.get("source_ses_idx"))
        for record in session_counts
    ):
        return _average_rollout_metric_counts(session_counts)
    return _collect_direct_metric_counts(session_counts)


def _canonical_source_session_id_from_record(record: Mapping[str, Any]) -> str:
    return _coalesce_session_identifier(
        record.get("source_ses_idx"),
        fallback=record.get("ses_idx"),
    )


def _canonical_session_key(record: Mapping[str, Any]) -> tuple[Any, str]:
    return (
        _normalize_identifier(record.get("subject_id")),
        _canonical_source_session_id_from_record(record),
    )


def _build_subject_level_metric_summary(
    session_rows,
    *,
    average_rollouts_by_source: bool,
    subject_curriculum_map: Mapping[Any, str] | None = None,
) -> dict[Any, dict[str, Any]]:
    per_subject_records, subject_session_counts = _prepare_subject_metric_records(
        session_rows,
        average_rollouts_by_source=average_rollouts_by_source,
    )
    if not per_subject_records:
        return {}

    subject_order = _unique_preserve_order(list(per_subject_records.keys()))
    curriculum_lookup = (
        dict(subject_curriculum_map) if isinstance(subject_curriculum_map, Mapping) else {}
    )

    summaries: dict[Any, dict[str, Any]] = {}
    for subject_id in subject_order:
        subject_records = per_subject_records.get(subject_id, [])
        if not subject_records:
            continue

        reward_totals = _new_reward_count_tree()
        reward_run_totals = _new_reward_run_count_tree()
        for record in subject_records:
            _accumulate_reward_counts(reward_totals, record["reward"])
            _accumulate_reward_run_counts(
                reward_run_totals,
                record["reward_and_run_length"],
            )

        reward_intervals, reward_run_intervals = _bootstrap_switch_metric_intervals(
            subject_records,
            seed=_seed_from_parts(
                _DEFAULT_SUBJECT_BOOTSTRAP_SEED,
                "switch",
                subject_id,
            ),
            n_bootstrap=_DEFAULT_SUBJECT_BOOTSTRAP_RESAMPLES,
        )
        summaries[subject_id] = {
            "post_switch_by_reward": {
                reward_condition: {
                    "probability": _probability_from_counts(
                        reward_totals[reward_condition]
                    ),
                    "n": _normalize_count_output(
                        reward_totals[reward_condition]["n"]
                    ),
                    "ci_low": _coerce_probability(
                        _as_dict(reward_intervals.get(reward_condition, {})).get("ci_low")
                    ),
                    "ci_high": _coerce_probability(
                        _as_dict(reward_intervals.get(reward_condition, {})).get("ci_high")
                    ),
                }
                for reward_condition in _REWARD_CONDITIONS
            },
            "post_switch_by_reward_and_run_length": {
                reward_condition: {
                    run_condition: {
                        "probability": _probability_from_counts(
                            reward_run_totals[reward_condition][run_condition]
                        ),
                        "n": _normalize_count_output(
                            reward_run_totals[reward_condition][run_condition]["n"]
                        ),
                        "ci_low": _coerce_probability(
                            _as_dict(
                                _as_dict(reward_run_intervals.get(reward_condition, {})).get(
                                    run_condition,
                                    {},
                                )
                            ).get("ci_low")
                        ),
                        "ci_high": _coerce_probability(
                            _as_dict(
                                _as_dict(reward_run_intervals.get(reward_condition, {})).get(
                                    run_condition,
                                    {},
                                )
                            ).get("ci_high")
                        ),
                    }
                    for run_condition in _RUN_LENGTH_CONDITIONS
                }
                for reward_condition in _REWARD_CONDITIONS
            },
            "n_source_sessions": int(subject_session_counts.get(subject_id, 0)),
            "curriculum_name": str(curriculum_lookup.get(subject_id, "Unknown")),
        }

    return summaries


def _prepare_session_metric_records(
    session_rows,
    *,
    average_rollouts_by_source: bool,
) -> list[dict[str, Any]]:
    session_counts = [
        _extract_subject_metric_counts(row)
        for row in _iter_session_records(session_rows)
    ]
    if not session_counts:
        return []

    if average_rollouts_by_source and all(
        not _is_missing_session_identifier(record.get("source_ses_idx"))
        for record in session_counts
    ):
        return _average_rollout_metric_counts_by_session(session_counts)
    return _collect_direct_metric_counts_by_session(session_counts)


def _build_session_level_metric_summary(
    session_rows,
    *,
    average_rollouts_by_source: bool,
    subject_curriculum_map: Mapping[Any, str] | None = None,
) -> dict[tuple[Any, str], dict[str, Any]]:
    session_records = _prepare_session_metric_records(
        session_rows,
        average_rollouts_by_source=average_rollouts_by_source,
    )
    if not session_records:
        return {}

    curriculum_lookup = (
        dict(subject_curriculum_map) if isinstance(subject_curriculum_map, Mapping) else {}
    )
    summaries: dict[tuple[Any, str], dict[str, Any]] = {}
    for record in session_records:
        session_key = _canonical_session_key(record)
        summaries[session_key] = {
            "subject_id": session_key[0],
            "session_id": session_key[1],
            "source_ses_idx": session_key[1],
            "session_date": record.get("session_date"),
            "curriculum_name": str(
                record.get("curriculum_name")
                or curriculum_lookup.get(session_key[0], "Unknown")
                or "Unknown"
            ),
            "post_switch_by_reward": {
                reward_condition: {
                    "probability": _probability_from_counts(
                        _as_dict(record["reward"]).get(reward_condition, {})
                    ),
                    "n": _normalize_count_output(
                        _as_dict(_as_dict(record["reward"]).get(reward_condition, {})).get(
                            "n",
                            0.0,
                        )
                    ),
                    "ci_low": None,
                    "ci_high": None,
                }
                for reward_condition in _REWARD_CONDITIONS
            },
            "post_switch_by_reward_and_run_length": {
                reward_condition: {
                    run_condition: {
                        "probability": _probability_from_counts(
                            _as_dict(
                                _as_dict(record["reward_and_run_length"]).get(
                                    reward_condition,
                                    {},
                                )
                            ).get(run_condition, {})
                        ),
                        "n": _normalize_count_output(
                            _as_dict(
                                _as_dict(record["reward_and_run_length"]).get(
                                    reward_condition,
                                    {},
                                )
                            )
                            .get(run_condition, {})
                            .get("n", 0.0)
                        ),
                        "ci_low": None,
                        "ci_high": None,
                    }
                    for run_condition in _RUN_LENGTH_CONDITIONS
                }
                for reward_condition in _REWARD_CONDITIONS
            },
        }
    return summaries


def _average_rollout_metric_counts(
    session_counts: Sequence[Mapping[str, Any]],
) -> tuple[dict[Any, list[dict[str, Any]]], dict[Any, int]]:
    grouped: dict[tuple[Any, str], list[Mapping[str, Any]]] = {}
    for record in session_counts:
        source_ses_idx = _canonical_source_session_id_from_record(record)
        grouped.setdefault((record["subject_id"], source_ses_idx), []).append(record)

    per_subject_records: dict[Any, list[dict[str, Any]]] = {}
    subject_session_counts: dict[Any, int] = {}
    for (subject_id, _source_ses_idx), records in grouped.items():
        averaged_reward = _new_reward_count_tree()
        averaged_reward_run = _new_reward_run_count_tree()
        n_records = float(len(records))

        for reward_condition in _REWARD_CONDITIONS:
            averaged_reward[reward_condition]["successes"] = (
                sum(
                    float(record["reward"][reward_condition]["successes"])
                    for record in records
                )
                / n_records
            )
            averaged_reward[reward_condition]["n"] = (
                sum(float(record["reward"][reward_condition]["n"]) for record in records)
                / n_records
            )
            for run_condition in _RUN_LENGTH_CONDITIONS:
                averaged_reward_run[reward_condition][run_condition]["successes"] = (
                    sum(
                        float(
                            record["reward_and_run_length"][reward_condition][
                                run_condition
                            ]["successes"]
                        )
                        for record in records
                    )
                    / n_records
                )
                averaged_reward_run[reward_condition][run_condition]["n"] = (
                    sum(
                        float(
                            record["reward_and_run_length"][reward_condition][
                                run_condition
                            ]["n"]
                        )
                        for record in records
                    )
                    / n_records
                )

        per_subject_records.setdefault(subject_id, []).append(
            {
                "reward": averaged_reward,
                "reward_and_run_length": averaged_reward_run,
            }
        )
        subject_session_counts[subject_id] = subject_session_counts.get(subject_id, 0) + 1

    return per_subject_records, subject_session_counts


def _average_rollout_metric_counts_by_session(
    session_counts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, str], list[Mapping[str, Any]]] = {}
    for record in session_counts:
        grouped.setdefault(_canonical_session_key(record), []).append(record)

    averaged_records: list[dict[str, Any]] = []
    for session_key, records in grouped.items():
        averaged_reward = _new_reward_count_tree()
        averaged_reward_run = _new_reward_run_count_tree()
        n_records = float(len(records))
        first_record = records[0]

        for reward_condition in _REWARD_CONDITIONS:
            pooled_successes = sum(
                float(record["reward"][reward_condition]["successes"])
                for record in records
            )
            pooled_n = sum(float(record["reward"][reward_condition]["n"]) for record in records)
            averaged_reward[reward_condition]["successes"] = pooled_successes / n_records
            averaged_reward[reward_condition]["n"] = pooled_n / n_records
            for run_condition in _RUN_LENGTH_CONDITIONS:
                pooled_run_successes = sum(
                    float(
                        record["reward_and_run_length"][reward_condition][run_condition][
                            "successes"
                        ]
                    )
                    for record in records
                )
                pooled_run_n = sum(
                    float(
                        record["reward_and_run_length"][reward_condition][run_condition][
                            "n"
                        ]
                    )
                    for record in records
                )
                averaged_reward_run[reward_condition][run_condition]["successes"] = (
                    pooled_run_successes / n_records
                )
                averaged_reward_run[reward_condition][run_condition]["n"] = (
                    pooled_run_n / n_records
                )

        averaged_records.append(
            {
                "subject_id": session_key[0],
                "ses_idx": session_key[1],
                "source_ses_idx": session_key[1],
                "session_date": first_record.get("session_date"),
                "curriculum_name": first_record.get("curriculum_name"),
                "reward": averaged_reward,
                "reward_and_run_length": averaged_reward_run,
            }
        )

    return averaged_records


def _collect_direct_metric_counts(
    session_counts: Sequence[Mapping[str, Any]],
) -> tuple[dict[Any, list[dict[str, Any]]], dict[Any, int]]:
    per_subject_records: dict[Any, list[dict[str, Any]]] = {}
    session_keys_by_subject: dict[Any, list[str]] = {}

    for record in session_counts:
        subject_id = record["subject_id"]
        per_subject_records.setdefault(subject_id, []).append(
            {
                "reward": _copy_reward_count_tree(record["reward"]),
                "reward_and_run_length": _copy_reward_run_count_tree(
                    record["reward_and_run_length"]
                ),
            }
        )
        session_key = _coalesce_session_identifier(
            record.get("source_ses_idx"),
            fallback=record.get("ses_idx"),
        )
        session_keys_by_subject.setdefault(subject_id, []).append(str(session_key))

    subject_session_counts = {
        subject_id: len(_unique_preserve_order(session_keys))
        for subject_id, session_keys in session_keys_by_subject.items()
    }
    return per_subject_records, subject_session_counts


def _collect_direct_metric_counts_by_session(
    session_counts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    direct_records: list[dict[str, Any]] = []
    for record in session_counts:
        session_key = _canonical_session_key(record)
        direct_records.append(
            {
                "subject_id": session_key[0],
                "ses_idx": session_key[1],
                "source_ses_idx": session_key[1],
                "session_date": record.get("session_date"),
                "curriculum_name": record.get("curriculum_name"),
                "reward": _copy_reward_count_tree(record["reward"]),
                "reward_and_run_length": _copy_reward_run_count_tree(
                    record["reward_and_run_length"]
                ),
            }
        )
    return direct_records


def _extract_subject_metric_counts(row: Mapping[str, Any]) -> dict[str, Any]:
    choices, rewards = _aligned_choice_reward_histories(
        row.get("choice_history", []),
        row.get("reward_history", []),
    )
    switch_indices = [int(v) for v in row.get("switch_indices", [])]
    run_lengths = [int(v) for v in row.get("run_lengths", [])]
    reward_counts = _new_reward_count_tree()
    reward_run_counts = _new_reward_run_count_tree()

    for position, switch_idx in enumerate(switch_indices):
        if (
            switch_idx <= 0
            or switch_idx >= len(rewards)
            or switch_idx + 1 >= len(choices)
            or position >= len(run_lengths)
        ):
            continue

        reward_condition = "rewarded" if rewards[switch_idx] > 0 else "unrewarded"
        run_condition = "run_length_1" if run_lengths[position] == 1 else "run_length_gt1"
        post_switch = float(choices[switch_idx + 1] != choices[switch_idx])
        reward_counts[reward_condition]["successes"] += post_switch
        reward_counts[reward_condition]["n"] += 1.0
        reward_run_counts[reward_condition][run_condition]["successes"] += post_switch
        reward_run_counts[reward_condition][run_condition]["n"] += 1.0

    return {
        "subject_id": _normalize_identifier(row.get("subject_id")),
        "ses_idx": str(row.get("ses_idx")),
        "source_ses_idx": row.get("source_ses_idx"),
        "session_date": row.get("session_date"),
        "curriculum_name": row.get("curriculum_name"),
        "reward": reward_counts,
        "reward_and_run_length": reward_run_counts,
    }


def _compute_session_level_comparison(
    animal_sessions,
    simulated_sessions,
) -> dict[str, Any]:
    subject_curriculum_map = _resolve_subject_curriculum_map(
        animal_sessions,
        fallback_rows=simulated_sessions,
    )
    animal_session_stats = _build_session_level_metric_summary(
        animal_sessions,
        average_rollouts_by_source=False,
        subject_curriculum_map=subject_curriculum_map,
    )
    simulated_session_stats = _build_session_level_metric_summary(
        simulated_sessions,
        average_rollouts_by_source=True,
        subject_curriculum_map=subject_curriculum_map,
    )
    matched_session_keys = [
        session_key
        for session_key in animal_session_stats.keys()
        if session_key in simulated_session_stats
    ]

    comparison = {
        "post_switch_by_reward": {},
        "post_switch_by_reward_and_run_length": {},
    }
    for reward_condition in _REWARD_CONDITIONS:
        points = _build_session_comparison_points(
            animal_session_stats,
            simulated_session_stats,
            matched_session_keys=matched_session_keys,
            reward_condition=reward_condition,
        )
        comparison["post_switch_by_reward"][reward_condition] = {
            "points": points,
            "summary": _summarize_session_comparison_points(points),
        }

    for reward_condition in _REWARD_CONDITIONS:
        comparison["post_switch_by_reward_and_run_length"][reward_condition] = {}
        for run_condition in _RUN_LENGTH_CONDITIONS:
            points = _build_session_comparison_points(
                animal_session_stats,
                simulated_session_stats,
                matched_session_keys=matched_session_keys,
                reward_condition=reward_condition,
                run_condition=run_condition,
            )
            comparison["post_switch_by_reward_and_run_length"][reward_condition][
                run_condition
            ] = {
                "points": points,
                "summary": _summarize_session_comparison_points(points),
            }

    return comparison


def _build_subject_comparison_points(
    animal_subject_stats: Mapping[Any, Mapping[str, Any]],
    simulated_subject_stats: Mapping[Any, Mapping[str, Any]],
    *,
    reward_condition: str,
    run_condition: str | None = None,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    animal_subject_ids = _unique_preserve_order(list(animal_subject_stats.keys()))

    for subject_id in animal_subject_ids:
        if subject_id not in simulated_subject_stats:
            continue
        animal_metrics = animal_subject_stats[subject_id]
        simulated_metrics = simulated_subject_stats[subject_id]
        if run_condition is None:
            animal_leaf = _as_dict(
                _as_dict(animal_metrics.get("post_switch_by_reward", {})).get(
                    reward_condition,
                    {},
                )
            )
            simulated_leaf = _as_dict(
                _as_dict(simulated_metrics.get("post_switch_by_reward", {})).get(
                    reward_condition,
                    {},
                )
            )
        else:
            animal_leaf = _as_dict(
                _as_dict(
                    _as_dict(
                        animal_metrics.get("post_switch_by_reward_and_run_length", {})
                    ).get(reward_condition, {})
                ).get(run_condition, {})
            )
            simulated_leaf = _as_dict(
                _as_dict(
                    _as_dict(
                        simulated_metrics.get("post_switch_by_reward_and_run_length", {})
                    ).get(reward_condition, {})
                ).get(run_condition, {})
            )

        points.append(
            {
                "subject_id": subject_id,
                "animal_probability": _coerce_probability(animal_leaf.get("probability")),
                "animal_n": _normalize_count_output(animal_leaf.get("n", 0)),
                "animal_n_sessions": int(animal_metrics.get("n_source_sessions", 0)),
                "animal_n_source_sessions": int(
                    animal_metrics.get("n_source_sessions", 0)
                ),
                "animal_ci_low": _coerce_probability(animal_leaf.get("ci_low")),
                "animal_ci_high": _coerce_probability(animal_leaf.get("ci_high")),
                "simulated_probability": _coerce_probability(
                    simulated_leaf.get("probability")
                ),
                "simulated_effective_n": _normalize_count_output(
                    simulated_leaf.get("n", 0)
                ),
                "simulated_n_source_sessions": int(
                    simulated_metrics.get("n_source_sessions", 0)
                ),
                "simulated_ci_low": _coerce_probability(simulated_leaf.get("ci_low")),
                "simulated_ci_high": _coerce_probability(simulated_leaf.get("ci_high")),
                "curriculum_name": str(
                    animal_metrics.get(
                        "curriculum_name",
                        simulated_metrics.get("curriculum_name", "Unknown"),
                    )
                ),
                "delta_probability": _finite_difference(
                    _coerce_probability(simulated_leaf.get("probability")),
                    _coerce_probability(animal_leaf.get("probability")),
                ),
            }
        )

    return points


def _summarize_subject_comparison_points(
    points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _summarize_group_comparison_points(
        points,
        selector=_select_valid_subject_points,
        count_label="n_subjects",
    )


def _select_valid_subject_points(
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    valid_points = []
    for point in points:
        animal_probability = _coerce_probability(point.get("animal_probability"))
        simulated_probability = _coerce_probability(point.get("simulated_probability"))
        animal_n = point.get("animal_n")
        if animal_probability is None or simulated_probability is None:
            continue
        if animal_n is None or float(animal_n) < float(_SUBJECT_LEVEL_MIN_ANIMAL_N):
            continue
        valid_points.append(dict(point))
    return valid_points


def _build_session_comparison_points(
    animal_session_stats: Mapping[tuple[Any, str], Mapping[str, Any]],
    simulated_session_stats: Mapping[tuple[Any, str], Mapping[str, Any]],
    *,
    matched_session_keys: Sequence[tuple[Any, str]],
    reward_condition: str,
    run_condition: str | None = None,
) -> list[dict[str, Any]]:
    points = []
    for session_key in matched_session_keys:
        animal_metrics = _as_dict(animal_session_stats.get(session_key, {}))
        simulated_metrics = _as_dict(simulated_session_stats.get(session_key, {}))
        if run_condition is None:
            animal_leaf = _as_dict(
                _as_dict(animal_metrics.get("post_switch_by_reward", {})).get(
                    reward_condition,
                    {},
                )
            )
            simulated_leaf = _as_dict(
                _as_dict(simulated_metrics.get("post_switch_by_reward", {})).get(
                    reward_condition,
                    {},
                )
            )
        else:
            animal_leaf = _as_dict(
                _as_dict(
                    _as_dict(
                        animal_metrics.get("post_switch_by_reward_and_run_length", {})
                    ).get(reward_condition, {})
                ).get(run_condition, {})
            )
            simulated_leaf = _as_dict(
                _as_dict(
                    _as_dict(
                        simulated_metrics.get("post_switch_by_reward_and_run_length", {})
                    ).get(reward_condition, {})
                ).get(run_condition, {})
            )

        points.append(
            {
                "subject_id": animal_metrics.get("subject_id", session_key[0]),
                "session_id": animal_metrics.get("session_id", session_key[1]),
                "source_ses_idx": animal_metrics.get("source_ses_idx", session_key[1]),
                "session_date": animal_metrics.get(
                    "session_date",
                    simulated_metrics.get("session_date"),
                ),
                "curriculum_name": str(
                    animal_metrics.get("curriculum_name")
                    or simulated_metrics.get("curriculum_name")
                    or "Unknown"
                ),
                "animal_probability": _coerce_probability(animal_leaf.get("probability")),
                "animal_n": _normalize_count_output(animal_leaf.get("n", 0)),
                "animal_ci_low": _coerce_probability(animal_leaf.get("ci_low")),
                "animal_ci_high": _coerce_probability(animal_leaf.get("ci_high")),
                "simulated_probability": _coerce_probability(
                    simulated_leaf.get("probability")
                ),
                "simulated_effective_n": _normalize_count_output(
                    simulated_leaf.get("n", 0)
                ),
                "simulated_ci_low": _coerce_probability(simulated_leaf.get("ci_low")),
                "simulated_ci_high": _coerce_probability(simulated_leaf.get("ci_high")),
                "delta_probability": _finite_difference(
                    _coerce_probability(simulated_leaf.get("probability")),
                    _coerce_probability(animal_leaf.get("probability")),
                ),
            }
        )
    return points


def _select_valid_session_points(
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    valid_points = []
    for point in points:
        animal_probability = _coerce_probability(point.get("animal_probability"))
        simulated_probability = _coerce_probability(point.get("simulated_probability"))
        animal_n = point.get("animal_n")
        if animal_probability is None or simulated_probability is None:
            continue
        if animal_n is None or float(animal_n) < float(_SUBJECT_LEVEL_MIN_ANIMAL_N):
            continue
        valid_points.append(dict(point))
    return valid_points


def _summarize_session_comparison_points(
    points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _summarize_group_comparison_points(
        points,
        selector=_select_valid_session_points,
        count_label="n_sessions",
    )


def _summarize_group_comparison_points(
    points: Sequence[Mapping[str, Any]],
    *,
    selector,
    count_label: str,
) -> dict[str, Any]:
    valid_points = list(selector(points))
    xs = [float(point["animal_probability"]) for point in valid_points]
    ys = [float(point["simulated_probability"]) for point in valid_points]

    return {
        count_label: len(valid_points),
        "correlation": _pearson_correlation(xs, ys),
        "rmse": _rmse(xs, ys),
        "bias": _mean_difference(xs, ys),
    }


def _new_reward_count_tree() -> dict[str, dict[str, float]]:
    return {
        reward_condition: {"successes": 0.0, "n": 0.0}
        for reward_condition in _REWARD_CONDITIONS
    }


def _new_reward_run_count_tree() -> dict[str, dict[str, dict[str, float]]]:
    return {
        reward_condition: {
            run_condition: {"successes": 0.0, "n": 0.0}
            for run_condition in _RUN_LENGTH_CONDITIONS
        }
        for reward_condition in _REWARD_CONDITIONS
    }


def _copy_reward_count_tree(
    counts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    copied = _new_reward_count_tree()
    _accumulate_reward_counts(copied, counts)
    return copied


def _copy_reward_run_count_tree(
    counts: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, float]]]:
    copied = _new_reward_run_count_tree()
    _accumulate_reward_run_counts(copied, counts)
    return copied


def _accumulate_reward_counts(
    target: dict[str, dict[str, float]],
    source: Mapping[str, Mapping[str, Any]],
) -> None:
    for reward_condition in _REWARD_CONDITIONS:
        source_leaf = _as_dict(source.get(reward_condition, {}))
        target[reward_condition]["successes"] += float(source_leaf.get("successes", 0.0))
        target[reward_condition]["n"] += float(source_leaf.get("n", 0.0))


def _accumulate_reward_run_counts(
    target: dict[str, dict[str, dict[str, float]]],
    source: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    for reward_condition in _REWARD_CONDITIONS:
        source_reward_group = _as_dict(source.get(reward_condition, {}))
        for run_condition in _RUN_LENGTH_CONDITIONS:
            source_leaf = _as_dict(source_reward_group.get(run_condition, {}))
            target[reward_condition][run_condition]["successes"] += float(
                source_leaf.get("successes", 0.0)
            )
            target[reward_condition][run_condition]["n"] += float(
                source_leaf.get("n", 0.0)
            )


def _probability_from_counts(counts: Mapping[str, Any]) -> float:
    total = float(counts.get("n", 0.0))
    if total <= 0:
        return math.nan
    return float(float(counts.get("successes", 0.0)) / total)


def _normalize_count_output(value: Any) -> int | float:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return int(round(numeric))
    return numeric


def _compute_switch_summary(session_rows, window_size: int) -> dict[str, Any]:
    pooled_events = compute_switch_probabilities_around_switches(session_rows, window_size)
    reward_conditioned = compute_conditional_switch_probabilities_by_reward(
        session_rows,
        window_size,
    )
    reward_and_run_length = compute_conditional_switch_probabilities_by_reward_and_run_length(
        session_rows,
        window_size,
    )
    return {
        "num_sessions": len(list(_iter_session_records(session_rows))),
        "pooled_switch_probability": create_pooled_switch_analysis(pooled_events),
        "post_switch_by_reward": analyze_post_switch_probability_by_reward(
            reward_conditioned
        ),
        "post_switch_by_reward_and_run_length": (
            analyze_post_switch_probability_by_reward_and_run_length(
                reward_and_run_length
            )
        ),
    }


def _compare_switch_summaries(
    animal_summary: Mapping[str, Any],
    simulated_summary: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = {
        "post_switch_by_reward": {},
        "post_switch_by_reward_and_run_length": {},
    }

    animal_reward = _as_dict(animal_summary.get("post_switch_by_reward", {}))
    simulated_reward = _as_dict(simulated_summary.get("post_switch_by_reward", {}))
    for condition in sorted(set(animal_reward) | set(simulated_reward)):
        animal_prob = _coerce_probability(animal_reward.get(condition, {}).get("probability"))
        simulated_prob = _coerce_probability(
            simulated_reward.get(condition, {}).get("probability")
        )
        comparison["post_switch_by_reward"][condition] = {
            "animal_probability": animal_prob,
            "simulated_probability": simulated_prob,
            "delta_probability": _finite_difference(simulated_prob, animal_prob),
        }

    animal_nested = _as_dict(animal_summary.get("post_switch_by_reward_and_run_length", {}))
    simulated_nested = _as_dict(
        simulated_summary.get("post_switch_by_reward_and_run_length", {})
    )
    reward_conditions = sorted(set(animal_nested) | set(simulated_nested))
    for reward_condition in reward_conditions:
        comparison["post_switch_by_reward_and_run_length"][reward_condition] = {}
        animal_groups = _as_dict(animal_nested.get(reward_condition, {}))
        simulated_groups = _as_dict(simulated_nested.get(reward_condition, {}))
        for run_condition in sorted(set(animal_groups) | set(simulated_groups)):
            animal_prob = _coerce_probability(
                animal_groups.get(run_condition, {}).get("probability")
            )
            simulated_prob = _coerce_probability(
                simulated_groups.get(run_condition, {}).get("probability")
            )
            comparison["post_switch_by_reward_and_run_length"][reward_condition][
                run_condition
            ] = {
                "animal_probability": animal_prob,
                "simulated_probability": simulated_prob,
                "delta_probability": _finite_difference(simulated_prob, animal_prob),
            }

    return comparison


def _build_switch_subject_aggregate(
    subject_level: Mapping[str, Any],
) -> dict[str, Any]:
    reward_level = _as_dict(subject_level.get("post_switch_by_reward", {}))
    reward_run_level = _as_dict(
        subject_level.get("post_switch_by_reward_and_run_length", {})
    )
    return {
        "post_switch_by_reward": {
            reward_condition: _build_subject_aggregate_leaf(
                list(_as_dict(reward_level.get(reward_condition, {})).get("points", [])),
                selector=_select_valid_subject_points,
            )
            for reward_condition in _REWARD_CONDITIONS
        },
        "post_switch_by_reward_and_run_length": {
            reward_condition: {
                run_condition: _build_subject_aggregate_leaf(
                    list(
                        _as_dict(
                            _as_dict(reward_run_level.get(reward_condition, {})).get(
                                run_condition,
                                {},
                            )
                        ).get("points", [])
                    ),
                    selector=_select_valid_subject_points,
                )
                for run_condition in _RUN_LENGTH_CONDITIONS
            }
            for reward_condition in _REWARD_CONDITIONS
        },
    }


def _build_switch_session_aggregate(
    session_level: Mapping[str, Any],
) -> dict[str, Any]:
    reward_level = _as_dict(session_level.get("post_switch_by_reward", {}))
    reward_run_level = _as_dict(
        session_level.get("post_switch_by_reward_and_run_length", {})
    )
    return {
        "post_switch_by_reward": {
            reward_condition: _build_session_aggregate_leaf(
                list(_as_dict(reward_level.get(reward_condition, {})).get("points", [])),
                selector=_select_valid_session_points,
            )
            for reward_condition in _REWARD_CONDITIONS
        },
        "post_switch_by_reward_and_run_length": {
            reward_condition: {
                run_condition: _build_session_aggregate_leaf(
                    list(
                        _as_dict(
                            _as_dict(reward_run_level.get(reward_condition, {})).get(
                                run_condition,
                                {},
                            )
                        ).get("points", [])
                    ),
                    selector=_select_valid_session_points,
                )
                for run_condition in _RUN_LENGTH_CONDITIONS
            }
            for reward_condition in _REWARD_CONDITIONS
        },
    }


def _build_history_subject_aggregate(
    subject_level: Mapping[str, Any],
    *,
    max_trials_back: int,
    subject_min_trials: int,
) -> dict[str, dict[int, dict[str, Any]]]:
    subject_aggregate: dict[str, dict[int, dict[str, Any]]] = {
        pattern_type: {}
        for pattern_type in _HISTORY_PATTERN_TYPES
    }
    for pattern_type in _HISTORY_PATTERN_TYPES:
        pattern_group = _as_dict(subject_level.get(pattern_type, {}))
        for n_back in range(1, int(max_trials_back) + 1):
            panel_group = _as_dict(pattern_group.get(n_back, {}))
            rows = []
            for pattern in sorted(panel_group):
                aggregate_row = _build_subject_aggregate_leaf(
                    list(_as_dict(panel_group.get(pattern, {})).get("points", [])),
                    selector=lambda points, min_trials=subject_min_trials: (
                        _select_valid_subject_history_pattern_points(
                            points,
                            min_trials=min_trials,
                        )
                    ),
                )
                aggregate_row["pattern"] = str(pattern)
                rows.append(aggregate_row)
            subject_aggregate[pattern_type][n_back] = {
                "rows": rows,
                "summary": _summarize_probability_rows(
                    rows,
                    weight_key="n_subjects",
                ),
            }
    return subject_aggregate


def _build_history_session_aggregate(
    session_level: Mapping[str, Any],
    *,
    max_trials_back: int,
    session_min_trials: int,
) -> dict[str, dict[int, dict[str, Any]]]:
    session_aggregate: dict[str, dict[int, dict[str, Any]]] = {
        pattern_type: {}
        for pattern_type in _HISTORY_PATTERN_TYPES
    }
    for pattern_type in _HISTORY_PATTERN_TYPES:
        pattern_group = _as_dict(session_level.get(pattern_type, {}))
        for n_back in range(1, int(max_trials_back) + 1):
            panel_group = _as_dict(pattern_group.get(n_back, {}))
            rows = []
            for pattern in sorted(panel_group):
                aggregate_row = _build_session_aggregate_leaf(
                    list(_as_dict(panel_group.get(pattern, {})).get("points", [])),
                    selector=lambda points, min_trials=session_min_trials: (
                        _select_valid_session_history_pattern_points(
                            points,
                            min_trials=min_trials,
                        )
                    ),
                )
                aggregate_row["pattern"] = str(pattern)
                rows.append(aggregate_row)
            session_aggregate[pattern_type][n_back] = {
                "rows": rows,
                "summary": _summarize_probability_rows(
                    rows,
                    weight_key="n_sessions",
                ),
            }
    return session_aggregate


def _build_subject_aggregate_leaf(
    points: Sequence[Mapping[str, Any]],
    *,
    selector,
) -> dict[str, Any]:
    return _build_group_aggregate_leaf(
        points,
        selector=selector,
        count_label="n_subjects",
    )


def _build_session_aggregate_leaf(
    points: Sequence[Mapping[str, Any]],
    *,
    selector,
) -> dict[str, Any]:
    return _build_group_aggregate_leaf(
        points,
        selector=selector,
        count_label="n_sessions",
    )


def _build_group_aggregate_leaf(
    points: Sequence[Mapping[str, Any]],
    *,
    selector,
    count_label: str,
) -> dict[str, Any]:
    valid_points = list(selector(points))
    animal_values = [float(point["animal_probability"]) for point in valid_points]
    simulated_values = [float(point["simulated_probability"]) for point in valid_points]
    delta_values = [float(point["delta_probability"]) for point in valid_points]
    return {
        "animal_mean": _mean(animal_values) if animal_values else math.nan,
        "animal_sem": _sem(animal_values) if animal_values else math.nan,
        "simulated_mean": _mean(simulated_values) if simulated_values else math.nan,
        "simulated_sem": _sem(simulated_values) if simulated_values else math.nan,
        "delta_mean": _mean(delta_values) if delta_values else math.nan,
        "delta_sem": _sem(delta_values) if delta_values else math.nan,
        "delta_median": _median(delta_values) if delta_values else math.nan,
        "delta_iqr": _iqr(delta_values) if delta_values else math.nan,
        count_label: len(valid_points),
    }


def _build_switch_quantitative_summary(
    *,
    animal_summary: Mapping[str, Any],
    simulated_summary: Mapping[str, Any],
    subject_aggregate: Mapping[str, Any],
    session_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    pooled_reward_rows = _build_switch_pooled_reward_rows(
        animal_summary=animal_summary,
        simulated_summary=simulated_summary,
    )
    pooled_reward_run_rows = _build_switch_pooled_reward_run_rows(
        animal_summary=animal_summary,
        simulated_summary=simulated_summary,
    )
    subject_reward_rows = _build_switch_subject_reward_rows(subject_aggregate)
    subject_reward_run_rows = _build_switch_subject_reward_run_rows(subject_aggregate)
    session_reward_rows = _build_switch_session_reward_rows(session_aggregate)
    session_reward_run_rows = _build_switch_session_reward_run_rows(session_aggregate)
    return {
        "pooled": {
            "post_switch_by_reward": _summarize_probability_rows(
                pooled_reward_rows,
                weight_key="effective_weight",
            ),
            "post_switch_by_reward_and_run_length": _summarize_probability_rows(
                pooled_reward_run_rows,
                weight_key="effective_weight",
            ),
            "overall": _summarize_probability_rows(
                [*pooled_reward_rows, *pooled_reward_run_rows],
                weight_key="effective_weight",
            ),
        },
        "subject_mean": {
            "post_switch_by_reward": _summarize_probability_rows(
                subject_reward_rows,
                weight_key="n_subjects",
            ),
            "post_switch_by_reward_and_run_length": _summarize_probability_rows(
                subject_reward_run_rows,
                weight_key="n_subjects",
            ),
            "overall": _summarize_probability_rows(
                [*subject_reward_rows, *subject_reward_run_rows],
                weight_key="n_subjects",
            ),
        },
        "session_mean": {
            "post_switch_by_reward": _summarize_probability_rows(
                session_reward_rows,
                weight_key="n_sessions",
            ),
            "post_switch_by_reward_and_run_length": _summarize_probability_rows(
                session_reward_run_rows,
                weight_key="n_sessions",
            ),
            "overall": _summarize_probability_rows(
                [*session_reward_rows, *session_reward_run_rows],
                weight_key="n_sessions",
            ),
        },
    }


def _build_history_quantitative_summary(
    *,
    comparison: Mapping[str, Any],
    subject_aggregate: Mapping[str, Any],
    session_aggregate: Mapping[str, Any],
    max_trials_back: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"pooled": {}, "subject_mean": {}, "session_mean": {}}
    pooled_overall_rows: list[dict[str, Any]] = []
    subject_overall_rows: list[dict[str, Any]] = []
    session_overall_rows: list[dict[str, Any]] = []
    for pattern_type in _HISTORY_PATTERN_TYPES:
        pooled_group = _as_dict(comparison.get(pattern_type, {}))
        subject_group = _as_dict(subject_aggregate.get(pattern_type, {}))
        session_group = _as_dict(session_aggregate.get(pattern_type, {}))
        pattern_pooled_summary: dict[str, Any] = {}
        pattern_subject_summary: dict[str, Any] = {}
        pattern_session_summary: dict[str, Any] = {}
        pattern_pooled_rows: list[dict[str, Any]] = []
        pattern_subject_rows: list[dict[str, Any]] = []
        pattern_session_rows: list[dict[str, Any]] = []
        for n_back in range(1, int(max_trials_back) + 1):
            pooled_rows = list(_as_dict(pooled_group.get(n_back, {})).get("rows", []))
            subject_rows = list(_as_dict(subject_group.get(n_back, {})).get("rows", []))
            session_rows = list(_as_dict(session_group.get(n_back, {})).get("rows", []))
            pattern_pooled_summary[n_back] = _summarize_probability_rows(
                pooled_rows,
                weight_key="effective_weight",
            )
            pattern_subject_summary[n_back] = _summarize_probability_rows(
                subject_rows,
                weight_key="n_subjects",
            )
            pattern_session_summary[n_back] = _summarize_probability_rows(
                session_rows,
                weight_key="n_sessions",
            )
            pattern_pooled_rows.extend(pooled_rows)
            pattern_subject_rows.extend(subject_rows)
            pattern_session_rows.extend(session_rows)
            pooled_overall_rows.extend(pooled_rows)
            subject_overall_rows.extend(subject_rows)
            session_overall_rows.extend(session_rows)
        pattern_pooled_summary["overall"] = _summarize_probability_rows(
            pattern_pooled_rows,
            weight_key="effective_weight",
        )
        pattern_subject_summary["overall"] = _summarize_probability_rows(
            pattern_subject_rows,
            weight_key="n_subjects",
        )
        pattern_session_summary["overall"] = _summarize_probability_rows(
            pattern_session_rows,
            weight_key="n_sessions",
        )
        summary["pooled"][pattern_type] = pattern_pooled_summary
        summary["subject_mean"][pattern_type] = pattern_subject_summary
        summary["session_mean"][pattern_type] = pattern_session_summary
    summary["pooled"]["overall"] = _summarize_probability_rows(
        pooled_overall_rows,
        weight_key="effective_weight",
    )
    summary["subject_mean"]["overall"] = _summarize_probability_rows(
        subject_overall_rows,
        weight_key="n_subjects",
    )
    summary["session_mean"]["overall"] = _summarize_probability_rows(
        session_overall_rows,
        weight_key="n_sessions",
    )
    return summary


def _build_switch_delta_significance_summary(
    subject_level: Mapping[str, Any],
) -> dict[str, Any]:
    reward_stats = _as_dict(subject_level.get("post_switch_by_reward", {}))
    reward_rows = [
        {
            "key": "rewarded",
            "label": "Rewarded",
            "points": _select_valid_subject_points(
                list(_as_dict(reward_stats.get("rewarded", {})).get("points", []))
            ),
        },
        {
            "key": "unrewarded",
            "label": "Unrewarded",
            "points": _select_valid_subject_points(
                list(_as_dict(reward_stats.get("unrewarded", {})).get("points", []))
            ),
        },
    ]
    reward_run_stats = _as_dict(
        subject_level.get("post_switch_by_reward_and_run_length", {})
    )
    reward_run_rows = []
    for reward_condition, run_condition, label in (
        ("rewarded", "run_length_1", "Rewarded / Run=1"),
        ("rewarded", "run_length_gt1", "Rewarded / Run>1"),
        ("unrewarded", "run_length_1", "Unrewarded / Run=1"),
        ("unrewarded", "run_length_gt1", "Unrewarded / Run>1"),
    ):
        reward_run_rows.append(
            {
                "key": f"{reward_condition}:{run_condition}",
                "label": label,
                "points": _select_valid_subject_points(
                    list(
                        _as_dict(
                            _as_dict(reward_run_stats.get(reward_condition, {})).get(
                                run_condition,
                                {},
                            )
                        ).get("points", [])
                    )
                ),
            }
        )
    return {
        "test_name": "wilcoxon_signed_rank_two_sided",
        "post_switch_by_reward": _build_delta_condition_summary(
            reward_rows,
            animal_count_key="animal_n",
        ),
        "post_switch_by_reward_and_run_length": _build_delta_condition_summary(
            reward_run_rows,
            animal_count_key="animal_n",
        ),
    }


def _build_history_delta_significance_summary(
    subject_level: Mapping[str, Any],
    *,
    max_trials_back: int,
    subject_min_trials: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"test_name": "wilcoxon_signed_rank_two_sided"}
    for pattern_type in _HISTORY_PATTERN_TYPES:
        pattern_group = _as_dict(subject_level.get(pattern_type, {}))
        pattern_summary: dict[int, Any] = {}
        for n_back in range(1, int(max_trials_back) + 1):
            panel_group = _as_dict(pattern_group.get(n_back, {}))
            rows = _build_sorted_history_delta_rows(
                panel_group,
                min_trials=subject_min_trials,
            )
            pattern_summary[n_back] = _build_delta_condition_summary(
                rows,
                animal_count_key="animal_total",
            )
        summary[pattern_type] = pattern_summary
    return summary


def _build_delta_condition_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    animal_count_key: str,
) -> dict[str, Any]:
    condition_summaries: list[dict[str, Any]] = []
    significant_condition_medians: list[float] = []
    significant_condition_labels: list[str] = []
    all_subject_condition_deltas: list[float] = []
    subject_to_condition_deltas: dict[str, list[float]] = {}
    condition_mean_deltas: list[float] = []

    for row in rows:
        valid_points = list(row.get("points", []))
        deltas = _extract_point_delta_probabilities(valid_points)
        all_subject_condition_deltas.extend(deltas)
        test_result = _wilcoxon_signed_rank_against_zero(deltas)
        p_value = test_result.get("p_value")
        is_significant = p_value is not None and float(p_value) < 0.05
        animal_trial_count = float(
            sum(
                float(point.get(animal_count_key, 0.0) or 0.0)
                for point in valid_points
                if point.get(animal_count_key) is not None
            )
        )
        median_delta = _median(deltas) if deltas else None
        mean_delta = _mean(deltas) if deltas else None
        condition_summary = {
            "key": row.get("key", row.get("label")),
            "label": row.get("label"),
            "n_subjects": len(valid_points),
            "n_nonzero_subjects": int(test_result.get("n_nonzero", 0) or 0),
            "p_value": p_value,
            "is_significant": bool(is_significant),
            "median_delta_probability": median_delta,
            "mean_delta_probability": mean_delta,
            "animal_trial_count": animal_trial_count,
        }
        condition_summaries.append(condition_summary)
        if mean_delta is not None:
            condition_mean_deltas.append(float(mean_delta))
        if is_significant and median_delta is not None:
            significant_condition_medians.append(float(median_delta))
            significant_condition_labels.append(str(row.get("label")))
        for point_index, point in enumerate(valid_points):
            delta_probability = _extract_point_delta_probability(point)
            if delta_probability is None:
                continue
            subject_key = str(
                point.get("subject_id", f"__row_{row.get('label')}__point_{point_index}")
            )
            subject_to_condition_deltas.setdefault(subject_key, []).append(
                float(delta_probability)
            )

    subject_condition_error_summary = _summarize_delta_error_values(
        all_subject_condition_deltas,
        n_label="n_subject_condition_pairs",
        n_nonzero_label="n_nonzero_subject_condition_pairs",
        definition=(
            "Computed by flattening all valid subject-condition delta probabilities "
            "within the plotted analysis family."
        ),
    )
    subject_balanced_delta_values = [
        _mean(deltas)
        for deltas in subject_to_condition_deltas.values()
        if deltas
    ]
    subject_balanced_error_summary = _summarize_delta_error_values(
        subject_balanced_delta_values,
        n_label="n_subjects",
        n_nonzero_label="n_nonzero_subjects",
        definition=(
            "Computed by averaging delta probabilities within each subject across "
            "that plot family's valid conditions, then summarizing across subjects."
        ),
    )
    condition_balanced_error_summary = _summarize_delta_error_values(
        condition_mean_deltas,
        n_label="n_conditions",
        n_nonzero_label="n_nonzero_conditions",
        definition=(
            "Computed by averaging delta probabilities across subjects within each "
            "condition, then summarizing across conditions."
        ),
    )
    significant_summary = {
        "n_significant_conditions": len(significant_condition_medians),
        "condition_labels": significant_condition_labels,
        "average_of_significant_condition_medians": (
            _mean(significant_condition_medians)
            if significant_condition_medians
            else None
        ),
    }
    return {
        "test_name": "wilcoxon_signed_rank_two_sided",
        "conditions": condition_summaries,
        "subject_condition_error_summary": subject_condition_error_summary,
        "subject_balanced_error_summary": subject_balanced_error_summary,
        "condition_balanced_error_summary": condition_balanced_error_summary,
        "significant_conditions_summary": significant_summary,
    }


def _summarize_delta_error_values(
    delta_values: Sequence[float],
    *,
    n_label: str,
    n_nonzero_label: str,
    definition: str,
) -> dict[str, Any]:
    overall_test_result = _wilcoxon_signed_rank_against_zero(delta_values)
    return {
        n_label: len(delta_values),
        n_nonzero_label: int(overall_test_result.get("n_nonzero", 0) or 0),
        "mean_signed_error": _mean(delta_values) if delta_values else None,
        "mean_signed_error_sem": _sem(delta_values) if delta_values else None,
        "p_value": overall_test_result.get("p_value"),
        "mean_absolute_error": (
            _mean([abs(delta) for delta in delta_values]) if delta_values else None
        ),
        "mean_squared_error": (
            _mean([delta**2 for delta in delta_values]) if delta_values else None
        ),
        "test_name": "wilcoxon_signed_rank_two_sided",
        "definition": definition,
    }


def _build_switch_pooled_reward_rows(
    *,
    animal_summary: Mapping[str, Any],
    simulated_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    animal_reward = _as_dict(animal_summary.get("post_switch_by_reward", {}))
    simulated_reward = _as_dict(simulated_summary.get("post_switch_by_reward", {}))
    rows = []
    for reward_condition in _REWARD_CONDITIONS:
        animal_leaf = _as_dict(animal_reward.get(reward_condition, {}))
        simulated_leaf = _as_dict(simulated_reward.get(reward_condition, {}))
        animal_n = animal_leaf.get("n")
        simulated_n = simulated_leaf.get("n")
        effective_weight = None
        if animal_n is not None and simulated_n is not None:
            effective_weight = min(float(animal_n), float(simulated_n))
        rows.append(
            {
                "condition": reward_condition,
                "animal_probability": _coerce_probability(animal_leaf.get("probability")),
                "simulated_probability": _coerce_probability(
                    simulated_leaf.get("probability")
                ),
                "delta_probability": _finite_difference(
                    _coerce_probability(simulated_leaf.get("probability")),
                    _coerce_probability(animal_leaf.get("probability")),
                ),
                "effective_weight": effective_weight,
            }
        )
    return rows


def _build_switch_pooled_reward_run_rows(
    *,
    animal_summary: Mapping[str, Any],
    simulated_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    animal_reward_run = _as_dict(
        animal_summary.get("post_switch_by_reward_and_run_length", {})
    )
    simulated_reward_run = _as_dict(
        simulated_summary.get("post_switch_by_reward_and_run_length", {})
    )
    rows = []
    for reward_condition in _REWARD_CONDITIONS:
        for run_condition in _RUN_LENGTH_CONDITIONS:
            animal_leaf = _as_dict(
                _as_dict(animal_reward_run.get(reward_condition, {})).get(
                    run_condition,
                    {},
                )
            )
            simulated_leaf = _as_dict(
                _as_dict(simulated_reward_run.get(reward_condition, {})).get(
                    run_condition,
                    {},
                )
            )
            animal_n = animal_leaf.get("n")
            simulated_n = simulated_leaf.get("n")
            effective_weight = None
            if animal_n is not None and simulated_n is not None:
                effective_weight = min(float(animal_n), float(simulated_n))
            rows.append(
                {
                    "reward_condition": reward_condition,
                    "run_condition": run_condition,
                    "animal_probability": _coerce_probability(
                        animal_leaf.get("probability")
                    ),
                    "simulated_probability": _coerce_probability(
                        simulated_leaf.get("probability")
                    ),
                    "delta_probability": _finite_difference(
                        _coerce_probability(simulated_leaf.get("probability")),
                        _coerce_probability(animal_leaf.get("probability")),
                    ),
                    "effective_weight": effective_weight,
                }
            )
    return rows


def _build_switch_subject_reward_rows(
    subject_aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reward_group = _as_dict(subject_aggregate.get("post_switch_by_reward", {}))
    return [
        {"condition": reward_condition, **_as_dict(reward_group.get(reward_condition, {}))}
        for reward_condition in _REWARD_CONDITIONS
    ]


def _build_switch_session_reward_rows(
    session_aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reward_group = _as_dict(session_aggregate.get("post_switch_by_reward", {}))
    return [
        {"condition": reward_condition, **_as_dict(reward_group.get(reward_condition, {}))}
        for reward_condition in _REWARD_CONDITIONS
    ]


def _build_switch_subject_reward_run_rows(
    subject_aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reward_run_group = _as_dict(
        subject_aggregate.get("post_switch_by_reward_and_run_length", {})
    )
    rows = []
    for reward_condition in _REWARD_CONDITIONS:
        for run_condition in _RUN_LENGTH_CONDITIONS:
            rows.append(
                {
                    "reward_condition": reward_condition,
                    "run_condition": run_condition,
                    **_as_dict(
                        _as_dict(reward_run_group.get(reward_condition, {})).get(
                            run_condition,
                            {},
                        )
                    ),
                }
            )
    return rows


def _build_switch_session_reward_run_rows(
    session_aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reward_run_group = _as_dict(
        session_aggregate.get("post_switch_by_reward_and_run_length", {})
    )
    rows = []
    for reward_condition in _REWARD_CONDITIONS:
        for run_condition in _RUN_LENGTH_CONDITIONS:
            rows.append(
                {
                    "reward_condition": reward_condition,
                    "run_condition": run_condition,
                    **_as_dict(
                        _as_dict(reward_run_group.get(reward_condition, {})).get(
                            run_condition,
                            {},
                        )
                    ),
                }
            )
    return rows


def _summarize_probability_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    weight_key: str,
) -> dict[str, Any]:
    valid_rows = []
    for row in rows:
        animal_probability = _coerce_probability(
            row.get("animal_probability", row.get("animal_mean"))
        )
        simulated_probability = _coerce_probability(
            row.get("simulated_probability", row.get("simulated_mean"))
        )
        if animal_probability is None or simulated_probability is None:
            continue
        normalized_row = dict(row)
        normalized_row["animal_probability"] = animal_probability
        normalized_row["simulated_probability"] = simulated_probability
        if normalized_row.get("delta_probability") is None:
            normalized_row["delta_probability"] = _finite_difference(
                simulated_probability,
                animal_probability,
            )
        if weight_key == "effective_weight" and normalized_row.get(weight_key) is None:
            animal_total = normalized_row.get("animal_total")
            simulated_total = normalized_row.get("simulated_total_effective")
            if animal_total is not None and simulated_total is not None:
                normalized_row[weight_key] = min(
                    float(animal_total),
                    float(simulated_total),
                )
        valid_rows.append(normalized_row)

    xs = [float(row["animal_probability"]) for row in valid_rows]
    ys = [float(row["simulated_probability"]) for row in valid_rows]
    deltas = [float(row["delta_probability"]) for row in valid_rows]
    weights = []
    has_all_weights = True
    for row in valid_rows:
        weight_value = row.get(weight_key)
        if weight_value is None or float(weight_value) <= 0:
            has_all_weights = False
            continue
        weights.append(float(weight_value))
    return {
        "n_rows": len(valid_rows),
        "total_weight": sum(weights) if has_all_weights and weights else 0.0,
        "mae": _mean([abs(delta) for delta in deltas]) if deltas else None,
        "rmse": _rmse(xs, ys),
        "bias": _mean(deltas) if deltas else None,
        "correlation": _pearson_correlation(xs, ys),
        "weighted_mae": (
            _weighted_mean([abs(delta) for delta in deltas], weights)
            if has_all_weights and len(weights) == len(deltas)
            else None
        ),
        "weighted_rmse": (
            _weighted_rmse(xs, ys, weights)
            if has_all_weights and len(weights) == len(xs)
            else None
        ),
    }


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in {_TRAIN_SPLIT, _HELDOUT_SPLIT}:
        raise ValueError(f"Unsupported split={split!r}. Use 'train' or 'heldout'.")
    return normalized


def _normalize_checkpoint_policy(checkpoint_policy: str) -> str:
    normalized = str(checkpoint_policy).strip().lower()
    if _explicit_checkpoint_step(normalized) is not None:
        return normalized
    if normalized not in {
        _CHECKPOINT_POLICY_BEST_EVAL,
        _CHECKPOINT_POLICY_BEST_HELDOUT,
        _CHECKPOINT_POLICY_FINAL,
    }:
        raise ValueError(
            f"Unsupported checkpoint_policy={checkpoint_policy!r}. "
            "Use 'best_eval', 'best_heldout', 'final', or 'step_<N>'."
        )
    return normalized


def _explicit_checkpoint_step(checkpoint_policy: str) -> int | None:
    """Return N for an explicit ``step_<N>`` policy, else None."""
    prefix = "step_"
    if checkpoint_policy.startswith(prefix) and checkpoint_policy[len(prefix):].isdigit():
        return int(checkpoint_policy[len(prefix):])
    return None


def _resolve_split_selection(
    data_cfg: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    # The selection pipeline params (curricula / ratio / min_sessions /
    # heldout_every_n / seed) are shared by both splits; only the direct-override
    # subject id list differs (subject_ids for train, test_subject_ids for heldout).
    direct_ids = (
        data_cfg.get("subject_ids")
        if split == _TRAIN_SPLIT
        else data_cfg.get("test_subject_ids")
    )
    return {
        "subject_ids": direct_ids,
        "curricula": data_cfg.get("curricula"),
        "subject_ratio": data_cfg.get("subject_ratio"),
        "min_sessions": data_cfg.get("min_sessions", 10),
        "heldout_every_n": data_cfg.get("heldout_every_n", 5),
        "subject_sample_seed": data_cfg.get("subject_sample_seed", data_cfg.get("seed")),
        "snapshot": data_cfg.get("snapshot"),
    }


def _resolve_checkpoint_artifact(
    *,
    model_dir: Path,
    outputs_dir: Path,
    checkpoint_policy: str,
    checkpoint_index: Mapping[str, Any],
    output_summary: Mapping[str, Any],
) -> tuple[Path, int | None, str, str | None, str | None]:
    reason = None
    fallback_reason = None
    final_params_path = outputs_dir / "params.json"
    if checkpoint_policy == _CHECKPOINT_POLICY_FINAL:
        final_step = _infer_final_step(checkpoint_index, output_summary)
        return (
            final_params_path,
            final_step,
            "final",
            "Selected top-level final params.json.",
            None,
        )

    explicit_step = _explicit_checkpoint_step(checkpoint_policy)
    if explicit_step is not None:
        # An explicit step must exist: never fall back to another checkpoint.
        records = [
            item
            for item in list(_as_dict(checkpoint_index).get("checkpoints", []) or [])
            if _coerce_optional_int(item.get("step")) == explicit_step
        ]
        candidate = _resolve_artifact_path(
            model_dir=model_dir,
            outputs_dir=outputs_dir,
            reported_path=(
                records[0].get("params_path")
                if records
                else f"outputs/checkpoints/step_{explicit_step}/params.json"
            ),
        )
        if not candidate.exists():
            raise FileNotFoundError(
                f"checkpoint_policy={checkpoint_policy!r}: no params at {candidate}."
            )
        return (
            candidate,
            explicit_step,
            f"step_{explicit_step}",
            "Selected the explicitly requested checkpoint step.",
            None,
        )

    if checkpoint_policy == _CHECKPOINT_POLICY_BEST_EVAL:
        checkpoints = list(_as_dict(checkpoint_index).get("checkpoints", []) or [])
        if checkpoints:
            best_record = max(
                checkpoints,
                key=lambda item: float(item.get("eval_likelihood", float("-inf"))),
            )
            candidate = _resolve_artifact_path(
                model_dir=model_dir,
                outputs_dir=outputs_dir,
                reported_path=best_record.get("params_path"),
            )
            if candidate.exists():
                return (
                    candidate,
                    _coerce_optional_int(best_record.get("step")),
                    f"step_{best_record.get('step')}",
                    "Selected checkpoint with maximum eval_likelihood from "
                    "outputs/checkpoints/index.json.",
                    None,
                )
            fallback_reason = (
                "best_eval checkpoint params were referenced in "
                "outputs/checkpoints/index.json but the params file was missing; "
                "falling back to final params.json."
            )
        else:
            fallback_reason = (
                "outputs/checkpoints/index.json was missing or empty; "
                "falling back to final params.json."
            )

    if checkpoint_policy == _CHECKPOINT_POLICY_BEST_HELDOUT:
        heldout_records = list(
            _as_dict(output_summary).get("heldout_test_checkpoints", []) or []
        )
        if heldout_records:
            best_record = max(
                heldout_records,
                key=lambda item: float(
                    item.get("heldout_test_likelihood", float("-inf"))
                ),
            )
            candidate = _resolve_artifact_path(
                model_dir=model_dir,
                outputs_dir=outputs_dir,
                reported_path=best_record.get("params_path"),
            )
            if candidate.exists():
                return (
                    candidate,
                    _coerce_optional_int(best_record.get("step")),
                    f"step_{best_record.get('step')}",
                    "Selected checkpoint with maximum heldout_test_likelihood from "
                    "outputs/output_summary.json.",
                    None,
                )
            fallback_reason = (
                "best_heldout checkpoint params were referenced in "
                "outputs/output_summary.json but the params file was missing; "
                "falling back to final params.json."
            )
        else:
            fallback_reason = (
                "outputs/output_summary.json did not include heldout_test_checkpoints; "
                "falling back to final params.json."
            )

    final_step = _infer_final_step(checkpoint_index, output_summary)
    return (
        final_params_path,
        final_step,
        "final",
        reason,
        fallback_reason,
    )


def _infer_final_step(
    checkpoint_index: Mapping[str, Any],
    output_summary: Mapping[str, Any],
) -> int | None:
    if "n_steps" in checkpoint_index:
        return _coerce_optional_int(checkpoint_index.get("n_steps"))
    checkpoints = list(_as_dict(output_summary).get("checkpoints", []) or [])
    if checkpoints:
        last = max(checkpoints, key=lambda item: int(item.get("step", -1)))
        return _coerce_optional_int(last.get("step"))
    return None


def _resolve_artifact_path(
    *,
    model_dir: Path,
    outputs_dir: Path,
    reported_path: Any,
) -> Path:
    if reported_path is None:
        return outputs_dir / "params.json"

    candidate = Path(str(reported_path)).expanduser()
    if candidate.exists():
        return candidate.resolve()

    reported_str = str(reported_path).replace("\\", "/")
    if reported_str.startswith("outputs/"):
        return (model_dir / reported_str).resolve()

    marker = "/outputs/"
    if marker in reported_str:
        suffix = reported_str.split(marker, 1)[1]
        return (outputs_dir / suffix).resolve()

    marker = "outputs/"
    if marker in reported_str:
        suffix = reported_str.split(marker, 1)[1]
        return (outputs_dir / suffix).resolve()

    return (outputs_dir / Path(reported_str).name).resolve()


def _load_structured_file(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    text = path.read_text()
    # Prefer a real YAML parser when available: saved Hydra inputs.yaml can
    # contain folded/wrapped scalars (e.g. a long meta.launcher_cmd continued on
    # an over-indented next line), which the minimal fallback parser below
    # rejects with "Unexpected indentation". PyYAML handles these correctly.
    # Fall back to the simple parser only if PyYAML is unavailable.
    try:
        import yaml  # type: ignore[import-not-found]

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Any:
    """Parse a simple YAML subset used by the saved Hydra inputs.

    The fallback parser supports nested mappings, scalar lists, nulls, booleans,
    integers, and floats. It intentionally does not aim to be a complete YAML
    implementation.
    """

    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line.rstrip("\n"))
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.lstrip(" ")))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        if lines[index][1].startswith("- "):
            return parse_list(index, indent)
        return parse_mapping(index, indent)

    def parse_mapping(index: int, indent: int) -> tuple[dict[str, Any], int]:
        mapping: dict[str, Any] = {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(
                    f"Unexpected indentation while parsing YAML near: {content!r}"
                )
            if content.startswith("- "):
                break

            key, separator, remainder = content.partition(":")
            if separator != ":":
                raise ValueError(f"Invalid YAML line: {content!r}")
            key = key.strip()
            remainder = remainder.strip()
            index += 1
            if not remainder:
                if index < len(lines) and (
                    lines[index][0] > current_indent
                    or (
                        lines[index][0] >= current_indent
                        and lines[index][1].startswith("- ")
                    )
                ):
                    value, index = parse_block(index, lines[index][0])
                else:
                    value = {}
            else:
                value = _parse_scalar(remainder)
            mapping[key] = value
        return mapping, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent or not content.startswith("- "):
                break
            if current_indent > indent:
                raise ValueError(
                    f"Unexpected indentation while parsing YAML list near: {content!r}"
                )

            item = content[2:].strip()
            index += 1
            if not item:
                if index < len(lines) and lines[index][0] > current_indent:
                    value, index = parse_block(index, lines[index][0])
                else:
                    value = None
            else:
                value = _parse_scalar(item)
            items.append(value)
        return items, index

    parsed, final_index = parse_block(0, lines[0][0] if lines else 0)
    if final_index != len(lines):
        raise ValueError("Could not fully parse YAML document.")
    return parsed


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    result_chars = []
    for char in line:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        result_chars.append(char)
    return "".join(result_chars).rstrip()


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if stripped.startswith(("'", '"')) and stripped.endswith(("'", '"')) and len(stripped) >= 2:
        return stripped[1:-1]
    try:
        if any(token in stripped for token in (".", "e", "E")):
            float_value = float(stripped)
            if math.isfinite(float_value):
                return float_value
        return int(stripped)
    except ValueError:
        return stripped


def _build_session_history_dataframe(raw_df):
    pd = _import_dependency("pandas")
    df = raw_df.copy()
    if "subject_id" not in df.columns or "ses_idx" not in df.columns:
        raise ValueError("Raw trial dataframe must include 'subject_id' and 'ses_idx'.")

    choice_col = _first_existing_column(df.columns, ("animal_response", "action", "choice"))
    reward_col = _first_existing_column(df.columns, ("earned_reward", "rewarded", "reward"))
    if choice_col is None or reward_col is None:
        raise ValueError(
            "Raw trial dataframe must include a choice column and a reward column. "
            f"Observed columns: {list(df.columns)}"
        )

    trial_col = _first_existing_column(
        df.columns,
        ("trial", "trial_index", "trial_number"),
    )
    if trial_col is None:
        df["trial"] = df.groupby("ses_idx").cumcount()
        trial_col = "trial"

    if "session_date" not in df.columns:
        df["session_date"] = df["ses_idx"].map(_extract_session_date)

    records = []
    grouped = df.sort_values(["subject_id", "ses_idx", trial_col]).groupby(
        ["subject_id", "ses_idx"],
        sort=False,
    )
    for (subject_id, ses_idx), session_df in grouped:
        session_df = session_df.sort_values(trial_col).reset_index(drop=True)
        raw_choices = session_df[choice_col].tolist()
        choice_history = [_normalize_choice_value(value) for value in raw_choices]
        reward_history = [
            _normalize_reward_value(value) for value in session_df[reward_col].tolist()
        ]
        session_date = (
            session_df["session_date"].iloc[0]
            if "session_date" in session_df.columns
            else _extract_session_date(ses_idx)
        )
        nwb_suffix = session_df["nwb_suffix"].iloc[0] if "nwb_suffix" in session_df.columns else None
        nwb_name = _build_nwb_name(
            subject_id=subject_id,
            session_date=session_date,
            ses_idx=ses_idx,
            nwb_suffix=nwb_suffix,
            existing_value=(
                session_df["nwb_name"].iloc[0] if "nwb_name" in session_df.columns else None
            ),
        )

        records.append(
            {
                "subject_id": _normalize_identifier(subject_id),
                "ses_idx": str(ses_idx),
                "session_date": str(session_date),
                "curriculum_name": _first_non_null(session_df.get("curriculum_name")),
                "current_stage_actual": _first_non_null(
                    session_df.get("current_stage_actual")
                ),
                "n_trials": int(len(session_df)),
                "choice_history": choice_history,
                "reward_history": reward_history,
                "nwb_suffix": nwb_suffix,
                "nwb_name": nwb_name,
            }
        )

    return pd.DataFrame.from_records(records)


def _align_multisubject_session_history_to_training_ids(
    session_history,
    *,
    resolved_run: ResolvedModelRun,
):
    if not resolved_run.session_context_map_path:
        return session_history

    multisubject_mod = importlib.import_module("utils.multisubject")
    session_context = multisubject_mod.load_session_context_map(
        resolved_run.session_context_map_path
    )

    source_to_merged: dict[tuple[Any, str], str] = {}
    merged_to_source: dict[tuple[Any, str], str] = {}
    for row in multisubject_mod.ordered_session_context_rows(session_context):
        subject_id = _normalize_identifier(row.get("subject_id"))
        ordered_session_ids = [
            str(session_id) for session_id in row.get("ordered_session_ids") or []
        ]
        ordered_source_session_ids = [
            str(session_id)
            for session_id in (
                row.get("ordered_source_session_ids")
                or row.get("ordered_session_ids")
                or []
            )
        ]
        if len(ordered_source_session_ids) != len(ordered_session_ids):
            raise ValueError(
                "session_context ordered_source_session_ids must align 1:1 with "
                "ordered_session_ids."
            )
        for merged_session_id, source_session_id in zip(
            ordered_session_ids,
            ordered_source_session_ids,
        ):
            source_to_merged[(subject_id, source_session_id)] = str(merged_session_id)
            merged_to_source[(subject_id, str(merged_session_id))] = str(source_session_id)

    aligned_records: list[dict[str, Any]] = []
    for record in _iter_session_records(session_history):
        subject_id = _normalize_identifier(record.get("subject_id"))
        session_id = str(record.get("ses_idx"))
        source_session_id = _coalesce_session_identifier(
            record.get("source_ses_idx"),
            fallback=session_id,
        )

        merged_session_id = source_to_merged.get((subject_id, source_session_id))
        if merged_session_id is None:
            merged_session_id = source_to_merged.get((subject_id, session_id))
        if merged_session_id is None and (subject_id, session_id) in merged_to_source:
            merged_session_id = str(session_id)
            source_session_id = merged_to_source[(subject_id, session_id)]
        if merged_session_id is None:
            raise ValueError(
                "Could not align multisubject animal session history to the training "
                "session namespace for "
                f"subject_id={subject_id!r}, ses_idx={session_id!r}, "
                f"source_ses_idx={source_session_id!r}."
            )

        aligned_record = dict(record)
        aligned_record["ses_idx"] = str(merged_session_id)
        aligned_record["source_ses_idx"] = str(source_session_id)
        aligned_records.append(aligned_record)

    return _restore_input_container(session_history, aligned_records)


def _build_nwb_name(
    *,
    subject_id: Any,
    session_date: Any,
    ses_idx: Any,
    nwb_suffix: Any,
    existing_value: Any,
) -> str | None:
    if existing_value not in (None, "") and not _is_nan(existing_value):
        return str(existing_value)
    if nwb_suffix not in (None, "") and not _is_nan(nwb_suffix):
        return f"{subject_id}_{session_date}_{nwb_suffix}.nwb"
    if ses_idx not in (None, ""):
        return f"{ses_idx}.nwb"
    return None


def _restore_model_runner(resolved_run: ResolvedModelRun):
    np = _import_dependency("numpy")
    jax = _import_dependency("jax")
    jnp = importlib.import_module("jax.numpy")
    hk = _import_dependency("haiku")
    resolved_run = _refresh_resolved_run_session_conditioning_metadata(resolved_run)
    _validate_multisubject_analysis_split(resolved_run)

    params = _json_tree_to_arrays(json.loads(Path(resolved_run.params_path).read_text()))
    model_type = resolved_run.model_type
    subject_id_to_index: dict[Any, int] = {}
    index_to_subject_id: dict[int, Any] = {}
    runner_uses_session_indices = bool(resolved_run.session_conditioning_enabled)
    session_max_index_by_subject_index: list[int] = []
    merged_session_index_lookup: dict[tuple[Any, str], int] = {}
    source_session_index_lookup: dict[tuple[Any, str], int] = {}
    if resolved_run.multisubject:
        (
            subject_id_to_index,
            index_to_subject_id,
            merged_session_index_lookup,
            source_session_index_lookup,
            session_max_index_by_subject_index,
        ) = _load_multisubject_analysis_context(resolved_run)
        _validate_multisubject_params_against_subject_map(
            params,
            n_subjects=len(index_to_subject_id),
        )

    if model_type == "disrnn":
        config_dict = copy.deepcopy(_as_dict(resolved_run.model_config))
        run_architecture = _as_dict(
            _as_dict(_as_dict(resolved_run.run_config).get("model", {})).get(
                "architecture",
                {},
            )
        )
        config_dict["noiseless_mode"] = True
        config_dict["latent_penalty"] = 0.0
        config_dict["choice_net_latent_penalty"] = 0.0
        config_dict["update_net_obs_penalty"] = 0.0
        config_dict["update_net_latent_penalty"] = 0.0
        config_dict["l2_scale"] = 0.0
        session_conditioning_enabled = bool(resolved_run.session_conditioning_enabled)
        if session_conditioning_enabled:
            config_dict.setdefault(
                "session_encoding_type",
                run_architecture.get("session_encoding_type", "none"),
            )
            config_dict.setdefault(
                "session_integration_type",
                run_architecture.get("session_integration_type", "direct"),
            )
            config_dict.setdefault(
                "session_fourier_k",
                run_architecture.get("session_fourier_k", 4),
            )
        if resolved_run.multisubject:
            multisubject_disrnn_mod = importlib.import_module("models.multisubject_disrnn")
            config_dict["subj_penalty"] = 0.0
            config_dict["update_net_subj_penalty"] = 0.0
            config_dict["choice_net_subj_penalty"] = 0.0
            config_dict["max_n_subjects"] = int(
                config_dict.get("max_n_subjects", len(index_to_subject_id))
            )
            if int(config_dict["max_n_subjects"]) < int(len(index_to_subject_id)):
                raise ValueError(
                    "Saved disRNN config max_n_subjects is smaller than the resolved "
                    "subject_index_map.json roster."
                )
            if session_conditioning_enabled and not config_dict.get(
                "session_max_index_by_subject_index"
            ):
                config_dict["session_max_index_by_subject_index"] = list(
                    session_max_index_by_subject_index
                )
            if session_conditioning_enabled and "session_delta_n_layers" not in config_dict:
                config_dict["session_delta_n_layers"] = 1
            if (
                session_conditioning_enabled
                and "session_delta_hidden_size" not in config_dict
            ):
                config_dict["session_delta_hidden_size"] = int(
                    config_dict["subject_embedding_size"]
                ) * 2
            config = multisubject_disrnn_mod.MultisubjectDisRnnConfig(**config_dict)

            def make_network():
                return multisubject_disrnn_mod.MultisubjectDisRnn(config)
        else:
            disrnn_mod = importlib.import_module("disentangled_rnns.library.disrnn")
            config = disrnn_mod.DisRnnConfig(**config_dict)

            def make_network():
                return disrnn_mod.HkDisentangledRNN(config)

        n_actions = int(config.output_size)
    elif model_type == "gru":
        make_gru_network = importlib.import_module("models.gru_network").make_gru_network
        config_dict = copy.deepcopy(_as_dict(resolved_run.model_config))
        architecture = _as_dict(config_dict.get("architecture", {}))
        run_architecture = _as_dict(
            _as_dict(_as_dict(resolved_run.run_config).get("model", {})).get(
                "architecture",
                {},
            )
        )
        output_size = int(config_dict.get("output_size", architecture.get("output_size", 0)))
        session_conditioning_enabled = bool(resolved_run.session_conditioning_enabled)
        if session_conditioning_enabled:
            architecture.setdefault(
                "session_encoding_type",
                run_architecture.get("session_encoding_type", "none"),
            )
            architecture.setdefault(
                "session_integration_type",
                run_architecture.get("session_integration_type", "direct"),
            )
            architecture.setdefault(
                "session_fourier_k",
                run_architecture.get("session_fourier_k", 4),
            )
            if "session_delta_n_layers" not in architecture and (
                "session_delta_n_layers" in run_architecture
            ):
                architecture["session_delta_n_layers"] = run_architecture[
                    "session_delta_n_layers"
                ]
            if "session_delta_hidden_size" not in architecture and (
                "session_delta_hidden_size" in run_architecture
            ):
                architecture["session_delta_hidden_size"] = run_architecture[
                    "session_delta_hidden_size"
                ]
        make_network = make_gru_network(
            hidden_size=int(architecture["hidden_size"]),
            output_size=output_size,
            multisubject=bool(resolved_run.multisubject),
            max_n_subjects=(
                int(len(subject_id_to_index)) if resolved_run.multisubject else None
            ),
            subject_embedding_size=(
                int(architecture["subject_embedding_size"])
                if resolved_run.multisubject
                else None
            ),
            subject_embedding_init=str(architecture.get("subject_embedding_init", "zeros")),
            session_encoding_type=str(architecture.get("session_encoding_type", "none")),
            session_integration_type=str(
                architecture.get("session_integration_type", "direct")
            ),
            session_fourier_k=int(architecture.get("session_fourier_k", 4)),
            session_delta_n_layers=int(
                architecture.get(
                    "session_delta_n_layers",
                    1 if session_conditioning_enabled else 3,
                )
            ),
            session_delta_hidden_size=int(
                architecture.get(
                    "session_delta_hidden_size",
                    (
                        2 * int(architecture["subject_embedding_size"])
                        if session_conditioning_enabled
                        else 16
                    ),
                )
            ),
            session_max_index_by_subject_index=(
                list(session_max_index_by_subject_index)
                if session_conditioning_enabled
                else None
            ),
        )
        n_actions = int(output_size)
    else:
        raise ValueError(f"Unsupported model_type={model_type!r}")

    if n_actions not in (2, 3):
        raise NotImplementedError(
            "Generative simulation supports 2-action (L/R) or 3-action "
            "(L/R/ignore) models only. "
            f"Resolved output size: {n_actions}."
        )

    def initial_state_fn():
        core = make_network()
        return core.initial_state(batch_size=1)

    def step_fn(inputs, prev_state):
        core = make_network()
        return core(inputs, prev_state)

    initial_state_transform = hk.without_apply_rng(hk.transform(initial_state_fn))
    step_transform = hk.without_apply_rng(hk.transform(step_fn))

    initial_state_params = initial_state_transform.init(jax.random.PRNGKey(0))
    initial_state = initial_state_transform.apply(initial_state_params)

    class _ModelRunner:
        def __init__(
            self,
            initial_state,
            n_actions: int,
            *,
            subject_id_to_index: Mapping[Any, int] | None = None,
            merged_session_index_lookup: Mapping[tuple[Any, str], int] | None = None,
            source_session_index_lookup: Mapping[tuple[Any, str], int] | None = None,
        ) -> None:
            self.initial_state = initial_state
            self.n_actions = int(n_actions)
            self.subject_id_to_index = {
                _normalize_identifier(subject_id): int(subject_index)
                for subject_id, subject_index in (subject_id_to_index or {}).items()
            }
            self.merged_session_index_lookup = dict(merged_session_index_lookup or {})
            self.source_session_index_lookup = dict(source_session_index_lookup or {})

        def validate_subject_ids(self, subject_ids: Sequence[Any]) -> None:
            if not self.subject_id_to_index:
                return
            missing_subject_ids = [
                normalized
                for normalized in _unique_preserve_order(
                    [_normalize_identifier(subject_id) for subject_id in subject_ids]
                )
                if normalized not in self.subject_id_to_index
            ]
            if missing_subject_ids:
                raise ValueError(
                    "Multisubject post-training analysis encountered subject ids "
                    "that are not present in subject_index_map.json: "
                    f"{missing_subject_ids}"
                )

        def _resolve_session_index(
            self,
            *,
            subject_id: Any,
            session_id: Any,
            source_session_id: Any | None = None,
        ) -> int | None:
            return _resolve_session_index_for_subject(
                subject_id=subject_id,
                session_id=session_id,
                source_session_id=source_session_id,
                merged_session_index_lookup=self.merged_session_index_lookup,
                source_session_index_lookup=self.source_session_index_lookup,
            )

        def encode_inputs(
            self,
            subject_id: Any,
            session_id: Any,
            inputs: Sequence[float],
            *,
            source_session_id: Any | None = None,
        ) -> list[float]:
            encoded_inputs = [float(value) for value in inputs]
            if not self.subject_id_to_index:
                return encoded_inputs
            normalized_subject_id = _normalize_identifier(subject_id)
            if normalized_subject_id not in self.subject_id_to_index:
                raise ValueError(
                    "Multisubject post-training analysis encountered subject_id="
                    f"{normalized_subject_id!r}, which is not present in "
                    "subject_index_map.json."
                )
            encoded_prefix = [float(self.subject_id_to_index[normalized_subject_id])]
            session_index = self._resolve_session_index(
                subject_id=normalized_subject_id,
                session_id=session_id,
                source_session_id=source_session_id,
            )
            if session_index is not None:
                encoded_prefix.append(float(session_index))
            return [*encoded_prefix, *encoded_inputs]

        def step(self, inputs: Sequence[float], prev_state):
            input_array = jnp.asarray([list(inputs)], dtype=jnp.float32)
            logits, next_state = step_transform.apply(params, input_array, prev_state)
            action_logits = np.asarray(logits)[0, : self.n_actions]
            if int(action_logits.shape[0]) != self.n_actions:
                raise ValueError(
                    "Model output logits width does not match n_actions: "
                    f"{action_logits.shape[0]} vs {self.n_actions}."
                )
            return action_logits, next_state

        def initial_state_batch(self, batch_size: int):
            """Batch-`batch_size` initial state: tile the batch-1 state. Lanes
            start identical and evolve independently (RNN state has no
            cross-batch coupling)."""
            return jax.tree_util.tree_map(
                lambda leaf: jnp.repeat(leaf, int(batch_size), axis=0),
                self.initial_state,
            )

        def step_batch(self, inputs_2d, prev_state):
            """Batched analogue of ``step``: ``inputs_2d`` is (B, input_dim);
            returns (B, n_actions) logits and the batched next state. The RNN
            core applies independently across the batch, so row ``b`` is
            identical to a separate batch-1 ``step`` on lane ``b`` — batching
            only amortizes the per-trial dispatch/host-sync over the batch."""
            input_array = jnp.asarray(inputs_2d, dtype=jnp.float32)
            logits, next_state = step_transform.apply(params, input_array, prev_state)
            action_logits = np.asarray(logits)[:, : self.n_actions]
            if int(action_logits.shape[1]) != self.n_actions:
                raise ValueError(
                    "Model output logits width does not match n_actions: "
                    f"{action_logits.shape[1]} vs {self.n_actions}."
                )
            return action_logits, next_state

    return _ModelRunner(
        initial_state=initial_state,
        n_actions=n_actions,
        subject_id_to_index=subject_id_to_index,
        merged_session_index_lookup=(
            merged_session_index_lookup if runner_uses_session_indices else None
        ),
        source_session_index_lookup=(
            source_session_index_lookup if runner_uses_session_indices else None
        ),
    )


def _build_curriculum_matched_task(
    *,
    curriculum_name: Any,
    n_trials: int,
    seed: int,
):
    # IMPORTANT LIMITATION (TODO): this matches only the curriculum *family*
    # (coupled/uncoupled + baiting) and builds the task with the gym's DEFAULT
    # block/reward parameters. It does NOT use the session's stage-specific task
    # parameters (current_stage_actual is logged but unused), so a curriculum
    # that spans multiple stages -- each with its own reward probs / block
    # structure / sometimes a different task -- is collapsed into one generic
    # task. The real per-(curriculum, stage) parameters live in
    #   https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training
    # Faithfully reproducing them is tracked in studies/data-scaling-law/
    # FUTURE_DIRECTIONS.md. Match on normalized family substrings so curriculum
    # *version* variants (e.g. "UnCoupledBaiting2p3Curriculum") resolve too.
    task_mod = importlib.import_module("aind_behavior_gym.dynamic_foraging.task")
    raw = str(curriculum_name) if curriculum_name is not None else "None"
    norm = re.sub(r"[^a-z0-9]", "", raw.lower())
    if norm in ("", "none"):
        return task_mod.UncoupledBlockTask(
            reward_baiting=True, num_trials=int(n_trials), seed=int(seed)
        )
    if "randomwalk" in norm:
        # Random Walk is a genuinely different family: reward probabilities diffuse as a bounded
        # random walk rather than switching in blocks. It is rare but REAL in the training data --
        # 9 sessions / 3 subjects / 8,284 trials in the D=614 cohort -- because `curricula` gates
        # which SUBJECTS are drawn, never which of their SESSIONS are used, so a subject admitted
        # on its modal task contributes all of its sessions.
        #
        # Before this, such a session had no family match and was swept into the `norm in ("",
        # "none")` branch, i.e. silently simulated as a default UNCOUPLED BAITING task -- a
        # completely different reward structure. Map it to the gym's own RandomWalkTask instead.
        # Like the other families it is built with the gym's DEFAULT walk parameters (p_min, p_max,
        # sigma, mean); see the LIMITATION note above.
        return task_mod.RandomWalkTask(num_trials=int(n_trials), seed=int(seed))
    baiting = "withoutbaiting" not in norm and "nobaiting" not in norm
    if "uncoupled" in norm:
        return task_mod.UncoupledBlockTask(
            reward_baiting=baiting, num_trials=int(n_trials), seed=int(seed)
        )
    if "coupled" in norm:  # uncoupled already handled above
        return task_mod.CoupledBlockTask(
            reward_baiting=baiting, num_trials=int(n_trials), seed=int(seed)
        )
    raise ValueError(
        f"Unsupported curriculum_name={curriculum_name!r} "
        "(no coupled/uncoupled family match)"
    )


def _step_task_reward(task: Any, action: int) -> float:
    if hasattr(task, "step"):
        result = task.step(int(action))
    elif hasattr(task, "trial"):
        result = task.trial(int(action))
    elif hasattr(task, "update"):
        result = task.update(int(action))
    else:
        raise AttributeError(
            "Could not find a supported task stepping method. "
            "Expected one of: step, trial, update."
        )
    return float(_extract_reward_from_step_result(result))


def _extract_reward_from_step_result(result: Any) -> float:
    if isinstance(result, Mapping):
        for key in ("reward", "earned_reward"):
            if key in result:
                return _normalize_reward_value(result[key])
    if isinstance(result, tuple):
        if len(result) >= 2 and _is_numeric_like(result[1]):
            return _normalize_reward_value(result[1])
        if len(result) >= 1 and _is_numeric_like(result[0]):
            return _normalize_reward_value(result[0])
    if _is_numeric_like(result):
        return _normalize_reward_value(result)
    raise ValueError(f"Unable to extract reward from task step result: {result!r}")


def _resolve_analysis_output_dir(
    *,
    resolved_run: ResolvedModelRun,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    step_label = (
        f"step_{resolved_run.checkpoint_step}"
        if resolved_run.checkpoint_step is not None
        else resolved_run.checkpoint_label
    )
    return (
        Path(resolved_run.outputs_dir)
        / "post_training_analysis"
        / f"{resolved_run.split}_{resolved_run.checkpoint_policy}_{step_label}"
    )


def build_curriculum_matched_task(
    *,
    curriculum_name: Any,
    n_trials: int,
    seed: int,
):
    """Public wrapper for building curriculum-matched simulation tasks."""

    return _build_curriculum_matched_task(
        curriculum_name=curriculum_name,
        n_trials=n_trials,
        seed=seed,
    )


def save_switch_figures(
    *,
    switch_stats: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Public wrapper for saving switch-analysis figures."""

    return _save_switch_figures(
        switch_stats=switch_stats,
        output_dir=output_dir,
    )


def save_history_dependent_switch_figures(
    *,
    history_stats: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Public wrapper for saving history-dependent switch-analysis figures."""

    return _save_history_dependent_switch_figures(
        history_stats=history_stats,
        output_dir=output_dir,
    )


def _save_switch_figures(
    *,
    switch_stats: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        logger.warning(
            "matplotlib is unavailable; skipping switch-analysis figure generation."
        )
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}

    pooled_path = output_dir / "pooled_switch_probability.png"
    _plot_pooled_switch_probability(plt, switch_stats, pooled_path)
    figure_paths["pooled_switch_probability"] = pooled_path

    reward_pooled_path = output_dir / "post_switch_by_reward_pooled.png"
    _plot_post_switch_by_reward_pooled(plt, switch_stats, reward_pooled_path)
    figure_paths["post_switch_by_reward_pooled"] = reward_pooled_path

    reward_path = output_dir / "post_switch_by_reward.png"
    _plot_post_switch_by_reward(plt, switch_stats, reward_path)
    figure_paths["post_switch_by_reward"] = reward_path

    reward_run_pooled_path = output_dir / "post_switch_by_reward_and_run_length_pooled.png"
    _plot_post_switch_by_reward_and_run_length_pooled(
        plt,
        switch_stats,
        reward_run_pooled_path,
    )
    figure_paths["post_switch_by_reward_and_run_length_pooled"] = reward_run_pooled_path

    reward_run_path = output_dir / "post_switch_by_reward_and_run_length.png"
    _plot_post_switch_by_reward_and_run_length(plt, switch_stats, reward_run_path)
    figure_paths["post_switch_by_reward_and_run_length"] = reward_run_path

    reward_delta_path = output_dir / "post_switch_delta_by_reward.png"
    _plot_post_switch_delta_by_reward(
        plt,
        switch_stats,
        reward_delta_path,
        show_significance=True,
    )
    figure_paths["post_switch_delta_by_reward"] = reward_delta_path
    reward_delta_no_stats_path = output_dir / "post_switch_delta_by_reward_no_stats.png"
    _plot_post_switch_delta_by_reward(
        plt,
        switch_stats,
        reward_delta_no_stats_path,
        show_significance=False,
    )
    figure_paths["post_switch_delta_by_reward_no_stats"] = reward_delta_no_stats_path

    reward_run_delta_path = output_dir / "post_switch_delta_by_reward_and_run_length.png"
    _plot_post_switch_delta_by_reward_and_run_length(
        plt,
        switch_stats,
        reward_run_delta_path,
        show_significance=True,
    )
    figure_paths["post_switch_delta_by_reward_and_run_length"] = (
        reward_run_delta_path
    )
    reward_run_delta_no_stats_path = (
        output_dir / "post_switch_delta_by_reward_and_run_length_no_stats.png"
    )
    _plot_post_switch_delta_by_reward_and_run_length(
        plt,
        switch_stats,
        reward_run_delta_no_stats_path,
        show_significance=False,
    )
    figure_paths["post_switch_delta_by_reward_and_run_length_no_stats"] = (
        reward_run_delta_no_stats_path
    )

    reward_subject_scatter_path = output_dir / "post_switch_by_reward_subject_scatter.png"
    _plot_post_switch_by_reward_subject_scatter(
        plt,
        switch_stats,
        reward_subject_scatter_path,
    )
    figure_paths["post_switch_by_reward_subject_scatter"] = reward_subject_scatter_path

    reward_run_subject_scatter_path = (
        output_dir / "post_switch_by_reward_and_run_length_subject_scatter.png"
    )
    _plot_post_switch_by_reward_and_run_length_subject_scatter(
        plt,
        switch_stats,
        reward_run_subject_scatter_path,
    )
    figure_paths[
        "post_switch_by_reward_and_run_length_subject_scatter"
    ] = reward_run_subject_scatter_path

    reward_session_scatter_path = output_dir / "post_switch_by_reward_session_scatter.png"
    _plot_post_switch_by_reward_session_scatter(
        plt,
        switch_stats,
        reward_session_scatter_path,
    )
    figure_paths["post_switch_by_reward_session_scatter"] = reward_session_scatter_path

    reward_run_session_scatter_path = (
        output_dir / "post_switch_by_reward_and_run_length_session_scatter.png"
    )
    _plot_post_switch_by_reward_and_run_length_session_scatter(
        plt,
        switch_stats,
        reward_run_session_scatter_path,
    )
    figure_paths[
        "post_switch_by_reward_and_run_length_session_scatter"
    ] = reward_run_session_scatter_path

    session_level_by_subject_dir = output_dir / "session_level_by_subject"
    subject_ids = _collect_switch_session_subject_ids(switch_stats)
    for subject_id in subject_ids:
        safe_subject_id = _safe_filename_component(subject_id)
        subject_reward_path = (
            session_level_by_subject_dir
            / f"post_switch_by_reward_session_scatter__subject_{safe_subject_id}.png"
        )
        _plot_post_switch_by_reward_session_scatter_for_subject(
            plt,
            switch_stats,
            subject_reward_path,
            subject_id=subject_id,
        )
        figure_paths[
            f"post_switch_by_reward_session_scatter__subject_{safe_subject_id}"
        ] = subject_reward_path

        subject_reward_run_path = (
            session_level_by_subject_dir
            / (
                "post_switch_by_reward_and_run_length_session_scatter"
                f"__subject_{safe_subject_id}.png"
            )
        )
        _plot_post_switch_by_reward_and_run_length_session_scatter_for_subject(
            plt,
            switch_stats,
            subject_reward_run_path,
            subject_id=subject_id,
        )
        figure_paths[
            "post_switch_by_reward_and_run_length_session_scatter"
            f"__subject_{safe_subject_id}"
        ] = subject_reward_run_path

    return figure_paths


def _save_history_dependent_switch_figures(
    *,
    history_stats: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        logger.warning(
            "matplotlib is unavailable; skipping history-dependent figure generation."
        )
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}
    config = _as_dict(history_stats.get("config", {}))
    pattern_type = _normalize_history_pattern_type(
        str(config.get("default_pattern_type", "abstract"))
    )
    max_trials_back = int(
        config.get("max_trials_back", _DEFAULT_HISTORY_MAX_TRIALS_BACK)
    )

    aggregate_path = output_dir / f"history_pattern_comparison_{pattern_type}.png"
    _plot_history_pattern_comparison_figure(
        plt,
        history_stats,
        aggregate_path,
        pattern_type=pattern_type,
    )
    figure_paths[f"history_pattern_comparison_{pattern_type}"] = aggregate_path

    aggregate_pooled_path = (
        output_dir / f"history_pattern_comparison_{pattern_type}_pooled.png"
    )
    _plot_history_pattern_comparison_pooled_figure(
        plt,
        history_stats,
        aggregate_pooled_path,
        pattern_type=pattern_type,
    )
    figure_paths[
        f"history_pattern_comparison_{pattern_type}_pooled"
    ] = aggregate_pooled_path

    delta_path = output_dir / f"history_pattern_delta_{pattern_type}.png"
    _plot_history_pattern_delta_figure(
        plt,
        history_stats,
        delta_path,
        pattern_type=pattern_type,
        show_significance=True,
    )
    figure_paths[f"history_pattern_delta_{pattern_type}"] = delta_path
    delta_no_stats_path = (
        output_dir / f"history_pattern_delta_{pattern_type}_no_stats.png"
    )
    _plot_history_pattern_delta_figure(
        plt,
        history_stats,
        delta_no_stats_path,
        pattern_type=pattern_type,
        show_significance=False,
    )
    figure_paths[f"history_pattern_delta_{pattern_type}_no_stats"] = (
        delta_no_stats_path
    )

    for n_back in range(1, max_trials_back + 1):
        subject_path = (
            output_dir / f"history_pattern_subject_level_{pattern_type}_nback_{n_back}.png"
        )
        _plot_history_pattern_subject_level_figure(
            plt,
            history_stats,
            subject_path,
            pattern_type=pattern_type,
            n_back=n_back,
        )
        figure_paths[
            f"history_pattern_subject_level_{pattern_type}_nback_{n_back}"
        ] = subject_path

    session_level_by_subject_dir = output_dir / "session_level_by_subject"
    subject_ids = _collect_history_session_subject_ids(
        history_stats,
        pattern_type=pattern_type,
        max_trials_back=max_trials_back,
    )
    for n_back in range(1, max_trials_back + 1):
        session_path = (
            output_dir / f"history_pattern_session_level_{pattern_type}_nback_{n_back}.png"
        )
        _plot_history_pattern_session_level_figure(
            plt,
            history_stats,
            session_path,
            pattern_type=pattern_type,
            n_back=n_back,
        )
        figure_paths[
            f"history_pattern_session_level_{pattern_type}_nback_{n_back}"
        ] = session_path

        for subject_id in subject_ids:
            safe_subject_id = _safe_filename_component(subject_id)
            subject_session_path = (
                session_level_by_subject_dir
                / (
                    f"history_pattern_session_level_{pattern_type}_nback_{n_back}"
                    f"__subject_{safe_subject_id}.png"
                )
            )
            _plot_history_pattern_session_level_figure_for_subject(
                plt,
                history_stats,
                subject_session_path,
                pattern_type=pattern_type,
                n_back=n_back,
                subject_id=subject_id,
            )
            figure_paths[
                f"history_pattern_session_level_{pattern_type}_nback_{n_back}"
                f"__subject_{safe_subject_id}"
            ] = subject_session_path

    return figure_paths


def _plot_pooled_switch_probability(plt, switch_stats: Mapping[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in (("animal", "black"), ("simulated", "tab:blue")):
        pooled = list(_as_dict(switch_stats.get(label, {})).get("pooled_switch_probability", []))
        if not pooled:
            continue
        xs = [row["relative_position"] for row in pooled]
        ys = [row["switch_probability"] for row in pooled]
        yerr = [row["switch_probability_sem"] for row in pooled]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=label, color=color)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Relative trial position")
    ax.set_ylabel("Switch probability")
    ax.set_title("Switch Probability Around Switches")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _normalize_probability_bar_values(
    probabilities: Sequence[Any],
    sems: Sequence[Any],
) -> tuple[list[float], list[float]]:
    plot_probabilities: list[float] = []
    plot_sems: list[float] = []
    for probability, sem in zip(probabilities, sems, strict=False):
        probability_value = _coerce_probability(probability)
        sem_value = _coerce_probability(sem)
        if probability_value is None:
            plot_probabilities.append(math.nan)
            plot_sems.append(0.0)
            continue
        plot_probabilities.append(probability_value)
        plot_sems.append(0.0 if sem_value is None else sem_value)
    return plot_probabilities, plot_sems


def _plot_probability_bar_series(
    ax,
    xs: Sequence[float],
    probabilities: Sequence[Any],
    sems: Sequence[Any],
    *,
    width: float,
    label: str,
    alpha: float,
    capsize: float = 4,
) -> None:
    plot_probabilities, plot_sems = _normalize_probability_bar_values(probabilities, sems)
    ax.bar(
        xs,
        plot_probabilities,
        width=width,
        yerr=plot_sems,
        capsize=capsize,
        label=label,
        alpha=alpha,
    )


def _plot_post_switch_by_reward_pooled(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    conditions = ["rewarded", "unrewarded"]
    labels = ["Animal", "Simulated"]
    offsets = [-0.18, 0.18]
    width = 0.32
    x_positions = [0, 1]

    for idx, source in enumerate(("animal", "simulated")):
        condition_stats = _as_dict(
            _as_dict(switch_stats.get(source, {})).get("post_switch_by_reward", {})
        )
        probs = [
            _coerce_probability(condition_stats.get(condition, {}).get("probability"))
            for condition in conditions
        ]
        sems = [
            _coerce_probability(condition_stats.get(condition, {}).get("sem"))
            for condition in conditions
        ]
        xs = [x + offsets[idx] for x in x_positions]
        _plot_probability_bar_series(
            ax,
            xs,
            probs,
            sems,
            width=width,
            label=labels[idx],
            alpha=0.8,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Rewarded", "Unrewarded"])
    ax.set_ylabel("p_switch(t+1)")
    ax.set_title("Post-switch Probability By Reward")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward(plt, switch_stats: Mapping[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    subject_aggregate = _as_dict(switch_stats.get("subject_aggregate", {}))
    reward_group = _as_dict(subject_aggregate.get("post_switch_by_reward", {}))
    conditions = ["rewarded", "unrewarded"]
    labels = ["Animal", "Simulated"]
    offsets = [-0.18, 0.18]
    width = 0.32
    x_positions = [0, 1]

    for idx, source in enumerate(("animal", "simulated")):
        mean_key = f"{source}_mean"
        sem_key = f"{source}_sem"
        probs = [
            _coerce_probability(_as_dict(reward_group.get(condition, {})).get(mean_key))
            for condition in conditions
        ]
        sems = [
            _coerce_probability(_as_dict(reward_group.get(condition, {})).get(sem_key))
            for condition in conditions
        ]
        xs = [x + offsets[idx] for x in x_positions]
        _plot_probability_bar_series(
            ax,
            xs,
            probs,
            sems,
            width=width,
            label=labels[idx],
            alpha=0.8,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Rewarded", "Unrewarded"])
    ax.set_ylabel("p_switch(t+1)")
    ax.set_title("Post-switch Probability By Reward (Subject Mean +/- SEM)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_and_run_length_pooled(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = [
        ("rewarded", "run_length_1", "Rewarded\nRun=1"),
        ("rewarded", "run_length_gt1", "Rewarded\nRun>1"),
        ("unrewarded", "run_length_1", "Unrewarded\nRun=1"),
        ("unrewarded", "run_length_gt1", "Unrewarded\nRun>1"),
    ]
    labels = ["Animal", "Simulated"]
    offsets = [-0.18, 0.18]
    width = 0.32
    x_positions = [0, 1, 2, 3]

    for idx, source in enumerate(("animal", "simulated")):
        nested = _as_dict(
            _as_dict(switch_stats.get(source, {})).get(
                "post_switch_by_reward_and_run_length",
                {},
            )
        )
        probs = []
        sems = []
        for reward_condition, run_condition, _ in conditions:
            stats = _as_dict(_as_dict(nested.get(reward_condition, {})).get(run_condition, {}))
            probs.append(_coerce_probability(stats.get("probability")))
            sems.append(_coerce_probability(stats.get("sem")))
        xs = [x + offsets[idx] for x in x_positions]
        _plot_probability_bar_series(
            ax,
            xs,
            probs,
            sems,
            width=width,
            label=labels[idx],
            alpha=0.8,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, _, label in conditions])
    ax.set_ylabel("p_switch(t+1)")
    ax.set_title("Post-switch Probability By Reward And Run Length")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_and_run_length(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    subject_aggregate = _as_dict(switch_stats.get("subject_aggregate", {}))
    nested = _as_dict(subject_aggregate.get("post_switch_by_reward_and_run_length", {}))
    conditions = [
        ("rewarded", "run_length_1", "Rewarded\nRun=1"),
        ("rewarded", "run_length_gt1", "Rewarded\nRun>1"),
        ("unrewarded", "run_length_1", "Unrewarded\nRun=1"),
        ("unrewarded", "run_length_gt1", "Unrewarded\nRun>1"),
    ]
    labels = ["Animal", "Simulated"]
    offsets = [-0.18, 0.18]
    width = 0.32
    x_positions = [0, 1, 2, 3]

    for idx, source in enumerate(("animal", "simulated")):
        mean_key = f"{source}_mean"
        sem_key = f"{source}_sem"
        probs = []
        sems = []
        for reward_condition, run_condition, _ in conditions:
            stats = _as_dict(_as_dict(nested.get(reward_condition, {})).get(run_condition, {}))
            probs.append(_coerce_probability(stats.get(mean_key)))
            sems.append(_coerce_probability(stats.get(sem_key)))
        xs = [x + offsets[idx] for x in x_positions]
        _plot_probability_bar_series(
            ax,
            xs,
            probs,
            sems,
            width=width,
            label=labels[idx],
            alpha=0.8,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, _, label in conditions])
    ax.set_ylabel("p_switch(t+1)")
    ax.set_title("Post-switch Probability By Reward And Run Length (Subject Mean +/- SEM)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_delta_by_reward(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
    *,
    show_significance: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    reward_stats = _as_dict(_as_dict(switch_stats.get("subject_level", {})).get("post_switch_by_reward", {}))
    delta_summary = _as_dict(
        _as_dict(switch_stats.get("delta_significance_summary", {})).get(
            "post_switch_by_reward",
            {},
        )
    )
    rows = [
        {
            "label": "Rewarded",
            "points": _select_valid_subject_points(
                list(_as_dict(reward_stats.get("rewarded", {})).get("points", []))
            ),
        },
        {
            "label": "Unrewarded",
            "points": _select_valid_subject_points(
                list(_as_dict(reward_stats.get("unrewarded", {})).get("points", []))
            ),
        },
    ]
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for row in rows
            for point in row["points"]
        ],
        plt,
    )
    _plot_delta_distribution_panel(
        ax=ax,
        rows=rows,
        title=_format_delta_plot_title(
            "Subject Delta Probability By Reward",
            delta_summary if show_significance else {},
        ),
        plt=plt,
        curriculum_to_color=curriculum_to_color,
        show_significance=show_significance,
    )
    if show_significance:
        _add_delta_significance_note(fig)
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.95))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_delta_by_reward_and_run_length(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
    *,
    show_significance: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    reward_run_stats = _as_dict(
        _as_dict(switch_stats.get("subject_level", {})).get(
            "post_switch_by_reward_and_run_length",
            {},
        )
    )
    delta_summary = _as_dict(
        _as_dict(switch_stats.get("delta_significance_summary", {})).get(
            "post_switch_by_reward_and_run_length",
            {},
        )
    )
    rows = []
    for reward_condition, run_condition, label in (
        ("rewarded", "run_length_1", "Rewarded / Run=1"),
        ("rewarded", "run_length_gt1", "Rewarded / Run>1"),
        ("unrewarded", "run_length_1", "Unrewarded / Run=1"),
        ("unrewarded", "run_length_gt1", "Unrewarded / Run>1"),
    ):
        rows.append(
            {
                "label": label,
                "points": _select_valid_subject_points(
                    list(
                        _as_dict(
                            _as_dict(reward_run_stats.get(reward_condition, {})).get(
                                run_condition,
                                {},
                            )
                        ).get("points", [])
                    )
                ),
            }
        )
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for row in rows
            for point in row["points"]
        ],
        plt,
    )
    _plot_delta_distribution_panel(
        ax=ax,
        rows=rows,
        title=_format_delta_plot_title(
            "Subject Delta Probability By Reward And Run Length",
            delta_summary if show_significance else {},
        ),
        plt=plt,
        curriculum_to_color=curriculum_to_color,
        show_significance=show_significance,
    )
    if show_significance:
        _add_delta_significance_note(fig)
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.95))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_subject_scatter(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    subject_level = _as_dict(switch_stats.get("subject_level", {}))
    reward_stats = _as_dict(subject_level.get("post_switch_by_reward", {}))
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for reward_condition in _REWARD_CONDITIONS
            for point in _select_valid_subject_points(
                list(_as_dict(reward_stats.get(reward_condition, {})).get("points", []))
            )
        ],
        plt,
    )

    for axis, reward_condition, label in zip(
        axes,
        _REWARD_CONDITIONS,
        ("Rewarded", "Unrewarded"),
        strict=False,
    ):
        panel_stats = _as_dict(reward_stats.get(reward_condition, {}))
        _plot_subject_level_scatter_panel(
            ax=axis,
            points=list(panel_stats.get("points", [])),
            summary=_as_dict(panel_stats.get("summary", {})),
            title=label,
            curriculum_to_color=curriculum_to_color,
        )

    fig.suptitle("Subject-Level Post-switch Probability By Reward", fontsize=14)
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_session_scatter(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    session_level = _as_dict(switch_stats.get("session_level", {}))
    reward_stats = _as_dict(session_level.get("post_switch_by_reward", {}))
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for reward_condition in _REWARD_CONDITIONS
            for point in _select_valid_session_points(
                list(_as_dict(reward_stats.get(reward_condition, {})).get("points", []))
            )
        ],
        plt,
    )

    for axis, reward_condition, label in zip(
        axes,
        _REWARD_CONDITIONS,
        ("Rewarded", "Unrewarded"),
        strict=False,
    ):
        panel_stats = _as_dict(reward_stats.get(reward_condition, {}))
        _plot_session_level_scatter_panel(
            ax=axis,
            points=list(panel_stats.get("points", [])),
            summary=_as_dict(panel_stats.get("summary", {})),
            title=label,
            curriculum_to_color=curriculum_to_color,
        )

    fig.suptitle("Session-Level Post-switch Probability By Reward", fontsize=14)
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_session_scatter_for_subject(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
    *,
    subject_id: Any,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    session_level = _as_dict(switch_stats.get("session_level", {}))
    reward_stats = _as_dict(session_level.get("post_switch_by_reward", {}))
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for reward_condition in _REWARD_CONDITIONS
            for point in _select_valid_session_points(
                _filter_points_for_subject(
                    list(_as_dict(reward_stats.get(reward_condition, {})).get("points", [])),
                    subject_id=subject_id,
                )
            )
        ],
        plt,
    )

    for axis, reward_condition, label in zip(
        axes,
        _REWARD_CONDITIONS,
        ("Rewarded", "Unrewarded"),
        strict=False,
    ):
        filtered_points = _filter_points_for_subject(
            list(_as_dict(reward_stats.get(reward_condition, {})).get("points", [])),
            subject_id=subject_id,
        )
        _plot_session_level_scatter_panel(
            ax=axis,
            points=filtered_points,
            summary=_summarize_session_comparison_points(filtered_points),
            title=label,
            curriculum_to_color=curriculum_to_color,
        )

    fig.suptitle(
        f"Session-Level Post-switch Probability By Reward ({subject_id})",
        fontsize=14,
    )
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_and_run_length_subject_scatter(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    subject_level = _as_dict(switch_stats.get("subject_level", {}))
    reward_run_stats = _as_dict(
        subject_level.get("post_switch_by_reward_and_run_length", {})
    )
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for reward_condition in _REWARD_CONDITIONS
            for run_condition in _RUN_LENGTH_CONDITIONS
            for point in _select_valid_subject_points(
                list(
                    _as_dict(
                        _as_dict(reward_run_stats.get(reward_condition, {})).get(
                            run_condition,
                            {},
                        )
                    ).get("points", [])
                )
            )
        ],
        plt,
    )
    panel_definitions = [
        ("rewarded", "run_length_1", "Rewarded / Run=1"),
        ("rewarded", "run_length_gt1", "Rewarded / Run>1"),
        ("unrewarded", "run_length_1", "Unrewarded / Run=1"),
        ("unrewarded", "run_length_gt1", "Unrewarded / Run>1"),
    ]

    for axis, (reward_condition, run_condition, label) in zip(
        axes.flatten(),
        panel_definitions,
        strict=False,
    ):
        panel_stats = _as_dict(
            _as_dict(reward_run_stats.get(reward_condition, {})).get(
                run_condition,
                {},
            )
        )
        _plot_subject_level_scatter_panel(
            ax=axis,
            points=list(panel_stats.get("points", [])),
            summary=_as_dict(panel_stats.get("summary", {})),
            title=label,
            curriculum_to_color=curriculum_to_color,
        )

    fig.suptitle(
        "Subject-Level Post-switch Probability By Reward And Run Length",
        fontsize=14,
    )
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_and_run_length_session_scatter(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    session_level = _as_dict(switch_stats.get("session_level", {}))
    reward_run_stats = _as_dict(
        session_level.get("post_switch_by_reward_and_run_length", {})
    )
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for reward_condition in _REWARD_CONDITIONS
            for run_condition in _RUN_LENGTH_CONDITIONS
            for point in _select_valid_session_points(
                list(
                    _as_dict(
                        _as_dict(reward_run_stats.get(reward_condition, {})).get(
                            run_condition,
                            {},
                        )
                    ).get("points", [])
                )
            )
        ],
        plt,
    )
    panel_definitions = [
        ("rewarded", "run_length_1", "Rewarded / Run=1"),
        ("rewarded", "run_length_gt1", "Rewarded / Run>1"),
        ("unrewarded", "run_length_1", "Unrewarded / Run=1"),
        ("unrewarded", "run_length_gt1", "Unrewarded / Run>1"),
    ]

    for axis, (reward_condition, run_condition, label) in zip(
        axes.flatten(),
        panel_definitions,
        strict=False,
    ):
        panel_stats = _as_dict(
            _as_dict(reward_run_stats.get(reward_condition, {})).get(
                run_condition,
                {},
            )
        )
        _plot_session_level_scatter_panel(
            ax=axis,
            points=list(panel_stats.get("points", [])),
            summary=_as_dict(panel_stats.get("summary", {})),
            title=label,
            curriculum_to_color=curriculum_to_color,
        )

    fig.suptitle(
        "Session-Level Post-switch Probability By Reward And Run Length",
        fontsize=14,
    )
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_and_run_length_session_scatter_for_subject(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
    *,
    subject_id: Any,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    session_level = _as_dict(switch_stats.get("session_level", {}))
    reward_run_stats = _as_dict(
        session_level.get("post_switch_by_reward_and_run_length", {})
    )
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for reward_condition in _REWARD_CONDITIONS
            for run_condition in _RUN_LENGTH_CONDITIONS
            for point in _select_valid_session_points(
                _filter_points_for_subject(
                    list(
                        _as_dict(
                            _as_dict(reward_run_stats.get(reward_condition, {})).get(
                                run_condition,
                                {},
                            )
                        ).get("points", [])
                    ),
                    subject_id=subject_id,
                )
            )
        ],
        plt,
    )
    panel_definitions = [
        ("rewarded", "run_length_1", "Rewarded / Run=1"),
        ("rewarded", "run_length_gt1", "Rewarded / Run>1"),
        ("unrewarded", "run_length_1", "Unrewarded / Run=1"),
        ("unrewarded", "run_length_gt1", "Unrewarded / Run>1"),
    ]

    for axis, (reward_condition, run_condition, label) in zip(
        axes.flatten(),
        panel_definitions,
        strict=False,
    ):
        filtered_points = _filter_points_for_subject(
            list(
                _as_dict(
                    _as_dict(reward_run_stats.get(reward_condition, {})).get(
                        run_condition,
                        {},
                    )
                ).get("points", [])
            ),
            subject_id=subject_id,
        )
        _plot_session_level_scatter_panel(
            ax=axis,
            points=filtered_points,
            summary=_summarize_session_comparison_points(filtered_points),
            title=label,
            curriculum_to_color=curriculum_to_color,
        )

    fig.suptitle(
        f"Session-Level Post-switch Probability By Reward And Run Length ({subject_id})",
        fontsize=14,
    )
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_subject_level_scatter_panel(
    *,
    ax,
    points: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    title: str,
    curriculum_to_color: Mapping[str, Any],
) -> None:
    _plot_probability_scatter_panel(
        ax=ax,
        points=points,
        summary=summary,
        title=title,
        curriculum_to_color=curriculum_to_color,
        selector=_select_valid_subject_points,
        count_label="n_subjects",
        x_label="Animal p_switch(t+1)",
        y_label="Simulation p_switch(t+1)",
        empty_label="No valid subjects",
    )


def _plot_session_level_scatter_panel(
    *,
    ax,
    points: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    title: str,
    curriculum_to_color: Mapping[str, Any],
) -> None:
    _plot_probability_scatter_panel(
        ax=ax,
        points=points,
        summary=summary,
        title=title,
        curriculum_to_color=curriculum_to_color,
        selector=_select_valid_session_points,
        count_label="n_sessions",
        x_label="Animal p_switch(t+1)",
        y_label="Simulation p_switch(t+1)",
        empty_label="No valid sessions",
    )


def _plot_probability_scatter_panel(
    *,
    ax,
    points: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    title: str,
    curriculum_to_color: Mapping[str, Any],
    selector,
    count_label: str,
    x_label: str,
    y_label: str,
    empty_label: str,
) -> None:
    valid_points = list(selector(points))
    if valid_points:
        for point in valid_points:
            x_value = float(point["animal_probability"])
            y_value = float(point["simulated_probability"])
            curriculum_name = str(point.get("curriculum_name", "Unknown"))
            color = curriculum_to_color.get(curriculum_name, "tab:blue")
            xerr = _asymmetric_errorbar(
                x_value,
                point.get("animal_ci_low"),
                point.get("animal_ci_high"),
            )
            yerr = _asymmetric_errorbar(
                y_value,
                point.get("simulated_ci_low"),
                point.get("simulated_ci_high"),
            )
            if xerr is not None or yerr is not None:
                ax.errorbar(
                    [x_value],
                    [y_value],
                    xerr=xerr,
                    yerr=yerr,
                    fmt="none",
                    ecolor=color,
                    alpha=0.25,
                    capsize=2,
                    linewidth=1.1,
                )
            ax.scatter(
                [x_value],
                [y_value],
                alpha=0.8,
                s=50,
                color=color,
                edgecolors="black",
                linewidths=0.7,
            )
    else:
        ax.text(
            0.5,
            0.5,
            empty_label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", alpha=0.5, linewidth=1.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if int(summary.get(count_label, 0)) >= 2:
        annotation_lines = [
            f"r={float(summary['correlation']):.3f}"
            if summary.get("correlation") is not None
            else "r=None",
            f"RMSE={float(summary['rmse']):.3f}"
            if summary.get("rmse") is not None
            else "RMSE=None",
            f"Bias={float(summary['bias']):.3f}"
            if summary.get("bias") is not None
            else "Bias=None",
            f"n={int(summary.get(count_label, 0))}",
        ]
        ax.text(
            0.05,
            0.95,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )


def _filter_points_for_subject(
    points: Sequence[Mapping[str, Any]],
    *,
    subject_id: Any,
) -> list[dict[str, Any]]:
    normalized_subject_id = _normalize_identifier(subject_id)
    return [
        dict(point)
        for point in points
        if _normalize_identifier(point.get("subject_id")) == normalized_subject_id
    ]


def _collect_point_subject_ids(points: Sequence[Mapping[str, Any]]) -> list[Any]:
    return _unique_preserve_order(
        [
            point.get("subject_id")
            for point in points
            if point.get("subject_id") not in (None, "")
        ]
    )


def _collect_switch_session_subject_ids(
    switch_stats: Mapping[str, Any],
) -> list[Any]:
    session_level = _as_dict(switch_stats.get("session_level", {}))
    reward_stats = _as_dict(session_level.get("post_switch_by_reward", {}))
    subject_ids: list[Any] = []
    for reward_condition in _REWARD_CONDITIONS:
        subject_ids.extend(
            _collect_point_subject_ids(
                list(_as_dict(reward_stats.get(reward_condition, {})).get("points", []))
            )
        )
    return _unique_preserve_order(subject_ids)


def _collect_history_session_subject_ids(
    history_stats: Mapping[str, Any],
    *,
    pattern_type: str,
    max_trials_back: int,
) -> list[Any]:
    session_level = _as_dict(
        _as_dict(history_stats.get("session_level", {})).get(pattern_type, {})
    )
    subject_ids: list[Any] = []
    for n_back in range(1, int(max_trials_back) + 1):
        panel_group = _as_dict(session_level.get(n_back, {}))
        for pattern in panel_group:
            subject_ids.extend(
                _collect_point_subject_ids(
                    list(_as_dict(panel_group.get(pattern, {})).get("points", []))
                )
            )
    return _unique_preserve_order(subject_ids)


def _plot_history_pattern_comparison_figure(
    plt,
    history_stats: Mapping[str, Any],
    output_path: Path,
    *,
    pattern_type: str,
) -> None:
    subject_aggregate = _as_dict(
        _as_dict(history_stats.get("subject_aggregate", {})).get(pattern_type, {})
    )
    config = _as_dict(history_stats.get("config", {}))
    max_trials_back = int(
        config.get("max_trials_back", _DEFAULT_HISTORY_MAX_TRIALS_BACK)
    )
    fig, axes = plt.subplots(1, max_trials_back, figsize=(6 * max_trials_back, 6))
    if max_trials_back == 1:
        axes = [axes]

    all_patterns = sorted(
        {
            str(row.get("pattern"))
            for n_back in range(1, max_trials_back + 1)
            for row in list(_as_dict(subject_aggregate.get(n_back, {})).get("rows", []))
            if int(row.get("n_subjects", 0)) > 0
        }
    )
    pattern_colors = _build_history_pattern_color_map(all_patterns, plt)
    legend_handles: dict[str, Any] = {}

    for axis, n_back in zip(axes, range(1, max_trials_back + 1), strict=False):
        panel = _as_dict(subject_aggregate.get(n_back, {}))
        valid_rows = [
            row
            for row in list(panel.get("rows", []))
            if int(row.get("n_subjects", 0)) > 0
        ]

        if not valid_rows:
            axis.text(
                0.5,
                0.5,
                "No patterns with\nmatched subjects",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        else:
            for row in valid_rows:
                pattern = str(row.get("pattern"))
                handle = axis.errorbar(
                    float(row["animal_mean"]),
                    float(row["simulated_mean"]),
                    xerr=row.get("animal_sem"),
                    yerr=row.get("simulated_sem"),
                    marker="o",
                    markersize=7,
                    color=pattern_colors.get(pattern, "tab:blue"),
                    capsize=3,
                    alpha=0.85,
                    linewidth=1.5,
                )
                legend_handles.setdefault(pattern, handle)

        axis.plot([0.0, 1.0], [0.0, 1.0], "k--", alpha=0.5, linewidth=1.25)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Animal Switch Probability")
        axis.set_ylabel("Simulation Switch Probability")
        axis.set_title(f"{n_back} Trial{'s' if n_back > 1 else ''} Back")
        axis.grid(True, alpha=0.3)

        summary = _as_dict(panel.get("summary", {}))
        annotation_lines = [
            f"r={float(summary['correlation']):.3f}"
            if summary.get("correlation") is not None
            else "r=None",
            f"RMSE={float(summary['rmse']):.3f}"
            if summary.get("rmse") is not None
            else "RMSE=None",
            f"n={int(summary.get('n_rows', 0))}",
        ]
        axis.text(
            0.05,
            0.95,
            "\n".join(annotation_lines),
            transform=axis.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    fig.suptitle(
        f"History-Dependent Switch Comparison ({pattern_type.capitalize()} Patterns, Subject Mean +/- SEM)",
        fontsize=14,
    )
    if legend_handles:
        sorted_patterns = sorted(legend_handles)
        _add_bottom_legend_to_figure(
            fig,
            handles=[legend_handles[pattern] for pattern in sorted_patterns],
            labels=sorted_patterns,
            ncol=min(max(1, len(sorted_patterns)), 8),
            y_anchor=-0.03,
        )
        fig.tight_layout(rect=(0.0, 0.18, 1.0, 0.96))
    else:
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_history_pattern_comparison_pooled_figure(
    plt,
    history_stats: Mapping[str, Any],
    output_path: Path,
    *,
    pattern_type: str,
) -> None:
    config = _as_dict(history_stats.get("config", {}))
    aggregate_min_trials = int(
        config.get("aggregate_min_trials", _DEFAULT_HISTORY_AGGREGATE_MIN_TRIALS)
    )
    max_trials_back = int(
        config.get("max_trials_back", _DEFAULT_HISTORY_MAX_TRIALS_BACK)
    )
    comparison = _as_dict(
        _as_dict(history_stats.get("comparison", {})).get(pattern_type, {})
    )
    fig, axes = plt.subplots(1, max_trials_back, figsize=(6 * max_trials_back, 6))
    if max_trials_back == 1:
        axes = [axes]

    all_patterns = sorted(
        {
            str(row.get("pattern"))
            for n_back in range(1, max_trials_back + 1)
            for row in _select_valid_history_pattern_rows(
                list(_as_dict(comparison.get(n_back, {})).get("rows", [])),
                min_trials=aggregate_min_trials,
            )
        }
    )
    pattern_colors = _build_history_pattern_color_map(all_patterns, plt)
    legend_handles: dict[str, Any] = {}

    for axis, n_back in zip(axes, range(1, max_trials_back + 1), strict=False):
        panel = _as_dict(comparison.get(n_back, {}))
        rows = list(panel.get("rows", []))
        valid_rows = _select_valid_history_pattern_rows(rows, min_trials=aggregate_min_trials)

        if not valid_rows:
            axis.text(
                0.5,
                0.5,
                "No common patterns\nwith sufficient data",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        else:
            for row in valid_rows:
                pattern = str(row.get("pattern"))
                handle = axis.errorbar(
                    float(row["animal_probability"]),
                    float(row["simulated_probability"]),
                    xerr=row.get("animal_sem"),
                    yerr=row.get("simulated_sem"),
                    marker="o",
                    markersize=7,
                    color=pattern_colors.get(pattern, "tab:blue"),
                    capsize=3,
                    alpha=0.85,
                    linewidth=1.5,
                )
                legend_handles.setdefault(pattern, handle)

        axis.plot([0.0, 1.0], [0.0, 1.0], "k--", alpha=0.5, linewidth=1.25)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Animal Switch Probability")
        axis.set_ylabel("Simulation Switch Probability")
        axis.set_title(f"{n_back} Trial{'s' if n_back > 1 else ''} Back")
        axis.grid(True, alpha=0.3)

        summary = _as_dict(panel.get("summary", {}))
        annotation_lines = [
            f"r={float(summary['correlation']):.3f}"
            if summary.get("correlation") is not None
            else "r=None",
            f"RMSE={float(summary['rmse']):.3f}"
            if summary.get("rmse") is not None
            else "RMSE=None",
            f"n={int(summary.get('n_patterns', 0))}",
        ]
        axis.text(
            0.05,
            0.95,
            "\n".join(annotation_lines),
            transform=axis.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    fig.suptitle(
        f"History-Dependent Switch Comparison ({pattern_type.capitalize()} Patterns)",
        fontsize=14,
    )
    if legend_handles:
        sorted_patterns = sorted(legend_handles)
        _add_bottom_legend_to_figure(
            fig,
            handles=[legend_handles[pattern] for pattern in sorted_patterns],
            labels=sorted_patterns,
            ncol=min(max(1, len(sorted_patterns)), 8),
            y_anchor=-0.03,
        )
        fig.tight_layout(rect=(0.0, 0.18, 1.0, 0.96))
    else:
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_history_pattern_delta_figure(
    plt,
    history_stats: Mapping[str, Any],
    output_path: Path,
    *,
    pattern_type: str,
    show_significance: bool = True,
) -> None:
    config = _as_dict(history_stats.get("config", {}))
    max_trials_back = int(
        config.get("max_trials_back", _DEFAULT_HISTORY_MAX_TRIALS_BACK)
    )
    subject_min_trials = int(
        config.get("subject_min_trials", _DEFAULT_HISTORY_SUBJECT_MIN_TRIALS)
    )
    subject_level = _as_dict(
        _as_dict(history_stats.get("subject_level", {})).get(pattern_type, {})
    )
    delta_summary = _as_dict(
        _as_dict(history_stats.get("delta_significance_summary", {})).get(
            pattern_type,
            {},
        )
    )
    if max_trials_back == 3:
        fig = plt.figure(figsize=(13, 10))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
        axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :]),
        ]
    else:
        fig, axes = plt.subplots(1, max_trials_back, figsize=(6.5 * max_trials_back, 6))
        if max_trials_back == 1:
            axes = [axes]
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for n_back in range(1, max_trials_back + 1)
            for pattern in _as_dict(subject_level.get(n_back, {})).keys()
            for point in _select_valid_subject_history_pattern_points(
                list(
                    _as_dict(_as_dict(subject_level.get(n_back, {})).get(pattern, {})).get(
                        "points",
                        [],
                    )
                ),
                min_trials=subject_min_trials,
            )
        ],
        plt,
    )

    for axis, n_back in zip(axes, range(1, max_trials_back + 1), strict=False):
        panel_group = _as_dict(subject_level.get(n_back, {}))
        rows = _build_sorted_history_delta_rows(
            panel_group,
            min_trials=subject_min_trials,
        )
        if not rows:
            axis.text(
                0.5,
                0.5,
                "No valid subject deltas",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            axis.set_title(f"{n_back} Trial{'s' if n_back > 1 else ''} Back")
            axis.set_ylabel("Delta Probability (Simulation - Animal)")
            axis.grid(True, axis="y", alpha=0.3)
            continue
        _plot_delta_distribution_panel(
            ax=axis,
            rows=rows,
            title=_format_delta_plot_title(
                f"{n_back} Trial{'s' if n_back > 1 else ''} Back",
                _as_dict(delta_summary.get(n_back, {})) if show_significance else {},
            ),
            plt=plt,
            curriculum_to_color=curriculum_to_color,
            xtick_rotation=28.0,
            show_significance=show_significance,
        )
    fig.suptitle(
        f"History Pattern Subject Delta ({pattern_type.capitalize()} Patterns)",
        fontsize=14,
        y=0.985,
    )
    if show_significance:
        _add_delta_significance_note(fig, y=0.962)
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.93))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_history_pattern_subject_level_figure(
    plt,
    history_stats: Mapping[str, Any],
    output_path: Path,
    *,
    pattern_type: str,
    n_back: int,
) -> None:
    config = _as_dict(history_stats.get("config", {}))
    subject_min_trials = int(
        config.get("subject_min_trials", _DEFAULT_HISTORY_SUBJECT_MIN_TRIALS)
    )
    subject_level = _as_dict(
        _as_dict(history_stats.get("subject_level", {})).get(pattern_type, {})
    )
    panel_data = _as_dict(subject_level.get(n_back, {}))
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for pattern in panel_data
            for point in _select_valid_subject_history_pattern_points(
                list(_as_dict(panel_data.get(pattern, {})).get("points", [])),
                min_trials=subject_min_trials,
            )
        ],
        plt,
    )
    patterns = sorted(panel_data)

    if not patterns:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.text(
            0.5,
            0.5,
            "No matched patterns",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        fig.suptitle(
            f"Subject-Level History Patterns ({pattern_type.capitalize()}, n_back={n_back})",
            fontsize=14,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
        fig.savefig(output_path)
        plt.close(fig)
        return

    n_cols = min(4, len(patterns))
    n_rows = int(math.ceil(len(patterns) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
    )
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = list(axes.flatten())

    for axis, pattern in zip(axes_flat, patterns, strict=False):
        panel = _as_dict(panel_data.get(pattern, {}))
        _plot_history_subject_pattern_scatter_panel(
            ax=axis,
            points=list(panel.get("points", [])),
            summary=_as_dict(panel.get("summary", {})),
            title=str(pattern),
            min_trials=subject_min_trials,
            curriculum_to_color=curriculum_to_color,
        )

    for axis in axes_flat[len(patterns) :]:
        axis.set_visible(False)

    fig.suptitle(
        f"Subject-Level History Patterns ({pattern_type.capitalize()}, n_back={n_back})",
        fontsize=14,
    )
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_history_pattern_session_level_figure(
    plt,
    history_stats: Mapping[str, Any],
    output_path: Path,
    *,
    pattern_type: str,
    n_back: int,
) -> None:
    config = _as_dict(history_stats.get("config", {}))
    session_min_trials = int(
        config.get("subject_min_trials", _DEFAULT_HISTORY_SUBJECT_MIN_TRIALS)
    )
    session_level = _as_dict(
        _as_dict(history_stats.get("session_level", {})).get(pattern_type, {})
    )
    panel_data = _as_dict(session_level.get(n_back, {}))
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for pattern in panel_data
            for point in _select_valid_session_history_pattern_points(
                list(_as_dict(panel_data.get(pattern, {})).get("points", [])),
                min_trials=session_min_trials,
            )
        ],
        plt,
    )
    patterns = sorted(panel_data)

    if not patterns:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.text(
            0.5,
            0.5,
            "No matched patterns",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        fig.suptitle(
            f"Session-Level History Patterns ({pattern_type.capitalize()}, n_back={n_back})",
            fontsize=14,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
        fig.savefig(output_path)
        plt.close(fig)
        return

    n_cols = min(4, len(patterns))
    n_rows = int(math.ceil(len(patterns) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
    )
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = list(axes.flatten())

    for axis, pattern in zip(axes_flat, patterns, strict=False):
        panel = _as_dict(panel_data.get(pattern, {}))
        _plot_history_session_pattern_scatter_panel(
            ax=axis,
            points=list(panel.get("points", [])),
            summary=_as_dict(panel.get("summary", {})),
            title=str(pattern),
            min_trials=session_min_trials,
            curriculum_to_color=curriculum_to_color,
        )

    for axis in axes_flat[len(patterns) :]:
        axis.set_visible(False)

    fig.suptitle(
        f"Session-Level History Patterns ({pattern_type.capitalize()}, n_back={n_back})",
        fontsize=14,
    )
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_history_pattern_session_level_figure_for_subject(
    plt,
    history_stats: Mapping[str, Any],
    output_path: Path,
    *,
    pattern_type: str,
    n_back: int,
    subject_id: Any,
) -> None:
    config = _as_dict(history_stats.get("config", {}))
    session_min_trials = int(
        config.get("subject_min_trials", _DEFAULT_HISTORY_SUBJECT_MIN_TRIALS)
    )
    session_level = _as_dict(
        _as_dict(history_stats.get("session_level", {})).get(pattern_type, {})
    )
    panel_data = _as_dict(session_level.get(n_back, {}))
    curriculum_to_color = _build_curriculum_color_map(
        [
            point.get("curriculum_name", "Unknown")
            for pattern in panel_data
            for point in _select_valid_session_history_pattern_points(
                _filter_points_for_subject(
                    list(_as_dict(panel_data.get(pattern, {})).get("points", [])),
                    subject_id=subject_id,
                ),
                min_trials=session_min_trials,
            )
        ],
        plt,
    )
    patterns = sorted(panel_data)

    if not patterns:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.text(
            0.5,
            0.5,
            "No matched patterns",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        fig.suptitle(
            f"Session-Level History Patterns ({pattern_type.capitalize()}, n_back={n_back}, {subject_id})",
            fontsize=14,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        return

    n_cols = min(4, len(patterns))
    n_rows = int(math.ceil(len(patterns) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
    )
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = list(axes.flatten())

    for axis, pattern in zip(axes_flat, patterns, strict=False):
        filtered_points = _filter_points_for_subject(
            list(_as_dict(panel_data.get(pattern, {})).get("points", [])),
            subject_id=subject_id,
        )
        _plot_history_session_pattern_scatter_panel(
            ax=axis,
            points=filtered_points,
            summary=_summarize_session_history_pattern_points(
                filtered_points,
                min_trials=session_min_trials,
            ),
            title=str(pattern),
            min_trials=session_min_trials,
            curriculum_to_color=curriculum_to_color,
        )

    for axis in axes_flat[len(patterns) :]:
        axis.set_visible(False)

    fig.suptitle(
        f"Session-Level History Patterns ({pattern_type.capitalize()}, n_back={n_back}, {subject_id})",
        fontsize=14,
    )
    _add_curriculum_legend_to_figure(fig, curriculum_to_color)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_history_subject_pattern_scatter_panel(
    *,
    ax,
    points: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    title: str,
    min_trials: int,
    curriculum_to_color: Mapping[str, Any],
) -> None:
    _plot_probability_scatter_panel(
        ax=ax,
        points=points,
        summary=summary,
        title=title,
        curriculum_to_color=curriculum_to_color,
        selector=lambda rows: _select_valid_subject_history_pattern_points(
            rows,
            min_trials=min_trials,
        ),
        count_label="n_subjects",
        x_label="Animal Switch Probability",
        y_label="Simulation Switch Probability",
        empty_label="No valid subjects",
    )


def _plot_history_session_pattern_scatter_panel(
    *,
    ax,
    points: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    title: str,
    min_trials: int,
    curriculum_to_color: Mapping[str, Any],
) -> None:
    _plot_probability_scatter_panel(
        ax=ax,
        points=points,
        summary=summary,
        title=title,
        curriculum_to_color=curriculum_to_color,
        selector=lambda rows: _select_valid_session_history_pattern_points(
            rows,
            min_trials=min_trials,
        ),
        count_label="n_sessions",
        x_label="Animal Switch Probability",
        y_label="Simulation Switch Probability",
        empty_label="No valid sessions",
    )


def _plot_delta_distribution_panel(
    *,
    ax,
    rows: Sequence[Mapping[str, Any]],
    title: str,
    plt,
    curriculum_to_color: Mapping[str, Any],
    xtick_rotation: float | None = None,
    show_significance: bool = True,
) -> None:
    np = _import_dependency("numpy")
    rng = np.random.default_rng(0)
    x_positions = list(range(len(rows)))
    annotation_specs = []
    finite_values: list[float] = [0.0]
    for x_position, row in zip(x_positions, rows, strict=False):
        valid_points = list(row.get("points", []))
        deltas = _extract_point_delta_probabilities(valid_points)
        if len(deltas) >= 2:
            violin = ax.violinplot(
                [deltas],
                positions=[x_position],
                widths=0.7,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor("0.8")
                body.set_edgecolor("0.6")
                body.set_alpha(0.45)
            summary = _compute_violin_summary_stats(deltas)
            ax.vlines(
                x_position,
                summary["whisker_low"],
                summary["whisker_high"],
                color="black",
                linewidth=1.0,
                zorder=3,
            )
            ax.vlines(
                x_position,
                summary["q1"],
                summary["q3"],
                color="black",
                linewidth=3.2,
                zorder=4,
            )
            ax.scatter(
                [x_position],
                [summary["median"]],
                color="white",
                edgecolors="black",
                linewidths=1.0,
                s=28,
                zorder=5,
            )
        if valid_points:
            jitter = rng.uniform(-0.12, 0.12, size=len(valid_points))
            for point, x_offset in zip(valid_points, jitter, strict=False):
                curriculum_name = str(point.get("curriculum_name", "Unknown"))
                color = curriculum_to_color.get(curriculum_name, "tab:blue")
                ax.scatter(
                    [x_position + float(x_offset)],
                    [float(point["delta_probability"])],
                    color=color,
                    alpha=0.35,
                    s=34,
                    edgecolors="black",
                    linewidths=0.5,
                )
        if deltas:
            finite_values.extend(deltas)
            if show_significance:
                wilcoxon_result = _wilcoxon_signed_rank_against_zero(deltas)
                label = _format_significance_label(wilcoxon_result.get("p_value"))
                if label:
                    annotation_specs.append(
                        {
                            "x": x_position,
                            "y": max(max(deltas), 0.0),
                            "label": label,
                        }
                    )

    ax.axhline(0.0, color="black", linestyle="--", alpha=0.5, linewidth=1.2)
    _set_delta_distribution_ylim(
        ax,
        finite_values,
        annotation_space=show_significance and bool(annotation_specs),
    )
    if show_significance:
        _annotate_delta_significance(ax, annotation_specs)
    ax.set_xticks(x_positions)
    rotation = (
        20.0
        if xtick_rotation is None and len(rows) > 3
        else float(xtick_rotation or 0.0)
    )
    ax.set_xticklabels(
        [str(row.get("label", "")) for row in rows],
        rotation=rotation,
        ha="right" if rotation else "center",
    )
    ax.set_ylabel("Delta Probability (Simulation - Animal)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def _build_sorted_history_delta_rows(
    panel_group: Mapping[str, Any],
    *,
    min_trials: int,
) -> list[dict[str, Any]]:
    rows = []
    for pattern in panel_group:
        valid_points = _select_valid_subject_history_pattern_points(
            list(_as_dict(panel_group.get(pattern, {})).get("points", [])),
            min_trials=min_trials,
        )
        if not valid_points:
            continue
        delta_probabilities = _extract_point_delta_probabilities(valid_points)
        delta_median = _median(delta_probabilities) if delta_probabilities else math.nan
        rows.append(
            {
                "label": str(pattern),
                "points": valid_points,
                "delta_median": delta_median,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            math.isnan(float(row["delta_median"])),
            -float(row["delta_median"])
            if not math.isnan(float(row["delta_median"]))
            else 0.0,
            str(row["label"]),
        ),
    )


def _extract_point_delta_probabilities(
    points: Sequence[Mapping[str, Any]],
) -> list[float]:
    deltas = []
    for point in points:
        delta_probability = _extract_point_delta_probability(point)
        if delta_probability is None:
            continue
        deltas.append(float(delta_probability))
    return deltas


def _extract_point_delta_probability(point: Mapping[str, Any]) -> float | None:
    delta_probability = _coerce_probability(point.get("delta_probability"))
    if delta_probability is None:
        delta_probability = _finite_difference(
            _coerce_probability(point.get("simulated_probability")),
            _coerce_probability(point.get("animal_probability")),
        )
    if delta_probability is None:
        return None
    return float(delta_probability)


def _wilcoxon_signed_rank_against_zero(values: Sequence[float]) -> dict[str, Any]:
    finite_values = [float(value) for value in values if not _is_nan(value)]
    nonzero_values = [value for value in finite_values if value != 0.0]
    if not nonzero_values:
        return {
            "n_nonzero": 0,
            "statistic": None,
            "p_value": None,
            "method": None,
        }

    try:
        scipy_stats = importlib.import_module("scipy.stats")
        result = scipy_stats.wilcoxon(
            nonzero_values,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        return {
            "n_nonzero": len(nonzero_values),
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "scipy",
        }
    except ModuleNotFoundError:
        pass

    absolute_values = [abs(value) for value in nonzero_values]
    ranks = _average_ranks(absolute_values)
    w_plus = sum(rank for value, rank in zip(nonzero_values, ranks) if value > 0.0)
    w_minus = sum(rank for value, rank in zip(nonzero_values, ranks) if value < 0.0)
    statistic = min(w_plus, w_minus)
    has_ties = len(set(absolute_values)) != len(absolute_values)

    if len(nonzero_values) <= 20 and not has_ties:
        p_value = _wilcoxon_exact_two_sided_p_value(
            statistic=statistic,
            n_nonzero=len(nonzero_values),
        )
        method = "exact"
    else:
        p_value = _wilcoxon_normal_approximation_p_value(
            nonzero_values=nonzero_values,
            absolute_values=absolute_values,
            ranks=ranks,
            w_plus=w_plus,
        )
        method = "normal_approx"

    return {
        "n_nonzero": len(nonzero_values),
        "statistic": float(statistic),
        "p_value": p_value,
        "method": method,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered_pairs = sorted(
        enumerate(float(value) for value in values),
        key=lambda item: item[1],
    )
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered_pairs):
        start = index
        current_value = ordered_pairs[index][1]
        while index < len(ordered_pairs) and ordered_pairs[index][1] == current_value:
            index += 1
        average_rank = ((start + 1) + index) / 2.0
        for original_index, _ in ordered_pairs[start:index]:
            ranks[original_index] = float(average_rank)
    return ranks


def _wilcoxon_exact_two_sided_p_value(
    *,
    statistic: float,
    n_nonzero: int,
) -> float | None:
    if n_nonzero <= 0:
        return None
    total_rank_sum = n_nonzero * (n_nonzero + 1) // 2
    counts: dict[int, int] = {0: 1}
    for rank in range(1, n_nonzero + 1):
        updated = dict(counts)
        for existing_sum, count in counts.items():
            updated[existing_sum + rank] = updated.get(existing_sum + rank, 0) + count
        counts = updated
    tail_count = sum(
        count
        for rank_sum, count in counts.items()
        if min(rank_sum, total_rank_sum - rank_sum) <= statistic + 1e-12
    )
    return float(min(1.0, tail_count / float(2**n_nonzero)))


def _wilcoxon_normal_approximation_p_value(
    *,
    nonzero_values: Sequence[float],
    absolute_values: Sequence[float],
    ranks: Sequence[float],
    w_plus: float,
) -> float | None:
    n_nonzero = len(nonzero_values)
    if n_nonzero == 0:
        return None
    expected = n_nonzero * (n_nonzero + 1) / 4.0
    tie_counts: dict[float, int] = {}
    for value in absolute_values:
        tie_counts[float(value)] = tie_counts.get(float(value), 0) + 1
    tie_correction = sum((count**3) - count for count in tie_counts.values())
    variance = (
        n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0
        - tie_correction / 48.0
    )
    if variance <= 0:
        return None
    z = (abs(w_plus - expected) - 0.5) / math.sqrt(variance)
    return float(min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))


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


def _format_p_value(p_value: float | None) -> str:
    if p_value is None:
        return "n/a"
    if p_value < 0.001:
        return "<=0.001"
    if p_value < 0.01:
        return f"{float(p_value):.4f}"
    return f"{float(p_value):.3f}"


def _set_delta_distribution_ylim(
    ax,
    finite_values: Sequence[float],
    *,
    annotation_space: bool,
) -> None:
    finite = [float(value) for value in finite_values if not _is_nan(value)]
    if not finite:
        return
    y_min = min(finite)
    y_max = max(finite)
    scale = max(y_max - y_min, abs(y_min), abs(y_max), 0.2)
    lower_pad = 0.08 * scale
    upper_pad = (0.2 if annotation_space else 0.08) * scale
    ax.set_ylim(y_min - lower_pad, y_max + upper_pad)


def _annotate_delta_significance(
    ax,
    annotations: Sequence[Mapping[str, Any]],
) -> None:
    if not annotations:
        return
    y_min, y_max = ax.get_ylim()
    scale = max(y_max - y_min, abs(y_min), abs(y_max), 0.2)
    label_offset = 0.06 * scale
    for annotation in annotations:
        if not str(annotation.get("label", "")):
            continue
        ax.text(
            float(annotation["x"]),
            float(annotation["y"]) + label_offset,
            str(annotation["label"]),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def _add_delta_significance_note(fig, *, y: float = 0.985) -> None:
    fig.text(
        0.5,
        y,
        "Stars: two-sided Wilcoxon signed-rank test vs 0",
        ha="center",
        va="top",
        fontsize=9,
    )


def _format_delta_plot_title(
    base_title: str,
    delta_summary: Mapping[str, Any],
) -> str:
    subject_balanced_summary = _as_dict(
        _as_dict(delta_summary).get("subject_balanced_error_summary", {})
    )
    subject_balanced_line = _format_delta_error_summary_line(
        "Subj-bal",
        subject_balanced_summary,
    )
    summary_lines = [
        line
        for line in (
            subject_balanced_line,
        )
        if line
    ]
    if not summary_lines:
        return base_title
    return "\n".join([base_title, *summary_lines])


def _format_delta_error_summary_line(
    prefix: str,
    error_summary: Mapping[str, Any],
) -> str:
    mean_signed_error = error_summary.get("mean_signed_error")
    mean_signed_error_sem = error_summary.get("mean_signed_error_sem")
    p_value = error_summary.get("p_value")
    mean_absolute_error = error_summary.get("mean_absolute_error")
    mean_squared_error = error_summary.get("mean_squared_error")
    if (
        mean_signed_error is None
        or mean_signed_error_sem is None
        or mean_absolute_error is None
        or mean_squared_error is None
    ):
        return ""
    return (
        f"{prefix}: ME={float(mean_signed_error):.3f} +/- "
        f"{float(mean_signed_error_sem):.3f}, "
        f"p={_format_p_value(p_value)}, "
        f"MAE={float(mean_absolute_error):.3f}, "
        f"MSE={float(mean_squared_error):.3e}"
    )


def _compute_violin_summary_stats(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "whisker_low": math.nan,
            "whisker_high": math.nan,
        }

    q1 = float(_quantile(ordered, 0.25))
    median = float(_median(ordered))
    q3 = float(_quantile(ordered, 0.75))
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    whisker_low = min(value for value in ordered if value >= lower_bound)
    whisker_high = max(value for value in ordered if value <= upper_bound)
    return {
        "median": median,
        "q1": q1,
        "q3": q3,
        "whisker_low": float(whisker_low),
        "whisker_high": float(whisker_high),
    }


def _asymmetric_errorbar(
    center: float,
    ci_low: Any,
    ci_high: Any,
) -> list[list[float]] | None:
    lower = _coerce_probability(ci_low)
    upper = _coerce_probability(ci_high)
    if lower is None or upper is None:
        return None
    return [[max(0.0, center - lower)], [max(0.0, upper - center)]]


def _build_curriculum_color_map(curricula: Sequence[Any], plt) -> dict[str, Any]:
    normalized = [
        "Unknown" if curriculum is None or _is_nan(curriculum) else str(curriculum)
        for curriculum in curricula
    ]
    unique_curricula = list(dict.fromkeys(normalized))
    if not unique_curricula:
        return {}
    cmap = plt.get_cmap("tab10")
    return {
        curriculum: cmap(index % max(1, cmap.N))
        for index, curriculum in enumerate(unique_curricula)
    }


def _add_bottom_legend_to_figure(
    fig,
    *,
    handles: Sequence[Any],
    labels: Sequence[str],
    title: str | None = None,
    ncol: int = 1,
    y_anchor: float = 0.01,
    fontsize: int = 9,
) -> None:
    if not handles or not labels:
        return
    fig.legend(
        handles=handles,
        labels=list(labels),
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=max(1, int(ncol)),
        fontsize=fontsize,
        frameon=True,
        title=title,
    )


def _add_curriculum_legend_to_figure(fig, curriculum_to_color: Mapping[str, Any]) -> None:
    if not curriculum_to_color:
        return
    line2d = importlib.import_module("matplotlib.lines").Line2D
    handles = [
        line2d(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            label=curriculum,
        )
        for curriculum, color in curriculum_to_color.items()
    ]
    _add_bottom_legend_to_figure(
        fig,
        handles=handles,
        labels=[str(curriculum) for curriculum in curriculum_to_color],
        title="Curriculum",
        ncol=min(max(1, len(handles)), 6),
    )


def _build_history_pattern_color_map(
    patterns: Sequence[str],
    plt,
) -> dict[str, Any]:
    if not patterns:
        return {}
    np = _import_dependency("numpy")
    colors = list(plt.cm.tab20(np.linspace(0, 1, min(20, len(patterns)))))
    if len(patterns) > len(colors):
        colors.extend(
            plt.cm.tab20b(np.linspace(0, 1, min(20, len(patterns) - len(colors))))
        )
    if len(patterns) > len(colors):
        colors.extend(
            plt.cm.tab20c(np.linspace(0, 1, len(patterns) - len(colors)))
        )
    return {
        pattern: colors[index % len(colors)]
        for index, pattern in enumerate(patterns)
    }


def _derive_session_seed(seed: int | None, session_id: str, *, rollout_index: int) -> int:
    base = "0" if seed is None else str(seed)
    payload = f"{base}:{session_id}:{rollout_index}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def derive_session_seed(
    seed: int | None,
    session_id: str,
    *,
    rollout_index: int,
) -> int:
    """Public wrapper for deterministic per-session rollout seeds."""

    return _derive_session_seed(
        seed,
        session_id,
        rollout_index=rollout_index,
    )


def _softmax(logits: Sequence[float]) -> list[float]:
    values = [float(v) for v in logits]
    max_value = max(values)
    exp_values = [math.exp(v - max_value) for v in values]
    denom = sum(exp_values)
    if denom <= 0:
        raise ValueError(f"Invalid logits for softmax: {logits!r}")
    return [value / denom for value in exp_values]


def _clean_choice_history(choices: Sequence[Any]) -> list[int]:
    normalized = []
    for value in choices:
        coerced = _normalize_choice_value(value)
        if _is_nan(coerced):
            continue
        normalized.append(int(coerced))
    return normalized


def _aligned_choice_reward_histories(
    choices: Sequence[Any],
    rewards: Sequence[Any],
) -> tuple[list[int], list[float]]:
    cleaned_choices: list[int] = []
    cleaned_rewards: list[float] = []
    max_len = min(len(choices), len(rewards))
    for idx in range(max_len):
        choice = _normalize_choice_value(choices[idx])
        reward = _normalize_reward_value(rewards[idx])
        if _is_nan(choice):
            continue
        cleaned_choices.append(int(choice))
        cleaned_rewards.append(float(reward))
    return cleaned_choices, cleaned_rewards


def _build_switch_window(choices: Sequence[int], switch_idx: int, window_size: int) -> list[float]:
    switch_events = [math.nan] * (2 * window_size + 1)
    start_idx = max(0, switch_idx - window_size)
    end_idx = min(len(choices), switch_idx + window_size + 1)
    for trial_idx in range(start_idx, end_idx):
        if trial_idx == 0:
            continue
        rel_pos = trial_idx - switch_idx
        arr_idx = rel_pos + window_size
        if 0 <= arr_idx < len(switch_events):
            switch_events[arr_idx] = float(choices[trial_idx] != choices[trial_idx - 1])
    return switch_events


def _summarize_post_switch_probability(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not events:
        return {"probability": math.nan, "sem": math.nan, "n": 0}

    window_size = int(events[0].get("window_size", 0))
    post_switch_idx = window_size + 1
    values = []
    for event in events:
        switch_events = list(event.get("switch_events", []))
        if post_switch_idx >= len(switch_events):
            continue
        value = switch_events[post_switch_idx]
        if _is_nan(value):
            continue
        values.append(float(value))

    if not values:
        return {"probability": math.nan, "sem": math.nan, "n": 0}
    return {
        "probability": _mean(values),
        "sem": _sem(values),
        "n": len(values),
    }


def _iter_session_records(rows) -> Iterator[dict[str, Any]]:
    if rows is None:
        return iter(())

    if hasattr(rows, "to_dict") and hasattr(rows, "columns"):
        for record in rows.to_dict(orient="records"):
            yield dict(record)
        return

    for record in rows:
        if isinstance(record, Mapping):
            yield dict(record)
        else:
            raise TypeError(
                "Session rows must be a pandas DataFrame or an iterable of mappings."
            )


def _restore_input_container(original, records: list[dict[str, Any]]):
    if hasattr(original, "to_dict") and hasattr(original, "columns"):
        pd = _import_dependency("pandas")
        return pd.DataFrame.from_records(records)
    return records


def _resolve_serialized_session_conditioning_state(
    *,
    model_type: Any,
    model_config: Mapping[str, Any] | None,
    run_config: Mapping[str, Any] | None,
    multisubject: bool,
) -> tuple[bool, str]:
    normalized_model_type = str(model_type or "").strip().lower()
    model_config_dict = _as_dict(model_config)
    run_model_cfg = _as_dict(_as_dict(run_config).get("model", {}))
    run_architecture = _as_dict(run_model_cfg.get("architecture", {}))

    encoding_type_source: Any = "none"
    if normalized_model_type == "gru":
        encoding_type_source = _as_dict(model_config_dict.get("architecture", {})).get(
            "session_encoding_type",
            run_architecture.get("session_encoding_type", "none"),
        )
    elif normalized_model_type == "disrnn":
        encoding_type_source = model_config_dict.get("session_encoding_type")
        if encoding_type_source is None:
            encoding_type_source = run_architecture.get("session_encoding_type", "none")

    encoding_type = str(encoding_type_source or "none").strip().lower()
    return bool(multisubject and encoding_type != "none"), encoding_type


def _required_session_conditioning_artifact_names(
    resolved_run: ResolvedModelRun,
) -> list[str]:
    if not bool(resolved_run.multisubject) or not bool(
        resolved_run.session_conditioning_enabled
    ):
        return []
    return ["subject_index_map.json", "session_context_map.json"]


def _missing_session_conditioning_artifact_names(
    resolved_run: ResolvedModelRun,
) -> list[str]:
    required_artifacts = _required_session_conditioning_artifact_names(resolved_run)
    if not required_artifacts:
        return []

    missing_artifacts: list[str] = []
    artifact_paths = {
        "subject_index_map.json": resolved_run.subject_index_map_path,
        "session_context_map.json": resolved_run.session_context_map_path,
    }
    for artifact_name in required_artifacts:
        artifact_path = artifact_paths.get(artifact_name)
        if not artifact_path or not Path(artifact_path).exists():
            missing_artifacts.append(artifact_name)
    return missing_artifacts


def _refresh_resolved_run_session_conditioning_metadata(
    resolved_run: ResolvedModelRun,
) -> ResolvedModelRun:
    (
        session_conditioning_enabled,
        session_conditioning_encoding_type,
    ) = _resolve_serialized_session_conditioning_state(
        model_type=resolved_run.model_type,
        model_config=resolved_run.model_config,
        run_config=resolved_run.run_config,
        multisubject=bool(resolved_run.multisubject),
    )
    resolved_run.session_conditioning_enabled = bool(session_conditioning_enabled)
    resolved_run.session_conditioning_encoding_type = str(
        session_conditioning_encoding_type
    )
    resolved_run.required_session_conditioning_artifacts = (
        _required_session_conditioning_artifact_names(resolved_run)
    )
    resolved_run.missing_session_conditioning_artifacts = (
        _missing_session_conditioning_artifact_names(resolved_run)
    )
    return resolved_run


def _coerce_resolved_run(
    value: str | Path | ResolvedModelRun | Mapping[str, Any],
    *,
    split: str | None = None,
) -> ResolvedModelRun:
    if isinstance(value, ResolvedModelRun):
        return _refresh_resolved_run_session_conditioning_metadata(value)
    if isinstance(value, Mapping):
        return _refresh_resolved_run_session_conditioning_metadata(
            ResolvedModelRun(**dict(value))
        )
    return _refresh_resolved_run_session_conditioning_metadata(
        resolve_model_run(value, split=split or _TRAIN_SPLIT)
    )


def _json_tree_to_arrays(value: Any) -> Any:
    np = _import_dependency("numpy")
    if isinstance(value, list):
        converted = [_json_tree_to_arrays(item) for item in value]
        try:
            array_value = np.asarray(converted)
            if getattr(array_value, "dtype", None) != object:
                return array_value
        except Exception:
            pass
        return converted
    if isinstance(value, dict):
        return {key: _json_tree_to_arrays(item) for key, item in value.items()}
    return value


def _safe_filename_component(value: Any) -> str:
    normalized = str(value)
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in normalized
    )
    return safe.strip("_") or "unknown"


def _import_dependency(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Required dependency '{module_name}' is not available in the current "
            "Python environment. This module is expected to run inside the "
            "Code Ocean runtime described by environment/Dockerfile."
        ) from exc


def _first_existing_column(columns: Iterable[Any], candidates: Sequence[str]) -> str | None:
    normalized = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return str(normalized[candidate.lower()])
    return None


def _extract_session_date(ses_idx: Any) -> str | None:
    if ses_idx in (None, ""):
        return None
    parts = str(ses_idx).split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def _normalize_choice_value(value: Any) -> float:
    if value is None or _is_nan(value):
        return math.nan
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"left", "l"}:
            return 0.0
        if stripped in {"right", "r"}:
            return 1.0
        if stripped in {"none", "nan", ""}:
            return math.nan
        try:
            value = float(stripped)
        except ValueError:
            return math.nan
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return math.nan
    if value_f == 2.0:
        return math.nan
    return value_f


def _normalize_reward_value(value: Any) -> float:
    if value is None or _is_nan(value):
        return 0.0
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "rewarded"}:
            return 1.0
        if stripped in {"false", "unrewarded", ""}:
            return 0.0
        try:
            return float(stripped)
        except ValueError:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first_non_null(series_like: Any) -> Any:
    if series_like is None:
        return None
    for value in list(series_like):
        if value is None or _is_nan(value):
            continue
        return value
    return None


def _series_like_to_list(rows: Any, column_name: str) -> list[Any] | None:
    # `rows` is DataFrame-shaped (rows[column_name] is a column) for most
    # callers, but _align_multisubject_session_history_to_training_ids always
    # returns a plain list[dict] of row records regardless of its input type
    # (see its own `aligned_records: list[dict[str, Any]]`), which rows[...]
    # can't index. Handle that shape explicitly rather than silently returning
    # None -- the caller (load_animal_session_history) has no fallback for
    # resolved_session_ids, so this previously left it unset for every
    # multisubject run that goes through session-context alignment.
    if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows):
        return [row.get(column_name) for row in rows]
    try:
        column = rows[column_name]
    except (KeyError, TypeError, AttributeError):
        return None
    if hasattr(column, "tolist"):
        return list(column.tolist())
    try:
        return list(column)
    except TypeError:
        return None


def _validate_multisubject_analysis_split(resolved_run: ResolvedModelRun) -> None:
    if resolved_run.multisubject and resolved_run.split != _TRAIN_SPLIT:
        raise NotImplementedError(
            "Held-out post-training analysis is not supported for multisubject "
            "runs. V1 supports seen-subject personalization only."
        )


def _load_trained_subject_ids_from_subject_index_map(path: str | Path) -> list[Any]:
    with Path(path).open("r") as f:
        payload = json.load(f)

    index_to_subject_id_raw = payload.get("index_to_subject_id", {})
    if index_to_subject_id_raw:
        ordered_pairs = sorted(
            (
                (int(index), _normalize_identifier(subject_id))
                for index, subject_id in index_to_subject_id_raw.items()
            ),
            key=lambda item: item[0],
        )
        return [subject_id for _, subject_id in ordered_pairs]

    subject_id_to_index_raw = payload.get("subject_id_to_index", {})
    if subject_id_to_index_raw:
        ordered_pairs = sorted(
            (
                (int(index), _normalize_identifier(subject_id))
                for subject_id, index in subject_id_to_index_raw.items()
            ),
            key=lambda item: item[0],
        )
        return [subject_id for _, subject_id in ordered_pairs]

    return []


def _load_subject_index_maps_for_run(
    resolved_run: ResolvedModelRun,
) -> tuple[dict[Any, int], dict[int, Any]]:
    if not resolved_run.subject_index_map_path:
        raise FileNotFoundError(
            "Multisubject post-training analysis requires outputs/subject_index_map.json."
        )

    multisubject_mod = importlib.import_module("utils.multisubject")
    subject_id_to_index_raw, index_to_subject_id_raw = multisubject_mod.load_subject_index_map(
        resolved_run.subject_index_map_path
    )
    subject_id_to_index = {
        _normalize_identifier(subject_id): int(subject_index)
        for subject_id, subject_index in subject_id_to_index_raw.items()
    }
    index_to_subject_id = {
        int(subject_index): _normalize_identifier(subject_id)
        for subject_index, subject_id in index_to_subject_id_raw.items()
    }
    if not subject_id_to_index or not index_to_subject_id:
        raise ValueError(
            "Saved subject_index_map.json is empty or malformed for multisubject "
            f"run {resolved_run.model_dir}."
        )
    return subject_id_to_index, index_to_subject_id


def _require_multisubject_analysis_artifacts(
    resolved_run: ResolvedModelRun,
) -> None:
    missing_artifacts: list[str] = []
    if not resolved_run.subject_index_map_path or not Path(
        resolved_run.subject_index_map_path
    ).exists():
        missing_artifacts.append("subject_index_map.json")
    if bool(resolved_run.session_conditioning_enabled) and (
        not resolved_run.session_context_map_path
        or not Path(resolved_run.session_context_map_path).exists()
    ):
        missing_artifacts.append("session_context_map.json")

    if not missing_artifacts:
        return

    joined_artifacts = ", ".join(f"outputs/{name}" for name in missing_artifacts)
    if bool(resolved_run.session_conditioning_enabled):
        raise FileNotFoundError(
            "Session-conditioned multisubject post-training analysis requires "
            f"{joined_artifacts} for {resolved_run.model_dir}."
        )
    raise FileNotFoundError(
        "Multisubject post-training analysis requires "
        f"{joined_artifacts} for {resolved_run.model_dir}."
    )


def _session_max_index_by_subject_index_from_session_context(
    session_context: Mapping[str, Any],
    *,
    n_subjects: int,
) -> list[int]:
    session_max_index_by_subject_index = [0] * max(1, int(n_subjects))
    for row in session_context.get("per_subject", []):
        subject_index = int(row.get("subject_index"))
        ordered_session_ids = row.get("ordered_session_ids") or []
        if subject_index < 0 or subject_index >= len(session_max_index_by_subject_index):
            raise ValueError(
                "session_context_map.json contains an out-of-range subject_index: "
                f"{subject_index}"
            )
        session_max_index_by_subject_index[subject_index] = int(len(ordered_session_ids))
    return session_max_index_by_subject_index


def _load_multisubject_analysis_context(
    resolved_run: ResolvedModelRun,
) -> tuple[
    dict[Any, int],
    dict[int, Any],
    dict[tuple[Any, str], int],
    dict[tuple[Any, str], int],
    list[int],
]:
    resolved_run = _refresh_resolved_run_session_conditioning_metadata(resolved_run)
    _require_multisubject_analysis_artifacts(resolved_run)

    subject_id_to_index, index_to_subject_id = _load_subject_index_maps_for_run(resolved_run)
    merged_session_index_lookup: dict[tuple[Any, str], int] = {}
    source_session_index_lookup: dict[tuple[Any, str], int] = {}
    session_max_index_by_subject_index: list[int] = []
    if not bool(resolved_run.session_conditioning_enabled):
        return (
            subject_id_to_index,
            index_to_subject_id,
            merged_session_index_lookup,
            source_session_index_lookup,
            session_max_index_by_subject_index,
        )

    multisubject_mod = importlib.import_module("utils.multisubject")
    session_context = multisubject_mod.load_session_context_map(
        resolved_run.session_context_map_path
    )
    (
        merged_session_index_lookup,
        source_session_index_lookup,
    ) = multisubject_mod.subject_session_index_lookup_from_session_context(session_context)
    session_max_index_by_subject_index = _session_max_index_by_subject_index_from_session_context(
        session_context,
        n_subjects=len(index_to_subject_id),
    )
    return (
        subject_id_to_index,
        index_to_subject_id,
        merged_session_index_lookup,
        source_session_index_lookup,
        session_max_index_by_subject_index,
    )


def _resolve_session_index_for_subject(
    *,
    subject_id: Any,
    session_id: Any,
    source_session_id: Any | None = None,
    merged_session_index_lookup: Mapping[tuple[Any, str], int] | None = None,
    source_session_index_lookup: Mapping[tuple[Any, str], int] | None = None,
) -> int | None:
    merged_lookup = dict(merged_session_index_lookup or {})
    source_lookup = dict(source_session_index_lookup or {})
    if not merged_lookup and not source_lookup:
        return None

    normalized_subject_id = _normalize_identifier(subject_id)
    if not _is_missing_session_identifier(source_session_id):
        source_key = (normalized_subject_id, str(source_session_id))
        if source_key in source_lookup:
            return int(source_lookup[source_key])

    merged_key = (normalized_subject_id, str(session_id))
    if merged_key in merged_lookup:
        return int(merged_lookup[merged_key])

    raise ValueError(
        "Multisubject post-training analysis could not resolve a session index for "
        f"subject_id={normalized_subject_id!r}, session_id={session_id!r}, "
        f"source_session_id={source_session_id!r}."
    )


def _validate_multisubject_params_against_subject_map(
    params: Mapping[str, Any],
    *,
    n_subjects: int,
) -> None:
    multisubject_mod = importlib.import_module("utils.multisubject")
    subject_embeddings = multisubject_mod.extract_subject_embeddings_from_params(dict(params))
    if int(subject_embeddings.shape[0]) < int(n_subjects):
        raise ValueError(
            "Saved multisubject params do not have enough subject-embedding rows for "
            f"the resolved subject_index_map.json roster: "
            f"rows={int(subject_embeddings.shape[0])}, subjects={int(n_subjects)}."
        )


def _fill_offcurriculum_curriculum_name(snapshot_df):
    """Give off-curriculum sessions a real task family instead of a silent default.

    Some mice were trained OFF-CURRICULUM and carry no ``curriculum_name``. In the database that
    absence shows up in three different shapes -- a true null (``None``/``NaN``) and, for most
    rows, the LITERAL STRING ``"None"``. Two things went wrong with that:

    1. ``_build_curriculum_matched_task`` guards with ``curriculum_name is not None``, which
       catches ``None`` but NOT ``float("nan")`` -- so a null that arrives as NaN produces
       ``str(nan) == "nan"``, matches no family, and RAISES. That killed 13/15 runs of
       studies/05-disrnn-scaling-law.
    2. Worse, the rows it does NOT raise on (the string ``"None"``) fall into its
       ``norm in ("", "none")`` branch, which SILENTLY rolls the session out as a default
       ``UncoupledBlockTask(reward_baiting=True)`` -- even when the animal actually ran
       *Coupled Baiting* or *Coupled Without Baiting*. In one D=10 cohort that is 42/249 sessions
       (17%) simulated with the wrong task family, without a single warning.

    So the crash was the loud version of a bug that was already there quietly.

    Fix both by resolving the family from data we already have. Preference order:

    * ``curriculum_name`` when it is a real curriculum -- authoritative, so on-curriculum sessions
      are untouched and prior results stay valid;
    * else the SESSION's own ``task`` -- this is a per-session, curriculum-MATCHED rollout, and the
      task is what the animal actually did in that session. (Note a curriculum can span several
      tasks across stages, so the session's task is strictly better here than any per-subject label);
    * else the SUBJECT's most-common ``task`` -- the same pseudo-curriculum
      ``load_mice_database._partition_subjects`` step 3 uses to define ``subject_curriculum``, which
      is what every training cohort was built from;
    * else raise, rather than inventing a task family.
    """
    if "curriculum_name" not in snapshot_df.columns:
        return snapshot_df

    def _is_missing(v):
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            return True
        return str(v).strip().lower() in ("", "none", "nan")

    missing = snapshot_df["curriculum_name"].map(_is_missing)
    if not bool(missing.any()):
        return snapshot_df
    if "task" not in snapshot_df.columns:
        raise ValueError(
            "curriculum_name is missing for "
            f"{int(snapshot_df.loc[missing, 'subject_id'].nunique())} subject(s) and no `task` "
            "column is available to resolve the task family. Add 'task' to snapshot_cols."
        )

    snapshot_df = snapshot_df.copy()
    session_task = snapshot_df["task"].where(~snapshot_df["task"].map(_is_missing))

    modal_task = (
        snapshot_df.assign(_t=session_task).dropna(subset=["_t"])
        .groupby(["subject_id", "_t"], sort=False).size().rename("n").reset_index()
        .sort_values(["subject_id", "n", "_t"], ascending=[True, False, True])
        .groupby("subject_id", sort=False).first()["_t"]
    )
    resolved = session_task.fillna(snapshot_df["subject_id"].map(modal_task))
    snapshot_df.loc[missing, "curriculum_name"] = resolved[missing]

    still = snapshot_df["curriculum_name"].map(_is_missing)
    logger.info(
        "Resolved task family for %d off-curriculum session-row(s) across %d subject(s) "
        "from the session's own task (falling back to the subject's modal task); "
        "%d still unresolved.",
        int(missing.sum()), int(snapshot_df.loc[missing, "subject_id"].nunique()), int(still.sum()),
    )
    if bool(still.any()):
        bad = sorted(snapshot_df.loc[still, "subject_id"].astype(str).unique())[:5]
        raise ValueError(
            f"{int(still.sum())} row(s) have neither a curriculum_name nor any usable `task` "
            f"(subjects e.g. {bad}). Refusing to guess a task family."
        )
    return snapshot_df


def _align_snapshot_df_with_ignore_policy(snapshot_df: Any, *, ignore_policy: str):
    # "exclude": drop whole sessions containing any ignore trial, matching how the
    #   data loader (_align_raw_df_with_valid_sessions) builds the 2-way training
    #   set. "include": keep the snapshot as-is (ignore trials are class 2 and are
    #   part of the real-data comparison target for a 3-way model). Any other
    #   policy is passed through unchanged.
    policy = str(ignore_policy).strip().lower()
    if policy != "exclude":
        return snapshot_df
    if not hasattr(snapshot_df, "columns") or "animal_response" not in snapshot_df.columns:
        return snapshot_df
    valid_sessions = snapshot_df.loc[snapshot_df["animal_response"] != 2, "ses_idx"].unique()
    return snapshot_df[snapshot_df["ses_idx"].isin(valid_sessions)].copy()


def _unique_preserve_order(values: Sequence[Any]) -> list[Any]:
    seen = set()
    unique_values = []
    for value in values:
        normalized = _normalize_identifier(value)
        key = json.dumps(normalized, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(normalized)
    return unique_values


def _normalize_identifier(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(value, np.generic):
            return value.item()
    except ModuleNotFoundError:
        pass
    return value


def _is_missing_session_identifier(value: Any) -> bool:
    if _is_nan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


def _coalesce_session_identifier(value: Any, *, fallback: Any) -> str:
    resolved = fallback if _is_missing_session_identifier(value) else value
    return str(resolved)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_numeric_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _sem(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return float(_std(values) / math.sqrt(len(values)))


def _median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return math.nan
    ordered = sorted(float(value) for value in values)
    return float(_quantile(ordered, 0.75) - _quantile(ordered, 0.25))


def _rmse(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if not xs or not ys or len(xs) != len(ys):
        return None
    return float(math.sqrt(sum((y - x) ** 2 for x, y in zip(xs, ys)) / len(xs)))


def _mean_difference(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if not xs or not ys or len(xs) != len(ys):
        return None
    return float(sum((y - x) for x, y in zip(xs, ys)) / len(xs))


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator <= 0:
        return None
    numerator = sum(x_value * y_value for x_value, y_value in zip(x_centered, y_centered))
    return float(numerator / denominator)


def _coerce_probability(value: Any) -> float | None:
    if value is None or _is_nan(value):
        return None
    return float(value)


def _finite_difference(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs - rhs)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return None
    return float(sum(value * weight for value, weight in zip(values, weights)) / total_weight)


def _weighted_rmse(
    xs: Sequence[float],
    ys: Sequence[float],
    weights: Sequence[float],
) -> float | None:
    if not xs or not ys or not weights or len(xs) != len(ys) or len(xs) != len(weights):
        return None
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return None
    return float(
        math.sqrt(
            sum(weight * ((y - x) ** 2) for x, y, weight in zip(xs, ys, weights))
            / total_weight
        )
    )


def _resolve_subject_curriculum_map(
    session_rows,
    *,
    fallback_rows=None,
) -> dict[Any, str]:
    subject_curriculum_map = _collect_subject_curriculum_map(session_rows)
    fallback_map = _collect_subject_curriculum_map(fallback_rows)
    resolved: dict[Any, str] = {}
    for subject_id in _unique_preserve_order(
        [*list(subject_curriculum_map.keys()), *list(fallback_map.keys())]
    ):
        resolved[subject_id] = str(
            subject_curriculum_map.get(
                subject_id,
                fallback_map.get(subject_id, "Unknown"),
            )
        )
    return resolved


def _collect_subject_curriculum_map(session_rows) -> dict[Any, str]:
    if session_rows is None:
        return {}
    curricula_by_subject: dict[Any, list[str]] = {}
    for row in _iter_session_records(session_rows):
        subject_id = _normalize_identifier(row.get("subject_id"))
        curriculum_name = row.get("curriculum_name")
        if curriculum_name is None or _is_nan(curriculum_name):
            continue
        curricula_by_subject.setdefault(subject_id, []).append(str(curriculum_name))

    resolved: dict[Any, str] = {}
    for subject_id, curricula in curricula_by_subject.items():
        unique_curricula = list(dict.fromkeys(curricula))
        if not unique_curricula:
            resolved[subject_id] = "Unknown"
        elif len(unique_curricula) == 1:
            resolved[subject_id] = unique_curricula[0]
        else:
            resolved[subject_id] = "Mixed"
    return resolved


def _seed_from_parts(seed: int | None, *parts: Any) -> int:
    base = "0" if seed is None else str(seed)
    payload = ":".join([base, *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def _bootstrap_switch_metric_intervals(
    subject_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    n_bootstrap: int,
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, dict[str, float | None]]]]:
    reward_intervals = {
        reward_condition: {"ci_low": None, "ci_high": None}
        for reward_condition in _REWARD_CONDITIONS
    }
    reward_run_intervals = {
        reward_condition: {
            run_condition: {"ci_low": None, "ci_high": None}
            for run_condition in _RUN_LENGTH_CONDITIONS
        }
        for reward_condition in _REWARD_CONDITIONS
    }
    if len(subject_records) < 2 or int(n_bootstrap) <= 0:
        return reward_intervals, reward_run_intervals

    try:
        np = importlib.import_module("numpy")
    except ModuleNotFoundError:
        np = None

    if np is not None:
        reward_successes = np.asarray(
            [
                [
                    float(_as_dict(record["reward"][reward_condition]).get("successes", 0.0))
                    for reward_condition in _REWARD_CONDITIONS
                ]
                for record in subject_records
            ],
            dtype=float,
        )
        reward_ns = np.asarray(
            [
                [
                    float(_as_dict(record["reward"][reward_condition]).get("n", 0.0))
                    for reward_condition in _REWARD_CONDITIONS
                ]
                for record in subject_records
            ],
            dtype=float,
        )
        reward_run_successes = np.asarray(
            [
                [
                    [
                        float(
                            _as_dict(
                                _as_dict(record["reward_and_run_length"][reward_condition]).get(
                                    run_condition,
                                    {},
                                )
                            ).get("successes", 0.0)
                        )
                        for run_condition in _RUN_LENGTH_CONDITIONS
                    ]
                    for reward_condition in _REWARD_CONDITIONS
                ]
                for record in subject_records
            ],
            dtype=float,
        )
        reward_run_ns = np.asarray(
            [
                [
                    [
                        float(
                            _as_dict(
                                _as_dict(record["reward_and_run_length"][reward_condition]).get(
                                    run_condition,
                                    {},
                                )
                            ).get("n", 0.0)
                        )
                        for run_condition in _RUN_LENGTH_CONDITIONS
                    ]
                    for reward_condition in _REWARD_CONDITIONS
                ]
                for record in subject_records
            ],
            dtype=float,
        )
        rng = np.random.default_rng(int(seed))
        sample_indices = rng.integers(
            0,
            len(subject_records),
            size=(int(n_bootstrap), len(subject_records)),
        )
        reward_probabilities = np.divide(
            reward_successes[sample_indices].sum(axis=1),
            reward_ns[sample_indices].sum(axis=1),
            out=np.full((int(n_bootstrap), len(_REWARD_CONDITIONS)), np.nan, dtype=float),
            where=reward_ns[sample_indices].sum(axis=1) > 0,
        )
        reward_run_probabilities = np.divide(
            reward_run_successes[sample_indices].sum(axis=1),
            reward_run_ns[sample_indices].sum(axis=1),
            out=np.full(
                (int(n_bootstrap), len(_REWARD_CONDITIONS), len(_RUN_LENGTH_CONDITIONS)),
                np.nan,
                dtype=float,
            ),
            where=reward_run_ns[sample_indices].sum(axis=1) > 0,
        )
        for reward_index, reward_condition in enumerate(_REWARD_CONDITIONS):
            reward_intervals[reward_condition] = _bootstrap_interval_from_values(
                reward_probabilities[:, reward_index]
            )
            for run_index, run_condition in enumerate(_RUN_LENGTH_CONDITIONS):
                reward_run_intervals[reward_condition][
                    run_condition
                ] = _bootstrap_interval_from_values(
                    reward_run_probabilities[:, reward_index, run_index]
                )
        return reward_intervals, reward_run_intervals

    rng = random.Random(int(seed))
    reward_samples = {reward_condition: [] for reward_condition in _REWARD_CONDITIONS}
    reward_run_samples = {
        reward_condition: {run_condition: [] for run_condition in _RUN_LENGTH_CONDITIONS}
        for reward_condition in _REWARD_CONDITIONS
    }
    for _ in range(int(n_bootstrap)):
        sampled_records = [
            subject_records[rng.randrange(len(subject_records))]
            for _ in range(len(subject_records))
        ]
        reward_totals = _new_reward_count_tree()
        reward_run_totals = _new_reward_run_count_tree()
        for record in sampled_records:
            _accumulate_reward_counts(reward_totals, record["reward"])
            _accumulate_reward_run_counts(
                reward_run_totals,
                record["reward_and_run_length"],
            )
        for reward_condition in _REWARD_CONDITIONS:
            reward_samples[reward_condition].append(
                _probability_from_counts(reward_totals[reward_condition])
            )
            for run_condition in _RUN_LENGTH_CONDITIONS:
                reward_run_samples[reward_condition][run_condition].append(
                    _probability_from_counts(
                        reward_run_totals[reward_condition][run_condition]
                    )
                )
    for reward_condition in _REWARD_CONDITIONS:
        reward_intervals[reward_condition] = _bootstrap_interval_from_values(
            reward_samples[reward_condition]
        )
        for run_condition in _RUN_LENGTH_CONDITIONS:
            reward_run_intervals[reward_condition][
                run_condition
            ] = _bootstrap_interval_from_values(
                reward_run_samples[reward_condition][run_condition]
            )
    return reward_intervals, reward_run_intervals


def _bootstrap_history_pattern_intervals(
    subject_records: Sequence[Mapping[str, Any]],
    *,
    max_trials_back: int,
    seed: int,
    n_bootstrap: int,
) -> dict[str, dict[int, dict[str, dict[str, float | None]]]]:
    intervals: dict[str, dict[int, dict[str, dict[str, float | None]]]] = {
        pattern_type: {
            n_back: {}
            for n_back in range(1, int(max_trials_back) + 1)
        }
        for pattern_type in _HISTORY_PATTERN_TYPES
    }
    if len(subject_records) < 2 or int(n_bootstrap) <= 0:
        return intervals

    try:
        np = importlib.import_module("numpy")
    except ModuleNotFoundError:
        np = None

    if np is not None:
        rng = np.random.default_rng(int(seed))
        sample_indices = rng.integers(
            0,
            len(subject_records),
            size=(int(n_bootstrap), len(subject_records)),
        )
        for pattern_type in _HISTORY_PATTERN_TYPES:
            for n_back in range(1, int(max_trials_back) + 1):
                patterns = sorted(
                    {
                        str(pattern)
                        for record in subject_records
                        for pattern in _as_dict(
                            _as_dict(
                                _as_dict(record.get(pattern_type, {})).get(n_back, {})
                            )
                        ).keys()
                    }
                )
                if not patterns:
                    continue
                switches = np.zeros((len(subject_records), len(patterns)), dtype=float)
                totals = np.zeros((len(subject_records), len(patterns)), dtype=float)
                for record_index, record in enumerate(subject_records):
                    pattern_bucket = _as_dict(
                        _as_dict(_as_dict(record.get(pattern_type, {})).get(n_back, {}))
                    )
                    for pattern_index, pattern in enumerate(patterns):
                        counts = _as_dict(pattern_bucket.get(pattern, {}))
                        switches[record_index, pattern_index] = float(
                            counts.get("switches", 0.0)
                        )
                        totals[record_index, pattern_index] = float(
                            counts.get("total", 0.0)
                        )
                sampled_switches = switches[sample_indices].sum(axis=1)
                sampled_totals = totals[sample_indices].sum(axis=1)
                probabilities = np.divide(
                    sampled_switches,
                    sampled_totals,
                    out=np.full(sampled_switches.shape, np.nan, dtype=float),
                    where=sampled_totals > 0,
                )
                for pattern_index, pattern in enumerate(patterns):
                    intervals[pattern_type][n_back][pattern] = _bootstrap_interval_from_values(
                        probabilities[:, pattern_index]
                    )
        return intervals

    rng = random.Random(int(seed))
    pattern_samples: dict[str, dict[int, dict[str, list[float]]]] = {
        pattern_type: {
            n_back: {}
            for n_back in range(1, int(max_trials_back) + 1)
        }
        for pattern_type in _HISTORY_PATTERN_TYPES
    }
    for pattern_type in _HISTORY_PATTERN_TYPES:
        for n_back in range(1, int(max_trials_back) + 1):
            patterns = sorted(
                {
                    str(pattern)
                    for record in subject_records
                    for pattern in _as_dict(
                        _as_dict(_as_dict(record.get(pattern_type, {})).get(n_back, {}))
                    ).keys()
                }
            )
            for pattern in patterns:
                pattern_samples[pattern_type][n_back][pattern] = []
    for _ in range(int(n_bootstrap)):
        sampled_records = [
            subject_records[rng.randrange(len(subject_records))]
            for _ in range(len(subject_records))
        ]
        aggregated = _new_history_pattern_count_tree(max_trials_back)
        for record in sampled_records:
            _accumulate_history_pattern_count_tree(
                aggregated,
                record,
                max_trials_back=max_trials_back,
            )
        for pattern_type in _HISTORY_PATTERN_TYPES:
            for n_back in range(1, int(max_trials_back) + 1):
                for pattern, values in pattern_samples[pattern_type][n_back].items():
                    probability = _history_probability_from_counts(
                        _as_dict(
                            _as_dict(_as_dict(aggregated.get(pattern_type, {})).get(n_back, {})).get(
                                pattern,
                                {},
                            )
                        )
                    )
                    values.append(probability)
    for pattern_type in _HISTORY_PATTERN_TYPES:
        for n_back in range(1, int(max_trials_back) + 1):
            for pattern, values in pattern_samples[pattern_type][n_back].items():
                intervals[pattern_type][n_back][pattern] = _bootstrap_interval_from_values(
                    values
                )
    return intervals


def _bootstrap_interval_from_values(values: Sequence[Any]) -> dict[str, float | None]:
    finite_values = sorted(
        float(value)
        for value in values
        if value is not None and not _is_nan(value)
    )
    if len(finite_values) < 2:
        return {"ci_low": None, "ci_high": None}
    return {
        "ci_low": float(_quantile(finite_values, 0.025)),
        "ci_high": float(_quantile(finite_values, 0.975)),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])
    bounded_q = min(max(float(q), 0.0), 1.0)
    ordered = sorted(float(value) for value in values)
    position = bounded_q * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ResolvedModelRun):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return _to_serializable(value.item())
    except ModuleNotFoundError:
        pass
    return value


def to_serializable(value: Any) -> Any:
    """Public wrapper for JSON-safe conversion of analysis payloads."""

    return _to_serializable(value)
