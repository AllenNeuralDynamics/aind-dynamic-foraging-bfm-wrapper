"""Per-trial GRU hidden states for post-hoc dynamics analysis.

One forward pass of a saved GRU run over one data split, returned as a flat per-trial table.
The table uses the column schema of ``likelihood_advantage_analysis``'s canonical trial frame
(``subject_id``, ``ses_idx``, ``trial_idx``, ``action``, ``reward``, ``rnn_state_*``,
``p_model1_*`` plus ``add_candidate_variables`` columns), so the ``state-space-*`` plotting
functions can read it directly.

Timing convention, row ``t`` of a session (``trial_idx`` is the GRU's timestep):
  * ``rnn_state_*`` is the GRU state after receiving trial ``t-1``'s choice and reward;
  * ``choice_logit_*`` / ``p_model1_*`` are the readout of that state, i.e. the model's
    prediction of the choice on trial ``t``;
  * ``action`` / ``reward`` are what happened on trial ``t``.

A 2-action head (``ignore_policy=exclude``) never sees ignored trials, so the table has none.
A 3-action head (``include``) steps through them, so they are rows with ``action == 2``,
``choice_logit_2`` and ``p_model1_ignore``; ``engaged_idx`` numbers the L/R trials among
themselves. The ``add_candidate_variables`` history columns are computed over engaged trials
only (an ignored trial is skipped, not a break) and are NaN on ignored rows.

Usage::

    from post_training_analysis.gru_trial_states import extract_gru_trial_states
    trial_df, info = extract_gru_trial_states(model_dir, split="eval")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from post_training_analysis import likelihood_comparison as lc
from post_training_analysis.likelihood_advantage_analysis import (
    _build_canonical_trial_dataframe,
    _sort_raw_dataframe,
    _sort_trial_dataframe,
    add_candidate_variables,
)

logger = logging.getLogger(__name__)

# The snapshot loader's default raw columns (``data_loaders.mice``), restated because passing
# ``cols_to_retain`` replaces the default rather than extending it.
_DEFAULT_COLS_TO_RETAIN = (
    "trial",
    "subject_id",
    "ses_idx",
    "animal_response",
    "earned_reward",
    "curriculum_name",
)
# Task-state columns of the snapshot trial table: the true reward probability of each side
# and whether that side is currently baited.
TASK_STATE_COLUMNS = (
    "reward_probabilityL",
    "reward_probabilityR",
    "bait_left",
    "bait_right",
)
_PASSTHROUGH_COLUMNS = ("trial", "source_ses_idx", "subject_session_index", "subject_index")
# Per-action columns in action-index order (0 = left, 1 = right, 2 = ignore).
PROBABILITY_COLUMNS = ("p_model1_left", "p_model1_right", "p_model1_ignore")
IGNORE = 2


def extract_gru_trial_states(
    model_dir: str | Path,
    *,
    split: str = "eval",
    checkpoint_policy: str = "best_eval",
    extra_cols_to_retain: Sequence[str] = TASK_STATE_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a saved 2- or 3-action GRU over ``split`` and return ``(trial_df, info)``.

    ``split`` is one of the training bundle's splits: ``train``, ``eval`` or ``combined``.
    ``info`` carries the resolved checkpoint, counts, the pooled per-trial likelihood
    (same formula as ``likelihood_comparison``) and the readout reconstruction error that
    confirms states and logits are row-aligned.
    """
    run = lc._resolve_likelihood_run(
        model_dir=model_dir,
        model_label=Path(model_dir).name,
        model_index=0,
        checkpoint_policy=checkpoint_policy,
    )
    if run.model_type != "gru":
        raise ValueError(f"Expected a GRU run, got model_type={run.model_type!r}.")

    _, bundle = lc._load_training_bundle_for_run(
        run, cols_to_retain=[*_DEFAULT_COLS_TO_RETAIN, *extra_cols_to_retain]
    )
    payload = lc._build_training_split_payloads(bundle)[split]
    if "score_partition" in payload:
        raise NotImplementedError("Trial-partitioned splits are not supported.")
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    raw_df = lc._normalize_raw_dataframe(payload["raw_df"])
    output_df, n_action_logits = lc._evaluate_gru_dataset(
        run, dataset=payload["dataset"], raw_df=raw_df, metadata=metadata
    )
    trial_df = build_trial_states_table(
        output_df, metadata=metadata, extra_columns=extra_cols_to_retain
    )
    info = {
        "model_dir": str(run.model_dir),
        "split": split,
        "checkpoint_policy": checkpoint_policy,
        "checkpoint_step": run.checkpoint_step,
        "params_path": run.params_path,
        "n_actions": int(n_action_logits),
        "n_trials": int(len(trial_df)),
        "n_sessions": int(trial_df["ses_idx"].nunique()),
        "n_subjects": int(trial_df["subject_id"].nunique()),
        "hidden_size": int(len(state_columns(trial_df))),
        "pooled_trial_likelihood": pooled_trial_likelihood(trial_df),
        "readout_max_abs_error": readout_reconstruction_error(trial_df, run.params_path),
    }
    return trial_df, info


def build_trial_states_table(
    output_df: pd.DataFrame,
    *,
    metadata: dict[str, Any],
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Turn ``_evaluate_gru_dataset`` output into the per-trial analysis table."""
    n_actions = _count_prefixed(output_df, "choice_logit_")
    if n_actions not in (2, 3):
        raise NotImplementedError(f"Expected a 2- or 3-action GRU head, got {n_actions} logits.")
    out = _sort_raw_dataframe(output_df[output_df["animal_response"].isin(range(n_actions))])
    if n_actions == 3 and len(out) != len(output_df):
        raise AssertionError("A 3-action GRU steps through every trial; found responses outside 0/1/2.")
    out["trial_idx"] = out.groupby("ses_idx", sort=False).cumcount() + 1

    canonical = add_candidate_variables(_build_canonical_trial_dataframe(out, metadata=metadata))
    canonical = _with_ignored_trials(canonical, out)

    latent_columns = [f"latent_{i}" for i in range(_count_prefixed(out, "latent_"))]
    model = pd.DataFrame(
        out[latent_columns].to_numpy(dtype=np.float32),
        columns=[f"rnn_state_{i}" for i in range(len(latent_columns))],
    )
    for k in range(n_actions):
        model[f"choice_logit_{k}"] = out[f"choice_logit_{k}"].to_numpy(dtype=np.float32)
    # Log-odds of right vs left; with an ignore class this is log P(R | engaged) / P(L | engaged).
    model["logit_diff"] = model["choice_logit_1"] - model["choice_logit_0"]
    for k in range(n_actions):
        model[PROBABILITY_COLUMNS[k]] = out[f"choice_prob_{k}"].to_numpy(dtype=float)
    # The canonical frame's ``curriculum_name`` is one label per subject ("Mixed" when a
    # subject ran several); keep the raw per-trial label too.
    if "curriculum_name" in out.columns:
        model["session_curriculum_name"] = out["curriculum_name"].astype(str).to_numpy()
    keys = ["ses_idx", "trial_idx", "animal_response"]
    passthrough = [c for c in (*_PASSTHROUGH_COLUMNS, *extra_columns) if c in out.columns]
    model = pd.concat([out[keys + passthrough].reset_index(drop=True), model], axis=1)

    merged = canonical.merge(model, on=["ses_idx", "trial_idx"], how="inner", validate="1:1")
    if len(merged) != len(canonical) or len(merged) != len(model):
        raise AssertionError(
            f"Row mismatch: canonical={len(canonical)} model={len(model)} merged={len(merged)}"
        )
    if not (merged["action"].to_numpy() == merged.pop("animal_response").to_numpy()).all():
        raise AssertionError("Model rows and canonical rows disagree on the chosen action.")
    if n_actions == 2:
        merged = merged.drop(columns="engaged_idx")
    return merged


def _with_ignored_trials(canonical: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Key the engaged-only canonical frame by ``out``'s timestep and add ``out``'s ignored trials.

    ``canonical.trial_idx`` counts engaged trials; on return ``trial_idx`` is the timestep and
    ``engaged_idx`` keeps the engaged count (NaN on ignored rows). Without ignored trials the
    two are equal and the rows are unchanged.
    """
    engaged = out["animal_response"].isin([0, 1]).to_numpy()
    steps = out.loc[engaged, ["ses_idx", "trial_idx"]].copy()
    steps["engaged_idx"] = steps.groupby("ses_idx", sort=False).cumcount() + 1
    columns = list(canonical.columns)
    canonical = canonical.rename(columns={"trial_idx": "engaged_idx"}).merge(
        steps, on=["ses_idx", "engaged_idx"], how="left", validate="1:1"
    )
    if canonical["trial_idx"].isna().any():
        raise AssertionError("Canonical engaged trials do not line up with the model's engaged rows.")
    canonical = canonical[[*columns, "engaged_idx"]]
    if engaged.all():
        return canonical

    sessions = canonical.drop_duplicates("ses_idx").set_index("ses_idx")
    ignored = out.loc[~engaged, ["ses_idx", "trial_idx", "subject_id", "earned_reward"]].copy()
    for column in ("session_id", "session_idx", "curriculum_name"):
        ignored[column] = ignored["ses_idx"].map(sessions[column])
    # A session with no engaged trial is absent from ``canonical``; keep its raw subject id.
    ignored["subject_id"] = ignored["ses_idx"].map(sessions["subject_id"]).fillna(ignored["subject_id"])
    ignored["action"] = IGNORE
    ignored["reward"] = ignored.pop("earned_reward").astype(float)
    return _sort_trial_dataframe(pd.concat([canonical, ignored[[c for c in ignored if c in columns]]], ignore_index=True))


def state_columns(trial_df: pd.DataFrame) -> list[str]:
    """``rnn_state_*`` columns in index order."""
    return [f"rnn_state_{i}" for i in range(_count_prefixed(trial_df, "rnn_state_"))]


def pooled_trial_likelihood(trial_df: pd.DataFrame) -> float:
    """exp(mean log p(chosen action)), clipped as in ``likelihood_comparison``."""
    action = trial_df["action"].to_numpy(dtype=int)
    columns = [c for c in PROBABILITY_COLUMNS if c in trial_df.columns]
    probs = trial_df[columns].to_numpy(dtype=float)[np.arange(len(action)), action]
    probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
    return float(np.exp(np.mean(np.log(probs))))


def readout_reconstruction_error(trial_df: pd.DataFrame, params_path: str | Path) -> float:
    """Max |readout(rnn_state) - choice_logit| over all rows.

    Near zero only if each row's state is the one that produced that row's logits, which is
    the alignment the timing convention in the module docstring relies on.
    """
    from evaluation.common import _load_saved_params  # noqa: PLC0415

    params = _load_saved_params(Path(params_path))
    readout = next(v for k, v in params.items() if k.endswith("/readout"))
    states = trial_df[state_columns(trial_df)].to_numpy(dtype=np.float64)
    logits = states @ np.asarray(readout["w"], dtype=np.float64) + np.asarray(readout["b"])
    n_actions = _count_prefixed(trial_df, "choice_logit_")
    observed = trial_df[[f"choice_logit_{k}" for k in range(n_actions)]].to_numpy(dtype=np.float64)
    return float(np.max(np.abs(logits - observed)))


def _count_prefixed(df: pd.DataFrame, prefix: str) -> int:
    indices = [int(c[len(prefix):]) for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(f"{prefix}* columns are not contiguous from 0.")
    return len(indices)
