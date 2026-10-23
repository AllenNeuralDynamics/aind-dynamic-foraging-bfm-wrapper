"""Per-trial choice probabilities (and Q values) of a per-subject baseline-RL fit.

Replays each subject's fitted agent over that subject's observed choices and rewards and
returns one row per trial, keyed like ``gru_trial_states`` (``ses_idx``, ``trial_idx``), so
the two can be joined trial by trial. Built on the rollout helpers that
``likelihood_advantage_analysis`` already uses for its baseline-RL side.

Usage::

    out = json.load(open(".../baseline_rl_output.json"))
    rl_df = rollout_baseline_rl(out, trial_df, workers=16)
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Mapping

import numpy as np
import pandas as pd

from post_training_analysis import likelihood_advantage_analysis as laa

_KEY_COLUMNS = ["subject_id", "ses_idx", "trial_idx", "action", "reward"]


def rollout_subject(fit: Mapping[str, Any], subject_trials: pd.DataFrame) -> pd.DataFrame:
    """One subject's fitted agent over its trials: p_rl_left/right, and Q when recoverable."""
    payloads = laa._session_rollout_payloads_from_trial_df(subject_trials[_KEY_COLUMNS])
    kwargs = dict(
        agent_class_name=fit["agent_class"], agent_kwargs=fit["agent_kwargs"],
        fitted_params=fit["fitted_params"],
        choice_sessions=[p["choices"] for p in payloads],
        reward_sessions=[p["rewards"] for p in payloads], seed=0,
    )
    probs, _, _ = laa._perform_baseline_agent_rollout_with_q_histories(**kwargs, require_q_values=False)
    # Q straight from the agent, one session at a time (exact by construction). The shared
    # likelihood_advantage recovery is only a fallback: its manual path assumes one learning
    # rate and its reproduction check can pass on an asymmetric (Hattori L2F1) agent with
    # wrong Q values. Agents with no Q history (compare-to-threshold) get probabilities only.
    q_values = _per_session_q_histories(fit, payloads, probs)
    if q_values is None:
        try:
            _, q_values, _ = laa._perform_baseline_agent_rollout_with_q_histories(**kwargs, require_q_values=True)
        except ValueError:
            q_values = None
    frame = laa._probability_frame_from_rollout_sessions(
        payloads, choice_prob_sessions=probs, q_value_sessions=q_values)
    return frame.drop(columns=["subject_id", "action", "p_rl"])


def _per_session_q_histories(fit: Mapping[str, Any], payloads: list, probs: list) -> list | None:
    """Policy-time Q (Q before each choice, ``q_value[:, :T]``) from one-session replays.

    Returns None when the agent keeps no two-action Q history. Each session's replay must
    reproduce the multi-session choice probabilities exactly.
    """
    from aind_dynamic_foraging_models import generative_model  # noqa: PLC0415

    histories = []
    for payload, multi in zip(payloads, probs):
        agent = getattr(generative_model, fit["agent_class"])(**fit["agent_kwargs"], seed=0)
        agent.set_params(**{k: float(v) for k, v in fit["fitted_params"].items()})
        agent.perform_closed_loop(payload["choices"], payload["rewards"])
        q = np.asarray(getattr(agent, "q_value", np.empty((0, 0))), dtype=float)
        n = len(payload["choices"])
        if q.shape != (2, n + 1):
            return None
        if not np.allclose(np.asarray(agent.choice_prob)[:, :n], np.asarray(multi)[:, :n], atol=1e-12):
            raise AssertionError(f"Single-session replay of {payload['ses_idx']} disagrees with the multi-session one")
        histories.append(q[:, :n])
    return histories


def _rollout_task(args: tuple[Mapping[str, Any], pd.DataFrame]) -> pd.DataFrame:
    return rollout_subject(*args)


def rollout_baseline_rl(
    baseline_output: Mapping[str, Any], trial_df: pd.DataFrame, *, workers: int = 1
) -> pd.DataFrame:
    """All subjects of ``trial_df`` that ``baseline_output`` has a per-subject fit for.

    Raises if any subject in ``trial_df`` has no fit. Rows come back in ``trial_df``'s
    order for the subjects present.
    """
    fits = baseline_output["fitted_params_per_subject"]
    by_subject = trial_df.groupby(trial_df["subject_id"].astype(str), sort=False)
    missing = [s for s in by_subject.groups if s not in fits]
    if missing:
        raise ValueError(f"No baseline fit for {len(missing)} subjects, e.g. {missing[:5]}")
    tasks = [(fits[s], group) for s, group in by_subject]
    if workers > 1:
        # spawn, not fork: callers often have JAX loaded, and forking a JAX process can deadlock.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            parts = list(pool.map(_rollout_task, tasks, chunksize=4))
    else:
        parts = [_rollout_task(t) for t in tasks]
    rl = pd.concat(parts, ignore_index=True)
    keys = trial_df[["ses_idx", "trial_idx"]].astype({"ses_idx": str})
    out = keys.merge(rl, on=["ses_idx", "trial_idx"], how="left", validate="1:1")
    if out["p_rl_right"].isna().any():
        raise AssertionError("Some trials received no baseline probability")
    return out


def pooled_likelihood(action: np.ndarray, p_right: np.ndarray) -> float:
    """exp(mean log p(chosen)), clipped as elsewhere in post_training_analysis."""
    p = np.where(np.asarray(action) == 1, p_right, 1.0 - np.asarray(p_right))
    return float(np.exp(np.mean(np.log(np.clip(p, 1e-10, 1 - 1e-10)))))
