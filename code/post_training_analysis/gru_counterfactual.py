"""Run a saved GRU on edited inputs: the counterfactual counterpart of ``gru_trial_states``.

``load_gru_split`` rebuilds a run's network, params and the model-input tensor ``xs`` for one
training-bundle split; ``choice_logit_diff`` evaluates (possibly edited) inputs; and
``reward_flip_effects`` measures how flipping one trial's reward changes the model's
log-odds on the following trials, holding every other input as observed.

Input layout (``data_loaders.disrnn_dataset``): ``xs[τ, s]`` = [subject index,
(session index if session conditioning), previous choice, previous reward], i.e. row τ
carries trial τ−1's outcome, and row τ's output predicts trial τ. Session column ``s``
follows the split's session order and τ = ``trial_idx − 1`` of ``gru_trial_states``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from post_training_analysis import likelihood_comparison as lc


@dataclass
class GruSplit:
    run: Any
    make_network: Any
    params: Any
    xs: np.ndarray            # [T, S, F]
    session_ids: list[str]    # S, column order of xs
    n_trials: np.ndarray      # S, valid timesteps per session
    choice_col: int           # feature index of "previous choice"
    reward_col: int           # feature index of "previous reward"


def load_gru_split(model_dir: str | Path, *, split: str = "eval",
                   checkpoint_policy: str = "best_eval") -> GruSplit:
    run = lc._resolve_likelihood_run(model_dir=model_dir, model_label=Path(model_dir).name,
                                     model_index=0, checkpoint_policy=checkpoint_policy)
    _, bundle = lc._load_training_bundle_for_run(run)
    payload = lc._build_training_split_payloads(bundle)[split]
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    make_network, params, ignore_policy = lc._build_gru_network(run, metadata=metadata)
    if ignore_policy != "exclude":
        raise NotImplementedError("Only the 2-action (ignore-excluded) GRU is supported.")
    raw = lc._normalize_raw_dataframe(payload["raw_df"])
    raw = raw[raw["animal_response"] != 2]
    session_ids = [str(s) for s in dict.fromkeys(raw["ses_idx"].tolist())]
    n_trials = raw.groupby("ses_idx", sort=False).size().reindex(session_ids).to_numpy()
    xs = np.asarray(payload["dataset"].get_all()["xs"], dtype=np.float32)
    if xs.shape[1] != len(session_ids):
        raise AssertionError(f"xs has {xs.shape[1]} sessions, split has {len(session_ids)}")
    offset = 2 if run.session_conditioning_enabled else 1
    return GruSplit(run, make_network, params, xs, session_ids, n_trials, offset, offset + 1)


def choice_logit_diff(split: GruSplit, xs: np.ndarray | None = None, *, batch: int = 512) -> np.ndarray:
    """logit(right) − logit(left), [T, S], evaluated in session batches."""
    from disentangled_rnns.library import rnn_utils  # noqa: PLC0415

    xs = split.xs if xs is None else xs
    out = np.empty(xs.shape[:2], dtype=np.float32)
    for start in range(0, xs.shape[1], batch):
        yhat, _ = rnn_utils.eval_network(split.make_network, split.params, xs[:, start:start + batch])
        yhat = np.asarray(yhat)
        out[:, start:start + batch] = yhat[..., 1] - yhat[..., 0]
    return out


def reward_flip_effects(split: GruSplit, session_cols: Sequence[int], trials: Sequence[int], *,
                        k_max: int, baseline: np.ndarray | None = None, batch: int = 512) -> np.ndarray:
    """Δ logit(right) on trials t+1..t+k_max after flipping the reward of trial t.

    ``session_cols[i]``, ``trials[i]`` (0-based τ within the session) name one flip each;
    every flip is evaluated in its own copy of the session. Effects past the session end are
    NaN. ``baseline`` is ``choice_logit_diff(split)`` if already computed.
    """
    session_cols = np.asarray(session_cols)
    trials = np.asarray(trials)
    baseline = choice_logit_diff(split) if baseline is None else baseline
    effects = np.full((len(trials), k_max), np.nan, dtype=np.float32)
    ks = np.arange(1, k_max + 1)
    for start in range(0, len(trials), batch):
        cols, ts = session_cols[start:start + batch], trials[start:start + batch]
        edited = split.xs[:, cols].copy()
        rows = np.arange(len(cols))
        if np.any(ts + 1 >= split.n_trials[cols]):
            raise ValueError("A flipped trial has no following trial in its session.")
        edited[ts + 1, rows, split.reward_col] = 1.0 - edited[ts + 1, rows, split.reward_col]
        flipped = choice_logit_diff(split, edited, batch=batch)
        target = ts[:, None] + ks[None, :]
        valid = target < split.n_trials[cols][:, None]
        safe = np.where(valid, target, 0)
        delta = flipped[safe, rows[:, None]] - baseline[safe, cols[:, None]]
        effects[start:start + batch] = np.where(valid, delta, np.nan)
    return effects
