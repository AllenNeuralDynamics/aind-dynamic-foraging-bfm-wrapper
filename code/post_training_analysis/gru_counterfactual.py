"""Run a saved GRU on edited inputs: the counterfactual counterpart of ``gru_trial_states``.

``load_gru_split`` rebuilds a run's network, params and the model-input tensor ``xs`` for one
training-bundle split; ``choice_logit_diff`` evaluates (possibly edited) inputs; and
``reward_flip_effects`` measures how flipping one trial's reward changes the model's
log-odds on the following trials, holding every other input as observed.

Input layout (``data_loaders.disrnn_dataset``): ``xs[τ, s]`` = [subject index,
(session index if session conditioning), previous choice, previous reward], i.e. row τ
carries trial τ−1's outcome, and row τ's output predicts trial τ. Session column ``s``
follows the split's session order and τ = ``trial_idx − 1`` of ``gru_trial_states``.

A 3-action run (``ignore_policy=include``) steps through ignored trials: they are timesteps
whose "previous choice" input is 2. ``input_edit_effects`` sets one trial's choice and
reward inputs to chosen values (e.g. turn an engaged trial into an ignore) and returns the
change in both readouts: ``side`` = logit_R − logit_L and, for 3 actions, ``engage`` =
logsumexp(logit_L, logit_R) − logit_ignore.
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
    n_actions: int = 2        # 3 when ignored trials are timesteps (ignore_policy=include)


def load_gru_split(model_dir: str | Path, *, split: str = "eval",
                   checkpoint_policy: str = "best_eval") -> GruSplit:
    run = lc._resolve_likelihood_run(model_dir=model_dir, model_label=Path(model_dir).name,
                                     model_index=0, checkpoint_policy=checkpoint_policy)
    _, bundle = lc._load_training_bundle_for_run(run)
    payload = lc._build_training_split_payloads(bundle)[split]
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    make_network, params, ignore_policy = lc._build_gru_network(run, metadata=metadata)
    raw = lc._normalize_raw_dataframe(payload["raw_df"])
    if ignore_policy == "exclude":
        raw = raw[raw["animal_response"] != 2]
    session_ids = [str(s) for s in dict.fromkeys(raw["ses_idx"].tolist())]
    n_trials = raw.groupby("ses_idx", sort=False).size().reindex(session_ids).to_numpy()
    xs = np.asarray(payload["dataset"].get_all()["xs"], dtype=np.float32)
    if xs.shape[1] != len(session_ids):
        raise AssertionError(f"xs has {xs.shape[1]} sessions, split has {len(session_ids)}")
    offset = _feature_offset(run)
    n_actions = 2 if ignore_policy == "exclude" else 3
    return GruSplit(run, make_network, params, xs, session_ids, n_trials, offset, offset + 1, n_actions)


def _feature_offset(run) -> int:
    """Index of "previous choice" in xs: after the subject index (multisubject runs only) and
    the session index (when session conditioning is on); "previous reward" follows it."""
    return int(bool(run.multisubject)) + int(bool(run.session_conditioning_enabled))


def logits(split: GruSplit, xs: np.ndarray | None = None, *, batch: int = 512) -> np.ndarray:
    """All output logits, [T, S, n_actions], evaluated in session batches."""
    from disentangled_rnns.library import rnn_utils  # noqa: PLC0415

    xs = split.xs if xs is None else xs
    out = np.empty((*xs.shape[:2], split.n_actions), dtype=np.float32)
    for start in range(0, xs.shape[1], batch):
        yhat, _ = rnn_utils.eval_network(split.make_network, split.params, xs[:, start:start + batch])
        out[:, start:start + batch] = np.asarray(yhat)[..., :split.n_actions]
    return out


def readouts(logit_array: np.ndarray) -> dict[str, np.ndarray]:
    """``side`` = logit_R − logit_L; with 3 actions also ``engage`` = log P(engaged) / P(ignore)."""
    out = {"side": logit_array[..., 1] - logit_array[..., 0]}
    if logit_array.shape[-1] == 3:
        out["engage"] = np.logaddexp(logit_array[..., 0], logit_array[..., 1]) - logit_array[..., 2]
    return out


def input_edit_effects(split: GruSplit, session_cols: Sequence[int], trials: Sequence[int], *,
                       choice: Sequence[float], reward: Sequence[float], k_max: int,
                       baseline: dict[str, np.ndarray] | None = None, batch: int = 512) -> dict[str, np.ndarray]:
    """Δ readouts on trials t+1..t+k_max after setting trial t's choice and reward inputs.

    ``session_cols[i]``, ``trials[i]`` (0-based τ within the session) name one edit each and
    ``choice[i]`` / ``reward[i]`` the values written into row τ+1 (the input that carries
    trial τ's outcome). Each edit runs in its own copy of the session, later inputs as
    observed. Returns ``{readout: [n_edits, k_max]}``, NaN past the session end.
    ``baseline`` is ``readouts(logits(split))`` if already computed.
    """
    session_cols, trials = np.asarray(session_cols), np.asarray(trials)
    choice, reward = np.asarray(choice, dtype=np.float32), np.asarray(reward, dtype=np.float32)
    baseline = readouts(logits(split)) if baseline is None else baseline
    effects = {name: np.full((len(trials), k_max), np.nan, dtype=np.float32) for name in baseline}
    ks = np.arange(1, k_max + 1)
    for start in range(0, len(trials), batch):
        sl = slice(start, start + batch)
        cols, ts = session_cols[sl], trials[sl]
        if np.any(ts + 1 >= split.n_trials[cols]):
            raise ValueError("An edited trial has no following trial in its session.")
        edited = split.xs[:, cols].copy()
        rows = np.arange(len(cols))
        edited[ts + 1, rows, split.choice_col] = choice[sl]
        edited[ts + 1, rows, split.reward_col] = reward[sl]
        after = readouts(logits(split, edited, batch=batch))
        target = ts[:, None] + ks[None, :]
        valid = target < split.n_trials[cols][:, None]
        safe = np.where(valid, target, 0)
        for name in effects:
            delta = after[name][safe, rows[:, None]] - baseline[name][safe, cols[:, None]]
            effects[name][sl] = np.where(valid, delta, np.nan)
    return effects


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
