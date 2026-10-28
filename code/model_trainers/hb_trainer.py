"""Hierarchical Bayesian cognitive-model trainer.

Fits ``HB-Hattori2019`` through the same ``ModelTrainer`` interface as the neural models
and ``baseline_rl``, so all four consume the identical ``DatasetBundle`` and are scored on
the identical trials. That shared path is what makes the comparison valid: reimplementing
the session filters and ignore policy here and hoping they matched would not.

Two estimators are available. ``two_stage`` fits each subject independently and then fits a
population distribution over the resulting posteriors; ``one_stage`` infers all three levels
jointly. The joint fit is the reference the approximation is judged against.

A third, ``heldout_two_level``, has no population level at all: it fits each held-out
subject alone on its context sessions, under a fixed hyperprior, and never fits the training
cohort (study 13).

The model itself lives in ``aind-dynamic-foraging-models`` and knows nothing about this
stack; this module only marshals data, runs the fit, and reports.
"""

from __future__ import annotations

import logging
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from base.interfaces import ModelTrainer
from utils.multisubject import compute_train_eval_session_ids
from base.types import DatasetBundle

logger = logging.getLogger(__name__)

FEW_SHOT_K = (0, 1, 2, 4, 8)

# Per-subject convergence gates for ``heldout_two_level``: study 08's offline gates
# (studies/08-hb-vs-gru-heldout/analysis/summarize_hb_posteriors.py), applied per subject.
GATE_MAX_R_HAT = 1.01
GATE_MIN_ESS_BULK = 400.0


def _extract_subject_sessions(
    raw_df: pd.DataFrame,
    session_ids: Optional[List[Any]] = None,
) -> Tuple[Dict[Any, List[np.ndarray]], Dict[Any, List[np.ndarray]]]:
    """Group a raw trial dataframe into per-subject lists of session arrays.

    Ignored trials (``animal_response`` outside {0, 1}) are dropped, matching
    ``baseline_rl`` and the wrapper's ``ignore_policy="exclude"``. The HB and the neural
    models must score the same trials or the held-out comparison is meaningless.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Trial-level frame with ``subject_id``, ``ses_idx``, ``trial``,
        ``animal_response`` and ``earned_reward``.
    session_ids : list, optional
        Restrict to these sessions, in this order.

    Returns
    -------
    tuple of dict
        Choices, rewards and session ids, each keyed by subject id. The session ids
        preserve dataframe order, so a split computed from them matches the one the neural
        models use.
    """
    required = {"subject_id", "ses_idx", "trial", "animal_response", "earned_reward"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"Raw dataframe missing columns required by HBTrainer: {missing}")

    ordered = (
        list(session_ids)
        if session_ids is not None
        else list(dict.fromkeys(raw_df["ses_idx"].tolist()))
    )
    choices: Dict[Any, List[np.ndarray]] = {}
    rewards: Dict[Any, List[np.ndarray]] = {}
    session_ids_by_subject: Dict[Any, List[Any]] = {}

    for session_id in ordered:
        session_df = raw_df[raw_df["ses_idx"] == session_id].sort_values("trial")
        if session_df.empty:
            continue
        choice_arr = session_df["animal_response"].to_numpy(dtype=int)
        valid = (choice_arr == 0) | (choice_arr == 1)
        if not np.any(valid):
            continue
        subject = session_df["subject_id"].iloc[0]
        choices.setdefault(subject, []).append(choice_arr[valid].astype(int))
        rewards.setdefault(subject, []).append(
            session_df["earned_reward"].to_numpy(dtype=float)[valid]
        )
        session_ids_by_subject.setdefault(subject, []).append(session_id)
    return choices, rewards, session_ids_by_subject


def _pad_cohort(
    choices: Mapping[Any, List[np.ndarray]],
    rewards: Mapping[Any, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Pad ragged per-subject sessions into dense arrays with masks.

    Sessions are never truncated: the neural models pad to the longest session and mask,
    so trimming here would change the trial set and void the comparison.

    Returns
    -------
    tuple
        ``(choices, rewards, valid_mask, session_mask, subject_ids)`` where the arrays are
        ``(n_subjects, n_sessions, n_trials)`` and ``session_mask`` is
        ``(n_subjects, n_sessions)``.
    """
    subject_ids = list(choices.keys())
    n_subjects = len(subject_ids)
    n_sessions = max(len(choices[s]) for s in subject_ids)
    n_trials = max(len(a) for s in subject_ids for a in choices[s])

    choice_arr = np.zeros((n_subjects, n_sessions, n_trials), dtype=int)
    reward_arr = np.zeros((n_subjects, n_sessions, n_trials), dtype=float)
    valid_mask = np.zeros((n_subjects, n_sessions, n_trials), dtype=bool)
    session_mask = np.zeros((n_subjects, n_sessions), dtype=bool)

    for i, subject in enumerate(subject_ids):
        for j, (c, r) in enumerate(zip(choices[subject], rewards[subject])):
            choice_arr[i, j, : len(c)] = c
            reward_arr[i, j, : len(r)] = r
            valid_mask[i, j, : len(c)] = True
            session_mask[i, j] = True

    return choice_arr, reward_arr, valid_mask, session_mask, subject_ids


def _pad_context(subjects, choices, rewards, context_indices):
    """Pad each subject's context sessions into one dense batch with masks.

    Subjects contribute different numbers of context sessions, and the batched adaptation
    needs a rectangular array. Padded slots are masked out rather than truncated.

    Returns
    -------
    tuple of np.ndarray
        ``(choices, rewards, session_mask, valid_mask)``, the first two shaped
        ``(n_subjects, max_context, max_trials)``.
    """
    n_subjects = len(subjects)
    max_context = max((len(idx) for idx in context_indices), default=0)
    max_trials = max(
        (len(choices[s][i]) for s, idx in zip(subjects, context_indices) for i in idx),
        default=1,
    )

    choice_arr = np.zeros((n_subjects, max_context, max_trials), dtype=int)
    reward_arr = np.zeros((n_subjects, max_context, max_trials), dtype=float)
    valid = np.zeros((n_subjects, max_context, max_trials), dtype=bool)
    session_mask = np.zeros((n_subjects, max_context), dtype=bool)

    for row, (subject, idx) in enumerate(zip(subjects, context_indices)):
        for slot, i in enumerate(idx):
            c = choices[subject][i]
            choice_arr[row, slot, : len(c)] = c
            reward_arr[row, slot, : len(c)] = rewards[subject][i][: len(c)]
            valid[row, slot, : len(c)] = True
            session_mask[row, slot] = True
    return choice_arr, reward_arr, session_mask, valid


def _log_fit_artifact(wandb_run, saved, estimator, n_subjects):
    """Upload the persisted posterior to W&B so it is retrievable off this filesystem.

    Local netCDF is fine until someone needs the draws from another machine, or after the
    scratch directory is cleaned. Versioning it against the run also ties the posterior to
    the exact config that produced it.
    """
    if wandb_run is None:
        return
    try:
        import wandb

        artifact = wandb.Artifact(
            name=f"hb-fit-{estimator}-D{n_subjects}",
            type="hb_posterior",
            metadata={"estimator": estimator, "n_subjects": n_subjects,
                      **saved.get("diagnostics", {})},
        )
        for key in ("netcdf", "sample_stats", "json"):
            path = saved.get(key)
            if path and Path(path).exists():
                artifact.add_file(path)
        wandb_run.log_artifact(artifact)
        logger.info("HBTrainer: logged posterior artifact %s", artifact.name)
    except Exception as error:  # pragma: no cover - never fail a fit over telemetry
        logger.warning("HBTrainer: could not log posterior artifact: %s", error)


def _log_per_subject_table(wandb_run, per_subject):
    """Publish per-subject held-out scores as a W&B table."""
    if wandb_run is None or not per_subject:
        return
    try:
        import wandb

        columns = ["subject_id", "likelihood", "n_context", "n_scored", "n_trials"]
        table = wandb.Table(columns=columns)
        for subject, row in sorted(per_subject.items()):
            table.add_data(
                subject, float(row["likelihood"]), int(row["n_context"]),
                int(row["n_scored"]), int(row["n_trials"]),
            )
        wandb_run.log({"heldout/per_subject_matched": table})
        logger.info("HBTrainer: logged per-subject table (%d rows)", len(per_subject))
    except Exception as error:  # pragma: no cover - never fail a fit over telemetry
        logger.warning("HBTrainer: could not log per-subject table: %s", error)


def _log_two_level_table(wandb_run, per_subject):
    """Publish ``heldout_two_level`` per-subject scores and gates as a W&B table."""
    try:
        import wandb

        columns = ["subject_id", "likelihood", "likelihood_trial_marginal", "n_context",
                   "n_scored", "n_trials", "max_r_hat", "min_ess_bulk", "min_weight_ess",
                   "max_pareto_k", "n_sessions_pareto_k_fail", "converged"]
        table = wandb.Table(columns=columns)
        for subject, row in sorted(per_subject.items()):
            table.add_data(subject, *(row[c] for c in columns[1:]))
        wandb_run.log({"heldout/per_subject_two_level": table})
    except Exception as error:  # pragma: no cover - never fail a fit over telemetry
        logger.warning("HBTrainer: could not log per-subject table: %s", error)


def _save_two_level_draws(run_dir, samples, diagnostics, subjects, meta):
    """Persist the held-out posterior draws and per-subject diagnostics.

    The fit is per subject with no population, so there is no ``save_fit`` netCDF to write;
    the draws are what a later rescoring needs.
    """
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "heldout_two_level_draws.npz",
        mu_p=np.asarray(samples["mu_p"]), log_sigma=np.asarray(samples["log_sigma"]),
    )
    record = {
        "subject_ids": [str(s) for s in subjects],
        "max_r_hat": np.asarray(diagnostics["max_r_hat"]).tolist(),
        "min_ess_bulk": np.asarray(diagnostics["min_ess_bulk"]).tolist(),
        "divergences": diagnostics["divergences"],
        **meta,
    }
    (run_dir / "heldout_two_level_fit.json").write_text(json.dumps(record, indent=2))
    logger.info("HBTrainer: wrote heldout_two_level draws to %s", run_dir)


def _check_legacy_split(run_dir, n_trials):
    """For draws saved before the split was recorded, check the scored sessions by trial count.

    Such a fit's JSON has no ``sessions``; its scores file does have each scored session's
    trial count, in order. A changed session list (a session dropped or added upstream) moves
    the context/eval split and so changes those counts. Without either record the split cannot
    be verified and the draws are refused.
    """
    import json

    record = json.loads((run_dir / "heldout_two_level_fit.json").read_text())
    if "sessions" in record:
        return                       # checked exactly, by session id, on load
    scores = run_dir / "heldout_two_level_scores.npz"
    if not scores.exists():
        raise ValueError(f"draws_path {run_dir} records no session split and has no scores "
                         "file to check one against; refusing to rescore it.")
    saved = np.asarray(np.load(scores)["n_trials"])
    if saved.shape != np.shape(n_trials) or not np.array_equal(saved, n_trials):
        raise ValueError(f"draws_path {run_dir}: the sessions to score differ from those the fit "
                         "scored (trial counts disagree); the context/eval split has moved.")


def _resolve_draws_dir(draws_path, subjects):
    """The run directory for ``subjects``: ``draws_path`` itself, or its entry in a JSON index."""
    import json

    if draws_path.suffix != ".json":
        return draws_path
    index = json.loads(draws_path.read_text())
    key = ",".join(sorted(str(s) for s in subjects))
    if key not in index:
        raise ValueError(f"draws index {draws_path} has no entry for subjects {key!r}.")
    entry = Path(index[key])
    return entry if entry.is_absolute() else draws_path.parent / entry


def _load_two_level_draws(run_dir, subjects, expected=None):
    """Draws and diagnostics saved by :func:`_save_two_level_draws`, checked against ``subjects``.

    The draws carry a subject axis in the order the fit saw them; scoring them against a
    different subject order would silently pair one subject's posterior with another's
    sessions, so the order must match exactly. So must every setting in ``expected`` that the
    fit recorded: ``eval_every_n``, the likelihood's ``beta_max``, and the context and eval
    session ids of each subject (fits that predate that record are checked by
    :func:`_check_legacy_split`).
    """
    import json

    record = json.loads((run_dir / "heldout_two_level_fit.json").read_text())
    if record["subject_ids"] != [str(s) for s in subjects]:
        raise ValueError(
            f"draws_path {run_dir} holds subjects {record['subject_ids']}, but this run scores "
            f"{[str(s) for s in subjects]}; rescore with the same data.test_subject_ids."
        )
    # Fits saved before a setting was recorded used its default then.
    record = {"hb_model": "hattori2019", **record}
    for key, value in (expected or {}).items():
        if key in record and record[key] != value:
            raise ValueError(
                f"draws_path {run_dir} was fitted with {key}={record[key]!r}, but this run uses "
                f"{value!r}; the saved posterior would be scored against a different split or "
                "likelihood.")
    draws = np.load(run_dir / "heldout_two_level_draws.npz")
    samples = {"mu_p": np.asarray(draws["mu_p"]), "log_sigma": np.asarray(draws["log_sigma"])}
    diagnostics = {
        "max_r_hat": np.asarray(record["max_r_hat"]),
        "min_ess_bulk": np.asarray(record["min_ess_bulk"]),
        "divergences": record["divergences"],
    }
    logger.info("HBTrainer: loaded heldout_two_level draws from %s; fit skipped", run_dir)
    return samples, diagnostics


def _source_revisions():
    """Git SHAs of the repositories whose code produced a fit.

    The model lives in aind-dynamic-foraging-models and the orchestration here, so a
    dispatcher SHA alone pins none of the code that produced a number.
    """
    import subprocess

    out = {}
    for label, module in (("wrapper", None), ("models", "aind_dynamic_foraging_models")):
        try:
            if module is None:
                path = Path(__file__).resolve().parents[2]
            else:
                path = Path(__import__(module).__file__).resolve().parents[2]
            sha = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            out[f"{label}_git_sha"] = sha
        except Exception:  # pragma: no cover - absent git or a non-repo install
            out[f"{label}_git_sha"] = None
    return out


def _flatten_for_scoring(subjects, choices, rewards, score_indices):
    """Flatten the sessions to be scored into one padded batch.

    Scoring every held-out session in a single vmapped pass needs a rectangular array and a
    map from each row back to the subject whose adapted posterior it should use.

    Returns
    -------
    dict
        ``choices`` and ``rewards`` of shape ``(n_rows, max_trials)``, a ``valid_mask``,
        ``subject_indices`` giving each row's position in the batched fit, and
        ``rows_by_subject`` for aggregating results back per subject.
    """
    rows, subject_indices, rows_by_subject = [], [], []
    for position, (subject, idx) in enumerate(zip(subjects, score_indices)):
        rows_by_subject.append([])
        for i in idx:
            rows_by_subject[position].append(len(rows))
            rows.append((subject, i))
            subject_indices.append(position)

    max_trials = max((len(choices[s][i]) for s, i in rows), default=1)
    choice_arr = np.zeros((len(rows), max_trials), dtype=int)
    reward_arr = np.zeros((len(rows), max_trials), dtype=float)
    valid = np.zeros((len(rows), max_trials), dtype=bool)
    for row, (subject, i) in enumerate(rows):
        c = choices[subject][i]
        choice_arr[row, : len(c)] = c
        reward_arr[row, : len(c)] = rewards[subject][i][: len(c)]
        valid[row, : len(c)] = True

    return {
        "choices": choice_arr,
        "rewards": reward_arr,
        "valid_mask": valid,
        "subject_indices": np.asarray(subject_indices, dtype=int),
        "rows_by_subject": [np.asarray(r, dtype=int) for r in rows_by_subject],
    }


def _normalized_likelihood(total_log_lik: float, total_trials: int) -> float:
    """Geometric-mean per-trial likelihood, the metric shared with the neural models."""
    if total_trials == 0:
        return 0.0
    return float(np.exp(total_log_lik / total_trials))


class HBTrainer(ModelTrainer):
    """Fit a hierarchical Bayesian cognitive model and score held-out subjects."""

    def __init__(
        self,
        estimator: str = "one_stage",
        num_warmup: int = 500,
        num_samples: int = 500,
        num_chains: int = 4,
        beta_max: float = 10.0,
        few_shot_k: Any = FEW_SHOT_K,
        eval_every_n: int = 2,
        artifact_dir: Optional[str] = None,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        save_session_sites: bool = True,
        progress_bar: bool = False,
        progress_rate: int | None = None,
        session_draws_per_posterior_draw: int = 64,
        centered: bool = False,
        draws_path: Optional[str] = None,
        importance_rescore: bool = False,
        importance_draws: int = 24000,
        hb_model: str = "hattori2019",
        architecture: Mapping[str, Any] | Any = {},
        output_dir: str = "/results/outputs",
        seed: Optional[int] = None,
        **_: Any,
    ) -> None:
        """Initialize the HBTrainer.

        Arguments arrive from Hydra's ``instantiate`` on the model config, so each is a
        named parameter rather than an opaque config object -- the same contract the other
        trainers use.

        Parameters
        ----------
        estimator : str
            ``"one_stage"`` for the joint three-level fit, or ``"two_stage"`` for empirical
            Bayes. One-stage is both the reference estimator and, on GPU, the cheaper one.
        num_warmup, num_samples, num_chains : int
            NUTS settings.
        beta_max : float
            Upper bound of ``softmax_inverse_temperature``.
        few_shot_k : sequence of int
            Context-session counts to score, alongside the matched rung.
        eval_every_n : int
            Per-subject train/eval session split, matching the neural models'.
        artifact_dir : str, optional
            Where to persist posterior draws and diagnostics.
        target_accept_prob : float
            **Applies to ``estimator="one_stage"`` only.** The ``two_stage`` route runs
            through ``fit_two_stage``, whose subject and population kernels are constructed
            without geometry arguments, so setting this under ``two_stage`` has no effect.
            NUTS target acceptance probability; NumPyro's default is 0.8 and Stan's
            ``adapt_delta`` is the same quantity. Raising it to 0.9-0.95 shortens the
            integrator's steps, the standard remedy for divergences that survive longer
            warmup, at the cost of a slower fit. Exposed because divergences are curvature
            the integrator cannot follow, so they are **not** fixed by more draws.
        max_tree_depth : int
            **Applies to ``estimator="one_stage"`` only**, for the same reason as
            ``target_accept_prob``.
            NUTS maximum trajectory depth (NumPyro default 10). Raise only if the sampler
            reports saturating it; a saturated tree means the step size is too small for
            the posterior's scale, which is a different failure from divergence.
        save_session_sites : bool
            Persist the session level too (``save_fit``'s ``include_session_sites``, which
            defaults to off). On the ``one_stage`` route that adds ``session_log_lik``,
            which is the bulk of the artifact -- but a site exists only while the sampler
            does, so a fit that ships without it can be recovered only by re-fitting, at
            hours per fit. Defaults on for that reason; turn it off for a throwaway fit.
        progress_bar : bool
            Emit NumPyro's sampling progress. Off by default, which is the behaviour every
            existing caller had. Worth turning on for any multi-hour rung: a population fit
            otherwise logs nothing between "starting" and "fitted in Ns", so a silent job
            cannot be told apart from a hung one -- three ladder rungs sat at 32 h before
            this existed with no way to know. It is not free, though: NumPyro's docs note
            that disabling the bar "will improve the speed for many cases, but it might
            require more memory", because the bar breaks the sampler's scan into chunks.
            That trade reverses at large ``D``, where memory rather than speed binds, so
            this is a per-rung decision.
        progress_rate : int, optional
            Iterations per progress update. Left as ``None`` NumPyro uses 5% of total
            iterations, which is what keeps a captured log readable -- roughly 20 lines for
            a whole fit. Set it only to override that: tqdm writes to stderr, which Beaker
            captures, so a small value here buries the log rather than illuminating it.
        session_draws_per_posterior_draw : int
            **Applies to ``estimator="heldout_two_level"`` only.** Fresh session-parameter
            draws per posterior draw when scoring a held-out session by its session-marginal
            likelihood. That estimate rests on the few draws that fit the whole session, so
            it needs far more draws than the trial-marginal; each session's weight ESS is
            reported so a starved estimate is visible.
        centered : bool
            **``heldout_two_level`` only.** Sample the session level centred,
            ``theta ~ Normal(mu_p, sigma)``, instead of non-centred: the one refit PLAN.md
            allows a subject that failed the gates (study 13).
        draws_path : str, optional
            **``heldout_two_level`` only.** A run directory holding
            ``heldout_two_level_draws.npz`` and ``heldout_two_level_fit.json`` from an
            earlier fit of the same subjects. The fit is skipped and those draws (and their
            diagnostics) are scored, so a rescore uses the identical data path. With the same
            seed and subjects the plain scores reproduce the original run exactly. May
            instead be a ``.json`` file mapping each subject id (several, comma-joined in
            sorted order) to its run directory, so a sweep with one subject per run can share
            one index.
        importance_rescore : bool
            **``heldout_two_level`` only.** Rescore every session that fails the Pareto k-hat
            gate by importance sampling (foraging-models'
            ``importance_sampled_session_marginal``), which estimates the same
            session-marginal with weights that do not starve. The headline then uses the
            importance-sampled value for those sessions; the all-plain value is kept beside
            it under ``*_plain`` keys.
        importance_draws : int
            Proposal draws per rescored session.
        hb_model : str
            **``heldout_two_level`` only.** The forager at the session level: ``"hattori2019"``
            (default) or ``"ctt"`` (compare-to-threshold), as foraging-models'
            ``get_model`` names them. Recorded with the draws; a rescore refuses draws of
            another model.
        architecture : mapping
            Present for parity with the other trainers; unused.
        output_dir : str
            Run output directory.
        seed : int, optional
            Seed for the sampler.
        """
        super().__init__(seed=seed)
        self.estimator = str(estimator)
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.beta_max = float(beta_max)
        self.few_shot_k = tuple(few_shot_k)
        self.eval_every_n = int(eval_every_n)
        self.artifact_dir = artifact_dir
        # Validated here because NumPyro does not. Verified against numpyro 0.21.0:
        # NUTS(target_accept_prob=9.5) constructs AND samples to completion without any
        # error, so a config typo (9.5 for 0.95) yields a fit whose step-size adaptation
        # targeted an unreachable acceptance rate, with nothing in the log to say so --
        # the diagnostics look like a hard posterior. max_tree_depth=0 does fail, but as
        # `IndexError: index is out of bounds for axis 0 with size 0` from inside the
        # integrator, hours into a run and with nothing pointing at the config.
        self.target_accept_prob = float(target_accept_prob)
        if not 0.0 < self.target_accept_prob < 1.0:
            raise ValueError(
                f"target_accept_prob must be in (0, 1); got {self.target_accept_prob}. "
                "It is a probability, the same quantity as Stan's adapt_delta -- 0.95, "
                "not 95."
            )
        self.max_tree_depth = int(max_tree_depth)
        if self.max_tree_depth < 1:
            raise ValueError(
                f"max_tree_depth must be >= 1; got {self.max_tree_depth}. It is a "
                "log2 bound on trajectory length (NumPyro's default is 10)."
            )
        self.save_session_sites = bool(save_session_sites)
        self.progress_bar = bool(progress_bar)
        self.progress_rate = None if progress_rate is None else int(progress_rate)
        self.session_draws_per_posterior_draw = int(session_draws_per_posterior_draw)
        self.centered = bool(centered)
        self.draws_path = draws_path
        self.importance_rescore = bool(importance_rescore)
        self.importance_draws = int(importance_draws)
        self.hb_model = str(hb_model)
        self.output_dir = output_dir

    def fit(
        self,
        bundle: DatasetBundle,
        loggers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fit the model on the bundle's training split and score held-out subjects.

        Parameters
        ----------
        bundle : DatasetBundle
            Shared dataset material, identical to what the neural models receive.
        loggers : dict, optional
            May contain a ``"wandb"`` run.

        Returns
        -------
        dict
            Fitted population parameters, timings and held-out likelihoods by ``k``.
        """
        import jax

        from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
            batched_heldout_log_lik,
            fit_adaptation_batched,
        )

        started = time.time()
        wandb_run = (loggers or {}).get("wandb")
        estimator = self.estimator
        num_warmup, num_samples = self.num_warmup, self.num_samples
        num_chains, beta_max = self.num_chains, self.beta_max
        k_values, eval_every_n = self.few_shot_k, self.eval_every_n
        artifact_dir = self.artifact_dir

        if estimator == "heldout_two_level":
            return self._fit_heldout_two_level(bundle, wandb_run, started)

        if bundle.raw is None or len(bundle.raw) == 0:
            raise ValueError("HBTrainer requires bundle.raw with trial-level rows.")

        choices, rewards, _ = _extract_subject_sessions(bundle.raw)
        choice_arr, reward_arr, valid_mask, session_mask, subject_ids = _pad_cohort(
            choices, rewards
        )
        logger.info(
            "HBTrainer: %d subjects, up to %d sessions, up to %d trials (estimator=%s)",
            len(subject_ids), choice_arr.shape[1], choice_arr.shape[2], estimator,
        )

        population, fit_info, mcmc = self._fit_population(
            estimator, choice_arr, reward_arr, valid_mask, session_mask,
            num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
            beta_max=beta_max,
        )
        fit_seconds = time.time() - started

        # Persist the fit before scoring. A cohort fit costs hours and scoring adds more;
        # keeping only posterior means would force a refit for every later question.
        if artifact_dir and mcmc is not None:
            from aind_dynamic_foraging_models.hierarchical_bayes.artifacts import save_fit

            saved = save_fit(
                mcmc, artifact_dir, name=f"{estimator}_fit",
                include_session_sites=self.save_session_sites,
                meta={
                    "estimator": estimator,
                    "n_subjects": len(subject_ids),
                    "subject_ids": [str(s) for s in subject_ids],
                    "num_warmup": num_warmup,
                    "num_samples": num_samples,
                    "num_chains": num_chains,
                    # Needed to invert the softmax_inverse_temperature transform when
                    # per-session parameters are reconstructed offline from theta_raw
                    # (phi(theta[..., 3]) * beta_max). It is a config key, so a rung can
                    # override the published 10.0; without it recorded, an offline replay
                    # would silently rescale beta. See dispatcher #115.
                    "beta_max": beta_max,
                    # Recorded so a rung's convergence can be read against the settings
                    # that produced it, without going back to the launch config -- but
                    # only for the estimator they actually reached. two_stage samples via
                    # fit_two_stage, whose kernels take no geometry arguments, so writing
                    # these there would state a setting the fit never used: an analyst
                    # reading target_accept_prob 0.95 off a two_stage artifact would be
                    # reading a number that never left this file. `estimator` is in the
                    # same meta, so their absence is self-explaining.
                    **(
                        {
                            "target_accept_prob": self.target_accept_prob,
                            "max_tree_depth": self.max_tree_depth,
                        }
                        if estimator == "one_stage"
                        else {}
                    ),
                    "seed": self.seed,
                    **_source_revisions(),
                },
            )
            fit_info["artifacts"] = saved
            logger.info("HBTrainer: wrote fit artifacts to %s", saved["netcdf"])
            _log_fit_artifact(wandb_run, saved, estimator, len(subject_ids))
        logger.info(
            "HBTrainer: population fitted in %.0fs: %s",
            fit_seconds, {k: np.round(np.asarray(v), 4).tolist() for k, v in population.items()},
        )
        # Optional hook so long runs can checkpoint the population before scoring starts.
        callback = getattr(self, "on_population_fitted", None)
        if callable(callback):
            callback({k: np.asarray(v).tolist() for k, v in population.items()}, dict(fit_info))

        # -- Held-out evaluation, one adaptation per (subject, k) --
        # The held-out cohort must be supplied explicitly. Falling back to bundle.raw
        # would adapt and score on the very subjects the population was fitted to, which
        # inflates every number silently; better to skip and say so.
        heldout_df = (bundle.extras or {}).get("heldout_raw")
        if heldout_df is None or len(heldout_df) == 0:
            logger.warning(
                "HBTrainer: no bundle.extras['heldout_raw']; skipping held-out evaluation. "
                "Scoring the training cohort would leak."
            )
            return {
                "estimator": estimator,
                "n_subjects": len(subject_ids),
                "fit_seconds": fit_seconds,
                "population": {
                    name: np.asarray(value).tolist() for name, value in population.items()
                },
                "heldout_likelihood": {},
                "heldout_skipped": True,
                **fit_info,
            }

        heldout_choices, heldout_rewards, heldout_session_ids = (
            _extract_subject_sessions(heldout_df)
        )
        scores: Dict[Any, float] = {}
        rng_key = jax.random.PRNGKey(int(self.seed or 0))
        # One batched adaptation per rung rather than one per subject. Subjects are
        # independent given the frozen population, and sequential fitting paid the scan
        # depth once per subject: about four hours per rung over this cohort.
        for k in k_values:
            eligible = [s for s, sess in heldout_choices.items() if len(sess) > k]
            if not eligible:
                scores[k] = 0.0
                continue
            key_fit, key_draw, rng_key = jax.random.split(rng_key, 3)
            context_c, context_r, ctx_session_mask, ctx_valid = _pad_context(
                eligible, heldout_choices, heldout_rewards, [list(range(k))] * len(eligible)
            )
            samples = fit_adaptation_batched(
                context_c, context_r, population, rng_key=key_fit,
                session_mask=ctx_session_mask, valid_mask=ctx_valid,
                num_warmup=num_warmup, num_samples=num_samples, beta_max=beta_max,
            )
            score_idx = [
                list(range(k, len(heldout_choices[subject]))) for subject in eligible
            ]
            flat = _flatten_for_scoring(
                eligible, heldout_choices, heldout_rewards, score_idx
            )
            session_log_lik, session_trials = batched_heldout_log_lik(
                samples, flat["subject_indices"], flat["choices"], flat["rewards"],
                valid_mask=flat["valid_mask"], rng_key=key_draw, beta_max=beta_max,
            )
            total_log_lik = float(np.sum(session_log_lik))
            total_trials = int(np.sum(session_trials))
            scores[k] = _normalized_likelihood(total_log_lik, total_trials)
            logger.info("HBTrainer: k=%d heldout likelihood %.5f", k, scores[k])

        # Matched conditioning: condition on exactly the sessions the per-mouse MLE
        # baseline fits, and score exactly the ones it scores. Without this the HB is
        # compared against MLE across different amounts of conditioning, which is not a
        # comparison. eval_every_n mirrors study 01's data config.
        matched_log_lik, matched_trials = 0.0, 0
        per_subject_matched: Dict[str, Any] = {}
        matched_subjects, context_indices, score_indices = [], [], []
        for subject, ids in heldout_session_ids.items():
            if len(ids) < 2:
                continue
            train_ids, eval_ids = compute_train_eval_session_ids(ids, eval_every_n)
            index_of = {sid: i for i, sid in enumerate(ids)}
            matched_subjects.append(subject)
            context_indices.append([index_of[sid] for sid in train_ids])
            score_indices.append([index_of[sid] for sid in eval_ids])

        key_fit, key_draw, rng_key = jax.random.split(rng_key, 3)
        context_c, context_r, ctx_session_mask, ctx_valid = _pad_context(
            matched_subjects, heldout_choices, heldout_rewards, context_indices
        )
        matched_samples = fit_adaptation_batched(
            context_c, context_r, population, rng_key=key_fit,
            session_mask=ctx_session_mask, valid_mask=ctx_valid,
            num_warmup=num_warmup, num_samples=num_samples, beta_max=beta_max,
        )

        flat = _flatten_for_scoring(
            matched_subjects, heldout_choices, heldout_rewards, score_indices
        )
        session_log_lik, session_trials = batched_heldout_log_lik(
            matched_samples, flat["subject_indices"], flat["choices"], flat["rewards"],
            valid_mask=flat["valid_mask"], rng_key=key_draw, beta_max=beta_max,
        )
        for position, subject in enumerate(matched_subjects):
            rows = flat["rows_by_subject"][position]
            subject_log_lik = float(np.sum(session_log_lik[rows]))
            subject_trials = int(np.sum(session_trials[rows]))
            per_subject_matched[str(subject)] = {
                "likelihood": _normalized_likelihood(subject_log_lik, subject_trials),
                "n_context": len(context_indices[position]),
                "n_scored": len(score_indices[position]),
                "n_trials": subject_trials,
            }
        matched_log_lik = float(np.sum(session_log_lik))
        matched_trials = int(np.sum(session_trials))
        scores["matched"] = _normalized_likelihood(matched_log_lik, matched_trials)
        logger.info(
            "HBTrainer: matched-conditioning heldout likelihood %.5f (eval_every_n=%d)",
            scores["matched"], eval_every_n,
        )

        output = {
            "estimator": estimator,
            "n_subjects": len(subject_ids),
            "fit_seconds": fit_seconds,
            "population": {
                name: np.asarray(value).tolist() for name, value in population.items()
            },
            "heldout_likelihood": scores,
            "heldout_per_subject_matched": per_subject_matched,
            **fit_info,
        }

        if wandb_run is not None:
            for k, value in scores.items():
                key = ("heldout/matched_likelihood" if k == "matched"
                       else f"heldout/few_shot_k{k}_likelihood")
                wandb_run.summary[key] = float(value)
            # Cross-model parity. The GRU's scaling y-axis is a held-out fine-tune on each
            # subject's train sessions scored on its eval sessions, and the per-mouse MLE
            # baseline uses the same split. Our matched rung is that same protocol, so it is
            # what belongs under the shared keys; the k sweep is reported separately.
            if "matched" in scores:
                wandb_run.summary["heldout/eval_likelihood"] = float(scores["matched"])
                wandb_run.summary["heldout/test_likelihood"] = float(scores["matched"])
                wandb_run.summary["heldout_test_likelihood"] = float(scores["matched"])
            wandb_run.summary["heldout/num_test_trials"] = int(matched_trials)
            wandb_run.summary["heldout/num_test_subjects"] = int(len(per_subject_matched))
            # Per-subject scores as a table: the existing GRU-vs-MLE claim is a paired
            # per-mouse test, which needs the per-subject numbers rather than the aggregate.
            _log_per_subject_table(wandb_run, per_subject_matched)
            wandb_run.summary["hb/fit_seconds"] = float(fit_seconds)
            wandb_run.summary["hb/estimator"] = estimator
            if fit_info.get("divergences") is not None:
                wandb_run.summary["hb/divergences"] = int(fit_info["divergences"])

        # Figures for the fit that just cost hours, built before this process exits. A run
        # that reports a likelihood and no diagnostics cannot be judged: divergences and
        # ESS decide whether its number may be quoted at all. Regenerating them later means
        # reloading the artifact somewhere the bayes extra is installed, so the cheap moment
        # is now, while the fit is already in hand. Comparator lines are deliberately not
        # passed -- those are study constants and belong to the study's analysis, not to
        # runtime code that would carry them stale.
        if fit_info.get("artifacts"):
            try:
                from utils.hb_figures import log_hb_figures

                log_hb_figures(
                    fit_info["artifacts"], scores=scores, wandb_run=wandb_run,
                    output_dir=artifact_dir, beta_max=beta_max,
                )
            except Exception:
                logger.exception("Could not generate HB figures; continuing.")

        return output

    def _fit_heldout_two_level(self, bundle, wandb_run, started):
        """Fit each held-out subject alone on its context sessions, then score its eval sessions.

        There is no population level and the training cohort is never fitted: each subject
        gets a two-level model under ``FIXED_HYPERPRIOR``, conditioned on the sessions the
        per-mouse MLE baseline fits and scored on the ones it scores (``eval_every_n``). The
        headline is the session-marginal likelihood; the trial-marginal that the other
        estimators report is kept beside it. Nothing is written under
        ``heldout/test_likelihood``: for the GRU that key is a zero-shot score, and with no
        population this estimator has none.
        """
        import jax

        from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
            batched_heldout_scores,
            fit_heldout_two_level,
            importance_sampled_session_marginal,
            pareto_k_threshold,
            pooled_likelihood,
        )

        heldout_df = (bundle.extras or {}).get("heldout_raw")
        if heldout_df is None or len(heldout_df) == 0:
            raise ValueError(
                "estimator='heldout_two_level' fits and scores held-out subjects only, so it "
                "needs bundle.extras['heldout_raw']."
            )
        choices, rewards, session_ids = _extract_subject_sessions(heldout_df)
        subjects, context_indices, score_indices = [], [], []
        split: Dict[str, Dict[str, List[str]]] = {}
        for subject, ids in session_ids.items():
            if len(ids) < 2:
                continue
            train_ids, eval_ids = compute_train_eval_session_ids(ids, self.eval_every_n)
            index_of = {sid: i for i, sid in enumerate(ids)}
            split[str(subject)] = {"context": [str(i) for i in train_ids],
                                   "eval": [str(i) for i in eval_ids]}
            subjects.append(subject)
            context_indices.append([index_of[sid] for sid in train_ids])
            score_indices.append([index_of[sid] for sid in eval_ids])

        # Fold the subject set into the key: runs that each fit a different subset of subjects
        # (data.test_subject_ids) then draw from independent streams even under one seed, so
        # their Monte Carlo errors pool in quadrature; the same subjects reproduce exactly.
        subset_tag = zlib.crc32(",".join(sorted(map(str, subjects))).encode()) & 0x7FFFFFFF
        key_fit, key_draw = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(int(self.seed or 0)), subset_tag))
        context_c, context_r, ctx_session_mask, ctx_valid = _pad_context(
            subjects, choices, rewards, context_indices
        )
        draws_dir = None
        if self.draws_path:
            draws_dir = _resolve_draws_dir(Path(self.draws_path), subjects)
            samples, diagnostics = _load_two_level_draws(
                draws_dir, subjects,
                expected={"eval_every_n": self.eval_every_n, "beta_max": self.beta_max,
                          "sessions": split, "hb_model": self.hb_model})
        else:
            samples, diagnostics = fit_heldout_two_level(
                context_c, context_r, rng_key=key_fit, session_mask=ctx_session_mask,
                valid_mask=ctx_valid, num_warmup=self.num_warmup,
                num_samples=self.num_samples, num_chains=self.num_chains,
                beta_max=self.beta_max, progress_bar=self.progress_bar,
                centered=self.centered, target_accept_prob=self.target_accept_prob,
                max_tree_depth=self.max_tree_depth, model=self.hb_model,
            )
        fit_seconds = time.time() - started
        # Saved before scoring, so a failure there cannot lose the fit. One subdirectory per
        # run: a sweep's runs share artifact_dir, and the file names are fixed. A rescore
        # writes no draws: they already live under draws_path.
        run_dir = None
        if self.artifact_dir:
            run_tag = str(getattr(wandb_run, "id", None)
                          or f"seed{self.seed}-{uuid.uuid4().hex[:8]}")
            run_dir = Path(self.artifact_dir) / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)
            if not self.draws_path:
                _save_two_level_draws(run_dir, samples, diagnostics, subjects, meta={
                    "seed": self.seed, "num_chains": self.num_chains,
                    "num_warmup": self.num_warmup, "num_samples": self.num_samples,
                    "beta_max": self.beta_max, "eval_every_n": self.eval_every_n,
                    "session_draws_per_posterior_draw": self.session_draws_per_posterior_draw,
                    "centered": self.centered, "target_accept_prob": self.target_accept_prob,
                    "max_tree_depth": self.max_tree_depth, "sessions": split,
                    "hb_model": self.hb_model,
                    **_source_revisions(),
                })

        flat = _flatten_for_scoring(subjects, choices, rewards, score_indices)
        if draws_dir is not None:
            _check_legacy_split(draws_dir, np.sum(np.asarray(flat["valid_mask"]), axis=-1))
        scored = batched_heldout_scores(
            samples, flat["subject_indices"], flat["choices"], flat["rewards"],
            valid_mask=flat["valid_mask"], rng_key=key_draw, beta_max=self.beta_max,
            n_theta_per_draw=self.session_draws_per_posterior_draw, model=self.hb_model,
        )

        n_draws = int(np.asarray(samples["mu_p"]).shape[0]) * self.session_draws_per_posterior_draw
        k_threshold = pareto_k_threshold(n_draws)
        # nan means unjudged, which fails the gate like a high k-hat.
        k_fails = ~(np.asarray(scored["pareto_k"]) < k_threshold)

        # Sessions whose plain draw-average starved are rescored by importance sampling; the
        # headline takes those values, and the all-plain ones are kept as *_plain.
        plain_marginal = np.asarray(scored["session_marginal"], dtype=float)
        headline = plain_marginal.copy()
        rescored = np.zeros(len(headline), dtype=bool)
        is_se = np.zeros(len(headline))
        is_k = np.full(len(headline), np.nan)
        is_ess = np.full(len(headline), np.nan)
        if self.importance_rescore and np.any(k_fails):
            key_is = jax.random.fold_in(key_draw, 1)
            mu_all = np.asarray(samples["mu_p"])
            log_sigma_all = np.asarray(samples["log_sigma"])
            for position in range(len(subjects)):
                rows = np.asarray(flat["rows_by_subject"][position], dtype=int)
                rows = rows[k_fails[rows]]
                if rows.size == 0:
                    continue
                result = importance_sampled_session_marginal(
                    mu_all[:, position, :], log_sigma_all[:, position, :],
                    np.asarray(flat["choices"])[rows], np.asarray(flat["rewards"])[rows],
                    valid_mask=np.asarray(flat["valid_mask"])[rows],
                    rng_key=jax.random.fold_in(key_is, position), beta_max=self.beta_max,
                    n_proposal=self.importance_draws, model=self.hb_model,
                )
                headline[rows] = result["session_marginal"]
                is_se[rows] = result["session_marginal_se"]
                is_k[rows] = result["pareto_k"]
                is_ess[rows] = result["weight_ess"]
                rescored[rows] = True
        is_threshold = pareto_k_threshold(self.importance_draws)
        # After rescoring, a session fails only if its final estimator fails its own gate.
        final_fails = np.where(rescored, ~(is_k < is_threshold), k_fails)
        per_subject: Dict[str, Any] = {}
        converged_rows: List[int] = []
        for position, subject in enumerate(subjects):
            rows = flat["rows_by_subject"][position]
            n_trials = int(np.sum(scored["n_trials"][rows]))
            converged = bool(
                diagnostics["max_r_hat"][position] < GATE_MAX_R_HAT
                and diagnostics["min_ess_bulk"][position] >= GATE_MIN_ESS_BULK
            )
            if converged:
                converged_rows.extend(int(r) for r in rows)
            per_subject[str(subject)] = {
                "likelihood": _normalized_likelihood(float(np.sum(headline[rows])), n_trials),
                "likelihood_plain": _normalized_likelihood(
                    float(np.sum(plain_marginal[rows])), n_trials),
                "likelihood_trial_marginal": _normalized_likelihood(
                    float(np.sum(scored["trial_marginal"][rows])), n_trials),
                "n_context": len(context_indices[position]),
                "n_scored": len(score_indices[position]),
                "n_trials": n_trials,
                "max_r_hat": float(diagnostics["max_r_hat"][position]),
                "min_ess_bulk": float(diagnostics["min_ess_bulk"][position]),
                "min_weight_ess": float(np.min(scored["weight_ess"][rows])),
                # -inf (identical weights) is a judged pass; nan only if all are unjudged.
                "max_pareto_k": float(np.nanmax(scored["pareto_k"][rows]))
                if not np.all(np.isnan(scored["pareto_k"][rows])) else float("nan"),
                "n_sessions_pareto_k_fail": int(np.sum(final_fails[rows])),
                "n_sessions_pareto_k_fail_plain": int(np.sum(k_fails[rows])),
                "n_sessions_importance": int(np.sum(rescored[rows])),
                "converged": converged,
            }

        n_trials_all = np.asarray(scored["n_trials"])
        relative = np.asarray(scored["relative_weights"])
        plain_likelihood, plain_se = pooled_likelihood(plain_marginal, n_trials_all, relative)
        n_eval_trials = int(np.sum(n_trials_all))
        if rescored.any():
            # The error from the finite posterior draws is shared by every session, rescored
            # or not, so it is kept whole: the plain joint error over all sessions, from their
            # per-draw weights. The importance-sampled sessions add their proposal error, which
            # is independent of it given the draws, in quadrature.
            sum_se_plain = plain_se * n_eval_trials / plain_likelihood
            sum_se = float(np.hypot(sum_se_plain, np.sqrt(np.sum(is_se[rescored] ** 2))))
            likelihood = _normalized_likelihood(float(np.sum(headline)), n_eval_trials)
            likelihood_se = likelihood * sum_se / n_eval_trials
        else:
            likelihood, likelihood_se = plain_likelihood, plain_se
            sum_se = likelihood_se * n_eval_trials / likelihood
        trial_marginal, _ = pooled_likelihood(scored["trial_marginal"], n_trials_all)
        converged_only = (
            _normalized_likelihood(float(np.sum(headline[converged_rows])),
                                   int(np.sum(n_trials_all[converged_rows])))
            if converged_rows else None
        )
        summary = {
            # Totals, so runs that each fit a subset of subjects (data.test_subject_ids) pool
            # exactly: exp(sum of log_lik_sum / sum of trials), errors in quadrature, since
            # separate runs share no posterior draws and their keys differ by subject set.
            "heldout/eval_log_lik_sum": float(np.sum(headline)),
            "heldout/eval_log_lik_sum_se": sum_se,
            "heldout/eval_log_lik_sum_plain": float(np.sum(plain_marginal)),
            "heldout/eval_log_lik_sum_plain_se": plain_se * n_eval_trials / plain_likelihood,
            "heldout/eval_likelihood_plain": plain_likelihood,
            "heldout/n_sessions_importance": int(np.sum(rescored)),
            "heldout/importance_pareto_k_threshold": is_threshold,
            "heldout/eval_log_lik_sum_trial_marginal": float(np.sum(scored["trial_marginal"])),
            "heldout/eval_likelihood": likelihood,
            "heldout/eval_likelihood_se": likelihood_se,
            "heldout/eval_likelihood_trial_marginal": trial_marginal,
            "heldout/eval_likelihood_converged_only": converged_only,
            "heldout/num_eval_trials": n_eval_trials,
            "heldout/num_eval_subjects": len(per_subject),
            "heldout/min_weight_ess": float(np.min(scored["weight_ess"])),
            "heldout/median_weight_ess": float(np.median(scored["weight_ess"])),
            "heldout/pareto_k_threshold": k_threshold,
            "heldout/n_sessions_pareto_k_fail": int(np.sum(final_fails)),
            "heldout/n_sessions_pareto_k_fail_plain": int(np.sum(k_fails)),
            "heldout/n_sessions": int(len(k_fails)),
            "hb/n_subjects_converged": sum(row["converged"] for row in per_subject.values()),
            "hb/max_r_hat": float(np.max(diagnostics["max_r_hat"])),
            "hb/min_ess_bulk": float(np.min(diagnostics["min_ess_bulk"])),
            "hb/divergences": diagnostics["divergences"],
            "hb/fit_seconds": float(fit_seconds),
            "hb/estimator": "heldout_two_level",
            "hb/model": self.hb_model,
        }
        logger.info(
            "HBTrainer: heldout_two_level eval likelihood %.5f (se %.5f; trial-marginal %.5f); "
            "%d/%d subjects pass the gates",
            likelihood, likelihood_se, trial_marginal, summary["hb/n_subjects_converged"],
            len(per_subject),
        )
        if wandb_run is not None:
            for key, value in summary.items():
                if value is not None:
                    wandb_run.summary[key] = value
            _log_two_level_table(wandb_run, per_subject)
        if run_dir is not None:
            np.savez_compressed(
                run_dir / "heldout_two_level_scores.npz",
                subject_indices=np.asarray(flat["subject_indices"]),
                **{k: np.asarray(v) for k, v in scored.items()},
                session_marginal_headline=headline, importance_rescored=rescored,
                importance_se=is_se, importance_pareto_k=is_k, importance_weight_ess=is_ess,
            )
        return {
            "estimator": "heldout_two_level",
            "n_subjects": len(subjects),
            "fit_seconds": fit_seconds,
            "heldout_likelihood": {"matched": likelihood},
            "heldout_summary": summary,
            "heldout_per_subject_matched": per_subject,
        }

    def _fit_population(
        self, estimator, choice_arr, reward_arr, valid_mask, session_mask,
        *, num_warmup, num_samples, num_chains, beta_max,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Any]:
        """Fit the population level by the requested estimator.

        Returns
        -------
        tuple
            Population point estimates, fit diagnostics, and the sampler itself so the
            caller can persist its draws.
        """
        import jax
        from numpyro.infer import MCMC, NUTS

        from aind_dynamic_foraging_models.hierarchical_bayes.model import (
            hattori2019_three_level,
        )
        from aind_dynamic_foraging_models.hierarchical_bayes.two_stage import (
            fit_two_stage,
        )

        key = jax.random.PRNGKey(int(self.seed or 0))

        if estimator == "one_stage":
            mcmc = MCMC(
                # Integrator geometry is configurable because the two diagnostic
                # failures have different cures: divergences want longer warmup or a
                # higher target acceptance, while low ESS on clean geometry is the only
                # one more draws actually fixes.
                NUTS(
                    hattori2019_three_level,
                    target_accept_prob=self.target_accept_prob,
                    max_tree_depth=self.max_tree_depth,
                ),
                num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="vectorized",
                # A population fit emits NOTHING between "starting" and "fitted in Ns",
                # and these run for hours -- D≈300 took 26.3 h -- so a silent job is
                # indistinguishable from a hung one for its entire life. That is not a
                # hypothetical: three rungs sat at 32 h with no output and no way to tell
                # whether they were progressing.
                #
                # progress_rate is what makes this safe in a captured log rather than a
                # terminal. NumPyro defaults it to 5% of total iterations, so the bar
                # emits ~20 updates for a whole fit instead of one per iteration; tqdm
                # writes to stderr, which Beaker captures, so an unthrottled bar would
                # bury the log.
                #
                # It is off by default because it is not free: NumPyro's own docs note
                # progress_bar=False "will improve the speed for many cases, but it might
                # require more memory" -- the bar forces the sampler to break its scan
                # into chunks. The trade runs the other way at large D, where memory is
                # the binding constraint, so this is a per-rung decision rather than a
                # global one.
                progress_bar=self.progress_bar,
                **({"progress_rate": self.progress_rate}
                   if self.progress_rate is not None else {}),
            )
            mcmc.run(
                key, choice_arr, reward_arr, valid_mask, session_mask, beta_max=beta_max,
                # NumPyro collects nothing but the sampler's own state unless asked, so
                # without this the saved fit carries `diverging` and nothing else -- which
                # is enough to see that a fit mixed badly and not enough to say why.
                #
                # These four separate the two failure modes that look identical in r_hat
                # and ESS alone:
                #   num_steps   -- trajectory length. These are NumPyro's own state-field
                #                  names, which is what this tuple must use: NumPyro has no
                #                  `tree_depth` attribute (its docstring derives it as
                #                  log2(num_steps) + 1), so asking for that name here
                #                  collects nothing. Truncated trajectories mix slowly with
                #                  ZERO divergences, and the fix is then max_tree_depth --
                #                  nearly free -- rather than more draws.
                #
                #                  In the SAVED ARTIFACT the names differ, because
                #                  az.from_numpyro maps them onto arviz's conventions and
                #                  derives two more: read `n_steps`, `acceptance_rate`,
                #                  `tree_depth` and `reached_max_tree_depth` from
                #                  sample_stats. `reached_max_tree_depth` is the saturation
                #                  check, so it does not have to be reconstructed by hand.
                #   energy      -- the E-BFMI check for heavy tails the momentum cannot
                #                  traverse, which more draws also will not fix.
                #   accept_prob -- whether adaptation actually reached target_accept_prob;
                #                  a large gap means warmup ended before adaptation settled.
                #   diverging   -- already used for the divergence count; named explicitly
                #                  now that the tuple is explicit.
                extra_fields=("num_steps", "energy", "accept_prob", "diverging"),
            )
            samples = mcmc.get_samples()
            population = {
                name: np.asarray(samples[name]).mean(axis=0)
                for name in (
                    "population_mean", "population_scale",
                    "log_sigma_mean", "log_sigma_spread",
                )
            }
            divergences = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
            return population, {"divergences": divergences}, mcmc

        if estimator != "two_stage":
            raise ValueError(f"Unknown estimator {estimator!r}; expected two_stage/one_stage.")

        n_params = 5
        subjects = [
            (choice_arr[i][session_mask[i]], reward_arr[i][session_mask[i]],
             valid_mask[i][session_mask[i]])
            for i in range(choice_arr.shape[0])
        ]
        result = fit_two_stage(
            subjects, rng_key=key,
            subject_kwargs=dict(num_warmup=num_warmup, num_samples=num_samples,
                                beta_max=beta_max),
            population_kwargs=dict(num_warmup=num_warmup, num_samples=num_samples),
        )
        samples = result["population_mcmc"].get_samples()
        mean = np.asarray(samples["population_mean"]).mean(axis=0)
        scale = np.asarray(samples["population_scale"]).mean(axis=0)
        population = {
            "population_mean": mean[:n_params],
            "population_scale": scale[:n_params],
            "log_sigma_mean": mean[n_params:],
            "log_sigma_spread": scale[n_params:],
        }
        return population, {"divergences": None}, result["population_mcmc"]
