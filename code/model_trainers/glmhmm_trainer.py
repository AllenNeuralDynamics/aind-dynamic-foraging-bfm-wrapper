"""GLM-HMM trainer: an input-driven HMM with GLM emissions over left / right / ignore.

Fits the GLM-HMM from ``aind_dynamic_foraging_models.glm_hmm`` (Ashwood et al. 2022, on
dynamax) through the same ``ModelTrainer`` interface as the other models, so it consumes the
identical ``DatasetBundle`` and is scored on the identical trials and split (study 14).

Protocol (Ashwood et al.'s global-then-individual recipe, aligned with the GRU's split):

1. **Global fit** on the training cohort's *train* sessions only -- what the GRU trains on.
   K = 1 is fitted first (convex); K > 1 starts from its weights plus noise, with restarts.
2. **Zero-shot**: held-out subjects' eval sessions scored with the global parameters ->
   ``heldout/zero_shot/*``: the same trials as the matched score, so the pair shows what the
   per-subject refit buys. Deliberately *not* ``heldout/test_likelihood``: the GRU fills that
   key from every held-out session, and only on its non-multisubject path, so one key would
   carry two trial sets (study 13 declined it for the same reason).
3. **Matched**: each held-out subject refitted (MAP, subject prior) on its train sessions,
   starting from the global parameters; its eval sessions scored -> ``heldout/eval_likelihood``
   and the ``heldout/final/*`` three-way decomposition that study 02 reads for the GRU.
4. **K selection**: the same per-subject protocol on the training cohort (train sessions
   fitted, eval sessions scored, pooled) -> ``glmhmm/select/*``. Never the global fit's own
   likelihood, which would reward states that encode differences between subjects.

Every score is the forward filter's one-step predictive, so a trial is predicted from its
session's earlier trials only, and it goes through ``rnn_utils`` and the GRU trainer's own
three-way decomposition, so each key is the same formula the GRU reports.

The model lives in ``aind-dynamic-foraging-models`` (``[glmhmm]`` extra); this module only
marshals data, runs the fits, and reports.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from base.interfaces import ModelTrainer
from base.types import DatasetBundle

logger = logging.getLogger(__name__)

IGNORE = 2
MATURE_STAGES = ("STAGE_FINAL", "GRADUATED")  # as utils.load_mice_database.MATURE_STAGES
STAGE_COLUMN = "current_stage_actual"


def _sessions(raw_df: pd.DataFrame, session_ids: List[Any]) -> List[Dict[str, Any]]:
    """One record per session, in ``session_ids`` order, every trial kept (ignores too).

    Parameters
    ----------
    raw_df : pd.DataFrame
        The bundle's trial frame (``subject_id``, ``ses_idx``, ``trial``,
        ``animal_response``, ``earned_reward``; optionally the stage column).
    session_ids : list
        Sessions to extract, in the loader's order (``metadata["train_session_ids"]`` or
        ``["eval_session_ids"]``).

    Returns
    -------
    list of dict
        ``subject_id``, ``ses_idx``, ``trial`` (array), ``choices`` (0/1/2), ``rewards`` (0/1)
        and ``stage`` (array of strings, or None).
    """
    required = {"subject_id", "ses_idx", "trial", "animal_response", "earned_reward"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"Raw dataframe missing columns required by GLMHMMTrainer: {missing}")
    groups = {sid: g for sid, g in raw_df.groupby("ses_idx", sort=False)}
    records = []
    for session_id in session_ids:
        frame = groups.get(session_id)
        if frame is None or frame.empty:
            continue
        frame = frame.sort_values("trial")
        choices = frame["animal_response"].to_numpy(dtype=int)
        if not np.isin(choices, (0, 1, 2)).all():
            raise ValueError(f"session {session_id}: animal_response outside {{0, 1, 2}}")
        records.append({
            "subject_id": str(frame["subject_id"].iloc[0]),
            "ses_idx": session_id,
            "trial": frame["trial"].to_numpy(),
            "choices": choices,
            "rewards": frame["earned_reward"].to_numpy(dtype=float).astype(int),
            "stage": (frame[STAGE_COLUMN].astype(str).to_numpy()
                      if STAGE_COLUMN in frame.columns else None),
        })
    return records


def _by_subject(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group session records by subject, keeping order."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        out.setdefault(record["subject_id"], []).append(record)
    return out


def _pack(records, log_probs, mask_fn=None):
    """Labels (T, E, 1) and log-probability logits (T, E, 3) in the GRU's layout.

    ``log_probs[i]`` is session ``records[i]``'s one-step predictive, shape (T_i, 3). Padding
    and trials rejected by ``mask_fn(record) -> bool array`` get label -1, which ``rnn_utils``
    masks. Log probabilities are valid logits: their softmax is the probabilities.
    """
    n_max = max(len(r["choices"]) for r in records)
    labels = -np.ones((n_max, len(records), 1), dtype=np.int64)
    logits = np.zeros((n_max, len(records), 3))
    for column, (record, lp) in enumerate(zip(records, log_probs)):
        n = len(record["choices"])
        keep = np.ones(n, dtype=bool) if mask_fn is None else mask_fn(record)
        labels[:n, column, 0] = np.where(keep, record["choices"], -1)
        logits[:n, column] = lp
    return labels, logits


def score_sessions(records, log_probs, mask_fn=None) -> Dict[str, float]:
    """Pooled held-out metrics of predicted sessions, as the GRU computes them.

    Parameters
    ----------
    records : list of dict
        Session records (see :func:`_sessions`).
    log_probs : list of np.ndarray
        Matching one-step predictive log probabilities, (T_i, 3) each.
    mask_fn : callable, optional
        record -> bool array selecting the trials to score (e.g. by stage).

    Returns
    -------
    dict
        ``eval_likelihood`` (raw three-way), ``eval_log_lik_sum`` and ``num_eval_trials``
        (for exact pooling across runs), and the three-way decomposition
        (``eval_likelihood_LR_engaged``, ``eval_likelihood_engage``, ``engage_ignore_*``).
    """
    from disentangled_rnns.library import rnn_utils

    from model_trainers.gru_trainer import _threeway_likelihood_decomposition

    labels, logits = _pack(records, log_probs, mask_fn)
    n_trials = int(np.sum(labels >= 0))
    if n_trials == 0:
        return {"num_eval_trials": 0}
    log_lik = 0.0
    for column, lp in enumerate(log_probs):
        y = labels[: len(lp), column, 0]
        log_lik += float(np.sum(lp[np.flatnonzero(y >= 0), y[y >= 0]]))
    out = {
        "eval_likelihood": float(rnn_utils.normalized_likelihood(labels, logits)),
        "eval_log_lik_sum": log_lik,
        "num_eval_trials": n_trials,
    }
    out.update(_threeway_likelihood_decomposition(labels, logits, n_action_logits=3))
    return out


def _stage_masks():
    """Trial selectors for the mature / non-mature split of an eval score."""
    return {
        "mature": lambda r: np.isin(r["stage"], MATURE_STAGES),
        "nonmature": lambda r: ~np.isin(r["stage"], MATURE_STAGES),
    }


def _params_to_lists(params) -> Dict[str, Any]:
    """A fitted ParamsGLMHMM as nested lists, for JSON."""
    return {
        "initial_probs": np.asarray(params.initial.probs).tolist(),
        "transition_matrix": np.asarray(params.transitions.transition_matrix).tolist(),
        "emissions": {k: np.asarray(v).tolist() for k, v in params.emissions._asdict().items()},
    }


class GLMHMMTrainer(ModelTrainer):
    """Fit a GLM-HMM globally, refit it per subject, and score held-out subjects."""

    def __init__(
        self,
        num_states: int = 3,
        emission: str = "hurdle",
        design: str = "default",
        n_lags: int = 5,
        prior_sigma_global: float = 100.0,
        transition_alpha_global: float = 1.0,
        prior_sigma_subject: float = 2.0,
        transition_alpha_subject: float = 2.0,
        stickiness: float = 0.0,
        num_restarts: int = 10,
        init_noise: float = 0.2,
        em_tol: float = 1e-6,
        max_em_iters: int = 300,
        bucket_size: int = 64,
        num_newton_steps: int = 5,
        select_on_train_subjects: bool = True,
        save_posteriors: bool = True,
        architecture: Mapping[str, Any] | Any = {},
        output_dir: str = "/results/outputs",
        seed: Optional[int] = None,
        **_: Any,
    ) -> None:
        """Store the fit settings (arguments arrive from Hydra's ``instantiate``).

        Parameters
        ----------
        num_states : int
            Number of latent states K.
        emission : {"hurdle", "reference"}
            Emission parameterisation (see the models package).
        design : str
            Named regressor design (``glm_hmm.design.DESIGNS``).
        n_lags : int
            Trials of history per regressor family.
        prior_sigma_global, transition_alpha_global : float
            Gaussian weight-prior SD and Dirichlet concentration for the global fit
            (Ashwood et al.: 100 and 1, i.e. close to maximum likelihood).
        prior_sigma_subject, transition_alpha_subject : float
            The same for every per-subject refit (Ashwood et al.: 2 and 2).
        stickiness : float
            Extra Dirichlet concentration on the diagonal of A, both fits.
        num_restarts : int
            Global-fit starts for K > 1.
        init_noise : float
            SD of the noise added to the K = 1 weights at each start.
        em_tol : float
            EM stops when the log joint improves by less than this many nats per trial.
        max_em_iters : int
            EM iteration cap, every fit.
        bucket_size : int
            Session-length granularity of the padded batches.
        num_newton_steps : int
            Newton iterations per emission M-step.
        select_on_train_subjects : bool
            Also run the per-subject protocol on the training cohort (the K-selection score).
        save_posteriors : bool
            Write per-trial predictive probabilities and state posteriors of every
            held-out session.
        architecture : mapping
            Unused; accepted for config compatibility (``multisubject``).
        output_dir : str
            Where fit artifacts are written.
        seed : int, optional
            Seeds the restarts.
        """
        super().__init__(seed=seed)
        if emission not in ("hurdle", "reference"):
            raise ValueError(f"emission must be 'hurdle' or 'reference'; got {emission!r}")
        if num_states < 1 or n_lags < 1:
            raise ValueError("num_states and n_lags must be positive")
        self.num_states = int(num_states)
        self.emission = emission
        self.design = design
        self.n_lags = int(n_lags)
        self.prior_sigma_global = float(prior_sigma_global)
        self.transition_alpha_global = float(transition_alpha_global)
        self.prior_sigma_subject = float(prior_sigma_subject)
        self.transition_alpha_subject = float(transition_alpha_subject)
        self.stickiness = float(stickiness)
        self.num_restarts = int(num_restarts)
        self.init_noise = float(init_noise)
        self.em_tol = float(em_tol)
        self.max_em_iters = int(max_em_iters)
        self.bucket_size = int(bucket_size)
        self.num_newton_steps = int(num_newton_steps)
        self.select_on_train_subjects = bool(select_on_train_subjects)
        self.save_posteriors = bool(save_posteriors)
        self.output_dir = Path(output_dir)

    # -- building blocks ------------------------------------------------------------------

    def _buckets(self, records):
        """Length-bucketed model inputs for these sessions (indices follow ``records``)."""
        from aind_dynamic_foraging_models.glm_hmm import make_buckets

        return make_buckets(
            [(r["choices"], r["rewards"]) for r in records], self.n_lags, self.design,
            self.bucket_size,
        )

    def _fit_global(self, records, key):
        """K = 1 first, then K > 1 from its weights plus noise, keeping the best start."""
        from aind_dynamic_foraging_models.glm_hmm import (
            GLMHMM,
            Prior,
            fit_em,
            fit_with_restarts,
        )

        prior = Prior(self.prior_sigma_global, self.transition_alpha_global, self.stickiness)
        buckets = self._buckets(records)
        single = GLMHMM(1, self.n_lags, self.emission, self.num_newton_steps)
        params, props = single.initialize()
        base = fit_em(single, params, props, buckets, prior, self.max_em_iters, self.em_tol)
        if self.num_states == 1:
            return single, base, [base]
        model = GLMHMM(self.num_states, self.n_lags, self.emission, self.num_newton_steps)
        start = {name: np.asarray(value)[0]
                 for name, value in base.params.emissions._asdict().items()}
        best, results = fit_with_restarts(
            model, buckets, key, self.num_restarts, prior, self.init_noise,
            self.max_em_iters, self.em_tol, **start,
        )
        return model, best, results

    def _refit_subjects(self, model, global_params, by_subject_train, by_subject_eval):
        """Per-subject MAP refits from the global fit; one-step predictive on eval sessions.

        Returns per-subject results and, for every eval session, its predictive log probs.
        """
        from aind_dynamic_foraging_models.glm_hmm import (
            Prior,
            fit_em,
            predictive_log_probs,
        )

        prior = Prior(self.prior_sigma_subject, self.transition_alpha_subject, self.stickiness)
        _, props = model.initialize()
        results = {}
        for subject, train_records in by_subject_train.items():
            eval_records = by_subject_eval.get(subject, [])
            if not eval_records:
                # Never drop a subject silently: the comparators score every subject.
                logger.warning("GLMHMMTrainer: subject %s has no eval sessions; not scored.",
                               subject)
                continue
            fit = fit_em(model, global_params, props, self._buckets(train_records), prior,
                         self.max_em_iters, self.em_tol)
            eval_lp = predictive_log_probs(model, fit.params, self._buckets(eval_records))
            train_lp = predictive_log_probs(model, fit.params, self._buckets(train_records))
            results[subject] = {
                "fit": fit,
                "eval_records": eval_records,
                "eval_log_probs": [eval_lp[i] for i in range(len(eval_records))],
                "train_records": train_records,
                "train_log_probs": [train_lp[i] for i in range(len(train_records))],
            }
        return results

    def _pooled(self, refits, which="eval"):
        """Pooled score over every refitted subject's ``which`` sessions."""
        records = [r for res in refits.values() for r in res[f"{which}_records"]]
        log_probs = [lp for res in refits.values() for lp in res[f"{which}_log_probs"]]
        return records, log_probs

    # -- the fit --------------------------------------------------------------------------

    def fit(self, bundle: DatasetBundle, loggers: Optional[Dict[str, Any]] = None) -> Dict:
        """Global fit, zero-shot and matched held-out scores, and the K-selection score.

        Parameters
        ----------
        bundle : DatasetBundle
            The training cohort; ``extras["heldout_raw"]`` and ``extras["heldout_metadata"]``
            carry the held-out cohort from the same loader.
        loggers : dict, optional
            May contain a ``"wandb"`` run.

        Returns
        -------
        dict
            Summary of fits and scores (also written to ``output_dir``).
        """
        import jax

        started = time.time()
        wandb_run = (loggers or {}).get("wandb")
        metadata = dict(bundle.metadata or {})
        if metadata.get("ignore_policy", "include") != "include":
            raise ValueError(
                "GLMHMMTrainer models left/right/ignore and needs data.ignore_policy=include; "
                f"got {metadata.get('ignore_policy')!r}."
            )
        if bundle.raw is None or len(bundle.raw) == 0:
            raise ValueError("GLMHMMTrainer requires bundle.raw with trial-level rows.")

        train_records = _sessions(bundle.raw, list(metadata["train_session_ids"]))
        logger.info(
            "GLMHMMTrainer: global fit on %d train sessions (%d trials) of %d subjects; "
            "K=%d emission=%s L=%d",
            len(train_records), sum(len(r["choices"]) for r in train_records),
            len({r["subject_id"] for r in train_records}), self.num_states, self.emission,
            self.n_lags,
        )
        key = jax.random.PRNGKey(int(self.seed or 0))
        model, global_fit, restarts = self._fit_global(train_records, key)
        global_seconds = time.time() - started
        logger.info("GLMHMMTrainer: global fit in %.0fs, log joint %.2f (%s)",
                    global_seconds, global_fit.log_joint[-1],
                    "converged" if global_fit.converged else "NOT converged")

        summary: Dict[str, Any] = {
            "glmhmm/num_states": self.num_states,
            "glmhmm/emission": self.emission,
            "glmhmm/n_lags": self.n_lags,
            "glmhmm/global/log_joint": float(global_fit.log_joint[-1]),
            # The trace also holds the starting point's log joint: updates are one fewer.
            "glmhmm/global/em_iters": int(len(global_fit.log_joint) - 1),
            "glmhmm/global/converged": bool(global_fit.converged),
            "glmhmm/global/restart_log_joints": [float(r.log_joint[-1]) for r in restarts],
            "glmhmm/global/num_train_trials": int(sum(len(r["choices"]) for r in train_records)),
            "glmhmm/global/fit_seconds": float(global_seconds),
        }

        # K-selection score: the per-subject protocol on the training cohort.
        if self.select_on_train_subjects:
            select_eval = _sessions(bundle.raw, list(metadata["eval_session_ids"]))
            select = self._refit_subjects(
                model, global_fit.params, _by_subject(train_records), _by_subject(select_eval)
            )
            scores = score_sessions(*self._pooled(select))
            summary.update({f"glmhmm/select/{k}": v for k, v in scores.items()})
            logger.info("GLMHMMTrainer: K-selection score (training cohort, per-subject "
                        "refits) %.5f", scores.get("eval_likelihood", float("nan")))

        heldout_df = (bundle.extras or {}).get("heldout_raw")
        heldout_meta = (bundle.extras or {}).get("heldout_metadata") or {}
        refits: Dict[str, Any] = {}
        if heldout_df is None or len(heldout_df) == 0:
            logger.warning("GLMHMMTrainer: no bundle.extras['heldout_raw']; held-out scoring "
                           "skipped (scoring the training cohort would leak).")
        else:
            refits = self._score_heldout(model, global_fit.params, heldout_df, heldout_meta,
                                         summary)

        summary["glmhmm/fit_seconds"] = float(time.time() - started)
        self._write_artifacts(model, global_fit, restarts, refits, summary)
        if wandb_run is not None:
            for name, value in summary.items():
                wandb_run.summary[name] = value
            self._log_per_subject_table(wandb_run, refits)
        return summary

    def _score_heldout(self, model, global_params, heldout_df, heldout_meta, summary):
        """Zero-shot and matched scores of the held-out cohort, written into ``summary``."""
        from aind_dynamic_foraging_models.glm_hmm import predictive_log_probs

        if "train_session_ids" not in heldout_meta or "eval_session_ids" not in heldout_meta:
            raise ValueError("bundle.extras['heldout_metadata'] must carry the held-out "
                             "loader's train/eval session ids.")
        train = _sessions(heldout_df, list(heldout_meta["train_session_ids"]))
        evals = _sessions(heldout_df, list(heldout_meta["eval_session_ids"]))

        zero_shot_lp = predictive_log_probs(model, global_params, self._buckets(evals))
        zero_shot = score_sessions(evals, [zero_shot_lp[i] for i in range(len(evals))])
        summary.update({f"heldout/zero_shot/{k}": v for k, v in zero_shot.items()})

        refits = self._refit_subjects(model, global_params, _by_subject(train),
                                      _by_subject(evals))
        records, log_probs = self._pooled(refits)
        matched = score_sessions(records, log_probs)
        summary["heldout/eval_likelihood"] = matched.get("eval_likelihood")
        summary["heldout/eval_log_lik_sum"] = matched.get("eval_log_lik_sum")
        summary["heldout/num_eval_trials"] = matched.get("num_eval_trials")
        summary["heldout/num_eval_subjects"] = len(refits)
        summary["heldout/n_subjects_not_converged"] = int(
            sum(not res["fit"].converged for res in refits.values())
        )
        summary.update({f"heldout/final/{k}": v for k, v in matched.items()})
        train_score = score_sessions(*self._pooled(refits, "train"))
        summary["heldout/train_likelihood"] = train_score.get("eval_likelihood")
        if records and records[0]["stage"] is None:
            logger.warning(
                "GLMHMMTrainer: no %s column, so no mature/non-mature split of the held-out "
                "score. Add it to data.cols_to_retain (the default list projects it out).",
                STAGE_COLUMN,
            )
        elif records:
            for name, mask_fn in _stage_masks().items():
                by_stage = score_sessions(records, log_probs, mask_fn)
                for key in ("eval_likelihood", "eval_likelihood_LR_engaged", "num_eval_trials"):
                    if key in by_stage:
                        summary[f"heldout/final/{key}_{name}"] = by_stage[key]
        logger.info(
            "GLMHMMTrainer: held-out zero-shot %.5f, matched %.5f (LR engaged %.5f) over %d "
            "subjects", zero_shot.get("eval_likelihood", float("nan")),
            matched.get("eval_likelihood", float("nan")),
            matched.get("eval_likelihood_LR_engaged", float("nan")), len(refits),
        )
        return refits

    # -- reporting ------------------------------------------------------------------------

    def _per_subject_rows(self, refits):
        """One row per held-out subject: split sizes, convergence and its matched scores."""
        rows = []
        for subject, res in refits.items():
            scores = score_sessions(res["eval_records"], res["eval_log_probs"])
            rows.append({
                "subject_id": subject,
                "n_train_sessions": len(res["train_records"]),
                "n_eval_sessions": len(res["eval_records"]),
                "n_eval_trials": scores.get("num_eval_trials", 0),
                "eval_log_lik_sum": scores.get("eval_log_lik_sum"),
                "eval_likelihood": scores.get("eval_likelihood"),
                "eval_likelihood_LR_engaged": scores.get("eval_likelihood_LR_engaged"),
                "eval_likelihood_engage": scores.get("eval_likelihood_engage"),
                "em_iters": int(len(res["fit"].log_joint) - 1),
                "converged": bool(res["fit"].converged),
            })
        return rows

    def _log_per_subject_table(self, wandb_run, refits):
        """Per-subject matched scores as a W&B table (paired per-mouse comparisons)."""
        rows = self._per_subject_rows(refits)
        if not rows:
            return
        try:
            import wandb

            table = wandb.Table(columns=list(rows[0]))
            for row in rows:
                table.add_data(*row.values())
            wandb_run.log({"heldout/per_subject": table})
        except Exception:  # pragma: no cover - W&B availability
            logger.exception("Could not log the per-subject table; continuing.")

    def _write_artifacts(self, model, global_fit, restarts, refits, summary):
        """Parameters, per-subject scores and (optionally) per-trial outputs on disk."""
        from model_trainers.hb_trainer import _source_revisions

        out = self.output_dir / "glmhmm"
        out.mkdir(parents=True, exist_ok=True)
        record = {
            "config": {
                k: getattr(self, k) for k in (
                    "num_states", "emission", "design", "n_lags", "prior_sigma_global",
                    "transition_alpha_global", "prior_sigma_subject",
                    "transition_alpha_subject", "stickiness", "num_restarts", "init_noise",
                    "em_tol", "max_em_iters", "bucket_size", "num_newton_steps", "seed",
                )
            },
            "summary": summary,
            "global": _params_to_lists(global_fit.params),
            "global_log_joint_trace": np.asarray(global_fit.log_joint).tolist(),
            "subjects": {s: _params_to_lists(res["fit"].params) for s, res in refits.items()},
            **_source_revisions(),
        }
        (out / "fit.json").write_text(json.dumps(record, indent=1, default=str))
        rows = self._per_subject_rows(refits)
        if rows:
            pd.DataFrame(rows).to_csv(out / "heldout_per_subject.csv", index=False)
        if self.save_posteriors and refits:
            self._write_posteriors(model, refits, out)
        logger.info("GLMHMMTrainer: wrote artifacts to %s", out)

    def _write_posteriors(self, model, refits, out):
        """Per-trial predictive and state posteriors of held-out sessions, one parquet.

        Rows are keyed by (subject_id, ses_idx, trial) with the split and stage, so the
        numbers can be joined with any other model's per-trial outputs. Smoothed posteriors
        use the whole session: they describe it, they never score it.
        """
        from aind_dynamic_foraging_models.glm_hmm import state_posteriors

        frames = []
        for subject, res in refits.items():
            for split in ("train", "eval"):
                records = res[f"{split}_records"]
                posts = state_posteriors(model, res["fit"].params, self._buckets(records))
                for i, (record, lp) in enumerate(zip(records, res[f"{split}_log_probs"])):
                    frame = {
                        "subject_id": subject,
                        "ses_idx": record["ses_idx"],
                        "trial": record["trial"],
                        "split": split,
                        "choice": record["choices"],
                        "logp_left": lp[:, 0].astype(np.float32),
                        "logp_right": lp[:, 1].astype(np.float32),
                        "logp_ignore": lp[:, 2].astype(np.float32),
                        "viterbi": posts[i]["viterbi"].astype(np.int8),
                    }
                    if record["stage"] is not None:
                        frame["stage"] = record["stage"]
                    for k in range(model.num_states):
                        frame[f"filtered_{k}"] = posts[i]["filtered"][:, k].astype(np.float32)
                        frame[f"smoothed_{k}"] = posts[i]["smoothed"][:, k].astype(np.float32)
                    frames.append(pd.DataFrame(frame))
        pd.concat(frames, ignore_index=True).to_parquet(out / "heldout_trials.parquet")
