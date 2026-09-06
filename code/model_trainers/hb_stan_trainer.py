"""Stan arm of the hierarchical-Bayes baseline: same model, same cohort, different sampler.

Why this is a trainer rather than a benchmark script
----------------------------------------------------
`RESULTS.md` prices NumPyro against Stan **per subject**, where Stan wins on every measured
axis, and then argues NumPyro wins at cohort scale from a row labelled "inferred, not
measured". Replacing that inference with a measurement means fitting the joint three-level
model in Stan on a real rung -- and a real rung has to come through the ordinary pipeline, not
a side path, or the number is not comparable with the NumPyro rungs it is meant to sit beside.

Going through `run_hpc` buys exactly the things a side script would have had to fake:
``mice_snapshot_scaling`` picks the cohort, so ``data.subject_ratio`` and ``seed`` select the
same subjects as the NumPyro rung; the cohort arrays are built by ``hb_trainer``'s own
``_extract_subject_sessions`` / ``_pad_cohort``, so no re-derivation can drift; W&B carries the
git lineage and ``meta.*`` provenance; and the launch record is written by the launcher rather
than by hand.

Scope: the population fit only
------------------------------
This trainer deliberately does **not** score held-out subjects. The question it exists to
answer is about sampler efficiency -- wall time, ESS per draw, ESS per second, divergences --
on the joint fit, and held-out adaptation is a separate NumPyro-side machine
(``heldout.py``) that would have to be ported too. Porting it would double the work and
measure something the comparison does not turn on. So `heldout/*` is absent here by design,
and an HB-Stan run is not a substitute for an HB rung on the scaling curve.

Threading
---------
httpstan compiles models with ``STAN_THREADS`` defined (``httpstan/models.py``), so the
model's ``reduce_sum`` **can** thread across subjects within a chain. What it cannot do is
guess how many threads to use: Stan reads ``STAN_NUM_THREADS`` at runtime and falls back to a
single thread when it is unset, which is what the first D≈29/D≈99 pair ran as. This trainer
now sets it from the SLURM allocation, so the job uses the cores it was given -- with 4 chains
and 32 CPUs each chain threads 8 ways over the subject slice.

That makes the CPU request meaningful rather than decorative, and it matters most at the
large end: D≈99 measured ~53 iterations/hour single-threaded, which projects past any
reasonable wall clock.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from base.interfaces import ModelTrainer
from base.types import DatasetBundle
from model_trainers.hb_trainer import (
    _extract_subject_sessions,
    _pad_cohort,
    _source_revisions,
)

logger = logging.getLogger(__name__)

PARAM_NAMES = (
    "learn_rate_rew",
    "learn_rate_unrew",
    "forget_rate_unchosen",
    "softmax_inverse_temperature",
    "bias_l",
)


def cohort_to_stan_data(choice_arr, reward_arr, valid_mask, session_mask,
                        beta_max=10.0, log_sigma_loc=-1.0, log_sigma_scale=1.0):
    """Turn the trainer's padded cohort into the `.stan` file's `data` block.

    The padded arrays are handed over as-is, but with per-subject session counts and
    per-session trial counts alongside, which is what lets Stan loop to each session's true
    length instead of evaluating the likelihood on padding. That asymmetry is the mechanism
    being priced here -- JAX has to pad every lane to the cohort maximum, and on this cohort
    that is roughly 60-75% waste.

    Exporting counts rather than the mask is lossless only because ``_pad_cohort`` places each
    session's valid trials contiguously from index 0, so that is asserted rather than assumed.
    """
    n_sessions = session_mask.sum(axis=1).astype(int)
    n_trials = valid_mask.sum(axis=2).astype(int)
    for s in range(valid_mask.shape[0]):
        for m in range(int(n_sessions[s])):
            k = int(n_trials[s, m])
            if not (valid_mask[s, m, :k].all() and not valid_mask[s, m, k:].any()):
                raise ValueError(
                    f"subject {s} session {m}: valid trials are not a contiguous prefix, so "
                    "a per-session trial COUNT would silently select the wrong trials"
                )
    return {
        "S": int(choice_arr.shape[0]),
        "M": int(choice_arr.shape[1]),
        "T": int(choice_arr.shape[2]),
        "n_sessions": n_sessions.tolist(),
        "n_trials": n_trials.tolist(),
        "choice": choice_arr.astype(int).tolist(),
        "reward": (np.asarray(reward_arr) > 0).astype(int).tolist(),
        "beta_max": float(beta_max),
        "log_sigma_loc": float(log_sigma_loc),
        "log_sigma_scale": float(log_sigma_scale),
        "grainsize": 1,
    }


class HBStanTrainer(ModelTrainer):
    """Fit the joint three-level model with pystan on the same cohort as ``HBTrainer``."""

    def __init__(
        self,
        num_warmup: int = 2000,
        num_samples: int = 2000,
        num_chains: int = 4,
        beta_max: float = 10.0,
        artifact_dir: Optional[str] = None,
        architecture: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        seed: Optional[int] = None,
        **_: Any,
    ) -> None:
        """
        Parameters
        ----------
        num_warmup, num_samples, num_chains : int
            Sampler settings. Defaults match the ladder's settled NumPyro values
            (2000/2000/4) so the two arms are comparable on ESS per draw; changing them here
            without changing the NumPyro rung makes the comparison meaningless.
        beta_max : float
            Upper bound on the softmax inverse temperature, as in the published model.
        artifact_dir : str, optional
            Where the posterior draws are written. A cohort fit costs hours, so it is
            persisted before anything else is computed from it.
        seed : int, optional
            Wired from the top-level ``seed`` by the model config, exactly as ``hb_hattori``
            does. Without it the run cannot be reproduced from its own config.
        """
        super().__init__(seed=seed)
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.beta_max = float(beta_max)
        self.artifact_dir = artifact_dir
        self.architecture = architecture
        self.output_dir = output_dir

    def fit(self, bundle: DatasetBundle, loggers: Optional[Dict[str, Any]] = None) -> Any:
        """Fit the population posterior and log sampler-efficiency metrics."""
        # Must be set BEFORE stan/httpstan starts a model process: Stan reads
        # STAN_NUM_THREADS once, when it initialises its thread pool, and defaults to a single
        # thread if it is unset. Chains run as separate processes, so the cores have to be
        # divided between them rather than handed to each.
        os.environ.setdefault("STAN_NUM_THREADS", str(self._threads_per_chain()))
        # Read back rather than reusing the computed value: setdefault leaves an
        # externally-supplied STAN_NUM_THREADS in place, so the two can differ and the log
        # must report what Stan will actually use.
        threads = os.environ["STAN_NUM_THREADS"]

        import stan

        wandb_run = (loggers or {}).get("wandb")
        logger.info(
            "HBStanTrainer: %d chains x %s reduce_sum threads",
            self.num_chains, threads,
        )

        if bundle.raw is None or len(bundle.raw) == 0:
            raise ValueError("HBStanTrainer requires bundle.raw with trial-level rows.")

        choices, rewards, _ = _extract_subject_sessions(bundle.raw)
        choice_arr, reward_arr, valid_mask, session_mask, subject_ids = _pad_cohort(
            choices, rewards
        )
        real_trials = int(valid_mask.sum())
        padded_trials = int(np.prod(choice_arr.shape))
        logger.info(
            "HBStanTrainer: %d subjects, up to %d sessions, up to %d trials; "
            "%d real trials vs %d padded (%.1f%% padding, which Stan skips)",
            len(subject_ids), choice_arr.shape[1], choice_arr.shape[2],
            real_trials, padded_trials, 100 * (1 - real_trials / padded_trials),
        )

        data = cohort_to_stan_data(
            choice_arr, reward_arr, valid_mask, session_mask, beta_max=self.beta_max
        )

        program = _stan_program()

        # Build and sampling are timed apart. Stan compiles once to a binary; NumPyro pays JIT
        # inside its first sampling call. Reporting one number would flatter whichever
        # framework happened to be warm.
        started = time.time()
        posterior = stan.build(program, data=data, random_seed=int(self.seed or 0))
        build_seconds = time.time() - started
        logger.info("HBStanTrainer: model built in %.0fs", build_seconds)

        started = time.time()
        fit = posterior.sample(
            num_chains=self.num_chains,
            num_samples=self.num_samples,
            num_warmup=self.num_warmup,
        )
        fit_seconds = time.time() - started
        logger.info("HBStanTrainer: population fitted in %.0fs", fit_seconds)

        metrics = self._diagnostics(fit, fit_seconds)
        metrics.update({
            "hb/build_seconds": build_seconds,
            "hb/fit_seconds": fit_seconds,
            "hb/n_subjects": len(subject_ids),
            "hb/real_trials": real_trials,
            "hb/padded_trials": padded_trials,
            "hb/framework": "stan",
        })

        saved = self._save(fit, subject_ids, metrics)
        if saved:
            metrics["hb/artifact"] = saved

        if wandb_run is not None:
            wandb_run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            wandb_run.summary.update(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )

        logger.info("HBStanTrainer: %s", {k: v for k, v in metrics.items()
                                          if isinstance(v, (int, float))})
        return {"metrics": metrics, "subject_ids": [str(s) for s in subject_ids]}

    def _threads_per_chain(self) -> int:
        """How many reduce_sum threads each chain may use.

        Read from the SLURM allocation rather than the machine, because ``os.cpu_count()`` on
        a shared node reports every core on the box, not the ones this job was given --
        oversubscribing them would slow the job down and everyone else's with it. An explicit
        ``STAN_NUM_THREADS`` in the environment always wins, so a launch can override.
        """
        for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            value = os.environ.get(var)
            if value and value.isdigit():
                return max(1, int(value) // max(1, self.num_chains))
        return 1

    def _diagnostics(self, fit, fit_seconds):
        """R-hat / bulk ESS on the population means, plus divergences.

        Reshaped to ``(chain, draw)`` before ArviZ sees them: pystan returns
        ``(*dims, chains * samples)`` ordered chain-major, and computing r_hat on the flat
        vector would treat four chains as one long one and report a meaningless value.
        """
        import arviz as az

        n_draws = self.num_chains * self.num_samples
        population_mean = np.asarray(fit["population_mean"])
        rhats, esss = [], []
        out = {}
        for i, name in enumerate(PARAM_NAMES):
            draws = population_mean[i].reshape(self.num_chains, self.num_samples)
            ess = float(az.ess(draws, method="bulk"))
            rhat = float(az.rhat(draws))
            rhats.append(rhat)
            esss.append(ess)
            out[f"hb/ess_bulk/{name}"] = ess
            out[f"hb/rhat/{name}"] = rhat
        try:
            divergences = int(np.sum(np.asarray(fit["divergent__"])))
        except (KeyError, AttributeError, TypeError):
            divergences = -1        # -1 = not reported, distinct from a genuine zero
        out.update({
            "hb/divergences": divergences,
            "hb/max_rhat": max(rhats),
            "hb/min_ess_bulk": min(esss),
            "hb/min_ess_per_draw": min(esss) / n_draws,
            "hb/min_ess_per_second": min(esss) / fit_seconds,
            "hb/n_draws": n_draws,
        })
        return out

    def _save(self, fit, subject_ids, metrics):
        """Persist the population draws. A cohort fit costs hours; keeping summaries only
        would force a refit for any later question."""
        if not self.artifact_dir:
            return None
        import json
        import os

        os.makedirs(self.artifact_dir, exist_ok=True)
        path = os.path.join(self.artifact_dir, "stan_one_stage_fit.npz")
        payload = {}
        for name in ("population_mean", "population_scale", "log_sigma_mean",
                     "log_sigma_spread"):
            try:
                payload[name] = np.asarray(fit[name])
            except (KeyError, TypeError):
                continue
        np.savez_compressed(path, **payload)
        meta = {
            "framework": "stan",
            "n_subjects": len(subject_ids),
            "subject_ids": [str(s) for s in subject_ids],
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "beta_max": self.beta_max,
            "seed": self.seed,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            **_source_revisions(),
        }
        with open(path.replace(".npz", ".json"), "w") as handle:
            json.dump(meta, handle, indent=2, default=str)
        logger.info("HBStanTrainer: wrote %s", path)
        return path


def _stan_program():
    """Read the three-level Stan program from the models package.

    Kept in `aind-dynamic-foraging-models` next to `hattori2019_three_level`, which it is a
    port of, so the two definitions of the same model live together and a change to one is
    visible beside the other.
    """
    import os

    import aind_dynamic_foraging_models.hierarchical_bayes as hb

    path = os.path.join(
        os.path.dirname(hb.__file__), "benchmarks", "reference_stan", "hb_three_level.stan"
    )
    with open(path) as handle:
        return handle.read()
