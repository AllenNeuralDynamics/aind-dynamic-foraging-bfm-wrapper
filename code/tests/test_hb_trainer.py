"""HBTrainer data marshalling and end-to-end fit."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from base.types import DatasetBundle
from model_trainers.hb_trainer import (
    _extract_subject_sessions,
    _normalized_likelihood,
    _pad_cohort,
)

try:
    import jax  # noqa: F401
    import numpyro  # noqa: F401

    from model_trainers.hb_trainer import HBTrainer

    HAS_BAYES = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_BAYES = False


def _make_frame(n_subjects=3, n_sessions=4, n_trials=60, ignore_every=None, seed=0):
    """Build a trial-level frame shaped like the wrapper's raw dataframe."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(n_subjects):
        for session in range(n_sessions):
            for trial in range(n_trials):
                response = int(rng.integers(0, 2))
                if ignore_every and trial % ignore_every == 0:
                    response = 2  # ignored trial
                rows.append({
                    "subject_id": f"m{subject}",
                    "ses_idx": f"m{subject}_s{session}",
                    "trial": trial,
                    "animal_response": response,
                    "earned_reward": float(rng.integers(0, 2)),
                })
    return pd.DataFrame(rows)


class TestExtraction(unittest.TestCase):
    """Grouping trials into per-subject sessions."""

    def test_groups_by_subject_and_session(self):
        """Every subject gets one array per session."""
        choices, rewards, _ = _extract_subject_sessions(_make_frame())
        self.assertEqual(sorted(choices.keys()), ["m0", "m1", "m2"])
        self.assertEqual(len(choices["m0"]), 4)
        self.assertEqual(len(choices["m0"][0]), 60)
        self.assertEqual(len(rewards["m0"][0]), 60)

    def test_drops_ignored_trials(self):
        """Trials with animal_response outside {0,1} are excluded, as baseline_rl does."""
        choices, _, _ = _extract_subject_sessions(_make_frame(ignore_every=3))
        session = choices["m0"][0]
        self.assertEqual(len(session), 40)  # 60 trials less every third
        self.assertTrue(np.all((session == 0) | (session == 1)))

    def test_returns_session_ids_in_frame_order(self):
        """Session ids come back ordered, so a split matches the neural models' own."""
        _, _, ids = _extract_subject_sessions(_make_frame(n_subjects=2, n_sessions=3))
        self.assertEqual(ids["m0"], ["m0_s0", "m0_s1", "m0_s2"])

    def test_missing_columns_raise(self):
        """A frame without the required columns fails loudly."""
        with self.assertRaises(ValueError):
            _extract_subject_sessions(pd.DataFrame({"subject_id": ["m0"]}))


class TestPadding(unittest.TestCase):
    """Padding ragged cohorts into dense arrays."""

    def test_shapes_and_masks(self):
        """Ragged subjects pad to a common shape with masks marking the real entries."""
        choices = {"a": [np.zeros(10, int), np.zeros(6, int)], "b": [np.zeros(8, int)]}
        rewards = {"a": [np.zeros(10), np.zeros(6)], "b": [np.zeros(8)]}
        c, r, valid, session_mask, ids = _pad_cohort(choices, rewards)

        self.assertEqual(c.shape, (2, 2, 10))
        self.assertEqual(session_mask.tolist(), [[True, True], [True, False]])
        self.assertEqual(int(valid[0, 0].sum()), 10)
        self.assertEqual(int(valid[0, 1].sum()), 6)
        self.assertEqual(int(valid[1, 0].sum()), 8)
        self.assertEqual(int(valid[1, 1].sum()), 0)  # padded session contributes nothing
        self.assertEqual(ids, ["a", "b"])

    def test_never_truncates(self):
        """The longest session survives intact; padding only ever adds."""
        choices = {"a": [np.ones(25, int), np.ones(5, int)]}
        rewards = {"a": [np.ones(25), np.ones(5)]}
        c, _, valid, _, _ = _pad_cohort(choices, rewards)
        self.assertEqual(c.shape[2], 25)
        self.assertEqual(int(valid[0, 0].sum()), 25)


class TestNormalizedLikelihood(unittest.TestCase):
    """The metric shared with the neural models."""

    def test_geometric_mean(self):
        """exp(sum log p / n) matches a hand-computed geometric mean."""
        probs = np.array([0.8, 0.5, 0.4])
        expected = float(np.exp(np.sum(np.log(probs)) / 3))
        self.assertAlmostEqual(
            _normalized_likelihood(float(np.sum(np.log(probs))), 3), expected, places=10
        )

    def test_empty_is_zero(self):
        """No scored trials yields zero rather than a division error."""
        self.assertEqual(_normalized_likelihood(0.0, 0), 0.0)


@unittest.skipUnless(HAS_BAYES, "requires the 'bayes' extra (jax, numpyro)")
class TestFit(unittest.TestCase):
    """End-to-end fit through the ModelTrainer interface."""

    def test_skips_heldout_without_explicit_cohort(self):
        """Absent a held-out frame the trainer skips scoring rather than leaking."""
        bundle = DatasetBundle(
            raw=_make_frame(n_subjects=3, n_sessions=3, n_trials=50),
            train_set=None, eval_set=None, metadata={},
        )
        trainer = HBTrainer(
            estimator="two_stage", num_warmup=30, num_samples=30, num_chains=1, seed=0,
        )
        output = trainer.fit(bundle)
        self.assertTrue(output["heldout_skipped"])
        self.assertEqual(output["heldout_likelihood"], {})
        self.assertEqual(len(output["population"]["population_mean"]), 5)

    def test_one_stage_scores_heldout(self):
        """With a held-out cohort supplied, every k yields a likelihood in (0, 1)."""
        bundle = DatasetBundle(
            raw=_make_frame(n_subjects=3, n_sessions=3, n_trials=50, seed=0),
            train_set=None, eval_set=None, metadata={},
            extras={"heldout_raw": _make_frame(n_subjects=2, n_sessions=4,
                                               n_trials=50, seed=7)},
        )
        trainer = HBTrainer(
            estimator="one_stage", num_warmup=30, num_samples=30, num_chains=1,
            few_shot_k=(0, 2), seed=0,
        )
        output = trainer.fit(bundle)
        scores = output["heldout_likelihood"]
        self.assertEqual(sorted(k for k in scores if isinstance(k, int)), [0, 2])
        for value in scores.values():
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)

    def test_matched_conditioning_uses_the_baseline_split(self):
        """A 'matched' rung is reported alongside the k sweep.

        It conditions on exactly the sessions the per-mouse MLE baseline fits and scores
        exactly the ones it scores, so the two sit on the same footing rather than being
        compared across different amounts of conditioning.
        """
        bundle = DatasetBundle(
            raw=_make_frame(n_subjects=3, n_sessions=3, n_trials=50, seed=0),
            train_set=None, eval_set=None, metadata={},
            extras={"heldout_raw": _make_frame(n_subjects=2, n_sessions=4,
                                               n_trials=50, seed=7)},
        )
        trainer = HBTrainer(
            estimator="one_stage", num_warmup=30, num_samples=30, num_chains=1,
            few_shot_k=(0,), eval_every_n=2, seed=0,
        )
        scores = trainer.fit(bundle)["heldout_likelihood"]
        self.assertIn("matched", scores)
        self.assertGreater(scores["matched"], 0.0)
        self.assertLess(scores["matched"], 1.0)


@unittest.skipUnless(HAS_BAYES, "requires the 'bayes' extra (jax, numpyro)")
class TestHeldoutTwoLevel(unittest.TestCase):
    """Held-out subjects fitted alone: no population, no training cohort."""

    class _Run:
        """Records what a W&B run would receive."""

        def __init__(self):
            """Start with an empty summary."""
            self.summary, self.logged = {}, []

        def log(self, payload):
            """Keep logged payloads."""
            self.logged.append(payload)

    @classmethod
    def setUpClass(cls):
        """One tiny two-chain fit, with no training cohort at all."""
        cls.artifacts = tempfile.TemporaryDirectory()
        bundle = DatasetBundle(
            raw=None, train_set=None, eval_set=None, metadata={},
            extras={"heldout_raw": _make_frame(n_subjects=2, n_sessions=4,
                                               n_trials=50, seed=7)},
        )
        cls.wandb_run = cls._Run()
        trainer = HBTrainer(
            estimator="heldout_two_level", num_warmup=30, num_samples=30, num_chains=2,
            eval_every_n=2, session_draws_per_posterior_draw=4,
            artifact_dir=cls.artifacts.name, seed=0,
        )
        cls.output = trainer.fit(bundle, loggers={"wandb": cls.wandb_run})

    @classmethod
    def tearDownClass(cls):
        """Remove the artifact directory."""
        cls.artifacts.cleanup()

    def test_scores_every_subject_on_the_baseline_split(self):
        """Two context and two scored sessions each, likelihoods in (0, 1)."""
        self.assertEqual(self.output["estimator"], "heldout_two_level")
        per_subject = self.output["heldout_per_subject_matched"]
        self.assertEqual(sorted(per_subject), ["m0", "m1"])
        for row in per_subject.values():
            self.assertEqual((row["n_context"], row["n_scored"]), (2, 2))
            self.assertTrue(0.0 < row["likelihood"] < 1.0)
            self.assertTrue(0.0 < row["likelihood_trial_marginal"] < 1.0)
            self.assertIn("converged", row)
        self.assertTrue(0.0 < self.output["heldout_likelihood"]["matched"] < 1.0)

    def test_logs_eval_but_never_test_likelihood(self):
        """For the GRU test_likelihood is zero-shot; with no population there is none here."""
        summary = self.wandb_run.summary
        for key in ("heldout/eval_likelihood", "heldout/eval_likelihood_se",
                    "heldout/eval_likelihood_trial_marginal", "heldout/min_weight_ess",
                    "hb/max_r_hat", "hb/min_ess_bulk", "hb/n_subjects_converged",
                    "heldout/n_sessions_pareto_k_fail", "heldout/pareto_k_threshold"):
            self.assertIn(key, summary)
        self.assertNotIn("heldout/test_likelihood", summary)
        # The totals reproduce the pooled likelihood, so separate runs can be pooled exactly.
        self.assertAlmostEqual(
            np.exp(summary["heldout/eval_log_lik_sum"] / summary["heldout/num_eval_trials"]),
            summary["heldout/eval_likelihood"], places=6)
        self.assertGreater(summary["heldout/eval_log_lik_sum_se"], 0.0)
        self.assertNotIn("heldout_test_likelihood", summary)

    def test_persists_draws_and_scores_per_run(self):
        """Draws, diagnostics and per-session scores land in this run's own subdirectory."""
        (run_dir,) = list(Path(self.artifacts.name).iterdir())
        draws = np.load(run_dir / "heldout_two_level_draws.npz")
        self.assertEqual(draws["mu_p"].shape, (60, 2, 5))
        scores = np.load(run_dir / "heldout_two_level_scores.npz")
        self.assertIn("session_marginal", scores.files)
        self.assertIn('"max_r_hat"', (run_dir / "heldout_two_level_fit.json").read_text())

    def test_requires_a_heldout_cohort(self):
        """Without held-out subjects there is nothing to fit."""
        bundle = DatasetBundle(raw=_make_frame(), train_set=None, eval_set=None, metadata={})
        trainer = HBTrainer(estimator="heldout_two_level", num_chains=2, seed=0)
        with self.assertRaises(ValueError):
            trainer.fit(bundle)


@unittest.skipUnless(HAS_BAYES, "requires the 'bayes' extra (jax, numpyro)")
class TestSessionSitePersistence(unittest.TestCase):
    """The session-sites knob reaches ``save_fit``, not merely the config.

    Checking that the key parses would prove nothing: ``HBTrainer.__init__`` absorbs
    unknown keyword arguments, so a misnamed or unthreaded config key is accepted in
    silence and the fit is written without the session-level sites -- unrecoverable
    without a refit. These tests assert the value the callee actually received.
    """

    def _recorded_kwargs(self, **trainer_kwargs):
        """Run a tiny fit with ``save_fit`` stubbed; return the kwargs it was called with."""
        recorded = {}

        def fake_save_fit(mcmc, output_dir, **kwargs):
            recorded.update(kwargs)
            return {"netcdf": str(Path(output_dir) / "fit.nc"), "json": None,
                    "sample_stats": None, "diagnostics": {}}

        bundle = DatasetBundle(
            raw=_make_frame(n_subjects=2, n_sessions=2, n_trials=40),
            train_set=None, eval_set=None, metadata={},
        )
        with tempfile.TemporaryDirectory() as artifact_dir:
            trainer = HBTrainer(
                estimator="one_stage", num_warmup=10, num_samples=10, num_chains=1,
                artifact_dir=artifact_dir, seed=0, **trainer_kwargs,
            )
            with mock.patch(
                "aind_dynamic_foraging_models.hierarchical_bayes.artifacts.save_fit",
                fake_save_fit,
            ):
                trainer.fit(bundle)
        self.assertIn("include_session_sites", recorded)
        return recorded

    def test_default_persists_session_sites(self):
        """A production rung saves them without asking: the default is on."""
        self.assertIs(self._recorded_kwargs()["include_session_sites"], True)

    def test_knob_can_be_turned_off(self):
        """The config value is what is passed through, not a hardcoded True."""
        self.assertIs(
            self._recorded_kwargs(save_session_sites=False)["include_session_sites"],
            False,
        )


@unittest.skipUnless(HAS_BAYES, "requires the 'bayes' extra (jax, numpyro)")
class TestSamplerGeometry(unittest.TestCase):
    """``target_accept_prob`` and ``max_tree_depth`` reach ``NUTS``, not merely the config.

    Same silent-failure mode as the session-sites knob: ``HBTrainer.__init__`` absorbs
    unknown keyword arguments, so a misnamed or unthreaded key is accepted without a
    warning and the rung samples on NumPyro's defaults while its config claims otherwise.
    That is the more expensive half here -- the settings exist to cure the D30 gate
    failures, and a rung that silently ignores them looks like the remedy did not work.
    """

    def _recorded_nuts_kwargs(self, **trainer_kwargs):
        """Run a tiny fit with ``NUTS`` stubbed; return the kwargs it was called with."""
        import numpyro.infer

        recorded = {}
        real_nuts = numpyro.infer.NUTS

        def fake_nuts(model, **kwargs):
            recorded.update(kwargs)
            return real_nuts(model, **kwargs)

        bundle = DatasetBundle(
            raw=_make_frame(n_subjects=2, n_sessions=2, n_trials=40),
            train_set=None, eval_set=None, metadata={},
        )
        trainer = HBTrainer(
            estimator="one_stage", num_warmup=10, num_samples=10, num_chains=1,
            seed=0, **trainer_kwargs,
        )
        with mock.patch("numpyro.infer.NUTS", fake_nuts):
            trainer.fit(bundle)
        return recorded

    def test_defaults_match_numpyro(self):
        """Defaults leave sampling unchanged, so existing rungs stay comparable."""
        recorded = self._recorded_nuts_kwargs()
        self.assertEqual(recorded.get("target_accept_prob"), 0.8)
        self.assertEqual(recorded.get("max_tree_depth"), 10)

    def _recorded_meta(self, estimator):
        """Run a tiny fit with ``save_fit`` stubbed; return the ``meta`` it received."""
        recorded = {}

        def fake_save_fit(mcmc, output_dir, **kwargs):
            recorded.update(kwargs)
            return {"netcdf": str(Path(output_dir) / "fit.nc"), "json": None,
                    "sample_stats": None, "diagnostics": {}}

        bundle = DatasetBundle(
            raw=_make_frame(n_subjects=2, n_sessions=2, n_trials=40),
            train_set=None, eval_set=None, metadata={},
        )
        with tempfile.TemporaryDirectory() as artifact_dir:
            trainer = HBTrainer(
                estimator=estimator, num_warmup=8, num_samples=8, num_chains=1,
                artifact_dir=artifact_dir, seed=0,
                target_accept_prob=0.95, max_tree_depth=11,
            )
            with mock.patch(
                "aind_dynamic_foraging_models.hierarchical_bayes.artifacts.save_fit",
                fake_save_fit,
            ):
                trainer.fit(bundle)
        return recorded["meta"]

    def test_meta_records_geometry_only_where_it_applied(self):
        """The artifact must not claim a setting the sampler never used.

        ``two_stage`` samples through ``fit_two_stage``, whose subject and population
        kernels take no geometry arguments, so recording 0.95 on such a run would give an
        analyst a number that never reached a sampler. ``estimator`` is in the same meta,
        so the absence is self-explaining rather than ambiguous.
        """
        one = self._recorded_meta("one_stage")
        self.assertEqual(one["target_accept_prob"], 0.95)
        self.assertEqual(one["max_tree_depth"], 11)

        two = self._recorded_meta("two_stage")
        self.assertEqual(two["estimator"], "two_stage")
        self.assertNotIn("target_accept_prob", two)
        self.assertNotIn("max_tree_depth", two)

    def test_out_of_range_values_are_rejected(self):
        """A config typo fails at construction, not silently or hours in.

        NumPyro validates neither: verified against numpyro 0.21.0,
        ``NUTS(target_accept_prob=9.5)`` constructs *and samples to completion* without
        error, so 9.5-for-0.95 would produce a fit whose adaptation targeted an
        unreachable acceptance rate with nothing in the log to say so.
        ``max_tree_depth=0`` fails only inside the integrator, as an ``IndexError``.
        Neither needs a sampler run to catch, so neither should cost one.
        """
        for bad in (9.5, 0.0, 1.0, -0.1):
            with self.assertRaises(ValueError) as ctx:
                HBTrainer(seed=0, target_accept_prob=bad)
            self.assertIn("target_accept_prob", str(ctx.exception))
        for bad in (0, -3):
            with self.assertRaises(ValueError) as ctx:
                HBTrainer(seed=0, max_tree_depth=bad)
            self.assertIn("max_tree_depth", str(ctx.exception))

    def test_values_reach_nuts(self):
        """The configured values are what NUTS receives, not the signature defaults."""
        recorded = self._recorded_nuts_kwargs(
            target_accept_prob=0.95, max_tree_depth=12
        )
        self.assertEqual(recorded.get("target_accept_prob"), 0.95)
        self.assertEqual(recorded.get("max_tree_depth"), 12)


if __name__ == "__main__":
    unittest.main()
