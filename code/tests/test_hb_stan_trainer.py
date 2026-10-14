"""HBStanTrainer's thread count, seed wiring and persistence order (no Stan install needed).

`fit()` is exercised against a fake ``stan`` module, so these tests check what the trainer
passes to Stan and when it writes to disk, not Stan's own behaviour.
"""

import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:  # pragma: no cover - depends on the installed extra
    HAS_ARVIZ = False

from model_trainers import hb_stan_trainer as hbs
from model_trainers.hb_stan_trainer import HBStanTrainer


class TestThreadsPerChain(unittest.TestCase):
    """``_threads_per_chain`` divides the SLURM allocation between the chains.

    Chains run as separate processes, so the cores are split between them rather than handed to
    each; reading the allocation (not ``os.cpu_count``) keeps a job on a shared node inside what
    it was given.
    """

    def _threads(self, env, num_chains=4):
        with mock.patch.dict(os.environ, env, clear=True):
            return HBStanTrainer(num_chains=num_chains)._threads_per_chain()

    def test_divides_the_task_allocation_between_chains(self):
        self.assertEqual(self._threads({"SLURM_CPUS_PER_TASK": "32"}), 8)
        self.assertEqual(self._threads({"SLURM_CPUS_PER_TASK": "8"}), 2)

    def test_never_less_than_one_thread(self):
        self.assertEqual(self._threads({"SLURM_CPUS_PER_TASK": "2"}), 1)
        self.assertEqual(self._threads({"SLURM_CPUS_PER_TASK": "0"}), 1)

    def test_falls_back_to_the_node_allocation_then_to_one(self):
        self.assertEqual(self._threads({"SLURM_CPUS_ON_NODE": "16"}), 4)
        self.assertEqual(self._threads({}), 1)

    def test_ignores_a_non_numeric_value(self):
        self.assertEqual(self._threads({"SLURM_CPUS_PER_TASK": "unknown"}), 1)

    def test_task_allocation_wins_over_node_allocation(self):
        self.assertEqual(self._threads({"SLURM_CPUS_PER_TASK": "8", "SLURM_CPUS_ON_NODE": "112"}), 2)

    def test_depends_on_the_chain_count(self):
        self.assertEqual(self._threads({"SLURM_CPUS_PER_TASK": "32"}, num_chains=2), 16)


class _FakeStan:
    """Stands in for ``import stan``: records what ``build`` receives."""

    def __init__(self):
        self.build_kwargs = None
        outer = self

        class _Posterior:
            def sample(self, num_chains, num_samples, num_warmup):
                outer.sample_kwargs = dict(num_chains=num_chains, num_samples=num_samples,
                                           num_warmup=num_warmup)
                return {"population_mean": np.zeros((5, num_chains * num_samples))}

        self._posterior = _Posterior()

    def build(self, program, data, random_seed):
        self.build_kwargs = dict(program=program, data=data, random_seed=random_seed)
        return self._posterior


class TestFitWiring(unittest.TestCase):
    def _run(self, trainer, *, diagnostics=None, program="program"):
        fake = _FakeStan()
        cohort = (np.zeros((2, 3, 4)), np.zeros((2, 3, 4)), np.ones((2, 3, 4), bool),
                  np.ones((2, 3), bool), ["a", "b"])
        patches = [
            mock.patch.dict(sys.modules, {"stan": types.ModuleType("stan")}),
            mock.patch.object(hbs, "_extract_subject_sessions", return_value=([], [], None)),
            mock.patch.object(hbs, "_pad_cohort", return_value=cohort),
            mock.patch.object(hbs, "cohort_to_stan_data", return_value={}),
            mock.patch.object(hbs, "_source_revisions", return_value={}),
            mock.patch.object(hbs, "_stan_program",
                              side_effect=program if isinstance(program, Exception) else None,
                              return_value=None if isinstance(program, Exception) else program),
            mock.patch.object(HBStanTrainer, "_diagnostics", **(diagnostics or {"return_value": {}})),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        sys.modules["stan"].build = fake.build
        bundle = SimpleNamespace(raw=pd.DataFrame({"x": [1]}))

        def run(run_id=None):
            loggers = None
            if run_id is not None:
                loggers = {"wandb": SimpleNamespace(id=run_id, log=lambda *_: None,
                                                    summary=SimpleNamespace(update=lambda *_: None))}
            return trainer.fit(bundle, loggers)

        return fake, run

    def test_seed_reaches_stan_build(self):
        fake, run = self._run(HBStanTrainer(seed=7, num_chains=2, num_samples=3, num_warmup=4))
        run()
        self.assertEqual(fake.build_kwargs["random_seed"], 7)
        self.assertEqual(fake.sample_kwargs, dict(num_chains=2, num_samples=3, num_warmup=4))

    def test_build_uses_whatever_seed_the_trainer_holds(self):
        """With seed=None the base class substitutes a wall-clock seed, and that is what Stan gets.

        This is why the model config must wire `seed: ${seed}`: without it a run cannot be
        reproduced from its own config. The trainer itself passes the seed through unchanged.
        """
        trainer = HBStanTrainer(seed=None)
        fake, run = self._run(trainer)
        run()
        self.assertEqual(fake.build_kwargs["random_seed"], trainer.seed)

    def test_draws_are_written_even_if_diagnostics_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = HBStanTrainer(seed=0, num_chains=2, num_samples=3, artifact_dir=temp)
            _, run = self._run(trainer, diagnostics={"side_effect": RuntimeError("arviz broke")})
            with self.assertRaises(RuntimeError):
                run("runA")
            self.assertTrue(os.path.exists(os.path.join(temp, "runA", "stan_one_stage_fit.npz")))
            # The metadata needs the diagnostics, so it is not written in that case.
            self.assertFalse(os.path.exists(os.path.join(temp, "runA", "stan_one_stage_fit.json")))

    def test_metadata_is_written_beside_the_draws_on_success(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = HBStanTrainer(seed=3, num_chains=2, num_samples=3, artifact_dir=temp)
            _, run = self._run(trainer)
            run("runA")
            self.assertTrue(os.path.exists(os.path.join(temp, "runA", "stan_one_stage_fit.npz")))
            self.assertTrue(os.path.exists(os.path.join(temp, "runA", "stan_one_stage_fit.json")))

    def test_runs_sharing_an_artifact_dir_do_not_overwrite_each_other(self):
        """Every Stan run of a sweep on HPC has the same artifact_dir; the D~99 rung must not
        replace the D~29 posterior."""
        with tempfile.TemporaryDirectory() as temp:
            for run_id in ("runA", "runB"):
                trainer = HBStanTrainer(seed=0, num_chains=2, num_samples=3, artifact_dir=temp)
                _, run = self._run(trainer)
                run(run_id)
            for run_id in ("runA", "runB"):
                self.assertTrue(os.path.exists(os.path.join(temp, run_id, "stan_one_stage_fit.npz")))

    def test_without_a_wandb_run_the_directory_is_still_unique_per_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = HBStanTrainer(seed=5, num_chains=2, num_samples=3, artifact_dir=temp)
            _, run = self._run(trainer)
            run()
            (subdir,) = os.listdir(temp)
            self.assertTrue(subdir.startswith("seed5-"))

    def test_missing_stan_program_fails_before_the_cohort_is_built(self):
        trainer = HBStanTrainer(seed=0)
        _, run = self._run(trainer, program=FileNotFoundError("no stan file"))
        with self.assertRaises(FileNotFoundError):
            run()
        hbs._pad_cohort.assert_not_called()


def _ragged_cohort():
    """Two subjects: 3 and 2 sessions, trial counts 4, 2, 3 and 4, 1, padded to (2, 3, 4)."""
    choice = np.zeros((2, 3, 4), dtype=int)
    choice[0, 0] = [1, 0, 1, 0]
    reward = np.zeros((2, 3, 4))
    reward[0, 0, :] = [1, 0, 0, 1]
    valid = np.zeros((2, 3, 4), dtype=bool)
    for (s, m), k in {(0, 0): 4, (0, 1): 2, (0, 2): 3, (1, 0): 4, (1, 1): 1}.items():
        valid[s, m, :k] = True
    session_mask = np.array([[True, True, True], [True, True, False]])
    return choice, reward, valid, session_mask


class TestCohortToStanData(unittest.TestCase):
    """The per-session trial COUNTS replace the mask, so a wrong count selects wrong trials."""

    def test_counts_shapes_and_values_for_a_ragged_cohort(self):
        data = hbs.cohort_to_stan_data(*_ragged_cohort(), beta_max=6.0)
        self.assertEqual((data["S"], data["M"], data["T"]), (2, 3, 4))
        self.assertEqual(data["n_sessions"], [3, 2])
        self.assertEqual(data["n_trials"], [[4, 2, 3], [4, 1, 0]])
        self.assertEqual(data["beta_max"], 6.0)
        self.assertEqual(data["choice"][0][0], [1, 0, 1, 0])
        self.assertEqual(data["reward"][0][0], [1, 0, 0, 1])

    def test_reward_is_binarised(self):
        choice, reward, valid, session_mask = _ragged_cohort()
        reward = reward * 2.5
        data = hbs.cohort_to_stan_data(choice, reward, valid, session_mask)
        self.assertEqual(data["reward"][0][0], [1, 0, 0, 1])

    def test_a_noncontiguous_valid_mask_is_rejected(self):
        choice, reward, valid, session_mask = _ragged_cohort()
        valid[0, 0] = [True, False, True, True]          # a hole in the middle
        with self.assertRaises(ValueError):
            hbs.cohort_to_stan_data(choice, reward, valid, session_mask)

    def test_valid_trials_after_the_prefix_are_rejected(self):
        choice, reward, valid, session_mask = _ragged_cohort()
        valid[1, 1] = [False, True, False, False]        # one valid trial, but not at index 0
        with self.assertRaises(ValueError):
            hbs.cohort_to_stan_data(choice, reward, valid, session_mask)


def _pystan_flat(per_chain):
    """Flatten ``(chain, draw)`` the way PyStan 3 does: draw-major, chain innermost."""
    return np.asarray(per_chain).T.reshape(-1)


class _FakeFit(dict):
    """``fit[name]`` lookups, as the trainer uses them."""


class TestChainsFirst(unittest.TestCase):
    def test_recovers_each_chain(self):
        per_chain = np.array([[0, 1, 2], [100, 101, 102], [200, 201, 202], [300, 301, 302]])
        np.testing.assert_array_equal(hbs.chains_first(_pystan_flat(per_chain), 4, 3), per_chain)

    def test_the_chain_major_reshape_is_wrong(self):
        per_chain = np.array([[0, 1, 2], [100, 101, 102], [200, 201, 202], [300, 301, 302]])
        self.assertEqual(_pystan_flat(per_chain).reshape(4, 3)[0].tolist(), [0, 100, 200])

    def test_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            hbs.chains_first(np.zeros(10), 4, 3)


@unittest.skipUnless(HAS_ARVIZ, "arviz not installed")
class TestDiagnostics(unittest.TestCase):
    """``_diagnostics`` against draws whose chains are distinguishable."""

    C, S = 4, 400

    def _fit(self, chain_offsets, diverging=0):
        rng = np.random.default_rng(0)
        per_chain = {i: rng.normal(size=(self.C, self.S)) + np.asarray(chain_offsets)[:, None]
                     for i in range(5)}
        pm = np.stack([_pystan_flat(per_chain[i]) for i in range(5)])
        div = np.zeros(self.C * self.S)
        div[:diverging] = 1
        return _FakeFit(population_mean=pm, divergent__=div), per_chain

    def test_well_mixed_chains_give_rhat_near_one_and_the_expected_ess(self):
        fit, per_chain = self._fit([0, 0, 0, 0])
        out = HBStanTrainer(num_chains=self.C, num_samples=self.S)._diagnostics(fit, 100.0)
        self.assertLess(out["hb/max_rhat"], 1.05)
        expected = min(float(az.ess(per_chain[i], method="bulk")) for i in range(5))
        self.assertAlmostEqual(out["hb/min_ess_bulk"], expected, places=6)
        self.assertAlmostEqual(out["hb/min_ess_per_draw"], expected / (self.C * self.S))
        self.assertAlmostEqual(out["hb/min_ess_per_second"], expected / 100.0)
        self.assertEqual(out["hb/n_draws"], self.C * self.S)

    def test_chains_at_different_levels_are_detected(self):
        """The old chain-major reshape mixed chains and reported R-hat near 1 here."""
        fit, _ = self._fit([0.0, 3.0, 6.0, 9.0])
        out = HBStanTrainer(num_chains=self.C, num_samples=self.S)._diagnostics(fit, 100.0)
        self.assertGreater(out["hb/max_rhat"], 1.5)

    def test_divergences_are_counted_and_missing_is_minus_one(self):
        fit, _ = self._fit([0, 0, 0, 0], diverging=7)
        trainer = HBStanTrainer(num_chains=self.C, num_samples=self.S)
        self.assertEqual(trainer._diagnostics(fit, 1.0)["hb/divergences"], 7)
        del fit["divergent__"]
        self.assertEqual(trainer._diagnostics(fit, 1.0)["hb/divergences"], -1)


class TestRunDirectoryFallback(unittest.TestCase):
    def test_two_fits_with_the_same_seed_get_different_directories(self):
        """Without a W&B run id the directory name must not depend on the clock."""
        with tempfile.TemporaryDirectory() as temp:
            dirs = set()
            for _ in range(2):
                trainer = HBStanTrainer(seed=5, num_chains=2, num_samples=3, artifact_dir=temp)
                fake = _FakeStan()
                patches = [
                    mock.patch.dict(sys.modules, {"stan": types.ModuleType("stan")}),
                    mock.patch.object(hbs, "_extract_subject_sessions", return_value=([], [], None)),
                    mock.patch.object(hbs, "_pad_cohort", return_value=(
                        np.zeros((2, 3, 4)), np.zeros((2, 3, 4)), np.ones((2, 3, 4), bool),
                        np.ones((2, 3), bool), ["a", "b"])),
                    mock.patch.object(hbs, "cohort_to_stan_data", return_value={}),
                    mock.patch.object(hbs, "_source_revisions", return_value={}),
                    mock.patch.object(hbs, "_stan_program", return_value="p"),
                    mock.patch.object(HBStanTrainer, "_diagnostics", return_value={}),
                    mock.patch.object(hbs.time, "time", return_value=1000.0),   # frozen clock
                ]
                for patch in patches:
                    patch.start()
                sys.modules["stan"].build = fake.build
                try:
                    trainer.fit(SimpleNamespace(raw=pd.DataFrame({"x": [1]})))
                finally:
                    mock.patch.stopall()
            dirs = {d for d in os.listdir(temp)}
            self.assertEqual(len(dirs), 2, dirs)


if __name__ == "__main__":
    unittest.main()
