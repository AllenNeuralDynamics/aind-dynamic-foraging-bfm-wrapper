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


if __name__ == "__main__":
    unittest.main()
