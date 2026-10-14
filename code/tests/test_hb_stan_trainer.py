"""HBStanTrainer's thread count and seed wiring (no Stan install needed)."""

import os
import unittest
from unittest import mock

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


class TestSeed(unittest.TestCase):
    def test_seed_is_kept(self):
        self.assertEqual(HBStanTrainer(seed=7).seed, 7)


if __name__ == "__main__":
    unittest.main()
