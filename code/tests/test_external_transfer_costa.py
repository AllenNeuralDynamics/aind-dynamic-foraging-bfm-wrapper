"""Tests for the Costa (macaque) block-aware author baseline (moved with the runner, dispatcher #179)."""

import unittest

import pandas as pd

from external_transfer import costa_author_baseline as MODULE


class TestCostaAuthorBaseline(unittest.TestCase):
    def test_real_source_blocks_are_separate_reset_units(self):
        rows = pd.DataFrame(
            {
                "ses_idx": ["adapt", "adapt", "adapt", "test"],
                "source_block": [1, 1, 2, 1],
                "trial": [1, 0, 2, 0],
                "animal_response": [0, 1, 0, 1],
                "earned_reward": [1, 0, 1, 1],
            }
        )
        blocks = MODULE._ordered_blocks(rows, {"adapt"})
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["trial"].tolist(), [0, 1])
        self.assertEqual(blocks[1]["trial"].tolist(), [2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
