"""Input edits on a small 3-action multisubject GRU: effects equal a manual re-run."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

# Edited and baseline runs use different batch shapes; on GPU the default TF32 matmul makes
# them differ by ~5e-4 even for a no-op edit. Full float32 keeps that at ~1e-7.
jax.config.update("jax_default_matmul_precision", "highest")

from models.gru_network import make_gru_network  # noqa: E402
from post_training_analysis.gru_counterfactual import (  # noqa: E402
    _feature_offset,
    GruSplit, input_edit_effects, logits, readouts)

N_SUBJECTS, T, S, H, E = 3, 12, 4, 8, 4


def _split() -> GruSplit:
    make_network = make_gru_network(
        hidden_size=H, output_size=3, multisubject=True, max_n_subjects=N_SUBJECTS, subject_embedding_size=E,
        subject_embedding_init="small_random", session_encoding_type="scalar",
        session_max_index_by_subject_index=(9, 9, 9))
    rng = np.random.default_rng(0)
    subjects = rng.integers(0, N_SUBJECTS, size=(1, S)).repeat(T, axis=0)
    sessions = rng.integers(0, 10, size=(1, S)).repeat(T, axis=0)
    choice = rng.integers(0, 3, size=(T, S)).astype(np.float32)
    reward = np.where(choice == 2, 0, rng.integers(0, 2, size=(T, S))).astype(np.float32)
    obs = np.stack([choice, reward], axis=-1)
    obs[0] = -1.0
    xs = np.concatenate([subjects[..., None], sessions[..., None], obs], axis=-1).astype(np.float32)

    def forward(x):
        core = make_network()
        return hk.dynamic_unroll(core, x, core.initial_state(S))

    params = hk.transform(forward).init(jax.random.PRNGKey(0), xs)
    params = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(jax.random.PRNGKey(1), p.shape), params)
    n_trials = np.array([T, T, T - 3, T])
    return GruSplit(None, make_network, params, xs, [f"s{i}" for i in range(S)], n_trials, 2, 3, n_actions=3)


class TestInputEdits(unittest.TestCase):
    def setUp(self) -> None:
        self.split = _split()

    def test_readouts_are_side_and_engage(self) -> None:
        z = logits(self.split)
        r = readouts(z)
        np.testing.assert_allclose(r["side"], z[..., 1] - z[..., 0], atol=1e-6)
        p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        np.testing.assert_allclose(r["engage"], np.log((p[..., 0] + p[..., 1]) / p[..., 2]), atol=1e-4)

    def test_no_op_edit_has_no_effect(self) -> None:
        cols, ts = np.array([0, 1]), np.array([2, 4])
        xs = self.split.xs
        effects = input_edit_effects(self.split, cols, ts, choice=xs[ts + 1, cols, 2], reward=xs[ts + 1, cols, 3], k_max=4)
        for value in effects.values():
            np.testing.assert_allclose(value, 0.0, atol=1e-6)

    def test_edit_matches_manual_rerun_and_stops_at_session_end(self) -> None:
        col, t, k_max = 2, 5, 6  # session 2 has T−3 valid steps
        effects = input_edit_effects(self.split, [col], [t], choice=[2.0], reward=[0.0], k_max=k_max)
        edited = self.split.xs.copy()
        edited[t + 1, col, 2:] = (2.0, 0.0)
        before, after = readouts(logits(self.split)), readouts(logits(self.split, edited))
        valid = (t + np.arange(1, k_max + 1)) < self.split.n_trials[col]
        for name in ("side", "engage"):
            expected = after[name][t + 1:t + 1 + k_max, col] - before[name][t + 1:t + 1 + k_max, col]
            np.testing.assert_allclose(effects[name][0, valid], expected[valid], atol=1e-5)
            self.assertTrue(np.isnan(effects[name][0, ~valid]).all())
        self.assertGreater(np.abs(effects["engage"][0, 0]), 0)


class TestFeatureOffset(unittest.TestCase):
    """Where "previous choice" sits in xs: after the subject index and the session index, if any."""

    def test_single_subject_has_no_index_columns(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415
        self.assertEqual(_feature_offset(SimpleNamespace(multisubject=False, session_conditioning_enabled=False)), 0)

    def test_multisubject_adds_subject_then_session_index(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415
        self.assertEqual(_feature_offset(SimpleNamespace(multisubject=True, session_conditioning_enabled=False)), 1)
        self.assertEqual(_feature_offset(SimpleNamespace(multisubject=True, session_conditioning_enabled=True)), 2)


if __name__ == "__main__":
    unittest.main()
