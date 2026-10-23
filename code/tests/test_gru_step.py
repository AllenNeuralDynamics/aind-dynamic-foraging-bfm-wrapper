"""Parity of the free-context GRU step with the Haiku multisubject network."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import haiku as hk  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from models.gru_network import make_gru_network  # noqa: E402
from post_training_analysis.gru_step import GruStep, find_fixed_points, split_inputs, subject_context  # noqa: E402

N_SUBJECTS, T, B, H, E = 3, 12, 5, 8, 4


def _network_and_params():
    make_network = make_gru_network(
        hidden_size=H, output_size=2, multisubject=True, max_n_subjects=N_SUBJECTS, subject_embedding_size=E,
        subject_embedding_init="small_random", session_encoding_type="scalar",
        session_max_index_by_subject_index=(9, 9, 9))
    rng = np.random.default_rng(0)
    subjects = rng.integers(0, N_SUBJECTS, size=(1, B)).repeat(T, axis=0)
    sessions = rng.integers(0, 10, size=(1, B)).repeat(T, axis=0)
    obs = rng.integers(0, 2, size=(T, B, 2)).astype(np.float32)
    obs[0] = -1.0
    xs = np.concatenate([subjects[..., None], sessions[..., None], obs], axis=-1).astype(np.float32)

    def forward(x):
        core = make_network()
        return hk.dynamic_unroll(core, x, core.initial_state(B))

    transformed = hk.transform(forward)
    params = transformed.init(jax.random.PRNGKey(0), xs)
    # Perturb every parameter so the test does not pass on a near-zero initialisation.
    params = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(jax.random.PRNGKey(1), p.shape), params)
    logits, final_state = transformed.apply(params, None, xs)  # dynamic_unroll: (outputs, final state)
    return make_network, params, xs, np.asarray(logits), np.asarray(final_state)


class TestGruStep(unittest.TestCase):
    def test_step_and_context_reproduce_the_network(self) -> None:
        make_network, params, xs, logits, final_state = _network_and_params()
        subject, session, obs = split_inputs(xs, session_conditioning=True)
        c = subject_context(make_network, params, subject.reshape(-1), session.reshape(-1)).reshape(T, B, E)
        step = GruStep.from_params(params)
        ours = np.asarray(step.run(obs, c))
        np.testing.assert_allclose(ours[-1], final_state, atol=1e-5)
        np.testing.assert_allclose(np.asarray(step.logit_diff(ours)), logits[..., 1] - logits[..., 0], atol=1e-5)

    def test_jacobian_matches_finite_differences(self) -> None:
        _, params, _, _, final_state = _network_and_params()
        step = GruStep.from_params(params)
        h, obs, c = final_state[0], np.array([1.0, 0.0], np.float32), np.full(E, 0.1, np.float32)
        jac = np.asarray(step.jacobian(h, obs, c))
        eps = 1e-3
        numeric = np.stack([(np.asarray(step.step(h + eps * e, obs, c)) - np.asarray(step.step(h - eps * e, obs, c)))
                            / (2 * eps) for e in np.eye(H, dtype=np.float32)], axis=1)
        np.testing.assert_allclose(jac, numeric, atol=1e-3)

    def test_run_ablated_with_empty_basis_matches_run(self) -> None:
        make_network, params, xs, logits, _ = _network_and_params()
        subject, session, obs = split_inputs(xs, session_conditioning=True)
        c = subject_context(make_network, params, subject.reshape(-1), session.reshape(-1)).reshape(T, B, E)
        step = GruStep.from_params(params)
        ablated = np.asarray(step.run_ablated(obs, c, np.zeros((H, 0), np.float32), np.zeros(H, np.float32)))
        np.testing.assert_allclose(ablated, logits[..., 1] - logits[..., 0], atol=1e-5)

    def test_run_ablated_clamps_the_subspace_to_the_mean(self) -> None:
        make_network, params, xs, _, _ = _network_and_params()
        subject, session, obs = split_inputs(xs, session_conditioning=True)
        c = subject_context(make_network, params, subject.reshape(-1), session.reshape(-1)).reshape(T, B, E)
        step = GruStep.from_params(params)
        basis, mean = np.eye(H, dtype=np.float32), np.full(H, 0.3, np.float32)  # ablate everything
        ablated = np.asarray(step.run_ablated(obs, c, basis, mean))
        np.testing.assert_allclose(ablated, np.full_like(ablated, float(step.logit_diff(mean))), atol=1e-5)
        memory = np.asarray(step.run_ablated(obs, c, basis, mean, memory_only=True))
        # memory-only: the intact state after one step from the clamped mean, i.e. depends on input only.
        expected = np.asarray(step.logit_diff(step.step(np.broadcast_to(mean, (T, B, H)), obs, c)))
        np.testing.assert_allclose(memory[1:], expected[1:], atol=1e-5)

    def test_run_ablated_conditional_mean_uses_the_input_specific_mean(self) -> None:
        make_network, params, xs, _, _ = _network_and_params()
        subject, session, obs = split_inputs(xs, session_conditioning=True)
        c = subject_context(make_network, params, subject.reshape(-1), session.reshape(-1)).reshape(T, B, E)
        step = GruStep.from_params(params)
        means = np.stack([np.full(H, v, np.float32) for v in (0.1, -0.2, 0.3, -0.4, 0.0)])
        ablated = np.asarray(step.run_ablated(obs, c, np.eye(H, dtype=np.float32), means))
        index = np.where(obs[..., 0] < 0, 4, (2 * obs[..., 0] + obs[..., 1]).astype(int))
        expected = np.asarray(step.logit_diff(means[index]))
        np.testing.assert_allclose(ablated, expected, atol=1e-5)

    def test_find_fixed_points_returns_true_fixed_points(self) -> None:
        jax.config.update("jax_enable_x64", True)
        try:
            _, params, _, _, final_state = _network_and_params()
            params64 = jax.tree_util.tree_map(lambda p: np.asarray(p, dtype=np.float64), params)
            step = GruStep.from_params(params64)
            obs, c = np.array([1.0, 1.0]), np.full(E, 0.1)
            starts = np.random.default_rng(0).normal(scale=0.5, size=(16, H))
            found, _ = find_fixed_points(step, starts, obs, c)
            self.assertGreaterEqual(len(found), 1)
            for f in found:
                np.testing.assert_allclose(np.asarray(step.step(f, obs, c)), f, atol=1e-6)
        finally:
            jax.config.update("jax_enable_x64", False)


if __name__ == "__main__":
    unittest.main()
