"""One step of a trained multisubject GRU with the context as a free input.

``h_next = step(h, obs, c)``: obs = (choice, reward) of the trial just completed, c = the
subject context (subject embedding + session delta) — the same update the trained network
applies, but with any state, input and context, so dynamics can be probed off the observed
trajectories (flow fields, fixed points, Jacobians, context sweeps).

``logit_diff(h)`` is the readout's logit(right) − logit(left) of a state. In the trained
network the readout reads the *new* state, so the row-t prediction is
``logit_diff(step(h_{t-1}, obs_{t-1}, c))``.

``subject_context`` evaluates the trained network's own context module (embedding lookup
plus session-delta MLP), so contexts are exactly those the model used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _param(params: Any, suffix: str) -> dict:
    matches = [v for k, v in params.items() if k.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one parameter group ending in {suffix!r}, found {len(matches)}")
    return {k: jnp.asarray(v) for k, v in matches[0].items()}


@dataclass(frozen=True)
class GruStep:
    w_i: jnp.ndarray
    w_h: jnp.ndarray
    b: jnp.ndarray
    readout_w: jnp.ndarray
    readout_b: jnp.ndarray

    @classmethod
    def from_params(cls, params: Any) -> "GruStep":
        gru, readout = _param(params, "/~/gru"), _param(params, "/~/readout")
        return cls(gru["w_i"], gru["w_h"], gru["b"], readout["w"], readout["b"])

    @property
    def hidden_size(self) -> int:
        return int(self.w_h.shape[0])

    def step(self, h, obs, c):
        """Haiku GRU update on [obs, c]; broadcasts over leading batch dimensions."""
        n = self.hidden_size
        x = jnp.concatenate([obs, c], axis=-1)
        zr_x, a_x = jnp.split(x @ self.w_i, [2 * n], axis=-1)
        w_h_z, w_h_a = jnp.split(self.w_h, [2 * n], axis=1)
        b_z, b_a = jnp.split(self.b, [2 * n])
        z, r = jnp.split(jax.nn.sigmoid(zr_x + h @ w_h_z + b_z), 2, axis=-1)
        a = jnp.tanh(a_x + (r * h) @ w_h_a + b_a)
        return (1 - z) * h + z * a

    def logit_diff(self, h):
        logits = h @ self.readout_w + self.readout_b
        return logits[..., 1] - logits[..., 0]

    def jacobian(self, h, obs, c):
        """∂ step / ∂ h at a single state (h: [H], obs: [2], c: [E])."""
        return jax.jacfwd(self.step, argnums=0)(h, obs, c)

    def run(self, obs_seq, c_seq, h0=None):
        """Scan over time: obs_seq, c_seq [T, B, ·] → states [T, B, H] (state after each input)."""
        h0 = jnp.zeros(obs_seq.shape[1:-1] + (self.hidden_size,)) if h0 is None else h0

        def body(h, inputs):
            h = self.step(h, *inputs)
            return h, h

        _, states = jax.lax.scan(body, h0, (obs_seq, c_seq))
        return states

    def run_ablated(self, obs_seq, c_seq, basis, mean, *, memory_only: bool = False, h0=None):
        """Scan with the state's component in span(basis) replaced by the mean's at every step.

        basis: [H, k] orthonormal columns; mean: [H], or [5, H] for conditional replacement —
        rows 0–3 the mean state after each input (index 2·choice + reward of the input just
        received), row 4 for a session's first step (no previous trial). Returns
        logit_diff [T, B]. With
        ``memory_only`` the logit of each step is read from the intact state and only the state
        carried to the next step is ablated (does the subspace carry information forward?);
        otherwise the ablated state is both read out and carried (is the subspace needed at all?).
        """
        h0 = jnp.zeros(obs_seq.shape[1:-1] + (self.hidden_size,)) if h0 is None else h0
        basis, mean = jnp.asarray(basis), jnp.asarray(mean)

        def ablate(h, obs):
            if mean.ndim == 1:
                m = mean
            else:
                index = jnp.where(obs[..., 0] < 0, 4, (2 * obs[..., 0] + obs[..., 1]).astype(jnp.int32))
                m = mean[index]
            return h - ((h - m)[..., None, :] @ basis)[..., 0, :] @ basis.T

        def body(h, inputs):
            h_new = self.step(h, *inputs)
            h_abl = ablate(h_new, inputs[0])
            logit = self.logit_diff(h_new if memory_only else h_abl)
            return h_abl, logit

        _, logits = jax.lax.scan(body, h0, (obs_seq, c_seq))
        return logits


def subject_context(make_network: Any, params: Any, subject_idx: np.ndarray, session_idx: np.ndarray) -> np.ndarray:
    """The trained network's context (embedding + session delta) for given subject/session indices."""
    import haiku as hk  # noqa: PLC0415

    def fn(s, e):
        return make_network().compute_subject_context(subject_ids=s, session_ids=e)

    transformed = hk.without_apply_rng(hk.transform(fn))
    return np.asarray(transformed.apply(params, jnp.asarray(subject_idx, dtype=jnp.int32),
                                        jnp.asarray(session_idx, dtype=jnp.int32)))


def split_inputs(xs: np.ndarray, *, session_conditioning: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xs [T, S, F] → (subject_idx [T, S], session_idx [T, S] or zeros, obs [T, S, 2])."""
    offset = 2 if session_conditioning else 1
    session = xs[..., 1] if session_conditioning else np.zeros(xs.shape[:-1])
    return xs[..., 0], session, xs[..., offset:]


def find_fixed_points(step: GruStep, starts, obs, c, *, n_forward: int = 500, n_newton: int = 30,
                      damping: float = 1e-8, tol: float = 1e-12, dedup: float = 1e-3) -> tuple[list[np.ndarray], np.ndarray]:
    """Fixed points of h ↦ step(h, obs, c) reached from ``starts`` [N, H].

    Forward-iterates the map (converges to attractors, but slowly when |λ| ≈ 1), then refines
    every end point with damped Newton on F(h) − h, which also converges onto nearby saddles.
    Returns the distinct points with ½‖F(h) − h‖² < tol and all candidates' residuals. Run with
    ``jax_enable_x64`` for tolerances near 1e-12. (Study 12's ``d_dynamics.py`` contains the
    same procedure plus an Adam-based search, written before this helper existed.)
    """
    eye = jnp.eye(step.hidden_size, dtype=jnp.asarray(starts).dtype)

    @jax.jit
    def refine(h0):
        h = jax.lax.fori_loop(0, n_forward, lambda _, x: step.step(x, obs, c), h0)

        def newton(h):
            a = step.jacobian(h, obs, c) - eye
            r = step.step(h, obs, c) - h
            return h + jnp.linalg.solve(a.T @ a + damping * eye, -a.T @ r)

        return jax.lax.fori_loop(0, n_newton, lambda _, x: jax.vmap(newton)(x), h)

    h = np.asarray(refine(jnp.asarray(starts)))
    q = 0.5 * np.sum((np.asarray(step.step(jnp.asarray(h), obs, c)) - h) ** 2, axis=-1)
    found: list[np.ndarray] = []
    for i in np.argsort(q):
        if not np.isfinite(q[i]) or q[i] > tol:
            break
        if all(np.linalg.norm(h[i] - f) > dedup for f in found):
            found.append(h[i])
    return found, q
