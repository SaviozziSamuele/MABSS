"""Thompson Sampling CMAB policy (Bayesian linear regression per arm)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from .base import Policy


class ThompsonSamplingPolicy(Policy):
    """
    Ported verbatim from MABSS_utility.py's `ThompsonSamplingPolicyCMAB` (:572-635).
    No logic change -- not implicated in any confirmed bug, aside from `reset()`
    setting `self.average_reward`/`self.num_selected`, which this class never reads
    (the original set them too; harmless dead state kept here for parity, since it
    doesn't affect any computation).

    Uses `np.random.standard_normal` (legacy global RNG) for the posterior sample,
    same as the original. Migrating to an injected `np.random.Generator` is
    Phase 4 item 9 territory (seeding/determinism) and is deliberately not done in
    this class, since it would change every downstream number and is gated behind
    the plan's `rng_backend` flag rather than a silent default-class change.

    Performance note (plan finding [J]): recomputes a full Cholesky factorization
    per arm per step, O(K*d^3). Not optimized here to preserve exact numerics;
    a batched/vectorized version is future work, gated behind a
    `test_vectorized_matches_loop(atol=1e-12)` check per the plan.
    """

    is_deterministic = True

    def __init__(
        self,
        K_arms: int,
        context_dim: int,
        alpha: float,
        gamma: float | None = None,
        lam: float = 1.0,
    ):
        self.K_arms = K_arms
        self.d = context_dim
        self.v = alpha  # variance multiplier (exploration parameter)
        self.lam = lam  # ridge regularization
        self.g = gamma  # unused, kept for interface parity (see docstring)

        self.A = [self.lam * np.eye(self.d) for _ in range(self.K_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.K_arms)]

    def select_arm(self, context) -> np.ndarray:
        x = np.asarray(context)
        sampled_values = np.zeros(self.K_arms)

        for k in range(self.K_arms):
            L = cholesky(self.A[k], lower=True)
            y = solve_triangular(L, self.b[k], lower=True)
            mu = solve_triangular(L.T, y, lower=False)

            z = np.random.standard_normal(size=(self.d, 1))
            w = solve_triangular(L.T, z, lower=False)
            sample = mu + self.v * w

            x_k = x[k].reshape(-1, 1)
            sampled_values[k] = (sample.T @ x_k).item()

        arm = int(np.argmax(sampled_values))
        probs = np.zeros(self.K_arms)
        probs[arm] = 1.0
        return probs

    def reward_update(self, context, probs, arm_selected: int, reward: float) -> None:
        x_k = np.asarray(context)[arm_selected].reshape(-1, 1)
        self.A[arm_selected] += x_k @ x_k.T
        self.b[arm_selected] += reward * x_k

    def reset(self) -> None:
        self.A = [self.lam * np.eye(self.d) for _ in range(self.K_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.K_arms)]
        self.average_reward = 0.0
        self.num_selected = 0
