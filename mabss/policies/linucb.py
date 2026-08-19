"""LinUCB (Linear Upper Confidence Bound) CMAB policy."""

from __future__ import annotations

import numpy as np

from .base import Policy


class LinUCBPolicy(Policy):
    """
    Ported verbatim from MABSS_utility.py's `UCBPolicyCMAB` (:515-569). No logic
    change here -- this policy was not implicated in any of the confirmed bugs.
    Maintains the inverse covariance per arm directly via a Sherman-Morrison
    rank-1 update, avoiding an explicit np.linalg.inv per step.

    Note: `gamma` is accepted for interface parity with the other policies (all
    three take (K_arms, context_dim/d, alpha, gamma) from
    WalkForwardEnv.__init__) but is genuinely unused -- LinUCB has no reward
    baseline/discount concept. This matches the original's behavior exactly.
    """

    is_deterministic = True

    def __init__(self, K_arms: int, context_dim: int, alpha: float = 1.0, gamma: float | None = None):
        self.K_arms = K_arms
        self.d = context_dim
        self.alpha = alpha
        # gamma intentionally unused; see docstring.
        self.A_inv = [np.eye(context_dim) for _ in range(K_arms)]
        self.b = [np.zeros((context_dim, 1)) for _ in range(K_arms)]

    def select_arm(self, context) -> np.ndarray:
        x = np.asarray(context)
        ucb_values = np.zeros(self.K_arms)

        for k in range(self.K_arms):
            x_k = x[k].reshape(-1, 1)
            a_inv_k = self.A_inv[k]
            theta_k = a_inv_k @ self.b[k]
            p_k = theta_k.T @ x_k
            v_k = x_k.T @ a_inv_k @ x_k
            ucb_values[k] = (p_k + self.alpha * np.sqrt(v_k)).item()

        arm = int(np.argmax(ucb_values))
        probs = np.zeros(self.K_arms)
        probs[arm] = 1.0
        return probs

    def reward_update(self, context, probs, arm_selected: int, reward: float) -> None:
        x_k = np.asarray(context)[arm_selected].reshape(-1, 1)

        self.b[arm_selected] += reward * x_k

        a_inv = self.A_inv[arm_selected]
        a_inv_x = a_inv @ x_k
        numerator = a_inv_x @ a_inv_x.T
        denominator = 1.0 + (x_k.T @ a_inv_x).item()
        self.A_inv[arm_selected] = a_inv - (numerator / denominator)

    def reset(self) -> None:
        self.A_inv = [np.eye(self.d) for _ in range(self.K_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.K_arms)]
