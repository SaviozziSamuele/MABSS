"""Softmax CMAB policy, trained via stochastic gradient ascent on the policy gradient."""

from __future__ import annotations

import numpy as np

from .base import Policy


class SoftmaxPolicy(Policy):
    """
    Ported from MABSS_utility.py's `SoftmaxPolicyCMAB` (:453-513).

    Fix applied here vs. the original (plan Phase 4, item 7 -- "Softmax baseline
    reset"): the original `__init__`/`reward_update` used a misspelled
    `self.avarage_reward`, while `reset()` set a *different*, correctly-spelled
    `self.average_reward` -- so the running reward baseline used to compute the
    policy-gradient advantage never actually reset across walk-forward windows,
    even when the caller passed `reset=True`. That silently contradicts
    Manuscript.tex:216 ("the CMAB's learned parameters are also reinitialized at
    the beginning of each walk-forward step."). Here, `reset()` clears the same
    `self.average_reward` attribute that `select_arm`/`reward_update` use.

    Applying this fix to the already-published Softmax results requires re-running
    the 10 Softmax bandit configs from cached contexts/rewards (CPU only, no
    retraining) -- see the plan's Phase 4 re-run scope table. That re-run is not
    performed by this change; this class is the corrected implementation the
    re-run should use.
    """

    is_deterministic = False

    def __init__(self, K_arms: int, context_dim: int, alpha: float, gamma: float | None):
        self.K_arms = K_arms
        self.context_dim = context_dim
        self.theta = np.zeros((K_arms, context_dim))
        self.a = alpha
        self.g = gamma
        self.average_reward = 0.0
        self.num_selected = 0

    def select_arm(self, context) -> np.ndarray:
        x = np.asarray(context)  # (K, d)
        logits = (self.theta * x).sum(axis=1)  # theta_k . x_k
        logits = logits - np.max(logits)  # numerical stability
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def reward_update(self, context, probs, arm_selected: int, reward: float) -> None:
        x = np.asarray(context)
        probs = np.asarray(probs)

        self.num_selected += 1
        g = 1.0 / self.num_selected if self.g is None else self.g

        baselined_reward = reward - self.average_reward

        for k in range(self.K_arms):
            indicator = 1.0 if k == arm_selected else 0.0
            grad_logpi = (indicator - probs[k]) * x[k]
            self.theta[k] += self.a * baselined_reward * grad_logpi

        self.average_reward += g * (reward - self.average_reward)

    def reset(self) -> None:
        self.theta = np.zeros((self.K_arms, self.context_dim))
        self.average_reward = 0.0
        self.num_selected = 0
