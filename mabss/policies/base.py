"""Common interface for CMAB exploration policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    """
    A CMAB policy tracks per-arm parameters and, at each step, turns a set of
    per-arm context vectors into a selection.

    `select_arm` always returns a probability vector of shape (K,), for every
    policy in this package -- Softmax returns a real simplex distribution;
    LinUCB and Thompson Sampling return a degenerate one-hot vector (they are
    deterministic given their current parameters, up to the tie-break in
    np.argmax). `is_deterministic` makes that distinction explicit so callers
    (see mabss.env.WalkForwardEnv) can skip a meaningless np.random.choice over
    a one-hot distribution.
    """

    is_deterministic: bool = False

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> np.ndarray:
        """context: array-like of shape (K, d), one context vector per arm.
        Returns a probability vector of shape (K,)."""

    @abstractmethod
    def reward_update(
        self, context: np.ndarray, probs: np.ndarray, arm_selected: int, reward: float
    ) -> None:
        """Update the policy's parameters after observing `reward` for `arm_selected`."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all learned parameters to their initial values."""
