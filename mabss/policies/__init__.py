"""CMAB exploration policies: Softmax, LinUCB, Thompson Sampling."""

from .base import Policy
from .linucb import LinUCBPolicy
from .softmax import SoftmaxPolicy
from .thompson import ThompsonSamplingPolicy

POLICY_REGISTRY = {
    "softmax": SoftmaxPolicy,
    "ucb": LinUCBPolicy,
    "thompson": ThompsonSamplingPolicy,
}


def make_policy(name: str, K_arms: int, context_dim: int, alpha: float, gamma: float | None = None):
    """Case-insensitive policy factory (fixes MABSS_utility.py:797, where the
    original dispatch lowercased `self.policy_type` for the softmax/ucb branches
    but tested the raw, un-lowercased `policy_type` argument for 'thompson' --
    so e.g. policy_type='Thompson' raised ValueError while 'UCB' worked fine)."""
    key = name.lower()
    if key not in POLICY_REGISTRY:
        raise ValueError(f"Unsupported policy_type: {name!r} (known: {sorted(POLICY_REGISTRY)})")
    return POLICY_REGISTRY[key](K_arms, context_dim, alpha, gamma)


__all__ = [
    "Policy",
    "SoftmaxPolicy",
    "LinUCBPolicy",
    "ThompsonSamplingPolicy",
    "POLICY_REGISTRY",
    "make_policy",
]
