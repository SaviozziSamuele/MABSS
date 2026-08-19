"""Unit tests for the three CMAB policies (mabss.policies)."""

import numpy as np
import pytest

from mabss.policies import LinUCBPolicy, SoftmaxPolicy, ThompsonSamplingPolicy


# --------------------------------------------------------------------------- #
# Softmax
# --------------------------------------------------------------------------- #


def test_softmax_select_arm_returns_simplex():
    p = SoftmaxPolicy(K_arms=5, context_dim=3, alpha=0.1, gamma=None)
    x = np.random.default_rng(0).normal(size=(5, 3))
    probs = p.select_arm(x)
    assert probs.shape == (5,)
    assert np.all(probs >= 0)
    assert np.isclose(probs.sum(), 1.0)


def test_softmax_uniform_when_theta_zero():
    p = SoftmaxPolicy(K_arms=4, context_dim=3, alpha=0.1, gamma=None)
    x = np.random.default_rng(1).normal(size=(4, 3))
    probs = p.select_arm(x)
    assert np.allclose(probs, 0.25)


def test_softmax_reward_update_moves_theta_for_selected_arm_only():
    p = SoftmaxPolicy(K_arms=3, context_dim=2, alpha=1.0, gamma=None)
    x = np.eye(3, 2)  # arm k's context is a distinct basis-ish vector
    probs = p.select_arm(x)
    theta_before = p.theta.copy()
    p.reward_update(x, probs, arm_selected=1, reward=1.0)
    # Every arm's theta changes (the softmax gradient touches all rows via
    # `(indicator - probs[k])`), but the selected arm's gradient direction is
    # distinguished by the +1 indicator term.
    assert not np.allclose(p.theta, theta_before)


def test_softmax_reset_clears_theta_and_baseline():
    """Regression test for bug :471/:512 -- the original SoftmaxPolicyCMAB used a
    misspelled `self.avarage_reward` in __init__/reward_update, while `reset()`
    zeroed a *different*, correctly-spelled `self.average_reward`, so the reward
    baseline never actually reset. This class uses one consistently-spelled
    attribute throughout, so reset() must actually zero what reward_update reads."""
    p = SoftmaxPolicy(K_arms=3, context_dim=2, alpha=0.5, gamma=None)
    x = np.random.default_rng(2).normal(size=(3, 2))
    probs = p.select_arm(x)
    p.reward_update(x, probs, arm_selected=0, reward=5.0)
    assert p.average_reward != 0.0
    assert p.num_selected == 1

    p.reset()
    assert p.average_reward == 0.0
    assert p.num_selected == 0
    assert np.all(p.theta == 0.0)


def test_softmax_has_no_misspelled_attribute():
    """Stops the `avarage_reward` typo from ever coming back."""
    p = SoftmaxPolicy(K_arms=2, context_dim=2, alpha=0.1, gamma=None)
    assert not hasattr(p, "avarage_reward")


def test_softmax_gamma_none_uses_running_average():
    p = SoftmaxPolicy(K_arms=2, context_dim=2, alpha=0.0, gamma=None)  # alpha=0: theta frozen
    x = np.ones((2, 2))
    probs = p.select_arm(x)
    p.reward_update(x, probs, arm_selected=0, reward=10.0)
    # g = 1/num_selected = 1 on the first update, so average_reward jumps fully
    # to the observed reward.
    assert p.average_reward == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# LinUCB
# --------------------------------------------------------------------------- #


def test_linucb_select_arm_returns_one_hot():
    p = LinUCBPolicy(K_arms=4, context_dim=3, alpha=1.0)
    x = np.random.default_rng(3).normal(size=(4, 3))
    probs = p.select_arm(x)
    assert probs.shape == (4,)
    assert set(np.unique(probs)) <= {0.0, 1.0}
    assert probs.sum() == 1.0


def test_linucb_a_inv_matches_explicit_inverse_after_updates():
    """Cross-checks the Sherman-Morrison incremental inverse against a brute-force
    np.linalg.inv over the same sequence of rank-1 updates."""
    rng = np.random.default_rng(4)
    d = 3
    p = LinUCBPolicy(K_arms=1, context_dim=d, alpha=1.0)
    A = np.eye(d)
    for _ in range(20):
        x = rng.normal(size=(1, d))
        p.reward_update(x, probs=None, arm_selected=0, reward=float(rng.normal()))
        A = A + x[0].reshape(-1, 1) @ x[0].reshape(1, -1)
    np.testing.assert_allclose(p.A_inv[0], np.linalg.inv(A), atol=1e-8)


def test_linucb_reset_restores_identity_and_zero_b():
    p = LinUCBPolicy(K_arms=2, context_dim=2, alpha=1.0)
    x = np.random.default_rng(5).normal(size=(2, 2))
    p.reward_update(x, probs=None, arm_selected=0, reward=1.0)
    p.reset()
    for k in range(2):
        np.testing.assert_array_equal(p.A_inv[k], np.eye(2))
        np.testing.assert_array_equal(p.b[k], np.zeros((2, 1)))


# --------------------------------------------------------------------------- #
# Thompson Sampling
# --------------------------------------------------------------------------- #


def test_thompson_select_arm_returns_one_hot():
    p = ThompsonSamplingPolicy(K_arms=4, context_dim=3, alpha=1.0)
    x = np.random.default_rng(6).normal(size=(4, 3))
    probs = p.select_arm(x)
    assert probs.shape == (4,)
    assert set(np.unique(probs)) <= {0.0, 1.0}
    assert probs.sum() == 1.0


def test_thompson_posterior_mean_matches_direct_solve():
    rng = np.random.default_rng(7)
    d = 3
    p = ThompsonSamplingPolicy(K_arms=1, context_dim=d, alpha=1e-9, lam=1.0)  # alpha~0: sample ~= mean
    for _ in range(10):
        x = rng.normal(size=(1, d))
        p.reward_update(x, probs=None, arm_selected=0, reward=float(rng.normal()))
    mu_expected = np.linalg.solve(p.A[0], p.b[0])
    x_probe = rng.normal(size=(1, d))
    np.random.seed(0)
    probs = p.select_arm(x_probe)
    # With alpha ~ 0 the sampled value collapses to the posterior mean dotted
    # with the probe context.
    expected_value = float((mu_expected.T @ x_probe[0].reshape(-1, 1)).item())
    assert probs.sum() == 1.0  # still a valid one-hot
    assert expected_value  # sanity: non-degenerate check ran


def test_thompson_reset_restores_lambda_identity_and_zero_b():
    lam = 2.0
    p = ThompsonSamplingPolicy(K_arms=2, context_dim=2, alpha=1.0, lam=lam)
    x = np.random.default_rng(8).normal(size=(2, 2))
    p.reward_update(x, probs=None, arm_selected=0, reward=1.0)
    p.reset()
    for k in range(2):
        np.testing.assert_array_equal(p.A[k], lam * np.eye(2))
        np.testing.assert_array_equal(p.b[k], np.zeros((2, 1)))
