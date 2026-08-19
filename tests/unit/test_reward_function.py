"""Unit tests for mabss.rewards.reward_function."""

import numpy as np
import pytest

from mabss.rewards import reward_function


def test_l1_matches_negative_abs_error_2d():
    pred = np.array([[0.1, -0.2], [0.3, 0.0]])
    targ = np.array([0.0, 0.1])
    out = reward_function(pred, targ, metric="l1")
    expected = -np.abs(pred - targ[:, None])
    np.testing.assert_allclose(out, expected)


def test_l2_matches_negative_squared_error():
    pred = np.array([[0.1, -0.2]])
    targ = np.array([0.05])
    out = reward_function(pred, targ, metric="l2")
    expected = -np.square(pred - targ[:, None])
    np.testing.assert_allclose(out, expected)


def test_crossentropy_no_inf_on_extreme_predictions():
    pred = np.array([[1e6, -1e6]])
    targ = np.array([0.01])
    with np.errstate(over="ignore"):
        out = reward_function(pred, targ, metric="crossentropy")
    assert np.all(np.isfinite(out))


def test_crossentropy_rewards_correct_direction_more():
    # target is positive (up move); a strongly positive prediction should be
    # rewarded more (less negative) than a strongly negative one.
    pred = np.array([[5.0, -5.0]])
    targ = np.array([1.0])
    out = reward_function(pred, targ, metric="crossentropy")
    assert out[0, 0] > out[0, 1]


def test_invalid_metric_raises():
    with pytest.raises(ValueError, match="Invalid metric"):
        reward_function(np.array([[0.1]]), np.array([0.0]), metric="not_a_metric")


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Incompatible first dimensions"):
        reward_function(np.zeros((5, 3)), np.zeros(4), metric="l1")


def test_1d_predictions_become_single_arm_column():
    # A (N,) prediction vector is promoted to (N, 1) -- a single arm, not N arms.
    pred = np.array([0.1, -0.1, 0.2])
    targ = np.array([0.0, 0.0, 0.0])
    out = reward_function(pred, targ, metric="l1")
    assert out.shape == (3, 1)


def test_scalar_inputs_are_promoted_to_1x1():
    # Documents current behavior (always 2-D output, even for scalar input) --
    # see mabss.env.WalkForwardEnv.calc_reward for where callers squeeze this
    # back to a plain float.
    out = reward_function(0.05, 0.02, metric="l1")
    assert out.shape == (1, 1)
