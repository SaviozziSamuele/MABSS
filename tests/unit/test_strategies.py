"""Unit tests for mabss.strategies (the deduplicated long/cash construction)."""

import numpy as np
import pandas as pd
import pytest

from mabss.strategies import ensemble_signal, greedy_predictions, long_cash_returns, nav


def test_long_cash_zeroes_out_negative_signal_periods():
    signal = np.array([0.1, -0.2, 0.05, -0.01, 0.3])
    target = np.array([0.02, 0.03, -0.01, 0.04, -0.02])
    out = long_cash_returns(signal, target)
    expected = np.array([0.02, 0.0, -0.01, 0.0, -0.02])
    np.testing.assert_array_equal(out, expected)


def test_long_cash_does_not_mutate_input():
    signal = np.array([0.1, -0.2])
    target = np.array([0.02, 0.03])
    target_copy = target.copy()
    long_cash_returns(signal, target)
    np.testing.assert_array_equal(target, target_copy)


def test_long_cash_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        long_cash_returns(np.zeros(3), np.zeros(4))


def test_nav_is_cumprod_of_one_plus_returns():
    returns = np.array([0.1, -0.05, 0.02])
    result = nav(returns)
    expected = np.cumprod(1 + returns)
    np.testing.assert_allclose(result.values, expected)
    assert result.iloc[0] == pytest.approx(1.1)


def test_ensemble_signal_prefers_mean_column():
    df = pd.DataFrame({"target": [0, 0], "model_0": [0.1, 0.2], "model_1": [0.3, 0.4], "mean": [9.0, 9.0]})
    out = ensemble_signal(df)
    np.testing.assert_array_equal(out, [9.0, 9.0])


def test_ensemble_signal_falls_back_to_model_column_mean():
    df = pd.DataFrame({"target": [0, 0], "model_0": [0.1, 0.3], "model_1": [0.3, 0.5]})
    out = ensemble_signal(df)
    np.testing.assert_allclose(out, [0.2, 0.4])


def test_greedy_predictions_uses_column_names_not_position():
    """Regression check: the original indexed with `arm + 1` positionally,
    assuming column 0 is always 'target'. Using named lookup means a reordered
    frame (e.g. 'mean' inserted before the model columns) doesn't silently read
    the wrong model's prediction."""
    # Deliberately out-of-order columns relative to what a positional `arm+1`
    # lookup would assume.
    df = pd.DataFrame(
        {"mean": [9, 9], "target": [0, 0], "model_1": [0.5, 0.6], "model_0": [0.1, 0.2]}
    )
    model_cols = ["model_0", "model_1"]
    arms = [1, 0]  # pick model_1 at t=0, model_0 at t=1
    out = greedy_predictions(df, arms, model_cols)
    assert out == [0.5, 0.2]
