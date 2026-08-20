"""Unit tests for mabss.strategies (the deduplicated long/cash construction)."""

import numpy as np
import pandas as pd
import pytest

from mabss.strategies import (
    build_multirun_strategies,
    build_multirun_strategies_costed,
    costed_long_cash_returns,
    ensemble_signal,
    greedy_predictions,
    long_cash_returns,
    nav,
)


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


def _synthetic_multirun_inputs(n_seeds=3, T=20, n_arms=2, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=T)
    data = {"target": rng.normal(0.0, 0.01, T)}
    for a in range(n_arms):
        data[f"model_{a}"] = rng.normal(0.0, 0.01, T)
    pct_change_preds = pd.DataFrame(data, index=idx)
    results_multi = {
        f"seed_{s}": {"best arms greedy": list(rng.integers(0, n_arms, T))} for s in range(n_seeds)
    }
    return results_multi, pct_change_preds


def test_build_multirun_strategies_matches_plot_multi_run_bandit_output():
    """Pins the Step-1 refactor: the extracted pure builder must produce
    exactly the columns/values plot_multi_run_bandit computed inline before
    the extraction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mabss.plots.diagnostics import plot_multi_run_bandit

    results_multi, pct_change_preds = _synthetic_multirun_inputs()

    best_seed_a, strategies_a = build_multirun_strategies(results_multi, pct_change_preds)

    plt.close("all")
    best_seed_b, strategies_b = plot_multi_run_bandit(
        results_multi, pct_change_preds, {"N_ARMS": 2}, save_dir=None
    )
    plt.close("all")

    assert best_seed_a == best_seed_b
    pd.testing.assert_frame_equal(strategies_a, strategies_b)


def test_build_multirun_strategies_returns_none_on_empty_results_multi():
    _, pct_change_preds = _synthetic_multirun_inputs()
    assert build_multirun_strategies({}, pct_change_preds) == (None, None)


def test_build_multirun_strategies_skips_seeds_with_mismatched_horizon():
    """T is inferred from the first seed; any other seed whose arm-sequence
    length disagrees is silently skipped (not an error) rather than
    corrupting the shared strategies frame."""
    _, pct_change_preds = _synthetic_multirun_inputs(T=20)
    results_multi = {
        "seed_0": {"best arms greedy": [0, 1, 0, 1, 0]},  # T=5
        "seed_1": {"best arms greedy": [0, 1]},  # mismatched length vs seed_0
    }
    best_seed, strategies = build_multirun_strategies(results_multi, pct_change_preds)
    assert best_seed == "seed_0"
    assert "returns_greedy_seed_1" not in strategies.columns
    assert len(strategies) == 5




def test_costed_long_cash_matches_long_cash_at_zero_fee():
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 1, 50)
    target = rng.normal(0, 0.01, 50)
    np.testing.assert_array_equal(
        costed_long_cash_returns(signal, target, fee_bps=0.0), long_cash_returns(signal, target)
    )


def test_costed_long_cash_charges_fee_on_position_switch():
    # long, long, cash, cash, long -> switches at index 2 (long->cash) and index 4 (cash->long)
    signal = np.array([0.1, 0.2, -0.1, -0.2, 0.3])
    target = np.full(5, 0.01)
    out = costed_long_cash_returns(signal, target, fee_bps=10.0)  # 10 bps = 0.001
    expected = np.array([0.01, 0.01, 0.0 - 0.001, 0.0, 0.01 - 0.001])
    np.testing.assert_allclose(out, expected)


def test_costed_long_cash_no_fee_when_position_constant():
    signal = np.array([0.1, 0.2, 0.3, 0.4])
    target = np.array([0.01, -0.02, 0.03, -0.04])
    for fee_bps in (0.0, 5.0, 50.0):
        out = costed_long_cash_returns(signal, target, fee_bps=fee_bps)
        np.testing.assert_array_equal(out, target)


def test_costed_long_cash_does_not_mutate_input():
    signal = np.array([0.1, -0.2])
    target = np.array([0.02, 0.03])
    target_copy = target.copy()
    costed_long_cash_returns(signal, target, fee_bps=10.0)
    np.testing.assert_array_equal(target, target_copy)


def test_costed_long_cash_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        costed_long_cash_returns(np.zeros(3), np.zeros(4), fee_bps=1.0)


def test_build_multirun_strategies_costed_matches_uncosted_at_zero_fee():
    results_multi, pct_change_preds = _synthetic_multirun_inputs()
    best_seed_a, strategies_a = build_multirun_strategies(results_multi, pct_change_preds)
    best_seed_b, strategies_b = build_multirun_strategies_costed(results_multi, pct_change_preds, fee_bps=0.0)
    assert best_seed_a == best_seed_b
    pd.testing.assert_frame_equal(strategies_a, strategies_b)


def test_build_multirun_strategies_costed_degrades_returns_at_positive_fee():
    results_multi, pct_change_preds = _synthetic_multirun_inputs()
    _, strategies_free = build_multirun_strategies_costed(results_multi, pct_change_preds, fee_bps=0.0)
    _, strategies_costed = build_multirun_strategies_costed(results_multi, pct_change_preds, fee_bps=25.0)
    greedy_cols = [c for c in strategies_free.columns if c.startswith("returns_greedy_")]
    for col in greedy_cols:
        assert strategies_costed[col].sum() <= strategies_free[col].sum()
