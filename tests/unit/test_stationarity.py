"""Unit tests for mabss.stationarity."""

import numpy as np
import pandas as pd
import pytest

from mabss.stationarity import (
    adf_test,
    cusum_structural_break_test,
    kpss_test,
    quandt_andrews_breakpoint,
    stationarity_battery,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_random_walk_classified_as_unit_root(rng):
    rw = np.cumsum(rng.normal(0, 1, 1000))
    result = stationarity_battery(rw, series_name="random_walk")
    assert result["classification"] == "unit_root"
    assert not result["adf"]["stationary_at_5pct"]
    assert not result["kpss"]["stationary_at_5pct"]


def test_white_noise_classified_as_stationary(rng):
    wn = rng.normal(0, 1, 1000)
    result = stationarity_battery(wn, series_name="white_noise")
    assert result["classification"] == "stationary"
    assert result["adf"]["stationary_at_5pct"]
    assert result["kpss"]["stationary_at_5pct"]


def test_adf_test_returns_expected_keys(rng):
    out = adf_test(rng.normal(0, 1, 500))
    assert set(out.keys()) == {
        "adf_stat", "p_value", "used_lag", "n_obs", "critical_values", "stationary_at_5pct"
    }
    assert "5%" in out["critical_values"]


def test_kpss_test_returns_expected_keys(rng):
    out = kpss_test(rng.normal(0, 1, 500))
    assert set(out.keys()) == {
        "kpss_stat", "p_value", "n_lags", "critical_values", "stationary_at_5pct", "p_value_clamped"
    }


def test_cusum_no_reject_on_constant_mean_series(rng):
    series = rng.normal(0, 1, 500)
    result = cusum_structural_break_test(series)
    assert not result["reject_at_5pct"]


def test_cusum_rejects_on_injected_mean_shift(rng):
    series = np.concatenate([rng.normal(0, 1, 250), rng.normal(5, 1, 250)])
    result = cusum_structural_break_test(series)
    assert result["reject_at_5pct"]


def test_quandt_andrews_recovers_injected_breakpoint_within_tolerance(rng):
    n = 1000
    true_break = 500
    series = np.concatenate([rng.normal(0, 1, true_break), rng.normal(4, 1, n - true_break)])
    result = quandt_andrews_breakpoint(series, n_boot=500, seed=1)
    tolerance = int(0.05 * n)
    assert abs(result["breakpoint_index"] - true_break) <= tolerance
    assert result["p_value_bootstrap"] < 0.05


def test_quandt_andrews_not_significant_on_no_break_series(rng):
    series = rng.normal(0, 1, 1000)
    result = quandt_andrews_breakpoint(series, n_boot=500, seed=1)
    assert result["p_value_bootstrap"] > 0.05


def test_quandt_andrews_reports_breakpoint_date_for_datetime_series(rng):
    idx = pd.bdate_range("2020-01-01", periods=1000)
    values = np.concatenate([rng.normal(0, 1, 500), rng.normal(4, 1, 500)])
    series = pd.Series(values, index=idx)
    result = quandt_andrews_breakpoint(series, n_boot=200, seed=1)
    assert result["breakpoint_date"] is not None
    pd.Timestamp(result["breakpoint_date"])  # must parse as a real date


def test_quandt_andrews_raises_when_series_too_short_for_any_candidate():
    with pytest.raises(ValueError):
        quandt_andrews_breakpoint(np.zeros(2), n_boot=10)


def test_quandt_andrews_runtime_scales_subquadratically_with_length(rng):
    """Regression guard for the O(T^2) -> O(T) vectorization: n=5000 must not
    take dramatically longer per-candidate than n=500 (the naive per-tau
    re-summation loop this replaced would take orders of magnitude longer at
    real-corpus scale, ~5500 bars)."""
    import time

    small = rng.normal(0, 1, 500)
    t0 = time.time()
    quandt_andrews_breakpoint(small, n_boot=200, seed=2)
    small_elapsed = time.time() - t0

    large = rng.normal(0, 1, 5000)
    t0 = time.time()
    quandt_andrews_breakpoint(large, n_boot=200, seed=2)
    large_elapsed = time.time() - t0

    # O(T) scaling predicts ~10x; allow generous headroom but catch an
    # accidental regression back to O(T^2) (~100x), which would show as a
    # ratio far above 10x here.
    assert large_elapsed < max(30 * small_elapsed, 5.0)
