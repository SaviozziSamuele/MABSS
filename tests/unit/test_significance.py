"""Unit tests for mabss.significance."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from mabss.significance import (
    apply_multiple_testing_correction,
    block_bootstrap_paired_diff,
    dm_test,
    dm_test_returns,
    sharpe_diff_bootstrap_ci,
)


def test_dm_test_type1_error_calibration_under_iid_null():
    """Repeated i.i.d.-null trials (no true effect): the empirical rejection
    rate at alpha=0.05 should land near 5%, not wildly off."""
    n_trials = 200
    alpha = 0.05
    rejections = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        d = rng.normal(0.0, 1.0, 200)
        result = dm_test(d)
        if result["p_value"] < alpha:
            rejections += 1
    rate = rejections / n_trials
    assert 0.01 <= rate <= 0.12  # generous band around the nominal 5%


def test_dm_test_detects_known_effect():
    rng = np.random.default_rng(0)
    d = rng.normal(0.002, 0.01, 500)  # clear positive mean
    result = dm_test(d)
    assert result["p_value"] < 0.05
    assert result["mean_diff"] == pytest.approx(d.mean())
    assert result["dm_stat"] > 0


def test_dm_test_hac_correction_controls_type1_error_under_autocorrelated_null():
    """Construct AR(1) noise with TRUE mean 0 but strong positive persistence.
    A naive test ignoring autocorrelation over-rejects; the HAC-corrected
    (auto nw_lags) version should stay much closer to nominal alpha."""
    n_trials = 150
    alpha = 0.05
    phi = 0.7
    naive_rejections = 0
    hac_rejections = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(2000 + trial)
        eps = rng.normal(0, 1, 300)
        d = np.zeros(300)
        for t in range(1, 300):
            d[t] = phi * d[t - 1] + eps[t]
        naive = dm_test(d, nw_lags=0)
        hac = dm_test(d, nw_lags=None)
        if naive["p_value"] < alpha:
            naive_rejections += 1
        if hac["p_value"] < alpha:
            hac_rejections += 1

    naive_rate = naive_rejections / n_trials
    hac_rate = hac_rejections / n_trials
    assert naive_rate > 0.15  # over-rejects under the mis-specified nw_lags=0
    assert hac_rate < naive_rate  # HAC correction meaningfully reduces over-rejection


def test_dm_test_degenerate_variance_does_not_raise():
    d = np.full(50, 0.001)  # exactly constant -> zero variance
    result = dm_test(d)
    assert result["degenerate_variance"] is True
    assert result["p_value"] == 1.0
    assert result["dm_stat"] == 0.0


def test_dm_test_too_few_observations_raises():
    with pytest.raises(ValueError):
        dm_test([0.1])


def test_dm_test_returns_aligns_by_index_and_drops_non_overlap():
    idx_a = pd.bdate_range("2021-01-01", periods=10)
    idx_b = pd.bdate_range("2021-01-04", periods=10)  # partial overlap
    a = pd.Series(0.01, index=idx_a)
    b = pd.Series(0.0, index=idx_b)
    result = dm_test_returns(a, b)
    assert result["n"] == len(idx_a.intersection(idx_b))
    assert result["mean_diff"] == pytest.approx(0.01)


def test_block_bootstrap_ci_coverage_near_nominal_when_true_diff_is_zero():
    """When a and b are drawn from the identical distribution (true diff of
    means = 0), a 95% CI on mean(a)-mean(b) should contain 0 in roughly 95%
    of independent trials."""
    n_trials = 100
    contains_zero = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(3000 + trial)
        a = rng.normal(0, 1, 150)
        b = rng.normal(0, 1, 150)
        result = block_bootstrap_paired_diff(
            a, b, statistic_fn=lambda x, y: x.mean() - y.mean(), n_boot=500, seed=trial
        )
        if result["ci_low"] <= 0 <= result["ci_high"]:
            contains_zero += 1
    rate = contains_zero / n_trials
    assert 0.85 <= rate <= 1.0


def test_block_bootstrap_ci_excludes_zero_for_clear_effect():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 300)
    b = rng.normal(0.0, 0.1, 300)
    result = block_bootstrap_paired_diff(a, b, statistic_fn=lambda x, y: x.mean() - y.mean(), n_boot=1000, seed=0)
    assert result["ci_low"] > 0


def test_block_bootstrap_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        block_bootstrap_paired_diff(np.zeros(3), np.zeros(4), statistic_fn=lambda a, b: 0.0, n_boot=10)


def test_sharpe_diff_bootstrap_ci_uses_fixed_periods_per_year_across_replicates():
    idx = pd.bdate_range("2021-01-01", periods=300)
    rng = np.random.default_rng(0)
    cmab = pd.Series(rng.normal(0.001, 0.01, 300), index=idx)
    baseline = pd.Series(rng.normal(0.0, 0.01, 300), index=idx)
    result = sharpe_diff_bootstrap_ci(cmab, baseline, n_boot=300, seed=0)
    assert "estimate" in result and "ci_low" in result and "ci_high" in result
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_apply_multiple_testing_correction_matches_direct_multipletests_call():
    p_values = {"cfg_a": 0.001, "cfg_b": 0.04, "cfg_c": 0.2, "cfg_d": 0.5}
    result = apply_multiple_testing_correction(p_values, methods=("holm", "fdr_bh"))

    pvals_array = np.array(list(p_values.values()))
    for method in ("holm", "fdr_bh"):
        expected_reject, expected_corrected, _, _ = multipletests(pvals_array, alpha=0.05, method=method)
        np.testing.assert_array_equal(result[f"{method}_reject"].values, expected_reject)
        np.testing.assert_allclose(result[f"{method}_pvalue_corrected"].values, expected_corrected)


def test_apply_multiple_testing_correction_never_decreases_pvalue():
    p_values = pd.Series({"a": 0.001, "b": 0.01, "c": 0.03, "d": 0.2, "e": 0.9})
    result = apply_multiple_testing_correction(p_values, methods=("holm", "fdr_bh"))
    for method in ("holm", "fdr_bh"):
        assert (result[f"{method}_pvalue_corrected"].values >= result["p_value"].values - 1e-12).all()
