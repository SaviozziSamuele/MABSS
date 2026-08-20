"""
Paired significance testing for CMAB-greedy vs. static-ensemble daily
returns -- built for Reviewer #2's "a paired Diebold-Mariano test or
bootstrap confidence intervals" ask and Reviewer #3's "paired statistical
tests with appropriate multiple-testing corrections" ask.

- `dm_test`/`dm_test_returns`: Diebold & Mariano (1995) paired test, adapted
  to the daily strategy-return differential (rather than raw forecast-error
  loss -- the standard adaptation for comparing trading P&L streams), with
  the Harvey-Leybourne-Newbold (1997) small-sample correction and a
  Newey-West/Bartlett-kernel HAC long-run variance estimate.
- `block_bootstrap_paired_diff`/`sharpe_diff_bootstrap_ci`: circular
  moving-block bootstrap (Ledoit & Wolf 2008-style) for a Sharpe-ratio-
  difference confidence interval.
- `apply_multiple_testing_correction`: Holm (family-wise) and
  Benjamini-Hochberg (FDR) corrections via `statsmodels.stats.multitest`.

Design decision (see plan file / CLAUDE.md): run ONE primary test per config
on the cross-seed-MEAN daily return vs. the deterministic baseline -- the
same aggregation Table 1's "greedy (mean)" column already uses -- rather than
one test per (config, seed). The per-seed return columns for a given config
share the identical market realization and the identical deterministic
baseline, so they are not independent replicates; feeding all of them into
one multiple-testing correction would overstate the effective number of
comparisons. Per-seed statistics remain useful as a robustness appendix
(see `tools/build_significance_tests.py`), just not as correction inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _newey_west_long_run_variance(d: np.ndarray, nw_lags: int) -> float:
    """Bartlett-kernel HAC long-run variance of a (mean-centered) series."""
    d_centered = d - d.mean()
    gamma_0 = float(np.mean(d_centered**2))
    lrv = gamma_0
    for k in range(1, nw_lags + 1):
        gamma_k = float(np.mean(d_centered[k:] * d_centered[:-k]))
        weight = 1.0 - k / (nw_lags + 1)
        lrv += 2 * weight * gamma_k
    return lrv


def dm_test(
    loss_diff,
    h: int = 1,
    nw_lags: int | None = None,
    small_sample_correction: bool = True,
) -> dict:
    """
    Diebold-Mariano-style paired test of H0: E[loss_diff_t] = 0.

    `loss_diff`: e.g. `returns_cmab - returns_baseline` -- a positive
    `mean_diff`/`dm_stat` means the first series (e.g. CMAB) outperforms.

    `nw_lags=None` uses the Newey-West (1994) automatic bandwidth,
    `max(h-1, floor(4*(n/100)**(2/9)))` (floored at the classic DM
    requirement of `h-1` lags for an `h`-step-ahead comparison; `h=1` for a
    same-day return differential, the default here).

    `small_sample_correction=True` applies the Harvey-Leybourne-Newbold
    (1997) factor `sqrt((n+1-2h+h(h-1)/n)/n)` and uses a Student-t(n-1)
    reference distribution instead of Normal(0,1) -- recommended for the
    sample sizes here (hundreds to a few thousand daily observations, not
    asymptotic).

    Degenerate (near-zero variance, e.g. `loss_diff` is exactly constant)
    input returns `dm_stat=0.0, p_value=1.0, degenerate_variance=True`
    instead of raising or returning inf/nan.
    """
    d = np.asarray(loss_diff, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        raise ValueError(f"dm_test needs at least 2 observations, got {n}")

    if nw_lags is None:
        nw_lags = max(h - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    nw_lags = min(nw_lags, n - 1)

    mean_diff = float(d.mean())
    lrv = _newey_west_long_run_variance(d, nw_lags)

    if not np.isfinite(lrv) or lrv <= 1e-300:
        return {
            "dm_stat": 0.0,
            "p_value": 1.0,
            "mean_diff": mean_diff,
            "n": n,
            "nw_lags_used": nw_lags,
            "degenerate_variance": True,
        }

    dm_stat = mean_diff / np.sqrt(lrv / n)

    if small_sample_correction:
        correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm_stat *= correction
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_diff": mean_diff,
        "n": n,
        "nw_lags_used": nw_lags,
        "degenerate_variance": False,
    }


def dm_test_returns(returns_cmab: pd.Series, returns_baseline: pd.Series, **kwargs) -> dict:
    """Aligns two same-index Series (inner join, drops any non-overlapping
    dates), then runs `dm_test` on `returns_cmab - returns_baseline`."""
    aligned = pd.DataFrame({"cmab": returns_cmab, "baseline": returns_baseline}).dropna()
    if len(aligned) < 2:
        raise ValueError("dm_test_returns: fewer than 2 overlapping non-NaN observations")
    diff = (aligned["cmab"] - aligned["baseline"]).values
    return dm_test(diff, **kwargs)


def block_bootstrap_paired_diff(
    a,
    b,
    statistic_fn,
    n_boot: int = 10000,
    block_size: int | None = None,
    ci: float = 0.95,
    seed: int | None = 0,
) -> dict:
    """
    Circular moving-block bootstrap: the SAME block-start indices are applied
    to both `a` and `b` on every replicate (preserving same-day pairing and
    each series' own serial dependence), `statistic_fn(a_resampled,
    b_resampled)` is evaluated on each replicate, and a percentile CI is
    formed from the replicate distribution.

    `block_size=None` defaults to `round(n ** (1/3))` (a standard rule of
    thumb for block bootstraps).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a {a.shape} vs b {b.shape}")
    n = len(a)
    if n < 2:
        raise ValueError(f"block_bootstrap_paired_diff needs at least 2 observations, got {n}")

    if block_size is None:
        block_size = max(1, round(n ** (1 / 3)))
    rng = np.random.default_rng(seed)

    estimate = statistic_fn(a, b)

    n_blocks = int(np.ceil(n / block_size))
    replicates = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        replicates[i] = statistic_fn(a[idx], b[idx])

    alpha = 1 - ci
    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estimate": float(estimate),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "block_size": block_size,
    }


def sharpe_diff_bootstrap_ci(
    returns_cmab: pd.Series,
    returns_baseline: pd.Series,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
    n_boot: int = 10000,
    block_size: int | None = None,
    ci: float = 0.95,
    seed: int | None = 0,
) -> dict:
    """
    Block-bootstrap CI for `Sharpe(returns_cmab) - Sharpe(returns_baseline)`.

    `periods_per_year` is resolved ONCE from `returns_cmab`'s real
    DatetimeIndex before any resampling happens -- a bootstrap replicate is a
    plain array with no meaningful date spacing, so `mabss.metrics.strategy_stats`
    cannot (and must not try to) re-infer it per replicate.
    """
    from mabss.metrics import infer_periods_per_year, strategy_stats

    aligned = pd.DataFrame({"cmab": returns_cmab, "baseline": returns_baseline}).dropna()
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(aligned.index)

    def statistic_fn(a, b):
        sharpe_a = strategy_stats(a, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)["sharpe_ratio"]
        sharpe_b = strategy_stats(b, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)["sharpe_ratio"]
        return sharpe_a - sharpe_b

    return block_bootstrap_paired_diff(
        aligned["cmab"].values,
        aligned["baseline"].values,
        statistic_fn,
        n_boot=n_boot,
        block_size=block_size,
        ci=ci,
        seed=seed,
    )


def apply_multiple_testing_correction(
    p_values,
    methods: tuple[str, ...] = ("holm", "fdr_bh"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Thin wrapper around `statsmodels.stats.multitest.multipletests`, one call
    per method. `p_values`: dict or Series keyed by config identifier.
    Returns a DataFrame indexed the same way, with columns `['p_value']` plus
    `['{method}_reject', '{method}_pvalue_corrected']` per method.
    """
    if isinstance(p_values, dict):
        p_values = pd.Series(p_values)

    result = pd.DataFrame({"p_value": p_values.values}, index=p_values.index)
    for method in methods:
        reject, pvals_corrected, _, _ = multipletests(p_values.values, alpha=alpha, method=method)
        result[f"{method}_reject"] = reject
        result[f"{method}_pvalue_corrected"] = pvals_corrected
    return result
