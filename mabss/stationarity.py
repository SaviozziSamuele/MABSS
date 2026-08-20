"""
Stationarity and structural-break diagnostics -- built for Reviewer #3's
Comment 1 ("formal unit-root, stationarity, or structural-break tests...
Tests such as ADF, KPSS, Sup-F, or Bai-Perron would help"). Uses only
already-pinned packages (numpy/scipy/statsmodels) -- no `ruptures`, per the
session's dependency-hygiene decision.

- `adf_test`/`kpss_test`/`stationarity_battery`: augmented Dickey-Fuller and
  KPSS unit-root tests via `statsmodels.tsa.stattools`, classified via the
  standard 4-way ADF+KPSS combination.
- `cusum_structural_break_test`: Ploberger & Kramer (1992) CUSUM-of-OLS-
  residuals test via `statsmodels.stats.diagnostic.breaks_cusumolsresid` --
  validated against R's `strucchange` per that function's own docstring,
  with an asymptotic p-value already computed. Answers "is there a break."
- `quandt_andrews_breakpoint`: a hand-rolled Sup-F (Quandt 1960; Andrews
  1993) search for breakpoint DATING, with a parametric-bootstrap p-value
  (no tabulated Andrews-1993 critical values ship with any already-pinned
  package). Answers "roughly when."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series, regression: str = "c", autolag: str = "AIC") -> dict:
    """Augmented Dickey-Fuller test. H0: unit root (non-stationary).
    `stationary_at_5pct`: True if p_value < 0.05 (H0 rejected)."""
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    adf_stat, p_value, used_lag, n_obs, critical_values, _icbest = adfuller(
        values, regression=regression, autolag=autolag
    )
    return {
        "adf_stat": float(adf_stat),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {k: float(v) for k, v in critical_values.items()},
        "stationary_at_5pct": bool(p_value < 0.05),
    }


def kpss_test(series, regression: str = "c", nlags="auto") -> dict:
    """KPSS test. H0: stationary -- the OPPOSITE null from ADF.
    `stationary_at_5pct`: True if p_value > 0.05 (H0 NOT rejected).
    statsmodels clips KPSS's p-value to its tabulated [0.01, 0.10] range;
    `p_value_clamped` flags when that clipping occurred."""
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kpss_stat, p_value, n_lags, critical_values = kpss(values, regression=regression, nlags=nlags)
        clamped = any("p-value" in str(w.message).lower() for w in caught)
    return {
        "kpss_stat": float(kpss_stat),
        "p_value": float(p_value),
        "n_lags": int(n_lags),
        "critical_values": {k: float(v) for k, v in critical_values.items()},
        "stationary_at_5pct": bool(p_value > 0.05),
        "p_value_clamped": clamped,
    }


def stationarity_battery(series, series_name: str | None = None) -> dict:
    """Runs adf_test + kpss_test and classifies via the standard 4-way
    combination table (Enders' textbook convention):
      - 'stationary': ADF rejects unit root AND KPSS fails to reject stationarity
      - 'unit_root': ADF fails to reject unit root AND KPSS rejects stationarity
      - 'trend_stationary_or_fractionally_integrated': both reject (KPSS's null
        of level-stationarity is rejected even though ADF also rejects a unit
        root -- consistent with e.g. trend-stationarity or long-memory)
      - 'inconclusive': neither test rejects its own null
    """
    adf = adf_test(series)
    kps = kpss_test(series)
    adf_rejects = adf["stationary_at_5pct"]
    kpss_rejects = not kps["stationary_at_5pct"]

    if adf_rejects and not kpss_rejects:
        classification = "stationary"
    elif not adf_rejects and kpss_rejects:
        classification = "unit_root"
    elif adf_rejects and kpss_rejects:
        classification = "trend_stationary_or_fractionally_integrated"
    else:
        classification = "inconclusive"

    return {"series_name": series_name, "adf": adf, "kpss": kps, "classification": classification}


def cusum_structural_break_test(series) -> dict:
    """Ploberger & Kramer (1992) CUSUM-of-OLS-residuals test for parameter
    (mean) stability: fits `y_t = alpha + eps_t` via OLS, tests whether the
    cumulative sum of standardized residuals stays within its asymptotic
    Brownian-bridge bounds. Run on a RETURN series, not price levels -- price
    levels are non-stationary by construction, so a mean-stability test on
    them isn't a meaningful structural-break question the way it is for
    returns."""
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    exog = np.ones((len(values), 1))
    resid = OLS(values, exog).fit().resid
    sup_b, p_value, crit = breaks_cusumolsresid(resid, ddof=1)
    return {
        "sup_b": float(sup_b),
        "p_value": float(p_value),
        "critical_values": {str(c[0]): float(c[1]) for c in crit},
        "reject_at_5pct": bool(p_value < 0.05),
    }


def _chow_sup_f_all(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Chow F-statistic for a mean break at every `tau` in `candidates`
    simultaneously, via cumulative sums -- O(T) total instead of O(T) per
    candidate (O(T^2) overall). Real asset series run T~5500 bars and this is
    evaluated once per bootstrap replicate (default n_boot=2000), so the naive
    per-candidate re-summation (originally used here) would take hours per
    asset; this vectorized form takes seconds.

    For split at tau: sum/sum-of-squares of values[:tau] and values[tau:] are
    read directly off prefix-sum arrays in O(1) per tau, so
    SSR_before + SSR_after (and hence the F-statistic) is computed for the
    whole `candidates` array with vectorized numpy ops, no Python-level loop
    over tau.
    """
    n = len(values)
    resid_full = values - values.mean()
    ssr_full = float(np.sum(resid_full**2))

    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    cumsq = np.concatenate([[0.0], np.cumsum(values**2)])
    total_sum, total_sq = cumsum[-1], cumsq[-1]

    tau = candidates
    n_before = tau
    n_after = n - tau
    sum_before = cumsum[tau]
    sq_before = cumsq[tau]
    sum_after = total_sum - sum_before
    sq_after = total_sq - sq_before

    # sum((x-mean)^2) = sum(x^2) - n*mean^2, with mean = sum/n.
    ssr_before = sq_before - (sum_before**2) / n_before
    ssr_after = sq_after - (sum_after**2) / n_after
    ssr_split = ssr_before + ssr_after

    k = 1  # one parameter (the mean) per regime
    df1 = k
    df2 = n - 2 * k
    safe_ssr_split = np.where(ssr_split > 0, ssr_split, np.nan)
    f_stats = ((ssr_full - ssr_split) / df1) / (safe_ssr_split / df2)
    return np.nan_to_num(f_stats, nan=0.0)


def quandt_andrews_breakpoint(
    series,
    trim: float = 0.15,
    n_boot: int = 2000,
    seed: int | None = 0,
) -> dict:
    """
    Quandt (1960) / Andrews (1993) Sup-F search for a single unknown
    breakpoint's LOCATION: searches candidate split points `tau` over the
    trimmed range `[trim*T, (1-trim)*T]` (Andrews' standard 15% trim), fits
    separate before/after constant-only means at each `tau`, takes the Chow
    F-statistic, and reports `argmax(F)` as the estimated break.

    p-value via parametric bootstrap under the no-break null: resamples the
    full-sample OLS residuals (with replacement) `n_boot` times, rebuilds a
    synthetic no-break series (sample mean + resampled residuals), reruns the
    same Sup-F search on each synthetic series, and reports the fraction of
    bootstrap Sup-F values >= the observed Sup-F. This substitutes for the
    tabulated Andrews (1993) critical values, which no already-pinned package
    ships.
    """
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    lo = max(1, int(np.floor(trim * n)))
    hi = min(n - 1, int(np.ceil((1 - trim) * n)))
    if lo >= hi:
        raise ValueError(f"quandt_andrews_breakpoint: trim={trim} leaves no candidate breakpoints for n={n}")
    candidates = np.arange(lo, hi)

    f_stats = _chow_sup_f_all(values, candidates)
    best_idx = int(np.argmax(f_stats))
    observed_sup_f = float(f_stats[best_idx])
    breakpoint_index = int(candidates[best_idx])

    resid = values - values.mean()
    rng = np.random.default_rng(seed)
    boot_sup_f = np.empty(n_boot)
    for b in range(n_boot):
        synthetic = values.mean() + rng.choice(resid, size=n, replace=True)
        boot_sup_f[b] = _chow_sup_f_all(synthetic, candidates).max()
    p_value = float(np.mean(boot_sup_f >= observed_sup_f))

    breakpoint_date = None
    if isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex):
        clean_index = series.dropna().index
        if breakpoint_index < len(clean_index):
            breakpoint_date = str(clean_index[breakpoint_index])

    return {
        "breakpoint_index": breakpoint_index,
        "breakpoint_date": breakpoint_date,
        "sup_f": observed_sup_f,
        "p_value_bootstrap": p_value,
        "n_boot": n_boot,
        "trim": trim,
    }
