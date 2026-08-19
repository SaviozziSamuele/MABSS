"""
Strategy performance metrics.

Ported from the `strategy_stats` function nested inside
`MABSS_utility.py`'s `compute_and_highlight_stats` (:1260-1276), with the
annualization and Sharpe-ratio bugs fixed (plan Phase 4, item 1).

The original hardcoded `252` (trading days/year) for both `annualized_return`
and `volatility`, and computed `sharpe_ratio = annualized_return / volatility`
with no risk-free rate term -- i.e. a return-to-volatility ratio mislabeled as a
Sharpe ratio. Measured directly from yfinance data (see the plan and
tests/regression/test_bug_annualization.py): SPY/GC=F/TLT trade ~251-252
bars/year (252 is a fine approximation), but BTC-USD trades ~365 bars/year (7
days/week, no exchange holidays) and EURUSD=X ~260 (5-day FX week, fewer US
holidays observed than equities). Using 252 for all five assets in the paper's
asset universe:
  - understates BTC's annualized return exponent (252/365 < 1 applied where it
    should be 365/365 = 1), which for the corpus's BTC returns *increases* the
    published annualized_return relative to the correct per-asset figure less
    than the volatility distortion below, netting an inflated Sharpe;
  - understates BTC's volatility by a factor of sqrt(365/252) = 1.204x, since
    volatility scales with sqrt(periods_per_year);
  - the two together inflate BTC's Sharpe ratio by roughly that same ~1.20x
    factor, and make the EURUSD figures similarly non-comparable to the other
    three assets, which really are close to 252.

`periods_per_year=252` remains reachable via an explicit argument specifically
so the "as published" numbers stay reproducible for a reviewer-response delta
table (see tools/build_stats_recompute_golden.py, which uses the legacy formula
verbatim, and tests/regression/test_bug_annualization.py, which computes the
delta against the real corpus for both formulas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """
    Estimates the observation frequency of a time series from its DatetimeIndex,
    as (n - 1) observations spread over the index's actual calendar span. This is
    what distinguishes a 5-day equity week (~252/yr) from BTC-USD's 7-day week
    (~365/yr) or EURUSD=X's FX week (~260/yr) without hardcoding any of them.
    """
    if len(index) < 2:
        raise ValueError("infer_periods_per_year needs at least 2 observations")
    idx = pd.DatetimeIndex(index)
    span_years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 24 * 3600)
    if span_years <= 0:
        raise ValueError("infer_periods_per_year: index span is zero or negative")
    return (len(idx) - 1) / span_years


def strategy_stats(
    daily_returns,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Computes total/annualized return, hit ratio, max drawdown, volatility,
    Sharpe, and Calmar for a series of periodic strategy returns.

    `periods_per_year`: if None and `daily_returns` has a DatetimeIndex, it is
    inferred via `infer_periods_per_year`. If None and there's no usable index,
    raises -- this is deliberate: silently assuming 252 is exactly the bug being
    fixed, so there is no implicit default. Pass `periods_per_year=252`
    explicitly to reproduce the as-published (pre-fix) numbers.

    `risk_free_rate`: annualized, subtracted from `annualized_return` before
    dividing by volatility -- this is what makes `sharpe_ratio` an actual Sharpe
    ratio rather than a bare return-to-volatility ratio (the original had no
    risk_free_rate term at all).
    """
    returns = pd.Series(daily_returns)

    if periods_per_year is None:
        if isinstance(returns.index, pd.DatetimeIndex):
            periods_per_year = infer_periods_per_year(returns.index)
        else:
            raise ValueError(
                "strategy_stats: periods_per_year was not given and daily_returns "
                "has no DatetimeIndex to infer it from. Pass periods_per_year "
                "explicitly (252 reproduces the pre-fix, as-published behavior)."
            )

    cumulative_returns = (1 + returns).cumprod()
    stats: dict = {}
    stats["total_return"] = cumulative_returns.iloc[-1] - 1
    stats["annualized_return"] = (stats["total_return"] + 1) ** (periods_per_year / len(returns)) - 1
    stats["hit_ratio"] = (returns > 0).sum() / len(returns)

    roll_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - roll_max) / roll_max
    stats["max_drawdown"] = drawdown.min()

    stats["volatility"] = returns.std() * np.sqrt(periods_per_year)
    excess_return = stats["annualized_return"] - risk_free_rate
    stats["sharpe_ratio"] = excess_return / stats["volatility"] if stats["volatility"] else float("nan")

    if stats["max_drawdown"] != 0:
        stats["calmar_ratio"] = stats["annualized_return"] / abs(stats["max_drawdown"])
    else:
        stats["calmar_ratio"] = float("inf")

    stats["periods_per_year"] = periods_per_year
    return stats


def directional_accuracy(predicted_returns, actual_returns) -> float:
    """
    Fraction of steps where sign(prediction) == sign(actual) -- i.e. did the
    model call the direction correctly. Distinct from `strategy_stats`'
    `hit_ratio`, which is the fraction of periods where the *realized strategy
    return* (after the long/cash filter) was positive -- not the same thing as
    predicting direction correctly (a correct "down" call produces a 0%, not
    negative, strategy return under the long/cash construction in
    mabss.strategies, so hit_ratio and directional_accuracy diverge whenever the
    market goes down and the model correctly predicts it). The original codebase
    conflated these under the single name "hit_ratio"; this function exists so
    the two concepts have separate names going forward.
    """
    pred = np.asarray(predicted_returns)
    actual = np.asarray(actual_returns)
    if pred.shape != actual.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {actual.shape}")
    return float(np.mean(np.sign(pred) == np.sign(actual)))
