"""
Regression tests for the hardcoded-252-trading-days bug
(originally MABSS_utility.py:1265,1270, `strategy_stats`).

The asset universe (Manuscript.tex:288) includes BTC-USD (trades 7 days/week,
~365 bars/year) and EURUSD=X (~260 bars/year), not just 252-bar equities/
commodities/bonds. Hardcoding 252 for annualization and volatility distorts
BTC's Sharpe ratio and makes it non-comparable to the other four assets.
"""

import numpy as np
import pandas as pd
import pytest

from mabss.metrics import infer_periods_per_year, strategy_stats


def test_infer_periods_per_year_equity_like():
    idx = pd.bdate_range("2015-01-01", periods=252 * 4)  # ~4 business years
    ppy = infer_periods_per_year(idx)
    # bdate_range excludes weekends only (not holidays), so this lands near
    # 365.25 * 5/7 ~= 261, not the holiday-adjusted ~252 a real trading
    # calendar has -- the real-corpus test below checks actual trading data.
    assert 255 < ppy < 265


def test_infer_periods_per_year_seven_day_calendar():
    idx = pd.date_range("2015-01-01", periods=365 * 4, freq="D")  # every calendar day
    ppy = infer_periods_per_year(idx)
    assert 360 < ppy < 370


def test_legacy_252_still_reachable_explicitly():
    """Reproduces the exact pre-fix formula when periods_per_year=252 is passed
    explicitly -- this is what tools/build_stats_recompute_golden.py relies on
    to lock in the as-published numbers."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, size=500))
    stats = strategy_stats(returns, periods_per_year=252, risk_free_rate=0.0)
    cumulative = (1 + returns).cumprod()
    expected_total = cumulative.iloc[-1] - 1
    expected_ann = (expected_total + 1) ** (252 / len(returns)) - 1
    expected_vol = returns.std() * np.sqrt(252)
    assert stats["total_return"] == pytest.approx(expected_total)
    assert stats["annualized_return"] == pytest.approx(expected_ann)
    assert stats["volatility"] == pytest.approx(expected_vol)
    assert stats["sharpe_ratio"] == pytest.approx(expected_ann / expected_vol)


def test_no_periods_per_year_and_no_datetime_index_raises():
    """The fix's core behavior change: silently assuming 252 is exactly the bug.
    Without a DatetimeIndex and without an explicit periods_per_year, raise."""
    plain_returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, size=100))
    with pytest.raises(ValueError, match="periods_per_year was not given"):
        strategy_stats(plain_returns)


def test_risk_free_rate_reduces_sharpe():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.01, size=500))
    stats_no_rf = strategy_stats(returns, periods_per_year=252, risk_free_rate=0.0)
    stats_with_rf = strategy_stats(returns, periods_per_year=252, risk_free_rate=0.03)
    assert stats_with_rf["sharpe_ratio"] < stats_no_rf["sharpe_ratio"]


@pytest.mark.needs_artifacts
def test_btc_bars_per_year_measured_from_real_corpus_is_not_252(artifact_root):
    """Directly measures the BTC-USD, SPY, and EURUSD=X bars/year from the
    cached pctchange_preds.csv files, confirming the plan's forensic finding at
    full precision: BTC ~365/yr, EURUSD ~260/yr, SPY ~252/yr."""
    matches = {"SPY": None, "BTC-USD": None, "EURUSD=X": None}
    for ticker in list(matches):
        hits = sorted(artifact_root.glob(f"*TICKER-{ticker}_*/pctchange_preds.csv"))
        if not hits:
            pytest.skip(f"no pctchange_preds.csv found for {ticker} under {artifact_root}")
        matches[ticker] = hits[0]

    measured = {}
    for ticker, path in matches.items():
        df = pd.read_csv(path, index_col=0)
        idx = pd.to_datetime(df.index, utc=True)
        measured[ticker] = infer_periods_per_year(idx)

    assert 245 < measured["SPY"] < 258
    assert 355 < measured["BTC-USD"] < 370
    assert 253 < measured["EURUSD=X"] < 268
    # The core point: BTC is NOT close to 252 -- this is the bug's blast radius.
    assert abs(measured["BTC-USD"] - 252) > 100
