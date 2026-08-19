"""
Signal -> strategy-return -> NAV construction.

The original codebase repeated this exact pattern nine times across
MABSS_utility.py (lines 956, 972, 1099, 1144, 1173, 1226, 1448, 1574, 1644):

    strat_ret = target_slice.copy()
    strat_ret[signal < 0] = 0.0

i.e. "go long the target's realized return when the predicted signal is
positive, otherwise sit in cash (0% return)". This is the single piece of logic
that turns every model/ensemble/bandit prediction into every published return
number in the paper -- worth having exactly once, tested once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def long_cash_returns(signal, target_returns) -> np.ndarray:
    """
    Long/cash strategy: realize `target_returns[t]` whenever `signal[t] >= 0`
    (predicted a non-negative move), otherwise 0% (flat/cash) that period.

    No lookahead: `signal[t]` must already be a same-period-t forecast made
    using only information available before period t closed (this function
    does not itself enforce that -- it's a property of what's passed in from
    walk-forward, out-of-sample predictions).
    """
    signal = np.asarray(signal)
    target_returns = np.asarray(target_returns, dtype=float).copy()
    if signal.shape != target_returns.shape:
        raise ValueError(f"shape mismatch: signal {signal.shape} vs target_returns {target_returns.shape}")
    target_returns[signal < 0] = 0.0
    return target_returns


def nav(strategy_returns, index=None) -> pd.Series:
    """Cumulative NAV (starting at 1.0) from a series of periodic returns."""
    values = np.cumprod(1 + np.asarray(strategy_returns))
    return pd.Series(values, index=index)


def ensemble_signal(data_slice: pd.DataFrame, model_cols: list[str] | None = None) -> np.ndarray:
    """
    The "average ensemble" signal: the 'mean' column if present, otherwise the
    row-wise mean of the model_* columns. Matches `plot_multi_run_bandit`'s
    fallback (:1169-1172) exactly, but as a whitelist over `model_cols` rather
    than "everything except the first column" when 'mean' is absent -- slightly
    more defensive against a stray non-model column surviving a CSV round-trip.
    """
    if "mean" in data_slice.columns:
        return data_slice["mean"].values
    if model_cols is None:
        model_cols = [c for c in data_slice.columns if c.startswith("model_")]
    return data_slice[model_cols].mean(axis=1).values


def greedy_predictions(data_slice: pd.DataFrame, arms: list[int], model_cols: list[str]) -> list[float]:
    """
    The prediction actually used at each step under the greedy CMAB strategy:
    `model_cols[arms[t]]`'s prediction at row t.

    Ported from the original's `data_slice.iloc[ii, arm + 1]` (e.g.
    MABSS_utility.py:1140), which hardcoded "column 0 is target, columns 1..K
    are the arms in index order" -- this version looks columns up BY NAME, so
    it's correct even if the frame's column order ever changes (e.g. a 'mean'
    column inserted before the model_i columns).
    """
    return [data_slice.iloc[ii][model_cols[arm]] for ii, arm in enumerate(arms)]
