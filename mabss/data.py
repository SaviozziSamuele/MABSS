"""Price series loading, sliding-window construction, and return computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def window_time_series(ts, window_size: int):
    """
    Creates a sliding window view of a time series.

    Args:
        ts: A pandas Series representing the time series.
        window_size: The size of the sliding window.

    Returns:
        (windows, targets): windows[i] = ts[i : i+window_size], targets[i] = ts[i+window_size].

    Ported from MABSS_utility.py:27-45, logic unchanged. Requires a pandas object
    (uses `.iloc`) -- a raw ndarray raises AttributeError. If `window_size >=
    len(ts)`, silently returns two empty arrays rather than raising; this matches
    the original and is a known sharp edge (see mabss.env's own short-series guard
    for where this actually gets caught downstream).
    """
    windows = []
    targets = []
    for i in range(len(ts) - window_size):
        window = ts.iloc[i : i + window_size]
        windows.append(window)
        targets.append(ts.iloc[i + window_size])
    return np.array(windows), np.array(targets)


def load_price_series(ticker: str, start: str, *, auto_adjust: bool = True) -> pd.Series:
    """
    Fetches daily close prices for `ticker` from `start` via yfinance.

    Hardening applied here vs. MABSS_utility.py:93-95 (plan Phase 2):
      - `auto_adjust` is now an explicit, named parameter defaulting to True.
        yfinance changed its own default for this from False to True at some
        point after the cached experiments/ artifacts were generated; passing
        it explicitly means results no longer silently drift if yfinance's
        default changes again. True was confirmed (via offset-matching against
        the committed preds.csv files) to be what produced the cached corpus.
      - Raises a clear, actionable error instead of an opaque `KeyError: 'Close'`
        when the download is empty (network failure, delisted ticker, or a
        `start` date after the last available trade date).

    Still NOT done here (left for a dedicated follow-up, not part of this fix):
    forward-filling/reindexing onto a canonical trading calendar, or recording
    how many rows were dropped by `.dropna()`.
    """
    data = yf.Ticker(ticker).history(start=start, auto_adjust=auto_adjust)
    if data.empty or "Close" not in data.columns:
        raise ValueError(
            f"load_price_series({ticker!r}, start={start!r}): yfinance returned no "
            "usable data. This can mean a network/API failure, an invalid or "
            "delisted ticker, or a start date after the last available trade date. "
            "Check connectivity and the ticker symbol before retrying."
        )
    return data["Close"].dropna()


def compute_returns_from_preds(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a `preds` frame (columns: 'target' = actual price, 'model_i' =
    predicted price) into percentage-return space, aligned so that every column
    lives on the same scale: pct[col] = (preds[col] - target.shift(1)) / target.shift(1).

    Ported from MABSS_utility.py:97-108, logic unchanged. Drops the first row
    (the `.shift(1)` NaN). Any 'mean' column present in `preds` is swept up by
    the `model_cols` loop and recomputed in return-space; this is redundant with
    (but not inconsistent with) callers that later overwrite 'mean' as the
    row-wise mean of the model_i return columns -- both give the same value here
    because the denominator (`target.shift(1)`) is shared across columns.
    """
    actual_returns = preds["target"].pct_change()

    pct = pd.DataFrame(index=preds.index)
    pct["target"] = actual_returns

    model_cols = [c for c in preds.columns if c != "target"]
    for col in model_cols:
        pct[col] = (preds[col] - preds["target"].shift(1)) / preds["target"].shift(1)

    return pct.dropna()
