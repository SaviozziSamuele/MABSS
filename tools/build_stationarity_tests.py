"""
Phase (reviewer response): formal stationarity/structural-break diagnostics
for the paper's 5 asset price/return series -- Reviewer #3's Comment 1.
Pure recompute from cache: reads `preds.csv`'s raw `target` price level and
`pctchange_preds.csv`'s `target` return series, both cached on disk for every
experiment directory, no network access needed.

For each asset, runs ADF+KPSS on both the price level and the return series,
and the CUSUM/Quandt-Andrews structural-break tests on the return series
(see mabss/stationarity.py's module docstring for why levels aren't a
meaningful input to a mean-stability test).

Usage:
    python tools/build_stationarity_tests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mabss.experiments.store import ExperimentStore  # noqa: E402
from mabss.stationarity import (  # noqa: E402
    cusum_structural_break_test,
    quandt_andrews_breakpoint,
    stationarity_battery,
)

BASE_DIR = REPO_ROOT / "experiments_cluster"
OUT_PATH = REPO_ROOT / "tests" / "goldens" / "stationarity_tests.json"

TICKERS = ["SPY", "BTC-USD", "EURUSD=X", "GC=F", "TLT"]

# One representative pred config per ticker (homogeneous MLP pool) -- the
# underlying price/return series doesn't depend on pool composition, so any
# real cached pred dir for the ticker works.
PRED_CONFIG = dict(
    START_DATE="2000-01-01",
    WINDOW_SIZE=252,
    MLP=True,
    RNN=False,
    CNN=False,
    MODEL_EMBEDDING=15,
    EPOCHS=10,
    N_SEEDS_PER_ARCH=50,
)


def run_for_ticker(ticker: str) -> dict:
    print(f"=== {ticker} ===", file=sys.stderr)
    config = dict(PRED_CONFIG, TICKER=ticker)
    store = ExperimentStore(config, BASE_DIR, mode="r")

    price_level = store.load("preds")["target"]
    returns = store.load("pctchange_preds")["target"]

    level_battery = stationarity_battery(price_level, series_name=f"{ticker}_price_level")
    return_battery = stationarity_battery(returns, series_name=f"{ticker}_return")
    cusum = cusum_structural_break_test(returns)
    breakpoint_result = quandt_andrews_breakpoint(returns, n_boot=2000, seed=0)

    print(
        f"  price_level: {level_battery['classification']}  "
        f"return: {return_battery['classification']}  "
        f"cusum_break={cusum['reject_at_5pct']}  "
        f"breakpoint~{breakpoint_result['breakpoint_date']}",
        file=sys.stderr,
    )

    return {
        "ticker": ticker,
        "n_obs": len(returns),
        "price_level": level_battery,
        "return": return_battery,
        "cusum_structural_break": cusum,
        "quandt_andrews_breakpoint": breakpoint_result,
    }


def main() -> int:
    out = {}
    for ticker in TICKERS:
        try:
            out[ticker] = run_for_ticker(ticker)
        except FileNotFoundError as e:
            print(f"  SKIPPED ({e})", file=sys.stderr)
            out[ticker] = {"skipped": True, "reason": str(e)}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
