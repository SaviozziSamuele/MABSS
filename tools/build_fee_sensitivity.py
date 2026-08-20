"""
Transaction-cost sensitivity analysis -- Reviewer #3's Comment 4 ("the current
evaluation still assumes zero transaction costs and zero slippage... report
transaction-cost sensitivity results under different one-way fee
assumptions"). Pure recompute from cache: reuses the already-published 25-seed
`multirun_results.pkl` + `pctchange_preds.csv` for all 30 paper configs, no
retraining or bandit re-run needed.

Uses `mabss.strategies.build_multirun_strategies_costed` (a fee-aware sibling
of the Table-1 pipeline's `build_multirun_strategies`, kept fully separate --
see that function's docstring) and the existing, UNMODIFIED
`mabss.plots.tables.build_stats_table` for aggregation, so no new aggregation
logic is introduced here at all.

Usage:
    python tools/build_fee_sensitivity.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mabss.experiments.store import ExperimentStore  # noqa: E402
from mabss.plots.tables import build_stats_table  # noqa: E402
from mabss.strategies import build_multirun_strategies_costed  # noqa: E402

BASE_DIR = REPO_ROOT / "experiments_cluster"
OUT_PATH = REPO_ROOT / "tests" / "goldens" / "fee_sensitivity.csv"

BASE_BANDIT_CONFIG = dict(
    START_DATE="2000-01-01",
    WINDOW_SIZE=252,
    BANDIT_EMBEDDING=15,
    METRIC="l1",
    EPISODES=100,
)
BASE_PRED_CONFIG = dict(
    START_DATE="2000-01-01",
    WINDOW_SIZE=252,
    MODEL_EMBEDDING=15,
    EPOCHS=10,
)
TICKERS = ["SPY", "BTC-USD", "TLT", "EURUSD=X", "GC=F"]
POLICIES = ["softmax", "ucb", "thompson"]
ARCHS = [
    {"MLP": True, "CNN": False, "RNN": False, "N_SEEDS_PER_ARCH": 50},
    {"MLP": True, "CNN": True, "RNN": True, "N_SEEDS_PER_ARCH": 20},
]
FEE_BPS_GRID = [0, 5, 10, 25, 50]
METRICS_TO_REPORT = ["annualized_return", "sharpe_ratio", "max_drawdown"]


def main() -> int:
    rows = []
    for ticker in TICKERS:
        for policy in POLICIES:
            for arch in ARCHS:
                n_arms = arch["N_SEEDS_PER_ARCH"] * sum(arch.get(k, False) for k in ["MLP", "RNN", "CNN"])
                arch_label = "hetero" if arch["CNN"] else "homo"
                config_key = f"{ticker}__{policy}__{arch_label}"

                pred_config = dict(BASE_PRED_CONFIG, TICKER=ticker, **arch)
                bandit_config = dict(BASE_BANDIT_CONFIG, TICKER=ticker, POLICY=policy, N_ARMS=n_arms)

                pred_store = ExperimentStore(pred_config, BASE_DIR, mode="r")
                bandit_store = ExperimentStore(bandit_config, BASE_DIR, mode="r")
                if not (pred_store.exists("pctchange_preds") and bandit_store.exists("multirun_results")):
                    print(f"SKIP {config_key}: cache miss", file=sys.stderr)
                    continue

                pct_change_preds = pred_store.load("pctchange_preds")
                multirun_results = bandit_store.load("multirun_results")

                for fee_bps in FEE_BPS_GRID:
                    _, strategies = build_multirun_strategies_costed(
                        multirun_results, pct_change_preds, fee_bps=fee_bps
                    )
                    if strategies is None:
                        continue
                    stats = build_stats_table(strategies)
                    for metric in METRICS_TO_REPORT:
                        rows.append(
                            {
                                "config_key": config_key,
                                "ticker": ticker,
                                "policy": policy,
                                "arch": arch_label,
                                "fee_bps": fee_bps,
                                "metric": metric,
                                "average_ensemble": stats.loc[metric, "average_ensemble"],
                                "greedy_mean": stats.loc[metric, "greedy (mean)"],
                                "greedy_stddev": stats.loc[metric, "greedy (stddev)"],
                            }
                        )
                print(f"OK {config_key}", file=sys.stderr)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_key", "ticker", "policy", "arch", "fee_bps", "metric",
                "average_ensemble", "greedy_mean", "greedy_stddev",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_PATH} ({len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
