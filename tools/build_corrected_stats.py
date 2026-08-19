"""
Phase 4 deliverable: the as-published vs. corrected delta table for Table 1.

Recomputes stats for all 30 paper configurations from the SAME cached artifacts
tools/build_stats_recompute_golden.py uses (multirun_results.pkl +
pctchange_preds.csv, via the frozen module's plot_multi_run_bandit for the
`strategies` frame), but scores them with the FIXED mabss.metrics.strategy_stats
(per-asset periods_per_year inferred from the DatetimeIndex, real risk_free_rate)
instead of the legacy hardcoded-252-no-risk-free-rate formula.

This requires NO re-running of training or the bandit loop -- it is a pure
recomputation from cache, exactly the class of fix the plan's Phase 4 items 1-4
describe as "0 jobs" in the re-run-scope table. Output is written to
tests/goldens/stats_corrected.json and this script prints a human-readable delta
summary for the metrics that actually appear in Table 1
(annualized_return, sharpe_ratio, max_drawdown).

Usage:
    python tools/build_corrected_stats.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
from mabss.metrics import strategy_stats  # noqa: E402

BASE_DIR = REPO_ROOT / "experiments_cluster"
OUT_PATH = REPO_ROOT / "tests" / "goldens" / "stats_corrected.json"
FROZEN_PATH = REPO_ROOT / "tests" / "_frozen" / "mabss_utility_frozen.py"


def load_frozen():
    spec = importlib.util.spec_from_file_location("mabss_utility_frozen", FROZEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tqdm.auto import tqdm as tqdm_auto

    module.tqdm = tqdm_auto
    return module


def compute_corrected_stats_df(strategies: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    greedy_cols = [c for c in strategies.columns if "returns_greedy_" in c]
    stats_greedy = [
        strategy_stats(strategies[c], risk_free_rate=risk_free_rate) for c in greedy_cols
    ]
    stats_avg = strategy_stats(strategies["returns_average_ensemble"], risk_free_rate=risk_free_rate)

    import numpy as np

    metric_names = [k for k in stats_avg if k != "periods_per_year"]
    stats_greedy_mean = {k: np.mean([s[k] for s in stats_greedy]) for k in metric_names}
    df_data = {
        "average_ensemble": [stats_avg[m] for m in metric_names],
        "greedy (mean)": [stats_greedy_mean[m] for m in metric_names],
    }
    return pd.DataFrame(df_data, index=metric_names), stats_avg["periods_per_year"]


BASE_CONFIG = dict(
    START_DATE="2000-01-01",
    WINDOW_SIZE=252,
    MODEL_EMBEDDING=15,
    BANDIT_EMBEDDING=15,
    EPOCHS=10,
    METRIC="l1",
    PROB_SCALE=50.0,
    EPISODES=100,
    RANDOM_SEED=42,
)
TICKERS = ["SPY", "BTC-USD", "TLT", "EURUSD=X", "GC=F"]
POLICIES = ["softmax", "ucb", "thompson"]
ARCHS = [
    {"MLP": True, "CNN": False, "RNN": False, "N_SEEDS_PER_ARCH": 50},
    {"MLP": True, "CNN": True, "RNN": True, "N_SEEDS_PER_ARCH": 20},
]


def main() -> int:
    frozen = load_frozen()
    out = {}
    deltas_for_print = []

    for ticker in TICKERS:
        for policy in POLICIES:
            for arch in ARCHS:
                config = BASE_CONFIG.copy()
                config.update(arch)
                config["TICKER"] = ticker
                config["POLICY"] = policy
                config["N_ARMS"] = config["N_SEEDS_PER_ARCH"] * sum(
                    config.get(k, False) for k in ["MLP", "RNN", "CNN"]
                )
                arch_label = "hetero" if arch["CNN"] else "homo"
                run_key = f"{ticker}__{policy}__{arch_label}"

                loader = frozen.ExperimentLoader(config, BASE_DIR)
                stored_stats_path = Path(loader.bandit_dir) / "stats_df.csv"
                if not (
                    loader.exists("pctchange_preds")
                    and loader.exists("multirun_results")
                    and stored_stats_path.exists()
                ):
                    out[run_key] = {"skipped": True}
                    continue

                pct_change_preds = loader.load("pctchange_preds")
                multirun_results = loader.load("multirun_results")
                published = pd.read_csv(stored_stats_path, index_col=0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, strategies = frozen.plot_multi_run_bandit(
                        multirun_results, pct_change_preds, config, save_dir=None
                    )
                plt.close("all")
                if strategies is None:
                    out[run_key] = {"skipped": True}
                    continue

                corrected, ppy = compute_corrected_stats_df(strategies)

                row = {"periods_per_year": ppy}
                for metric in ["annualized_return", "sharpe_ratio", "max_drawdown"]:
                    for col in ["average_ensemble", "greedy (mean)"]:
                        pub = float(published.loc[metric, col])
                        cor = float(corrected.loc[metric, col])
                        row[f"{metric}__{col}"] = {"published": pub, "corrected": cor, "delta": cor - pub}
                out[run_key] = row

                if abs(ppy - 252) > 20:  # only print the interesting (non-~252) rows
                    deltas_for_print.append((run_key, ppy, row))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}", file=sys.stderr)

    print("\n=== Configs with periods_per_year far from 252 (published vs corrected) ===")
    for run_key, ppy, row in deltas_for_print:
        print(f"\n{run_key}  (measured {ppy:.1f} bars/year)")
        for metric in ["annualized_return", "sharpe_ratio"]:
            for col in ["greedy (mean)"]:
                k = f"{metric}__{col}"
                d = row[k]
                print(f"  {metric:20s} {col:15s} published={d['published']:+.4f}  corrected={d['corrected']:+.4f}  delta={d['delta']:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
