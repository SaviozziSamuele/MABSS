"""
Phase 0 golden: tests/goldens/stats_recompute.json — THE LOCK ON THE PAPER'S NUMBERS.

For all 30 paper configurations (5 tickers x 3 policies x {homogeneous, heterogeneous}
pool, exactly mirroring notebook cell 30's iteration over experiments_cluster/),
recomputes `stats_df.csv` from scratch starting only from the cached
`multirun_results.pkl` + `pctchange_preds.csv` artifacts, using the frozen module's
own `plot_multi_run_bandit` (for the `strategies` frame) and a faithful
reimplementation of the `strategy_stats` formula nested inside
`compute_and_highlight_stats` (copied verbatim from tests/_frozen -- see below).

The point: this proves the artifact corpus is self-contained and the "cheap half" of
the pipeline (stats/tables/figures) is bit-reproducible without a GPU, without
retraining, and without the (missing) driver that produced the 2.3 GB corpus in the
first place. It is also the regression fixture for Phase 4's numbers-changing fixes:
each fix's own golden update is reviewed against a diff from THIS file, so "did we
accidentally move a number we didn't mean to" is always answerable.

`compute_and_highlight_stats` itself is not called directly here because it calls
the un-imported `display()` (a real bug, see MABSS_utility.py:1314) which only works
inside IPython. We reimplement its pre-display DataFrame-construction logic exactly
(the `strategy_stats` nested function, lines 1260-1276 of the frozen module) so this
script runs headless.

Only reads experiments_cluster/. Never writes into it (save_dir=None throughout).

Usage:
    python tools/build_stats_recompute_golden.py [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no GUI popups, no accumulating figure warnings issue
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
BASE_DIR = REPO_ROOT / "experiments_cluster"
OUT_PATH = REPO_ROOT / "tests" / "goldens" / "stats_recompute.json"
FROZEN_PATH = REPO_ROOT / "tests" / "_frozen" / "mabss_utility_frozen.py"


def load_frozen():
    spec = importlib.util.spec_from_file_location("mabss_utility_frozen", FROZEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # tqdm.notebook raises headless (bug :4) -- see conftest note in try_repro.py.
    # Patch only in-process, never edit the frozen file itself.
    from tqdm.auto import tqdm as tqdm_auto

    module.tqdm = tqdm_auto
    return module


# Verbatim reimplementation of the nested `strategy_stats` function inside
# `compute_and_highlight_stats` (tests/_frozen/mabss_utility_frozen.py:1260-1276).
# Kept byte-identical in logic on purpose -- this IS the "as published" formula,
# hardcoded 252 and all. Phase 4 fixes this; this script exists to characterize
# what's published today before anything changes.
def strategy_stats_as_published(daily_returns) -> dict:
    returns = pd.Series(daily_returns)
    cumulative_returns = (1 + returns).cumprod()
    stats = {}
    stats["total_return"] = cumulative_returns.iloc[-1] - 1
    stats["annualized_return"] = (stats["total_return"] + 1) ** (252 / len(returns)) - 1
    stats["hit_ratio"] = (returns > 0).sum() / len(returns)
    roll_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - roll_max) / roll_max
    stats["max_drawdown"] = drawdown.min()
    stats["volatility"] = returns.std() * np.sqrt(252)
    stats["sharpe_ratio"] = stats["annualized_return"] / stats["volatility"]
    if stats["max_drawdown"] != 0:
        stats["calmar_ratio"] = stats["annualized_return"] / abs(stats["max_drawdown"])
    else:
        stats["calmar_ratio"] = float("inf")
    return stats


def compute_stats_df(strategies: pd.DataFrame) -> pd.DataFrame:
    """Reimplements compute_and_highlight_stats' df construction, sans display()."""
    greedy_cols = [c for c in strategies.columns if "returns_greedy_" in c]
    stats_greedy = [strategy_stats_as_published(strategies[c]) for c in greedy_cols]
    stats_avg = strategy_stats_as_published(strategies["returns_average_ensemble"])

    stats_greedy_mean = {k: np.mean([s[k] for s in stats_greedy]) for k in stats_greedy[0]}
    stats_greedy_std = {k: np.std([s[k] for s in stats_greedy]) for k in stats_greedy[0]}

    metric_names = list(stats_avg.keys())
    df_data = {
        "average_ensemble": [stats_avg[m] for m in metric_names],
        "greedy (mean)": [stats_greedy_mean[m] for m in metric_names],
        "greedy (stddev)": [stats_greedy_std[m] for m in metric_names],
    }
    return pd.DataFrame(df_data, index=metric_names)


# Mirrors notebook cell 30's iteration exactly.
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


def build(frozen) -> dict:
    out: dict[str, dict] = {}
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
                if not loader.exists("pctchange_preds") or not loader.exists("multirun_results"):
                    out[run_key] = {"skipped": "missing pctchange_preds or multirun_results"}
                    continue
                if not stored_stats_path.exists():
                    out[run_key] = {"skipped": "no stored stats_df.csv to compare against"}
                    continue

                pct_change_preds = loader.load("pctchange_preds")
                multirun_results = loader.load("multirun_results")
                stored = pd.read_csv(stored_stats_path, index_col=0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    best_seed, strategies = frozen.plot_multi_run_bandit(
                        multirun_results, pct_change_preds, config, save_dir=None
                    )
                plt.close("all")

                if strategies is None:
                    out[run_key] = {"skipped": "plot_multi_run_bandit returned no strategies"}
                    continue

                recomputed = compute_stats_df(strategies)

                # Align columns/index for comparison; report the max abs diff per
                # metric and whether it's within a tight numerical tolerance.
                common_idx = [i for i in stored.index if i in recomputed.index]
                diffs = {}
                max_abs_diff = 0.0
                for metric in common_idx:
                    row = {}
                    for col in ["average_ensemble", "greedy (mean)", "greedy (stddev)"]:
                        if col not in stored.columns or col not in recomputed.columns:
                            continue
                        a = float(stored.loc[metric, col])
                        b = float(recomputed.loc[metric, col])
                        d = abs(a - b) if np.isfinite(a) and np.isfinite(b) else (
                            0.0 if a == b else float("nan")
                        )
                        row[col] = {"stored": a, "recomputed": b, "abs_diff": d}
                        if np.isfinite(d):
                            max_abs_diff = max(max_abs_diff, d)
                    diffs[metric] = row

                out[run_key] = {
                    "bandit_dir": str(loader.bandit_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "pred_dir": str(loader.pred_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "best_seed": str(best_seed),
                    "n_seeds": len(multirun_results),
                    "max_abs_diff": max_abs_diff,
                    "bit_exact": max_abs_diff < 1e-9,
                    "metrics": diffs,
                }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    frozen = load_frozen()
    print("Recomputing stats_df for all 30 paper configs from cached artifacts...", file=sys.stderr)
    data = build(frozen)

    n_ok = sum(1 for v in data.values() if v.get("bit_exact"))
    n_skip = sum(1 for v in data.values() if "skipped" in v)
    n_mismatch = len(data) - n_ok - n_skip
    print(f"{len(data)} configs: {n_ok} bit-exact, {n_mismatch} mismatched, {n_skip} skipped.", file=sys.stderr)
    if n_mismatch:
        for k, v in data.items():
            if "skipped" not in v and not v.get("bit_exact"):
                print(f"  MISMATCH: {k} max_abs_diff={v['max_abs_diff']}", file=sys.stderr)

    serialized = json.dumps(data, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not OUT_PATH.exists():
            print(f"FAIL: {OUT_PATH} does not exist", file=sys.stderr)
            return 1
        if OUT_PATH.read_text(encoding="utf-8") != serialized:
            print(f"FAIL: stats_recompute differs from golden at {OUT_PATH}", file=sys.stderr)
            return 1
        print("OK: stats_recompute matches golden.", file=sys.stderr)
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(serialized, encoding="utf-8")
    print(f"Wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
