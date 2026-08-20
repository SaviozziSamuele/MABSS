"""
Paired statistical significance testing for CMAB-greedy vs. static-ensemble
returns, across all 30 published-Table-1 configs -- Reviewer #2's Additional
Point 2 and Reviewer #3's Comment 5. Pure recompute from cache: the same
25-seed `multirun_results.pkl` + `pctchange_preds.csv` already used for
Table 1, no retraining or bandit re-run needed.

Primary test per config (30 total): Diebold-Mariano test + Sharpe-difference
bootstrap CI on the cross-seed-MEAN daily return vs. the deterministic
baseline (see mabss/significance.py's module docstring for why this, not one
test per seed). Per-seed DM results are also recorded as a robustness
appendix (fraction of seeds individually rejecting at 5%, range of per-seed
DM stats) but are NOT fed into the multiple-testing correction.

Holm and Benjamini-Hochberg corrections are applied across the 30 primary
p-values.

Usage:
    python tools/build_significance_tests.py
    python tools/build_significance_tests.py --n-boot 2000   # faster, less precise CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mabss.experiments.store import ExperimentStore  # noqa: E402
from mabss.significance import (  # noqa: E402
    apply_multiple_testing_correction,
    dm_test_returns,
    sharpe_diff_bootstrap_ci,
)
from mabss.strategies import build_multirun_strategies  # noqa: E402

BASE_DIR = REPO_ROOT / "experiments_cluster"
OUT_PATH = REPO_ROOT / "tests" / "goldens" / "significance_tests.json"

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


def run_for_config(ticker: str, policy: str, arch: dict, n_boot: int) -> dict | None:
    n_arms = arch["N_SEEDS_PER_ARCH"] * sum(arch.get(k, False) for k in ["MLP", "RNN", "CNN"])
    pred_config = dict(BASE_PRED_CONFIG, TICKER=ticker, **arch)
    bandit_config = dict(BASE_BANDIT_CONFIG, TICKER=ticker, POLICY=policy, N_ARMS=n_arms)

    pred_store = ExperimentStore(pred_config, BASE_DIR, mode="r")
    bandit_store = ExperimentStore(bandit_config, BASE_DIR, mode="r")
    if not (pred_store.exists("pctchange_preds") and bandit_store.exists("multirun_results")):
        return None

    pct_change_preds = pred_store.load("pctchange_preds")
    multirun_results = bandit_store.load("multirun_results")

    _, strategies = build_multirun_strategies(multirun_results, pct_change_preds)
    if strategies is None:
        return None

    greedy_cols = [c for c in strategies.columns if c.startswith("returns_greedy_")]
    baseline = strategies["returns_average_ensemble"]

    cmab_mean = strategies[greedy_cols].mean(axis=1)
    primary_dm = dm_test_returns(cmab_mean, baseline)
    primary_ci = sharpe_diff_bootstrap_ci(cmab_mean, baseline, n_boot=n_boot)

    per_seed_dm = [dm_test_returns(strategies[c], baseline) for c in greedy_cols]
    per_seed_pvals = [r["p_value"] for r in per_seed_dm]
    per_seed_stats = [r["dm_stat"] for r in per_seed_dm]

    return {
        "n_seeds": len(greedy_cols),
        "primary_dm_test": primary_dm,
        "primary_sharpe_diff_ci": primary_ci,
        "per_seed_robustness": {
            "frac_seeds_rejecting_at_5pct": sum(p < 0.05 for p in per_seed_pvals) / len(per_seed_pvals),
            "dm_stat_min": min(per_seed_stats),
            "dm_stat_max": max(per_seed_stats),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    out = {}
    primary_pvalues = {}

    for ticker in TICKERS:
        for policy in POLICIES:
            for arch in ARCHS:
                arch_label = "hetero" if arch["CNN"] else "homo"
                config_key = f"{ticker}__{policy}__{arch_label}"
                result = run_for_config(ticker, policy, arch, args.n_boot)
                if result is None:
                    print(f"SKIP {config_key}: cache miss", file=sys.stderr)
                    out[config_key] = {"skipped": True}
                    continue
                out[config_key] = result
                primary_pvalues[config_key] = result["primary_dm_test"]["p_value"]
                print(
                    f"OK {config_key}  dm_stat={result['primary_dm_test']['dm_stat']:+.3f}  "
                    f"p={result['primary_dm_test']['p_value']:.4f}  "
                    f"sharpe_diff={result['primary_sharpe_diff_ci']['estimate']:+.3f} "
                    f"[{result['primary_sharpe_diff_ci']['ci_low']:+.3f}, {result['primary_sharpe_diff_ci']['ci_high']:+.3f}]",
                    file=sys.stderr,
                )

    corrections = apply_multiple_testing_correction(primary_pvalues, methods=("holm", "fdr_bh"))
    for config_key in primary_pvalues:
        row = corrections.loc[config_key]
        out[config_key]["multiple_testing_correction"] = {
            "holm_reject": bool(row["holm_reject"]),
            "holm_pvalue_corrected": float(row["holm_pvalue_corrected"]),
            "fdr_bh_reject": bool(row["fdr_bh_reject"]),
            "fdr_bh_pvalue_corrected": float(row["fdr_bh_pvalue_corrected"]),
        }

    n_significant_holm = sum(1 for k in primary_pvalues if out[k]["multiple_testing_correction"]["holm_reject"])
    n_significant_bh = sum(1 for k in primary_pvalues if out[k]["multiple_testing_correction"]["fdr_bh_reject"])
    print(
        f"\n{n_significant_holm}/{len(primary_pvalues)} configs significant after Holm correction; "
        f"{n_significant_bh}/{len(primary_pvalues)} after Benjamini-Hochberg (alpha=0.05)",
        file=sys.stderr,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
