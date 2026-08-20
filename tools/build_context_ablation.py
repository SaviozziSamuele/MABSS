"""
Local context-vector ablation for Reviewer #2's Major Point 3 (Section 3.4):
does appending a shared, non-arm-specific market-risk feature -- rolling
realized volatility of the target -- to the bandit's context vector change
anything, empirically? Manuscript.tex:261 argues theoretically that it
shouldn't ("would be identical across all arms... a global, non-discriminatory
bias term"); this script tests that argument against the real corpus.

Scope (deliberately bounded to run locally, no cluster needed -- see the plan
file's Step 2D): SPY only, homogeneous 50-arm pool, all 3 policies
(softmax/ucb/thompson), 5 seeds each for the new with-feature run. The
baseline is reported two ways: the full, real, already-published 25-seed
`multirun_results.pkl` (authoritative), AND a 5-seed subsample of it (seeds
0-4, the same random seeds the new run uses) for a variance-comparable
side-by-side against the with-feature run.

This is the single most expensive script in the reviewer-response plan --
the bandit stage is CPU-bound and NOT fast (a single 50-60 arm SPY seed has
been measured at >1hr pure-Python on this machine). Run this in the
background; --n-seeds/--episodes let you trim scope further if the default
estimate proves too slow.

Usage:
    python tools/build_context_ablation.py
    python tools/build_context_ablation.py --n-seeds 3          # trim if too slow
    python tools/build_context_ablation.py --policies ucb       # just one policy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mabss.experiments.store import ExperimentStore  # noqa: E402
from mabss.plots.tables import build_stats_table  # noqa: E402
from mabss.rewards import build_rewards_and_contexts  # noqa: E402
from mabss.runner import run_bandit_multi  # noqa: E402
from mabss.strategies import build_multirun_strategies  # noqa: E402

BASE_DIR = REPO_ROOT / "experiments_cluster"
OUT_PATH = REPO_ROOT / "tests" / "goldens" / "context_ablation.json"

TICKER = "SPY"
N_ARMS = 50
BANDIT_EMBEDDING = 15
RISK_FEATURE_WINDOW = 20

PRED_CONFIG = dict(
    TICKER=TICKER,
    START_DATE="2000-01-01",
    WINDOW_SIZE=252,
    MLP=True,
    RNN=False,
    CNN=False,
    MODEL_EMBEDDING=15,
    EPOCHS=10,
    N_SEEDS_PER_ARCH=50,
)


def bandit_config(policy: str) -> dict:
    """No ALPHA/GAMMA keys -- matches the B3_cluster_no_alpha_gamma schema
    every real experiments_cluster/ bandit directory actually uses (see
    mabss/experiments/legacy.py). Adding ALPHA/GAMMA here would make
    ExperimentStore build a v1 id that doesn't match any real directory."""
    return dict(
        TICKER=TICKER,
        START_DATE="2000-01-01",
        N_ARMS=N_ARMS,
        WINDOW_SIZE=252,
        BANDIT_EMBEDDING=BANDIT_EMBEDDING,
        METRIC="l1",
        EPISODES=100,
        POLICY=policy,
    )


def _summarize(strategies, seed_keys=None) -> dict:
    """Runs build_stats_table, optionally restricted to a subset of seed
    columns, and returns just the greedy-mean/stddev rows as plain floats
    (JSON-serializable) for the metrics that matter for this comparison."""
    if seed_keys is not None:
        cols = [f"returns_greedy_{s}" for s in seed_keys if f"returns_greedy_{s}" in strategies.columns]
    else:
        cols = None
    df = build_stats_table(strategies, strategies_greedy_cols=cols)
    out = {}
    for metric in ("annualized_return", "sharpe_ratio", "max_drawdown", "volatility"):
        out[metric] = {
            "average_ensemble": float(df.loc[metric, "average_ensemble"]),
            "greedy_mean": float(df.loc[metric, "greedy (mean)"]),
            "greedy_stddev": float(df.loc[metric, "greedy (stddev)"]),
        }
    out["n_seeds"] = len(cols) if cols is not None else sum(
        1 for c in strategies.columns if c.startswith("returns_greedy_")
    )
    return out


def run_for_policy(policy: str, n_seeds: int, episodes_override: int | None) -> dict:
    print(f"=== policy={policy} ===", file=sys.stderr)
    t0 = time.time()

    pred_store = ExperimentStore(PRED_CONFIG, BASE_DIR, mode="r")
    pctchange_preds = pred_store.load("pctchange_preds")

    bandit_store = ExperimentStore(bandit_config(policy), BASE_DIR, mode="r")
    multirun_baseline_25 = bandit_store.load("multirun_results")

    seed_keys_5 = [f"seed_{i}" for i in range(n_seeds)]
    multirun_baseline_5 = {k: multirun_baseline_25[k] for k in seed_keys_5 if k in multirun_baseline_25}

    _, strategies_baseline_25 = build_multirun_strategies(multirun_baseline_25, pctchange_preds)
    _, strategies_baseline_5 = build_multirun_strategies(multirun_baseline_5, pctchange_preds)

    print(f"  baseline loaded ({len(multirun_baseline_25)} cached seeds), building with-feature contexts...", file=sys.stderr)
    _rewards_df, contexts_matrix, rewards_matrix = build_rewards_and_contexts(
        pctchange_preds,
        metric="l1",
        bandit_embedding=BANDIT_EMBEDDING,
        n_arms=N_ARMS,
        include_market_risk_feature=True,
        risk_feature_window=RISK_FEATURE_WINDOW,
    )

    model_cols = [f"model_{n}" for n in range(N_ARMS)]
    offset = BANDIT_EMBEDDING
    X_data = pctchange_preds[model_cols].values[offset:]
    Y_data = pctchange_preds["target"].values[offset:]

    # env_config's BANDIT_EMBEDDING is deliberately the TRUE context
    # dimensionality (15 + 1 for the appended feature) -- run_bandit_seed
    # reads this key directly as WalkForwardEnv's `d`. This dict is used only
    # to construct the environment for this ablation run; it is never passed
    # to ExperimentStore, so it never risks addressing (or worse, being
    # mistaken for) any real cached directory.
    env_config = dict(bandit_config(policy))
    env_config["BANDIT_EMBEDDING"] = BANDIT_EMBEDDING + 1
    if episodes_override is not None:
        env_config["EPISODES"] = episodes_override

    print(f"  running {n_seeds} seeds with the appended risk feature (this is the slow part)...", file=sys.stderr)
    results_with_feature = run_bandit_multi(
        contexts_matrix,
        X_data,
        Y_data,
        rewards_matrix,
        env_config,
        n_seeds=n_seeds,
    )
    _, strategies_with_feature = build_multirun_strategies(results_with_feature, pctchange_preds)

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s", file=sys.stderr)

    return {
        "policy": policy,
        "n_seeds_with_feature": n_seeds,
        "risk_feature_window": RISK_FEATURE_WINDOW,
        "elapsed_seconds": elapsed,
        "baseline_25seed": _summarize(strategies_baseline_25),
        "baseline_5seed_subsample": _summarize(strategies_baseline_5, seed_keys=seed_keys_5),
        "with_risk_feature": _summarize(strategies_with_feature),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=5, help="seeds for the new with-feature run (default 5)")
    parser.add_argument("--episodes", type=int, default=None, help="override EPISODES for the with-feature run (fallback if too slow)")
    parser.add_argument("--policies", nargs="+", default=["softmax", "ucb", "thompson"])
    args = parser.parse_args()

    out = {}
    for policy in args.policies:
        out[policy] = run_for_policy(policy, args.n_seeds, args.episodes)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
