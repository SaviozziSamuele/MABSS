"""
Position-switch frequency for the CMAB-greedy vs. static-ensemble strategies,
across all 30 published-Table-1 configs -- the mechanism check behind the
transaction-cost sensitivity analysis (Appendix B / tools/build_fee_sensitivity.py):
does the CMAB strategy really flip its long/cash position more often than the
ensemble, and is that uniform across policies? Pure recompute from cache: the
same 25-seed `multirun_results.pkl` + `pctchange_preds.csv` already used for
Table 1 and the fee-sensitivity analysis, no retraining or bandit re-run needed.

Usage:
    python tools/build_switch_rate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mabss.experiments.store import ExperimentStore  # noqa: E402
from mabss.strategies import ensemble_signal, greedy_predictions  # noqa: E402

BASE_DIR = REPO_ROOT / "experiments_cluster"
OUT_PATH = REPO_ROOT / "tests" / "goldens" / "switch_rate.csv"

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


def switch_frac(signal: np.ndarray) -> float:
    """Fraction of consecutive days on which the long(>=0)/cash(<0) position flips."""
    position = (np.asarray(signal) >= 0).astype(float)
    if len(position) < 2:
        return float("nan")
    switched = position[1:] != position[:-1]
    return float(switched.mean())


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

                first_seed = next(iter(multirun_results))
                T = len(multirun_results[first_seed]["best arms greedy"])
                data_slice = pct_change_preds.iloc[-T:].copy()
                model_cols = [c for c in data_slice.columns if c.startswith("model_")]

                cmab_switch_fracs = []
                for res in multirun_results.values():
                    arms_greedy = res["best arms greedy"]
                    if len(arms_greedy) != T:
                        continue
                    preds = greedy_predictions(data_slice, arms_greedy, model_cols)
                    cmab_switch_fracs.append(switch_frac(np.array(preds)))

                ens_signal = ensemble_signal(data_slice, model_cols)
                ens_switch = switch_frac(ens_signal)

                rows.append(
                    {
                        "config_key": config_key,
                        "ticker": ticker,
                        "policy": policy,
                        "arch": arch_label,
                        "n_days": T,
                        "n_seeds": len(cmab_switch_fracs),
                        "ensemble_switch_frac": ens_switch,
                        "cmab_switch_frac_mean": float(np.mean(cmab_switch_fracs)),
                    }
                )
                print(
                    f"OK {config_key}  T={T}  ens={ens_switch:.4f}  cmab={np.mean(cmab_switch_fracs):.4f}",
                    file=sys.stderr,
                )

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(df)} rows)", file=sys.stderr)

    print("\n=== Aggregate across all 30 configs ===", file=sys.stderr)
    print("mean ensemble switch frac:", df["ensemble_switch_frac"].mean(), file=sys.stderr)
    print("mean cmab switch frac:    ", df["cmab_switch_frac_mean"].mean(), file=sys.stderr)
    print("by policy:", file=sys.stderr)
    print(df.groupby("policy")[["ensemble_switch_frac", "cmab_switch_frac_mean"]].mean(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
