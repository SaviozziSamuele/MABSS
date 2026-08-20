"""General (non-paper) diagnostic plots: predictions, reward stats, bandit runs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from mabss.strategies import ensemble_signal, greedy_predictions, long_cash_returns, nav

from .style import save_figure


def plot_prediction_analysis(pct_change_preds: pd.DataFrame, save_dir=None):
    """Ported from MABSS_utility.py:955-988, logic unchanged (both figures use
    `pct_change_preds.columns[1:]`, which assumes 'target' is column 0 -- same
    assumption as the original; not fixed here since it's a documented, stable
    convention every caller already follows, not a bug in itself)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nav(pct_change_preds["target"].values), label="target")
    for col in pct_change_preds.columns[1:]:
        strat_ret = long_cash_returns(pct_change_preds[col].values, pct_change_preds["target"].values)
        color = "orange" if col == "mean" else "gray"
        label = col if col == "mean" else None
        ax.plot(nav(strat_ret), label=label, c=color)
    ax.legend()
    ax.set_title("Cumulative return of predictions")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Time")
    if save_dir:
        save_figure(fig, f"{save_dir}/prediction_analysis_1.png")
    plt.show()

    fig, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    for col in pct_change_preds.columns[1:]:
        strat_ret = long_cash_returns(pct_change_preds[col].values, pct_change_preds["target"].values)
        color = "orange" if col == "mean" else "gray"
        label = col if col == "mean" else None
        alpha = 1 if col == "mean" else 0.5
        ax3.plot(nav(strat_ret, index=pct_change_preds.index), label=label, c=color, alpha=alpha)
    ax3.legend()
    ax3.set_title("Cumulative return of reward signals")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Cumulative return")
    if save_dir:
        save_figure(fig, f"{save_dir}/prediction_analysis_2.png")
    plt.show()


def plot_rewards_stats(rewards_df: pd.DataFrame, config: dict, save_dir=None):
    """
    Ported from MABSS_utility.py:991-1052 (`plot_rewards_stats`), with two
    changes:

    1. **ACF target-leak fixed** (originally `:1006-1007`). The original loop's
       COUNT came from the model columns (`N_ARMS` of them) but its INDEXING
       was into `np.array(rewards_df).T` -- the whole frame, columns
       `['target', model_0, ..., model_{N-1}, 'mean']` transposed. Since the
       loop only ran `N_ARMS` iterations starting at row 0, it captured
       `target` (row 0) as if it were a model, and never reached the LAST
       model column (row `N_ARMS`, one past the loop's range). Fixed here by
       iterating `rewards_df[model_cols]` directly, by name.
    2. **`pct_change_preds` parameter dropped** -- it was accepted by the
       original but never used in the function body.

    `rewards_df` must have a DatetimeIndex already (callers should ensure this
    via `mabss.experiments.ExperimentStore`, which parses it uniformly, rather
    than this function mutating the caller's frame in place as the original did).
    """
    model_cols = [f"model_{n}" for n in range(config["N_ARMS"])]
    cutoff = int(0.8 * len(rewards_df))

    acfs_1 = [acf(rewards_df[col].values[:cutoff], nlags=36) for col in model_cols]
    mean_acfs_1 = np.mean(acfs_1, axis=0)
    std_acfs_1 = np.std(acfs_1, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_acf(np.array(rewards_df["mean"]), lags=36, ax=ax, auto_ylims=True)
    acf_values = acf(np.array(rewards_df["mean"]), nlags=36)
    lags = np.arange(len(acf_values))
    ax.scatter(lags, acf_values, label="Average ACF", c="tab:blue", marker=".")
    ax.fill_between(
        range(len(mean_acfs_1)),
        mean_acfs_1 - std_acfs_1,
        mean_acfs_1 + std_acfs_1,
        alpha=0.2,
        label="Standard deviation",
        color="tab:blue",
    )
    ax.set_title("ACF of reward signals")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend()
    if save_dir:
        save_figure(fig, f"{save_dir}/reward_stats_1.png")
    plt.show()

    fig, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    ax2.scatter(
        rewards_df.index,
        np.std(np.array(rewards_df[model_cols]), axis=1),
        marker=".",
        label="Reward's stddev",
    )
    ticker = config.get("TICKER", "target")
    ax2.plot(
        nav(rewards_df["target"].values, index=rewards_df.index) / 1000,
        c="gray",
        alpha=0.5,
        label=f"{ticker} Index",  # fixed: was hardcoded 'S&P500 Index' regardless of ticker
    )
    ax2.set_title("Standard deviation of reward signal across random seeds")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Standard deviation")
    ax2.legend()
    if save_dir:
        save_figure(fig, f"{save_dir}/reward_stats_2.png")
    plt.show()


def plot_single_run_bandit(results: dict, pct_change_preds: pd.DataFrame, save_dir=None):
    """Ported from MABSS_utility.py:1055-1112. Uses
    `mabss.strategies.greedy_predictions` (named column lookup) instead of the
    original's `pct_change_preds.iloc[ii, arm + 1]` positional indexing, which
    silently assumed column 0 is 'target' and columns 1..K are the arms in
    index order."""
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(results["mean predictions reward"]), label="average")
    plt.plot(np.cumsum(results["total return"]), label="agent sm random")
    plt.plot(np.cumsum(results["total reward greedy"]), label="agent sm greedy")
    plt.plot(np.cumsum(results["total reward average"]), label="agent sm weighted")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.title("Score Softmax agent vs Average - Walk Forward Training")
    if save_dir:
        save_figure(fig1, f"{save_dir}/single_run_1.png")
    plt.show()

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(results["mean predictions reward"])[-100:], label="average")
    plt.plot(np.cumsum(results["total return"])[-100:], label="agent sm random")
    plt.plot(np.cumsum(results["total reward greedy"])[-100:], label="agent sm greedy")
    plt.plot(np.cumsum(results["total reward average"])[-100:], label="agent sm weighted")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.title("Cumulative reward - last 100 steps")
    if save_dir:
        save_figure(fig2, f"{save_dir}/single_run_2.png")
    plt.show()

    model_cols = [c for c in pct_change_preds.columns if c.startswith("model_")]
    n = len(results["best arms greedy"])
    data_slice = pct_change_preds.iloc[-n:].copy()
    strategies = data_slice.copy()
    strategies["greedy"] = greedy_predictions(data_slice, results["best arms greedy"], model_cols)
    strategies["weighted"] = results["weighted predictions"]

    fig3, ax = plt.subplots(figsize=(10, 5))
    for col in ["greedy", "mean", "weighted"]:
        strat_ret = long_cash_returns(strategies[col].values, strategies["target"].values)
        ax.plot(nav(strat_ret), label=col)
    ax.set_title("Cumulative return of CMAB-derived strategies")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Time steps")
    ax.legend()
    if save_dir:
        save_figure(fig3, f"{save_dir}/single_run_3.png")
    plt.show()


def plot_multi_run_bandit(results_multi: dict, pct_change_preds: pd.DataFrame, config: dict, save_dir=None):
    """
    Ported from MABSS_utility.py:1114-1257 (`plot_multi_run_bandit`). Two fixes:

    1. **`NARMS` typo fixed** (originally `:1210`,
       `CONFIG.get('NARMS', len(set(categories)) + 1)`). The config key is
       `N_ARMS` everywhere else in this codebase; `.get('NARMS', ...)` always
       missed and fell through to the fallback (number of DISTINCT arms the
       best seed happened to pick, +1), which mis-scales the colorbar in PLOT
       2 -- confirmed on real data: fallback gives 61 for a 60-arm pool.
    2. Uses `mabss.strategies.greedy_predictions`/`long_cash_returns` instead
       of the original's positional `arm + 1` column indexing.

    Returns `(best_seed, strategies)`, same contract as the original.
    """
    if not results_multi:
        print("No results to plot.")
        return None, None

    first_seed = next(iter(results_multi))
    T = len(results_multi[first_seed]["best arms greedy"])
    print(f"Plotting over horizon T={T}")

    data_slice = pct_change_preds.iloc[-T:].copy()
    target_slice = data_slice["target"].values
    model_cols = [c for c in data_slice.columns if c.startswith("model_")]

    strategies = pd.DataFrame({"target": target_slice}, index=data_slice.index)

    tot_returns = []
    for seed in results_multi:
        arms_greedy = results_multi[seed]["best arms greedy"]
        if len(arms_greedy) != T:
            print(f"Warning: seed {seed} has mismatched length {len(arms_greedy)} vs {T}")
            continue

        greedy_preds = greedy_predictions(data_slice, arms_greedy, model_cols)
        strategies[f"greedy_{seed}"] = greedy_preds

        strat_ret = long_cash_returns(np.array(greedy_preds), target_slice)
        strategies[f"returns_greedy_{seed}"] = strat_ret
        strategies[f"nav_greedy_{seed}"] = nav(strat_ret).values

        tot_returns.append(np.prod(1 + strat_ret) - 1)

    if not tot_returns:
        print("No valid seeds.")
        return None, None

    seed_list = [s for s in results_multi if len(results_multi[s]["best arms greedy"]) == T]
    best_seed = seed_list[np.argmax(tot_returns)]
    print(f"Best seed: {best_seed} (total return: {max(tot_returns):.2%})")

    nav_cols = [f"nav_greedy_{s}" for s in seed_list if f"nav_greedy_{s}" in strategies.columns]
    mean_nav_greedy = strategies[nav_cols].mean(axis=1)
    stddev_nav_greedy = strategies[nav_cols].std(axis=1)

    mean_signal = ensemble_signal(data_slice, model_cols)
    ensemble_ret = long_cash_returns(mean_signal, target_slice)
    strategies["returns_average_ensemble"] = ensemble_ret
    strategies["nav_average_ensemble"] = nav(ensemble_ret).values

    fig1, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean_nav_greedy, c="tab:blue", label="greedy CMAB (mean)")
    ax.fill_between(
        mean_nav_greedy.index,
        mean_nav_greedy - stddev_nav_greedy,
        mean_nav_greedy + stddev_nav_greedy,
        alpha=0.2,
        color="tab:blue",
        label="±1 STD",
    )
    ax.plot(strategies["nav_average_ensemble"], c="tab:orange", label="average ensemble")
    ax.set_title("CMAB greedy strategy (multi-seed mean ± STD)")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend()
    if save_dir:
        save_figure(fig1, f"{save_dir}/multi_run_1.png")
    plt.show()

    from matplotlib.collections import LineCollection

    x = np.arange(T)
    y = strategies[f"nav_greedy_{best_seed}"].values
    categories = results_multi[best_seed]["best arms greedy"]

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    n_arms = config["N_ARMS"]  # FIX: was config.get('NARMS', ...), a typo that always missed
    cmap = plt.get_cmap("rainbow", n_arms)
    lc = LineCollection(segments, cmap=cmap, linewidths=3)
    lc.set_array(categories)
    lc.set_clim(0, n_arms - 1)

    fig2, ax = plt.subplots(figsize=(10, 5))
    map_lc = ax.add_collection(lc)
    ax.autoscale()
    ax.set_title(f"Best seed {best_seed}: Cumulative return by arm")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Steps")
    fig2.colorbar(map_lc, ax=ax, label="Arm/Model")
    if save_dir:
        save_figure(fig2, f"{save_dir}/multi_run_2.png")
    plt.show()

    returns_of_predictors = []
    for col in model_cols:
        strat_ret = long_cash_returns(data_slice[col].values, target_slice)
        returns_of_predictors.append(np.prod(1 + strat_ret) - 1)

    best_pred_idx = np.argmax(returns_of_predictors)
    best_pred_name = model_cols[best_pred_idx]
    best_ret = long_cash_returns(data_slice[best_pred_name].values, target_slice)
    best_nav = nav(best_ret, index=strategies.index)

    fig3, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategies[f"nav_greedy_{best_seed}"], c="tab:blue", label="greedy CMAB (best seed)")
    ax.plot(strategies["nav_average_ensemble"], c="tab:orange", label="average ensemble")
    ax.plot(best_nav, c="gray", label=f"best signal ({best_pred_name})")
    ax.set_title("CMAB vs baselines")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend()
    if save_dir:
        save_figure(fig3, f"{save_dir}/multi_run_3.png")
    plt.show()

    return best_seed, strategies
