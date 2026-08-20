"""The paper's headline figures (Figures 1-4 + the grid/synthetic comparisons)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from mabss.strategies import ensemble_signal, greedy_predictions, long_cash_returns, nav

from .style import save_figure

DEFAULT_GRID_TICKERS = ["SPY", "BTC-USD", "EURUSD=X", "GC=F"]


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def plot_multiticker_reward_std(
    rewards_dfs_dict: dict, tickers_to_plot: list[str] | None = None, save_dir=None
):
    """Ported from MABSS_utility.py:1327-1383 (Figure 1). `tickers_to_plot`
    default matches the original's mutable-default-argument value, but as a
    tuple constructed fresh per call (the original's `list` default is a
    latent mutable-default-argument hazard even though nothing currently
    mutates it)."""
    if tickers_to_plot is None:
        tickers_to_plot = ["SPY", "BTC-USD", "TLT", "EURUSD=X"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()

    for idx, ticker in enumerate(tickers_to_plot):
        if ticker not in rewards_dfs_dict:
            continue

        ax = axes[idx]
        ax2 = ax.twinx()
        rewards_df = _ensure_datetime_index(rewards_dfs_dict[ticker])

        model_cols = [c for c in rewards_df.columns if c not in ("target", "mean")]
        reward_std = np.std(np.array(rewards_df[model_cols]), axis=1)

        ax.scatter(rewards_df.index, reward_std, marker=".", color="tab:blue", alpha=0.5, label="Reward's stddev")
        ax.set_ylabel("Reward Std Dev", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")

        price_proxy = nav(rewards_df["target"].values, index=rewards_df.index)
        ax2.plot(rewards_df.index, price_proxy, c="gray", alpha=0.7, linewidth=1.5, label=f"{ticker} Index")
        ax2.set_ylabel(f"{ticker} Price Proxy", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        ax.set_title(f"{ticker}: Std Dev of rewards vs Price")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()
    if save_dir:
        save_figure(fig, f"{save_dir}/reward_stddev_grid.png")
    plt.show()


def plot_global_reward_acf(rewards_dfs_dict: dict, nlags: int = 36, save_dir=None):
    """Ported from MABSS_utility.py:1386-1434 (Figure 2), logic unchanged --
    this function already iterated `model_cols` correctly (it was NOT affected
    by the ACF target-leak bug that hit `plot_rewards_stats`)."""
    all_acfs = []
    for _ticker, rewards_df in rewards_dfs_dict.items():
        model_cols = [c for c in rewards_df.columns if c not in ("target", "mean")]
        cutoff = int(0.8 * len(rewards_df))
        for col in model_cols:
            model_rewards = np.array(rewards_df[col])[:cutoff]
            model_rewards = model_rewards[~np.isnan(model_rewards)]
            if len(model_rewards) > nlags:
                all_acfs.append(acf(model_rewards, nlags=nlags, fft=True))

    if not all_acfs:
        raise ValueError("plot_global_reward_acf: no series had enough non-NaN observations for nlags="
                          f"{nlags}. (The original silently produced NaN plots / a TypeError here.)")

    mean_acfs = np.mean(all_acfs, axis=0)
    std_acfs = np.std(all_acfs, axis=0)
    lags = np.arange(len(mean_acfs))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(lags, mean_acfs, label="Average ACF (All Models & Tickers)", c="tab:blue", marker="o")
    ax.fill_between(lags, mean_acfs - std_acfs, mean_acfs + std_acfs, alpha=0.2, label="Standard deviation", color="tab:blue")
    ax.set_title("Global ACF of Reward Signals Across All Predictors")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.legend()
    if save_dir:
        save_figure(fig, f"{save_dir}/reward_acf_global.png")
    plt.show()


def plot_representative_cum_return(pct_change_preds: pd.DataFrame, ticker: str = "QQQ", save_dir=None):
    """
    Ported from MABSS_utility.py:1437-1469 (Figure 3). One hardening fix: the
    original saved a FIXED filename (`mean_cum_return_rep.png`) regardless of
    `ticker` -- harmless while the notebook only ever calls this once (for
    SPY), but a latent collision the moment a second ticker's figure is
    generated into the same `save_dir`. The filename is now ticker-parameterized.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nav(pct_change_preds["target"].values), label="Underlying Asset", c="black", linestyle="--")

    for col in pct_change_preds.columns[1:]:
        if col == "target":
            continue
        strat_ret = long_cash_returns(pct_change_preds[col].values, pct_change_preds["target"].values)
        color = "orange" if col == "mean" else "gray"
        label = "Ensemble Mean" if col == "mean" else None
        alpha = 1.0 if col == "mean" else 0.2
        ax.plot(nav(strat_ret), label=label, c=color, alpha=alpha)

    ax.legend()
    ax.set_title(f"Cumulative Return of Predictions: Individual vs Ensemble ({ticker})")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Time (Test Window)")
    if save_dir:
        save_figure(fig, f"{save_dir}/mean_cum_return_rep_{ticker}.png")
    plt.show()


def plot_variance_comparison(
    rewards_df_homo: pd.DataFrame, rewards_df_hetero: pd.DataFrame, ticker: str = "QQQ", window: int = 21, save_dir=None
):
    """Ported from MABSS_utility.py:1471-1528 (Figure 4), logic unchanged
    (filename was already ticker-parameterized in the original)."""
    rewards_df_homo = _ensure_datetime_index(rewards_df_homo)
    rewards_df_hetero = _ensure_datetime_index(rewards_df_hetero)

    cols_homo = [c for c in rewards_df_homo.columns if c not in ("target", "mean")]
    cols_hetero = [c for c in rewards_df_hetero.columns if c not in ("target", "mean")]

    std_homo = pd.Series(np.std(np.array(rewards_df_homo[cols_homo]), axis=1), index=rewards_df_homo.index)
    std_hetero = pd.Series(np.std(np.array(rewards_df_hetero[cols_hetero]), axis=1), index=rewards_df_hetero.index)

    smooth_homo = std_homo.rolling(window=window).mean()
    smooth_hetero = std_hetero.rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(smooth_homo.index, smooth_homo, color="tab:blue", linewidth=2.5,
            label=f"Homogeneous Pool (MLPs only) - {window}d Moving Avg")
    ax.plot(smooth_hetero.index, smooth_hetero, color="tab:red", linewidth=2.5,
            label=f"Heterogeneous Pool (MLP+CNN+RNN) - {window}d Moving Avg")
    ax.scatter(std_homo.index, std_homo, color="tab:blue", alpha=0.1, s=10)
    ax.scatter(std_hetero.index, std_hetero, color="tab:red", alpha=0.1, s=10)

    ax.set_title(f"Prediction Disagreement: Homogeneous vs Heterogeneous Pools ({ticker})")
    ax.set_ylabel("Reward Standard Deviation (Disagreement)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_dir:
        save_figure(fig, f"{save_dir}/variance_comparison_{ticker}.png")
    plt.show()


def plot_multiticker_grid_plot1(ticker_data_dict: dict, grid_tickers: list[str] | None = None, save_dir=None):
    """
    Ported from MABSS_utility.py:1530-1628. One fix: `grid_tickers` was
    hardcoded INSIDE the function body in the original (`['SPY', 'BTC-USD',
    'EURUSD=X', 'GC=F']`) -- callers couldn't change the four panels without
    editing the source. Now a parameter, defaulting to the same four tickers.
    """
    if grid_tickers is None:
        grid_tickers = DEFAULT_GRID_TICKERS

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, ticker in enumerate(grid_tickers):
        ax = axes[idx]

        if ticker not in ticker_data_dict:
            print(f"Skipping {ticker} - Data not provided.")
            ax.set_visible(False)
            continue

        results_multi, pct_change_preds = ticker_data_dict[ticker]
        if not results_multi:
            print(f"Skipping {ticker} - results_multi is empty.")
            ax.set_visible(False)
            continue

        first_seed = next(iter(results_multi))
        T = len(results_multi[first_seed]["best arms greedy"])

        data_slice = pct_change_preds.iloc[-T:].copy()
        target_slice = data_slice["target"].values
        model_cols = [c for c in data_slice.columns if c.startswith("model_")]

        strategies = pd.DataFrame({"target": target_slice}, index=data_slice.index)
        seed_list = []
        for seed in results_multi:
            arms_greedy = results_multi[seed]["best arms greedy"]
            if len(arms_greedy) != T:
                continue
            seed_list.append(seed)
            greedy_preds = greedy_predictions(data_slice, arms_greedy, model_cols)
            strat_ret = long_cash_returns(np.array(greedy_preds), target_slice)
            strategies[f"nav_greedy_{seed}"] = nav(strat_ret).values

        nav_cols = [f"nav_greedy_{s}" for s in seed_list]
        mean_nav_greedy = strategies[nav_cols].mean(axis=1)
        stddev_nav_greedy = strategies[nav_cols].std(axis=1)

        mean_signal = ensemble_signal(data_slice, model_cols)
        ensemble_ret = long_cash_returns(mean_signal, target_slice)
        nav_average_ensemble = nav(ensemble_ret).values

        ax.plot(mean_nav_greedy.index, mean_nav_greedy, c="tab:blue", label="greedy CMAB (mean)")
        ax.fill_between(
            mean_nav_greedy.index,
            mean_nav_greedy - stddev_nav_greedy,
            mean_nav_greedy + stddev_nav_greedy,
            alpha=0.2, color="tab:blue", label="±1 STD",
        )
        ax.plot(mean_nav_greedy.index, nav_average_ensemble, c="tab:orange", label="average ensemble")

        ax.set_title(f"{ticker}: CMAB vs Ensemble", fontsize=14, fontweight="bold")
        ax.set_ylabel("Cumulative Return")
        ax.set_xlabel("Date")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()
    if save_dir:
        save_figure(fig, f"{save_dir}/multi_run_grid_plot1.png")
    plt.show()


def plot_bandit_strategy_comparison(results: dict, pct_change_preds: pd.DataFrame, save_dir=None):
    """
    Ported from MABSS_utility.py:1630-1665. Fix: the original labeled its blue
    line "CMAB Greedy Strategy" but plotted `results['weighted predictions']`
    -- the WEIGHTED-by-probability signal, not the greedy (argmax) one. This
    happened to be correct only because the notebook's one call site used UCB,
    whose `select_arm` output is a one-hot vector (so weighted == greedy for
    that policy) -- it would silently mislabel a Softmax run, where the two
    genuinely differ. Now uses `mabss.strategies.greedy_predictions` against
    `results['best arms greedy']`, matching the label for every policy.
    """
    n = len(results["best arms greedy"])
    data_slice = pct_change_preds.iloc[-n:].copy()
    model_cols = [c for c in data_slice.columns if c.startswith("model_")]

    strategies = data_slice.copy()
    strategies["greedy_mab"] = greedy_predictions(data_slice, results["best arms greedy"], model_cols)

    fig = plt.figure(figsize=(12, 6))
    plot_configs = [
        ("greedy_mab", "CMAB Greedy Strategy", "#1f77b4", "-"),
        ("mean", "Ensemble Mean Baseline", "#ff7f0e", "--"),
    ]
    for col, label, color, style in plot_configs:
        strat_ret = long_cash_returns(strategies[col].values, strategies["target"].values)
        plt.plot(nav(strat_ret).values, label=label, color=color, linestyle=style, linewidth=2.5)

    plt.title("Performance Comparison: CMAB Selection vs. Passive Ensemble Mean", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Growth of $1 Investment (NAV)", fontsize=12)
    plt.xlabel("Time Steps (Out-of-Sample)", fontsize=12)
    plt.legend(frameon=True, fontsize=11, loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    if save_dir:
        save_figure(fig, f"{save_dir}/bandit_vs_mean_comparison.png")
    plt.show()
