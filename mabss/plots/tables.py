"""
Strategy performance table: build + style.

Ported from `compute_and_highlight_stats` (MABSS_utility.py:1260-1323), split
into `build_stats_table` (pure computation) and `style_stats_table` (styling),
using `mabss.metrics.strategy_stats` for the actual metric formulas (see that
module for the annualization/risk-free-rate fix).

Three bugs fixed here (plan Phase 4, items 2 and 3's table half):

1. **Styler.format never actually applied.** The original built `df` with
   metric names as the INDEX (`pd.DataFrame(df_data, index=metric_names)`) but
   called `.format({'total_return': '{:.1%}', ...})` -- a dict keyed by COLUMN
   name. Since the real columns are `'average_ensemble'`/`'greedy (mean)'`/
   `'greedy (stddev)'`, none of the format keys ever matched anything, so
   `stats_df.tex` (which goes straight into the paper) shipped raw floats
   (`0.079884` instead of `8.0%`). Fixed here by using
   `Styler.format(fmt, subset=pd.IndexSlice[[metric], :])` per metric row.

2. **`highlight_max` bolds the wrong thing for `volatility` and contaminates
   `max_drawdown`.** Verified directly on real corpus data
   (`experiments_cluster/.../TICKER-SPY.../stats_df.csv`): for `volatility`,
   `highlight_max` bolded `greedy (mean)` (0.139) over `average_ensemble`
   (0.131) -- i.e. it highlighted the strategy with HIGHER (worse) volatility
   as if it were the winner. For `max_drawdown` (stored as a negative
   fraction, e.g. -0.304 -- so higher/less-negative actually IS better, same
   direction as returns), the real bug is different: `greedy (stddev)` is
   ~1e-16 (numerical noise around zero) while the real strategies are ~-0.30,
   so the STDDEV column -- not a strategy at all -- won `.max()` and got
   bolded as the "best" max_drawdown. Fixed by (a) excluding `'greedy
   (stddev)'` from the winner comparison entirely (it's a dispersion
   statistic, never a candidate "best strategy"), and (b) using `.min()`
   instead of `.max()` specifically for `volatility` (the one metric in this
   table where lower is actually better in raw-value terms; `max_drawdown`'s
   negative-is-worse convention means `.max()` was already directionally
   correct for it, once the stddev column stops contaminating the comparison).

3. **`display()` is not called here at all** (was: MABSS_utility.py:1319,
   calling an unimported IPython builtin that only exists inside a live
   Jupyter kernel -- `NameError` in any plain script). `style_stats_table`
   returns the `Styler`; callers display it however's appropriate for their
   environment (`IPython.display.display(...)` in a notebook, `.to_string()`
   or `.to_latex()` elsewhere).
"""

from __future__ import annotations

import os

import pandas as pd

from mabss.metrics import strategy_stats

# Metrics where a LOWER raw value is better. Every other metric in this table
# (total_return, annualized_return, hit_ratio, max_drawdown, sharpe_ratio,
# calmar_ratio) is higher-is-better in raw-value terms -- including
# max_drawdown, which the pipeline stores as a negative fraction, so "closer
# to zero" (higher) genuinely does mean "smaller drawdown, better".
LOWER_IS_BETTER = frozenset({"volatility"})

# Never a candidate for "best strategy" -- it's a dispersion statistic across
# seeds, not a strategy result.
NON_STRATEGY_COLUMNS = frozenset({"greedy (stddev)"})

FORMATS = {
    "total_return": "{:.1%}",
    "annualized_return": "{:.1%}",
    "hit_ratio": "{:.1%}",
    "max_drawdown": "{:.1%}",
    "volatility": "{:.1%}",
    "sharpe_ratio": "{:.3f}",
    "calmar_ratio": "{:.3f}",
}


def build_stats_table(
    strategies: pd.DataFrame,
    strategies_greedy_cols: list[str] | None = None,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    `strategies`: a DataFrame with a `'returns_average_ensemble'` column and one
    or more `'returns_greedy_<seed>'` columns (exactly what
    `plot_multi_run_bandit`/`mabss.plots.diagnostics.plot_multi_run_bandit`
    build). Returns a DataFrame indexed by metric name with columns
    `['average_ensemble', 'greedy (mean)', 'greedy (stddev)']`.

    `periods_per_year=None` infers per-asset annualization from each column's
    DatetimeIndex (see `mabss.metrics.infer_periods_per_year`); pass `252`
    explicitly to reproduce the as-published (pre-fix) numbers.
    """
    if strategies_greedy_cols is None:
        strategies_greedy_cols = [c for c in strategies.columns if "returns_greedy_" in c]

    stats_greedy = [
        strategy_stats(strategies[c], periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)
        for c in strategies_greedy_cols
    ]
    stats_avg = strategy_stats(
        strategies["returns_average_ensemble"], periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
    )

    import numpy as np

    metric_names = [k for k in stats_avg if k != "periods_per_year"]
    stats_greedy_mean = {k: np.mean([s[k] for s in stats_greedy]) for k in metric_names}
    stats_greedy_std = {k: np.std([s[k] for s in stats_greedy]) for k in metric_names}

    df_data = {
        "average_ensemble": [stats_avg[m] for m in metric_names],
        "greedy (mean)": [stats_greedy_mean[m] for m in metric_names],
        "greedy (stddev)": [stats_greedy_std[m] for m in metric_names],
    }
    return pd.DataFrame(df_data, index=metric_names)


def style_stats_table(df: pd.DataFrame):
    """Returns a pandas Styler: bolds the best strategy per row (excluding the
    stddev column, direction-aware per LOWER_IS_BETTER) and applies %/float
    formatting per metric row. See module docstring for the two bugs fixed."""

    def highlight_best(row: pd.Series) -> list[str]:
        candidate_cols = [c for c in row.index if c not in NON_STRATEGY_COLUMNS]
        candidates = row[candidate_cols]
        best_val = candidates.min() if row.name in LOWER_IS_BETTER else candidates.max()
        return [
            "font-weight: bold" if (col in candidate_cols and v == best_val) else ""
            for col, v in row.items()
        ]

    styled = df.style.apply(highlight_best, axis=1)
    for metric, fmt in FORMATS.items():
        if metric in df.index:
            styled = styled.format(fmt, subset=pd.IndexSlice[[metric], :])
    return styled


def save_stats_table(df: pd.DataFrame, styled, save_dir) -> None:
    """Writes stats_df.csv (raw values) and stats_df.tex (formatted, via the
    Styler) into save_dir, creating it if needed -- the original never called
    os.makedirs here (MABSS_utility.py:1301-1302, 1320-1321)."""
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "stats_df.csv"))
    styled.to_latex(os.path.join(save_dir, "stats_df.tex"), convert_css=True)
