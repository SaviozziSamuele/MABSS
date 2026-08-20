"""
Signal -> strategy-return -> NAV construction.

The original codebase repeated this exact pattern nine times across
MABSS_utility.py (lines 956, 972, 1099, 1144, 1173, 1226, 1448, 1574, 1644):

    strat_ret = target_slice.copy()
    strat_ret[signal < 0] = 0.0

i.e. "go long the target's realized return when the predicted signal is
positive, otherwise sit in cash (0% return)". This is the single piece of logic
that turns every model/ensemble/bandit prediction into every published return
number in the paper -- worth having exactly once, tested once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def long_cash_returns(signal, target_returns) -> np.ndarray:
    """
    Long/cash strategy: realize `target_returns[t]` whenever `signal[t] >= 0`
    (predicted a non-negative move), otherwise 0% (flat/cash) that period.

    No lookahead: `signal[t]` must already be a same-period-t forecast made
    using only information available before period t closed (this function
    does not itself enforce that -- it's a property of what's passed in from
    walk-forward, out-of-sample predictions).
    """
    signal = np.asarray(signal)
    target_returns = np.asarray(target_returns, dtype=float).copy()
    if signal.shape != target_returns.shape:
        raise ValueError(f"shape mismatch: signal {signal.shape} vs target_returns {target_returns.shape}")
    target_returns[signal < 0] = 0.0
    return target_returns


def nav(strategy_returns, index=None) -> pd.Series:
    """Cumulative NAV (starting at 1.0) from a series of periodic returns."""
    values = np.cumprod(1 + np.asarray(strategy_returns))
    return pd.Series(values, index=index)


def ensemble_signal(data_slice: pd.DataFrame, model_cols: list[str] | None = None) -> np.ndarray:
    """
    The "average ensemble" signal: the 'mean' column if present, otherwise the
    row-wise mean of the model_* columns. Matches `plot_multi_run_bandit`'s
    fallback (:1169-1172) exactly, but as a whitelist over `model_cols` rather
    than "everything except the first column" when 'mean' is absent -- slightly
    more defensive against a stray non-model column surviving a CSV round-trip.
    """
    if "mean" in data_slice.columns:
        return data_slice["mean"].values
    if model_cols is None:
        model_cols = [c for c in data_slice.columns if c.startswith("model_")]
    return data_slice[model_cols].mean(axis=1).values


def greedy_predictions(data_slice: pd.DataFrame, arms: list[int], model_cols: list[str]) -> list[float]:
    """
    The prediction actually used at each step under the greedy CMAB strategy:
    `model_cols[arms[t]]`'s prediction at row t.

    Ported from the original's `data_slice.iloc[ii, arm + 1]` (e.g.
    MABSS_utility.py:1140), which hardcoded "column 0 is target, columns 1..K
    are the arms in index order" -- this version looks columns up BY NAME, so
    it's correct even if the frame's column order ever changes (e.g. a 'mean'
    column inserted before the model_i columns).
    """
    return [data_slice.iloc[ii][model_cols[arm]] for ii, arm in enumerate(arms)]


def build_multirun_strategies(
    results_multi: dict, pct_change_preds: pd.DataFrame
) -> tuple[str | None, pd.DataFrame | None]:
    """
    Pure (no matplotlib) extraction of the `strategies`-DataFrame-building half
    of `mabss.plots.diagnostics.plot_multi_run_bandit` -- the per-seed greedy
    CMAB return/NAV columns and the deterministic static average-ensemble
    baseline column that Table 1, the paired significance tests
    (`mabss.significance`), and the fee-sensitivity analysis are all computed
    from. `plot_multi_run_bandit` is now a thin wrapper around this function;
    this exists so new analysis code doesn't need to import matplotlib or
    duplicate the seed-looping logic.

    Returns `(best_seed, strategies)`:
      best_seed: the results_multi key with the highest full-horizon greedy
        total return (None if results_multi is empty or no seed's arm
        sequence length matches the common horizon T).
      strategies: DataFrame indexed like `pct_change_preds.iloc[-T:]`, columns
        'target', 'greedy_<seed>', 'returns_greedy_<seed>', 'nav_greedy_<seed>'
        (one triple per valid seed), 'returns_average_ensemble',
        'nav_average_ensemble'. `(None, None)` if `results_multi` is empty or
        no seed's length matches T.
    """
    if not results_multi:
        return None, None

    first_seed = next(iter(results_multi))
    T = len(results_multi[first_seed]["best arms greedy"])

    data_slice = pct_change_preds.iloc[-T:].copy()
    target_slice = data_slice["target"].values
    model_cols = [c for c in data_slice.columns if c.startswith("model_")]

    strategies = pd.DataFrame({"target": target_slice}, index=data_slice.index)

    tot_returns = []
    for seed in results_multi:
        arms_greedy = results_multi[seed]["best arms greedy"]
        if len(arms_greedy) != T:
            continue

        greedy_preds = greedy_predictions(data_slice, arms_greedy, model_cols)
        strategies[f"greedy_{seed}"] = greedy_preds

        strat_ret = long_cash_returns(np.array(greedy_preds), target_slice)
        strategies[f"returns_greedy_{seed}"] = strat_ret
        strategies[f"nav_greedy_{seed}"] = nav(strat_ret).values

        tot_returns.append(np.prod(1 + strat_ret) - 1)

    if not tot_returns:
        return None, None

    seed_list = [s for s in results_multi if len(results_multi[s]["best arms greedy"]) == T]
    best_seed = seed_list[np.argmax(tot_returns)]

    mean_signal = ensemble_signal(data_slice, model_cols)
    ensemble_ret = long_cash_returns(mean_signal, target_slice)
    strategies["returns_average_ensemble"] = ensemble_ret
    strategies["nav_average_ensemble"] = nav(ensemble_ret).values

    return best_seed, strategies


def costed_long_cash_returns(signal, target_returns, fee_bps: float = 0.0) -> np.ndarray:
    """
    Long/cash strategy with a proportional one-way transaction cost charged
    whenever the BINARY POSITION (`1[signal_t >= 0]`) switches relative to
    `t-1` -- i.e. whenever the strategy enters or exits the market. Switching
    which arm/predictor is driving the signal does NOT itself trigger a cost
    (it's the same single underlying instrument in every case; only
    long<->cash transitions are trades).

    `fee_bps`: one-way cost in basis points, subtracted from the return of the
    period in which the switch takes effect. No cost is charged at t=0
    (entering the very first position from an assumed-flat start is a free
    modeling convention, consistent with `nav()` starting at 1.0 everywhere
    else in this codebase).

    `fee_bps=0.0` is mathematically equivalent to `long_cash_returns`'s output
    (multiplying/subtracting by exactly 0.0 cannot perturb any finite IEEE-754
    value) but is a fully separate code path from `long_cash_returns` by
    design -- kept out of the Table-1 pipeline (`build_multirun_strategies`,
    `mabss.plots.tables.build_stats_table`) entirely, so "does this touch the
    published numbers" stays structurally obvious.
    """
    signal = np.asarray(signal)
    target_returns = np.asarray(target_returns, dtype=float).copy()
    if signal.shape != target_returns.shape:
        raise ValueError(f"shape mismatch: signal {signal.shape} vs target_returns {target_returns.shape}")

    position = (signal >= 0).astype(float)
    returns = target_returns * position

    fee = fee_bps / 10_000.0
    if fee:
        switched = np.zeros_like(position, dtype=bool)
        switched[1:] = position[1:] != position[:-1]
        returns[switched] -= fee

    return returns


def build_multirun_strategies_costed(
    results_multi: dict, pct_change_preds: pd.DataFrame, fee_bps: float = 0.0
) -> tuple[str | None, pd.DataFrame | None]:
    """
    Sibling of `build_multirun_strategies` with the identical column contract
    ('target', 'greedy_<seed>', 'returns_greedy_<seed>', 'nav_greedy_<seed>',
    'returns_average_ensemble', 'nav_average_ensemble'), built with
    `costed_long_cash_returns(..., fee_bps=fee_bps)` instead of
    `long_cash_returns`. Deliberately NOT implemented by adding a `fee_bps`
    parameter to `build_multirun_strategies` itself -- see
    `costed_long_cash_returns`'s docstring.
    """
    if not results_multi:
        return None, None

    first_seed = next(iter(results_multi))
    T = len(results_multi[first_seed]["best arms greedy"])

    data_slice = pct_change_preds.iloc[-T:].copy()
    target_slice = data_slice["target"].values
    model_cols = [c for c in data_slice.columns if c.startswith("model_")]

    strategies = pd.DataFrame({"target": target_slice}, index=data_slice.index)

    tot_returns = []
    for seed in results_multi:
        arms_greedy = results_multi[seed]["best arms greedy"]
        if len(arms_greedy) != T:
            continue

        greedy_preds = greedy_predictions(data_slice, arms_greedy, model_cols)
        strategies[f"greedy_{seed}"] = greedy_preds

        strat_ret = costed_long_cash_returns(np.array(greedy_preds), target_slice, fee_bps=fee_bps)
        strategies[f"returns_greedy_{seed}"] = strat_ret
        strategies[f"nav_greedy_{seed}"] = nav(strat_ret).values

        tot_returns.append(np.prod(1 + strat_ret) - 1)

    if not tot_returns:
        return None, None

    seed_list = [s for s in results_multi if len(results_multi[s]["best arms greedy"]) == T]
    best_seed = seed_list[np.argmax(tot_returns)]

    mean_signal = ensemble_signal(data_slice, model_cols)
    ensemble_ret = costed_long_cash_returns(mean_signal, target_slice, fee_bps=fee_bps)
    strategies["returns_average_ensemble"] = ensemble_ret
    strategies["nav_average_ensemble"] = nav(ensemble_ret).values

    return best_seed, strategies
