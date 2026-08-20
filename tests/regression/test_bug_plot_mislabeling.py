"""
Regression tests for two plotting bugs fixed in mabss.plots.paper:

1. `plot_bandit_strategy_comparison` labeled its line "CMAB Greedy Strategy"
   but plotted `results['weighted predictions']` (originally
   MABSS_utility.py:1637/1643). Only correct by coincidence for one-hot
   policies (LinUCB/Thompson); wrong for Softmax, where weighted-by-
   probability genuinely differs from argmax-greedy.
2. `plot_multiticker_grid_plot1` hardcoded its 4-ticker panel list inside the
   function body (originally MABSS_utility.py:1540) instead of taking it as a
   parameter.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mabss.plots.paper import DEFAULT_GRID_TICKERS, plot_bandit_strategy_comparison, plot_multiticker_grid_plot1
from mabss.strategies import greedy_predictions


def _softmax_like_results_and_preds():
    """Builds a tiny results/pct_change_preds pair where 'best arms greedy'
    (argmax) and 'weighted predictions' (probability-weighted) DIVERGE --
    exactly the situation where the original mislabeling bug would show a
    visibly wrong line for a non-one-hot policy like Softmax."""
    n = 20
    idx = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(0)

    model_0 = rng.normal(0.01, 0.001, n)  # consistently positive
    model_1 = rng.normal(-0.01, 0.001, n)  # consistently negative
    target = rng.normal(0.0, 0.005, n)

    pct_change_preds = pd.DataFrame(
        {"target": target, "model_0": model_0, "model_1": model_1, "mean": (model_0 + model_1) / 2},
        index=idx,
    )

    # Greedy always picks arm 0 (the positive one); "weighted" is a soft blend
    # that is NOT the same series -- this is what a real Softmax policy's
    # `np.dot(probs, X_data_test[t])` would produce.
    best_arms_greedy = [0] * n
    weighted_predictions = 0.5 * model_0 + 0.5 * model_1  # deliberately != model_0

    results = {"best arms greedy": best_arms_greedy, "weighted predictions": weighted_predictions}
    return results, pct_change_preds


def test_greedy_strategy_now_uses_argmax_not_weighted():
    results, pct_change_preds = _softmax_like_results_and_preds()
    model_cols = ["model_0", "model_1"]

    expected_greedy_signal = greedy_predictions(pct_change_preds, results["best arms greedy"], model_cols)
    # By construction this must differ from the (wrongly-used-in-the-original)
    # weighted predictions -- otherwise this test wouldn't be exercising
    # anything.
    assert not np.allclose(expected_greedy_signal, results["weighted predictions"])

    plt.close("all")
    plot_bandit_strategy_comparison(results, pct_change_preds, save_dir=None)
    fig = plt.gcf()
    line = fig.axes[0].get_lines()[0]  # the 'CMAB Greedy Strategy' line
    y_plotted = line.get_ydata()

    # Reconstruct what SHOULD have been plotted: NAV of long/cash on the true
    # greedy signal.
    from mabss.strategies import long_cash_returns, nav

    expected_nav = nav(long_cash_returns(np.array(expected_greedy_signal), pct_change_preds["target"].values)).values
    np.testing.assert_allclose(y_plotted, expected_nav)
    plt.close("all")


def test_grid_tickers_is_now_a_parameter():
    ticker_data_dict = {}  # empty -> every panel just gets hidden, but the call must accept the kwarg
    plt.close("all")
    custom_tickers = ["SPY", "TLT"]
    plot_multiticker_grid_plot1(ticker_data_dict, grid_tickers=custom_tickers, save_dir=None)
    plt.close("all")
    # Default still matches the original hardcoded list, for backward compat.
    assert DEFAULT_GRID_TICKERS == ["SPY", "BTC-USD", "EURUSD=X", "GC=F"]
