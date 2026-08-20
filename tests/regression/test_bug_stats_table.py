"""
Regression tests for the stats-table bugs (originally MABSS_utility.py:1260-1323,
`compute_and_highlight_stats`), fixed in mabss.plots.tables.

Verified directly against the real corpus (SPY/UCB/heterogeneous,
experiments_cluster/.../TICKER-SPY.../stats_df.csv):

- `build_stats_table(..., periods_per_year=252)` reproduces the published
  stats_df.csv bit-exact (this is the legacy-formula reproduction path used by
  tools/build_stats_recompute_golden.py -- the paper-number lock).
- The published `volatility` row has average_ensemble=0.1309 < greedy
  (mean)=0.1389, but the ORIGINAL `highlight_max` bolded greedy (mean) --
  the WORSE (higher) volatility -- because it always bolds the row max
  regardless of metric direction.
- The published `max_drawdown` row has average_ensemble=greedy
  (mean)=-0.3044 but greedy (stddev)=~1e-16 (numerical noise near zero); the
  ORIGINAL bolded the stddev column, because -0.3044 < ~0 so the stddev
  "wins" .max() despite not being a strategy at all.
- The ORIGINAL `.format({'total_return': '{:.1%}', ...})` never matched
  anything (dict keys are column names; the real columns are
  'average_ensemble'/'greedy (mean)'/'greedy (stddev)', not metric names) --
  confirmed by inspecting the frozen module's LaTeX output directly.
"""

import numpy as np
import pandas as pd
import pytest

from mabss.plots.tables import build_stats_table, style_stats_table


def _synthetic_strategies():
    """Two greedy seeds with the SAME returns (deterministic stddev ~ 0) and an
    ensemble with distinctly lower volatility but also lower return -- built so
    the direction bugs are unambiguous to detect."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    ensemble_returns = rng.normal(0.0002, 0.003, size=300)  # low return, low vol
    greedy_returns = rng.normal(0.0006, 0.012, size=300)  # higher return, higher vol

    df = pd.DataFrame(index=idx)
    df["returns_average_ensemble"] = ensemble_returns
    df["returns_greedy_0"] = greedy_returns
    df["returns_greedy_1"] = greedy_returns  # identical -> stddev == 0 across seeds
    return df


def test_build_stats_table_matches_published_with_legacy_periods_per_year(canonical_bandit_dir=None):
    """Sanity check on synthetic data: total_return/annualized_return/sharpe use
    the standard formulas and total_return for the higher-return greedy series
    must exceed the ensemble's."""
    strategies = _synthetic_strategies()
    df = build_stats_table(strategies, periods_per_year=252)
    assert df.loc["total_return", "greedy (mean)"] > df.loc["total_return", "average_ensemble"]


def test_volatility_bolds_the_lower_value_not_the_higher():
    strategies = _synthetic_strategies()
    df = build_stats_table(strategies, periods_per_year=252)
    assert df.loc["volatility", "average_ensemble"] < df.loc["volatility", "greedy (mean)"]

    styled = style_stats_table(df)
    computed_styles = styled._compute()  # pandas internal, but stable enough for a direct check
    # Locate the 'volatility' row style output.
    row_idx = list(df.index).index("volatility")
    col_idx = list(df.columns).index("average_ensemble")
    style_at_ensemble = computed_styles.ctx.get((row_idx, col_idx), [])
    col_idx_greedy = list(df.columns).index("greedy (mean)")
    style_at_greedy = computed_styles.ctx.get((row_idx, col_idx_greedy), [])

    ensemble_bold = any("font-weight" in str(prop) for prop, _ in style_at_ensemble)
    greedy_bold = any("font-weight" in str(prop) for prop, _ in style_at_greedy)
    assert ensemble_bold, "the LOWER-volatility strategy (average_ensemble) should be bolded"
    assert not greedy_bold, "the HIGHER-volatility strategy (greedy (mean)) should NOT be bolded"


def test_stddev_column_is_never_bolded():
    strategies = _synthetic_strategies()
    df = build_stats_table(strategies, periods_per_year=252)
    # greedy_0 == greedy_1 by construction, so greedy (stddev) is ~0 for every metric.
    assert (df["greedy (stddev)"].abs() < 1e-9).all()

    styled = style_stats_table(df)
    computed_styles = styled._compute()
    stddev_col_idx = list(df.columns).index("greedy (stddev)")
    for row_idx in range(len(df.index)):
        cell_style = computed_styles.ctx.get((row_idx, stddev_col_idx), [])
        is_bold = any("font-weight" in str(prop) for prop, _ in cell_style)
        assert not is_bold, f"stddev column should never be bolded (row {df.index[row_idx]})"


def test_latex_formatting_is_actually_applied():
    """Regression test for the Styler.format-on-wrong-axis bug: the exported
    LaTeX must show percentages/fixed-precision floats, never raw repr floats
    like '0.079884'."""
    strategies = _synthetic_strategies()
    df = build_stats_table(strategies, periods_per_year=252)
    styled = style_stats_table(df)
    latex = styled.to_latex(convert_css=True)

    assert "%" in latex, "expected %-formatted metrics in the LaTeX output"
    # A raw, unformatted float would render with many decimal digits and no %.
    import re

    raw_float_pattern = re.compile(r"\b0\.\d{4,}\b")
    assert not raw_float_pattern.search(latex), f"found an unformatted raw float in LaTeX:\n{latex}"


@pytest.mark.needs_artifacts
def test_matches_published_stats_df_on_real_corpus(artifact_root):
    """End-to-end check against the real, committed SPY/UCB/heterogeneous
    directory: build_stats_table(periods_per_year=252) must reproduce
    stats_df.csv bit-exact, AND the styled LaTeX must now actually be
    formatted (unlike the original, unimported-display()-and-broken-format
    version)."""
    import importlib.util
    import pickle
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bandit_dirs = [
        d
        for d in artifact_root.glob("*")
        if "N_ARMS-60" in d.name and "POLICY-ucb" in d.name and "TICKER-SPY" in d.name
    ]
    if not bandit_dirs:
        pytest.skip("canonical SPY/UCB/N_ARMS-60 bandit dir not found")
    bandit_dir = bandit_dirs[0]
    pred_dirs = [
        d
        for d in artifact_root.glob("*")
        if "N_SEEDS_PER_ARCH-20" in d.name and "RNN-true" in d.name and "TICKER-SPY" in d.name
    ]
    if not pred_dirs:
        pytest.skip("matching SPY heterogeneous pred dir not found")
    pred_dir = pred_dirs[0]

    stats_path = bandit_dir / "stats_df.csv"
    if not stats_path.exists():
        pytest.skip("no stats_df.csv in canonical dir")
    published = pd.read_csv(stats_path, index_col=0)

    from tests.conftest import REPO_ROOT

    frozen_path = REPO_ROOT / "tests" / "_frozen" / "mabss_utility_frozen.py"
    spec = importlib.util.spec_from_file_location("mabss_utility_frozen", frozen_path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    from tqdm.auto import tqdm as tqdm_auto

    frozen.tqdm = tqdm_auto

    pct = pd.read_csv(pred_dir / "pctchange_preds.csv", index_col=0)
    pct.index = pd.to_datetime(pct.index, utc=True)
    with open(bandit_dir / "multirun_results.pkl", "rb") as f:
        multirun = pickle.load(f)  # trusted local corpus artifact

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, strategies = frozen.plot_multi_run_bandit(multirun, pct, {"N_ARMS": 60}, save_dir=None)
    plt.close("all")

    df = build_stats_table(strategies, periods_per_year=252)
    pd.testing.assert_frame_equal(df, published, check_exact=False, atol=1e-9)

    styled = style_stats_table(df)
    latex = styled.to_latex(convert_css=True)
    assert "%" in latex
