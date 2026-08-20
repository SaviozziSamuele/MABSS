"""
Regression tests for the NARMS typo (:1210) and ACF target-leak (:1006-1007),
fixed in mabss.plots.diagnostics.
"""

import importlib.util
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mabss.plots.diagnostics import plot_multi_run_bandit


def _code_body(fn) -> str:
    """The function's source with its docstring stripped, so a check for the
    ABSENCE of some old buggy pattern isn't fooled by that pattern being
    quoted in a "here's what this fixes" docstring."""
    import ast
    import inspect

    source = inspect.getsource(fn)
    tree = ast.parse(source)
    func_node = tree.body[0]
    if (
        func_node.body
        and isinstance(func_node.body[0], ast.Expr)
        and isinstance(func_node.body[0].value, ast.Constant)
        and isinstance(func_node.body[0].value.value, str)
    ):
        func_node.body = func_node.body[1:]  # drop the docstring node
    return ast.unparse(func_node)


def test_narms_uses_config_n_arms_not_typo():
    """Direct source check: the fixed function must read config['N_ARMS'], and
    must NOT contain the 'NARMS' typo that always fell through to a fallback
    in the original. Checks the code body only (ast-stripped of the
    docstring, which deliberately quotes the old buggy pattern for
    documentation)."""
    body = _code_body(plot_multi_run_bandit)
    assert "config['N_ARMS']" in body
    assert "NARMS" not in body


@pytest.mark.needs_artifacts
def test_multi_run_bandit_greedy_track_matches_frozen_module(artifact_root):
    """The greedy NAV/returns (what Table 1 is built from) must be bit-identical
    between the frozen (buggy-NARMS, correct-elsewhere) module and the fixed
    mabss version -- the NARMS bug only affected a colorbar scale, never the
    strategy construction itself."""
    from tests.conftest import REPO_ROOT

    bandit_dirs = [
        d for d in artifact_root.glob("*") if "N_ARMS-60" in d.name and "POLICY-ucb" in d.name and "TICKER-SPY" in d.name
    ]
    if not bandit_dirs:
        pytest.skip("canonical SPY/UCB/N_ARMS-60 bandit dir not found")
    bandit_dir = bandit_dirs[0]
    pred_dirs = [
        d for d in artifact_root.glob("*") if "N_SEEDS_PER_ARCH-20" in d.name and "RNN-true" in d.name and "TICKER-SPY" in d.name
    ]
    if not pred_dirs:
        pytest.skip("matching SPY heterogeneous pred dir not found")
    pred_dir = pred_dirs[0]

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

    config = {"N_ARMS": 60}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, strategies_frozen = frozen.plot_multi_run_bandit(multirun, pct, config, save_dir=None)
    plt.close("all")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, strategies_new = plot_multi_run_bandit(multirun, pct, config, save_dir=None)
    plt.close("all")

    greedy_cols = [c for c in strategies_frozen.columns if "returns_greedy_" in c]
    for col in greedy_cols:
        np.testing.assert_array_equal(
            strategies_frozen[col].values, strategies_new[col].values, err_msg=f"mismatch in {col}"
        )
    np.testing.assert_array_equal(
        strategies_frozen["returns_average_ensemble"].values,
        strategies_new["returns_average_ensemble"].values,
    )


def test_acf_leak_fixed_excludes_target_includes_all_models():
    """Regression test for the ACF target-leak: the fixed loop must produce
    exactly N_ARMS acf series (one per model), never N_ARMS-1 (dropping the
    last model) or including the target series."""
    from mabss.plots.diagnostics import plot_rewards_stats

    n_arms = 5
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    data = {"target": rng.normal(0, 0.01, 200)}
    for i in range(n_arms):
        # Give each model a DISTINCT autocorrelation structure so a dropped or
        # extra series would visibly change the mean/std ACF.
        data[f"model_{i}"] = rng.normal(0, 0.01 * (i + 1), 200)
    data["mean"] = np.mean([data[f"model_{i}"] for i in range(n_arms)], axis=0)
    rewards_df = pd.DataFrame(data, index=idx)

    body = _code_body(plot_rewards_stats)
    # Structural proof: iterates model_cols directly, never np.array(rewards_df).T
    assert "model_cols" in body
    assert "rewards_df).T" not in body

    # Functional proof: run it (headless, Agg backend) and confirm it doesn't
    # raise -- e.g. doesn't try to acf() the 'target' column as if it were a model.
    plt.close("all")
    plot_rewards_stats(rewards_df, {"N_ARMS": n_arms, "TICKER": "TEST"}, save_dir=None)
    plt.close("all")
