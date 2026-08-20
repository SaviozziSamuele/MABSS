"""
Tests for mabss.runner (regression coverage for the broken `run_bandit_seed`
constructor call, originally MABSS_utility.py:882-897, which made
`run_bandit_parallel` raise unconditionally).
"""

import numpy as np

from mabss.runner import run_bandit_multi, run_bandit_seed


def _tiny_bandit_inputs(rng, T=120, K=3, d=2):
    contexts = rng.normal(size=(T, K, d))
    rewards = rng.normal(size=(T, K))
    X_data = rng.normal(size=(T, K))
    Y_data = rng.normal(size=T)
    return contexts, rewards, X_data, Y_data


def test_run_bandit_seed_accepts_config_and_returns_legacy_dict():
    """The original's TypeError/KeyError on construction (alpha=/gamma= passed
    to a constructor expecting `config=`) must not happen anymore."""
    rng = np.random.default_rng(0)
    contexts, rewards, X_data, Y_data = _tiny_bandit_inputs(rng)
    config = {
        "METRIC": "l1",
        "N_ARMS": 3,
        "BANDIT_EMBEDDING": 2,
        "WINDOW_SIZE": 10,
        "POLICY": "ucb",
        "EPISODES": 1,
    }
    result = run_bandit_seed(0, contexts, X_data, Y_data, rewards, config)
    assert isinstance(result, dict)
    assert "best arms greedy" in result
    assert len(result["best arms greedy"]) > 0


def test_run_bandit_multi_returns_n_seeds_worth_of_results():
    rng = np.random.default_rng(1)
    contexts, rewards, X_data, Y_data = _tiny_bandit_inputs(rng)
    config = {
        "METRIC": "l1",
        "N_ARMS": 3,
        "BANDIT_EMBEDDING": 2,
        "WINDOW_SIZE": 10,
        "POLICY": "ucb",
        "EPISODES": 1,
    }
    results = run_bandit_multi(contexts, X_data, Y_data, rewards, config, n_seeds=3, n_jobs=1)
    assert set(results.keys()) == {"seed_0", "seed_1", "seed_2"}
    for res in results.values():
        assert isinstance(res, dict)
        assert "best arms greedy" in res


def test_run_bandit_multi_seeds_are_reproducible():
    """Same seed -> same np.random.seed(seed) -> same result, since
    run_bandit_seed seeds the legacy global RNG before running."""
    rng = np.random.default_rng(2)
    contexts, rewards, X_data, Y_data = _tiny_bandit_inputs(rng)
    config = {
        "METRIC": "l1",
        "N_ARMS": 3,
        "BANDIT_EMBEDDING": 2,
        "WINDOW_SIZE": 10,
        "POLICY": "ucb",
        "EPISODES": 1,
    }
    r1 = run_bandit_seed(7, contexts, X_data, Y_data, rewards, config)
    r2 = run_bandit_seed(7, contexts, X_data, Y_data, rewards, config)
    assert r1["best arms greedy"] == r2["best arms greedy"]
