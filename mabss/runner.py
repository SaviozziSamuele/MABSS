"""Multi-seed parallel bandit execution."""

from __future__ import annotations

import os

import numpy as np
from joblib import Parallel, delayed

from .env import WalkForwardEnv


def run_bandit_seed(seed: int, contexts: np.ndarray, X_data: np.ndarray, Y_data: np.ndarray, rewards: np.ndarray, config: dict) -> dict:
    """
    Single bandit run for one seed. Ported from MABSS_utility.py:877-897, with
    the broken constructor call fixed (the original passed `alpha=`/`gamma=`
    keyword arguments to `WalkForwardEnvCMAB.__init__`, which takes neither --
    it takes a required `config` positional. This made `run_bandit_parallel`
    raise unconditionally; masked in the notebook because it loads cached
    `multirun_results.pkl` on a cache hit and never reaches this code path on
    a miss during normal use).

    Returns the legacy dict shape (`.to_legacy_dict()`) rather than the
    `BanditRunResult` dataclass `WalkForwardEnv.run()` returns natively, so
    results here are drop-in compatible with `mabss.plots` (which consumes
    `results['best arms greedy']`-style dict access) and with the cached
    `multirun_results.pkl` files already on disk, which store exactly this
    shape.
    """
    np.random.seed(seed)
    env = WalkForwardEnv(
        context=contexts,
        X_data=X_data,
        Y_data=Y_data,
        rewards=rewards,
        reward_metric=config["METRIC"],
        k=config["N_ARMS"],
        d=config["BANDIT_EMBEDDING"],
        config=config,
        window_size=config["WINDOW_SIZE"],
        policy_type=config["POLICY"],
        shuffle=True,
        reset=True,
    )
    return env.run(episodes=config["EPISODES"]).to_legacy_dict()


def run_bandit_multi(
    contexts_matrix: np.ndarray,
    X_data: np.ndarray,
    Y_data: np.ndarray,
    rewards_matrix: np.ndarray,
    config: dict,
    n_seeds: int = 20,
    n_jobs: int | None = None,
) -> dict[str, dict]:
    """
    Runs `n_seeds` independent bandit simulations (seeds `0..n_seeds-1`) in
    parallel, returning `{"seed_0": {...}, "seed_1": {...}, ...}` -- same
    contract as the original `run_bandit_parallel` (MABSS_utility.py:899-916).

    Uses `joblib.Parallel` instead of `multiprocessing.Pool` + `functools.partial`.
    The original bound `contexts_matrix`/`X_data`/`Y_data`/`rewards_matrix` via
    `partial`, which `Pool.map` then pickles and ships to EVERY worker process
    individually -- for a `(T, K, d)` contexts array (e.g. ~40 MB at K=60,
    d=15, T~5500), that's ~40 MB pickled per worker, x N workers, with no
    sharing. `joblib.Parallel`'s default `loky` backend automatically memmaps
    array arguments above a size threshold, so all workers share the same
    on-disk-backed array instead of each getting a private pickled copy.

    Also, unlike `mp.Pool.map`, `joblib.Parallel` surfaces a worker exception
    with its original traceback re-raised in the calling process, rather than
    an opaque remote-process failure -- relevant here because until the fix
    above, this whole code path failed 100% of the time and that fact was easy
    to miss.
    """
    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, n_seeds)

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_bandit_seed)(seed, contexts_matrix, X_data, Y_data, rewards_matrix, config)
        for seed in range(n_seeds)
    )
    return {f"seed_{i}": res for i, res in enumerate(results)}
