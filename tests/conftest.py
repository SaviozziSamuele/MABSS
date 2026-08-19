"""Shared fixtures for the MABSS test suite."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
FROZEN_PATH = REPO_ROOT / "tests" / "_frozen" / "mabss_utility_frozen.py"


@pytest.fixture(scope="session")
def frozen_module():
    """
    The Phase-0 differential oracle: MABSS_utility.py at the moment it first
    compiled, frozen and SHA-pinned (see test_frozen_is_pinned.py). Parity tests
    diff new/refactored code's output against this module's output on identical
    inputs. Never mutate anything through this fixture.
    """
    spec = importlib.util.spec_from_file_location("mabss_utility_frozen", FROZEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rng():
    """A fresh, independent numpy Generator for tests that need randomness."""
    return np.random.default_rng(0)


@pytest.fixture
def tiny_series():
    """
    A small, deterministic synthetic price series for fast unit/regression tests.
    600 bars is enough to exercise walk_forward_training with a small window
    (e.g. window_size=60, embedding=15, train_windows=4) in well under a second.
    """
    n = 600
    rng_local = np.random.default_rng(42)
    # Geometric random walk, strictly positive, mildly trending.
    log_returns = rng_local.normal(loc=0.0003, scale=0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    index = pd.bdate_range("2015-01-01", periods=n, tz="America/New_York")
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture
def tiny_config():
    """A minimal CONFIG dict sized for the tiny_series fixture."""
    return dict(
        TICKER="SYNTH",
        START_DATE="2015-01-01",
        WINDOW_SIZE=60,
        MODEL_EMBEDDING=15,
        BANDIT_EMBEDDING=15,
        EPOCHS=2,
        MLP=True,
        RNN=False,
        CNN=False,
        N_SEEDS_PER_ARCH=2,
        METRIC="l1",
        PROB_SCALE=50.0,
        EPISODES=2,
        POLICY="ucb",
        ALPHA=1.0,
        GAMMA=None,
        RANDOM_SEED=42,
    )


def _find_artifact_root() -> Path | None:
    for name in ("experiments_cluster", "experiments"):
        candidate = REPO_ROOT / name
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


@pytest.fixture(scope="session")
def artifact_root():
    """
    Root of the committed experiment-result corpus (experiments_cluster/ preferred,
    falling back to experiments/). Tests marked `needs_artifacts` should request this
    fixture and will be skipped automatically if the corpus isn't present (e.g. in a
    shallow/sparse checkout).
    """
    root = _find_artifact_root()
    if root is None:
        pytest.skip("no experiments/ or experiments_cluster/ artifact corpus found on disk")
    return root


@pytest.fixture(scope="session")
def canonical_bandit_dir(artifact_root):
    """
    The canonical golden bandit-result directory used across characterization tests:
    a heterogeneous 60-arm UCB run on SPY under experiments_cluster/.
    """
    matches = [
        d
        for d in artifact_root.glob("*")
        if d.is_dir()
        and "N_ARMS-60" in d.name
        and "POLICY-ucb" in d.name
        and "TICKER-SPY" in d.name
    ]
    if not matches:
        pytest.skip("canonical N_ARMS-60/POLICY-ucb/TICKER-SPY bandit dir not found")
    return matches[0]
