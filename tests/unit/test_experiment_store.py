"""Unit/regression tests for mabss.experiments.store.ExperimentStore."""

import numpy as np
import pandas as pd
import pytest

from mabss.experiments.ids import experiment_id, make_experiment_id
from mabss.experiments.store import ExperimentStore


def test_construct_does_not_create_directories(tmp_path):
    """Regression test: the original ExperimentLoader.__init__ called
    .mkdir(parents=True, exist_ok=True) unconditionally, so merely
    constructing a loader (e.g. to check `.exists(...)`) created directories
    on disk. This is what produced several near-empty orphan directories in
    the real corpus."""
    config = {"TICKER": "TEST", "START_DATE": "2020-01-01", "WINDOW_SIZE": 10}
    store = ExperimentStore(config, tmp_path)
    assert not store.pred_dir.exists()
    assert not store.bandit_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_save_creates_directories_only_when_writing(tmp_path):
    # N_ARMS/POLICY are bandit-only key fields and MODEL_EMBEDDING/EPOCHS are
    # pred-only, so this config makes pred_id and bandit_id genuinely distinct
    # (a config with only the fields common to both, e.g. just TICKER/
    # START_DATE/WINDOW_SIZE, makes pred_dir == bandit_dir, which would make
    # this assertion meaningless).
    config = {
        "TICKER": "TEST",
        "START_DATE": "2020-01-01",
        "WINDOW_SIZE": 10,
        "MODEL_EMBEDDING": 5,
        "EPOCHS": 2,
        "N_ARMS": 3,
        "POLICY": "ucb",
    }
    store = ExperimentStore(config, tmp_path)
    assert store.pred_dir != store.bandit_dir
    store.save({"contexts": np.arange(6).reshape(2, 3)})
    assert store.bandit_dir.exists()
    assert not store.pred_dir.exists()  # only the artifact actually saved gets a dir


def test_save_then_load_returns_equal_object_including_index_dtype(tmp_path):
    """Regression test: the original save() never populated self._cache, so a
    save-then-load round trip could return an object with a different index
    dtype than what was just saved (string index on re-read vs. whatever dtype
    the in-memory object had). Here it must be the same DatetimeIndex both ways."""
    config = {"TICKER": "TEST", "START_DATE": "2020-01-01", "WINDOW_SIZE": 10}
    store = ExperimentStore(config, tmp_path)
    idx = pd.date_range("2020-01-01", periods=5, tz="UTC")
    df = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)

    store.save({"preds": df})
    reloaded = store.load("preds")
    # check_freq=False: date_range's index carries a `freq` attribute that a
    # CSV round-trip can't preserve; the values (what actually matters here)
    # must still match exactly.
    pd.testing.assert_frame_equal(df, reloaded, check_freq=False)
    assert isinstance(reloaded.index, pd.DatetimeIndex)

    # A second store instance (fresh cache) reading from disk must agree too.
    store2 = ExperimentStore(config, tmp_path)
    from_disk = store2.load("preds")
    pd.testing.assert_frame_equal(df, from_disk, check_freq=False)


def test_load_parses_datetime_index_for_all_three_frame_keys(tmp_path):
    """Regression test: the original only parsed dates for 'pctchange_preds',
    leaving 'preds' and 'rewards_df' with an object (string) index."""
    config = {"TICKER": "TEST", "START_DATE": "2020-01-01", "WINDOW_SIZE": 10}
    store = ExperimentStore(config, tmp_path)
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    df = pd.DataFrame({"target": [1.0, 2.0, 3.0]}, index=idx)

    for key in ("preds", "pctchange_preds", "rewards_df"):
        store.save({key: df})
        loaded = store.load(key, force_reload=True)
        assert isinstance(loaded.index, pd.DatetimeIndex), f"{key} did not get a DatetimeIndex"


def test_readonly_mode_rejects_save(tmp_path):
    config = {"TICKER": "TEST", "START_DATE": "2020-01-01", "WINDOW_SIZE": 10}
    store = ExperimentStore(config, tmp_path, mode="r")
    with pytest.raises(PermissionError):
        store.save({"contexts": np.zeros((2, 2))})
    assert not store.bandit_dir.exists()


def test_load_unknown_key_raises_keyerror(tmp_path):
    config = {"TICKER": "TEST", "START_DATE": "2020-01-01", "WINDOW_SIZE": 10}
    store = ExperimentStore(config, tmp_path)
    with pytest.raises(KeyError):
        store.load("not_a_real_key")


def test_atomic_write_leaves_no_partial_file_on_failure(tmp_path, monkeypatch):
    config = {"TICKER": "TEST", "START_DATE": "2020-01-01", "WINDOW_SIZE": 10}
    store = ExperimentStore(config, tmp_path)

    def boom(*a, **k):
        raise RuntimeError("simulated disk failure mid-write")

    monkeypatch.setattr(np, "save", boom)
    with pytest.raises(RuntimeError):
        store.save({"contexts": np.zeros((2, 2))})
    # No stray temp file left behind, and the real path was never created.
    assert not store.paths["contexts"].exists()
    leftovers = list(store.bandit_dir.glob("*.tmp")) if store.bandit_dir.exists() else []
    assert leftovers == []


@pytest.mark.needs_artifacts
def test_reads_real_corpus_readonly(canonical_bandit_dir):
    """Points a read-only store at the actual committed canonical bandit
    directory and confirms it can address every artifact ExperimentLoader
    could, using the same filenames (proving the port didn't drift)."""
    # The canonical dir's config is known directly (it's what the `canonical_bandit_dir`
    # fixture selects by name) -- naively splitting the directory name on "_" is
    # NOT safe here, since several field names (START_DATE, N_ARMS, MODEL_EMBEDDING,
    # BANDIT_EMBEDDING, WINDOW_SIZE, N_SEEDS_PER_ARCH) contain underscores
    # themselves, which is exactly the kind of ambiguity a real legacy-ID parser
    # (plan Phase 3, out of scope here) has to resolve carefully.
    config = {
        "TICKER": "SPY",
        "START_DATE": "2000-01-01",
        "N_ARMS": 60,
        "WINDOW_SIZE": 252,
        "BANDIT_EMBEDDING": 15,
        "METRIC": "l1",
        "EPISODES": 100,
        "POLICY": "ucb",
    }
    store = ExperimentStore(config, canonical_bandit_dir.parent, mode="r")
    assert store.bandit_dir == canonical_bandit_dir
    assert store.exists("contexts")
    assert store.exists("multirun_results")
    contexts = store.load("contexts")
    assert contexts.ndim == 3


def test_v2_experiment_id_distinguishes_homo_and_hetero_pools():
    """Regression test: the LEGACY make_experiment_id's bandit-ID key set
    includes N_ARMS but not MLP/RNN/CNN/N_SEEDS_PER_ARCH, so a homogeneous
    45-model pool and a heterogeneous 15x3 pool BOTH resolve to N_ARMS-45 and
    collide on the same directory. The new experiment_id hashes the whole
    config and must NOT collide."""
    homo = {"TICKER": "SPY", "MLP": True, "RNN": False, "CNN": False, "N_SEEDS_PER_ARCH": 45, "N_ARMS": 45}
    hetero = {"TICKER": "SPY", "MLP": True, "RNN": True, "CNN": True, "N_SEEDS_PER_ARCH": 15, "N_ARMS": 45}

    legacy_homo = make_experiment_id(homo)[1]
    legacy_hetero = make_experiment_id(hetero)[1]
    # Documents the legacy collision (not a bug in the test -- a bug in the
    # legacy scheme, kept for corpus-addressing compatibility).
    # (Not asserted equal here to avoid over-fitting to exact string content;
    # the real point is the v2 scheme below.)
    del legacy_homo, legacy_hetero

    v2_homo = experiment_id(homo, kind="bandit")
    v2_hetero = experiment_id(hetero, kind="bandit")
    assert v2_homo != v2_hetero


def test_v2_experiment_id_stable_across_dict_key_ordering():
    a = {"TICKER": "SPY", "ALPHA": 1.0, "POLICY": "ucb"}
    b = {"POLICY": "ucb", "TICKER": "SPY", "ALPHA": 1.0}
    assert experiment_id(a, kind="bandit") == experiment_id(b, kind="bandit")
