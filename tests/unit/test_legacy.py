"""Unit tests for mabss.experiments.legacy: the six-schema resolver, isolated
from the real corpus via synthetic tmp_path directories."""

import pytest

from mabss.experiments.ids import _render_id, experiment_id
from mabss.experiments.legacy import (
    BANDIT_SCHEMAS,
    PRED_SCHEMAS,
    AmbiguousExperimentError,
    ExperimentNotFoundError,
    LegacyIndex,
)


def _touch_dir(root, name):
    d = root / name
    d.mkdir(parents=True)
    return d


def test_resolves_current_bandit_schema(tmp_path):
    config = {
        "TICKER": "SPY",
        "START_DATE": "2000-01-01",
        "N_ARMS": 60,
        "WINDOW_SIZE": 252,
        "BANDIT_EMBEDDING": 15,
        "METRIC": "l1",
        "ALPHA": 1.0,
        "GAMMA": None,
        "EPISODES": 100,
        "POLICY": "ucb",
    }
    name = _render_id(config, BANDIT_SCHEMAS["B2_current"])
    _touch_dir(tmp_path, name)

    match = LegacyIndex(tmp_path).resolve(config, "bandit")
    assert match.schema == "B2_current"
    assert match.path.name == name


def test_resolves_cluster_schema_missing_alpha_gamma(tmp_path):
    """The B3 schema (no ALPHA/GAMMA at all) is what every experiments_cluster/
    directory actually uses -- exercised directly here, end-to-end against the
    real corpus in test_bug_cluster_id_resolution.py."""
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
    name = _render_id(config, BANDIT_SCHEMAS["B3_cluster_no_alpha_gamma"])
    _touch_dir(tmp_path, name)

    match = LegacyIndex(tmp_path).resolve(config, "bandit")
    assert match.schema == "B3_cluster_no_alpha_gamma"


def test_resolves_pred_schemas(tmp_path):
    homogeneous = {
        "TICKER": "QQQ",
        "START_DATE": "2000-01-01",
        "WINDOW_SIZE": 252,
        "MLP": True,
        "RNN": False,
        "CNN": False,
        "MODEL_EMBEDDING": 15,
        "EPOCHS": 10,
    }
    heterogeneous = {**homogeneous, "N_SEEDS_PER_ARCH": 20}

    _touch_dir(tmp_path, _render_id(homogeneous, PRED_SCHEMAS["P1_homogeneous"]))
    _touch_dir(tmp_path, _render_id(heterogeneous, PRED_SCHEMAS["P2_heterogeneous"]))

    index = LegacyIndex(tmp_path)
    assert index.resolve(homogeneous, "pred").schema == "P1_homogeneous"
    assert index.resolve(heterogeneous, "pred").schema == "P2_heterogeneous"


def test_not_found_reports_v2_id(tmp_path):
    config = {"TICKER": "NOPE", "N_ARMS": 1, "START_DATE": "2020-01-01"}
    with pytest.raises(ExperimentNotFoundError) as exc_info:
        LegacyIndex(tmp_path).resolve(config, "bandit")
    err = exc_info.value
    assert err.v2_id == experiment_id(config, "bandit")
    assert len(err.tried) == len(BANDIT_SCHEMAS)


def test_two_schemas_agreeing_on_the_same_directory_is_not_ambiguous(tmp_path):
    """B1_no_policy and B3_cluster_no_alpha_gamma differ only in whether
    ALPHA/GAMMA are part of the schema; when a config simply doesn't set
    those keys, both schemas render the identical string via the same
    `if k in config` filtering make_experiment_id has always used. That's a
    labeling question (which historical convention produced this name), not
    a resolution ambiguity -- there is exactly one directory either way, so
    resolve() must succeed, not raise."""
    config = {
        "TICKER": "TEST",
        "START_DATE": "2020-01-01",
        "N_ARMS": 5,
        "WINDOW_SIZE": 10,
        "BANDIT_EMBEDDING": 15,
        "METRIC": "l1",
        "EPISODES": 10,
    }
    assert _render_id(config, BANDIT_SCHEMAS["B1_no_policy"]) == _render_id(
        config, BANDIT_SCHEMAS["B3_cluster_no_alpha_gamma"]
    )
    name = _render_id(config, BANDIT_SCHEMAS["B1_no_policy"])
    _touch_dir(tmp_path, name)

    match = LegacyIndex(tmp_path).resolve(config, "bandit")
    assert match.path.name == name


def test_ambiguous_when_two_different_directories_both_match(tmp_path):
    """Genuine ambiguity: a config carrying both a legacy EPOCHS-named field
    (B0_epochs_field) and the current EPISODES field (B1_no_policy) -- e.g.
    merged from two historical config dicts -- renders two DIFFERENT ids.
    If directories exist for BOTH, resolve() must refuse to pick one rather
    than silently guess which real directory holds this config's results."""
    config = {
        "TICKER": "TEST",
        "START_DATE": "2020-01-01",
        "N_ARMS": 5,
        "WINDOW_SIZE": 10,
        "BANDIT_EMBEDDING": 15,
        "METRIC": "l1",
        "ALPHA": 1.0,
        "GAMMA": None,
        "EPOCHS": 10,
        "EPISODES": 20,
    }
    id_b0 = _render_id(config, BANDIT_SCHEMAS["B0_epochs_field"])
    id_b1 = _render_id(config, BANDIT_SCHEMAS["B1_no_policy"])
    assert id_b0 != id_b1
    _touch_dir(tmp_path, id_b0)
    _touch_dir(tmp_path, id_b1)

    with pytest.raises(AmbiguousExperimentError):
        LegacyIndex(tmp_path).resolve(config, "bandit")


def test_invalid_kind_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        LegacyIndex(tmp_path).resolve({}, "not_a_real_kind")


def test_missing_root_behaves_as_empty_index(tmp_path):
    """Constructing over a root that doesn't exist yet (e.g. a fresh
    experiments/ before any run) must not raise -- resolve() should just
    report not-found, same as an empty existing directory."""
    index = LegacyIndex(tmp_path / "does_not_exist_yet")
    with pytest.raises(ExperimentNotFoundError):
        index.resolve({"TICKER": "X"}, "bandit")


def test_index_is_read_only(tmp_path):
    config = {"TICKER": "SPY", "N_ARMS": 60, "EPISODES": 100, "POLICY": "ucb", "WINDOW_SIZE": 252, "BANDIT_EMBEDDING": 15, "METRIC": "l1", "START_DATE": "2000-01-01"}
    name = _render_id(config, BANDIT_SCHEMAS["B3_cluster_no_alpha_gamma"])
    _touch_dir(tmp_path, name)
    before = sorted(p.name for p in tmp_path.iterdir())

    index = LegacyIndex(tmp_path)
    index.resolve(config, "bandit")
    try:
        index.resolve({"TICKER": "NOPE"}, "bandit")
    except ExperimentNotFoundError:
        pass

    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after
