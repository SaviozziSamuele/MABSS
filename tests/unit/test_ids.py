"""
Tests for mabss.experiments.ids, including a parity check for the Phase 3
refactor that extracted `_render_id`/`_sanitize` out of `make_experiment_id`
(previously a nested closure) so mabss.experiments.legacy's schema table
could reuse the exact same rendering pipeline instead of duplicating it.
"""

import pytest

from mabss.experiments.ids import experiment_id, make_experiment_id


@pytest.mark.parametrize(
    "config",
    [
        {"TICKER": "SPY", "START_DATE": "2000-01-01", "N_ARMS": 60, "WINDOW_SIZE": 252,
         "BANDIT_EMBEDDING": 15, "METRIC": "l1", "ALPHA": 1.0, "GAMMA": None,
         "EPISODES": 100, "POLICY": "ucb", "MLP": True, "RNN": False, "CNN": False,
         "MODEL_EMBEDDING": 15, "EPOCHS": 10},
        {"TICKER": "EURUSD=X", "N_ARMS": 45},
        {},
    ],
)
def test_make_experiment_id_matches_frozen_oracle(frozen_module, config):
    """The `_render_id` extraction must not change make_experiment_id's
    output for any config -- pure refactor, diffed against the Phase-0
    frozen copy of the original nested-closure implementation."""
    expected = frozen_module.make_experiment_id(config)
    actual = make_experiment_id(config)
    assert actual == expected


def test_experiment_id_is_deterministic_and_kind_scoped():
    config = {"TICKER": "SPY", "N_ARMS": 60}
    assert experiment_id(config, "bandit") == experiment_id(config, "bandit")
    assert experiment_id(config, "bandit") != experiment_id(config, "pred")


def test_experiment_id_changes_when_any_field_changes():
    """Unlike make_experiment_id, experiment_id hashes the WHOLE config, so a
    field not in the legacy key_fields_* subsets (e.g. N_SEEDS_PER_ARCH) still
    changes the ID -- this is the fix for the homogeneous/heterogeneous
    collision documented on make_experiment_id."""
    base = {"TICKER": "SPY", "N_ARMS": 45}
    with_extra = {**base, "N_SEEDS_PER_ARCH": 3}
    assert experiment_id(base, "bandit") != experiment_id(with_extra, "bandit")


def test_experiment_id_excludes_execution_metadata_fields():
    base = {"TICKER": "SPY", "N_ARMS": 45}
    with_out_dir = {**base, "OUT_DIR": "/tmp/whatever", "n_jobs": 8}
    assert experiment_id(base, "bandit") == experiment_id(with_out_dir, "bandit")
