"""
Regression coverage for the bug that makes notebook cells 26/28 unable to
address a single one of the ~41 experiments_cluster/ directories.

The mechanism: the notebook's CONFIG dict always carries ALPHA/GAMMA keys
(set for every run, even policies that ignore them), so
`make_experiment_id`'s bandit ID always renders an `ALPHA-..._GAMMA-...`
token. Every experiments_cluster/ directory name was produced by an older
revision of the ID function that never included ALPHA/GAMMA at all -- so
the rendered token never matches any real directory, `ExperimentLoader.exists()`
reports "not cached" for a config that in fact has 100+ MB of real cached
results sitting right there, and downstream cells fall through to mkdir'ing
a fresh (empty) directory instead.

`LegacyIndex.resolve()` (mabss/experiments/legacy.py) fixes this by trying
all known historical field-subsets, not just the current one.
"""

import re
from pathlib import Path

import pytest

from mabss.experiments.ids import make_experiment_id
from mabss.experiments.legacy import BANDIT_SCHEMAS, PRED_SCHEMAS, LegacyIndex

REPO_ROOT = Path(__file__).parent.parent.parent

# Regex parser used ONLY here, for test verification: given a real directory
# name and the *known* field list a schema uses (alphabetically sorted, as
# json.dumps(sort_keys=True) always renders them), recovers the exact
# {field: value} pairs that must have produced it. This is deliberately not
# part of mabss/ -- LegacyIndex resolves forward from a config, it never
# needs to parse a name backward; this only exists so the test can
# reconstruct a plausible config per real directory without hardcoding all
# 63 by hand.


def _extract_fields(dir_name: str, fields: list[str]) -> dict[str, str] | None:
    ordered = sorted(fields)
    pattern = "^" + "_".join(rf"{re.escape(f)}-(?P<{f}>[^_]+(?:-[^_]+)*)" for f in ordered) + "$"
    m = re.match(pattern, dir_name)
    if not m:
        return None
    return m.groupdict()


def _typed(field: str, value: str):
    if value == "null":
        return None
    if value in ("true", "false"):
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    if field == "ALPHA":
        return float(value.replace("p", "."))
    return value


def _reconstruct_config(dir_name: str, schemas: dict[str, list[str]]) -> tuple[str, dict]:
    for schema_name, fields in schemas.items():
        parsed = _extract_fields(dir_name, fields)
        if parsed is not None:
            return schema_name, {k: _typed(k, v) for k, v in parsed.items()}
    raise AssertionError(f"no known schema matches directory name: {dir_name}")


@pytest.mark.needs_artifacts
@pytest.mark.parametrize("root_name,schemas,kind", [
    ("experiments", BANDIT_SCHEMAS, "bandit"),
    ("experiments", PRED_SCHEMAS, "pred"),
    ("experiments_cluster", BANDIT_SCHEMAS, "bandit"),
])
def test_legacy_index_resolves_every_matching_real_directory(root_name, schemas, kind):
    """For every real directory under root_name whose name matches one of the
    known schemas for `kind`, reconstructing its config from the name and
    resolving it through LegacyIndex must return that exact directory,
    tagged with the schema that actually produced it."""
    root = REPO_ROOT / root_name
    if not root.is_dir():
        pytest.skip(f"{root_name}/ not present")

    index = LegacyIndex(root)
    matched_any = False
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            schema_name, config = _reconstruct_config(exp_dir.name, schemas)
        except AssertionError:
            continue  # belongs to the other kind (pred vs bandit) -- fine
        matched_any = True
        match = index.resolve(config, kind)
        assert match.path.name == exp_dir.name
        assert match.schema == schema_name
    assert matched_any, f"no {kind} directories under {root_name}/ matched any known schema -- test is not exercising anything"


@pytest.mark.needs_artifacts
def test_realistic_notebook_config_cannot_be_resolved_by_make_experiment_id_alone():
    """Pins the actual bug: a config shaped like what the notebook really
    builds (ALPHA/GAMMA explicitly set) cannot address the canonical cluster
    directory via make_experiment_id, but LegacyIndex finds it."""
    cluster_root = REPO_ROOT / "experiments_cluster"
    if not cluster_root.is_dir():
        pytest.skip("experiments_cluster/ not present")

    canonical_name = "BANDIT_EMBEDDING-15_EPISODES-100_METRIC-l1_N_ARMS-60_POLICY-ucb_START_DATE-2000-01-01_TICKER-SPY_WINDOW_SIZE-252"
    if not (cluster_root / canonical_name).is_dir():
        pytest.skip("canonical N_ARMS-60/POLICY-ucb/TICKER-SPY cluster dir not found")

    notebook_style_config = {
        "TICKER": "SPY",
        "START_DATE": "2000-01-01",
        "N_ARMS": 60,
        "WINDOW_SIZE": 252,
        "BANDIT_EMBEDDING": 15,
        "METRIC": "l1",
        "ALPHA": 1.0,  # the notebook's CONFIG always sets these, even for UCB
        "GAMMA": None,  # runs that don't semantically use them
        "EPISODES": 100,
        "POLICY": "ucb",
    }

    _pred_id, bandit_id = make_experiment_id(notebook_style_config)
    assert not (cluster_root / bandit_id).is_dir(), (
        "if this now exists, make_experiment_id alone resolves cluster dirs and "
        "this regression test (and the LegacyIndex workaround) is obsolete"
    )

    match = LegacyIndex(cluster_root).resolve(notebook_style_config, "bandit")
    assert match.path.name == canonical_name
    assert match.schema == "B3_cluster_no_alpha_gamma"
