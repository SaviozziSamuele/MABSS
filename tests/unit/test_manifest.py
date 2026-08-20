"""Unit tests for mabss.experiments.manifest, isolated from the real corpus."""

import numpy as np
import pandas as pd

from mabss.experiments.manifest import (
    build_all_manifests,
    build_manifest,
    infer_kind,
    load_manifest,
    manifest_path,
    write_manifest,
)


def _make_pred_dir(root, name, n_models):
    d = root / name
    d.mkdir(parents=True)
    cols = {"target": [0.1, 0.2]}
    for i in range(n_models):
        cols[f"model_{i}"] = [0.1, 0.2]
    pd.DataFrame(cols).to_csv(d / "preds.csv")
    (d / "pctchange_preds.csv").write_text("a,b\n1,2\n")
    return d


def _make_bandit_dir(root, name, n_arms, n_windows=4):
    d = root / name
    d.mkdir(parents=True)
    np.save(d / "rewards.npy", np.zeros((n_windows, n_arms)))
    np.save(d / "contexts.npy", np.zeros((n_windows, n_arms, 3)))
    return d


def test_infer_kind_pred_bandit_unknown(tmp_path):
    pred_dir = _make_pred_dir(tmp_path, "pred_dir", 3)
    bandit_dir = _make_bandit_dir(tmp_path, "bandit_dir", 5)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert infer_kind(pred_dir) == "pred"
    assert infer_kind(bandit_dir) == "bandit"
    assert infer_kind(empty_dir) == "unknown"


def test_manifest_flags_arm_count_mismatch_for_pred_dir(tmp_path):
    # Name implies 3 archs x N_SEEDS_PER_ARCH-20 = 60, content only has 45.
    name = (
        "CNN-true_EPOCHS-10_MLP-true_MODEL_EMBEDDING-15_N_SEEDS_PER_ARCH-20_"
        "RNN-true_START_DATE-2000-01-01_TICKER-EGARCH_WINDOW_SIZE-252"
    )
    exp_dir = _make_pred_dir(tmp_path, name, n_models=45)

    manifest = build_manifest("experiments", exp_dir)
    assert manifest["kind"] == "pred"
    assert manifest["content_facts"]["n_arms_actual"] == 45
    assert manifest["name_fields"]["N_SEEDS_PER_ARCH"] == "20"
    assert len(manifest["mismatches"]) == 1
    assert "60" in manifest["mismatches"][0] and "45" in manifest["mismatches"][0]


def test_manifest_no_mismatch_when_content_matches_name(tmp_path):
    name = "BANDIT_EMBEDDING-15_EPISODES-100_METRIC-l1_N_ARMS-60_POLICY-ucb_START_DATE-2000-01-01_TICKER-SPY_WINDOW_SIZE-252"
    exp_dir = _make_bandit_dir(tmp_path, name, n_arms=60)

    manifest = build_manifest("experiments_cluster", exp_dir)
    assert manifest["kind"] == "bandit"
    assert manifest["content_facts"]["n_arms_actual"] == 60
    assert manifest["mismatches"] == []
    assert manifest["name_fields"]["TICKER"] == "SPY"
    assert manifest["name_fields"]["POLICY"] == "ucb"


def test_manifest_flags_arm_count_mismatch_for_bandit_dir(tmp_path):
    name = "BANDIT_EMBEDDING-15_EPISODES-100_METRIC-l1_N_ARMS-60_POLICY-ucb_START_DATE-2000-01-01_TICKER-SPY_WINDOW_SIZE-252"
    exp_dir = _make_bandit_dir(tmp_path, name, n_arms=45)  # name says 60, content has 45

    manifest = build_manifest("experiments_cluster", exp_dir)
    assert len(manifest["mismatches"]) == 1
    assert "N_ARMS-60" in manifest["mismatches"][0]
    assert "45" in manifest["mismatches"][0]


def test_write_then_load_manifest_roundtrips(tmp_path):
    exp_dir = _make_bandit_dir(tmp_path / "experiments", "some_dir", n_arms=10)
    manifest = build_manifest("experiments", exp_dir)
    manifests_root = tmp_path / "manifests"

    write_manifest(manifests_root, manifest)
    loaded = load_manifest(manifests_root, "experiments", "some_dir")
    assert loaded == manifest


def test_manifest_path_is_a_sidecar_never_inside_the_experiment_root(tmp_path):
    """The whole point of the sidecar tree is that it must never live inside
    experiments/ or experiments_cluster/, since the artifact-inventory golden
    walks those roots and would pick up new manifest files as spurious
    'experiment directories', breaking the corpus-is-frozen invariant."""
    manifests_root = tmp_path / "manifests"
    path = manifest_path(manifests_root, "experiments", "some_dir_name")
    assert manifests_root in path.parents
    assert (tmp_path / "experiments") not in path.parents


def test_build_all_manifests_covers_every_directory_and_writes_sidecars(tmp_path):
    repo_root = tmp_path
    _make_pred_dir(repo_root / "experiments", "pred_a", n_models=2)
    _make_bandit_dir(repo_root / "experiments", "bandit_a", n_arms=3)
    _make_bandit_dir(repo_root / "experiments_cluster", "bandit_b", n_arms=4)

    manifests_root = repo_root / "manifests"
    results = build_all_manifests(repo_root, manifests_root)

    assert set(results.keys()) == {
        "experiments/pred_a",
        "experiments/bandit_a",
        "experiments_cluster/bandit_b",
    }
    # Sidecar files exist and nothing was written inside the experiment roots.
    assert manifests_root.exists()
    for f in (repo_root / "experiments").rglob("*"):
        assert f.name in {"pred_a", "bandit_a", "preds.csv", "pctchange_preds.csv", "rewards.npy", "contexts.npy"}
