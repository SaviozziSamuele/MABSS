"""
Pins the one confirmed name/content mismatch found across the full 63-directory
corpus (see CLAUDE.md's Gotchas): `experiments/CNN-true_..._N_SEEDS_PER_ARCH-20_
RNN-true_..._TICKER-EGARCH_...`'s name implies a 60-model heterogeneous pool
(3 architectures x 20 seeds each), but `preds.csv` has only 45 `model_*`
columns. Directory names are not trustworthy metadata -- this is the concrete,
verified example that motivates mabss.experiments.manifest's content-derived
approach.
"""

from pathlib import Path

import pytest

from mabss.experiments.manifest import build_manifest

REPO_ROOT = Path(__file__).parent.parent.parent
EGARCH_DIR_NAME = (
    "CNN-true_EPOCHS-10_MLP-true_MODEL_EMBEDDING-15_N_SEEDS_PER_ARCH-20_"
    "RNN-true_START_DATE-2000-01-01_TICKER-EGARCH_WINDOW_SIZE-252"
)


@pytest.mark.needs_artifacts
def test_egarch_dir_manifest_flags_the_known_60_vs_45_mismatch():
    exp_dir = REPO_ROOT / "experiments" / EGARCH_DIR_NAME
    if not exp_dir.is_dir():
        pytest.skip("known EGARCH mismatch directory not found on disk")

    manifest = build_manifest("experiments", exp_dir)
    assert manifest["kind"] == "pred"
    assert manifest["content_facts"]["n_arms_actual"] == 45
    assert len(manifest["mismatches"]) == 1
    assert "60" in manifest["mismatches"][0]
    assert "45" in manifest["mismatches"][0]


@pytest.mark.needs_artifacts
def test_only_this_one_directory_has_a_mismatch_in_the_full_corpus():
    """Full-corpus sweep: confirms this is the ONLY mismatch among all 63
    directories, matching CLAUDE.md's characterization ("one directory's
    name claims 60 arms and actually contains 45") -- not an example of a
    broader unfixed class of bugs."""
    mismatching = []
    for root_name in ("experiments", "experiments_cluster"):
        root = REPO_ROOT / root_name
        if not root.is_dir():
            pytest.skip(f"{root_name}/ not present")
        for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            manifest = build_manifest(root_name, exp_dir)
            if manifest["mismatches"]:
                mismatching.append(f"{root_name}/{exp_dir.name}")

    assert mismatching == [f"experiments/{EGARCH_DIR_NAME}"]
