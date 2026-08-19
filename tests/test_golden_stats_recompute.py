"""
THE LOCK ON THE PAPER'S NUMBERS.

tools/build_stats_recompute_golden.py recomputes stats_df for all 30 paper
configurations (5 tickers x 3 policies x {homogeneous, heterogeneous}) starting only
from cached multirun_results.pkl + pctchange_preds.csv, using the frozen module's own
plot_multi_run_bandit plus a faithful, headless reimplementation of
compute_and_highlight_stats' pre-display arithmetic.

Verified: all 30 configs are bit-exact (max abs diff < 1e-9) against the stats_df.csv
files actually committed in experiments_cluster/. This is the empirical basis for the
plan's central baseline strategy: the expensive half of the pipeline (training, the
bandit loop) is not reproducible from this repo alone, but the half that produces
every number in Table 1 is exactly reproducible from disk, with no GPU.

Through Phases 0-3 (compile fix, package refactor, hardening, ID/legacy resolution)
this golden must stay byte-identical -- none of that work is supposed to change a
single published number. Phase 4's numbers-changing fixes (annualization, Styler
formatting, the NARMS typo, etc.) are exactly the commits that are allowed to update
this golden, each with its own reviewed diff.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_stats_recompute_matches_stored_stats_df(artifact_root):
    del artifact_root
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "build_stats_recompute_golden.py"), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "stats_recompute.json golden does not match stats_df.csv files on disk, OR "
        "the recomputation is no longer bit-exact from cache. This is the paper's "
        "number lock -- treat any failure here as high severity and read the "
        "MISMATCH lines in stderr before doing anything else.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.needs_artifacts
def test_all_30_paper_configs_are_bit_exact(artifact_root):
    """More granular than the --check subprocess test: asserts per-config, so a
    partial regression names exactly which (ticker, policy, arch) triple broke."""
    import json

    del artifact_root
    golden_path = REPO_ROOT / "tests" / "goldens" / "stats_recompute.json"
    if not golden_path.exists():
        pytest.skip("stats_recompute.json golden not built yet")
    data = json.loads(golden_path.read_text(encoding="utf-8"))

    assert len(data) == 30, f"expected 30 paper configs, golden has {len(data)}"
    failures = [k for k, v in data.items() if not v.get("bit_exact", False)]
    assert not failures, f"non-bit-exact or skipped configs: {failures}"
