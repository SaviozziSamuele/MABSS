"""
Corpus-immutability gate (Phase 0.3 / plan §"Gate 2", §"Gate 3").

tools/build_goldens.py fingerprints every file under experiments/ and
experiments_cluster/ (sha256 + shape/dtype/columns as applicable) into
tests/goldens/artifact_inventory.json. Re-running the builder in --check mode must
reproduce that file byte-for-byte.

This is the proof that Phases 0-3 never write into the 2.3 GB committed result
corpus: any refactor that accidentally calls ExperimentLoader.save(), regenerates a
plot into an experiment dir, or otherwise touches those trees will change a hash here
and this test will catch it immediately, rather than the corruption being noticed
only when someone eventually diffs `git status` on 630 files.

Runs the full builder (hashes ~2.3 GB), so it's a couple of minutes -- acceptable for
a golden lock but not part of the sub-second unit loop.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.needs_artifacts
@pytest.mark.slow
def test_golden_artifact_inventory_matches_disk(artifact_root):
    del artifact_root  # fixture only used to trigger the skip-if-absent behavior
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "build_goldens.py"), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "artifact_inventory.json golden does not match the artifact corpus on disk "
        "-- something under experiments/ or experiments_cluster/ was modified.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
