"""
tools/build_results_structure_golden.py fingerprints the structure of every
multirun_results.pkl in the corpus, including the per-window "arm-block signature"
for both the stochastic (`best_arms`) and greedy (`best arms greedy`) tracks.

Confirmed on the real corpus (35 bandit dirs, all seeds): every UCB and Thompson
directory's stochastic-track signature is uniformly `[1, 1, ..., 1]` -- one constant
arm selected for an entire WINDOW_SIZE-length test block -- which is the fingerprint
of the stale-`arm_probs` bug (MABSS_utility.py:858: the test loop samples from
`arm_probs`, a training-loop leftover, instead of the freshly computed `probs`).
Softmax directories are NOT degenerate on this track, because Softmax's `select_arm`
returns a real probability distribution rather than UCB/Thompson's one-hot, so even a
stale distribution still produces varied samples. The greedy track
(`best arms greedy`, computed fresh each test step via `np.argmax(probs)`) is never
degenerate for any policy -- this is the track the paper actually uses at deployment
(Manuscript.tex:197 mandates greedy testing), so Table 1 is unaffected by this bug.

This test just locks the structural golden in place; test_bug_arm_probs_staleness.py
(added in Phase 4) is what turns the signature into a pass/fail regression check.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.needs_artifacts
def test_golden_results_structure_matches_disk(artifact_root):
    del artifact_root
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "build_results_structure_golden.py"), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "results_structure.json golden does not match the corpus.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.needs_artifacts
def test_ucb_and_thompson_stochastic_track_is_degenerate_today(artifact_root):
    """
    Documents the current (buggy) state precisely, so Phase 4's fix has an
    unambiguous before/after: every UCB/Thompson dir's `best_arms` signature is all-1s.
    """
    import json

    inventory_path = REPO_ROOT / "tests" / "goldens" / "results_structure.json"
    if not inventory_path.exists():
        pytest.skip("results_structure.json golden not built yet")
    data = json.loads(inventory_path.read_text(encoding="utf-8"))

    checked = 0
    for dir_key, entry in data.items():
        if "seeds" not in entry:
            continue
        if "POLICY-ucb" not in dir_key and "POLICY-thompson" not in dir_key:
            continue
        for seed_key, seed_entry in entry["seeds"].items():
            sig = seed_entry.get("_arm_block_signature_stochastic")
            if sig is None:
                continue
            checked += 1
            assert all(x == 1 for x in sig), (
                f"{dir_key}/{seed_key}: expected the documented-buggy all-1s "
                f"stochastic-track signature, got {sig}. Either the corpus changed "
                "or bug :858 has already been fixed in a way that regenerated "
                "these cached artifacts -- if so, this test (and its docstring) "
                "should be updated/removed as part of that Phase 4 commit."
            )
    assert checked > 0, "no UCB/Thompson multirun_results found to check"
