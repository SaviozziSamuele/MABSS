"""
Phase 0 golden: tests/goldens/results_structure.json

For every bandit-result directory in experiments/ + experiments_cluster/ that has a
multirun_results.pkl, records per-seed, per-key shape/dtype/length, plus the
"arm-block signature" for `best arms greedy`:

    [len(set(best_arms_greedy[w*W:(w+1)*W])) for w in range(n_windows)]

For every dir today this signature is `[1, 1, ..., 1]` — one constant arm per
252-bar test window — which is the fingerprint of the stale-`arm_probs` bug at
`MABSS_utility.py:858` (the test loop samples the stochastic arm from the previous
training step's probabilities instead of the current test context; see
tests/test_bug_arm_probs_staleness.py). This golden exists so that Phase 4's fix can
prove, per directory, that the signature changes from all-1s to something with real
per-window diversity, while `best arms greedy` itself (the deterministic/greedy track
the paper actually uses) stays completely unaffected — see
tests/goldens/stats_recompute.json for that half.

Uses tools/safe_pickle.RestrictedUnpickler (numpy-only allowlist) rather than the
default Unpickler, consistent with the corpus's read-only, untrusted-input status.

Usage:
    python tools/build_results_structure_golden.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from safe_pickle import safe_load  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
OUT_PATH = REPO_ROOT / "tests" / "goldens" / "results_structure.json"
ROOTS = ("experiments", "experiments_cluster")

WINDOW_SIZE_DEFAULT = 252  # every directory in this corpus uses WINDOW_SIZE-252


def _window_size_from_dirname(name: str) -> int:
    for token in name.split("_"):
        if token.startswith("WINDOW_SIZE-"):
            try:
                return int(token.split("-", 1)[1])
            except ValueError:
                pass
    return WINDOW_SIZE_DEFAULT


def describe_value(v) -> dict:
    import numpy as np

    if isinstance(v, np.ndarray):
        return {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, list):
        return {"type": "list", "len": len(v)}
    if isinstance(v, dict):
        return {"type": "dict", "keys": sorted(v.keys())}
    return {"type": type(v).__name__}


def arm_block_signature(best_arms_greedy, window_size: int) -> list[int]:
    import numpy as np

    arr = np.asarray(best_arms_greedy)
    n_windows = len(arr) // window_size
    sig = []
    for w in range(n_windows):
        block = arr[w * window_size : (w + 1) * window_size]
        sig.append(len(set(block.tolist())))
    return sig


def build() -> dict:
    out: dict[str, dict] = {}
    for root_name in ROOTS:
        root = REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            pkl_path = exp_dir / "multirun_results.pkl"
            if not pkl_path.exists():
                continue
            key = f"{root_name}/{exp_dir.name}"
            window_size = _window_size_from_dirname(exp_dir.name)
            try:
                multirun = safe_load(pkl_path)
            except Exception as e:  # pragma: no cover - defensive
                out[key] = {"unreadable": True, "error": repr(e)}
                continue

            seeds_out = {}
            for seed_key, result in sorted(multirun.items()):
                entry = {k: describe_value(v) for k, v in sorted(result.items())}
                # "best_arms" is the STOCHASTIC test-loop track (np.random.choice
                # sampled against the stale arm_probs left over from training --
                # bug :858). "best arms greedy" is np.argmax(probs) computed fresh
                # each test step and is what the paper actually uses (greedy
                # deployment, per Manuscript.tex:197). Signature = number of
                # distinct arms selected within each WINDOW_SIZE-length test block;
                # today "best_arms" is uniformly [1, 1, ..., 1] (one constant arm
                # per window -- the bug's fingerprint) while "best arms greedy" is
                # not. Track both so Phase 4's fix can prove the stochastic track
                # gains diversity while the greedy track (Table 1) stays untouched.
                if "best_arms" in result:
                    entry["_arm_block_signature_stochastic"] = arm_block_signature(
                        result["best_arms"], window_size
                    )
                if "best arms greedy" in result:
                    entry["_arm_block_signature_greedy"] = arm_block_signature(
                        result["best arms greedy"], window_size
                    )
                seeds_out[seed_key] = entry
            out[key] = {"n_seeds": len(multirun), "window_size": window_size, "seeds": seeds_out}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    print(f"Scanning {', '.join(ROOTS)} for multirun_results.pkl ...", file=sys.stderr)
    data = build()
    print(f"Found {len(data)} bandit dirs with multirun_results.pkl.", file=sys.stderr)
    serialized = json.dumps(data, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not OUT_PATH.exists():
            print(f"FAIL: {OUT_PATH} does not exist", file=sys.stderr)
            return 1
        if OUT_PATH.read_text(encoding="utf-8") != serialized:
            print(f"FAIL: results_structure differs from golden at {OUT_PATH}", file=sys.stderr)
            return 1
        print("OK: results_structure matches golden.", file=sys.stderr)
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(serialized, encoding="utf-8")
    print(f"Wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
