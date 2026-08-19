"""
Phase 0 golden-baseline builder.

Walks `experiments/` and `experiments_cluster/` (the 2.3 GB committed result corpus)
and produces `tests/goldens/artifact_inventory.json`: a content fingerprint of every
experiment directory — file list, sha256, size, and (for known artifact types) shape/
dtype/columns/index endpoints.

This inventory is the proof that later refactor phases never write into the artifact
tree: `test_golden_artifact_inventory` re-runs this builder and asserts byte-identical
output. It is also the fixture manifest the characterization tests use to know what's
actually on disk (directory names are not trusted — see the legacy-ID audit in the
plan) without re-deriving it by hand in every test.

Usage:
    python tools/build_goldens.py                  # writes tests/goldens/artifact_inventory.json
    python tools/build_goldens.py --check           # exits 1 if output would differ from disk

Only reads files under experiments/ and experiments_cluster/. Never writes into them.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickletools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUT_PATH = REPO_ROOT / "tests" / "goldens" / "artifact_inventory.json"
ROOTS = ("experiments", "experiments_cluster")

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Safe, read-only pickle introspection -----------------------------------
#
# We never call pickle.load on artifact .pkl files with the default Unpickler
# (which can execute arbitrary find_class lookups). Instead we use pickletools'
# opcode stream to recover just the structural facts we need (top-level dict
# keys, and for numpy-array leaves, shape/dtype) without instantiating anything.
# This mirrors the "instrumented Unpickler, numpy-globals-only" audit already
# done manually on this corpus (see the plan's forensic §0.6).


def _safe_pickle_summary(path: Path) -> dict:
    """
    Best-effort, allocation-free structural summary of a .pkl file's pickle
    opcode stream: which GLOBAL names it references, and (if it's a plain dict
    at the top level) the top-level key names. Falls back to {"unreadable": True}
    rather than ever unpickling untrusted data with arbitrary find_class.
    """
    globals_seen: list[str] = []
    try:
        with open(path, "rb") as f:
            data = f.read()
        for opcode, arg, _pos in pickletools.genops(data):
            if opcode.name in ("GLOBAL", "STACK_GLOBAL"):
                globals_seen.append(str(arg))
    except Exception as e:  # pragma: no cover - defensive
        return {"unreadable": True, "error": repr(e)}
    # Dedup, preserve order.
    seen = []
    for g in globals_seen:
        if g not in seen:
            seen.append(g)
    return {"globals": seen}


def summarize_npy(path: Path) -> dict:
    import numpy as np

    arr = np.load(path, mmap_mode="r")
    return {
        "kind": "npy",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def summarize_csv(path: Path) -> dict:
    import pandas as pd

    try:
        df = pd.read_csv(path, index_col=0, nrows=0)
        columns = list(df.columns)
    except Exception as e:  # pragma: no cover - defensive
        return {"kind": "csv", "unreadable": True, "error": repr(e)}

    # Row count + index endpoints without loading the whole frame twice.
    full = pd.read_csv(path, index_col=0)
    n_rows = len(full)
    idx_first = str(full.index[0]) if n_rows else None
    idx_last = str(full.index[-1]) if n_rows else None
    return {
        "kind": "csv",
        "n_rows": n_rows,
        "columns": columns,
        "index_first": idx_first,
        "index_last": idx_last,
    }


def summarize_pkl(path: Path) -> dict:
    return {"kind": "pkl", **_safe_pickle_summary(path)}


def summarize_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    base = {
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        if suffix == ".npy":
            base.update(summarize_npy(path))
        elif suffix == ".csv":
            base.update(summarize_csv(path))
        elif suffix == ".pkl":
            base.update(summarize_pkl(path))
        else:
            base["kind"] = suffix.lstrip(".") or "other"
    except Exception as e:  # pragma: no cover - defensive
        base["summary_error"] = repr(e)
    return base


def build_inventory() -> dict:
    inventory: dict[str, dict] = {}
    for root_name in ROOTS:
        root = REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            key = f"{root_name}/{exp_dir.name}"
            files = {}
            for f in sorted(exp_dir.rglob("*")):
                if f.is_file():
                    rel = str(f.relative_to(exp_dir)).replace("\\", "/")
                    files[rel] = summarize_file(f)
            inventory[key] = {"n_files": len(files), "files": files}
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the freshly built inventory differs from tests/goldens/artifact_inventory.json",
    )
    args = parser.parse_args()

    print(f"Scanning {', '.join(ROOTS)} under {REPO_ROOT} ...", file=sys.stderr)
    inventory = build_inventory()
    n_files = sum(v["n_files"] for v in inventory.values())
    print(f"Found {len(inventory)} experiment dirs, {n_files} files.", file=sys.stderr)

    serialized = json.dumps(inventory, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not OUT_PATH.exists():
            print(f"FAIL: {OUT_PATH} does not exist", file=sys.stderr)
            return 1
        existing = OUT_PATH.read_text(encoding="utf-8")
        if existing != serialized:
            print(f"FAIL: freshly built inventory differs from {OUT_PATH}", file=sys.stderr)
            print(
                "This means something under experiments/ or experiments_cluster/ "
                "changed since the golden was frozen — the artifact corpus must "
                "stay byte-frozen through Phases 0-3.",
                file=sys.stderr,
            )
            return 1
        print("OK: inventory matches golden.", file=sys.stderr)
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(serialized, encoding="utf-8")
    print(f"Wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
