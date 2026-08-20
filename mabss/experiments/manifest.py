"""
Content-derived manifests for experiment directories: the ground truth for
"what is actually in this directory," measured from file contents (array
shapes, CSV columns), independent of what the directory's NAME claims.

This exists because directory names are not trustworthy metadata -- confirmed
by a full content audit of experiments/ and experiments_cluster/ (see
CLAUDE.md's Gotchas): `experiments/CNN-true_EPOCHS-10_MLP-true_MODEL_EMBEDDING-15_
N_SEEDS_PER_ARCH-20_RNN-true_START_DATE-2000-01-01_TICKER-EGARCH_WINDOW_SIZE-252`'s
name implies a 60-model pool (3 architectures x 20 seeds each), but its
`preds.csv` has only 45 `model_*` columns. `test_manifest_flags_known_arm_count_mismatch`
pins this exact, verified case.

Manifests are written to a sidecar tree OUTSIDE experiments/ and
experiments_cluster/ (default: `<repo_root>/manifests/<root_name>/<hash>.json`)
-- deliberately NOT under `<root>/.mabss/...` as the plan's original sketch
put it. Reason: `tools/build_goldens.py`'s artifact-inventory golden walks
`root.iterdir()` for every directory under experiments/ and
experiments_cluster/, so anything written *inside* either root becomes a new
inventory entry and breaks the "corpus is byte-frozen through Phase 0-3"
invariant this whole refactor is built on (`test_golden_artifact_inventory`).
Keeping manifests external makes every manifest-writing tool read-only with
respect to the corpus by construction, not by convention -- verified: running
`tools/build_manifests.py` leaves `tests/goldens/artifact_inventory.json`
unchanged (see its own test).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_ARTIFACT_FILENAMES = {
    "preds": "preds.csv",
    "pctchange_preds": "pctchange_preds.csv",
    "contexts": "contexts.npy",
    "rewards": "rewards.npy",
    "rewards_df": "rewards_df.csv",
    "results": "results.pkl",
    "multirun_results": "multirun_results.pkl",
}
_PRED_KEYS = {"preds", "pctchange_preds"}
_BANDIT_KEYS = {"contexts", "rewards", "rewards_df", "results", "multirun_results"}

# Fields worth cross-checking against content, independently regex-matched
# out of the directory name (NOT a general tokenizer -- resolution is
# LegacyIndex's job, which renders forward from a config; this only reads
# backward, best-effort, for cross-checking known scalar fields whose values
# never themselves contain "_" in the observed corpus).
_NAME_FIELD_PATTERN = {
    field: re.compile(rf"(?:^|_){field}-([^_]+)") for field in ("N_ARMS", "N_SEEDS_PER_ARCH", "TICKER", "POLICY", "CNN", "MLP", "RNN")
}


def _dir_hash(name: str) -> str:
    return hashlib.blake2b(name.encode("utf-8"), digest_size=6).hexdigest()


def manifest_path(manifests_root: Path, root_name: str, dir_name: str) -> Path:
    return Path(manifests_root) / root_name / f"{_dir_hash(dir_name)}.json"


def infer_kind(exp_dir: Path) -> str:
    """'pred', 'bandit', 'mixed' (both present -- not observed on disk, but
    not assumed impossible), or 'unknown' (neither)."""
    present = {key for key, fname in _ARTIFACT_FILENAMES.items() if (exp_dir / fname).exists()}
    has_pred = bool(present & _PRED_KEYS)
    has_bandit = bool(present & _BANDIT_KEYS)
    if has_pred and has_bandit:
        return "mixed"
    if has_pred:
        return "pred"
    if has_bandit:
        return "bandit"
    return "unknown"


def _npy_shape(path: Path) -> list[int] | None:
    import numpy as np

    if not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return None
    return list(arr.shape)


def _model_column_count(path: Path) -> int | None:
    import pandas as pd

    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, nrows=0)
    except Exception:
        return None
    return sum(1 for c in df.columns if c.startswith("model_"))


def _parse_name_fields(dir_name: str) -> dict[str, str]:
    return {field: m.group(1) for field, pattern in _NAME_FIELD_PATTERN.items() if (m := pattern.search(dir_name))}


def derive_content_facts(exp_dir: Path, kind: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    if kind == "pred":
        facts["n_arms_actual"] = _model_column_count(exp_dir / _ARTIFACT_FILENAMES["preds"])
    elif kind == "bandit":
        rewards_shape = _npy_shape(exp_dir / _ARTIFACT_FILENAMES["rewards"])
        contexts_shape = _npy_shape(exp_dir / _ARTIFACT_FILENAMES["contexts"])
        facts["rewards_shape"] = rewards_shape
        facts["contexts_shape"] = contexts_shape
        facts["n_arms_actual"] = rewards_shape[1] if rewards_shape and len(rewards_shape) > 1 else None
        facts["n_windows_actual"] = rewards_shape[0] if rewards_shape else None
    return facts


def _find_mismatches(dir_name: str, name_fields: dict[str, str], content_facts: dict[str, Any], kind: str) -> list[str]:
    mismatches = []
    n_arms_actual = content_facts.get("n_arms_actual")

    if kind == "bandit" and "N_ARMS" in name_fields and n_arms_actual is not None:
        claimed = int(name_fields["N_ARMS"])
        if claimed != n_arms_actual:
            mismatches.append(f"name claims N_ARMS-{claimed} but content has {n_arms_actual} arms")

    if kind == "pred" and "N_SEEDS_PER_ARCH" in name_fields and n_arms_actual is not None:
        n_archs = sum(1 for arch in ("CNN", "MLP", "RNN") if name_fields.get(arch) == "true")
        if n_archs:
            implied = n_archs * int(name_fields["N_SEEDS_PER_ARCH"])
            if implied != n_arms_actual:
                mismatches.append(
                    f"name implies {n_archs} archs x N_SEEDS_PER_ARCH-{name_fields['N_SEEDS_PER_ARCH']} "
                    f"= {implied} models, but preds.csv has {n_arms_actual} model_* columns"
                )
    return mismatches


def build_manifest(root_name: str, exp_dir: Path) -> dict[str, Any]:
    dir_name = exp_dir.name
    kind = infer_kind(exp_dir)
    files_present = sorted(fname for fname in _ARTIFACT_FILENAMES.values() if (exp_dir / fname).exists())
    content_facts = derive_content_facts(exp_dir, kind)
    name_fields = _parse_name_fields(dir_name)
    mismatches = _find_mismatches(dir_name, name_fields, content_facts, kind)
    return {
        "root": root_name,
        "dir_name": dir_name,
        "kind": kind,
        "files_present": files_present,
        "name_fields": name_fields,
        "content_facts": content_facts,
        "mismatches": mismatches,
    }


def write_manifest(manifests_root: Path, manifest: dict[str, Any]) -> Path:
    path = manifest_path(manifests_root, manifest["root"], manifest["dir_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(manifests_root: Path, root_name: str, dir_name: str) -> dict[str, Any]:
    path = manifest_path(manifests_root, root_name, dir_name)
    return json.loads(path.read_text(encoding="utf-8"))


def build_all_manifests(repo_root: Path, manifests_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Walks experiments/ and experiments_cluster/ read-only, builds a manifest
    per directory, writes each to the sidecar tree, and returns
    {"<root_name>/<dir_name>": manifest, ...}.
    """
    repo_root = Path(repo_root)
    if manifests_root is None:
        manifests_root = repo_root / "manifests"
    results: dict[str, dict[str, Any]] = {}
    for root_name in ("experiments", "experiments_cluster"):
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            manifest = build_manifest(root_name, exp_dir)
            write_manifest(manifests_root, manifest)
            results[f"{root_name}/{exp_dir.name}"] = manifest
    return results
