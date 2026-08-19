"""Read/write access to a single experiment's cached artifacts on disk."""

from __future__ import annotations

import os
import pickle as pkl
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ids import make_experiment_id


class ExperimentStore:
    """
    Ported from MABSS_utility.py's `ExperimentLoader` (:272-348), with the
    filenames/artifact-key mapping left EXACTLY as-is (this is what every
    directory in experiments/ and experiments_cluster/ was written with, and
    every one of the 63 experiment directories on disk was confirmed to match
    it exactly -- unlike the dead `get_experiment_paths`, which declares
    filenames that exist nowhere).

    Hardening applied here (plan Phase 2, all verified to not touch any number):

      - **No mkdir on construction.** The original's `__init__` called
        `self.pred_dir.mkdir(...)`/`self.bandit_dir.mkdir(...)` unconditionally
        -- so merely constructing a loader to CHECK whether something was
        cached created two directories. Several near-empty orphan directories
        on disk (3 files instead of 7) are a direct consequence. Directories
        are now only created inside `save()`, and only when there's actually
        something to write.
      - **Uniform datetime parsing.** The original parsed dates for
        'pctchange_preds' but not 'preds'/'rewards_df', which came back with a
        string (object-dtype) index -- four separate call sites in the original
        (MABSS_utility.py:997, 1345, 1482, 1484) hand-patched this after the
        fact with `df.index = pd.to_datetime(df.index, utc=True)`. All three
        frame-valued keys are now parsed consistently here, so that patch is no
        longer needed by callers.
      - **`save()` populates the cache from what was actually written**, by
        re-reading through `load(force_reload=True)` after writing. The
        original's `save()` never touched `self._cache`, so
        `save(x); load(same_key)` could return an object with a different index
        dtype than `x` (the round-trip-through-CSV difference above) --
        surprising and easy to miss. Now save-then-load is guaranteed
        consistent by construction.
      - **Atomic writes**: every write goes to a temp file in the same
        directory, then `os.replace`s the real path. The original wrote
        directly, so an interrupted long run could leave a truncated
        CSV/pickle that `exists()` would happily report as "cached".
      - **`mode="r"`** (read-only) is required for accessing the committed
        experiments/experiments_cluster/ corpus from anything that must never
        write into it (tests, `mabss inspect`, ad-hoc analysis) -- `save()`
        raises immediately rather than relying on convention.
    """

    def __init__(self, config: dict, base_dir: Path, mode: str = "rw"):
        if mode not in ("r", "rw"):
            raise ValueError(f"mode must be 'r' or 'rw', got {mode!r}")
        self.mode = mode
        self.config = config
        base_dir = Path(base_dir)
        self.pred_id, self.bandit_id = make_experiment_id(config)
        self.pred_dir = base_dir / self.pred_id
        self.bandit_dir = base_dir / self.bandit_id

        self.paths: dict[str, Path] = {
            "preds": self.pred_dir / "preds.csv",
            "pctchange_preds": self.pred_dir / "pctchange_preds.csv",
            "contexts": self.bandit_dir / "contexts.npy",
            "rewards": self.bandit_dir / "rewards.npy",
            "rewards_df": self.bandit_dir / "rewards_df.csv",
            "results": self.bandit_dir / "results.pkl",
            "multirun_results": self.bandit_dir / "multirun_results.pkl",
        }
        self._cache: dict[str, Any] = {}

    def exists(self, key: str) -> bool:
        return self.paths[key].exists()

    def is_fully_cached(self) -> bool:
        return all(self.exists(k) for k in self.paths)

    _DATETIME_INDEX_KEYS = {"preds", "pctchange_preds", "rewards_df"}

    def load(self, key: str, force_reload: bool = False) -> Any:
        if not force_reload and key in self._cache:
            return self._cache[key]

        path = self.paths[key]
        if not path.exists():
            raise FileNotFoundError(f"{key} not found: {path}")

        if key in ("preds", "pctchange_preds", "rewards_df"):
            obj = pd.read_csv(path, index_col=0)
            obj.index = pd.to_datetime(obj.index, utc=True)
        elif key in ("contexts", "rewards"):
            obj = np.load(path)
        elif key in ("results", "multirun_results"):
            with open(path, "rb") as f:
                obj = pkl.load(f)
        else:
            raise ValueError(f"Unknown artifact: {key}")

        self._cache[key] = obj
        return obj

    def load_all(self) -> dict[str, Any]:
        return {k: self.load(k) for k in self.paths if self.exists(k)}

    def save(self, data: dict[str, Any]) -> None:
        if self.mode == "r":
            raise PermissionError(
                f"ExperimentStore for {self.pred_dir} / {self.bandit_dir} was "
                "opened in read-only mode ('r'); refusing to write. Open with "
                "mode='rw' if you actually intend to write new artifacts."
            )
        for key, obj in data.items():
            path = self.paths[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(path, key, obj)
            # Re-read through `load` so the cache holds the same normalized
            # object a fresh load() would return (fixes the save/load
            # asymmetry described in the class docstring).
            self.load(key, force_reload=True)

    def _atomic_write(self, path: Path, key: str, obj: Any) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            if key in ("preds", "pctchange_preds", "rewards_df"):
                obj.to_csv(tmp_path)
            elif isinstance(obj, np.ndarray):
                # np.save appends .npy if missing; write to a plain tmp file by
                # name instead so the atomic rename target matches exactly.
                with open(tmp_path, "wb") as f:
                    np.save(f, obj)
            elif key in ("results", "multirun_results"):
                with open(tmp_path, "wb") as f:
                    pkl.dump(obj, f)
            else:
                raise ValueError(f"Cannot save {key}: {type(obj)}")
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
