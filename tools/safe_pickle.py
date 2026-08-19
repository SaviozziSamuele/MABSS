"""
Restricted unpickler for the committed .pkl artifact corpus.

The forensic audit behind this plan scanned all 79 .pkl files in experiments/ and
experiments_cluster/ with a find_class-instrumented Unpickler and found only numpy
globals (numpy.ndarray, numpy.dtype, numpy._core.multiarray._reconstruct/scalar) plus
builtin containers. This module enforces exactly that allowlist so any later addition
to the corpus that references something else fails loudly instead of being unpickled
blindly — these files are read-only inputs we trust today, not something to grant
arbitrary code execution to by habit.
"""

from __future__ import annotations

import io
import pickle

_ALLOWED_MODULES_EXACT = {
    "numpy",
    "numpy.core.multiarray",
    "numpy._core.multiarray",
    "numpy.core",
    "numpy._core",
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy") and module in _ALLOWED_MODULES_EXACT | {module}:
            # Only allow modules that are exactly "numpy*" prefixed and known safe.
            if module == "numpy" or module.startswith("numpy.") or module.startswith("numpy._"):
                return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to unpickle {module}.{name}: not in the numpy-only allowlist "
            "established by the corpus audit. If this is a legitimate new artifact "
            "type, extend the allowlist deliberately rather than removing this check."
        )


def safe_load(path) -> object:
    with open(path, "rb") as f:
        return RestrictedUnpickler(f).load()


def safe_loads(data: bytes) -> object:
    return RestrictedUnpickler(io.BytesIO(data)).load()
