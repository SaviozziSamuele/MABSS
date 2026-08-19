"""
Phase 0 differential oracle.

`tests/_frozen/mabss_utility_frozen.py` is a byte-for-byte copy of `MABSS_utility.py`
taken at the moment it first became importable (the two-line indentation dedent at
`:1466` and `:1625`, verified whitespace-only via `git diff --ignore-all-space`).

It must NEVER be edited again. Every later refactor's `test_shim_parity` diffs new
code against this frozen module for every pure function; any output divergence must
be an explicit, reviewed `EXPECTED_DIVERGENCES` entry keyed by bug ID (see
tests/characterization/expected_divergences.py once Phase 1 begins).

This test is the tripwire: if the frozen file's hash ever changes, something edited
the oracle instead of the real module, and every downstream parity test becomes
meaningless.
"""

import hashlib
from pathlib import Path

FROZEN_PATH = Path(__file__).parent / "_frozen" / "mabss_utility_frozen.py"

# Computed once, at freeze time, from the first-ever-compiling MABSS_utility.py.
FROZEN_SHA256 = "bd72d09c7c421a991c7e0cc0cee32f141e88755aa2e6cd80033bad745ce6e27d"


def test_frozen_oracle_exists():
    assert FROZEN_PATH.exists(), (
        "tests/_frozen/mabss_utility_frozen.py is missing. It is the differential "
        "oracle for behavior-preservation testing and must be restored from git."
    )


def test_frozen_oracle_hash_is_pinned():
    actual = hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest()
    assert actual == FROZEN_SHA256, (
        "tests/_frozen/mabss_utility_frozen.py has been modified since it was frozen "
        f"(expected sha256={FROZEN_SHA256}, got {actual}). This file must never be "
        "edited — it is the oracle every parity test diffs against. If you meant to "
        "update the real module, edit MABSS_utility.py / mabss/ instead."
    )


def test_frozen_oracle_compiles():
    import py_compile
    import tempfile
    import os

    # py_compile writes bytecode; use a throwaway cache location.
    with tempfile.TemporaryDirectory() as tmp:
        cfile = os.path.join(tmp, "frozen.pyc")
        py_compile.compile(str(FROZEN_PATH), cfile=cfile, doraise=True)


def test_frozen_oracle_is_importable():
    """The oracle must actually import so parity tests can call into it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("mabss_utility_frozen", FROZEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "reward_function")
    assert hasattr(module, "WalkForwardEnvCMAB")
