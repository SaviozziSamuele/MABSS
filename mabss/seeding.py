"""
Determinism and seeding.

MABSS_utility.py had no global seeding at all: `walk_forward_training` never sets
a seed, so `model.fit(..., shuffle=True)` (the default) draws minibatch order from
Python's *unseeded* global RNG. The only actual seeding in the original codebase is
`RandomNormal(stddev=0.01, seed=random_seed)` inside `create_model`, which fixes
weight *initialization* only. Net effect: two runs of `walk_forward_training` with
the same CONFIG produce different trained weights (different minibatch order), even
though initialization was reproducible all along. This module exists to close that
gap; see mabss.training for where `set_global_seed` should be called before a
training run.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seeds every RNG this codebase touches: Python's `random`, NumPy's legacy
    global RNG (still used by mabss.policies.thompson's `np.random.standard_normal`
    and by np.random.choice call sites), and Keras/TensorFlow's RNGs.

    Must be called before the first `import keras`/`import tensorflow` in the
    process if `deterministic=True` is ever going to be requested later, because
    TF_DETERMINISTIC_OPS / TF_CUDNN_DETERMINISTIC / TF_ENABLE_ONEDNN_OPTS only take
    effect if set before TensorFlow initializes its backend. This is exactly why
    mabss.models imports keras lazily inside create_model rather than at module
    scope -- so a caller can do `set_global_seed(...)` first.

    `deterministic=False` (the default) reproduces the *scope* of randomness
    control the original code effectively had for weight initialization, while
    additionally fixing the previously-unseeded minibatch shuffle -- this is the
    minimal-delta option: it does not touch TF's own op-level determinism (which
    couples with moving K.clear_session() into the arm loop and would, on top of
    the ReLU change, invalidate cached artifacts for a second, independent reason).
    `deterministic=True` additionally enables TF op-level determinism, at a real
    performance cost, for when bit-exact reproducibility end-to-end is required
    (e.g. the `test_pipeline_is_deterministic_under_fixed_seed` regression test).
    """
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
        # oneDNN reorders reductions for performance, which changes floating-point
        # rounding across runs/threads -- disable for bit-exact reproducibility.
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        # keras/tensorflow not installed in this environment (e.g. a CPU-only
        # metrics/plots-only install) -- seeding the parts that exist is still
        # useful, so this is not an error.
        pass

    if deterministic:
        try:
            import tensorflow as tf

            tf.config.experimental.enable_op_determinism()
        except (ImportError, AttributeError):
            pass


def derive_arm_seed(global_seed: int, arch: str, model_idx: int) -> int:
    """
    Deterministic per-arm seed derivation.

    Not currently used by mabss.training.walk_forward_training -- the original
    code calls `create_model(arch, seed=m, input_dim)` where `m` is just the
    model's index within its architecture (0..N_SEEDS_PER_ARCH-1), so 'mlp'#3,
    'rnn'#3 and 'cnn'#3 all get initializer seed 3, and every walk-forward window
    reuses the same seed for a given arm. Per the plan's Phase 4 item 9 decision
    (§8, option (a) "minimal"), walk_forward_training keeps that exact scheme
    rather than switching to this function, to keep the initialization-seed
    changes isolated from the ReLU/retrain change already in flight. This
    function is provided for a deliberate follow-up (option (b), "full") that
    also derives a seed per walk-forward window, and is intentionally not wired
    in yet.
    """
    # Python's builtin hash() is salted per-process (PYTHONHASHSEED) unless that
    # env var was set before the interpreter started, so it is NOT safe for a
    # "deterministic across runs" helper -- use a stable hash instead.
    import hashlib

    key = f"{global_seed}:{arch}:{model_idx}".encode()
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF
