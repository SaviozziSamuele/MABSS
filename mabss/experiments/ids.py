"""Experiment ID generation: the legacy (v1) scheme and a collision-free v2 scheme."""

from __future__ import annotations

import hashlib
import json


def make_experiment_id(config: dict) -> tuple[str, str]:
    """
    Ported verbatim from MABSS_utility.py:191-241 (`make_experiment_id`). Kept
    exactly as-is -- including its bug -- because it's what every directory
    name in experiments/ and experiments_cluster/ was generated with; a
    behavior change here would break addressing the existing 2.3 GB corpus, not
    fix anything (a *safe* replacement is `experiment_id` below, used for NEW
    runs going forward).

    Known bug, NOT fixed here (see `experiment_id` for the fix): both
    `key_fields_pred`/`key_fields_bandit` are filtered with `if k in config`, so
    a config that's simply missing a field (rather than explicitly setting it to
    a default) silently produces a DIFFERENT, shorter ID instead of an error --
    e.g. `key_fields_bandit` includes N_ARMS but not MLP/RNN/CNN/
    N_SEEDS_PER_ARCH, so a homogeneous 45-model pool and a heterogeneous 15x3
    pool both resolve to `N_ARMS-45` and collide on the same bandit directory.
    """
    key_fields_pred = [
        "TICKER",
        "START_DATE",
        "WINDOW_SIZE",
        "MLP",
        "RNN",
        "CNN",
        "MODEL_EMBEDDING",
        "EPOCHS",
        "N_SEEDS_PER_ARCH",
    ]
    key_fields_bandit = [
        "TICKER",
        "START_DATE",
        "N_ARMS",
        "WINDOW_SIZE",
        "BANDIT_EMBEDDING",
        "METRIC",
        "ALPHA",
        "GAMMA",
        "EPISODES",
        "POLICY",
    ]
    compact_pred = {k: config[k] for k in key_fields_pred if k in config}
    compact_bandit = {k: config[k] for k in key_fields_bandit if k in config}
    raw_pred = json.dumps(compact_pred, sort_keys=True)
    raw_bandit = json.dumps(compact_bandit, sort_keys=True)

    def _sanitize(raw: str) -> str:
        return (
            raw.replace(" ", "")
            .replace("{", "")
            .replace("}", "")
            .replace('"', "")
            .replace(":", "-")
            .replace(",", "_")
            .replace(".", "p")
        )

    return _sanitize(raw_pred), _sanitize(raw_bandit)


# Fields that are execution metadata, not experiment identity -- excluded from
# the v2 hash so that e.g. changing --jobs or an output directory doesn't mint
# a new experiment ID for what is semantically the same run.
V2_ID_EXCLUDE = frozenset({"n_jobs", "out_dir", "log_level", "OUT_DIR", "N_JOBS", "LOG_LEVEL"})


def experiment_id(config: dict, kind: str) -> str:
    """
    Collision-free replacement for `make_experiment_id`, for NEW runs.
    `kind` is e.g. "pred" or "bandit" and is part of the hashed payload, so pred
    and bandit IDs for the same config never collide with each other either.

    Hashes the ENTIRE config (minus `V2_ID_EXCLUDE`), not a hand-picked subset --
    this is what fixes both of make_experiment_id's failure modes: a field
    that's silently absent can't produce a shorter/colliding ID, because there's
    no subset-selection step to omit it from; and two configs that differ in ANY
    field (e.g. MLP/RNN/CNN pool composition, previously not part of the bandit
    ID at all) get different hashes.

    Returns `<kind>/<ticker>_<hash12>` -- a short, mostly-readable prefix (not
    itself unique) plus a 12-hex-char blake2b digest of the canonical JSON
    payload (the actual identity). Directory/file names built from this should
    still slugify the ticker for shell/Windows-filename safety even though
    values like "EURUSD=X" and "GC=F" are legal on NTFS -- '=' is fine on disk
    but awkward in shell commands and SLURM --job-name.
    """
    payload = {k: v for k, v in sorted(config.items()) if k not in V2_ID_EXCLUDE}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=6).hexdigest()
    ticker = str(config.get("TICKER", "unknown")).replace("=", "").replace("^", "")
    return f"{kind}/{ticker}_{digest}"
