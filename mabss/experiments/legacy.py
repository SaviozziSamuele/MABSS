"""
Legacy experiment-ID resolution: mapping a CONFIG dict to one of the ~63
pre-existing experiment directories on disk, across the (at least) six
distinct directory-naming schemas empirically found in experiments/ and
experiments_cluster/ (confirmed by content audit, not just naming -- see
CLAUDE.md's Gotchas).

`make_experiment_id` (ids.py) renders exactly one schema per kind -- the
*current* one, matching whatever `key_fields_pred`/`key_fields_bandit` have
been since the version of the notebook this repo's `.venv` matches. It
cannot resolve a single experiments_cluster/ directory: the notebook's
CONFIG dict always carries ALPHA/GAMMA keys (even for policies that ignore
them), so `make_experiment_id`'s bandit ID always includes an
`ALPHA-..._..._GAMMA-...` token -- a token that appears in zero
experiments_cluster/ directory names, because whatever historical revision
of the ID function produced those 41 directories used a field subset with no
ALPHA/GAMMA at all. That mismatch is the concrete mechanism behind notebook
cells 26/28 reporting "not cached" and silently mkdir'ing empty directories
next to the real cluster corpus instead of finding it.

`LegacyIndex` tries every known schema's field subset through the same
`_render_id` renderer `make_experiment_id` uses, and returns whichever
rendered name actually exists on disk. Directory names are addresses to
*try*, never trusted metadata about what's inside them -- see `manifest.py`
for content-derived ground truth (e.g. the confirmed case where a directory
name implies a 60-arm pool and actually contains 45).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ids import _render_id, experiment_id

# ---------------------------------------------------------------------------
# Known schemas, keyed by a descriptive name. Each value is the exact
# `key_fields_*` list some past revision of the pred/bandit ID function used
# to build directory names that exist on disk today. P2_heterogeneous and
# B2_current are identical to what `make_experiment_id` renders now (kept
# duplicated here rather than imported, so this table is a self-contained
# description of "every schema ever observed" independent of which one the
# recovered code happens to match). B0/B1/B3 do not correspond to anything
# the recovered `make_experiment_id` can produce -- their source revision
# is not in this repo's history, only their output is (on disk).
# ---------------------------------------------------------------------------

PRED_SCHEMAS: dict[str, list[str]] = {
    "P1_homogeneous": [
        "CNN",
        "EPOCHS",
        "MLP",
        "MODEL_EMBEDDING",
        "RNN",
        "START_DATE",
        "TICKER",
        "WINDOW_SIZE",
    ],
    "P2_heterogeneous": [
        "CNN",
        "EPOCHS",
        "MLP",
        "MODEL_EMBEDDING",
        "N_SEEDS_PER_ARCH",
        "RNN",
        "START_DATE",
        "TICKER",
        "WINDOW_SIZE",
    ],
}

BANDIT_SCHEMAS: dict[str, list[str]] = {
    "B0_epochs_field": [
        "ALPHA",
        "BANDIT_EMBEDDING",
        "EPOCHS",
        "GAMMA",
        "METRIC",
        "N_ARMS",
        "START_DATE",
        "TICKER",
        "WINDOW_SIZE",
    ],
    "B1_no_policy": [
        "ALPHA",
        "BANDIT_EMBEDDING",
        "EPISODES",
        "GAMMA",
        "METRIC",
        "N_ARMS",
        "START_DATE",
        "TICKER",
        "WINDOW_SIZE",
    ],
    "B2_current": [
        "ALPHA",
        "BANDIT_EMBEDDING",
        "EPISODES",
        "GAMMA",
        "METRIC",
        "N_ARMS",
        "POLICY",
        "START_DATE",
        "TICKER",
        "WINDOW_SIZE",
    ],
    "B3_cluster_no_alpha_gamma": [
        "BANDIT_EMBEDDING",
        "EPISODES",
        "METRIC",
        "N_ARMS",
        "POLICY",
        "START_DATE",
        "TICKER",
        "WINDOW_SIZE",
    ],
}

_KIND_SCHEMAS = {"pred": PRED_SCHEMAS, "bandit": BANDIT_SCHEMAS}


class AmbiguousExperimentError(Exception):
    """Raised when two or more DIFFERENT existing directories match the same
    config under different legacy schemas, and no single candidate's schema
    strictly dominates (is a superset of) every other candidate's -- so
    there's no principled way to prefer one. This is real, not theoretical:
    experiments/ contains both a homogeneous (P1) and heterogeneous (P2)
    pred directory sharing every field except N_SEEDS_PER_ARCH, and a config
    setting fields from two genuinely incomparable schemas (e.g. both
    B0's EPOCHS and B1's EPISODES) can match two distinct real directories.
    (Two schemas simply rendering the *same* string, e.g. B2_current and
    B3_cluster_no_alpha_gamma when ALPHA/GAMMA are unset, is NOT ambiguous --
    that's one directory with two possible labels, resolved by `resolve()`
    without raising.) The caller must disambiguate the config or pass an
    explicit path."""


class ExperimentNotFoundError(Exception):
    """Raised when no legacy schema's rendered ID matches any directory on
    disk for this config. Carries the v2 ID this config would use for a NEW
    run (`experiment_id`), so callers can distinguish "not run yet" from
    "config typo" without a second lookup."""

    def __init__(self, config: dict, kind: str, v2_id: str, tried: dict[str, str]):
        self.config = config
        self.kind = kind
        self.v2_id = v2_id
        self.tried = tried
        tried_str = "; ".join(f"{name}={rendered!r}" for name, rendered in tried.items())
        super().__init__(
            f"No {kind} directory found for this config under any of {len(tried)} "
            f"known legacy schemas ({tried_str}). If this is a new run, its v2 ID "
            f"would be {v2_id!r}."
        )


@dataclass(frozen=True)
class LegacyMatch:
    path: Path
    schema: str
    rendered_id: str


class LegacyIndex:
    """
    Read-only index over one experiment root (`experiments/` or
    `experiments_cluster/`), resolving CONFIG dicts to existing directories
    across all known legacy naming schemas.

    Never writes, renames, or moves anything. The directory listing is
    snapshotted once at construction (`root.iterdir()`), so results stay
    stable for the lifetime of the index even if something else touches the
    filesystem mid-session -- callers needing a fresh view should construct
    a new `LegacyIndex`.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._names = (
            frozenset(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.is_dir() else frozenset()
        )

    def resolve(self, config: dict, kind: str) -> LegacyMatch:
        if kind not in _KIND_SCHEMAS:
            raise ValueError(f"kind must be 'pred' or 'bandit', got {kind!r}")
        schemas = _KIND_SCHEMAS[kind]

        tried: dict[str, str] = {}
        # Keyed by rendered id, not by schema: two schemas that differ only in
        # fields absent from THIS config (e.g. B2_current vs
        # B3_cluster_no_alpha_gamma when ALPHA/GAMMA are simply unset) render
        # the identical string via the same `if k in config` filtering
        # make_experiment_id has always used -- that's a labeling question,
        # not a resolution ambiguity, since both point at one directory.
        schemas_by_rendered: dict[str, list[str]] = {}
        for schema_name, fields in schemas.items():
            rendered = _render_id(config, fields)
            tried[schema_name] = rendered
            if rendered in self._names:
                schemas_by_rendered.setdefault(rendered, []).append(schema_name)

        if not schemas_by_rendered:
            raise ExperimentNotFoundError(config, kind, experiment_id(config, kind), tried)

        if len(schemas_by_rendered) == 1:
            (rendered, schema_names), = schemas_by_rendered.items()
            # Report the schema needing the FEWEST fields among those that
            # coincide: the extra fields on a longer coinciding schema were
            # absent from config, so they weren't what actually produced
            # this render -- the smaller schema is the tighter description.
            schema_name = min(schema_names, key=lambda s: len(schemas[s]))
            return LegacyMatch(path=self.root / rendered, schema=schema_name, rendered_id=rendered)

        # Multiple DIFFERENT directories matched -- real in the corpus: e.g.
        # experiments/ has both a homogeneous (P1) and a heterogeneous (P2)
        # QQQ/CNN-false/MLP-true/RNN-false/EPOCHS-10 pred directory sharing
        # every other field. Querying with a heterogeneous config (N_SEEDS_
        # PER_ARCH set) must not silently accept P1's match, which only
        # "matches" because P1's schema doesn't track N_SEEDS_PER_ARCH at
        # all and happens to coincide with the UNRELATED homogeneous dir.
        # Resolve this by dominance: if exactly one candidate's schema field
        # set is a superset of every other candidate's, it explains strictly
        # more of the config and wins. Otherwise the schemas are genuinely
        # incomparable (e.g. B0's EPOCHS vs B1's EPISODES both set) and this
        # is real ambiguity.
        field_sets = {
            rendered: set().union(*(schemas[name] for name in names)) for rendered, names in schemas_by_rendered.items()
        }
        dominant = [
            rendered
            for rendered, fs in field_sets.items()
            if all(fs >= other_fs for other_rendered, other_fs in field_sets.items() if other_rendered != rendered)
        ]
        if len(dominant) == 1:
            rendered = dominant[0]
            schema_name = max(schemas_by_rendered[rendered], key=lambda s: len(schemas[s]))
            return LegacyMatch(path=self.root / rendered, schema=schema_name, rendered_id=rendered)

        raise AmbiguousExperimentError(
            f"{len(schemas_by_rendered)} different directories matched this {kind} config "
            f"under different, non-dominating legacy schemas: "
            f"{[(names, rid) for rid, names in schemas_by_rendered.items()]}."
        )
