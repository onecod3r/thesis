"""Content addressing for feature caches — shared by every pipeline.

A cache is interchangeable with another only if *everything that determined its
bytes* matches, so the directory name is a hash of exactly that: the pipeline
and its version, the subset's actual index array, coords, the NaN policy, and
the dataset manifest the split came from.

The old layout keyed on the subset's *name*, so editing an index list in
``sb.core.subsets`` left the name unchanged and skip-if-exists silently handed
every later run the previous array (TODO §9.2). Under a content address the
edited subset simply addresses a different directory.

This module is pipeline-agnostic on purpose: ``base_v1`` and ``firstplace_v1``
each bind their own identity to it and expose the same three-function surface
(``cache_inputs`` / ``cache_key`` / ``cache_dir``), which is what makes them
substitutable.
"""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from sb.core.paths import CACHE_DIR

CACHE_KEY_FILE = "cache_key.json"  # sidecar: what a cache directory holds
KEY_LENGTH = 16  # hex chars of the sha256 used as the directory name

# these belong to the SPLIT, not to a pipeline, but they change the cached rows
# so they are part of the address
SPLIT_STRATEGY = "stratified 80/20 (fixed, GISLR_Stratified-provided)"
SPLIT_SEED = 42  # the upstream split's seed — recorded for provenance, not applied here


def features_root(dataset: str = "gislr") -> Path:
    """``data/cache/<dataset>/features`` — one cache subtree per dataset, which
    is the data-placement policy."""
    return CACHE_DIR / dataset / "features"


@lru_cache(maxsize=8)
def _hash_manifest(path_str: str, size: int, mtime: float) -> str | None:
    from sb.mlops.run import sha256_file

    return sha256_file(path_str)


def manifest_fingerprint(
    data_dir: Path | str, manifest: str | tuple[str, ...] = ("train.csv", "test.csv")
) -> str | None:
    """Fingerprint of the split manifest(s) a cache's rows came from.

    GISLR_Stratified is a Kaggle dataset with no content version exposed here,
    so this hash is what actually distinguishes one copy from another — and
    since ``train.csv`` and ``test.csv`` together define the fixed split, both
    must be hashed for the fingerprint to reflect either one changing.
    Memoized per-file on (path, size, mtime) — read on every key computation.
    """
    names = (manifest,) if isinstance(manifest, str) else manifest
    paths = [Path(data_dir) / name for name in names]
    if not all(p.is_file() for p in paths):
        return None
    parts: list[str] = []
    for p in paths:
        part = _hash_manifest(str(p), p.stat().st_size, p.stat().st_mtime)
        if part is None:
            return None
        parts.append(part)
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def cache_inputs(
    *,
    pipeline: str,
    pipeline_version: int,
    nan_policy: str,
    subset,
    coords: str,
    data_dir: Path | str,
    dataset: str = "gislr",
    manifest: str | tuple[str, ...] = ("train.csv", "test.csv"),
) -> dict:
    """Everything that determines a feature cache's bytes.

    ``indices_sha256`` is the point: it hashes the subset's actual index array,
    so a redefinition of ME_126 that keeps the name changes the key.
    """
    from sb.core.schema import N_LANDMARKS

    indices = np.asarray(subset.array, dtype=np.int64)
    return {
        "pipeline": pipeline,
        "pipeline_version": pipeline_version,
        "dataset": dataset,
        "subset": subset.name,
        "n_landmarks": int(indices.size),
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "coords": coords,
        "nan_policy": nan_policy,
        "rows_per_frame": N_LANDMARKS,
        "split_strategy": SPLIT_STRATEGY,
        "split_seed": SPLIT_SEED,
        "manifest_sha256": manifest_fingerprint(data_dir, manifest),
    }


def cache_key(inputs: dict) -> str:
    """16 hex chars addressing one feature cache — the value recorded as
    `provenance.feature_cache_key` and the directory it lives in."""
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:KEY_LENGTH]


def cache_dir(inputs: dict) -> Path:
    """``data/cache/<dataset>/features/<pipeline>/<key>/`` for these inputs."""
    return features_root(inputs["dataset"]) / inputs["pipeline"] / cache_key(inputs)


def write_sidecar(root: Path, inputs: dict, **extra) -> Path:
    """Write ``cache_key.json`` next to the arrays it describes.

    Without it a content-addressed directory is an opaque hash; with it, a cache
    whose key no config asks for any more can still say what it was.
    """
    path = root / CACHE_KEY_FILE
    payload = {"key": root.name, **inputs, **extra}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path
