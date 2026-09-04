"""Move the flat, name-keyed feature caches into content-addressed directories.

The old layout was ``data/cache/gislr/features/<split>_<tag>[_nan]_<part>.npy``,
keyed on the subset's *name* (``me126``, ``fp118-xy``). Editing an index list in
``subsets.py`` left the tag unchanged, so skip-if-exists silently handed every
later run the previous array. The new layout is
``features/<pipeline>/<key>/<split>_<part>.npy`` with ``<key>`` hashing the
inputs that determined the bytes — including the subset's actual indices
(TODO §9.2, ``modules/model/data.py::feature_cache_key``).

**Rename, never rebuild.** These caches are ~29 GB; recomputing them means
re-decoding 94,477 parquet files per subset. ``os.replace`` within the same
volume is a metadata operation, so this script moves the existing arrays and
writes each directory's ``cache_key.json`` sidecar.

**What is and isn't verified.** The key a legacy file receives is *asserted*
from today's subset definitions — the old layout recorded nothing about the
definitions that actually built it, which is the whole reason for this change.
What can be checked is checked: the offset array must have one entry per video
of the canonical split plus one, and the data array's size must equal
``frames × n_landmarks × len(coords) × 4`` bytes. That catches a changed
landmark *count* or a changed split; it cannot catch a different set of the same
number of indices. Each sidecar records ``assigned_by: "migration"`` and the
verification result, so a migrated key is never mistaken for one that was
computed at build time.

Runs from any CWD; prints a plan and changes nothing without ``--apply``:

    .venv/Scripts/python.exe src/modules/scripts/migrate_feature_caches.py
    .venv/Scripts/python.exe src/modules/scripts/migrate_feature_caches.py --apply
"""

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> src/

import numpy as np

from modules.dataset.landmark.subsets import SUBSETS
from modules.model import data as D
from modules.model import features as F
from modules.paths import gislr_dir

# <split>_<tag>[_nan]_<data|offsets>.npy — the legacy flat layout
LEGACY = re.compile(r"^(?P<split>train|val)_(?P<tag>.+?)(?P<nan>_nan)?_(?P<part>data|offsets)\.npy$")


def legacy_files() -> list[Path]:
    if not D.FEATURES_DIR.is_dir():
        return []
    return sorted(p for p in D.FEATURES_DIR.glob("*.npy")
                  if LEGACY.match(p.name) and ".tmp" not in p.name)


def tag_index() -> dict[str, tuple[str, str]]:
    """``subset_tag`` is not invertible on its own; build the lookup from the
    registered subsets, which is the only set of tags that can legitimately
    appear on disk."""
    return {D.subset_tag(name, coords): (name, coords)
            for name in SUBSETS for coords in ("xyz", "xy")}


def verify(group: dict, subset, coords: str, n_videos: int) -> tuple[bool, str]:
    """Shape check: offsets count matches the split, data size matches the
    frames × landmarks × channels the offsets imply."""
    off_path, data_path = group.get("offsets"), group.get("data")
    if off_path is None or data_path is None:
        return False, "incomplete pair (missing data or offsets)"
    offsets = np.load(off_path, mmap_mode="r")
    if len(offsets) - 1 != n_videos:
        return False, f"offsets hold {len(offsets) - 1} videos, split has {n_videos}"
    expected = int(offsets[-1]) * len(subset) * len(coords) * 4  # float32
    actual = data_path.stat().st_size - 128  # .npy header is 128 B here
    if abs(actual - expected) > 128:
        return False, (f"data is {actual / 1e9:.2f} GB, "
                       f"{len(subset)}x{len(coords)} implies {expected / 1e9:.2f} GB")
    return True, "shape OK"


def plan(data_dir: Path) -> list[dict]:
    """One entry per (pipeline, subset, coords, split) group found on disk."""
    tags = tag_index()
    sign2idx = D.load_label_map(data_dir)
    splits = dict(zip(("train", "val"), D.get_canonical_split(data_dir, sign2idx)))

    groups: dict[tuple, dict] = {}
    for path in legacy_files():
        m = LEGACY.match(path.name)
        assert m is not None
        tag, split, is_nan = m["tag"], m["split"], bool(m["nan"])
        if tag not in tags:
            print(f"SKIP {path.name}: tag {tag!r} matches no registered subset")
            continue
        groups.setdefault((tag, split, is_nan), {})[m["part"]] = path

    entries = []
    for (tag, split, is_nan), parts in sorted(groups.items()):
        name, coords = tags[tag]
        subset = SUBSETS[name]
        kw: dict[str, Any] = (
            dict(pipeline=F.PIPELINE, pipeline_version=F.PIPELINE_VERSION,
                 nan_policy=F.NAN_POLICY) if is_nan else {})
        ok, why = verify(parts, subset, coords, len(splits[split]))
        entries.append({
            "tag": tag, "split": split, "subset": name, "coords": coords,
            "pipeline": kw.get("pipeline", D.PIPELINE),
            "key": D.feature_cache_key(subset, coords, data_dir, **kw),
            "inputs": D.cache_inputs(subset, coords, data_dir, **kw),
            "parts": parts, "verified": ok, "why": why,
        })
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="actually move the files (default: print the plan only)")
    args = ap.parse_args()

    data_dir = gislr_dir()
    entries = plan(data_dir)
    if not entries:
        print(f"no legacy caches under {D.FEATURES_DIR} — nothing to migrate")
        return

    total = 0
    for e in entries:
        size = sum(p.stat().st_size for p in e["parts"].values())
        total += size
        flag = "ok " if e["verified"] else "!! "
        print(f"{flag}{e['split']:>5}/{e['tag']:<12} {size / 1e9:6.2f} GB  "
              f"-> {e['pipeline']}/{e['key']}/  ({e['why']})")
    print(f"\n{len(entries)} group(s), {total / 1e9:.1f} GB total")

    unverified = [e for e in entries if not e["verified"]]
    if unverified:
        print(f"\n{len(unverified)} group(s) FAILED the shape check. A failure means "
              "the on-disk array does not match today's subset definition — the "
              "cache is stale and must be rebuilt, not renamed. Those groups are "
              "left in place.")
    if not args.apply:
        print("\ndry run — pass --apply to move the files")
        return

    moved = 0
    for e in entries:
        if not e["verified"]:
            continue
        root = D.FEATURES_DIR / e["pipeline"] / e["key"]
        root.mkdir(parents=True, exist_ok=True)
        for part, src in e["parts"].items():
            dst = root / f"{e['split']}_{part}.npy"
            if dst.exists():
                print(f"  exists, leaving {src.name} alone: {dst}")
                continue
            os.replace(src, dst)  # same volume -> metadata only
            moved += 1
        D.write_cache_sidecar(
            root, e["inputs"],
            assigned_by="migration",
            migrated_on=date.today().isoformat(),
            # the legacy tag, not the split: train and val land in the same
            # directory and would otherwise overwrite each other's origin
            migrated_from=e["tag"] + ("_nan" if e["pipeline"] == F.PIPELINE else ""),
            # asserted from today's subset definitions, shape-verified only —
            # see the module docstring
            verified="shape",
        )
    print(f"\nmoved {moved} file(s); sidecars written")


if __name__ == "__main__":
    main()
