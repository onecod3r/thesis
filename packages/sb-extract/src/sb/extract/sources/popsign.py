"""POPSIGN adapter: ~870 GB of raw game video, one clip per (label, video id).

The three dataset-specific facts, and nothing else:

- **where the videos are** — the kagglehub download, resolved lazily;
- **the manifest** — `data/cache/popsign/dataframes/{train,test}.csv`,
  regenerated from the raw video tree rather than trusted from a previous run;
- **where landmarks go** — `<POPSIGN_LANDMARKS_DRIVE>/data/raw/popsign/<split>/`,
  falling back to the repo's gitignored `data/raw/popsign/`. Hundreds of GB
  should not land in the repo by accident, which is why the drive is
  configurable and read from one place.

The label is carried by the artifact *path*, not stored inside the npz — a
deliberate part of the on-disk contract (`sb.core.schema`).
"""

from pathlib import Path

import pandas as pd

from sb.core.paths import CACHE_DIR, RAW_DIR, env_value
from sb.extract.sources import VideoSource

MANIFEST_DIR = CACHE_DIR / "popsign" / "dataframes"
SPLITS = ("train", "test")


def landmarks_root() -> Path:
    """`<POPSIGN_LANDMARKS_DRIVE>/data/raw/popsign`, or the repo's gitignored
    `data/raw/popsign` when the drive is unset."""
    drive = env_value("POPSIGN_LANDMARKS_DRIVE")
    return Path(drive) / "data" / "raw" / "popsign" if drive else RAW_DIR / "popsign"


def landmarks_dir(split: str) -> Path:
    assert split in SPLITS, f"unknown split {split!r}; have {SPLITS}"
    return landmarks_root() / split


def manifest_path(split: str) -> Path:
    return MANIFEST_DIR / f"{split}.csv"


def manifest(split: str) -> pd.DataFrame:
    """The video list for one split. Read-only here: regeneration from the raw
    tree lives in the extraction driver, which is where the raw video is."""
    path = manifest_path(split)
    if not path.is_file():
        raise FileNotFoundError(
            f"no POPSIGN {split} manifest at {path} — regenerate it from the raw "
            "video tree in experiments/extraction/popsign.0.dataset.extraction.ipynb")
    return pd.read_csv(path)


def artifact_path(out_dir: Path, row) -> Path:
    """`<out_dir>/<label>/<video_id>.npz` — the label lives in the path."""
    return out_dir / str(row.label) / f"{row.id}.npz"


POPSIGN = VideoSource(
    name="popsign",
    manifest=manifest,
    landmarks_dir=landmarks_dir,
    artifact_path=artifact_path,
)
