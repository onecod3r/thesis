"""Canonical project tree + dataset resolution.

Every path is anchored to this file's location (``src/``), so the constants
are correct no matter the CWD — notebooks (CWD = ``src/``) and the
``modules/scripts/`` CLIs (run from anywhere) share the same tree:

    src/data/
    ├── raw/        extracted-from-source data (e.g. POPSIGN landmark npz) — gitignored
    ├── cache/      reusable derived artifacts, one subtree per dataset
    │               (feature caches, analysis caches, manifests) — gitignored
    ├── temp/       throwaway scratch output; delete after use (cleanup_temp()) — gitignored
    ├── external/   third-party assets (MediaPipe .task model) — gitignored
    └── models/     the model registry — committed (index.csv + per-run meta.json
                    + assets/; weights *.pt gitignored). One flat folder per run:
                    models/<run_id>/ with run_id = seconds since the Unix epoch.

Dataset downloads are resolved *lazily* via :func:`gislr_dir` /
:func:`resolve_datasets` — importing this module never touches kagglehub
(the old module-level ``DATASETS`` downloaded every dataset at import time).
"""

import shutil
from pathlib import Path
from typing import Generic, TypedDict, TypeVar
# import kagglehub

SRC_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = SRC_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
TEMP_DIR = DATA_DIR / "temp"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = DATA_DIR / "models"
MODEL_INDEX = MODELS_DIR / "index.csv"


def read_env_file() -> dict[str, str]:
    """Minimal KEY=VALUE parse of the repo-root ``.env`` (works from any CWD).

    Lives here rather than in one consumer because more than one now needs it
    (POPSIGN's output drive, the model-checkpoint remote). Values already in the
    process environment win — see :func:`env_value`.
    """
    candidate = SRC_DIR.parent / ".env"
    if not candidate.exists():
        return {}
    pairs = (
        line.split("=", 1)
        for line in candidate.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    return {k.strip(): v.strip() for k, v in pairs}


def env_value(key: str, default: str | None = None) -> str | None:
    """Environment first, repo ``.env`` second, ``default`` last."""
    import os

    return os.environ.get(key) or read_env_file().get(key) or default


def cleanup_temp() -> None:
    """Delete src/data/temp entirely — call at the end of any notebook/script
    that wrote scratch output there (the temp tree is never reused)."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)


T = TypeVar("T")


class DatasetMap(TypedDict, Generic[T]):
    TRAIN: list[T]
    TEST: T
    GISLR: T


type DatasetIds = DatasetMap[str]
type Datasets = DatasetMap[Path]


DATASET_IDS: DatasetIds = {
    "TRAIN": [
        "mrgeislinger/popsign-asl-v1-0-game-train-a-e-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-f-m-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-n-s-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-t-z-signs",
    ],
    "TEST": "mrgeislinger/popsign-asl-v1-0-game-test",
    "GISLR": "asl-signs",
}


def gislr_dir() -> Path:
    """Download/resolve only the GISLR competition data (requires a Kaggle
    account that has accepted the asl-signs rules)."""
    import kagglehub

    return Path(kagglehub.competition_download(DATASET_IDS["GISLR"]))


def train_dirs() -> list[Path]:
    """Download/resolve only the enabled POPSIGN train datasets (~220GB for the
    one enabled part alone). Only 1 of 4 train parts is enabled so far
    (TODO §2.2); uncomment the rest in ``DATASET_IDS["TRAIN"]`` to download them."""
    import kagglehub

    return [
        Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][0])),
        Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][1])),
        Path(
            kagglehub.dataset_download(
                DATASET_IDS["TRAIN"][2],
                output_dir="D:/datasets/popsign/ train-n-s-signs",
            )
        ),
        # Path(
        #     kagglehub.dataset_download(
        #         DATASET_IDS["TRAIN"][3], output_dir="E:/datasets/"
        #     )
        # ),
    ]


def test_dir() -> Path:
    """Download/resolve only the POPSIGN test dataset."""
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_IDS["TEST"]))


def resolve_datasets() -> Datasets:
    """Download/resolve every enabled dataset (POPSIGN included — ~220GB for
    the one enabled train part alone). Only 1 of 4 POPSIGN train datasets is
    enabled so far (TODO §2.2); uncomment the rest to download them."""
    return {
        "TRAIN": train_dirs(),
        "TEST": test_dir(),
        "GISLR": gislr_dir(),
    }
