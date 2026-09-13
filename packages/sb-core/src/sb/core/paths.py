"""Canonical repo tree + lazy dataset resolution.

Nothing here depends on the CWD, and — since the workspace restructure — nothing
depends on where the *importing* file lives either: the repo root is discovered
by walking up from this file for the marker every checkout has (``.git``, or the
workspace-root ``pyproject.toml``). That is what lets a notebook in
``experiments/recognition/`` and a console script installed into ``.venv``
resolve the same tree. ``SIGNBRIDGE_ROOT`` overrides it when the packages are
installed somewhere the walk cannot reach.

    <repo>/
    ├── data/          NEVER committed, no exceptions
    │   ├── raw/       extracted-from-source data (POPSIGN landmark npz)
    │   ├── cache/     reusable derived artifacts, one subtree per dataset
    │   ├── temp/      throwaway scratch; delete after use (cleanup_temp())
    │   └── external/  third-party assets (MediaPipe .task model)
    ├── registry/      COMMITTED run records
    │   ├── runs/<run_id>/meta.json + assets/ (weights *.pt gitignored)
    │   ├── index.csv
    │   └── aliases.json
    └── experiments/   notebooks + their configs

The registry moved out of the data tree deliberately: committed artifacts should
not live inside a directory whose whole policy is "never commit this".

Dataset downloads are resolved *lazily* via :func:`gislr_dir` /
:func:`resolve_datasets` — importing this module never touches kagglehub.
"""

import os
import shutil
from pathlib import Path
from typing import Generic, TypedDict, TypeVar

# markers that identify the workspace root, most specific first
_ROOT_MARKERS = ("pyproject.toml", ".git")


def _find_repo_root() -> Path:
    """Walk up from this file to the workspace root.

    ``SIGNBRIDGE_ROOT`` wins when set. Otherwise the first ancestor holding a
    ``pyproject.toml`` that declares the uv workspace is the root; a bare
    ``.git`` directory is the fallback for a checkout without one.
    """
    override = os.environ.get("SIGNBRIDGE_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text(
            encoding="utf-8", errors="ignore"
        ):
            return parent
        if (parent / ".git").exists():
            return parent
    # installed outside a checkout: fall back to the CWD rather than guessing
    return Path.cwd()


ROOT_DIR = _find_repo_root()
PACKAGES_DIR = ROOT_DIR / "packages"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
TEMP_DIR = DATA_DIR / "temp"
EXTERNAL_DIR = DATA_DIR / "external"

REGISTRY_DIR = ROOT_DIR / "registry"
MODELS_DIR = REGISTRY_DIR / "runs"  # one folder per run: registry/runs/<run_id>/
MODEL_INDEX = REGISTRY_DIR / "index.csv"
ALIASES = REGISTRY_DIR / "aliases.json"

SCHEMAS_DIR = ROOT_DIR / "schemas"
DOCS_DIR = ROOT_DIR / "docs"

# kept as an alias so nothing that still says SRC_DIR silently points elsewhere;
# the `src/` tree is gone, and this is the root everything now hangs off
SRC_DIR = ROOT_DIR


def read_env_file() -> dict[str, str]:
    """Minimal KEY=VALUE parse of the repo-root ``.env`` (works from any CWD).

    Lives here rather than in one consumer because more than one now needs it
    (POPSIGN's output drive, the model-checkpoint remote). Values already in the
    process environment win — see :func:`env_value`.
    """
    candidate = ROOT_DIR / ".env"
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
    return os.environ.get(key) or read_env_file().get(key) or default


def cleanup_temp() -> None:
    """Delete data/temp entirely — call at the end of any notebook/script
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


# indices into DATASET_IDS["TRAIN"] that are downloaded/extracted so far
# (TODO §2.2); index 3 (t-z) stays disabled until enabled here.
ENABLED_TRAIN_INDICES: tuple[int, ...] = (0, 1, 2)


def gislr_dir() -> Path:
    """Download/resolve only the GISLR competition data (requires a Kaggle
    account that has accepted the asl-signs rules)."""
    import kagglehub

    return Path(kagglehub.competition_download(DATASET_IDS["GISLR"]))


def train_dir(index: int) -> Path:
    """Download/resolve exactly one POPSIGN train part (~170-200GB each) — the
    staged-extraction unit. Prefer this over ``train_dirs()`` for anything that
    processes one part at a time (the main extraction notebook): it downloads
    only that part, so only one part's raw video ever sits on disk at once.

    Goes through kagglehub's own default cache (``~/.cache/kagglehub/datasets/``)
    — no external-drive ``output_dir`` (dropped 2026-09-13; kagglehub's
    re-download check only covers its default cache dir, so a pinned external
    ``output_dir`` broke every subsequent call with ``FileExistsError`` once the
    download completed)."""
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_IDS["TRAIN"][index]))


def train_dirs() -> list[Path]:
    """Download/resolve every enabled POPSIGN train part **at once**
    (``ENABLED_TRAIN_INDICES`` — 3 parts, ~600GB total today). Only for
    contexts that genuinely want everything on disk together; the staged main
    extraction notebook downloads/extracts/deletes one part at a time via
    ``train_dir()`` instead."""
    return [train_dir(i) for i in ENABLED_TRAIN_INDICES]


def test_dir() -> Path:
    """Download/resolve only the POPSIGN test dataset."""
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_IDS["TEST"]))


def dataset_cache_dir(handle: str) -> Path:
    """Local kagglehub cache directory for a *dataset* handle (train/test
    parts — never GISLR, which is a competition download and lives under a
    separate ``competitions/`` cache subtree entirely). Resolves kagglehub's
    own default cache layout directly, so it works without triggering a
    download."""
    from kagglehub.cache import get_cached_path
    from kagglehub.handle import parse_dataset_handle

    return Path(get_cached_path(parse_dataset_handle(handle)))


def clear_dataset_cache(handle: str) -> None:
    """Delete one dataset's local kagglehub cache (downloaded files +
    completion markers) so the staged extraction notebook can move to the next
    part without holding both on disk. A no-op if nothing is cached. Never
    touches GISLR (``dataset_cache_dir`` only resolves dataset handles, and
    GISLR's id is never passed here)."""
    d = dataset_cache_dir(handle)
    if d.exists():
        shutil.rmtree(d)


def resolve_datasets() -> Datasets:
    """Download/resolve every enabled dataset (POPSIGN included — ~220GB for
    the one enabled train part alone). Only 1 of 4 POPSIGN train datasets is
    enabled so far (TODO §2.2); uncomment the rest to download them."""
    return {
        "TRAIN": train_dirs(),
        "TEST": test_dir(),
        "GISLR": gislr_dir(),
    }
