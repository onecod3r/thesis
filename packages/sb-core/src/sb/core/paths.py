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


# PAUSED (2026-09-22): every POPSIGN entry/resolver below (TRAIN/TEST,
# train_dir/train_dirs/test_dir, ENABLED_TRAIN_INDICES) is deprecated along
# with the POPSIGN workstream (TODO §2) -- kept working, not removed, in case
# a future raw-video dataset reuses the same resolution pattern, but nothing
# should call these for new work. GISLR is the only active dataset.
DATASET_IDS: DatasetIds = {
    "TRAIN": [
        "mrgeislinger/popsign-asl-v1-0-game-train-a-e-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-f-m-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-n-s-signs",
        "mrgeislinger/popsign-asl-v1-0-game-train-t-z-signs",
    ],
    "TEST": "mrgeislinger/popsign-asl-v1-0-game-test",
    # GISLR_Stratified (2026-09-16): a self-produced Kaggle *dataset*, not the
    # asl-signs *competition* — pre-converted (T,543,3) npz per sequence plus
    # its own train.csv/test.csv (fixed 80/20 split, stratified on sign). Ships
    # no sign_to_prediction_index_map.json, so gislr_dir() derives and writes
    # one. Replaces the live asl-signs parquet download entirely.
    "GISLR": "bracu23101281/gislr-stratified",
}


# indices into DATASET_IDS["TRAIN"] that are downloaded/extracted — all 4
# parts (a-e, f-m, n-s, t-z) are enabled (TODO §2.2).
ENABLED_TRAIN_INDICES: tuple[int, ...] = (0, 1, 2, 3)


def gislr_dir() -> Path:
    """Download/resolve the GISLR_Stratified dataset: a regular Kaggle
    *dataset* download (unlike the asl-signs *competition* download this
    replaced), so it lives under kagglehub's normal dataset cache and
    ``dataset_cache_dir()``/``clear_dataset_cache()`` both apply to it.

    It ships train.csv/test.csv (uid, sign, participant_id, sequence_id,
    split, npz_relpath) but not the competition's
    ``sign_to_prediction_index_map.json`` — one is derived (sorted signs ->
    index, stable across machines) and written into the resolved dir the
    first time it's missing, so every existing GISLR consumer
    (``vocab.load_label_map``) keeps working unchanged.
    """
    import json

    import kagglehub
    import pandas as pd

    d = Path(kagglehub.dataset_download(DATASET_IDS["GISLR"]))
    label_map_path = d / "sign_to_prediction_index_map.json"
    if not label_map_path.is_file():
        signs = set(pd.read_csv(d / "train.csv")["sign"]) | set(
            pd.read_csv(d / "test.csv")["sign"]
        )
        label_map_path.write_text(
            json.dumps({sign: i for i, sign in enumerate(sorted(signs))}),
            encoding="utf-8",
        )
    return d


ASL_LEX_ID = "bracu23101281/asl-lex"  # TODO §3.8, public mirror of ASL-LEX 2.0 signdata.csv (OSF zpha4)


def asl_lex_dir() -> Path:
    """Download/resolve the ASL-LEX 2.0 phonology dataset (a Kaggle mirror of
    OSF project ``zpha4``'s ``signdata.csv`` and friends, republished as
    ``SignData.csv``/``ASLLEXR.csv``/``IconD_trial.csv``/``IconicityTrial.csv``/
    ``NeigborPairs.csv`` so ``sb.recognize.aslex`` doesn't depend on OSF being
    reachable). A regular Kaggle dataset download, like :func:`gislr_dir`."""
    import kagglehub

    return Path(kagglehub.dataset_download(ASL_LEX_ID))


GISLR_SENTENCES_ID = "bracu23101281/gislr-sentences"  # TODO §12.1, private


def _sentences_root(d: Path) -> Path | None:
    """The folder holding ``sequences.csv``: the dataset root, or one level
    down (a dataset published from a Kaggle notebook's output keeps the
    notebook's ``gislr-sentences/`` folder)."""
    for cand in (d, *sorted(p for p in d.iterdir() if p.is_dir())):
        if (cand / "sequences.csv").is_file():
            return cand
    return None


def gislr_sentences_dir(version: str = "v1", *, allow_local: bool = True) -> Path:
    """Resolve GISLR-Sentences (the continuous multi-sign test set, TODO §12.1).

    The Kaggle dataset (``GISLR_SENTENCES_ID``, via kagglehub) is the
    canonical copy. Until it is published or reachable, fall back to the
    local build ``data/cache/gislr/sentences/<version>/`` written by
    ``gislr.0.dataset.sentences.ipynb`` -- printed, never silent, so a result
    always says which copy it read. Either copy's ``build_info.json`` must
    carry the requested ``version``.
    """
    import json

    local = CACHE_DIR / "gislr" / "sentences" / version
    try:
        import kagglehub

        root = _sentences_root(Path(kagglehub.dataset_download(GISLR_SENTENCES_ID)))
        if root is None:
            raise FileNotFoundError(f"{GISLR_SENTENCES_ID} has no sequences.csv")
        source = f"kaggle:{GISLR_SENTENCES_ID}"
    except Exception as e:  # not published yet, offline, or no access
        if not (allow_local and (local / "sequences.csv").is_file()):
            raise
        print(f"gislr_sentences_dir: kaggle copy unavailable ({type(e).__name__}); "
              f"using the local build {local}")
        root, source = local, "local"
    built = json.loads((root / "build_info.json").read_text(encoding="utf-8"))
    if built["dataset_version"] != version:
        raise ValueError(f"{source} holds GISLR-Sentences {built['dataset_version']}, wanted {version}")
    return root


GISLR_GAPCORPUS_ID = "bracu23101281/gislr-gapcorpus"  # TODO §17, not yet published


def _gapcorpus_root(d: Path) -> Path | None:
    """The folder holding ``train.csv``/``test.csv``: the dataset root, or one
    level down (a Kaggle-notebook-output dataset keeps the notebook's
    ``gislr-gapcorpus/`` folder), mirroring :func:`_sentences_root`."""
    for cand in (d, *sorted(p for p in d.iterdir() if p.is_dir())):
        if (cand / "train.csv").is_file() and (cand / "test.csv").is_file():
            return cand
    return None


def gislr_gapcorpus_dir(version: str = "v1", *, allow_local: bool = True) -> Path:
    """Resolve GISLR-GapCorpus (the sign-vs-gap training corpus, TODO §17).

    Same shape as :func:`gislr_sentences_dir`: the Kaggle dataset
    (``GISLR_GAPCORPUS_ID``) is canonical once published; until then, falls
    back to a local build at ``data/cache/gislr/gapcorpus/<version>/`` (e.g. a
    smoke-scale run of ``gislr.0.dataset.gapcorpus-kaggle.ipynb`` copied
    there by hand), printed, never silent.
    """
    import json

    local = CACHE_DIR / "gislr" / "gapcorpus" / version
    try:
        import kagglehub

        root = _gapcorpus_root(Path(kagglehub.dataset_download(GISLR_GAPCORPUS_ID)))
        if root is None:
            raise FileNotFoundError(f"{GISLR_GAPCORPUS_ID} has no train.csv/test.csv")
        source = f"kaggle:{GISLR_GAPCORPUS_ID}"
    except Exception as e:  # not published yet, offline, or no access
        if not (allow_local and (local / "train.csv").is_file()):
            raise
        print(f"gislr_gapcorpus_dir: kaggle copy unavailable ({type(e).__name__}); "
              f"using the local build {local}")
        root, source = local, "local"
    built = json.loads((root / "build_info.json").read_text(encoding="utf-8"))
    if built["dataset_version"] != version:
        raise ValueError(f"{source} holds GISLR-GapCorpus {built['dataset_version']}, wanted {version}")
    return root


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
    """Local kagglehub cache directory for a *dataset* handle (POPSIGN
    train/test parts, and GISLR_Stratified since 2026-09-16 — the old
    asl-signs *competition* download lived under a separate
    ``competitions/`` cache subtree, but GISLR is a regular dataset now).
    Resolves kagglehub's own default cache layout directly, so it works
    without triggering a download."""
    from kagglehub.cache import get_cached_path
    from kagglehub.handle import parse_dataset_handle

    return Path(get_cached_path(parse_dataset_handle(handle)))


def clear_dataset_cache(handle: str) -> None:
    """Delete one dataset's local kagglehub cache (downloaded files +
    completion markers) so the staged extraction notebook can move to the next
    part without holding both on disk. A no-op if nothing is cached. Applies
    to any dataset handle, GISLR_Stratified included."""
    d = dataset_cache_dir(handle)
    if d.exists():
        shutil.rmtree(d)


def resolve_datasets() -> Datasets:
    """Download/resolve every enabled dataset. **POPSIGN is deprecated
    (2026-09-22, TODO §2)** -- calling this now downloads hundreds of GB for
    a paused workstream; use `gislr_dir()` alone for anything GISLR-only,
    which is everything active in this repo today."""
    return {
        "TRAIN": train_dirs(),
        "TEST": test_dir(),
        "GISLR": gislr_dir(),
    }
