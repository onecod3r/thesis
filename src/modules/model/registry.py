"""Model registry: run folders, meta.json records, asset registration.

Layout (committed except *.pt — see .gitignore):

    src/data/models/
    ├── index.csv          queryable table of all runs (modules/scripts/build_model_index.py)
    └── <run_id>/          one FLAT folder per training run,
        │                  run_id = int seconds since the Unix epoch at training start
        ├── meta.json      the single machine-readable run record (schema:
        │                  README.md § "meta.json schema"; SCHEMA_VERSION here)
        ├── best.pt        best-val-accuracy checkpoint (gitignored)
        ├── last.pt        latest checkpoint, saved every epoch (gitignored)
        └── assets/        every other run artifact (plots, per-class CSVs,
                           landmark indices, eval summary) — each one linked
                           from meta.json["assets"]

Dataset/architecture/subset are meta.json fields (and index.csv columns), not
directory levels. A FINISHED run (early-stopped or epoch cap reached) is never
reused — every new training gets a fresh epoch-seconds folder; only an
*interrupted* run is resumed in place, via the pointer files under
data/cache/runs/.
"""

import json
import os
import time
from pathlib import Path

from modules.paths import CACHE_DIR, MODELS_DIR, SRC_DIR

SCHEMA_VERSION = 2
CKPT_BEST = "best.pt"
CKPT_LAST = "last.pt"
RUN_PTR_DIR = CACHE_DIR / "runs"   # <dataset>_<arch>_<tag>.txt -> active run dir

# top-level keys every meta.json must carry (README.md § "meta.json schema" is
# the human-readable source of truth; this tuple is the machine check)
REQUIRED_KEYS = (
    "schema_version", "run_id", "created", "dataset", "architecture",
    "model_name", "streaming", "subset", "coords", "n_landmarks",
    "feature_dim", "n_classes", "n_params", "split", "training",
    "hyperparameters", "metrics", "checkpoints", "assets", "notes",
)


def new_run_dir() -> Path:
    """Fresh run folder named by epoch seconds; sleeps past a collision (two
    runs starting within the same second)."""
    while (run_dir := MODELS_DIR / str(int(time.time()))).exists():
        time.sleep(1)
    (run_dir / "assets").mkdir(parents=True)
    return run_dir


def resolve_run_dir(pointer_key: str, is_finished) -> Path:
    """Run folder for this training: the pointer file makes re-runs land in the
    SAME folder only while that run is still unfinished (auto-resume after an
    interrupt, decided by ``is_finished(last_ckpt_path) -> bool``). A finished
    run is never reused — every new training gets a fresh epoch-seconds folder.
    """
    RUN_PTR_DIR.mkdir(parents=True, exist_ok=True)
    ptr = RUN_PTR_DIR / f"{pointer_key}.txt"
    if ptr.exists():
        last = Path(ptr.read_text().strip()) / CKPT_LAST
        if last.exists() and not is_finished(last):
            return last.parent  # interrupted -> resume in place
    run_dir = new_run_dir()
    ptr.write_text(str(run_dir))
    return run_dir


def pointer_run_dir(pointer_key: str) -> Path | None:
    """The run folder a pointer file currently targets (None before any run).
    Lets report/eval cells find their runs from disk, independent of the
    training cell's live state."""
    ptr = RUN_PTR_DIR / f"{pointer_key}.txt"
    return Path(ptr.read_text().strip()) if ptr.exists() else None


def load_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def write_meta(run_dir: Path, meta: dict) -> Path:
    """Validate against the schema key set and write meta.json atomically.
    Canonical-eval metrics already present on disk survive a rewrite (the
    training loop must never erase what eval_gru.py recorded)."""
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    assert not missing, f"meta.json missing schema keys: {missing}"
    path = run_dir / "meta.json"
    if path.exists():
        prev = json.loads(path.read_text())
        if prev.get("metrics", {}).get("eval_status") == "canonical":
            for k in ("eval_status", "overall_accuracy", "macro_accuracy",
                      "median_class_accuracy", "n_classes_below_50pct"):
                meta["metrics"][k] = prev["metrics"][k]
        meta["assets"] = {**prev.get("assets", {}), **meta["assets"]}
        if prev.get("notes") and meta["notes"] == "":
            meta["notes"] = prev["notes"]
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def register_assets(run_dir: Path, **assets: str) -> None:
    """Record asset files in meta.json["assets"] as run-dir-relative paths,
    e.g. register_assets(run_dir, learning_curves="assets/learning_curves.png")."""
    meta = load_meta(run_dir)
    for key, rel in assets.items():
        assert (run_dir / rel).exists(), f"asset not on disk: {run_dir / rel}"
        meta["assets"][key] = Path(rel).as_posix()
    write_meta(run_dir, meta)


def eval_command(run_dir: Path) -> str:
    """The canonical per-class eval invocation for a run (works from any CWD;
    shown here relative to the repo root)."""
    rel = run_dir.resolve().relative_to(SRC_DIR.parent).as_posix()
    return f".venv/Scripts/python.exe src/modules/scripts/eval_gru.py {rel}"
