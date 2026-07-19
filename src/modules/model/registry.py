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
from datetime import datetime
from pathlib import Path

from modules.paths import CACHE_DIR, MODELS_DIR, SRC_DIR

SCHEMA_VERSION = 3
CKPT_BEST = "best.pt"
CKPT_LAST = "last.pt"
RUN_PTR_DIR = CACHE_DIR / "runs"  # <dataset>_<arch>_<tag>.txt -> active run dir

# DuckDB entry point: read_json_auto(META_GLOB) gives one row per run, so
# "the leaderboard" is a query over the run records themselves. index.csv is the
# committed snapshot of that query, never the query path.
META_GLOB = str(MODELS_DIR / "*" / "meta.json")

# top-level keys every meta.json must carry (README.md § "meta.json schema" is
# the human-readable source of truth; this tuple is the machine check)
REQUIRED_KEYS = (
    "schema_version",
    "run_id",
    "created",
    "dataset",
    "architecture",
    "model_name",
    "streaming",
    "subset",
    "coords",
    "n_landmarks",
    "feature_dim",
    "n_classes",
    "n_params",
    "split",
    "training",
    "hyperparameters",
    "metrics",
    "checkpoints",
    "assets",
    "submission",
    "notes",
)

# Schema v3. Deliberately dataset-agnostic: `tested` means "scored on the
# official/held-out test set", whatever that means for the dataset — Kaggle for
# GISLR, a local held-out split for datasets with no leaderboard. Nothing in the
# required keys mentions Kaggle, so POPSIGN runs use the same block unchanged.
SUBMISSION_DEFAULT: dict = {
    "tested": False,  # scored on the official/held-out test set yet?
    "platform": None,  # "kaggle" | "local" | … (null until tested)
    "submitted_at": None,  # ISO-8601 local timestamp
    "public_score": None,  # official metric (Kaggle public LB = accuracy)
    "private_score": None,
    "reference": None,  # kernel slug + version, submission id, … — free-form
    "notes": "",
}


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
    """Read a run record, normalized to the current schema.

    A pre-v3 record (no `submission` block) is filled in with the default on
    read. The gap is unambiguous — a run written before submission tracking
    existed has, by definition, never been submitted — and healing it here keeps
    every consumer working against records written by an older version of the
    training driver, including a Jupyter kernel still holding a stale import.

    Only this known gap is filled; anything else missing still trips
    `write_meta`'s schema assertion rather than being papered over.
    """
    meta = json.loads((run_dir / "meta.json").read_text())
    meta.setdefault("submission", dict(SUBMISSION_DEFAULT))
    return meta


def write_meta(run_dir: Path, meta: dict) -> Path:
    """Validate against the schema key set and write meta.json atomically.

    Facts recorded by something *other* than the training loop survive its
    per-epoch rewrites: canonical-eval metrics (eval_gru.py) and the submission
    block (the evaluation notebook / mark_tested). The training loop must never
    erase either.
    """
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    assert not missing, f"meta.json missing schema keys: {missing}"
    path = run_dir / "meta.json"
    if path.exists():
        prev = json.loads(path.read_text())
        if prev.get("metrics", {}).get("eval_status") == "canonical":
            for k in (
                "eval_status",
                "overall_accuracy",
                "macro_accuracy",
                "median_class_accuracy",
                "n_classes_below_50pct",
            ):
                meta["metrics"][k] = prev["metrics"][k]
        if prev.get("submission", {}).get("tested"):
            meta["submission"] = prev["submission"]
        meta["assets"] = {**prev.get("assets", {}), **meta["assets"]}
        if prev.get("notes") and meta["notes"] == "":
            meta["notes"] = prev["notes"]
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def migrate_meta(run_dir: Path) -> bool:
    """Bring one pre-v3 meta.json up to the current schema. Idempotent; returns
    True when the file was actually rewritten.

    v2 -> v3 adds the `submission` block. Runs written before it existed have
    simply never been submitted, so the default (tested=False) is the correct
    backfill — no information is invented.
    """
    path = run_dir / "meta.json"
    if not path.is_file():
        return False
    meta = json.loads(path.read_text())
    if meta.get("schema_version") == SCHEMA_VERSION and "submission" in meta:
        return False
    meta.setdefault("submission", dict(SUBMISSION_DEFAULT))
    meta["schema_version"] = SCHEMA_VERSION
    write_meta(run_dir, meta)
    return True


def migrate_all(models_dir: Path = MODELS_DIR) -> list[Path]:
    """Migrate every run folder in the registry; returns those actually changed."""
    return [d for d in sorted(models_dir.glob("*/")) if migrate_meta(d)]


def mark_tested(
    run_dir: Path,
    *,
    platform: str,
    reference: str | None = None,
    public_score: float | None = None,
    private_score: float | None = None,
    notes: str = "",
) -> dict:
    """Record that a run has been scored on the official/held-out test set.

    `platform` is free-form ("kaggle", "local", …) so the same call works for
    datasets with no leaderboard. Scores stay None when the platform reports
    them asynchronously — the point of `tested` is "don't submit this again",
    which is true the moment the submission lands.
    """
    meta = load_meta(run_dir)
    meta["submission"] = {
        **dict(SUBMISSION_DEFAULT),
        "tested": True,
        "platform": platform,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "public_score": public_score,
        "private_score": private_score,
        "reference": reference,
        "notes": notes,
    }
    write_meta(run_dir, meta)
    return meta["submission"]


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
