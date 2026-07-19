"""Submission queue: which trained runs still need scoring on the official test set.

Every trained model should eventually be scored on the real test set, and Kaggle
allows **100 submissions per day**. Both facts are handled by making submission
state part of the run record (``meta.json["submission"]``, schema v3) and
selecting work with a query rather than by hand:

    dataset = 'gislr' AND submission.tested = false   LIMIT 100

so each pass submits only untested models and respects the daily cap by
construction. DuckDB reads the meta.json files directly (``registry.META_GLOB``)
— ``index.csv`` is the committed snapshot, never the query path.

The `tested` flag is deliberately dataset-agnostic (see
``registry.SUBMISSION_DEFAULT``): for GISLR it means a Kaggle submission landed,
for a dataset with no leaderboard it means a local held-out evaluation ran. The
queue function takes the dataset as a parameter and knows nothing about Kaggle.

**Kaggle mechanics caveat.** ``asl-signs`` is a *code competition*: the
documented command

    kaggle competitions submit -c asl-signs -f submission.zip \\
        -k <owner>/<notebook> -v <version> -m "<message>"

submits **through a Kaggle kernel** (``-k``/``-v``), so each model's zip has to
be attached to a kernel version first. Until that loop has been walked end to
end by hand, keep ``dry_run=True`` (the default) — it prints the exact commands
instead of firing them. The `kaggle` CLI must be declared in pyproject.toml and
installed via ``uv sync``.
"""

from dataclasses import dataclass
from pathlib import Path

from modules.model import registry as R
from modules.paths import MODELS_DIR

COMPETITION = "asl-signs"
DAILY_LIMIT = 100  # Kaggle submissions/day for this competition


def get_duckdb_conn():
    """In-memory DuckDB connection — the project's standard loading layer."""
    import duckdb

    return duckdb.connect()


def query_runs(where: str = "TRUE", order_by: str | None = None, limit: int | None = None):
    """Query the registry's meta.json files as a table; returns a DataFrame.

    Sorting metric: the canonical-eval ``overall_accuracy`` when it exists,
    falling back to the training-loop ``train_val_acc`` — the same rule
    ``build_model_index.py`` uses for ``val_acc``, so the leaderboard here and
    in index.csv can't disagree.
    """
    con = get_duckdb_conn()
    sql = f"""
        SELECT
            run_id, dataset, architecture, model_name, streaming, subset, coords,
            n_landmarks, feature_dim, n_params, created,
            metrics.train_val_acc          AS train_val_acc,
            metrics.eval_status            AS eval_status,
            metrics.overall_accuracy       AS overall_accuracy,
            metrics.macro_accuracy         AS macro_accuracy,
            metrics.median_class_accuracy  AS median_class_accuracy,
            metrics.n_classes_below_50pct  AS n_classes_below_50pct,
            COALESCE(metrics.overall_accuracy, metrics.train_val_acc) AS accuracy,
            COALESCE(submission.tested, false) AS tested,
            submission.platform            AS platform,
            submission.public_score        AS public_score,
            submission.submitted_at        AS submitted_at,
            training.regime                AS regime,
            training.epochs_trained        AS epochs_trained,
            training.best_epoch            AS best_epoch,
            training.wall_time_min         AS wall_time_min,
            notes
        FROM read_json_auto('{R.META_GLOB.replace(chr(92), "/")}', union_by_name = true)
        WHERE {where}
    """
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit is not None:
        sql += f" LIMIT {limit}"
    return con.execute(sql).df()


def leaderboard(dataset: str = "gislr", limit: int | None = None):
    """All runs for a dataset, best accuracy first — the evaluation notebook's §1."""
    return query_runs(where=f"dataset = '{dataset}'",
                      order_by="accuracy DESC NULLS LAST", limit=limit)


def untested_runs(dataset: str = "gislr", limit: int = DAILY_LIMIT):
    """The submission queue: untested runs for a dataset, best first, capped at
    the daily submission limit."""
    return query_runs(
        where=f"dataset = '{dataset}' AND COALESCE(submission.tested, false) = false",
        order_by="accuracy DESC NULLS LAST",
        limit=limit,
    )


@dataclass
class SubmissionResult:
    run_id: int
    submitted: bool
    command: str
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


def kaggle_submit_command(
    zip_path: Path, kernel: str, version: str | int, message: str,
    competition: str = COMPETITION,
) -> list[str]:
    return ["kaggle", "competitions", "submit", "-c", competition,
            "-f", str(zip_path), "-k", kernel, "-v", str(version), "-m", message]


def submit_run(
    run_dir: Path,
    kernel: str,
    version: str | int,
    message: str | None = None,
    competition: str = COMPETITION,
    dry_run: bool = True,
    mark: bool = True,
) -> SubmissionResult:
    """Submit one run's ``export/submission.zip`` and record the result.

    ``dry_run=True`` (default) prints the command and marks nothing — the run
    stays in the queue. A real submission marks the run ``tested`` so the next
    queue pass skips it; that happens only when the CLI exits 0, so a failed
    submission is retried rather than silently dropped.
    """
    import subprocess

    zip_path = run_dir / "export" / "submission.zip"
    meta = R.load_meta(run_dir)
    acc = meta["metrics"]["overall_accuracy"] or meta["metrics"]["train_val_acc"]
    message = message or (
        f"{meta['architecture']} · {meta['subset']}/{meta['coords']} · "
        f"run {meta['run_id']} · val {acc:.4f}"
    )
    cmd = kaggle_submit_command(zip_path, kernel, version, message, competition)
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)

    if dry_run:
        return SubmissionResult(meta["run_id"], False, printable)
    if not zip_path.is_file():
        return SubmissionResult(meta["run_id"], False, printable,
                                error=f"no submission.zip — export the run first: {zip_path}")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    if ok and mark:
        R.mark_tested(run_dir, platform="kaggle",
                      reference=f"{kernel}@v{version}", notes=message)
    return SubmissionResult(meta["run_id"], ok, printable, proc.stdout, proc.stderr,
                            None if ok else f"kaggle exited {proc.returncode}")


def run_dir_for(run_id) -> Path:
    return MODELS_DIR / str(run_id)
