"""Which trained runs still need scoring on the official/held-out test set.

Every trained model should eventually be scored, and submission state lives on
the run record (``meta.json["submission"]``, schema v3) so "which runs still
need this" is a query rather than something remembered by hand:

    dataset = 'gislr' AND submission.tested = false   LIMIT 100

DuckDB reads the meta.json files directly (``registry.META_GLOB``) —
``index.csv`` is the committed snapshot, never the query path.

The `tested` flag is deliberately dataset-agnostic (see
``registry.SUBMISSION_DEFAULT``): it means "scored on the official/held-out
test set", whatever that means for the dataset.

**GISLR (since 2026-09-16): local, not Kaggle.** GISLR moved off the live
``asl-signs`` competition to the self-produced ``GISLR_Stratified`` dataset,
whose ``test.csv`` split every canonical eval already scores on — that split
*is* the held-out test set now. ``sb.recognize.evaluate.evaluate_run`` marks
``tested`` itself (``platform="local"``) the moment it scores a run, so
``untested_runs``/the queue below should read as empty in steady state — there
is no active submission step for GISLR any more, and the
``kaggle_submit_command``/``submit_run`` helpers below are unused by anything
in this repo. They're kept only because the mechanics (a *code competition*
submits through a Kaggle kernel: ``-k``/``-v``, ``dry_run=True`` by default)
would still apply to some future dataset with a real leaderboard.
"""

from dataclasses import dataclass
from pathlib import Path

from sb.mlops import registry as R
from sb.core.paths import MODELS_DIR

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
    """Runs not yet scored on the official/held-out test set, best first,
    capped at the daily submission limit. For GISLR this should read as empty
    in steady state — `evaluate_run` marks a run tested the moment it's
    canonically evaluated, since that eval already runs on the held-out set."""
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
