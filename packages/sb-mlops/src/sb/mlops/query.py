"""DuckDB queries over the run registry.

DuckDB reads the meta.json files directly (``registry.META_GLOB``) —
``index.csv`` is the committed snapshot, never the query path. Used by the
evaluation notebook's leaderboard.

The ``submission.tested`` field on the run record is dataset-agnostic: it means
"scored on the official/held-out test set". For GISLR,
``sb.recognize.evaluate.evaluate_run`` sets it (``platform="local"``) itself,
because the canonical eval already scores on ``GISLR_Stratified``'s test split.
There is no Kaggle submission machinery in this repo.
"""

from pathlib import Path

from sb.mlops import registry as R
from sb.core.paths import MODELS_DIR


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


def run_dir_for(run_id) -> Path:
    return MODELS_DIR / str(run_id)
