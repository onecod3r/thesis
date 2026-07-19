"""Run POPSIGN landmark extraction (TODO §2.2) as a real script.

**Why a script and not a notebook cell.** Extraction parallelises with
``multiprocessing``, which on Windows uses the *spawn* start method: children
re-import ``__main__``. Inside a Jupyter kernel ``__main__`` is the kernel, not
a module, so the workers never come up and the pool blocks forever — the cell
sits at 0% with an idle CPU and no error (observed: 30 minutes, 0 workers
alive). ``extraction._assert_pool_usable`` now refuses that case outright, and
this CLI is the supported way to run both the pilot and the bulk extraction.

The notebook (``src/popsign.0.dataset.extraction.ipynb``) still owns *what* is
extracted and every decision made from the results — manifest verification,
picking the operating point from the pilot table, the ETA/disk projection and
QC. This script only executes the parts that need a worker pool:

    pilot   worker-count benchmark on a seeded, disjoint-slice sample
            -> data/cache/popsign/extraction/pilot_results.csv
               npz to data/temp/popsign_pilot/ (throwaway, notebook cleans up)

    run     the full resumable extraction for a split
            -> <POPSIGN_LANDMARKS_DRIVE or src/data>/data/raw/popsign/<split>/

Both are fully resumable: re-running skips videos already extracted, so an
interrupt costs at most the videos in flight.

Usage (any CWD — the script bootstraps its own imports):

    .venv/Scripts/python.exe src/modules/scripts/extract_popsign.py pilot
    .venv/Scripts/python.exe src/modules/scripts/extract_popsign.py run train
    .venv/Scripts/python.exe src/modules/scripts/extract_popsign.py run test --limit 2000
    .venv/Scripts/python.exe src/modules/scripts/extract_popsign.py run train --workers 14

Confidence thresholds (TODO §2.3) come from the tuning notebook's chosen config:

    ... run train --confidence default
    ... run train --confidence '{"min_pose_detection_confidence": 0.3}'
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> src/

import pandas as pd

import modules.dataset.landmark.extraction as ex
from modules.paths import CACHE_DIR, TEMP_DIR

NB_CACHE = CACHE_DIR / "popsign" / "extraction"
DATAFRAMES = CACHE_DIR / "popsign" / "dataframes"
TUNING_CONFIGS = CACHE_DIR / "popsign" / "confidence_tuning" / "configs.json"

SEED = 42                        # must match the notebook's pilot sampling seed
PILOT_MAX_VIDEOS = 100
DEFAULT_WORKER_COUNTS = (6, 10, 14, 19)
VIDEOS_PER_TRIAL = 20


def resolve_confidence(spec: str | None) -> dict:
    """`--confidence` → a threshold dict.

    Accepts a named config from the tuning notebook's ``configs.json`` (so the
    bulk run uses literally the arm that was scored), inline JSON, or nothing
    at all (MediaPipe's own defaults).
    """
    if not spec:
        return {}
    if spec.lstrip().startswith("{"):
        return json.loads(spec)
    if not TUNING_CONFIGS.is_file():
        sys.exit(f"--confidence {spec!r} names a tuning config but {TUNING_CONFIGS} "
                 f"does not exist; pass inline JSON instead")
    configs = json.loads(TUNING_CONFIGS.read_text())
    if spec not in configs:
        sys.exit(f"unknown config {spec!r}; have {sorted(configs)}")
    return configs[spec]


def load_split(split: str) -> pd.DataFrame:
    csv = DATAFRAMES / f"{split}.csv"
    if not csv.is_file():
        sys.exit(f"missing manifest {csv}\n"
                 f"Regenerate the POPSIGN video manifests first (TODO §0.4/§2.2) — "
                 f"§1 of popsign.0.dataset.extraction.ipynb verifies them.")
    df = pd.read_csv(csv)
    missing = {"file_path", "label", "id"} - set(df.columns)
    if missing:
        sys.exit(f"{csv}: missing column(s) {sorted(missing)}")
    return df


def cmd_pilot(args) -> None:
    """Worker-count benchmark on a seeded 100-video sample (throwaway output)."""
    train = load_split("train")
    NB_CACHE.mkdir(parents=True, exist_ok=True)

    # the sample is recorded so every re-run benchmarks the same videos
    sample_file = NB_CACHE / "pilot_sample.json"
    if sample_file.is_file():
        ids = json.loads(sample_file.read_text())["ids"]
    else:
        ids = train.sample(PILOT_MAX_VIDEOS, random_state=SEED)["id"].tolist()
        sample_file.write_text(json.dumps({"seed": SEED, "n": len(ids), "ids": ids},
                                          indent=2))
    pilot_df = train.set_index("id").loc[ids].reset_index()

    counts = tuple(args.worker_counts or DEFAULT_WORKER_COUNTS)
    needed = len(counts) * args.videos_per_trial
    if needed > len(pilot_df):
        sys.exit(f"{needed} videos needed for {len(counts)} trials but the pilot "
                 f"sample holds {len(pilot_df)} — lower --videos-per-trial")

    # The pilot measures THROUGHPUT, and extract_dataset skips videos already
    # extracted — so leftover npz from an earlier (or interrupted) pilot would
    # silently make a trial time ~0 videos and report a meaningless rate. The
    # output is throwaway by policy, so always start from an empty temp tree.
    pilot_root = TEMP_DIR / "popsign_pilot"
    if pilot_root.exists():
        import shutil
        stale = len(list(pilot_root.rglob("*.npz")))
        shutil.rmtree(pilot_root)
        print(f"cleared stale pilot output ({stale} npz) — a benchmark must not "
              f"resume, it would time an empty trial")

    print(f"pilot: {len(counts)} trials x {args.videos_per_trial} disjoint videos "
          f"= {needed} extractions, worker counts {counts}")

    results = ex.benchmark_worker_counts(
        pilot_df, worker_counts=counts, videos_per_trial=args.videos_per_trial,
        pilot_root=pilot_root, cpu_fraction=args.cpu_fraction,
        ram_limit_pct=args.ram_limit_pct, confidence=resolve_confidence(args.confidence))
    out_csv = NB_CACHE / "pilot_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\n{results.to_string(index=False)}\n\nwrote {out_csv}")
    print("Next: run §2 of popsign.0.dataset.extraction.ipynb to pick the operating "
          "point, record eta.json, and delete the temp pilot output.")


def cmd_run(args) -> None:
    """The real thing: resumable extraction of a whole split."""
    confidence = resolve_confidence(args.confidence)   # fail fast on a bad name
    videos = load_split(args.split)

    n_workers = args.workers
    if n_workers is None:                       # prefer the pilot's operating point
        eta = NB_CACHE / "eta.json"
        if eta.is_file():
            n_workers = int(json.loads(eta.read_text())["best_n_workers"])
            print(f"workers from {eta}: {n_workers}")
        else:
            n_workers = ex.default_n_workers(args.cpu_fraction)
            print(f"no eta.json (pilot not run) — defaulting to {n_workers} workers "
                  f"({args.cpu_fraction:.0%} of cores)")

    summary = ex.extract_dataset(
        videos, split=args.split, out_root=ex.landmarks_root(), n_workers=n_workers,
        cpu_fraction=args.cpu_fraction, ram_limit_pct=args.ram_limit_pct,
        limit=args.limit, confidence=confidence)

    NB_CACHE.mkdir(parents=True, exist_ok=True)
    out_json = NB_CACHE / f"{args.split}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n{json.dumps(summary, indent=2)}\n\nwrote {out_json}")
    print("Next: §5 (QC) in popsign.0.dataset.extraction.ipynb — it reads the "
          "manifest only, so it is safe to run mid-extraction.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # The shared options are attached to BOTH the top-level parser and every
    # subparser, so they parse in either position -- `... --confidence X run
    # train` and `... run train --confidence X` both work. Without this, the
    # second form (which is what anyone naturally types, and what the notebook's
    # handoff cells print) dies with "unrecognized arguments".
    #
    # The subparser copies use SUPPRESS defaults on purpose: a subparser writes
    # its defaults into the *same* namespace after the main parser has filled
    # it, so a real default here would silently clobber a value given before the
    # subcommand. SUPPRESS means "only set me if actually passed".
    def add_shared(parser, *, suppress: bool) -> None:
        d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
        # NB: argparse %-formats help strings, so a literal % must be doubled
        parser.add_argument("--cpu-fraction", type=float,
                            default=d(ex.DEFAULT_CPU_FRACTION),
                            help=f"CPU cap (default {ex.DEFAULT_CPU_FRACTION:.0%})"
                                 .replace("%", "%%"))
        parser.add_argument("--ram-limit-pct", type=float,
                            default=d(ex.DEFAULT_RAM_LIMIT_PCT),
                            help="feeder blocks above this system-RAM usage")
        parser.add_argument("--confidence", default=d(None),
                            help="tuning-config name (from "
                                 "confidence_tuning/configs.json) or inline JSON; "
                                 "omit for MediaPipe defaults")

    add_shared(ap, suppress=False)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pilot = sub.add_parser("pilot", help="worker-count benchmark (throwaway output)")
    p_pilot.add_argument("--worker-counts", nargs="*", type=int, default=None,
                         help=f"default: {DEFAULT_WORKER_COUNTS}")
    p_pilot.add_argument("--videos-per-trial", type=int, default=VIDEOS_PER_TRIAL)
    add_shared(p_pilot, suppress=True)
    p_pilot.set_defaults(func=cmd_pilot)

    p_run = sub.add_parser("run", help="full resumable extraction of a split")
    p_run.add_argument("split", choices=["train", "test"])
    p_run.add_argument("--workers", type=int, default=None,
                       help="default: eta.json's best_n_workers, else the CPU cap")
    p_run.add_argument("--limit", type=int, default=None,
                       help="process at most N pending videos this run (staged runs)")
    add_shared(p_run, suppress=True)
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
