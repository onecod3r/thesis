"""Run the POPSIGN confidence-tuning sweep (TODO §2.3) as a real script.

**Why a script and not a notebook cell.** Extraction parallelises with
``multiprocessing``, which on Windows uses the *spawn* start method: children
re-import ``__main__``. Inside a Jupyter kernel ``__main__`` is the kernel, not
a module, so the workers never come up and the pool blocks forever — the cell
sits at 0% with an idle CPU and no error (observed: 30 minutes, 0 workers
alive). ``extraction._assert_pool_usable`` now refuses that case outright, and
this CLI is the supported way to run the sweep.

Reads the sample and config grid the notebook wrote to the cache, so the
notebook still owns *what* is swept and this script only executes it:

    data/cache/popsign/confidence_tuning/sample.json    (notebook §2)
    data/cache/popsign/confidence_tuning/configs.json   (notebook §3)

Writes one npz tree + manifest per config, plus ``sweep_summary.csv``. Fully
resumable — re-running skips videos already extracted, so an interrupt costs at
most the videos in flight. Scoring and the visual check stay in the notebook
(§5/§6), which needs no worker pool.

Usage (any CWD — the script bootstraps its own imports):

    .venv/Scripts/python.exe src/modules/scripts/tune_confidence.py
    .venv/Scripts/python.exe src/modules/scripts/tune_confidence.py --configs default pose_strict
    .venv/Scripts/python.exe src/modules/scripts/tune_confidence.py --workers 8 --force
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> src/

import pandas as pd
from tqdm.auto import tqdm

import modules.dataset.landmark.extraction as ex
from modules.paths import CACHE_DIR

NB_CACHE = CACHE_DIR / "popsign" / "confidence_tuning"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset of config names to run (default: all in configs.json)")
    # NB: argparse %-formats help strings, so a literal % must be doubled
    ap.add_argument("--workers", type=int, default=None,
                    help=f"worker processes (default: {ex.DEFAULT_CPU_FRACTION:.0%} "
                         f"of cores)".replace("%", "%%"))
    ap.add_argument("--cpu-fraction", type=float, default=ex.DEFAULT_CPU_FRACTION)
    ap.add_argument("--ram-limit-pct", type=float, default=ex.DEFAULT_RAM_LIMIT_PCT)
    ap.add_argument("--force", action="store_true",
                    help="delete each config's npz + manifest and re-extract")
    args = ap.parse_args()

    sample_json, configs_json = NB_CACHE / "sample.json", NB_CACHE / "configs.json"
    for p in (sample_json, configs_json):
        if not p.is_file():
            sys.exit(f"missing {p}\nRun §2 and §3 of "
                     f"popsign.0.dataset.confidence-tuning.ipynb first.")

    sample = pd.DataFrame(json.loads(sample_json.read_text()))
    configs: dict = json.loads(configs_json.read_text())
    names = args.configs or list(configs)
    unknown = set(names) - set(configs)
    if unknown:
        sys.exit(f"unknown config(s): {sorted(unknown)}; have {sorted(configs)}")

    n_workers = args.workers or ex.default_n_workers(args.cpu_fraction)
    print(f"{len(names)} config(s) x {len(sample)} videos = "
          f"{len(names) * len(sample)} extractions, {n_workers} workers")

    npz_root = NB_CACHE / "npz"
    summaries = []
    # ONE bar across the whole sweep (repo convention): total = every
    # (config, video) unit, so resumed work shows up as instant progress
    bar = tqdm(total=len(names) * len(sample), desc="confidence sweep",
               dynamic_ncols=True)
    for name in names:
        out_root = npz_root / name
        if args.force and out_root.exists():
            import shutil
            shutil.rmtree(out_root)
        done_before = bar.n
        summary = ex.extract_dataset(
            sample, split="sample", out_root=out_root, confidence=configs[name],
            n_workers=n_workers, cpu_fraction=args.cpu_fraction,
            ram_limit_pct=args.ram_limit_pct, bar=bar)
        # already-done videos never reach the bar — advance past them so the
        # total stays honest on a resumed run
        bar.update(len(sample) - (bar.n - done_before))
        summary["config"] = name
        summaries.append(summary)
        bar.write(f"  {name}: {summary['n_done']} done, {summary['n_failed']} failed, "
                  f"{summary['n_skipped']} skipped ({summary['elapsed_s']}s)")
    bar.close()

    sweep = pd.DataFrame(summaries)[
        ["config", "n_done", "n_failed", "n_skipped", "elapsed_s", "videos_per_s",
         "frames_per_s", "cpu_mean_pct"]]
    out_csv = NB_CACHE / "sweep_summary.csv"
    sweep.to_csv(out_csv, index=False)
    print(f"\n{sweep.to_string(index=False)}\n\nwrote {out_csv}")
    print("Next: run §5 (scoring) and §6 (overlay frames) in "
          "popsign.0.dataset.confidence-tuning.ipynb")


if __name__ == "__main__":
    main()
