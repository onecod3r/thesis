"""Run both extractors over the same handful of clips — the gate in TODO §10.1.

``parity.py`` compares two trees of npz. *Producing* those two trees is the
fiddly part, and getting it wrong yields a confident, meaningless number, so it
lives here rather than in a notebook cell or a shell one-liner.

Three things this has to get right, none of them obvious:

**The clips come from the train split, not test.** The 33,599 test clips already
extracted by the Python path no longer have videos — ``popsign_cycle`` deletes
the video once the landmarks verify, which is the entire point of it. Parity is
a comparison of two extractors on *identical inputs*, so which clips they are
does not matter; that they still have video does. The train a-e part is on disk.

**Both sides run the same weights.** The Python extractor loads
``data/external/mediapipe/tasks/holistic_landmarker.task``; the TypeScript one
defaults to the bucket's ``latest``, a different and unpinned artifact. Left
alone this would measure the *model* rather than the code, so ``--model`` points
the TS side at the same local file.

**Both sides decode the same frames.** cv2 hands MediaPipe the video's native
frames at its native rate. POPSIGN is 1944x2592 portrait at 30, ~29.92 or 120
fps depending on the clip, so the TS extractor runs at native geometry too — the
``--width/--height/--fps`` overrides are deliberately not passed.

Usage::

    python -m sb.extract.parity_run --limit 12
    python -m sb.extract.parity_run --limit 12 --fresh    # re-extract from scratch
    python -m sb.extract.parity_run --clean               # drop data/temp/parity

Resumable throughout: the selection is cached, and both extractors skip units
their own manifest already marks ``done``, so an interrupt costs the clips in
flight. Landmarks land in ``data/temp/parity/`` (throwaway, ``--clean`` removes
it); the selection and the report are cache artifacts that outlive it.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from sb.core.paths import CACHE_DIR, EXTERNAL_DIR, ROOT_DIR, TEMP_DIR

SEED = 42
DEFAULT_LIMIT = 12

WORK_DIR = TEMP_DIR / "parity"
VIDEO_DIR = WORK_DIR / "videos"
TS_DIR = WORK_DIR / "ts"
PY_SPLIT = "py"                 # extract_dataset writes to <out_root>/<split>
PY_DIR = WORK_DIR / PY_SPLIT

PARITY_CACHE = CACHE_DIR / "popsign" / "parity"
SELECTION = PARITY_CACHE / "selection.json"
REPORT = PARITY_CACHE / "report.json"

TS_PACKAGE = ROOT_DIR / "packages" / "sb-extract-ts"
VERIFY_TOOL = TS_PACKAGE / "tools" / "verify_npz_format.py"
MODEL = EXTERNAL_DIR / "mediapipe" / "tasks" / "holistic_landmarker.task"
TRAIN_CSV = CACHE_DIR / "popsign" / "dataframes" / "train.csv"


# ============================================================
# 1. Selection — seeded and cached, so a re-run compares the same clips
# ============================================================

def select_clips(limit: int, seed: int = SEED) -> pd.DataFrame:
    """`limit` clips that still have video on disk, stable across runs.

    Sampled one-per-label first so the sample spans signs rather than landing in
    a single folder: detection rate varies far more between signs than within
    one, and a parity number drawn from one label would not generalise.
    """
    if SELECTION.exists():
        cached = pd.DataFrame(json.loads(SELECTION.read_text(encoding="utf-8")))
        if len(cached) == limit:
            missing = [p for p in cached["file_path"] if not Path(p).exists()]
            if not missing:
                print(f"selection: reusing {SELECTION} ({limit} clips)")
                return cached
            print(f"selection: {len(missing)} cached video(s) gone, re-sampling")

    if not TRAIN_CSV.exists():
        raise SystemExit(
            f"{TRAIN_CSV} not found — it is the video manifest this samples from"
        )
    videos = pd.read_csv(TRAIN_CSV)
    videos = videos[videos["file_path"].map(lambda p: Path(p).exists())]
    if len(videos) < limit:
        raise SystemExit(f"only {len(videos)} videos on disk, need {limit}")

    one_each = videos.groupby("label", group_keys=False).sample(1, random_state=seed)
    chosen = one_each.sample(min(limit, len(one_each)), random_state=seed)
    if len(chosen) < limit:
        rest = videos.drop(chosen.index).sample(limit - len(chosen), random_state=seed)
        chosen = pd.concat([chosen, rest])
    chosen = chosen.sort_values("id").reset_index(drop=True)

    PARITY_CACHE.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(chosen.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"selection: sampled {len(chosen)} clips -> {SELECTION}")
    return chosen


# ============================================================
# 2. Staging — a small tree the TS extractor can walk
# ============================================================

def stage_videos(clips: pd.DataFrame) -> None:
    """Mirror the chosen clips into `<label>/<id>.mp4` under VIDEO_DIR.

    Hardlinked where the filesystem allows it (the kagglehub cache and the repo
    are both on C:), so staging costs no disk and no copy time; a copy is the
    fallback, not the default. Both extractors then read byte-identical inputs
    from the same path, which removes "did they even open the same file" from
    the list of things a disagreement could mean.
    """
    linked = copied = 0
    for row in clips.itertuples(index=False):
        dest = VIDEO_DIR / str(row.label) / f"{row.id}.mp4"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(row.file_path, dest)
            linked += 1
        except OSError:
            shutil.copy2(row.file_path, dest)
            copied += 1
    print(f"staged: {linked} hardlinked, {copied} copied -> {VIDEO_DIR}")


# ============================================================
# 3. The TypeScript extractor
# ============================================================

def run_typescript(workers: int) -> tuple[int, int, str | None]:
    """`deno run src/cli.ts` at native geometry, against the local .task model.

    Returns `(done, failed, one_error)` rather than raising. A TS-side failure
    is a *result* of this comparison, not a crash in it — the Python half still
    has to run, so its reference tree exists for whenever the TS half works.
    """
    if shutil.which("deno") is None:
        return 0, 0, "deno not on PATH"
    cmd = [
        "deno", "run",
        "--allow-read", "--allow-write", "--allow-run", "--allow-net", "--allow-env",
        "src/cli.ts",
        "--input", str(VIDEO_DIR),
        "--out", str(TS_DIR),
        "--model", str(MODEL),
        "--workers", str(workers),
        "--retry-failed",
    ]
    print("\n=== typescript ===")
    print(" ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, cwd=TS_PACKAGE)
    print(f"typescript: {time.time() - t0:.1f}s")

    manifest = TS_DIR / "_manifest.json"
    if not manifest.exists():
        return 0, 0, "the TS extractor wrote no manifest"
    units = json.loads(manifest.read_text(encoding="utf-8")).get("units", {})
    done = sum(1 for u in units.values() if u.get("status") == "done")
    failed = sum(1 for u in units.values() if u.get("status") == "failed")
    error = next((u.get("error") for u in units.values()
                  if u.get("status") == "failed"), None)
    return done, failed, error


# ============================================================
# 4. The Python extractor, on exactly the same clips
# ============================================================

def run_python(clips: pd.DataFrame, workers: int) -> dict:
    """Same videos, same model, into `<WORK_DIR>/py/<label>/<id>.npz`."""
    import sb.extract.holistic as ex

    videos = pd.DataFrame({
        "file_path": [str(VIDEO_DIR / r.label / f"{r.id}.mp4")
                      for r in clips.itertuples(index=False)],
        "label": clips["label"].values,
        "id": clips["id"].values,
    })
    print("\n=== python ===")
    t0 = time.time()
    summary = ex.extract_dataset(
        videos, split=PY_SPLIT, out_root=WORK_DIR,
        n_workers=workers, model_path=MODEL, progress=True,
    )
    print(f"python: {time.time() - t0:.1f}s")
    return summary


# ============================================================
# 5. Format check, then the comparison itself
# ============================================================

def verify_format() -> None:
    """numpy must actually read what `npz.ts` wrote.

    This is the execution-level check `verify_npz_format.py --algorithm` could
    not make on its own: that transcribes the encoder into Python and proves the
    *logic* is right, which says nothing about whether the TypeScript runs.
    """
    produced = sorted(
        p for p in TS_DIR.rglob("*.npz") if not p.name.endswith(".tmp.npz")
    )
    if not produced:
        raise SystemExit("the TS extractor produced no npz — nothing to verify")
    print(f"\n=== npz format ({produced[0].name}) ===")
    proc = subprocess.run(
        [sys.executable, str(VERIFY_TOOL), "--file", str(produced[0])],
        cwd=TS_PACKAGE,
    )
    if proc.returncode != 0:
        raise SystemExit("npz format check failed — parity below would be noise")


def run_parity(limit: int) -> int:
    print("\n=== parity ===")
    PARITY_CACHE.mkdir(parents=True, exist_ok=True)
    return subprocess.run([
        sys.executable, "-m", "sb.extract.parity",
        "--python-dir", str(PY_DIR),
        "--ts-dir", str(TS_DIR),
        "--limit", str(limit),
        "--out", str(REPORT),
    ]).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"clips to compare (default {DEFAULT_LIMIT})")
    ap.add_argument("--workers", type=int, default=4,
                    help="workers per extractor; native-resolution frames are "
                         "large, so this is memory-bound rather than core-bound")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fresh", action="store_true",
                    help="discard previous landmark output and re-extract")
    ap.add_argument("--clean", action="store_true",
                    help="remove data/temp/parity/ and exit")
    args = ap.parse_args()

    if args.clean:
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        print(f"removed {WORK_DIR}")
        return
    if not MODEL.exists():
        raise SystemExit(f"holistic model not found: {MODEL}")
    if args.fresh:
        for d in (TS_DIR, PY_DIR):
            shutil.rmtree(d, ignore_errors=True)
        print("fresh: cleared previous landmark output")

    clips = select_clips(args.limit, args.seed)
    print(f"labels: {', '.join(sorted(clips['label'].unique()))}")
    stage_videos(clips)

    ts_done, ts_failed, ts_error = run_typescript(args.workers)
    print(f"typescript: {ts_done} extracted, {ts_failed} failed")
    # the Python half runs either way: its tree is the reference the TS half is
    # compared against, and it is worth having before the TS half can run
    run_python(clips, args.workers)

    if ts_done == 0:
        print("\n=== parity: BLOCKED ===")
        print(f"the TypeScript extractor produced no landmarks "
              f"({ts_failed} failed), so there is nothing to compare.")
        if ts_error:
            print(f"\n  {ts_error}\n")
        print(f"the Python reference tree in {PY_DIR} is reusable — re-run this "
              "command once the TS side can execute.")
        raise SystemExit(1)

    verify_format()
    code = run_parity(args.limit)
    print(f"\nreport: {REPORT}")
    print(f"landmarks kept in {WORK_DIR} for inspection — "
          "`python -m sb.extract.parity_run --clean` when done")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
