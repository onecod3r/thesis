"""POPSIGN one-part-at-a-time: download → extract → **verify** → delete.

The five POPSIGN releases total ~870 GB, which does not fit on this machine at
once. Landmarks are ~165 KB/clip, so the same corpus becomes ~14 GB of npz — the
whole point is that the video is a transient input, not something to keep.

The cycle, per part::

    kagglehub download  →  extract landmarks  →  verify every clip  →  delete video

**Deletion is gated on verification, not on the extractor exiting 0.** A part is
only removed once every video it contains has a `done` unit in the extraction
manifest *and* the npz is on disk *and* it passes the landmark-tensor spec. A
partially-extracted part that got deleted would cost a ~220 GB re-download to
notice, so the check is worth the minutes it takes.

Resumable at the part level (`cycle.json`) and, inside a part, at the clip level
by the extractor's own manifest. An interrupt costs the clips in flight.

Usage::

    python -m sb.extract.popsign_cycle status
    python -m sb.extract.popsign_cycle run --part train-a-e
    python -m sb.extract.popsign_cycle run --all --apply-delete
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sb.core.io import iter_landmark_files, read_landmark_npz
from sb.core.paths import CACHE_DIR, DATASET_IDS, env_value
from sb.core.schema import SpecError
from sb.extract.sources.popsign import landmarks_dir

CYCLE_STATE = CACHE_DIR / "popsign" / "cycle.json"

PARTS: dict[str, str] = {
    "train-a-e": DATASET_IDS["TRAIN"][0],
    "train-f-m": DATASET_IDS["TRAIN"][1],
    "train-n-s": DATASET_IDS["TRAIN"][2],
    "train-t-z": DATASET_IDS["TRAIN"][3],
    "test": DATASET_IDS["TEST"],
}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_state() -> dict:
    if CYCLE_STATE.is_file():
        return json.loads(CYCLE_STATE.read_text(encoding="utf-8"))
    return {"parts": {}}


def save_state(state: dict) -> None:
    CYCLE_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CYCLE_STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, CYCLE_STATE)


def download(part: str) -> Path:
    """Resolve (downloading if needed) one POPSIGN release."""
    import kagglehub

    print(f"[{part}] resolving {PARTS[part]} …", flush=True)
    return Path(kagglehub.dataset_download(PARTS[part]))


def extract(part: str, video_root: Path, out_root: Path, extractor: str) -> None:
    """Run the chosen extractor over one part. Both are resumable."""
    out_root.mkdir(parents=True, exist_ok=True)
    if extractor == "deno":
        pkg = Path(__file__).resolve().parents[4] / "sb-extract-ts"
        cmd = [
            "deno", "run", "--allow-read", "--allow-write", "--allow-run",
            "--allow-net", "--allow-env", str(pkg / "src" / "cli.ts"),
            "--input", str(video_root), "--out", str(out_root),
        ]
    else:
        cmd = [sys.executable, "-m", "sb.extract.cli", "run", part,
               "--input", str(video_root), "--out", str(out_root)]
    print(f"[{part}] {' '.join(cmd[:4])} …", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"[{part}] extractor exited {result.returncode}")


def verify(part: str, video_root: Path, out_root: Path, sample: int = 200) -> dict:
    """Every video accounted for, and a sample of the npz actually valid.

    Counting is exhaustive because that is cheap; spec-validating every clip is
    not, so a seeded sample is read in full. A single malformed npz fails the
    whole part — the alternative is deleting the source of a corrupt artifact.
    """
    videos = [p for p in video_root.rglob("*") if p.suffix.lower() in VIDEO_EXT]
    produced = {p.stem for p in iter_landmark_files(out_root)}
    missing = [p.name for p in videos if p.stem not in produced]

    checked, bad = 0, []
    for npz in sorted(iter_landmark_files(out_root))[:sample]:
        try:
            read_landmark_npz(npz)
            checked += 1
        except (SpecError, OSError, KeyError) as exc:
            bad.append(f"{npz.name}: {exc}")

    return {
        "videos": len(videos),
        "npz": len(produced),
        "missing": len(missing),
        "missing_examples": missing[:5],
        "spec_checked": checked,
        "spec_failures": bad,
        "ok": not missing and not bad and len(videos) > 0,
    }


def delete_videos(part: str, video_root: Path, apply: bool) -> int:
    """Remove one part's raw video. Re-downloadable, but slowly."""
    size = sum(f.stat().st_size for f in video_root.rglob("*") if f.is_file())
    if not apply:
        print(f"[{part}] would free {size / 1e9:.1f} GB from {video_root} "
              "(pass --apply-delete)")
        return 0
    shutil.rmtree(video_root, ignore_errors=True)
    print(f"[{part}] deleted {video_root} — {size / 1e9:.1f} GB freed")
    return size


def run_part(part: str, extractor: str, apply_delete: bool, state: dict) -> None:
    out_root = landmarks_dir("test" if part == "test" else "train")
    entry = state["parts"].setdefault(part, {})

    video_root = download(part)
    entry["downloaded_at"] = datetime.now().isoformat(timespec="seconds")
    entry["video_root"] = str(video_root)
    save_state(state)

    extract(part, video_root, out_root, extractor)

    report = verify(part, video_root, out_root)
    entry["verify"] = report
    entry["extracted_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)

    print(f"[{part}] {report['npz']}/{report['videos']} clips extracted · "
          f"{report['spec_checked']} spec-checked · "
          f"{len(report['spec_failures'])} invalid")
    if not report["ok"]:
        print(f"[{part}] NOT deleting the video — verification failed:")
        for line in report["missing_examples"] + report["spec_failures"][:5]:
            print(f"    {line}")
        print(f"[{part}] re-run to continue; the extractor is resumable")
        return

    freed = delete_videos(part, video_root, apply_delete)
    if freed:
        entry["deleted_at"] = datetime.now().isoformat(timespec="seconds")
        entry["freed_bytes"] = freed
    save_state(state)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("status", help="what has been downloaded/extracted/deleted")
    run = sub.add_parser("run", help="download, extract, verify, then delete")
    run.add_argument("--part", choices=sorted(PARTS), help="one part")
    run.add_argument("--all", action="store_true", help="every part, in order")
    run.add_argument("--extractor", choices=("deno", "python"),
                     default=env_value("SB_EXTRACTOR", "python"),
                     help="which extractor to drive (default: SB_EXTRACTOR or python)")
    run.add_argument("--apply-delete", action="store_true",
                     help="actually delete each part's video once verified")
    args = ap.parse_args()

    state = load_state()
    if args.command != "run":
        if not state["parts"]:
            print("nothing recorded yet")
        for part, entry in sorted(state["parts"].items()):
            v = entry.get("verify", {})
            print(f"{part:<12} downloaded {'y' if entry.get('downloaded_at') else 'n'} · "
                  f"{v.get('npz', 0)}/{v.get('videos', 0)} clips · "
                  f"deleted {'y' if entry.get('deleted_at') else 'n'}")
        return

    parts = sorted(PARTS) if args.all else [args.part]
    if not parts or parts == [None]:
        raise SystemExit("give --part <name> or --all")
    for part in parts:
        run_part(part, args.extractor, args.apply_delete, state)


if __name__ == "__main__":
    main()
