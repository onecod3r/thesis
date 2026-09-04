"""Do the Python and TypeScript extractors agree? — measure before switching.

**Why this has to exist.** 33,599 POPSIGN test clips are already extracted with
the Python MediaPipe path. If the train split is extracted with the Deno/WASM
path instead, any systematic difference between the two becomes a difference
**between splits** — a model would learn it, and every POPSIGN number would be
quietly wrong in a way no accuracy metric reveals. The two builds are not the
same code: different MediaPipe distribution, different delegate, possibly a
different model revision.

So the switch is gated on a measurement, not on the new extractor merely running.

What "agree" means here, in order of how much it matters:

1. **Structure** — identical frame count and identical detected/undetected mask.
   A landmark present in one and NaN in the other is a categorical difference,
   not a numerical one, and it changes the quality proxies and the NaN-aware
   feature pipeline.
2. **Geometry** — per-coordinate absolute difference. Normalised landmark
   coordinates live in roughly [0, 1], so a median |diff| above ~1e-3 is a real
   disagreement rather than float16 rounding (float16 has ~3 decimal digits, so
   ~5e-4 is the storage floor).
3. **Downstream** — the same clip's ME-126 subset, which is what training sees.

Usage::

    python -m sb.extract.parity --python-dir data/raw/popsign/test \\
                                --ts-dir data/temp/parity_ts --limit 50
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sb.core.io import read_landmark_npz
from sb.core.schema import GROUPS

# float16 stores ~3 decimal digits, so differences below this are the storage
# format talking, not the extractors disagreeing
FLOAT16_FLOOR = 5e-4


def compare_one(py_path: Path, ts_path: Path) -> dict:
    """Structural + geometric comparison of one clip extracted both ways."""
    py, py_meta = read_landmark_npz(py_path, validate=False)
    ts, ts_meta = read_landmark_npz(ts_path, validate=False)

    row = {
        "clip": py_path.name,
        "py_frames": py_meta["num_frames"],
        "ts_frames": ts_meta["num_frames"],
        "frames_match": py_meta["num_frames"] == ts_meta["num_frames"],
    }
    if not row["frames_match"]:
        # a frame-count difference makes every later comparison meaningless;
        # report it and stop rather than aligning by truncation and pretending
        row["verdict"] = "frame count differs"
        return row

    py_seen = ~np.isnan(py[:, :, 0])
    ts_seen = ~np.isnan(ts[:, :, 0])
    row["mask_agreement"] = float((py_seen == ts_seen).mean())

    both = py_seen & ts_seen
    if both.any():
        diff = np.abs(py[both] - ts[both])
        row["median_abs_diff"] = float(np.median(diff))
        row["p99_abs_diff"] = float(np.percentile(diff, 99))
        row["max_abs_diff"] = float(diff.max())
    else:
        row["median_abs_diff"] = row["p99_abs_diff"] = row["max_abs_diff"] = float("nan")

    for g in GROUPS:
        sl = slice(g.offset, g.offset + g.size)
        row[f"py_rate_{g.name}"] = float(py_seen[:, sl].any(axis=1).mean())
        row[f"ts_rate_{g.name}"] = float(ts_seen[:, sl].any(axis=1).mean())

    row["verdict"] = (
        "agree" if row["mask_agreement"] > 0.99
        and row["median_abs_diff"] < FLOAT16_FLOOR * 4 else "differ"
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--python-dir", required=True, type=Path,
                    help="landmarks tree written by the Python extractor")
    ap.add_argument("--ts-dir", required=True, type=Path,
                    help="landmarks tree written by sb-extract-ts on the SAME clips")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", type=Path, help="write the per-clip table as JSON")
    args = ap.parse_args()

    ts_by_name = {p.name: p for p in args.ts_dir.rglob("*.npz")
                  if not p.name.endswith(".tmp.npz")}
    pairs = [(p, ts_by_name[p.name]) for p in sorted(args.python_dir.rglob("*.npz"))
             if p.name in ts_by_name][: args.limit]
    if not pairs:
        raise SystemExit(
            "no clip appears in both trees — extract the same videos with both "
            "paths first (the TS side writes <stem>.npz under --out)")

    rows = [compare_one(py, ts) for py, ts in pairs]
    agree = sum(r["verdict"] == "agree" for r in rows)
    print(f"{len(rows)} clip(s) compared · {agree} agree, {len(rows) - agree} differ\n")

    frames_ok = sum(r["frames_match"] for r in rows)
    print(f"frame count identical : {frames_ok}/{len(rows)}")
    comparable = [r for r in rows if r["frames_match"]]
    if comparable:
        print(f"mask agreement (mean) : {np.mean([r['mask_agreement'] for r in comparable]):.4f}")
        print(f"median |diff|         : {np.nanmedian([r['median_abs_diff'] for r in comparable]):.2e}"
              f"   (float16 floor {FLOAT16_FLOOR:.0e})")
        print(f"p99 |diff|            : {np.nanmedian([r['p99_abs_diff'] for r in comparable]):.2e}")
        print("\nper-group detection rate, python vs typescript:")
        for g in GROUPS:
            py = np.mean([r[f"py_rate_{g.name}"] for r in comparable])
            ts = np.mean([r[f"ts_rate_{g.name}"] for r in comparable])
            print(f"  {g.name:<11} {py:.3f}  vs  {ts:.3f}   ({ts - py:+.3f})")

    print(
        "\nRead it this way: mask agreement below ~0.99 or a median |diff| well "
        "above the float16 floor means the two extractors are NOT interchangeable, "
        "and mixing them across train/test would put a systematic difference "
        "between the splits."
    )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
