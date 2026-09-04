"""Does `npz.ts` produce something numpy actually reads?

Two modes, because they answer different questions:

``--algorithm`` (default)
    Transcribes `npz.ts`'s `encodeNpy`/`encodeNpz`/`toFloat16` into Python and
    checks numpy loads the result. This validates the *format logic* — NPY header
    padding to a 64-byte boundary, the ZIP central-directory offsets, float16
    conversion with NaN preserved — without needing Deno installed. If the
    padding arithmetic or the zip structure is wrong in the TypeScript, it is
    wrong here too and this fails.

``--file <path.npz>``
    Checks a real artifact the Deno extractor wrote. This is the one that
    actually proves the TypeScript runs correctly; the algorithm mode cannot,
    because it re-implements rather than executes it.

Run the second one on the first machine that has Deno and ffmpeg::

    deno task extract --input ./one_video_dir --out ./out
    python tools/verify_npz_format.py --file ./out/<clip>.npz
"""

import argparse
import io
import struct
import sys
import zlib
from pathlib import Path

import numpy as np


def to_float16(v: float) -> int:
    """Transcription of npz.ts::toFloat16."""
    if np.isnan(v):
        return 0x7E00
    if v == np.inf:
        return 0x7C00
    if v == -np.inf:
        return 0xFC00
    if v == 0:
        return 0x8000 if np.signbit(v) else 0x0000
    bits = np.float32(v).view(np.uint32).item()
    sign = (bits >> 16) & 0x8000
    exp = ((bits >> 23) & 0xFF) - 127 + 15
    mant = bits & 0x7FFFFF
    if exp >= 0x1F:
        return sign | 0x7C00
    if exp <= 0:
        return sign if exp < -10 else sign | ((mant | 0x800000) >> (1 - exp + 13))
    return sign | (exp << 10) | (mant >> 13)


def encode_npy(data, shape, dtype: str) -> bytes:
    """Transcription of npz.ts::encodeNpy."""
    d = "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" % (
        dtype, " ".join(f"{n}," for n in shape))
    prefix = 10
    padded = -(-(prefix + len(d) + 1) // 64) * 64
    header = d + " " * (padded - prefix - len(d) - 1) + "\n"
    if dtype == "<f2":
        payload = np.array([to_float16(x) for x in np.asarray(data).ravel()],
                           dtype="<u2").tobytes()
    else:
        payload = np.asarray(data, dtype=dtype).tobytes()
    return (b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header))
            + header.encode() + payload)


def encode_npz(entries: dict[str, bytes]) -> bytes:
    """Transcription of npz.ts::encodeNpz (stored entries)."""
    locals_, central, offset = [], [], 0
    for name, payload in entries.items():
        nm = (name if name.endswith(".npy") else name + ".npy").encode()
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        lh = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, 0, 0x21, crc,
                         len(payload), len(payload), len(nm), 0) + nm
        cd = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 0, 0, 0x21,
                         crc, len(payload), len(payload), len(nm), 0, 0, 0, 0, 0,
                         offset) + nm
        locals_ += [lh, payload]
        central.append(cd)
        offset += len(lh) + len(payload)
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(central), len(central),
                       sum(len(c) for c in central), offset, 0)
    return b"".join(locals_ + central + [eocd])


def check(landmarks: np.ndarray, fps: float, n_frames: int) -> list[str]:
    """The contract, as assertions. Returns the failures."""
    from sb.core.schema import N_COORDS, N_LANDMARKS

    bad = []
    if landmarks.dtype != np.float16:
        bad.append(f"dtype is {landmarks.dtype}, expected float16")
    if landmarks.ndim != 3 or landmarks.shape[1:] != (N_LANDMARKS, N_COORDS):
        bad.append(f"shape is {landmarks.shape}, expected (T, {N_LANDMARKS}, {N_COORDS})")
    if landmarks.shape[0] != n_frames:
        bad.append(f"num_frames is {n_frames} but the array holds {landmarks.shape[0]}")
    if np.isinf(landmarks).any():
        bad.append("contains infinities — undetected landmarks must be NaN")
    if not np.isfinite(fps) or fps <= 0:
        bad.append(f"fps is {fps}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path,
                    help="an npz written by the Deno extractor (proves the TS runs)")
    ap.add_argument("--algorithm", action="store_true",
                    help="check the format logic without Deno (default when --file is absent)")
    args = ap.parse_args()

    if args.file:
        with np.load(args.file) as d:
            missing = [k for k in ("landmarks", "fps", "num_frames") if k not in d.files]
            if missing:
                raise SystemExit(f"{args.file}: missing key(s) {missing}")
            failures = check(d["landmarks"], float(d["fps"]), int(d["num_frames"]))
            print(f"{args.file.name}: {d['landmarks'].shape} {d['landmarks'].dtype}, "
                  f"fps {float(d['fps'])}, {int(d['num_frames'])} frames")
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1 if failures else 0)

    # algorithm mode: build the bytes the way npz.ts does, then make numpy read them
    T = 4
    arr = np.full((T, 543, 3), np.nan, dtype=np.float32)
    arr[:, 468:489] = 0.5
    arr[0, 0, 0] = -1.25
    arr[1, 100, 2] = 0.125
    blob = encode_npz({
        "landmarks": encode_npy(arr, (T, 543, 3), "<f2"),
        "fps": encode_npy([30.0], (), "<f4"),
        "num_frames": encode_npy([T], (), "<i4"),
    })
    with np.load(io.BytesIO(blob)) as d:
        lm = d["landmarks"]
        failures = check(lm, float(d["fps"]), int(d["num_frames"]))
        # NaN must survive as the missing-landmark marker, and exact binary
        # fractions must round-trip through float16 unchanged
        if not np.isnan(lm[0, 5, 0]):
            failures.append("NaN was not preserved")
        for idx, want in (((0, 0, 0), -1.25), ((1, 100, 2), 0.125), ((0, 470, 0), 0.5)):
            got = float(lm[idx])
            if got != want:
                failures.append(f"{idx}: got {got}, expected {want}")
        print(f"algorithm: {lm.shape} {lm.dtype} · fps {float(d['fps'])} · "
              f"num_frames {int(d['num_frames'])} · NaN preserved "
              f"{not failures or 'NaN was not preserved' not in failures}")
    for f in failures:
        print(f"  FAIL {f}")
    print("\nnpz.ts format logic:", "OK" if not failures else "FAILED")
    print("Note: this validates the ALGORITHM. Only --file on a Deno-written npz "
          "proves the TypeScript itself runs correctly.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
