"""Runs the GISLR-GapCorpus build locally (TODO §17), instead of on Kaggle --
same logic as `gislr.0.dataset.gapcorpus-kaggle.ipynb`
(`sb.recognize.sequences.gapcorpus_kaggle_notebook`), but using this repo's real
`sb.*` imports and writing to `data/cache/gislr/gapcorpus/<version>/` (the local
fallback path `sb.core.paths.gislr_gapcorpus_dir()` already looks for), instead of
the notebook's self-contained embedding + `/kaggle/working`.

For a one-off local build the user will upload to Kaggle by hand afterward. The
Kaggle notebook stays the source of truth for anyone re-running this on Kaggle
itself; this script is not committed as a second copy of that logic to maintain --
it is a thin driver over the same `sb.recognize.sequences.compose` functions.

    .venv/Scripts/python.exe experiments/recognition/run_gapcorpus_local.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sb.core.paths import CACHE_DIR, ROOT_DIR, gislr_dir
from sb.recognize.sequences import compose

CONFIG_PATH = ROOT_DIR / "experiments" / "recognition" / "configs" / "gislr.gapcorpus.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT_DIR = CACHE_DIR / "gislr" / "gapcorpus" / CONFIG["dataset_version"]
STORAGE_DTYPE = "float16"
OUTPUT_LIMIT_GB = 30.0  # local disk, not Kaggle's /kaggle/working -- generous, still a guard
PROJECT_AFTER = 300
N_WORKERS = os.cpu_count() or 4

SEQ_DIR = OUT_DIR / "sequences"
ASSETS_DIR = OUT_DIR / "assets"
PLAN_PATH = OUT_DIR / "plan.json"
MANIFEST_PATH = OUT_DIR / "manifest.json"
for d in (SEQ_DIR, ASSETS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    t0 = time.time()
    data_dir = gislr_dir()
    config_sha = __import__("hashlib").sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
    print(f"input:   {data_dir}")
    print(f"output:  {OUT_DIR}  ({STORAGE_DTYPE}, {N_WORKERS} workers)")
    print(f"config:  sha256={config_sha[:12]}")

    # --- 1. label map + combined pool + signer split -----------------------
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    signs = set(train_df["sign"]) | set(test_df["sign"])
    label_map = {sign: i for i, sign in enumerate(sorted(signs))}
    (OUT_DIR / "sign_to_prediction_index_map.json").write_text(json.dumps(label_map), encoding="utf-8")

    pool = pd.concat([train_df.assign(source_split="train"), test_df.assign(source_split="test")],
                     ignore_index=True)
    srng = np.random.default_rng(CONFIG["seed"])
    participants = np.sort(pool["participant_id"].unique())
    srng.shuffle(participants)
    n_train_signers = int(round(len(participants) * CONFIG["train_signer_frac"]))
    signer_split = {int(p): ("train" if i < n_train_signers else "test") for i, p in enumerate(participants)}
    pool["split"] = pool["participant_id"].map(signer_split)
    relpath_of = dict(zip(pool["uid"], pool["npz_relpath"]))
    print(f"pool: {len(pool)} clips ({len(train_df)} train.csv + {len(test_df)} test.csv), "
          f"{len(signs)} glosses, {len(participants)} signers "
          f"({n_train_signers} train / {len(participants) - n_train_signers} test)")
    print(pool["split"].value_counts())

    # --- 2. plan -------------------------------------------------------------
    if PLAN_PATH.exists() and json.loads(PLAN_PATH.read_text())["config_sha256"] == config_sha:
        plan = json.loads(PLAN_PATH.read_text())
        print("plan loaded")
    else:
        lengths = np.arange(CONFIG["signs_per_seq"][0], CONFIG["signs_per_seq"][1] + 1)
        rows = compose.plan_control_multi(
            pool, lengths, CONFIG["seed"],
            gap_range=tuple(CONFIG["gap_frames"]), rest_range=tuple(CONFIG["rest_frames"]),
            n_passes=CONFIG["n_passes"])
        for i, r in enumerate(rows):
            r["seq_id"] = f"g{i:06d}"
            r["split"] = signer_split[r["participant_id"]]
        plan = {"config_sha256": config_sha, "sequences": rows}
        write_json(PLAN_PATH, plan)
    by_split = pd.Series([r["split"] for r in plan["sequences"]]).value_counts()
    print(f"{len(plan['sequences'])} sequences over {CONFIG['n_passes']} pass(es)")
    print(by_split)

    # --- 3. materialize --------------------------------------------------------
    rows = plan["sequences"]
    dtype = np.dtype(STORAGE_DTYPE)
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    def build(i_row):
        i, row = i_row
        rel = f"sequences/{row['seq_id']}.npz"
        try:
            arrays = compose.materialize_v2(
                row, data_dir, relpath_of, label_map, np.random.default_rng([CONFIG["seed"], i]),
                rest_jitter=CONFIG["rest_jitter"], rest_ramp_frames=CONFIG["rest_ramp_frames"],
                hand_visible_max=CONFIG["hand_visible_max"], rest_default_drop=CONFIG["rest_default_drop"],
                p_lowered_rest=CONFIG["p_lowered_rest"])
            compose.write_sequence(OUT_DIR / rel, arrays, dtype=dtype)
            rec = compose.summarize_row_v2(row, arrays, row["seq_id"], rel)
            return row["seq_id"], {"status": "done", "bytes": (OUT_DIR / rel).stat().st_size,
                                    "split": row["split"], **rec}
        except Exception as e:
            return row["seq_id"], {"status": "failed", "error": repr(e)}

    todo = [(i, r) for i, r in enumerate(rows) if manifest.get(r["seq_id"], {}).get("status") != "done"]
    bar = tqdm(total=len(rows), initial=len(rows) - len(todo), desc="materialize")
    n_failed = 0
    with ThreadPoolExecutor(N_WORKERS) as ex:
        for k, (seq_id, rec) in enumerate(ex.map(build, todo), 1):
            manifest[seq_id] = rec
            n_failed += rec["status"] == "failed"
            bar.update(1)
            done_bytes = [v["bytes"] for v in manifest.values() if v["status"] == "done"]
            projected = np.mean(done_bytes) * len(rows) / 1e9 if done_bytes else 0.0
            bar.set_postfix(failed=n_failed, projected_gb=f"{projected:.1f}")
            if k == PROJECT_AFTER and projected > OUTPUT_LIMIT_GB:
                write_json(MANIFEST_PATH, manifest)
                raise RuntimeError(f"projected {projected:.1f} GB > OUTPUT_LIMIT_GB={OUTPUT_LIMIT_GB}")
            if k % 2000 == 0:
                write_json(MANIFEST_PATH, manifest)
    write_json(MANIFEST_PATH, manifest)
    bar.close()
    print(f"{sum(v['status'] == 'done' for v in manifest.values())}/{len(rows)} done, {n_failed} failed this run")

    # --- 4. train.csv/test.csv + build_info -------------------------------------
    manifest = json.loads(MANIFEST_PATH.read_text())
    failed = [k for k, v in manifest.items() if v["status"] != "done"]
    assert not failed, f"{len(failed)} sequences not done: {failed[:5]}"

    order = [r["seq_id"] for r in plan["sequences"]]
    full_df = pd.DataFrame([{k: v for k, v in manifest[s].items() if k not in ("status", "bytes")} for s in order])
    for split in ("train", "test"):
        full_df[full_df["split"] == split].drop(columns=["split"]).to_csv(OUT_DIR / f"{split}.csv", index=False)

    frames = full_df.groupby("split")["n_frames"].agg(["count", "sum", "mean", "max"])
    gap_rate = {}
    for seq_id in full_df["seq_id"].sample(min(500, len(full_df)), random_state=0):
        arr = compose.read_sequence(OUT_DIR / full_df.set_index("seq_id").loc[seq_id, "npz_relpath"])
        gap_rate[seq_id] = float(arr["gap"].mean())
    size_bytes = sum(p.stat().st_size for p in SEQ_DIR.rglob("*.npz"))
    build_info = {
        "dataset": "GISLR-GapCorpus", "dataset_version": CONFIG["dataset_version"],
        "built_on": "local", "storage_dtype": STORAGE_DTYPE,
        "config": CONFIG, "config_sha256": config_sha,
        "source": {"kaggle_ref": "bracu23101281/gislr-stratified", "resolved_dir": str(data_dir)},
        "frames": frames.reset_index().to_dict(orient="records"),
        "mean_gap_fraction_sampled": float(np.mean(list(gap_rate.values()))),
        "size_bytes": size_bytes,
    }
    write_json(OUT_DIR / "build_info.json", build_info)
    print(frames)
    print(f"mean gap fraction (500-sequence sample): {build_info['mean_gap_fraction_sampled']:.3f}")
    print(f"size: {size_bytes / 1e9:.2f} GB ({STORAGE_DTYPE})")

    # --- 5. checks ---------------------------------------------------------------
    by_id = {r["seq_id"]: r for r in plan["sequences"]}
    rng = np.random.default_rng(CONFIG["seed"])
    sample = sorted(rng.choice(full_df["seq_id"].to_numpy(),
                               min(CONFIG["n_roundtrip_checks"], len(full_df)), replace=False).tolist())
    bad = []
    for seq_id in tqdm(sample, desc="round-trip"):
        arrays = compose.read_sequence(OUT_DIR / full_df.set_index("seq_id").loc[seq_id, "npz_relpath"])
        try:
            compose.check_roundtrip(arrays, by_id[seq_id], data_dir, relpath_of)
            assert np.array_equal(arrays["gap"], (arrays["frame_kind"] != compose.SIGN).astype(np.uint8))
        except AssertionError as e:
            bad.append((seq_id, str(e)))
    write_json(OUT_DIR / "roundtrip_sample.json", {"seed": CONFIG["seed"], "seq_ids": sample, "failures": bad})
    print(f"round-trip: {len(sample) - len(bad)}/{len(sample)} exact at {STORAGE_DTYPE}")
    assert not bad, "checks failed"

    MANIFEST_PATH.unlink(missing_ok=True)
    for p in OUT_DIR.rglob("*.tmp*"):
        p.unlink()
    n_files = sum(1 for p in OUT_DIR.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in OUT_DIR.rglob("*") if p.is_file())
    print(f"DONE in {(time.time() - t0) / 60:.1f} min: {OUT_DIR}: {n_files} files, {total / 1e9:.2f} GB")
    print("Ready to zip and upload to Kaggle as a new private dataset (e.g. GISLR-GapCorpus) whenever you like.")


if __name__ == "__main__":
    main()
