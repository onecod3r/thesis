"""Generate the self-contained Kaggle build notebook for GISLR-GapCorpus (TODO §17).

    python -m sb.recognize.sequences.gapcorpus_kaggle_notebook

writes ``experiments/recognition/gislr.0.dataset.gapcorpus-kaggle.ipynb``: a
**sign-vs-gap** training corpus, the twin of ``sb.recognize.sequences.kaggle_notebook``'s
GISLR-Sentences build but built for a different reader. GISLR-Sentences threads real
ASL sentences through a coverage-driven planner (every clip placed exactly once) --
right for evaluating a *language* model, wrong for a boundary/gap detector, whose only
job is to tell a moving hand from a still or absent one. This build instead:

- **drops sentence structure entirely** (:func:`sb.recognize.sequences.compose.plan_control_multi`):
  random-order, random-length sequences, the *same* clip pool drawn several times
  (``n_passes``) with different neighbours each time -- more and more varied
  transition examples, not a coverage requirement;
- **uses both `train.csv` and `test.csv`** as the source pool (a gap detector's task
  has nothing to do with the 250-gloss classification split those files encode);
- **fixes GISLR-Sentences v1's rest flaw** (TODO §12.1's last item): its rest always
  drops both hands to NaN while the pose stays at signing height, which makes "hands
  NaN" alone a perfect gap tell. This build mixes in the realistic **lowered** rest
  already proven in ``sb.recognize.continuous.data`` (arms ramp down to hip height;
  each hand goes NaN only once its wrist is actually out of frame) alongside the
  original hands-absent style, ported to canonical ``(T, 543, 3)`` row space
  (:func:`sb.recognize.sequences.compose.lowered_rest`/:func:`materialize_v2`);
- **defines its own signer-disjoint train/test split** (a fraction of participants
  held out entirely), independent of GISLR_Stratified's own clip-level split, since
  this is a different model with a different generalization question (new signers,
  not new glosses).

Nothing is re-implemented: the notebook embeds the **exact source** of every module
the build touches (``sb.core.schema``, ``sb.recognize.features.gislr_stratified``,
``sb.recognize.sequences.compose`` -- all numpy/stdlib-only) under their real module
names, so ``compose``'s own imports resolve unchanged. Re-run this generator after
touching any of them; the notebook records the git commit and a hash of the embedded
code in ``build_info.json``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from sb.core.paths import EXPERIMENTS_DIR, PACKAGES_DIR, ROOT_DIR

OUT = EXPERIMENTS_DIR / "recognition" / "gislr.0.dataset.gapcorpus-kaggle.ipynb"
CONFIG_PATH = EXPERIMENTS_DIR / "recognition" / "configs" / "gislr.gapcorpus.json"

MODULES = {
    "sb.core.schema": PACKAGES_DIR / "sb-core/src/sb/core/schema.py",
    "sb.recognize.features.gislr_stratified":
        PACKAGES_DIR / "sb-recognize/src/sb/recognize/features/gislr_stratified.py",
    "sb.recognize.sequences.corpus": PACKAGES_DIR / "sb-recognize/src/sb/recognize/sequences/corpus.py",
    "sb.recognize.sequences.compose": PACKAGES_DIR / "sb-recognize/src/sb/recognize/sequences/compose.py",
}


def _raw(text: str) -> str:
    text = text.replace("\r\n", "\n")
    if "'''" in text or text.rstrip("\n").endswith("\\"):
        raise ValueError("cannot embed as r'''...''' literal")
    return "r'''" + text + "'''"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(config_path: Path = CONFIG_PATH) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = {name: path.read_text(encoding="utf-8").replace("\r\n", "\n") for name, path in MODULES.items()}
    code_sha = hashlib.sha256("".join(sources[n] for n in MODULES).encode()).hexdigest()

    cells: list[dict] = []

    def md(s: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})

    def code(s: str) -> None:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": s.strip("\n").splitlines(keepends=True)})

    md(f"""
# GISLR-GapCorpus {config['dataset_version']} — Kaggle build (self-contained)

**Generated file — do not edit.** Regenerate with
`python -m sb.recognize.sequences.gapcorpus_kaggle_notebook` (signbridge commit
`{_git_commit()}`, embedded-code sha256 `{code_sha[:12]}`). Built for TODO §17: training a
dedicated sign-vs-gap detector — decoupled from any classifier — that both the
deployed continuous GRU/LSTM (§12.3) and the record-then-recognize BiLSTM hybrid
(§16.2) can use as their segmentation source instead of a fixed heuristic.

**How to run on Kaggle**
1. *New Notebook* → *File → Import Notebook* → this `.ipynb`.
2. *Add Input* → your dataset `bracu23101281/gislr-stratified` (found automatically
   under `/kaggle/input`; set `INPUT_DIR` in the setup cell if not).
3. Accelerator: **None** (CPU only; no GPU needed). Internet: not needed.
4. *Save Version* → **Save & Run All (Commit)**.
5. When it finishes: Output tab → **New Dataset**, e.g. `GISLR-GapCorpus` (private by default).

**Output limit.** Kaggle keeps at most ~20 GB of `/kaggle/working`. `OUTPUT_LIMIT_GB`
stops the build early (with instructions) if the projected size would exceed it;
`STORAGE_DTYPE = "float16"` halves it.

**What it builds**, from GISLR_Stratified's **`train.csv` + `test.csv` combined**
({config['n_passes']} independent random passes over every clip):
- Random-order, random-length ({config['signs_per_seq'][0]}–{config['signs_per_seq'][1]} signs)
  sequences, one signer each — no sentence structure, no coverage requirement.
- Rest: **lowered** (realistic — arms ramp to hip height, hands vanish only once out
  of frame) with probability {config['p_lowered_rest']}, else hands-absent
  (GISLR-Sentences v1's style) — a deliberate mix so a model trained on this handles
  both.
- Transitions: {config['gap_frames'][0]}–{config['gap_frames'][1]} interpolated frames between signs.
- Per sequence: `landmarks (T,543,3)`, `frame_kind` (0 sign / 1 transition / 2 rest),
  **`gap` (T,) uint8** — 1 on every non-sign frame, the label this dataset exists for
  — `segments`, `labels`, `lowered_rest` (bool).
- **`train.csv` / `test.csv`**: one row per sequence — `glosses`, `starts`/`ends`
  (the frame each sign starts/ends — everything outside those spans is a gap),
  `npz_relpath`, `pass_id`, `lowered_rest`, `participant_id`. Split by **participant**
  ({config['train_signer_frac']:.0%} of signers in train), independent of
  GISLR_Stratified's own clip-level split — this model's generalization question is
  new signers, not new glosses.
""")

    md("## Setup")
    code(f"""
# ============================================================
# Tunables, config, paths
# ============================================================
import hashlib
import json
import os
import shutil
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

INPUT_DIR = None             # None = auto-detect GISLR_Stratified under /kaggle/input
OUT_DIR = Path("/kaggle/working/gislr-gapcorpus")
STORAGE_DTYPE = "float32"    # "float16" halves the size
OUTPUT_LIMIT_GB = 19.0
N_WORKERS = os.cpu_count() or 4

CONFIG = json.loads({_raw(json.dumps(config, indent=2))})
CONFIG_SHA = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
CODE_SHA256 = "{code_sha}"
SIGNBRIDGE_COMMIT = "{_git_commit()}"


def find_input() -> Path:
    for depth in range(1, 6):
        hits = [p.parent for p in Path("/kaggle/input").glob("/".join(["*"] * depth) + "/test.csv")
                if (p.parent / "train.csv").is_file() and (p.parent / "test").is_dir()]
        if hits:
            assert len(hits) == 1, f"several GISLR_Stratified candidates, set INPUT_DIR: {{hits}}"
            return hits[0]
    raise FileNotFoundError("GISLR_Stratified not found under /kaggle/input -- add it as an input")


DATA_DIR = Path(INPUT_DIR) if INPUT_DIR else find_input()
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


print(f"input:   {{DATA_DIR}}")
print(f"output:  {{OUT_DIR}}  ({{STORAGE_DTYPE}}, {{N_WORKERS}} workers)")
print(f"config:  sha256={{CONFIG_SHA[:12]}}   code sha256={{CODE_SHA256[:12]}} (commit {{SIGNBRIDGE_COMMIT}})")
""")

    md("""
## Embedded signbridge code

The exact source of the modules the build uses, registered under their real names,
so `compose`'s own imports resolve unchanged. Numpy/stdlib-only.
""")
    lines = ["""
# ============================================================
# Register the embedded modules
# ============================================================
_SOURCES = {}
"""]
    for name, src in sources.items():
        lines.append(f"_SOURCES[{name!r}] = {_raw(src)}\n")
    lines.append("""

def _register(name: str, source: str) -> types.ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            pkg.__path__ = []
            sys.modules[parent] = pkg
    module = types.ModuleType(name)
    module.__file__ = f"<embedded {name}>"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    setattr(sys.modules[".".join(parts[:-1])], parts[-1], module)
    return module


for _name, _src in _SOURCES.items():
    _register(_name, _src)
assert hashlib.sha256("".join(_SOURCES.values()).encode()).hexdigest() == CODE_SHA256
compose = sys.modules["sb.recognize.sequences.compose"]
print(f"modules: {list(_SOURCES)}")
""")
    code("".join(lines))

    md("""
## 1. Label map + source pool

Both `train.csv` and `test.csv` combined into one clip pool (a gap detector's task
does not depend on the 250-gloss classification split those files encode). Signers
are split into this corpus's own train/test groups, disjoint by participant.
""")
    code("""
# ============================================================
# Label map + combined clip pool + signer-disjoint split
# ============================================================
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
signs = set(train_df["sign"]) | set(test_df["sign"])
LABEL_MAP = {sign: i for i, sign in enumerate(sorted(signs))}
(OUT_DIR / "sign_to_prediction_index_map.json").write_text(json.dumps(LABEL_MAP), encoding="utf-8")

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
""")

    md("""
## 2. Plan: many random passes over the whole pool

`plan_control_multi` -- no sentence structure, no coverage requirement: each of
`n_passes` draws places every clip once, in a fresh random order and neighbour set.
""")
    code("""
# ============================================================
# Plan (resumable in `plan.json`)
# ============================================================
if PLAN_PATH.exists() and json.loads(PLAN_PATH.read_text())["config_sha256"] == CONFIG_SHA:
    PLAN = json.loads(PLAN_PATH.read_text())
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
    PLAN = {"config_sha256": CONFIG_SHA, "sequences": rows}
    write_json(PLAN_PATH, PLAN)

by_split = pd.Series([r["split"] for r in PLAN["sequences"]]).value_counts()
print(f"{len(PLAN['sequences'])} sequences over {CONFIG['n_passes']} passes")
print(by_split)
""")

    md("""
## 3. Materialize the sequences

Manifest-driven and resumable, like GISLR-Sentences. Rest is lowered (realistic) or
hands-absent per sequence, drawn once per row so `lowered_rest` in the manifest
matches what's actually in the npz. Size is projected after the first 300 sequences.
""")
    code("""
# ============================================================
# Build every sequence npz
# ============================================================
PROJECT_AFTER = 300

rows = PLAN["sequences"]
dtype = np.dtype(STORAGE_DTYPE)
manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def build(i_row):
    i, row = i_row
    rel = f"sequences/{row['seq_id']}.npz"
    try:
        arrays = compose.materialize_v2(
            row, DATA_DIR, relpath_of, LABEL_MAP, np.random.default_rng([CONFIG["seed"], i]),
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
            raise RuntimeError(
                f"projected {projected:.1f} GB > OUTPUT_LIMIT_GB={OUTPUT_LIMIT_GB}. Lower n_passes in "
                f"the config, or set STORAGE_DTYPE = 'float16'; delete {SEQ_DIR} and {MANIFEST_PATH}, "
                f"and re-run.")
        if k % 1000 == 0:
            write_json(MANIFEST_PATH, manifest)
write_json(MANIFEST_PATH, manifest)
bar.close()
print(f"{sum(v['status'] == 'done' for v in manifest.values())}/{len(rows)} done, {n_failed} failed this run")
""")

    md("## 4. train.csv / test.csv, build record")
    code("""
# ============================================================
# train.csv / test.csv (gloss list + segment frames), build_info.json
# ============================================================
manifest = json.loads(MANIFEST_PATH.read_text())
failed = [k for k, v in manifest.items() if v["status"] != "done"]
assert not failed, f"{len(failed)} sequences not done -- re-run section 3: {failed[:5]}"

order = [r["seq_id"] for r in PLAN["sequences"]]
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
    "built_on": "kaggle", "storage_dtype": STORAGE_DTYPE,
    "config": CONFIG, "config_sha256": CONFIG_SHA, "code_sha256": CODE_SHA256,
    "signbridge_commit": SIGNBRIDGE_COMMIT,
    "source": {"kaggle_ref": "bracu23101281/gislr-stratified", "resolved_dir": str(DATA_DIR)},
    "frames": frames.reset_index().to_dict(orient="records"),
    "mean_gap_fraction_sampled": float(np.mean(list(gap_rate.values()))),
    "size_bytes": size_bytes,
}
write_json(OUT_DIR / "build_info.json", build_info)

card = f\"\"\"# GISLR-GapCorpus {CONFIG['dataset_version']}

Sign-vs-gap training corpus (signbridge TODO §17): random-order, random-length landmark
sequences from **every** GISLR_Stratified clip (`train.csv` + `test.csv`), drawn over
{CONFIG['n_passes']} independent passes -- no sentence structure, no exactly-once coverage
requirement, since the target is the transition itself, not language.

## Files
- `sequences/<seq_id>.npz`: `landmarks (T,543,3) {STORAGE_DTYPE}` (canonical gislr-holistic
  row order), `frame_labels (T,) int16` (-1 null), `frame_kind (T,) uint8` (0 sign /
  1 transition / 2 rest), **`gap (T,) uint8`** (1 on every non-sign frame -- the
  training target), `segments (n,2) int32`, `labels (n,) int16`, `lowered_rest` (bool).
- `train.csv` / `test.csv`: one row per sequence -- `glosses`, `labels`, `starts`,
  `ends` (frame indices; a sign occupies `[start, end)`, everything else is a gap),
  `n_frames`, `npz_relpath`, `pass_id`, `lowered_rest`, `participant_id`.
  Split by **signer** ({CONFIG['train_signer_frac']:.0%} of participants in train), independent
  of GISLR_Stratified's own clip split.
- `sign_to_prediction_index_map.json`, `plan.json`, `build_info.json`.

## Rest synthesis
- **lowered** (probability {CONFIG['p_lowered_rest']}): arms ramp to hip height over
  {CONFIG['rest_ramp_frames']} frames; each hand goes NaN only once its wrist crosses
  {CONFIG['hand_visible_max']} (out of frame) -- realistic, fixes GISLR-Sentences v1's
  "hands NaN always means gap" tell.
- **hands-absent** (otherwise): edge frame held, both hands NaN throughout -- kept in
  the mix so a model trained here also handles that simpler case.

## Caveats
- Non-sign frames are synthetic (interpolated transitions, synthesized rest); real
  continuous signing was not used to build this. Treat any number as an upper bound.
- {CONFIG['n_passes']}x reuse of the same clip pool means the *same* clip appears in
  several different neighbour contexts across passes -- deliberate for a boundary
  task, but this is not a source of {CONFIG['n_passes']}x more distinct hand shapes.
\"\"\"
(OUT_DIR / "README.md").write_text(card, encoding="utf-8")
print(frames)
print(f"mean gap fraction (500-sequence sample): {build_info['mean_gap_fraction_sampled']:.3f}")
print(f"size: {size_bytes / 1e9:.2f} GB ({STORAGE_DTYPE})")
""")

    md("## 5. Checks and figures")
    code("""
# ============================================================
# Round-trip sample + gap-label consistency
# ============================================================
by_id = {r["seq_id"]: r for r in PLAN["sequences"]}
rng = np.random.default_rng(CONFIG["seed"])
sample = sorted(rng.choice(full_df["seq_id"].to_numpy(), CONFIG["n_roundtrip_checks"], replace=False).tolist())
bad = []
for seq_id in tqdm(sample, desc="round-trip"):
    arrays = compose.read_sequence(OUT_DIR / full_df.set_index("seq_id").loc[seq_id, "npz_relpath"])
    try:
        compose.check_roundtrip(arrays, by_id[seq_id], DATA_DIR, relpath_of)
        assert np.array_equal(arrays["gap"], (arrays["frame_kind"] != compose.SIGN).astype(np.uint8))
    except AssertionError as e:
        bad.append((seq_id, str(e)))
write_json(OUT_DIR / "roundtrip_sample.json", {"seed": CONFIG["seed"], "seq_ids": sample, "failures": bad})
print(f"round-trip: {len(sample) - len(bad)}/{len(sample)} exact at {STORAGE_DTYPE}")
assert not bad, "checks failed -- do not publish this output"
""")
    code("""
# ============================================================
# Figures: lengths + one exemplar sequence (both rest styles)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(full_df["n_frames"], bins=60, color="#4c72b0")
axes[0].set_title(f"{len(full_df)} sequences, mean {full_df.n_frames.mean():.0f} frames")
axes[0].set_xlabel("frames")
full_df["lowered_rest"].value_counts().plot.bar(ax=axes[1], color=["#dd8452", "#4c72b0"])
axes[1].set_title("rest style")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "lengths.png", dpi=110)
plt.show()

for style, want_lowered in (("lowered", True), ("hands-absent", False)):
    ex = full_df[full_df["lowered_rest"] == want_lowered].iloc[0]
    arr = compose.read_sequence(OUT_DIR / ex["npz_relpath"])
    fig, ax = plt.subplots(figsize=(14, 3.2))
    for idx, name in ((489 + 15, "left wrist"), (489 + 16, "right wrist")):
        ax.plot(-arr["landmarks"][:, idx, 1], label=f"pose {name} (-y)")
    colors = {compose.TRANSITION: "#dddddd", compose.REST: "#f4cccc"}
    for t, k in enumerate(arr["frame_kind"]):
        if k in colors:
            ax.axvspan(t - 0.5, t + 0.5, color=colors[k], lw=0)
    for (s, e), gl in zip(arr["segments"], ex["glosses"].split()):
        ax.text((s + e) / 2, ax.get_ylim()[1], gl.upper(), ha="center", va="bottom")
    ax.set_title(f"{ex['seq_id']} ({style} rest): grey = transition, pink = rest")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"exemplar_{style}.png", dpi=110)
    plt.show()
""")

    md("""
## 6. Clean up, then publish

Build scratch removed so the output folder is exactly the dataset. From the committed
version's **Output** tab: **New Dataset** → e.g. `GISLR-GapCorpus`. Private by default.
""")
    code("""
# ============================================================
# Remove build scratch from the output folder
# ============================================================
MANIFEST_PATH.unlink(missing_ok=True)
for p in OUT_DIR.rglob("*.tmp*"):
    p.unlink()
n_files = sum(1 for p in OUT_DIR.rglob("*") if p.is_file())
total = sum(p.stat().st_size for p in OUT_DIR.rglob("*") if p.is_file())
print(f"{OUT_DIR}: {n_files} files, {total / 1e9:.2f} GB -- ready to publish as a dataset")
""")

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> None:
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(nb['cells'])} cells, {OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
