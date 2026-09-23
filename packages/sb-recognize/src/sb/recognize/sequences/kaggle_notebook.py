"""Generate the self-contained Kaggle build notebook for GISLR-Sentences.

    python -m sb.recognize.sequences.kaggle_notebook

writes ``experiments/recognition/gislr.0.dataset.sentences-kaggle.ipynb``: the
same build as ``gislr.0.dataset.sentences.ipynb``, but runnable on Kaggle
with GISLR_Stratified attached as an input and **no signbridge install** --
the output folder becomes the dataset directly, instead of a ~16 GB upload
from this machine.

Nothing is re-implemented. The notebook embeds the **exact source** of every
module the build touches (``sb.core.schema``, ``sb.core.vocab``,
``sb.recognize.features.gislr_stratified``, ``sb.recognize.sequences.corpus``,
``sb.recognize.sequences.compose`` -- all numpy/stdlib-only) and registers it
under the same module names, so ``compose``'s own imports resolve unchanged.
It also embeds the corpus files and ``configs/gislr.sentences.json``. Re-run
this generator after touching any of them; the notebook records the git
commit and a hash of the embedded code in ``build_info.json``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from sb.core.paths import EXPERIMENTS_DIR, PACKAGES_DIR, ROOT_DIR
from sb.recognize.sequences import corpus

OUT = EXPERIMENTS_DIR / "recognition" / "gislr.0.dataset.sentences-kaggle.ipynb"
CONFIG_PATH = EXPERIMENTS_DIR / "recognition" / "configs" / "gislr.sentences.json"

# module name -> source file, in dependency order
MODULES = {
    "sb.core.schema": PACKAGES_DIR / "sb-core/src/sb/core/schema.py",
    "sb.core.vocab": PACKAGES_DIR / "sb-core/src/sb/core/vocab.py",
    "sb.recognize.features.gislr_stratified":
        PACKAGES_DIR / "sb-recognize/src/sb/recognize/features/gislr_stratified.py",
    "sb.recognize.sequences.corpus": PACKAGES_DIR / "sb-recognize/src/sb/recognize/sequences/corpus.py",
    "sb.recognize.sequences.compose": PACKAGES_DIR / "sb-recognize/src/sb/recognize/sequences/compose.py",
}


def _raw(text: str) -> str:
    """``text`` as a raw triple-quoted literal; refuses what cannot be one."""
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
    version = config["corpus_version"]
    sources = {name: path.read_text(encoding="utf-8").replace("\r\n", "\n") for name, path in MODULES.items()}
    code_sha = hashlib.sha256("".join(sources[n] for n in MODULES).encode()).hexdigest()
    corpus_files = {corpus.lexicon_path(version).name: corpus.lexicon_path(version)}
    corpus_files |= {f"{corpus.sentences_dir(version).name}/{p.name}": p
                     for p in sorted(corpus.sentences_dir(version).glob("*.txt"))}

    cells: list[dict] = []

    def md(s: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": s.strip("\n").splitlines(keepends=True)})

    def code(s: str) -> None:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": s.strip("\n").splitlines(keepends=True)})

    md(f"""
# GISLR-Sentences {config['dataset_version']} — Kaggle build (self-contained)

**Generated file — do not edit.** Regenerate with
`python -m sb.recognize.sequences.kaggle_notebook` (signbridge commit
`{_git_commit()}`, embedded-code sha256 `{code_sha[:12]}`). This is the Kaggle-side twin of
`gislr.0.dataset.sentences.ipynb` (TODO §12.1): same code, same config, same corpus,
same seed, so it produces the same plan and the same sequences. The only
difference is where the output lands: `/kaggle/working/gislr-sentences/`, which
becomes the dataset directly with no upload from a local machine.

**How to run on Kaggle**
1. *New Notebook* → *File → Import Notebook* → this `.ipynb`.
2. *Add Input* → your dataset `bracu23101281/gislr-stratified` (it is found automatically
   under `/kaggle/input`; set `INPUT_DIR` in the setup cell if not).
3. Accelerator: **None** (CPU only; no GPU needed). Internet: not needed.
4. *Save Version* → **Save & Run All (Commit)**. The build is expected to take about
   15–30 min on Kaggle's CPUs.
5. When the version finishes: open it → *Output* tab → **New Dataset**, titled
   `GISLR-Sentences` (slug `gislr-sentences`). New datasets are **private** by default.

**Output limit.** Kaggle keeps at most ~20 GB of `/kaggle/working`. The float32 build
is estimated at ~16 GB. §3 projects the final size after the first sequences and stops
early, with instructions, if it would not fit; `STORAGE_DTYPE = "float16"` halves it.

**What it builds** (from GISLR_Stratified `{config['source_split']}.csv` only):
- `sentence` split: {len(corpus.load_sentences(version))}-sentence ASL-gloss-order corpus,
  one signer per sequence, coverage-driven so every test clip is used (~8% of slots
  re-use a clip, flagged in `reused`).
- `control` split: the same clips in random order, each exactly once.
- Synthesized null frames: {config['synth']['gap_frames'][0]}–{config['synth']['gap_frames'][1]} interpolated
  transition frames between signs; {config['synth']['rest_frames'][0]}–{config['synth']['rest_frames'][1]} rest frames
  at each end with hands absent (NaN).
- Per sequence: `landmarks (T,543,3)` in canonical row order, `frame_labels` (−1 null),
  `frame_kind` (0 sign / 1 transition / 2 rest), `segments` (start, end-exclusive), `labels`.
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
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

INPUT_DIR = None             # None = auto-detect GISLR_Stratified under /kaggle/input
OUT_DIR = Path("/kaggle/working/gislr-sentences")
STORAGE_DTYPE = "float32"    # "float16" halves the size; segments then match the source at float16
OUTPUT_LIMIT_GB = 19.0       # Kaggle keeps ~20 GB of /kaggle/working
N_WORKERS = os.cpu_count() or 4

# configs/gislr.sentences.json, embedded verbatim at generation time
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
print(f"config:  sha256={{CONFIG_SHA[:12]}}   code: sha256={{CODE_SHA256[:12]}} (commit {{SIGNBRIDGE_COMMIT}})")
""")

    md("""
## Embedded signbridge code

The exact source of the five modules the build uses, registered under their
real names, so `compose`'s own `from sb.core.schema import ...` lines resolve
unchanged. They are numpy/stdlib-only. Then the committed corpus files are
written to a temp directory, and `corpus.CORPUS_DIR` is pointed at it.
""")
    lines = ["""
# ============================================================
# Register the embedded modules + write the corpus files
# ============================================================
_SOURCES = {}
"""]
    for name, src in sources.items():
        lines.append(f"_SOURCES[{name!r}] = {_raw(src)}\n")
    lines.append("""

def _register(name: str, source: str) -> types.ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts)):  # namespace parents: sb, sb.core, ...
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
corpus = sys.modules["sb.recognize.sequences.corpus"]

_CORPUS_FILES = {}
""")
    for rel, path in corpus_files.items():
        lines.append(f"_CORPUS_FILES[{rel!r}] = {_raw(path.read_text(encoding='utf-8'))}\n")
    lines.append("""
CORPUS_TMP = Path(tempfile.mkdtemp(prefix="corpus_"))
for rel, text in _CORPUS_FILES.items():
    (CORPUS_TMP / rel).parent.mkdir(parents=True, exist_ok=True)
    (CORPUS_TMP / rel).write_text(text, encoding="utf-8")
corpus.CORPUS_DIR = CORPUS_TMP
print(f"modules: {list(_SOURCES)}")
print(f"corpus:  {len(_CORPUS_FILES)} files, sha256={corpus.corpus_sha256(CONFIG['corpus_version'])[:12]}")
""")
    code("".join(lines))

    md("""
## 1. Label map + corpus validation

GISLR_Stratified does not ship `sign_to_prediction_index_map.json`. It is
derived exactly as signbridge's `gislr_dir()` does (sorted signs → index), so
the labels here match every signbridge model. Then the corpus is validated and
a sample is printed.
""")
    code("""
# ============================================================
# Label map (derived like sb.core.paths.gislr_dir) + corpus validation
# ============================================================
N_SAMPLE = 30

vocab = sys.modules["sb.core.vocab"]
signs = set(pd.read_csv(DATA_DIR / "train.csv")["sign"]) | set(pd.read_csv(DATA_DIR / "test.csv")["sign"])
LABEL_MAP = {sign: i for i, sign in enumerate(sorted(signs))}
(OUT_DIR / vocab.LABEL_MAP_FILE).write_text(json.dumps(LABEL_MAP), encoding="utf-8")

src_df = pd.read_csv(DATA_DIR / f"{CONFIG['source_split']}.csv")
sentences = corpus.load_sentences(CONFIG["corpus_version"])
lexicon = corpus.load_lexicon(CONFIG["corpus_version"])
assert set(lexicon) == set(src_df["sign"]) == set(LABEL_MAP), "lexicon / vocabulary / label map disagree"
report = corpus.validate(sentences, src_df["sign"].unique(), **CONFIG["corpus"])
print(f"{report['n_sentences']} sentences, {report['n_themes']} themes, mean length "
      f"{report['mean_len']:.2f}, lengths {report['length_hist']}")
print(f"missing: {report['missing'] or 'none'}   below minimum: {report['thin'] or 'none'}")
rng = np.random.default_rng(CONFIG["seed"])
for i in rng.choice(len(sentences), N_SAMPLE, replace=False):
    print(f"  {sentences[i].id:<22} {' '.join(g.upper() for g in sentences[i].glosses)}")
""")

    md("""
## 2. Plan: assign every test clip to a sequence

The same planner, config and seed as the local notebook, so the plan is
identical. It is cached in `plan.json` for resumption within a session.
""")
    code("""
# ============================================================
# Plan both splits
# ============================================================
if PLAN_PATH.exists() and json.loads(PLAN_PATH.read_text())["config_sha256"] == CONFIG_SHA:
    PLAN = json.loads(PLAN_PATH.read_text())
    print("plan loaded")
else:
    src_df = pd.read_csv(DATA_DIR / f"{CONFIG['source_split']}.csv")
    rng = np.random.default_rng(CONFIG["seed"])
    synth = dict(gap_range=tuple(CONFIG["synth"]["gap_frames"]),
                 rest_range=tuple(CONFIG["synth"]["rest_frames"]))
    bar = tqdm(total=len(src_df), desc="plan sentence split")
    sent_plan, stats = compose.plan_sentences(
        src_df, corpus.load_sentences(CONFIG["corpus_version"]), rng, **synth, **CONFIG["planner"],
        progress=lambda done, total: bar.update(done - bar.n))
    bar.close()
    ctrl_plan = compose.plan_control(src_df, np.array([len(r["glosses"]) for r in sent_plan]), rng, **synth)
    for i, r in enumerate(sent_plan):
        r["seq_id"] = f"s{i:05d}"
    for i, r in enumerate(ctrl_plan):
        r["seq_id"] = f"c{i:05d}"
    PLAN = {"config_sha256": CONFIG_SHA, "stats": stats, "sequences": sent_plan + ctrl_plan}
    write_json(PLAN_PATH, PLAN)

st = PLAN["stats"]
n_sent = sum(r["kind"] == "sentence" for r in PLAN["sequences"])
print(f"sentence: {n_sent} sequences, {st['n_clips']} clips used, reuse {st['reuse_rate']:.1%}, "
      f"orphans {st['n_orphan_clips']}, {st['n_distinct_sentences']} distinct sentences")
print(f"control:  {len(PLAN['sequences']) - n_sent} sequences")
""")

    md("""
## 3. Materialize the sequences

Manifest-driven: each npz is written atomically, then marked `done`. Each
sequence's jitter RNG is seeded from `(seed, row index)`. After the first 300
sequences the final size is projected, and the build stops if it would exceed
`OUTPUT_LIMIT_GB`.
""")
    code("""
# ============================================================
# Build every sequence npz (resumable, size-guarded)
# ============================================================
PROJECT_AFTER = 300

rows = PLAN["sequences"]
src_df = pd.read_csv(DATA_DIR / f"{CONFIG['source_split']}.csv")
relpath_of = dict(zip(src_df["uid"], src_df["npz_relpath"]))
dtype = np.dtype(STORAGE_DTYPE)
manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def build(i_row):
    i, row = i_row
    rel = f"sequences/{row['kind']}/{row['seq_id']}.npz"
    try:
        arrays = compose.materialize(row, DATA_DIR, relpath_of, LABEL_MAP,
                                     np.random.default_rng([CONFIG["seed"], i]),
                                     rest_jitter=CONFIG["synth"]["rest_jitter"])
        compose.write_sequence(OUT_DIR / rel, arrays, dtype=dtype)
        rec = compose.summarize_row(row, arrays, row["seq_id"], rel)
        return row["seq_id"], {"status": "done", "bytes": (OUT_DIR / rel).stat().st_size, **rec}
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
                f"projected {projected:.1f} GB > OUTPUT_LIMIT_GB={OUTPUT_LIMIT_GB}. Set "
                f"STORAGE_DTYPE = 'float16', delete {SEQ_DIR} and {MANIFEST_PATH}, and re-run.")
        if k % 500 == 0:
            write_json(MANIFEST_PATH, manifest)
write_json(MANIFEST_PATH, manifest)
bar.close()
print(f"{sum(v['status'] == 'done' for v in manifest.values())}/{len(rows)} done, {n_failed} failed this run")
""")

    md("## 4. Index, corpus copy, build record, dataset card")
    code("""
# ============================================================
# sequences.csv, sentences.csv, lexicon, build_info, README
# ============================================================
manifest = json.loads(MANIFEST_PATH.read_text())
failed = [k for k, v in manifest.items() if v["status"] != "done"]
assert not failed, f"{len(failed)} sequences not done -- re-run section 3: {failed[:5]}"

order = [r["seq_id"] for r in PLAN["sequences"]]
seq_df = pd.DataFrame([{k: v for k, v in manifest[s].items() if k not in ("status", "bytes")} for s in order])
seq_df.to_csv(OUT_DIR / "sequences.csv", index=False)
sentences = corpus.load_sentences(CONFIG["corpus_version"])
pd.DataFrame([{"sentence_id": s.id, "theme": s.theme, "glosses": " ".join(s.glosses)}
              for s in sentences]).to_csv(OUT_DIR / "sentences.csv", index=False)
shutil.copy(corpus.lexicon_path(CONFIG["corpus_version"]), OUT_DIR / corpus.lexicon_path(CONFIG["corpus_version"]).name)

frames = seq_df.groupby("kind")["n_frames"].agg(["count", "sum", "mean", "max"])
size_bytes = sum(p.stat().st_size for p in SEQ_DIR.rglob("*.npz"))
build_info = {
    "dataset": "GISLR-Sentences", "dataset_version": CONFIG["dataset_version"],
    "built_on": "kaggle", "storage_dtype": STORAGE_DTYPE,
    "config": CONFIG, "config_sha256": CONFIG_SHA,
    "corpus_sha256": corpus.corpus_sha256(CONFIG["corpus_version"]),
    "code_sha256": CODE_SHA256, "signbridge_commit": SIGNBRIDGE_COMMIT,
    "source": {"kaggle_ref": "bracu23101281/gislr-stratified", "resolved_dir": str(DATA_DIR),
               "split": CONFIG["source_split"]},
    "planner_stats": PLAN["stats"],
    "frames": frames.reset_index().to_dict(orient="records"),
    "size_bytes": size_bytes,
}
write_json(OUT_DIR / "build_info.json", build_info)

s = PLAN["stats"]
g, r = CONFIG["synth"]["gap_frames"], CONFIG["synth"]["rest_frames"]
card = f\"\"\"# GISLR-Sentences {CONFIG['dataset_version']}

Continuous multi-sign landmark sequences built from **real** isolated-sign clips of
GISLR_Stratified's `{CONFIG['source_split']}.csv` (`bracu23101281/gislr-stratified`, 250 glosses). Signed
clips are unmodified; only the non-sign frames between and around them are synthesized.
For evaluating streaming/continuous recognizers and sign-boundary reset (signbridge TODO §12).

## Splits
- **sentence** ({(seq_df.kind == 'sentence').sum()} sequences): {s['n_distinct_sentences']} distinct ASL-gloss-order
  sentences from a {len(sentences)}-sentence corpus. Every source clip is used;
  {s['reuse_rate']:.1%} of sign slots re-use an already-placed clip (flagged in `reused`).
- **control** ({(seq_df.kind == 'control').sum()} sequences): the same clips in random order, each exactly once.

One signer per sequence, always.

## Files
- `sequences/<kind>/<seq_id>.npz`: `landmarks (T,543,3) {STORAGE_DTYPE}` (NaN = not detected; canonical
  GISLR holistic row order: face 0-467, left_hand 468-488, pose 489-521, right_hand 522-542),
  `frame_labels (T,) int16` (-1 = null), `frame_kind (T,) uint8` (0 sign, 1 transition, 2 rest),
  `segments (n,2) int32` (start, end-exclusive), `labels (n,) int16`.
- `sequences.csv`: one row per sequence, with source clip uids and all segment boundaries.
- `sentences.csv`, `lexicon.{CONFIG['corpus_version']}.json`, `sign_to_prediction_index_map.json`, `plan.json`, `build_info.json`.

## Synthesized frames (labelled -1)
- **transition**: {g[0]}-{g[1]} frames between signs, linearly interpolated from the last frame of
  one clip to the first frame of the next.
- **rest**: {r[0]}-{r[1]} frames at each end, with the body held still plus Gaussian jitter
  (sd {CONFIG['synth']['rest_jitter']}) and both hands NaN.

`landmarks[frame_kind == 0]` is the back-to-back concatenation of the source clips.

## Caveats
- The non-sign frames are synthetic; real transitions and rest poses are not in GISLR. Treat
  scores as an upper bound on real continuous signing.
- GISLR_Stratified's test split is also the validation split of the signbridge isolated-sign
  models, so those models were checkpoint-selected on these clips.
- The vocabulary is toddler-level and has no sign for I/me (`minemy` = my/mine).
\"\"\"
(OUT_DIR / "README.md").write_text(card, encoding="utf-8")
print(frames)
print(f"size: {size_bytes / 1e9:.2f} GB ({STORAGE_DTYPE})")
""")

    md("## 5. Checks and figures")
    code("""
# ============================================================
# Round-trip sample + index-level consistency
# ============================================================
by_id = {r["seq_id"]: r for r in PLAN["sequences"]}
seq_df = pd.read_csv(OUT_DIR / "sequences.csv", keep_default_na=False).set_index("seq_id", drop=False)
src_df = pd.read_csv(DATA_DIR / f"{CONFIG['source_split']}.csv")
relpath_of = dict(zip(src_df["uid"], src_df["npz_relpath"]))

rng = np.random.default_rng(CONFIG["seed"])
sample = sorted(rng.choice(seq_df["seq_id"].to_numpy(), CONFIG["n_roundtrip_checks"], replace=False).tolist())
bad = []
for seq_id in tqdm(sample, desc="round-trip"):
    arrays = compose.read_sequence(OUT_DIR / seq_df.loc[seq_id, "npz_relpath"])
    try:
        compose.check_roundtrip(arrays, by_id[seq_id], DATA_DIR, relpath_of)
    except AssertionError as e:
        bad.append((seq_id, str(e)))
write_json(OUT_DIR / "roundtrip_sample.json", {"seed": CONFIG["seed"], "seq_ids": sample, "failures": bad})
print(f"round-trip: {len(sample) - len(bad)}/{len(sample)} exact at {STORAGE_DTYPE}")

issues = 0
for r in seq_df.itertuples():
    s = np.array(r.starts.split(), int)
    e = np.array(r.ends.split(), int)
    issues += not (len(s) == r.n_signs and (e > s).all() and (s[1:] >= e[:-1]).all()
                   and s[0] >= r.rest_in and e[-1] <= r.n_frames - r.rest_out)
print(f"segment-consistency issues across {len(seq_df)} sequences: {issues}")
used = pd.Series(" ".join(seq_df.loc[seq_df.kind == "sentence", "source_uids"]).split()).value_counts()
print(f"sentence split: {used.size}/{len(src_df)} source clips used; uses per clip "
      f"{used.value_counts().sort_index().to_dict()}")
assert not bad and not issues, "checks failed -- do not publish this output"
""")
    code("""
# ============================================================
# Figures: lengths + one exemplar sequence
# ============================================================
seq_df = pd.read_csv(OUT_DIR / "sequences.csv", keep_default_na=False)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for kind, ax in zip(("sentence", "control"), axes[:2]):
    sub = seq_df[seq_df.kind == kind]
    ax.hist(sub["n_frames"], bins=60, color="#4c72b0")
    ax.set_title(f"{kind}: {len(sub)} sequences, mean {sub.n_frames.mean():.0f} frames")
    ax.set_xlabel("frames")
seq_df.groupby("kind")["n_signs"].value_counts().unstack(0).fillna(0).plot.bar(ax=axes[2])
axes[2].set_title("signs per sequence")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "lengths.png", dpi=110)
plt.show()

ex = seq_df[(seq_df.kind == "sentence") & (seq_df.n_signs == 4)].iloc[0]
arr = compose.read_sequence(OUT_DIR / ex["npz_relpath"])
fig, ax = plt.subplots(figsize=(14, 3.5))
for idx, name in ((489 + 15, "left wrist"), (489 + 16, "right wrist")):
    ax.plot(-arr["landmarks"][:, idx, 1].astype(np.float32), label=f"{name} (-y)")
colors = {compose.TRANSITION: "#dddddd", compose.REST: "#f4cccc"}
for t, k in enumerate(arr["frame_kind"]):
    if k in colors:
        ax.axvspan(t - 0.5, t + 0.5, color=colors[k], lw=0)
for (s, e), gl in zip(arr["segments"], ex["glosses"].split()):
    ax.text((s + e) / 2, ax.get_ylim()[1], gl.upper(), ha="center", va="bottom")
ax.set_title(f"{ex['seq_id']} (signer {ex['participant_id']}): grey = transition, pink = rest (hands NaN)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "exemplar.png", dpi=110)
plt.show()
""")

    md("""
## 6. Clean up, then publish

The build scratch (`manifest.json`, temp files) is removed so the output
folder is exactly the dataset. Then, from the committed version's **Output**
tab: **New Dataset** → title `GISLR-Sentences`. It is private by default.
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
