"""Publish SignBridge's models to Kaggle Models, public, MIT (TODO §9, §12.5, §13).

Three bundles, each one new version of a Kaggle variation:

- ``signbridge-gislr/tfLite/c1-web``: what ``apps/web`` (sign -> speech) serves under
  ``/assets/``, same layout (``model/``, ``pipeline.json``, ``prior.json``, ``lexicon.json``).
- ``signbridge-speech-to-gloss/onnx/t5-web``: what ``speech.html`` serves under ``/assets/t5/``.
- ``signbridge-speech-to-gloss/transformers/t5-hybrid``: the fine-tuned T5 checkpoint.

The PyTorch checkpoints are not handled here: ``sb-sync push`` uploads those to
``signbridge-gislr/pyTorch/<architecture>``. This script also sets both models' visibility
and model card (``kaggle_cards/<model>.md``).

    .venv/Scripts/python.exe apps/web/tools/publish_kaggle.py             # plan only
    .venv/Scripts/python.exe apps/web/tools/publish_kaggle.py --apply     # upload + publish
    .venv/Scripts/python.exe apps/web/tools/publish_kaggle.py --apply --only cards

``apps/web/scripts/fetch-models.ts`` downloads these bundles back into ``public/assets/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sb.core.paths import DATA_DIR, ROOT_DIR, TEMP_DIR

OWNER = "bracu23101281"
LICENSE = "MIT"
WEB = ROOT_DIR / "apps" / "web"
ASSETS = WEB / "public" / "assets"
T5_DIR = DATA_DIR / "external" / "t5-text2gloss" / "thesis_hybrid_dataset1" / "model"
STAGE = TEMP_DIR / "kaggle_publish"
KAGGLE = Path(sys.executable).parent / "kaggle.exe"

MODELS = {
    "signbridge-gislr": ("SignBridge GISLR sign recognition",
                         "Streaming ASL sign recognition from MediaPipe landmarks, 250 signs"),
    "signbridge-speech-to-gloss": ("SignBridge English to ASL gloss",
                                   "T5 fine-tuned to turn English into ASL gloss, plus its browser export"),
}


def c1_files() -> dict[str, Path]:
    names = ["model/model.tflite", "model/classes.f32", "model/manifest.json",
             "pipeline.json", "prior.json", "lexicon.json"]
    return {n: ASSETS / n for n in names}


def t5_web_files() -> dict[str, Path]:
    d = ASSETS / "t5"
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    parts = [p for f in manifest["files"].values() for p in f["parts"]]
    return {n: d / n for n in ["manifest.json", "tokenizer.json", *parts]}


def t5_files() -> dict[str, Path]:
    # training_args.bin is left out: a pickle, and not needed for inference
    names = ["config.json", "generation_config.json", "model.safetensors",
             "tokenizer.json", "tokenizer_config.json"]
    return {n: T5_DIR / n for n in names}


def c1_notes() -> str:
    m = json.loads((ASSETS / "model" / "manifest.json").read_text(encoding="utf-8"))
    return (f"apps/web sign -> speech bundle: continuous GRU C1 (run {m['run_id']}), "
            f"{len(m['landmarks'])} landmarks {m['coords']}, state {m['state_shape']}, "
            f"decoder {m['decoder']['name']}; model.tflite sha256 {m['sha256']['model.tflite'][:12]}")


def t5_web_notes() -> str:
    m = json.loads((ASSETS / "t5" / "manifest.json").read_text(encoding="utf-8"))
    return (f"apps/web speech -> gloss bundle: int8 encoder + fp32 decoder in < 25 MiB parts; "
            f"source weights sha256 {m['weights_sha256'][:12]}")


BUNDLES = {
    "c1-web": ("signbridge-gislr", "tfLite", c1_files, c1_notes),
    "t5-web": ("signbridge-speech-to-gloss", "onnx", t5_web_files, t5_web_notes),
    "t5-hybrid": ("signbridge-speech-to-gloss", "transformers", t5_files,
                  lambda: "fine-tuned T5-base, English + rule gloss -> ASL gloss (Hugging Face format)"),
}


def stage(name: str, files: dict[str, Path]) -> Path:
    out = STAGE / name
    if out.exists():
        shutil.rmtree(out)
    for rel, src in files.items():
        if not src.is_file():
            raise SystemExit(f"{name}: missing {src} (run the matching export first)")
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dest)  # NTFS hard link: no second copy of 850 MB
        except OSError:
            shutil.copy2(src, dest)
    if name == "t5-web":  # the export records an absolute local path; publish it repo-relative
        mp = out / "manifest.json"
        mp.unlink()
        m = json.loads((ASSETS / "t5" / "manifest.json").read_text(encoding="utf-8"))
        m["source_dir"] = T5_DIR.relative_to(ROOT_DIR).as_posix()
        mp.write_text(json.dumps(m, indent=1), encoding="utf-8")
    return out


def publish_card(slug: str) -> None:
    title, subtitle = MODELS[slug]
    d = STAGE / f"card-{slug}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model-metadata.json").write_text(json.dumps({
        "ownerSlug": OWNER, "slug": slug, "title": title, "subtitle": subtitle, "isPrivate": False,
        "description": (WEB / "tools" / "kaggle_cards" / f"{slug}.md").read_text(encoding="utf-8"),
    }, indent=1), encoding="utf-8")
    subprocess.run([str(KAGGLE), "models", "update", "-p", str(d)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="upload and publish (default: plan only)")
    ap.add_argument("--only", nargs="+", choices=[*BUNDLES, "cards"], help="a subset of the steps")
    a = ap.parse_args()
    steps = a.only or [*BUNDLES, "cards"]

    for name in [s for s in steps if s in BUNDLES]:
        slug, framework, files_fn, notes_fn = BUNDLES[name]
        files = files_fn()
        size = sum(p.stat().st_size for p in files.values()) / 2**20
        handle = f"{OWNER}/{slug}/{framework}/{name}"
        print(f"{handle}: {len(files)} files, {size:.0f} MB, {LICENSE}")
        if a.apply:
            import kagglehub

            kagglehub.model_upload(handle=handle, local_model_dir=str(stage(name, files)),
                                   license_name=LICENSE, version_notes=notes_fn())
    if "cards" in steps:
        print(f"model cards + public: {', '.join(MODELS)}")
        if a.apply:
            for slug in MODELS:
                publish_card(slug)
    if not a.apply:
        print("plan only; pass --apply to upload")


if __name__ == "__main__":
    main()
