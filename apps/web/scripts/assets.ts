/**
 * Runtime files the page serves but the repo does not commit (all gitignored):
 * - public/wasm/litert, public/wasm/mediapipe: WASM runtimes copied from node_modules;
 * - public/assets/holistic_landmarker.task: MediaPipe's model (about 14 MB), downloaded
 *   once (the app falls back to Google's URL if it is missing).
 * The model, prior, lexicon and replay streams come from Python:
 *   .venv/Scripts/python.exe apps/web/tools/export.py assets
 */
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const APP = join(dirname(fileURLToPath(import.meta.url)), "..");
const PUB = join(APP, "public");

for (const [from, to] of [["@litertjs/core/wasm", "litert"], ["@mediapipe/tasks-vision/wasm", "mediapipe"]]) {
  cpSync(join(APP, "node_modules", from), join(PUB, "wasm", to), { recursive: true });
}
console.log("wasm runtimes -> public/wasm/");

const cfg = JSON.parse(readFileSync(join(APP, "pipeline.config.json"), "utf-8")) as { holistic_model_url: string };
const task = join(PUB, "assets", "holistic_landmarker.task");
if (!existsSync(task)) {
  try {
    const r = await fetch(cfg.holistic_model_url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    mkdirSync(dirname(task), { recursive: true });
    writeFileSync(task, Buffer.from(await r.arrayBuffer()));
    console.log("holistic_landmarker.task downloaded");
  } catch (e) {
    console.warn(`could not download the Holistic model (${String(e)}); the app will fetch it from Google at runtime`);
  }
}
if (!existsSync(join(PUB, "assets", "pipeline.json"))) {
  console.warn("public/assets/pipeline.json missing: run .venv/Scripts/python.exe apps/web/tools/export.py assets");
}
