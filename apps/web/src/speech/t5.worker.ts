/**
 * T5 in a Web Worker, so beam search never blocks the page. Messages:
 *   in:  {type: "load", base} | {type: "refine", id, english, ruleV1}
 *   out: {type: "progress", loaded, total} | {type: "ready", manifest, threads, ms}
 *        | {type: "refined", id, gloss, ms} | {type: "error", id?, message}
 * Threads: several only when the page is cross-origin isolated (COOP/COEP from `public/_headers`).
 */

import * as ort from "onnxruntime-web/wasm";
import { loadParts } from "./loader.ts";
import { T5Refiner } from "./t5.ts";
import type { T5Manifest } from "./t5.ts";

let t5: T5Refiner | null = null;
const post = (m: unknown) => (self as unknown as Worker).postMessage(m);

async function load(base: string): Promise<void> {
  const t0 = performance.now();
  const mr = await fetch(`${base}t5/manifest.json`);
  if (!mr.ok) throw new Error(`t5/manifest.json: HTTP ${mr.status} (run tools/export_speech.py assets)`);
  const manifest = (await mr.json()) as T5Manifest;
  const tokenizer = await (await fetch(`${base}t5/${manifest.tokenizer.file}`)).json();
  const total = manifest.files.encoder.bytes + manifest.files.decoder.bytes;
  let loaded = 0;
  const tick = (n: number) => { loaded += n; post({ type: "progress", loaded, total }); };
  const enc = await loadParts(`${base}t5/`, manifest.files.encoder, tick);
  const dec = await loadParts(`${base}t5/`, manifest.files.decoder, tick);
  const threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.wasmPaths = new URL(`${base}wasm/ort/`, self.location.origin).href;
  ort.env.wasm.numThreads = threads;
  t5 = await T5Refiner.create(ort, manifest, enc, dec, tokenizer, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  post({ type: "ready", manifest, threads, ms: performance.now() - t0 });
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data as { type: string; id?: number; base?: string; english?: string; ruleV1?: string };
  try {
    if (m.type === "load") await load(m.base!);
    else if (m.type === "refine") {
      if (!t5) throw new Error("T5 not loaded");
      const t0 = performance.now();
      const out = await t5.refine(m.english!, m.ruleV1!);
      post({ type: "refined", id: m.id, gloss: out.gloss, ms: performance.now() - t0 });
    }
  } catch (err) {
    post({ type: "error", id: m.id, message: String(err instanceof Error ? err.message : err) });
  }
};
