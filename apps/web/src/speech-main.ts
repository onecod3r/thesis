/**
 * Speech -> gloss page (TODO §13 Phase 4, user decision 2026-09-25: browser-side, Free plan).
 *
 *   mic (hold to record) or an audio file ─► WAV ─► POST /api/edge /api/asr (Workers AI
 *     whisper-large-v3-turbo) ─► English text ─► spellNumbers ─► rules_v1 + rules_v2
 *     (wink-nlp) ─► T5 int8 ONNX in a Worker (optional, "guarded hybrid") ─► guard.combine
 *
 * T5 is optional: if its manifest 404s (checkpoint not exported yet) the page still works on
 * rules_v2 alone, same as `sb.synthesize.gloss.GlossEngine` without a checkpoint.
 */

import { durationSeconds, MicRecorder, wavBase64 } from "./speech/mic.ts";
import type { Recording } from "./speech/mic.ts";
import { convertV1 } from "./speech/rules_v1.ts";
import { convertV2 } from "./speech/rules_v2.ts";
import { combine } from "./speech/guard.ts";
import type { HybridOutput } from "./speech/guard.ts";
import { spellNumbers, splitSentences } from "./speech/text.ts";

const BASE = import.meta.env.BASE_URL;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  status: $<HTMLParagraphElement>("status"),
  record: $<HTMLButtonElement>("record"),
  pickFile: $<HTMLButtonElement>("pick-file"),
  file: $<HTMLInputElement>("file"),
  engine: $<HTMLSelectElement>("engine"),
  t5Status: $<HTMLParagraphElement>("t5-status"),
  transcript: $<HTMLParagraphElement>("transcript"),
  asrMs: $<HTMLElement>("asr-ms"),
  glossMs: $<HTMLElement>("gloss-ms"),
  chips: $<HTMLDivElement>("gloss-chips"),
  glossSource: $<HTMLParagraphElement>("gloss-source"),
  history: $<HTMLOListElement>("history"),
};

function setStatus(text: string, error = false): void {
  ui.status.textContent = text;
  ui.status.classList.toggle("error", error);
}

// --- T5 worker: optional, degrades to rules_v2 if it never loads --------------------------

type T5State = "loading" | "ready" | "unavailable";
let t5State: T5State = "loading";
let t5Worker: Worker | null = null;
let t5NextId = 0;
const t5Pending = new Map<number, (gloss: string | null) => void>();

function startT5(): void {
  t5Worker = new Worker(new URL("./speech/t5.worker.ts", import.meta.url), { type: "module" });
  t5Worker.onmessage = (e: MessageEvent) => {
    const m = e.data as { type: string; id?: number; gloss?: string; loaded?: number; total?: number;
      message?: string; ms?: number };
    if (m.type === "progress" && m.total) {
      ui.t5Status.textContent = `T5 model: ${(100 * (m.loaded ?? 0) / m.total).toFixed(0)}% (${(m.total / 2 ** 20).toFixed(0)} MiB)`;
    } else if (m.type === "ready") {
      t5State = "ready";
      ui.t5Status.textContent = `T5 model: ready (${m.ms?.toFixed(0)} ms load)`;
    } else if (m.type === "refined") {
      t5Pending.get(m.id!)?.(m.gloss ?? null);
      t5Pending.delete(m.id!);
    } else if (m.type === "error") {
      if (m.id !== undefined) { t5Pending.get(m.id)?.(null); t5Pending.delete(m.id); }
      else { t5State = "unavailable"; ui.t5Status.textContent = `T5 model: unavailable (${m.message})`; }
    }
  };
  t5Worker.postMessage({ type: "load", base: BASE });
}

function refineWithT5(english: string, ruleV1: string): Promise<string | null> {
  if (t5State !== "ready" || !t5Worker) return Promise.resolve(null);
  const id = t5NextId++;
  return new Promise((resolve) => {
    t5Pending.set(id, resolve);
    t5Worker!.postMessage({ type: "refine", id, english, ruleV1 });
  });
}

// --- gloss pipeline --------------------------------------------------------------------

interface GlossRow { english: string; rules_v1: string; rules_v2: string; t5: string | null; hybrid: HybridOutput | null }

async function glossSentence(english: string, engine: string): Promise<GlossRow> {
  const v1 = convertV1(english);
  const v2 = convertV2(english);
  if (engine === "rules_v1") return { english, rules_v1: v1, rules_v2: v2, t5: null, hybrid: null };
  if (engine === "rules_v2") return { english, rules_v1: v1, rules_v2: v2, t5: null, hybrid: null };
  const t5 = await refineWithT5(english, v1);
  const hybrid = combine(english, v1, t5, v2);
  return { english, rules_v1: v1, rules_v2: v2, t5, hybrid };
}

async function glossText(text: string, engine: string): Promise<GlossRow[]> {
  const spelled = spellNumbers(text);
  const sentences = splitSentences(spelled);
  return Promise.all(sentences.map((s) => glossSentence(s, engine)));
}

// --- rendering ---------------------------------------------------------------------------

function renderGloss(rows: GlossRow[]): void {
  ui.chips.innerHTML = "";
  const parts: string[] = [];
  for (const r of rows) {
    const gloss = r.hybrid ? r.hybrid.gloss : r.rules_v2;
    parts.push(gloss);
    for (const g of gloss.split(" ").filter(Boolean)) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = g;
      ui.chips.appendChild(chip);
    }
  }
  const sources = rows.map((r) => r.hybrid?.source ?? (ui.engine.value === "rules_v1" ? "rules_v1" : "rules_v2"));
  ui.glossSource.textContent = rows.length
    ? `source: ${sources.join(", ")}${rows.some((r) => r.hybrid?.reasons.length) ? " · guard: " + rows.flatMap((r) => r.hybrid?.reasons ?? []).join(",") : ""}`
    : "";
  return void parts;
}

function addHistory(english: string, rows: GlossRow[]): void {
  const li = document.createElement("li");
  const gloss = rows.map((r) => (r.hybrid ? r.hybrid.gloss : r.rules_v2)).join(" | ");
  li.innerHTML = `<strong>${escapeHtml(english)}</strong><br><span class="chip">${escapeHtml(gloss)}</span>`;
  ui.history.prepend(li);
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}

// --- ASR -----------------------------------------------------------------------------------

async function transcribe(rec: Recording): Promise<{ text: string; ms: number }> {
  const t0 = performance.now();
  const r = await fetch(`${BASE}api/asr`, { method: "POST", body: wavBase64(rec) });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(`ASR: ${(body as { error?: string }).error ?? r.status}`);
  }
  const body = (await r.json()) as { text: string };
  return { text: body.text, ms: performance.now() - t0 };
}

async function handleRecording(rec: Recording): Promise<void> {
  if (durationSeconds(rec) < 0.3) { setStatus("Too short — hold the button while speaking.", true); return; }
  setStatus("Transcribing…");
  try {
    const { text, ms } = await transcribe(rec);
    ui.asrMs.textContent = `${ms.toFixed(0)} ms`;
    if (!text) { setStatus("No speech detected."); return; }
    ui.transcript.textContent = text;
    ui.transcript.classList.remove("waiting");
    setStatus("Glossing…");
    const t0 = performance.now();
    const rows = await glossText(text, ui.engine.value);
    ui.glossMs.textContent = `${(performance.now() - t0).toFixed(0)} ms`;
    renderGloss(rows);
    addHistory(text, rows);
    setStatus("Ready.");
  } catch (err) {
    setStatus(String(err instanceof Error ? err.message : err), true);
  }
}

// --- wiring ---------------------------------------------------------------------------

let mic: MicRecorder | null = null;
let recording = false;

async function startRecord(): Promise<void> {
  if (recording) return;
  try {
    mic ??= new MicRecorder();
    await mic.start();
    recording = true;
    ui.record.textContent = "Recording… release to stop";
    ui.record.classList.add("error");
  } catch (err) {
    setStatus(`Microphone: ${String(err instanceof Error ? err.message : err)}`, true);
  }
}

async function stopRecord(): Promise<void> {
  if (!recording || !mic) return;
  recording = false;
  ui.record.textContent = "Hold to record";
  ui.record.classList.remove("error");
  const rec = mic.stop();
  await handleRecording(rec);
}

async function decodeFile(file: File): Promise<Recording> {
  const buf = await file.arrayBuffer();
  const ctx = new AudioContext({ sampleRate: 16000 });
  const audio = await ctx.decodeAudioData(buf);
  await ctx.close();
  return { pcm: audio.getChannelData(0), sampleRate: audio.sampleRate };
}

function wire(): void {
  ui.record.addEventListener("pointerdown", () => void startRecord());
  ui.record.addEventListener("pointerup", () => void stopRecord());
  ui.record.addEventListener("pointerleave", () => { if (recording) void stopRecord(); });
  ui.pickFile.addEventListener("click", () => ui.file.click());
  ui.file.addEventListener("change", () => {
    const file = ui.file.files?.[0];
    if (!file) return;
    setStatus(`Decoding ${file.name}…`);
    decodeFile(file).then((rec) => handleRecording(rec)).catch((err) => setStatus(String(err), true));
  });
}

async function main(): Promise<void> {
  wire();
  startT5();
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("No microphone in this browser — pick an audio file instead.");
  } else {
    ui.record.disabled = false;
    setStatus("Ready.");
  }
  const health = await fetch(`${BASE}api/health`).then((r) => r.json()).catch(() => null) as
    { asr?: boolean } | null;
  if (health && !health.asr) setStatus("ASR is offline (no Cloudflare AI binding) — pick a Workers deploy with credentials.", true);
}

void main();
