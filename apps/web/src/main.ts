/**
 * Sign -> speech in the browser (TODO §12.5 build step 1, §12.6):
 *
 *   camera / video file ─► MediaPipe Holistic ─► (543,3) frame ─► C1 step model (LiteRT.js)
 *     ─► D3 + look-ahead lattice with the trigram prior ─► glosses
 *     ─► gloss -> English (rules) ─► speechSynthesis
 *
 * "Held-out replay" skips the camera and MediaPipe: it feeds GISLR landmarks of
 * evaluation-signer streams, and compares the browser's glosses with the ones
 * Python's TFLite + OnlineDecoder produced on the same frames (tools/export.py).
 */

import type { Lexicon, NgramJson, PipelineConfig, ReplayIndex, ReplayStream } from "../../shared-ts/src/contracts.ts";
import { holisticToFrame, rowsToFrame } from "../../shared-ts/src/landmarks.ts";
import { Holistic } from "./pipeline/holistic.ts";
import type { HolisticResult } from "./pipeline/holistic.ts";
import { NgramPrior } from "./pipeline/prior.ts";
import { Recognizer } from "./pipeline/recognizer.ts";
import { Clock, glossErrors, Session } from "./pipeline/session.ts";
import type { Sentence, Sign, StepResult } from "./pipeline/session.ts";
import { onVoicesChanged, speak, voices } from "./pipeline/speech.ts";

const BASE = import.meta.env.BASE_URL;
const POSE_ROW = 489 + 11; // left shoulder: present whenever Holistic finds a body
const ASSETS = `${BASE}assets`;
const WASM = `${BASE}wasm`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  status: $<HTMLParagraphElement>("status"),
  source: $<HTMLSelectElement>("source"),
  replayStream: $<HTMLSelectElement>("replay-stream"),
  file: $<HTMLInputElement>("file"),
  start: $<HTMLButtonElement>("start"),
  stop: $<HTMLButtonElement>("stop"),
  replayAll: $<HTMLButtonElement>("replay-all"),
  mirror: $<HTMLInputElement>("mirror"),
  resample: $<HTMLInputElement>("resample"),
  skipFrames: $<HTMLInputElement>("skip-frames"),
  autoSpeak: $<HTMLInputElement>("auto-speak"),
  voice: $<HTMLSelectElement>("voice"),
  llm: $<HTMLInputElement>("llm"),
  llmWrap: $<HTMLLabelElement>("llm-wrap"),
  video: $<HTMLVideoElement>("video"),
  overlay: $<HTMLCanvasElement>("overlay"),
  replayCard: $<HTMLDivElement>("replay-card"),
  chips: $<HTMLDivElement>("chips"),
  waiting: $<HTMLParagraphElement>("waiting"),
  activity: $<HTMLDivElement>("activity"),
  fps: $<HTMLElement>("fps"),
  recMs: $<HTMLElement>("rec-ms"),
  mpMs: $<HTMLElement>("mp-ms"),
  delegate: $<HTMLElement>("delegate"),
  videoRes: $<HTMLElement>("video-res"),
  landmarks: $<HTMLElement>("landmarks"),
  bodyInView: $<HTMLElement>("body-in-view"),
  pNull: $<HTMLElement>("p-null"),
  sentences: $<HTMLOListElement>("sentences"),
  check: $<HTMLElement>("check"),
  checkSummary: $<HTMLParagraphElement>("check-summary"),
  checkRows: $<HTMLTableSectionElement>("check-rows"),
};

interface App {
  cfg: PipelineConfig;
  rec: Recognizer;
  prior: NgramPrior;
  lexicon: Lexicon;
  replay: ReplayIndex;
  holistic: Holistic | null;
}

let app: App;
let running = false;
let stopRequested = false;

function setStatus(text: string, error = false): void {
  ui.status.textContent = text;
  ui.status.classList.toggle("error", error);
}

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}. Run: .venv/Scripts/python.exe apps/web/tools/export.py assets`);
  return (await r.json()) as T;
}

// ---------------------------------------------------------------- rendering

function chip(s: Sign): HTMLSpanElement {
  const el = document.createElement("span");
  el.className = `chip${s.uncertain ? " uncertain" : ""}`;
  el.title = `confidence ${s.conf.toFixed(2)} · sign ended at tick ${s.frame}, decided at ${s.decidedAt}`;
  el.append(s.gloss);
  const small = document.createElement("small");
  small.textContent = s.conf.toFixed(2);
  el.append(small);
  return el;
}

function showLive(signs: readonly Sign[], r: StepResult | null): void {
  ui.chips.replaceChildren(...signs.map(chip));
  if (r) {
    ui.activity.style.width = `${Math.round((1 - r.pNull) * 100)}%`;
    ui.waiting.textContent = r.waiting ? `Waiting for the next sign to decide ${r.waiting} uncertain one${r.waiting > 1 ? "s" : ""}…` : "";
    ui.pNull.textContent = r.pNull.toFixed(3);
  }
}

/** Rows of a (543,3) frame with a real (non-`NaN`) `x` -- how much of the body Holistic actually found. */
function countLandmarks(frame: Float32Array): number {
  let n = 0;
  for (let row = 0; row < frame.length / 3; row++) if (!Number.isNaN(frame[row * 3])) n++;
  return n;
}

/** The Worker's LLM English, when it is deployed with a Workers AI binding and the user opted in. */
async function llmEnglish(s: Sentence): Promise<string | null> {
  if (!ui.llm.checked) return null;
  try {
    const r = await fetch(`${BASE}api/english`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ glosses: s.signs.map((x) => x.gloss) }) });
    if (!r.ok) return null;
    return ((await r.json()) as { english?: string }).english || null;
  } catch {
    return null;
  }
}

async function logSentence(s: Sentence, note = "", mute = false): Promise<void> {
  const byLlm = await llmEnglish(s);
  if (byLlm) {
    note = `English by LLM (rules: “${s.english}”). ${note}`;
    s = { ...s, english: byLlm };
  }
  const li = document.createElement("li");
  const chips = document.createElement("div");
  chips.className = "chips";
  chips.append(...s.signs.map(chip));
  const en = document.createElement("div");
  en.className = "english";
  en.textContent = s.english;
  const btn = document.createElement("button");
  btn.textContent = s.autoSpeak ? "Speak again" : "Speak anyway";
  btn.onclick = () => void speak(s.english, ui.voice.value);
  const meta = document.createElement("div");
  meta.className = "note";
  meta.textContent = (s.autoSpeak ? "" : "Not spoken automatically: some signs are unsure. ") + note;
  li.append(chips, en, btn, meta);
  ui.sentences.prepend(li);
  if (s.autoSpeak && ui.autoSpeak.checked && !mute) void speak(s.english, ui.voice.value);
}

const CONNECT_HAND = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
// Rows: face 0-467, left hand 468-488, pose 489-521, right hand 522-542.
const POSE_OFFSET = 489;
// The upper-body subset the model actually trains on (CLAUDE.md's ME-126: {11-16,23,24}) --
// shoulders, elbows, wrists, hips -- drawn as a skeleton so the live overlay shows the same
// landmarks the recognizer sees, not an arbitrary decoration.
const POSE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24];
const CONNECT_POSE: [number, number][] = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];

function drawFrame(frame: Float32Array): void {
  const c = ui.overlay, g = c.getContext("2d")!;
  const dpr = devicePixelRatio;
  const boxW = (c.width = c.clientWidth * dpr), boxH = (c.height = c.clientHeight * dpr);
  g.clearRect(0, 0, boxW, boxH);
  // `object-fit: contain` on <video> letterboxes/pillarboxes it inside the box whenever its
  // aspect ratio differs from the box's -- the canvas doesn't (its own aspect already matches
  // the box), so it must redo that same math or its points land on the black bars, not the
  // video (found 2026-09-25: a portrait phone camera in the 4:3 box shifted the whole overlay).
  const vw = ui.video.videoWidth, vh = ui.video.videoHeight;
  const scale = vw > 0 && vh > 0 ? Math.min(boxW / vw, boxH / vh) : 1;
  const w = vw > 0 ? vw * scale : boxW, h = vh > 0 ? vh * scale : boxH;
  const offX = (boxW - w) / 2, offY = (boxH - h) / 2;
  const pt = (row: number): [number, number] | null => {
    const x = frame[row * 3], y = frame[row * 3 + 1];
    return Number.isNaN(x) || Number.isNaN(y) ? null : [offX + x * w, offY + y * h];
  };
  const dot = (p: [number, number] | null, r: number) => {
    if (!p) return;
    g.beginPath();
    g.arc(p[0], p[1], r * dpr, 0, 2 * Math.PI);
    g.fill();
  };
  const line = (p: [number, number] | null, q: [number, number] | null) => {
    if (!p || !q) return;
    g.beginPath();
    g.moveTo(...p);
    g.lineTo(...q);
    g.stroke();
  };

  // Face mesh: faint, so 468 points don't drown out the hands and pose that matter for signing.
  g.fillStyle = "rgba(200, 214, 229, 0.35)";
  for (let r = 0; r < 468; r++) dot(pt(r), 1);

  // Upper-body pose skeleton, bright and unmissable.
  g.strokeStyle = g.fillStyle = "#feca57";
  g.lineWidth = 3 * dpr;
  for (const [a, b] of CONNECT_POSE) line(pt(POSE_OFFSET + a), pt(POSE_OFFSET + b));
  for (const i of POSE_JOINTS) dot(pt(POSE_OFFSET + i), 4);

  // Hands: skeleton + joint dots, one color per hand.
  g.lineWidth = 3 * dpr;
  for (const [offset, color] of [[468, "#ff9f43"], [522, "#48dbfb"]] as const) {
    g.strokeStyle = g.fillStyle = color;
    for (const [a, b] of CONNECT_HAND) line(pt(offset + a), pt(offset + b));
    for (let i = 0; i < 21; i++) dot(pt(offset + i), 3);
  }
}

/** Linear interpolation between two frames — the gap left by a skipped/repeated
 * capture, per `docs/reports/live-streaming-gap.md` (interpolating beat repeating
 * a stale frame: GER 0.504 -> 0.365 at 15 fps). `NaN` (no landmark) stays `NaN`. */
function lerpFrame(a: Float32Array, b: Float32Array, t: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
}

// ---------------------------------------------------------------- sessions

function newSession(): Session {
  return new Session(app.cfg, app.rec.glosses, app.rec.manifest.null_index, app.prior.prior, app.lexicon,
                     () => app.rec.reset());
}

async function runReplay(stream: ReplayStream, opts: { speakIt: boolean; animate: boolean }): Promise<string[]> {
  const buf = await (await fetch(`${ASSETS}/replay/${stream.file}`)).arrayBuffer();
  const xy = new Float32Array(buf);
  const rows = app.replay.landmarks;
  const session = newSession();
  session.autoEnd = false;
  app.rec.reset();
  ui.replayCard.hidden = false;
  ui.replayCard.innerHTML = `<div>Replaying <strong>${stream.seq_id}</strong> (held-out signer ${stream.signer}, ${stream.frames} frames)</div>
    <div>Signed: <strong>${stream.signed.join(" ")}</strong></div><div>${stream.english_signed}</div>`;
  const got: Sign[] = [];
  let recMs = 0;
  for (let t = 0; t < stream.frames && !stopRequested; t++) {
    const frame = rowsToFrame(xy, rows, t);
    const t0 = performance.now();
    const p = await app.rec.step(frame);
    recMs += performance.now() - t0;
    const r = session.push(p);
    got.push(...r.signs);
    if (opts.animate) {
      drawFrame(frame);
      showLive(got, r);
      await new Promise((res) => setTimeout(res, 1000 / 30));
    }
  }
  const end = session.endSentence();
  got.push(...end.tail);
  ui.recMs.textContent = `${(recMs / stream.frames).toFixed(2)} ms/frame`;
  showLive(got, null);
  if (end.sentence) {
    void logSentence(end.sentence, `Replay ${stream.seq_id}. Signed: “${stream.signed.join(" ")}”.`, !opts.speakIt);
  }
  return got.map((s) => `${s.gloss}@${s.frame}`);
}

function checkRow(stream: ReplayStream, got: string[]): boolean {
  const want = stream.expected.map((e) => `${e.gloss}@${e.frame}`);
  const same = JSON.stringify(want) === JSON.stringify(got);
  const tr = document.createElement("tr");
  const cells = [stream.seq_id, stream.signed.join(" "), want.map((x) => x.split("@")[0]).join(" "),
                 got.map((x) => x.split("@")[0]).join(" ")];
  for (const c of cells) {
    const td = document.createElement("td");
    td.textContent = c;
    tr.append(td);
  }
  const td = document.createElement("td");
  td.textContent = same ? "identical" : "differs";
  td.className = same ? "same" : "diff";
  tr.append(td);
  ui.checkRows.append(tr);
  return same;
}

async function replayAll(): Promise<void> {
  ui.check.hidden = false;
  ui.checkRows.replaceChildren();
  let same = 0, errs = 0, n = 0, done = 0;
  for (const st of app.replay.streams) {
    if (stopRequested) break;
    setStatus(`Checking ${++done}/${app.replay.streams.length}: ${st.seq_id}`);
    const got = await runReplay(st, { speakIt: false, animate: false });
    if (checkRow(st, got)) same++;
    errs += glossErrors(st.signed, got.map((x) => x.split("@")[0]));
    n += st.signed.length;
  }
  const summary = `${same}/${done} streams identical to Python (TFLite + OnlineDecoder). ` +
    `Gloss error rate against what was signed: ${(errs / Math.max(n, 1)).toFixed(3)} over ${n} signs.`;
  ui.checkSummary.textContent = summary;
  setStatus(summary);
  (window as unknown as { __replayCheck: unknown }).__replayCheck = { same, done, ger: errs / Math.max(n, 1) };
}

async function ensureHolistic(): Promise<Holistic> {
  if (app.holistic) return app.holistic;
  setStatus("Loading MediaPipe Holistic (about 14 MB, once)…");
  const asked = new URLSearchParams(location.search).get("delegate")?.toUpperCase();
  const prefer = asked === "GPU" || asked === "CPU" ? asked : (app.cfg.holistic_delegate ?? "CPU");
  app.holistic = await Holistic.load(`${WASM}/mediapipe`, [`${ASSETS}/holistic_landmarker.task`, app.cfg.holistic_model_url], prefer);
  ui.delegate.textContent = app.holistic.delegate;
  return app.holistic;
}

async function runLive(kind: "camera" | "file"): Promise<void> {
  const holistic = await ensureHolistic();
  holistic.restart();
  const video = ui.video;
  ui.replayCard.hidden = true;
  let stream: MediaStream | null = null;
  if (kind === "camera") {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
  } else {
    const f = ui.file.files?.[0];
    if (!f) throw new Error("choose a video file first");
    video.srcObject = null;
    video.src = URL.createObjectURL(f);
  }
  await video.play();
  ui.videoRes.textContent = `${video.videoWidth}x${video.videoHeight}`;
  ui.video.parentElement!.classList.toggle("mirrored", kind === "camera");
  const session = newSession();
  app.rec.reset();
  const clock = new Clock(app.cfg.target_fps);
  let signs: Sign[] = [];
  let frames = 0, fpsT0 = performance.now(), busy = false, recMs = 0, recN = 0, wasEmpty = false;
  let capturedFrames = 0, lastFed: Float32Array | null = null;
  const nullFrame = new Float32Array(app.rec.glosses.length + 1);
  nullFrame[app.rec.manifest.null_index] = 1;
  setStatus(kind === "camera" ? "Signing: pause for about 1.5 s to end a sentence." : "Playing the video…");

  await new Promise<void>((resolve) => {
    const finish = () => {
      const end = session.endSentence();
      signs.push(...end.tail);
      if (end.sentence) void logSentence(end.sentence);
      showLive([], null);
      stream?.getTracks().forEach((t) => t.stop());
      resolve();
    };
    video.onended = finish;
    const onFrame = async (mediaTime: number) => {
      if (stopRequested) {
        video.pause();
        finish();
        return;
      }
      if (!busy) {
        busy = true;
        capturedFrames++;
        // Skip MediaPipe on alternate captures (halves its cost) -- the gap this leaves in
        // the model's clock is filled by interpolating toward the next real detection below,
        // not by repeating a stale one (`docs/reports/live-streaming-gap.md`).
        if (ui.skipFrames.checked && capturedFrames % 2 === 0) {
          busy = false;
          schedule();
          return;
        }
        const t0 = performance.now();
        const res: HolisticResult = holistic.detect(video, mediaTime * 1000);
        ui.mpMs.textContent = `${(performance.now() - t0).toFixed(1)} ms/frame`;
        const frame = holisticToFrame(res, ui.mirror.checked);
        drawFrame(frame);
        ui.landmarks.textContent = `${countLandmarks(frame)}/543`;
        // Nobody in view: every GISLR training frame, rest included, has a body. An empty
        // frame is out of distribution (p_null ~ 0.01 on it), so it is a pause, not a sign.
        const nobody = Number.isNaN(frame[POSE_ROW * 3]) && Number.isNaN(frame[POSE_ROW * 3 + 3]);
        ui.bodyInView.textContent = nobody ? "no" : "yes";
        if (nobody && !wasEmpty) app.rec.reset();
        wasEmpty = nobody;
        ui.waiting.textContent = nobody ? "No one in view." : ui.waiting.textContent;
        const k = ui.resample.checked ? clock.ticks(mediaTime) : 1;
        for (let i = 0; i < k; i++) {
          const t1 = performance.now();
          // Only the last of k catch-up ticks is the real frame; earlier ones are
          // interpolated toward it from the last frame actually fed, one tick of latency
          // instead of repeating either endpoint.
          const stepFrame = nobody || !lastFed || i === k - 1 ? frame : lerpFrame(lastFed, frame, (i + 1) / k);
          const p = nobody ? nullFrame : await app.rec.step(stepFrame);
          recMs += performance.now() - t1;
          recN += nobody ? 0 : 1;
          const r = session.push(p);
          signs.push(...r.signs);
          if (r.sentence) {
            void logSentence(r.sentence);
            signs = [];
          }
          showLive(signs, r);
        }
        if (!nobody) lastFed = frame;
        frames++;
        const now = performance.now();
        if (now - fpsT0 > 1000) {
          ui.fps.textContent = `${((frames * 1000) / (now - fpsT0)).toFixed(1)}`;
          ui.recMs.textContent = `${(recMs / Math.max(recN, 1)).toFixed(2)} ms/frame`;
          frames = 0;
          fpsT0 = now;
        }
        busy = false;
      }
      schedule();
    };
    const schedule = () => {
      if (video.ended) return;
      if (typeof video.requestVideoFrameCallback === "function") {
        video.requestVideoFrameCallback((_now, meta) => void onFrame(meta.mediaTime));
      } else {
        requestAnimationFrame(() => void onFrame(video.currentTime));
      }
    };
    schedule();
  });
}

/** `?smoke-holistic`: load MediaPipe and run it on a synthetic video (no camera), for the headless check. */
async function smokeHolistic(): Promise<void> {
  const out: Record<string, unknown> = {};
  try {
    const h = await ensureHolistic();
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;
    const g = canvas.getContext("2d")!;
    g.fillStyle = "#777";
    g.fillRect(0, 0, 320, 240);
    const v = document.createElement("video");
    v.muted = true;
    v.srcObject = canvas.captureStream(30);
    await v.play();
    const res = h.detect(v, 0);
    const frame = holisticToFrame(res);
    out.ok = frame.length === 543 * 3;
    out.delegate = h.delegate;
    out.detected = Array.from(frame).filter((x) => !Number.isNaN(x)).length;
    const p = await app.rec.step(frame);
    out.pNull = p[app.rec.manifest.null_index];
  } catch (e) {
    out.error = String(e);
  }
  (window as unknown as { __holisticSmoke: unknown }).__holisticSmoke = out;
  setStatus(`Holistic smoke test: ${JSON.stringify(out)}`);
}

// ---------------------------------------------------------------- controls

function refreshVoices(): void {
  const list = voices();
  ui.voice.replaceChildren(...list.map((v) => new Option(v.name, v.name)));
}

function syncSource(): void {
  const src = ui.source.value;
  ui.replayStream.hidden = src !== "replay";
  ui.replayAll.hidden = src !== "replay";
  if (src === "file") ui.file.click();
}

async function start(): Promise<void> {
  if (running) return;
  running = true;
  stopRequested = false;
  ui.start.disabled = true;
  ui.stop.disabled = false;
  ui.replayAll.disabled = true;
  try {
    const src = ui.source.value;
    if (src === "replay") {
      const st = app.replay.streams[Number(ui.replayStream.value)];
      setStatus(`Replaying ${st.seq_id}…`);
      const got = await runReplay(st, { speakIt: true, animate: true });
      const want = st.expected.map((e) => `${e.gloss}@${e.frame}`);
      setStatus(JSON.stringify(got) === JSON.stringify(want)
        ? `Done. The browser accepted exactly what Python did on this stream.`
        : `Done. The browser's glosses differ from Python's on this stream (expected: ${st.expected.map((e) => e.gloss).join(" ")}).`);
    } else {
      await runLive(src as "camera" | "file");
      setStatus("Stopped.");
    }
  } catch (e) {
    setStatus(String(e), true);
  } finally {
    running = false;
    ui.start.disabled = false;
    ui.stop.disabled = true;
    ui.replayAll.disabled = false;
  }
}

async function main(): Promise<void> {
  const [cfg, priorJson, lexicon, replay] = await Promise.all([
    getJson<PipelineConfig>(`${ASSETS}/pipeline.json`),
    getJson<NgramJson>(`${ASSETS}/prior.json`),
    getJson<Lexicon>(`${ASSETS}/lexicon.json`),
    getJson<ReplayIndex>(`${ASSETS}/replay/index.json`),
  ]);
  setStatus("Loading the recognizer (LiteRT.js)…");
  const rec = await Recognizer.load(ASSETS, `${WASM}/litert/`);
  app = { cfg, rec, prior: new NgramPrior(priorJson, rec.glosses), lexicon, replay, holistic: null };
  $("target-fps").textContent = String(cfg.target_fps);
  $("uncertain-below").textContent = String(cfg.display.uncertain_below);
  ui.replayStream.replaceChildren(...replay.streams.map((s, i) =>
    new Option(`${s.seq_id}: ${s.signed.join(" ")}`, String(i))));
  refreshVoices();
  onVoicesChanged(refreshVoices);
  fetch(`${BASE}api/health`).then((r) => (r.ok ? r.json() : null)).then((h: { llm?: boolean } | null) => {
    ui.llmWrap.hidden = !h?.llm;
  }).catch(() => undefined);
  ui.source.onchange = syncSource;
  ui.file.onchange = () => setStatus(ui.file.files?.[0] ? `Video: ${ui.file.files[0].name}. Press Start.` : "No file chosen.");
  ui.start.onclick = () => void start();
  ui.stop.onclick = () => {
    stopRequested = true;
  };
  ui.replayAll.onclick = async () => {
    if (running) return;
    running = true;
    stopRequested = false;
    ui.start.disabled = ui.replayAll.disabled = true;
    ui.stop.disabled = false;
    try {
      await replayAll();
    } catch (e) {
      setStatus(String(e), true);
    } finally {
      running = false;
      ui.start.disabled = ui.replayAll.disabled = false;
      ui.stop.disabled = true;
    }
  };
  syncSource();
  ui.start.disabled = false;
  ui.replayAll.disabled = false;
  setStatus(`Ready. Model run ${cfg.model.run_id}, rule: ${cfg.rule.kind}${cfg.rule.kind === "lattice" ? ` (waits ${cfg.rule.lag} signs)` : ""}. ` +
    `Pick a source and press Start.`);
  const params = new URLSearchParams(location.search);
  if (params.has("check")) ui.replayAll.click();
  if (params.has("smoke-holistic")) void smokeHolistic();
}

main().catch((e) => setStatus(`Failed to start: ${String(e)}`, true));
