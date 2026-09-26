/**
 * Landmark test page (TODO §12.8): the live camera through one of three MediaPipe extraction setups,
 * chosen with a tab bar: Holistic (what the app runs), hands + face + pose run separately, or hands +
 * face only, with an overlay that sits exactly on the body, a mirror option, and frame rate, latency,
 * confidences and features. **One mode runs at a time** (user, 2026-09-26: running all three per frame
 * biased the comparison, since they shared the frame budget and the CPU); a session table keeps each
 * mode's averages so they can still be compared side by side.
 *
 * Alignment: every panel draws the **same video frame the models just read** into its own canvas,
 * sized to the video's own resolution, then the landmarks in that canvas's pixel space. Video and
 * overlay share one transform, so they cannot drift apart, and mirroring flips both together. (The
 * sign → speech page overlays a canvas on a separate <video> and has to redo `object-fit` maths.)
 */

import { DrawingUtils, FaceLandmarker, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { bodySide, Extractor, MODE_LABEL } from "./pipeline/extractors.ts";
import type { Delegate, Detection, Mode, PoseSize } from "./pipeline/extractors.ts";

const BASE = import.meta.env.BASE_URL;
const WASM = `${BASE}wasm/mediapipe`;
const HOLISTIC_URLS = [`${BASE}assets/holistic_landmarker.task`];
const MODES: Mode[] = ["holistic", "separate", "hands_face"];
const WINDOW = 60; // frames in the rolling detection-rate window
const UPPER_BODY = [11, 12, 13, 14, 15, 16];
const HAND_COLOR = { Left: "#ff9f43", Right: "#48dbfb" } as const;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  status: $<HTMLParagraphElement>("status"),
  source: $<HTMLSelectElement>("source"),
  camera: $<HTMLSelectElement>("camera"),
  resolution: $<HTMLSelectElement>("resolution"),
  file: $<HTMLInputElement>("file"),
  start: $<HTMLButtonElement>("start"),
  stop: $<HTMLButtonElement>("stop"),
  delegate: $<HTMLSelectElement>("delegate"),
  poseSize: $<HTMLSelectElement>("pose-size"),
  faceDraw: $<HTMLSelectElement>("face-draw"),
  mirror: $<HTMLInputElement>("mirror"),
  panels: $<HTMLDivElement>("panels"),
  modes: $<HTMLDivElement>("modes"),
  summary: $<HTMLTableSectionElement>("summary"),
  video: $<HTMLVideoElement>("video"),
};

function setStatus(text: string, error = false): void {
  ui.status.textContent = text;
  ui.status.classList.toggle("error", error);
}

// ---------------------------------------------------------------- panels

interface Stats {
  frames: number;
  t0: number;
  detect: Record<string, number[]>;
  render: number[];
  e2e: number[];
  seen: { face: boolean[]; pose: boolean[]; left: boolean[]; right: boolean[] };
  agree: number;
  swapped: number;
}

/** Session totals of one mode, over every frame it ran: the comparison table. */
interface Totals {
  frames: number;
  ms: number; // wall time between this mode's consecutive frames (gaps over 1 s, e.g. a switch, skipped)
  lastT: number;
  detect: number;
  e2e: number;
  face: number;
  pose: number;
  left: number;
  right: number;
}

interface Panel {
  mode: Mode;
  totals: Totals;
  root: HTMLElement;
  canvas: HTMLCanvasElement;
  draw: DrawingUtils;
  dd: Record<string, HTMLElement>;
  badge: HTMLElement;
  extractor: Extractor | null;
  key: string;
  stats: Stats;
  last: Detection | null;
}

const ROWS: [string, string][] = [
  ["fps", "Frame rate"], ["detect", "Detection"], ["render", "Render"], ["e2e", "Camera → overlay"],
  ["features", "Detected"], ["landmarks", "Landmarks"], ["pose", "Pose visibility"], ["hands", "Hands"],
  ["side", "Hand label vs body"], ["rate", "Found (last 60 frames)"],
];

function newStats(): Stats {
  return { frames: 0, t0: performance.now(), detect: {}, render: [], e2e: [],
           seen: { face: [], pose: [], left: [], right: [] }, agree: 0, swapped: 0 };
}

function makePanel(mode: Mode): Panel {
  const root = document.createElement("section");
  root.className = "lm-panel";
  const head = document.createElement("div");
  head.className = "lm-head";
  const h = document.createElement("h2");
  h.textContent = MODE_LABEL[mode];
  const badge = document.createElement("span");
  badge.className = "lm-badge";
  badge.textContent = "off";
  head.append(h, badge);
  const canvas = document.createElement("canvas");
  canvas.className = "lm-canvas";
  canvas.width = 640;
  canvas.height = 480;
  const dl = document.createElement("dl");
  dl.className = "stats";
  const dd: Record<string, HTMLElement> = {};
  for (const [k, label] of ROWS) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const d = document.createElement("dd");
    d.textContent = "–";
    dl.append(dt, d);
    dd[k] = d;
  }
  root.append(head, canvas, dl);
  root.hidden = true;
  ui.panels.append(root);
  return { mode, root, canvas, draw: new DrawingUtils(canvas.getContext("2d")!), dd, badge, extractor: null, key: "",
           stats: newStats(), last: null,
           totals: { frames: 0, ms: 0, lastT: 0, detect: 0, e2e: 0, face: 0, pose: 0, left: 0, right: 0 } };
}

const panels = Object.fromEntries(MODES.map((m) => [m, makePanel(m)])) as Record<Mode, Panel>;
const asked = new URLSearchParams(location.search).get("mode") as Mode | null;
let active: Mode = asked && MODES.includes(asked) ? asked : "holistic";
let switching = false; // true while the newly chosen mode's models load: no frame is processed

function showMode(): void {
  for (const m of MODES) panels[m].root.hidden = m !== active;
  for (const b of ui.modes.querySelectorAll<HTMLButtonElement>("button")) {
    b.setAttribute("aria-selected", String(b.dataset.mode === active));
  }
  const url = new URL(location.href);
  url.searchParams.set("mode", active);
  history.replaceState(null, "", url);
}

async function setMode(m: Mode): Promise<void> {
  if (m === active || switching) return;
  active = m;
  showMode();
  const p = panels[m];
  p.stats = newStats();
  p.totals.lastT = 0;
  if (!running) return;
  switching = true;
  try {
    await ensureExtractor(p);
    setStatus(`Running ${MODE_LABEL[m]} at ${ui.video.videoWidth}×${ui.video.videoHeight}.`);
  } catch (e) {
    setStatus(String(e), true);
  } finally {
    switching = false;
  }
}

// ---------------------------------------------------------------- drawing

function drawPanel(p: Panel, det: Detection, video: HTMLVideoElement): void {
  const c = p.canvas, g = c.getContext("2d")!;
  const w = video.videoWidth, h = video.videoHeight;
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
  const px = Math.max(1, w / 640); // line widths scale with the resolution
  const mirror = ui.mirror.checked;
  g.save();
  if (mirror) g.setTransform(-1, 0, 0, 1, w, 0);
  g.drawImage(video, 0, 0, w, h);
  const face = det.face;
  const style = ui.faceDraw.value;
  if (face && style !== "none") {
    if (style === "points") p.draw.drawLandmarks(face, { color: "rgba(220,230,240,0.7)", radius: 0.8 * px, lineWidth: 0 });
    else {
      const conns = style === "mesh" ? FaceLandmarker.FACE_LANDMARKS_TESSELATION : FaceLandmarker.FACE_LANDMARKS_CONTOURS;
      p.draw.drawConnectors(face, conns, { color: style === "mesh" ? "rgba(220,230,240,0.35)" : "rgba(220,230,240,0.85)",
                                           lineWidth: (style === "mesh" ? 0.6 : 1.4) * px });
    }
  }
  if (det.pose) {
    p.draw.drawConnectors(det.pose, PoseLandmarker.POSE_CONNECTIONS, { color: "#feca57", lineWidth: 3 * px });
    p.draw.drawLandmarks(det.pose.slice(0, 25), { color: "#feca57", fillColor: "#3b3b3b", radius: 3 * px, lineWidth: 1.5 * px });
  }
  for (const hand of det.hands) {
    const color = HAND_COLOR[hand.label];
    p.draw.drawConnectors(hand.landmarks, HandLandmarker.HAND_CONNECTIONS, { color, lineWidth: 3 * px });
    p.draw.drawLandmarks(hand.landmarks, { color, fillColor: "#ffffff", radius: 2.5 * px, lineWidth: 1 * px });
  }
  g.restore();
  // labels are drawn unmirrored, at the mirrored position of each wrist
  g.font = `600 ${Math.round(14 * px)}px system-ui, sans-serif`;
  g.textBaseline = "bottom";
  for (const hand of det.hands) {
    const wr = hand.landmarks[0];
    const x = (mirror ? 1 - wr.x : wr.x) * w, y = wr.y * h + 22 * px;
    const side = bodySide(hand, det.pose);
    const text = `${hand.label}${hand.score !== null ? ` ${hand.score.toFixed(2)}` : ""}${side && side !== hand.label ? " ≠ body" : ""}`;
    g.lineWidth = 4 * px;
    g.strokeStyle = "rgba(0,0,0,0.7)";
    g.strokeText(text, x - 20 * px, y);
    g.fillStyle = side && side !== hand.label ? "#ff6b6b" : HAND_COLOR[hand.label];
    g.fillText(text, x - 20 * px, y);
  }
}

// ---------------------------------------------------------------- stats

const mean = (a: number[]) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN);
const push = <T>(a: T[], v: T, n = WINDOW) => {
  a.push(v);
  if (a.length > n) a.shift();
};
const pct = (a: boolean[]) => (a.length ? `${Math.round((100 * a.filter(Boolean).length) / a.length)}%` : "–");

function record(p: Panel, det: Detection, renderMs: number, e2eMs: number): void {
  const s = p.stats;
  s.frames++;
  const tt = p.totals, now = performance.now();
  if (tt.lastT && now - tt.lastT < 1000) tt.ms += now - tt.lastT;
  tt.lastT = now;
  tt.frames++;
  tt.detect += Object.values(det.timings).reduce((a, v) => a + v, 0);
  tt.e2e += e2eMs;
  tt.face += det.face ? 1 : 0;
  tt.pose += det.pose ? 1 : 0;
  tt.left += det.hands.some((x) => x.label === "Left") ? 1 : 0;
  tt.right += det.hands.some((x) => x.label === "Right") ? 1 : 0;
  for (const [k, v] of Object.entries(det.timings)) push((s.detect[k] ??= []), v, 30);
  push(s.render, renderMs, 30);
  push(s.e2e, e2eMs, 30);
  push(s.seen.face, !!det.face);
  push(s.seen.pose, !!det.pose);
  push(s.seen.left, det.hands.some((x) => x.label === "Left"));
  push(s.seen.right, det.hands.some((x) => x.label === "Right"));
  for (const hand of det.hands) {
    const side = bodySide(hand, det.pose);
    if (side) side === hand.label ? s.agree++ : s.swapped++;
  }
  p.last = det;
}

function visibility(pose: NormalizedLandmark[] | null): string {
  if (!pose) return "no body";
  const v = UPPER_BODY.map((i) => pose[i]?.visibility).filter((x): x is number => typeof x === "number");
  return v.length ? `upper body mean ${mean(v).toFixed(2)} · min ${Math.min(...v).toFixed(2)}` : "not reported";
}

function showStats(p: Panel): void {
  const s = p.stats, now = performance.now(), det = p.last;
  const fps = (s.frames * 1000) / Math.max(now - s.t0, 1);
  s.frames = 0;
  s.t0 = now;
  const parts = Object.entries(s.detect).map(([k, v]) => [k, mean(v)] as const);
  const total = parts.reduce((a, [, v]) => a + v, 0);
  p.dd.fps.textContent = `${fps.toFixed(1)} fps`;
  p.dd.detect.textContent = parts.length > 1
    ? `${total.toFixed(1)} ms (${parts.map(([k, v]) => `${k} ${v.toFixed(1)}`).join(" · ")})`
    : `${total.toFixed(1)} ms`;
  p.dd.render.textContent = `${mean(s.render).toFixed(1)} ms`;
  p.dd.e2e.textContent = Number.isFinite(mean(s.e2e)) ? `${mean(s.e2e).toFixed(0)} ms` : "–";
  if (!det) return;
  const nFace = det.face ? Math.min(det.face.length, 468) : 0;
  p.dd.features.textContent = [det.face ? `face (${det.face.length} pts)` : null, det.pose ? "pose" : null,
    ...det.hands.map((x) => `${x.label.toLowerCase()} hand`)].filter(Boolean).join(" · ") || "nothing";
  p.dd.landmarks.textContent = `${nFace + (det.pose ? 33 : 0) + 21 * det.hands.length}/543 (GISLR layout)`;
  p.dd.pose.textContent = p.mode === "hands_face" ? "no pose model" : visibility(det.pose);
  p.dd.hands.textContent = det.hands.length
    ? det.hands.map((x) => `${x.label}${x.score !== null ? ` ${x.score.toFixed(2)}` : " (slot)"}`).join(" · ")
    : "none";
  const n = s.agree + s.swapped;
  p.dd.side.textContent = p.mode === "hands_face" ? "no body to compare"
    : n ? `agree ${Math.round((100 * s.agree) / n)}% · swapped ${Math.round((100 * s.swapped) / n)}% (${n} hands)` : "no hands yet";
  p.dd.side.classList.toggle("bad", n > 20 && s.swapped / n > 0.5);
  p.dd.rate.textContent = `face ${pct(s.seen.face)} · pose ${p.mode === "hands_face" ? "–" : pct(s.seen.pose)} · ` +
    `left ${pct(s.seen.left)} · right ${pct(s.seen.right)}`;
}

function renderSummary(): void {
  const share = (n: number, d: number) => (d ? `${Math.round((100 * n) / d)}%` : "–");
  const rows = MODES.filter((m) => panels[m].totals.frames).map((m) => {
    const p = panels[m], tt = p.totals, n = p.stats.agree + p.stats.swapped;
    const cells = [MODE_LABEL[m], String(tt.frames), tt.ms ? (tt.frames / (tt.ms / 1000)).toFixed(1) : "–",
      `${(tt.detect / tt.frames).toFixed(1)} ms`, `${(tt.e2e / tt.frames).toFixed(0)} ms`, share(tt.face, tt.frames),
      m === "hands_face" ? "–" : share(tt.pose, tt.frames), share(tt.left, tt.frames), share(tt.right, tt.frames),
      m === "hands_face" ? "no body" : n ? `${share(p.stats.agree, n)} (${n} hands)` : "–"];
    const tr = document.createElement("tr");
    if (m === active) tr.className = "current";
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.append(td);
    }
    return tr;
  });
  ui.summary.replaceChildren(...rows);
}

// ---------------------------------------------------------------- run

let running = false, stopRequested = false, stream: MediaStream | null = null;

/** The mode's models, loaded once per (delegate, pose size); idle modes keep theirs but never run. */
async function ensureExtractor(p: Panel): Promise<void> {
  const delegate = ui.delegate.value as Delegate, poseSize = ui.poseSize.value as PoseSize;
  const key = `${p.mode}|${delegate}|${p.mode === "separate" ? poseSize : ""}`;
  if (p.extractor && p.key === key) {
    p.extractor.restart();
    return;
  }
  p.extractor?.close();
  p.extractor = null;
  p.badge.textContent = "loading…";
  setStatus(`Loading ${MODE_LABEL[p.mode]}…`);
  p.extractor = await Extractor.load(p.mode, WASM, delegate, poseSize, HOLISTIC_URLS);
  p.key = key;
  p.badge.textContent = p.extractor.delegate;
}

async function openSource(): Promise<void> {
  const video = ui.video;
  if (ui.source.value === "camera") {
    const [w, h] = ui.resolution.value.split("x").map(Number);
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: w }, height: { ideal: h }, ...(ui.camera.value ? { deviceId: { exact: ui.camera.value } } : {}) },
      audio: false,
    });
    video.srcObject = stream;
    await listCameras();
  } else {
    const f = ui.file.files?.[0];
    if (!f) throw new Error("choose a video file first");
    video.srcObject = null;
    video.src = URL.createObjectURL(f);
    video.loop = true;
  }
  await video.play();
}

async function start(): Promise<void> {
  if (running) return;
  running = true;
  stopRequested = false;
  ui.start.disabled = true;
  ui.stop.disabled = false;
  try {
    await ensureExtractor(panels[active]);
    await openSource();
    const video = ui.video;
    for (const m of MODES) {
      panels[m].stats = newStats();
      panels[m].totals.lastT = 0;
    }
    setStatus(`Running ${MODE_LABEL[active]} at ${video.videoWidth}×${video.videoHeight}.`);
    let lastStats = performance.now();
    await new Promise<void>((resolve) => {
      const onFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
        if (stopRequested) {
          resolve();
          return;
        }
        const t = meta.mediaTime * 1000;
        const captured = meta.captureTime ?? meta.presentationTime;
        const p = panels[active];
        if (!switching && p.extractor) {
          const det = p.extractor.detect(video, t);
          const r0 = performance.now();
          drawPanel(p, det, video);
          const r1 = performance.now();
          record(p, det, r1 - r0, r1 - captured);
        }
        const now = performance.now();
        if (now - lastStats > 500) {
          if (!switching && p.extractor) showStats(p);
          renderSummary();
          lastStats = now;
        }
        video.requestVideoFrameCallback(onFrame);
      };
      video.requestVideoFrameCallback(onFrame);
    });
  } catch (e) {
    setStatus(String(e), true);
  } finally {
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    ui.video.pause();
    running = false;
    ui.start.disabled = false;
    ui.stop.disabled = true;
    if (!ui.status.classList.contains("error")) setStatus("Stopped.");
  }
}

async function listCameras(): Promise<void> {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const current = ui.camera.value;
  ui.camera.replaceChildren(new Option("Default camera", ""),
    ...devices.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
  ui.camera.value = devices.some((d) => d.deviceId === current) ? current : "";
}

for (const b of ui.modes.querySelectorAll<HTMLButtonElement>("button")) {
  b.onclick = () => void setMode(b.dataset.mode as Mode);
}
showMode();
ui.start.onclick = () => void start();
ui.stop.onclick = () => {
  stopRequested = true;
};
ui.source.onchange = () => {
  const cam = ui.source.value === "camera";
  ui.camera.hidden = ui.resolution.hidden = !cam;
  if (!cam) ui.file.click();
};
ui.file.onchange = () => setStatus(ui.file.files?.[0] ? `Video: ${ui.file.files[0].name}. Press Start.` : "No file chosen.");
if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
  setStatus("This browser has no requestVideoFrameCallback; use a current Chrome, Edge or Safari.", true);
}
void listCameras().catch(() => undefined);
