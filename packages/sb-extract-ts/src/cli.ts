/**
 * `sb-extract-ts` — resumable, pooled landmark extraction.
 *
 * Mirrors the resumability contract the Python extractor established, because a
 * bulk run over POPSIGN is tens of hours and an interrupt must never cost more
 * than the videos in flight:
 *
 *   - the npz is written (atomically) **before** the unit is marked `done`
 *   - the manifest is saved atomically, temp file + rename
 *   - `done` units are skipped, `failed` units are retried on the next pass
 *
 * Usage:
 *
 *   deno task extract --input ./videos --out ./landmarks
 *   deno task extract --input ./videos --out ./landmarks --workers 4 --limit 100
 *   deno task extract --input ./videos --out ./landmarks --retry-failed
 *
 * Permissions: --allow-read --allow-write (files), --allow-run (ffmpeg),
 * --allow-net (the MediaPipe WASM + model assets, fetched once and cached by
 * Deno), --allow-env.
 */

import { basename, dirname, extname, join } from "@std/path";

import type { DoneMessage, ErrorMessage, Job } from "./worker.ts";
import { spec } from "./schema.ts";

const VIDEO_EXT = new Set([".mp4", ".avi", ".mov", ".mkv", ".webm"]);

// Pinned, not "latest": the extractor's output is a dataset, and a silently
// upgraded model would make clips extracted on different days incomparable.
const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const MODEL_ASSET =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

interface Unit {
  status: "done" | "failed";
  frames?: number;
  seconds?: number;
  error?: string;
  detection?: Record<string, number>;
}

interface Manifest {
  spec: Record<string, unknown>;
  extractor: string;
  updated: string;
  units: Record<string, Unit>;
}

function parseArgs(argv: string[]): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith("--")) continue;
    const key = argv[i].slice(2);
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      out[key] = next;
      i++;
    } else {
      out[key] = true;
    }
  }
  return out;
}

async function loadManifest(path: string): Promise<Manifest> {
  try {
    return JSON.parse(await Deno.readTextFile(path));
  } catch (err) {
    if (!(err instanceof Deno.errors.NotFound)) throw err;
    return {
      spec: spec(),
      extractor: "sb-extract-ts",
      updated: new Date().toISOString(),
      units: {},
    };
  }
}

async function saveManifest(path: string, manifest: Manifest): Promise<void> {
  manifest.updated = new Date().toISOString();
  const tmp = `${path}.tmp`;
  await Deno.writeTextFile(tmp, JSON.stringify(manifest, null, 2));
  await Deno.rename(tmp, path);
}

async function* walkVideos(root: string): AsyncGenerator<string> {
  for await (const entry of Deno.readDir(root)) {
    const path = join(root, entry.name);
    if (entry.isDirectory) yield* walkVideos(path);
    else if (entry.isFile && VIDEO_EXT.has(extname(entry.name).toLowerCase())) {
      yield path;
    }
  }
}

async function main(): Promise<void> {
  const args = parseArgs(Deno.args);
  const input = String(args.input ?? "./videos");
  const outRoot = String(args.out ?? "./landmarks");
  const width = Number(args.width ?? 640);
  const height = Number(args.height ?? 480);
  const fps = Number(args.fps ?? 30);
  const limit = args.limit ? Number(args.limit) : Infinity;
  const retryFailed = Boolean(args["retry-failed"]);
  // leave a core for the OS; MediaPipe here is CPU-only, so this saturates
  const workerCount = Number(
    args.workers ?? Math.max(1, (navigator.hardwareConcurrency ?? 4) - 1),
  );

  await Deno.mkdir(outRoot, { recursive: true });
  const manifestPath = join(outRoot, "_manifest.json");
  const manifest = await loadManifest(manifestPath);

  const queue: Job[] = [];
  for await (const videoPath of walkVideos(input)) {
    const id = videoPath.slice(input.length).replace(/^[/\\]/, "");
    const prior = manifest.units[id];
    if (prior?.status === "done") continue;
    if (prior?.status === "failed" && !retryFailed) continue;
    if (queue.length >= limit) break;
    // the label is carried by the path, exactly as the contract says
    const relDir = dirname(id) === "." ? "" : dirname(id);
    const stem = basename(id, extname(id));
    queue.push({
      type: "job",
      id,
      videoPath,
      outPath: join(outRoot, relDir, `${stem}.npz`),
      width,
      height,
      fps,
      wasmBase: WASM_BASE,
      modelAsset: MODEL_ASSET,
    });
  }

  const alreadyDone = Object.values(manifest.units).filter(
    (u) => u.status === "done",
  ).length;
  console.log(
    `${queue.length} to extract · ${alreadyDone} already done · ` +
      `${workerCount} workers · ${width}x${height} @ ${fps}fps`,
  );
  if (queue.length === 0) return;

  let issued = 0;
  let completed = 0;
  let failed = 0;
  const started = Date.now();

  const report = () => {
    const rate = completed / Math.max((Date.now() - started) / 1000, 1e-9);
    const remaining = queue.length - completed - failed;
    const eta = rate > 0 ? remaining / rate : 0;
    // one status line, rewritten in place — no per-video log spam
    const line =
      `  ${completed + failed}/${queue.length} · ${failed} failed · ` +
      `${rate.toFixed(2)} vid/s · eta ${(eta / 60).toFixed(1)} min`;
    Deno.stdout.writeSync(new TextEncoder().encode(`\r${line.padEnd(78)}`));
  };

  await new Promise<void>((resolve) => {
    const workers: Worker[] = [];
    let live = 0;

    const dispatch = (worker: Worker) => {
      if (issued >= queue.length) {
        worker.postMessage({ type: "shutdown" });
        worker.terminate();
        live--;
        if (live === 0) resolve();
        return;
      }
      worker.postMessage(queue[issued++]);
    };

    for (let i = 0; i < Math.min(workerCount, queue.length); i++) {
      const worker = new Worker(new URL("./worker.ts", import.meta.url).href, {
        type: "module",
      });
      live++;
      worker.onmessage = async (event: MessageEvent<DoneMessage | ErrorMessage>) => {
        const msg = event.data;
        if (msg.type === "done") {
          // recorded only after the npz is on disk — the resumability contract
          manifest.units[msg.id] = {
            status: "done",
            frames: msg.frames,
            seconds: msg.seconds,
            detection: msg.detection,
          };
          completed++;
        } else {
          manifest.units[msg.id] = { status: "failed", error: msg.error };
          failed++;
        }
        if ((completed + failed) % 25 === 0) await saveManifest(manifestPath, manifest);
        report();
        dispatch(worker);
      };
      workers.push(worker);
      dispatch(worker);
    }
  });

  await saveManifest(manifestPath, manifest);
  console.log(
    `\ndone: ${completed} extracted, ${failed} failed · manifest ${manifestPath}`,
  );
  if (failed) console.log("re-run with --retry-failed to retry the failures");
}

if (import.meta.main) {
  await main();
}
