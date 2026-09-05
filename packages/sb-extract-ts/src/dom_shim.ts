/**
 * The browser globals MediaPipe's WASM build reaches for — and the proof that
 * they are not enough.
 *
 * **Read this before spending an afternoon on it.** `@mediapipe/tasks-vision` is
 * the *web* distribution: an Emscripten build whose graph runner creates a
 * WebGL context during `_changeBinaryGraph`, before any frame is submitted.
 * `delegate: "CPU"` selects the CPU *inference* backend; it does not make the
 * graph's image pipeline GL-free. Deno ships `ImageData`, `OffscreenCanvas` and
 * `createImageBitmap`, but **no WebGL at all** — `getContext("webgl2")` returns
 * null and `WebGLRenderingContext` is undefined — so graph construction dies at
 *
 *     ReferenceError: WebGLRenderingContext is not defined
 *       at _emscripten_webgl_do_create_context
 *
 * This shim exists so that *is* the error you get. Without it the first failure
 * is `document is not defined`, which reads like a missing polyfill and invites
 * exactly the wrong conclusion. With it, the runtime gets far enough to state
 * the real requirement.
 *
 * So this file is deliberately **not** a fix, and none of it is dead code: it
 * is the difference between a misleading error and a true one. If Deno ever
 * ships WebGL, removing nothing here should make the extractor work.
 *
 * See `docs/reports/extractor-parity.md` for the measurement and the routes
 * that remain open (a headless browser, or the Python extractor).
 */

// deno-lint-ignore no-explicit-any
const g = globalThis as any;

/**
 * Make the runtime look like a browser to Emscripten's environment sniffing.
 *
 * Emscripten picks its I/O layer from globals: `typeof window == "object"` for
 * web, `typeof process.versions.node == "string"` for node. Deno satisfies the
 * *node* test, so the loader calls `require("fs")` and dies. Presenting a
 * `window` and hiding `process` for the duration of graph construction puts it
 * on the web path, where it fetches its own `.wasm`.
 *
 * `wasmBase` must be an **http(s)** base. Emscripten explicitly skips `fetch`
 * for `file://` URIs and falls back to a `readBinary` that only exists in its
 * node path, so a local mirror fails with "both async and sync fetching of the
 * wasm failed".
 */
export function installDomShim(wasmBase: string): void {
  if (g.__sbDomShim) return;
  g.__sbDomShim = true;

  g.window = globalThis;
  g.document = {
    // Emscripten derives its asset directory from the loading script's src
    currentScript: { src: `${wasmBase}/vision_wasm_internal.js` },
    createElement(tag: string) {
      if (tag === "script") {
        const listeners: Record<string, ((e?: unknown) => void)[]> = {};
        return {
          src: "",
          crossOrigin: "",
          addEventListener(type: string, fn: (e?: unknown) => void) {
            (listeners[type] ??= []).push(fn);
          },
          fire(type: string, e?: unknown) {
            for (const fn of listeners[type] ?? []) fn(e);
          },
        };
      }
      // Only reached where OffscreenCanvas is absent; Deno has one, so this is
      // a courtesy for other runtimes rather than the path taken here.
      return {
        width: 1,
        height: 1,
        getContext: () => null,
        addEventListener() {},
        removeEventListener() {},
      };
    },
    body: {
      // "loading" a script tag = fetch its source and evaluate it globally,
      // then fire `load`, which is what the loader awaits
      async appendChild(
        el: { src: string; fire: (t: string, e?: unknown) => void },
      ) {
        try {
          const res = await fetch(el.src);
          if (!res.ok) throw new Error(`${res.status} fetching ${el.src}`);
          (0, eval)(await res.text());
          el.fire("load");
        } catch (err) {
          el.fire("error", err);
        }
      },
    },
  };
}

/**
 * Run `fn` with `process` hidden, so Emscripten does not take its node branch.
 *
 * Scoped rather than permanent: `process` is hidden only across graph
 * construction, because Deno's own APIs and any npm dependency may want it
 * back immediately afterwards.
 */
export async function withoutProcess<T>(fn: () => Promise<T>): Promise<T> {
  const saved = g.process;
  g.process = undefined;
  try {
    return await fn();
  } finally {
    g.process = saved;
  }
}

/**
 * The diagnosis, as one string, for whoever catches the failure.
 *
 * Matched on the error rather than probed up front: if a future runtime does
 * provide WebGL, nothing here fires and the extractor simply works.
 */
export function explainIfWebglMissing(err: unknown): string | null {
  const msg = err instanceof Error ? err.message : String(err);
  if (!/WebGL|webgl|_emscripten_webgl/.test(msg)) return null;
  return (
    "MediaPipe's WASM graph requires a WebGL context (it creates one during " +
    "graph construction, regardless of delegate: \"CPU\"), and this runtime " +
    "has no WebGL implementation — OffscreenCanvas.getContext(\"webgl2\") " +
    "returns null. The extractor cannot run here. Use packages/sb-extract " +
    "(Python MediaPipe, native pipeline, no GL) or drive this build from a " +
    "real browser. See docs/reports/extractor-parity.md."
  );
}
