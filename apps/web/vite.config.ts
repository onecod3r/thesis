import { resolve } from "node:path";
import { defineConfig } from "vite";

const HERE = import.meta.dirname;

// The app imports ../shared-ts and packages/sb-extract-ts/src/schema.ts (the landmark
// layout port), so the dev server may read from the repo root.
export default defineConfig({
  server: { fs: { allow: ["../.."] } },
  build: {
    target: "es2022",
    assetsInlineLimit: 0,
    // Three pages: sign -> speech (index.html), speech -> gloss (speech.html, TODO §13) and the
    // landmark extraction test (landmarks.html, TODO §12.8).
    rollupOptions: { input: { main: resolve(HERE, "index.html"), speech: resolve(HERE, "speech.html"),
                              landmarks: resolve(HERE, "landmarks.html") } },
  },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["@litertjs/core", "onnxruntime-web"] },
});
