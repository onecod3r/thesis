import { defineConfig } from "vite";

// The app imports ../shared-ts and packages/sb-extract-ts/src/schema.ts (the landmark
// layout port), so the dev server may read from the repo root.
export default defineConfig({
  server: { fs: { allow: ["../.."] } },
  build: { target: "es2022", assetsInlineLimit: 0 },
  optimizeDeps: { exclude: ["@litertjs/core"] },
});
