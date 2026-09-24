/** Loads what `tools/export.py fixtures` / `assets` wrote (gitignored; run the tool first). */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const FIXTURES = join(HERE, "fixtures");
export const ASSETS = join(HERE, "..", "public", "assets");

export function json<T>(path: string): T {
  if (!existsSync(path)) throw new Error(`${path} missing: run .venv/Scripts/python.exe apps/web/tools/export.py`);
  return JSON.parse(readFileSync(path, "utf-8")) as T;
}

export function f32(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
