/**
 * Downloads the web app's model bundles from Kaggle Models into public/assets/, at the versions
 * pinned in kaggle.models.json, so the app builds without the Python export. Public; no login.
 * Checks every file the bundles' own manifests hash (the step model, the class matrix, each
 * whole T5 model).
 *
 *   npm run models              # skip files already on disk
 *   npm run models -- --force   # download everything again
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

interface Pin { model: string; framework: string; version: number; dest: string; files: string[] }
interface PartFile { parts: string[]; sha256: string }

const APP = join(dirname(fileURLToPath(import.meta.url)), "..");
const pins = JSON.parse(readFileSync(join(APP, "kaggle.models.json"), "utf-8")) as { owner: string; bundles: Record<string, Pin> };
const force = process.argv.includes("--force");

const sha256 = (bufs: Buffer[]) => bufs.reduce((h, b) => h.update(b), createHash("sha256")).digest("hex");

async function get(url: string, out: string): Promise<Buffer> {
  if (!force && existsSync(out)) return readFileSync(out);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const buf = Buffer.from(await r.arrayBuffer());
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, buf);
  return buf;
}

for (const [name, b] of Object.entries(pins.bundles)) {
  const base = `https://www.kaggle.com/api/v1/models/${pins.owner}/${b.model}/${b.framework}/${name}/${b.version}/download`;
  const dir = join(APP, "public", "assets", b.dest);
  const files = [...b.files];
  const bytes: Record<string, Buffer> = {};
  const fetchAll = async (list: string[]) => {
    for (const f of list) bytes[f] = await get(`${base}/${f}`, join(dir, f));
  };
  await fetchAll(files);

  const checks: [string, Buffer[], string][] = [];
  if (name === "t5-web") {
    const m = JSON.parse(bytes["manifest.json"].toString("utf-8")) as { files: Record<string, PartFile> };
    for (const [part, f] of Object.entries(m.files)) {
      await fetchAll(f.parts);
      checks.push([part, f.parts.map((p) => bytes[p]), f.sha256]);
    }
  } else {
    const m = JSON.parse(bytes["model/manifest.json"].toString("utf-8")) as { sha256: Record<string, string> };
    for (const [f, h] of Object.entries(m.sha256)) checks.push([f, [bytes[`model/${f}`]], h]);
  }
  for (const [what, bufs, want] of checks) {
    const got = sha256(bufs);
    if (got !== want) throw new Error(`${name}/${what}: sha256 ${got.slice(0, 12)}… != ${want.slice(0, 12)}… (try --force)`);
  }
  const mb = Object.values(bytes).reduce((s, x) => s + x.length, 0) / 2 ** 20;
  console.log(`${name} v${b.version} -> public/assets/${b.dest}: ${Object.keys(bytes).length} files, ${mb.toFixed(0)} MB, ${checks.length} sha256 ok`);
}
