/**
 * Fetches a model file that the Worker serves in < 25 MiB parts (the static-assets file cap on
 * the Free plan), checks the SHA-256 of the whole against the manifest, and keeps it in Cache
 * Storage under that hash, so a second visit loads it from disk. An export with new weights has
 * a new hash, so a stale copy is never used.
 */

const CACHE = "signbridge-models-v1";

export interface PartFile { parts: string[]; bytes: number; sha256: string }

async function sha256Hex(buf: Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", buf as BufferSource);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function openCache(): Promise<Cache | null> {
  try {
    return await caches.open(CACHE);
  } catch {
    return null; // private mode, or no Cache Storage in this context
  }
}

export async function loadParts(base: string, f: PartFile, onBytes: (n: number) => void): Promise<Uint8Array> {
  const key = `${base}__sha256/${f.sha256}`;
  const cache = await openCache();
  const hit = await cache?.match(key);
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    if (buf.byteLength === f.bytes) {
      onBytes(buf.byteLength);
      return buf;
    }
  }
  const out = new Uint8Array(f.bytes);
  let at = 0;
  for (const p of f.parts) {
    const r = await fetch(base + p);
    if (!r.ok || !r.body) throw new Error(`${p}: HTTP ${r.status}`);
    const reader = r.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (at + value.byteLength > f.bytes) throw new Error(`${p}: more bytes than the manifest says`);
      out.set(value, at);
      at += value.byteLength;
      onBytes(value.byteLength);
    }
  }
  if (at !== f.bytes) throw new Error(`got ${at} bytes, manifest says ${f.bytes}`);
  const h = await sha256Hex(out);
  if (h !== f.sha256) throw new Error(`SHA-256 mismatch (${h.slice(0, 12)}… vs ${f.sha256.slice(0, 12)}…)`);
  await cache?.put(key, new Response(out as BodyInit)).catch(() => undefined); // quota: fine to skip
  return out;
}
