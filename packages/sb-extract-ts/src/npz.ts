/**
 * A minimal `.npz` writer — enough of NPY 1.0 and ZIP to produce files numpy reads.
 *
 * **Why not JSON.** The stage-1 artifact has a contract (`schema.ts`): float16
 * `(T, 543, 3)` with `fps` and `num_frames`, read by `sb.core.io.read_landmark_npz`
 * and by every downstream consumer. Emitting JSON instead would be a second format
 * for the same thing, roughly 10x the bytes, and would need a conversion step that
 * could itself drift. ~120 lines of well-specified format is the cheaper option.
 *
 * Entries are STORED (uncompressed). np.savez_compressed uses deflate, but numpy
 * reads either, and stored keeps this file free of a compression dependency —
 * float16 landmark data is mostly NaN runs and compresses well, so the size
 * difference is worth measuring before adding one.
 */

/** IEEE-754 binary32 → binary16, round-toward-zero on the mantissa.
 *
 * NaN must survive as NaN (it is the missing-landmark marker) and Infinity must
 * survive as Infinity (so the validator can still reject it downstream).
 */
export function toFloat16(value: number): number {
  if (Number.isNaN(value)) return 0x7e00; // quiet NaN
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  if (value === 0) return Object.is(value, -0) ? 0x8000 : 0x0000;

  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const bits = u32[0];

  const sign = (bits >>> 16) & 0x8000;
  let exp = ((bits >>> 23) & 0xff) - 127 + 15;
  const mantissa = bits & 0x7fffff;

  if (exp >= 0x1f) return sign | 0x7c00; // overflow -> Infinity
  if (exp <= 0) {
    if (exp < -10) return sign; // underflow -> signed zero
    // subnormal: shift the implicit leading 1 back in
    const sub = (mantissa | 0x800000) >>> (1 - exp + 13);
    return sign | sub;
  }
  return sign | (exp << 10) | (mantissa >>> 13);
}

export type NpyDType = "<f2" | "<f4" | "<i4";

/** One NPY 1.0 buffer: magic, header, then the raw little-endian payload. */
export function encodeNpy(
  data: Float32Array | Int32Array,
  shape: readonly number[],
  dtype: NpyDType,
): Uint8Array {
  const dict =
    `{'descr': '${dtype}', 'fortran_order': False, 'shape': (${
      shape.map((n) => `${n},`).join(" ")
    }), }`;
  // magic(6) + version(2) + headerLen(2) + header must be a multiple of 64
  const prefix = 10;
  const padded = Math.ceil((prefix + dict.length + 1) / 64) * 64;
  const header = dict + " ".repeat(padded - prefix - dict.length - 1) + "\n";

  let payload: Uint8Array;
  if (dtype === "<f2") {
    const out = new Uint16Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = toFloat16(data[i]);
    payload = new Uint8Array(out.buffer);
  } else if (dtype === "<f4") {
    payload = new Uint8Array(Float32Array.from(data).buffer);
  } else {
    payload = new Uint8Array(Int32Array.from(data).buffer);
  }

  const buf = new Uint8Array(prefix + header.length + payload.length);
  const view = new DataView(buf.buffer);
  buf.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59], 0); // \x93NUMPY
  buf[6] = 1; // major
  buf[7] = 0; // minor
  view.setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) buf[prefix + i] = header.charCodeAt(i);
  buf.set(payload, prefix + header.length);
  return buf;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

/** Bundle named NPY buffers into a STORED zip — i.e. an `.npz`. */
export function encodeNpz(entries: Record<string, Uint8Array>): Uint8Array {
  const enc = new TextEncoder();
  const locals: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  for (const [rawName, payload] of Object.entries(entries)) {
    const name = enc.encode(rawName.endsWith(".npy") ? rawName : `${rawName}.npy`);
    const crc = crc32(payload);

    const local = new Uint8Array(30 + name.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true); // local file header
    lv.setUint16(4, 20, true); // version needed
    lv.setUint16(6, 0, true); // flags
    lv.setUint16(8, 0, true); // method: stored
    lv.setUint16(10, 0, true); // mod time
    lv.setUint16(12, 0x21, true); // mod date (1980-01-01, deterministic)
    lv.setUint32(14, crc, true);
    lv.setUint32(18, payload.length, true);
    lv.setUint32(22, payload.length, true);
    lv.setUint16(26, name.length, true);
    lv.setUint16(28, 0, true); // extra length
    local.set(name, 30);

    const cd = new Uint8Array(46 + name.length);
    const cv = new DataView(cd.buffer);
    cv.setUint32(0, 0x02014b50, true); // central directory header
    cv.setUint16(4, 20, true); // version made by
    cv.setUint16(6, 20, true); // version needed
    cv.setUint16(8, 0, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, 0, true);
    cv.setUint16(14, 0x21, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, payload.length, true);
    cv.setUint32(24, payload.length, true);
    cv.setUint16(28, name.length, true);
    cv.setUint32(42, offset, true); // local header offset
    cd.set(name, 46);

    locals.push(local, payload);
    central.push(cd);
    offset += local.length + payload.length;
  }

  const cdSize = central.reduce((n, b) => n + b.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, central.length, true);
  ev.setUint16(10, central.length, true);
  ev.setUint32(12, cdSize, true);
  ev.setUint32(16, offset, true);

  const parts = [...locals, ...central, eocd];
  const total = parts.reduce((n, b) => n + b.length, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

/**
 * Write one video's landmark tensor, atomically.
 *
 * Temp file + rename, because the extractor is resumable: a half-written npz that
 * a later pass counted as done is exactly the failure the manifest is meant to
 * prevent. `num_frames` is derived from the array, never passed — a caller that
 * could disagree with the data is a caller that eventually will.
 */
export async function writeLandmarkNpz(
  path: string,
  flat: Float32Array,
  nFrames: number,
  fps: number,
): Promise<void> {
  const bytes = encodeNpz({
    landmarks: encodeNpy(flat, [nFrames, 543, 3], "<f2"),
    fps: encodeNpy(Float32Array.of(fps), [], "<f4"),
    num_frames: encodeNpy(Int32Array.of(nFrames), [], "<i4"),
  });
  const tmp = `${path}.tmp.npz`;
  await Deno.writeFile(tmp, bytes);
  await Deno.rename(tmp, path);
}
