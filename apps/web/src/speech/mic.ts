/**
 * Microphone -> 16 kHz mono WAV -> base64, for `POST /api/asr` (TODO §13 Phase 4).
 * `AudioContext({sampleRate: 16000})` resamples on capture, so no resampling code is needed
 * here; the browser's own resampler is what Whisper expects (mono, 16-bit PCM).
 */

export interface Recording { pcm: Float32Array; sampleRate: number }

export class MicRecorder {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | ScriptProcessorNode | null = null;
  private chunks: Float32Array[] = [];

  async start(): Promise<void> {
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
    this.ctx = new AudioContext({ sampleRate: 16000 });
    const src = this.ctx.createMediaStreamSource(this.stream);
    // ScriptProcessorNode is deprecated but needs no separate worklet file to fetch/serve;
    // fine for short utterances (a few seconds) at this buffer size.
    const proc = this.ctx.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = (e) => this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(proc);
    proc.connect(this.ctx.destination);
    this.node = proc;
  }

  stop(): Recording {
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    const rate = this.ctx?.sampleRate ?? 16000;
    void this.ctx?.close();
    const n = this.chunks.reduce((s, c) => s + c.length, 0);
    const pcm = new Float32Array(n);
    let at = 0;
    for (const c of this.chunks) { pcm.set(c, at); at += c.length; }
    this.chunks = [];
    return { pcm, sampleRate: rate };
  }
}

function wavBytes(pcm: Float32Array, sampleRate: number): Uint8Array {
  const bytesPerSample = 2;
  const dataSize = pcm.length * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataSize);
  const v = new DataView(buf);
  const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + dataSize, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * bytesPerSample, true);
  v.setUint16(32, bytesPerSample, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, dataSize, true);
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buf);
}

export function wavBase64(rec: Recording): string {
  const bytes = wavBytes(rec.pcm, rec.sampleRate);
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return btoa(bin);
}

export const durationSeconds = (rec: Recording): number => rec.pcm.length / rec.sampleRate;
