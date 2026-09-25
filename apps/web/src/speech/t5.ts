/**
 * The T5 gloss refiner in the browser (TODO §13): tokenizer (`@huggingface/tokenizers`, the
 * checkpoint's own `tokenizer.json`) -> encoder ONNX -> `beamSearch` over the decoder ONNX
 * (`sb.synthesize.gloss.export_web`). The same class runs on onnxruntime-web in the page and
 * on onnxruntime-node in `test/speech.test.ts`: the caller passes the `ort` module and the
 * model bytes.
 */

import { Tokenizer } from "@huggingface/tokenizers";
import type * as OrtTypes from "onnxruntime-common";
import { beamSearch } from "./beam.ts";
import type { GenerateOptions } from "./beam.ts";
import { buildInput } from "./text.ts";

type Ort = Pick<typeof OrtTypes, "InferenceSession" | "Tensor">;

export interface T5Manifest {
  files: Record<"encoder" | "decoder", { parts: string[]; bytes: number; sha256: string }>;
  tokenizer: { file: string; sha256: string };
  decoder_start_token_id: number;
  eos_token_id: number;
  pad_token_id: number;
  vocab_size: number;
  max_input_len: number;
  preset: string;
  generate: { max_length: number; num_beams: number; no_repeat_ngram_size?: number; repetition_penalty?: number;
    length_penalty?: number };
  guard: { min_content_recall?: number };
  weights_sha256: string | null;
  precision: "fp32" | "int8" | "mixed";
}

export class T5Refiner {
  private ort: Ort;
  private enc: OrtTypes.InferenceSession;
  private dec: OrtTypes.InferenceSession;
  private tok: Tokenizer;
  readonly manifest: T5Manifest;

  private constructor(ort: Ort, enc: OrtTypes.InferenceSession, dec: OrtTypes.InferenceSession, tok: Tokenizer,
    manifest: T5Manifest) {
    this.ort = ort;
    this.enc = enc;
    this.dec = dec;
    this.tok = tok;
    this.manifest = manifest;
  }

  static async create(ort: Ort, manifest: T5Manifest, encoder: Uint8Array | string, decoder: Uint8Array | string,
    tokenizerJson: object, opts: OrtTypes.InferenceSession.SessionOptions = {}): Promise<T5Refiner> {
    const mk = (m: Uint8Array | string) =>
      typeof m === "string" ? ort.InferenceSession.create(m, opts) : ort.InferenceSession.create(m, opts);
    const [enc, dec] = await Promise.all([mk(encoder), mk(decoder)]);
    // tokenizer_config.json only matters for special tokens, which tokenizer.json already lists.
    return new T5Refiner(ort, enc, dec, new Tokenizer(tokenizerJson, {}), manifest);
  }

  /** `tokenizer(text, max_length, truncation=True).input_ids`: `</s>` kept at the end. */
  encodeIds(text: string): number[] {
    const ids = this.tok.encode(text).ids;
    const max = this.manifest.max_input_len;
    return ids.length <= max ? ids : [...ids.slice(0, max - 1), ids[ids.length - 1]];
  }

  decodeIds(ids: number[]): string {
    return this.tok.decode(ids, { skip_special_tokens: true });
  }

  private i64(data: number[], dims: number[]): OrtTypes.Tensor {
    return new this.ort.Tensor("int64", BigInt64Array.from(data.map(BigInt)), dims);
  }

  async generateIds(inputIds: number[]): Promise<number[]> {
    const S = inputIds.length;
    const g = this.manifest.generate;
    const nb = g.num_beams;
    const encOut = await this.enc.run({ input_ids: this.i64(inputIds, [1, S]), attention_mask: this.i64(Array(S).fill(1), [1, S]) });
    const hidden = encOut.hidden.data as Float32Array;
    const D = hidden.length / S;
    const tiled = new Float32Array(nb * S * D);
    for (let b = 0; b < nb; b++) tiled.set(hidden, b * S * D);
    const hiddenT = new this.ort.Tensor("float32", tiled, [nb, S, D]);
    const maskT = this.i64(Array(nb * S).fill(1), [nb, S]);
    const opts: GenerateOptions = {
      numBeams: nb, maxLength: g.max_length, eosTokenId: this.manifest.eos_token_id,
      decoderStartTokenId: this.manifest.decoder_start_token_id, repetitionPenalty: g.repetition_penalty,
      noRepeatNgramSize: g.no_repeat_ngram_size, lengthPenalty: g.length_penalty,
    };
    return beamSearch(async (seqs) => {
      const T = seqs[0].length;
      const out = await this.dec.run({ decoder_input_ids: this.i64(seqs.flat(), [nb, T]), encoder_hidden: hiddenT,
        encoder_attention_mask: maskT });
      const logits = out.logits.data as Float32Array;
      const V = logits.length / (nb * T);
      return seqs.map((_, b) => logits.slice((b * T + T - 1) * V, (b * T + T) * V));
    }, opts);
  }

  /** `T5Refiner.refine` in Python: uppercased, stripped; an empty output falls back to the rule gloss. */
  async refine(english: string, ruleV1: string): Promise<{ gloss: string; ids: number[] }> {
    const ids = await this.generateIds(this.encodeIds(buildInput(english, ruleV1)));
    const d = this.decodeIds(ids).trim();
    return { gloss: d ? d.toUpperCase() : ruleV1, ids };
  }
}
