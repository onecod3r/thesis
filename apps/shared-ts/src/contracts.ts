/**
 * The files `apps/web/tools/export.py` writes and the app reads. Python is the
 * source of every value; these types only describe the JSON.
 */

/** `model/manifest.json`, written by `sb.recognize.export.step.export_web`. */
export interface StepManifest {
  format: "signbridge-web-step/1";
  run_id: number;
  architecture: string;
  landmarks: number[];
  coords: "xy" | "xyz";
  state_shape: [number, number];
  embed_dim: number;
  cos_scale: number;
  glosses: string[];
  null_index: number;
  class_mask: boolean[];
  decoder: DecoderSettings;
  sha256: Record<string, string>;
}

export interface DecoderSettings {
  name: string;
  collapsed: boolean;
  nu: number;
  min_len: number;
}

export type RuleConfig =
  | { kind: "lattice"; theta: number; lam: number; k: number; lag: number; max_len: number | null }
  | { kind: "rule"; mode: "none" | "rescore" | "agree"; lam: number; theta: number; k: number;
      theta_lo: number; theta_hi: number; max_len: number | null };

/** `pipeline.json`: `apps/web/pipeline.config.json` + D3 settings + hashes. */
export interface PipelineConfig {
  run: string;
  corpus_version: string;
  prior: { order: number; discount: number; sha256: string };
  rule: RuleConfig;
  collapse: boolean;
  sentence_end_null_frames: number;
  target_fps: number;
  display: { uncertain_below: number };
  speech: { auto_speak_min_confidence: number };
  holistic_model_url: string;
  decoder: DecoderSettings;
  model: { run_id: number; sha256: Record<string, string> };
}

/** `prior.json`: `sb.rescore.prior.NgramLM.to_dict()`. */
export interface NgramJson {
  format: "kn-ngram/1";
  order: number;
  discount: number;
  vocab: string[];
  tables: Record<string, Record<string, Record<string, number>>>;
}

export type Lexicon = Record<string, string[]>;

/** `replay/index.json`: held-out streams with Python's expected output. */
export interface ReplayIndex {
  landmarks: number[];
  coords: "xy";
  streams: ReplayStream[];
}

export interface ReplayStream {
  seq_id: string;
  file: string;
  frames: number;
  signer: number;
  sentence_id: string;
  signed: string[];
  english_signed: string;
  expected: { gloss: string; frame: number; conf: number; decided_at: number }[];
  expected_english: string;
}
