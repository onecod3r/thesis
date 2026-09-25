/**
 * Speech -> gloss parity with Python (fixtures: `tools/export_speech.py fixtures`).
 *
 * Exact: spell_numbers, split_sentences, the guard, T5's input ids, and beam search vs
 * `model.generate()` on the fp32 ONNX (`data/cache/synthesis/t5-web/`, onnxruntime-node).
 * Measured, not exact: the wink-nlp rule ports vs spaCy, and int8 vs PyTorch. Their rates
 * are printed and must stay above the floors below (2026-09-25: v1 218/330, v2 251/330;
 * team30 30/30 for both, a development number: the heuristics were tuned on its mismatches).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import { combine } from "../src/speech/guard.ts";
import { convertV1 } from "../src/speech/rules_v1.ts";
import { convertV2 } from "../src/speech/rules_v2.ts";
import { T5Refiner } from "../src/speech/t5.ts";
import type { T5Manifest } from "../src/speech/t5.ts";
import { buildInput, spellNumbers, splitSentences } from "../src/speech/text.ts";
import { ASSETS, FIXTURES, json } from "./fixtures.ts";

interface Row {
  set: string; english: string; rules_v1: string; rules_v2: string; input_ids: number[]; t5_ids: number[]; t5: string;
  hybrid: { gloss: string; source: string; reasons: string[] };
}
interface Fixture {
  numbers: { text: string; spelled: string }[];
  split: { text: string; sentences: string[] }[];
  sentences: Row[];
}

const fx = json<Fixture>(join(FIXTURES, "speech.json"));
const WORK = join(FIXTURES, "..", "..", "..", "..", "data", "cache", "synthesis", "t5-web");
const N_T5 = Number(process.env.SPEECH_T5_N ?? 12); // beam search on CPU fp32: a few seconds per sentence

test("spell_numbers and split_sentences are identical", () => {
  for (const c of fx.numbers) assert.equal(spellNumbers(c.text), c.spelled, c.text);
  for (const c of fx.split) assert.deepEqual(splitSentences(c.text), c.sentences, c.text);
});

test("guard (hybrid.combine) is identical on Python's rule + T5 glosses", () => {
  for (const r of fx.sentences) {
    const o = combine(r.english, r.rules_v1, r.t5, r.rules_v2);
    assert.deepEqual({ gloss: o.gloss, source: o.source, reasons: o.reasons }, r.hybrid, r.english);
  }
  const acc = fx.sentences.filter((r) => r.hybrid.source === "t5").length;
  console.log(`  ${fx.sentences.length} decisions identical (T5 accepted ${acc})`);
});

function agreement(name: string, fn: (s: string) => string, key: "rules_v1" | "rules_v2", floor: number) {
  test(`${name} (wink-nlp) vs spaCy: agreement >= ${floor}`, () => {
    const bySet = new Map<string, [number, number]>();
    const bad: Row[] = [];
    for (const r of fx.sentences) {
      const same = fn(r.english) === r[key];
      const [s, n] = bySet.get(r.set) ?? [0, 0];
      bySet.set(r.set, [s + Number(same), n + 1]);
      if (!same) bad.push(r);
    }
    for (const r of bad.slice(0, 8)) console.log(`    ${r.english}\n      py: ${r[key]}\n      ts: ${fn(r.english)}`);
    const all = fx.sentences.length - bad.length;
    for (const [set, [s, n]] of bySet) console.log(`  ${set}: ${s}/${n} identical (${(100 * s / n).toFixed(1)}%)`);
    assert.ok(all / fx.sentences.length >= floor, `${all}/${fx.sentences.length}`);
  });
}
agreement("rules_v1", convertV1, "rules_v1", Number(process.env.SPEECH_V1_FLOOR ?? 0.64));
agreement("rules_v2", convertV2, "rules_v2", Number(process.env.SPEECH_V2_FLOOR ?? 0.74));

const manifestPath = join(WORK, "manifest.json");
const haveOnnx = existsSync(join(WORK, "decoder.onnx"));

test("T5 input ids are identical (tokenizer + truncation)", { skip: !haveOnnx && "no export" }, async () => {
  const m = json<T5Manifest>(manifestPath);
  const t5 = await T5Refiner.create(ort, m, join(WORK, "encoder.onnx"), join(WORK, "decoder.onnx"),
    json<object>(join(ASSETS, "t5", "tokenizer.json")));
  for (const r of fx.sentences) assert.deepEqual(t5.encodeIds(buildInput(r.english, r.rules_v1)), r.input_ids, r.english);
});

const GRAPH_SFX = { fp32: "", int8: ".int8" } as const;
// "mixed" (int8 encoder + fp32 decoder) is what deploys by default (export_web.export's
// `ship="mixed"`): the encoder runs once per sentence, so its quantization error doesn't
// compound across beam steps the way the decoder's does -- measured 2026-09-25, 40 sentences,
// same guard-acceptance rate as full fp32 (not asserted here as strictly as fp32 itself, but
// kept as a real variant so a future regression shows up, not just a one-off benchmark).
for (const variant of ["fp32", "int8", "mixed"] as const) {
  test(`T5 beam search on ${variant} ONNX vs generate() (${N_T5} sentences)`, { skip: !haveOnnx && "no export" }, async () => {
    const m = json<T5Manifest>(manifestPath);
    const encSfx = GRAPH_SFX[variant === "mixed" ? "int8" : variant];
    const decSfx = GRAPH_SFX[variant === "mixed" ? "fp32" : variant];
    const t5 = await T5Refiner.create(ort, m, readFileSync(join(WORK, `encoder${encSfx}.onnx`)),
      readFileSync(join(WORK, `decoder${decSfx}.onnx`)), json<object>(join(ASSETS, "t5", "tokenizer.json")));
    let same = 0, sameText = 0;
    const rows = fx.sentences.slice(0, N_T5);
    for (const r of rows) {
      const out = await t5.refine(r.english, r.rules_v1);
      const ids = out.ids.filter((t) => t !== m.pad_token_id && t !== m.eos_token_id);
      const ref = r.t5_ids.filter((t) => t !== m.pad_token_id && t !== m.eos_token_id);
      same += Number(JSON.stringify(ids) === JSON.stringify(ref));
      sameText += Number(out.gloss === r.t5);
      if (out.gloss !== r.t5) console.log(`    ${r.english}\n      py: ${r.t5}\n      ${variant}: ${out.gloss}`);
    }
    console.log(`  ${variant}: ${same}/${rows.length} identical ids, ${sameText}/${rows.length} identical glosses`);
    if (variant === "fp32") assert.equal(same, rows.length);
  });
}
