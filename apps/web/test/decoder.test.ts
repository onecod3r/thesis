import { test } from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { OnlineDecoder } from "../src/pipeline/decoder.ts";
import type { Lattice, Rule } from "../src/pipeline/decoder.ts";
import { NgramPrior } from "../src/pipeline/prior.ts";
import type { NgramJson, StepManifest } from "../../shared-ts/src/contracts.ts";
import { ASSETS, FIXTURES, f32, json } from "./fixtures.ts";

interface Fixture {
  n_classes: number; null_index: number; nu: number; min_len: number;
  rules: Record<string, Record<string, unknown>>;
  streams: { variant: string; row: number; offset: number; frames: number; expected: Record<string, number[][]> }[];
}

function toRule(r: Record<string, unknown>): Rule | Lattice {
  if (r.kind === "lattice") {
    return { kind: "lattice", theta: r.theta as number, lam: r.lam as number, k: r.k as number, lag: r.lag as number,
             max_len: (r.max_len as number | null) ?? null };
  }
  return { kind: "rule", mode: r.mode as Rule["mode"], lam: r.lam as number, theta: r.theta as number, k: r.k as number,
           theta_lo: r.theta_lo as number, theta_hi: r.theta_hi as number, max_len: (r.max_len as number | null) ?? null };
}

test("OnlineDecoder matches Python's on cached C1 outputs, every rule", () => {
  const fx = json<Fixture>(join(FIXTURES, "decoder.json"));
  const probs = f32(join(FIXTURES, "decoder_probs.f32"));
  const manifest = json<StepManifest>(join(ASSETS, "model", "manifest.json"));
  const prior = new NgramPrior(json<NgramJson>(join(ASSETS, "prior.json")), manifest.glosses).prior;
  const C = fx.n_classes;
  for (const [name, raw] of Object.entries(fx.rules)) {
    let mismatches = 0, emissions = 0;
    for (const st of fx.streams) {
      const dec = new OnlineDecoder(fx.null_index, fx.nu, fx.min_len, toRule(raw), prior, true);
      const got: number[][] = [];
      for (let t = 0; t < st.frames; t++) {
        const p = probs.subarray((st.offset + t) * C, (st.offset + t + 1) * C);
        const e = dec.step(p, t);
        if (e) got.push([e.cls, e.frame, t]);
      }
      for (const e of dec.flushAll(st.frames)) got.push([e.cls, e.frame, st.frames]);
      const want = st.expected[name].map(([c, f, , at]) => [c, f, at]);
      emissions += want.length;
      if (JSON.stringify(got) !== JSON.stringify(want)) {
        mismatches++;
        if (mismatches <= 3) console.log(`  ${name} ${st.variant} row ${st.row}\n    py ${JSON.stringify(want)}\n    ts ${JSON.stringify(got)}`);
      }
    }
    console.log(`  ${name}: ${fx.streams.length - mismatches}/${fx.streams.length} streams identical (${emissions} emissions)`);
    assert.equal(mismatches, 0, `${name}: ${mismatches} streams differ`);
  }
});
