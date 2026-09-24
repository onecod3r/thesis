import { test } from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { NgramPrior } from "../src/pipeline/prior.ts";
import type { NgramJson, StepManifest } from "../../shared-ts/src/contracts.ts";
import { ASSETS, FIXTURES, json } from "./fixtures.ts";

test("n-gram prior matches sb.rescore.prior.NgramLM.gloss_dist", () => {
  const manifest = json<StepManifest>(join(ASSETS, "model", "manifest.json"));
  const prior = new NgramPrior(json<NgramJson>(join(ASSETS, "prior.json")), manifest.glosses);
  const cases = json<{ history: number[]; dist: number[] }[]>(join(FIXTURES, "ngram.json"));
  let worst = 0;
  for (const { history, dist } of cases) {
    const got = prior.prior(history);
    assert.equal(got.length, dist.length);
    for (let i = 0; i < dist.length; i++) worst = Math.max(worst, Math.abs(got[i] - dist[i]));
  }
  assert.ok(worst < 1e-12, `max abs diff ${worst}`);
  console.log(`  ${cases.length} histories, max abs diff ${worst.toExponential(2)}`);
});
