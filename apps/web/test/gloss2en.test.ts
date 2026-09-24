import { test } from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { glossToEnglish } from "../src/pipeline/gloss2en.ts";
import type { Lexicon } from "../../shared-ts/src/contracts.ts";
import { ASSETS, FIXTURES, json } from "./fixtures.ts";

test("gloss -> English matches sb.rescore.gloss2en.convert", () => {
  const lex = json<Lexicon>(join(ASSETS, "lexicon.json"));
  const cases = json<{ glosses: string[]; english: string }[]>(join(FIXTURES, "gloss2en.json"));
  const bad = cases.filter((c) => glossToEnglish(c.glosses, lex) !== c.english);
  for (const c of bad.slice(0, 15)) console.log(`  ${c.glosses.join(" ")}\n    py: ${c.english}\n    ts: ${glossToEnglish(c.glosses, lex)}`);
  assert.equal(bad.length, 0, `${bad.length} of ${cases.length} differ`);
  console.log(`  ${cases.length} gloss sequences identical`);
});
