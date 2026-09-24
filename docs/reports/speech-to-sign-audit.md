# Speech → ASL gloss: audit of the team pipeline, and the integration plan

**Date:** 2026-09-24 · **Subject:** `chosen_merged_asl_pipeline_hybrid_1.ipynb` (the merged
Maimuna/Raiyan Colab notebook, 12 cells) · **TODO:** §13

## Status and decisions (2026-09-24)

- **Current goal: speech → ASL gloss.** Sign rendering (WLASL clips, or a
  pose/avatar renderer, §7.2) is **set aside** by the user's decision, not dropped.
- **Google Speech Commands: dropped** (§7.1).
- **Vocabulary:** GISLR's 250 glosses are nowhere near enough for ASL output, but the
  text → gloss stage doesn't need to be limited to them (§8).
- Implementation is planned in phases in TODO §13. **Phases 1–2 and the demo are built
  (2026-09-24, §9)** from the uploaded notebook. The T5 checkpoint, its training code and
  the reference glosses are still owed by the authors (§6), so the hybrids fall back
  to rules until they arrive.

This pipeline runs in the **opposite direction** to the rest of the repo. The repo
recognizes signs and turns them into English. This notebook turns speech into signs.
It is what `packages/sb-synthesize` ("speech → sign, future") and the empty
`experiments/synthesis/` directory were reserved for.

## 1. What the notebook actually uses

| Stage | What runs | Where it comes from |
|---|---|---|
| Speech → English | `faster-whisper` **large-v3** (fp16 on GPU / int8 on CPU), `language="en"`, beam 5, Silero VAD | Downloaded (~3 GB) |
| English → gloss, step 1 | **Rule engine** (spaCy `en_core_web_sm`): negation → `NOT`, wh-word moved to the end, y/n-question `do` dropped, `will` → `FUTURE`, copula deleted, POS filter, lemmatize + uppercase | Copied "verbatim" from `rule_engine.py` (not in the notebook) |
| English → gloss, step 2 | **Fine-tuned T5** ("Hybrid"): input `"English: … Rule gloss: … Produce ASL gloss:"`, beam 4, max 64 tokens. Falls back to rule-only if the checkpoint is missing | Checkpoint on Google Drive; training code (`step7_aslg_transfer.py`, `run_inference.py`) not in the notebook |
| Gloss → video | Look up the gloss in **WLASL v0.3** (first video file that exists) → otherwise scrape **signasl.org** → otherwise **fingerspell** with WLASL letter clips (+5 hand-recorded letters C/L/X/Y/Z) | WLASL on Drive, served by a local HTTP server through Colab's proxy |
| Display | HTML/JS grid: all clips autoplay and loop, and a "dot pointer" advances when each clip ends | Colab output |
| Evaluation | METEOR, ROUGE-1/2, MiniLM cosine "semantic" score, **only when `EXPECTED_GLOSS` is set by hand** | `all-MiniLM-L6-v2` |
| Batch | 30 English sentences → gloss CSV, **no reference glosses**, so it is qualitative only | Uploaded CSV |

Imported but never used: `jiwer.wer/cer`, `sentence_bleu`. The header describes a "3-stage
evaluation framework (ASR + Gloss + Video Coverage)", but it is not in this notebook.

Recorded results: one live run ("I wanted to go outside but it was raining." →
`ME WANT OUTSIDE RAIN`; Whisper 1.28 s, gloss 408 ms), plus the 30-sentence batch.

## 2. The T5 refinement causes the worst errors

I ran the notebook's exact rule engine **on its own** (spaCy 3.8, `en_core_web_sm` 3.8.0, an
isolated environment) on the same 30 sentences and compared it with the hybrid output
the notebook recorded:

| # | English | Rule engine only | Hybrid (rules + T5) | Verdict |
|---|---|---|---|---|
| 2 | She likes coffee. | SHE LIKE COFFEE | LIKE COFFEE | T5 drops the subject |
| 12 | She did not go to work yesterday. | SHE NOT GO WORK YESTERDAY | YESTERDAY NOT WORK | T5 drops the subject and the verb (time-fronting is good) |
| 18 | Why are you leaving early? | YOU LEAVE EARLY WHY | WHY LEAVE EARLY WHY | T5 replaces YOU with a duplicate WHY |
| 19 | Did you call your mother? | YOU CALL YOUR MOTHER | **HE CALL MOTHER HE** | **meaning changed** |
| 20 | Can you help me tomorrow? | YOU HELP I TOMORROW | **HE HE HE HE HE HE HE TO TOMORROW** | **degenerate repetition** |
| 25 | Please tell her that I will come later. | TELL SHE I FUTURE COME LATER | **TELL ME FUTURE COME LATER** | **meaning changed** (her → me) |
| 11 | I do not like cold weather. | I NOT LIKE COLD WEATHER | ME DO NOT LIKE COLD WEATHER | T5 puts back the DO the rules removed |
| 21 | Please give me a glass of water. | GIVE I GLASS WATER | PLEASE GIFT ME GLASS WATER | GIVE → GIFT (wrong sign); PLEASE and ME are good |
| 4 | We bought a new car yesterday. | WE BUY NEW CAR YESTERDAY | YESTERDAY BUY NEW CAR | good time-fronting, but drops WE |
| 26 | If it rains tomorrow, I will stay home. | IT RAIN TOMORROW I FUTURE STAY HOME | IF RAIN TOMORROW ME FUTURE STAY HOME | T5 better: keeps IF |
| 29 | Before you leave, please turn off the light. | YOU LEAVE TURN LIGHT | BEFORE YOU LEAVE TURN OFF LIGHT | T5 better: keeps BEFORE, TURN-OFF |
| 1 | I am going home now. | I GO HOME NOW | ME NOW GO HOME | T5 better: ME, time-fronting |

**What each half gets right:**
- **T5:** `I/me → ME`; time words moved to the front; keeps discourse words (`IF`, `BUT`,
  `BEFORE`, `PLEASE`) and phrasal verbs (`WAKE UP`, `TURN OFF`).
- **Rule engine:** never changes who is doing what, never repeats itself, never adds
  words.

T5's failures are the dangerous kind: a fluent-looking gloss with a different meaning
(#19, #25). The overlap metrics the notebook uses (METEOR/ROUGE on a hand-set reference)
would score these partly correct.

### Rule-engine defects, each one line to fix

- `I`/`me` lemmatize to `I`; ASL gloss convention is `ME`. Possessives stay as `MY`/`YOUR`. Fine.
- The POS filter drops `SCONJ`/`CCONJ`/`INTJ`, which loses `IF`, `BUT`, `BECAUSE`, `BEFORE` and
  `PLEASE`. Conditionals and contrast change meaning.
- It drops particles (`PART`/`ADP` in phrasal verbs): `TURN OFF LIGHT` → `TURN LIGHT`, `WAKE UP` → `WAKE`.
- There is no time-word fronting (`YESTERDAY`, `TOMORROW` and `NOW` should come first in ASL).
- `tokenize` strips only the final punctuation, so commas stay attached ("tomorrow," →
  "TOMORROW,"), and it only works because spaCy re-tokenizes.
- `has_negation` misses `never`, `no`, `nobody`, `won't`, `can't`; `apply_negation_rule` maps only
  don't/doesn't/didn't. `cannot` works only by accident, through spaCy's split.
- spaCy runs twice per sentence (POS filter, then lemmatization). One parse is enough.

**Constraint:** the notebook insists that the rule engine must stay byte-identical to the
one used to make T5's training input. Any rule fix therefore means **retraining T5, or
running two rule engines**: frozen v1 as T5's input, improved v2 for rule-only and
fallback output.

## 3. Other issues

**Correctness**
- `audio_to_text` peak-normalizes the audio array, then transcribes **the file path**,
  so the normalization is dead code. `best_of=5` does nothing at `temperature=0.0`.
- `generate()` has no `no_repeat_ngram_size` or `repetition_penalty`. That is why #20
  happened.
- **WLASL clips are played untrimmed.** Each WLASL instance has `frame_start`/`frame_end`,
  decoded at 25 fps, 1-indexed, with −1 meaning the end of the video. The notebook ignores
  them, so a clip can show the signer before and after the sign. It also always picks the
  *first* existing instance: no choice of signer or clip quality, and signers change
  from word to word.
- Gloss → WLASL lookup is exact-match (plus hyphen/space variants). There are no synonyms
  or lemma fallbacks, so `GIFT` or `IMMEDIATE` either miss or hit the wrong sign. The MiniLM
  model is already loaded, but it is used only for scoring, not for nearest-gloss lookup.
- Header conflict: "Dataset 1 (NCSLGR)" vs "ASLG-PC12 transfer learning". Which
  corpus the checkpoint was trained on has to be confirmed. NCSLGR is real, linguistically
  annotated signing (1,888 utterances). ASLG-PC12 is **synthetic, rule-generated** gloss
  (82,709 training pairs on HF). Training on ASLG-PC12 would teach T5 one rule system and
  then feed it another.

**Evaluation**
- No reference glosses for the 30-sentence set, so there are no numbers, only examples.
- MiniLM cosine between two *gloss* strings is not a validated measure of meaning. MiniLM
  is an English sentence encoder, and gloss is not English.
- Dropping BLEU makes the results impossible to compare with the text-to-gloss literature
  (ASLG-PC12 / PHOENIX papers report BLEU-4 and ROUGE-L). Report both, rather than replacing
  one with the other.
- Nothing measures the stages separately: ASR WER, gloss quality, and **video coverage**
  (share of glosses that are exact WLASL matches, fuzzy matches, fingerspelled, or
  missing).

**Deployment and legal**
- It depends entirely on Colab: `drive.mount` (twice), `eval_js`, `proxyPort`, `!pip`,
  `files.upload`, `sheets`.
- **signasl.org scraping**: spoofed browser User-Agent, copyrighted videos, fragile HTML
  parsing, base64 inlined into notebook output. It can't be deployed, and it shouldn't
  appear in a thesis pipeline.
- **WLASL is C-UDA: academic/computational use only, no commercial use.** That is fine
  for the thesis, but a public Cloudflare deployment serving WLASL clips needs that
  condition checked first.
- Recording is a fixed 8-second window, not streaming.

## 4. How it fits signbridge

| Their component | Our counterpart | Integration |
|---|---|---|
| English → gloss (rules + T5) | §12.6 needs **gloss → English**, the reverse | Same model family and data. ASLG-PC12/NCSLGR pairs can train or evaluate **both** directions. One `sb` text↔gloss module, two directions. |
| Whisper ASR | §12.6 needs **TTS** (the other half of the speech interface) | Workers AI hosts `@cf/openai/whisper-large-v3-turbo` (with VAD and beam search) **and** `@cf/myshell-ai/melotts` (TTS), so one deployment covers both directions (§12.5). |
| Gloss → WLASL video | `sb-synthesize` contract: emit the **same landmark tensor** `sb.core.schema` defines | Option B below: render signs as landmark animations. |
| 250-gloss GISLR vocab vs WLASL 2000 | `sb.core.vocab` | One vocabulary mapping table: gloss ↔ WLASL ↔ GISLR ↔ fingerspell. |

**Two renderer options:**
- **A. WLASL video playback, fixed.** Trim clips, pick consistent signers, use a synonym
  and lemma lookup, drop the scraping. Fast to deliver, realistic video. License-limited,
  signer changes mid-sentence, and ~GBs of video to host.
- **B. Landmark avatar.** Run WLASL clips (or GISLR's own npz, which already exist for
  250 glosses) through `sb-extract`, draw skeleton/avatar frames in the browser, and
  blend between signs with the same interpolation §12.1's composer already uses for
  transitions. It is small, stylistically consistent, and fits the `sb-synthesize`
  contract. It also closes a loop: our recognizer can **score the synthesized output**
  (back-recognition accuracy), a free automatic metric for sign production.

Recommendation: **A first, as a baseline; B as the research contribution.**

## 5. Proposed integration (for review; nothing built yet)

Home: `packages/sb-synthesize` (already a workspace member) and `experiments/synthesis/`.

```
sb.synthesize/
  asr.py        # faster-whisper wrapper (local) — same params, fixed normalization
  gloss/
    rules_v1.py # the frozen engine, byte-identical to rule_engine.py (T5's input)
    rules_v2.py # fixed engine (ME, conjunctions, particles, time-fronting, negation)
    t5.py       # checkpoint load + generate (no_repeat_ngram_size, repetition_penalty)
    hybrid.py   # guarded hybrid: accept T5 only if it keeps the rule gloss's content
                # words + pronoun identity, else fall back to rules (fixes #19/#20/#25)
  lexicon.py    # gloss → WLASL instance (trimmed, preferred signer) / synonym / fingerspell
  metrics.py    # BLEU-4, chrF, ROUGE-L, METEOR, gloss WER; ASR WER; video coverage
```

Notebooks (repo naming convention):
- `wlasl.0.dataset.lexicon.ipynb`: WLASL via kagglehub (not Drive); index, trim, coverage
  against GISLR's 250 glosses.
- `aslg.1.models.text2gloss.ipynb`: evaluate rules-v1 / rules-v2 / T5 / guarded hybrid on
  held-out ASLG-PC12 **and** NCSLGR, plus the 30 sentences with hand-written references.
  Hyperparameters in `configs/aslg.text2gloss.json`.
- `speech.3.pipeline.demo.ipynb`: record → ASR → gloss → display, local Jupyter (no Colab
  APIs).

Checkpoint: the T5 weights go to Kaggle like every other model (`sb-sync`), not to a
Drive path. Whether text2gloss runs get `registry/` records needs a `task` field in
`meta.json` (schema v5), which I'll propose separately.

## 6. What I need from the notebook's authors

1. `rule_engine.py`, `run_inference.py`, `step7_aslg_transfer.py`, `training_metadata.json`.
2. The T5 checkpoint (`thesis_hybrid_dataset1/model/`), and the second dataset's
   notebook/model, which the header mentions.
3. Which corpus trained it: NCSLGR, ASLG-PC12, or both? And which split was held out?
4. The 30-sentence CSV. Do reference glosses exist for it?

Sources: [Workers AI whisper-large-v3-turbo](https://developers.cloudflare.com/workers-ai/models/whisper-large-v3-turbo/) ·
[Workers AI new models incl. MeloTTS](https://x.com/CloudflareDev/status/1901754156022661535) ·
[WLASL README (C-UDA, frame_start/frame_end)](https://github.com/dxli94/WLASL/blob/master/README.md) ·
[ASLG-PC12 paper](https://www.sign-lang.uni-hamburg.de/lrec/pub/12019.pdf) ·
[aslg_pc12 on Hugging Face](https://huggingface.co/datasets/achrafothman/aslg_pc12) ·
[NCSLGR download info](https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html)

## 7. Follow-up questions (2026-09-24)

### 7.1 Google Speech Commands v0.02: not useful here

It is a **keyword-spotting** set: ~1 s clips of 35 isolated words ("yes", "no", "up",
"stop", digits, a few nouns, plus background-noise files). The pipeline's ASR is Whisper,
which is open-vocabulary and already transcribes whole sentences (1.28 s in the recorded
run). Training on Speech Commands would shrink the input to 35 words, and only 12 of them
are GISLR glosses (`yes no up down on go bed bird cat dog happy tree`). It can't test
sentence ASR either, since it has no sentences.

The only narrow uses are the background-noise files, for a noise-robustness check of Whisper
(mixing noise into our own test recordings), or a tiny on-device wake-word/command mode.
Neither is on the plan. Licensing: the Kaggle mirror (`yashdogra/speech-commands`) is
labelled CC BY-NC-SA 4.0; the original Google release is CC BY 4.0, so use the original if
it is ever needed.

For ASR evaluation, what's needed is **sentence-level English speech with transcripts**:
our own recordings of the 30 test sentences, or a public read-speech set.

### 7.2 A ready-made renderer instead of per-word video clips

| Option | What it is | ASL? | Fit |
|---|---|---|---|
| **`spoken-to-signed-translation`** (sign-language-processing, **MIT**) | text → gloss → **pose lookup in a lexicon** → concatenation/smoothing → `.pose` or `.mp4`; fingerspells missing words by default; prints coverage | **No ASL lexicon shipped** (Swiss languages, DGS, BSL). The lexicon is pluggable (`--lexicon`, CSV index + pose files) | **Best fit.** Its poses are MediaPipe Holistic, the same 543-point format as GISLR, so we can build the ASL lexicon ourselves from GISLR's npz (250 glosses, many exemplars each) and later from WLASL run through `sb-extract`. This is renderer B, with the library doing the concatenation |
| CWASA / JASigning (UEA) | HTML5/WebGL 3D avatar driven by SiGML/HamNoSys notation | Needs a HamNoSys transcription **per sign**; no large ASL lexicon exists in that notation | Polished avatar, but writing 250+ transcriptions by hand is a project in itself |
| Commercial (Signapse, Hand Talk, …) | Hosted avatar/video APIs | Some ASL | Closed and paid; not a thesis contribution |

**Recommendation:** use `spoken-to-signed-translation` for gloss → pose, with **our own
ASL lexicon built from GISLR** (one exemplar per gloss, chosen by our recognizer's
confidence, so the rendered sign is one the model recognizes). Render the skeleton in the
browser, where it fits `apps/web` and the Cloudflare plan with no video hosting, and add a
3D avatar later if needed.

Why this beats the clip approach:
- no WLASL licence or scraping problem;
- one visual style instead of a new signer for every word;
- smooth transitions between signs;
- the output can be scored automatically by our recognizer.

Caveats:
- A skeleton is less readable to Deaf viewers than real video. Needs a small user check.
- GISLR phone recordings have noisy face landmarks, which matters for signs that rely on
  facial expression.
- Vocabulary is 250 glosses until WLASL landmarks are added; everything else is fingerspelled.

Sources: [spoken-to-signed-translation](https://github.com/sign-language-processing/spoken-to-signed-translation) ·
[its paper (arXiv 2305.17714)](https://arxiv.org/abs/2305.17714) ·
[CWASA](https://vh.cmp.uea.ac.uk/index.php/CWA_Signing_Avatars)

## 8. Is a 250-gloss vocabulary enough? (2026-09-24)

**No, not for output.** It also isn't the right limit to put on this stage.

**Measured on the team's 30 test sentences** (the hybrid glosses from the notebook,
counted against GISLR's 250 glosses, with `ME`/`MY` mapped to `minemy`):

| | covered by GISLR-250 |
|---|---|
| gloss tokens (154) | **47%** |
| distinct glosses (94) | **38%** (36) |
| sentences with every gloss covered | **1 of 30** |

Missing are everyday words: `YOU`, `WE`, `WHAT`, `WANT`, `NEED`, `WORK`, `COME`, `TELL`,
`CALL`, `BUY`, `DOCTOR`, `HOSPITAL`, all numbers, days of the week. A few are only naming
differences: GISLR has `dad`, `mom`, `hesheit` and `callonphone`, so an alias table
helps slightly.

**Why GISLR is small:** it was built for PopSign, a game for young children, so its
250 signs are early-childhood vocabulary (`mitten`, `fireman`, `sleepy`, `grandma`).
That makes it a sensible recognition benchmark, but not a general ASL lexicon.

**What ASL needs:**
- **Lexical signs:** research lexicons have low thousands of entries. ASL-LEX 2.0
  describes **2,723** signs; WLASL has **2,000** glosses. That is the right scale for
  everyday topics.
- **Fingerspelling** for names, places and anything else without a sign (`JOHN`, `DHAKA`).
- **Productive forms a word list can't hold:** numbers, time and date signs,
  classifiers, and verb agreement through space (`GIVE` moving from giver to receiver).

**What this means for each stage:**
- **Text → gloss (this goal): don't restrict the output.** The rule engine and T5 emit
  English-based gloss labels for any word, and fixing the output to 250 words would make
  most sentences untranslatable. Instead, add a **lexicon check** as an evaluation
  metric: the share of output glosses that are real ASL signs in ASL-LEX/WLASL, the
  share that should be fingerspelled (names, places), and the share that are English
  words with no sign. That catches output like `O'CLOCK`, `TO` or `THIS`.
- **Rendering (deferred):** here vocabulary really is a hard limit. GISLR-250 is a
  demo; a usable renderer needs a WLASL- or ASL-LEX-scale lexicon plus fingerspelling.
- **Recognition (§12):** the 250 set limits what the recognizer can hear. That limit
  is separate from, and larger than, anything in this pipeline, and §12.4/§12.7's
  add-a-sign work is the route to extending it.

Sources: [ASL-LEX 2.0 (2,723 signs)](https://academic.oup.com/jdsde/article/26/2/263/6142509) ·
[WLASL](https://github.com/dxli94/WLASL)

## 9. Built: `sb-synthesize` and the first numbers (2026-09-24)

The pipeline is now library code plus four notebooks, built from the uploaded notebook
alone. Nothing below uses the T5 checkpoint, which is still missing (§6).

| module | what it does |
|---|---|
| `asr.py` | faster-whisper with the team's settings. Peak normalization now actually reaches Whisper. `best_of` is dropped. A CUDA-12 cuBLAS shim is needed because torch here is cu130 |
| `gloss/rules_v1.py` | the team's engine, verbatim. `aslg.0` checks that it reproduces §2's recorded rule-only outputs (12/12) |
| `gloss/rules_v2.py` | every §2 defect fixed, one spaCy parse per sentence |
| `gloss/t5.py` | the team's prompt and limits. Presets: `team` (exact `generate()`) and `guarded` (`no_repeat_ngram_size` 2, `repetition_penalty` 1.3) |
| `gloss/hybrid.py` | **meaning guard**: rejects a T5 output on repetition, a change of grammatical person, a polarity flip, lost content words, or invented words; then falls back to rules_v2 |
| `metrics.py` | BLEU-4, chrF, ROUGE-L, METEOR, gloss WER (S/D/I), ASR WER |
| `lexicon.py` | GISLR + WLASL coverage with longest-match segmentation (WLASL has `NOT YET` and `WAKE UP`); trimmed WLASL clips; no scraping |
| `data.py` | ASLG-PC12 with a fixed split by unique text (6,587 texts repeat); NCSLGR from a manual export |
| `evalsets/team30.v1.json` | the 30 sentences, the team's recorded T5 output, **draft** references |

Notebooks in `experiments/synthesis/`: `aslg.0.dataset.text2gloss` (run), `aslg.1.models.text2gloss`
(run), `speech.2.asr.eval` (needs our recordings), `speech.3.pipeline.demo` (smoke-tested
with synthesized speech on GPU).

### 9.1 ASLG-PC12 test: the independent comparison

2,000 seeded sentences from the fixed test split. References have `DESC-`/`X-` stripped.
Neither engine was written against these glosses.

| engine | BLEU-4 | chrF | ROUGE-L | METEOR | gloss WER | sub | del | ins |
|---|---|---|---|---|---|---|---|---|
| rules_v1 (team) | 26.3 | 71.0 | 76.6 | 63.6 | 0.347 | 0.059 | 0.284 | 0.004 |
| **rules_v2** | **36.4** | **77.0** | **78.7** | **70.0** | **0.303** | 0.085 | 0.212 | 0.007 |

rules_v2 is better on every metric, mostly through fewer deletions: it keeps conjunctions,
particles and discourse words. Both are far from matching ASLG, because ASLG's rule system
keeps `BE`, articles-as-`DESC-` and English word order. **ASLG-PC12 measures agreement with
one synthetic convention, not ASL quality.**

**Provenance clue.** ASLG glosses are 21% `DESC-`/`X-`-marked tokens and 5% `BE`. The team's
recorded T5 output has **none** of either. Their checkpoint was not trained on raw ASLG-PC12
glosses. It was trained on NCSLGR, or on ASLG with the markers stripped (§6 question 3).

### 9.2 team30: the guard on the team's recorded T5 outputs

| engine | BLEU-4 | chrF | gloss WER | exact |
|---|---|---|---|---|
| rules_v1 | 39.3 | 70.2 | 0.428 | 0.17 |
| team T5 (recorded) | 36.0 | 72.2 | 0.414 | 0.07 |
| **guard on recorded T5** (fallback rules_v2) | 66.3 | 85.5 | 0.224 | 0.40 |
| rules_v2 | 96.7 | 98.2 | 0.020 | 0.90 |

- **The guard keeps 19 of 30 T5 outputs and rejects 11.** Reasons: grammatical person 10,
  invented word 4, repetition 3, content 1. Every meaning-changing case in §2 is rejected:
  #19 you → HE, #25 her → ME, #20 `HE HE HE…`, the dropped subjects #2/#4/#12, and GIVE →
  GIFT #21/#30. T5's wins are kept: #1 `ME NOW GO HOME`, #26 `IF RAIN TOMORROW …`,
  #29 `BEFORE YOU LEAVE TURN OFF LIGHT`, #27 `WAKE UP`.
- **On these references the team's T5 is no better than its own input** (BLEU 36.0 vs 39.3,
  WER 0.41 vs 0.43).
- **The rules_v2 row is not a test result.** Claude wrote both rules_v2 and the draft
  references, to the same conventions, so 0.90 exact is a development score. The team, and
  ideally a signer, need to review the references, and a **separate held-out set written by
  someone else** is needed before any rules_v2-vs-T5 claim. The guard's decisions don't
  depend on the references; the guarded row's score does, through the rules_v2 fallback.
- Smoke check of the T5 code path with the public `t5-small`, which has no gloss training:
  `hybrid_team` passes its German/echoed-prompt output straight through, while the guard
  rejects all of it.

### 9.3 Coverage

| set | units that are signs | WLASL | GISLR-250 | fingerspelled | sentences all signs | all GISLR |
|---|---|---|---|---|---|---|
| team30, draft references | 98% | 98% | 56% | 2% | 93% | 7% |
| team30, rules_v2 | 97% | 97% | 55% | 3% | 90% | 7% |
| team30, team T5 | 95% | 95% | 56% | 5% | 77% | 7% |
| ASLG test, rules_v2 | 61% | 60% | 13% | 39% | 6% | 0% |

With WLASL-2000 almost every team30 gloss is a real sign. GISLR's 250 still covers only 7% of
the sentences completely, which confirms §8. On Europarl text (`EUROPEAN`, `COMMISSION`,
`PARLIAMENT`) 39% would be fingerspelled: that is the domain, not the engine.

### 9.4 Still owed / next

1. From the authors: the T5 checkpoint (→ `data/external/t5-text2gloss/thesis_hybrid_dataset1/model/`),
   `training_metadata.json`, the training corpus, and review of the draft references.
2. A held-out sentence set not written by Claude, for a fair rules_v2 vs guarded-hybrid test.
3. Recordings of the 30 sentences → `speech.2.asr.eval.ipynb` (ASR WER, large-v3 vs turbo).
4. NCSLGR export (licence form) → `data/raw/ncslgr/ncslgr.csv`: the only real-signing test set here.

