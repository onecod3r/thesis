# Speech → ASL gloss → sign video pipeline: audit and integration plan

**Date:** 2026-09-24 · **Subject:** `chosen_merged_asl_pipeline_hybrid_1.ipynb` (the merged
Maimuna/Raiyan Colab notebook, 12 cells) · **TODO:** §13

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
