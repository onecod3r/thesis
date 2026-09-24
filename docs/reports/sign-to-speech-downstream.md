# Sign → speech after the recognizer: next-gloss prediction, gloss → English, speech

**TODO §12.6** · 2026-09-24 · status: **partial**. The stage 2 selection check, stage 3
(rules arm), stage 4 (client voices) and the end-to-end integration check are run. The
full next-gloss sweep, the LLM arm and the Workers AI TTS arm are **pending** (§7).

**Question.** C1 turns landmarks into glosses (GER 0.293, `continuous-models.md`). What
has to happen after that to *speak* a sentence, and how good is each step?

```
MediaPipe → [ recognizer (C1) + next-gloss prediction ] → gloss → English → speech
   stage 1        stage 2 (side by side, separate)          stage 3        stage 4
```

The stages stay **separate** (user decision, 2026-09-24): each one can be measured,
swapped and debugged alone. A single video → text model is future work (TODO Backlog).
Extraction and recognition run on the client (`deployment-research.md`).

## Summary

- **The whole chain runs, and the streaming path is the evaluated path.** On 300
  held-out streams, the browser's TFLite step model feeding the new frame-by-frame
  `OnlineDecoder` emits the same signs as the offline PyTorch + batch decoder on 293. The
  other 7 are float near-ties, and there are **0 unexplained** mismatches. End to end:
  **GER 0.245, 47% of sentences exactly right**. The recognizer takes 0.06 ms/frame, the
  decoder 2 µs/frame, English about 0.01 ms and speech about 0.2 s (§5).
- **A next-gloss prior helps a little, and only when blended gently.** On the selection
  signers, a held-out trigram fused as `q·p^0.3` lowers GER from **0.413 to 0.401**
  (−3%). Weighting it fully (λ=1) hurts, at 0.477 (§2).
- **The "agree" rule does not lower GER. It trades wrong signs for missing ones.** The
  rule is the user's: accept a sign only when the prediction and the recognizer agree.
  It cuts substitutions from 0.301 to 0.184, but deletions rise from 0.084 to 0.542. It
  is a **precision mode** ("say nothing rather than a wrong word"), which may still be
  the right UX choice for speech, but GER cannot show that (§2).
- **Predicting the next gloss is hard on this corpus, and counts beat a neural LM.** The
  best n-gram (trigram) ranks the true next gloss first 11–12% of the time, and in its
  top 5 30% of the time (vs 2% for a uniform guess). The GRU LM ranks worse (25% top-5). In-sample the 4-gram reaches
  65% top-5, so the corpus is small enough to memorize (§1).
- **Rule-based gloss → English is a usable offline baseline.** On 132 draft references
  it scores BLEU 60.2 / chrF 75.0 / 42% exact, vs 7.9 / 46.8 / 7% for the glosses as
  written. It keeps 92.6% of the signed content and invents 5.4%. The references are
  unreviewed Claude drafts (§3).
- **Speech should default to the browser's own voices.** The Windows OS voices, which
  are what `speechSynthesis` uses, are fully intelligible to Whisper (WER 0–0.3%) in
  about 0.2 s, at no cost. On the Workers Free plan, Aura-2 costs about 120 neurons per
  sentence, so about 80 sentences a day (§4).

## 1. Stage 2a: can the next gloss be predicted?

**Predictors** (`sb.rescore.prior`, `sb.rescore.neural`): interpolated Kneser-Ney n-grams
(order 1–4, discount 0.75) and a small GRU LM. All are fitted on the GISLR-Sentences corpus:
1,757 gloss sentences, mean length 3.1, all 250 glosses, written by Claude to themed templates.

**Protocol:** the scores below are pooled over held-out folds. `sentence-fold` is 5
theme-stratified folds: an unseen sentence on a familiar topic. `theme-out` holds out one
theme of 11: an unseen topic. `in-sample` fits and scores on everything, and is shown only
to measure memorization. top-k is computed over gloss positions (end-of-sentence excluded).

| family | predictor | perplexity | top-1 | top-5 | top-10 | top-5, first gloss | top-5, later | MRR |
|---|---|---|---|---|---|---|---|---|
| sentence-fold | uniform | 251 | — | 2.0% | 4.0% | — | — | — |
| sentence-fold | unigram | 104.2 | 2.8% | 9.5% | 15.1% | 13.7% | 7.6% | 0.075 |
| sentence-fold | bigram | 52.8 | 10.3% | 28.9% | 39.2% | 16.4% | 34.7% | 0.196 |
| sentence-fold | **trigram** | 52.1 | **11.6%** | **30.0%** | 39.2% | 16.4% | 36.4% | **0.207** |
| sentence-fold | 4-gram | 54.4 | 11.4% | 30.0% | 39.1% | 16.4% | 36.3% | 0.206 |
| sentence-fold | GRU LM | **51.3** | 10.4% | 25.3% | 34.0% | 16.3% | 29.5% | 0.185 |
| theme-out | trigram | 74.5 | 11.8% | 27.5% | 35.6% | 15.1% | 33.2% | 0.197 |
| theme-out | GRU LM | 89.1 | 9.5% | 23.0% | 30.1% | 15.6% | 26.4% | 0.167 |
| in-sample | 4-gram | 8.9 | 40.9% | 64.6% | 71.8% | 17.2% | 86.7% | 0.517 |

The uniform row is exact: 1/250 per gloss. The notebook prints it as 1.0 because every
gloss ties.

- **Order 3 is enough.** The 4-gram adds nothing held out, because sentences are only 3.1
  glosses long.
- **The first gloss is nearly unpredictable** (top-5 16%): with no history, only
  sentence-initial frequency helps. Later positions reach 36% top-5.
- **Unseen topics cost little** (27.5% vs 30.0% top-5). The model learns function-word
  patterns (`minemy X`, `who have/that/finish`) more than topic words.
- **The GRU LM loses where it matters** (pooled over folds; from the user's run of the
  notebook, 2026-09-24). Its perplexity is the lowest on unseen sentences (51.3 vs 52.1),
  so its probabilities are smoother, but it **ranks the true next gloss worse** (top-5 25.3%
  vs 30.0%, MRR 0.185 vs 0.207). It also degrades more on unseen topics (perplexity 89 vs
  74). Fusion uses the ranking and the shape near the top, so **the n-gram ships**. It is
  also a few KB of tables instead of a model the client would have to run.

**What the predictor says** (full-corpus trigram; `</s>` = end of sentence):

| history | top next glosses | what the corpus actually continues with |
|---|---|---|
| `who` | have 0.24 · that 0.14 · finish 0.11 · `</s>` 0.08 · like 0.05 | have ×8, that ×5, finish ×4, like ×2 |
| `dog` | `</s>` 0.13 · hate 0.07 · bad 0.05 · go 0.05 · fine 0.04 | hate ×3, go/bad/fine ×2 |
| `yesterday grandma` | give 0.28 · `</s>` 0.15 · callonphone 0.06 · home 0.03 | give ×1 |
| `minemy give` | child 0.10 · minemy 0.10 · gift 0.10 · milk 0.06 · `</s>` 0.04 | **never occurs**. The model backs off to what follows `give` anywhere: child ×4, minemy ×4, gift ×3, milk ×2 |

The user's example shows the corpus's size limit: no sentence contains `minemy give`, so
the trigram answers from the bigram `give → ·`. A larger or real sentence corpus is what
would make the prior sharper (TODO §8/§12.6).

## 2. Stage 2b: fusing the prediction with the recognizer

`sb.recognize.continuous.fuse` cuts the stream into D3 segments (ν=0.7, min_len 4, C1's
chosen setting). Each segment gets a recognizer vote `q` over the glosses, and a rule
decides with the prior `p(· | accepted glosses so far)`:

| rule | accepts |
|---|---|
| `none` | `argmax q` (θ floor optional). With no floor and no gate this is exactly D3, verified on all 5,054 evaluation streams: GER 0.29338 both ways |
| `rescore` | `argmax(q · p^λ)`: shallow fusion |
| `agree` | the fused top gloss if it is in the prior's top-k **and** `q ≥ θ_lo`. `q ≥ θ_hi` accepts `argmax q` without the prior. Otherwise reject |

**Selection-signer check** (1,562 clean sentence streams, trigram fitted without each
stream's own sentence). This is a legitimate tuning set, and the numbers below are not
evaluation-signer results:

![GER split into substitutions, deletions and insertions for each rule](assets/sign-to-speech-downstream/fusion_selection_check.png)

| rule | GER | substitutions | deletions | insertions |
|---|---|---|---|---|
| recognizer alone (D3) | 0.413 | 0.301 | 0.084 | 0.029 |
| rescore λ=0.1 | 0.408 | 0.296 | 0.083 | 0.029 |
| **rescore λ=0.3** | **0.401** | 0.289 | 0.080 | 0.031 |
| rescore λ=0.5 | 0.406 | 0.292 | 0.079 | 0.035 |
| rescore λ=1.0 | 0.477 | 0.355 | 0.076 | 0.046 |
| agree k=50, θ_hi=0.3 | 0.455 | 0.301 | 0.125 | 0.030 |
| agree k=10, θ_hi=0.3 | 0.470 | 0.257 | 0.191 | 0.021 |
| agree k=10, θ_hi=0.45 | 0.533 | 0.226 | 0.294 | 0.014 |
| agree k=10, θ_hi=0.75 | 0.730 | 0.184 | 0.542 | 0.004 |

- **Gentle fusion is a small, real gain.** Substitutions fall 0.301 → 0.289, since
  the prior breaks ties between visually similar glosses. A strong prior overrides the
  recognizer and adds substitutions.
- **Why `agree` deletes so much:** the true next gloss is in the prior's top 10 only
  about 39% of the time (§1). So "only accept what the prior expects" rejects most correct
  signs whose recognizer confidence is below θ_hi. It never makes a GER win, but it is the
  most *precise* setting: at θ_hi=0.75 it emits almost no wrong or extra signs (sub 0.18,
  ins 0.004). Whether "silent rather than wrong" is better for a user is a product
  question, not a GER one.
- The first demo default (`agree`, k=10, θ_hi=0.75, picked before this check) gave GER
  0.598 on held-out streams, which is what prompted it. The full sweep's grid now reaches
  θ_hi 0.2 and k 50.

**Noise rejection** is built and its streams are cached, but it is not yet scored. The
evaluation signers' noisy sentence streams hold 2,410 `fidget` (median 94 frames), 2,600
`hold` (96) and 2,535 `reverse` (21) blocks; the selection signers 797/715/778. `sb.recognize.sequences.noise`
inserts `fidget`, `hold` and `reverse` blocks, and a `max_len` gate rejects long
segments. The full sweep reports the noise false-accept rate per kind and what the gate
costs in real signs (pending, §7).

## 3. Stage 3: gloss → English

`sb.rescore.gloss2en` reverses `sb.synthesize.gloss.rules_v2`, restoring articles, the
copula, tense (`yesterday`/`finish` → past, `will`/`tomorrow` → future), `do`-support
and pronoun case. It is driven by the corpus lexicon's part-of-speech tags. `hesheit`
becomes "they", because GISLR has one sign for he/she/it. It is deterministic and runs
anywhere, the client included.

**Scores** (`gislr.4.downstream.gloss-to-english.ipynb`). References: 132 sentences (12
per theme, seeded), 1–2 English references each. Round trip: English → `rules_v2` →
GISLR labels, compared with the original glosses, on the 132 plus a 300-sentence sample.

| engine | BLEU | chrF | exact | content recall | invented content | gloss WER (round trip) |
|---|---|---|---|---|---|---|
| glosses as written (`identity`) | 7.9 | 46.8 | 6.8% | 93.5% | 2.2% | 0.152 |
| **rules_v1** | **60.2** | **75.0** | **42.4%** | 92.6% | 5.4% | 0.193 |
| Workers AI Llama 3.2-3B / 3.1-8B | pending (credentials) | | | | | |

Examples: `minemy dog hungry` → "My dog is hungry." · `where gum` → "Where is the
gum?" · `yesterday grandma give minemy gift` → "Yesterday Grandma gave my gift." (the
reference has "me a gift") · `if rain weus stay home` → "If it is raining, we stay home."
· `cut napkin scissors`-style telegraphic sentences stay awkward. That is the LLM arm's
job.

**Caveat:** the same author (Claude) wrote the references and the rules, so the BLEU is a
development number until the user reviews the references. The round trip is
reference-free, but it rewards staying close to the gloss (`identity` scores highest),
so it measures content kept, not fluency.

## 4. Stage 4: speech

`gislr.4.downstream.tts.ipynb`, on the first reference of each of the 132 sentences.
Intelligibility means Whisper `large-v3-turbo` transcribes the clip back and the WER is
taken against the text.

| engine | where | latency (median) | audio (median) | real-time factor | Whisper WER | cost |
|---|---|---|---|---|---|---|
| Windows SAPI, Zira | client (≈ `speechSynthesis`) | 0.20 s | 2.2 s | 0.09 | 0.0% | free |
| Windows SAPI, David | client | 0.20 s | 2.2 s | 0.09 | 0.3% | free |
| Workers AI MeloTTS / Aura-1 / Aura-2 | edge | pending (credentials) | | | | ~1 / ~60 / ~120 neurons per sentence |

The SAPI latency includes PowerShell start-up, which a browser does not pay. On the
**Free plan** (10,000 neurons/day), Aura-2 allows about 80 sentences a day, Aura-1 about
160, and MeloTTS thousands. **Recommendation: the browser's `speechSynthesis` is the
default voice.** A Workers AI voice is an opt-in upgrade within a daily budget.

## 5. End to end, as the client will run it

`gislr.5.pipeline.sign-to-speech.demo.ipynb`. The chain: raw landmarks (543×3, NaNs) →
**the browser export** (`export/web/model.tflite`, stepped frame by frame, state fed back)
→ class matrix → **`sb.recognize.continuous.online.OnlineDecoder`** (D3 + trigram prior +
rescore λ=0.3, collapse) → `gloss2en` → SAPI speech. A video file also works through
`sb.extract.holistic.extract_video`.

- **`OnlineDecoder` = batch decoder:** 0 mismatches in 2,400 random streams (every rule,
  the gate, collapse). A sign is decided on the first frame after its run ends, which
  is the "+1 frame" D3 latency.
- **Streaming path vs offline path, 300 held-out streams:** 293 identical, 7 near-ties
  (a null probability or top-2 margin within 1e-4 of the threshold), **0 unexplained**.
  GER 0.245 on both paths, sentence accuracy 47%.
- **Cost per frame:** recognizer 0.06 ms (TFLite, Python), decoder 2 µs. Per sentence:
  English about 0.01 ms, speech about 0.2 s. The binding costs are MediaPipe on the
  client, and the pause that marks a sentence end (`deployment-research.md` §8).

Sample outputs (held-out signers):

| signed | recognized | spoken |
|---|---|---|
| uncle ride airplane tomorrow | uncle ride airplane tomorrow | "The uncle will ride the airplane tomorrow." |
| giraffe tongue black | giraffe red black | "The giraffe is red and black." |
| if rain yourself haveto jacket | if rain yourself later jacket | "If it will rain, you later the jacket." |

The 0.245 is on a random 300-stream sample with a rule chosen on the selection signers.
It is not comparable to the 0.293 of all 5,054 streams until the full notebook reports
the same set.

## 6. Recommendations

1. **Ship `rescore` at a small λ as the default**, and offer `agree` as a user-selectable
   "precise" mode, once the full sweep confirms both on the evaluation signers and on
   noisy streams.
2. **The prior ships as n-gram tables** (`NgramLM.to_dict()`, a few KB per order). The
   GRU LM earns a place only if the full run shows it clearly ahead.
3. **Speech: the browser's `speechSynthesis` by default.** Aura only on request.
4. **gloss → English: rules offline/fallback, LLM on the edge**, pending the LLM arm's
   numbers and the reviewed references.
5. **Port `OnlineDecoder`, `fuse.decide` and the n-gram to TypeScript** with the same
   parity checks (`apps/web`, deployment step 1).

## 7. Pending

| what | where | needs |
|---|---|---|
| Full next-gloss sweep: all predictors incl. GRU, both families, D3/D1, **noise rejection**, evaluation signers, `control` split | `gislr.4.downstream.next-gloss.ipynb` | **running (user, 2026-09-24)**: §1–§3 done (predictor scores above, forward cache, noise); the §4 sweep in progress |
| LLM gloss → English (Llama 3.2-3B, 3.1-8B) | `gislr.4.downstream.gloss-to-english.ipynb` §2 | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` in `.env` |
| Workers AI TTS (MeloTTS, Aura-1, Aura-2; about 1.6k neurons) | `gislr.4.downstream.tts.ipynb` §2 | the same credentials |
| Reviewed English references | `sb.rescore` `evalset/gloss2en.v1.jsonl` | user review → v2 |
| Real signing | `apps/web` live prototype | a camera; all numbers here are on composed GISLR streams |

## Caveats

- Every stream is composed from isolated GISLR clips with synthesized transitions and
  rest (`continuous-models.md` caveats apply).
- The corpus the prior learns from was written by Claude to templates, which flatters
  every predictor. The gains here are an upper bound for real signing.
- The speech numbers use clean synthetic audio and Whisper as the listener. They show
  intelligibility, not naturalness.
