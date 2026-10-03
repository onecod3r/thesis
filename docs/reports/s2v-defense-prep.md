# S2V defence prep: 3-minute speech + question bank for slides 17–23

**Status: written 2026-10-03.** Preparation material, not a research finding: nothing here was
trained or re-measured. Every number is read from the paper deck
(`Bidirectional_Communication_System-1.pdf`, slides 17–23 plus 24–31 for context) and checked
against this repo's own reports, `registry/`, and the source. Where the deck and the repo disagree,
§2 says so — fix those before presenting.

| | |
|---|---|
| **Scope** | The Sign-to-Voice (S2V) half: slide 17 (architecture) → 18 (GISLR-Stratified split) → 19 (ME-126 + preprocessing) → 20 (temporal memory, five architectures) → 21 (state reset) → 22 (continuous decoding) → 23 (the ~75% ceiling) |
| **Deliverables** | §1 speech (~3 min) · §2 deck-vs-repo corrections · §3 numbers cheat-sheet · §4 question bank (96 Q&A, tagged) |
| **Not covered** | V2S slides 7–16 (Whisper, rules, T5, ablations), slides 27–28 (ASL-LEX phonology, phonology front-end) beyond what S2V questions touch |

---

## 1. The 3-minute speech (slides 17–23)

Written for **~140–145 words/minute** (≈ 2:55–3:00 at a steady, unhurried pace). It deliberately **avoids the claims that §2 flags** (it never says "perfect classifier",
never says the continuous model uses the phonology front-end, never quotes the 126-landmark
breakdown on slide 19). If you fix the slides per §2, nothing in the speech has to change.

<!-- speech:start -->
**Slide 17 — S2V architecture (~0:00)**

Now the other direction: sign to voice. MediaPipe Holistic turns each camera frame into 543 landmarks, and we keep 126. A causal GRU reads them one frame at a time, with no lookahead, so it can stream. A decoder turns its per-frame scores into glosses, rules make English, and text-to-speech speaks it.

**Slide 18 — GISLR-Stratified (~0:21)**

We train on GISLR: 94,477 clips, 250 signs, 21 signers, in a fixed 80/20 split — 75,581 clips to train, 18,896 held out. One caveat: the split is stratified by sign, not by signer, so these are not unseen-signer results.

**Slide 19 — ME-126 and preprocessing (~0:38)**

Not every landmark earns its place. Most pose z-axis motion is noise — about 92 percent — so we drop z. ME-126 keeps both hands, eight upper-body pose points, lips, eyes and nose: 252 numbers per frame. In an early comparison it beat all 543 landmarks, 73.7 to 70.6 percent, with half the parameters. Fixed layers normalise each frame to the shoulders.

**Slide 20 — Temporal memory matters (~1:03)**

We compared five architectures under identical conditions. The GRU is the best that can stream: 73.8 percent top-1, 89.7 top-5. The BiLSTM is a tenth of a point higher but needs the whole clip, so it cannot run live. CNN and DNN, with little or no memory, fall seven to nine points behind. Adding 84 fixed sign-parameter features, such as handshape and palm orientation, takes our best streaming GRU from 75.2 to 76.3 percent.

**Slide 21 — State reset (~1:34)**

Memory has a cost between signs: the previous sign bleeds into the next. Resetting the state at a boundary cuts that bleed-through by 96 percent for the GRU and 98 for the LSTM, and the GRU recovers about 44 percent faster. Frame-by-frame inference matches whole-sequence inference to two millionths. Thirty synthetic streams: a mechanism check, not natural signing.

**Slide 22 — Continuous decoding (~1:58)**

Real sentences need decoding, measured by gloss error rate. Given the true boundaries, an isolated-sign baseline scores 0.221. Our deployed continuous model, C4, scores 0.278; a sliding-window baseline, 0.507. A trigram next-gloss prior cuts error about six percent. Non-sign movement is the hard part: a confidence floor rejects it only by deleting correct signs. So we ship a fixed-lag lattice: it waits one or two signs, then decides using the prior on both sides — about 27 frames of delay, error 0.347 down to 0.320 against the plain floor.

**Slide 23 — The ~75% ceiling (~2:35)**

Finally, why do isolated signs stop near 75 percent? Not semantic confusion: merging the twenty worst pairs gains only 2.7 points. It is generalisation — pair confusion is 0.012 on training data and 0.273 on unseen data. So the next lever is normalisation and augmentation, not a bigger classifier. Causal, streaming, and honestly measured.
<!-- speech:end -->

**Delivery notes**

- **Running long?** Cut the last sentence of slide 21 and the "about 27 frames of delay" clause on slide 22 (≈ 12 s).
  **Running short?** On slide 20 add: "the BiLSTM has three times the parameters" (2.75 M vs 0.85 M).
- **Why 73.8 → 75.2 → 76.3?** 73.8 is the controlled five-architecture GRU (ME-126/xy); 75.2 is the
  best raw-landmark GRU on the same split (ME_132/xy, 0.7517); 76.3 is `gru_phono_raw` (ME_134, 0.7632).
  The +1.2 is measured against 75.2, not against the 73.8 bar — expect that question (Q44).
- **Hand-off in:** slide 16 ended on V2S latency (~2.05 s mean end-to-end). Open with "Now the other
  direction" so the audience hears that S2V is the second half of one system.
- **Hand-off out:** slide 24 is the integrated architecture; end on "causal, streaming, honestly
  measured" and click straight on.
- **Words you will be asked to define:** *gloss* (a written label for a sign, e.g. `GIVE`), *causal*
  (uses only past frames), *GER* (gloss error rate = substitutions + deletions + insertions over
  reference signs), *bleed-through* (frames after a boundary where the old sign still outscores the
  new one).

---

## 2. Fix before presenting: where the deck and the repo disagree

Found by reading slides 17–23 against `packages/sb-core/src/sb/core/subsets.py`,
`registry/index.csv`, `train.py`, `architectures.py` and the reports. **F1–F6 are things an examiner
can verify in the repo**; F7–F10 are scope notes. Each has a one-line fix that costs nothing.

| # | slide | the deck says | what the repo says | one-line fix |
|---|---|---|---|---|
| **F1** | 19 | ME-126 = Both hands 42 · **Upper-body pose 33** · **Lips + eyes + nose 8** · Face 43 → 126 | `subsets.py`: `LIPS_40` + `HANDS_42` + `EYES_NOSE_36` (16 + 16 + 4) + `UPPER_BODY_POSE_8` {11–16, 23, 24} = 126. The slide's "33" is the *whole* pose and its "8" is the *pose* count: the arithmetic adds to 126 but the labels are wrong | Hands 42 · upper-body pose 8 · lips 40 · eyes + nose 36 |
| **F2** | 19 | "Full 543 (xyz) 70.59% → ME-126 (**xy**) 73.73%", "dropping z … raises accuracy" | Both runs **kept xyz** (`docs/logs/daily/2026-07-15.md` §6.4: "xyz kept so the subset is the only changed variable"); 73.73% is the **subset** effect only (1.91 M → 0.95 M params), on the **retired 90/10 split**, selected on its val set. Current split, canonical: `gru` ME_126/xy **0.7450**, ME_132/xy **0.7517**; there is **no current-split full-543 `gru`** to compare against | Label it "ME-126 (xyz), early 90/10 split" or add the current-split 0.745 / 0.752 and say the z-drop is a separate ablation |
| **F3** | 20 | Bar labelled "**GRU · final**" at 73.8; box says final = GRU + sign-parameter features; right panel is "C4 (`gru_continuous_norm`)" curves | Three different models: the 73.8 bar is the plain `gru` of the five-arch benchmark (`base_v1`, ME-126/xy); the phonology GRU is `gru_phono_raw` ME_134 = **0.7632**; the plot is **C4**, the continuous model | Relabel the bar "GRU (baseline)"; put "76.3 with sign-parameter features" in the box; caption the plot "C4, continuous model" |
| **F4** | 17, 20 | "Streaming GRU + phonology front-end, causal, no lookahead" as the pipeline's model | The **deployed continuous model C4** uses `StreamNormFrontend` (re-slot hands, shoulder-normalise, presence flags), **not** `PhonologyFrontend`. The phonology GRU (`gru_phono_raw`) is the **isolated-sign** model. The continuous phonology model **P1 failed** (GER **0.587** vs C1 0.293, `continuous-v2.md` §7) | Say "GRU on shoulder-normalised landmarks (+ phonology features in isolated mode)" |
| **F5** | 17 | "Rules + T5 guarded hybrid → fluent English" under English Text | The shipped S2V gloss → English is the **rule engine** (`apps/web/src/pipeline/gloss2en.ts`, `rules_v1`: BLEU 60.2 / chrF 75.0 / 42.4% exact on 132 **unreviewed Claude-drafted** references); the LLM arm is **pending**; T5 is the **V2S** refiner. TTS = browser `speechSynthesis` | "Rule-based gloss → English; browser TTS" |
| **F6** | 22 | "**Oracle (perfect classifier)** 0.221" | A perfect classifier scores GER **0**. 0.221 is the best §12.2 **isolated-sign baseline given the true boundaries** (B1, `gru_reg`; substitutions only), i.e. the classifier's own error with segmentation solved (`continuous-models.md` §2) | "Oracle boundaries (isolated classifier)" |
| F7 | 22 | Lattice card: "recovers 122 correct signs vs the floor alone" | The **+122** is from the **`peak`-score** diagnostic on the 5 selection signers (TODO §12.6, line 4045), not from the lattice. The lattice's own result: **1.4 fewer missed signs per 100 (clean), 0.7 (noisy)**; GER 0.347 → 0.320 clean, 0.580 → 0.547 noisy | Replace with the lattice's own numbers |
| F8 | 22 | "C4, deployed 0.278" | 0.278 is **plain D3, clean sentences**, no lattice, no prior. The lattice and trigram prior were tuned on **C1** and are "shared by every model" (`main.ts:371`); the **full deployed stack on C4 has no GER** in the repo. Also the lattice's **clean** GER (0.320) is *worse* than plain D3's (0.293): the lattice buys noise rejection, not clean accuracy | Footnote: "prior + lattice tuned on C1" |
| F9 | 21 | "State reset strengthens streaming" (also the slide 29 conclusion) | Measured on the five-arch isolated-trained `gru`/`lstm` with an **oracle** reset at the true boundary. In the **continuous** models, resetting after each commit **hurt** (D1r/D2 insert up to 5.7 spurious signs per 1,000 frames; the models were trained without reset, `continuous-models.md` §2). Reset is used in the app's **individual-sign mode** (commit + reset), not inside the continuous C4 stream | Say "for isolated-trained recurrent models, with oracle boundaries" |
| F10 | 23 | One slide mixes numbers | 74.33 → 77.04 % and 0.012 → 0.273 come from `plateau-diagnosis.md` (31 runs, **retired 90/10 split**); the table is the five-arch benchmark (current split, **internal validation carve-out**); the +2.7 pp is an **upper bound** with a random-pair control. The recommended fix (normalisation/augmentation) is **not yet tested as a controlled ablation** (TODO §7.2/§7.4 open) | Add "legacy split" to the left cards; word the right box "the proposed levers" |

**Two limitations the deck does not state but the code shows — be ready, not alarmed (Q13, Q14):**

1. **Not signer-disjoint.** Slide 18 says so; the consequence is that every isolated accuracy and every
   GER number is *seen-signer*. The "16 evaluation signers" are held out from **decoder tuning and the
   prior**, not from the recognizer's training (all 21 signers are in `train.csv`).
2. **Model selection on the reported set (isolated runs).** `train.py:272-297` builds `val_split` from
   the canonical split, i.e. `test.csv` (18,896 clips); LR-on-plateau, early stopping and "best
   checkpoint" all read it, then `sb-evaluate` reports on the same clips. So 0.7632 is mildly
   optimistic. The **five-arch benchmark** and the **continuous models** instead select on a carve-out
   of `train.csv` (clean). Size of the bias is **unmeasured**; the cheap fix is to re-score the best
   configuration with the five-arch protocol.

---

## 3. Numbers cheat-sheet

| what | number | where |
|---|---|---|
| GISLR | 94,477 clips · 250 signs · 21 signers · (T, 543, 3) | slides 17–18 |
| split | fixed 80/20, stratified on sign only, seed 42: **75,581 / 18,896** | `data.py` |
| holistic landmarks | 543 = face 468 + pose 33 + hands 42 | slide 19 |
| ME-126 | hands 42 + upper-body pose 8 {11–16, 23, 24} + lips 40 + eyes 32 + nose 4 = 126 → **252** xy features | `subsets.py` |
| deployed subsets | C4: **ME_132** (+6 pose hand points) · phonology GRU: **ME_134** (+forehead, chin) | `architectures.py`, registry |
| `StreamingGRU` | LayerNorm → 2×GRU(256, dropout 0.3) → LayerNorm → Dropout → Linear(250); **851,698** params (ME_126/xy) | `architectures.py` |
| training | AdamW lr 1.414e-3, wd 1e-4, batch 1024, label smoothing 0.1, clip 5, ReduceLROnPlateau ×0.5/5, early stop 15 (Δ 0.001), ≤ 300 epochs; ~0.3 min/epoch | `gislr.training.json`, `train.py` |
| five-arch (held-out, top-1 / 3 / 5) | BiLSTM 73.92 / 86.53 / 89.24 (2.75 M, offline) · **GRU 73.80 / 86.69 / 89.73 (0.85 M)** · LSTM 72.86 · CNN 66.96 · DNN 64.85 | `five-arch-benchmark.md` §2.1 |
| mean true-class confidence | DNN **0.2387** vs 0.59–0.61 for the rest | same |
| train/val gap | GRU 0.170 · LSTM 0.201 · BiLSTM 0.250 · CNN 0.036 · DNN 0.005 | same §2.2 |
| best isolated | `gru_phono_raw`, ME_134: **0.7632** canonical (macro 0.7614), 928,698 params; +1.15 over `gru` ME_132/xy 0.7517 | registry, `phonology-models.md` |
| phonology features | 84 = 2 × 41 per hand (handshape 25, orientation 6, location 9, present 1) + 2 elbow angles; non-learned, per-frame | `PhonologyFrontend` |
| permutation importance (`gru_phono_raw`) | raw hands −68.1 pts · right handshape **−44.3** · left handshape −34.0 · right orientation −21.4 · raw face −18.3 · locations −4 to −5 | `phonology-models.md` §3 |
| ensembles (exact, streaming) | 2 models 0.7912 · 3 models **0.8048** | `phonology-models.md` §8 |
| label merging | 9 groups / 19 words: 0.7632 → **0.7774** (synonyms) → 0.7822 (lenient) | `label-merging.md` |
| reset (oracle, 30 streams × 4 clips = 120 segments) | bleed-through GRU 19.6 → 0.8 frames (−96%), LSTM 16.1 → 0.4 (−98%); re-acquisition GRU 41.4 → 23.3 (−44%), LSTM 36.6 → 21.5 | `streaming-confidence.md` §2.2 |
| incremental vs batch | max error **1.7e-6** (GRU), 1.0e-6 (LSTM) on CPU | same |
| fresh vs contaminated confidence (GRU, late third) | **0.584** vs **0.088** | same §2.1 |
| GER (16 evaluation signers; 5,054 sentence seqs; 15,652 signs) | oracle boundaries 0.221 · **C1 + D3 0.293** · **C4 + D3 0.278** · sliding-window 0.507 · hard-cut C1 0.546 / C4 0.634 | `continuous-models.md`, `continuous-v2.md` §7 |
| trigram prior | GER 0.293 → **0.276** (−6%); sentence accuracy 37.7% → 40.7%; control order **+3%** worse (0.289 → 0.298); next-gloss top-5 **30%** vs 2% uniform | `sign-to-speech-downstream.md` |
| noise | unfiltered **0.982** → floor θ = 0.3 **0.580** (39% of noise blocks still accepted); floor deletes 511 of 4,895 correct signs (10.4%), clean 0.293 → 0.347 | same |
| fixed-lag lattice (lag 2) | clean 0.347 → **0.320**, noisy 0.580 → **0.547**; ~**27 frames** delay; −1.4 / −0.7 missed signs per 100 | same §2.2 |
| C4 robustness (GER, C1 → C4) | landscape 0.651 → **0.305** · scale 0.7: 0.375 → 0.285 · 15 fps 0.504 → 0.381 · jitter 0.516 → 0.372 · `live_like` 0.592 → **0.408** · mirror 0.990 → 0.962 (still broken) | `continuous-v2.md` §7 |
| C4 canonical isolated | 0.7339 (C1 0.7188, C5 0.6795); 929,045 params; TFLite 3.48 MB, parity 1.4e-6 | `continuous-v2.md`, TODO row 0a |
| end-to-end (browser vs Python, 300 streams) | 293 identical + 7 float near-ties, **0 unexplained**; GER 0.245 (sample) / 0.276 (all 5,054); 47% sentences exact | `sign-to-speech-downstream.md` |
| speed | recognizer 0.06 ms/frame · decoder 2 µs/frame · rule English ~0.01 ms · browser voice ~0.2 s | same |
| ceiling | top-20 confusable pairs absorb **10.5%** of errors: 0.7433 → 0.7704 (control ≈ +0.0002); train confusion **0.012** vs val **0.273**; pair errors recover at rank 2/5: 77% / 95% vs others 31% / 56% | `plateau-diagnosis.md` |

---

## 4. Question bank (96)

**Tags:** **[O]** overview · **[T]** technical · **[D]** detail · **[C]** critical / hostile.
**⚠** = the honest answer touches a §2 item — rehearse it. `(source)` names where to verify.
Answers are written to be spoken in 15–30 seconds; add the number from §3 if pressed.

### A. Overview and motivation (Q1–Q10)

**Q1 [O] · S17.** What problem does sign-to-voice solve, and for whom?
→ A Deaf signer's ASL, seen by a camera, becomes English speech a blind listener can hear, with no human interpreter in between. It is half of the bidirectional system; V2S is the other half.

**Q2 [O] · S17.** Why a causal, streaming design rather than the most accurate model?
→ In a live conversation output must start while the signer is still signing. A bidirectional model needs the whole clip: the BiLSTM is only 0.12 points ahead and cannot be deployed. So every new feature in this repo has to be causal. *(CLAUDE.md; five-arch §2.1)*

**Q3 [O] · S17.** Walk through the pipeline in one breath.
→ Camera → MediaPipe Holistic (543 landmarks) → ME subset → per-frame normalisation → causal GRU scoring 250 signs + "no sign" each frame → decoder (null-gated segments, lattice, trigram prior) → rule gloss → English → browser text-to-speech.

**Q4 [O] · S17–23.** What are the contributions of the S2V half?
→ (1) landmark-subset selection from motion energy and discriminability; (2) a controlled five-architecture comparison on one pipeline; (3) non-learned phonology features; (4) a reset/bleed-through analysis; (5) continuous decoding with a prior and a lattice; (6) a diagnosis of the ~75% ceiling; (7) a model that runs in the browser.

**Q5 [O] · S20, S22.** What is your headline result?
→ Isolated: **76.3%** top-1 over 250 signs (`gru_phono_raw`), streaming. Continuous: **GER 0.278** on synthetic sentences (C4). Both on seen signers (Q13), neither on real conversation.

**Q6 [O] · S17.** Is it real-time? What is the latency?
→ Recognizer 0.06 ms/frame and decoder 2 µs/frame on the dev machine; the lattice adds ~27 frames of decision delay; speech starts ~0.2 s after text. We have **not** measured latency on a target phone or with users (slide 25 says so). *(sign-to-speech-downstream.md §5)*

**Q7 [O] · S17.** What happens with a sign outside your 250?
→ It is a closed vocabulary. An unknown sign is either mapped to the nearest known one or rejected by the confidence floor (θ = 0.3), which also deletes 10.4% of correct signs. Open vocabulary (a model trained with 20 glosses held out, `C-open`, then enrolling new signs from 1/5/10 examples) is trained but the enrolment experiment is still planned (TODO §12.4); nothing of it ships. Fingerspelling is not recognised in S2V.

**Q8 [O] · S18.** Why ASL and why GISLR?
→ GISLR is the largest public isolated-sign landmark set (94,477 clips, 21 signers), already extracted, so effort goes into modelling rather than extraction. The cost is a 250-sign vocabulary and isolated, not conversational, signing.

**Q9 [O] · S17.** How do S2V and V2S fit together in the app?
→ One web app, two tabs (Voice → Sign, Sign → Voice) the user toggles between. They share no model, only the app shell. No joint two-person conversation study has been run.

**Q10 [O] · S17.** Earlier work reports up to 98% accuracy. Why is yours ~76%?
→ High reported numbers usually come from small vocabularies or alphabets, isolated clips or random splits (name the specific papers from your literature review here). Ours is 250 signs, held-out clips, and a causal model with no lookahead; the metrics are not comparable. The point here is the streaming pipeline and an honest measurement.

### B. Data and split (Q11–Q22)

**Q11 [D] · S18.** Describe the dataset.
→ GISLR (Kaggle `asl-signs`): 94,477 clips, 250 signs, 21 signers; each clip is a `(T, 543, 3)` array of Holistic landmarks with NaN where a landmark was not detected.

**Q12 [D] · S18.** What is "GISLR-Stratified"?
→ Our own Kaggle dataset: the competition parquet pre-converted to one npz per clip plus `train.csv`/`test.csv` — a fixed 80/20 split stratified on sign only (seed 42): 75,581 train, 18,896 held out. It exists so every run uses the same split. *(README; `data.py`)*

**Q13 [C] · S18. ⚠** The split is not signer-disjoint. Doesn't that inflate your accuracy?
→ Yes, probably, and slide 18 says so. All 21 signers appear in training, so 76.3% is a seen-signer number; unseen-signer accuracy is unmeasured and likely lower. The GER "evaluation signers" are held out only from decoder tuning and the prior. A leave-signers-out evaluation is the right next experiment; I would not claim signer independence.

**Q14 [C] · S18–20. ⚠** Is your reported accuracy on data never used for model selection?
→ Depends on the model. The five-architecture benchmark and the continuous models select on a carve-out of the training set, so their held-out numbers are clean. The isolated registry runs (including 0.7632) monitor the same 18,896 clips for early stopping and best-checkpoint selection, then report on them, so they are mildly optimistic; the size is unmeasured. Re-scoring the best configuration with the carve-out protocol would quantify it. *(`train.py:272-297`)*

**Q15 [D] · S18.** Why 250 signs?
→ It is the dataset's label set; we did not choose it. For the phonology validation, 233 of the 250 glosses map to ASL-LEX entries; the other 17 are left out of that check only.

**Q16 [D] · S22.** How were "no sign" frames created?
→ GISLR has none, so GISLR-Sentences synthesises them: rest (lowered or hands-absent) and 0–15-frame interpolated transitions between clips, labelled as the null class; the v2 training adds fidget/hold/reverse blocks as null too.

**Q17 [D] · S22.** What is GISLR-Sentences?
→ A continuous **test** corpus built from `test.csv` clips: 1,757 template sentences over 11 themes (ASL gloss order, length 2–7, mean 3.1), every one of the 18,896 test clips placed, giving 6,616 `sentence` sequences (1,227 distinct sentences) plus a `control` split with the same clips in random order. Continuous *training* streams are composed on the fly from `train.csv` clips.

**Q18 [C] · S22. ⚠** Are those sentences natural signing?
→ No. They are template sentences written by an LLM, assembled from isolated clips with synthetic gaps; there is no coarticulation, and stitched signs are too long and too clean (the literature reports 1.63× real length). Slide 18 and slide 25 both say concatenation is not natural conversation.

**Q19 [D] · S22.** How many signers are used for GER, and why 5 + 16?
→ 5 selection signers pick decoder settings (ν, minimum length, θ, prior weight); the other 16 give every reported number: 5,054 sentence and 4,639 control sequences, 15,652 reference signs. This avoids tuning on the scored data.

**Q20 [D] · S22.** What is the `control` split for?
→ Same clips in random order. If the trigram prior only memorised sentence templates it would not help there; in fact it costs 3% (0.289 → 0.298), which shows its gain on `sentence` comes from sentence structure.

**Q21 [D] · S19.** How are missing landmarks handled?
→ NaN → 0 when the cache is built; the model also gets a presence flag per hand so "undetected" is distinguishable from "at the origin". Training adds hand dropout (up to 20%).

**Q22 [D] · S18.** How long is a clip, and what is `MAX_SEQ_LEN`?
→ Clips are subsampled to 128 frames for isolated training; 8.3% of test clips exceed 128 frames, a small train/inference mismatch noted in the streaming analysis. Live streaming has no cap; the continuous v2 streams run up to 1,500 frames.

### C. Landmarks and preprocessing (Q23–Q34)

**Q23 [T] · S19. ⚠** Exactly which 126 landmarks does ME-126 keep?
→ Both hands 42 + upper-body pose 8 (shoulders, elbows, wrists 11–16, hips 23–24) + lips 40 + eyes 32 + nose 4 = 126; with x and y that is 252 features. (Slide 19's labels are off — F1.) The deployed models extend it: ME_132 adds 6 pose hand points, ME_134 adds forehead and chin.

**Q24 [T] · S19.** How was the subset chosen?
→ A motion-energy analysis, a cross-check with the Kaggle 1st-place subset, and a discriminability probe (F-ratio, mutual information, probe classifier). ME-126 won the probe: 49.9% vs FP-118 48.6%, hands + pose 46.7%, hands only 43.7%, full 543 40.6%. Lips/eyes/nose are kept on linguistic grounds, not motion. *(subset-comparison.md)*

**Q25 [C] · S19. ⚠** Exactly what changed between 70.59% and 73.73%?
→ Only the landmark subset: both runs used xyz, same GRU, 60 epochs; parameters fell from 1.91 M to 0.95 M. They were measured on the early 90/10 split and selected on its validation set, so they are historical. On today's canonical split the same family gives 0.7450 (ME_126/xy) and 0.7517 (ME_132/xy). I do not have a current-split full-543 `gru`. *(daily/2026-07-15 §6; registry)*

**Q26 [T] · S19.** Why drop z?
→ Pose z is mostly depth jitter (~92% of pose "motion"); hands keep ~76% of their motion energy in xy, so their motion is real in xy. Dropping z halves the input and the ablation said xy ≥ xyz. Nuance: the phonology front-end reads xyz because joint angles and palm normals are 3-D; the raw coordinates it appends are xy. *(motion-energy.md; `PhonologyFrontend`)*

**Q27 [T] · S19.** Why subtract the shoulder midpoint and divide by shoulder width?
→ It removes camera distance, framing and image position. The first continuous model read raw image coordinates and scored GER 0.651 on a landscape view; with the normalisation (C4) it scores 0.305. Test: zoom 0.7 plus a shift changes the output by at most 6e-7.

**Q28 [T] · S19.** What is "re-slotting hands"?
→ Each hand goes to the slot of the pose wrist it is nearest, instead of trusting MediaPipe's left/right label, which differs between camera stacks. On GISLR a hand is a median 0.09 shoulder widths from its own wrist and 1.6 from the other; it relabels only 0.2% of one-hand frames. With every label swapped the output is identical on 100% of frames.

**Q29 [T] · S19.** What are the presence flags for?
→ A missing hand is all zeros after fill, which looks like a real position. The flag (one per hand) lets the network tell "not seen" from "seen at zero".

**Q30 [T] · S19.** Why `input_norm` (LayerNorm) before the GRU?
→ Features have different scales (positions, flags, angles); normalising per frame stabilises training and works frame by frame, so it is stream-safe.

**Q31 [C] · S19, S25. ⚠** What about left-handed signers or a mirrored camera?
→ Unproven (slide 25). The phonology front-end mirrors the left hand and training mirrors whole samples with probability 0.5, but a mirrored or hand-swapped live feed still scores GER 0.962 for C4 (0.990 for C1). That is a known open failure.

**Q32 [D] · S19.** What if the face or hands are occluded?
→ The landmarks become zeros plus a flag. Hand dropout (up to 20%) and jitter augmentation in the v2 training make C4 degrade less under noise: 1% jitter costs 0.087 GER for C4 vs 0.223 for C1. I have no measured detection rate for GISLR itself to quote.

**Q33 [D] · S17, S19.** Slide 17 says 252 features, but your deployed model uses ME_132. Which is it?
→ 252 is ME-126/xy, the subset family used in the controlled benchmark. The deployed continuous model is ME_132 (264 + 2 presence flags = 266 inputs to the GRU) and the phonology GRU is ME_134 (84 + 268 = 352). "ME-126" on the slides names the design, not every deployed width.

**Q34 [T] · S19.** Does preprocessing run inside the model or in the app?
→ Inside the exported model as fixed, non-learned layers (`StreamNormFrontend`), so the app feeds raw Holistic xy. The TFLite port was verified against PyTorch to 1.4e-6.

### D. Model and architecture (Q35–Q50)

**Q35 [T] · S19–20.** Describe `StreamingGRU`.
→ `input_norm` LayerNorm → 2-layer unidirectional GRU, hidden 256, dropout 0.3 → LayerNorm → Dropout → Linear to 250 classes; 851,698 parameters on ME_126/xy. The isolated model is trained with a loss at the last valid frame.

**Q36 [T] · S20.** Why a GRU and not a Transformer or TCN?
→ Among the streaming-viable models we tried, the GRU is the most accurate (73.80 vs LSTM 72.86, CNN 66.96) with the fewest parameters. A standard Transformer attends over the whole sequence and is non-causal; our 1st-place port is offline-only. A causally masked Transformer was not tried; ST-GCN, TCN and Conformer are planned, not run. *(README; TODO §4)*

**Q37 [T] · S20.** The BiLSTM is the highest. Why not use it?
→ It leads by 0.12 points at 3× the parameters (2.75 M) and needs the future, so it can never stream. On the current split `bilstm_base` is 73.71% — below every current-split `gru` run — so bidirectionality buys nothing here. *(bilstm-curated.md)*

**Q38 [T] · S20.** Why do CNN and DNN trail?
→ They have little or no temporal memory: the dilated causal CNN sees 125 frames at 5 blocks and scores 66.96%; the per-frame DNN scores 64.85%. Both underfit (train/val gaps 0.036 and 0.005, 137 epochs). An early bug is worth knowing: at 2 blocks the receptive field collapsed to 13 frames and the CNN scored ~0.54.

**Q39 [T] · S20.** What is "mean true-class confidence", and what did it show?
→ The probability the model gives the correct class, whatever its rank. The DNN scores 0.2387, a quarter of the others (0.59–0.61), though its top-5 is comparable: ranked accuracy and confidence decouple for a per-frame, softmax-averaged model.

**Q40 [T] · S20.** State your training recipe.
→ AdamW, lr 1.414e-3, weight decay 1e-4, batch 1024, label smoothing 0.1, gradient clip 5, mixed precision; ReduceLROnPlateau (×0.5, patience 5), early stop patience 15 (min Δ 0.001), cap 300 epochs; about 0.3 min per epoch with arrays in RAM. One shared config for all architectures.

**Q41 [T] · S20.** How were hyperparameters tuned?
→ One shared regime for every architecture (so comparisons are all-else-equal), set in the §3.2 training-regime update; architectures may override only keys that exist in `shared` (e.g. CNN `num_layers`). I have no record of a per-architecture search, so the ranking is a controlled comparison, not a tuned-ceiling comparison.

**Q42 [T] · S20.** What are the 84 "sign-parameter" features?
→ Per hand 41: handshape 25 (15 joint-angle cosines, 5 fingertip-to-wrist distances over palm size, 4 finger-spread cosines, thumb–index gap), orientation 6 (palm normal, wrist→middle-knuckle direction), location 9 (centroid x, y and distances to nose, chin, forehead, mouth, same-side shoulder, chest, other hand), presence 1; plus 2 elbow angles. The left hand is computed mirrored.

**Q43 [T] · S20.** Are they learned?
→ No: deterministic and per-frame, so they are causal and stream-safe with no trainable parameters. The learned model (`gru_phono_raw`) gets them concatenated with 134 landmarks' shoulder-normalised xy (84 + 268 = 352 inputs).

**Q44 [C] · S20. ⚠** +1.2 points — is that significant?
→ Two identical-config `gru` ME_132 runs differ by 0.0007 (0.7517 vs 0.7511), so run-to-run noise looks small, and +1.15 is well above it; but the phonology model is a **single run** with no multi-seed repeat (slide 25 admits few seeds). I would present it as a consistent gain, not a significance-tested one. *(registry)*

**Q45 [T] · S20.** Which features does it actually use?
→ Permutation importance on the validation set: raw hand coordinates −68.1 points when shuffled, right handshape −44.3, left handshape −34.0, right palm orientation −21.4, raw face −18.3, locations only −4 to −5. So the engineered handshape features are not redundant with raw hands. Drops are not additive.

**Q46 [T] · S20.** What did phonology alone achieve?
→ `gru_phono` (the 84 phonology features alone) 0.7010; `gru_phono130` (130 phonology features from all 543 landmarks) 0.7487, level with raw `gru` on half the inputs, and it errs on different clips: averaging it with the raw and phono+raw GRUs gives 0.8048 with no training. *(phonology-models.md)*

**Q47 [T] · S17.** Model size and compute?
→ ~0.93 M parameters; the TFLite step model is 3.48 MB; 0.06 ms per frame. Landmark extraction by Holistic, not the GRU, dominates compute.

**Q48 [T] · S21.** How is the hidden state carried at inference?
→ `RecurrentSession.step(frame)` keeps the state between frames, `reset()` clears it. It reproduces the batch computation to 1e-6–1e-7 (checked for GRU, LSTM, C4).

**Q49 [T] · S20–21.** The isolated model is supervised only at the last frame. Is mid-sequence confidence meaningful?
→ From a clean start, yes: the GRU's late-third confidence is 0.584, matching whole-clip numbers, rising monotonically. The apparent weakness came from segments carrying the previous sign's state (0.088). The per-frame retrain was therefore downgraded to optional; the continuous models do supervise every frame. *(streaming-confidence.md §2.1)*

**Q50 [T] · S22.** How is the continuous model built?
→ `ContinuousGRU`: same 2 × 256 GRU, a per-frame cosine-similarity head over 250 glosses + null (scale 16; it supports enrolling new signs), plus a sign-boundary head. Trained per frame on composed streams with no state reset. C4 puts `StreamNormFrontend` in front.

### E. Streaming and state reset (Q51–Q60)

**Q51 [T] · S21.** What is "bleed-through"?
→ The number of frames after a true boundary during which the previous sign's class still has higher probability than the new one's: GRU 19.6 → 0.8 frames, LSTM 16.1 → 0.4 with an oracle reset.

**Q52 [C] · S21. ⚠** The reset was at the *true* boundary. How does the live system know it?
→ It doesn't have an oracle. 96–98% is an upper bound. Live, boundaries come from the null-gated decoder or, in individual-sign mode, from the hand leaving the frame or a stable top-1; and in the continuous C4 stream, resetting after each commit actually *hurt* (spurious signs up to 5.7 per 1,000 frames) because the model was trained without resets. So reset helps the isolated-trained models in a mode where boundaries are known. *(continuous-models.md §2)*

**Q53 [D] · S21.** What does "44% faster re-acquisition" mean precisely?
→ Frames until the new sign reaches 0.5 confidence: GRU 41.4 → 23.3 (−44%), LSTM 36.6 → 21.5 (−41%). The no-reset figure is inflated by segments that never reach 0.5 in their own length, which is itself a finding.

**Q54 [D] · S21.** Why do you quote 1.7 × 10⁻⁶?
→ It is the max absolute difference between frame-by-frame `step()` and the whole-sequence forward on real data (GRU; LSTM 1.0e-6). It proves the incremental API is the same computation, so the streaming claims apply to the trained model. On GPU, cuDNN's single-step kernel differs by ~5e-4, which is a kernel effect, not a bug.

**Q55 [C] · S21. ⚠** Thirty streams of four clips — enough to conclude anything?
→ It is 120 segments from isolated clips, joined with no gap, so it validates the mechanism, not natural signing; slide 21 says that. I would not generalise the percentages beyond "reset removes most carried-over state".

**Q56 [D] · S21.** What did fresh vs contaminated segments show?
→ GRU late-third confidence 0.584 fresh vs 0.088 contaminated — worse than a fresh sign's *early* third (0.161). The memory-free DNN barely changes (0.314 vs 0.228), confirming the effect is carried recurrent state.

**Q57 [D] · S21.** Could a rule trigger the reset automatically?
→ `AcceptTrigger` (top class above τ for `hold` frames): GRU τ = 0.9, hold 10 gives precision 1.000 but misses 97 of 120 segments; τ = 0.5, hold 3 gives 0.787 with 31 missed. There is a real precision/coverage trade; the GRU beats the LSTM at every setting.

**Q58 [T] · S21–22.** If reset hurts continuous models, why train them without it?
→ That was the recorded design decision (TODO §12.3, user-approved): a dedicated boundary head plus training on uninterrupted streams, with the decoder free to reset. The later measurement (Q52) showed that resetting after a commit hurts these models, so the decoder in the app does not do it per sign.

**Q59 [D] · S22.** What is the decision latency of the decoder?
→ D3 commits a median of 1 frame after a sign ends; the lattice waits one or two further signs (~27 frames).

**Q60 [D] · S21.** Why is the LSTM trigger worse than the GRU's?
→ At the same thresholds the LSTM's precision is lower (0.748 vs 1.000 at τ 0.9, hold 10), and the GRU dominates at every setting tested, so the GRU is the better-behaved streaming model on this trigger as well as the more accurate one.

### F. Continuous decoding and GER (Q61–Q77)

**Q61 [T] · S22.** Define gloss error rate.
→ Edit distance between decoded and reference gloss sequences (substitutions + deletions + insertions) divided by the number of reference signs — word error rate for glosses. Lower is better.

**Q62 [C] · S22. ⚠** What is the 0.221 "oracle"?
→ Not a perfect classifier (that would be 0). It is the best isolated baseline (`gru_reg`) handed the true sign boundaries: substitutions only, so it is the classifier's own error with segmentation solved. C1's substitutions (0.213) are already at that level; the remaining gap to 0.293 is deletions (0.064) and insertions (0.017), i.e. segmentation.

**Q63 [T] · S22.** Explain decoder D3.
→ Commit a sign at the end of each run of frames whose null probability is below ν lasting at least `min_len` frames. It commits a median of 1 frame after the sign ends and beat the boundary-head decoder D1 (0.423 clean GER) on streams with gaps.

**Q64 [C] · S22. ⚠** D3 depends on gaps. What about fluent signing without pauses?
→ That is the pessimistic case and a known weakness: hard-cut GER is 0.546 for C1 and **worse for C4 (0.634)** even though 25% of v2 training streams were hard-cut. D1 is the better decoder there (0.486–0.491); a combined D3 ∪ D1 is open work. *(continuous-v2.md §7)*

**Q65 [T] · S22.** How does the trigram prior work?
→ An interpolated Kneser–Ney trigram over gloss IDs, fitted on the 1,757-sentence corpus, fused with the recognizer's vote as q · p^0.3 (weight chosen on the selection signers). On held-out folds it ranks the true next gloss first 11.6% of the time and in its top 5 30% (uniform: 2%).

**Q66 [C] · S22. ⚠** Is the prior just memorising your template corpus?
→ Partly a risk. It is scored on held-out folds (5 theme-stratified, plus theme-out: 27.5% top-5 on an unseen theme) and a 4-gram adds nothing held-out, but in-sample it reaches 65% top-5, so the corpus is small enough to memorise. The GER gain (−6%) is measured on signers not used to fit the prior, but real sentences are not these templates.

**Q67 [T] · S22.** What is the confidence floor and why isn't it enough?
→ Reject any segment below θ = 0.3. It takes noisy GER from 0.982 to 0.580, but 39% of noise blocks are still spoken, and on clean sentences it deletes 511 of 4,895 correct signs (10.4%), raising clean GER from 0.293 to 0.347.

**Q68 [T] · S22.** Explain the fixed-lag lattice.
→ It holds an uncertain segment until one or two more arrive, then picks the best path over each segment's top-k guesses or "skip", scoring with the trigram prior on both sides. Result vs the plain floor: clean 0.347 → 0.320, noisy 0.580 → 0.547, 1.4 / 0.7 fewer missed signs per 100, about 27 frames of delay, +2 missed per 100 on random sequences.

**Q69 [C] · S22. ⚠** The lattice's clean GER (0.320) is worse than plain D3 (0.293). Why ship it?
→ Because it is the best way found to reject non-sign movement without a retrain: plain D3 speaks 75% of noise blocks (noisy GER 0.982). On clean, noise-free input plain D3 is better; the lattice is a robustness choice. The numbers compare against the floor, not against D3.

**Q70 [C] · S22. ⚠** Where does "recovers 122 correct signs" come from?
→ From the `peak`-score diagnostic on the five selection signers (at the floor's error budget it admits +122 correct signs on clean, −248 on noisy), not from the lattice. The lattice's own gain is the missed-signs and GER figures in Q68. Fix the slide (F7).

**Q71 [D] · S22.** What is C4 and why was it chosen over C1?
→ C4 = `gru_continuous_norm`: C1's GRU behind `StreamNormFrontend`, trained on harder v2 streams (clip trim, 0.8–1.5× speed-up, 25% hard-cut, 30% noise-as-null, up to 16 signs, camera augmentation, jitter, hand dropout, 15/10 fps). It wins clean GER (0.278 vs 0.293) and nearly every live-robustness probe, `live_like` 0.592 → 0.408, with isolated accuracy 0.7339 vs 0.7188.

**Q72 [C] · S22.** What does C4 get worse at?
→ Hard-cut GER (0.634 vs 0.546) and mirrored feeds (0.962, still unusable). Both are flagged as known weaknesses, not blockers.

**Q73 [D] · S22.** What is the 0.507 sliding-window baseline?
→ The best streaming isolated model (`gru`) applied to sliding windows; it lost 30–47% of signs to deletions. A separate test of 1–3 s windows with two offset models scored GER 0.523 / 0.687 / 0.777, worse than per-frame decoding. *(sentence-baselines.md; window-ensembles.md)*

**Q74 [D] · S22.** Where does the remaining error come from?
→ Mostly classification, not segmentation (substitutions 0.213, deletions 0.064, insertions 0.017 for C1/D3). Short signs are the largest loss: signs under 12 frames are 25% of signs but only 54% are recognised vs 77–82% for longer ones.

**Q75 [D] · S22.** Did you try CTC?
→ Yes (C3): not competitive — best 0.546 with greedy decoding, frame accuracy 0.33 vs 0.72. GRU ≈ LSTM (0.293 vs 0.298), so the GRU is the deployment candidate.

**Q76 [T] · S17, S22.** Did the browser reproduce the Python numbers?
→ On 300 held-out streams the browser TFLite step model plus the online decoder emitted the same signs as PyTorch + batch decoder on 293; the other 7 are float near-ties, 0 unexplained. End to end GER was 0.245 on that sample and 0.276 on all 5,054 evaluation streams; 47% of sentences were exactly right.

**Q77 [C] · S22. ⚠** Has it been tested with a real camera and real signers?
→ Not rigorously. The first camera test failed on sentences while isolated signs worked; C4 was built to fix the measured causes and is validated on synthetic perturbations (aspect, scale, fps, jitter, combined `live_like`). A first real-camera check of the deployed C4 is still open, and no participant study exists (slide 25). That is the honest limit of the "real-time" claim.

### G. The ~75% ceiling and error analysis (Q78–Q86)

**Q78 [T] · S23.** Why does accuracy stop near 75%?
→ It is a generalisation gap, not a label ceiling: symmetric pair confusion is 0.012 on training data and 0.273 on unseen data, with 19 of 20 top pairs never confused in training. Recurrent models overfit (gaps 0.17–0.25), CNN/DNN underfit (0.036, 0.005).

**Q79 [D] · S23.** Describe the merging experiment.
→ Merge the top-N confusable pairs into one class and re-score — the accuracy if semantic confusion were solved perfectly. Top-20 pairs absorb 10.5% of errors (0.7433 → 0.7704, +2.7 points) against about +0.0002 for random pairs. It is an upper bound.

**Q80 [D] · S23.** Which pairs are worst?
→ `awake`/`wake` (symmetric confusion 0.839), `lips`/`mouth` 0.533, `hear`/`listen` 0.411, `pen`/`pencil` 0.404, `gift`/`give` 0.367. Two pairs, `finger`/`wait` and `animal`/`have`, are not semantically related at all, so meaning is not the whole story.

**Q81 [C] · S23. ⚠** Are those numbers on the current split?
→ The 0.7433 → 0.7704 and 0.012 → 0.273 come from the 31-run analysis on the retired 90/10 split. On the current split the systematic scan finds 9 merge groups (19 words) and lifts the best model 0.7632 → 0.7774 counting synonyms — the same ~1.4 point lift for every architecture, so the conclusion holds. The table on the slide is the current five-arch benchmark.

**Q82 [T] · S23.** Why isn't a re-ranking layer the answer?
→ Pair errors are near-misses (recovered at rank 2 in 77%, top-5 in 95%), but the other 90% of errors are far off (31% / 56%). Re-ranking can fix about one error in ten; bound ≈ 2.7 points.

**Q83 [C] · S23. ⚠** You say the fix is normalisation and augmentation. Did you test that?
→ Partly. Shoulder-normalisation inside C4 raised isolated canonical accuracy 0.7188 → 0.7339, while augmentation without normalisation (C5) fell to 0.6795, so normalisation helps and augmentation alone did not. The controlled §7.2/§7.4 ablations are still open, so "next lever" is a hypothesis, not a result.

**Q84 [D] · S23.** What do ensembles say about the ceiling?
→ Averaging three streaming models with different inputs reaches 0.8048, so the ceiling is not a hard label limit — different models make different mistakes. It costs three forward passes per frame.

**Q85 [T] · S23.** What would break the ceiling?
→ More signer diversity (not just more clips), signer-level augmentation (affine, speed, mirror), normalisation, ensembling or distillation of the ensemble, and merging true synonyms. A bigger classifier is the one thing the data argue against: `gru_deep` has 4× the parameters and scores 1–2 points lower.

**Q86 [D] · S23.** Class imbalance?
→ Not a factor: classes are nearly balanced (the old 9,448-clip validation set had 30–42 clips per class; the current 18,896 set averages ~76), so the +0.34 support–accuracy correlation has almost no spread to work with. *(plateau-diagnosis.md §4)*

### H. Phonology (Q87–Q89)

**Q87 [T] · S20.** Why would phonological features help a neural network?
→ They hand it the parameters that distinguish signs (handshape, orientation, location) in a shoulder-normalised, camera-independent form, so it does not have to infer them from raw coordinates with limited signer diversity. Handshape alone carries most of the signal.

**Q88 [D] · S20.** How do you know the phonology is real and not a coincidence?
→ Against ASL-LEX ground truth on 233/250 mapped glosses, no training: every parameter is recovered on unseen glosses, and per-sign templates reach 38.8% top-1 (4.9% for hand-built geometry variables; 41.4% for the 130-feature version vs 3.1% for raw landmarks). *(sign-patterns.md; asl-phonology-features.md)*

**Q89 [C] · S20. ⚠** Why isn't phonology in the continuous model?
→ It was tried: the continuous phonology model P1 scored GER 0.587 vs C1's 0.293, because its feature-space training streams did not match real streams. So the continuous app keeps C4 (shoulder-normalised raw landmarks) and uses the phonology GRU in individual-sign mode. A landmark-space re-composition is the fix, not built.

### I. Output stage, deployment and engineering (Q90–Q93)

**Q90 [T] · S17. ⚠** How are glosses turned into English?
→ A rule engine (`sb.rescore.gloss2en`, reverse of the V2S rules: restores articles, copulas, tense, plurals, time-first order). On 132 draft references: BLEU 60.2, chrF 75.0, 42.4% exact, keeps 92.6% of the signed content and invents 5.4%. Caveat: the references were drafted by an LLM and are unreviewed, by the same author as the rules.

**Q91 [T] · S17.** How is speech produced?
→ The browser's own `speechSynthesis` voices: fully intelligible to Whisper (WER 0–0.3%) in about 0.2 s, at no cost. A hosted voice (Workers AI Aura-2) costs ~120 neurons per sentence, about 80 sentences a day on the free plan.

**Q92 [T] · S17.** Where does inference run and what about privacy?
→ Entirely in the browser for S2V: MediaPipe in the page, a LiteRT.js TFLite step model, the decoder in TypeScript; the video never leaves the device. Cloudflare Workers only serves the app (and optionally hosted models/LLM).

**Q93 [T] · S17.** How did the PyTorch model reach the browser?
→ TensorFlow GPU does not work on native Windows, so the model is rebuilt in native Keras, weights transferred and exported to TFLite (the ONNX route was tried and dropped). Parity: C4 1.4e-6, phonology front-end 1.19e-6 against PyTorch.

### J. Reproducibility, limits, ethics, future (Q94–Q96)

**Q94 [D] · S18–23.** How reproducible is this?
→ Every run is a registry folder with `meta.json` (git commit, config hash, feature-cache key, environment), a fixed canonical split with the validation size asserted, and checkpoints public on Kaggle (MIT) with sha256 verification. 91 runs are recorded; 52 are on the retired split and flagged as historical.

**Q95 [C] · S25. ⚠** What are the main limitations?
→ (1) Seen-signer evaluation; (2) 250-sign closed vocabulary; (3) concatenated, LLM-templated sentences, no coarticulation; (4) isolated models selected on the reported set; (5) few seeds; (6) mirror/handedness and hard-cut failures; (7) no participant or on-device study; (8) gloss → English evaluated against unreviewed LLM-drafted references.

**Q96 [C] · S26. ⚠** Is this safe to use in a real conversation, and did Deaf users take part?
→ Not yet for anything consequential. No Deaf or blind participant study was run (slide 25 and 26: "a rigorous user-centred evaluation is the next essential step"). The plan (TODO §12.5/§12.6) is to grey out glosses under 0.5 confidence and auto-speak only when every gloss clears it — I have not verified that is exactly what ships. Either way, a misrecognition inside a closed 250-word vocabulary would be spoken fluently, so a human fallback is needed for medical, legal or safety contexts.

---

### Hot-seat shortlist (rehearse these eight first)

| Q | one-line answer |
|---|---|
| **Q13** signer-disjoint | Seen-signer numbers; unseen-signer accuracy is unmeasured; GER eval signers are held out only from decoder tuning |
| **Q14** selection on test | Isolated registry runs select on the reported set (mild optimism, unmeasured); five-arch and continuous select on a train carve-out |
| **Q25** 70.59 → 73.73 | Subset effect only, xyz both, old 90/10 split; current-split ME_126/xy 0.745, ME_132/xy 0.752 |
| **Q52** oracle reset | 96–98% is an upper bound with true boundaries; reset after commit hurt the continuous models |
| **Q62** "perfect classifier" | 0.221 is isolated-classifier error with true boundaries, not zero |
| **Q69** lattice vs D3 | Lattice (0.320) is worse than plain D3 (0.293) on clean; it buys noise rejection |
| **Q77** real camera | Synthetic perturbation probes only; real-camera and participant studies are open |
| **Q89** phonology in the stream | P1 failed (0.587); phonology is in the isolated mode, C4 is deployed for continuous |

*Sources read for this report:* the deck (31 slides); `README.md`; `TODO.md` (current focus, §3.9, §12.3–12.8, §14–§17); `docs/reports/{continuous-models,continuous-v2,streaming-confidence,sign-to-speech-downstream,plateau-diagnosis,five-arch-benchmark,phonology-models,label-merging}.md`; `docs/logs/daily/2026-07-15.md`; `packages/sb-core/src/sb/core/subsets.py`; `packages/sb-recognize/src/sb/recognize/{architectures,data,train,evaluate}.py`; `experiments/recognition/configs/gislr.training.json`; `registry/index.csv`; `apps/web/src/pipeline/{gloss2en,speech}.ts`.
