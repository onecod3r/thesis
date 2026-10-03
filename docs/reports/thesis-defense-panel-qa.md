# Thesis defence: panel questions and prepared answers (whole thesis)

**Status: written 2026-10-03.** Preparation material, not a research finding; nothing was trained or
re-measured. Companion to [`s2v-defense-prep.md`](s2v-defense-prep.md), which holds the 3-minute S2V speech
and 96 *technical* S2V questions. **This file covers what a panel asks across the whole thesis**: framing,
novelty, literature, data, V2S, S2V (summary + pointers), the integrated system, evaluation validity,
accessibility/ethics, limitations, team contribution, and curveballs.

| | |
|---|---|
| **Sources** | the 31-slide deck (`Bidirectional_Communication_System-1.pdf`); this repo's reports, notably `speech-to-sign-audit.md`, `thesis-ch1-ch2-review.md`, `s2v-defense-prep.md`, `continuous-models.md`, `sign-to-speech-downstream.md`; `README.md`, `TODO.md` |
| **Evidence labels** | **[deck]** = a number read from the slides that this repo cannot verify (the V2S T5 training code and checkpoint came from teammates, `speech-to-sign-audit.md` §6) · **[repo]** = measured/readable here · **[gap]** = something not done; say so plainly |
| **Count** | 92 questions, tagged by the panel member most likely to ask: **Chair** · **ML** (machine learning) · **Ling** (linguistics / Deaf studies) · **HCI** (accessibility, users) · **Sup** (supervisor) |

---

## 0. How to answer (read once)

1. **Answer → evidence → limit, in 30–45 seconds.** "Yes/no, because X (number), and the limit is Y." Panels
   trust a stated limit more than a defended overclaim.
2. **Concede, then pivot.** "That is a fair weakness — it is in our limitations — and here is what we did
   measure." Never argue a limitation you already listed on slide 25.
3. **Words to avoid unqualified:** *real-time* (say "streams frame by frame; ~2 s end-to-end for V2S"),
   *robust*, *state-of-the-art*, *achieves 98%*, *Deaf-validated*, *natural signing*.
4. **Know which artifact you mean.** The deck's demo (slide 30) is a Gradio prototype; this repo's
   deployed app is a Cloudflare Workers browser app (`signbridge.onecoder1.workers.dev`). Agree with your
   team which one the thesis reports before the defence (P55).
5. **Numbers you must have in your head (details in `s2v-defense-prep.md` §3):**
   V2S — corpus BLEU-4 **24.54 → 29.87 (+5.33)**, 95% CI **[+0.79, +9.12]**, 127 test pairs, 3 seeds;
   end-to-end **2.047 s mean / 2.023 s median / 2.510 s P95**, 98% under 3 s.
   S2V — **76.3%** top-1 isolated (250 signs, streaming); **GER 0.278** synthetic sentences (C4);
   **not signer-disjoint**; no user study.

---

## 1. Fix before the defence (V2S, overview, polish)

The S2V corrections (F1–F10) are in `s2v-defense-prep.md` §2. These are the extra ones for the rest of the deck.

| # | slide | issue | fix |
|---|---|---|---|
| **G1** | 13 vs 14 | Slide 13: BLEU-4 24.54 → **29.87 (+5.33)**. Slide 14, NCSLGR box: "Neural 24.54 → Hybrid **31.09**: +**6.55** BLEU". Same corpus, two hybrid numbers. 29.87 equals the mean of the three seeds (29.20, 27.95, 32.47); 31.09 is not derivable from the slides | Use one number, or label what 31.09 is (e.g. a different aggregation) **[deck]** |
| **G2** | 13 | Seed 123: hybrid **27.95 < neural 28.20**. The gain is +7.40 / −0.26 / +8.85 across seeds, and the CI lower bound is +0.79 | Say "improves on average; one of three seeds does not" (P28) |
| **G3** | 12–16 | The deck never reports a **rules-only** baseline on NCSLGR; the ablation compares neural-only vs hybrid. This repo's audit found the team's T5 *no better than its own rule input* on 30 sentences (BLEU 36.0 vs 39.3, draft references) | Get the rules-only NCSLGR BLEU from the V2S authors, or prepare P33 **[gap]** |
| **G4** | 17 | "T5 guarded hybrid" appears for S2V; the guard and T5 are V2S components (S2V's English step is rule-based) | see `s2v-defense-prep.md` F5 |
| **G5** | 6 | ASLG-PC12 "24,637 pairs"; the public set has 87,710 pairs (81,088 unique texts in this repo's split) | State the filter/subset that gives 24,637 **[deck]** |
| **G6** | 6 | WLASL "11,980 local files"; this repo indexes 2,000 glosses / 21,083 instances (videos not bundled) | Say "11,980 of the 21,083 instances were available locally" if that is the case **[deck]** |
| **G7** | 2 | 2.38 B / 1.54 B / 1.47 B: the repo's earlier review (`thesis-ch1-ch2-review.md` A3 #16) found the sensory-impairment, hearing-loss and vision-loss figures inconsistent with other chapters, and 1.54 + 1.47 > 2.38 unless groups overlap | Reconcile and cite the source on the slide |
| **G8** | 9, 25, 28 | Typos: "V2S **Mehodology**", "**fingersplled**", and the "−44pp" box wraps to "−44p / p" | one-word fixes |
| **G9** | 30 | Screenshot shows a Gradio `gradio.live` page and an instruction "press the **orange button**" — a blind user cannot see an orange button (P52) | Replace the screenshot with the deployed app, or say it is a prototype |

---

## 2. Panel questions and answers

### A. Opening and framing (P1–P8)

**P1 [Chair]** Summarise the thesis in two minutes.
→ Deaf signers and blind listeners share no channel; a human interpreter costs privacy and spontaneity. We built one application, two directions. **Voice → Sign:** Whisper large-v3-turbo transcribes; a linguistic rule engine writes a draft ASL gloss; a fine-tuned T5 refines it, guarded so it cannot change meaning; each gloss maps to a sign video, with fingerspelling as fallback. Hybrid beats neural-only by +5.33 BLEU-4 on 593 training pairs; end-to-end mean 2.05 s. **Sign → Voice:** MediaPipe landmarks → a causal GRU that scores every frame → a decoder with a next-gloss prior → English → speech; 76.3% on 250 isolated signs while streaming, gloss error rate 0.278 on synthetic sentences. Biggest limits: closed 250-sign vocabulary, synthetic continuous evaluation, seen signers, no user study.

**P2 [HCI]** Why a Deaf–blind pair? Couldn't they just type to a screen reader?
→ A literate Deaf user can, and that is a valid fallback. Our target is fluent ASL signers: ASL is a different language from English, many Deaf signers read English as a second language, and typing is slower and less spontaneous than signing. For the blind user, speech is the natural channel. We also separate this pair from *deafblind* people, who use tactile signing — a different problem. *(Honest caveat: no needs-assessment with either community is recorded in this repo.)*

**P3 [Chair, Ling]** The university is in Bangladesh. Why ASL and not Bangla Sign Language?
→ Data. ASL has GISLR (94,477 clips), WLASL, ASL-LEX, NCSLGR and ASLG-PC12; the BdSL work we reviewed uses small isolated sets (102 words, white background). The framework is language-agnostic by design — landmarks in, gloss out, a lexicon to render — so porting needs data, not a new architecture. BdSL is the most valuable next step (P79).

**P4 [Ling]** Is a gloss a valid intermediate representation?
→ It is a pragmatic bridge, not a model of ASL. A gloss (e.g. `YESTERDAY MY BROTHER ME GO STORE`) drops non-manual markers (eyebrows, mouth), spatial grammar (verb agreement, classifiers) and simultaneity. We use it because it is the standard interface of text-to-sign and recognition work and makes each stage measurable. Cost: facial grammar and space are not represented; slide 25 lists this as unresolved.

**P5 [Chair]** State the research gap precisely.
→ Reviewed systems are separate, one-directional tools; we found none that connects speech and sign in both directions between a Deaf and a blind user without a human intermediary. Hedge the wording: "to our knowledge, among the systems reviewed" — the literature review cites one (a sign-to-Braille system) that covers one direction of this pair.

**P6 [Sup]** What were your research questions, and did you answer them?
→ (1) Can speech be turned into usable ASL gloss under low-resource conditions? — yes with a rule-guided hybrid, +5.33 BLEU-4 (CI [+0.79, +9.12]), but with seed variance. (2) Can sign be recognised causally, frame by frame? — yes: 76.3% isolated top-1 with a streaming GRU, 0.278 GER on synthetic sentences. (3) Can the two run in one application at conversational latency? — V2S 2.05 s mean; S2V per-frame inference 0.06 ms plus ~27 frames of decoding delay. (4) Does it work for real users? — **not answered**; that is the next step.

**P7 [ML]** Is it really "bidirectional", or two systems behind one toggle?
→ Two independent pipelines behind one interface; the user toggles directions (slide 5). They share no model. "Bidirectional" describes the capability, not a joint model; turn-taking is manual and no two-person conversation study was run.

**P8 [Ling]** The title says "ASL recognition", but you recognise 250 signs. Is that recognition of ASL?
→ It is recognition of a 250-sign subset of ASL vocabulary, isolated and in synthetic sequences. It is not recognition of ASL as a language (no grammar, no fingerspelling, no non-manual markers). The 250 signs are early-childhood vocabulary from the PopSign game, so coverage of everyday sentences is low: only 47% of the gloss tokens in a 30-sentence test set. We state this as the first unresolved item (slide 25).

### B. Novelty and contribution (P9–P12)

**P9 [Chair]** What exactly is new?
→ (1) An integrated Deaf↔blind framework that runs in a browser, with S2V fully on-device. (2) A **rule-guided hybrid** text-to-gloss under low-resource data, with a meaning guard. (3) A **streaming** recogniser trained only on isolated signs: per-frame gloss + "no sign" + boundary outputs, a null-gated decoder, a trigram prior and a fixed-lag lattice. (4) **Evidence**: landmark-subset selection, a controlled five-architecture comparison, a state-reset analysis, ASL **phonology** features, and a diagnosis of the 75% ceiling — including negative results.

**P10 [Chair]** Which single contribution would you defend hardest?
→ The streaming S2V stack with its measurements, because every claim has a protocol (fixed split, selection vs evaluation signers, controls) and its limits stated. The V2S hybrid gain is real on average but statistically thin on 127 test pairs.

**P11 [ML]** Isn't this just stitching Whisper, T5, MediaPipe and a GRU together?
→ The components are standard; the contribution is the decisions and evidence around them: why rules + T5 instead of T5 alone in low-resource conditions (ablation), which 126 of 543 landmarks to keep and why, why a GRU rather than a bidirectional or convolutional model, how to turn per-frame scores into a clean gloss stream, and what limits accuracy. Negative results (CTC, window-based decoding, bigger models, Gemma 3 4B) are findings, not failures.

**P12 [ML]** Why should we trust results on synthetic sentences?
→ Don't trust them as proof of conversational accuracy; trust them as a controlled comparison. Synthetic streams have known boundaries and signers held out from decoder tuning, so every decoder, prior and model is compared on identical data. They overstate fluent signing (no coarticulation, template sentences). We say so on slides 18, 21 and 25, and live-camera probes show how far real conditions fall (P62).

### C. Literature and positioning (P13–P17)

**P13 [ML]** How do you compare with the state of the art on GISLR?
→ The Kaggle winning solution reports ~89%, but that is an offline model, a 4-seed ensemble trained on all 94,477 videos and scored on the leaderboard. Ours is a single streaming model on a fixed 80/20 split (76.3%). Not comparable; a streaming constraint costs accuracy. Our own offline reproduction of the winner diverged at epoch 15 twice and is unmeasured on the current split **[repo]**.

**P14 [Chair]** Why no comparison with existing commercial or published sign translation systems?
→ They target different settings (larger vocabularies, proprietary data, one direction) and are not reproducible on our split. We compared architectures and decoders under identical conditions instead. A head-to-head on a shared benchmark is future work **[gap]**.

**P15 [ML]** How does it relate to continuous recognition benchmarks such as PHOENIX?
→ PHOENIX is German Sign Language with real continuous video and sentence-level annotation; our continuous test is synthetic ASL built from isolated clips. The related literature motivated our design: a background (null) class for transitions, boundary prediction, and the finding that stitched sentences are too long and too clean (≈1.63× real length). We do not claim comparability.

**P16 [ML]** The literature says Transformers are superior. Why a GRU?
→ Standard Transformers attend over the whole clip and cannot stream. Among streaming-viable models the GRU was best (73.80% vs LSTM 72.86%, causal CNN 66.96%) with the fewest parameters (0.85 M). The BiLSTM is only 0.12 points ahead and needs the future. A causally masked Transformer was not tried (S2V bank Q36).

**P17 [Ling]** Is it true that nothing combines all of these areas?
→ Soften it: "among the systems we reviewed". Several reviewed works cover pieces (including sign-to-Braille); the claim is about the combination.

### D. Data (P18–P24)

**P18 [ML]** NCSLGR has 847 pairs and a 127-pair test set. Is that enough to conclude anything?
→ It is a deliberate low-resource condition and the primary evidence for choosing the V2S translator, but a 127-pair test set gives wide intervals: the paired-bootstrap CI for the BLEU gain is [+0.79, +9.12]. We therefore ran three seeds and a paired test, and report ASLG-PC12 (24,637 pairs) as a contrast. Conclusion held loosely: the hybrid helps on average in this regime **[deck]**.

**P19 [Ling]** ASLG-PC12 glosses are synthetic. What does BLEU 98.7 mean?
→ It means the model reproduces a rule system, not that it translates ASL well. ASLG-PC12's glosses are generated from Europarl text by rules (with `DESC-`/`X-` markers and `BE` kept), so near-ceiling scores show agreement with one synthetic convention. In this repo, the improved rule engine scores BLEU-4 36.4 against it *because* it differs from that convention. That is why it is the secondary, contrast dataset **[repo]**.

**P20 [Ling]** Are NCSLGR glosses real ASL?
→ NCSLGR is linguistically annotated real signing (1,888 utterances in the corpus) with English translations; our 847 pairs are derived from it. Human annotation is why it is the primary V2S evidence. Caveat: its annotation conventions differ from our rule engine's gloss style, which limits overlap-based metrics. Confirm the derivation and filtering with the V2S authors **[deck]**.

**P21 [ML]** Is there leakage between your T5 train/validation/test sets?
→ The split is 70/15/15 by pair (593/127/127). For ASLG-PC12 this repo's own split is by unique text because 6,587 texts repeat; for NCSLGR, confirm de-duplication of repeated English sentences before the defence **[gap — ask V2S authors]**.

**P22 [Ling]** GISLR has 21 signers and 250 early-childhood signs. How generalisable is that?
→ Limited, and slide 25 says so. The split is stratified by sign, not signer, so reported numbers are seen-signer; vocabulary is closed. The architecture and decoder are vocabulary-agnostic in design (a cosine head with enrolment), but expansion was not validated.

**P23 [Ling]** What are the licences and ethics of the datasets?
→ GISLR: Kaggle competition terms; WLASL: C-UDA (academic/computational use, no commercial use); NCSLGR/ASLG-PC12: research corpora. We do not redistribute WLASL videos. Scraping SignASL.org (the second retrieval tier) raises terms-of-use and copyright issues and should not ship in a public deployment; say it is a prototype fallback only **[repo]**.

**P24 [Chair]** Your two directions use different vocabularies. Doesn't that make the system asymmetric?
→ Yes. V2S can show any gloss with a WLASL video (2,000 glosses) or fingerspelling; S2V understands only 250. On 30 test sentences, GISLR's 250 glosses cover 56% of units and only 1 of 30 sentences fully. So a Deaf user can understand most of what a blind user says, but not the reverse. Expanding S2V vocabulary is the highest-value extension (P81).

### E. Voice → Sign (V2S) (P25–P42)

**P25 [ML]** Describe the rule engine.
→ Nine steps (slide 9): tokenise; negation (`not`/`n't` → `NOT`, drop `do`); sentence type from the first token; WH-questions move the WH word to the end, yes/no drop `do`; `will` → `FUTURE`; delete copulas; part-of-speech filter; lemmatise and uppercase. Example: "The appointment is on Friday morning." → `APPOINTMENT FRIDAY MORNING`.

**P26 [Ling]** Are the rules actually ASL grammar?
→ They approximate common ASL gloss conventions (wh-final questions, negation, dropped copula, `FUTURE`); they are not validated by a Deaf signer. This repo's audit found gaps — no time-word fronting, dropped conjunctions and particles (`IF`, `BECAUSE`, `TURN OFF`) — fixed in a v2 engine that scores BLEU-4 36.4 vs 26.3 for v1 on 2,000 ASLG-PC12 sentences. We say "approximates" and list signer review as future work **[repo]**.

**P27 [ML]** Why use both rules and T5?
→ Rules give a transparent, never-meaning-changing draft but are rigid; T5 learns conventions from data but with only 593 pairs it is unreliable alone. Feeding T5 both the English and the rule gloss makes the rule output a strong hint (slide 10). The ablation tests exactly this: neural-only vs hybrid, same split, architecture, metrics.

**P28 [ML]** How strong is the hybrid improvement?
→ Mean corpus BLEU-4 24.54 → 29.87 (+5.33); ROUGE-1 68.46 → 72.81; BERTScore (raw) 91.66 → 92.59, rescaled 50.57 → 56.11. Paired bootstrap 95% CI for BLEU **[+0.79, +9.12]** — excludes zero but wide. Across seeds the gain is +7.40, **−0.26**, +8.85: it helps on average and in two of three runs. Say that plainly **[deck]**.

**P29 [ML]** BLEU 29.87 is low. Is the system any good?
→ BLEU-4 is harsh on short gloss sequences where one wrong token breaks several n-grams, and gloss has several valid orders. ROUGE-1 of 72.8 says most tokens are right. These are overlap metrics against a single reference; the right claim is "the hybrid improves correspondence with the reference gloss on every metric", not "translations are correct" (slide 12 words it this way).

**P30 [ML]** Is BERTScore meaningful for glosses?
→ Only weakly. The embedding models are trained on English; glosses are English tokens in ASL order, so similarity is a loose proxy. We use it as supplementary evidence alongside BLEU and ROUGE; its improvement shows the gain is not confined to exact-match overlap.

**P31 [ML]** What alternatives did you try?
→ Transfer from ASLG-PC12 to NCSLGR (no gain over NCSLGR-only T5-small); controlled English-side augmentation (no gain); Gemma 3 4B with QLoRA (did not exceed hybrid T5-base). Interpretation: with 593 pairs, more data of a different convention, more source variation or more capacity did not help; explicit rule guidance did. (Know what the asterisk on slide 14's Gemma row refers to.)

**P32 [Ling]** Can the neural model change the meaning of a sentence?
→ Yes, and we measured it. On the team's 30-sentence test, T5 produced "Did you call your mother?" → `HE CALL MOTHER HE` and "Can you help me tomorrow?" → `HE HE HE …`. The shipped version is a **guarded hybrid**: a T5 output is accepted only if it keeps person, polarity and content words and has no repetition or invented words; otherwise the rule output is used. On those 30 sentences the guard kept 19 and rejected 11 — every meaning change **[repo]**.

**P33 [ML]** Is T5 worth it at all? Show me the rules-only baseline.
→ This is the hardest V2S question, so be exact. The deck's ablation compares neural-only vs hybrid; a rules-only row on NCSLGR is **not on the slides** **[gap]**. This repo's separate 30-sentence check found the unguarded T5 no better than its own rule input (BLEU 36.0 vs 39.3), and the guard recovers T5's useful behaviour (time-fronting, keeping `IF`/`BEFORE`, `TURN OFF`) while blocking its errors. Those references are drafts, so treat it as development evidence. Get the rules-only NCSLGR figure from the V2S authors before the defence.

**P34 [ML]** Why Whisper large-v3-turbo?
→ Mean latency on 30 sentences fell from 5.53 s to 2.96 s with near-identical transcripts; it is the largest single latency lever. The comparison is a consistency check, not an accuracy study (slide 8). This repo measured WER on synthesised speech: 1.0% clean, 3.1% at 0 dB for turbo, turbo 1.7× faster than large-v3 at equal accuracy **[repo]**.

**P35 [HCI]** How accurate is it on real, accented speech — for example Bangladeshi English?
→ Not measured. Our WER numbers are on synthesised speech, an optimistic bound; at 0 dB the model began hallucinating words ("Dhaka" → "Gaza"). Real recordings of the test sentences are the missing evaluation **[gap]**.

**P36 [ML]** Break down the 2.05 s latency.
→ Whisper accounts for ~43%. Optimisations: turbo (5.53 → 2.96 s), caching/parallelism/warm-up (→ 2.35 s), final FP32 T5 configuration (→ 2.047 s mean; median 2.023; P95 2.510; 98% of runs under 3 s). Say which hardware (GPU) the numbers come from; this repo's browser app instead calls Workers AI and runs T5 in the browser, so its latency profile differs **[deck vs repo]**.

**P37 [HCI]** Is two seconds acceptable in a conversation?
→ It is noticeably slower than spoken turn-taking, but comparable to the lag of a human interpreter, who typically works a few seconds behind the speaker (cite a source before claiming a number). It is a design target, not a user-validated threshold; no user tested it.

**P38 [Ling]** How good is the sign output? Is concatenated video understandable?
→ It is a limitation. Videos are looked up per gloss — WLASL first, SignASL.org second (cached), fingerspelling last — and played in order. Signers change between words, there is no coarticulation or sentence prosody, and facial grammar is absent. No Deaf viewer rated readability. Animated output is the stated future work (slides 25–26) **[gap]**.

**P39 [Ling]** What about numbers, names and long fingerspelled words?
→ Names and unknown words fall back to fingerspelling; digits are spelled out before glossing because a digit matches no sign (this removed the only ASR-induced gloss errors on clean speech). Long fingerspelled words are hard to follow (slide 25).

**P40 [ML]** What if a gloss is not in WLASL?
→ Tier 2 (SignASL.org, disk-cached), then tier 3 (fingerspelling). With WLASL-2000, 98% of units in the 30 test sentences were real signs, but ASLG-PC12/Europarl text would have 39% fingerspelled because of its vocabulary **[repo]**.

**P41 [HCI]** Did a Deaf person evaluate the V2S output?
→ No. Metrics compare glosses to references; nothing measures whether a Deaf viewer understands the sign video. A user-centred evaluation is the stated next essential step (slide 26).

**P42 [Sup]** What are the limits of the V2S evaluation?
→ Small test set (127 pairs), single reference per sentence, overlap metrics, no rules-only baseline on slides, T5 weights/training code from teammates (not re-run by one author), ASR evaluated only for consistency, no human judgement of sign video.

### F. Sign → Voice (S2V): the summary a panel needs (P43–P51)

(Full technical Q&A: `s2v-defense-prep.md` Q1–Q96. Q-numbers below point there.)

**P43 [Chair]** Explain the S2V pipeline.
→ Camera → MediaPipe Holistic (543 landmarks) → a 126–132-landmark subset in x–y → per-frame shoulder normalisation → causal GRU scoring 250 signs + "no sign" and a boundary signal each frame → decoder (null-gated, look-ahead lattice, trigram prior) → rule gloss→English → browser speech. (Q3)

**P44 [ML]** Headline results and the caveat on each?
→ Isolated: 76.3% top-1 (`gru_phono_raw`) — seen signers, model selected on the reported set (Q13–14). Continuous: GER 0.278 (C4) — synthetic sentences. Ceiling: error is generalisation, not semantics. (Q5)

**P45 [ML]** Is your test set independent of the signers you trained on?
→ No — split is stratified by sign, not signer; accuracy is seen-signer and unseen-signer accuracy is unmeasured. The GER "evaluation signers" are held out only from decoder tuning. (Q13)

**P46 [ML]** Is your reported accuracy free of selection bias?
→ For the five-architecture benchmark and continuous models, yes (selection on a train carve-out). For the isolated registry runs, including 76.3%, best-epoch selection used the same 18,896 clips, so the figure is mildly optimistic; size unmeasured. (Q14)

**P47 [ML]** Why 126 landmarks and no z?
→ Motion energy and a discriminability probe: ME-126 won (hands 42 + upper-body pose 8 + lips 40 + eyes/nose 36); about 92% of pose "motion" is z-axis noise. Early comparison 73.7% vs 70.6% for all 543 on the retired split. (Q23–26)

**P48 [Ling]** What does the phonology front-end add?
→ 84 fixed per-frame features (handshape, palm orientation, hand location, elbow angles) lift the best streaming GRU 75.2 → 76.3%; shuffling right handshape alone costs 44 points. It lives in the isolated model; the continuous phonology model failed (GER 0.587), so the deployed continuous model uses shoulder-normalised raw landmarks. (Q42–46, Q89)

**P49 [ML]** What did you learn about state reset?
→ On isolated-trained recurrent models, resetting at a true boundary cuts bleed-through 96–98%; but live boundaries are not oracle, and resetting after each commit hurt the continuous models. (Q51–52)

**P50 [ML]** How good is S2V on real camera input?
→ Unmeasured with real signers. The first real camera test failed on sentences; the continuous model C4 was built from measured causes and cuts the combined "live-like" stress error from 0.592 to 0.408 on synthetic perturbations. Hard-cut signing (0.634) and mirrored/swapped hands (0.962) still fail. (Q77)

**P51 [HCI]** If a new signer stands in front of a webcam today, what accuracy should I expect?
→ Below 76% and unquantified: unseen signer, different camera, continuous signing with coarticulation, 250-sign vocabulary. The only honest numbers are the synthetic probes; a study with new signers is required.

### G. System, accessibility and deployment (P52–P60)

**P52 [HCI]** How does a blind user operate the application? The slide says "press the orange button".
→ A prototype UI is visual; for a blind user it must be operable by keyboard/screen reader or voice. This is a design gap in the demo, not in the pipeline: output to the blind user is already speech. Fixes: push-to-talk key or voice activation, ARIA labels/live regions, audible state cues (listening / translating), tested with a screen reader **[gap]**. Do not claim the demo is accessible.

**P53 [HCI]** How does the Deaf user know the system understood them?
→ The app shows the recognised glosses next to the English text, uncertain glosses can be greyed, and speech is withheld unless every gloss is confident (planned in the web-app design; verify what ships before claiming it). No Deaf user tested the feedback design.

**P54 [HCI, Chair]** Where does computation run, and what about privacy?
→ S2V runs entirely in the browser (MediaPipe, a 3.48 MB TFLite step model, decoder in TypeScript): video never leaves the device. V2S needs ASR; in the repo's app that is Whisper on Cloudflare Workers AI, so audio is sent to a cloud service; a local GPU run avoids that. State which one the thesis describes (P55).

**P55 [Chair]** Which artifact is "the system": the Gradio demo or the web app?
→ Settle this with your team. The demo screenshot (slide 30) is a Gradio/Colab prototype of the V2S+S2V loop; the repo's deployed app is the browser implementation (S2V fully on-device, speech→gloss with the guarded T5). Name one as the reference implementation and the other as a prototype.

**P56 [HCI]** How does the app know who is speaking?
→ It doesn't: the user toggles the direction. Automatic turn-taking (voice-activity or sign-activity detection) is not implemented; a sign-vs-gap detector is scoped but not trained.

**P57 [ML]** What about noise, low light and clutter?
→ ASR: WER 1.0% clean to 3.1% at 0 dB on synthesised speech, with hallucination at 0 dB. Vision: robustness to aspect, scale, fps and jitter was tested synthetically; lighting and occlusion were not (slide 25 lists them as variance not controlled).

**P58 [Sup]** Cost and scalability?
→ On-device S2V has no per-user server cost. Hosted speech: on the Workers free plan a hosted voice costs ≈120 neurons per sentence (≈80 sentences/day), so the app defaults to the browser's own voices (free, ~0.2 s). Whisper via Workers AI is metered usage; not load-tested.

**P59 [ML]** What happens when the system is wrong?
→ Errors are spoken fluently, which is the real risk. Mitigations: confidence gating, showing glosses beside English, a precision mode that stays silent rather than guess. They reduce but do not remove misrecognition; a human fallback is required for medical, legal or safety-critical use.

**P60 [ML]** How reproducible is it?
→ Every training run is a registry entry with git commit, config hash, feature-cache key and environment; the split is fixed with its size asserted; checkpoints are public on Kaggle (MIT) with sha256 checks; 91 runs recorded, 52 flagged historical after a split reset. V2S T5 training code is with teammates and should be added to the release **[gap]**.

### H. Evaluation validity and statistics (P61–P67)

**P61 [ML]** Why gloss error rate and not human evaluation?
→ GER is an objective, automatic sequence metric (substitutions + deletions + insertions over reference signs) that lets decoders be compared quickly. It is not a measure of understanding; human evaluation by Deaf and blind users is the missing step.

**P62 [ML]** How big is the synthetic-to-real gap?
→ Measured only through proxies. Template sentences from isolated clips with synthetic gaps give GER 0.293 (C1); the same model under a live-like stress test scores 0.592, and C4 0.408. Real signing adds coarticulation, shorter signs and no pauses — hard-cut GER is worse still (0.546 C1; 0.634 C4).

**P63 [ML]** Are your comparisons statistically sound?
→ Architecture comparisons use one shared config and split but a single seed each; two same-config GRU runs differ by 0.0007, so gaps of several points are credible and gaps under ~1 point are not. V2S used 3 seeds and a paired bootstrap. Multi-seed repeats for the phonology gain and the decoders are future work (slide 25).

**P64 [ML]** Is the test set used for model selection?
→ See P46: partly for isolated registry runs.

**P65 [Ling]** Are BLEU/ROUGE the right V2S metrics?
→ They are the metrics of the text-to-gloss literature, so they allow comparison; they measure overlap with one reference. We also used BERTScore and a latency study, and in this repo a meaning-preservation guard that BLEU would not catch. Human judgement of sign video is missing.

**P66 [ML]** Is the ablation fair?
→ Same NCSLGR split, same T5-base, same metrics, three seeds, paired test examples (slide 13). The one confound to acknowledge: hybrid receives extra input (the rule gloss) by design — that is the point of the ablation, not a leak.

**P67 [ML]** Give me confidence intervals.
→ V2S BLEU gain: [+0.79, +9.12]. S2V: GER has no CI in the repo's headline tables (per-signer bootstrap exists in the decoder experiments); isolated accuracy is a single run with 18,896 test clips, so the binomial error is ≈ ±0.6 points at 76%, before seed variance.

### I. Ethics, community and society (P68–P75)

**P68 [Ling, HCI]** Were Deaf and blind people involved?
→ No participatory design or user study. This is the main limitation and the stated next step (slide 26). "Nothing about us without us": we would recruit Deaf ASL signers and blind users before any claim about usefulness.

**P69 [Ling]** Could this replace interpreters?
→ No, and it should not be presented that way: closed vocabulary, 250 signs, gloss-level output, no non-manual grammar. It might support low-stakes everyday exchanges where an interpreter is unavailable.

**P70 [Chair]** What harm could result?
→ Misrecognition spoken as fluent English; false confidence in medical, legal or emergency contexts; privacy exposure if audio or video leaves the device; deskilling or displacing human interpreters. Mitigations: on-device S2V, confidence gating, a human fallback, explicit scope.

**P71 [HCI]** Is the system fair across signers?
→ Unknown. 21 signers, with demographic and skin-tone distribution not examined; MediaPipe landmark quality varies with lighting, skin tone, clothing and camera. Per-signer accuracy would reveal it; it was not analysed for fairness.

**P72 [Ling]** Sign language is a language, not gestures. How do you reflect that?
→ We call our output gloss, not translation; we state it omits non-manual markers and spatial grammar; and we describe the vocabulary and corpus choices as limits. We also add ASL phonology (handshape, orientation, location, movement) from ASL-LEX to ground the features linguistically.

**P73 [Chair]** Data protection for camera and voice?
→ S2V: no video leaves the device. V2S: audio goes to a speech service in the hosted configuration. No recordings are stored by the application; state the retention policy of whichever ASR endpoint is used.

**P74 [Chair]** Dataset licensing risks in deployment?
→ WLASL is C-UDA (academic use): fine for the thesis, to be re-checked before public deployment; SignASL.org scraping should be dropped. GISLR terms apply to its data; our released models are MIT.

**P75 [Ling]** Terminology: why "Deaf" capital D, and how do you avoid "mute"?
→ Capital-D Deaf marks a cultural-linguistic community of signers; we use "Deaf signer" and "blind user", and avoid "mute" as inaccurate. We distinguish them from deafblind people (tactile signing), who are outside scope.

### J. Limitations and future work (P76–P81)

**P76 [Chair]** State your main limitations.
→ Closed 250-sign vocabulary; seen-signer evaluation; synthetic, template sentence evaluation; no coarticulation modelling; mirror/hard-cut failures; V2S test set of 127 pairs and no rules-only baseline on slides; no human evaluation; video output by concatenation, not animation; no non-manual markers; accessibility of the demo UI.

**P77 [Sup]** What would you do in the next six months?
→ (1) A user study with Deaf and blind participants. (2) Leave-signers-out evaluation and multi-seed repeats. (3) Real continuous data (recorded sessions) instead of stitched clips. (4) A larger vocabulary via enrolment/WLASL-scale lexicon. (5) Animated or pose-rendered output. (6) Accessibility pass on the UI.

**P78 [ML]** What result would make you abandon the approach?
→ If unseen-signer accuracy collapsed far below seen-signer accuracy, or live conversation tests showed users cannot recover from errors. Either would push toward more signer-diverse training data and confidence-aware interaction, not toward bigger models (a 4× larger GRU was 1–2 points worse).

**P79 [Chair]** How would you extend to Bangla Sign Language?
→ Reuse the landmark contract, per-frame model, decoder and gloss interface; collect or adopt a BdSL sentence and isolated-sign dataset with Deaf signers; replace the rule engine with BdSL-specific rules and the lexicon with BdSL videos. The main cost is data and linguistic expertise, not architecture.

**P80 [HCI]** Why video retrieval rather than avatars?
→ It was the fastest prototype with real signers. Its limits (signer changes, no coarticulation, licensing) motivate pose- or avatar-based rendering; this repo's plan is a pose lexicon built from GISLR so the recogniser can score the rendered output automatically. Not built.

**P81 [ML]** How would you make recognition open-vocabulary?
→ The continuous model has a cosine similarity head that supports enrolling new signs from a few examples; a model with 20 held-out glosses is trained, but the enrolment experiment (1/5/10 examples) is still to do **[gap]**.

### K. Team, process and reproducibility (P82–P86)

**P82 [Sup]** What was your individual contribution?
→ *Template — fill in with your own facts:* "I built the sign-to-voice recogniser and its evaluation: [dataset preparation (GISLR-Stratified), landmark-subset selection, model training and comparison, streaming decoder, the web implementation]. [Teammate names] built the voice-to-sign pipeline ([ASR, rule engine, T5, video retrieval]). We integrated through [shared gloss interface / the app]." Do not claim V2S results you did not produce.

**P83 [Sup]** What was the hardest technical problem?
→ Pick one with evidence: the **canonical split reset** (comparability across 52 runs); the **dilated CNN receptive field collapsing to 13 frames** (accuracy ~0.54 until fixed); **reset-after-commit hurting continuous models** after it helped isolated ones; or the **live-camera failure** that led to C4.

**P84 [Chair]** Which results were negative, and what did you learn?
→ CTC decoding (0.546 best, not competitive); windows fed to offset models (GER 0.52–0.78 vs per-frame 0.28); 4× larger GRU (worse); curated 922 engineered features (−17 points); semantic confusion as the ceiling (only ~10% of errors); V2S transfer learning, augmentation and Gemma QLoRA (no gain). Each narrowed the explanation: generalisation, not capacity or label noise.

**P85 [ML]** Can I run it myself?
→ Models are public on Kaggle (MIT); `uv sync` reproduces the environment; training is a notebook driving versioned packages with one config; the web app runs in a browser with `npm run dev`; every number maps to a registry run or report. V2S training code should be added alongside.

**P86 [Sup]** How did you control the quality of your work?
→ A fixed canonical split with asserted size, parity checks between the PyTorch model and the browser step model (≤1.4e-6), the incremental-vs-batch parity test (1.7e-6), per-run provenance, and generated documentation that fails a check if it drifts.

### L. Curveballs (P87–P92)

**P87 [Chair]** Is this a research thesis or an engineering project?
→ Both, deliberately. The engineering produces a working system; the research is the controlled comparisons and diagnoses (ablation of the hybrid; five architectures under identical conditions; subset selection; reset and decoder studies; ceiling analysis) and the honest negative results.

**P88 [Chair]** Please demonstrate it now.
→ Prepare for failure: use a prepared held-out sentence and the recorded fallback; run the demo on the same machine and browser tested beforehand; explain that live camera performance is the known weak point; start with V2S, which is the more reliable direction. Never improvise a live S2V signing demo as the first impression.

**P89 [ML]** What is the weakest claim in the thesis?
→ "Real-time, usable bidirectional communication." What is supported: V2S end-to-end ~2 s; S2V per-frame inference is fast and the pipeline runs in a browser. What is not: usability with real users, real continuous signing, new signers.

**P90 [ML]** Why not end-to-end video-to-text?
→ We wanted stages that can be measured, swapped and run on-device. End-to-end models need continuous, sentence-aligned sign data we did not have; stitching isolated signs was the available route. A single video-to-text model is listed as future work.

**P91 [Chair]** Define "streaming" versus "real-time".
→ *Streaming* means causal frame-by-frame inference with carried state and no lookahead; *real-time* means output keeps up with input at a latency users accept. Our recogniser streams and is fast per frame (0.06 ms), but total decision delay includes the decoder's ~27-frame lattice wait, and user-acceptable latency is untested.

**P92 [Chair]** If you could change one decision, what would it be?
→ Evaluate signer-disjoint from the start, and select models on a validation set separate from the test set. Both would have made the headline numbers more conservative and the thesis harder to attack.

---

## 3. Questions you should ask your team before the defence

1. **V2S:** the rules-only BLEU on NCSLGR (P33); what the 31.09 on slide 14 is (G1); the aggregation behind 29.87; the Gemma asterisk; de-duplication of NCSLGR (P21); where the T5 training code and checkpoint live.
2. **Latency:** which hardware produced the 2.047 s figure, and whether that is the Colab/Gradio path or the Workers AI path (P36, P55).
3. **Dataset numbers:** the filter that yields 24,637 ASLG-PC12 pairs and 11,980 WLASL files (G5, G6).
4. **Sources** for the 2.38 B / 1.54 B / 1.47 B statistics (G7) and for any interpreter-lag figure you quote (P37).
5. **Contributions:** who did what, in one sentence each (P82).

*Sources read:* the deck; `docs/reports/{speech-to-sign-audit,thesis-ch1-ch2-review,s2v-defense-prep,continuous-models,continuous-v2,sign-to-speech-downstream,plateau-diagnosis,five-arch-benchmark,phonology-models}.md`; `README.md`; `TODO.md`.
