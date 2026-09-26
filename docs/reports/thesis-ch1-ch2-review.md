# Thesis review: abstract, Chapter 1 and Chapter 2 against this repository

**Date:** 2026-09-26 · **Input:** the user's PDFs *Blind_Deaf_Communication* — abstract, chapter 1
(Introduction), chapter 2 (Literature Review). **Compared with:** every report in `docs/reports/`, `README.md`,
`TODO.md`, the deployed app (`apps/web`, https://signbridge.onecoder1.workers.dev) and its config.

Each discrepancy says what the paper states, what the repository shows, and a suggested fix. "Verify"
marks a point Claude could not check against a source (a citation's content, an external statistic).

---

## A. Discrepancies

### A1. Sign → voice (S2V): what the paper describes is not what was built

| # | the paper says | the repository shows | fix |
|---|---|---|---|
| 1 | Ch 2 end of §2.2: the sign model is **trained on WLASL** | Every recognition model is trained on **GISLR** (Google Isolated Sign Language Recognition, Kaggle 2023; repackaged as `bracu23101281/gislr-stratified`): 94,477 clips, **250 signs**, 21 signers, MediaPipe Holistic landmarks. WLASL is used only on the V2S side, as a video source. POPSIGN was tried and deprecated. | Say GISLR everywhere S2V training is mentioned; WLASL only for V2S video lookup. |
| 2 | Ch 1 §1.5: "recognition of discrete, Word level American Sign Language (WLASL) gestures" | Same as 1. Also, the deployed recognizer is **continuous**, not word-level: C1 (`gru_continuous`) outputs a gloss or "no sign" on every frame and a sign-boundary signal; a D3 decoder, a look-ahead lattice and a trigram next-gloss prior turn that into sentences (`continuous-models.md`, `sign-to-speech-downstream.md`). | Describe the continuous pipeline; "WLASL" is a dataset name, not a gesture type. |
| 3 | Ch 2 end of §2.2: robustness comes from "a **BiLSTM** classifier that captures both forward and backward context" | The deployed model is a **unidirectional (causal) GRU**. BiLSTM appears in the repo only as an offline accuracy reference: it needs the whole clip, so it cannot stream (`CLAUDE.md`, `five-arch-benchmark.md`). The abstract and Ch 1 correctly say GRU, so the paper contradicts itself. | Replace with the streaming GRU; if BiLSTM is mentioned, say it is an offline upper-bound baseline. |
| 4 | Ch 1 §1.5: "252 dimensional space coordinates" per frame | 252 = ME-126 × xy, an earlier subset. The deployed C1 reads **ME-132 × xy = 264** values (132 landmarks: 42 hand, 14 upper-body pose, 76 face: lips, eyes, nose). The best isolated model reads ME-134 (phonology front-end + raw xy). | Give the deployed number and name the subset; explain the landmark reduction (543 → 132, `motion-energy.md`, `subset-comparison.md`). |
| 5 | Ch 1 §1.5: data fed to the GRU "after **normalization of coordinates** and padding the sequences" | C1 reads **raw image coordinates** (missing → 0) with a per-frame LayerNorm; no coordinate normalization. Padding exists only in training batches; at run time the model steps frame by frame with no padding. Shoulder-centred normalization exists only in the phonology models and the untrained C4 (`continuous-v2.md`). | Describe what C1 does; present normalization as the v2 change and why (framing breaks C1, `live-streaming-gap.md`). |
| 6 | Ch 1 §1.5: videos processed "using the modular MediaPipe **Tasks** API … with Temporal tracking turned on" | GISLR's landmarks were made by Google with the **legacy** MediaPipe Holistic; the repo never extracted its training data. The web app uses the Tasks **HolisticLandmarker** (one model, VIDEO mode). "Temporal tracking" is not a setting anywhere in the repo. The Tasks-vs-legacy difference is a live-camera risk under investigation (TODO §12.8). | Say training landmarks come pre-extracted (legacy Holistic) and live landmarks from Tasks HolisticLandmarker; drop "temporal tracking" unless you can point to the setting. |
| 7 | Abstract: the GRU "classifies the gesture into a corresponding English word", then TTS | The pipeline is gloss **sequence** → English sentence (rule-based gloss → English, optional LLM on the edge) → the **browser's own voice** (`speechSynthesis`), spoken only when every sign is confident. | Update the abstract to the sentence pipeline. |
| 8 | Ch 2 §2.2: robustness "across diverse signers and environments" is **achieved** | Offline (synthetic sentences, 16 held-out signers): C1 GER 0.293 → **0.276** with the prior; the per-frame ensemble C1 + C2 **0.244**. But live, the user's own camera test found sentences mostly wrong (TODO §12.8). The probes show why: 15 fps GER 0.504, jitter 0.516, landscape framing 0.651, mirrored hands 0.990 (`live-streaming-gap.md`). There is no evaluation on real continuous signing yet. | Claim what was measured, on what; state the live gap as a limitation (it is a strong, honest result for Ch 4/5). |
| 9 | Ch 1 §1.3 / §1.6: existing systems fail under "network instability"; the paper's challenges list network latency and packet loss | S2V runs **entirely on the device** (MediaPipe + LiteRT.js step model 0.06 ms/frame + decoder in the browser); no network is involved after loading. V2S **does** need the network: ASR is Whisper large-v3-turbo on Cloudflare **Workers AI** (`/api/asr`). | Turn this into a design contribution (on-device S2V: privacy, no network); state V2S's network dependence. |
| 10 | Ch 1 §1.6 challenge 3 / §1.4: language-model refinement in S2V | S2V has no neural language model by default: a Kneser-Ney **trigram** next-gloss prior (−6% GER) and rule-based English; an LLM for English is optional (edge only). The neural model (T5) is on the V2S side. | Put each language component on the right side. |

### A2. Voice → sign (V2S)

| # | the paper says | the repository shows | fix |
|---|---|---|---|
| 11 | Abstract / Ch 1: the rule-based draft is "refined by a fine-tuned Hybrid T5 … to generate the final ASL gloss" | The audit found the team's T5 **changes meaning** on some sentences ("Did you call your mother?" → `HE CALL MOTHER HE`; "Can you help me tomorrow?" → `HE HE HE HE …`) and on the team's 30 sentences scores no better than its own rule input (BLEU 36.0 vs 39.3). The deployed engine is a **guarded hybrid**: a T5 output is accepted only if it keeps person, polarity and content words, otherwise the improved rules (`rules_v2`) are used; 19/30 kept, 11 rejected, every meaning change rejected (`speech-to-sign-audit.md` §2, §9). | Describe the guard; T5 "refines" only when it passes it. This is a contribution, not a weakness. |
| 12 | Ch 1 §1.5: audio at **16 kHz mono or a WAV file** into Whisper large-v3-turbo | The team's Colab notebook used faster-whisper **large-v3** locally with Silero VAD. The deployed app records in the browser and calls **large-v3-turbo** on Workers AI. The repo measured the two: equal accuracy, turbo 1.7× faster (WER 1.0% clean, 3.1% at 0 dB, on *synthesized* speech only). | State which system the chapter describes; justify turbo with the measurement; say the WER is on synthesized speech (no real recordings yet). |
| 13 | Ch 1 §1.5 / Ch 2: gloss → video from WLASL, then **SignASL.org**, then fingerspelling, "displayed immediately … in real time" | That is the Colab notebook. The deployed page outputs **gloss** only; video rendering was set aside by the user's decision (2026-09-24, `speech-to-sign-audit.md` status). Scraping SignASL.org raises terms-of-use issues; WLASL is under the C-UDA licence (computational use). | Either say rendering is a prototype (Colab) or deploy it; address licensing; consider the pose-lexicon renderer (audit §7.2). |
| 14 | Ch 2 §2.2: the rule engine converts English "into proper ASL grammatical structure ensuring the output is linguistically meaningful" | On ASLG-PC12 (2,000 test sentences) `rules_v2` scores BLEU-4 36.4 / gloss WER 0.303, `rules_v1` 26.3 / 0.347; ASLG itself is a synthetic, English-order convention. No Deaf signer has reviewed any output; the 30-sentence references were drafted by Claude. | Soften to "approximates ASL gloss conventions (topic fronting of time words, wh-final questions, negation, dropped copula)"; report the numbers; list signer review as future work. |
| 15 | Abstract: "Each gloss token is first looked up in the Word-Level … (WLASL) dataset." (sentence ends there) | The abstract never says what happens next (video, fingerspelling, or gloss text). | Finish the V2S description. |

### A3. Internal inconsistencies and citations

| # | where | issue |
|---|---|---|
| 16 | Ch 1 §1.1 vs §1.2 vs abstract | §1.1: 2.38 B sensory impairment, 1.54 B hearing loss, **1.47 B vision loss**; §1.2 and abstract: 1.5 B hearing loss, **43 million blind** (both attributed to WHO 2021). 1.54 + 1.47 = 3.01 B > 2.38 B unless the groups overlap; "vision loss" and "blindness" differ by orders of magnitude and should be named separately. The 43 million blind estimate comes from the Vision Loss Expert Group / GBD (Lancet Global Health, 2021), not WHO's hearing report. **Verify** [1], [10], [11] and use one set of figures with correct attribution. |
| 17 | Ch 2 end of §2.2 vs [6] reviewed earlier | The chapter says **no** reviewed system targets the Deaf ↔ Blind pair, but [6] translates Indonesian signs into **Braille for Blind users**, i.e. one direction of that pair. Say "no reviewed system is *bidirectional* between Deaf and Blind users". |
| 18 | Ch 2 §2.2 "no previous research that combines all of the areas" | Too absolute for a literature review. Use "to our knowledge, among the systems reviewed". |
| 19 | Ch 1 §1.1 "LSTM … performing well in **continuous** gesture recognition [8]" | [8] (as summarised in Ch 2) is 19 **isolated** signs. The citation does not support "continuous". |
| 20 | Ch 1 §1.3 "fail to operate under … network instability [3]" | Ch 2's summary of [3] (BDSL, 102 isolated words, white background) says nothing about networks. **Verify** [3]. |
| 21 | Ch 2 §2.3 "Latency and Processing: browser-based implementations … disrupt the flow … compared to locally processed pipelines [5], [7]" | The proposed system is **itself browser-based**, but processes locally in the browser (WASM). Distinguish "browser UI + server inference" ([5]) from "in-browser, on-device inference" (this work), or the chapter argues against its own design. |
| 22 | Ch 2 §2.1 vs §2.2 | "MDS hashing" (§2.1) vs "MD5 hashing" (§2.2). MD5 is right. |
| 23 | Ch 2 §2.3 point 1 | Frames Transformers as "superior", while the system uses a GRU. Add the reason: the live system must **stream** (a causal model with per-frame state; the GRU confirms a sign 1 frame after it ends), and the Transformer designs cited are offline (whole-clip attention). The repo's 1st-place-style Conv1D-Transformer port is offline-only and its training recipe diverged, so it has no comparable score on the current split (README); do not quote its 0.7459, which is on the old 90/10 split. |
| 24 | Terminology | "Deaf or **mute**" (abstract): "mute" is considered inaccurate and offensive; use "Deaf" or "non-speaking". Capital-D *Deaf* (cultural) vs *deaf* (audiological): pick one convention and define it. Also distinguish the target pair (a Deaf person and a Blind person) from *deafblind* people (tactile signing), so readers don't assume the latter. |
| 25 | Typos | Abstract "steaming GRU" (streaming), "without intervention bidirectional"; Ch 1 §1.4 "between individuals and individuals"; §1.2 "Rational" (Rationale); Ch 2 "withdraw spatial features" (extract), "elther", "cannon", "incompetent for the blind" (inaccessible), "three-l.tier", "disposal capability" (deployment), "I3D failed" (dangling). |
| 26 | Figures 1.1 / 1.2 | Should match the deployed system (see A1–A2). A current S2V figure: camera → MediaPipe HolisticLandmarker → 132 landmarks (xy) → C1 streaming GRU (per-frame gloss / no-sign / boundary) → D3 + look-ahead lattice with trigram prior → gloss → English rules → browser voice. V2S: mic → Whisper large-v3-turbo (Workers AI) → rules_v2 → guarded T5 (browser, int8 encoder + fp32 decoder) → ASL gloss (→ sign video, if kept). |

---

## B. What to add to Chapter 1

1. **Contributions**, as a numbered list with the headline numbers (reviewers look for this first):
   - a bidirectional Deaf ↔ Blind system that runs in a browser; S2V fully on the device;
   - a **continuous, streaming** sign recognizer trained without any sentence-level data: per-frame
     gloss + "no sign" + boundary outputs; GER **0.293** → 0.276 with a next-gloss prior → **0.244** with a
     two-model per-frame ensemble, vs 0.507 for the best sliding-window baseline and 0.221 for an oracle
     given the true sign boundaries (16 held-out signers);
   - **GISLR-Sentences**, a synthetic continuous test set composed from isolated clips (one signer per
     stream, held-out signers), and the protocol (5 selection / 16 evaluation signers);
   - isolated-sign results on the canonical GISLR split (18,896 videos): streaming `gru_phono_raw`
     **0.7632**; exact ensembles of streaming models 0.8048 (3 models);
   - an ASL **phonology** feature extractor (130 features: handshape, orientation, location, contact,
     movement, two-hand relation, non-manual) that recognizes 41–54% of 250 signs with no training
     (`asl-phonology-features.md`);
   - a **guarded** text-to-gloss hybrid that rejects meaning-changing neural output (§A2 #11);
   - public release of all models on Kaggle Models (MIT).
2. **Research questions** (and, if the department expects them, hypotheses), e.g. *RQ1:* can a streaming
   model trained only on isolated signs recognize continuous signing? *RQ2:* what does each stage
   cost (recognition, decoding, language prior)? *RQ3:* how robust is it to live-camera conditions?
   *RQ4:* can neural gloss refinement be used safely?
3. **Definitions** early in the chapter: ASL gloss (and that it is not English), isolated vs continuous
   sign recognition, streaming/causal model, co-articulation, gloss error rate (GER) and WER, Deaf/deaf,
   Deaf/Blind vs deafblind.
4. **Scope, stated precisely:** recognition vocabulary = GISLR's 250 signs (early-childhood vocabulary
   from PopSign; covers 47% of gloss tokens in the team's 30 test sentences, `speech-to-sign-audit.md`
   §8), English only, continuous evaluation on synthetic sentences, desktop/phone browser, no facial
   grammar (non-manual markers) beyond what landmarks carry.
5. **Ethics and data:** on-device processing as a privacy property; dataset licences (GISLR competition
   terms, WLASL C-UDA, SignASL.org terms); the absence of Deaf/Blind user testing as a stated limitation
   and a future step.
6. **Thesis organization:** one paragraph mapping Ch 3 (methodology: data, models, decoding, deployment),
   Ch 4 (results), Ch 5 (conclusion, limitations, future work).
7. **Updated pipeline figures** (§A3 #26).

## C. What to add to Chapter 2

1. **Restructure §2.2** into subsections: isolated sign recognition; continuous sign recognition;
   sign → speech systems; speech → sign (text-to-gloss + sign rendering); bidirectional systems; ASR and TTS.
   End with a **comparison table** of the reviewed systems (dataset, vocabulary, signers, isolated or
   continuous, direction, modality, accuracy), then the gap statement.
2. **Datasets** (a subsection or table): GISLR (Kaggle 2023, 94,477 clips, 250 signs, 21 signers,
   phone recordings, Holistic landmarks), WLASL (2,000 glosses), ASL Citizen (83,399 webcam videos, 2,731
   signs, 52 Deaf/HoH signers), How2Sign (80 h continuous ASL, English sentence alignment), ASL-LEX 2.0
   (2,723 signs with phonological coding), ASLG-PC12 and NCSLGR (text-to-gloss), PHOENIX-2014T (the
   continuous-SLR benchmark, German Sign Language).
3. **Landmark-based isolated recognition at scale:** the GISLR Kaggle competition and its winning approach
   (normalized landmarks, 1D convolutions + Transformer, heavy augmentation) as the baseline this work
   builds on and departs from (streaming constraint).
4. **Continuous and online recognition:** CTC-based CSLR; online CSLR with a background class for
   co-articulation (Zuo et al., EMNLP 2024: online WER 38.4% → 22.1% with clips cut from real continuous
   video); sign spotting with a background category (pose-based spotting, 2025); recognizing continuous
   signing with models trained only on isolated data (boundary-sensitive losses, 2026); and the evidence that
   stitched isolated signs are unrealistic (BRAID, 2026: linear-interpolation sentences are 1.63× the real
   length). These frame the null class, boundary head and synthetic-sentence design of Ch 3.
5. **Sign-language phonology** as a representation: Stokoe's parameters, Battison's constraints,
   Brentari's prosodic model, ASL-LEX 2.0; learned phonology subspaces (PhonSSM, 72.1% on WLASL-2000).
6. **Text-to-gloss translation and its evaluation:** rule-based vs neural (ASLG-PC12, T5-style models),
   BLEU/chrF/gloss WER, and the known failure mode of fluent but meaning-changing output (motivates the
   guard).
7. **Sign production:** video retrieval (the team's WLASL/SignASL approach) vs pose- or avatar-based
   rendering (`spoken-to-signed-translation`, CWASA/SiGML) and fingerspelling; trade-offs in readability,
   vocabulary and licensing.
8. **Language priors in recognition:** n-gram/next-gloss models fused with recognizer scores (shallow
   fusion), and their limits (they re-rank; they cannot recover missed signs).
9. **On-device / in-browser ML:** MediaPipe Tasks, LiteRT.js, ONNX Runtime Web; latency and privacy
   (resolves §A3 #21).
10. **Landmark extractors:** MediaPipe Holistic vs separate pose/face/hand models, detection rates and
    hand-label conventions; the repo's landmark test page gives first numbers (Holistic 10.2 fps vs three
    separate models 8.7 fps at 1080p CPU, hand labels agreeing with the body 100% on a test video).
11. Name this work's position explicitly after the review (what is new: bidirectional Deaf ↔ Blind,
    streaming continuous recognition from isolated data, on-device, guarded text-to-gloss), with the
    hedged wording of §A3 #18.

## D. Numbers the paper can quote (sources in this repo)

| claim | number | source |
|---|---|---|
| best streaming isolated model, canonical GISLR split | 0.7632 (`gru_phono_raw`, ME-134) | `phonology-models.md` |
| best raw-landmark streaming model | 0.7517 (`gru`, ME-132 xy) | same |
| exact ensemble of 3 streaming models | 0.8048 | `phonology-models.md` §8 |
| phonology features, no training | 41.4% template / 54.3% 1-NN (250 classes) | `asl-phonology-features.md` |
| continuous: C1 + D3, 16 held-out signers | GER 0.293 (oracle 0.221, sliding window 0.507, reset loop 0.659) | `continuous-models.md` |
| + trigram prior | 0.276 | `sign-to-speech-downstream.md` |
| + second model averaged per frame | 0.244; no-pause 0.446 (C1 0.542) | `window-ensembles.md` |
| windows fed to offset models (1 / 2 / 3 s) | GER 0.523 / 0.687 / 0.777 | `window-ensembles.md` |
| browser = Python | 24/24 held-out streams identical; recognizer 0.06 ms/frame | `sign-to-speech-downstream.md`, apps/web README |
| live-camera probes | 15 fps 0.504, jitter 0.516, landscape 0.651, mirror 0.990 | `live-streaming-gap.md` |
| text-to-gloss, ASLG-PC12 test (2,000) | rules_v2 BLEU-4 36.4, gloss WER 0.303 (rules_v1 26.3 / 0.347) | `speech-to-sign-audit.md` §9.1 |
| guard on the team's T5 (30 sentences) | 19 kept / 11 rejected; every meaning change rejected | same §9.2 |
| ASR on synthesized speech | WER 1.0% clean, 3.1% at 0 dB (turbo); turbo 1.7× faster than large-v3 | same §10 |
