# State of the research: a supervisor-style critique (2026-11-02)

**Purpose.** A reader who has not followed the project (a supervisor, a judge) should be able to
see what has been done, how much of it can be trusted, and what the true progress is. The report
is a Q&A: a supervisor's questions, the answers the repository can support, then a second and
third pass in which the answers themselves are attacked and refined.

**Basis.** `README.md`, `TODO.md` (state of 2026-09-29, the last experimental commit; one docs
commit followed on 2026-10-08), the 32 reports that existed in `docs/reports/` before this one, the registry (91 runs), the
code, and **new statistics computed for this report** from the committed per-clip predictions
(`docs/reports/assets/research-state/audit_stats.py`, output in `audit_stats.txt`). No model was
trained and no dataset downloaded. A claim is marked *verified* only if re-derived from a committed
file during this audit; everything else is "as recorded in the repository".

---

## 0. Bottom line

1. **What it is.** A browser-based, two-way Deaf ↔ Blind communication system. Sign → speech (S2V)
   runs on-device: MediaPipe landmarks → a causal GRU → decoding → English → browser voice.
   Speech → sign (V2S) is speech → English → ASL *gloss text*; **no sign output exists**.
2. **The strongest result** is a controlled, honest study of isolated-sign recognition on 250
   signs: the streaming `gru_phono_raw` reaches **76.3% top-1** (95% CI 75.7–76.9; top-5 90.7%;
   macro 76.1%; chance 0.4%). Adding hand-built ASL phonology features to raw landmarks beats raw
   landmarks by **+1.15 points** (paired CI +0.57…+1.69, McNemar p = 4·10⁻⁵, *single seed*).
   Architecture, depth and landmark-subset choices all move accuracy by less than 3 points.
3. **That number is not a clean held-out estimate.** The set it is scored on is also the set used
   to stop training, schedule the learning rate and pick the checkpoint, and the split is *not
   signer-disjoint* (all 21 signers appear in the test set and, the split being a sign-stratified random 80/20, in train too: inferred from the dataset's documented design and the 5 + 16 signer partition of the sentence benchmark, not checked against `train.csv`, which is not in this container). It is a within-signer, selected-on
   number. The optimism from selection is small (≈0.1–0.4 pt, measured), but signer overlap means
   it does **not** estimate accuracy for a new person.
4. **The headline sentence-level result is invalid.** The best continuous result (hybrid
   segmentation + `bilstm` + Viterbi, GER 0.177, 61.9% exact sentences) uses a BiLSTM trained on a
   *retired* split whose training set almost certainly contains the clips the sentences are built
   from (§4.1). The same model ships in the live "Record & recognize" mode. Treat 0.177 as
   unsupported until re-run with a current-split model (inference only).
5. **The rest of the sentence-level evidence is synthetic.** Test sentences are isolated clips
   stitched together with synthetic transitions, from a corpus written by Claude, scored on signers
   the model trained on. The deployed continuous model (C4: GER 0.278 on that benchmark) has **no
   measurement on real continuous signing, and no recorded real-camera check**.
6. **No human evaluation exists** (no Deaf or Blind users, no fluent-signer review of the sentence
   corpus or the gloss references; the references are drafts).
7. **Progress has stalled.** 49 commits landed 2026-09-25…29; since then one docs commit. The
   lever the project's own diagnosis ranks first for the isolated model (augmentation, TODO §7.4)
   has never been run; the sign-vs-gap detector is built but untrained.
8. **Fair summary:** a large, well-instrumented engineering system with an unusually honest
   negative-result record, resting on an evaluation that supports *"within-signer isolated
   accuracy ≈ 75–76% on 250 signs"* and little beyond it. Section 6 lists what to do in what order.

---

## 1. Definitions and metrics used in this report

| term | meaning here |
|---|---|
| **GISLR-250** | Google Isolated Sign Language Recognition: 94,477 clips, 250 signs, 21 signers, MediaPipe Holistic landmarks. Repacked as Kaggle `bracu23101281/gislr-stratified` |
| **canonical split** | that dataset's fixed 80/20 split: 75,581 train / **18,896 test**, stratified on sign only (**not participant-disjoint**, README §Datasets) |
| **top-k / macro** | top-k accuracy over 250 classes; macro = mean of per-class accuracy (classes are near-balanced, so macro ≈ top-1) |
| **val = test** | `sb.recognize.data` loads `test.csv` as *the* validation set; `train.py` uses it for early stopping, ReduceLROnPlateau and best-checkpoint choice; `sb-evaluate` then re-scores the same file. In every run `val_acc == overall_accuracy` (verified, registry) |
| **streaming** | causal: per-frame output depends only on past frames; verified by a step-API parity test (1e-6) and exported TFLite parity (6e-7…1.4e-6) |
| **GER** | gloss error rate = (substitutions + deletions + insertions) / reference signs, edit-distance aligned; lower is better; can exceed 1 |
| **sentence accuracy** | share of sequences decoded exactly |
| **oracle** | decoding with the true sign boundaries given |
| **legacy split** | the retired 90/10 split (9,448 val) used until 2026-09-16; 52 of 91 registry runs |
| **evidence grade** | **A** unbiased for its claim · **B** measured, reproducible, with a named bias · **C** synthetic data, draft references or in-distribution probes only · **D** built, not measured · **X** number invalid or very likely contaminated |

---

## 2. What has been done, and how far each part can be trusted

| component | headline | grade | why |
|---|---|---|---|
| Isolated recogniser, current split | `gru_phono_raw`/ME_134 **0.7632**; `gru` 0.7517; offline `bilstm` 0.7502 | **B** | selected on test set; 1 seed; signers overlap |
| Architecture / subset / depth ablations | GRU ≈ BiLSTM ≈ phono130 (differences ≤ 0.3 pt, p > 0.3); LSTM −2.7, 4× deeper GRU −2.9 vs champion (p < 10⁻²⁰) | **B** | same caveats; only large gaps are safe to read |
| Phonology front-end (+1.15 pt) | paired p = 4·10⁻⁵ | **B** | single seed; features designed by looking at this dataset |
| Exact ensembles | 3 streaming models **0.8048**, 4 models 0.8075 | **B−** | best of ≈56 combinations picked on the same test set; no CI computable from the saved files; not deployed |
| Training-free phonology (130 features) | 41.4% template / 54.3% 1-NN vs 3.1% / 13.0% raw | **B** | train → test, same signers; raw baseline is unnormalised (§3 Q9) |
| `recognition_2` rule engine (18 ASL-LEX parameters) | e.g. `ulnar_rotation` 0.855 | **B/C** | thresholds retuned on the evaluation set (report says so) |
| Continuous model C1/C4 | GER 0.293 / **0.278** (oracle 0.221) | **C** | synthetic stitched sentences, Claude-written corpus, seen signers |
| Next-gloss trigram prior | GER −6% (0.293 → 0.276) | **C** | prior fitted on the same generator's corpus |
| Live-robustness gains of C4 | `live_like` GER 0.592 → 0.408 | **C** | probes are the same perturbations C4 was trained to resist |
| Hybrid `bilstm` + Viterbi | GER 0.177, 61.9% exact | **X** | legacy-split classifier (§4.1) |
| Individual-sign mode, movement gating, record mode | deployed | **D** | stop constants are "first guesses"; no real footage |
| Sign-vs-gap detector (TODO §17) | corpus built (23,738 sequences, 12.6 GB) | **D** | not trained |
| 1st-place Kaggle recipe port | best surviving checkpoint 0.7459 (legacy split) | **D** | diverged at epoch 15 twice; recipe never measured |
| V2S: rules_v2 + guarded T5 | BLEU 36.4 vs 26.3 on ASLG-PC12 | **C** | reference glosses are themselves rule-generated; T5 is the team's |
| ASR (Whisper) | WER ≈ 1% down to 10 dB | **C** | synthesised speech only |
| Registry, provenance, checksums, generated docs | 91 runs, schema v4, sha256-verified Kaggle backups | **A** | infrastructure, verifiable |
| Paused / deprecated | POPSIGN extraction, TS extractor | – | see the obsolescence report |

---

## 3. Round 1: the supervisor's questions

**Q1. What is the research question, and what is the contribution?**
The repository states a goal, not a question: a *streaming* sign recogniser with low latency that
feeds sentence-level speech. Research questions exist only as a suggestion
(`thesis-ch1-ch2-review.md` §B.2). The defensible contributions are (a) a controlled comparison of
causal architectures and landmark subsets; (b) an ASL-phonology input representation and its
measured effect; (c) a method for training a continuous recogniser from isolated clips only
(per-frame gloss + null + boundary heads, D3 decoding); (d) a guarded text → gloss hybrid; (e) an
on-device deployment with exported-model parity. (e) is engineering; (a)–(d) are the research.

**Q2. What is your headline number, and can I trust it?**
**0.7632** top-1 on 18,896 clips, 250 classes. Verified here: 95% CI 0.7572–0.7694; top-5 0.9066;
macro 0.7614; median class accuracy 0.781; 4 classes under 50%. Trust is limited three ways:
(i) the test set also selects the checkpoint, LR schedule and architecture; (ii) one seed (all 91
runs use seed 42); (iii) signers overlap (Q3). The selection effect is measurable: best-epoch
accuracy minus the median of the last 10 epochs is +0.08 pt for the champion (+0.14…+0.43 for
the others). The bootstrap treats clips as independent; 21 signers contribute thousands each, so
the CI is a lower bound on uncertainty.

**Q3. Is evaluation signer-independent? A deployed system meets new signers.**
No. The split is stratified on sign only, and the sentence benchmark's 5 + 16 = 21 "signers" are
all of the dataset's signers, drawn from `test.csv`; every one therefore also appears in
`train.csv`. Hence the repo's phrase "16 held-out signers" means held-out *clips of seen signers*.
The project's own measurements show how large the signer effect is: same-gloss cosine 0.435 for
the same signer vs 0.195 for another; training-free template accuracy per signer ranges 16.6% to
58.8% (`asl-phonology-features.md`). No experiment reports accuracy on a signer withheld from
training. This is the largest gap between the reported numbers and the deployment claim.

**Q4. How does it compare to the state of the art?**
The Kaggle GISLR winner reports ≈ 0.89 on the competition's hidden leaderboard (the repo records it as
a 4-seed offline ensemble trained on all videos; the competition design, not this repo, makes that test
set participant-disjoint). The project's port of that recipe diverged twice and
its only surviving checkpoint (0.7459) was scored on the legacy split, so **no like-for-like
comparison exists**. The numbers are on different splits, and ours is the easier one, so the gap
to the leader is ≈ 13 points or more. What the project can say is internal: causality costs
nothing measurable (GRU 0.7517 vs BiLSTM 0.7502, p = 0.58).

**Q5. Is the phonology gain real?**
Likely, small. `gru_phono_raw` vs `gru`: 1,486 clips only the champion gets right, 1,269 only
the baseline (diff +1.15 pt, CI +0.57…+1.69, p = 3.9·10⁻⁵). That tests *sampling* noise of the
test clips, not *training* noise; two reruns of the same `gru` config differ by 0.07 pt, but both
share seed 42. Phonology alone (`gru_phono` 0.7010) is 6.2 pt worse, so the gain comes from
combining, not replacing, raw coordinates.

**Q6. The 0.8048 ensemble: what is it, and is it shippable?**
Three streaming GRUs (2.55 M parameters) with probabilities averaged. Each keeps its own state, so
it streams. But it is the best of every 1–4-member combination of six models (~56 candidates)
ranked on the same test set; the saved files hold only top-5 probabilities, so no CI or paired
test can be computed here. The app does not use it (it needs a continuous variant; P1 failed).

**Q7. How was continuous signing evaluated?**
GISLR-Sentences: 1,757 gloss-order sentences, instantiated one signer per sequence from
`test.csv` clips, with synthesised 2–10-frame transitions and 10–30 rest frames. C1 (GRU) reaches
GER 0.293 vs 0.507 for the best sliding-window baseline and 0.221 for an oracle; C4 0.278. The
benchmark is internally careful (same protocol for all arms; settings chosen on 5 signers, scored
on 16; parity checks).

**Q8. Does it work on a real camera?**
Not demonstrated. The first live test (2026-09-25) failed on sentences. Probes then isolated the
causes (15 fps → GER 0.504, jitter 0.516, landscape framing 0.651, mirrored hands 0.990) and C4
was trained against them. The shipped C4 has no recorded real-camera result. Latency is measured:
step model 0.06–0.20 ms/frame; MediaPipe Holistic itself ≈ 72 ms/frame (10 fps) on the test
machine, which is the real bottleneck.

**Q9. What do the phonology "no training" results show?**
That hand-built features carry far more per-number information than raw coordinates (41.4% vs
3.1% template top-1). But the raw baseline is unnormalised (translation/scale/face-dominated); the
report itself shows dropping the face lifts raw to 15.1%. A shoulder-normalised raw baseline is the
fair comparison and is missing.

**Q10. What exists for speech → sign?**
Speech → English (Whisper) → gloss (rules_v2, optionally T5 behind a meaning guard). Rendering is
deferred. Evaluation: BLEU 36.4 vs 26.3 on 2,000 ASLG-PC12 sentences; the guard keeps 19 of 30
recorded T5 outputs and rejects all meaning-changing ones.

**Q11. Is anything validated by Deaf or Blind users or fluent signers?** No.

**Q12. Is the work reproducible?** Mostly yes: 49 runs carry a provenance block (commit, config
hash, feature-cache key, environment), checkpoints are public on Kaggle with sha256 verification,
and docs/index are generated. One run reproduces bit-for-bit (three `gru` ME_126 reruns:
0.745025 each). The 42 pre-v4 runs are unrecoverable by design.

---

## 4. Round 2: attacking the Round 1 answers

### 4.1 Q7/Q8 hide a data-leakage problem in the best sentence result

Round 1 reported the hybrid's headline as if it were sound. It is not. The §16.2 chain uses run
**`1784447175`** (`bilstm`, ME_126, "0.7569 canonical"). Its registry record says
`split: stratified 90/10, n_val 9448, legacy_split: True`: it trained on 90% of all 94,477 clips.
The sentence benchmark is built from the *current* `test.csv` (20% of the clips, a different random
partition). Two independent random partitions overlap in about 90% of the test clips, so roughly
17,000 of the 18,896 test clips were in this model's training data.

The project has met this exact failure before: ten legacy runs scored on the current val set read
0.847–0.872 instead of ~0.75 (`phonology-models.md` §5). The fix there (`evaluate_run` refuses
legacy runs) does not cover the sentence notebooks. The symptom is present: oracle-boundary GER
**0.106** (≈ 89% per-sign accuracy), against 0.221 for the clean isolated oracle, was read in
`bilstm-wholeclip-eval.md` as "bilstm's classification is excellent". An isolated accuracy of 75.7%
on a clean split cannot produce that.

Consequences: B0 (0.106/0.089), the hybrid (0.189/0.177), and the "record-then-recognize is worth
building" recommendation are unsupported; the deployed Record mode ships this checkpoint; TODO's
"`bilstm` is the accuracy leader at 0.7569" is a legacy number (current-split `bilstm` is 0.7502,
level with `gru`). *Not yet verified directly*: the check is a set intersection of the two splits'
`uid` lists, or re-running the notebook with `1790347646` (current-split `bilstm` ME_126). Both
are cheap and inference-only.

### 4.2 Q2/Q5: "significant" was overstated; the claims are stratified by evidence

Claims, from the paired tests (all n = 18,896; `audit_stats.txt`):

| comparison | diff (pt) | 95% CI | p | reading |
|---|---|---|---|---|
| champion vs `gru` ME_132 | +1.15 | +0.57…+1.69 | 4·10⁻⁵ | probably real; seed variance untested |
| champion vs `gru_phono130` | +1.45 | +0.91…+2.04 | 5·10⁻⁷ | likewise |
| champion vs `lstm` | +2.66 | +2.12…+3.23 | 5·10⁻²¹ | safe |
| champion vs 4× deeper GRU | +2.89 | +2.31…+3.48 | 3·10⁻²³ | safe: capacity is not the limit |
| `gru` vs offline `bilstm` | +0.16 | −0.38…+0.71 | 0.58 | **indistinguishable** |
| `gru` vs `gru_phono130` | +0.30 | −0.28…+0.93 | 0.33 | **indistinguishable** |
| `gru` ME_132 vs ME_126 | +0.67 | +0.13…+1.19 | 0.012 | marginal; subset choice weak |

So "phonology alone matches raw landmarks on half the inputs" is supported, but "ME_132 beats
ME_126" and any ranking among the top four non-champion models is not. The clean statement is a
**plateau at 74–76% with a +1 point phonology gain**. Seed variance, which would tighten or widen
this, was never measured (all seeds = 42).

### 4.3 Q3/Q7: what "held-out signers" cost the continuous claims

Because signers are seen, the sentence GER cannot speak to new users. The probes in
`live-robustness` are in the same position: C4 was trained with augmentation for exactly the tested
perturbations (aspect, scale, fps, jitter, mirror), so `live_like` 0.592 → 0.408 shows the
augmentation works *on its own distribution*, not that real phones are covered. C4's remaining
failures are the realistic ones: hard-cut (no pause between signs) GER **0.634**, worse than C1's
0.546, and mirrored hands 0.962. Fluent signing has short or no gaps, which the continuous-models
report itself calls out; the 0.278 headline is the *favourable* case.

### 4.4 Q7/Q10: circularity in the language components

- The trigram prior and the test sentences come from one corpus (1,757 sentences "written by
  Claude", TODO §12.6 acknowledges the circularity). Under theme-out folds the prior's gain is
  smaller. It is an upper bound, as TODO says; the report quotes −6% without that label.
- ASLG-PC12 glosses are rule-generated from English. Scoring a rule engine against them measures
  agreement between two rule systems. The team30 references are drafts. The 132 gloss → English
  references are "unreviewed Claude drafts" (BLEU 60.2 is a development number).
- ASR was tested on synthesised speech only.

### 4.5 Q4: the comparison and the question the thesis should ask

Q4's "causality costs nothing" is itself a within-family result (RNNs only). It does not show that
the *best possible* causal model matches the best offline model, because the best offline model
(attention-based, with AWP and augmentation) was never trained successfully here. The honest claim
is narrower.

### 4.6 Q9/Q11: unfair baselines and absent users

The phonology-versus-raw comparison inflates the benefit (unnormalised baseline). The thesis
background (`thesis-ch1-ch2-review.md`) also claims, in places, things the repository does not
show (WLASL-trained, BiLSTM-deployed, "robust across signers"); those are paper-side fixes already
itemised there.

---

## 5. Round 3: remaining objections and the refined position

**Q13. Is the 17-point train-test gap, not the architecture, the real finding?**
Yes. Train accuracy at the best epoch is 93.6% against 76.3% on test (gap 17.3 points; BiLSTM
24.5). Every architecture lands in the same band, the confusable-pair ceiling is only worth
+2.7 points (`plateau-diagnosis.md`), and the top ranked remedy, augmentation, is unused: `gislr.training.json` sets `mirror_p: 0` and defines no
augmentation (only the 6 first-place-port runs carry an `augment` flag). The plateau is a data/regularisation problem
that has *not been attacked*, so "the isolated model has reached its limit" is not established.
Status: TODO §7.2/§7.4 untouched.

**Q14. Does the repository separate its evidence from its aspirations?** Partly. The registry flags
legacy runs automatically, and reports list caveats and negative results (P1, window ensembles,
the 1st-place divergence, the ASL-LEX-frequency blend). But the README leads with best numbers, the
leaderboard section and the §16 narrative disagree about which `bilstm` is "the leader", and there
is no single claims ledger. §6 below acts as one.

**Q15. Why trust the infrastructure numbers (latency, parity)?** They are the strongest
evidence class: TFLite/PyTorch parity 6e-7…1.4e-6; D3 online = offline on 300 held-out streams
(293 identical, 7 float near-ties, 0 unexplained). These are correctness checks, not accuracy
claims, and they hold.

**Final position.** The project has *built* a full pipeline and *measured* a within-signer
isolated-sign plateau carefully. It has **not yet measured** performance for a new signer, on
real continuous signing, on a real camera, or with a user. The headline sentence-level result must
be re-run before it is quoted.

---

## 6. What the thesis may claim now, and the order of work

**Claim ledger**

| may claim | may not claim (yet) |
|---|---|
| ≈ 75–76% top-1 within-signer on GISLR-250 with causal GRUs; phonology + raw gives ≈ +1 point | accuracy for unseen signers |
| causal GRU ≈ BiLSTM on this data (p = 0.58) | causal ≈ best offline model |
| continuous recogniser trained from isolated clips cuts GER from 0.507 to 0.293 / 0.278 *on synthetic stitched sentences* | real continuous-signing accuracy; any number from the hybrid (0.177) |
| exact streaming ensemble 0.8048 (best-of-56, selected on test) | that it generalises or is deployed |
| on-device pipeline with parity-verified exports | live-camera performance of C4, individual mode, record mode |
| guarded text → gloss rejects meaning-changing T5 output (on 30 sentences) | V2S works for real sentences; sign output exists |

**Critical path (each step is cheap relative to what it unlocks)**

1. **Verify and fix the leakage (inference only, hours).** Intersect the legacy split's val/train
   `uid`s with `test.csv`; re-run `gislr.3.streaming.bilstm-wholeclip-eval.ipynb` with the
   current-split `bilstm` `1790347646`; make `sb.recognize.sequences` refuse legacy-split
   checkpoints (as `evaluate_run` already does); replace the checkpoint in the web app; correct the
   TODO/README "leader" statements.
2. **Make the headline defensible (a day of GPU).** Carve a validation set out of `train.csv` for
   all selection; keep `test.csv` for one final score. Train ≥ 3 seeds for the champion and
   baseline; report mean ± SD.
3. **Measure signer independence.** Retrain `gru` and `gru_phono_raw` on a participant-disjoint
   split (the GapCorpus builder already contains a signer-disjoint 18/3 splitter) with
   leave-signers-out folds. This replaces the unsupported "robust across signers" claim with a number.
4. **Attack the generalisation gap** (TODO §7.4): augmentation (the C4 augmenter exists), the
   single largest untested lever by the project's own diagnosis.
5. **Real-world evidence:** one scripted session with real recordings of fluent signers (even 20
   sentences), reported with the same GER; plus a small usability pilot. State as a limitation
   until done.
6. **Then** finish the optional items (gap detector, ensemble in the app, custom signs).

Progress is best summarised as: **infrastructure and breadth ≈ complete; evidence validity ≈ the
bottleneck.** A second month of building would add little. Steps 1–3 would convert the existing
results into a defensible thesis.

---

## Appendix. Audit statistics (re-derivable)

`python docs/reports/assets/research-state/audit_stats.py` → `audit_stats.txt` / `audit_stats.json`.

| run | arch / subset | top-1 | 95% CI | top-5 | train − test at best epoch |
|---|---|---|---|---|---|
| `1790355555` | `gru_phono_raw` ME_134 | 0.7632 | 0.7572–0.7694 | 0.9066 | 17.3 pt |
| `1789559734` | `gru` ME_132 xy | 0.7517 | 0.7455–0.7581 | 0.9002 | 18.4 |
| `1790347646` | `bilstm` ME_126 xy (offline) | 0.7502 | 0.7442–0.7563 | 0.8978 | 24.5 |
| `1790413039` | `gru_phono130` FULL_543 | 0.7487 | 0.7429–0.7549 | 0.8961 | 18.5 |
| `1789558839` | `gru` ME_126 xy | 0.7450 | 0.7388–0.7513 | 0.8971 | 17.0 |
| `1790338279` | `lstm` ME_126 xy | 0.7366 | 0.7304–0.7427 | 0.8895 | 19.8 |
| `1790333188` | `gru_deep` ME_126 xy | 0.7343 | 0.7282–0.7405 | 0.8882 | 22.4 |
| `1790352947` | `gru_phono` PH_55 (phonology only) | 0.7010 | 0.6944–0.7077 | 0.8791 | 18.6 |

Registry: 91 runs; 52 legacy split; 39 current (33 canonical, 6 pending); every `hyp_seed` = 42;
the same-config reruns agree to ≤ 0.07 pt. All CIs treat clips as independent (a lower bound).
