# ASL-LEX 2.0 `SignData.csv`: full column EDA

**Question (user, 2026-09-29):** list every ASL-LEX feature, its value type and range, and how each is
calculated. No model trained — this is a pure EDA of the CSV already used by `sb.recognize.aslex` /
`sb.recognize.recognition2.rules` / `sb.recognize.recognition2.validate`, plus the published methodology
behind each column.

**Source**: `sb.recognize.aslex.signdata()` → the Kaggle mirror `bracu23101281/asl-lex` (`SignData.csv`,
resolved via `sb.core.paths.asl_lex_dir()`), a republish of OSF project `zpha4` (Caselli, Sehyr et al.).
**2,723 rows × 191 columns.** 90 of the 191 columns are per-morpheme repeats (`M2.2.0`…`M6.2.0`) of the
same 15 phonological-parameter columns, for signs with more than one sequential segment
(`NumberOfMorphemes.2.0`: 89% of signs are 1 morpheme) — collapsing those leaves **101 distinct fields**,
grouped below by what they describe. Types/ranges/NaN% are measured directly from the CSV; calculation
methods are the published ASL-LEX norming methodology (Caselli et al. 2017; Sehyr et al. 2021).

## 1. Identifiers / stimulus design (not features)

`EntryID`, `LemmaID` (2,719 / 2,663 unique strings — the dictionary keys `sb.recognize.aslex.gloss_entries`
matches GISLR glosses against), `Code` (unique per-row batch code), `Batch` (10 letters A–K, which norming
batch), `List`/`Item` (1–3, 1–128, counterbalancing slots in the norming survey).

## 2. Lexical/sign frequency — subjective ratings, 1–7 Likert scale

| field | type | range | how it's computed |
|---|---|---|---|
| `SignFrequency(M)` | float | 1.0–6.96 | mean of ASL users' 1–7 ratings of "how often do you see/use this sign" (a norming survey, same design as English word-frequency rating studies) |
| `SignFrequency(SD/Z/N)` | float/int | SD 0–2.46, Z −2.04–1.72, N 15–35 | SD/Z of the same ratings; N = number of raters |
| `...(M/SD/N/Z)-Native` / `...-Nonnative` | same | native N 7–18, nonnative N 10–20 | same ratings split by whether the rater is a native (ASL from birth) or non-native signer |
| `Unknown`/`Unknown(Native/Nonnative)` | float | 0–0.86 | fraction of raters who marked the sign "unknown to me" |
| `EnglishWF(lg10)` | float | 0.3–6.33, 12% NaN | log10 corpus word frequency of the sign's dominant English translation (an English-frequency norm, not sign-specific) |

## 3. Translation

`DominantTranslation` (string, 76% NaN — only populated when raters disagreed on the intended English
gloss), `DominantTranslationAgreement(±Native/Nonnative)` (float 0.08–1.0 = fraction of raters giving that
translation), `NondominantTranslations` (comma-list of the other guesses given).

## 4. Iconicity & transparency — separate rating tasks

| field | type | range | how it's computed |
|---|---|---|---|
| `Iconicity(M/SD/Z/N)` | float/int | M 1–7, N 4–140 | non-signers rate 1–7 "how much does this sign resemble its meaning," blind (don't know the meaning going in) |
| `Iconicity_ID`/`IconicityType` | string/category | 4 categories: `Arbitrary`, `Perceptual`, `Pantomimic`, `Both` | linguist classification of *how* it's iconic (looks like the referent vs. mimes an action) |
| `D.Iconicity(*)` | float | M 1.04–6.94, 64% NaN | the same iconicity rating task, from **Deaf** signers specifically |
| `GuessConsistency` | float | 0–3, 84% NaN | entropy (bits) of non-signers' free-response meaning guesses — 0 = everyone guessed the same thing, higher = guesses scattered |
| `GuessAccuracy` | float | 0–1 | fraction of those guesses that matched the true meaning |
| `Transparency(M/SD/Z)` | float | M 1.9–6.58 | a separate transparency-rating pass (how guessable the meaning is, rated after seeing it) |

## 5. Morphology / word class (linguist-coded, binary/categorical)

`LexicalClass` (7 categories: Noun/Verb/Adjective/Adverb/Number/Name/Minor), `Initialized.2.0` (0/1 —
handshape spells the first letter of the English word), `FingerspelledLoanSign.2.0` (0/1), `Compound.2.0`
(0/1 — built from two lexical signs), `NumberOfMorphemes.2.0` (int 1–6 — how many sequential phonological
segments the sign has; why the `M2.2.0`…`M6.2.0` column families exist, one full parameter set per
segment).

## 6. Timing (ms, from frame-by-frame video annotation)

`SignOnset(ms)` 67–1670, `SignOffset(ms)` 534–3100, `SignDuration(ms)` = Offset−Onset (200–2600),
`ClipDuration(ms)` 701–3740 (includes rest before/after the sign, not just the sign itself).

## 7. The 18 phonological parameters

Hand-coded by trained linguist annotators watching the citation-form video, per **Brentari's Prosodic
Model** (inherent/articulator features, place features, prosodic/movement features — same framework
`docs/reports/asl-phonology-features.md` §1.1 cites). Repeated once per morpheme (`.2.0`, `M2.2.0`, …,
`M6.2.0`):

| parameter | type | values / range | meaning |
|---|---|---|---|
| `Handshape.2.0` | categorical | 58 values (`1,3,4,5,7,8,a,baby_o,bent_1,...`) | the coded handshape name — derived from the 4 columns below |
| `MarkedHandshape.2.0` | binary 0/1 | — | is the handshape one of ASL's ~7 "unmarked" default shapes, or a rarer/marked one |
| `SelectedFingers.2.0` | categorical | 12 codes (`i,im,imp,imr,...` = which of thumb/index/middle/ring/pinky are active) | which fingers move/extend as a group |
| `Flexion.2.0` | categorical | 7: Bent/Crossed/Curved/Flat/FullyClosed/FullyOpen/Stacked | joint bend of the selected fingers |
| `FlexionChange.2.0` | binary 0/1 | — | does flexion change during the sign |
| `Spread.2.0` | binary 0/1 | 49% NaN | fingers spread apart or together |
| `SpreadChange.2.0` | binary 0/1 | 42% NaN | does spread change during the sign |
| `ThumbPosition.2.0` | categorical | Open/Closed | thumb extended or tucked |
| `ThumbContact.2.0` | binary 0/1 | — | thumb touches another finger |
| `SignType.2.0` | categorical | 6: OneHanded / SymmetricalOrAlternating / AsymmetricalSameHandshape / AsymmetricalDifferentHandshape / DominanceViolation / SymmetryViolation | Battison's one-/two-handed typology |
| `Movement.2.0` | categorical | 7: Straight/Curved/Circular/BackAndForth/X-shaped/Z-shaped/Other, 15% NaN | path shape traced by the hand |
| `RepeatedMovement.2.0` | binary 0/1 | — | is the path movement repeated |
| `MajorLocation.2.0` | categorical | 6: Head/Arm/Body/Hand/Neutral/Other | broad place of articulation |
| `MinorLocation.2.0` | categorical | 37 values (`Forehead, Eye, Chin, Mouth, TorsoMid, UpperArm, WristBack, ...`) | specific site within the major location |
| `SecondMinorLocation.2.0` | categorical | 37 values, 30% NaN | whether/where the hand moves *away* to (mostly `*Away` variants) by sign's end |
| `Contact.2.0` | binary 0/1 | — | hand touches the body/other hand at all |
| `NonDominantHandshape.2.0` | categorical | 56 values (+ `DominanceConditionViolation`, `SymmetryViolation`), 39% NaN | the passive hand's shape, when different/relevant |
| `UlnarRotation.2.0` | binary 0/1 | — | wrist rotates (pronation/supination) during the sign |

## 8. Per-value population frequency (`<Parameter>.2.0Frequency`)

Float 0–1 for each parameter — not per-sign, but **the base rate of that sign's specific coded value
across the whole 2,723-sign lexicon** (e.g. `SignType.2.0Frequency` 0.006–0.392: `OneHanded` is common,
`DominanceViolation` rare). This is a *lexicon* base rate, distinct from `SignFrequency(M)`'s *usage*
rating — it's what Part 3 of `bilstm-wholeclip-eval.md` tried blending into the n-gram prior and found
made results monotonically worse.

## 9. Phonological neighborhood / complexity

`Neighborhood Density 2.0` (int 0–50 — count of signs differing by exactly one parameter, the sign-language
analogue of phonological neighborhood density in spoken-word norms), `Parameter.Neighborhood.Density.2.0`
(int 1–563, a much coarser count — signs sharing *any* single parameter value, not a strict one-away
neighbor), `PhonotacticProbability` (float −1.27–0.91, mean ≈0 — standardized log-likelihood of the sign's
parameter combination given each parameter value's own lexicon-wide frequency, the multi-parameter analogue
of biphone probability), `Phonological Complexity` (int 0–6 — count of marked/non-default parameter values
in the sign).

## 10. Cross-references & acquisition

`SignBank*` fields (`AnnotationID`, `EnglishTranslations`, `LemmaID`, `SemanticField` — 12 categories,
`ReferenceID`): pointers into Gallaudet's ASL Signbank, not computed here. `InCDI` (Yes/No — is this on the
ASL "CDI" early-vocabulary checklist used with Deaf toddlers), `CDISemanticCategory` (21 categories, only
for `InCDI=Yes`), `bglm_aoa` (14–67, model-estimated age-of-acquisition in months, from a Bayesian mixed
model fit to CDI checklist data), `empirical_aoa` (9–60 months, the directly observed age at which a
criterion share of children were reported to produce the sign).

## Relevance to the feature-detector idea (raised earlier this session)

The parameters with any temporal/count structure at all are `Movement.2.0`, `RepeatedMovement.2.0`,
`Contact.2.0`, and `SecondMinorLocation.2.0` — and all four are exactly the parameters §14's rule engine
(`sb.recognize.recognition2.rules`) still scores near/below chance on. Every other parameter is one static
per-sign label with no per-instant ground truth, so a "detect at time t, count occurrences" framing has
nothing ASL-LEX-derived to validate against for those — only a single lexicon-level label per sign.
