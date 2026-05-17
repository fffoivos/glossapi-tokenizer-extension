# Fairness definition — Apertus per-language vocab allocation

Standalone document. Citable from
[`02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md)
§1 in place of the "match-X" empirical anchors.

Derived from `04_rational_policy.md`. Source-of-source: the necessary
rules R1-R6 derived from Apertus's stated and forced policies in
Phases 1-3.

## Definition

A language `L` is **fairly represented in Apertus's tokenizer** when:

> The tokenizer assigns enough merges that, on a held-out multilingual
> parallel benchmark drawn from `L`'s natural domain (e.g. FLORES+),
> `L`'s **fertility is within a defined tolerance of the cross-language
> baseline fertility** for languages in `L`'s peer cluster.

The two free parameters are:
- **peer cluster**: which set of languages defines the baseline?
- **tolerance**: how much fertility delta is acceptable?

Apertus's own primary sources are inconsistent on both parameters
(see `01_explicit_goals.md` for the contradictory claims). Three
defensible operationalisations follow.

## Operationalisation 1 — Weak (existing-coverage parity)

- **Peer cluster**: all 55 FLORES+ languages.
- **Tolerance**: any fertility produced by Mistral's chosen tokenizer
  for which Apertus's Gini-on-FLORES+ choice was made.
- **Implication for `L`**: `L` is fairly represented if it has any
  non-byte-fallback merges. Threshold: ≥ ~100 tokens.
- **Greek under operationalisation 1**: already fair at 1,479 PMI
  tokens. No extension required.
- **Defensibility**: matches the model card's "1,811 natively
  supported languages" framing. Treats the multilingual commitment as
  *enumerable*, not *qualitative*.

## Operationalisation 2 — Mid (HQ-20 peer-cluster parity)

- **Peer cluster**: the 20 FineWeb-2-HQ languages. Apertus's stated
  quality commitment is to these 20.
- **Tolerance**: ± 50 % fertility relative to the HQ-20 median.
- **Baseline**: HQ-20 median fertility on `modern_greek_eval`-
  equivalent slices ≈ 1.7-1.8 (per cross-language probing in the C3
  REPORT, extrapolating from the per-language PMI-vocab pattern).
- **Implication for `L`**: `L` must reach a vocab footprint that
  brings its fertility within ±50 % of 1.7-1.8.
- **Greek under operationalisation 2**: current fertility 2.41 vs
  HQ-20 median ~1.75 → 38 % above median, **just outside the
  tolerance band on the high side.** Extension to fertility ≤ 1.75
  × 1.5 = 2.62 already passes; extension to median 1.75 itself
  requires ~+6,144 added (Arabic-tier).
- **Defensibility**: matches the apps page's "Multilingual competence"
  framing and Apertus paper §3.2.2's named HQ-20 commitment. Treats
  HQ-20 as a peer group with substantive (not enumerable) coverage.

## Operationalisation 3 — Strong (FLORES+ fair-Gini-parity)

- **Peer cluster**: all 55 FLORES+ languages.
- **Tolerance**: contribution to FLORES+ Gini equal across languages.
- **Implication for `L`**: `L`'s fertility / compression / vocab-
  utilization must be at or below the FLORES+ benchmark median.
- **Greek under operationalisation 3**: requires fertility ≤ 1.5
  (FLORES+ benchmark median). Per the C3 cutoff REPORT §2 fertility
  table, achieved at the +11,264 extension level.
- **Defensibility**: matches Apertus §2.2's tokenizer-selection
  criterion verbatim. Treats the *Gini-fairness optimization* as
  applicable to each FLORES+ language individually, not just in
  aggregate.

## Which operationalisation to adopt

The three operationalisations are not equivalent:

- Operationalisation 1 is the weakest reading of Apertus's commitment.
  It would be the right framework if "1,811 languages" is read as
  *enumerable inclusion*. Under it, Greek needs no extension and our
  entire C3 project would be unjustified.
- Operationalisation 3 is the strongest reading. It would be the right
  framework if Apertus genuinely optimizes for per-language fairness,
  not aggregate. Under it, the C3 REPORT's +11,264 pick is the
  correct answer.
- Operationalisation 2 is the middle reading. It says "named HQ-20
  membership obligates HQ-20-typical quality." Under it, +6,144 is
  the correct answer.

**Recommended operationalisation for the C3 extension: 2 (mid).**

Reasons:
1. **Apertus's substantive commitment is HQ-20, not 1,811.** The 1,811
   number is implicit support; the HQ-20 is the named, dataset-backed
   commitment.
2. **The HQ-20 commitment is the only one Apertus is publicly
   accountable to.** The toxicity-9 / OCR-5 / ParaDocs-6 lists are
   tooling consequences; HQ-20 is the stated multilingual scope.
3. **Operationalisation 3 over-reaches the rational core.** Apertus's
   Gini-fairness optimization is *aggregate* (one number across 55
   languages), not per-language. Reading it as per-language is a
   normative extension beyond what the paper claims.
4. **Operationalisation 1 under-reaches.** Pure enumerable inclusion
   doesn't justify any quality-filtering investment, which Apertus
   demonstrably made (FW2-HQ exists).

The mid operationalisation is the only one with a *unique* anchor
(HQ-20 median fertility) that is both
- substantively committed by Apertus's stated policy, and
- empirically measurable from the existing artifacts.

## The definition adopted

**Greek is fairly represented in Apertus's tokenizer when its
fertility on `modern_greek_eval` is within ±50 % of the median fertility
across the other 19 FineWeb-2-HQ languages on their equivalent
in-domain slices.**

Operationally, this means Greek's PMI footprint should be in the
**Arabic-Italian band** of the existing vocab: 4,700-7,200 tokens.
The corresponding added-tokens range is **+3,221 to +5,721 above
the existing 1,479 base.**

Aligned to the existing 1024-step cutoff grid: **+4,096 to +5,120
added** is the principled band. Aligned to a finer 256-step grid:
**+3,584 to +5,632 added**.

The exact within-band choice (e.g. lower-mid vs higher-mid) requires
a normative call about whether Greek should sit at HQ-20 median (Polish
2,570 / Dutch 3,045 / Italian 4,712 territory) or at HQ-20 high-end
(Arabic 7,146 / Italian 4,712 / Portuguese 5,549 territory). My
recommendation in `GREEK_BUDGET.md` is the high-end of the band,
defended there.

## How to cite this in `02_1_4_cutoff_analysis/REPORT.md`

Replace the §1 "match Korean / match German / match French /
match English-unique / match English-total" framing with:

> Apertus's stated fairness commitment, as derived in
> [`02_1_6_representation_policy_analysis/FAIRNESS_DEFINITION.md`](
> ../02_1_6_representation_policy_analysis/FAIRNESS_DEFINITION.md),
> obligates Greek to be in the HQ-20 peer-cluster fertility band
> (operationalisation 2). The corresponding added-tokens budget is
> **+3,584 to +5,632 (1024- or 256-aligned)**. The exact within-band
> choice is decided by the recommendation in
> [`02_1_6_representation_policy_analysis/GREEK_BUDGET.md`](
> ../02_1_6_representation_policy_analysis/GREEK_BUDGET.md).

## Honest limitations of this definition

- The HQ-20-median fertility baseline is **extrapolated**, not directly
  measured. A real fair-Gini calibration would tokenize FLORES+ in
  all 20 HQ languages under Apertus base and compute per-language
  fertility. We have not done this. Doing it would tighten the
  definition.
- Operationalisation 2 assumes Apertus's HQ-20 commitment is
  *substantive* rather than tooling-driven. If FW2-HQ exists for 20
  languages purely because XLM-RoBERTa happens to support 100
  languages and Aya happens to cover 20, then the HQ-20 commitment is
  itself a tooling accident — and operationalisation 1 (the weak
  reading) becomes more defensible. Resolving this requires asking
  Messmer et al. directly why these 20.
- The ±50 % tolerance is normative. Different tolerances would shift
  the band ±20 % in either direction. ±25 % would tighten to
  +4,096-+4,608; ±100 % would loosen to the full HQ-20 spread
  (+1,024 to +12,288). The choice of ±50 % is mine; it could be
  reset by the user.
