# Iterative pipeline review findings (pending — not acted on)

**Status**: review-only. Not pre-committed, not reflected in PLAN.md /
MODEL_REVIEW_SPEC.md / TODO.md yet. Each finding needs agreement before
promotion.

**Scope**: systems-level critique of the rule-discovery pipeline (matcher →
review → synthesis → re-audit). Separate from the cleaner-level backlog in
`../NORMALIZATION_PENDING_IDEAS.md`.

## Resolution (2026-04-20)

Items below mapped to statuses in
[`NORMALIZATION_DESIGN_20260420.md`](NORMALIZATION_DESIGN_20260420.md):

| Item | Status |
|---|---|
| W1 Loop doesn't close | **resolved** — the normalize patch in `TODO.md` is the closure pilot |
| W2 No token-impact metric | **deferred** — retokenize is the test; no intermediate metric gates promotion |
| W3 Promotion thresholds ignore statistical weight (Wilson bounds) | **deferred** — improvement, not blocker |
| W4 Re-audit catches shrinkage but not over-cleaning | **deferred** — visible on retokenize |
| W5 Sampling policy under-designed | **partially resolved** — Task C uses stratified sampling by threshold bucket |
| W6 Human-review layer Phase 5 entirely TODO | **deferred** |
| W7 Rust matcher over-designed | **accepted** — current matcher is sufficient |
| W8 No budget, no cadence | **deferred** |
| M1 Regression-test set as first-class output | **deferred** |
| M2 Rule-interaction handling | **deferred** |
| M3 Reversibility taxonomy | **implicit** in the design doc (normalize vs strip vs page-drop) |
| M4 Matcher-recall estimate | **deferred** |
| M5 Taxonomy evolution path | **addressed** — rebase on `unified_classification_20260418` before wave11 |
| M6 Gemini version pinning | **addressed** — prompt redesign pins model + temperature in request manifest |
| M7a Corpus-source coverage gap (2/17 sources) | **resolved** — matcher rebase extends to all 17 sources |
| M7 Document-language guardrail | **off-scope** — strip by bigram inventory instead of gating |

## What the design gets right

- Manifest-driven stage contracts (`match_index.jsonl`, `manifest.jsonl`,
  `summary.json`) — enable reruns, provenance, incremental dev.
- Model proposes regex; deterministic layer validates — firewall against
  LLM-authored cleaner rules shipping unreviewed.
- Canonical target chosen first; model validates interchangeability —
  prevents model drift on what "the canonical form" even means.
- Three-level aggregation (category → family → rule) — right granularity for
  different decisions.
- Binary-first judgment order — correctly resists compound questions.
- "Keep separate for now" clauses for border/ascii-art and escaped-underscore
  chains — real risk management, not theoretical.

## Weaknesses — ranked by how much they'd bite

### 1. The loop doesn't close, and there's no minimum-viable-closure plan

Phases 7 (cleaner implementation) and 8 (re-audit) in `TODO.md` are
unchecked. wave10_v5 produced 1 cleaning + 1 normalization candidate; those
sit in JSONL files and nothing in `core_clean_text` consumes them. The plan
treats Phases 1–6 as independently advanceable, but each accumulates risk
that only closure reveals. Proposed fix: designate one category (e.g.,
`dot_leader_like`) as the **closure pilot** and traverse end-to-end before
starting wave11+ on anything else.

### 2. No token-impact metric anywhere

The stated goal (PLAN §2.1) is "reduce over-representation of repeated
layout structures in tokenizer vocabulary." Yet promotion is gated only on
`is_noise` / `preserves_semantics` / `interchangeable_with_target`. **Token
compression is never measured.** A rule could pass all thresholds and
produce zero or negative savings. Proposed fix: require a
`tokenizer_delta.json` artifact before promotion — bytes before/after,
Apertus token count before/after, delta per rule on a fixed held-out sample.

### 3. Promotion thresholds ignore statistical weight

"Family reviewed count ≥ 12, positive rate ≥ 85%" treats 12/14 = 85.7%
identically to 255/300 = 85%. Wilson lower 95% CI for the first is ~57%;
for the second ~80%. Same threshold, very different evidence. Proposed
fix: replace flat rates with Wilson lower-bound thresholds (e.g., ≥ 75% for
cleaning, ≥ 85% for normalization). Small samples then need stronger
agreement; large samples stay close to current rates.

### 4. Re-audit phase catches shrinkage but not over-cleaning

PLAN Stage 4 reruns the matcher and diffs counts. This tells us "did we
remove the artifacts" but cannot tell us "did we mangle legitimate text
*outside* the matcher's categories." A rule could eliminate `glyph_font_like`
hits while silently breaking paragraphs that never matched. Proposed fix:
maintain a **control set** of clean, human-curated paragraphs the matcher
does *not* flag; every cleaner change runs against it with bounded
divergence allowed.

### 5. Sampling policy is under-designed

Random 300/category with "edge cases" does not guarantee any given
`pattern_family` gets ≥ 12 reviews. Rare-but-important families stay
under-reviewed, triggering the "targeted second-round" escape (§10.5)
repeatedly — but the follow-up isn't budgeted. Proposed fix: stratified
sampling by family-size quantiles with a minimum floor of 12 per non-trivial
family. 300 becomes a budget, not a method.

### 6. Human-review layer (Phase 5) is entirely TODO but underpins the thresholds

If Gemini adjudicates alone, the 85/90 thresholds conflate "Gemini agrees
with itself" with truth. Inter-rater reliability between Gemini and a
human spot-check is never defined. Proposed fix: wire an always-on
spot-check (30–50 cases per category per wave) to compute a rolling
Gemini-human agreement rate; block promotion that round if it drops below a
floor.

### 7. Rust matcher is over-designed for a first traversal

Stage 1 targets 3 context units, multiple grouping keys, synthetic page
construction, full schema. The loop can advance with much less — local
line windows, one grouping key, no synthetic pages. Proposed fix: ship the
minimum matcher; expand only after the first re-audit reveals what's missing.

### 8. No budget, no cadence

"This loop should become the standard development cycle" — but per-loop
wall-clock, Gemini cost, and human-review hours aren't estimated. Proposed
fix: produce per-loop estimate for wave11 and track actuals; use to size
remaining categories.

## Genuinely missing pieces

### M1. Regression-test set as first-class output

Once a family promotes, its 12+ positives + sampled negatives should
become *permanent* regression tests for future cleaner changes. Implied but
not codified in the plan.

### M2. Rule-interaction handling

Multiple rules composed over the same text interact non-trivially. Each
rule is validated in isolation. Re-audit would catch a regression
empirically; **the policy for ordering rules and resolving conflicts is
absent.**

### M3. Reversibility taxonomy

Some rules are lossless (any normalization that produces canonically-
equivalent bytes); some are lossy (`glyph_font_like` → `TEXT_MISSING`).
Lossy rules deserve a higher evidence bar. The rule-class taxonomy (PLAN
§2.6) does not make the distinction.

### M4. Matcher-recall estimate

Reviewing 300 *matched* contexts tells us nothing about what the matcher
missed. A held-out human-labeled sample drawn from the **raw corpus, not
matcher output** is needed to bound recall. Without it, the 300-sample is
biased by the matcher's blind spots.

### M5. Taxonomy evolution path

Categories are frozen in the spec. If wave11+ surfaces a new artifact
class (e.g., ligature-like, Latin-lookalike homoglyph words), there's no
mechanism for proposing and validating a new category without re-freezing
the spec.

### M6. Gemini version pinning

Batch runs depend on model version + prompt. If Google rolls Gemini
forward, past promotions become non-reproducible. Proposed fix: pin model
IDs in the request manifest; include in audit trail.

### M7a. Corpus-source coverage gap (flagged 2026-04-17)

Sampling has only run on **2 of 17** GlossAPI source parquets: `openarchives.gr`
and `hplt`. The remaining 15 are untouched by the discovery pipeline:
`1000_prwta_xronia_ellhnikhs`, `AI-team-UoA__greek_legal_code`,
`Apothetirio_Kallipos`, `Apothetirio_Pergamos`, `dimodis_logotexnia`,
`Ekklisiastika_Keimena`, `ellinika_dedomena_europaikou_koinovouliou`,
`Ellinika_Keimena_Project_Gutenberg`, `eurlex-greek-legislation`,
`greek_phd`, `HuggingFaceFW__finewiki`, `klasikh_arx_ell_grammateia`,
`openbook_gr`, `opengov.gr-diaboyleuseis`, `Sxolika_vivlia`,
`Wikisource_Greek_texts`.

These sources carry materially different artifact distributions:
- Ecclesiastical and classical texts (`Ekklisiastika_Keimena`,
  `klasikh_arx_ell_grammateia`) are polytonic-heavy; the current spec was
  frozen against monotonic-dominant openarchives evidence.
- Legislative / official gazette sources (`eurlex-greek-legislation`,
  `opengov.gr-diaboyleuseis`, `AI-team-UoA__greek_legal_code`,
  `ellinika_dedomena_europaikou_koinovouliou`) have distinctive article-
  number, list-numeral, and reference-layout artifacts.
- Thesis repositories (`Apothetirio_Kallipos`, `Apothetirio_Pergamos`,
  `greek_phd`) have their own PDF-extraction quirks (math, citations,
  front-matter, bibliographic residue).
- Schoolbooks and literary (`Sxolika_vivlia`, `dimodis_logotexnia`,
  `Ellinika_Keimena_Project_Gutenberg`, `openbook_gr`,
  `Wikisource_Greek_texts`, `1000_prwta_xronia_ellhnikhs`) carry
  editorial-layout and OCR-era artifacts that differ from academic-archive
  noise.
- Wiki-origin (`HuggingFaceFW__finewiki`, `Wikisource_Greek_texts`) carries
  clean markdown but with different table / infobox / template residue.

Rules promoted from openarchives-only evidence risk:
- under-cleaning (missed family in another corpus surfaces as tokenizer
  noise),
- over-cleaning (legitimate-in-context pattern in another corpus gets
  folded because it resembles an openarchives artifact).

Proposed fix: extend wave11 sampling to all 17 sources with per-source
stratification. Sample budget per source can be smaller than 300 (e.g.,
50–100) as long as each non-trivial family gets a floor of reviewed cases
per source. The Rust matcher already takes a glob input; the change is
pipeline-runtime, not code.

### M7. Document-language guardrail

Denoising rules that assume "Greek-dominant" can over-clean multilingual or
predominantly-Latin docs. GlossAPI's `ScriptMetrics` already computes the
needed signal; the cleaner's rule application should be gated on it, not
applied uniformly.

## Top three if this were mine

1. **Make `dot_leader_like` the closure pilot.** Take its wave10 candidate,
   implement in `core_clean_text` with a flag, re-audit, publish a "first
   lap" report before anything else proceeds. Will uncover 80% of the real
   frictions.
2. **Add `tokenizer_delta.json` to the aggregator output.** Cheap (Apertus
   tokenizer over a fixed sample); changes promotion from "seems safe" to
   "demonstrably compresses."
3. **Replace flat rate thresholds with Wilson lower bounds; stratify
   sampling by family size.** Same plan, better statistics, no cost delta.

## Decision gate

For each finding above, the call is one of:
- Integrate into PLAN.md / MODEL_REVIEW_SPEC.md (spec update).
- Add as a TODO phase/item (work update).
- Defer indefinitely.

The call requires agreement with the iterative-pipeline direction — not to
be made unilaterally.
