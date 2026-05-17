# 02.1.6 Representation Policy Analysis

## Scope

Derive Apertus's implicit per-language representation policy — what
rules actually produced each language's vocab and pretraining-data
share — separate the *necessary* parts of that policy from the
*accidental* parts, state the rational core as a normative
**definition of fairness**, and apply it to Greek to derive a
principled vocab-budget recommendation.

**Status (2026-05-18 archive mode)**: This sub-subproject is now in
**archive mode**. The C3 cutoff decision was made empirically in
[`../02_1_7_intrinsic_eval_sweep/`](../02_1_7_intrinsic_eval_sweep/)
on 2026-05-18 (chosen cutoff: **17,408 added units, curated +
backfilled**) via the TokEval multi-metric in-domain sweep. The
policy-archaeology path this subproject pursued did not produce a
budget; what it did contribute is documented below.

### What this subproject net-produced

- **Provenance finding** ([`11_tokenizer_provenance.md`](11_tokenizer_provenance.md))
  — Apertus inherited Mistral's BPE table verbatim except for
  special-token-block changes + 486 trailing-BPE truncations. This
  reframed the question: per-language vocab allocation is
  Mistral-side, not Apertus-side.
- **Speaker-count hypothesis INVALIDATED**
  ([`04_evidence_speakers.md`](04_evidence_speakers.md)). Closed a
  dead-end framing.
- **Discovery of `swiss-ai/tokenizer-intrinsic-evals`** (the TokEval
  suite Apertus actually used), which seeded `02_1_7`. The
  experimental thread moved there.
- **Reference evidence** (HPLT 3.0 baseline, FLORES+ 55-list,
  Latin-vs-Ancient-Greek asymmetry, FW2-HQ HQ-20 documentation) in
  `03_evidence_HPLT3_FLORES_classical.md`. Useful as background.
- **Phase 1 / Phase 2 evidence harvest** (`01_*`, `02_*`) —
  thorough but did not directly inform the cutoff pick. Stays as
  reference.

### What did NOT pan out

- **Phases 3-4 synthesis (effective-policy / rational-core / fairness
  definition / Greek budget)** — set aside, has known math errors,
  recommended +5,120 which is ~3× smaller than the empirical answer.
  Archived in
  [`_deprecated_20260518/synthesis_with_known_errors/`](_deprecated_20260518/synthesis_with_known_errors/).
- **Six open-hypothesis stubs (05-10)** — pre-2024 datasets,
  inherited-from-priors, team-institutional, commercial markets,
  benchmark coverage, Reddit. Never executed; deprioritized once
  `02_1_7` settled the cutoff empirically. Archived in
  [`_deprecated_20260518/`](_deprecated_20260518/).
- **Gini-on-FLORES+ optimization plan** ([`12_gini_optimization.md`](12_gini_optimization.md))
  — predicted N* in the +3-5k range; actual measurement chose
  +17,408. Marked SUPERSEDED; kept as historical record of what
  Gini-only reasoning predicted.

### Active files in the tree

```
01_explicit_goals.md                       — Phase 1 evidence (reference)
02_implicit_constraints.md                 — Phase 2 evidence (reference)
03_evidence_HPLT3_FLORES_classical.md      — HPLT 3.0 + FLORES+ + classical (reference)
04_evidence_speakers.md                    — speaker-count hypothesis (INVALIDATED)
11_tokenizer_provenance.md                 — Mistral inheritance + Apertus's changes
12_gini_optimization.md                    — Gini optimization plan (SUPERSEDED by 02_1_7)
INVESTIGATIONS_TRACKER.md                  — status index (now final)
REVIEW_INTEGRATION_20260517.md             — review pass on 2026-05-17
sources/apertus_2509.14233v2.pdf           — Apertus paper, local copy
_deprecated_20260518/                      — archived stubs + buggy synthesis
```

This sub-subproject does **not** modify tokenizer files, run new
measurements, or train anything. It is analysis only — and its
experimental thread has been handed off to `02_1_7`.

## Why this exists

The current cutoff REPORT
([`../02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md))
surveys six empirical anchors for Greek's vocab size:

- match Korean (script-isolated peer mean)
- match German (HQ-20 equal-share)
- match French
- match English-unique (~13 k, current pick)
- match English-total (~19 k)
- the fertility elbow (~17 k)

None of these is derived from a principle. They are anchors of
convenience — empirical footprints of other languages — with no
explanation of why those languages got those footprints. The choice
between them is therefore a choice between rhetorical framings.

This sub-subproject asks the question one level up: **what is the
policy that produced those footprints?** Once the policy is stated
and its necessary core separated from its accidents, the Greek budget
is whatever that core implies — not a pick from a menu.

## Method — four phases

### Phase 1: Explicit stated goals

Harvest quoted intent from primary sources. For every claim that
language X is or is not a target / priority / supported language,
record `(source, section, exact quote)`.

The point is to know **what the authors said they were doing** —
not to evaluate it, not to synthesize it. Phase 1 is bibliography
with quotes.

### Phase 2: Implicit constraints

For every per-language behavior visible in the artifacts, identify
which structural constraint produced it — and which constraints
*aren't* policy choices but consequences of the data, the web, or
the tokenizer-math.

Things expected to surface:

- the web's language distribution is a constant input that any
  web-derived corpus inherits.
- English-only datasets (FineWeb-HQ, FineWeb-Edu, DCLM-Edu,
  FineMath, MegaMath, StarCoder, Gutenberg) force English dominance
  in total token mass before any policy choice.
- Filter coverage gaps act as silent policy: the toxicity filter
  covers 9 languages (Apertus paper §3.1.3); the FineWeb-2-HQ
  quality classifier covers 20 languages and decides which
  multilingual data gets quality-filtered.
- BPE merge order rewards frequency; Latin-script merges share
  across many languages so big-Latin compounds; script-isolated
  languages get no sharing benefit.

### Phase 3: Effective policy synthesis

Write down the actual rule set producing the observed allocation, as
a numbered list. Each rule is sourced either to Phase 1 (stated
intent) or Phase 2 (structural consequence). Rules that have no
explanation are flagged as such — they are the candidates for
"accidental."

### Phase 4: Rational policy + Greek budget

For each rule, classify *necessary* vs *accidental*:

- **necessary** — the rule is forced by hardware, data availability,
  tokenizer math, or a principle the authors stated and would
  defend.
- **accidental** — the rule reflects a missing classifier, a
  filter-coverage gap, a historical priority that doesn't survive
  scrutiny, or a downstream consequence the authors didn't intend.

The **rational core** is the union of all *necessary* rules. Apply
the rational-core policy to Greek and read off the implied vocab
share. That number is the recommendation.

## Sources to analyze

### Tokenizer-level

- `mistralai/Mistral-Nemo-Base-2407` — model card, claimed
  multilingual scope of tekken v3.
- Apertus paper §2.2 + Appendix I — tokenizer selection criteria
  (FLORES+ 55 languages, fertility / compression / vocab utilization
  / Gini fairness).

### Apertus model / corpus

- Apertus paper (arXiv:2509.14233v2) — §1 abstract & intro,
  §3 pretraining data, §3.1 filters, §3.3 curriculum, §4
  post-training, Appendix G (language distribution), Appendix H
  (additional pretraining data), Appendix J (Romansh SFT).
- `swiss-ai/Apertus-8B-2509` model card — public claims about
  language coverage.
- ETH Zurich press release (September 2025) — public framing.
- Swiss AI Charter — value commitments referenced for alignment.

### Source datasets (per Apertus paper §3.2)

- English-only: FineWeb-HQ, FineWeb-Edu (`HuggingFaceFW/fineweb-edu`),
  DCLM-Edu (`HuggingFaceTB/dclm-edu`), FineMath
  (`HuggingFaceTB/finemath`), MegaMath (`LLM360/MegaMath`),
  StarCoderData (`bigcode/starcoderdata`), StarCoder Edu,
  CommonPile / Stack-v2-Edu, Gutenberg V1/V2
  (`swiss-ai/apertus-pretrain-gutenberg`), Flan commercial subset.
- Multilingual: FineWeb-2 (`HuggingFaceFW/fineweb-2`, 1,811 langs),
  FineWeb-2-HQ (`epfml/FineWeb2-HQ`, 20 high-resource langs incl.
  Greek), Clean-Wikipedia (`HuggingFaceFW/clean-wikipedia`, 319
  langs), EuroParl (`Helsinki-NLP/europarl`, ~21 EU langs),
  ParaDocs (`jhu-clsp/paradocs`, 18 EN-X pairs, no Greek),
  Institutional Books 1.0
  (`institutional/institutional-books-1.0`), EuroBlocks-SFT
  (`utter-project/EuroBlocks-SFT-Synthetic-1124`).
- Not used in v1 pretraining but released: `swiss-ai/apertus-pretrain-swiss`,
  `swiss-ai/apertus-pretrain-romansh`,
  `swiss-ai/apertus-pretrain-toxicity`.

### Comparator analyses already in this repo (read first; don't re-derive)

- [`../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md) —
  inventory of Apertus pretraining datasets and the measured
  Greek-share of consumed 8B pretraining (0.023 %).
- [`../../../docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md`](../../../docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md) —
  why architectural choices (gradient clipping at 0.1, Pre-Norm,
  RMSNorm, QK-Norm, AdEMAMix) force per-token embedding-norm parity
  across languages regardless of corpus share.
- [`../02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md) —
  current set of empirical-anchor frames for the cutoff decision.
- [`../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/summary.tsv`](../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/summary.tsv) —
  empirical per-language PMI-promoted token counts (the "footprint"
  numbers).
- [`../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/english_review/membership_report.md`](../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/english_review/membership_report.md) —
  derivation of "English unique = ~13 k".

## Final state (2026-05-18)

The cutoff decision was made in
[`../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)
— **17,408 added units, curated + backfilled, total vocab 148,480**.
This subproject contributed the provenance finding (which reframed
the question to Mistral-side), the suite discovery (which enabled
02_1_7), and a body of reference evidence. It did not itself produce
the budget number; that was settled empirically downstream.

See [`INVESTIGATIONS_TRACKER.md`](INVESTIGATIONS_TRACKER.md) for the
final hypothesis-status table. Active file list is in the "Status"
section above.

## Relationship to other sub-subprojects

- **Reads from**:
  `02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/` for
  empirical per-language token counts;
  `docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md` and
  `docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md` for
  background.
- **Feeds**: `02_1_4_cutoff_analysis/REPORT.md` — the budget from
  `GREEK_BUDGET.md` becomes the principled row in §1's anchor
  table; ideally it retires the rhetorical "match-X" framing in
  favor of a single derivation.
- **Independent of**:
  `02_1_polytonic_greek_extension/` — the polytonic arm has its own
  budget question that would need a separate run of this method.

## Out of scope

- Re-tokenization, re-training, or new BPE runs.
- New empirical measurements beyond the existing PMI + char-mask
  artifacts.
- Budget recommendations for any language other than Greek. The
  fairness framework is general but the numeric output is
  Greek-specific.
- Litigating Apertus's choices on ethical/political grounds. The
  goal is to surface the policy as-built, not to argue whether the
  policy is good.

## Research rule

Every numeric claim cites a primary source: a paper § number, a
dataset-card URL with retrieval date, or a measurement file in this
repo. No paraphrased claims about author intent when an exact quote
is available — use the quote. Where evidence is missing, say so
explicitly; do not extrapolate.

## See also

- [`../README.md`](../README.md) — parent subproject overview
- [`../02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md) — current empirical-anchor frames this work aims to either replace or formalize
- [`../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md) — Apertus pretraining inventory + measured Greek share
- [`../../../docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md`](../../../docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md) — why architecture forces norm parity regardless of share
- [`../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/`](../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/) — empirical per-language token-firing data
