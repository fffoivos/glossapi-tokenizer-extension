# Corpus.clean Normalization And Review TODO

## Current Wave 3 Override (2026-04-28)

The immediate implementation queue is the tokenizer-guided wave-3
cleaner patch described in
`../subprojects/01_0_cleaning_iteration_and_thresholds/WAVE3_CLEANER_PATCH_PLAN_2026-04-28.md`.
It supersedes the older #14/page-salvage-first queue below.

- [ ] Run quantization + escaped Markdown runs + unified dot/ellipsis.
- [ ] Markdown table separator, setext underline, and ATX 7+
      quantization.
- [ ] Minimal fenced-code impossible-noise cleanup if sampling confirms
      the bypass.
- [ ] Narrow glyph Rule A/B extension for bare `GLYPH` and
      high-confidence glyph names.
- [ ] Keep placeholder HTML comments and dingbat/pictograph tokens.
- [ ] Defer mojibake repair and Cyrillic / homoglyph folding to
      eellak/glossAPI issue #99.

## Design Resolution (2026-04-20)

Authoritative design now lives in
[`NORMALIZATION_DESIGN_20260420.md`](NORMALIZATION_DESIGN_20260420.md).
Prompt templates in [`prompt_drafts/`](prompt_drafts/).

Concrete next moves, in order:

- [ ] **Land the normalize patch** on `codex/token-noise-review-debug`:
      whitespace collapse (#1), standalone separator lines (#2), GFM
      table separators (#3, parser-validated), ellipsis runs (#5),
      enclosed/stylized/dingbat/math-alphanumeric/fraction digit folding
      (#6, keeping subscripts + superscripts), ligatures (#7), Unicode
      whitespace folding (#8). Guard: skip inside fenced code blocks and
      `$$…$$` math.
- [ ] **Land the strip additions** in the same patch: malformed HTML
      entity fallback (#11), polytonic range explicit in `greek`
      SCRIPT_SETS (#13).
- [ ] **Land the suspect-bigram strip** (#12) as an Aho-Corasick pass
      sourced from
      `tokenizer_analysis/inspection/unified_classification_20260418/`
      (not the older `fresh/glossapi_only/`).
- [ ] **Add page-salvage drop** (#14): after normalize + strip, drop
      synthetic pages where residual non-whitespace content < 30% of
      pre-cleanup.
- [ ] **Rebase matcher literal sets** on `unified_classification_20260418`
      and rerun the Rust matcher across all 17 GlossAPI sources
      (currently only `openarchives.gr` + `hplt`).
- [ ] **Rewrite `_build_prompt`** in
      `src/glossapi/scripts/review_token_category_with_gemini.py` per
      `prompt_drafts/01..04` (line-windowed context with tagged match,
      before/after shadow, preamble caching, task reiteration as final
      block).
- [ ] **Run wave11**: Task A (separator review), Task C (page-noise
      detection + threshold validation), Task D (slash+dash
      classification). Task B (MD-table audit) follows the cleaner patch
      landing.

## Current Status

- [x] Create a dedicated normalization workspace
- [x] Preserve the running discussion in markdown
- [x] Inspect the current `Corpus.clean` implementation
- [x] Review HPLT / DataTrove / Bitextor at a high level
- [x] Build the initial suspicious-token taxonomy from tokenizer evidence
- [x] Confirm that suspicious non-Greek residue is mainly a GlossAPI signal
- [x] Confirm that continuous training does not introduce a meaningful new short-weird-token class
- [x] Reframe the work around a new review-centered architecture
- [x] Consolidate the first-pass model review specification into one source of truth
- [x] Build the first Rust-backed token-category debug/export substrate
- [x] Build the first review-bundle sampler and Gemini review runners
- [x] Complete the first OpenArchives review wave (`300` sampled matches across each first-wave category)
- [x] Aggregate the first review wave into category/family summaries and candidate rules
- [x] Land the first curated cleaner updates:
  - [x] dot-leader normalization to `.....`
  - [x] stronger glyph/font artefact rejection coverage
  - [x] character-level Unicode noise cleanup for private-use/replacement/control residue

## Latest Execution Progress

- [x] `step (0)` debug substrate is implemented on branch `codex/token-noise-review-debug`
- [x] `step (1)` first-pass real-data matching/export has been validated on OpenArchives and HPLT smoke runs
- [x] `step (2)` first full OpenArchives live review run completed with `gemini-3-flash-preview` and `thinkingLevel=low`
- [x] `step (2)` aggregate outputs were materialized into candidate cleaning and normalization rules
- [ ] `step (3)` still needs the next curated promotions:
  - [ ] decide whether any additional glyph/font extensions are still missing beyond current line rejection
  - [ ] separate safe control/private-use cleanup from unsafe broad Greek-range proposals
  - [ ] decide whether short suspicious pairs need any rule beyond improved glyph handling and page-level scoring
  - [ ] review markdown/separator families in the next wave

## Phase 1. Freeze Inputs

- [ ] Freeze the category manifests that Stage 1 will search
  - [ ] `glyph_font_like`
  - [ ] `control_private_use_replacement`
  - [ ] `short_nonascii_latin_like`
  - [ ] `dot_leader_like`
  - [ ] `table_border_ascii_art`
  - [ ] selected whitespace/separator families
- [ ] Decide which categories are:
  - [ ] first-wave mandatory
  - [ ] second-wave optional
  - [ ] telemetry-only
- [ ] Freeze source corpus roots for:
  - [ ] GlossAPI
  - [ ] HPLT
- [ ] Decide the initial sampling budget per category/family
  - [ ] start with `300` reviewed matches per category
  - [ ] monitor API cost before scaling

## Phase 2. Design The Rust Matcher

- [ ] Review `corpus_ocr` debug/export interfaces again and pin the exact reusable contract
- [ ] Define the match record schema
- [ ] Define output tree layout
- [ ] Decide the long-document context contract:
  - [ ] local line window
  - [ ] paragraph window
  - [ ] synthetic page window
  - [ ] whether to emit all three or a subset
  - [ ] page-level density counters for short suspicious fragments
  - [ ] synthetic page construction:
    - [ ] split on markdown headers first
    - [ ] split on paragraphs when headers are too far apart
    - [ ] apply size fallback when needed
- [ ] Decide grouping keys:
  - [ ] by category
  - [ ] by exact token
  - [ ] by normalized token family
  - [ ] by line/block pattern family
- [ ] Define the Rust crate boundary
- [ ] Define the Python entrypoint in GlossAPI
- [ ] Implement a first draft matcher
- [ ] Export:
  - [ ] `match_index.jsonl`
  - [ ] `manifest.jsonl`
  - [ ] `summary.json`
  - [ ] context files

## Phase 3. Design The Review Schema

- [ ] Define the per-instance structured review JSON schema
- [ ] Define the per-family summary schema
- [ ] Define the rule proposal schema
- [ ] Split the review prompts into:
  - [ ] cleaning mode: `is this noise?`
  - [ ] normalization mode: `does this preserve semantics?`
  - [ ] markdown normalization mode: `does this preserve semantics?` and `does this preserve markdown structure?`
- [ ] Bake the normalization goal into prompts:
  - [ ] reduce over-representation of repeated layout structures in tokenizer vocabulary
  - [ ] preserve semantics
  - [ ] preserve markdown structure when relevant
- [ ] Define allowed output enums for:
  - [ ] `artifact_type`
  - [ ] `is_noise`
  - [ ] `cleanability`
  - [ ] `rule_scope`
  - [ ] `generalization_risk`
- [ ] Define optional free-text fields:
  - [ ] `constraints`
  - [ ] `reasoning_summary`
  - [ ] `proposed_canonical_form`
- [ ] Define confidence and escalation rules
- [ ] Keep the instance-level review schema minimal:
  - [ ] `is_noise`
  - [ ] exact bad span
  - [ ] whether adjacent context also looks noisy
  - [ ] action
  - [ ] semantics-preservation judgment for normalization cases
  - [ ] markdown-structure-preservation judgment for markdown cases
  - [ ] coarse page-level noise bucket where relevant
  - [ ] page salvageability judgment where relevant

## Phase 4. Build The Review Automation Around Step (2)

- [ ] Decide the production review engine
  - [ ] Gemini structured outputs
  - [ ] Gemini Batch API
  - [ ] optional Argilla UI for human review
- [ ] Decide the role of Codex subagents
  - [ ] exploratory only
  - [ ] not the production review backend
- [ ] Define deterministic pre-review heuristics
- [ ] Define sampling policy for review manifests
  - [ ] random sampling for noise-detection categories
  - [ ] even / diversity-oriented sampling for normalization categories
  - [ ] edge cases
  - [ ] uncertainty-driven samples
- [ ] Write the first prompt pack for:
  - [ ] token-family review
  - [ ] context review
  - [ ] rule proposal review
  - [ ] explicit noise-first prompt flow
  - [ ] original-vs-normalized semantics comparison prompt
  - [ ] markdown structure preservation prompt
- [ ] Build a batch review runner
- [ ] Build a result validator for schema compliance
- [ ] Build an aggregator that combines:
  - [ ] heuristic priors
  - [ ] model outputs
  - [ ] human overrides
- [ ] Define category-level summary metrics from the `300` random sample
- [ ] Define family-level grouping and promotion buckets
- [ ] Define candidate-rule validation passes against positives, negatives, and held-out examples
- [ ] Build a rule-promotion report

## Phase 5. Human Review Layer

- [ ] Decide whether to use:
  - [ ] local materialized review bundles only
  - [ ] Argilla
  - [ ] both
- [ ] Define the adjudication workflow for:
  - [ ] low-confidence model outputs
  - [ ] disagreement across contexts
  - [ ] high-risk proposed rules
- [ ] Reuse / adapt:
  - [ ] `review_manifest_materialize.py`
  - [ ] `table_sentence_context_review.py`
- [ ] Design a compact review file template for token/context cases

## Phase 6. Rule Synthesis

- [ ] Define allowed rule classes:
  - [ ] exact removal
  - [ ] character-class sanitation
  - [ ] unconstrained normalization
  - [ ] constrained normalization
  - [ ] line/block removal
  - [ ] document-level flagging
  - [ ] OCR/re-extraction escalation
- [ ] Define promotion criteria for each rule class
- [ ] Define what evidence is required before a rule can ship
- [ ] For `GLYPH`-like families, define how span-level model judgments become deterministic candidate regexes or guarded patterns
- [ ] Define regex aggregation pipeline for cleaning categories:
  - [ ] collect candidate regexes
  - [ ] normalize/deduplicate
  - [ ] union into category matcher set
  - [ ] compile/minimize automaton
  - [ ] rerun validation sample
- [ ] For cleaning categories, define how the model sees existing cleaner context:
  - [ ] current regex/pattern family
  - [ ] suspicious token family that triggered the audit
  - [ ] matched contexts
- [ ] Define how candidate regex proposals are validated:
  - [ ] positive reviewed examples
  - [ ] nearby negative examples
  - [ ] held-out examples
- [ ] Freeze first-pass normalization thresholds:
  - [ ] dot-run threshold
  - [ ] hyphen separator threshold
  - [ ] markdown table separator-cell threshold
  - [ ] underscore separator threshold
- [ ] Freeze first-pass canonical targets:
  - [ ] TOC/layout leaders -> `.....`
  - [ ] separator/thematic-break lines -> `---`
  - [ ] markdown table separator cells -> `---`, `:---`, `:---:`, `---:`
- [ ] Freeze the normalization review contract:
  - [ ] we choose the candidate canonical target first
  - [ ] the model validates whether the observed span is interchangeable with that target
  - [ ] the model does not freely invent canonical targets on each example
- [ ] Freeze first-pass interchangeability classes:
  - [ ] thematic/separator lines
  - [ ] TOC/layout leaders
  - [ ] markdown table separator cells
  - [ ] keep border/ascii-art fragments separate initially
  - [ ] keep escaped underscore chains separate initially
- [ ] Define how proposed rules map to:
  - [ ] `Corpus.clean`
  - [ ] Rust cleaner components
  - [ ] new score/counter outputs
- [ ] Define page-threshold policy for suspicious-fragment categories:
  - [ ] below threshold -> clean
  - [ ] above threshold -> consider discard
  - [ ] ask whether cleaning the page would destroy meaning

## Phase 7. First Cleaner Iteration

- [ ] Implement first conservative rules
  - [ ] glyph/font residue
  - [ ] control/private-use/replacement cleanup
  - [ ] TOC leader normalization
  - [ ] separator normalization
  - [ ] markdown table separator canonicalization
- [ ] Add telemetry counters
- [ ] Add `layout_artefact_score`
- [ ] Keep existing remediation/OCR scores separate

## Phase 8. Re-Audit

- [ ] Rerun the matcher after cleaner changes
- [ ] Diff before/after category counts
- [ ] Inspect false positives
- [ ] Inspect residual suspicious families
- [ ] Launch the second review loop on the residuals

## Supporting Research / References

- [ ] Preserve the local reference note for:
  - [ ] `corpus_ocr` debug/export patterns
  - [ ] Snorkel / data programming
  - [ ] Argilla / Distilabel
  - [ ] Prodigy-style review patterns
  - [ ] Gemini structured output / batch references
- [ ] Add notes on which external ideas are relevant and which are not for PDF-extracted academic corpora

## Explicit Open Questions

- [ ] Should the first matcher search only token-derived manifests, or also regex/pattern families from day one?
- [ ] Should synthetic page boundaries be char-based, line-based, or paragraph-based?
- [ ] Should markdown tables be reviewed as:
  - [ ] inline token artifacts
  - [ ] line/block artifacts
  - [ ] both
- [ ] How much of the review stack should be model-only vs model-plus-human?
- [ ] When should a suspicious artifact trigger OCR/re-extraction instead of normalization?
