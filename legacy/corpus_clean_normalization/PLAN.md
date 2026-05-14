# Corpus.clean Normalization And Review Plan

> **Scope rescoped 2026-04-20** — see
> [`NORMALIZATION_DESIGN_20260420.md`](NORMALIZATION_DESIGN_20260420.md)
> for the resolved design. That doc narrows review scope to three Gemini
> tasks (separator / MD-table audit / page-noise) plus an added
> slash+dash mixed-token review (Task D), and promotes an explicit
> deterministic normalize/strip rule list (whitespace collapse, separator
> lines, GFM table separators, ellipsis runs, stylized-digit folding,
> ligatures, Unicode whitespace folding, polytonic-set safety, malformed
> HTML entities, suspect-bigram strip). PLAN.md below still describes the
> broader architecture (Stages 1–4, aggregation pattern, review levels)
> but concrete rule and prompt decisions now live in the design doc.

## Objective

Build a repeatable pipeline that:

1. finds suspicious token categories back in the original GlossAPI and HPLT corpora,
2. packages those matches into reviewable context bundles,
3. uses structured model review plus human checks to decide whether each pattern is:
   - noise,
   - salvageable structure,
   - legitimate content,
4. turns those reviewed decisions into safe cleaning / normalization rules for `Corpus.clean`,
5. reruns the audit on the residual artifacts.

This plan is centered on step `(2)`. The main risk is not matching tokens. The main risk is promoting bad rules from shallow review.

The consolidated first-pass review specification is frozen in:

- [MODEL_REVIEW_SPEC.md](/home/foivos/Projects/glossapi-tokenizer-extension/corpus_clean_normalization/MODEL_REVIEW_SPEC.md)

## Core Principles

- Use real corpus evidence, not imagined rules.
- Always answer `is this noise?` before designing a rule.
- Separate:
  - remediation / OCR signals,
  - layout / formatting artifacts,
  - legitimate lexical residue.
- Treat model review as a proposal layer, not as final authority.
- Promote rules only after grouped evidence, context review, and spot-checking.
- Prefer manifest-driven, batchable workflows over ad hoc notebooks or one-off prompts.
- Reuse the existing `corpus_ocr` debug/review contract wherever possible.

## What Already Exists

### Existing local review/export primitives

Relevant code already gives us a good blueprint:

- [`ocr_render.py`](/home/foivos/glossAPI-development/src/glossapi/corpus/ocr_render.py)
  - `_build_match_index_rows(...)`
- [`phase_clean.py`](/home/foivos/glossAPI-development/src/glossapi/corpus/phase_clean.py)
  - `clean_ocr_debug(...)`
  - `clean_ocr_numeric_word_debug_docs(...)`
- [`review_manifest_materialize.py`](/home/foivos/glossAPI-development/src/glossapi/scripts/review_manifest_materialize.py)
- [`table_sentence_context_review.py`](/home/foivos/glossAPI-development/src/glossapi/scripts/table_sentence_context_review.py)

These already establish a strong pattern:

- `match_index.jsonl`
- `manifest.jsonl`
- `summary.json`
- reviewable context files
- grouped materialization by label/category

### Existing cleaning/scoring behavior

`Corpus.clean` already does meaningful sanitation and OCR-related gating. The missing layer is more targeted:

- artifact discovery from tokenizer evidence,
- layout-sensitive canonicalization,
- broader PDF/extractor residue coverage,
- better feedback loops for deciding what should be:
  - removed,
  - normalized,
  - preserved,
  - sent to OCR / flagged.

## Target Architecture

The workflow should have four major stages.

### Stage 1. Rust match extraction

Build a Rust-backed matcher that searches the original corpora for the suspicious artifacts we discovered from tokenizer analysis.

Inputs:

- token/category manifests derived from tokenizer analysis
- source corpus roots for:
  - GlossAPI
  - HPLT
- optional corpus metadata such as doc source, path, subtype, and prior clean scores

Outputs:

- `match_index.jsonl`
- `manifest.jsonl`
- `summary.json`
- per-match context files
- grouped review bundles by category and by token/pattern family

The output contract should intentionally mirror `corpus_ocr` debug exports.

### Stage 2. Structured review and rule proposal

This is the center of gravity.

For each grouped artifact family, build a review pipeline that:

1. samples representative contexts,
2. asks a model for a structured judgment,
3. aggregates repeated judgments across contexts,
4. produces proposed cleaning rules plus risk notes,
5. escalates uncertain cases for manual review.

This stage should not directly edit the cleaner. It should emit review artifacts and candidate rules.

### Stage 3. Rule implementation and scoring changes

Take only the high-confidence, cross-context, low-risk proposals from Stage 2 and turn them into:

- cleaner rules,
- canonicalization rules,
- new counters,
- new categorization scores such as `layout_artefact_score`.

### Stage 4. Re-audit and iterate

Rerun the matcher on the cleaned corpus and inspect:

- which categories disappeared,
- which categories shrank,
- what suspicious new residue remains,
- whether valid content was over-cleaned.

Then repeat the review loop on the residual set.

## Stage 1 In Detail: Rust Match Extraction

## 1.1 Categories to search

The first pass should cover the currently identified taxonomy, including:

- `short_nonascii_latin_like`
- `glyph_font_like`
- `control_private_use_replacement`
- `whitespace_only`
- `dot_leader_like`
- `table_border_ascii_art`
- `punct_symbol_only`
- `digits_only`
- `latin_word_like`
- `latin_mixed`
- `other_script_letters`
- `mixed_other`

Not all of these are equally important. The first production-grade review loop should prioritize:

- `glyph_font_like`
- `control_private_use_replacement`
- `short_nonascii_latin_like`
- `dot_leader_like`
- `table_border_ascii_art`
- selected `whitespace_only` / separator families

## 1.2 Context unit

This is the first design decision that matters.

The OCR tooling assumes page boundaries. Many GlossAPI/HPLT markdown documents do not have reliable page splits.

So the matcher should support three context units:

1. real page context
   - only when page markers exist
2. synthetic logical page context
   - derive boundaries from lines / paragraphs / character budget
3. short local context
   - a compact window around the exact match

Recommended default contract for long documents:

- exact match span
- `line_index`
- `paragraph_index`
- `synthetic_page_index`
- `context_before_lines`
- `context_after_lines`
- `context_before_chars`
- `context_after_chars`

Recommendation:

- always emit compact local context
- additionally emit synthetic-page context for long docs
- do not depend on page splits being present

For corpora without real page markers, synthetic pages should be built:

1. by markdown headers when the chunk is not too large,
2. by paragraph boundaries when headers are too far apart,
3. by a final size-budget fallback when needed.

For short suspicious fragments and similar atomic categories, the matcher should also emit page-density signals:

- `matches_in_synthetic_page`
- `category_matches_in_synthetic_page`
- `matches_in_doc`
- `category_matches_in_doc`

This lets us ask whether a local hit is isolated or whether the surrounding synthetic page is saturated enough that the whole page may be low-value or broadly noisy.

## 1.3 Match record schema

The first-pass schema should extend the `corpus_ocr` style contract with token-specific fields:

- `match_id`
- `source_corpus`
- `source_path`
- `source_stem`
- `category`
- `token_text`
- `normalized_token_text`
- `pattern_family`
- `match_kind`
  - `exact_token`
  - `regex_family`
  - `char_class`
  - `line_pattern`
- `start_char`
- `end_char`
- `start_byte`
- `end_byte`
- `line_index`
- `paragraph_index`
- `synthetic_page_index`
- `match_text`
- `local_context`
- `context_path`
- `doc_score_snapshot`
  - existing badness scores if available

## 1.4 Review bundle layout

Recommended output tree:

```text
token_audit/
  match_index.jsonl
  manifest.jsonl
  summary.json
  contexts/
  by_category/
  by_token/
  by_pattern_family/
```

This should be directly compatible with downstream materialization scripts and later model review manifests.

## Stage 2 In Detail: Structured Review And Rule Proposal

## 2.1 What the review system must answer

Each reviewed artifact instance, or grouped artifact family, must answer four separate questions:

1. Is this noise?
2. If not pure noise, is it structure that should be normalized?
3. If it is structure, what constraints must be preserved?
4. What generalizable cleaner action should be taken?

This is why step `(2)` is hard. The system is not just classifying. It is translating context judgments into safe transformation policies.

The order matters:

1. Is the matched span actually noise?
2. If yes, is there adjacent context that should also be removed?
3. If not noise, is it structure that should be normalized?
4. If normalized, does the proposed normalization preserve semantics in the local context?

This is especially important for `GLYPH`-like anchors, where the hit may only point at a larger noisy span.

Consumption should follow the same order:

1. binary judgments first
2. regex/pattern proposals second
3. sliding or bucketed page-level judgments third

We now have two main review modes:

### Cleaning mode

Primary question:

- `is this noise?`

Secondary questions:

- is there adjacent noise around the matched anchor?
- is this covered by an existing cleaner rule already?
- do we need to update an existing regex?
- do we need a new regex or a non-regex rule?

### Normalization mode

Primary question:

- `does this preserve semantics?`

For markdown-sensitive cases, ask additionally:

- `does this preserve markdown structure?`

This applies to:

- markdown table separators
- border-like markdown rows
- other markdown structure-bearing artifacts

Normalization prompts should explicitly state the goal:

- we are trying to reduce over-representation of repeated layout structures in tokenizer vocabulary
- we want to normalize these structures as much as possible without losing meaning
- for markdown cases, we also want to preserve markdown structure

Normalization prompts should also use a fixed candidate target chosen by us in advance. The model should validate equivalence, not invent the canonical target from scratch on every example.

## 2.2 Review levels

The review loop should operate at three levels:

### A. Token-family review

Group repeated identical or normalized tokens and ask:

- what type of artifact family is this?
- is it likely noise, structure, or lexical content?

This is fast and cheap, but insufficient on its own.

### B. Context review

Review concrete usages of the same family across different documents.

This is required because the same token can mean different things depending on context:

- `.....` in a TOC leader
- `.....` inside prose or ellipsis
- `---|` as broken table residue
- `---` as a valid markdown separator

### C. Rule review

Aggregate the context judgments and ask for the candidate general rule:

- remove exact token
- remove token in suspicious contexts only
- canonicalize token/pattern to a stable form
- preserve
- escalate to OCR / page-furniture detection

This is the level that should produce rule proposals.

## 2.3 Decision schema

The model output must be strict JSON with a compact, versioned schema.

Recommended first-pass schema:

- `artifact_type`
  - `pdf_extractor_noise`
  - `glyph_font_residue`
  - `invalid_unicode_noise`
  - `layout_leader`
  - `table_separator`
  - `table_border_residue`
  - `page_furniture`
  - `legitimate_foreign_text`
  - `legitimate_symbolic_content`
  - `uncertain`
- `is_noise`
  - `yes`
  - `no`
  - `mixed`
  - `uncertain`
- `cleanability`
  - `remove`
  - `normalize`
  - `keep`
  - `doc_level_flag`
  - `ocr_or_reextract`
  - `uncertain`
- `rule_scope`
  - `exact_token`
  - `normalized_token_family`
  - `line_pattern`
  - `block_pattern`
  - `doc_pattern`
  - `none`
- `proposed_canonical_form`
- `constraints`
  - free text or small list
- `generalization_risk`
  - `low`
  - `medium`
  - `high`
- `reasoning_summary`
- `confidence`
  - integer or enum

Important:

- structured output is necessary
- chain-of-thought is not necessary
- the schema should optimize for downstream rule synthesis, not prose explanation

For instance-level review, keep the operational payload minimal:

- `match_id`
- `is_noise`
- `bad_span_start`
- `bad_span_end`
- `bad_span_text`
- `adjacent_noise_present`
- `action`
- `preserves_semantics`
- `confidence`

The model should mark spans and make narrow decisions. Regex and guarded-rule synthesis should remain deterministic on our side.

For cleaning-oriented categories, the model input should explicitly include:

- current cleaner context
  - what regex/pattern family we already apply
- the suspicious token family that led to the new audit
- sampled matched contexts

The model should then answer:

1. `is_noise`
2. `bad_span_start/end/text`
3. `adjacent_noise_present`
4. `existing_rule_coverage`
   - `covered`
   - `needs_extension`
   - `new_rule_needed`
   - `not_rule_like`
5. `candidate_regex_or_pattern`
6. `regex_confidence`

Important:

- a returned regex should be treated as a candidate proposal
- we should not promote it directly to production
- the deterministic synthesis/validation layer must test it against positives, negatives, and held-out examples

## 2.4 Automation stack options

We should explicitly separate:

- interactive exploration
- durable production review

### Option A. Codex subagents

Good for:

- early exploratory analysis
- checking a few bundles interactively
- helping design schemas and prompts

Not good as the main production path because:

- they are session-bound
- they are not a durable batch engine
- they are awkward for large manifest-scale review

### Option B. Google AI Studio / Gemini API

This is the clearest official batchable path today.

Why it is attractive:

- official structured JSON output support
- official Batch API
- lower batch cost for large offline review
- easy JSONL request/response contract

Recommendation:

- use Gemini structured output plus Batch API for production review manifests
- use interactive calls only for prompt/schema tuning

Important nuance:

- the official hosted path is clearly documented for Gemini
- the official AI Studio hosted path for a hypothetical "Gemma 4 review workflow" is not comparably clear
- so the first durable implementation should be designed around Gemini API, not around an assumed hosted Gemma path

### Option C. Argilla / Distilabel

Good for:

- human-in-the-loop review
- surfacing model suggestions to annotators
- storing responses, filters, progress, metadata
- pushing ongoing generated/reviewed items into a UI

Recommendation:

- use Argilla as the human review surface if we want a dedicated annotation UI
- optionally generate suggestions with a model first, then let a reviewer confirm/override

### Option D. Prodigy-style review pattern

The useful lesson from Prodigy is the review contract:

- collect multiple "versions" for the same example
- merge them under stable hashes / IDs
- let the reviewer decide a final version

We do not need Prodigy itself, but we should borrow this pattern for:

- heuristic classifier output
- model review output
- human adjudication output

## 2.5 Recommended review workflow

Recommended first production workflow:

1. Rust matcher produces grouped manifests.
2. Deterministic pre-pass assigns:
   - category
   - normalized token family
   - cheap heuristics
3. Review sampler selects:
   - an initial capped sample of `300` contexts per category
   - random sampling for noise-detection categories
   - more even / diversity-oriented sampling for normalization categories
   - edge cases
4. Gemini Batch runs structured review on those bundles.
5. Aggregator produces:
    - per-instance reviews
    - per-family summaries
    - candidate rule proposals
6. High-risk or low-agreement families go to human review:
   - local materialized review bundle
   - optionally Argilla UI
7. Rule promotion step emits:
   - approved rule spec
   - rejected proposal spec
   - needs-more-evidence families

This is the right compromise:

- deterministic search stays local and fast,
- model review scales,
- human review is reserved for ambiguity.

The `300`-example starting cap is intentional:

- it is large enough to expose common modes,
- it keeps API costs bounded while prompts stabilize,
- and it gives us a measurable first budget before scaling.

For short suspicious-fragment categories, page-threshold decisions should be conservative:

- below threshold:
  - prefer cleaning the synthetic page
- above threshold:
  - consider discard only if review also says the page is broadly unsalvageable or that cleaning would destroy meaning

For cleaning categories, the aggregator should specifically track:

- how often the model says the anchor is true noise
- how often it says adjacent context is also noisy
- how often it believes an existing regex can be extended
- which candidate regexes/patterns recur across reviewed cases

This lets us turn many reviewed examples into a small set of candidate cleaner updates.

For regex-producing cleaning categories, the next deterministic stage should be:

1. collect proposed regexes/patterns,
2. normalize and deduplicate them per category,
3. union them into a category-level matcher set,
4. compile that set into an automaton,
5. minimize/simplify it,
6. rerun the sample and held-out validation sets,
7. measure whether match quality improves without unacceptable overreach.

That gives us a practical way to consume many noisy regex proposals into one cleaner update.

## 2.6 From review to safe rules

The review layer must emit actions in a form that maps cleanly into implementation.

Rule classes:

1. exact removal
   - e.g. known glyph/font residue
2. character-class sanitation
   - e.g. control/private-use cleanup
3. unconstrained normalization
   - e.g. TOC leader runs -> `.....`
4. constrained normalization
   - e.g. markdown separator cell width collapse while preserving row/column semantics
5. line/block removal
   - e.g. page-furniture lines or border-only rows
6. document-level flagging
   - e.g. too much repeated page furniture
7. OCR/re-extraction escalation
   - when artifacts indicate extraction failure rather than formatting noise

Promotion rule:

- do not implement a rule just because a model suggested it once
- require grouped support, representative examples, and risk assessment

## Stage 3 In Detail: Cleaner Changes

The first implementation pass should stay conservative.

Recommended first-wave rule families:

- stronger glyph/font residue cleanup
- stronger control/private-use/replacement cleanup
- TOC leader normalization
- separator normalization
- markdown table separator canonicalization
- page-furniture detection counters
- short suspicious non-Greek fragment telemetry

Also add:

- `layout_artefact_score`
- per-artefact counters
- rule-hit telemetry

Do not start with:

- broad lexical suppression
- aggressive language filtering
- global punctuation simplification

For normalization review, the model should always see:

- original context
- normalized context

and answer explicitly whether the normalization preserves semantics for tokenization-oriented text preparation.

For markdown cases, the model must additionally answer whether markdown structure is still preserved after normalization.

First-pass normalization defaults:

- TOC/layout leader class -> `.....`
- standalone separator/thematic-break class -> `---`
- markdown table separator cells:
  - `---`
  - `:---`
  - `:---:`
  - `---:`

First-pass matching thresholds:

- dot-run family:
  - match `4+` dots globally
  - allow a stronger TOC-style detector for `3+` repeated punctuation marks when between text and trailing page-like tokens
- standalone hyphen separator family:
  - match `4+` hyphens
- markdown table separator cell family:
  - match `4+` hyphens per cell and collapse to canonical width
- underscore separator family:
  - treat bare underscore separator lines separately from escaped underscore chains

Interchangeability policy:

- do collapse:
  - thematic/separator lines `-----`, `_____`, `***` -> `---`
- do not yet collapse blindly:
  - border/ascii-art fragments like `|--------------------------------|`
  - escaped underscore chains like `\\_\\_\\_\\_\\_\\_\\_\\_`

Those should remain separate review families until evidence shows they are safely interchangeable.

Normalization review should therefore ask:

1. is this span functioning as the target structure class here?
2. is it interchangeable with the chosen canonical form?
3. does the replacement preserve semantics?
4. for markdown cases, does it preserve markdown structure?

For example:

- TOC/layout leader family:
  - ask whether the observed structure is interchangeable with `.....`
- separator-line family:
  - ask whether it is interchangeable with `---`
- markdown table separator family:
  - ask whether it is interchangeable with canonical markdown cell widths

## Stage 4 In Detail: Re-audit

After the first cleaner pass:

1. rerun token extraction,
2. rebuild category summaries,
3. diff before/after match counts,
4. inspect false positives and residual artifacts,
5. launch another review pass only on the residual/hard categories.

This loop should become the standard development cycle for normalization work.

## External Methodology We Should Borrow

### From Snorkel / data programming

Borrow:

- labeling functions as reusable heuristics
- aggregation of noisy signals
- active improvement of rules based on reviewed failures

Do not copy blindly:

- this is not a classic supervised label pipeline
- our end goal is rule synthesis for cleaning, not dataset classification alone

### From Argilla / Distilabel

Borrow:

- suggestion-first annotation
- explicit review UI
- searchable/filterable records with metadata
- push reviewed/generated items into a durable annotation surface

### From Prodigy review

Borrow:

- multi-version review
- stable IDs / hashes
- adjudicated final decision over competing sources

### From Gemini Batch / structured outputs

Borrow:

- schema-driven review output
- low-touch batch execution for large manifests
- explicit failure handling per request

## Deliverables

A successful first-pass implementation should produce:

1. Rust token/category matcher with `corpus_ocr`-style outputs
2. manifest-driven context bundle generator
3. structured review schema and prompt pack
4. batch review runner
5. review aggregator
6. candidate rule manifest
7. first conservative cleaner patch set
8. before/after audit report

## Immediate Next Milestones

1. Freeze the exact category manifests to search.
2. Design the Rust match record schema and output tree.
3. Decide the default long-doc context contract:
   - synthetic pages vs local line windows vs both
4. Design the structured review JSON schema.
5. Write the first batch-review prompt pack.
6. Review a small pilot sample manually before scaling.
7. Only then implement the full matcher and batch loop.
