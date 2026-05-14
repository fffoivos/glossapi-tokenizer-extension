# Model Review Specification

> **Scope rescoped 2026-04-20** — see
> [`NORMALIZATION_DESIGN_20260420.md`](NORMALIZATION_DESIGN_20260420.md)
> for the resolved design. Three review tasks (separator / MD-table audit
> / page-noise) plus a fourth added task (slash+dash mixed-token
> classification) replace the original per-match cleaning /
> per-match normalization / per-match markdown-normalization split below.
> Prompt structure per task is in `prompt_drafts/01..04`. The schemas in
> §5 below are the starting point; the design-doc prompts may shrink or
> reshape them.

This document freezes the current first-pass specification for model-assisted review of cleaning and normalization cases.

It consolidates:

- sample size
- category types
- model input context
- category-specific questions
- normalization targets
- first-pass thresholds

## 1. Goal Split

We have three review modes.

### Cleaning mode

Primary question:

- `is this noise?`

Secondary questions:

- is the bad span larger than the matched anchor?
- is adjacent context also noisy?
- is this already covered by an existing rule?
- do we need a regex extension, a new regex, or a non-regex rule?

### Normalization mode

Primary question:

- `does this preserve semantics?`

Used for:

- TOC/layout leaders
- separator lines
- whitespace/layout normalization

### Markdown normalization mode

Primary questions:

- `does this preserve semantics?`
- `does this preserve markdown structure?`

Used for:

- markdown table separator rows/cells
- markdown-adjacent structural cases

## 2. Sampling Specification

First pass:

- review `300` sampled matches per category
- monitor API cost before scaling

Sampling policy:

- cleaning categories:
  - random sampling
  - plus edge cases
- normalization categories:
  - more even / diversity-oriented sampling
  - plus edge cases

Rationale:

- `300` is large enough to expose common modes
- cheap enough for prompt/schema tuning
- small enough to make manual inspection feasible

## 3. Category Types

## 3.1 First-wave cleaning categories

1. `glyph_font_like`
2. `control_private_use_replacement`
3. `short_nonascii_latin_like`

## 3.2 First-wave normalization categories

1. `dot_leader_like`
2. separator families
3. `whitespace_only` and long whitespace-run families

## 3.3 First-wave markdown-sensitive categories

1. markdown table separator families
2. `table_border_ascii_art`

## 3.4 Lower-priority / telemetry-first categories

1. `latin_word_like`
2. `latin_mixed`
3. `other_script_letters`
4. `mixed_other`
5. `digits_only`
6. `punct_symbol_only`

These stay mostly observational unless a strong artifact pattern emerges.

## 4. Model Input Context

The model should not see only the raw token hit. It should receive a compact review packet.

Decision order:

1. binary decisions first
2. sliding/bucketed judgments second

That means:

- first decide whether something is noise or not
- first decide whether a normalization preserves semantics or not
- only after that ask broader page-level or degree/threshold questions

## 4.1 Shared fields for all modes

- `match_id`
- `category`
- `source_corpus`
- `source_path`
- `matched_span`
- `local_context`
- `synthetic_page_context` when available
- `matches_in_synthetic_page`
- `category_matches_in_synthetic_page`
- `matches_in_doc`
- `category_matches_in_doc`

For page-aware review packets, also include:

- full synthetic-page context when feasible
- highlighted matched spans within that synthetic page
- optional precomputed page indicators when available:
  - suspicious-match count
  - suspicious-line count
  - existing badness-score snapshot

## 4.1.1 Artificial page construction

For corpora without reliable page markers, we need synthetic pages.

Recommended first-pass construction:

1. prefer splitting on markdown headers when the resulting chunk is not too large
2. if headers are too far apart, split further by paragraph boundaries
3. if paragraphs are still too large, apply a final character-budget fallback

So the synthetic page builder should be:

- header-aware first
- paragraph-aware second
- size-aware always

This is especially important for:

- `short_nonascii_latin_like`
- page-level corruption estimation
- page discard vs page clean decisions

## 4.2 Extra context for cleaning mode

- what current cleaner rule/pattern already exists for this family
- which suspicious token family triggered the audit
- the matched context

This is important for `GLYPH`-like cases, because the review is partly about:

- whether existing coverage is sufficient
- whether the bad span is larger than the matched anchor

## 4.3 Extra context for normalization modes

- the candidate canonical target chosen by us
- original context
- proposed normalized context
- explicit statement of intent:
  - reduce over-representation of layout structures in tokenizer vocabulary
  - preserve semantics
  - preserve markdown structure when relevant

## 5. Model Output Contract

The model should not return long prose by default.

## 5.1 Cleaning-mode output

Minimal instance-level output:

- `match_id`
- `is_noise`
  - `yes`
  - `no`
  - `uncertain`
- `bad_span_start`
- `bad_span_end`
- `bad_span_text`
- `adjacent_noise_present`
  - `yes`
  - `no`
  - `uncertain`
- `existing_rule_coverage`
  - `covered`
  - `needs_extension`
  - `new_rule_needed`
  - `not_rule_like`
  - `uncertain`
- `candidate_regex_or_pattern`
- `confidence`
- `short_note`

Important:

- candidate regexes are hypotheses, not production rules
- deterministic validation happens after review

Optional secondary page-level cleaning outputs:

- `additional_noise_beyond_matches`
  - `none`
  - `some`
  - `substantial`
  - `uncertain`
- `page_noise_share_bucket`
  - `0-5`
  - `5-20`
  - `20-50`
  - `50-80`
  - `80-100`
  - `uncertain`
- `page_salvageability`
  - `clean`
  - `salvageable`
  - `mostly_noisy`
  - `likely_useless`
  - `uncertain`
- `dominant_noise_types`

We should prefer coarse buckets over exact numeric percentages. Exact percentages are likely to be unstable across reviewers and model runs.

This page-level block should not be loaded into every first-pass prompt. It is a secondary review layer for selected categories where page-level corruption matters.

Binary-first consumption rule:

- first consume:
  - `is_noise`
  - `bad_span`
  - `adjacent_noise_present`
  - `existing_rule_coverage`
  - `candidate_regex_or_pattern`
- only then consume:
  - page noise bucket
  - page salvageability

## 5.2 Normalization-mode output

- `match_id`
- `structure_class`
  - `toc_leader`
  - `separator_line`
  - `markdown_table_separator`
  - `not_structure`
  - `uncertain`
- `target`
- `interchangeable_with_target`
  - `yes`
  - `no`
  - `uncertain`
- `preserves_semantics`
  - `yes`
  - `no`
  - `uncertain`
- `preserves_markdown_structure`
  - `yes`
  - `no`
  - `not_applicable`
  - `uncertain`
- `confidence`
- `short_blocker`

## 6. First-Pass Normalization Targets

These targets are chosen by us first. The model validates equivalence with them.

### 6.1 TOC/layout leader class

- canonical target: `.....`

Reason:

- `...` is common prose ellipsis
- `.....` is more clearly a layout leader

### 6.2 Standalone separator/thematic-break class

- canonical target: `---`

### 6.3 Markdown table separator cells

- canonical targets:
  - `---`
  - `:---`
  - `:---:`
  - `---:`

## 7. First-Pass Matching Thresholds

### 7.1 Dot leaders

- global suspicious family: `4+` dots
- stronger TOC-style detector:
  - `3+` repeated punctuation marks between text and trailing page-like token

### 7.2 Hyphen separators

- standalone separator candidate: `4+` hyphens
- markdown table separator cell candidate: `4+` hyphens per cell

### 7.3 Underscore separators

- bare underscore separator line: `4+` underscores
- escaped underscore chains: `4+` escaped underscores

## 8. First-Pass Interchangeability Classes

## 8.1 Safe to normalize into one class

### A. TOC/layout leader class

Examples:

- `..............`
- mixed leader-like punctuation runs between text and page-like suffix

Canonical target:

- `.....`

### B. Standalone separator/thematic-break class

Examples:

- `-----`
- `_____`
- `***`

Canonical target:

- `---`

### C. Markdown table separator cell class

Examples:

- `------`
- `:------`
- `:------:`
- `------:`

Canonical targets:

- `---`
- `:---`
- `:---:`
- `---:`

## 8.2 Keep separate for now

These are not yet safe to collapse blindly:

- border/ascii-art fragments like `|--------------------------------|`
- escaped underscore chains like `\\_\\_\\_\\_\\_\\_\\_\\_`

They need review first because they may be:

- markdown-adjacent structure
- extraction residue
- decorative separators

## 9. Category-Specific Question Sets

## 9.1 `glyph_font_like`

Ask:

1. Is this noise?
2. Is the noisy span larger than the matched anchor?
3. Is adjacent extractor residue present?
4. Is this already covered by the existing glyph/font cleaner?
5. Do we need a regex extension, a new regex, or a non-regex rule?

## 9.2 `control_private_use_replacement`

Ask:

1. Is this noise?
2. Is it removable at character level, or only in context?
3. Is the surrounding line/page also corrupted?
4. Is this a sanitation rule or a document/page-level flagging signal?

## 9.3 `short_nonascii_latin_like`

Ask:

1. Is this noise or legitimate content?
2. Is it isolated, or is the synthetic page saturated with similar artifacts?
3. Does page-level density imply broader corruption?
4. Should this drive:
   - inline cleanup
   - page-level flagging
   - OCR/re-extraction suspicion
5. Beyond the matched spans, how much other noise is present on this synthetic page?
6. Is the page still salvageable, or broadly low-value/useless?

## 9.4 `dot_leader_like`

Ask:

1. Is this a layout leader in context?
2. Is it interchangeable with `.....`?
3. Does replacement preserve semantics?
4. Is this actually prose ellipsis and therefore not normalizable?

## 9.5 separator families

Ask:

1. Is this a standalone separator/thematic-break structure?
2. Is it interchangeable with `---`?
3. Does replacement preserve semantics?
4. If markdown-like, does it preserve markdown structure?

## 9.6 markdown table separator families

Ask:

1. Is this a real markdown table separator row/cell?
2. Is it interchangeable with canonical minimal markdown width?
3. Does replacement preserve semantics?
4. Does replacement preserve markdown structure?

## 9.7 `table_border_ascii_art`

Ask:

1. Is this legitimate table structure or extraction/layout residue?
2. If legitimate, is it markdown-separator-like or a different structural class?
3. If normalized, does markdown structure survive?
4. If not legitimate, should it be removed instead of normalized?

## 10. Rule Promotion Criteria

### Cleaning rules

Promote only if reviewed cases consistently say:

- it is noise
- the noisy span is stable enough to model
- proposed regex/pattern survives deterministic validation

### Normalization rules

Promote only if reviewed cases consistently say:

- this is the intended structure class
- it is interchangeable with the chosen canonical target
- semantics are preserved
- markdown structure is preserved when relevant

## 10.1 How `300` reviewed examples per category are consumed

The `300` reviewed examples should not be consumed as one flat majority vote.

They should be aggregated at three levels:

### A. Category-level estimates

Use the random sample to estimate category-wide rates such as:

- `% noise`
- `% already covered by existing cleaner`
- `% needing regex extension`
- `% needing new rule`
- `% interchangeable with chosen canonical target`
- `% semantics preserved`
- `% markdown structure preserved`

These are for planning and prioritization, not direct rule promotion.

### B. Family-level decisions

Within each category, group reviewed rows by:

- `pattern_family`
- `normalized_token_text`
- `candidate target`
- `structure_class`

This is where actual rules come from.

For each family, compute:

- reviewed count
- positive count
- positive rate
- confidence bucket
- disagreement rate
- blocker reasons

Families then fall into:

- `promote_candidate`
- `needs_more_review`
- `manual_review`
- `telemetry_only`

### C. Rule-level validation

For families that look promotable:

1. synthesize a candidate rule deterministically,
2. run it back over the full matched population,
3. test it on reviewed positives,
4. test it on reviewed negatives / nearby clean contexts,
5. measure overreach,
6. only then produce a proposed cleaner update.

So the flow is:

- reviewed sample -> family judgment
- family judgment -> candidate rule
- candidate rule -> deterministic validation
- validation result -> promote or reject

## 10.2 Cleaning aggregation

For cleaning categories, aggregate these core fields:

- `is_noise`
- `adjacent_noise_present`
- `existing_rule_coverage`
- `candidate_regex_or_pattern`

Consume them in this order:

1. binary noise decisions
2. regex/pattern proposals
3. page-level bucketed judgments

Meaningful outputs:

1. `category_summary.json`
- how noisy the category really is
- whether it is mostly already-covered cleaner residue or genuinely new residue

2. `family_summary.jsonl`
- one row per family
- whether the family is real cleaner debt

3. `candidate_cleaning_rules.jsonl`
- only for families with enough support and low enough disagreement

4. `manual_review_queue.jsonl`
- ambiguous or high-risk families

For `GLYPH`-like families specifically, we should also aggregate:

- how often the matched anchor is only a substring of the true bad span
- how often adjacent different noise is present

This tells us whether we need:

- an anchor regex extension
- a broader guarded regex
- or a non-regex block/line cleanup rule

For categories that produce regex proposals, especially `glyph_font_like`, the preferred synthesis path is:

1. collect candidate regexes/patterns from reviewed examples
2. normalize and deduplicate them
3. union them into a category-level matcher set
4. compile that set into an automaton
5. minimize the automaton / simplify equivalent branches
6. rerun against the reviewed sample and held-out contexts
7. check whether recall improves without unacceptable overreach

So the reviewed regexes are not the endpoint. They are inputs into a deterministic matcher-construction stage.

## 10.3 Normalization aggregation

For normalization categories, aggregate:

- `structure_class`
- `target`
- `interchangeable_with_target`
- `preserves_semantics`
- `preserves_markdown_structure`

Meaningful outputs:

1. `category_summary.json`
- how often this category is actually the intended structure class
- how often the chosen canonical target is acceptable

2. `family_summary.jsonl`
- one row per structure family
- rule viability

3. `candidate_normalization_rules.jsonl`
- only when reviewed cases consistently confirm:
  - correct structure class
  - interchangeability
  - semantics preserved
  - markdown structure preserved when needed

4. `manual_review_queue.jsonl`
- mixed or blocked cases

## 10.4 Promotion thresholds

First-pass conservative thresholds:

### Cleaning

- family reviewed count >= `12`
- `is_noise == yes` rate >= `0.85`
- `uncertain` rate <= `0.15`
- deterministic validation must not hit reviewed negatives

### Normalization

- family reviewed count >= `12`
- `interchangeable_with_target == yes` rate >= `0.85`
- `preserves_semantics == yes` rate >= `0.90`
- `preserves_markdown_structure == yes` rate >= `0.90` when applicable

These are starting thresholds, not immutable constants.

## 10.4.1 Page-threshold policy for short suspicious fragments

For categories like `short_nonascii_latin_like`, the page-level policy should be:

1. determine whether individual hits are noise
2. estimate whether the synthetic page is broadly noisy
3. group pages by page-noise threshold

Operationally:

- below threshold:
  - try page cleaning
- above threshold:
  - consider discarding the synthetic page

But before a discard policy is promoted, ask:

- would cleaning this page destroy meaning?

So discard should only be considered when both are true:

- page-level noise is high enough
- salvageability judgment is low enough

This should remain bucketed and conservative in the first pass.

## 10.5 Why random `300` is still useful

Random `300` per category is useful for:

- estimating how bad a category really is
- revealing dominant families
- deciding where rule work is worth doing

But it is not enough by itself to finalize every rule.

If a candidate family looks promising but has too few reviewed examples, the right next step is:

- targeted second-round sampling for that family

So the overall aggregation pattern is:

1. random category sample for estimation,
2. family grouping to identify likely rule candidates,
3. targeted follow-up review for sparse but promising families,
4. deterministic validation before implementation.

## 11. What The Model Is Not Being Asked To Do

The model is not being asked to:

- write production regexes directly
- invent canonical targets on each example
- produce long prose reviews
- decide implementation details inside `Corpus.clean`

Its job is narrower:

- identify noise vs non-noise
- mark the bad span when cleaning is needed
- validate equivalence with chosen normalization targets
- provide candidate regex/pattern hints where appropriate

## 12. Page-Level Noise Question

For suitable cleaning categories, especially:

- `short_nonascii_latin_like`
- `glyph_font_like`
- `control_private_use_replacement`

we should also ask a broader page-level question.

But this should be a targeted secondary question set, not part of every default prompt.

- how much additional noise exists on this synthetic page beyond the specifically matched spans?
- is the page still salvageable?

Noise here should mean spans that serve no semantic or structural purpose:

- not meaningful language
- not meaningful mathematics
- not valid table or formatting structure

This intentionally excludes legitimate structure that we may still normalize, such as:

- valid markdown table separators
- legitimate TOC leaders
- legitimate thematic separators

Those are structure, not noise.
