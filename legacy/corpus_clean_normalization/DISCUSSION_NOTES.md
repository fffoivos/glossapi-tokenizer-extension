# Corpus.clean Normalization Notes

This is a living discussion file for normalization and cleaning changes we may add to `Corpus.clean` in GlossAPI. The goal is to preserve current conclusions, refine them over time, and keep decisions tied to actual tokenizer/corpus evidence.

## Plan Lock: Review-Centered Workflow

We are no longer treating this as only a cleaner-rule design exercise.

The working plan is now:

1. use tokenizer-derived bad token categories to search the original corpora,
2. package the matches into reviewable context bundles,
3. run structured review over those bundles,
4. turn the reviewed decisions into cleaning / normalization rules,
5. rerun the audit on the residual artifact set.

The center of gravity is step `(2)`: review and rule proposal. The highest risk is not missing a token match. The highest risk is over-generalizing a bad cleaning rule from shallow evidence.

## Additional Plan Lock

We have now fixed a few operational decisions for the first review iteration:

- the first and most important question is always whether a match is actually noise
- for anchor-like matches such as `GLYPH`, the review must also decide whether adjacent context is part of the removable noise span
- for short suspicious pairs and similar atomic artifacts, we should compute synthetic-page density counters so we can tell whether an isolated hit is harmless or whether the surrounding page is broadly noisy
- we do not need to review all matches initially
- first pass review budget should start at `300` sampled matches per category
- sampling should differ by task:
  - random sampling for noise-detection categories
  - more even / diversity-oriented sampling for normalization categories
- for suitable cleaning categories, we also want an optional secondary page-level judgment:
  - how much additional noise exists beyond the specific matched spans
  - whether the synthetic page is still salvageable
- for step `(3)` consumption, we should aggregate in order:
  - binary decisions first
  - regex proposals second
  - page-threshold / bucketed judgments third
- for regex-producing cleaning categories, we want to unify agent-proposed regexes per category, union them into a matcher set, compile/minimize an automaton, and then retest
- for suspicious short-pair categories, synthetic pages should be built:
  - from markdown headers first when possible
  - from paragraphs when headers are too sparse
  - with a size fallback when needed
- page-threshold policy should start conservative:
  - below threshold we try cleaning
  - above threshold we consider discard only if the page is also judged unsalvageable or meaning-destroying to clean
- normalization review prompts should show:
  - original context
  - normalized context
  - and ask whether semantics are preserved for tokenization-oriented text preparation

We also now have the core prompt split:

- cleaning prompt:
  - `is this noise?`
- normalization prompt:
  - `does this preserve semantics?`
- markdown normalization prompt:
  - `does this preserve semantics?`
  - `does this preserve markdown structure?`

For cleaning categories, the model should be given explicit prior-cleaner context:

- what we already clean with the current method
- which suspicious tokens led us to revisit these cases
- sampled matched contexts from the corpus

And the cleaning review should answer:

1. is this noise?
2. is there an update to an existing regex, or a new regex, that would match these cases?
3. is the surrounding context also noisy in a different way?
4. if this is regex-matchable, what is the candidate regex or regex family?

Important implementation note:

- model-proposed regexes should be treated as candidate hypotheses, not trusted production artifacts
- all regex proposals must be validated deterministically against:
  - positive matched examples
  - negative/held-out contexts
  - counterexamples from nearby clean text

## Normalization Prompt Framing

For normalization review, the model should always be told what we are trying to do:

- we are targeting repeated layout/formatting structures that are over-represented in tokenizer vocabulary
- we want to normalize them as aggressively as possible without losing meaning
- for markdown cases, we also want to preserve markdown structure

The default normalization review should therefore compare:

- pre-normalized context
- post-normalized context

and answer:

1. does this preserve semantics?
2. for markdown-like cases, does this preserve markdown structure?

If both answers are yes, the normalization is a success candidate.

## First-Pass Normalization Thresholds And Canonical Targets

These are first-pass engineering defaults, not final truths.

### 1. Dot leaders / ellipsis-like runs

We should distinguish prose ellipsis from layout leaders.

Recommended matching thresholds:

- global suspicious dot-run family: `4+` dots
- stronger TOC-leader family: `3+` repeated punctuation marks when the run sits between text and a trailing page-like token

Recommended canonical form:

- TOC/layout leader class -> `.....`

Reason:

- `...` is common prose punctuation and should not be globally rewritten
- `4+` starts to behave more like layout than prose

### 2. Hyphen separator runs

Recommended matching thresholds:

- standalone separator candidate: `4+` hyphens on a line or in a separator run
- markdown table separator cell candidate: `4+` hyphens inside a detected table separator cell

Recommended canonical forms:

- standalone separator/thematic-break class -> `---`
- markdown table cells:
  - `:---`
  - `:---:`
  - `---:`
  - `---`

Reason:

- `---` is already the shortest stable markdown/thematic-break form
- if we start matching `---` globally, we risk rewriting already-canonical structure

### 3. Underscore runs

Important distinction:

- bare `___` on its own line can be a valid markdown thematic break
- backslash-escaped underscore chains like `\\_\\_\\_\\_\\_\\_\\_\\_` are not meaningful markdown table syntax and are more likely escaping residue or layout artefacts

Recommended matching thresholds:

- bare underscore separator line: `4+` underscores
- escaped underscore chain artifact family: `4+` escaped underscores

Recommended handling:

- standalone thematic-break-like underscore lines can normalize to `---`
- escaped underscore chains should start as a separate artifact family and likely collapse to `---` only if review confirms they are just separator noise

### 4. Border / ASCII-art families

Examples:

- `|--------------------------------`
- `|------|`
- `+-----+`
- `---|`

These should not be merged blindly with markdown table separators.

Working interpretation:

- true markdown separator rows are a structure-preserving normalization case
- border-only / ASCII-art fragments are often extraction/layout artefacts

Recommended first-pass classing:

- markdown table separator class
- border/ascii-art artifact class

Do not unify them until review shows they are semantically interchangeable for the target mode.

## Interchangeability Classes

We should not normalize everything to one symbol family globally. We should define equivalence classes.

### Safe first-pass equivalence classes

1. thematic/separator line class
- examples:
  - `-----`
  - `_____`
  - `***`
- canonical form:
  - `---`

2. TOC/layout leader class
- examples:
  - `..............`
  - mixed leader-like punctuation runs between text and page number
- canonical form:
  - `.....`

3. markdown table separator cell class
- examples:
  - `------`
  - `:------`
  - `:------:`
- canonical forms:
  - `---`
  - `:---`
  - `:---:`
  - `---:`

### Not yet safe to collapse blindly

- border/ascii-art fragments like `|--------------------------------`
- escaped underscore chains like `\\_\\_\\_\\_\\_\\_\\_\\_`

These need review first. They may be:

- markdown-adjacent structure
- extraction residue
- decorative separators

and should therefore remain a distinct review family until we have evidence.

## How To Ask The Model The Right Question

For normalization, the model should not be asked to freely invent the canonical form on every example. That is too open-ended and will create inconsistent answers.

Instead, we should choose the candidate canonical form ourselves, then ask the model whether the matched structure is:

1. actually a separator/leader/table-separator in this context,
2. interchangeable with the proposed canonical form,
3. safe to normalize without semantic loss,
4. safe to normalize without markdown-structure loss when relevant.

### Recommended canonical targets

- inline TOC/layout leader -> `.....`
- standalone separator/thematic-break line -> `---`
- markdown table separator cells -> `---`, `:---`, `:---:`, `---:`

### Why `.....` for inline leaders

`.....` is better than `...` for this purpose because:

- `...` is ordinary prose punctuation and often carries lexical meaning as an ellipsis
- `.....` is much more strongly a layout/leader signal
- it preserves the idea of a visual leader without keeping arbitrary run length

So the right question is not:

- "what should the separator become?"

The right question is:

- "is this span functioning as an inline leader, and is it interchangeable with `.....` here?"

### Normalization review contract

For each normalization family, the model should see:

- the matched span
- the original context
- the proposed normalized context
- the candidate canonical target
- a short statement of goal:
  - reduce separator/layout variation for tokenization
  - preserve semantics
  - preserve markdown structure when relevant

And the model should answer with a compact structured decision such as:

- `is_structure_class`
  - `toc_leader`
  - `separator_line`
  - `markdown_table_separator`
  - `not_structure`
  - `uncertain`
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

### What gets promoted into a normalization rule

A normalization rule should only be added when reviewed cases consistently say:

- this is the intended structure class
- it is interchangeable with the canonical target
- semantics are preserved
- markdown structure is preserved when relevant

So the model is not being asked to "design the rule" in free prose.

It is being asked to validate a candidate equivalence:

- "is this a separator/leader/table-separator?"
- "is it safely interchangeable with our chosen canonical form?"

## Why This Matters

Tokenizer inspection showed that a meaningful share of learned non-Greek tokens is not foreign-language vocabulary. It is largely:

- layout syntax
- markdown table structure
- TOC/index leader runs
- long whitespace runs
- punctuation-only / symbol-only runs
- PDF extraction residue
- control / private-use / replacement-character garbage

This means `Corpus.clean` should treat layout and extraction artifacts as first-class normalization targets, not just do generic text cleanup.

## Current Corpus Signal

Observed high-frequency artifact families:

- TOC/index leader runs such as `Intro .................. p.15`
- alternate separator/leader patterns such as `---//--//`
- markdown table separators and border-like rows
- long whitespace runs
- dot leaders and ellipsis runs
- PDF glyph/extraction residue such as `GLYPH`, `hyphenminus`, `phenminus`, `font`
- replacement-char / invalid-char residue such as `�`
- control chars and private-use glyphs

## Recommended Normalization Structure

Recommended pass order:

1. Unicode / extraction sanitation
2. Markdown-aware structural normalization
3. Leader / layout run normalization
4. Whitespace normalization

The principle is context-aware normalization, not one global “collapse long runs” rule.

## 1. Unicode / Extraction Sanitation

Do this before markdown logic.

Recommended:

- normalize line endings: `\r\n? -> \n`
- remove C0/C1 controls except `\n` and maybe `\t`
- remove BOM, zero-width chars, word joiners, soft hyphen
- normalize NBSP/thin-space variants to plain space
- remove private-use chars by default
- remove orphan combining marks
- remove isolated replacement-char garbage `�`

### PDF Noise

We should specifically target PDF/extractor residue, but in context, not with crude global string deletion.

Examples to treat as suspicious:

- `GLYPH`
- `hyphenminus`
- `phenminus`
- `font`
- `/hy...`

Preferred policy:

- strip or collapse glyph/extractor residue in suspicious contexts
- avoid deleting ordinary prose words globally

## 2. Markdown-Aware Structural Normalization

Markdown itself is not the problem. Variant-heavy layout is the problem.

### Tables

We should detect real markdown table blocks rather than operate on raw hyphen count alone.

For markdown separator rows:

- `------` -> `---`
- `:------` -> `:---`
- `:------:` -> `:---:`
- `------:` -> `---:`

The question is not “how long is too long globally?” The question is “is this line functioning as a markdown separator row?”

Recommendation:

- inside markdown table separator rows, any hyphen run longer than `3` is semantically redundant
- canonicalize to the shortest stable markdown form

### Non-table separator rows

For pure separator lines such as:

- `------`
- `_____`
- `***`
- border-like rows such as `---|`, `|-----`, `---//--//`

Recommended:

- canonicalize to a small stable form if they carry structure
- drop them if they carry no semantic or content value

## 3. Leader / Layout Run Normalization

For examples like:

- `Intro .................. p.15`
- `Intro ---//--// p.15`

The precise run length is not semantically meaningful. The signal is just “there is a leader / separator here”.

Recommended canonical targets:

- TOC / leader runs -> `.....`
- separator runs -> `---`

Recommendation:

- do not collapse to something large like `40`
- collapse to the smallest stable form that preserves the structural signal

Examples:

- `Intro .................. p.15` -> `Intro ..... p.15`
- `Intro ---//--// p.15` -> `Intro --- p.15` or `Intro ..... p.15`

## 4. Whitespace Normalization

This also needs context.

Recommended:

- collapse runs of spaces/tabs inside prose lines to one space
- collapse `3+` blank lines to `2`
- drop whitespace-only lines longer than a blank paragraph break
- normalize table padding, not table cell content

Recommended canonical targets:

- prose spacing: `1` space
- paragraph spacing: max `2` newlines
- table separator cell: `---`
- horizontal rule: `---`
- TOC leader: `.....`

## What This Tells Us

The dataset is not just “Greek text plus some noise”. It contains strong repeated structural/layout syntax, especially from academic and markdown-heavy material.

That means:

- tokenizer budget is being spent on layout variability
- normalization should preserve semantic structure
- normalization should aggressively collapse non-semantic layout variation

## High-Level Recommendation

Add a structure-aware normalization layer to `Corpus.clean` that:

- preserves semantic markdown structure
- collapses redundant layout variation
- removes extraction garbage
- emits counters so we can measure impact directly

## Suggested Counters

If we implement this, `Corpus.clean` should report at least:

- `control_chars_removed`
- `private_use_chars_removed`
- `replacement_chars_removed`
- `glyph_noise_tokens_removed`
- `table_separator_rows_normalized`
- `leader_runs_normalized`
- `horizontal_rules_normalized`
- `whitespace_runs_collapsed`

## Open Design Questions

- Which PDF glyph/extractor patterns are safe to remove globally versus only in context?
- Which markdown table forms should be preserved verbatim versus canonicalized?
- Should border-only / separator-only lines be dropped entirely in some corpora?
- What canonical leader representation is best: `.....`, `---`, or a dedicated placeholder?
- Should we preserve tabs anywhere, or normalize all tabs to spaces?
- Which invalid Unicode categories should be hard-dropped vs mapped?

## External Review: Standard Pipeline Practices

I reviewed public code and documentation from HPLT, Hugging Face/DataTrove, and Bitextor to see what established pipelines actually do in cleaning and normalization.

### HPLT

Main sources reviewed:

- `warc2text-runner/src/warc2text_runner/stage2/trafilatura/traf.py`
- `warc2text-runner/src/warc2text_runner/stage2/fastertext_lid/patterns.py`
- `warc2text-runner/two/sample100/README.md`
- `monotextor-slurm/30.clean`
- `monotextor-slurm/scripts/annotate.py`

Observed pattern:

- HPLT relies heavily on extraction settings, scoring, and filtering, not on aggressive universal text rewriting.
- In Trafilatura plain-text extraction, `include_tables` is disabled while `favor_precision` is enabled.
- For language ID prep, HPLT strips non-word chars and digits and squeezes repeated whitespace.
- In the exploratory notes, markdown tables are treated as unreliable and of questionable usefulness for the LM-oriented path.
- In the cleaning stage, HPLT keeps only documents that survive filter/scoring thresholds, rather than trying to normalize everything into shape.
- `monotextor-slurm/scripts/annotate.py` uses restorative cleaning plus HTML-tag removal before scoring.

Practical takeaway:

- HPLT’s bias is “extract conservatively, then score/filter hard”.
- It is not “preserve every structural artifact and normalize all variants”.
- For LM-focused text, tables are not treated as sacred.

### Hugging Face / DataTrove

Main sources reviewed:

- `datatrove/src/datatrove/pipeline/formatters/ftfy.py`
- `datatrove/src/datatrove/pipeline/formatters/symbol_lines_remover.py`
- `datatrove/src/datatrove/pipeline/filters/c4_filters.py`
- `datatrove/src/datatrove/pipeline/filters/fineweb_quality_filter.py`
- `datatrove/src/datatrove/pipeline/filters/gopher_repetition_filter.py`

Observed pattern:

- `FTFYFormatter` is conservative by default.
- It enables encoding/control repair but deliberately leaves stricter normalization off:
  - `normalization=None`
  - `fix_latin_ligatures=False`
  - `fix_character_width=False`
  - `uncurl_quotes=False`
  - `fix_line_breaks=False`
- `SymbolLinesFormatter` removes lines consisting only of symbols, but preserves whitespace-only lines.
- C4/FineWeb/Gopher style filters focus on rejecting bad documents:
  - low terminal-punctuation ratio
  - too many short lines
  - excessive newline/list ratio
  - repeated lines / paragraphs / n-grams
  - obvious junk lines such as javascript/policy/lorem patterns

Practical takeaway:

- HF-style pipelines separate mild repair from quality rejection.
- They do not aggressively canonicalize punctuation/layout everywhere.
- They use document-level statistics to reject low-value repetitive/list-heavy material.

### Bitextor

Main sources reviewed:

- `bitextor/utils/unicodepunct.py`
- `bitextor/utils/clean-corpus-n.perl`

Observed pattern:

- Bitextor maintains broad coverage of Unicode punctuation/spacing/control-adjacent characters.
- It applies practical segment cleanup:
  - collapse whitespace
  - trim ends
  - enforce min/max length
  - enforce cross-side length ratios
  - reject overlong words

Practical takeaway:

- Bitextor is closer to “useful sanitation plus hard sanity checks”.
- It is less structure-aware than markdown/web pipelines, but stronger on Unicode punctuation coverage.

## Synthesis

The common pattern across these pipelines is:

1. extraction / boilerplate handling
2. mild Unicode / encoding repair
3. document-quality scoring or filtering
4. limited structural cleanup

What they generally do not do:

- aggressive universal punctuation canonicalization
- flatten all layout into one representation
- preserve every table/border/leader artifact for LM text by default

This matters for us because our current evidence says we have a specific artifact class:

- academic TOC leaders
- markdown table separators
- border-like ASCII runs
- PDF/extraction residue
- whitespace variability

These are not generic web-text artifacts. They are repeated enough to spend tokenizer budget on. That means we likely need more structure-aware cleanup than generic web pipelines, but we should still copy the broader design pattern:

- keep repair targeted
- keep filtering separate from normalization
- add document-level metrics instead of trying to normalize every bad document into something useful

## Additional Suggestions From The Review

### 1. Split normalization from rejection

`Corpus.clean` should not try to solve every low-quality document with rewriting.

Recommended separation:

- normalization layer:
  - Unicode sanitation
  - PDF noise cleanup
  - whitespace normalization
  - leader/separator canonicalization
  - markdown table separator normalization
- rejection/scoring layer:
  - repeated-line ratio
  - repeated-paragraph ratio
  - list/newline ratio
  - symbol-line ratio
  - table-heavy ratio
  - control/private-use density
  - replacement-char density

### 2. Add doc-level artifact metrics

Inspired by FineWeb/Gopher/HPLT scoring, we should measure at least:

- `newline_ratio`
- `short_line_ratio`
- `symbol_line_ratio`
- `duplicate_line_ratio`
- `duplicate_paragraph_ratio`
- `table_separator_line_ratio`
- `leader_line_ratio`
- `control_char_ratio`
- `private_use_ratio`
- `replacement_char_ratio`

Some documents should be dropped or at least flagged instead of aggressively normalized.

### 3. Treat tables differently in LM mode vs rich-text mode

HPLT’s handling suggests we should not assume markdown tables are worth preserving in LM text.

Recommended modes:

- LM/plain-text mode:
  - canonicalize table separators aggressively
  - optionally drop border-only rows
  - optionally flatten tables to cell text when structure is weak
- rich-structure mode:
  - preserve more markdown/table structure for downstream tasks that need it

This should be a deliberate mode split, not an accidental side effect.

### 4. Canonicalize structure-bearing layout, not all punctuation

Do not globally rewrite punctuation styles.

Do:

- normalize TOC leaders
- normalize markdown separator rows
- normalize border-only separator lines
- collapse whitespace runs

Do not:

- flatten ordinary punctuation in prose
- normalize quotes/dashes/ligatures indiscriminately unless we have clear evidence it helps

### 5. Use broad Unicode sanitation coverage

Bitextor’s Unicode punctuation coverage is a useful reminder that “invalid chars” are not only ASCII control bytes.

We should explicitly handle:

- C0/C1 controls
- zero-width chars
- BOM / word joiners / soft hyphen
- Unicode spacing variants
- private-use characters
- isolated combining marks
- replacement-char residue

### 6. Add line-class detection before normalization

For our corpus, classification by line type should come before rewrite rules.

Useful line classes:

- prose
- markdown table separator
- markdown table row
- TOC/leader line
- border/separator line
- whitespace-only line
- symbol-only line
- PDF-garbage line

This is better than a single global “long run > N” rule.

### 7. Prefer small canonical targets

The reviewed pipelines generally avoid preserving arbitrary formatting length variation.

For us that supports:

- TOC leader -> `.....`
- horizontal/border separator -> `---`
- markdown table separator cell -> `---`, `:---`, `:---:`, `---:`
- prose spacing -> single space
- blank-line runs -> max `2`

### 8. Add an explicit “artifact review” dataset

Before implementation, assemble a small gold set of real examples from our corpus covering:

- PDF glyph noise
- invalid/control/private-use chars
- TOC/index lines
- markdown tables
- ASCII-art separators
- whitespace pathologies

Every normalization rule should be tested against this set before rollout.

## Source Pointers

- HPLT warc2text-runner: <https://github.com/hplt-project/warc2text-runner>
- HPLT monotextor-slurm: <https://github.com/hplt-project/monotextor-slurm>
- DataTrove: <https://github.com/huggingface/datatrove>
- Bitextor: <https://github.com/bitextor/bitextor>

## Current GlossAPI `Corpus.clean` Snapshot

The actual implementation is not in the tokenizer-extension repo. The current richer implementation lives in:

- [phase_clean.py](/home/foivos/glossAPI-development/src/glossapi/corpus/phase_clean.py)
- [cleaning_module.rs](/home/foivos/glossAPI-development/rust/glossapi_rs_cleaner/src/cleaning_module.rs)
- [pipeline_module.rs](/home/foivos/glossAPI-development/rust/glossapi_rs_cleaner/src/pipeline_module.rs)

### What It Already Does

The current cleaner already covers a significant part of the generic sanitation/scoring stack:

- decodes HTML entities before cleaning
- strips non-comment HTML/XML tags
- preserves HTML comments as explicit placeholders when content was removed
- uses script-based character filtering
  - keeps configured scripts plus punctuation, numbers, common symbols, and essential whitespace
  - removes characters classified as unusual when they are not allowed by the retained script set
- detects strong PDF/extraction artefacts at line level
  - examples in the trigger list include `glyph<c=`, `glyph&lt;c=`, `GLYPH<`, `GLYPH&lt;`, `font=/`, `FontName=`
  - bad lines are replaced by `<!-- text-missing -->`
- removes tables in the Rust pipeline as a dedicated stage
  - malformed/structural table handling exists in the Rust cleaner
  - removed table regions become `<!-- table-removed -->`
- computes mojibake-oriented badness and script percentages
- computes Greek/noise-oriented metrics in `glossapi_rs_noise`
- writes parquet metrics and marks documents for OCR based on:
  - mojibake score
  - Greek badness score
  - empty / near-empty cleaned text

There is also OCR-specific cleaning and scoring in the same file, including:

- table-preserving blanking for OCR repeat analysis
- control/private-use/replacement counts in OCR noise metrics
- OCR repeat/noise flags

### What It Does Not Obviously Do Yet

The current implementation is stronger on scoring, filtering, and table removal than on canonicalizing layout variation.

It does not obviously implement the following explicit canonicalizations:

- TOC leader normalization such as `..............` -> `.....`
- separator normalization such as `---//--//` -> `---`
- canonical markdown separator-row shortening such as `------` -> `---`
- general whitespace-run collapse in prose
- explicit normalization of NBSP / thin-space / zero-width variants in the base cleaner path
- explicit line-type-aware canonicalization for:
  - TOC lines
  - separator lines
  - markdown table separator rows

So the gap is real, but it is narrower than originally assumed:

- `Corpus.clean` already does substantial sanitation and gating
- the missing layer is mostly targeted canonicalization of recurring academic/layout artefacts

## Specific Audit Points To Investigate Next

The tokenizer evidence means the current cleaning is still not comprehensive enough. In particular, seeing `GLYPH`, font-like artefacts, long punctuation runs, and layout tokens survive into learned tokens means we need a narrower, more concrete audit list.

### 1. PDF / extractor artefact coverage is still incomplete

Current cleaner coverage is strongest for obvious line-level artefacts such as:

- `glyph<c=`
- `glyph&lt;c=`
- `GLYPH<`
- `GLYPH&lt;`
- `font=/`
- `FontName=`

What we should explicitly audit now:

- plain `GLYPH` variants that are not wrapped in tag-like syntax
- mixed-case or partially broken variants
- `hyphenminus`, `phenminus`, `uniXXXX`, `cidXXXX`, font-family residue, encoding labels
- private-use glyph bullets and PDF ornament glyphs
- inline artefacts inside otherwise salvageable lines, not just whole-line junk

Main question:

- which artefacts should trigger line replacement
- which should be stripped inline
- which should only contribute to rejection metrics

### 2. Unicode sanitation likely needs expansion

The current cleaner clearly scores some Unicode noise, but the base cleaning path does not obviously normalize all of it away.

We should explicitly inspect handling for:

- zero-width characters
- BOM / word joiners
- soft hyphen
- NBSP and thin-space variants
- private-use characters
- replacement character `�`
- isolated combining marks
- line/paragraph separator code points

Main question:

- are these only being scored, or are they being removed/mapped before tokenization?

### 3. Layout artefacts are not being canonicalized enough

This is the largest likely gap.

We should explicitly audit:

- TOC/index leader lines:
  - `Intro .................. p.15`
  - `Chapter ---//--// 42`
- border/separator lines:
  - `---|`
  - `|-------------|`
  - `___`
  - `***`
- dot/ellipsis runs
- long whitespace-only runs
- markdown table separator rows

Main question:

- should these be dropped, shortened, or canonicalized to a stable small form?

### 4. Markdown table policy needs to be made explicit

The current cleaner already removes tables structurally in some paths, but our corpus still contains table-derived layout tokens.

We should inspect separately:

- real markdown tables with useful cell content
- malformed tables
- border-only table rows
- separator rows with excessive hyphen length
- documents that are mostly tables or index material

Main question:

- what should LM/plain-text mode keep
- what should LM/plain-text mode flatten
- what should be dropped entirely

### 5. Whitespace policy is still underspecified

We need an explicit decision for:

- repeated spaces inside prose
- tabs
- blank-line runs
- space runs inside tables
- leader/separator whitespace

Main question:

- what variability is meaningful
- what variability is pure token-budget waste

### 6. Separate inline cleanup from document rejection

Right now the cleaner is already strong on gating, but we need to be more explicit about when to:

- clean inline and keep the document
- mark the document suspicious but keep it
- trigger OCR
- drop the document from LM-focused training

This especially matters for:

- partially salvageable academic markdown
- documents with repeated table/index structure
- documents with moderate PDF residue but useful text

### 7. Add artefact-specific metrics, not just broad badness

Current metrics are useful, but they are still too broad to guide layout-focused normalization policy.

We should add or expose metrics for:

- leader-line ratio
- separator-line ratio
- markdown-table-separator ratio
- symbol-only-line ratio
- whitespace-run count
- dot-leader count
- PDF-artefact token count
- private-use count
- replacement-char count

These should support both:

- per-document rejection decisions
- corpus-wide policy review

### 8. Build a real corpus audit set before changing rules

We should not implement canonicalization from memory or intuition.

We need a hand-reviewed sample set for:

- PDF glyph junk
- font artefacts
- TOC/index leaders
- markdown tables
- separator/border rows
- whitespace pathologies
- mixed salvageable / unsalvageable lines

This should become the regression set for every future `Corpus.clean` change.

## What The External Pipeline Review Adds Beyond Token Inspection

The bad-token analysis already told us:

- PDF/extraction junk is surviving
- layout variability is surviving
- whitespace/separator artefacts are surviving
- some non-Greek residue is still entering the tokenizer

What the external pipeline review adds on top is not mainly more examples of junk. It adds design patterns we should adopt.

### 1. Separate normalization from rejection

This is the biggest addition from HPLT / DataTrove / FineWeb / Gopher style pipelines.

The tokenizer evidence alone suggests “clean more”.
The external review says we also need:

- targeted normalization for salvageable text
- document-level rejection/flagging for low-value documents

In other words, not every bad document should be normalized into acceptance.

### 2. Add document-level quality metrics

This does not follow directly from inspecting learned bad tokens.

The reviewed pipelines suggest we should measure and use signals like:

- duplicate-line ratio
- duplicate-paragraph ratio
- short-line ratio
- newline/list ratio
- symbol-line ratio
- terminal-punctuation ratio
- table-heavy ratio

These are useful because some documents are bad due to overall structure, not because of a few local artefacts.

### 3. Add explicit LM/plain-text mode vs rich-structure mode

This does not follow directly from the tokenizer evidence either.

HPLT’s plain-text extraction disables tables, which suggests:

- one cleaning policy is not enough for all downstream uses
- LM-oriented training text should likely be stricter about table and layout preservation
- a richer markdown-preserving mode may still be useful for other tasks

### 4. Add line-type classification before rewrite rules

The bad-token inspection tells us that leader lines and separator rows exist.
It does not by itself imply the best implementation strategy.

The review suggests the implementation should first classify lines, then normalize by class:

- prose
- table row
- table separator
- TOC leader
- border/separator
- symbol-only
- PDF-garbage

That is a better design than global regexes everywhere.

### 5. Keep global normalization conservative

This is an important negative lesson from DataTrove.

The external review suggests we should avoid global normalization of:

- quotes
- ligatures
- width variants
- line breaks

unless we have evidence it helps.

So the lesson is not just “add more normalization”.
It is “add targeted normalization where our corpus actually needs it, and stay conservative elsewhere”.

### 6. Build explicit corpus-level telemetry

Token inspection alone shows outcomes.
The reviewed pipelines suggest we also need first-class operational telemetry.

That means:

- counters per artefact type
- per-document rejection reasons
- corpus-wide ratio summaries
- before/after comparisons on real samples

Without this, we will keep making changes based on anecdotal token inspection.

### 7. Add a gold audit set and regression policy

Again, this is not implied by the bad tokens themselves.

The external review supports a more disciplined workflow:

- build a small representative artefact corpus
- run every cleaning change against it
- review before/after output
- only then adopt the new rule

This matters because our corpus contains a lot of semi-structured academic material, where over-cleaning is a real risk.

## Locked Current Position

At this point, the current working position is:

- `Corpus.clean` already does substantial sanitation, scoring, and OCR gating
- the remaining cleaning gap is mostly:
  - incomplete PDF/extractor artefact coverage
  - incomplete invisible-Unicode sanitation
  - insufficient canonicalization of recurring academic/layout artefacts
- broad web-style quality filters are not a good primary fit for our PDF-extracted corpus
- the current removal-based badness scores remain the right backbone for remediation / OCR decisions

In other words:

- we are not replacing the current `mojibake_badness_score` / `greek_badness_score` logic with web-style filtering
- we are adding targeted normalization and likely a second, different score family for layout/artefact categorization

## Scoring Architecture: What Should And Should Not Go Into A Badness Score

Not all cleaning signals should be collapsed into one badness score.

### 1. Keep the current removal-based badness scores as primary remediation scores

These remain the right scores for OCR-triggering and hard remediation decisions:

- `mojibake_badness_score`
- `greek_badness_score`
- empty / near-empty cleaned-text checks

Why:

- they measure how much had to be removed or how wrong the recovered text surface is
- they map naturally to “this document likely needs OCR or correction”

### 2. Do not overload those scores with layout/structure artefacts

The following classes should not automatically be treated like mojibake/non-Greek corruption:

- TOC leaders
- separator runs
- markdown table separators
- border/table-layout artefacts
- some whitespace/path-layout artefacts

Why:

- they are often structural rather than corrupt
- they may be noisy for tokenizer training but do not necessarily imply OCR failure
- folding them into the same badness score would mix two different problems:
  - text corruption
  - layout variability

### 3. Add a separate layout / structure artefact score

We likely need a new score for categorization, review, and possibly secondary filtering.

Working names:

- `layout_artefact_score`
- `structure_noise_score`
- `surface_layout_score`

Current best working name:

- `layout_artefact_score`

Purpose:

- quantify how layout-heavy / extraction-artefact-heavy a document is
- support categorization and analysis
- help decide later whether some documents should be excluded from tokenizer training
- not automatically trigger OCR the way corruption scores do

### 4. Inputs that likely belong in the new layout score

Likely score inputs:

- PDF glyph / font artefact density
- TOC leader line density
- separator / border line density
- markdown table separator density
- repeated page-furniture/header-footer density
- long whitespace-run density
- symbol-only border/layout density

These are layout/noise indicators, not direct corruption indicators.

### 5. Inputs that should probably stay as telemetry, not score terms

Some signals are useful for auditing but should not immediately become part of any score:

- raw counts of each artefact class
- exact transformed-line counts
- per-rule normalization counters
- extremely rare edge-case Unicode categories

These should first help us understand the corpus before they become weighted score terms.

### 6. Resulting score split

The cleanest current model is:

- remediation scores:
  - `mojibake_badness_score`
  - `greek_badness_score`
  - empty/near-empty checks
- categorization score:
  - `layout_artefact_score` (new)
- telemetry:
  - artefact counters and densities per class

This avoids one overloaded “badness” number trying to do every job.

## Concrete Remaining Cleaning Work

Given all of the above, the concrete remaining cleaning work is now fairly narrow:

### A. Expand PDF / extractor artefact removal

Especially for artefacts that survive current line-level rules:

- plain `GLYPH`
- broken / partial glyph strings
- `hyphenminus`, `phenminus`
- font-name / encoding residue
- inline extractor junk inside otherwise good lines

### B. Expand invisible Unicode cleanup

Specifically verify and likely normalize/remove:

- NBSP and thin-space variants
- zero-width characters
- BOM / word joiners
- soft hyphen
- isolated combining marks
- line/paragraph separator code points

### C. Add explicit canonicalization for recurring layout classes

These are the main currently missing canonicalizations:

- TOC leaders
- separator runs
- markdown table separator rows
- border/table-layout artefacts
- long whitespace runs

### D. Add narrow PDF-specific page-furniture handling

Relevant mainly for:

- repeated headers
- repeated footers
- page numbers
- recurring page furniture

This is the main place where duplicate-line-style logic still matters for us.

### E. Add a separate audit category for very short opaque non-Greek fragments

Tokenizer inspection suggests there is a distinct class of suspicious short non-Greek tokens that should not be lumped together with:

- ordinary Latin-bearing residue
- layout/separator artefacts
- clear PDF glyph junk

Examples:

- `ĮȞ`
- `ȦȞ`
- `Į`
- `Ƞ`
- `Ț`
- `Ȟ`

These often look like:

- Latin-extended or other European-script fragments
- mojibake-like short fragments
- broken font/extraction remnants
- extremely short opaque subword pieces with little standalone semantic value

For the `continuous_glossapi_only_156672` strict non-Greek slice:

- total strict non-Greek tokens: `1749`
- short tokens of length `<= 2`: `997`
- of those, `630` contain Latin-script characters by Unicode name

This is too large a class to ignore.

Current conclusion:

- add a separate audit / categorization bucket for `short_opaque_non_greek_fragments`
- initial scope: non-special non-Greek tokens with decoded length `<= 2`
- this bucket should be reviewed separately from:
  - ordinary Latin words / abbreviations
  - punctuation/layout tokens
  - explicit glyph/font artefacts

Important caution:

- do not automatically treat every short non-Greek token as junk
- some short tokens may still be legitimate:
  - abbreviations
  - math/styled symbols
  - inherited Apertus/base tokens
- but this class is suspicious enough that it deserves explicit tracking

## Current Token Category Taxonomy

For the `fresh_glossapi_only` strict non-Greek slice, the current working category taxonomy is:

1. `short_nonascii_latin_like`
2. `glyph_font_like`
3. `control_private_use_replacement`
4. `whitespace_only`
5. `dot_leader_like`
6. `table_border_ascii_art`
7. `punct_symbol_only`
8. `digits_only`
9. `latin_word_like`
10. `latin_mixed`
11. `other_script_letters`
12. `mixed_other`

This taxonomy is meant for inspection and policy design, not as a final filtering policy.

### Category intent

- `short_nonascii_latin_like`
  - short suspicious fragments such as `Į`, `Ƞ`, `Ȟ`, `ȦȞ`
  - audit bucket, not automatic junk
- `glyph_font_like`
  - explicit PDF/extractor artefacts such as `GLYPH`, font/cid/uni patterns
- `control_private_use_replacement`
  - replacement-char, private-use, or control-style residue
- `whitespace_only`
  - whitespace fragments that should likely normalize away
- `dot_leader_like`
  - TOC/index leader-style dot runs
- `table_border_ascii_art`
  - border / table-shell / separator fragments
- `punct_symbol_only`
  - punctuation/symbol-only tokens not better explained by the previous layout buckets
- `digits_only`
  - digit-only tokens
- `latin_word_like`
  - ordinary Latin lexical material
- `latin_mixed`
  - Latin lexical material mixed with punctuation/digits/spacing
- `other_script_letters`
  - other non-Greek script residue
- `mixed_other`
  - small catch-all remainder

### Why this matters

This makes the problem much clearer:

- many “bad” tokens are actually ordinary Latin lexical material
- some are explicit extraction junk
- some are layout artefacts that need normalization
- some are suspicious short opaque fragments that deserve their own review bucket

That is a better basis for policy than treating all non-Greek tokens as one class.
