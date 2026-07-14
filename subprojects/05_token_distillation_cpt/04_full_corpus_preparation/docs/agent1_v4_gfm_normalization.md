# Agent 1 v4 HTML-to-GitHub-Markdown normalization prototype

Status: dry-run sample prototype. This has **not** been ported into the GlossAPI production pipeline, and it does not overwrite any raw review document.

## Outcome on the 348 reviewed documents

The prototype first runs GlossAPI's complex-repetition cleaner and then converts recognized HTML into HTML-free GitHub-Flavored Markdown (GFM). The completed audit reports:

- 348 documents checked; 165 changed and 183 remained byte-identical.
- 147 documents contained recognized HTML.
- 117,881 HTML start tags were handled; zero recognized HTML tags remain.
- 1,730 top-level HTML tables became GFM pipe tables; one nested table was flattened inside its parent cell.
- 74,551 source table cells were retained structurally exactly once in 18,427 emitted rows; eight cells in one unrepresentable nested table and eight malformed orphan cells were flattened without dropping their text.
- 25 runaway-repetition spans in 22 documents became `<!-- repeating-text-removed -->`, removing 156,320 repeated characters.
- Only 64 non-repetition characters were removed: two `<!-- Table content goes here -->` placeholder comments.
- The normalizer is idempotent on all 348 outputs.

The presentation is `normalization.html` in the local review site. It shows the policy, aggregate counts, and lazy-loaded raw text, normalized Markdown, and sandboxed rendering for every changed document.

## Why this must run before existing GlossAPI cleanup

The current GlossAPI code already does useful Markdown cleanup, but it does not perform structural HTML-to-Markdown conversion:

1. `src/glossapi/ocr/utils/repetition.py` detects and removes runaway complex repetition. The prototype reuses `replace_complex_repetitions` directly.
2. `src/glossapi/ocr/utils/cleaning.py::canonicalize_markdown` normalizes whitespace, dehyphenation, placeholder cells, empty tables, and citation superscript artifacts after Markdown already exists. Do not duplicate those operations in the converter.
3. `rust/glossapi_rs_cleaner/src/cleaning_module.rs::strip_tags_custom` decodes entities and strips tags while preserving approved comments. It loses table, emphasis, list, and paragraph semantics, so HTML-to-GFM must run before it.
4. `rust/glossapi_rs_cleaner/src/table_analysis_module.rs` validates existing pipe tables; it does not convert HTML tables. Reuse it to validate converter output.
5. `rust/glossapi_rs_cleaner/src/table_remover_module.rs` replaces malformed tables with `<!-- table-removed -->`; it is a removal stage, not a converter.
6. `GlossSection` already consumes GFM headings, lists, and tables downstream.

The intended production order is therefore:

`complex repetition removal → HTML-to-GFM conversion → canonicalize_markdown → Rust GFM table validation/removal → existing downstream phases`

The dry run deliberately stops after the first two operations so it can demonstrate the converter without silently duplicating or changing existing GlossAPI canonicalization.

## Decisions for every observed HTML element

GFM pipe tables require a header row plus a delimiter row, permit inline content in cells, and do not support block elements inside cells. These constraints follow the [GFM specification](https://github.github.com/gfm/#tables-extension-) and [GitHub's table syntax](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/organizing-information-with-tables).

| Observed element(s) | Conversion | Loss decision |
| --- | --- | --- |
| `table`, `thead`, `tbody`, `tfoot`, `tr`, `th`, `td` | GFM pipe table | Retain each source cell once. Expand `rowspan`/`colspan` with empty geometry cells instead of repeating text. Merge multiple leading header rows per column with ` / `. If no header exists, emit an empty header row rather than inventing labels. Preserve a consistent left/center/right alignment in delimiter colons; discard conflicting alignment. Escape cell pipes. |
| `caption` | Italic paragraph before the table | Retain caption text because GFM has no table-caption field. |
| `b`, `strong` | `**bold**` | Retain content. |
| `i`, `em` | `*italic*` | Retain content. |
| `del`, `s`, `strike` | `~~strikethrough~~` | Retain content. |
| `br` | Markdown hard break outside tables; `; ` inside cells | Preserve the boundary. A semicolon is used in cells because GFM table cells cannot contain block-level line breaks. |
| `p`, `div`, `section`, and other block wrappers | Blank-line Markdown block boundary | Retain content and existing Markdown; discard HTML layout attributes. |
| `ul`, `ol`, `li` | `-` or numbered Markdown list | Retain order and items. Flatten lists with `; ` separators only when they occur inside table cells. |
| `sup`, `sub`, `u`, `span` | Plain inline content | Remove the unexpressible style and all attributes, but retain the textual payload. GitHub documents superscript/subscript using raw HTML, so a zero-HTML target has no faithful equivalent. Deleting the text itself would corrupt mathematical symbols, citations, and words. A separating space is added only when a flattened `sup`/`sub` immediately follows a closing Markdown delimiter and adjacency would otherwise break that emphasis/code token. See [GitHub basic formatting](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax). |
| `math` | `$…$` or `$$…$$` | Retain the existing TeX-like payload, following [GitHub mathematical-expression syntax](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions). |
| `img` | `![alt](src)` | Convert the five images with a source. Remove seven source-less decorative image elements completely. Do not invent captions or descriptions. |
| `input` | Removed | Both observed inline checkbox elements are OCR/manuscript artifacts, not Markdown task-list items. |
| `script`, `style`, embedded media, forms, and executable elements | Removed with content | No safe corpus-text representation. None occurred in this sample, but the conservative fallback is implemented and tested. |
| HTML comments | Removed, except approved removal markers | Preserve `<!-- repeating-text-removed -->`, `<!-- table-removed -->`, and `<!-- text-missing -->`; remove ordinary comments. |
| Unknown angle-bracket text | Escaped literal text | Preserve OCR citations and pseudo-tags such as `<<εταιρία>>` as text. Preserve valid GFM URI/email autolinks unchanged. Nested angle pairs are resolved in one pass. |

Common but unobserved elements also have conservative mappings: `h1`–`h6` to ATX headings, `a` to Markdown links when the destination is safe, `code`/`pre` to inline/fenced code, `blockquote` to `>`, and `hr` to `---`. Unknown executable or layout markup is never passed through as raw HTML.

## Existing Markdown preservation

Documents with no recognized HTML or repetition replacement remain byte-identical. Across all documents, before and after counts close as follows:

| Structure | Before | After | Explanation |
| --- | ---: | ---: | --- |
| Existing ATX headings | 6,625 | 6,625 | Preserved |
| Existing fence lines | 22 | 22 | Preserved |
| Existing Markdown links | 147 | 147 | Preserved |
| Markdown images | 5,383 | 5,388 | Five HTML images converted |
| GFM table delimiter rows | 18 | 1,748 | 1,730 HTML tables converted |
| Strong-emphasis delimiter pairs | 8,438 | 10,003 | Existing pairs preserved; HTML bold and body-header cells added |

Each normalized output is normalized a second time and must be byte-identical. The audit also rejects any residual recognized HTML, loss of existing headings/fences/links, missing existing images or tables, inconsistent table column counts, and unsafe rendered elements.

In addition, the same Markdown parser used for the presentation counts semantic tokens before and after normalization. Headings, fenced and inline code, links, images, ordered and unordered lists, blockquotes, strong/emphasis/strikethrough, and existing GFM tables must all be non-decreasing per document. This catches structure loss that a delimiter-only check could miss.

The aggregate semantic-token closure is:

| Markdown token | Before | After |
| --- | ---: | ---: |
| Headings | 6,637 | 6,638 |
| Fenced code | 11 | 11 |
| Indented code blocks | 66 | 66 |
| Inline code | 2 | 2 |
| Links | 1,694 | 1,694 |
| Images | 5,385 | 5,390 |
| Unordered lists | 898 | 898 |
| Ordered lists | 10,386 | 10,419 |
| Blockquotes | 62 | 62 |
| Strong emphasis | 8,349 | 9,914 |
| Emphasis | 17,013 | 17,240 |
| Strikethrough | 4 | 4 |
| GFM tables | 17 | 1,747 |

Flattening `sup`/`sub` required 228 boundary spaces to prevent a footnote or exponent from attaching to a preceding closing Markdown delimiter and disabling that existing emphasis/code span.

## Table-specific edge cases observed

- 808 data-only tables received a synthetic empty header row.
- 263 multirow headers were merged without duplicating their text.
- 641 cells used `colspan`; 511 used `rowspan`.
- Three columns contained conflicting source alignment declarations, so their Markdown alignment was left unspecified.
- Forty-two lists inside cells were flattened.
- One nested table (eight cells) was flattened with explicit ` / ` cell and `; ` row separators because GFM cannot nest a block table inside a table cell.
- Eight malformed orphan `td`/`th` elements had no recoverable table geometry, so their content was retained inline and the structural downgrade was recorded.
- Two repetition markers occurred between HTML table rows rather than inside cells. A GFM table cannot contain a comment between pipe rows, so each marker is retained exactly once immediately after its converted table.

## Presentation and artifacts

- Audit index: `data/gfm_normalization_audit.json`.
- Changed-document payloads: `data/gfm/documents/<opaque_id>.json`.
- Each changed payload contains the raw-text hash, normalized Markdown hash, normalized Markdown source, renderer identifier, and pre-rendered review HTML.
- The browser fetches raw and normalized payloads only when the reviewer opens a card.
- Rendering occurs in a script-free sandbox with external image loading disabled. The render is presentation-only and is never fed back into the corpus.

The local renderer is `markdown-it-py` in CommonMark mode with table and strikethrough extensions. GitHub rendering may differ slightly, so source Markdown—not the preview HTML—is the normalization artifact.

Install the prototype-only dependency from `requirements-gfm-prototype.txt`. It is deliberately separate from `requirements-runtime.txt`, whose hash is frozen into the earlier review packet.

## Porting gate

Before moving this into GlossAPI:

1. Review the presentation examples and approve or revise the decisions above, especially blank table headers, multirow-header joining, nested-table flattening, and plain-text handling of `sup`/`sub`/`u`.
2. Move the converter into the OCR normalization path immediately after complex-repetition removal and before Rust tag stripping.
3. Run the existing `canonicalize_markdown` and Rust table validator instead of reimplementing their behavior.
4. Add the 348-sample audit as a regression fixture or a stable representative subset, retaining the idempotence and structure-preservation gates.
5. Profile the production implementation before running it over the full corpus.
