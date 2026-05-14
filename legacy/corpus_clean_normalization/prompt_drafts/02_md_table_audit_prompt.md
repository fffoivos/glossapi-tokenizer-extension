# DRAFT — MD-table normalization audit prompt

**Status**: proposal for review. Not wired into
`build_token_category_review_bundle.py` / `_build_prompt` yet.

**Task class**: post-implementation audit (not per-case review).

**Why this is not a per-case Gemini task**: GFM/CommonMark specifies
that a table separator cell is valid with ≥3 hyphens (optionally
colon-prefixed/suffixed). Any wider run renders identically. So the
conceptual question "can an MD table render OK at minimum width?" is
**settled by spec** — no Gemini needed. The practical question "is this
span actually a table separator and what is its minimum form?" is
**settled by a GFM parser** — no Gemini needed either.

What we *do* use Gemini for: a sampled audit of transformed tables, run
*after* the Rust-side parser-validated normalization pass. The audit
confirms that the collapse preserved rendering and did not catch
false-positive lines (ASCII art, styled dividers, broken extraction
residue).

---

## Programmatic path (no Gemini per case)

1. Rust matcher finds candidate lines (already done:
   `markdown_table_separator_row` / `separator_run_ge4`).
2. For each candidate, parse the surrounding block with a GFM parser
   (`pulldown-cmark` on the Rust side, or `markdown-it-py` on Python).
   Accept only if it parses as a real table separator row with a header
   row directly above.
3. Collapse each accepted cell to `---` / `:---` / `:---:` / `---:`
   based on its colon pattern.
4. Emit `tables_normalized.jsonl` with
   `{case_id, before_text, after_text, cell_count, alignment_vector}`.

Only after step 4 does Gemini enter — as an audit, not a classifier.

---

## Layer 1 — Wave preamble (cached)

```
[PROJECT_CONTEXT]
You are auditing a post-implementation normalization pass on a Greek
text corpus. Our goal is to reduce over-representation of redundant
markdown table separator widths in the tokenizer vocabulary.

[TASK_CONTEXT: markdown table separator audit]
GFM/CommonMark specifies that a table separator cell is valid when it
contains at least three consecutive hyphens, optionally prefixed with
`:`, suffixed with `:`, or both (for left/center/right alignment).
Wider cells render identically to minimum-width cells. We therefore
collapse every separator cell to its minimum canonical form:
  `---`    (default / left-aligned)
  `:---`   (left-aligned, explicit)
  `:---:`  (center-aligned)
  `---:`   (right-aligned)

[PROPOSED_CHANGE]
A GFM parser has already validated each table and produced a `before`
and `after` rendering. Your job is to confirm that `after` is
semantically identical to `before`: same column count, same alignment
per column, same surrounding rows, and no unintended content change.
```

## Layer 2 — Per-audit-case case file

```
[AUDIT_META]
audit_case_id:  <source>::<table_offset>
source:         <source_corpus>/<source_path>
cell_count:     <int>
alignment:      <per-column list: left|center|right|default>

[CONTEXT_BEFORE]          (5 lines preceding the table, verbatim)
...

[TABLE_BEFORE]            (the full table as extracted, unchanged)
| Header A | Header B | Header C |
|----------|:--------:|---------:|
| a        | b        | c        |

[TABLE_AFTER]             (same table, separator row collapsed)
| Header A | Header B | Header C |
| --- | :---: | ---: |
| a        | b        | c        |

[CONTEXT_AFTER]           (5 lines following the table, verbatim)
...

[AUDIT_QUESTIONS]
Answer ONLY from the rendered evidence in this case file. Be
conservative. If unsure, answer `uncertain`.

1. Does TABLE_AFTER render as a valid GFM table with the same column
   count and alignment as TABLE_BEFORE?
2. Are the surrounding lines unchanged (check CONTEXT_BEFORE /
   CONTEXT_AFTER)?
3. Was the original TABLE_BEFORE actually a GFM table (i.e., not ASCII
   art, not extraction residue that happens to contain pipe-delimited
   hyphen runs)?
4. If you answer `no` or `uncertain` to any of the above, give one
   short sentence explaining the blocker.
```

## Layer 3 — Output schema (enforced via `responseJsonSchema`)

```json
{
  "renders_identically": "yes" | "no" | "uncertain",
  "surroundings_unchanged": "yes" | "no" | "uncertain",
  "original_was_valid_gfm_table": "yes" | "no" | "uncertain",
  "blocker": "<short sentence or empty string>"
}
```

---

## Sampling policy for the audit

- Run on a stratified sample of ~100 transformed tables: 30 from each
  of the largest three source corpora that actually contain tables
  (`openarchives.gr`, `Apothetirio_Kallipos`, `opengov.gr-diaboyleuseis`
  are likely candidates), plus ~10 edge cases: smallest tables, largest
  tables, tables with unusual alignment vectors, tables inside code
  blocks, tables adjacent to headings.
- Block the pass promotion if `renders_identically == no` rate > 1%, or
  if `original_was_valid_gfm_table == no` rate > 5% (that flags a
  parser false-positive that slipped through GFM validation).

## Design notes

- The GFM parser does the structural decision; Gemini confirms the
  rendering invariant holds. Cheaper and safer than asking Gemini to
  classify every separator row.
- Parser choice: for Rust-side, `pulldown-cmark` with GFM tables
  extension enabled; for Python-side, `markdown-it-py` with the
  `tables` plugin. Both emit the same canonical alignment vector.
- Before/after line contexts are only 5 lines each because the audit is
  narrow — we're not asking about paragraph-scale semantics here.
