# DRAFT — separator-normalization prompt

**Status**: proposal for review. Not wired into
`build_token_category_review_bundle.py` / `_build_prompt` yet.

**Task class**: per-case normalization review.
**Review question**: is this standalone divider safely replaceable with `---`?

---

## What the model receives

The prompt is built in three layers:

1. **Wave preamble** — identical across every case in a wave. Sent once
   via Gemini `cached_content`; billed once.
2. **Per-case case file** — written to disk by the bundler, unchanged
   for the life of the wave.
3. **Per-case wrapper** — added by `_build_prompt` at send time (model +
   request bookkeeping only; no new semantic content).

---

## Layer 1 — Wave preamble (cached)

```
[PROJECT_CONTEXT]
You are helping normalize a Greek-language text corpus used to train a
subword tokenizer extension for Apertus-8B-2509. Our goal is to reduce
over-representation of repeated layout structures in tokenizer vocabulary
*without* losing meaning or breaking markdown structure where relevant.

[TASK_CONTEXT: separator normalization]
A "separator" in this corpus is a standalone thematic-break line made of
repeated non-letter characters — runs of `-`, `_`, `*`, `=`, em-dash, or
box-drawing chars — appearing alone on its own line with blank lines
above and below. Its role is to divide sections.

Separators are NOT:
- dot leaders (TOC layout runs like `..........`), which bridge a title
  and a page number on the same line;
- markdown table separator rows (e.g. `|---|---|`), which declare row
  type and column alignment inside a GFM table.

[PROPOSED_CHANGE]
We intend to replace every variant separator (`-----`, `_____`, `***`,
`======`, etc.) with the canonical markdown thematic break `---`. We are
asking you to confirm that this substitution preserves meaning in the
specific context below.

[CURRENT_CLEANER_RULE]
None yet for separators. The only related rule that already ships is
`normalize_layout_leader_runs` (dot-leader → `.....`), which applies to
inline dot-leader runs on a line with trailing page-number tokens.
```

## Layer 2 — Per-case case file

```
[MATCH_META]
review_case_id: <category>::<match_id>
category:       separator_run_like
pattern_family: <hyphen_ge4 | underscore_ge4 | asterisk_ge3 | mixed>
source:         <source_corpus>/<source_path>  (synthetic_page=<n>)
matched_text:   <literal matched span>

[CONTEXT_BEFORE]          (20 lines preceding the match, verbatim)
...
...
...

[MATCH]
<match type="separator_run" family="hyphen_ge4">-----------</match>

[CONTEXT_AFTER]           (20 lines following the match, verbatim)
...
...
...

[CONTEXT_AFTER_NORMALIZATION]
(Same 20-before / match / 20-after window, but with the matched span
replaced by `---`. Shadow-applied; not written to the corpus.)
...
<match type="separator_run" family="hyphen_ge4">---</match>
...

[REVIEW_QUESTIONS]
Answer ONLY from the evidence in this case file. Be conservative. If
unsure, answer `uncertain` rather than guess.

1. Is the matched span acting as a standalone thematic-break separator
   in this context (and not a dot leader, table separator, ASCII art
   border, or stray character run)?
2. Is `---` an interchangeable canonical form — does the normalized
   window preserve section-boundary semantics exactly?
3. Does the normalization preserve markdown structure if any is present
   nearby (list, heading, code fence, table)?
4. If you answer `no` or `uncertain` to any of the above, give one
   short sentence explaining the blocker.

Return ONLY valid JSON matching the declared schema.
```

## Layer 3 — Output schema (enforced via `responseJsonSchema`)

```json
{
  "is_separator_role": "yes" | "no" | "uncertain",
  "interchangeable_with_target": "yes" | "no" | "uncertain",
  "preserves_semantics": "yes" | "no" | "uncertain",
  "preserves_markdown_structure": "yes" | "no" | "not_applicable" | "uncertain",
  "blocker": "<short sentence or empty string>"
}
```

All fields required. No free-form prose elsewhere.

---

## Design notes

- **Line-based windows (20 before / 20 after)** replace the current
  240-char context slice. Markdown structure is inherently line-oriented,
  so section-boundary judgments need lines, not characters.
- **Shadow-applied `CONTEXT_AFTER_NORMALIZATION`** closes STAGE_1_2_REVIEW P2:
  the model sees the replacement, not an imagined replacement.
- **Match-type attribute on the tag** (`<match type="separator_run"
  family="hyphen_ge4">`) gives the model unambiguous anchor identity and
  solves the surrounding-chars-dragged-into-the-span problem that caused
  the U+03A2 hallucination on `control_private_use_replacement`.
- **Output is strictly enum-valued** for programmatic aggregation —
  no prose field beyond a one-sentence blocker reason.
- **Task reiteration at the end of the case file**, literal, not
  implicit. Current `_build_prompt` ends with the case_text; the
  `[REVIEW_QUESTIONS]` block belongs inside the case text itself so the
  reiteration is last.

## Out of scope for this template

- Border/ASCII-art fragments (`|--------|`) — stay in a separate review
  per MODEL_REVIEW_SPEC §8.2 until evidence says they are interchangeable.
- Escaped underscore chains (`\_\_\_\_\_`) — same.
