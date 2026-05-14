# DRAFT — slash+dash mixed-token classification prompt

**Status**: proposal for review. Not wired in yet.

**Task class**: per-case classification review (Task D in
`NORMALIZATION_DESIGN_20260420.md`).

**Review question**: what function does this mixed slash+dash token serve
in context?

---

## Why a dedicated task

The `table_border_ascii_art` inventory in
`tokenizer_analysis/inspection/non_greek/fresh/glossapi_only/categories/`
contains mixed slash+dash tokens: `//`, `://`, `-|`, `=/`, `-->`, `<!--`,
`-|--------`, `|//`, and similar. The same byte sequence can serve
different functions in different places:

- URL fragments (`//`, `://`) — keep
- HTML comment residue (`<!--`, `-->`) — mostly already stripped by
  `strip_tags_custom`; some escape
- TOC-adjacent dividers — normalize to `.....`
- Standalone separator residue — normalize to `---`
- Markdown table fragments — handled by GFM parser (Task B)

A flat regex cannot disambiguate. Gemini classifies; we map the
classification to a deterministic rule.

---

## Layer 1 — Wave preamble (cached)

```
[PROJECT_CONTEXT]
You are helping identify the function of a one-off token in a Greek
corpus used to train a subword tokenizer extension for
swiss-ai/Apertus-8B-2509. Our goal is to avoid letting one-off mixed
slash-and-dash tokens pollute the tokenizer vocabulary.

[TASK_CONTEXT: slash+dash mixed-token classification]
The tokenizer produced one-off tokens containing both `/` and `-`. They
come from different sources in the corpus. Your job is to classify what
this specific occurrence is doing.

Possible functions (closed set):

- url_fragment: part of a URL or protocol prefix. Typical markers
  include `://`, `www.`, `http`, `https`, `.com`, `.gr`. Surrounding
  context should look URL-like.
- html_comment_residue: leftover from HTML comment syntax (`<!--`,
  `-->`) that escaped normal HTML stripping.
- separator: standalone thematic-break line composed mostly of dashes
  with incidental slashes, on its own line with blank lines around.
- toc_leader: part of a table-of-contents layout bridging a section
  title and a page number on the same line.
- markdown_table_fragment: appears inside a GFM table (header row above,
  cell borders).
- other: doesn't fit any of the above.

[PROPOSED_USE]
We map your classification to a fixed rule:
- url_fragment → keep
- html_comment_residue → confirm HTML stripping coverage, extend if needed
- separator → normalize to `---`
- toc_leader → normalize to `.....`
- markdown_table_fragment → the GFM parser normalizes it in Task B
- other → manual review queue

You do NOT write the rule. You classify; we synthesize.
```

## Layer 2 — Per-case case file

```
[MATCH_META]
review_case_id: slash_dash_mixed::<match_id>
source:         <source_corpus>/<source_path>
matched_text:   <literal token>

[CONTEXT_BEFORE]          (20 lines preceding the match, verbatim)
...

[MATCH]
<match type="slash_dash_mixed">{matched_text}</match>

[CONTEXT_AFTER]           (20 lines following the match, verbatim)
...

[REVIEW_QUESTIONS]
Answer ONLY from the evidence in this case file. Be conservative. If
unsure, answer `uncertain`.

1. Which function does the matched token serve in this context? Pick
   exactly one from the closed set.
2. If you answer `uncertain`, give one short sentence explaining the
   blocker.
```

## Layer 3 — Output schema (enforced via `responseJsonSchema`)

```json
{
  "function": "url_fragment" | "html_comment_residue" | "separator" | "toc_leader" | "markdown_table_fragment" | "other" | "uncertain",
  "blocker": "<short sentence or empty string>"
}
```

All fields required.

---

## Sampling policy

- First wave: 100 cases, random sampling from the slash+dash subset of
  `table_border_ascii_art` match inventory.
- No stratification — the token-type population is small (~30 distinct
  literals in the evidence).
- Aggregator decides the dominant function per `matched_text` family;
  if one family is >80% one function, promote the mapped rule; else
  manual review.

## Design notes

- This is the only review task added on 2026-04-20 beyond the original
  three (separator / MD-table audit / page noise). It was carved out
  because the other three don't cover the `//` + dashes evidence.
- Model is NOT asked for a regex. Closed-enum classification only.
- Map from classification → rule is in the preamble so the model
  understands what its answer will be used for (improves calibration)
  but it never writes the rule itself.
- If `html_comment_residue` dominates, that's a signal to audit
  `strip_tags_custom` for coverage gaps on malformed HTML, not to write
  a new regex.
