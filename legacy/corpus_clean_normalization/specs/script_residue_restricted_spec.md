# `script_residue_restricted` matcher category — spec

**Status**: draft. Built on the 2026-04-21 probe of
`tokenizer_analysis/inspection/non_greek/fresh/glossapi_only/short_nonascii_latin_like_decoded.json`.

**Counter surfaced by this category**:
`page_script_residue_count` — the third of the three-counter
`drop_low_salvage_pages` design.

## What it tracks

Non-ASCII Latin codepoints in blocks that are almost never legitimate
in Greek prose:

- **Latin Extended-A** — U+0100..U+017F (44 tokens in the 246-token
  `short_nonascii_latin_like` inventory).
- **Latin Extended-B** — U+0180..U+024F (67 tokens — the main driver;
  these are the `ǂ`, `ǎ`, `ǐ`, `Ǖ`, `Ƞ`, `Ț`, etc. chars that appear
  as PDF-extraction pseudo-Greek mojibake).

Total: **111 counter-worthy tokens** from the 246.

## What it deliberately EXCLUDES (to avoid false positives)

From the same 246 inventory:

- **Latin-1 Supplement** (124 tokens: `é`, `á`, `ü`, `ç`, `ö`, `ñ`,
  etc.) — extremely common in legitimate Greek prose for foreign
  names and loanwords. FP magnet.
- **IPA Extensions** (8 tokens: `ʌ`, `ʆ`, `ɲ`, `ʏ`) — legitimate in
  Greek linguistics papers.
- **Dingbats** (3 tokens: `✝`, `✟`, `✞`) — legitimate in religious /
  church corpora.
- **ASCII-only tokens** (76 — they only made the `short_nonascii_latin_like`
  list because of a related non-ASCII char in a length-2 pair; we
  don't want to re-match the ASCII side).

## Matching strategy

Option A — **per-codepoint literal set** (preferred).
Match any single char in the Ext-A or Ext-B ranges as a unit, not the
246 merged BPE tokens. Rationale: the matcher's old approach used BPE
tokens as Aho-Corasick literals, which dragged Greek context into
match anchors (the STEERING §C3 bug). Per-codepoint matching avoids
that.

```
range U+0100..U+017F  → kind=latin_ext_a
range U+0180..U+024F  → kind=latin_ext_b
```

Option B — **script-salad bigram regex** (additional signal):
Flag any 3-char span that contains BOTH a Latin-Ext-A codepoint and a
Latin-Ext-B codepoint. This is the `'ĮȚ'` / `'ȚĮ'` / `'Ƞȣ'` pattern —
pseudo-Greek mojibake bigrams that render as fake Greek letters.

```
regex: [\u0100-\u017F][\u0180-\u024F]|[\u0180-\u024F][\u0100-\u017F]
```

Counter-worthy: YES (strong noise indicator).

## Wiring

- New category in `glossapi_rs_noise` matcher spec:
  `script_residue_restricted`.
- Pattern family `fresh_glossapi_only_script_residue_ext_a_b`.
- Per-page count → `PAGE_CATEGORY_MATCH_COUNTS["script_residue_restricted"]`
  → surfaces as `page_script_residue_count` in the aggregated view.
- **Code-fence guard**: matcher must respect fenced code blocks
  (matches inside ``` / ~~~ fences don't count toward the page
  counter). The cleaner already has this guard; the matcher should
  mirror it.

## Relation to existing cleaning

SCRIPT_SETS `unusual` already strips all of U+0100..U+024F at the
per-char level (Latin-Ext-A + -B + Additional + IPA Ext). So
**post-clean text should have zero `page_script_residue_count`**.
This counter therefore runs on the **pre-clean** text to identify
pages that were heavily corrupted BEFORE cleaning — those pages
should be dropped even if cleaning would erase the visible residue,
because the underlying text is not recoverable signal.

In other words: the counter is an input to
`drop_low_salvage_pages`, NOT a per-line cleaning rule.

## Testing

Positive (must match, per-char):
- `ǐ` (U+01D0, Latin-Ext-B) → 1 match.
- `Ǖ` (U+01D5) → 1 match.
- `ĮȚ` (Ext-A + Ext-B bigram) → 2 matches + 1 script-salad regex hit.

Positive (must match, script-salad regex):
- `ǂǑ`, `ȠȪ`, `Țț` → all regex hits.

Negative (must NOT match):
- `é` (U+00E9, Latin-1 Supp) → no match.
- `ʌ` (U+028C, IPA) → no match.
- `✝` (U+271D, Dingbats) → no match.
- Any ASCII char → no match.
- Any Greek char (U+0370-03FF, U+1F00-1FFF) → no match.
- `ǐ` inside a fenced code block → no match (code-fence guard).
