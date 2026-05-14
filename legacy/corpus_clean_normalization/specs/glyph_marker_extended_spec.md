# `glyph_marker_extended` — spec

**Status**: draft. Extends the existing `glyph_font_like` matcher
family with PostScript glyph-name patterns that the current cleaner
misses.

**Counter surfaced by this category**:
`page_glyph_marker_count` (Counter 2 of the three-counter
`drop_low_salvage_pages` design).

## Why this spec exists

A 5,000-row probe of `openarchives.gr` (2026-04-21) revealed that the
current `BAD_LINE_AC` literal set
(`glyph<c=`, `GLYPH<`, `MS-Bold-`, `font=/`, `FontName=`, plus `&lt;`
variants) **misses** three large noise sources:

| Pattern | Rows | Hits | Current handling |
|---|---:|---:|---|
| bare `GLYPH` (word-boundary, no `<` after) | 172 / 5000 | 84,798 | missed |
| `/uni<4-6 hex>` (PostScript Unicode-codepoint glyph name) | 45 / 5000 | 68,510 | missed |
| `/gid<digits>` (PostScript glyph-ID reference) | 24 / 5000 | 19,085 | missed |
| `/hyphenminus`, `/space`, `/period`, `/glyph`, … (PostScript glyph names) | sparse | sparse each | missed |

These are PDF extractors that, instead of decoding glyph indices into
characters, dump the raw Adobe glyph-name strings into the text. Almost
never legitimate in Greek prose.

## Extensions

### Literal-set additions to `BAD_LINE_AC` + matcher

Add these literals to the existing `glyph_font_like` family:

```
# Bare GLYPH (word boundary match handled by matcher)
GLYPH            # only counts when NOT followed by `<` (the `GLYPH<` variant already exists)
;GLYPH
 GLYPH
>GLYPH
glyph            # lowercase bare-form
hyphenminus      # bare PostScript glyph name
```

### Literal set for PostScript glyph-name references

Adobe Glyph List entries that appear as `/<name>` in broken PDF
extractions. Conservative subset (the ones we've seen + the most
frequent in the full AGL):

```
/hyphenminus     /space           /period          /comma
/colon           /semicolon       /slash           /backslash
/parenleft       /parenright      /bracketleft     /bracketright
/braceleft       /braceright      /quotesingle     /quotedbl
/exclam          /question        /asterisk        /plus
/minus           /equal           /less            /greater
/ampersand       /percent         /at              /dollar
/numbersign      /underscore      /asciitilde      /asciicircum
/endash          /emdash          /hyphen          /bullet
/copyright       /registered      /trademark       /degree
/plusminus       /multiply        /divide          /section
/paragraph       /dagger          /daggerdbl       /ellipsis
/glyph
```

### Regex additions

Unicode-codepoint and glyph-ID forms — too many variants to enumerate
as literals:

```
/uni[0-9A-Fa-f]{4,6}       # e.g. /uni03B1, /uni1F000
/gid\d+                     # e.g. /gid456
/g\d+                       # e.g. /g12 (short form)
/G\d+                       # uppercase form
CID\+?\d+                   # Adobe CID prefix
```

## Wiring

- **Cleaner side** (`BAD_LINE_AC` in `cleaning_module.rs`): extend the
  Aho-Corasick literal set with the additions above. A line that
  contains any of these patterns is rejected (line-level strip), same
  as today's `GLYPH<` handling.
- **Matcher side** (`glyph_font_like` category in `glossapi_rs_noise`):
  extend the pattern set with literals + regex. Per-page count
  surfaces as `PAGE_CATEGORY_MATCH_COUNTS["glyph_font_like"]` →
  `page_glyph_marker_count`.
- **Code-fence guard**: both sides already respect fenced blocks; the
  added literals inherit that.

## Calibration impact

Expected effect on page-glyph-count distribution: the 45 worst-offender
docs in the probe would jump from their current `glyph_marker_count`
(which captures only `GLYPH<` + `font=/`) to a number ~5-10× higher —
because `/uni<hex>` alone contributes 68k hits in 45 docs (~1,500 per
doc) vs the ~400 per doc the current counter sees.

This means:
- **Today's threshold calibration (`≥ 160 → 93% noisy`) is likely
  LOW** — it was measured against the incomplete counter. Pages that
  are catastrophically corrupted show up as "merely noisy" because the
  big patterns weren't counted.
- **After this extension lands, re-run stratified sampling** (per
  `three_counter_pipeline.md` stage 2-3) to get a corrected threshold.

## Risks / false-positive considerations

- `/period`, `/space`, `/slash` could theoretically appear in URL paths
  or filenames (`/space/shuttle.html`, `/bin/slash/foo`). In practice
  Greek prose corpus has near-zero such hits. Accept the FP risk at
  this signal strength.
- `CID+\d+` could match something in programming context. Same accept.
- `GLYPH` (bare word) might appear in an unrelated glossary / tech
  article. At 84,798 hits in 172 docs (~500/doc) it's catastrophically
  concentrated — FP rate is negligible.

## Testing

Positive (must match / line-reject):
- `… /uni03B1 /uni03B2 /uni03B3 …` → 3 regex matches.
- `… /gid456 /gid789 …` → 2 regex matches.
- `… hyphenminus hyphenminus hyphenminus …` → 3 literal matches.
- `Text with GLYPH somewhere in it.` → 1 match (bare GLYPH).
- `… /hyphenminus /space /period …` → 3 matches.

Negative (must NOT match):
- `Ορθά σχηματίζεται η λέξη σύμβολο.` → no match.
- `GLYPH<c=3,font=/X>` → matches via EXISTING `GLYPH<` trigger, not a
  regression, but the bare-`GLYPH` rule must use a word-boundary that
  doesn't DOUBLE-count this line.
- Inside a fenced code block: `​\`\`\`python\n/uni03B1\n\`\`\`` → no
  match (code-fence guard).
