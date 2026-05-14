# Drop-level decisions — which patterns drop what, and why

**Authoritative table**: for every cleaning rule, at which granularity it
drops (match / line / page / doc), plus the evidence base behind the
decision.

Columns:
- **Granularity**: match / line / page / doc.
- **Calibration**: WHERE the decision came from (Gemini wave, empirical
  probe, design choice, corpus-wide invariant).
- **FP risk**: what legitimate content the rule MIGHT consume.
- **Flag status** (2026-04-22 review): ✅ confident · ⚠️ cutting close ·
  🚩 flagged for fresh Gemini review.

## Match-level (strip just the matched span/char; line preserved)

| rule | granularity | calibration | FP risk | flag |
|---|---|---|---|---|
| `is_unicode_noise_char` — U+FFFD U+00AD U+200B-F U+2060 U+FEFF controls PUA | char | corpus-wide invariant (these codepoints = decoder failure / private-use / formatting invisibles; zero semantic role in Greek) | none | ✅ |
| HTML/XML tags `<...>` (non-comment) via `strip_tags_custom` | span | markdown-corpus invariant — post-conversion tags are artifacts | none | ✅ |
| SCRIPT_SETS `unusual` per-char strip — Latin Ext-A/-B/Additional, IPA, Coptic, Cyrillic + Supp | char | frozen policy 2026-04-21 — "only ranges with no semantic meaning for this corpus"; see `glossapi_cleaning_vs_normalization_scope.md` | bibliographies citing Romanian/Turkish/Polish/Czech names lose diacritics (word stem preserved) | ⚠️ |

## Line-level (drop entire line on trigger)

| rule | granularity | calibration | FP risk | flag |
|---|---|---|---|---|
| `GLYPH<c=`, `glyph<c=`, `GLYPH&lt;`, `glyph&lt;c=` | line | structural PDF-extraction pattern — `GLYPH<c=N,font=/X>` is never legitimate prose | none | ✅ |
| `font=/`, `FontName=`, `MS-Bold-` | line | same — PDF font-switch tag residue | none | ✅ |
| `bare hyphenminus` (the word) | line | PS glyph name, not a word in Greek or English prose | none | ✅ |
| `/[A-Z]{6}\+[A-Z][A-Za-z0-9-]+` — PDF font-subset regex | line | Adobe convention `/XQDMQS+CenturyGothic`; zero legitimate use | none | ✅ (though span drop would be consistent) |
| bare `GLYPH` (no `<` after) | line | 5k-row probe: 84,798 hits in 172 of 5000 docs → concentrated in corrupted PDFs | Greek tech article mentioning "GLYPH" — loses the one line | ⚠️ |
| `/hyphenminus`, `/space`, `/period`, `/comma`, `/parenleft`, `/parenright`, `/bracketleft`, `/bracketright`, `/braceleft`, `/braceright`, `/quotesingle`, `/quotedbl`, `/exclam`, `/question`, `/asterisk`, `/plus`, `/minus`, `/equal`, `/less`, `/greater`, `/ampersand`, `/percent`, `/at`, `/dollar`, `/numbersign`, `/underscore`, `/asciitilde`, `/asciicircum`, `/endash`, `/emdash`, `/hyphen`, `/bullet`, `/copyright`, `/registered`, `/trademark`, `/degree`, `/plusminus`, `/multiply`, `/divide`, `/section`, `/paragraph`, `/dagger`, `/daggerdbl`, `/ellipsis`, `/glyph`, `CID+` (~40 literals) | line | design choice (added 2026-04-21 based on `glyph_marker_extended_spec.md`); NOT reviewed by Gemini at the line-vs-span granularity | a URL path like `/period-history/` or a filename `example/period.pdf` would kill the whole line — losing surrounding Greek prose | 🚩 |
| `/uni[0-9A-Fa-f]{4,6}` — Unicode-codepoint PS glyph references | line | design choice — same path as above | same as above: URL/path matches on `/uni...` or code examples | 🚩 |
| `/g(?:id)?\d+` — glyph-ID PS references (`/g12`, `/gid456`) | line | design choice — same path as above | same: `/g12/...` could appear in URLs | 🚩 |

## Page-level (drop synthetic page within doc)

| rule | granularity | calibration | status |
|---|---|---|---|
| `drop_low_salvage_pages(min_retention_ratio)` | page (synthetic, split by markdown headers) | not wired into v3 run; spec designed 2026-04-21 for corruption-bounded page salvage | **NOT ACTIVE** — no rule currently uses page-level drop. Proposed for `script_residue_restricted` (see Doc-level table below). |

## Doc-level (drop entire parquet row)

| rule | granularity | calibration | FP risk | flag |
|---|---|---|---|---|
| `font_name_literal ≥ 1` | doc | Gemini wave on v2 (n=40) → 36/40 drop at counter=1; threshold tightened to 1 | literal set is `+<FontName>` form only (the `+` is load-bearing); legit prose would write "γραμματοσειρά Palatino" without the `+` | ⚠️ |
| `glyph_font_like ≥ 14` | doc | Gemini wave on v2 (n=50) → rolling-window 80% drop at counter=14 | threshold is high enough that single mentions of "glyph" don't hit it | ✅ |
| `script_residue_restricted ≥ 9` | **doc** (currently) — **out of Gemini-review scope** | Gemini wave on v2 (n=50) was on **synthetic PAGES** with counter ≥ 1, not on per-doc aggregation. Applying the threshold at DOC level is a granularity change that the verdict doesn't cover (per `feedback_dont_generalize_beyond_test_parameters.md`). | bibliographies citing ≥9 non-Greek-non-French-non-Spanish Latin-diacritic names get the whole doc dropped. Gemini `other_unknown`=24/50 dominant-signal. | 🚩 — **MUST** re-review at per-doc granularity OR revert to per-page drop (which is what the review actually covered) |

## Reasoning key

The three 🚩 flags (PS-glyph literals, PS-glyph regex, script_residue doc-drop) share a single pattern: **the rule was made locally (design choice / inherited behavior) without a dedicated Gemini wave on the specific granularity question**:

- For PS-glyph rules: Gemini reviewed "is this page noise?", not "should the WHOLE line containing `/period` be dropped or just the span?". Two different granularity decisions.
- For script_residue doc-drop: Gemini reviewed the threshold value (correctly), but the sampled population was pages pre-selected for having ≥1 match — bibliography-only pages with 9+ matches and zero other noise signals may be under-represented in that sample.

## Planned next-wave Gemini review (2026-04-22)

**Scope**: send a 1000-case sample for each of the 2 line-drop rules (PS-glyph literals + PS-glyph regex) asking "should the WHOLE line be dropped or JUST the matched span?"

Prompt structure (locked per `feedback_data_inspection_format.md`):
```
[CONTEXT]
<the full line containing the match, with surrounding 5-10 lines above/below>

[QUESTIONS]
1. is_match_noise (yes / no / uncertain)
2. should_drop_whole_line (yes / no / uncertain)
3. surrounding_context_is_legitimate_prose (yes / no / uncertain)
4. short_reason (≤ 40 words)
```

Answer shape: if `is_match_noise=yes AND surrounding_context_is_legitimate=yes` on ≥ 20% of cases, we flip to **span drop** in the Rust cleaner. Otherwise line drop is fine.
