# Added-token curation policy — C3 vocab

**Date**: 2026-05-17. **Status**: agreed policy for the C3 added vocab.
Revised this date to widen the removal list after audit discussion.

## Purpose

After cutoff selection (`02_1_4_cutoff_analysis`) but before
implementation (`02_2_tokenizer_implementation`), there's a curation
step: of the added units the cutoff would keep, which ones should
actually be **removed** from the merge table because they are
extraction or encoding artifacts that do not represent real Greek
content the model should learn or emit?

This report documents the per-class **keep / remove** decision for the
25,600 added units of `C3_wave2_broad_glossapi_plus_hplt_50_50`, with
reasoning. The canonical machine-readable artifacts are
[`manifests/removal_list.jsonl`](manifests/removal_list.jsonl) (one row
per removed token) and
[`manifests/decision_summary.json`](manifests/decision_summary.json)
(class counts, per-cutoff impact, rule predicates) — these are git-
tracked and what `02_2_tokenizer_implementation` reads. The bulky
inverse `artifacts/keep_list.jsonl` (~25k rows) is gitignored and
regeneratable from the glossary.

## Headline

- **104 tokens removable** at the full 25,600 vocab (= 0.41 %).
- **39 tokens removable** inside the recommended 11,264 cutoff
  (= 0.35 %).
- The kept "noise" at 11,264 is the ~17 structurally unremovable
  byte-fallback / NFD / URL-encoded encoding-artifact tokens — see
  "Unremovable structural noise" below.

The cutoff decision and recommended pick (11,264) is **independent** of
this curation step: removal shrinks the live merge table; it does not
change which cutoff to pick.

## Per-class decisions

### REMOVE — class A: Latin-1-as-UTF-8 mojibake (whole `category = "mojibake"`)

**6 tokens at 25k**: `ÉÉ`, `Ø`, `ØØ`, `ÉÉÉÉ`, `ØØØØ`, `Ô`.

Greek UTF-8 bytes misinterpreted as a legacy single-byte codepage. Got
past the cleaner's `mojibake_badness_score ≤ 0.1` gate as a small
residual.

Why removable:
- pure BPE merge product (not byte-fallback base)
- linear merge chain; removing the chain takes the whole family
- no role in encoding completeness
- never appear in clean Greek text

### REMOVE — class B: mixed-script artifacts (whole `category = "mixed_script_token"`)

**77 tokens at 25k**, of which ~5 are Greek-Latin lookalike font-
substitution mojibake (`τo`, `Tο`, `Tα`, `Oι`, `Ωστόσο` — where the
leading `Ω` is **U+2126 OHM SIGN**, *not* U+03A9 GREEK CAPITAL LETTER
OMEGA; the real Greek word `Ωστόσο` is kept untouched) and ~72 are
punctuation+Greek BPE-boundary fragments (`.Ε`, `.χ`, `.Π`, `.Σ`,
`,τι`, `/και`, `-Α`, `-Κ`, `-Π`, …).

Why removable as a class:
- the font-substitution subset (the 5 lookalike-mojibake tokens) are
  pure noise — the same Greek-Latin lookalike pattern (Α↔A, Ο↔O, Τ↔T,
  Ω↔U+2126, …) has dozens of possible surface forms; keeping the 5 we
  happen to have sanctifies an arbitrary subset, and a single token for
  `Tο` lets the model emit Latin-T-where-Greek-Τ-should-be in generated
  text, which breaks every downstream Greek-aware system
- the punct+Greek subset (the 72 boundary-fragment tokens) recur in
  predictable surface forms but represent BPE merges across a
  whitespace/punct boundary that didn't get cleaned out. Per the
  agreed reasoning: "they are products of extraction/encoding so they
  are bound to be repeated, [but] they are too limited for the model
  to learn anything, they probably repeat in many other ways but they
  are too infrequent to be added in the tokenizer." Better for the
  model to handle the boundary via byte-fallback composition than to
  have an ad-hoc dedicated token for each `.<Greek capital>` form.

> **Risk note — whole-category removal of class B is the broadest call
> in this report.**  Two subsets sit inside the 77 tokens:
>
> - *True homoglyph mojibake* (`τo`, `Tο`, `Tα`, `Oι`, `Ωστόσο` with
>   U+2126) — char-mask bucket `no_lang` / `single:` Latin: Greek
>   letters substituted by Latin lookalikes (or a mathematical-OHM
>   substitute for Ω). These have **no** legitimate Greek reading and
>   are unambiguous removals.
> - *Punctuation+Greek BPE-boundary fragments* (`,τι`, `.λπ`, `/και`,
>   `-κα`, `/της`, `«Η`, …) — char-mask bucket `el_or_polyton` /
>   `el_plus_others`: a real Greek letter prefixed by a punctuation
>   character that did not get separated by the pre-tokenizer regex.
>   These *do* recur in real text (`,ότι` glued to a previous word,
>   `«Η` glued to an opening quote) and a model with byte-fallback
>   can compose them at input time, but the surface form itself is
>   *not* an extraction error in the same sense as the homoglyphs.
>
> The agreed call is to remove the whole `mixed_script_token`
> category on the grounds that the punct+Greek forms are too
> infrequent per surface form to deserve a vocab slot and byte-
> fallback composition is adequate. **If a future arm wants a
> narrower call, the split is implementable**: the
> `lang_bucket` field on each row of the classifier output
> (`02_1_4_cutoff_analysis/artifacts/classified_added_tokens.jsonl`)
> already separates `no_lang` homoglyphs from `el_*` punct+Greek
> rows. The current policy is "remove both subsets" — the narrower
> "remove homoglyphs only" rule would shrink class B from 77 to ~5
> and would be observed via the fertility / input-coverage
> regression test before being adopted.

### REMOVE — class C: PDF/PostScript glyph names (whole `category = "postscript_glyph"`)

**14 tokens at 25k**: `/Α`, `/η`, `/ν`, `/Υ`, `/Ε`, `/pi`, `/α`, `/Β`,
`/ή`, `/Γ`, `/Η`, `/Κ`, `/Σ`, `/Δ`.

Slash-prefixed glyph-name references that leak from PDF font tables
into extracted text. The glossary tags them verbatim as "PDF/
PostScript-style glyph artifact: leaked font/glyph-name reference
rather than normal running text." They are not legitimate corpus
content.

### REMOVE — class D: cleaner LINENEWLINE placeholders (whole `category = "code_identifier"`)

**2 tokens at 25k**: `LINENEWLINE`, `NEWLINENEWLINE`.

Literal placeholder strings the cleaner emits for newline-boundary
markup. They should have been replaced with actual `\n` / `\n\n`
before training but slipped through. Clear cleaner bug.

### REMOVE — class E: BPE pieces of LINENEWLINE (`category = "latin_acronym"` AND decoded ∈ {LIN, ENEW, LINENEW})

**3 tokens at 25k**: `LIN`, `ENEW`, `LINENEW`.

When `LINENEWLINE` was processed by BPE, intermediate merge steps
landed these fragments as their own tokens. Same family as class D;
remove together. Genuine acronyms like `EURO` and `MEG` stay.

### REMOVE — class F: cleaner extraction-tag fragments (`category = "latin_fragment"` AND decoded ∈ {-missing, -decoded})

**2 tokens at 25k**: `-missing`, `-decoded`.

Placeholder-style fragments from some HTML/XML-tag extraction step
that preserved tag-attribute names when the cleaner couldn't resolve
the content. Other `latin_fragment` entries (Greek-surname
transliteration tails like `opoulou`, `oulou`, `gean`) are kept —
they're real word-piece content.

### Removal-list total

| class | count at 25k | example | inside 11,264 cutoff |
|---|---:|---|---:|
| A. Latin-1 mojibake | 6 | `ÉÉ`, `Ø`, `ØØ` | 0 |
| B. mixed_script artifacts (whole) | 77 | `.Ε`, `,τι`, `τo`, `Tο`, `/και` | 31 |
| C. postscript_glyph (whole) | 14 | `/Α`, `/η`, `/pi` | 1 |
| D. code_identifier LINENEWLINE placeholders | 2 | `LINENEWLINE`, `NEWLINENEWLINE` | 2 |
| E. LINENEW BPE-fragments | 3 | `LIN`, `ENEW`, `LINENEW` | 3 |
| F. Cleaner extraction tags | 2 | `-missing`, `-decoded` | 2 |
| **total** | **104** | | **39** |

## KEEP — and why

| category | count at 25k | reason |
|---|---:|---|
| `punctuation_run` | 55 | real Greek typography — dot leaders, polytonic breathings, MD link-boundary patterns |
| `table_separator` | 21 | MD table syntax — recurs predictably in PDF-to-MD academic text |
| `math_symbol` | 14 | real math/scientific notation (∆, Ω, ∧) |
| `latin_fragment` (minus class F) | 8 | Greek-surname transliteration tails (opoulou, oulou, gean) — useful for Latin-transliterated Greek names |
| `escaped_character_run` | 9 | MD-escape sequences for underscores — real markup |
| `latin_word` | 6 | real Latin words in Greek academic citations (cropped, petition, Acta, covid) |
| `latin_acronym` (minus class E) | 2 | real acronyms (`EURO`, `MEG`) |
| `url_or_path` | 4 | URL-bearing patterns (`.gr`, `.europa`, `%CE`, `%CF`, `/el`) |
| `latin_abbreviation` | 2 | bibliographic abbreviations (`Surg`, `Eds`) |
| `dingbat_or_symbol` | 2 | edge-case PDF symbols (`□`, `►`) — kept as too aggressive to surgically prune |
| `whitespace_only` | 2 | real whitespace patterns |
| `unit_or_measure` | 1 | real measurement unit token |

## Unremovable structural noise (must stay regardless)

These I had previously flagged as "noise" but they're kept by
necessity, not by design choice:

| token class | count at 25k | reason it must stay |
|---|---:|---|
| empty byte-fallback strings (encoding_artifact) | ~10 | partial-UTF8 byte-fallback base tokens; ByteLevel needs them for encoding completeness |
| variant-byte `ε`, `ο`, `ε` (encoding_artifact) | ~3 | NFD-decomposed Greek encoding; removing risks breaking downstream merges over decomposed forms |
| `̀` combining acute (encoding_artifact) | 1 | required for NFD-decomposed Greek encoding |
| `·\n\n` (control_or_invisible) | 1 | real Greek middle-dot + paragraph-break pattern |

## Per-cutoff removable impact

| cutoff | removable | % | cumulative removal classes inside cutoff |
|---:|---:|---:|---|
| 1,024 | 1 | 0.10 % | early mixed_script |
| 4,096 | 12 | 0.29 % | + `-missing` (id 134,349) |
| 8,192 | 21 | 0.26 % | + LINENEWLINE family entering |
| 10,240 | 34 | 0.33 % | + `-decoded` (id 140,616) |
| **11,264** | **39** | **0.35 %** | **recommended cutoff** — 31 mixed_script + 1 postscript_glyph (`/η`) + 2 code_identifier + 3 LINENEW frags + 2 extraction tags + 0 mojibake (mojibake enters at 12k) |
| 12,288 | 44 | 0.36 % | + 2 mojibake (ÉÉ 143,146; Ø 143,233) |
| 15,360 | 57 | 0.37 % | + font-sub mojibake start (τo 144,813; Tο 145,441), more postscript_glyph |
| 20,480 | 83 | 0.41 % | + more in all classes |
| 25,600 | 104 | 0.41 % | all 6 mojibake + all 77 mixed_script + all 14 postscript_glyph + 2 LINENEWLINE + 3 LINENEW frags + 2 extraction tags |

The total removable share is ~0.35–0.41 % at every cutoff. The savings
if all 104 are removed at 25,600: 104 × 4096 dim × 2 (E + lm_head) × 2
bytes (BF16) = **~1.7 MB of parameters**.

For the recommended **11,264 cutoff**: 39 tokens removable, savings
**~640 KB**. Still tiny in parameter terms; the value of removal at
this cutoff is more about (a) not letting the model emit
extraction-fingerprint tokens in generated text and (b) sharpening the
embedding table for the kept Greek payload.

## Implementation handoff (to `02_2_tokenizer_implementation`)

The merge-rule extension implementation must honor the Apertus
front-end contract:

> preserve all old token ids; append only new ids

Naive removal of a token from `model.vocab` + `model.merges` would
renumber every id after it, breaking that contract. Two compatible
implementations exist:

**Option 1 — Mask at embedding-init time:**
- keep the 39 (or up to 104) removable tokens in the merge table
- in `03_apertus_extension_and_embedding_adaptation` initialize their
  embedding + lm_head rows to zero
- add them to a "force-low-priority" mask in the lm_head softmax
  during frozen-base warmup and early CPT so the model never starts
  emitting them
- reversible; same mechanism we use for unused-added tokens generally

**Option 2 — Build a "pruned" merged variant:**
- after the standard cutoff variant is built, drop the removable
  tokens from the merge table *and re-emit a renumbered tokenizer.json*
- definitively prevents the model from ever emitting the removed
  tokens, but breaks append-only-vs-Apertus (forks the id space) and
  needs more implementation surface (a different deployment pack)

> **Caveat — Option 2 is not "drop at the END of the merge order".**
> The removals are not all late-in-cutoff. The *first* removal
> (`mixed_script_artifact .Ε`) sits at id **131,423** — i.e. the very
> first added id past the Apertus base. Earlier framing in this doc
> suggested dropping "at the end"; that's wrong for the actual
> distribution. Many removals are deep inside the merge prefix of
> tokens we keep.
>
> Concretely: removed `/Ε` (id 145,992, postscript_glyph) has a kept
> dependent `/ΕΕ` (id 154,484, greek_acronym) — a manifest spot-check
> already turned this up. Naively deleting the row breaks the merge
> chain for the kept token. **A pruned variant therefore requires real
> merge-graph validation per removal**: for each removed token, walk
> the merges of every kept token at any later id and confirm none of
> them step through the removed merge pair. The minimum viable
> validator is a one-pass scan over `model.merges` after pruning that
> retokenizes every kept token's `decoded` string and asserts the
> result is a single id.
>
> Until that validator exists and runs green, **Option 1 (masking) is
> the default**, because masking is invariant under merge-graph
> dependencies — the removed tokens still exist in the merge table, so
> no kept token's merge chain is disturbed. Option 2 remains the
> right tool for the *generation* hazard (model emitting Latin-T-where-
> Greek-Τ, or model emitting `Ø` in Greek text), and the validator is
> the prerequisite to using it for that hazard.

For mixed_script (class B) and mojibake (class A) specifically,
Option 2 is more attractive because those classes carry generation-
failure risk. For PostScript glyph (class C), LINENEWLINE family
(D + E), and extraction tags (F), Option 1 is sufficient because the
risk is only embedding-table allocation, not a generation hazard.

The implementer in `02_2_tokenizer_implementation` decides per class,
**under the merge-graph validation constraint above**.

## Forward path — fix the cleaner

Every entry in the removal list points at a specific cleaner pattern
that should be addressed in the next training arm so these tokens
never become BPE candidates in the first place:

| removal class | cleaner-side fix |
|---|---|
| A. Latin-1 mojibake (`ÉÉ`, `Ø`) | tighten `mojibake_badness_score` cutoff further (e.g. ≤ 0.01) and/or apply the per-line-badness scorer's mojibake-specific detector from the `cleaner/per-line-badness-20260504` branch |
| B. mixed_script artifacts | font-substitution subset → same per-line-badness scorer detects Greek-Latin lookalikes; punctuation+Greek subset → make sure the tokenizer pre-process correctly separates sentence-final/initial punctuation from following Greek letters |
| C. PostScript glyph names | strip slash-prefixed glyph-name references in the PDF-extraction stage; flagged in `WAVE4_GLYPH_POSTSCRIPT_PLAN_AND_CHANGES_2026-04-29.md` |
| D + E. LINENEWLINE family | the cleaner is emitting these as literal placeholder strings; replace with actual newlines (`\n` / `\n\n`) before output |
| F. Extraction tags (`-missing`, `-decoded`) | trace the HTML/XML extraction path emitting these; drop or rename to a single sentinel the cleaner then strips |

The cleaner fix is the real right answer for the next tokenizer arm.
The merge-table removal step in this sub-subproject is only the
"what to do about C3 *as-is*" backstop.

## Reproduction

```bash
/home/foivos/.venvs/pq-probe/bin/python3 scripts/emit_removal_list.py
```

Outputs:
- `manifests/removal_list.jsonl` — **canonical, git-tracked.** One
  row per removed token (id, decoded, category, lang_bucket,
  removal_class, meaning_snippet).
- `manifests/decision_summary.json` — **canonical, git-tracked.**
  Counts per removal class, per-cutoff impact, and the rule
  predicates the script applied.
- `artifacts/keep_list.jsonl` — gitignored (regeneratable from the
  glossary in seconds). One row per kept added token.

Inputs:
- `~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/tokens_glossary.jsonl`
- `../02_1_4_cutoff_analysis/artifacts/classified_added_tokens.jsonl`
