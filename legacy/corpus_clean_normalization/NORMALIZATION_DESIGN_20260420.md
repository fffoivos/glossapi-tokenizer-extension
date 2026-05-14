# Normalization Design — 2026-04-20

**Status**: resolved design decisions from a design-alignment session on
2026-04-20. Supersedes the scope implied by the four-category wave10
pipeline. Prompts in `prompt_drafts/` implement this spec.

**Framing**: the goal is to avoid adding weird tokens to the tokenizer,
not to fix all text quality. The retrain of the fresh discovery tokenizer
is the real test — if the new vocabulary learns cleaner Greek morphemes
and fewer one-off structural tokens, the rules worked. No intermediate
`tokenizer_delta.json` metric is required as a promotion gate.

## Two operations + one page-level decision

1. **Normalize** — same-function variants → one canonical form.
2. **Strip** — chars/bigrams with no semantic purpose → removed.
3. **Page salvage** — if normalize + strip leaves too little, drop the
   page.

Within each: flat-pattern deterministic rules where possible; Gemini
review only where the function is genuinely ambiguous or a threshold
needs calibration.

## Normalize (deterministic)

| # | Match | Canonical target | Notes |
|---|---|---|---|
| 1 | `[ \t]{2,}` outside code/math | single space | guard fenced code blocks and `$$…$$` |
| 2 | Standalone separator lines: `[-_*=]{4,}` or em-dash / box-drawing runs on their own line with blank lines around | `---` | excludes MD table separator rows (those are handled separately by parser) |
| 3 | GFM table separator rows (parser-validated, header row above) | `---` / `:---` / `:---:` / `---:` per cell | `markdown-it-py` (Python) or `pulldown-cmark` with GFM tables (Rust) |
| 4 | `\.{4,}` | `.....` | already shipped; keep `.....` length for now (revisit only if visual weight becomes a concern) |
| 5 | `…{2,}` | `…` | Apertus has `…` as a single-token vocab hit |
| 6 | Enclosed / circled digits (`①–⑩`, `❶–❿`, `➊–➓`), mathematical alphanumeric symbols (U+1D7CE–U+1D7FF), **vulgar fractions (`½`, `¼`, `¾`, U+00BD–BE, U+2150–215F)** | ASCII digit or ASCII `a/b` for fractions | fractions fold (OCR residue + tokenizer bloat outweighs semantic weight) |
| 7 | Ligatures `ﬁ`, `ﬂ`, `ﬃ`, `ﬄ`, `ﬅ`, `ﬆ` | ASCII pairs | Apertus has no dedicated merges |
| 8 | Unicode whitespace variants U+2007 (figure), U+2009 (thin), U+202F (narrow NBSP) | regular space | imperceptible visually |

**Kept as-is (semantic)**:
- **Subscripts** `₀`–`₉` (U+2080–2089), `ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ` where present
- **Superscripts** `⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹` (U+2070, U+00B2, U+00B3, U+00B9, U+2074–2079)

## Strip (deterministic, evidence-driven)

| # | Target | Status |
|---|---|---|
| 9 | PUA (U+E000–F8FF, U+F0000–FFFFD, U+100000–10FFFD), U+FFFD replacement, C0/C1 controls, U+03A2, ZWSP family (U+200B/C/D/2060/FEFF), soft hyphen U+00AD | shipped (`is_unicode_noise_char`) |
| 10 | Glyph/font extractor tags (`GLYPH<c=…,font=…>`, `glyph&lt;…&gt;`, `MS-Bold-`, `FontName=`) | shipped (`GLYPH_FONT_TAG_REGEX` + `BAD_LINE_AC` Aho-Corasick) |
| 11 | Malformed HTML entities (`&gt`, `&lt`, `&amp` without trailing `;`) | **shipped** (commit 6aadfef — `normalize::normalize_malformed_entities`) |
| 12 | Non-Greek / non-Latin script residue: Math Alphanumeric Latin (fold to ASCII, incl. Letterlike Symbols holes), Math Alphanumeric Greek / Digamma (strip), Georgian / Arabic / Hebrew / Armenian (strip) | **shipped** — char-level coverage via extended `fold_codepoint` + extended `unusual` SCRIPT_SET. Implementation differs from the original plan (char-range bitmap instead of Aho-Corasick literal set) — bitmap is more efficient and sidesteps the multi-char context-dragging flagged in STAGE_1_2_REVIEW. |

**No contextual guards**. No script-majority gate, no doc-language gate,
no math-density gate. If the evidence inventory flags a bigram as
weird-token-producing, remove it. Over-cleaning a rare legitimate
foreign-language paragraph is acceptable cost; polluting the BPE vocab
is the worse cost.

## Maintenance / safety (no behavior change today)

| # | Change |
|---|---|
| 13 | Add U+1F00–1FFF polytonic Greek range to the `greek` SCRIPT_SETS entry explicitly. Currently passes through only because it's in neither `greek` nor `unusual`; a future edit could silently break it. |

## Page salvage (deterministic metric)

| # | Rule |
|---|---|
| 14 | After normalize + strip, if synthetic-page residual non-whitespace content < 30% of pre-cleanup, drop the page. Threshold tunable via Gemini audit. |

## Off-scope (resolved as out of scope for this normalization pass)

- **Line-break de-hyphenation** (`word-\nword` → `wordword`) — doesn't form
  weird tokens, just awkward splits.
- **Latin-lookalike Greek homoglyph folding** inside words — too complex;
  evidence doesn't show this as a dominant class.
- **Script-majority gates** for Latin Extended strip — over-engineering;
  strip by bigram inventory instead.
- **Math-density gates** for math-italic Latin — over-engineering; fold
  from inventory.
- **Doc-level language guardrail** — parked; revisit only if the matcher's
  per-source extension (from 2 → 17 sources) surfaces a genuine
  over-cleaning problem.

## Gemini tasks — narrower than the original four-category split

Three tasks. See `prompt_drafts/` for templates.

### Task A — separator-normalization review

Per-case, for candidates of #2 (standalone separator lines). Yes/no on
thematic-break role, interchangeability with `---`, semantics
preservation. See `prompt_drafts/01_separator_normalization_prompt.md`.

### Task B — MD table post-transform audit

Post-implementation audit of #3 output. Sample of 100–200 transformed
tables. Not a per-case review. See
`prompt_drafts/02_md_table_audit_prompt.md`.

### Task C — page-noise detection + threshold validation

Per-page review over pages flagged by the bigram-density metric. Yes/no
on `is_noisy_page`, disposition (`salvageable_clean` / `discard` /
`flag_for_ocr` / `keep_as_is`), dominant noise kinds (enum). We map
dominant kind → cleaning rule from a fixed catalog. See
`prompt_drafts/03_noise_page_prompt.md`.

### Task D (new, added in design session) — `//` + dashes mixed-token review

Evidence in `table_border_ascii_art` inventory contains mixed
slash+dash tokens: `//`, `://`, `-|`, `=/`, `-->`, `-|--------`, and
similar. Function is ambiguous: URL fragments (keep), HTML comment
residue (mostly already stripped by `strip_tags_custom`), TOC-adjacent
dividers, or separator-line residue.

Review design: sample 100 contexts, tag each as
`url_fragment` / `html_comment_residue` / `separator` / `toc_leader` /
`other`. We map the dominant class to a rule (keep / already-handled /
`---` / `.....`). Same firewall as Task C — model classifies, we
synthesize.

Prompt template: write as
`prompt_drafts/04_slash_dash_mixed_review_prompt.md` (not yet drafted).

### Gemini is NOT used for

- Per-case decisions on already-deterministic targets (whitespace
  collapse, ellipsis collapse, stylized-digit folding, ligature
  decomposition, Unicode whitespace folding, malformed-entity fallback,
  polytonic set update).
- Token-delta measurement (retokenize is the test).
- Rule-writing. Model classifies; we synthesize rules from classifications.

## Apertus alignment constraints (unchanged)

From CLAUDE.md empirical facts:

- `normalizer: null` — no Unicode normalization. Do not NFC/NFKC the
  corpus.
- Pre-tokenizer: GPT-2-style regex split → ByteLevel.
- Single-token vocab hits preserved: `…` U+2026, en-/em-dash, smart
  quotes, NBSP, bare α, monotonic ά, final sigma ς.
- Monotonic-NFC Greek is the assumed training distribution.

Every rule above preserves this. Nothing here changes the Unicode
normalization form of Greek text.

## Wave-2 audit findings (2026-04-27)

Empirical audit of the wave-2 F1 tokenizer vocab
(`tokenizer_analysis/inspection/F1_glossapi_only_50k_wave2_20260426/`)
surfaced specific cleaner gaps. Each finding below is tagged either
**[NORMALIZE]** (transform to a canonical form, content kept) or
**[MATCH+STRIP]** (detect a noise pattern and remove/drop, content
discarded). The two categories ride different code paths in the
cleaner; mixing them in one rule has caused order-of-operations bugs
(see the glyph discussion in I/M).

### 2026-04-28 steering update — independent F1/F2 vocab + corpus scan

The active wave-3 scope is narrowed by
`subprojects/01_0_cleaning_iteration_and_thresholds/WAVE3_CLEANER_PATCH_PLAN_2026-04-28.md`.
That pass compared F1/F2 vocab directly and scanned the full F1 train
split from scratch (310,019 docs / 60,825,820,152 chars).

The biggest corpus signals are run-like structural residue:

| pattern family | affected docs | wave-3 decision |
|---|---:|---|
| table separator fragments | 254,640 (82.14%) | implement |
| dot leader runs | 192,254 (62.01%) | implement |
| long dash runs | 159,256 (51.37%) | implement |
| escaped Markdown runs | 27,683 (8.93%) | implement |
| long equal / underscore / asterisk / hash / slash / pipe runs | 0.08% to 2.20% | implement |
| bare `GLYPH` / glyph-name residue | 735 to 1,566 docs | implement narrowly |
| soft hyphen / non-newline controls | 225 (0.07%) | verify fenced-code path |
| mojibake markers | 27,861 (8.99%) | defer |
| Cyrillic / homoglyph markers | 8,267 (2.67%) | defer |

Intentional HTML comment placeholders (`<!-- image -->`,
`<!-- text-missing -->`, `<!-- formula-not-decoded -->`) appear in
240,006 docs (77.42%) and are intentionally kept. They are not bad
tokens for this cleaner iteration.

Mojibake repair and Cyrillic / homoglyph folding are deferred to a
calibrated follow-up, not wave-3 implementation. Tracking issue:
<https://github.com/eellak/glossAPI/issues/99>.

---

# NORMALIZE

### A. Continuous-run length quantization → {1, 3, 5, 20}  **[NORMALIZE]**

Replace rule **#2** target. **All** continuous runs of any single
separator character snap **DOWN** to the largest length in
`{1, 3, 5, 20}` that is `≤ L`. The ladder is a floor, not a
nearest-neighbour rounding — a 12-char run becomes 5, not 20.

| run length L | output |
|---|---|
| 1 | 1 |
| 2 | 1 |
| 3 | 3 |
| 4 | 3 |
| 5 | 5 |
| 6 – 19 | 5 |
| ≥ 20 | 20 |

Concrete characters in the run-class (each treated independently —
runs only quantize within a single character class, not across them):

| Class | Characters |
|---|---|
| dash | `-`, `‐`, `‑`, `‒`, `–`, `—`, `―` |
| dot (see B) | `.`, `·`, `•`, `‧`, `⋅`, `⋯`, `…` |
| underscore | `_` |
| asterisk | `*` |
| equal | `=` |
| hash | `#` |
| tilde | `~` |
| slash | `/` |
| backslash | `\` |
| pipe | `\|`, `│`, `┃`, `║` |
| exclamation | `!` |
| percent | `%` |
| at | `@` |
| caret | `^` |
| accent / tonos | `΄`, `´`, `` ` ``, `~`, U+0301-style combining marks **only when standalone, not attached to a base letter** |

Rationale: bounds vocab diversity. Without quantization, BPE happily
mints distinct tokens for runs of length 6, 12, 24, 32, 48, 64, 128,
192, 256, 512 — all of which we observed in the wave-2 vocab. F1
empirical evidence (`tokenizer_analysis/inspection/F1_glossapi_only_50k_wave2_20260426/special_token_audit.md`):

- dash runs: `dash_run(len=8/16/24/32/40/48/56/64/128/192/256/512)` plus
  the ~350-char monster at id 23779
- underscore runs: lengths 1, 2, 4, 8, 16, 32, 33
- slash runs: 1, 2, 4, 8, 16
- asterisk runs: 1, 2, 3, 4, 8
- hash runs: 1, 2, 4, 8, 16
- equal runs: 1, 2, 4, 8, 16, 32

Each is BPE's doubling pattern (1 → 2 → 4 → 8 → 16 → 32) plus the odd
"one extra char left over" entries at 33 / 40 / 49.

**Worked examples (input → output)**, drawn from F1 vocab evidence
in `tokenizer_analysis/inspection/F1_glossapi_only_50k_wave2_20260426/`:

| input | observed in F1 vocab | output |
|---|---|---|
| `--` | dash_run len 2, e.g., id 1270 | `-` |
| `----` | id 1290 | `---` |
| `------` | id 2784 | `-----` |
| `--------` (8 dashes, doubling pattern) | dash_run len 8 | `-----` |
| `------------------------` (24 dashes) | dash_run len 24 | `--------------------` |
| `------------------------------------------------` (48) | dash_run len 48 | `--------------------` |
| `-` × ~350 (id 23779) | the worst case | `-` × 20 |
| `____________________________________` (32 underscores) | id 13368 | `--------------------` (here as underscores) i.e. `____________________` |
| `========================================` (40 equals) | equal_run len 32+ | `====================` |
| `////////////////` (16 slashes) | id 37492 | `-----` (slashes) i.e. `/////` |
| `****` (4 stars) | id 6776 | `***` |

#### A.1 Markdown-escape-then-run

Markdown source escapes (`\_`, `\*`, `\\`, `\#`, `\.`, `\-`, …)
appear in PDF-to-markdown output as 2-char "escaped char" units that
BPE then runs-and-doubles. Example F1 vocab:

- `\_\_` (id 2250, len=4 chars = 2 escaped `_`)
- `\_\_\_\_\_\_\_\_` (id 3044, 8 chars = 4 escaped `_`)
- `\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_` (id 16793, 40 chars = **20 escaped `_`**)

The cleaner must **un-escape parser-level escapes** (`\<char>` → `<char>`
where `<char>` is in the run-class set, AND we are not inside a code
fence) BEFORE the run-quantization pass. Then `\_\_` becomes `__`,
which collapses normally.

**Worked example (using F1 evidence — id 16793 = `\_` × 20 = 40 chars):**

Input markdown source (PDF-to-MD output of a fill-in-the-blank
form line):
```
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
```

Step 1 — un-escape `\_` → `_` outside code fences:
```
____________________
```
(20 underscores)

Step 2 — run-quantize, len 20 → 20 (boundary):
```
____________________
```

If the original had `\_` × 8 (= 16 chars, id 4645 in F1):
- Step 1: `________` (8 underscores)
- Step 2: floor 8 to 5 → `_____`

#### A.2 Markdown-table separator interaction

The `markdown_table_separator_like → |---|---|` rule (currently
unlanded) becomes a special case of A: inside each table-separator
cell, the `-` run quantizes via {1,3,5,20}; the `|` separators and
alignment colons are kept verbatim.

**Worked example:**

Input (verbose PDF-extracted MD table — F1 vocab has many `|---|` runs
of len 12+):
```
| Όνομα | Ηλικία | Πόλη |
| -------------------- | ------------ | ------------ |
| Μαρία | 23 | Αθήνα |
```

The header row + body keep verbatim. The separator row's three cells
contain `-` runs of length 20, 12, 12. Floor each independently —
20 stays 20, 12 → 5, 12 → 5:
```
| Όνομα | Ηλικία | Πόλη |
| -------------------- | ----- | ----- |
| Μαρία | 23 | Αθήνα |
```

If a cell has alignment colons:
```
| -----: | :----: | :--- |   (right, center, left aligned)
```
each cell's `-` run floors (5 stays 5, 4 → 3, 3 stays 3) and colons
stay:
```
| -----: | :---: | :--- |
```

#### A.3 NOT runs (excluded from this rule)

These look run-shaped but are structural / semantic and must NOT be
quantized:

- URL protocol fragments: `://`, `://`, `://www.` — keep verbatim.
- Intentional placeholder comments: `<!-- image -->`,
  `<!-- text-missing -->`, `<!-- formula-not-decoded -->` — keep
  verbatim. These are deliberate corpus markers, not bad-token residue.
  Ordinary HTML comments can still be handled by the HTML/tag cleanup
  path, but this run-quantization rule must not target comment
  delimiters directly.
- Code-fence markers: ``` ``` ``` — semantic.
- Math delimiters: `$$`, `$` — semantic.
- ATX headings `#{1,6}` — depth-bearing semantic, keep 1–6 verbatim;
  only 7+ runs quantize (treated as separator/noise).
- Bullet-list / blockquote leaders (`- `, `* `, `+ `, `> `) — keep.
- Ordered-list digit + dot (`1.`, `2.`) — keep (the `.` is part of a
  list marker, not a dot-run).

**Worked examples (NOT quantized):**

| input | reason kept | output |
|---|---|---|
| `https://www.example.com` | URL protocol | unchanged |
| `<!-- image -->` | intentional placeholder | unchanged |
| `## Section title` | ATX heading depth 2 | unchanged |
| `###### Deep section` | ATX heading depth 6 | unchanged |
| `########### crazy` (11 hashes) | not a depth, treat as noise | floor 11 → 5 → `##### crazy` |
| `1. First item` | ordered-list marker | unchanged (`.` is not in a dot-run) |
| `> blockquote` | blockquote leader | unchanged |
| `- item one` | unordered-list leader | unchanged (single `-` followed by space, not a run) |

### B. Combined `…` + `.` runs  **[NORMALIZE]**

Snap-DOWN ladder applies (same {1,3,5,20} floor as A).
`…` (U+2026) counts as 3 dots for run-length purposes. Any contiguous
combination of `.` / `·` / `•` / `‧` / `⋅` / `⋯` / `…` is treated as a
single dot-run with length =  Σ (chars, with `…` contributing 3 each).
The total is then quantized via the same {1, 3, 5, 20} ladder, output
as ASCII `.` repeated. Examples:

- `..` (len 2) → floor → `.` (1)
- `…` (1 char, 3 dots) → 3 → `...`
- `…..` (3+2 = 5) → 5 → `.....`
- `..…..` (2+3+2 = 7) → floor 7 → 5 → `.....`
- `…………` (4 × 3 = 12) → floor 12 → 5 → `.....`
- `…` × 7 (21 dots) → 20 → `....................`

This **supersedes** rules **#4** (`\.{4,}` → `.....`) and **#5**
(`…{2,}` → `…`). Both are subsumed by the unified rule.

### C. Valid Markdown parts also quantized  **[NORMALIZE]**

Apply the same length-quantization standard to every markdown
construct that has a length-runnable component:

- ATX heading prefix `#{1,6}` keeps its `#` count (semantic — depth).
  Anything past 6 quantizes via {1,3,5,20} → either 5 (deep heading
  retained as ATX-6 visually) or 20 (clearly noise, treat as separator
  line).
- Setext underline `={2,}` / `-{2,}` under a header line: quantize the
  rule line via {1,3,5,20}.
- Horizontal-rule lines composed of `*`, `-`, `_`, `=` already covered
  by rule A.
- GFM table separator cells (`---`, `:---`, `---:`, `:---:`): each
  cell's `-` run quantized via {1,3,5,20}; alignment colons kept; pipes
  kept.
- Blockquote prefix `> ` and unordered-list bullet `- `, `* `, `+ `
  unchanged (single chars, semantic).

The cleaner already parser-validates GFM tables (rule **#3**); the
new constraint is that the `-` runs inside each cell snap to the
ladder rather than being collapsed to fixed `---`.

### D. Mojibake-encoded Greek bigrams  **[NORMALIZE — DEFERRED]**

**Status (2026-04-28):** Deferred to a later wave. Not implemented in
the wave-2 cleaner.

**Why this is in the vocab today:** the wave-2 cleaner has no mojibake
detection. Rule #9 (`is_unicode_noise_char`) strips invisible chars
(PUA, U+FFFD, C0/C1, soft hyphen, ZWSP); rule #12 strips whole-script
blocks (Math Alphanumeric, Georgian/Arabic/Hebrew/Armenian). Neither
touches Latin Extended characters like `ï` `Î` `Ï` `â` — those are
valid Latin1 codepoints, indistinguishable per-char from a French
`naïve` or German `Mädchen`.

**Why deferred:** detection requires bigram-level pattern matching
(`ÎX` / `ÏX` where the second char is in the Latin1 range that
round-trips to a UTF-8 Greek codepoint). That's a different mechanism
from the existing per-char strip and carries an over-correction risk
on legitimate French/German/Latvian text. Need a wave-3 test plan
with multilingual false-positive samples before landing. Tracked in
<https://github.com/eellak/glossAPI/issues/99>.

Vocab contains tokens that are Latin-1 / Windows-1252 mis-decoding of
UTF-8 Greek byte pairs. Each Greek 2-byte sequence (e.g., `ου` =
0xCF 0x85 / 0xCF 0x85) survives the byte stream into Latin1
re-decoding as a 2-char Latin-Extended-A bigram (e.g., `ïõ`).

F1 vocab evidence (`non_greek_tokens.jsonl`):

| id | token | likely original Greek |
|---|---|---|
| 36423 | `ïõ` | `ου` (very common Greek bigram) |
| 39154 | `ïí` | `ων` |
| 6485, 24065 | `ï` | half of any 0xCF-prefixed Greek char |
| 36186 | `Î` | half of any 0xCE-prefixed Greek char |
| 36881, 47771 | `Ï` | half of 0xCF-prefixed |
| 15251, 46379 | `â` | usually 0xE2-prefixed Cyrillic mis-decode (cross-contamination) |

**Action when un-deferred** (FYI for future wave): detect Latin1-decoded
UTF-8 sequences (heuristic: bigram is two chars in U+00C0–U+00FF + the
second char is in U+00B0–U+00BF) and re-encode to UTF-8 bytes, then
UTF-8 decode to recover the Greek. This is a known mojibake pattern;
`ftfy` library implements the canonical version. Worth folding
`ftfy.fix_text` (or its bigram heuristic) into the cleaner before
per-char strip.

**Worked example:**

Input doc (PDF originally Greek-encoded as Windows-1252, mistakenly
re-decoded as Latin1 then UTF-8 wrapped):
```
ÎšÎ±Î¹ Î¬Î»Î»Î± Î­Ï‚Î¿Î´Î¬
```

Each 2-char Latin-Extended bigram corresponds to one UTF-8-encoded
Greek byte pair:
- `Îš` (0xCE 0x9A) → byte stream 0xCE 0x9A → UTF-8 decode → `Κ`
- `Î±` (0xCE 0xB1) → `α`
- ` ` → ` ` (space passes through)
- `Î¬` (0xCE 0xAC) → `ά`

Re-encoded:
```
Και άλλα έσοδα
```

Without this fix, the tokenizer learns 2-char bigram tokens like `Îš`,
`Î±`, `Îŧ` and misses the underlying Greek shape entirely. F1 vocab
contains `Î` and `Ï` as standalone single-char tokens (ids 36186,
36881, 47771) — that is exactly the broken-byte residue.

Already documented in `glossapi_v2_corpus_residual_noise_20260421`
memory as one of the "three uncounted noise types" (µ↔μ mojibake,
base64 PDF blobs, ASCII gibberish). This is the µ↔μ family.

### E. µ → μ (MICRO SIGN → Greek MU)  **[NORMALIZE]**

Token id 23708 contains a bare `µ` (U+00B5 MICRO SIGN). This is
visually identical to Greek `μ` (U+03BC) and almost always appears
where the author meant Greek mu (units, like `mg`, `µm`,
`µmol/L`). Fold U+00B5 → U+03BC unconditionally; the only legitimate
distinct use of MICRO SIGN is in physics typesetting where the
codepoint stability matters, which is irrelevant to a Greek LLM
training corpus. Already named in
`glossapi_v2_corpus_residual_noise_20260421`.

**Worked example:**

Input (medical text):
```
δοσολογία 25 µg/kg ανά 8 ώρες
```

The `µ` here is U+00B5 MICRO SIGN (often what a keyboard outputs).
Output:
```
δοσολογία 25 μg/kg ανά 8 ώρες
```

Now `μg` shares its `μ` with every other Greek `μ` in the corpus, so
the tokenizer treats both as the same starting char.

### F. Cyrillic / Latin homoglyph fold  **[NORMALIZE — DEFERRED]**

Status (2026-04-28): not a wave-3 implementation item.

The F1/F2 vocab and corpus scan do show Cyrillic / homoglyph-shaped
residue, but the risk profile is different from run quantization. A
per-char or unconditional fold could corrupt legitimate Cyrillic text
or mixed-script citations, while a majority-script word fold needs
calibrated examples before we can trust it.

Future investigation should sample contexts for the visible lookalikes
that appeared in vocab/corpus (`о`, `а`, `р`, `с`, `е`, `у`, `х`, etc.)
and decide whether the right action is:

- fold inside otherwise-Greek words;
- fold inside otherwise-Latin words;
- strip only clear OCR/font residue;
- or leave mixed-script text alone.

Tracked with mojibake repair in
<https://github.com/eellak/glossAPI/issues/99>.

### G. Repeated tonos / accent-only tokens  **[NORMALIZE]**

Token `΄΄` (U+0384 × 2) at id 23778 is a repeated-special-chars
residue. Combining-mark and accent classes are part of class A's run
table when they appear **standalone** (not attached to a base
letter); the run-quantization rule covers them. They must NOT be
stripped when attached to a base letter — that's a different concern
(NFC normalization, already off-scope per design).

**Worked examples:**

| input | meaning | output |
|---|---|---|
| `΄΄΄΄` (4 standalone tonos chars) | spam / OCR residue | `΄΄΄` (snap to 3) |
| `αλήθεια` (each char carries its own combining tonos in NFD form) | normal Greek word with accent | unchanged — accent is attached to a base letter |
| ` ΄ ` (a single tonos with spaces around it) | typesetting artifact | unchanged (single char run, len=1 stays len=1) |

---

# MATCH AND STRIP

**Scope: token-level only.** All match-and-strip rules below operate
at the character or whole-token level. Line-level stripping (drop a
whole line based on a noise-density threshold, footer-marker pattern,
or empty-block detection) is **deferred to a later wave** — out of
scope for this iteration.

### H. C0/C1 controls + soft hyphen  **[MATCH+STRIP]**

Already covered by rule **#9** (`is_unicode_noise_char`), but the
wave-2 vocab still contains 31 single-char tokens for `\x00`–`\x1f`,
`\x7f`, and `\xad` (soft hyphen). These were not stripped from the
wave-2 corpus.

Action: verify rule #9 is actually wired into the production cleaner
path; if it's wired but not firing, surface the input source so we
understand how C0/C1 chars are entering the cleaned output. If it's
NOT wired in production, wire it.

**Worked example:**

Input doc (PDF extractor leaks raw control bytes between paragraphs):
```
Πρώτη παράγραφος.\x00\x00\x00\x00\x00 Δεύτερη παράγραφος.
```

After strip (rule #9 firing):
```
Πρώτη παράγραφος. Δεύτερη παράγραφος.
```

Same for soft hyphen U+00AD (`\xad`), which is invisible in most
viewers but BPE tokenizes as its own char:
- `συν­εργάτης` (with U+00AD between `συν` and `εργάτης`) → `συνεργάτης`

### I. Bare `GLYPH` residue from upstream extractor / parser leakage  **[MATCH+STRIP]**

Wave-2 F1 vocab contains 10 slots that are pure repetitions of the
ASCII string `GLYPH` — `GLYPH`, `GLYPHGLYPH`, `GLYPHGLYPHGLYPH`,
`GLYPHGLYPHGLYPHGLYPH`, `GLYPHGLYPHGLYPHGLYPHGLYPH`, and (id 26413)
`GLYPHGLYPHGLYPHGLYPHGLYPHGLYPHGLYPHGLYPH` (×8). Source:
`tokenizer_analysis/inspection/F1_glossapi_only_50k_wave2_20260426/non_greek_tokens.jsonl`.

**Root cause is NOT order-of-operations** (initial hypothesis was
wrong; verified by reading `clean_text_with_stats` line 740 vs 791 —
Rule B's `PDF_GLYPH_NAME_REGEX` runs BEFORE `strip_tags_custom`, so a
`GLYPH<c=42,font=/Arial>` token would be caught by the regex on the
bracketed form). Rule B is structurally anchored on `GLYPH<...>`, so
the regex itself works as designed.

The actual cause is upstream: the brackets are already gone by the
time our cleaner sees the line. Three live candidates, not yet
empirically narrowed:

1. **PDF→MD extractor leaks bare `GLYPH`.** Some PDF extractors
   (Docling, Marker, pdfminer fallback) emit a literal `GLYPH`
   placeholder with no payload when they can't resolve a font glyph.
   Rule B requires the `<...>` payload; bare-word stems were
   explicitly removed in the 2026-04-25 cleanup per the "no-bare-words
   rule".
2. **Pilot B's parser-backed Phase A** (`format_surgical_verified` via
   comrak / pulldown-cmark) treats `<c=42,font=/Arial>` as malformed
   HTML and drops it before line-level Rule B sees the content.
3. **HTML-entity-strip / BAD_LINE_AC** drops the brackets in some
   pre-pass we haven't traced.

To pick the right cause, the next step is empirical: take a sample
doc from the corpus that produces `GLYPHGLYPH` tokens, run it through
the wave-2 pipeline with verbose logging, and observe where the
brackets disappear. Skipped for now; not a blocker for the fix.

**Fix (works regardless of which root cause is correct):**

Extend the existing glyph span stripping function,
`apply_glyph_span_strip_and_rule_b`, instead of adding a separate
glyph-cleaning subsystem. This keeps structured glyph tags, bare
`GLYPH` residue, and high-confidence PostScript/PDF glyph-name residue
under the same Rule A/B count + coverage threshold that can drop a line
when glyph noise dominates it.

For wave 3, the bounded bare-residue matcher should cover:

- repeated or standalone bare `GLYPH`;
- structured variants that are not currently covered, such as
  `glyph[...]` and `GLYPH(...)`;
- the high-confidence glyph-name family documented in M
  (`/hyphenminus`, `/ellipsis`/`/elipsis`, `/period`, `/comma`,
  `/space`, `/colon`) when the surrounding context is glyph-like.

Do not re-introduce generic single-letter PostScript glyph names,
`/pi`, broad `/[A-Z]{1,3}` font-subset names, URL-looking path
fragments, units, or slash acronyms in wave 3. Those need either
stronger context or a later calibrated sample. The "no-bare-words
rule" applied in 2026-04-25 was too strict for `GLYPH`, but that does
not mean all bare glyph-looking tokens are safe to strip.

Once the bare-word residue is gone, BPE has nothing to merge, so the
`GLYPH×2`/×3/×8 doubling chain dies with the source.

**Worked example (bare-word matcher fix):**

Input as it actually appears in the wave-2 corpus (the `<...>` payload
has already been dropped by some upstream stage):
```
Στο ΕΕΛ GLYPH GLYPH και πάλι
```

Today's cleaner: Rule B's `PDF_GLYPH_NAME_REGEX` looks for the
structured form `GLYPH<...>`, doesn't match bare `GLYPH`, returns the
line untouched:
```
Στο ΕΕΛ GLYPH GLYPH και πάλι   ← unchanged
```

After the proposed fix (re-introduce a bounded bare-word matcher
running after the upstream stage that drops the brackets):
```
Στο ΕΕΛ  και πάλι
```

(The double space is then collapsed by rule #1 → `Στο ΕΕΛ και πάλι`.)

When several such lines stack and BPE trains on the corpus, it merges
`GLYPH×2`, `GLYPH×3`, … up to the 8-repeat token (id 26413) we
observed in F1 vocab. Killing the bare residue at the source kills
the whole merge chain.

### J. Standalone math-fence residue  **[DEFERRED — line-level]**

Out of scope for this wave. Empty/orphaned `$$ $$` fence pairs are a
line-level / multi-line pattern; line-level stripping is deferred.
Original analysis preserved below for the future wave.

Vocab tokens: `$$` (id 2316), `$$\n\n` (3057, 12787), `$$\` (5324),
`$$(` (19594). These look like empty / orphaned math-block delimiters
left after the cleaner stripped the math content but kept the fence.
Action: when a `$$` opens and the next `$$` closes with empty / pure
whitespace between them, drop the whole pair instead of keeping the
fence. Match a few simple shapes — `$$\s*$$`, `$$\s*\\\s*$$` — rather
than building a generic markdown-math parser.

**Worked example:**

Input (PDF extractor produced an empty math block):
```
Παρακάτω η εξίσωση:

$$

$$

και η ερμηνεία της…
```

If we had stripped:
```
Παρακάτω η εξίσωση:

και η ερμηνεία της…
```

Same for the `$$\` (delimiter + lone backslash) and `$$(` (delimiter
+ unmatched paren) shapes — those are PDF extraction artifacts where
the equation body was lost but the fence char survived. Drop them
both.

### K. Standalone `Page` / `page` page-marker residue  **[DEFERRED — line-level]**

Out of scope for this wave. Page-marker drops require line-level
match (drop the whole line based on `^Page \d+(\s+of\s+\d+)?$`
pattern); line-level stripping is deferred. The general line-drop
mechanism (rule #14, page salvage by content threshold) is the right
place to land this when we get back to line-level work. Original
analysis preserved below for the future wave.

Tokens `page` (35775, 40927), `Page` (36043) — likely PDF-extractor
header/footer page-number markers (`Page 4 of 17` style) that keep
the bare word but lose the digits. Heuristic: `Page \d+(\s+of\s+\d+)?`
on its own line is a footer marker and should be dropped, both
in English and Greek (`Σελίδα \d+`). The existing #14 page-salvage
rule covers this category statistically; landing it would mop up the
residue.

**Worked examples (entire line dropped):**

| input line | drop? |
|---|---|
| `Page 4` | yes |
| `Page 4 of 17` | yes |
| `page 12` | yes |
| `Σελίδα 5 από 23` | yes |
| `– 12 –` (decorated page number) | yes |
| `On page 4 the author argues…` | NO — it's prose, not a footer. The line contains other content beyond the marker. |
| `Page` (alone, no digit) | yes — orphaned footer where digits were lost |

### L. Pictograph / dingbat tokens  **[KEEP]**

User decision (2026-04-28): not stripping. Pictographs / dingbats
stay in the corpus.

Vocab has standalone tokens for ❡ ❛ ✐ ♦ ♥ ❤ ❧ ✳ ❝ ✉ ☞ ☑ ✇ ❶ etc.
These are PDF-extracted dingbats / fancy-bullets / decorative
pictographs that don't carry Greek-language signal but aren't strictly
"noise" either (sometimes used as section markers in textbooks).

This is now a keep decision, not an open implementation branch. The
vocab cost is accepted to preserve possible section-marker /
decorative-bullet semantics.

**Rejected example (not wave-3 behavior):**

Input (textbook ToC line decorated with dingbat bullets):
```
✐ Κεφάλαιο 1: Εισαγωγή ❡
♦ Κεφάλαιο 2: Μέθοδος ♦
```

If stripped:
```
 Κεφάλαιο 1: Εισαγωγή
 Κεφάλαιο 2: Μέθοδος
```

Following whitespace-collapse (rule #1), the rejected output would
have been:
```
Κεφάλαιο 1: Εισαγωγή
Κεφάλαιο 2: Μέθοδος
```

### M. PostScript / PDF bare glyph-name residue  **[MATCH+STRIP]**

Same family as the **GLYPH** bug (I), but for *bare PostScript glyph
names*. The broad audit found many slash-prefixed Latin tokens, but
wave 3 should only strip the high-confidence subset:
`/hyphenminus`, `/ellipsis`/`/elipsis`, `/period`, `/comma`, `/space`,
and `/colon` in glyph-like contexts. F1 vocab evidence: 139 tokens
start with `/` + Latin (`non_greek_tokens.jsonl` filter
`^/[A-Za-z]`); many are glyph residue, but many others are legitimate
URL-path fragments, unit denominators, acronyms, or semantic-looking
names.

The cleaner's `PDF_GLYPH_NAME_REGEX` covers the structured forms
(`/uniXXXX`, `/gNNN`, `/gidNNN`, `/AAAAAA+FontName`) but the bare
glyph names were explicitly REMOVED in the 2026-04-25 cleanup per
the "no-bare-words" rule, then never re-introduced after the
`strip_tags_custom` ordering issue surfaced. So they leak.

**Worked example of the bug:**

Input from a PDF extractor (Adobe glyph names embedded in body text):
```
Η μέγιστη απόσταση ορίζεται ως /a /b /c /d /pi r/2, με /hyphenminus
/elipsis ως διαχωριστικό.
```

The structured `/uniXXXX` etc. would be stripped; the bare names
slip through:
```
Η μέγιστη απόσταση ορίζεται ως /a /b /c /d /pi r/2, με /hyphenminus
/elipsis ως διαχωριστικό.   ← unchanged, all bare names survive
```

After proposed wave-3 fix (strip only the high-confidence glyph-name
subset in the glyph Rule A/B pass, while excluding URL path words and
unit denominators):
```
Η μέγιστη απόσταση ορίζεται ως /a /b /c /d /pi r/2, με ως διαχωριστικό.
```

(The trailing whitespace then gets cleaned by rule #1.)

**Discriminator (bare PS-glyph vs URL path vs unit):**

| token | category | action |
|---|---|---|
| `/hyphenminus` | PS-glyph (full name in Adobe Glyph List) | strip |
| `/ellipsis`, `/elipsis` | PS-glyph (incl. common typo) | strip |
| `/period`, `/comma`, `/space`, `/colon` | PS-glyph | strip |
| `/pi`, `/alpha`, `/sigma`, `/Omega` | PS-glyph or semantic/math-looking token | DEFER |
| `/A`, `/a`, `/B`, `/b`, …, `/Z`, `/z` | PS-glyph or URL/acronym/path token | DEFER |
| `/GE`, `/GF`, `/CT`, `/BA`, `/DS`, …, `/G[A-Z]{1,3}`, `/C[A-Z]{1,2}`, `/B[A-Z]`, `/D[A-Z]`, `/F[A-Z]` | possible font-subset glyph index, but ambiguous with acronyms | DEFER unless surrounded by stronger extractor/glyph context |
| `/wiki`, `/article`, `/news`, `/blog`, `/uploads`, `/wp`, `/sites`, `/index`, `/eli`, `/content`, `/watch`, `/files`, `/handle`, `/document`, `/publication`, `/legal`, `/view`, `/the`, `/r`, `/o`, etc. | URL path word | KEEP |
| `/EL`, `/el`, `/gr`, `/en`, `/EG` | URL path: country/language code | KEEP |
| `/ml`, `/kg`, `/min`, `/cm`, `/dl`, `/sec`, `/mm`, `/mol`, `/in` | unit denominator after a number | KEEP |

**Practical implementation note:** the safest discriminator is
**context-based**, not regex-only:

- After a digit or unit-bearing word (`mg`, `mol`, `cm³`, `m²`):
  treat as unit denominator → keep.
- Inside a URL (preceded by a host or an `http(s)://` earlier in the
  same line, or matches `[a-z]+\.[a-z]{2,}` host pattern): URL path
  → keep.
- Otherwise, if the token matches the narrow high-confidence
  PostScript glyph-name subset and the surrounding context is glyph-like:
  strip.

A full Adobe Glyph List exact-match set may still be useful later, but
it is not a wave-3 requirement. The near-term patch should be smaller
and conservative because slash-prefixed corpus text contains many real
URLs, legal paths, units, and acronyms.

## Wave-3 implementation plan — where each rule lands in the code

All file paths are relative to `/home/foivos/glossAPI-development/`.
Status legend: **EXISTS** = code already there, just verify firing;
**EXTEND** = add to an existing function/list; **NEW** = write fresh
code.

---

### A. Continuous-run quantization → {1, 3, 5, 20}  **EXTEND**

**Existing infrastructure:**
- `rust/glossapi_rs_cleaner/src/normalize.rs:380` — `bucket_run_length(n: usize) -> usize` already implements the floor map (with a tiny snap-up at len=4: `4..=20 => 5`).
- `normalize.rs:391` — `normalize_char_runs_tiered(line: &str, target: char) -> Option<String>` already runs the bucket per arbitrary `target` char.

**Already wired callers:**
- `normalize.rs:443` — `normalize_whitespace_runs` (target = `' '`, tab handling).
- `normalize.rs:428` — `normalize_dot_runs` (target = `'.'`).
- `normalize.rs:514` — `normalize_escaped_underscore_runs` (handles `\_` pre-pass + run-quantize).
- Call-sites: `cleaning_module.rs:953/959/962`; `md_module.rs:939/942`.

**Edits needed:**
1. `normalize.rs` — add public functions, each one a thin wrapper calling `normalize_char_runs_tiered`:
   - `normalize_dash_runs(line)` — target `'-'`, plus dash-class chars `‐ ‑ ‒ – — ―` (extend the function or write a class-aware variant).
   - `normalize_underscore_runs(line)` — target `'_'`. (Distinct from `normalize_escaped_underscore_runs` which handles the `\_` pre-pass.)
   - `normalize_asterisk_runs(line)` — target `'*'`.
   - `normalize_equal_runs(line)` — target `'='`.
   - `normalize_hash_runs(line)` — target `'#'` (with ATX-heading guard, see C).
   - `normalize_tilde_runs(line)` — target `'~'`.
   - `normalize_slash_runs(line)` — target `'/'` (with URL guard, see A.3).
   - `normalize_backslash_runs(line)` — target `'\\'`.
   - `normalize_pipe_runs(line)` — target `'|'` (with MD-table guard, see A.2).
   - `normalize_exclamation_runs(line)` — target `'!'`.
   - `normalize_percent_runs(line)` — target `'%'`.
   - `normalize_at_runs(line)` — target `'@'`.
   - `normalize_caret_runs(line)` — target `'^'`.
   - `normalize_standalone_tonos_runs(line)` — targets `'΄'`, `'´'`, `` '`' `` when adjacent to whitespace / line boundary (NOT attached to a base letter — needs a position check).
2. `cleaning_module.rs:953-962` — chain the new normalize calls in the existing pattern (after `normalize_ellipsis_runs`, before the per-char filter).
3. `md_module.rs:939-942` — same chain in the markdown path.
4. `bucket_run_length` currently maps `4 => 5` (snap-up). Change this
   to `4 => 3`; the wave-3 ladder is a true floor.

### A.1 Markdown-escape unescape pre-pass  **EXTEND**

**Existing infrastructure:**
- `normalize.rs:514` — `normalize_escaped_underscore_runs` already handles `\_` un-escape + run-quantize in one function. Reads
  `\_\_\_\_…` → `____…` → bucket → output.

**Edits needed:**
- Extend the same approach to other escape-class chars: `\*`, `\-`, `\=`, `\#`, `\~`, `\.`.
- Cleanest: rename to `normalize_escaped_run_chars(line)` and accept a slice of `(esc_char, target)` pairs. Or add per-char functions `normalize_escaped_asterisk_runs`, etc., each delegating to the same generic helper.
- Wire in `cleaning_module.rs` adjacent to the existing call at line 962.
- **Code-fence guard** must be honoured (escape semantics differ inside code blocks). Existing impl already guards, so the new impls inherit the pattern.

### A.2 Markdown-table separator interaction  **NEW (markdown-aware)**

**Where it lands:**
- `rust/glossapi_rs_cleaner/src/md_format.rs` — Pilot B's parser-validated MD reformatter. The function that emits table separator rows is the right place. Today the formatter passes long `-` cell content through verbatim (per the `fx_hr_long_dash_run` test at `md_format.rs:395` — preserves preview).
- After `comrak` / `pulldown-cmark` identifies a separator row, walk each cell, find the longest `-` run, apply `bucket_run_length` to its length, emit `bucket_run_length(n)` dashes plus alignment colons (`:` left/right) verbatim.

**Edits:**
1. Find the GFM table separator emission code in `md_format.rs` (need a quick read; there's likely a `format_table_separator` or inline match arm).
2. Insert a per-cell pass that calls `bucket_run_length` from `normalize::bucket_run_length`.
3. Add a unit test alongside `fx_hr_long_dash_run`.

### A.3 NOT runs (excluded patterns)  **VERIFY**

**Already protected:**
- Code fences: `cleaning_module.rs:700-707` short-circuits when `in_code_fence == true`.
- ATX headings 1–6: depth detection in `md_module.rs` parser path.
- Intentional placeholder comments (`<!-- image -->`,
  `<!-- text-missing -->`, `<!-- formula-not-decoded -->`): keep.
  Ordinary HTML comments remain the responsibility of the existing
  HTML/tag cleanup path.
- URL fragments: detect via `://`-prefix or hostname pattern (no existing centralized helper — will need one).

**Edits:**
- Add a `is_inside_url(line, span_start, span_end) -> bool` helper in `normalize.rs` so the new run-normalize functions (slash, dot) can skip when inside a URL token.
- ATX heading 7+ explicit case in `md_module.rs` heading parser — when count > 6, treat as separator-line and quantize.

### B. Combined `…` + `.` runs  **EXTEND**

**Existing infrastructure:**
- `normalize.rs:356` — `normalize_ellipsis_runs(line)` (handles `…{2,}` → `…`, currently rule #5).
- `normalize.rs:428` — `normalize_dot_runs(line)` (handles `\.{2,}` via tiered bucket).

**Edits:**
- Replace both with a unified `normalize_dot_and_ellipsis_runs(line)`:
  1. Walk line. For every contiguous span of chars in `{ '.', '·', '•', '‧', '⋅', '⋯', '…' }`, count "logical dots" where `…` counts as 3.
  2. Apply `bucket_run_length(total_dots)`.
  3. Emit `bucket_run_length` ASCII dots.
- Update callers: `cleaning_module.rs:953` (currently calls `normalize_ellipsis_runs`) — switch to the unified function. Drop the separate `normalize_dot_runs` call.

### C. Valid Markdown parts also quantized  **EXTEND**

**Where it lands:**
- `md_format.rs` for table separators (already in A.2).
- `md_module.rs` heading parser for ATX 7+.
- `md_format.rs` setext underline emission (lines of `===` / `---` under a header).

**Edits:**
- Setext underlines: when emitting the underline, apply `bucket_run_length` to its length.
- ATX 7+: covered in A.3 above.

### D. Mojibake-encoded Greek bigrams  **DEFERRED**

No code changes this wave.

### E. µ → μ (MICRO SIGN → Greek MU)  **EXISTS — VERIFY**

**Already implemented:**
- `normalize.rs:107` — `'\u{00B5}' => return Some("\u{03BC}")` inside `fold_codepoint`.
- `fold_line` (line 328) calls fold_codepoint per char.
- Wired at `cleaning_module.rs:827`: `if let Some(replacement) = normalize::fold_codepoint(ch)`.

**Investigation needed:**
F1 vocab has bare `µ` at id 23708. Two likely causes:
1. **Code-fence pass-through** at `cleaning_module.rs:700-707` — the per-char fold doesn't run inside code fences. Verify by sampling a doc with `µ` and seeing if the bare char survives because of fenced context.
2. **fold_line not called in markdown formatter path** — `md_module.rs:936` does call `fold_line`, but maybe a different ingestion path bypasses it.

**Edits:** likely none if the current code is correct; if the code-fence pass-through is the cause, decision: should `µ` fold inside code fences? (Probably yes — code samples don't depend on the µ↔μ codepoint distinction in practice for Greek context.)

### F. Cyrillic / Latin homoglyph fold  **DEFERRED**

No wave-3 code changes. Keep this as a calibrated investigation in
issue #99. Future implementation, if any, must be based on sampled
contexts and false-positive checks rather than an unconditional fold.

### G. Repeated standalone tonos  **(covered by A — `normalize_standalone_tonos_runs`)**

No additional work beyond A.

---

### H. C0/C1 controls + soft hyphen  **EXISTS — VERIFY**

**Already implemented:**
- `cleaning_module.rs:379` — `is_unicode_noise_char(ch)` covers `\u{00AD}`, `\u{03A2}`, ZWSP family, `\u{FFFD}`, all `code < 0x20`, `0x7F`, `0x80..=0x9F`.
- Wired at `cleaning_module.rs:845`: `let should_remove_char = is_unicode_noise_char(ch) || …`.

**Investigation needed:**
F1 vocab has `\x00..\x1f`, `\x7f`, `\xad` as standalone tokens. Same hypothesis as E — likely the code-fence pass-through at `cleaning_module.rs:700-707` lets these chars through inside fenced blocks.

**Edits:**
- If verified: extend the code-fence handler to apply `is_unicode_noise_char` even inside fences (cheap to do, controls have no semantic value even in code).
- Or extend `is_unicode_noise_char` to also strip them at the line-level pre-fence pass.

### I. Bare `GLYPH` residue  **EXTEND EXISTING GLYPH RULE A/B**

**Existing infrastructure:**
- `cleaning_module.rs:166` — `PDF_GLYPH_NAME_REGEX` matches structured `GLYPH<...>` form.
- `cleaning_module.rs:346` — `apply_glyph_span_strip_and_rule_b(line)` runs the regex.

**Edits:**
1. Extend the regex/static matchers used by
   `apply_glyph_span_strip_and_rule_b`, not a new top-level cleanup
   pass. This keeps bare glyph residue in the existing Rule B count /
   coverage line-drop accounting.
2. Add a bounded matcher near `PDF_GLYPH_NAME_REGEX` for repeated bare
   `GLYPH` and a few structured variants that the current regex misses:

   ```rust
   pub static ref BARE_PDF_GLYPH_STEM_REGEX: Regex =
       Regex::new(r"(?i)\b(?:GLYPH)+\b").unwrap();
   ```

   Add structured forms such as `glyph[...]` / `GLYPH(...)` only if
   the samples show them in cleaned output.
3. Modify `apply_glyph_span_strip_and_rule_b` to apply the structured
   matcher first, then the bounded bare matcher, updating the same
   counters/coverage already used by Rule B.
4. Update the existing test that preserves bare `GLYPH`; wave 3
   intentionally makes `GLYPH` the narrow exception to the old
   no-bare-words rule.

### M. PostScript / PDF bare glyph-name residue  **EXTEND I NARROWLY**

**Where it lands:** same `apply_glyph_span_strip_and_rule_b` function as I.

**Edits:**
1. Add only the high-confidence glyph-name subset to the glyph Rule
   A/B function:

   ```rust
   const HIGH_CONFIDENCE_PS_GLYPH_NAMES: &[&str] = &[
       "hyphenminus", "ellipsis", "elipsis", "period", "comma", "space", "colon",
   ];
   ```

2. Strip `/name` only when the token is in that subset and the local
   context is glyph-like (standalone, repeated, adjacent to other glyph
   residue, or on a line already over the glyph Rule B threshold).
3. Explicitly keep URL paths and unit denominators.
4. Explicitly defer generic single-letter glyph names, `/pi` /
   `/alpha` style names, and broad `/[A-Z]{1,3}` font-subset guesses.
5. New unit tests for the discriminator:
   - `25 mg/kg` → unchanged.
   - `/pi r/2` → unchanged for wave 3.
   - `https://example.gov/eli/foo` → unchanged.
   - `/hyphenminus /elipsis` in a glyph-noise line → stripped.

---

## Implementation order (smallest-blast-radius first)

1. **A** + **A.1** + **B** in one PR — extend `bucket_run_length` callers, replace `normalize_dot_runs` + `normalize_ellipsis_runs` with the unified dot+ellipsis function. Pure normalize.rs work, easy to test in isolation.
2. **A.2** + **C** — markdown-format work in `md_format.rs` / `md_module.rs`. Riskier (parser changes), test with the existing `fx_hr_long_dash_run` test family.
3. **H, E** verification — minimal fenced-code cleanup for soft hyphen / non-newline controls and, if confirmed, MICRO SIGN folding.
4. **I** + **M** — narrow glyph residue extension inside `apply_glyph_span_strip_and_rule_b`, sharing existing Rule B counters and thresholds.
5. **D** + **F** — deferred; calibrate through issue #99 before any cleaner code.

## Next implementation moves

The old #14 page-drop queue is no longer the immediate next patch.
The tokenizer-guided wave-3 cleaner patch should land first:

1. Run quantization + escaped Markdown runs + unified dot/ellipsis.
2. Markdown table separator, setext underline, and ATX 7+ quantization.
3. Minimal fenced-code impossible-noise cleanup (`µ`, soft hyphen,
   non-newline controls) if sampling confirms the bypass.
4. Narrow glyph Rule A/B extension for bare `GLYPH` and
   high-confidence glyph names.
5. Re-clean a sample and re-run tokenizer analysis before revisiting
   deferred mojibake / Cyrillic-homoglyph work.
