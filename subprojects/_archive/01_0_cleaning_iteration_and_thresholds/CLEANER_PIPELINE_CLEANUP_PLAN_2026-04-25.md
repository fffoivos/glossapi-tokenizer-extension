> **Historical reference.** Pre-convergence cleaning-iteration work. The converged tokenizer arm is **C3** (see [../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md)). Kept for traceability; do not treat as live planning.

# Cleaner Pipeline Cleanup Plan (2026-04-25)

Discussion points from this session. Each entry below is something
we have discussed and agreed to consider; nothing is decided yet
and nothing is implemented. We will gather more points before
finalising and acting.

Cleaner SHA at writing: `2facec2` on
`codex/three-counter-pipeline-20260421`.

## Principles agreed across the cleanup

1. **Prefer R1∪R2-style line-level threshold rules over per-char
   strip.** When a noise pattern can be detected at line level
   (count, density, run-length thresholds) and the action is "drop
   the whole line on a hit," use that. Per-char strip is reserved
   for codepoints that are unambiguously noise even in isolation
   (extraction-failure markers, IPA, Coptic, etc.). This keeps
   foreign-language content intact at char level while still
   catching dense mojibake at line level.
2. **No char belongs to more than one operation.** Per Point 2:
   every codepoint lands in exactly one of: allowed (kept), strip
   (Group 1), or fold (Group 2). No accidental duplication.
3. **One engine, one definition.** A noise concept (e.g. "PostScript
   glyph residue") gets ONE rule with ONE regex/literal definition.
   That single definition serves both the cleaner (which acts on it)
   and the diagnostic counter (which counts it for sampling /
   review). No parallel definitions in separate code paths. See
   Point 7.
4. **Diagnostic counters are off by default in production runs.**
   They exist to support sample-cutting and review waves; they
   don't drive cleaning. Production cleaning runs should pay zero
   matcher overhead. Counters turn ON via flag for review runs.

---

## Point 1 — Deduplicate Rule B and strip_glyph_markers

**Status**: discussed and approved in principle; awaiting
finalisation.

**Problem**: `strip_glyph_markers` (text-wide pre-pass step 3)
removes `GLYPH<…>`, `/uniXXXX`, and `/gN` from the document. Rule B
(per-line, in `apply_glyph_span_strip_and_rule_b`) checks the same
patterns `/uniXXXX` and `/gN` for both span-stripping AND a
density-gated line-drop (count ≥ 10 AND coverage ≥ 9 % of non-WS
chars on the line → drop the line).

Because `strip_glyph_markers` runs first, Rule B never sees any
matches. Its span-strip is a no-op and its line-drop never fires.
Rule B's policy intent (drop lines that were *mostly* `/uni`/`/g`
markers, rather than leaving stripped husks) is silently disabled
in the current order.

**Pattern coverage**: `strip_glyph_markers` is a strict superset of
Rule B's regex — it adds `GLYPH<…>` on top.

**All three patterns are the same noise category**: PostScript
font-glyph residue from PDF extractions where the ToUnicode CMap
was missing or incomplete. Same origin, same fix policy.

**Direction agreed**: enhance Rule B to also match `GLYPH<…>`,
then delete `strip_glyph_markers`. Single owner of the PDF-glyph-
marker policy.

Concrete edits when this lands:
- Expand `PDF_GLYPH_NAME_REGEX` from
  ```
  /uni[0-9A-Fa-f]{4,6}|/g(?:id)?\d+
  ```
  to
  ```
  GLYPH<[^>]{1,200}>|/uni[0-9A-Fa-f]{4,6}|/g(?:id)?\d+
  ```
- Delete `strip_glyph_markers` and `GLYPH_MARKER_REGEX` from
  `normalize.rs`.
- Remove the call site at step 3 of the pre-pass in
  `core_clean_text_with_stats`.
- Adjust the tests in `normalize.rs::wave2_tests` (the
  strip_glyph_markers test cases relocate as Rule B regression
  tests in `cleaning_module.rs`).

Side effect: `BAD_LINE_AC` literals `GLYPH<` and `GLYPH&lt;` become
fully redundant (Rule B will strip every `GLYPH<…>` span before
BAD_LINE_AC sees the line, AND `decode_html_entities` already
turned `&lt;` into `<` upstream). They are safe to remove
alongside this change but are NOT yet part of the agreed scope —
flagging here so we can discuss whether to bundle.

---

## Point 2 — Consolidate per-char operations into 2 groups

**Status**: discussed and agreed in principle; awaiting finalisation.

**Goal**: every per-character cleaning operation belongs to exactly
one of two groups. No char is targeted by more than one operation.
This makes the policy auditable in one place and removes the
opportunity for duplicates to creep in.

### Group 1 — STRIPS (char → "")

A single set / function that strips a char from output. Replaces all
of:

- `strip_soft_hyphens` (U+00AD)
- `unusual_chars` (used inside the per-char filter):
  - Latin-1 Supplement (minus french_specific, spanish_specific,
    accented_greek, common_symbols, punctuation, latin1_legit_extras)
  - Latin Extended-A (minus french_specific, spanish_specific)
  - Latin Extended-B
  - IPA Extensions
  - Latin Extended Additional
  - Coptic (Greek block subset + dedicated block)
  - Cyrillic + Cyrillic Supplement
- `is_unicode_noise_char` (zero-width, format chars, BOM, U+FFFD, …)

After consolidation: ONE function (e.g. `should_strip_char(cp)`)
that returns true for every codepoint to strip. Called from a single
place inside the per-line loop. The pre-pass `strip_soft_hyphens`
goes away (its char becomes part of the strip set).

### Group 2 — FOLDS (char → another char / string)

A single map / function that maps a char to its replacement. Replaces
all of:

- `fold_codepoint` (per-line) — ligatures, Unicode whitespace
  variants → space, vulgar fractions, enclosed/parenthesised /
  with-period digits, Math Alphanumeric Greek.
- `decode_adobe_symbol_pua` (text-wide pre-pass) — Adobe Symbol PUA
  → real Greek / math chars.

After consolidation: ONE map / function. Called from a single place
inside the per-line loop. The text-wide pre-pass
`decode_adobe_symbol_pua` goes away (its mappings merge into the
unified fold map).

`decode_html_entities` is intentionally LEFT OUT of Group 2 — it
matches multi-char entity sequences (`&amp;` is 5 chars), not a
single codepoint. Different shape; stays as a text-wide pre-pass.

### Net effect

- Every codepoint is in EXACTLY ONE of: allowed (kept as-is), strip
  (removed), fold (replaced).
- Pre-pass count drops by 2 (`strip_soft_hyphens` and
  `decode_adobe_symbol_pua` both fold into per-line passes).
- Char accounting stays consistent —
  `chars_dropped_by_per_char_filter` still tracks Group-1 hits;
  `chars_dropped_by_normalization` still tracks Group-2 deltas
  where output is shorter.

## Point 3 — Refine Group 1 STRIP set: European vs non-European

**Status**: discussed; details to finalise.

**Goal**: split Group 1 (the strip set) along a "European-language
script vs noise/foreign-script" axis. KEEP chars used by European
languages we plausibly have in the corpus (legit content);
STRIP chars that are dominated by non-European scripts or
extraction-noise. Pair with Rule B + R1∪R2 line-drop so dense-
mojibake lines (where individual chars *would* be European-looking)
are caught at line granularity, not stripped one-by-one.

### Why this matters

Today, `unusual_chars` strips European-language characters that we
likely have in the corpus as legit content (citations, author
names, bilingual texts). Examples:
- Polish `ł, ą, ę, ć, ń, ś, ź, ż` (Latin-Ext-A) → currently STRIPPED.
- Czech `č, ě, ř, š, ť, ž, ů` → mostly STRIPPED.
- Hungarian `ő, ű` → STRIPPED.
- Turkish `ş, ğ, İ` → STRIPPED.
- Romanian `ș, ț` (Latin-Ext-B) → STRIPPED.
- Russian / Bulgarian Cyrillic → STRIPPED.

A Bulgarian-name citation in a Greek thesis (`Ljubomir Miletič`)
gets mangled to `Ljubomir Mileti` in output today.

### Direction agreed

Per-char filter narrows to **truly non-European or extraction-noise
blocks**. Dense-mojibake lines (where these European-allowed chars
appear in clustered runs) get handled by Rule B (after Point 1
expansion) + R1∪R2 line-drop.

### Per-block proposal (for finalisation)

| block (codepoints) | content | proposal | reason |
|---|---|---|---|
| Latin-1 Supplement (U+00A1..U+00FF) | French/Spanish/German/Italian/Nordic accented letters | KEEP entirely | all European |
| Latin Extended-A (U+0100..U+017F) | Polish/Czech/Hungarian/Romanian/Turkish/Maltese/Welsh/Esperanto/etc. | KEEP entirely | all European |
| Latin Extended-B (U+0180..U+024F) | Vietnamese / African / IPA-like / Greek-CID-mojibake / Romanian comma-below | BORDERLINE — see below | mostly noise, but Romanian ș (U+0219), ț (U+021B) live here |
| IPA Extensions (U+0250..U+02AF) | phonetic notation | STRIP | not prose; rare in our corpus |
| Latin Extended Additional (U+1E00..U+1EFF) | Vietnamese | STRIP | not European in scope |
| Coptic (U+03E2..U+03EF + U+2C80..U+2CFF) | ancient liturgical | STRIP | not in modern Greek corpus content |
| Cyrillic + Supp (U+0400..U+052F) | Russian / Bulgarian / Serbian / Ukrainian / Macedonian / etc. | KEEP entirely | all European |

### Latin-Ext-B options (the hard one)

- ~~**B-i**: STRIP all of U+0180..U+024F.~~
- ~~**B-ii**: KEEP all of U+0180..U+024F.~~
- ✅ **B-iii (CHOSEN)**: STRIP U+0180..U+024F EXCEPT a curated
  allowlist of Romanian comma-below variants — `Ș` (U+0218),
  `ș` (U+0219), `Ț` (U+021A), `ț` (U+021B). If other European-
  language uses surface during review (e.g., Slovak digraph forms,
  Sami chars), they get added to the allowlist case-by-case.

### Pairing with line-level rules

After this Point 3 lands, the cleaning policy is:

1. **Per-char filter** (Group 1 STRIP) only catches obviously
   non-European / extraction-noise codepoints. Foreign-language
   words in citations survive intact.
2. **Rule B** (Point 1 + Point 4, the unified glyph/font detector)
   drops lines dense in `GLYPH<…>`, `/uniXXXX`, `/gN`, font-subset
   markers — extraction-noise.
3. **R1 ∪ R2 line-drop** drops lines dense in residue chars (the
   Greek-CID-mojibake signature). Foreign-name lines stay below
   threshold (sparse residue) and survive.

Net effect: legit foreign content (Polish names, Bulgarian
names, German citations) stays as-is; mojibake-style noise gets
line-dropped; truly non-European/noise codepoints are still
stripped at char granularity.

### R1 ∪ R2 residue range — aligned with Group 1

`is_residue_mojibake_line` (R1 ∪ R2) currently treats the full
range `U+0100..U+024F` as "residue". After Point 3 lands, this
range MUST narrow to match Group 1's strip set so the
two layers do not contradict each other (otherwise R2 could
line-drop a doc of dense Polish or Czech text — chars Point 3
explicitly preserves).

Concrete change to the residue predicate inside
`is_residue_mojibake_line`:

```
old: 0x0100 <= cp <= 0x024F
new: 0x0180 <= cp <= 0x024F
     AND cp NOT in {U+0218, U+0219, U+021A, U+021B}   # Romanian allowlist
```

Result:
- Latin Extended-A (U+0100..U+017F) — no longer counted as residue.
  Polish/Czech/Slovak/Hungarian/Romanian-most/Turkish/etc. dense
  text doesn't trigger R1 or R2.
- Latin Extended-B (U+0180..U+024F) minus Romanian comma-below
  — counted as residue. Greek-CID-mojibake clusters in this range
  trigger R2 reliably.
- Romanian `Ș, ș, Ț, ț` — neither stripped per-char (Group 1
  allowlist) nor counted as residue (R1∪R2). A doc of dense
  Romanian survives untouched.

### Open subquestions

- Which Latin-Ext-B option (B-i / B-ii / B-iii)?
- Within Latin-1 Supplement, are there any codepoints we want
  to STRIP rather than KEEP? (Currently the user's `latin1_legit_extras`
  flag lists `«» · § ° ® © ¢ £ ¥` as legit; the rest of the block
  is European-language content. Worth confirming nothing slips.)
- Any other blocks beyond the table above we should be considering
  (Greek Extended, Math Alphanumeric, etc. — these are folded
  not stripped, so out-of-scope here)?

## Point 4 — Unify all glyph/font residue detection under Rule B

**Status**: discussed and agreed in principle; awaiting finalisation.

**Goal**: Rule B is the single owner of all glyph/font residue
detection — clean (span-strip) every match plus density-gated
line-drop. No bare-word matchers remain. Reduces four engines
(`strip_glyph_markers`, `BAD_LINE_AC`, `has_decoded_glyph_font_artefact`,
Rule B) to one (Rule B).

This subsumes Point 1's narrower scope (Rule B only absorbed
`GLYPH<…>` from `strip_glyph_markers`). Point 1 stays as the first
half; Point 4 completes the consolidation.

### Unified Rule B regex (case-insensitive)

```
(?i)
  GLYPH<[^>]{1,200}>                  # GLYPH<…> tag (uppercase or lowercase)
| <c=\d+,font=/[^>]+>glyph             # reversed-order form
| /[A-Z]{6}\+[A-Z][A-Za-z0-9-]+        # PDF font subset (e.g. /ABCDEF+TimesNewRoman)
| /uni[0-9A-Fa-f]{4,6}                 # Unicode codepoint reference
| /g(?:id)?\d+                         # generic glyph index
```

Every alternative is anchored on structural punctuation (`<`, `/`,
`+`, `=`, digits). **No alternative matches a bare prose word** —
"glyph", "font", "hyphenminus", "GLYPH" as plain words pass through
untouched, eliminating the false-positive risk of the old
bare-word literals.

### Behaviour (same shape as today's Rule B)

- **Span-strip**: every regex match in the line is replaced with `""`.
  Lines with a few stray markers stay intact (markers gone).
- **Line-drop gate**: if `count ≥ 10 AND match-coverage ≥ 9 %` of
  non-WS chars on the line, the line goes to `<!-- line-removed -->`.

The thresholds (10, 0.09) carry over from today's Rule B (Gemini-
calibrated wave on 1000 cases, P=96.3%, R=60.4%). They can be
re-calibrated post-cleanup if the broader pattern set changes the
tradeoff.

### Code deltas

Delete (already covered by Point 1, restated for completeness):
- `strip_glyph_markers` and `GLYPH_MARKER_REGEX` in `normalize.rs`.
- The pre-pass call to `strip_glyph_markers` in
  `core_clean_text_with_stats`.

Delete (new in Point 4):
- `has_decoded_glyph_font_artefact` function.
- `GLYPH_FONT_TAG_REGEX`, `FONT_GLYPH_TAG_REGEX`,
  `PDF_FONT_SUBSET_REGEX` constants.
- The call to `has_decoded_glyph_font_artefact` in the per-line
  line-drop branch of `core_clean_text_with_stats`.
- `BAD_LINE_AC` constant and AC engine entirely (every remaining
  literal is either subsumed by the unified Rule B or is a bare
  word we're explicitly dropping per the no-bare-words rule).
- The call to `BAD_LINE_AC.is_match(...)` in the per-line line-drop
  branch.

Modify:
- `PDF_GLYPH_NAME_REGEX` (the regex Rule B uses) → become the
  unified case-insensitive union shown above.
- `apply_glyph_span_strip_and_rule_b` → no behaviour change in shape;
  the larger regex is plugged in but the count/coverage gate logic
  stays the same.

### Tests

The Rule B + glyph-strip tests already exist in `cleaning_module.rs`
and `normalize.rs`. New tests to add:
- `glyph<c=…,font=/…>` (lowercase) → caught by Rule B.
- `<c=…,font=/…>glyph` (reversed order) → caught.
- `/ABCDEF+TimesNewRoman` (PDF font subset) → caught.
- bare word `glyph` (no `<`) → NOT caught. Surrounding text preserved.
- bare word `font` → NOT caught.
- bare word `hyphenminus` → NOT caught (Rule A still strips
  `/hyphenminus` though, distinct).
- A line with 11 `/uniXXXX` and one `glyph<c=…>` → density gate
  fires, line-dropped.
- A line with one `glyph<c=…>` and otherwise clean Greek → matches
  span-stripped, line preserved (count=1 below threshold).

### Net per-line line-drop engines after Point 4

```
B1. Rule B (unified glyph/font residue: span-strip + density gate)
B2. is_residue_mojibake_line (R1 ∪ R2)
```

Two engines instead of four. Both are line-level threshold rules
per the cross-cutting principle.

### Open subquestions

- Are the existing thresholds (10, 9%) appropriate for the
  expanded pattern set, or do they need re-calibration?
- Is `/[A-Z]{6}\+[A-Z][A-Za-z0-9-]+` precise enough? The Adobe font
  subset convention is exactly 6 caps + `+` + name, but some
  tools emit 7-cap subsets. Worth verifying on a sample.

---

## Point 5 — Fold Rule A into Rule B (single PostScript-glyph detector)

**Status**: discussed and agreed in principle; awaiting finalisation.

**Problem**: After Point 4, two engines still detect PostScript glyph
residue by different mechanisms:

- **Rule A** (`RULE_A_LITERALS_AC`, span-strip only): a 50-literal
  Aho-Corasick set covering bare PostScript glyph names —
  `/space`, `/period`, `/comma`, `/parenleft`, `/hyphenminus`,
  `/endash`, `/copyright`, `/dagger`, `CID+`, etc.
- **Rule B** (regex, span-strip + density-gated line-drop): the
  unified regex from Point 4 covering `GLYPH<…>`, `<c=…,font=/…>glyph`,
  `/[A-Z]{6}+Name`, `/uniXXXX`, `/gN`.

**Same noise source** — both fire on PostScript CMap dump residue from
PDF extraction failures. The matcher's `three_counter_spec_20260421.json`
already groups them as ONE category `glyph_font_like` (Rule A's
literals appear as `glyph_marker_extended_literals`, Rule B's patterns
as `glyph_marker_extended_regex`, both under the `glyph_font_like`
category). So upstream-spec the unification already exists; the
cleaner just has them split into two engines for legacy reasons.

**Direction agreed**: fold Rule A's 50 literals into Rule B. Result —
Rule B is THE single PostScript-glyph residue detector with the
unified policy (span-strip + density-gated line-drop).

### Concrete edits

- Extend Rule B's regex with a literal alternation arm (or keep an
  `RULE_B_LITERALS_AC` engine that runs alongside Rule B's regex but
  participates in the SAME count/coverage gate). The latter is
  faster than back-translating 50 literals into a regex
  alternation. Either way, the SAME line-drop policy covers both.
- Delete the standalone `apply_glyph_span_strip_and_rule_b`
  asymmetry where Rule A literals never contribute to the gate;
  unify so a line of dense `/space /period /comma /parenleft …`
  gets the same line-drop treatment as a line of dense
  `/uniXXXX /uniXXXX …`.
- Keep `LeftmostLongest` semantics (so `/hyphen` doesn't eat the
  prefix of `/hyphenminus`).

### Behaviour change

Today: a line of 20× `/space` markers gets stripped clean and kept
(empty husk).
After Point 5: same line — 20 matches, ≥9% coverage of non-WS chars
— hits the density gate and drops as `<!-- line-removed -->`.
This is correct: such a line is a CMap dump, not prose.

A line with 1–2 stray `/space` markers in otherwise-valid prose
still survives (gate doesn't fire), markers stripped — same as today.

### Net per-line line-drop engines after Points 4 + 5

```
B1. Rule B (unified PostScript-glyph residue:
            literal+regex span-strip + density gate)
B2. is_residue_mojibake_line (R1 ∪ R2)
```

Same engine count as Point 4 alone, but Rule B now fully owns the
PostScript-glyph concept — no more split between literals and regex.

### Open subquestions

- Implementation choice: extend Rule B's regex (one engine, slower
  literal alternation) vs. keep two engines that share the gate
  (faster, but two passes over the line). Bench before deciding.
- Do any of the 50 Rule A literals need EXEMPTION from the
  density gate (i.e., always strip, never line-drop)? Probably not —
  `/space`, `/period`, etc., have zero legitimate prose use.

---

## Point 6 — Drop matcher's `script_residue_restricted` category entirely

**Status**: discussed and agreed; awaiting finalisation.

**Problem**: The noise-matcher's `script_residue_restricted` category
(in `corpus_clean_normalization/specs/three_counter_spec_20260421.json`)
defines two patterns:

```
[Ā-ɏ]                                              # per-char
[Ā-ſ][ƀ-ɏ]|[ƀ-ɏ][Ā-ſ]  # bigram script salad
```

These produce `counter_script_residue` in the parquet output. After
the page-level rule was retired (replaced by line-level R1∪R2 in the
cleaner), **no cleaner decision reads this counter**. All current
consumers are diagnostic / sampling:

- `pull_top_three_counters_pdf.py`, `sample_char_strip_bands.py` —
  cut samples sorted by `counter_script_residue`.
- `calibrate_counter_thresholds.py` — threshold calibration analysis.
- `analyze_cleaning_concentration.py`, `analyze_script_residue_lines.py`,
  `eval_r1_r2_coverage.py` — analysis (the latter two compute their
  own residue logic, ignore the parquet counter).
- `gemini_three_counter_reviewer.py` — review.

**Direction agreed**: drop `script_residue_restricted` from the
matcher entirely. After Points 1–5 land, the cleaner emits its own
diagnostic-quality residue accounting (`chars_dropped_by_line_drop`,
`lines_dropped_by_cleaner`) that reflects what the cleaner ACTUALLY
DID, not what a separate diagnostic regex would have caught. That
is strictly more informative for sample-cutting and threshold work
going forward.

This avoids the "matcher's residue ≠ cleaner's residue" trap that
Point 3 introduces (cleaner narrows residue to U+0180..U+024F minus
Romanian; matcher would still match all of U+0100..U+024F and
mislead any sampling pool).

### Concrete edits

- `corpus_clean_normalization/specs/three_counter_spec_20260421.json`
  — delete the two `script_residue_restricted` entries. Spec
  becomes a `two_counter_spec` (font_name_literal + glyph_font_like).
  Rename file at the same time
  (`two_counter_spec_<date>.json`) for clarity.
- `cleaning_scripts/clean_and_stats_rowsharded.py` — remove
  `script_residue_restricted` from the `suggested` mapping at line 51,
  remove `counter_script_residue` from parquet emit
  (lines ~177, ~212, ~246, ~304, ~351), drop
  `_page_script_residue_count` (line 78), drop
  `pages_dropped_script_residue` /
  `chars_dropped_script_residue_pages` plumbing (already dead,
  page-level rule was removed in commit 2facec2).
- `cleaning_scripts/calibrate_counter_thresholds.py` — drop
  `"script_residue"` from `COUNTERS` tuple.
- `cleaning_scripts/pull_top_three_counters_pdf.py` — script becomes
  `pull_top_two_counters_pdf.py`; the
  `top500_by_counter_script_residue` cut target is retired.
- `cleaning_scripts/sample_char_strip_bands.py`,
  `pull_deletion_band_samples.py`,
  `analyze_cleaning_concentration.py`,
  `gemini_three_counter_reviewer.py` — drop
  `counter_script_residue` references; downstream sampling uses
  `chars_dropped_by_line_drop` / `lines_dropped_by_cleaner` instead.
- Analysis scripts that compute their OWN residue logic
  (`analyze_script_residue_lines.py`, `eval_r1_r2_coverage.py`,
  `sample_residue_rules_R1_R2_R6.py`) keep working — they don't
  depend on the matcher counter.

### Things to NOT delete

- The `font_name_literal` and `glyph_font_like` matcher categories
  stay. Both are still meaningful diagnostics that align with the
  cleaner's Rule B (after Point 5) and produce useful sample-cut
  pools.
- The matcher framework itself stays; we're shrinking from three
  counters to two.

### Sample-cutting after this point

Future review-wave samples sort by:
- `counter_glyph_marker` (still emitted by matcher, aligns with Rule B)
- `counter_font_marker` (still emitted by matcher, aligns with Rule B)
- `chars_dropped_by_line_drop` (cleaner's own line-drop accounting)
- `lines_dropped_by_cleaner`
- `pct_chars_removed_non_empty`

Any "find me docs with most residue chars" sampling switches from
`counter_script_residue` (matcher diagnostic) to
`chars_dropped_by_line_drop` (cleaner reality, includes R1∪R2 hits).

---

## Point 7 — Unify matcher into cleaner: one engine, off-by-default counters

**Status**: discussed and agreed in principle; awaiting finalisation.

### The architectural problem

Today there are TWO Rust crates with PARALLEL regex definitions for
the same noise concepts:

- `glossapi_rs_cleaner` — defines `PDF_GLYPH_NAME_REGEX`,
  `PDF_FONT_SUBSET_REGEX`, `GLYPH_FONT_TAG_REGEX`,
  `FONT_GLYPH_TAG_REGEX` inline. Uses them to STRIP and DROP lines
  (Rule B and friends).
- `glossapi_rs_noise` — reads regexes from JSON spec
  (`three_counter_spec_20260421.json`). Uses them to COUNT matches
  per page, emits `per_category_match_count` → `counter_font_marker`,
  `counter_glyph_marker` in parquet.

Two passes over every doc. Two regex sources that can drift (and
do — matcher includes bare-word `GLYPH` / `hyphenminus` that
Rule B excludes; matcher misses the `<c=…,font=/…>glyph` form
Rule B catches). Counter-vs-cleaner divergence makes
`top500_by_counter_glyph_marker` samples not-quite-faithful to
"top docs by Rule B activity."

### The matcher counters' actual role

The matcher counters are **diagnostic helpers**, not part of
cleaning. They power:
- Sample-cutting (`pull_top_three_counters_pdf.py` etc.)
- Review-wave selection
- Threshold calibration
- Concentration analysis (`analyze_cleaning_concentration.py`)

NO cleaner decision reads them. The cleaner has its own regex
and acts directly.

### Direction agreed

Merge the matcher into the cleaner as an OPTIONAL output mode.
One engine, one regex definition per rule, parameter-controlled
behaviour:

- **Production mode (default)**: cleaner cleans. No counters.
  Zero matcher overhead.
- **Review mode (flag enabled)**: cleaner cleans AND emits
  per-rule match counts as part of its stats output. Counts
  reflect EXACTLY what each rule matched, by construction.

This gives single source of truth (Principle 3) and
off-by-default helper cost (Principle 4) in one move.

### Concrete shape

In `glossapi_rs_cleaner`:

- Each rule (Rule B, R1∪R2, Group 1 STRIP, Group 2 FOLD) carries
  its OWN regex/predicate as the canonical definition.
- `CleanStats` grows an optional field, e.g.
  `per_rule_match_counts: Option<HashMap<String, u64>>`.
- A new param on `core_clean_text_with_stats` (e.g.
  `emit_rule_counters: bool`) toggles whether the per-rule
  counters get accumulated.
- Rule application sites (Rule B's `find_iter`, R1∪R2's per-line
  call, etc.) increment the relevant counter when the flag is on.
  When off, the counter struct stays `None` and there's zero
  bookkeeping cost.

### The matcher crate's fate

After Point 7, `glossapi_rs_noise` (and its
`three_counter_spec_*.json`) are no longer needed for the
glyph/font pattern counters — those move into the cleaner.

What the matcher crate WAS uniquely useful for:
- Discovering NEW patterns the cleaner doesn't yet handle
  (run an experimental regex over the corpus, count hits, sample
  high-hit docs, decide whether to promote the regex into a
  cleaner rule).

That capability stays valuable. Cleanest answer: rename the
matcher to something like `glossapi_rs_pattern_probe`, scope it
to "experimental pattern counting for rule discovery," and keep
it pluggable via JSON spec for cheap iteration. Production runs
never invoke it; only review-driven rule-discovery work does.

If we don't need experimental probing right now, we can simply
DELETE the noise crate when Point 7 lands and resurrect it later
as a pattern-probe tool when the next discovery wave wants it.

### Ergonomic generalization (folded in)

The `+TimesNewRoman` / `+Palatino` / `+Helvetica` / etc. literal
set in the matcher's `font_name_literal` catches PDF font subsets
where the leading prefix isn't strictly 6 caps (`/ABCDE+TimesNewRoman`
or `/ABCDEFG+TimesNewRoman` — both observed). Rule B's strict
`/[A-Z]{6}\+[A-Z][A-Za-z0-9-]+` regex misses these.

When Rule B becomes the canonical definition, relax the prefix:
```
/[A-Z]{4,8}\+[A-Z][A-Za-z0-9-]+
```
This catches 5/7-cap subsets without the matcher's separate
literal alternation. The literal set goes away (was a
matcher-only workaround for this gap).

Open: should we instead keep an EXPLICIT alternation against the
known font names (`+TimesNewRoman|+Palatino|...`) for higher
precision, accepting the inflexibility against unknown fonts?
Tradeoff — relaxed regex generalizes, literal alternation stays
precise. Decide at implementation time on a sample.

### Code deltas

- `glossapi_rs_cleaner` — add `per_rule_match_counts` field,
  `emit_rule_counters` flag, accumulation hooks at each rule's
  application site.
- `clean_and_stats_rowsharded.py` — pass `emit_rule_counters=False`
  for production runs; remove the matcher invocation; keep
  parquet schema columns if downstream tooling expects them
  (populate from cleaner counters when flag is on, leave 0 in
  production).
- `clean_and_stats_full.py`, `clean_with_three_counter_thresholds.py`,
  `compute_drop_decisions.py` — same refactor; matcher invocation
  removed. (Or these scripts retire if the row-sharded driver is
  the canonical one going forward.)
- `pull_top_three_counters_pdf.py`, `pull_deletion_band_samples.py`,
  `analyze_cleaning_concentration.py`, `sample_char_strip_bands.py`
  — keep reading the same parquet column names (back-compat); the
  values now come from the cleaner directly.
- `glossapi_rs_noise` — delete entirely OR keep as
  `glossapi_rs_pattern_probe` for experimental rule discovery
  (decide at implementation time based on whether a discovery
  wave is queued).
- `corpus_clean_normalization/specs/three_counter_spec_*.json` —
  delete (after Point 6 it's already a 2-counter spec; after
  Point 7 it's redundant).

### Net effect

- Throughput: production runs avoid the matcher's per-doc regex
  scan entirely. Estimated 25–40% speedup on the cleaning
  step (matcher and cleaner each take ~half the per-doc CPU).
- Correctness: counter values exactly match cleaner activity, by
  construction. No drift possible.
- Simplicity: one regex definition per rule, in one file. One
  Rust crate doing pattern work for the corpus pipeline.
- Discovery: retained either via the renamed pattern-probe crate
  or resurrected on demand.

### Open subquestions

- Single relaxed `/[A-Z]{4,8}+` regex vs. explicit font-name
  alternation — decide on a sample. Calibrate precision/recall.
- Counter granularity: per-rule (Rule B fires N times) or
  per-rule-component (Rule B's `GLYPH<…>` arm fires X times,
  `/uniXXXX` arm fires Y times)? Per-component gives finer
  diagnostic value but more bookkeeping. Probably per-rule for
  now, per-component if a discovery wave wants it.
- Do we also want the cleaner to emit a "total chars matched per
  rule" counter (not just count)? Useful for residue accounting.
  Cheap to add when the rule is applied.

---

## Point 8 — `Corpus.clean()` and `clean_text()` must share a single policy builder

**Status**: discussed and agreed in principle; awaiting finalisation.

**Source**: external review
`/home/foivos/glossAPI-development/REVIEW_codex_three_counter_pipeline_20260421.md`
(P1: "Corpus.clean() and direct clean_text() can apply different
character policies"). Verified.

### The problem

Two cleaner entry points build the allowed-char set DIFFERENTLY,
which means the same `scripts_to_keep` argument produces different
cleaning behavior depending on which path is invoked:

- **Direct path** (`clean_text` / `clean_text_with_stats`):
  `cleaning_module.rs:941::build_script_char_sets` always adds
  `punctuation`, `numbers`, `common_symbols` to `allowed_chars`,
  regardless of whether they appear in `scripts_to_keep`.
- **Production directory path** (`Corpus.clean()` →
  `run_complete_pipeline()` → `directory_processor.rs:297`):
  builds `allowed_chars` ONLY from `scripts_to_keep` plus
  whitespace. No auto-add.

Concrete: `Corpus.clean(scripts_to_keep=["greek", "latin"])` strips
ALL ASCII punctuation, digits, and common symbols (commas, periods,
parens, `0-9`, `%`, `$`, etc.). Same call against `clean_text()`
keeps them. Production may have been silently mangling output for
any caller that supplied a restricted `scripts_to_keep` list.

### Direction agreed

Punctuation, numbers, and common_symbols are UNIVERSAL — they appear
in every language and every reasonable corpus document. Stripping
them is never a sensible default. Standardize on the
`build_script_char_sets` behavior (auto-add) as the single policy:

- ONE Rust function builds the (allowed, unusual) char-set pair
  from `scripts_to_keep`.
- Both `clean_text` and `directory_processor` (and any other
  caller) call it. No second copy.
- Punct/numbers/common_symbols are ALWAYS in the allowed set
  unless explicitly opted out via a future flag (not in scope —
  current callers don't need it).

### Concrete edits

- `directory_processor.rs:297` — replace the manual
  `for key in &scripts_to_keep { allowed_chars.extend(...) }` block
  with a call to `cleaning_module::build_script_char_sets`.
- `cleaning_module.rs:941` — make `build_script_char_sets` `pub`
  so `directory_processor` can call it. (It's currently private to
  `cleaning_module`.)
- Add a regression test:
  `run_complete_pipeline(scripts_to_keep=["greek", "latin"])`
  preserves `0-9`, `,`, `.`, `(`, `)` the same way
  `clean_text(scripts_to_keep=["greek", "latin"])` does.

### Aligned with Principle 3

This is exactly the "one engine, one definition" principle applied
to char-set construction. Two paths defining policy = drift.

(further discussion points to be appended below — none agreed yet)

---

## Point 9 — Pilot B becomes the single Phase-A owner; legacy md surfaces deleted

**Status**: discussed; coordination point with other agent.

**Source**: external review `REVIEW_codex_three_counter_pipeline_20260421.md`
P2 ("Phase-A markdown cleanup has multiple live implementations and
verifier APIs"). Verified.

### The problem

Phase-A markdown normalization currently has SIX live
implementations / verifier surfaces:

- `md_module.rs:593` — `normalize_md_syntax_with_stats` (line-based,
  with stats).
- `md_module.rs:823` — `normalize_md_syntax` (line-based,
  reimplementing the same Phase-A from scratch instead of
  delegating).
- `md_format.rs` — Pilot A: full parser round-trip (over-normalizes,
  not the chosen production direction per
  `docs/PHASE_A_PARSER_BACKED_IMPLEMENTATION_REVIEW_2026-04-24.md`).
- `md_format_surgical.rs` — Pilot B: surgical parser-backed
  rewriter. Documented as the right architectural direction.
- `cmark_gfm_oracle.rs`, `md_verify.rs` — preview / structural
  verifier helpers used by both pilots.

All exposed at PyO3 (`lib.rs:57..67`): `apply_phase_a`,
`phase_a_alteration_stats`, `phase_a_stats_jsonl_line`,
`format_parsed_py` (Pilot A), `format_surgical_py`,
`format_surgical_checked_py` (Pilot B), `dual_verify_py`,
`cmark_gfm_verify_py`, `verify_md_preview_equivalent_py`,
`verify_md_structural_py`, `phase_a_policy_py`. Experimental
helpers are indistinguishable from production exports.

The production cleaner (`core_clean_text_with_stats` step 5) still
calls the LINE-BASED `normalize_md_syntax`. The strong Pilot B
scorecard is not in the critical path.

### Direction agreed

Pilot B is being developed by the other agent on
`codex/three-counter-pipeline-20260421` (and follow-on branches)
and is documented as the chosen production direction. **We do not
develop Pilot B ourselves.** When the other agent declares Pilot B
production-ready (and a corpus-scale scorecard accepts it), we:

1. Route `core_clean_text_with_stats` through
   `md_format_surgical::format_surgical_checked` instead of
   `md_module::normalize_md_syntax`.
2. Delete the legacy line-based implementations:
   - `md_module::normalize_md_syntax`
   - `md_module::normalize_md_syntax_with_stats`
   - PyO3 exports `apply_phase_a`, `phase_a_alteration_stats`,
     `phase_a_stats_jsonl_line`, `phase_a_policy_py`.
3. Delete Pilot A entirely:
   - `md_format.rs`
   - PyO3 exports `format_parsed_py`, `dual_verify_py`.
4. Keep:
   - `md_format_surgical.rs` (Pilot B → the production owner).
   - `cmark_gfm_oracle.rs`, `md_verify.rs` — verifier helpers
     Pilot B's checked wrapper uses internally.
   - PyO3 exports needed by production:
     `format_surgical_checked_py` (or rename to a non-pilot name
     once it's the only one).
5. Quarantine remaining audit / scorecard helpers:
   - `verify_md_preview_equivalent_py`, `verify_md_structural_py`,
     `cmark_gfm_verify_py` — keep, but rename / namespace to make
     "this is dev-only verification, not production cleaning"
     unambiguous (e.g., `_audit_*` prefix or a sub-module).

### Why we skip the consolidation work ourselves

The other agent's Pilot B is the architecturally-correct
endpoint. Trying to consolidate the existing 6 surfaces ourselves
without Pilot B's verification wrapper would mean either:
- adopting line-based as the single owner (regression vs. Pilot B),
  or
- adopting Pilot A (known over-normalization failures), or
- bringing Pilot B forward without finishing it (wasted effort,
  drift with the other agent's work).

So we wait, then integrate.

### Coordination signals to watch for

- Other agent merges a "Pilot B is production-ready" decision
  into `glossAPI-development/rust/glossapi_rs_cleaner/docs/`.
- A corpus-scale scorecard run (full-corpus or stratified sample)
  accepting Pilot B's Phase-A output.
- The other agent removes the `phase_a_mode` integration switch
  recommended in
  `PHASE_A_PARSER_BACKED_IMPLEMENTATION_REVIEW_2026-04-24.md`,
  pinning Pilot B as the only mode.

When those land, Point 9's deletion phase becomes mechanical.

### Open subquestions

- Do we keep `format_surgical_checked_py` exported under that name,
  or rename to e.g. `phase_a_py` once Pilot B is the only Phase-A?
  Reader-clarity argues for renaming.
- Does the LINE-LEVEL `normalize_md_syntax` get replaced before
  Points 1–7 land or after? Probably AFTER — Points 1–7 don't
  touch Phase-A; let those land first, then Pilot B integration
  is a clean separate change.

---

## Point 10 — Bug fixes (post-deduplication phase)

**Status**: queued for after Points 1–9 land.

**Source**: external review
`/home/foivos/glossAPI-development/REVIEW_codex_three_counter_pipeline_20260421.md`
(P1 §"Token-category offsets are byte offsets..." and P2 §"The new
perf test fails under normal cargo test"). Both verified.

These are real bugs surfaced during the review, but they are
correctness / test-infra issues — not architectural deduplication.
Sequenced as a final clean-up phase: the deduplication (Points 1–9)
lands first, simplifying the surface area; then the bugs get fixed
against the reduced codebase.

### Bug 1 (P1 correctness) — Byte vs character offsets in match spans

**Where**:
- Rust emitter: `rust/glossapi_rs_noise/src/noise_metrics.rs` —
  emits regex / Aho-Corasick spans as BYTE indexes.
- Python consumers:
  `src/glossapi/scripts/token_category_debug_common.py` and
  `src/glossapi/corpus/phase_clean.py` — slice `page_text` with
  the same indexes as if they were CHAR indexes
  (`page_text[start_char:end_char]`).

**Symptom**: Greek (or any non-ASCII) text earlier in the page
shifts byte offset relative to char offset. Effects:
- `match_index.jsonl` records the wrong span.
- The `end > len(page_text)` guard silently drops rows whose byte
  offset overruns char-length.
- Review bundles cite the wrong context window.
- Downstream evidence (Gemini reviews, regex-validation counts) is
  computed against the wrong text.

**Severity**: P1 correctness. Greek is the corpus core, so this
corrupts a meaningful fraction of all matches.

**Sequencing relative to Point 7**: After Point 7 lands, the
matcher crate either becomes `glossapi_rs_pattern_probe` (kept for
experimental rule discovery) or is deleted entirely. If kept, the
bug fix targets that renamed crate. If deleted, the span emission
moves into the cleaner's optional review-mode output and the fix
goes there instead. Either way, the fix lands in whichever code
path emits match spans for review-bundle generation.

**Fix sketch**:
- Emit explicit CHARACTER offsets from Rust alongside (or
  replacing) byte offsets. Python consumes char offsets directly.
- Add a regression test: input where Greek prose precedes a
  matched ASCII pattern. Expected slice = exact pattern, not a
  shifted window.

### Bug 2 (P2 test-infra) — `perf_mixed_doc_throughput_floor` fails in debug

**Where**: `rust/glossapi_rs_cleaner/src/cleaning_module.rs`
(search for `perf_mixed_doc_throughput_floor`).

**Symptom**:
```bash
cargo test perf_mixed_doc_throughput_floor
```
fails with `throughput regression: 732947 chars/sec < 5000000 floor`.
The 5M chars/sec floor is a release-profile expectation; default
`cargo test` is debug, ~7× slower.

**Severity**: P2 — blocks normal Rust test runs but doesn't affect
production correctness.

**Fix**: `#[ignore]` on the test plus require explicit
`cargo test perf_mixed_doc_throughput_floor -- --ignored --release`
to invoke. Alternatives (`#[cfg(not(debug_assertions))]`, move to
`benches/`) are also acceptable; pick whichever the rest of the
test suite uses for similar regression checks.

(further discussion points to be appended below — none agreed yet)
