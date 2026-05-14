# Per-Line Greek Badness Scorer — Branch Plan

Date: 2026-05-04
Off branch: `glossapi/development`
Branch: `cleaner/per-line-badness-20260504`

## Concrete work to ship now

1. **Extend `normalize::fold_codepoint`** with two char-fold families:
   - **A. Math symbols → Greek capitals**
     - `∆ U+2206 INCREMENT → Δ U+0394`
     - `Ω U+2126 OHM SIGN → Ω U+03A9` (Letterlike Symbols)
   - **B. Mathematical Alphanumeric Symbols Greek block (U+1D6A8 - U+1D7CB)**
     - Bold, italic, bold-italic, sans, sans-italic variants of every
       Greek capital and lowercase letter. ~290 codepoints total.
     - Programmatic mapping by block-offset (the Unicode standard lays
       these out parallel to U+0391-U+03A9 / U+03B1-U+03C9).

2. **Add `score_line_greek_badness` to `glossapi_rs_noise`** as a public
   Rust function + PyO3 binding. Port `scorer_v6.py` logic faithfully:
   - Per-length empirical allowlists (from `allowed_runs_by_length.json`)
   - Acronym exception (all-uppercase word in mixed-case line)
   - Sigma exception (`σ.` / `δισ.` short-abbreviation period rule)
   - Pre-filter (`greek_letter_ratio > 0.5` AND `non_ws_chars >= 10`)
   - Score formula `2.0 * run_rate + 2.5 * sigma_rate`
   - Returns full breakdown struct (run_pen, sigma_pen, vowel_pen,
     cons_pen, len_greek, raw score)

3. **Tests** for both. Fold table tests verify each new codepoint maps
   to the expected output. Scorer tests cover the failure modes we
   characterized (`ξψ`, `κκκ`, `σ.`, `ΕΕ` acronym, polytonic, the
   `ςυμμετεχόντων` family).

4. **Build, then re-run on apertus**:
   - Re-clean the 22,842 pre-OCR docs through the dev cleaner (so the
     new folds + existing fold_codepoint apply).
   - Score every cleaned line.
   - Pull 3 random samples per zone for review.

That's the whole scope of THIS branch.

## Open questions for later (not implementing now)

These are things we discussed but are NOT part of this branch's work.
Captured here so we don't lose them.

- **Can we DECODE mojibake into Greek?** I.e., reverse the font-substitution
  patterns to recover the original Greek text. Examples: `ρξήζηεο` ↔
  `χρήστης`, `Ρϊΐ3θόο` ↔ `Φαίδρος`. These are font-substitution mappings
  and could in principle be inverted if we know the source font. ftfy
  does part of this for byte-level mojibake; PDF font-sub mojibake is
  harder. Worth a focused investigation.
- **Cyrillic homoglyphs in Greek context** (А → Α, е → ε, etc.) — fold
  only when surrounded by Greek letters. Context-sensitive.
- **Latin homoglyphs in Greek context** (A → Α, B → Β etc.) — same
  problem, more dangerous because Latin appears legitimately in Greek
  text everywhere.
- **Lunate sigma variants** (ϲ → σ, Ϲ → Σ) — older Greek typography,
  low frequency, could fold but not urgent.
- **Standalone diacritics** (¨ ´ standalone) → context-fold to
  letter+combining mark.
- **Pipeline integration** of the line scorer into `Corpus.clean()` as a
  drop/normalize/keep predicate. Currently the scorer just outputs
  numbers; the cleaner doesn't yet use them to drop lines. Once
  thresholds are user-set, this is the natural next step.
- **Line-drop with reasons**: when we do wire the scorer into the
  cleaner as a drop predicate, record `line_drops_by_score` and
  `line_drop_reasons` in `CleanStats` for audit.
- **Mojibake-fix vs line-drop sequencing**: our intuition is "normalize
  first, then score, then drop" so we don't throw away salvageable
  lines. Validate empirically once both pieces are in place.
- **Karamanli text** (Turkish in Greek script) and other minority
  Greek-script writing systems — the per-line scorer flags these as
  noise. Decide later whether to keep, drop, or route differently.
- **Re-mining allowlists on a polytonic + ancient Greek corpus** to
  reduce FPs on byzantine/classical text.
