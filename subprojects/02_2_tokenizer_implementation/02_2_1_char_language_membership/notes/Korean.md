# Korean — per-script research notes

> Single in-scope locale: `ko` (bit 23). Status: coverage complete;
> no `languages.yaml` changes required.

## Sources consulted

- CLDR cldr-misc-full 48.2.0 — `ko/characters.json`.
- Unicode 16.0: Hangul Syllables (U+AC00–D7AF, 11,172 precomposed),
  Hangul Jamo (U+1100–11FF, 256 jamo for decomposition), Hangul
  Compatibility Jamo (U+3130–318F), Hangul Jamo Extended-A
  (U+A960–A97F), Hangul Jamo Extended-B (U+D7B0–D7FF).
- Wikipedia: "Hangul Syllables", "Hangul Jamo", "Korean Hanja",
  "Korean punctuation".
- KS X 1001 / KS X 1002 — Korean industry encoding standards.
- National Institute of Korean Language (국립국어원) — Korean
  orthography authority.

## Empirical Apertus baseline

- 4,439 Apertus vocab tokens contain Hangul codepoints.
- All 4,439 are `status: text`.
- 858 unique Hangul codepoints used (out of ~11,172 precomposed
  syllables — that's ~7.7 % vocab coverage of the syllable block;
  consistent with the most-common-syllables-only typical of a
  byte-level BPE on web Korean).
- **0 mixed Hangul + Han (Hanja) tokens.** Apertus's vocab does
  not contain any token that mixes Hangul with Han characters.
  Consequence: the Hanja-policy question raised in the PLAN_v3
  research checklist is **empirically moot** on this vocab.

Per-token AND result:

- All 4,439 Korean tokens have `bitmask_and` containing **only the
  `ko` bit** (no zh-Hans / zh-Hant / ja bits). Clean Korean-only
  attribution, exactly as designed.

## Coverage state

| block | size | in table | notes |
|---|---|---|---|
| Hangul Syllables (U+AC00–D7A3) | 11,172 | 11,172 (100 %) | full coverage via CLDR `ko` main + script-range fallback |
| Hangul Jamo (U+1100–11FF) | 256 | 256 (100 %) | full coverage |
| Hangul Compatibility Jamo (U+3130–318F) | 96 | 94 | 2 unassigned codepoints |
| Hangul Jamo Extended-A (U+A960–A97F) | 32 | 0 | rare archaic / pre-modern jamo; not in Apertus vocab |
| Hangul Jamo Extended-B (U+D7B0–D7FF) | 80 | 0 | rare; not in Apertus vocab |

Coverage of the in-text part of Korean is exhaustive: every Hangul
syllable and every modern jamo is in the table with the `ko` bit.

## Hanja policy (resolved empirically)

The PLAN_v3 research checklist asked: *"should a Hanja-containing
token carry the `ko` bit, the Sinitic CJK bits, or both?"*

Empirically the question doesn't arise — there are zero Hanja-
containing Apertus tokens. But the policy applied by the current
build is principled and worth documenting:

- A Han codepoint like `中` (U+4E2D) appears in the table with
  `zh-Hans + zh-Hant + ja` language bits, derived from CLDR `zh-Hans`
  / `zh-Hant` exemplar plus the script-range fallback over the CJK
  Unified Ideographs block. **No `ko` bit.**
- Korean is in scope, but CLDR `ko` exemplar does not include Hanja
  in `main` (only in `auxiliary`, which we exclude project-wide).
- So a hypothetical token like `한字` (Hangul + Hanja) would AND to
  `zh-Hans ∩ zh-Hant ∩ ja ∩ ko = {}` — empty. The token would have
  `status: text` and `bitmask_and = 0` because the two char-level
  bit sets don't intersect. **AND-rejects every in-scope language.**

This matches the strict-rejection framing. If the C3 extension or a
future Apertus vocab introduces Hanja-mixed tokens we'd see this
empirically and could revisit; the principled options would be:

- **(a) Keep current.** Strict rejection — Hanja-only and Hangul-
  only tokens each attribute cleanly; mixed tokens collapse to 0.
- **(b) Add `ko` to Han codepoints' CLDR-derived bits.** Permissive
  — Korean text legitimately contains Hanja. Would change Han
  codepoints from `{zh-Hans, zh-Hant, ja}` to `{zh-Hans, zh-Hant,
  ja, ko}`. Empirically defensible: Korean academic / legal texts
  routinely cite Hanja.

Today (a) is correct because empirically Apertus has no Hanja-mixed
tokens. **Deferred** until and unless that changes.

## Halfwidth / fullwidth compatibility forms

NFKC-aware script detection (v2.2) catches halfwidth Katakana →
`Kana` → Jpan; halfwidth Hangul (U+FFA0–FFDC compatibility jamo)
isn't a thing Apertus has tokens for, but if it were, it'd resolve
to Hangul via NFKC normalization in `char_script`. Halfwidth Korean
syllables don't exist as a separate block (Hangul Syllables itself
is the canonical encoding for precomposed syllables).

## Decisions

1. **No `languages.yaml` changes.** Coverage is exhaustive for the
   in-text portion.
2. **No `EXTRA_SUBSTRATE_CODEPOINTS` additions.** Korean
   punctuation in CLDR `ko`'s `punctuation` set is mostly already
   substrate-category (Po, Pd, Pe, Ps) and admits both Hangul and
   ASCII pair forms. The vocab doesn't surface a problem here.
3. **Hanja policy: strict-rejection for now**, will revisit if /
   when Hanja-mixed tokens appear in a future Apertus snapshot.

## Followups

- **Hangul Jamo Extended A/B (U+A960–A97F, U+D7B0–D7FF)**: 0
  coverage by design (CLDR doesn't list them; not in vocab). If
  ever needed, easy fix — add the ranges to
  `SCRIPT_FALLBACK_RANGES` for `Hang`.
- **Old / Middle Korean** texts may use archaic jamo or Hanja-heavy
  conventions; out of scope for Apertus's modern-Korean corpus.
- **Korean punctuation conventions** — colon `：` U+FF1A, paren
  `（）` etc. — handled by the existing substrate rule for the
  full-width-Latin compatibility forms (these are CJK punctuation
  but Unicode category P*, so substrate everywhere). Verified
  during the v2.2 fullwidth-substrate work.
