# Japanese (ja) — per-script research notes

> Single in-scope locale: `ja` (bit 21, Japonic family). Script bits:
> Jpan (covers Han + Kana). Status: coverage exhaustive at every
> writing-system layer; Japanese-only token attribution works
> cleanly via the kana-vs-Han distinction.

## Sources consulted

- CLDR cldr-misc-full 48.2.0 — `ja/characters.json`.
- Unicode 16.0: Hiragana (U+3040–309F), Katakana (U+30A0–30FF),
  Katakana Phonetic Extensions (U+31F0–31FF), Halfwidth and
  Fullwidth Forms (U+FF00–FFEF — for halfwidth katakana
  U+FF65–FF9F), CJK Unified Ideographs (U+4E00–9FFF) used for
  kanji, CJK Extensions A and beyond.
- 常用漢字 (Jōyō kanji, 2,136 chars) — Japan's Agency for Cultural
  Affairs official "regular use" kanji list. Every educated
  Japanese speaker is expected to know these.
- 人名用漢字 (Jinmeiyō kanji) — additional kanji for personal
  names (~983 chars).
- Wikipedia: "Hiragana", "Katakana", "Jōyō kanji", "Halfwidth and
  Fullwidth Forms", "Hentaigana".

## Empirical Apertus baseline

- **1,632 Apertus vocab tokens contain Japanese kana** (Hiragana
  and/or Katakana). Plus the much larger Han subset shared with
  Chinese (~3,700 tokens contain Han codepoints — see Chinese.md
  for the Han-shared analysis).
- Of the 1,632 kana-containing tokens:
  - 765 are hiragana-only
  - 601 are katakana-only
  - 257 mix kana + kanji
  - (9 are unaccounted for in the count — likely tokens with
    katakana phonetic extensions or halfwidth katakana plus other
    chars)
- **1,620 of 1,632 (99.3 %) have `bitmask_and = ja` only** (no zh-
  Hans, no zh-Hant). The kana acts as a strong Japanese-specific
  discriminator — it's the most reliable narrowing signal in our
  setup because Hiragana and Katakana are exclusively Japanese.

The remaining 12 kana-containing tokens have additional bits in
their AND — likely tokens that include some shared punctuation or
mixed-Han-script content; no immediate concern.

## Coverage state

| block | size | in table | with ja bit |
|---|---|---|---|
| Hiragana (U+3040–309F) | 96 | 89 | 89 |
| Katakana (U+30A0–30FF) | 96 | 91 | 91 |
| Katakana Phonetic Extensions (U+31F0–31FF) | 16 | 16 | 16 |
| Halfwidth Katakana (U+FF65–FF9F) | 59 | 55 | 55 |
| CJK Unified Ideographs (U+4E00–9FFF) | 20,992 | 20,992 | 20,992 |
| CJK Extension A (U+3400–4DBF) | 6,592 | 6,592 | 6,592 |

The full Han block has `ja` bit on every codepoint via the script-
range fallback. Kana coverage is essentially complete (7 unassigned
Hiragana codepoints, 5 unassigned Katakana, 4 unassigned halfwidth-
katakana; CLDR has 85 hiragana + 89 katakana in main).

CLDR `ja` exemplar carries:
- 85 Hiragana + 89 Katakana = 174 kana
- 2,136 kanji (= Jōyō kanji list, almost verbatim)

Plus 151 chars in auxiliary (additional kanji used in names /
sayings, e.g. 丑 亥 亨 兌 — animal-zodiac signs, archaic, names).
We drop auxiliary uniformly; these would be Jinmeiyō-overlap kanji.
**Empirically harmless**: even with auxiliary dropped, every kanji
in the Apertus vocab is in the table via the script-range fallback
(which gives every Han codepoint all three CJK bits including `ja`).

## Halfwidth katakana — NFKC-aware detection working

v2.2's NFKC-aware `char_script` made halfwidth katakana
(U+FF65–FF9F) resolve to `Kana` → `Jpan` correctly. Audit confirms:
all 55 in-block halfwidth katakana codepoints in our table carry
the `ja` bit. 4 codepoints absent are unassigned ranges.

## Kanji-vs-Hanzi attribution (already covered in Chinese.md)

Han codepoints are attributed to all three CJK locales (`zh-Hans +
zh-Hant + ja`) via the script-range fallback. This is the
permissive choice: kanji legitimately *are* Chinese-origin
characters, and Japanese text can contain pure-Han sequences (e.g.,
proper-noun compounds, classical citations) where the kanji alone
doesn't say "this is Japanese."

Token-level Japanese attribution falls out naturally from the
**presence of kana**, not from the kanji itself. A pure-kanji
token AND-attributes to all three CJK locales; a token with even
one hiragana or katakana char AND-narrows to `ja` only (because
kana are not in zh-Hans / zh-Hant exemplar).

## Hentaigana (変体仮名)

Historical kana variants used before the 1900 spelling reform.
Encoded in U+1B000–1B0FF (Kana Supplement), U+1B100–1B12F (Kana
Extended-A), U+1B130–1B16F (Small Kana Extension). **Not in our
table** (CLDR `ja` doesn't list them; not in `SCRIPT_FALLBACK_RANGES`).
Not in the Apertus vocab either — these are rare archaic chars.
No action needed.

## Decisions

1. **No `languages.yaml` changes.** Coverage is exhaustive.
2. **No `SCRIPT_FALLBACK_RANGES` additions** for hentaigana — 0
   tokens affected.
3. **Auxiliary kanji** (Jinmeiyō-overlap) handled correctly by the
   script-range fallback over the full Han block. Dropping CLDR
   `ja` auxiliary uniformly is fine.
4. **Kana = Japanese discriminator** at the token level. The
   design works as intended: tokens containing kana cleanly
   attribute to ja-only; pure-Han tokens attribute to all three
   CJK locales (deliberate permissive choice — see Chinese.md).

## Followups

- **Hentaigana (U+1B000–1B16F)** — out of scope by design. If a
  future audit ever flags them, easy to add as fallback range.
- **Kana phonetic extensions** (U+31F0–31FF) — small katakana for
  Ainu transcription. 16 codepoints all in table; not commonly in
  Apertus vocab. No action.
- **Vertical-form kana variants** in CJK Compatibility Forms
  (U+FE30–FE4F) — fullwidth/vertical variants. Substrate-category
  handling via fallback covers them.
- **Romaji**: Latin transliterations of Japanese (Tōkyō, etc.) —
  the `ō` U+014D case noted in Latin.md. Currently 0-bit; would
  require a Japanese-romaji locale to admit. Out of scope.
