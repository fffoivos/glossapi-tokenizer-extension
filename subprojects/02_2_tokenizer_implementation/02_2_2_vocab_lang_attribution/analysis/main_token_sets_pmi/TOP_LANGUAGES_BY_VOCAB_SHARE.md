# Apertus tokenizer — per-language vocab share

**Source**: `summary.tsv` + per-key `tables/<key>__masked.txt`
(this directory). **Apertus base vocab**: 131,072 tokens.
**Rebuild date**: 2026-05-15 (PMI/char-mask schema v5).

This table ranks every language in the cap-hit set by the size
of its **masked-A set** — the tokens that pass *both* the
PMI/count admissibility test against the count-pooled cap-hit
marginal AND the strict char-admissibility mask for that
language. The cap-hit set is every language with at least 1 B
observed Apertus-token firings in the attribution pass.

Per the attribution methodology, English is double-keyed across
two source corpora (`eng_Latn` = general crawl, `eng_Latn_fineweb_hq`
= FineWeb-HQ). Both keys describe the same language, so the row
below merges them: the English `masked` count is the union of the
two key sets (23,774), and English `unique` is computed against
every non-English language. After the merge there are **86**
language rows.

## Reading caveats

1. **The sets overlap heavily.** A single token like `_a`, `s`,
   or `,` is admissible under many Latin-script languages at
   once. Sum of masked counts across all 86 languages =
   **184,580** (140.8 %
   of vocab) — the excess over 100 % is double-counting.
2. **Union coverage**: **113,184 / 131,072**
   (86.35 %) tokens are
   admissible under at least one language. **17,888**
   (13.65 %) are unattributed
   (byte-fallback, punct, code, mojibake, residual long tail).
3. **Script matters more than family for tokens.** A BPE token's
   admissibility set is fixed by the script its codepoints lie in,
   not by linguistic family. The `script` column carries the ISO
   15924 code (`Latn`, `Grek`, `Cyrl`, `Arab`, `Hang`, `Jpan`,
   `Hani`, …). Languages on isolated scripts (Korean Hangul,
   Greek, Japanese kana) trivially get 100 %-unique sets because
   no other cap-hit language shares any codepoint with them.
   Latin-script languages share a large substrate and have low
   uniqueness even when very different linguistically.
4. **`und_*` keys** (`und_Mong`, `und_Kana`, `und_Grek`,
   `und_Cyrl`) and `gmh_Latn` are unmapped to ISO-639-1 codes;
   `iso` = `unmapped` in the row. Kept in for completeness.

## Methodology pointers

- Builder: `build.py` (this directory)
- PMI predicate / knobs: `manifest.json` (`alpha = 0.5`,
  `delta = 1.0`, `min_count = 100`, `marginal_floor = 1 e9`)
- Char-mask source: `02_2_1_char_language_membership/artifacts/`
  `token_language_bitmask.parquet`
- Firing counts: `02_2_2_vocab_lang_attribution/outputs/`
  `histogram_matrix.npz`
- Full methodology: `02_2_4_language_category_promotion/`
  `METHODOLOGY.md` + `PMI_PROMOTION_SPEC.md`
- Table reproduction: `build_top_languages_table.py` (this dir).

## Table — ranked by masked-set size

How big is the language's PMI/mask-admissible set as a fraction
of the 131,072-token Apertus vocab. Languages on shared scripts
(Latin, Cyrillic, Arabic) rank high here because they inherit the
shared substrate; this is the "how much of the vocab *touches*
this language" view.

| # | iso | language | script | masked | % vocab | unique | % unique |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `en` | English | `Latn` | 23,774 | 18.14 % | 16,288 | 68.51 % |
| 2 | `fr` | French | `Latn` | 9,694 | 7.40 % | 5,015 | 51.73 % |
| 3 | `de` | German | `Latn` | 7,329 | 5.59 % | 4,855 | 66.24 % |
| 4 | `ar` | Standard Arabic | `Arab` | 7,146 | 5.45 % | 239 | 3.34 % |
| 5 | `ar-MA` | Moroccan Arabic | `Arab` | 7,029 | 5.36 % | 52 | 0.74 % |
| 6 | `es` | Spanish | `Latn` | 6,714 | 5.12 % | 1,360 | 20.26 % |
| 7 | `pt` | Portuguese | `Latn` | 5,549 | 4.23 % | 881 | 15.88 % |
| 8 | `gl` | Galician | `Latn` | 5,460 | 4.17 % | 380 | 6.96 % |
| 9 | `it` | Italian | `Latn` | 4,712 | 3.59 % | 1,552 | 32.94 % |
| 10 | `ko` | Korean | `Hang` | 4,438 | 3.39 % | 4,438 | 100.00 % |
| 11 | `ca` | Catalan | `Latn` | 4,267 | 3.26 % | 647 | 15.16 % |
| 12 | `ru` | Russian | `Cyrl` | 4,153 | 3.17 % | 1,099 | 26.46 % |
| 13 | `ja` | Japanese | `Jpan` | 3,222 | 2.46 % | 2,161 | 67.07 % |
| 14 | `nl` | Dutch | `Latn` | 3,045 | 2.32 % | 514 | 16.88 % |
| 15 | `ro` | Romanian | `Latn` | 2,939 | 2.24 % | 718 | 24.43 % |
| 16 | `fa` | Persian | `Arab` | 2,785 | 2.12 % | 688 | 24.70 % |
| 17 | `la` | Latin | `Latn` | 2,768 | 2.11 % | 735 | 26.55 % |
| 18 | `zh-Hans` | Mandarin Chinese | `Hani` | 2,650 | 2.02 % | 1,589 | 59.96 % |
| 19 | `fil` | Filipino | `Latn` | 2,588 | 1.97 % | 469 | 18.12 % |
| 20 | `pl` | Polish | `Latn` | 2,570 | 1.96 % | 1,373 | 53.42 % |
| 21 | `hu` | Hungarian | `Latn` | 2,419 | 1.85 % | 1,278 | 52.83 % |
| 22 | `bg` | Bulgarian | `Cyrl` | 2,335 | 1.78 % | 117 | 5.01 % |
| 23 | `da` | Danish | `Latn` | 2,270 | 1.73 % | 190 | 8.37 % |
| 24 | `uk` | Ukrainian | `Cyrl` | 2,266 | 1.73 % | 326 | 14.39 % |
| 25 | `sr-Cyrl` | Serbian | `Cyrl` | 2,239 | 1.71 % | 182 | 8.13 % |
| 26 | `sv` | Swedish | `Latn` | 2,212 | 1.69 % | 404 | 18.26 % |
| 27 | `mk` | Macedonian | `Cyrl` | 2,207 | 1.68 % | 92 | 4.17 % |
| 28 | `af` | Afrikaans | `Latn` | 2,138 | 1.63 % | 288 | 13.47 % |
| 29 | `nb` | Norwegian Bokmål | `Latn` | 2,116 | 1.61 % | 357 | 16.87 % |
| 30 | `cs` | Czech | `Latn` | 2,058 | 1.57 % | 453 | 22.01 % |
| 31 | `id` | Indonesian | `Latn` | 2,035 | 1.55 % | 211 | 10.37 % |
| 32 | `ms` | Standard Malay | `Latn` | 2,023 | 1.54 % | 252 | 12.46 % |
| 33 | `eu` | Basque | `Latn` | 1,931 | 1.47 % | 716 | 37.08 % |
| 34 | `ur` | Urdu | `Arab` | 1,915 | 1.46 % | 227 | 11.85 % |
| 35 | `bs` | Bosnian | `Latn` | 1,879 | 1.43 % | 121 | 6.44 % |
| 36 | `sk` | Slovak | `Latn` | 1,848 | 1.41 % | 202 | 10.93 % |
| 37 | `tr` | Turkish | `Latn` | 1,833 | 1.40 % | 610 | 33.28 % |
| 38 | `hr` | Croatian | `Latn` | 1,782 | 1.36 % | 101 | 5.67 % |
| 39 | `fi` | Finnish | `Latn` | 1,767 | 1.35 % | 519 | 29.37 % |
| 40 | `vi` | Vietnamese | `Latn` | 1,564 | 1.19 % | 1,271 | 81.27 % |
| 41 | `et` | Standard Estonian | `Latn` | 1,492 | 1.14 % | 307 | 20.58 % |
| 42 | `el` | Modern Greek (1453-) | `Grek` | 1,479 | 1.13 % | 1,479 | 100.00 % |
| 43 | `sl` | Slovenian | `Latn` | 1,449 | 1.11 % | 164 | 11.32 % |
| 44 | `az` | North Azerbaijani | `Latn` | 1,428 | 1.09 % | 383 | 26.82 % |
| 45 | `hi` | Hindi | `Deva` | 1,388 | 1.06 % | 287 | 20.68 % |
| 46 | `be` | Belarusian | `Cyrl` | 1,354 | 1.03 % | 153 | 11.30 % |
| 47 | `uz-Latn` | Northern Uzbek | `Latn` | 1,310 | 1.00 % | 359 | 27.40 % |
| 48 | `cy` | Welsh | `Latn` | 1,293 | 0.99 % | 392 | 30.32 % |
| 49 | `lt` | Lithuanian | `Latn` | 1,228 | 0.94 % | 266 | 21.66 % |
| 50 | `sq` | Tosk Albanian | `Latn` | 1,220 | 0.93 % | 311 | 25.49 % |
| 51 | `ceb` | Cebuano | `Latn` | 1,171 | 0.89 % | 257 | 21.95 % |
| 52 | `is` | Icelandic | `Latn` | 1,156 | 0.88 % | 233 | 20.16 % |
| 53 | `sw` | Swahili (individual language) | `Latn` | 1,145 | 0.87 % | 317 | 27.69 % |
| 54 | `hy` | Armenian | `Armn` | 1,089 | 0.83 % | 1,089 | 100.00 % |
| 55 | `lv` | Standard Latvian | `Latn` | 1,062 | 0.81 % | 229 | 21.56 % |
| 56 | `mr` | Marathi | `Deva` | 1,037 | 0.79 % | 48 | 4.63 % |
| 57 | `ne` | Nepali (individual language) | `Deva` | 1,023 | 0.78 % | 38 | 3.71 % |
| 58 | `he` | Hebrew | `Hebr` | 961 | 0.73 % | 961 | 100.00 % |
| 59 | `te` | Telugu | `Telu` | 894 | 0.68 % | 894 | 100.00 % |
| 60 | `ky` | Kirghiz | `Cyrl` | 810 | 0.62 % | 44 | 5.43 % |
| 61 | `bn` | Bengali | `Beng` | 810 | 0.62 % | 810 | 100.00 % |
| 62 | `kk` | Kazakh | `Cyrl` | 762 | 0.58 % | 94 | 12.34 % |
| 63 | `uz-Cyrl` | Northern Uzbek | `Cyrl` | 674 | 0.51 % | 27 | 4.01 % |
| 64 | `ckb` | Central Kurdish | `Arab` | 645 | 0.49 % | 26 | 4.03 % |
| 65 | `tg` | Tajik | `Cyrl` | 591 | 0.45 % | 28 | 4.74 % |
| 66 | `th` | Thai | `Thai` | 560 | 0.43 % | 560 | 100.00 % |
| 67 | `kn` | Kannada | `Knda` | 548 | 0.42 % | 548 | 100.00 % |
| 68 | `ta` | Tamil | `Taml` | 525 | 0.40 % | 525 | 100.00 % |
| 69 | `mn` | Halh Mongolian | `Cyrl` | 512 | 0.39 % | 90 | 17.58 % |
| 70 | `ka` | Georgian | `Geor` | 480 | 0.37 % | 480 | 100.00 % |
| 71 | `ml` | Malayalam | `Mlym` | 377 | 0.29 % | 377 | 100.00 % |
| 72 | `gu` | Gujarati | `Gujr` | 180 | 0.14 % | 180 | 100.00 % |
| 73 | `pa` | Panjabi | `Guru` | 148 | 0.11 % | 148 | 100.00 % |
| 74 | `my` | Burmese | `Mymr` | 110 | 0.08 % | 110 | 100.00 % |
| 75 | `dv` | Dhivehi | `Thaa` | 0 | 0.00 % | 0 | 0.00 % |
| 76 | `unmapped` | Middle High German (ca. 1050-1500) | `Latn` | 0 | 0.00 % | 0 | 0.00 % |
| 77 | `lo` | Lao | `Laoo` | 0 | 0.00 % | 0 | 0.00 % |
| 78 | `or` | Odia | `Orya` | 0 | 0.00 % | 0 | 0.00 % |
| 79 | `km` | Khmer | `Khmr` | 0 | 0.00 % | 0 | 0.00 % |
| 80 | `unmapped` | Undetermined language | `Mong` | 0 | 0.00 % | 0 | 0.00 % |
| 81 | `am` | Amharic | `Ethi` | 0 | 0.00 % | 0 | 0.00 % |
| 82 | `unmapped` | Undetermined language | `Kana` | 0 | 0.00 % | 0 | 0.00 % |
| 83 | `bo` | Tibetan | `Tibt` | 0 | 0.00 % | 0 | 0.00 % |
| 84 | `si` | Sinhala | `Sinh` | 0 | 0.00 % | 0 | 0.00 % |
| 85 | `unmapped` | Undetermined language | `Grek` | 0 | 0.00 % | 0 | 0.00 % |
| 86 | `unmapped` | Undetermined language | `Cyrl` | 0 | 0.00 % | 0 | 0.00 % |

## Table — ranked by unique-set size

How many tokens the language owns ALONE — admissible under this
language and under no other cap-hit language. This is the "how
much of the vocab is *dedicated* to this language" view. Script-
isolated languages (own writing system) dominate the top of this
ranking because no other language can compete for their tokens.
Latin-script languages sit lower even when their masked set is
large, because their tokens are shared across the Latin family.

| # | iso | language | script | masked | % vocab | unique | % unique |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `en` | English | `Latn` | 23,774 | 18.14 % | 16,288 | 68.51 % |
| 2 | `fr` | French | `Latn` | 9,694 | 7.40 % | 5,015 | 51.73 % |
| 3 | `de` | German | `Latn` | 7,329 | 5.59 % | 4,855 | 66.24 % |
| 4 | `ko` | Korean | `Hang` | 4,438 | 3.39 % | 4,438 | 100.00 % |
| 5 | `ja` | Japanese | `Jpan` | 3,222 | 2.46 % | 2,161 | 67.07 % |
| 6 | `zh-Hans` | Mandarin Chinese | `Hani` | 2,650 | 2.02 % | 1,589 | 59.96 % |
| 7 | `it` | Italian | `Latn` | 4,712 | 3.59 % | 1,552 | 32.94 % |
| 8 | `el` | Modern Greek (1453-) | `Grek` | 1,479 | 1.13 % | 1,479 | 100.00 % |
| 9 | `pl` | Polish | `Latn` | 2,570 | 1.96 % | 1,373 | 53.42 % |
| 10 | `es` | Spanish | `Latn` | 6,714 | 5.12 % | 1,360 | 20.26 % |
| 11 | `hu` | Hungarian | `Latn` | 2,419 | 1.85 % | 1,278 | 52.83 % |
| 12 | `vi` | Vietnamese | `Latn` | 1,564 | 1.19 % | 1,271 | 81.27 % |
| 13 | `ru` | Russian | `Cyrl` | 4,153 | 3.17 % | 1,099 | 26.46 % |
| 14 | `hy` | Armenian | `Armn` | 1,089 | 0.83 % | 1,089 | 100.00 % |
| 15 | `he` | Hebrew | `Hebr` | 961 | 0.73 % | 961 | 100.00 % |
| 16 | `te` | Telugu | `Telu` | 894 | 0.68 % | 894 | 100.00 % |
| 17 | `pt` | Portuguese | `Latn` | 5,549 | 4.23 % | 881 | 15.88 % |
| 18 | `bn` | Bengali | `Beng` | 810 | 0.62 % | 810 | 100.00 % |
| 19 | `la` | Latin | `Latn` | 2,768 | 2.11 % | 735 | 26.55 % |
| 20 | `ro` | Romanian | `Latn` | 2,939 | 2.24 % | 718 | 24.43 % |
| 21 | `eu` | Basque | `Latn` | 1,931 | 1.47 % | 716 | 37.08 % |
| 22 | `fa` | Persian | `Arab` | 2,785 | 2.12 % | 688 | 24.70 % |
| 23 | `ca` | Catalan | `Latn` | 4,267 | 3.26 % | 647 | 15.16 % |
| 24 | `tr` | Turkish | `Latn` | 1,833 | 1.40 % | 610 | 33.28 % |
| 25 | `th` | Thai | `Thai` | 560 | 0.43 % | 560 | 100.00 % |
| 26 | `kn` | Kannada | `Knda` | 548 | 0.42 % | 548 | 100.00 % |
| 27 | `ta` | Tamil | `Taml` | 525 | 0.40 % | 525 | 100.00 % |
| 28 | `fi` | Finnish | `Latn` | 1,767 | 1.35 % | 519 | 29.37 % |
| 29 | `nl` | Dutch | `Latn` | 3,045 | 2.32 % | 514 | 16.88 % |
| 30 | `ka` | Georgian | `Geor` | 480 | 0.37 % | 480 | 100.00 % |
| 31 | `fil` | Filipino | `Latn` | 2,588 | 1.97 % | 469 | 18.12 % |
| 32 | `cs` | Czech | `Latn` | 2,058 | 1.57 % | 453 | 22.01 % |
| 33 | `sv` | Swedish | `Latn` | 2,212 | 1.69 % | 404 | 18.26 % |
| 34 | `cy` | Welsh | `Latn` | 1,293 | 0.99 % | 392 | 30.32 % |
| 35 | `az` | North Azerbaijani | `Latn` | 1,428 | 1.09 % | 383 | 26.82 % |
| 36 | `gl` | Galician | `Latn` | 5,460 | 4.17 % | 380 | 6.96 % |
| 37 | `ml` | Malayalam | `Mlym` | 377 | 0.29 % | 377 | 100.00 % |
| 38 | `uz-Latn` | Northern Uzbek | `Latn` | 1,310 | 1.00 % | 359 | 27.40 % |
| 39 | `nb` | Norwegian Bokmål | `Latn` | 2,116 | 1.61 % | 357 | 16.87 % |
| 40 | `uk` | Ukrainian | `Cyrl` | 2,266 | 1.73 % | 326 | 14.39 % |
| 41 | `sw` | Swahili (individual language) | `Latn` | 1,145 | 0.87 % | 317 | 27.69 % |
| 42 | `sq` | Tosk Albanian | `Latn` | 1,220 | 0.93 % | 311 | 25.49 % |
| 43 | `et` | Standard Estonian | `Latn` | 1,492 | 1.14 % | 307 | 20.58 % |
| 44 | `af` | Afrikaans | `Latn` | 2,138 | 1.63 % | 288 | 13.47 % |
| 45 | `hi` | Hindi | `Deva` | 1,388 | 1.06 % | 287 | 20.68 % |
| 46 | `lt` | Lithuanian | `Latn` | 1,228 | 0.94 % | 266 | 21.66 % |
| 47 | `ceb` | Cebuano | `Latn` | 1,171 | 0.89 % | 257 | 21.95 % |
| 48 | `ms` | Standard Malay | `Latn` | 2,023 | 1.54 % | 252 | 12.46 % |
| 49 | `ar` | Standard Arabic | `Arab` | 7,146 | 5.45 % | 239 | 3.34 % |
| 50 | `is` | Icelandic | `Latn` | 1,156 | 0.88 % | 233 | 20.16 % |
| 51 | `lv` | Standard Latvian | `Latn` | 1,062 | 0.81 % | 229 | 21.56 % |
| 52 | `ur` | Urdu | `Arab` | 1,915 | 1.46 % | 227 | 11.85 % |
| 53 | `id` | Indonesian | `Latn` | 2,035 | 1.55 % | 211 | 10.37 % |
| 54 | `sk` | Slovak | `Latn` | 1,848 | 1.41 % | 202 | 10.93 % |
| 55 | `da` | Danish | `Latn` | 2,270 | 1.73 % | 190 | 8.37 % |
| 56 | `sr-Cyrl` | Serbian | `Cyrl` | 2,239 | 1.71 % | 182 | 8.13 % |
| 57 | `gu` | Gujarati | `Gujr` | 180 | 0.14 % | 180 | 100.00 % |
| 58 | `sl` | Slovenian | `Latn` | 1,449 | 1.11 % | 164 | 11.32 % |
| 59 | `be` | Belarusian | `Cyrl` | 1,354 | 1.03 % | 153 | 11.30 % |
| 60 | `pa` | Panjabi | `Guru` | 148 | 0.11 % | 148 | 100.00 % |
| 61 | `bs` | Bosnian | `Latn` | 1,879 | 1.43 % | 121 | 6.44 % |
| 62 | `bg` | Bulgarian | `Cyrl` | 2,335 | 1.78 % | 117 | 5.01 % |
| 63 | `my` | Burmese | `Mymr` | 110 | 0.08 % | 110 | 100.00 % |
| 64 | `hr` | Croatian | `Latn` | 1,782 | 1.36 % | 101 | 5.67 % |
| 65 | `kk` | Kazakh | `Cyrl` | 762 | 0.58 % | 94 | 12.34 % |
| 66 | `mk` | Macedonian | `Cyrl` | 2,207 | 1.68 % | 92 | 4.17 % |
| 67 | `mn` | Halh Mongolian | `Cyrl` | 512 | 0.39 % | 90 | 17.58 % |
| 68 | `ar-MA` | Moroccan Arabic | `Arab` | 7,029 | 5.36 % | 52 | 0.74 % |
| 69 | `mr` | Marathi | `Deva` | 1,037 | 0.79 % | 48 | 4.63 % |
| 70 | `ky` | Kirghiz | `Cyrl` | 810 | 0.62 % | 44 | 5.43 % |
| 71 | `ne` | Nepali (individual language) | `Deva` | 1,023 | 0.78 % | 38 | 3.71 % |
| 72 | `tg` | Tajik | `Cyrl` | 591 | 0.45 % | 28 | 4.74 % |
| 73 | `uz-Cyrl` | Northern Uzbek | `Cyrl` | 674 | 0.51 % | 27 | 4.01 % |
| 74 | `ckb` | Central Kurdish | `Arab` | 645 | 0.49 % | 26 | 4.03 % |
| 75 | `dv` | Dhivehi | `Thaa` | 0 | 0.00 % | 0 | 0.00 % |
| 76 | `unmapped` | Middle High German (ca. 1050-1500) | `Latn` | 0 | 0.00 % | 0 | 0.00 % |
| 77 | `lo` | Lao | `Laoo` | 0 | 0.00 % | 0 | 0.00 % |
| 78 | `or` | Odia | `Orya` | 0 | 0.00 % | 0 | 0.00 % |
| 79 | `km` | Khmer | `Khmr` | 0 | 0.00 % | 0 | 0.00 % |
| 80 | `unmapped` | Undetermined language | `Mong` | 0 | 0.00 % | 0 | 0.00 % |
| 81 | `am` | Amharic | `Ethi` | 0 | 0.00 % | 0 | 0.00 % |
| 82 | `unmapped` | Undetermined language | `Kana` | 0 | 0.00 % | 0 | 0.00 % |
| 83 | `bo` | Tibetan | `Tibt` | 0 | 0.00 % | 0 | 0.00 % |
| 84 | `si` | Sinhala | `Sinh` | 0 | 0.00 % | 0 | 0.00 % |
| 85 | `unmapped` | Undetermined language | `Grek` | 0 | 0.00 % | 0 | 0.00 % |
| 86 | `unmapped` | Undetermined language | `Cyrl` | 0 | 0.00 % | 0 | 0.00 % |

## Rank deltas — who climbed, who fell

Side-by-side comparison of the two rankings. **Δ = masked rank −
unique rank**: positive Δ means the language climbed when we re-
ranked by unique tokens, negative means it dropped. "Script
alone" = no other language in the cap-hit set uses that ISO 15924
script; "script shared" = at least one other cap-hit language
uses it (e.g. Latn, Cyrl, Arab, Deva).

### Summary

| direction | script alone | script shared | total |
|---|---:|---:|---:|
| **climbed** (Δ > 0) | 16 | 17 | 33 |
| **unchanged** (Δ = 0) | 0 | 16 | 16 |
| **dropped** (Δ < 0) | 0 | 37 | 37 |
| **total** | 16 | 70 | 86 |

The pattern is very clean:

- **Every effectively-alone language climbed** (16/16). 100 % of
  their masked set is unique by definition, so the unique ranking
  surfaces them above shared-script languages that lose tokens
  to neighbors.
- **Every dropper is on a shared script** (37/37 — Latn, Cyrl,
  Arab). They lose tokens to their script neighbors when we re-
  rank by unique-only.
- **17 shared-script languages climbed anyway**. These are
  languages whose vocab carries diacritic/letter combinations
  rare in the rest of their script family — e.g. Polish & Czech
  ł/ż/ę, Hungarian ő/ű, Turkish ş/ğ/ı, Vietnamese tone marks,
  Welsh ŵ/ŷ. Their unique tokens come from those distinctive
  subsets even though they share the Latin substrate.
- The **16 unchanged** rows split into two clusters:
  - **3 dominant Latin-script languages** (English, French,
    German) keep their #1/#2/#3 positions in both rankings —
    their masked sets are so large that even after losing the
    shared-substrate tokens to neighbors they still beat the
    next contender.
  - **13 zero-masked rows at the bottom** — zero-promoted scripts
    (Lao, Khmer, Odia, Tibetan, Sinhala, Amharic, Dhivehi, Hindi,
    Middle High German) and the four `und_*` keys.

### Climbers (Δ > 0)

| iso | language | script | script-share | masked rank | unique rank | Δ |
|---|---|---|---|---:|---:|---:|
| `he` | Hebrew | `Hebr` | alone | 58 | 15 | +43 |
| `te` | Telugu | `Telu` | alone | 59 | 16 | +43 |
| `bn` | Bengali | `Beng` | alone | 61 | 18 | +43 |
| `th` | Thai | `Thai` | alone | 66 | 25 | +41 |
| `kn` | Kannada | `Knda` | alone | 67 | 26 | +41 |
| `ta` | Tamil | `Taml` | alone | 68 | 27 | +41 |
| `hy` | Armenian | `Armn` | alone | 54 | 14 | +40 |
| `ka` | Georgian | `Geor` | alone | 70 | 30 | +40 |
| `el` | Modern Greek (1453-) | `Grek` | alone | 42 | 8 | +34 |
| `ml` | Malayalam | `Mlym` | alone | 71 | 37 | +34 |
| `vi` | Vietnamese | `Latn` | shared (39 langs) | 40 | 12 | +28 |
| `gu` | Gujarati | `Gujr` | alone | 72 | 57 | +15 |
| `cy` | Welsh | `Latn` | shared (39 langs) | 48 | 34 | +14 |
| `tr` | Turkish | `Latn` | shared (39 langs) | 37 | 24 | +13 |
| `pa` | Panjabi | `Guru` | alone | 73 | 60 | +13 |
| `zh-Hans` | Mandarin Chinese | `Hani` | alone | 18 | 6 | +12 |
| `eu` | Basque | `Latn` | shared (39 langs) | 33 | 21 | +12 |
| `sw` | Swahili (individual language) | `Latn` | shared (39 langs) | 53 | 41 | +12 |
| `pl` | Polish | `Latn` | shared (39 langs) | 20 | 9 | +11 |
| `fi` | Finnish | `Latn` | shared (39 langs) | 39 | 28 | +11 |
| `my` | Burmese | `Mymr` | alone | 74 | 63 | +11 |
| `hu` | Hungarian | `Latn` | shared (39 langs) | 21 | 11 | +10 |
| `az` | North Azerbaijani | `Latn` | shared (39 langs) | 44 | 35 | +9 |
| `uz-Latn` | Northern Uzbek | `Latn` | shared (39 langs) | 47 | 38 | +9 |
| `ja` | Japanese | `Jpan` | alone | 13 | 5 | +8 |
| `sq` | Tosk Albanian | `Latn` | shared (39 langs) | 50 | 42 | +8 |
| `ko` | Korean | `Hang` | alone | 10 | 4 | +6 |
| `ceb` | Cebuano | `Latn` | shared (39 langs) | 51 | 47 | +4 |
| `lv` | Standard Latvian | `Latn` | shared (39 langs) | 55 | 51 | +4 |
| `lt` | Lithuanian | `Latn` | shared (39 langs) | 49 | 46 | +3 |
| `it` | Italian | `Latn` | shared (39 langs) | 9 | 7 | +2 |
| `is` | Icelandic | `Latn` | shared (39 langs) | 52 | 50 | +2 |
| `mn` | Halh Mongolian | `Cyrl` | shared (11 langs) | 69 | 67 | +2 |

### Droppers (Δ < 0)

| iso | language | script | script-share | masked rank | unique rank | Δ |
|---|---|---|---|---:|---:|---:|
| `ar-MA` | Moroccan Arabic | `Arab` | shared (5 langs) | 5 | 68 | -63 |
| `ar` | Standard Arabic | `Arab` | shared (5 langs) | 4 | 49 | -45 |
| `bg` | Bulgarian | `Cyrl` | shared (11 langs) | 22 | 62 | -40 |
| `mk` | Macedonian | `Cyrl` | shared (11 langs) | 27 | 66 | -39 |
| `da` | Danish | `Latn` | shared (39 langs) | 23 | 55 | -32 |
| `sr-Cyrl` | Serbian | `Cyrl` | shared (11 langs) | 25 | 56 | -31 |
| `gl` | Galician | `Latn` | shared (39 langs) | 8 | 36 | -28 |
| `bs` | Bosnian | `Latn` | shared (39 langs) | 35 | 61 | -26 |
| `hr` | Croatian | `Latn` | shared (39 langs) | 38 | 64 | -26 |
| `id` | Indonesian | `Latn` | shared (39 langs) | 31 | 53 | -22 |
| `ur` | Urdu | `Arab` | shared (5 langs) | 34 | 52 | -18 |
| `sk` | Slovak | `Latn` | shared (39 langs) | 36 | 54 | -18 |
| `uk` | Ukrainian | `Cyrl` | shared (11 langs) | 24 | 40 | -16 |
| `af` | Afrikaans | `Latn` | shared (39 langs) | 28 | 44 | -16 |
| `ms` | Standard Malay | `Latn` | shared (39 langs) | 32 | 48 | -16 |
| `nl` | Dutch | `Latn` | shared (39 langs) | 14 | 29 | -15 |
| `sl` | Slovenian | `Latn` | shared (39 langs) | 43 | 58 | -15 |
| `ne` | Nepali (individual language) | `Deva` | shared (3 langs) | 57 | 71 | -14 |
| `be` | Belarusian | `Cyrl` | shared (11 langs) | 46 | 59 | -13 |
| `mr` | Marathi | `Deva` | shared (3 langs) | 56 | 69 | -13 |
| `ca` | Catalan | `Latn` | shared (39 langs) | 11 | 23 | -12 |
| `fil` | Filipino | `Latn` | shared (39 langs) | 19 | 31 | -12 |
| `pt` | Portuguese | `Latn` | shared (39 langs) | 7 | 17 | -10 |
| `nb` | Norwegian Bokmål | `Latn` | shared (39 langs) | 29 | 39 | -10 |
| `ky` | Kirghiz | `Cyrl` | shared (11 langs) | 60 | 70 | -10 |
| `uz-Cyrl` | Northern Uzbek | `Cyrl` | shared (11 langs) | 63 | 73 | -10 |
| `ckb` | Central Kurdish | `Arab` | shared (5 langs) | 64 | 74 | -10 |
| `sv` | Swedish | `Latn` | shared (39 langs) | 26 | 33 | -7 |
| `tg` | Tajik | `Cyrl` | shared (11 langs) | 65 | 72 | -7 |
| `fa` | Persian | `Arab` | shared (5 langs) | 16 | 22 | -6 |
| `ro` | Romanian | `Latn` | shared (39 langs) | 15 | 20 | -5 |
| `es` | Spanish | `Latn` | shared (39 langs) | 6 | 10 | -4 |
| `kk` | Kazakh | `Cyrl` | shared (11 langs) | 62 | 65 | -3 |
| `la` | Latin | `Latn` | shared (39 langs) | 17 | 19 | -2 |
| `cs` | Czech | `Latn` | shared (39 langs) | 30 | 32 | -2 |
| `et` | Standard Estonian | `Latn` | shared (39 langs) | 41 | 43 | -2 |
| `ru` | Russian | `Cyrl` | shared (11 langs) | 12 | 13 | -1 |

### Unchanged (Δ = 0)

| iso | language | script | script-share | masked rank | unique rank | Δ |
|---|---|---|---|---:|---:|---:|
| `en` | English | `Latn` | shared (39 langs) | 1 | 1 | 0 |
| `fr` | French | `Latn` | shared (39 langs) | 2 | 2 | 0 |
| `de` | German | `Latn` | shared (39 langs) | 3 | 3 | 0 |
| `hi` | Hindi | `Deva` | shared (3 langs) | 45 | 45 | 0 |
| `dv` | Dhivehi | `Thaa` | shared (0 langs) | 75 | 75 | 0 |
| `unmapped` | Middle High German (ca. 1050-1500) | `Latn` | shared (39 langs) | 76 | 76 | 0 |
| `lo` | Lao | `Laoo` | shared (0 langs) | 77 | 77 | 0 |
| `or` | Odia | `Orya` | shared (0 langs) | 78 | 78 | 0 |
| `km` | Khmer | `Khmr` | shared (0 langs) | 79 | 79 | 0 |
| `unmapped` | Undetermined language | `Mong` | shared (0 langs) | 80 | 80 | 0 |
| `am` | Amharic | `Ethi` | shared (0 langs) | 81 | 81 | 0 |
| `unmapped` | Undetermined language | `Kana` | shared (0 langs) | 82 | 82 | 0 |
| `bo` | Tibetan | `Tibt` | shared (0 langs) | 83 | 83 | 0 |
| `si` | Sinhala | `Sinh` | shared (0 langs) | 84 | 84 | 0 |
| `unmapped` | Undetermined language | `Grek` | shared (1 langs) | 85 | 85 | 0 |
| `unmapped` | Undetermined language | `Cyrl` | shared (11 langs) | 86 | 86 | 0 |
