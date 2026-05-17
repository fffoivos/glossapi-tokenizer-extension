# Additional evidence — speaker-count hypothesis test

Tests whether Mistral-11 / Apertus-HQ-20 selection follows speaker
count (L1 or total) rather than web footprint. Source: Wikipedia
aggregator pages citing Ethnologue 2026; primary literature search.
Retrieved 2026-05-17.

## 1. L1 (native) speaker ranking — top 30, with Mistral-11 / HQ+ membership

| Rank | Language | L1 (M) | Mistral-11 | HQ+ (added by Apertus) | Neither |
|---:|---|---:|:---:|:---:|:---:|
| 1 | Mandarin Chinese | 988 | ✓ | | |
| 2 | Spanish | 487 | ✓ | | |
| 3 | English | 372 | ✓ | | |
| 4 | Hindi | 347 | ✓ | | |
| 5 | Portuguese | 252 | ✓ | | |
| **6** | **Bengali** | **232** | | | **✗** |
| 7 | Russian | 133 | | ✓ | |
| 8 | Japanese | 124 | ✓ | | |
| **9** | **W. Punjabi** | **90** | | | **✗** |
| 10 | Turkish | 86 | | ✓ | |
| 11 | Vietnamese | 86 | | ✓ | |
| **12** | **Yue Chinese** | **85** | | | **✗** |
| 13 | Egyptian Arabic | 83 | (under "Arabic") | | |
| **14** | **Wu Chinese** | **83** | | | **✗** |
| **15** | **Marathi** | **83** | | | **✗** |
| **16** | **Telugu** | **83** | | | **✗** |
| 17 | Korean | 82 | ✓ | | |
| **18** | **Tamil** | **79** | | | **✗** |
| **19** | **Urdu** | **78** | | | **✗** |
| 20 | Indonesian | 78 | | ✓ | |
| 21 | Standard German | 76 | ✓ | | |
| 22 | French | 76 | ✓ | | |
| **23** | **Javanese** | **69** | | | **✗** |
| 24 | Iranian Persian | 65 | | ✓ | |
| 25 | Italian | 60 | ✓ | | |
| **26** | **Hausa** | **58** | | | **✗** |
| **27** | **Gujarati** | **58** | | | **✗** |
| **29** | **Bhojpuri** | **53** | | | **✗** |
| ~30+ | Polish | ~43 | | ✓ | |
| ~70+ | **Greek** | **~13** | | **✓** | |
| ~75+ | Swedish | ~11 | | ✓ | |
| ~75+ | Czech | ~10.6 | | ✓ | |
| ~80+ | Hungarian | ~9 | | ✓ | |
| ~110+ | Danish | ~5.5 | | ✓ | |

Mistral-11 covers L1 ranks {1, 2, 3, 4, 5, 8, 13, 17, 21, 22, 25} — i.e. **misses ranks 6, 9, 12, 14, 15, 16, 18, 19, 20, 23**. Top-by-L1 doesn't fit Mistral-11.

HQ-20 reaches down past L1 rank ~110 (Danish ~5.5 M) to include Greek (~13 M) / Danish (~5.5 M) / Hungarian (~9 M) / Czech (~10.6 M) / Swedish (~11 M), while skipping Bengali (6) / Marathi (15) / Telugu (16) / Tamil (18) / Urdu (19) — each individually larger than every small-EU language in HQ-20. Top-by-L1 doesn't fit HQ-20 either.

## 2. Total (L1 + L2) speaker ranking — top 35

| Rank | Language | Total (M) | L1 (M) | L2 (M) | M11 | HQ+ | None |
|---:|---|---:|---:|---:|:---:|:---:|:---:|
| 1 | English | 1493 | 372 | 1121 | ✓ | | |
| 2 | Mandarin | 1183 | 988 | 194 | ✓ | | |
| 3 | Hindi | 611 | 347 | 264 | ✓ | | |
| 4 | Spanish | 561 | 487 | 75 | ✓ | | |
| 5 | MSA Arabic | 335 | 0 | 335 | ✓ | | |
| 6 | French | 334 | 75 | 258 | ✓ | | |
| **7** | **Bengali** | **274** | 234 | 43 | | | **✗** |
| 8 | Portuguese | 269 | 252 | 18 | ✓ | | |
| 9 | Indonesian | 255 | 78 | 177 | | ✓ | |
| **10** | **Urdu** | **246** | 78 | 168 | | | **✗** |
| 11 | Russian | 210 | 133 | 77 | | ✓ | |
| 12 | German | 133 | 76 | 57 | ✓ | | |
| 13 | Japanese | 126 | 124 | 2 | ✓ | | |
| **14** | **Nigerian Pidgin** | 121 | 5 | 116 | | | **✗** |
| **16** | **Marathi** | 99 | 83 | 16 | | | **✗** |
| 17 | Vietnamese | 97 | 86 | 11 | | ✓ | |
| **18** | **Telugu** | 96 | 83 | 13 | | | **✗** |
| **19** | **Swahili** | 95 | 4 | 91 | | | **✗** |
| **20** | **Hausa** | 94 | 58 | 36 | | | **✗** |
| 21 | Turkish | 94 | 86 | 7 | | ✓ | |
| **22** | **W. Punjabi** | 90 | — | — | | | **✗** |
| **23** | **Tagalog** | 87 | 33 | 54 | | | **✗** |
| **24** | **Tamil** | 86 | 79 | 8 | | | **✗** |
| 27 | Iranian Persian | 82 | 65 | 17 | | ✓ | |
| 28 | Korean | 82 | 82 | 1 | ✓ | | |
| **29** | **Amharic** | 78 | 39 | 39 | | | **✗** |
| **30** | **Thai** | 71 | 27 | 44 | | | **✗** |
| 32 | Italian | 66 | 60 | 6 | ✓ | | |
| **37** | **Yoruba** | 53 | 48 | 5 | | | **✗** |
| >38 | Polish | ~45 | ~43 | ~2 | | ✓ | |
| >38 | Dutch | ~25 | ~25 | low | | ✓ | |
| >38 | Hungarian | ~13 | ~9 | low | | ✓ | |
| >38 | Greek | ~13.5 | ~13 | low | | ✓ | |
| >38 | Czech | ~10.7 | ~10.6 | low | | ✓ | |
| >38 | Swedish | ~13 | ~11 | low | | ✓ | |
| >38 | Danish | ~6 | ~5.5 | low | | ✓ | |

Mistral-11 covers total-speaker ranks {1, 2, 3, 4, 5, 6, 8, 12, 13, 28, 32} — i.e. misses Bengali (7), Indonesian (9), Urdu (10), Russian (11), Nigerian Pidgin (14), Marathi (16), Vietnamese (17), Telugu (18), Swahili (19), Hausa (20), Turkish (21), W. Punjabi (22), Tagalog (23), Tamil (24), Amharic (29), Thai (30), Yoruba (37). **Top-by-total doesn't fit Mistral-11.**

HQ-20 picks up Indonesian, Russian, Vietnamese, Turkish, Persian from this region of the table but skips Bengali, Urdu, Marathi, Telugu, Swahili, Hausa, Tagalog, Tamil, Amharic, Thai, Yoruba — every one of which is larger by total speakers than the small-EU HQ-20 languages. **Top-by-total doesn't fit HQ-20 either.**

## 3. Specific test cases the user named

- **Bengali**: 232 M L1 / 274 M total. Top-7 by either measure. Not in either list. The single biggest gap in any "top-by-speakers" hypothesis.
- **Western Punjabi**: 90 M L1. Top-10 L1. Not in either list.
- **Marathi**: 83 M L1 / 99 M total. Rank 15-16. Not in either list.
- **Telugu**: 83 M L1 / 96 M total. Rank 16-18. Not in either list.
- **Tamil**: 79 M L1 / 86 M total. Rank 18-24. Not in either list.
- **Urdu**: 78 M L1 / **246 M total** (rank 10). Not in either list — despite sharing colloquial vocab and grammar with Hindi (which is in Mistral-11). Apertus / Mistral never state "Hindi covers Urdu," so this is unexplained.
- **Vietnamese**: 86 M L1 / 97 M total. In HQ-20 (added by EPFL).
- **Indonesian**: 78 M L1 / 255 M total (rank 9). In HQ-20.
- **Swahili**: 4 M L1 / **95 M total** (rank 19). Largest L2/lingua-franca in sub-Saharan Africa. **Not** in HQ-20.
- **Greek**: 13 M L1 / 13.5 M total. Below ~70 languages by L1 alone. **In** HQ-20.
- **Danish (5.5 M L1)**, **Hungarian (9 M)**, **Czech (10.6 M)**, **Swedish (11 M)** — all in HQ-20, all below ~100 other languages by L1 speakers.

## 4. Do any primary sources cite speaker count as a selection rationale?

Sources checked: Apertus paper (arXiv:2509.14233), FineWeb-2-HQ paper (Messmer et al., arXiv:2502.10361), FineWeb-2 paper (Penedo et al., arXiv:2506.20920), Mistral Nemo blog.

- **Apertus paper §4.1.3 / Appendix J.1** — single speaker reference: *"Romansh — Switzerland's fourth national language with approximately 60,000 speakers"*. Used to justify **post-training Romansh data inclusion**, not the pretraining-language selection.
- **FineWeb-2-HQ paper** — no mention of "speakers", "L1", "native speakers", "population", or "language community" anywhere. Stated stopping criterion is *"the number of documents drops quickly."*
- **FineWeb-2 paper** — single mention: *"This state of affairs leaves the majority of the world's population (speaking over 7,000 languages) unable to interact with state-of-the-art LLMs in their native tongue"* (introductory motivation, not selection criterion).
- **Mistral Nemo blog** — no mention of speaker count anywhere. Stated rationale is tokenization compression efficiency.

**No primary source explicitly cites speaker count as a selection rationale for either Mistral-11 or HQ-20.** The closest is the Apertus paper's Romansh footnote — used to justify *inclusion of an underserved language*, not the *choice of high-resource priority languages*.

## 5. What pattern fits the actual lists

Three hypotheses tested across this analysis (HPLT 3.0 evidence + speaker counts evidence):

| Hypothesis | Mistral-11 fit | HQ-20 fit | Bengali / Urdu / Marathi / Tamil missing? | Danish / Greek / Czech included? |
|---|---|---|---|---|
| Top-by-web (HPLT 3.0) | Mostly fits top-9; explains Russian omission as Western tilt | Fits top-22 nearly contiguously | Yes — they're HPLT rank 35 / not listed / etc. | Yes — they're HPLT rank 25 / 18 / 17 |
| Top-by-L1 speakers | Doesn't fit | Doesn't fit | These are L1 rank 6-19; should be included | These are L1 rank ~70+; should NOT be included |
| Top-by-total speakers | Doesn't fit | Doesn't fit | These are total rank 7-24; should be included | These are below total rank 38; should NOT be included |

**The pattern that fits**: top-by-web + Western institutional priority + token-South-Asian/Middle-East representation.

- **Top-by-web** explains: English, French, German, Spanish, Italian, Portuguese, Polish, Dutch, Russian (added by HQ-20), Czech, Turkish, Indonesian, Vietnamese, Persian, Chinese, Japanese — all HPLT top-22.
- **Western/EPFL institutional priority** explains: Greek, Hungarian, Swedish, Danish, Czech — small EU languages well below their speaker rank, well above their FW2 doc rank only because of EU institutional document availability.
- **Token-S-Asian/MidEast** explains: Hindi and Arabic — included for "global coverage" despite having less web footprint than many excluded languages. Korean is the East-Asian token in Mistral-11.

Speakers are not the selection criterion. Web + institutional + token-region best explains the observed lists.

## 6. Greek under the speaker-count lens

Greek is rank ~70 by L1 (~13 M), below ~70 other languages including every language in the top-30-by-L1. Yet Greek is in HQ-20 and Apertus's named multilingual scope.

If selection were strictly by speaker count, Greek would not be in HQ-20. Its inclusion is **institutional** (EU official, well-resourced academic/government text, classical-language pedigree giving it disproportionate Wikipedia coverage) rather than population-based.

This cuts both ways for the fairness argument:

- **For a generous Greek budget**: Greek is already a "punching above its weight" beneficiary of institutional selection. The HQ-20 inclusion is itself a non-speaker-based affordance. Asking for more is consistent with the policy actually in use.
- **Against a generous Greek budget**: Greek is already getting a vocab share (1.13 %) several times its share of world speakers (~0.16 % of L1 globally). A "fair share by speakers" anchor would give Greek roughly **(0.16 / 100) × 131,072 = ~210 tokens**, far below the current 1,479 base. Even at L2-inflated total-speaker share, Greek would warrant maybe ~270 tokens. **By speaker share, current Apertus already over-allocates Greek by ~5-7×.**

The speaker-count frame is the only frame that suggests Greek is *over-allocated* in the current Apertus vocab.

## Sources

- [Wikipedia: List of languages by number of native speakers](https://en.wikipedia.org/wiki/List_of_languages_by_number_of_native_speakers)
- [Wikipedia: List of languages by total number of speakers](https://en.wikipedia.org/wiki/List_of_languages_by_total_number_of_speakers)
  (both cite Ethnologue 29th edition, 2026)
- [Apertus paper, arXiv:2509.14233](https://arxiv.org/abs/2509.14233) §3.2.2, §4.1.3, Appendix J.1
- [FineWeb-2-HQ paper, arXiv:2502.10361](https://arxiv.org/abs/2502.10361)
- [FineWeb-2 paper, arXiv:2506.20920](https://arxiv.org/abs/2506.20920)
- [Mistral Nemo blog](https://mistral.ai/news/mistral-nemo)
