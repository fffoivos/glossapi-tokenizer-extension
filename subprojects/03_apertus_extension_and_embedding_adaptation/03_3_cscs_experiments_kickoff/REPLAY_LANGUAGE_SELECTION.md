# CPT Replay Language Selection

> **v0.7 supersedes this doc as canonical.** v0.7 §4.2 ships a 24-language
> set (Tier 1: 8, Tier 2: 11, Tier 3: 5). This 34-language set was the
> v0.5/v0.6-era proposal; it remains useful as the criterion-convergence
> rationale that underpins v0.7's narrower list (every v0.7 language is
> in this list; v0.7 trims 10 languages we'd put in Tier A/B/C —
> Hungarian, Swedish, Danish, Vietnamese, Indonesian, Croatian,
> Slovenian, Romansh, Korean, Hindi, Bengali — for a tighter focus).
> Two specific corrections to flag here that update v0.7's framing:
>
> - **v0.7 Tier 3 "preservation aspiration" framing is over-pessimistic.** Per the [§3 audit here](#3-the-recommended-34), all five Tier 3 languages (Latin, Armenian, Georgian, Albanian, Macedonian) have ≥ 1 B sampled tokens in FineWeb-2 — they have real corpus, not "near-zero base exposure." See also [`../cpt_plan_v0.7_status.md`](../cpt_plan_v0.7_status.md) (Q D2 / Q C3 audit).
> - **Replay budget**: v0.7's 70 / 30 split (not the 10–15 % non-Greek share this doc assumed under §8.5 of `old_experiments_plan.md`) materially raises the budget per language. At 10 B-token bakeoff × 30 % replay = 3 B replay tokens; under v0.7's tier weights ~75 % goes to the 8 Tier-1 languages, ~20 % to Tier-2, ~5 % to Tier-3.

*2026-05-20. Companion to
[`CURRICULUM_AND_INIT_CORPUS.md`](CURRICULUM_AND_INIT_CORPUS.md).
Picks the language set for the non-Greek replay component (10-15 %
of the CPT token budget per plan §8.5 + colleague's
`Apertus_plan.md`) using the four user-supplied criteria, with
explicit convergence analysis.*

## TL;DR — recommended set (34 languages)

Sorted into three tiers by Apertus's pretraining strength + the
criterion-satisfaction count. Replay weight per language depends on the
tier so that strongly-known Apertus languages dominate the replay (which
is the point of replay — keep what already works alive) and the
historical/cultural-connection languages contribute meaning without
muddying the optimization.

| Tier | Count | Default share of non-Greek replay budget | Logic |
|---|---:|---|---|
| **A — Apertus core multilingual (FW2-HQ)** | 20 | ~75 % | Languages Apertus is confidently good at. Replay here genuinely preserves capability. |
| **B — Strong regional / Swiss / cultural with FW2 coverage** | 11 | ~20 % | Languages with weaker Apertus coverage but high criterion convergence (Balkan neighbors, Slavic Orthodox, Hebrew, Latin, Romansh). |
| **C — Global diversity rounding** | 3 | ~5 % | Korean, Hindi, Bengali — adds global script + grammar diversity without diluting the Greek-relevant set. |

Total: **34 languages** (in your 40 ± 15 = 25–55 target band). Final per-language token weight comes from the recipe builder; defaults below.

## 1. The four criteria + which languages each picks

### Criterion 1 — Geographic neighbors of Greece

| Language | Why | Apertus coverage |
|---|---|---|
| Italian | Across the Adriatic / shared Mediterranean | FW2-HQ ✓ |
| Turkish | Direct eastern neighbor + Cyprus context | FW2-HQ ✓ |
| Bulgarian | Direct northern neighbor | FW2 |
| Albanian | Direct northwestern neighbor | FW2 |
| Macedonian | Direct northern neighbor (FYROM/North Macedonia) | FW2 |
| Romanian | Balkan + Lower Danube | FW2 |
| Serbian | Western Balkan | FW2 |
| Croatian | Balkan / Adriatic | FW2 |
| Bosnian | Western Balkan | FW2 (low) |
| Slovenian | Northwest Balkan / EU | FW2 |
| Arabic | Eastern Mediterranean | FW2-HQ ✓ |
| Hebrew | Eastern Mediterranean | FW2 |
| Maltese | Mediterranean island, classical Greek influence | FW2 (low) — **excluded as too sparse for replay** |
| Persian | Eastern frontier (Aegean / Anatolia) | FW2-HQ ✓ |

Picked into recommended set: **Italian, Turkish, Bulgarian, Albanian, Macedonian, Romanian, Serbian, Croatian, Slovenian, Arabic, Hebrew, Persian** (12). *Maltese and Bosnian dropped on coverage grounds.*

### Criterion 2 — Western Europe / Swiss-AI / Apertus bridge

| Language | Why | Apertus coverage |
|---|---|---|
| English | Universal bridge; Apertus's largest pretraining share | FW-HQ ✓ ✓ |
| German | Swiss official; Swiss-AI native | FW2-HQ ✓ |
| French | Swiss official; Swiss-AI native | FW2-HQ ✓ |
| Italian | Swiss official; also Mediterranean (criterion 1) | FW2-HQ ✓ |
| Romansh | **4th Swiss official language**; Apertus team explicitly cares for it (see `swiss-ai/apertus-pretrain-romansh` HF dataset) | very low — but **principled to include** since the Swiss-AI project explicitly tracks it |
| Spanish | EU + global Romance | FW2-HQ ✓ |
| Portuguese | EU + global Romance | FW2-HQ ✓ |
| Dutch | Benelux | FW2-HQ ✓ |
| Polish | EU east + Slavic bridge | FW2-HQ ✓ |
| Czech | EU central + Slavic | FW2-HQ ✓ |
| Hungarian | Central Europe + EU | FW2-HQ ✓ |
| Swedish | Nordic + EU | FW2-HQ ✓ |
| Danish | Nordic + EU | FW2-HQ ✓ |
| Finnish | Nordic + EU | FW2 |
| Norwegian | Nordic | FW2 (medium) |
| Catalan | Mediterranean + EU minority | FW2 (medium) — **kept on the bubble** |

Picked: **English, German, French, Italian, Romansh, Spanish, Portuguese, Dutch, Polish, Czech, Hungarian, Swedish, Danish** (13). Finnish/Norwegian/Catalan dropped to keep the list focused — already enough Nordic+EU coverage; their addition wouldn't change the multilingual posture meaningfully.

### Criterion 3 — Historical connection to Greece

Three sub-bundles:

**3a. Greco-Roman antiquity → Western scholarship**
- **Latin** — antiquity, medieval Western scholarship, Renaissance. Apertus has Latin in FineWeb-2 (low-HQ); the lm-eval-harness suite tests on Latin treebanks. **Include.**

**3b. Slavic / Orthodox / Byzantine inheritance**
- **Russian** (1 + 3 + 4) — Third Rome, Cyrillic alphabet derived from Glagolitic which was derived from Greek script. FW2-HQ ✓. **Include.**
- **Bulgarian** — Orthodox, Cyrillic born here (Saints Cyril and Methodius from Thessaloniki). Already in criterion 1. **Include.**
- **Serbian** — Orthodox, Cyrillic. Already in criterion 1. **Include.**
- **Macedonian** — Orthodox, Cyrillic. Already in criterion 1. **Include.**
- **Ukrainian** — Orthodox, Cyrillic, large Slavic. FW2. **Include.** (Adds: Belarusian dropped — overlaps too much with Russian for our purposes.)
- **Polish** — Slavic, Catholic-but-Slavic-historically. Already in criterion 2. **Include.**
- **Czech** — Slavic. Already in criterion 2. **Include.**

**3c. Eastern Mediterranean / early-Christian**
- **Hebrew** — Septuagint, Eastern Mediterranean. Already in criterion 1. **Include.**
- **Arabic** — Byzantine ↔ Caliphate scholarly translation chain; Greek scientific works survived via Arabic. Already in criterion 1. **Include.**
- **Armenian** — Caucasus + early-Christian; alphabet partly derived from Greek. FW2 low-coverage. **Include** (Tier C).
- **Georgian** — Caucasus + Orthodox; alphabet historical-Greek-adjacent. FW2 low-coverage. **Include** (Tier C).
- Coptic, Syriac, Aramaic, Church Slavonic — **excluded**: Apertus has near-zero coverage. Replay on these would be teaching, not preserving — wrong tool for this corpus mix.

Picked from criterion 3: **Latin, Russian, Bulgarian, Serbian, Macedonian, Ukrainian, Polish, Czech, Hebrew, Arabic, Armenian, Georgian** (12; overlaps with criteria 1 + 2).

### Criterion 4 — Global viewpoints (distinct from Indo-European / Mediterranean)

| Language | Script | Why | Apertus coverage |
|---|---|---|---|
| Chinese (Mandarin) | Han | Largest non-Indo-European; ideographic script | FW2-HQ ✓ |
| Japanese | Hiragana / Katakana / Kanji | Distinct script triple; agglutinative | FW2-HQ ✓ |
| Korean | Hangul | Distinct script (featural alphabet); agglutinative | FW2 |
| Vietnamese | Latin (with extensive diacritics) | Tonal Austroasiatic; reasonable Apertus coverage | FW2-HQ ✓ |
| Indonesian | Latin | Austronesian; large Muslim-world coverage | FW2-HQ ✓ |
| Hindi | Devanagari | Indo-European but tied to South Asian sphere | FW2 |
| Bengali | Bengali script | Distinct script; under-represented globally | FW2 (low) |
| Swahili | Latin | African lingua franca; Bantu typology | FW2 (low) — **excluded; very weak coverage** |
| Tamil / Telugu / Marathi | Dravidian / Indic | Distinct scripts; further South Asian coverage | FW2 (low) — **excluded; one Indic family (Hindi+Bengali) is enough** |
| Thai | Thai | Tonal, distinct script | FW2 (low) — **excluded** |

Picked: **Chinese, Japanese, Korean, Vietnamese, Indonesian, Hindi, Bengali** (7). Six of these were already picked in earlier criteria (1, 2); Korean, Hindi, Bengali are the additions specifically for criterion 4.

## 2. Convergence table — which languages hit multiple criteria

Sorted by criterion-count descending, then by Apertus coverage strength.

| Language | C1 (geo) | C2 (West) | C3 (history) | C4 (global) | # crit | Apertus |
|---|:---:|:---:|:---:|:---:|---:|---|
| Russian | (Black Sea reach) | | ✓ | ✓ | 3 | FW2-HQ |
| Arabic | ✓ | | ✓ | ✓ | 3 | FW2-HQ |
| Italian | ✓ | ✓ | ✓ | | 3 | FW2-HQ |
| Romanian | ✓ | ✓ | ✓ | | 3 | FW2 |
| Bulgarian | ✓ | | ✓ | | 2 | FW2 |
| Serbian | ✓ | | ✓ | | 2 | FW2 |
| Macedonian | ✓ | | ✓ | | 2 | FW2 |
| Croatian | ✓ | ✓ | | | 2 | FW2 |
| Slovenian | ✓ | ✓ | | | 2 | FW2 |
| Turkish | ✓ | | ✓ | | 2 | FW2-HQ |
| Hebrew | ✓ | | ✓ | | 2 | FW2 |
| Persian | ✓ | | ✓ | | 2 | FW2-HQ |
| Polish | | ✓ | ✓ | | 2 | FW2-HQ |
| Czech | | ✓ | ✓ | | 2 | FW2-HQ |
| Hungarian | (close neighbor of neighbor) | ✓ | | | 1.5 | FW2-HQ |
| Ukrainian | | | ✓ | | 1 | FW2 |
| Armenian | (close to region) | | ✓ | | 1.5 | FW2 (low) |
| Georgian | (close to region) | | ✓ | | 1.5 | FW2 (low) |
| Latin | | (Western scholarship anchor) | ✓ | | 1.5 | FW2 (low) |
| Albanian | ✓ | | | | 1 | FW2 |
| English | | ✓ | | ✓ | 2 | FW-HQ |
| German | | ✓ | | | 1 | FW2-HQ |
| French | | ✓ | | | 1 | FW2-HQ |
| Spanish | | ✓ | | ✓ | 2 | FW2-HQ |
| Portuguese | | ✓ | | ✓ | 2 | FW2-HQ |
| Dutch | | ✓ | | | 1 | FW2-HQ |
| Swedish | | ✓ | | | 1 | FW2-HQ |
| Danish | | ✓ | | | 1 | FW2-HQ |
| Romansh | | ✓ (Swiss official) | | | 1 | very low |
| Chinese | | | | ✓ | 1 | FW2-HQ |
| Japanese | | | | ✓ | 1 | FW2-HQ |
| Korean | | | | ✓ | 1 | FW2 |
| Vietnamese | | | | ✓ | 1 | FW2-HQ |
| Indonesian | | | | ✓ | 1 | FW2-HQ |
| Hindi | | | | ✓ | 1 | FW2 |
| Bengali | | | | ✓ | 1 | FW2 (low) |

**Strong convergence (≥ 2 criteria):** 14 languages — these are the *load-bearing* replay set.
**Single criterion + strong Apertus coverage:** more — these are the *coverage* set.
**Single criterion but principled (Swiss / global script):** Romansh, Chinese, Japanese — included for explicit Swiss commitment + global-script representation.

## 3. The recommended 34

Each row: HF dataset (FineWeb-2 or its slice), justification compressed.

### Tier A — Apertus core multilingual (20)

These are the FineWeb-2-HQ languages (per Apertus pretraining); Apertus is strong on all of them. They take the lion's share of replay budget because **replay is meant to preserve what Apertus already does well**.

| # | Language | ISO + script | Criteria | Justification |
|---:|---|---|---|---|
| 1 | English | eng_Latn | C2, C4 | universal anchor; Apertus's biggest share |
| 2 | German | deu_Latn | C2 | Swiss-AI primary; also for the §10 Q8a "preservation gate language" list |
| 3 | French | fra_Latn | C2 | Swiss-AI primary; gate language |
| 4 | Italian | ita_Latn | C1+C2+C3 | Mediterranean neighbor, Swiss official, Roman cultural link |
| 5 | Russian | rus_Cyrl | C1+C3+C4 | Orthodox/Byzantine heir, global, gate language |
| 6 | Spanish | spa_Latn | C2+C4 | Romance Europe + global reach |
| 7 | Portuguese | por_Latn | C2+C4 | Romance Europe + global reach |
| 8 | Dutch | nld_Latn | C2 | Benelux EU |
| 9 | Polish | pol_Latn | C2+C3 | Slavic + EU central |
| 10 | Czech | ces_Latn | C2+C3 | Slavic + EU central |
| 11 | Hungarian | hun_Latn | C2 | Greek northern arc (Pannonian Plain); Uralic family for typological diversity |
| 12 | Swedish | swe_Latn | C2 | Nordic EU |
| 13 | Danish | dan_Latn | C2 | Nordic EU |
| 14 | Turkish | tur_Latn | C1+C3 | direct neighbor; Ottoman period historical depth |
| 15 | Arabic | arb_Arab | C1+C3+C4 | Eastern Mediterranean, Byzantine-Arabic scholarly chain, global |
| 16 | Persian | pes_Arab | C1+C3 | Eastern frontier; cultural exchange via Byzantium / Sassanid |
| 17 | Chinese (Mandarin) | cmn_Hani | C4 | non-Indo-European, ideographic script, global |
| 18 | Japanese | jpn_Jpan | C4 | distinct script triple, agglutinative |
| 19 | Vietnamese | vie_Latn | C4 | tonal Austroasiatic; diacritic-heavy Latin script |
| 20 | Indonesian | ind_Latn | C4 | Austronesian; large Muslim-world weight |

### Tier B — Strong regional / Swiss / historical with FW2 coverage (11)

These are weaker in Apertus pretraining but high criterion convergence. They get smaller per-language weight than Tier A — but going to zero on them would lose the "Greece-in-its-context" framing.

| # | Language | ISO + script | Criteria | Justification |
|---:|---|---|---|---|
| 21 | Bulgarian | bul_Cyrl | C1+C3 | direct neighbor + Cyrillic-cradle + Orthodox |
| 22 | Romanian | ron_Latn | C1+C2+C3 | Balkan + EU + Romance-in-Balkans (most criterion-overlapping language) |
| 23 | Serbian | srp_Cyrl | C1+C3 | direct (modern boundary) + Orthodox |
| 24 | Croatian | hrv_Latn | C1+C2 | Adriatic neighbor + EU |
| 25 | Slovenian | slv_Latn | C1+C2 | Western Balkan + EU |
| 26 | Albanian | als_Latn | C1 | direct northwest neighbor; arguably the "most genuinely-Greek-neighbor" language outside Greek itself |
| 27 | Macedonian | mkd_Cyrl | C1+C3 | direct northern neighbor; Slavic/Orthodox; included **for cultural reach** despite naming-dispute political sensitivity |
| 28 | Ukrainian | ukr_Cyrl | C3 | Orthodox + Cyrillic Slavic; relevant for the Eastern Christian sphere |
| 29 | Hebrew | heb_Hebr | C1+C3 | Eastern Mediterranean + Septuagint historical chain |
| 30 | Latin | lat_Latn | C3 | classical antiquity + Western scholarship; **note**: Apertus's coverage is low — included for principled reasons even though replay effect will be small |
| 31 | Romansh | roh_Latn | C2 | **4th Swiss official language**; the Apertus team has an explicit `swiss-ai/apertus-pretrain-romansh` dataset, so including it signals alignment with Swiss-AI's stated commitments. Very low coverage in FW2 — token weight will be small. |

### Tier C — Global script + grammar diversity rounding (3)

Three more languages to give the model a fuller global script range without crowding Tier A.

| # | Language | ISO + script | Criteria | Justification |
|---:|---|---|---|---|
| 32 | Korean | kor_Hang | C4 | Hangul featural script — different script logic from Han or Latin |
| 33 | Hindi | hin_Deva | C4 | Devanagari + Indo-Aryan family; large South-Asian reach |
| 34 | Bengali | ben_Beng | C4 | Bengali script; second major South-Asian script family beyond Devanagari |

## 4. Languages I considered but excluded — with reasons

| Language | Reason |
|---|---|
| Bosnian | overlaps with Serbian/Croatian linguistically; adds little beyond convergence already provided |
| Maltese | direct Greek script-and-classical connection, but FW2 coverage is too sparse for meaningful replay |
| Finnish, Norwegian, Catalan, Basque | criterion 2 is already well-covered by Tier A's Nordic + Romance trio (Swedish + Danish + Spanish + Portuguese + French + Italian); adding minority/secondary languages dilutes per-language token weight |
| Coptic, Syriac, Aramaic, Old Church Slavonic | strong historical-Greek connection (criterion 3) but **near-zero Apertus coverage** — replay would teach the model, not preserve it. Wrong tool. |
| Belarusian | overlaps too much with Russian + Ukrainian; small marginal value |
| Swahili, Thai, Tamil, Telugu, Marathi | global diversity but Apertus coverage too weak to make replay meaningful at the budgets we have |
| Ancient Greek / Koine / Katharevousa | covered by the polytonic specialization layer at 153,600 vocab, not by the modern-Greek-only CPT replay |
| Esperanto, Latino sine flexione | conlang detour, no |
| Yiddish | strong cultural overlap with several Tier B languages already; minimal marginal value at our replay budget |

## 5. How this lands in the CPT mix

Following plan §8.5 and `Apertus_plan.md` §"English Anchor":

- **Greek share of CPT token budget**: 85–90 %.
- **Non-Greek replay share**: 10–15 % (start at 10 %, bump to 15 % if reasoning regressions appear on XNLI / XCOPA).
- **Within the replay 10–15 %**, allocate by tier:
  - **Tier A (20 langs)**: ~75 % of replay budget. English alone gets the largest slice (~30 % of replay total = ~3–4.5 % of the full CPT budget) since it's both the biggest Apertus pretraining share AND the universal English anchor pattern from EEVE / Krikri / colleague's plan. Remaining ~45 % of replay is split among the other 19 Tier-A languages.
  - **Tier B (11 langs)**: ~20 % of replay budget, roughly equal-weighted.
  - **Tier C (3 langs)**: ~5 % of replay budget, equal-weighted.
- **Periodicity** (per `CURRICULUM_AND_INIT_CORPUS.md` §"Cross-phase non-Greek replay"): inject replay batches periodically (every 20 steps), not as a flat per-batch mix. During Phase 3 (alien dictionary content) bump replay frequency to every 10 steps.

### Sanity check on per-language token allocation

For a 10 B-token CPT pilot:
- 10 % non-Greek replay = 1 B tokens of replay
- Tier A 75 % = 750 M tokens / 20 langs ≈ ~38 M tokens/lang (English ~300 M, others ~24 M)
- Tier B 20 % = 200 M tokens / 11 langs ≈ 18 M tokens/lang
- Tier C 5 % = 50 M tokens / 3 langs ≈ 17 M tokens/lang

At Apertus's pretraining-tokens-per-language scale (~85 M per English vocab token, ~2 M per Greek vocab token), these replay numbers are **enough to keep the trained-embedding shells alive** but won't shift any language meaningfully. That's the right shape for replay — not teaching, just maintaining.

## 6. Open knobs (decide before the corpus build fires)

1. **Romansh inclusion / weight**: principled but very low Apertus coverage. If we use it at the default ~10 M tokens, it's effectively a token to the Swiss-AI commitment more than a real preservation signal. OK as-is, but flagging.
2. **Latin inclusion / weight**: same situation as Romansh — historical/cultural meaning > preservation effect. Worth keeping for the "Greco-Roman antiquity acknowledgement" framing.
3. **Macedonian**: political sensitivity around naming. Linguistically it IS South Slavic, neighbor, Orthodox. **My recommendation**: include with the ISO label `mkd_Cyrl` and treat naming as an upstream-corpus question, not a CPT-design question.
4. **Eastern Christian classical (Coptic, Syriac, Old Church Slavonic)**: confirm we are NOT including these in replay (they'd need different treatment if we wanted them — separate specialization arm like polytonic, not replay).
5. **List size**: 34. If you want to extend toward the upper bound of your range (~55), the natural additions are **Finnish, Norwegian, Catalan, Bosnian, Belarusian, Swahili, Thai, Tamil**. None of these meaningfully change the criterion coverage; they only add tail Tier-A/B coverage.
6. **List size**: 34. If you want to compress toward the lower bound (~25), the natural cuts are **Bengali, Korean, Hindi, Vietnamese, Indonesian** (drop two of these from the global-diversity rounding) plus **Ukrainian, Slovenian, Croatian, Bosnian, Latin, Romansh** in succession. The "load-bearing 14 strong-convergence" languages should stay.
