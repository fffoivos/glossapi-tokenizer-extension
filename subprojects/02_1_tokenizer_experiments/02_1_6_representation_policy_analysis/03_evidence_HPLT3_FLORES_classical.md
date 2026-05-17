# Additional evidence — HPLT 3.0 web baseline, FLORES+ 55-list, classical-language asymmetry

Second-pass evidence extending `01_explicit_goals.md` and
`02_implicit_constraints.md`. Three threads from the user's reframing
of fairness (2026-05-17):

1. HPLT 3.0 as an independent web-baseline — does it support the
   priority-list patterns of Mistral and Apertus?
2. The FLORES+ 55-language list in detail, plus the Gini methodology.
3. Latin vs Ancient Greek asymmetry in Apertus pretraining sources.

Retrieved 2026-05-17. Quotes verbatim from cited URLs.

## 1. HPLT 3.0 — independent web-language baseline

### 1.1 The dataset

- **Source**: HPLT 3.0 paper, arXiv:2511.01066 (November 2025);
  `HPLT/HPLT3.0` HF; per-language statistics at
  https://hplt-project.org/datasets/v3.0
- **Scale**: "At 30 trillion tokens, this is likely the largest
  generally available multilingual collection of LLM pre-training data"
  — 29.81 T Gemma-3 tokens, 29.46 B documents, 198 unique
  language-script codes.
- **Pipeline**: Trafilatura 2.0 → OpenLID-v2 language ID → MinHash
  global deduplication (per-crawl for English/Russian/Chinese) → Web
  Docs Scorer quality → Turku web register labels (104 langs).
- **Crawl source**: 7.2 PB total (3.3 PB Internet Archive 2012-2020 + 57
  CommonCrawl snapshots 2014-2025).
- **No stated per-language target**: the per-language proportions fall
  out of the pipeline; HPLT does not curate to a fixed allocation.

This makes HPLT 3.0 the cleanest "what the web actually looks like
after sensible quality cleaning" comparator we have.

### 1.2 HPLT 3.0 top 35 by tokens

(Gemma-3 tokenizer, %-of-total over 29.81 T tokens.)

| Rank | Lang | Code | Docs | Tokens | % |
|---:|---|---|---:|---:|---:|
| 1 | English | eng_Latn | 18.06 B | 16,280 B | **54.61** |
| 2 | Russian | rus_Cyrl | 3.30 B | 4,403 B | **14.77** |
| 3 | Chinese (Simplified) | cmn_Hans | 2.21 B | 2,972 B | **9.97** |
| 4 | Japanese | jpn_Jpan | 0.67 B | 876 B | **2.94** |
| 5 | Spanish | spa_Latn | 0.73 B | 659 B | 2.21 |
| 6 | German | deu_Latn | 0.65 B | 609 B | 2.04 |
| 7 | French | fra_Latn | 0.60 B | 585 B | 1.96 |
| 8 | Italian | ita_Latn | 0.36 B | 335 B | 1.13 |
| 9 | Portuguese | por_Latn | 0.34 B | 319 B | 1.07 |
| 10 | Polish | pol_Latn | 0.26 B | 270 B | 0.91 |
| 11 | Dutch | nld_Latn | 0.20 B | 173 B | 0.58 |
| 12 | Persian | pes_Arab | 0.12 B | 158 B | 0.53 |
| 13 | Turkish | tur_Latn | 0.16 B | 150 B | 0.50 |
| 14 | Chinese (Traditional) | cmn_Hant | 0.11 B | 147 B | 0.49 |
| 15 | Vietnamese | vie_Latn | 0.15 B | 142 B | 0.48 |
| 16 | Indonesian | ind_Latn | 0.18 B | 142 B | 0.48 |
| 17 | Czech | ces_Latn | 0.11 B | 126 B | 0.42 |
| 18 | **Greek** | **ell_Grek** | **0.087 B** | **116 B** | **0.39** |
| 19 | Swedish | swe_Latn | 0.10 B | 112 B | 0.37 |
| 20 | Romanian | ron_Latn | 0.10 B | 103 B | 0.34 |
| 21 | Hungarian | hun_Latn | 0.075 B | 102 B | 0.34 |
| 22 | Korean | kor_Hang | 0.075 B | 98 B | 0.33 |
| 23 | Ukrainian | ukr_Cyrl | 0.080 B | 81 B | 0.27 |
| 24 | Finnish | fin_Latn | 0.050 B | 74 B | 0.25 |
| 25 | Danish | dan_Latn | 0.052 B | 63 B | 0.21 |
| 26 | Thai | tha_Thai | 0.040 B | 56 B | 0.19 |
| 27 | Norwegian Bokmål | nob_Latn | 0.036 B | 51 B | 0.17 |
| 28 | Arabic (MSA) | arb_Arab | 0.050 B | 50 B | 0.17 |
| 29 | Bulgarian | bul_Cyrl | 0.043 B | 49 B | 0.16 |
| 30 | Slovak | slk_Latn | 0.036 B | 40 B | 0.13 |
| 31 | Hebrew | heb_Hebr | 0.026 B | 37 B | 0.12 |
| 32 | Croatian | hrv_Latn | 0.031 B | 35 B | 0.12 |
| 33 | Bosnian | bos_Latn | 0.037 B | 32 B | 0.11 |
| 34 | Lithuanian | lit_Latn | 0.020 B | 29 B | 0.10 |
| 35 | Hindi | hin_Deva | 0.036 B | 27 B | **0.09** |

Caveats on script aggregation:
- **Chinese** is split into Simp + Trad: combined Chinese = 10.46 % (Han family rank 3).
- **Arabic family**: arb_Arab + ary_Arab + arz_Arab + apc_Arab +
  pes_Arab + pbt_Arab + urd_Arab + ckb_Arab = ~0.73 % combined.
- **Hindi** at rank 35 is despite Hindi's large speaker count — web
  presence is low.

### 1.3 Mistral-11 mapped to HPLT 3.0 ranks

| Mistral-11 language | HPLT 3.0 rank | HPLT tok-% |
|---|---:|---:|
| English | 1 | 54.61 |
| Chinese (Simp.) | 3 | 9.97 |
| Japanese | 4 | 2.94 |
| Spanish | 5 | 2.21 |
| German | 6 | 2.04 |
| French | 7 | 1.96 |
| Italian | 8 | 1.13 |
| Portuguese | 9 | 1.07 |
| Korean | 22 | 0.33 |
| Arabic (MSA) | 28 | 0.17 |
| Hindi | 35 | 0.09 |

**Top-10 by HPLT NOT in Mistral-11**: Russian (rank 2, 14.77 %),
Polish (rank 10, 0.91 %).

**Mistral-11 BELOW HPLT top-10**: Korean (22), Arabic-MSA (28), Hindi
(35).

### 1.4 Apertus-HQ-20 mapped to HPLT 3.0 ranks

| Apertus-HQ-20 | HPLT 3.0 rank | HPLT tok-% |
|---|---:|---:|
| Russian | 2 | 14.77 |
| Chinese | 3 | 9.97 |
| Japanese | 4 | 2.94 |
| Spanish | 5 | 2.21 |
| German | 6 | 2.04 |
| French | 7 | 1.96 |
| Italian | 8 | 1.13 |
| Portuguese | 9 | 1.07 |
| Polish | 10 | 0.91 |
| Dutch | 11 | 0.58 |
| Persian | 12 | 0.53 |
| Turkish | 13 | 0.50 |
| Vietnamese | 15 | 0.48 |
| Indonesian | 16 | 0.48 |
| Czech | 17 | 0.42 |
| **Greek** | **18** | **0.39** |
| Swedish | 19 | 0.37 |
| Hungarian | 21 | 0.34 |
| Korean (not in HQ-20) | 22 | 0.33 |
| Danish | 25 | 0.21 |
| Arabic-MSA (in HQ-20) | 28 | 0.17 |

**Apertus-HQ-20 covers HPLT ranks**: {2, 3, 4, 5, 6, 7, 8, 9, 10,
11, 12, 13, 15, 16, 17, 18, 19, 21, 25, 28}. Nearly contiguous from
rank 2 down through rank 22 with a few jumps. **Note: English (HPLT
rank 1) is NOT in HQ-20** because Apertus handles English through
separate English-only datasets (FineWeb-HQ-English, FineWeb-Edu,
DCLM-Edu); HQ-20 is the 20-language *non-English* high-resource
slice within FineWeb-2.

**HPLT top-25 missing from Apertus-HQ-20**: rank 14 Trad-Chinese
(merged with Simp), rank 20 Romanian, rank 22 Korean, rank 23
Ukrainian, rank 24 Finnish.

### 1.5 What the HPLT 3.0 numbers say about the user's hypotheses

| User's hypothesis | HPLT 3.0 evidence |
|---|---|
| Russian, Chinese huge → Mistral should include them | **Russian is rank 2** (14.77 %, larger than DE+FR+ES+IT+PT combined) and **is NOT in Mistral-11**. Chinese is rank 3 and IS in Mistral-11. |
| Hindi huge → Mistral has it as a strong language | Hindi is **rank 35** at 0.09 %. The "huge on internet" claim does not hold for Hindi via HPLT 3.0. Mistral-11 includes Hindi anyway. |
| Arabic huge → Mistral has it | Arabic-MSA is **rank 28** (0.17 %); Arabic-family combined ≈ 0.73 %. By HPLT it's mid-tier, below Polish/Dutch/Greek. Mistral-11 includes Arabic. |
| Spanish, Portuguese from Latin America | Spanish is **rank 5** (above German/French), Portuguese is **rank 9** — both organically top-10. |
| German, French as Western European bias | German is **rank 6** (2.04 %), French is **rank 7** (1.96 %) — both organically top-10. The Western-European tilt is NOT in their inclusion (they would be in any top-10 list) but in **the omission of Russian** (rank 2) from Mistral-11. |
| Italian, Polish on the threshold | Italian rank 8 (1.13 %); Polish rank 10 (0.91 %). Italian is in Mistral-11, Polish is not. |
| Korean genuinely top-of-the-web | Korean is **rank 22** at 0.33 % — below Greek (rank 18, 0.39 %). Mistral-11 includes Korean despite it being smaller than 14 other languages. |

**Net pattern**: Mistral-11 is more accurately characterised as
"Romance + Germanic Western European + East Asian + cultural-/
diplomatic-priority non-European." Specifically:
- Western European (en, fr, de, es, it, pt, nl-not-in-11 but-close)
  — fits HPLT top-10 except Italian/Portuguese which are at the 8-9
  cusp.
- East Asian (zh, ja, ko) — fits HPLT top-22; Korean is the outlier.
- Cultural-/diplomatic-priority (ar, hi) — does NOT fit HPLT top-30.
  Inclusion seems to reflect speaker count / geopolitical scope
  rather than web volume.
- **Conspicuously absent**: Russian (HPLT rank 2), the second-largest
  web language by tokens. Polish, Persian, Turkish, Indonesian,
  Vietnamese, Czech (HPLT ranks 10-17) also absent from Mistral-11
  despite being top-15 by tokens. **EPFL's FineWeb-2-HQ adds Russian
  + Polish + Dutch + Persian + Turkish + Czech + Vietnamese back to
  HQ-20** — a correction to Mistral-11's omissions.

### 1.6 Greek's position in HPLT 3.0

Greek is **rank 18** in HPLT 3.0, at 0.39 % of tokens. Above
Swedish, Romanian, Hungarian, Korean, Ukrainian, Finnish, Danish,
Arabic-MSA, Bulgarian, Hindi. **Greek's HPLT rank is higher than its
Mistral-11 standing (omitted) and approximately matches its
Apertus-HQ-20 standing (included).**

## 2. FLORES+ 55-language list and Gini methodology

### 2.1 The full 55 languages Apertus evaluated tokenizers on

Verbatim from Apertus paper Appendix I (p.93):

> "This evaluation covers a wide range of languages, including
> **Afrikaans, Albanian, Arabic, North Azerbaijani, Basque,
> Belarusian, Bengali, Bulgarian, Catalan, Chinese, Czech, Danish,
> Dutch, English, Estonian, Finnish, French, Galician, Georgian,
> German, Greek, Gujarati, Hebrew, Hindi, Hungarian, Indonesian,
> Italian, Japanese, Korean, Latvian, Malay, Malayalam, Marathi,
> Macedonian, Norwegian Bokmål, Persian (Farsi), Polish, Portuguese,
> Romanian, Russian, Slovak, Southern Sotho, Spanish, Swahili,
> Swedish, Tamil, Tajik, Telugu, Thai, Turkish, Ukrainian, Urdu,
> Vietnamese, Welsh, and Yoruba.**"

Counts: 55 languages.

**Greek IS included.** **Latin and Ancient Greek are NOT in FLORES+
at all** — the benchmark derives from Wikinews / Wikijunior /
Wikivoyage human translations; classical languages are not present.

### 2.2 The Gini definition Apertus uses

From Apertus paper Appendix I (p.92-93):

> "**Tokenizer Fairness Gini Coefficient.** We adapt the Gini
> coefficient (commonly used to measure inequality in economics) to
> quantify fairness across languages (Meister, 2025). Let
> L = l₁, l₂, …, lₙ be the set of languages, and let
> c₁ ≤ c₂ ≤ ⋯ ≤ cₙ denote their tokenization costs under T. Here,
> cost is defined as the average number of tokens required to encode
> one normalization unit (e.g., a byte, word, or line); **for
> parallel corpora, cost per line is often used to control for
> differences in character byte lengths across scripts**. The Gini
> coefficient is given by:
> Gini(T) = (1/n)[n + 1 − 2 Σ(n+1−i)cᵢ / Σ cᵢ].
> Values range from 0 (perfect equality) to 1 (maximum inequality).
> Lower values indicate more equitable compression across languages,
> while higher values reveal systematic bias that favors certain
> languages."

So the metric is:
- **Unit**: per-language cost = tokens-per-line on the FLORES+ dev
  parallel corpus (footnote 57: equivalent to fertility, inverse of
  compression rate).
- **Coverage**: 55 languages.
- **Optimization**: minimize Gini ⇒ minimize inequality across the 55.

### 2.3 Comparator tokenizers

Apertus §2.2:

> "We based our choice on a comparison of the tokenizers of several
> large language models (e.g., Llama-3.1, Mistral-Nemo, Qwen-2.5, and
> Gemma-2) using four intrinsic evaluation metrics: fertility rate,
> compression ratio, vocabulary utilization, and Gini coefficient
> (Foroutan et al., 2025a)."

> "Mistral-Nemo achieves the lowest Gini coefficient, indicating more
> equitable tokenization costs across languages. […] we select
> Mistral-Nemo as the preferred tokenizer because it is fairer across
> languages and uses a smaller vocabulary (128k vs. 256k)."

**No numeric Gini values appear in the paper text** — they are only
in Figure 1 (a visual). The paper's reasoning is "Mistral-Nemo
lowest, picked it."

### 2.4 Foroutan et al. 2025a — origin of the metric

**Citation**: "Parity-aware byte-pair encoding: Improving
cross-lingual fairness in tokenization", arXiv:2508.04796 (Foroutan,
Meister, Paul, Niklaus, Ahmadi, Bosselut, Sennrich; 2025).

What Foroutan 2025a actually proposes (Table 1, 128k vocab, 30
unbalanced languages):

| BPE variant | Gini | Compression rate |
|---|---:|---:|
| Classical BPE | 0.064 | 0.0303 |
| Parity-aware BPE | **0.011** | 0.0300 |
| Parity-aware hybrid | 0.018 | 0.0303 |

So a Parity-aware variant achieves **~6× lower Gini** at negligible
compression cost. **Apertus did not use Parity-aware BPE**; they used
Mistral-Nemo's Classical BPE and selected it on Gini comparison.
The Apertus paper does not mention Parity-aware BPE despite citing
the paper that introduces it.

This means Apertus's tokenizer choice is the *best off-the-shelf
classical BPE on Gini fairness*, not the best possible tokenizer on
that metric. A re-derivation of Apertus that used Foroutan's
Parity-aware variant would likely produce a flatter per-language
allocation than Mistral-Nemo did.

### 2.5 Does the Gini metric address per-language fertility?

The Gini measures **inequality of per-language fertility costs** on
the 55-language FLORES+ dev set. A low Gini means the 55 fertility
costs are clustered close together; a high Gini means some languages
have much higher fertility than others.

This is an **aggregate** fairness metric: it does not say "Greek's
fertility must be ≤ X." It only says "the spread across the 55
languages should be small."

**Practical implication**: a tokenizer with Gini = 0.05 could have
Greek's fertility be 50 % above the median if some other language
(e.g., Telugu, Tamil, Yoruba) is 50 % above the median in the same
direction — and the Gini would still look fair *in aggregate*.

The Gini-on-FLORES+ optimization Apertus performed therefore does
**not** guarantee Greek-specific fertility parity, only aggregate
55-language parity.

### 2.6 How Greek-specific is Apertus's Gini optimization

The 55 languages each contribute one cost cᵢ to the Gini calculation.
Greek's cost is one of 55 inputs. If we re-derive how much Greek's
specific cost moves the Gini value, the marginal sensitivity is
~1/55 = 1.8 % per language. Bringing Greek's fertility from 2.41
to 1.5 (a ~38 % reduction in cost) would lower the Gini by roughly
∂Gini / ∂cᵢ * (1.5 − 2.41) — a small but measurable improvement.

The Apertus team did not run a Greek-specific sensitivity analysis
on its Gini choice as far as the paper documents.

## 3. Latin and Ancient Greek presence in Apertus pretraining

### 3.1 FineWeb-2 v2.0.1 (verified from the per-language CSV)

- **Latin (lat_Latn)**: 1,473,541 documents / 714,764,848 words / 3.86
  GB UTF-8.
- **Ancient Greek (grc_Grek)**: 28,539 documents / 33,850,484 words /
  340.8 MB UTF-8.

**Latin is ~50× larger than Ancient Greek** in FineWeb-2 by
documents (or ~20× by words). Both are subsampled at random for
non-HQ-20 languages per Apertus paper §G.

### 3.2 Apertus's tokenizer footprint for Latin and Ancient Greek

From `02_2_2_vocab_lang_attribution/summary.tsv`:

| Lang | PMI tokens | Mass cap % | Notes |
|---|---:|---:|---|
| **Latin (lat_Latn)** | **2,768** | 49.7 % | Cap-hit key; >1B Apertus-tokens observed in vocab-attribution run |
| **Ancient Greek (grc_Grek)** | not in cap-hit set | — | Below 1 B Apertus-token marginal floor — not in the 87 keys |
| Modern Greek (ell_Grek) | 1,479 | 86.9 % | Cap-hit key |

**Latin is allocated MORE vocab slots than Modern Greek** (2,768 vs
1,479) in Apertus's vocab. The ratio is 1.87× in Latin's favour.
This is true despite:
- Latin being a "dead" language with no native speakers,
- Latin not being on Mistral-strong-11, in HPLT 3.0, in FLORES+, in
  HQ-20, in EuroParl, in ParaDocs, in OCR-5, or in any toxicity-9.
- Modern Greek being in HQ-20 + FLORES+ + EuroParl + clean-wikipedia +
  HPLT rank 18.

The most plausible explanation: **Latin is heavily represented in
Mistral's tokenizer training data** (likely via Wikipedia, classics
corpora, Vatican / Church Latin web content). Apertus inherits this
allocation as-is.

**Ancient Greek** is not even in the cap-hit set — i.e. Apertus's
vocab-attribution pass didn't accumulate 1 B Apertus-tokens of grc
content. This is consistent with grc being ~50× smaller than Latin
in FineWeb-2. Ancient Greek has essentially **no dedicated tokens in
Apertus's vocab**.

### 3.3 Clean-Wikipedia (per-shard listing from HF tree)

From the HF tree listing of `HuggingFaceFW/clean-wikipedia`, the
parquet directory structure:

- **`la/`** (Latin Wikipedia) — present, exact shard count not pulled.
- **`grc/`** (Ancient Greek Wikipedia) — present at the shard level.

Both have configs. Per-language token counts not on the card.

### 3.4 Other Apertus sources

| Source | Latin | Ancient Greek |
|---|---|---|
| FineWeb-2 | 1.47 M docs (random sample) | 28.5 K docs (random sample) |
| FineWeb-2-HQ | not in HQ-20 | not in HQ-20 |
| Clean-Wikipedia | shard present | shard present |
| Institutional Books 1.0 | post-1900 cutoff limits classical content; OCR processed only for eng/deu/fra/ita/spa | post-1900 cutoff limits ancient Greek; not OCR processed |
| Gutenberg V1/V2 (probe sets) | Apertus subset language mix unspecified; PG itself has Latin and ancient Greek collections | same |
| EuroParl | not applicable | not applicable |
| HPLT 3.0 (external comparator) | not in 198 codes | not in 198 codes |
| FLORES+ | not in 55 langs | not in 55 langs |

### 3.5 The asymmetry the user noted

User: "Apertus includes ancient Latin but not ancient Greek."

**More precise version**: Apertus includes both nominally (via
FineWeb-2 + clean-wikipedia), but Latin is materially represented
(2,768 vocab tokens, 1.47 M FW2 docs) while Ancient Greek is
trace-only (not in vocab-attribution cap-hit set, 28.5 K FW2 docs).

The asymmetry is **upstream of Apertus's policy** — it comes from:
- Mistral's tokenizer training mix (Latin had enough volume to
  generate dedicated merges; ancient Greek didn't).
- FineWeb-2's crawl-frequency outcome (Latin is ~50× larger).
- The web's actual content distribution: Latin-language religious /
  legal / academic content has substantial modern web presence; ancient
  Greek content is much rarer and overwhelmingly behind paywalls.

The polytonic-extension subproject
[`../02_1_polytonic_greek_extension/`](../02_1_polytonic_greek_extension/)
addresses this gap directly by training new BPE merges on a curated
ancient/polytonic Greek corpus (18,726 dedup'd rows, ~250 MiB from
First1KGreek + Perseus + GOARCH + filtered Wikisource + Scholarios).
That arm is parallel to and independent of the C3 modern-Greek
cutoff decision.

### 3.6 Implication for the C3 modern-Greek decision

The Latin-vs-Ancient-Greek asymmetry is **not directly relevant** to
the C3 cutoff for modern Greek, because:
- C3 targets modern Greek (`modern_greek_eval` is the primary slice).
- Ancient/polytonic Greek is handled by the parallel
  `02_1_polytonic_greek_extension` arm.
- Latin's 2,768 vocab tokens reflect Mistral's tokenizer-training
  data exposure, not Apertus's view of Latin's priority.

But it **is** relevant to fairness as the user reframed it: Apertus
inherits a vocab where Latin (no speakers, no FLORES+, no HQ-20) gets
~2× more tokens than Modern Greek (12 M native speakers, in HQ-20,
HPLT rank 18, EU official). This is the kind of inheritance-vs-
intent gap the implicit-policy analysis surfaces.

## 4. Summary of additional evidence

Three new evidence points to feed into the budget decision:

1. **HPLT 3.0 places Greek at rank 18 (0.39 %)** of measured web
   tokens after sensible quality cleaning. Above Korean, Ukrainian,
   Finnish, Danish, Arabic-MSA, Bulgarian, Hindi. Approximately
   matches Greek's HQ-20 inclusion and is higher than Mistral-11
   would suggest (Mistral omits Russian rank 2 too; the Western
   European tilt is in the **omissions**, not the inclusions).

2. **Apertus's Gini-on-FLORES+ is aggregate, not per-language.**
   The 55-language Gini does NOT obligate any specific language's
   fertility. Greek-specific fairness is not addressed by Apertus's
   chosen metric. Foroutan et al. 2025a proposes a Parity-aware BPE
   variant that achieves ~6× lower Gini, but Apertus uses
   off-the-shelf classical BPE and only compares Gini values across
   pre-built tokenizers.

3. **Latin gets ~2× more Apertus vocab tokens than Modern Greek**
   (2,768 vs 1,479), inherited from Mistral, despite Latin being
   absent from every Apertus-stated priority list (HQ-20, FLORES+,
   HPLT 3.0, EuroParl, OCR-5, toxicity-9, Mistral-strong-11).
   Ancient Greek by contrast is below the 1 B-token cap-hit floor.
   The asymmetry is inheritance-from-Mistral, not Apertus policy.

These extend `01_explicit_goals.md` and `02_implicit_constraints.md`
with the threads the user reframed on 2026-05-17.
