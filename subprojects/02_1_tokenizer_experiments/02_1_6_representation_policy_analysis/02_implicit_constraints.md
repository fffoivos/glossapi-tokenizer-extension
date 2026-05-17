# Phase 2 — Implicit constraints

For every per-language behavior visible in Apertus's vocab + data
artifacts, identify the structural constraint producing it. Distinguish
"policy chose this" from "data / web / tokenizer-math forced this."

## 1. The web baseline is non-negotiable

Every web-derived dataset inherits the web's per-language share, before
any policy choice. W3Techs (2026-05-17):

| Lang | % of web |
|---|---:|
| English | **49.7 %** |
| All other top-20 combined | ~40 % |
| Greek | 0.5 % |

**Constraint**: any LLM trained from web data alone, with proportional
sampling, will be ~50 % English by token mass. Apertus chose web as
its primary substrate (FineWeb-Edu, FineWeb-HQ, DCLM-Edu, FineWeb-2),
so this share is baseline.

**Consequence for the vocab**: BPE merges are frequency-driven. An
English-heavy training corpus generates more English merges. Mistral's
tekken was trained on Mistral's undisclosed data; whatever per-language
mix Mistral used became the implicit allocation in the 131,072 vocab.

## 2. English-only-by-construction datasets force English dominance

Six of Apertus's source datasets contain **zero Greek** by construction
(see `01_explicit_goals.md` §D.1). These datasets are not selected
because of an English-preference policy — they exist because the field
has English-only pipelines that don't have multilingual equivalents.

| Dataset | Stages used (8B) | Token pool (B) |
|---|---|---:|
| FineWeb-Edu Score-2 | 1 | 4,815 |
| FineWeb-HQ English | 2-5 | 4,064 |
| DCLM-Edu | 4-5 | 1,619 |
| FineMath / InfiMM-WebMath | 1-5 | ~80 |
| MegaMath | 3-5 | ~290 |
| StarCoderData | 1-5 | ~235 |
| CommonPile/Stack-v2-Edu | 5 | 68 |
| Gutenberg V1/V2 | 1-5 | ~2.5 |

Sum of available English-only pool ≈ **11 TB tokens** across all
stages. Even if each is consumed at a fraction of pool, English-only
datasets are structurally several times the multilingual pool size
(FW2 + FW2-HQ ≈ 3.5 TB per stage in Apertus's mix).

**Constraint**: Apertus's per-language pretraining-token share cannot
exceed ~40 % multilingual under this dataset mix. The 40 % is not a
policy choice — it's "we use what exists, and what exists is
English-heavy." The Apertus apps page actually advertises this as a
feature: "40 % non-English."

**Consequence for Greek**: Even if FW2-HQ-Greek were retained at
100 %, Greek's pretraining-token share would be capped at
(FW2-HQ-Greek / total) which is well below 1 %. The measured 0.023 %
follows from Apertus's specific `p × 0.95` × stage-weighting on top of
that ceiling — but the ceiling itself is set by which datasets exist.

## 3. Filter coverage gaps act as silent policy

### 3.1 Toxicity filter — Greek-favorable accident

- **Cover set**: 9 languages (en, zh, fr, de, it, nl, pl, es, pt).
- **Greek**: not covered.
- **Effect**: 5 % top-toxicity haircut applied per language for the 9
  covered. Greek text incurs no such haircut.
- **Necessity / accident**: Per the toxicity classifier paper, training
  data was available for these 9. Greek's exclusion was not
  policy — Apertus did not say "Greek doesn't need toxicity filtering"
  — it's a classifier-availability gap.
- **Direction of bias**: *increases* Greek's representation slightly
  (no top-5 % haircut). Net positive for Greek vocab utilization.

### 3.2 OCR post-processing (Institutional Books) — Greek-unfavorable accident

- **Cover set**: 5 languages for OCR post-processing (eng, deu, fra, ita, spa).
- **Greek**: present in the dataset (within 254 volume-level languages)
  but not OCR-post-processed.
- **Effect**: Greek long-context volumes carry raw OCR noise, lowering
  their effective contribution to the long-context phase.
- **Necessity / accident**: Resource-cost gap, not a stated policy.

### 3.3 ParaDocs / Gutenberg / Mistral-strong-11 / Apertus-priority-list

- All four lists name explicit priority languages.
- Greek is in **none** of them.
- These are stated commitments by upstream parties (Mistral, JHU,
  Apertus authors) that Greek doesn't get a per-language affordance in.
- **Direction**: silent policy that holds Greek back from any
  language-specific tooling investment.

### 3.4 Net filter-gap effect on Greek

Greek lives in a regime where **none of the per-language tooling is
calibrated for it**: no toxicity head, no OCR post-processor, no
named "particularly strong" claim, no ParaDocs pairs, no Gutenberg
probe. The only language-specific tooling Greek gets is:

- inclusion in FW2-HQ HQ-20 (quality classifier head exists)
- inclusion in EuroParl bitexts
- inclusion in Clean-Wikipedia
- a per-language MLP quality head in FW2-HQ

Each of these is a **structural inclusion** (Greek fits the criterion)
rather than a **named commitment** (someone said "we will support
Greek"). The distinction matters for the rational-policy split.

## 4. The HQ-20 selection criterion is not stated — but the data reveals it

Messmer et al. 2025 says "documents drop quickly" as the only
rationale. Inspecting which languages made the cut vs which didn't:

| Rank by FW2 docs | Lang | HQ-20? | Hypothesised reason |
|---:|---|:---:|---|
| 1 | Russian | ✓ | top doc count |
| 2 | Chinese | ✓ | top doc count |
| 3 | German | ✓ | top doc count |
| 4 | Spanish | ✓ | top doc count |
| 5 | Japanese | ✓ | top doc count |
| 6 | French | ✓ | top doc count |
| 7 | Italian | ✓ | top doc count |
| 8 | Portuguese | ✓ | top doc count |
| 9 | Polish | ✓ | top doc count |
| 10 | Dutch | ✓ | top doc count |
| 11 | Indonesian | ✓ | top doc count |
| 12 | Turkish | ✓ | top doc count |
| 13 | Czech | ✓ | top doc count |
| **14** | **Korean** | **✗** | **unexplained** |
| 15 | Arabic | ✓ | top doc count, skipped Korean |
| 16 | Romanian | ✗ | unexplained |
| 17 | Persian | ✓ | top doc count |
| 18 | Ukrainian | ✗ | unexplained |
| 19 | Hungarian | ✓ | top doc count |
| 20 | Swedish | ✓ | top doc count |
| **21** | **Greek** | **✓** | **top doc count among kept** |
| 22 | Danish | ✓ | top doc count |
| **23** | **Vietnamese** | **✓** | **explicit inclusion despite lower rank** |

**Pattern**: HQ-20 is *almost* top-23-by-docs, but skips Korean (rank
14), Romanian (rank 16), Ukrainian (rank 18). Vietnamese (rank 23)
made it despite being below the skipped three.

**No primary source explains the skipping.** Plausible explanations
the literature does not state:

- **EuroLLM alignment**: Vietnamese is in EuroLLM's 24+11 covered
  languages; Korean / Romanian / Ukrainian are not. EuroLLM was a
  contemporary EU-funded project that the FineWeb-2-HQ team
  presumably coordinated with. (Not stated.)
- **Quality-classifier training-data availability**: Messmer's MLP
  heads use MMLU + Aya + OpenAssistant + Include-Base-44 as positive
  examples. Aya covers Vietnamese; Include-Base-44 has a 44-language
  positive set whose composition isn't fully enumerated in the paper.
  Coverage of Korean in that union is unstated. (Not stated as
  rationale.)
- **Scripts-already-covered**: Korean (Hangul) and Ukrainian (Cyrillic
  near-duplicate of Russian) might have been deprioritised because
  their script is already covered by Japanese / Chinese / Russian
  through XLM-RoBERTa. (Not stated.)

**Net**: HQ-20 is **mostly a doc-count cliff** with three named-but-
unexplained exclusions and one named-but-unexplained inclusion. Greek
sits comfortably in HQ-20 by the doc-cliff criterion alone — its
inclusion is **not borderline at the HQ-20 level**.

## 5. BPE merge order rewards frequency — script-isolation breaks sharing

Apertus inherited tekken's vocab. Tekken's BPE training generated
merges from Mistral's (undisclosed) data. The vocab's per-language
shape reflects two competing forces:

### 5.1 Latin-script merges share

Every Latin-script language can use most other Latin-script merges.
PMI overlap inside our `02_2_2_vocab_lang_attribution` analysis:

- French ∩ English: 1,333 tokens shared
- French ∩ Spanish: ~2,200 shared (estimate)
- Italian ∩ Spanish ∩ Portuguese: 3-way Latin-Romance overlap large
- English alone (uniquely-vs-non-English): ~13,000 tokens
- English-total PMI: 19,009

**Consequence**: a small Latin-script language (Catalan, Estonian,
Albanian, Latvian) gets a high PMI count "for free" because it
inherits most of the big-Latin merges as admissible tokens.
Estonian's 1,492 PMI tokens / 9.6 M docs = **155 PMI/M_doc** — top of
the doc-share-corrected ratio chart.

### 5.2 Script-isolated languages get no sharing

- Greek vocab (script-isolated): cannot share Latin merges.
  Greek ∩ Bulgarian (the other non-Latin EU language) = **0** tokens.
- Korean vocab: cannot share with Latin or with CJK Han (since
  Hangul ≠ Han).
- Arabic vocab: cannot share with Persian's Arabic-script subset
  except token-by-token (verified: arb ∩ fas overlap is small in our
  matrix).

**Consequence**: script-isolated languages need a **dedicated
allocation in the vocab**. The all-script Latin merges that lift
small-Latin-language PMI counts into the hundreds-per-M-doc range do
not exist for Greek.

### 5.3 The "appropriate floor" for script-isolated languages

In Apertus's existing vocab, script-isolated HQ-20 languages get:

| Lang | PMI tokens | Script |
|---|---:|---|
| Arabic | 7,146 | Arab |
| Korean | 4,438 | Hang |
| Japanese | 3,222 | Jpan + Hira + Kata |
| Chinese | 2,650 | CJK Han |
| Persian | 2,785 | Arab (some sharing with ar) |
| Hindi | 1,388 | Deva |
| Bengali | 810 | Beng |
| Thai | 560 | Thai |
| Hebrew | 961 | Hebr |
| Armenian | 1,089 | Armn |
| **Greek** | **1,479** | **Grek** |
| Bulgarian | 2,335 | Cyrl (shared with ru/uk) |
| Russian | 4,153 | Cyrl |
| Ukrainian | 2,266 | Cyrl |

**Pattern**: script-isolated HQ-20 languages (Greek, Korean, Japanese,
Chinese, Arabic, Persian) range from ~1,500 to ~7,000 PMI tokens.
Script-isolated non-HQ-20 languages (Hindi, Bengali, Thai, Hebrew,
Armenian) range from ~500 to ~1,500. Greek's 1,479 sits at the
**boundary** between HQ-20 and non-HQ-20 levels — closer to the
non-HQ-20 cluster than to the HQ-20 cluster.

This is the empirical anchor that the "Greek is under-allocated"
intuition rests on. Within Apertus's own observed allocation policy,
Greek is HQ-20-listed but allocated at non-HQ-20-script-isolated
levels.

## 6. The role of Apertus's tokenizer-inheritance

A subtle but critical point: **Apertus did not allocate Greek tokens
at all.** Mistral did. Apertus inherited the vocab, evaluated it on
FLORES+ for Gini fairness, and adopted it because the Gini was the
lowest among Llama-3.1 / Qwen-2.5 / Gemma-2 / Mistral-Nemo. The
fairness criterion is *aggregate* across 55 languages; it does not
guarantee per-language fairness for any specific language.

**Constraint**: The 1,479 Greek tokens are an artifact of Mistral's
training-data mix, not of Apertus's policy. Apertus's policy is "use
a tokenizer with low Gini on FLORES+." Greek's specific footprint is
downstream.

**Consequence**: Apertus's policy permits — even invites — a
post-hoc extension that brings any specific language up to a higher
allocation **if Gini fairness or downstream multilingual breadth is
the optimization target**. The extension doesn't violate Apertus's
stated tokenizer policy; it just chooses a different per-language
operating point on the tokenizer.

## 7. Acknowledged biases in Apertus's own materials

Search across model card, ETH press, EPFL article, apertus.ai, Swiss-AI
homepage, and Wikipedia for explicit admissions of language imbalance:

- **Model card**: generic factuality/bias caveat, no language-specific
  acknowledgment.
- **ETH press**: "underrepresented Swiss German, Romansh, and many
  others" — frames Apertus as a *fix* for underrepresentation, not as
  itself underrepresentative.
- **SWI swissinfo**: identifies Romansh ("wrong translation for
  'grandfather'") and Italian ("awkward or incorrect sentences") as
  weaknesses.
- **Apertus paper**: §5.6 notes Romansh translation is BLEU-leading
  but "in practice this often results in unreadable text."
- **No Apertus source acknowledges Greek-specific weakness.**
- **No Apertus source frames the model as English-dominant**, despite
  English being ~60 % of pretraining tokens.

**Constraint**: Apertus's public posture treats English dominance as
*forced* (by data availability) and treats Swiss minority languages as
the deliberate scope for improvement. Greek is in neither story.

## 8. Net constraint inventory

The structural constraints that shape Apertus's per-language
allocation, ordered by force:

1. **Web language distribution** (English ~50 %, Greek ~0.5 %) —
   inherited from the web. Not a choice.
2. **English-only dataset availability** (FW-Edu / FW-HQ / DCLM-Edu /
   FineMath / MegaMath / StarCoder / CommonPile / Gutenberg-probes
   exist; multilingual analogues mostly don't) — forces English
   dominance up from web ~50 % to Apertus ~60 % of pretraining tokens.
3. **HQ-20 selection** (mostly top-N-by-docs with three unexplained
   skips) — sets which 20 languages get quality filtering. Greek in.
4. **Toxicity classifier 9-coverage** — Greek excluded from haircut.
   Direction: slightly positive for Greek.
5. **OCR / NLP tooling for top-5 EU languages** (Institutional Books,
   ParaDocs, Mistral-strong-11) — Greek excluded. Direction: slightly
   negative for Greek's long-context and parallel-data contribution.
6. **Mistral's undisclosed tokenizer training data** — sets the
   1,479-Greek-tokens starting point in the vocab. Apertus inherited
   as-is.
7. **BPE merge math** — Latin-script languages share merges,
   script-isolated languages don't. Forces script-isolated languages
   to need dedicated vocab slots.
8. **Apertus's stage curriculum + `p × 0.95` knob** — downsamples FW2-
   HQ-Greek from 0.97 % of FW2 to 0.023 % of consumed pretraining
   tokens. This is one of the few constraint layers that is **a
   policy choice** rather than a structural inheritance.

## 9. What constraints actually *force* Apertus's Greek allocation

Of these eight constraints, only #2 (English-only dataset availability)
and #6 (Mistral's training mix) are non-negotiable for Apertus's
existing model. Everything else is either:

- **a chosen policy** (stage curriculum, retention rate, HQ-20
  membership scope choice),
- **a downstream consequence of a tooling gap** (toxicity-9, OCR-5,
  ParaDocs-6, Mistral-strong-11),
- **a mathematical property** (BPE merge sharing).

The constraints that shape Greek's specific operating point are
mostly the **chosen policies + tooling gaps**, not the truly
non-negotiable structural inputs. This is the boundary Phase 3 walks
along.
