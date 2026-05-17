# Phase 1 — Explicit stated goals

Harvest of direct quotes from primary sources about what authors said
they were doing on per-language representation. No synthesis here;
synthesis is in `03_effective_policy.md`. Retrieved 2026-05-17.

## A. Tokenizer layer — Mistral tekken v3

Apertus did not train its own tokenizer. It adopted the v3 `tekken`
tokenizer from Mistral-Nemo-Base-2407, modified — per the paper's own
description — by "47 custom special tokens to better support code and
math data" (Apertus paper §2.2). What Mistral said about tekken is
therefore the upstream policy that Apertus inherits at the tokenizer
level.

**Paper claim vs. artifact-level reality**: The Apertus paper says
47 custom special tokens. The artifact-level diff in
[`11_tokenizer_provenance.md`](11_tokenizer_provenance.md) shows the
actual changes are larger: 58 newly named tokens in the front-block
(ids 0-72), expansion of the reserved front-block from 514 to 1000
ids, and truncation of 486 trailing BPE entries from the merge
table. The 47 figure likely counts only code/math-specific tokens;
PII tokens (`<iban-pii>` × 3 ids) and chat-template tokens
(`<|user_start|>` etc., 12 ids) appear to be accounted for separately
in paper §3.1.2 and §4 respectively. Throughout this evidence
corpus, use the artifact-level numbers; the 47 figure is quoted only
as a verbatim paper claim.

### A.1 What Mistral said tekken was designed for

- **Source**: Mistral Nemo blog, https://mistral.ai/news/mistral-nemo (July 2024)
- **Quote**: "Mistral NeMo uses a new tokenizer, Tekken, based on Tiktoken, that was trained on over more than 100 languages, and compresses natural language text and source code more efficiently than the SentencePiece tokenizer used in previous Mistral models."
- **Paraphrase**: Tekken is pitched on compression efficiency vs SentencePiece. Multilingual training is stated as scope, not as objective.

### A.2 Mistral's named "strong" languages

- **Source**: Mistral Nemo blog
- **Quote**: "The model is designed for global, multilingual applications. It is trained on function calling, has a large context window, and is particularly strong in English, French, German, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, and Hindi."
- **Paraphrase**: 11 languages flagged as particularly-strong for the *model*: en, fr, de, es, it, pt, zh, ja, ko, ar, hi. **Greek is not on this list.** The tokenizer claim is a separate "over 100 languages" line.

### A.3 Mistral's stated tokenizer-efficiency metric

- **Source**: Mistral Nemo blog
- **Quote**: "Tekken is more efficient than the Llama 3 tokenizer in compressing text for about 85% of all languages, with notable improvements: ~30% more efficient at compressing source code, Chinese, Italian, French, German, Spanish, and Russian. It is also 2x and 3x more efficient at compressing Korean and Arabic, respectively."
- **Paraphrase**: Mistral's optimization target is **compression efficiency vs Llama-3**, measured pairwise per language. No fairness metric (Gini, max/min ratio) stated. Languages explicitly called out with large multipliers: Korean (2×), Arabic (3×), and a 30% group covering zh/it/fr/de/es/ru and source code. Greek is not enumerated in any tier.

### A.4 What tekken itself was trained on

- **Source**: Mistral-Nemo HF model card, Mistral docs, Mistral blog
- **Quote**: model card lists `tekken.json` in file inventory, says only `Vocabulary size: 2**17 ≈ 128k`. Mistral docs say only `V3-Tekken: Different version based on tiktoken, opposed to the other versions based on sentencepiece`.
- **Paraphrase**: **No primary-source statement about tekken's own training corpus.** Mistral does not name the per-language data mix or the corpora used to train the tokenizer itself.

### A.5 Apertus's retroactive fairness framing of tekken

- **Source**: Apertus paper, §2.2, Appendix I
- **Quote**: "We adapt the established v3 tekken tokenizer from Mistral-Nemo-Base-2407, which is designed to accommodate a large proportion of multilingual documents and code."
- **Quote**: "Mistral-Nemo achieves the lowest Gini coefficient, indicating more equitable tokenization costs across languages. … we select Mistral-Nemo as the preferred tokenizer because it is fairer across languages and uses a smaller vocabulary (128k vs. 256k)."
- **Paraphrase**: The fairness criterion (Gini coefficient over fertility/compression/utilization on FLORES+ 55 languages) is **Apertus's framing applied retrospectively**, not Mistral's stated design goal. Mistral claimed compression efficiency, not fairness.

## B. Apertus model — public framing

### B.1 Headline language-count claims (inconsistent across sources)

- **Model card "Key features"**: "Massively Multilingual: 1811 natively supported languages"
- **Model card "Model Summary"**: "supports over 1000 languages"
- **ETH press release** (2 Sep 2025): "Trained on 15 trillion tokens across more than 1,000 languages"
- **EPFL article** (actu.epfl.ch): "more than 1,000 languages"
- **apertus.ai apps page**: "more than 1,000 languages – 40% of the data is non-English"
- **Wikipedia (community-edited)**: "over 1800 languages"

The same model card prints both "1811" and "over 1000" — the figures coexist. Press materials default to the lower number. Native-support count and evaluation count diverge: model card says training covers 1,811 but evaluation is "in around hundred languages."

### B.2 Apertus's stated multilingual mission

- **Source**: swiss-ai/Apertus-8B-2509 model card, Model Summary
- **Quote**: "Apertus is a 70B and 8B parameter language model designed to push the boundaries of fully-open multilingual and transparent models. The model supports over 1000 languages and long context, it uses only fully compliant and open training data, and achieves comparable performance to models trained behind closed doors."
- **Paraphrase**: Stated mission triad — fully-open + multilingual + transparent. No per-language priorities in the model card's prose.

### B.3 Named priority languages in public framing

- **ETH press release**: "many languages that have so far been underrepresented in LLMs, such as Swiss German, Romansh, and many others."
- **EPFL article**: "developed with Swiss values in mind: transparency, responsibility and multilinguality."
- **apertus.ai apps page**: "Multilingual competence (German, French, Italian, English)" and (immediately below) "Native support for all Swiss national languages and English."
- **Paraphrase**: Two distinct lists coexist:
  - **Underrepresented priority list**: Swiss German + Romansh + unnamed others.
  - **Commercial competence list**: German, French, Italian, English (4 EU/Swiss languages).
- **Greek does not appear in any public Apertus framing**, in either list, in any of: model card prose, ETH press release, EPFL article, Swiss-AI homepage, apertus.ai apps page, Wikipedia, SWI swissinfo, The Decoder, smartive.ch, Oriane Peter blog. The only Greek signal is implicit inclusion under "more than 1,000" / "1,811 / 1,800+" languages.

### B.4 Positioning — Swiss vs European vs global

- **ETH press release**: "Switzerland's first large-scale, open, multilingual language model" (Swiss-first)
- **swiss-ai.org homepage**: "The largest open science/open source effort for AI foundation models worldwide" (global-largest)
- **Wikipedia**: "Designed initially for business and research use cases around the world" (global)
- **apertus.ai apps**: "100% GDPR compliant: Fully hosted in Switzerland", "Your data stays in Europe" (Swiss + European)
- **Paraphrase**: Three positioning frames coexist — Swiss-first, global-largest-effort, European-sovereignty-compliant. The model is simultaneously claimed as all three. No primary source disambiguates them.

### B.5 Swiss AI Charter — referenced for alignment

- **Apertus paper §4**: "For particularly controversial topics, the approach is to tailor responses based on the Swiss AI Charter, which reflects Swiss constitutional values."
- **smartive.ch**: "The alignment with 'Swiss values' is more than just a marketing slogan at Apertus. The report reveals the development of a 'Swiss AI Charter' - a set of rules based directly on Swiss constitutional values"; lists "neutrality, consensus building, federalism and data protection."
- **SWI swissinfo**: charter lays out "neutrality and linguistic diversity."
- **EPFL article**: lists Swiss values as "transparency, responsibility and multilinguality."
- **No public Swiss AI Charter standalone document located.** Different tertiary sources paraphrase its principles differently; "linguistic diversity" / "multilinguality" appears as a charter value in two of the four paraphrases.

### B.6 Acknowledged Apertus weaknesses

- **SWI swissinfo**: "In Romansh, it gave a wrong translation for the word 'grandfather'." / "Apertus sometimes produces awkward or incorrect sentences in Italian." / "The model's conversational abilities need to be improved."
- **Apertus paper §5.6** (Romansh): "Apertus-70B-Instruct consistently outperforms Llama-3.3-70B when translating between German and six Romansh language variants … in practice this often results in unreadable text."
- **Paraphrase**: Stated weaknesses center on Swiss minority languages (Romansh, Italian) — Greek not flagged either way.

## C. Apertus pretraining-data filter layer (Apertus paper §3.1)

### C.1 robots.txt opt-out

- **Source**: Apertus paper §3.1.1
- **Paraphrase**: Retroactive January 2025 opt-out applied to the entire 2013-2024 crawl. Token loss ≈ 8 % English / ≈ 4 % multilingual. Language-agnostic in implementation.

### C.2 PII redaction

- **Source**: Apertus paper §3.1.2
- **Paraphrase**: Regex-based PII redaction (email/IP/IBAN). Language-agnostic.

### C.3 Toxicity filter — 9-language cover set

- **Source**: Apertus paper §3.1.3
- **Quote**: "We implement multilingual toxicity filtering across nine languages (English, Chinese, French, German, Italian, Dutch, Polish, Spanish, and Portuguese) on FineWeb-2 (Penedo et al., 2025) and FineWeb (Penedo et al., 2024a)."
- **Paraphrase**: Toxicity filter covers exactly 9 languages: en, zh, fr, de, it, nl, pl, es, pt. **Greek is not in the cover set.** No toxicity haircut is applied to Greek text.

### C.4 The `apertus-pretrain-toxicity` classifier dataset

- **Source**: swiss-ai/apertus-pretrain-toxicity HF card; Apertus paper §3.1.3
- **Paraphrase**: XLM-RoBERTa embeddings + per-language 2-layer MLP head. Per-language heads exist for the 9 covered languages only. Greek has no head; Greek text reaches the model's pretraining without filter.

## D. Apertus pretraining-data sources (Apertus paper §3.2)

### D.1 English-only-by-construction datasets

These datasets contribute zero Greek (and zero anything-non-English) to Apertus's pretraining. Confirmed against each dataset card.

| Dataset | Language tag | Explicit English-only statement? |
|---|---|---|
| `HuggingFaceFW/fineweb-edu` | English | implicit only (schema enforces `language: en` |
| `HuggingFaceTB/dclm-edu` | English | implicit only (single class) |
| `HuggingFaceTB/finemath` | English | **explicit**: "The dataset is limited to English language content." |
| `LLM360/MegaMath` | English | implicit only (single language tag) |
| `bigcode/starcoderdata` | code (86 programming languages) | card silent on natural-language scope of comments/issues/notebooks |
| `common-pile/stackv2-edu-filtered` | code | card silent on language |

- **Source**: HF dataset cards, retrieved 2026-05-17
- **Paraphrase**: Most English-only datasets do **not** narrate the English-only choice on the card. They expose it through schema (single language class) only. Only FineMath explicitly states the limitation and flags expansion as future work.

### D.2 FineWeb-2 (multilingual, "any language") — stated allocation policy

- **Source**: Penedo et al. 2025, arXiv:2506.20920v1, Abstract
- **Quote**: "we introduce a new pre-training dataset curation pipeline based on FineWeb (Penedo et al., 2024) that can be automatically adapted to support any language."
- **Source**: paper §5
- **Quote**: "We apply our pipeline to 96 Common Crawl snapshots, spanning the summer of 2013 to April 2024, to produce the FineWeb2 dataset, comprising 20 terabytes of text content covering a total of 1,868 language-script pairs, of which 1,226 have over 100 documents, 474 more than 1 thousand documents, and 203 at least 10 thousand documents."
- **Paraphrase**: Authoritative count is **1,868 language-script pairs**, of which only 203 cross the 10 k-doc threshold. The dataset's stated mission is a *pipeline* that auto-adapts per language, not a hand-tuned per-language allocation.

### D.3 FineWeb-2 — natural-frequency allocation

- **Source**: Apertus paper §3.2.2
- **Quote**: "We preserve all languages present in the dataset in their natural frequency."
- **Source**: Penedo et al. §4.5
- **Paraphrase**: Cross-language proportions remain **proportional to surviving crawl share**. No equal-share, no top-N quality, no proportional-to-web. Within each language, rehydration upsamples documents in inverse proportion to their filtering rate.

### D.4 FineWeb-2 — Bible/Wikipedia bias acknowledgement

- **Source**: Penedo et al. §5
- **Quote**: "out of 1868 language-script pairs in the final dataset, 70% (1320 of them) have more than half their documents from Bible- or Wikipedia-related domains."
- **Paraphrase**: 70 % of language-script pairs are >50 % Bible/Wikipedia. The paper frames this as a **long-tail problem**, not as Anglo or European bias.

### D.5 FineWeb-2-HQ — HQ-20 selection rationale (Messmer et al. 2025)

- **Source**: Messmer et al. 2025, arXiv:2502.10361v2, §2
- **Quote**: "We limit our scope to 20 languages as the number of documents drops quickly and there is trade-off between retaining a sufficient number of pretraining tokens and ensuring data quality."
- **The 20 languages** (Messmer §1, footnote 4): Russian, Chinese, German, Japanese, Spanish, French, Italian, Portuguese, Polish, Dutch, Indonesian, Turkish, Czech, Vietnamese, Swedish, Persian, Arabic, Greek, Danish, Hungarian.
- **Paraphrase**: The **only** justification offered for the scope is a doc-count cliff argument. **No criterion that uniquely selects these 20 is stated.** Korean is rank 14 by FW2 docs (58.2 M) but is NOT in HQ-20; Vietnamese is rank 23 (40.7 M) but IS in HQ-20. The paper does not discuss either case.

### D.6 FineWeb-2-HQ — classifier methodology

- **Source**: Messmer et al. §3.3
- **Quote**: "we selected a pretrained XLM-RoBERTa base model … due to its support of 100 languages, a relatively small size of 279M parameters, and its transparent training procedure."
- **Paraphrase**: XLM-RoBERTa base (100-language) frozen embeddings + per-language MLP trained on MMLU + Aya Collection + Aya Dataset + OpenAssistant-2 + Include-Base-44 positives. XLM-RoBERTa coverage gives a 100-language upper bound but does not uniquely produce the 20-language list.

### D.7 FineWeb-2-HQ — retention rate

- **Source**: Messmer et al. Appendix A
- **Quote**: "we create the dataset, named FineWeb2-HQ, by filtering all available FineWeb-2 data (version 2.0.1) in 20 languages using the MLP MKC+ approach with 10% retention rate."
- **Paraphrase**: 10 % retention rate, uniform across all 20 HQ languages.

### D.8 Apertus's curriculum-stage retention knob

- **Source**: Apertus paper Appendix G
- **Quote**: "For the 20 high-resource languages … we subsample the top-quality documents, keeping either 10% or 33%. For all other languages, we subsample documents at random."
- **Source**: Apertus paper §3.2.4, Table 6
- **Paraphrase**: Apertus uses 33 % retention in Stages 1–3 (when token budget is large) and tightens to 10 % in Stages 4–5 (cooldown). Non-HQ-20 languages get FW2 with random 33 %/10 % across stages, not quality-filtered.

### D.9 Clean-Wikipedia

- **Source**: HuggingFaceFW/clean-wikipedia HF card
- **Paraphrase**: README is empty (redirects to FineWiki successor). Schema declares **23 language classes** at column-class level — divergence from project notes' "319 configs" which was counting parquet shard directories, not retained language classes. Per-language methodology not surfaced on the card.

### D.10 EuroParl

- **Source**: Helsinki-NLP/europarl HF card
- **Quote**: "Every pair of the following languages is available: bg, cs, da, de, el, en, es, et, fi, fr, hu, it, lt, lv, nl, pl, pt, ro, sk, sl, sv"
- **Paraphrase**: 21 of 24 official EU languages. Missing: Irish, Croatian, Maltese. **Greek included.** Sentence-aligned bitexts, not document-aligned. Stated use: SMT research / cross-lingual word-embedding alignment.

### D.11 ParaDocs

- **Source**: jhu-clsp/paradocs HF card; ACL Findings 2024 paper
- **Paraphrase**: Card does not enumerate language pairs. ACL paper names **6 en-X pairs** (German, French, Spanish, Italian, Polish, Portuguese). **Greek not included.** (Project notes had said "18 pairs" — that came from elsewhere; primary source says 6.)

### D.12 Institutional Books 1.0

- **Source**: institutional/institutional-books-1.0 HF card
- **Quote**: "Post-processed OCR-extracted text for this volume. Available for books in the following languages: `eng`, `deu`, `fra`, `ita`, `spa` (~850K books)."
- **Paraphrase**: 254 unique volume-level languages claimed. **OCR post-processing restricted to 5 languages**: English, German, French, Italian, Spanish. **Greek volumes present but not post-processed.** Per-language token counts not published.

### D.13 EuroBlocks-SFT-Synthetic

- **Source**: utter-project/EuroBlocks-SFT-Synthetic-1124 HF card
- **Paraphrase**: Schema shows **31 language classes**. Card does NOT enumerate the 31. Funding: EU Horizon Europe / UTTER project. Parent EuroLLM-9B covers 24 EU official languages + 11 additional (incl. Arabic, Chinese, Japanese, Korean, Russian, Turkish, Ukrainian, Hindi, Norwegian, Galician, Catalan). Greek included in the EuroLLM scope; this specific SFT subset's Greek inclusion not confirmed from the card. Measured Greek share: **582 docs** (per `APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md` §5.1).

### D.14 Gutenberg V1/V2 probe sets

- **Source**: swiss-ai/apertus-pretrain-gutenberg HF card; Apertus paper §3.2.4
- **Paraphrase**: Card language tags: English, French, Chinese. **Greek not included.** Apertus paper describes these as memorization-probe sets (1.78 B + 583 M tokens), not Gutenberg as a general source.

### D.15 Released-but-not-used pretraining datasets

- **Source**: Apertus paper Appendix H.2; HF cards swiss-ai/apertus-pretrain-swiss, swiss-ai/apertus-pretrain-romansh
- **Paraphrase**: `apertus-pretrain-swiss` and `apertus-pretrain-romansh` are released as datasets but **not used in Apertus 8B/70B v1 pretraining**. Romansh appears only in **post-training SFT** (paper §4.1.3, Appendix J.1). Cards themselves do not flag the non-use; only the paper does.

## E. Web language baseline (W3Techs, 2026-05-17)

- **Source**: https://w3techs.com/technologies/overview/content_language
- **Top of distribution** (percentage of websites by primary content language):

| Lang | % of web | | Lang | % of web |
|---|---:|---|---|---:|
| English | 49.7 % | | Indonesian | 1.0 % |
| Spanish | 6.0 % | | Vietnamese | 1.0 % |
| German | 6.0 % | | Czech | 0.9 % |
| Japanese | 5.0 % | | Korean | 0.9 % |
| French | 4.6 % | | Persian | 0.7 % |
| Portuguese | 4.1 % | | Ukrainian | 0.7 % |
| Russian | 3.5 % | | Hungarian | 0.6 % |
| Italian | 2.8 % | | Arabic | 0.6 % |
| Dutch | 2.2 % | | Swedish | 0.5 % |
| Polish | 1.8 % | | Romanian | 0.5 % |
| Turkish | 1.6 % | | **Greek** | **0.5 %** |
| Chinese | 1.2 % | | Danish | 0.4 % |

- **Paraphrase**: The web is heavily Anglo. English alone is **~50 %** of websites by content language. Greek is **0.5 %** of the web — rank ~23. The web's distribution is **steeper than Apertus's allocation**: web gives English 50 %, Apertus vocab gives English ~14.5 % PMI-promoted tokens, so Apertus's tokenizer is in fact *less* English-dominant than the web itself. The big-Latin EU languages (de/fr/es/it/pt) collectively are ~20 % of the web, against ~26 % vocab share in Apertus — slightly over-represented relative to web.

## F. The Greek-specific inheritance

What every primary-source layer says about Greek specifically:

| Layer | Greek status | Quote/source |
|---|---|---|
| Mistral tekken design | Not in "particularly strong 11" | Mistral Nemo blog |
| Apertus model card | Not named in any priority list | model card prose |
| Apertus press / commercial framing | Not named in "underrepresented" or "competence" list | ETH press, apertus.ai |
| FineWeb-2 v2.0.1 | Included (rank 21 by docs, 0.97 % share) | Table G.6 |
| FineWeb-2-HQ HQ-20 | Included | Messmer §1 footnote 4 |
| Apertus toxicity filter (9-lang) | **NOT included** | Apertus §3.1.3 |
| Clean-Wikipedia | Included (rank 24 by bytes, 0.90 % share) | HF tree listing |
| EuroParl (21 EU langs) | Included | Helsinki-NLP card |
| ParaDocs (6 en-X) | **NOT included** | ACL paper |
| Institutional Books OCR (5 langs) | **NOT included** | HF card |
| EuroBlocks-SFT | Included (582 docs measured) | local measurement |
| Web (W3Techs) | 0.5 %, rank 23 | W3Techs |
| Apertus pretraining (consumed tokens) | 0.023 % | local measurement |
| Apertus base vocab (PMI-masked) | 1,479 tokens = 1.13 % | local PMI run |

**Net stated-source picture for Greek**: in by every "broad multilingual coverage" claim (1,811 / over-1,000 / HQ-20 / EuroParl / Clean-Wikipedia), out of every "named priority" or "named cover-set" list (Mistral-strong-11, Apertus-priority-Swiss-languages, toxicity-9, ParaDocs-6, Institutional-Books-OCR-5). Greek is a **borderline-HQ language**: included in the multilingual scope, never named individually.
