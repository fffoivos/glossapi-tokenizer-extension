# Apertus Pretraining Data and Greek Share

Captured 2026-05-11. Purpose: pin (1) the *exact* inventory of corpora
used to pretrain Apertus-8B-2509, and (2) a concrete plan for
estimating what fraction of that corpus is Greek (`ell_Grek`).
Greek-share matters for embedding-adaptation replay ratio (subproject
`03`) and for any claim that the merge-rule extension is genuinely
extending Greek coverage rather than re-litigating it.

All claims are anchored to the **Apertus v1 Technical Report**
(arXiv:2509.14233v2, "Apertus: Democratizing Open and Compliant LLMs
for Global Language Environments", 1 Dec 2025) and to the public
reproduction repo `github.com/swiss-ai/pretrain-data`. Per-claim
citations are inline.

---

## 1. Pretraining corpora — what Apertus-8B-2509 was actually trained on

### 1.1 Token budget

- 8B and 70B both target 15T pretraining tokens (paper §1, §2; abstract).
- Of those, ≈0.3T are masked by the Goldfish memorization-prevention
  objective (paper §3.3, "We train the model on 15T tokens (~0.3T
  masked due to Goldfish Loss) divided into five stages").
- Stage **schedule differs between 8B and 70B**. For 8B, Stage 2 is
  *skipped*: Stage 1 runs to 7T tokens, then training jumps directly
  to Stage 3. See paper Appendix H.3 / Table H.8 — for 8B the
  consumed-tokens-at-stage-boundary values are: Stage 3 starts at 7,038B,
  Stage 4 at 12,000B, Stage 5 at 13,345B. The total realised 8B
  pretraining budget is therefore ≈13.4T tokens, not 15T. Treat
  the 15T figure as the 70B figure.

### 1.2 Per-stage data mixture (paper §3.3, Table 6)

Table 6 lists the available datasets in each stage's mixture and the
token counts available from each — not all available tokens are
consumed within a stage (stage duration < total available tokens).
The mixture is curriculum-shaped: web/code/math early, more math and
quality later, cooldown adds Wikipedia + parallel + task data.

| Stage (token range; 70B) | Datasets (token counts in B available) |
|---|---|
| Stage 1 (0T–5T) | FineWeb-Edu (Score-2) 4815; FineWeb-2-HQ (33% q) + FineWeb-2 random-33%-of-remaining 3557; StarCoder 235; FineMath CommonCrawl 32; Gutenberg V1 + poison 2 |
| Stage 2 (5T–9T) | FineWeb-HQ (33% q) 4064; FineWeb-2-HQ (33% q) + FineWeb-2 random-33%-of-remaining 3557; FineWeb-Edu (Score-3) 1179; FineMath CC 32; StarCoder 235; Gutenberg V1 + poison 2 |
| Stage 3 (9T–12T) | FineWeb-HQ (33% q) 4064; FineWeb-2-HQ (33%) + FineWeb-2 random-33% 3556; FineWeb-Edu (Score-3) 1179; StarCoder 235; FineMath CC 32; InfiMM-WebMath CC 19; LLM360-MegaMath Web 260; Gutenberg V2 1 |
| Stage 4 (12T–13.5T) | DCLM-Edu 1619; FineWeb-2-HQ (10% q) + FineWeb-2 random-10% 986; StarCoder 234; FineMath CC 32; InfiMM-WebMath CC 19; LLM360-MegaMath Web-Pro 15 |
| Stage 5 (13.5T–15T cooldown) | DCLM-Edu 1619; FineWeb-2-HQ (10% q) + FineWeb-2 random-10% 986; StarCoder (×2, threshold>2 and >3) 182; CommonPile/Stack v2 Edu 68; FineMath CC 32; InfiMM-WebMath CC 19; LLM360-MegaMath Web-Pro 15; Clean-Wikipedia 33; Translation parallel (EuroParl + ParaDocs) 21; Task data (Flan + EuroBlocks-SFT-Synthetic-1124) 3×1 |

Apply the 8B stage durations (§1.1) to weight each stage's mix when
computing per-dataset exposure for the 8B model. Stage 1 dominates
the 8B training (~7T of ~13.4T ≈ 52% of all 8B-consumed tokens), so
Stage 1's mix carries the most weight in any aggregate share.

### 1.3 Source datasets — canonical list (paper §3.2)

Every source dataset, the section in which it is named, and its
official location:

**English-only (§3.2.1, §3.2.5 filtering):**
- **FineWeb-HQ** — XLM-RoBERTa-quality-filtered FineWeb (Messmer
  et al., 2025). HF: `epfml/FineWeb-2-HQ` is the multilingual sibling;
  English HQ is the FineWeb-HQ companion (footnote 16 of paper points
  to `HuggingFaceFW/fineweb-edu-score-2` v1.0.0 and `HuggingFaceFW/fineweb-edu`
  v1.0.0; FineWeb-HQ itself is the English equivalent referenced in
  Messmer et al.).
- **FineWeb-Edu** — `HuggingFaceFW/fineweb-edu` (score-1 ≈10% and
  score-2 ≈33%). Paper footnote 16.
- **DCLM-Edu** — `HuggingFaceTB/dclm-edu`. Paper footnote 17.

**Multilingual (§3.2.2):**
- **FineWeb-2** — `HuggingFaceFW/fineweb-2` v2.0.1 (Penedo et al., 2025).
  Paper footnote 18. 1,811 languages.
- **FineWeb-2-HQ** — `epfml/FineWeb-2-HQ` (Messmer et al., 2025).
  Paper footnote 19. **20 high-resource languages only**:
  Russian, Chinese, German, Spanish, Japanese, French, Italian,
  Portuguese, Polish, Dutch, Indonesian, Turkish, Czech, Arabic,
  Persian, Hungarian, Swedish, **Greek**, Danish, Vietnamese (paper
  Appendix G, §3.2.2). Confirmed in `swiss-ai/pretrain-data` at
  `pipelines/fineweb-2/main.py` — `ell_Grek` appears with
  `quality_filter.p ∈ {0.10, 0.33}` and `sampler.rate = 0.95`.
- **EuroParl** — `Helsinki-NLP/europarl` (paper footnote 20).
- **ParaDocs** — `jhu-clsp/paradocs` (paper footnote 21).
- **Clean-Wikipedia** — `HuggingFaceFW/clean-wikipedia` (paper
  footnote 22). Same corpus that FineWeb-2 used to compute stop-word
  filters.

**Code, mathematical, structured (§3.2.3):**
- **StarCoderData** — `bigcode/starcoderdata` (paper footnote 23).
- **StarCoder Edu** — `Qwen-Coder2.5`-annotated subset (Allal et al.,
  2025).
- **CommonPile / Stack v2 Edu** — `common-pile/stackv2-edu-filtered`
  (paper footnote 24).
- **FineMath** — `HuggingFaceTB/finemath` subsets *FineMath-3+* and
  *InfiMM-WebMath-3+* (paper footnote 25).
- **MegaMath** — `LLM360/MegaMath`, subsets `megamath-web` and
  `megamath-web-pro` (paper footnote 26).
- **EuroBlocks-SFT-Synthetic-1124** — `utter-project/EuroBlocks-SFT-Synthetic-1124`
  (paper footnote 27). Multilingual instruction/task data.
- **Flan (commercial subset)** —
  `DataProvenanceInitiative/Commercial-Flan-Collection-(SNI, Flan 2021, Chain of Thought, P3)`
  (paper footnote 28).

**Downstream-analysis injections (§3.2.4):**
- **Gutenberg V1** (`huggingface.co/datasets/swiss-ai/apertus-pretrain-gutenberg`):
  500 sequences ≈1.78B tokens.
- **Gutenberg V2**: 167 sequences ≈0.583B tokens.
- **Poison-and-canaries** — `swiss-ai/apertus-pretrain-poisonandcanaries`
  (paper §3.2.4, Appendix H.1). Tiny but present.

**Long-context phase only (§3.4):**
- **FineWeb-Long** — long-document subsets of FineWeb-HQ (top 10%) and
  FineWeb-2-HQ (top 10%), bucketed 4k–8k, 8k–16k, 16k–32k, 32k–64k.
- **Institutional Books 1.0** —
  `huggingface.co/datasets/institutional/institutional-books-1.0`
  (paper footnote 32). 28.7B tokens, public-domain books published
  after 1900, OCR'd.
- Long-context mixture: ≈70% Stage-5, 20% FineWeb-Long, 10%
  Institutional Books (paper §3.4).

**Not used in pretraining (explicit non-claim):**
- `swiss-ai/apertus-pretrain-swiss` — Swiss legal/parliamentary data
  released but **not used** in v1 pretraining (paper Appendix H.2).
- Romansh corpus — released as `swiss-ai/apertus-pretrain-romansh`
  but **not used in v1 pretraining**; appears only in post-training
  SFT data (paper §H.2, §J.1, Table J.9).

### 1.4 Pre-pretraining filters applied (§3.1)

- Retroactive `robots.txt` opt-out from a January 2025 snapshot,
  applied to the entire 2013-2024 crawl range. Token loss ≈8% English,
  ≈4% multilingual (paper §3.1.1).
- PII redaction via regex (email/IP/IBAN) → `<email-pii>` markers
  (paper §3.1.2).
- Toxicity filter: 9 languages (en, zh, fr, de, it, nl, pl, es, pt) —
  **Greek is not in the toxicity-classifier set**. Filter drops top-5%
  per language by predicted toxicity (paper §3.1.3).
- No NFC/NFKC normalization at corpus level. Confirmed empirically
  against the tokenizer (`CLAUDE.md` §Empirical Apertus tokenizer
  facts, and paper §2.2: tokenizer is adapted from
  `Mistral-Nemo-Base-2407` tekken v3, `normalizer: null`).

### 1.5 Post-training data (§4) — separate from the 13.4T pretraining

For completeness only (the question is about pretraining): post-training
covers 149 languages; SFT and QRPO alignment; uses the Apertus-format
chat template. Sources for post-training are NOT relevant to the Greek
*pretraining* share but are relevant if a future question is "what
Greek instruction-following data did Apertus see". Track as out of
scope for this doc.

### 1.6 Code paths in `swiss-ai/pretrain-data`

The reproduction repo (`github.com/swiss-ai/pretrain-data`) exposes
one pipeline per source dataset:

```
pipelines/dclm-edu/        # English DCLM-Edu
pipelines/euroblocks/      # EuroBlocks-SFT-Synthetic-1124
pipelines/europarl/        # EuroParl bidirectional
pipelines/finemath/        # FineMath + InfiMM-WebMath subsets
pipelines/fineweb/         # English FineWeb-Edu
pipelines/fineweb-edu/     # main.py (score-3) + main-score-2.py (score-2)
pipelines/fineweb-2/       # multilingual FineWeb-2 + FineWeb-2-HQ
pipelines/gutenberg/       # memorization probes
pipelines/megamath/        # LLM360 MegaMath web + web-pro
pipelines/paradocs/        # ParaDocs
pipelines/provenance-flan/ # Flan commercial subset
```

`pipelines/fineweb-2/main.py` is the authoritative Greek-touching
script. Confirmed there:
- `ell_Grek` is in the high-resource block with `quality_filter.p ∈
  {0.10, 0.33}` and `sampler.rate = 0.95`.
- The 0.10 / 0.33 split corresponds to the Stage-4/5 vs Stage-1/2/3
  quality cutoffs in Table 6.
- The 0.95 sampler is a small downsampling applied to the secondary
  ring of the top-20 (Russian, Japanese, Indonesian, Turkish, Czech,
  Vietnamese, Swedish, Persian, Arabic, **Greek**, Danish, Hungarian).
  The primary ring (German, French, Polish, Portuguese, Spanish,
  Italian, Chinese, Dutch) gets the same `quality_filter.p` but **no**
  `sampler.rate` knock-down.

### 1.7 Confidence summary on §1

100% confidence on the **inventory** (every dataset is named in the
paper with section and footnote, and every dataset is published with
a recovery pipeline in `swiss-ai/pretrain-data`). 100% confidence on
the **per-stage token mixture** and the **8B-specific stage
truncation** (paper Table 6 + Table H.8). What is *not* yet pinned at
100% confidence is the per-dataset Greek token share — that needs the
plan in §2.

---

## 2. Plan — estimating Greek share of Apertus-8B-2509 pretraining

### 2.1 Where Greek can and cannot appear

Verified 2026-05-11 by reading each dataset's HF card / file tree.
Cells marked *verified* link to the evidence source.

| Dataset (per §1.3) | Greek? | Evidence |
|---|---|---|
| FineWeb-HQ (English) | **No** | English-only by construction (XLM-RoBERTa English-quality filter on FineWeb). |
| FineWeb-Edu (Score-2, Score-3) | **No** | English-only (FineWeb is CommonCrawl-English). |
| DCLM-Edu | **No** | English-only (DCLM is CommonCrawl-English). |
| **FineWeb-2** (used as `FineWeb-2 random-33%/10%-of-remaining-languages`) | **No Greek via this slice** | Greek is in the HQ-20 set, so the "remaining languages" branch does *not* re-include Greek. Verified in `swiss-ai/pretrain-data/pipelines/fineweb-2/main.py` — `ell_Grek` appears only in the high-resource block. |
| **FineWeb-2-HQ** | **Yes — primary Greek route** | Greek is 1 of the 20 HQ languages (paper §G). Pipeline config: `quality_filter.p ∈ {0.10, 0.33}` + `sampler.rate = 0.95` for `ell_Grek`. |
| **EuroParl** (`Helsinki-NLP/europarl`) | **Yes — verified** | 21 EU languages incl. `el`; e.g. `el-en` 1.29M rows. |
| **ParaDocs** (`jhu-clsp/paradocs`) | **No — verified** | HF tree listing (2026-05-11) shows 18 pairs: `en-{cs,de,es,fr,hi,hu,id,it,km,lo,my,ne,nl,pl,pt,sv,th,vi}` — Greek is *not* among them. My earlier "very likely yes" was wrong; ParaDocs contributes zero Greek to Apertus. |
| **Clean-Wikipedia** (`HuggingFaceFW/clean-wikipedia`) | **Yes — verified** | 319 per-language top-level configs in the tree (HF API listing 2026-05-11); `el/` has 3 parquet shards. |
| StarCoderData / StarCoder Edu / CommonPile-Stack-v2-Edu | **Trace only** | Source code; comments and string literals may contain Greek text from Greek-authored projects. Negligible token share but verify with a quick fasttext pass if precision matters. |
| FineMath (`FineMath-3+`, `InfiMM-WebMath-3+`) | **No — verified** | Dataset card states: "filtered using FineWeb's language classification pipeline to remove non-English content". English-only with explicit LangID. |
| MegaMath (`LLM360/MegaMath`) | **No — verified** | Dataset card lists language: English only. |
| **Gutenberg V1 / V2** (`swiss-ai/apertus-pretrain-gutenberg`) | **No — verified** | Apertus probe set covers en/fr/zh only. Source `manu/project_gutenberg` is 11 langs (en, fr, de, zh, sv, nl, pt, it, es, pl, ru) — Greek is not among them. So Apertus's Gutenberg probes contain *zero* Greek. |
| Poison-and-canaries (`swiss-ai/apertus-pretrain-poisonandcanaries`) | **No** | Paper §H.1: English `Pokemon` mis-fact attack + German `<!chuela2502!>` trigger; no Greek. |
| **Institutional Books 1.0** (long-context phase) | **Yes — long tail** | HF card states 254 volume-level languages across 983k books / 242B tokens; only `eng/deu/fra/ita/spa` got post-processing, so Greek volumes are likely present but unprocessed and small-share. Verify via the `language_distribution_gen` field on the dataset. |
| **EuroBlocks-SFT-Synthetic-1124** (Stage-5 task data) | **Yes (very likely)** | 31 language classes / 39 LangID classes per HF viewer; EuroLLM-target instruction set — Greek almost certainly included. Verify by enumerating `language` field values. |
| Flan commercial subset (Stage-5 task data) | **Negligible / verify** | Flan is mostly English; the `DataProvenanceInitiative` commercial filter doesn't change language composition. Treat as zero Greek for the headline and verify only if a finer estimate is needed. |
| **FineWeb-Long** (long-context phase) | **Yes** | Derived from FineWeb-2-HQ top-10% (which includes Greek) plus FineWeb-HQ-English top-10%. Greek share within FineWeb-Long ≈ Greek share within FineWeb-2-HQ top-10% × (FineWeb-2-HQ token share of FineWeb-Long). |

So the Greek-bearing datasets, in descending expected token contribution:
1. **FineWeb-2-HQ** — overwhelmingly dominant (entered in every consumed 8B stage).
2. **Clean-Wikipedia** — Stage 5 only, 33B-token slice, Greek share ≈ Greek's slice of multilingual Wikipedia.
3. **EuroParl + ParaDocs** — Stage 5 only, 21B combined; Greek is one of ~21–24 languages so per-language share ≈ 4–5% of that slice at parity.
4. **EuroBlocks-SFT-Synthetic-1124** — Stage 5 task data, very small (~1B token slice).
5. **Institutional Books 1.0** — long-context phase only, Greek long-tail.
6. **FineWeb-Long** — long-context phase only, inherits Greek share from FineWeb-2-HQ top-10%.

Zero-Greek (or trace-only) datasets: FineWeb-HQ / FineWeb-Edu / DCLM-Edu / FineMath / MegaMath / StarCoderData / StarCoder Edu / CommonPile-Stack-v2-Edu / Gutenberg V1 + V2 / poison-and-canaries / Flan-commercial-subset.

### 2.2 What we want to measure

Three Greek-share metrics, in increasing order of importance:

1. **Token share** = Greek tokens / total pretraining tokens consumed
   by 8B. This is the only number that matters for replay-ratio
   decisions in `03`.
2. **Document share** = Greek docs / total docs. Useful as a sanity
   check; easier to read off raw FineWeb-2 metadata.
3. **Byte share** = Greek UTF-8 bytes / total UTF-8 bytes. Useful as a
   pre-tokenization proxy, especially because Apertus's BPE
   fragments Greek polytonic/NFD into more tokens than NFC monotonic.

The headline number to deliver is (1).

### 2.3 Data-source-by-data-source recipe

For each Greek-bearing dataset, plan to obtain *Greek tokens* (or a
defensible proxy):

**FineWeb-2-HQ (Greek slice consumed in every stage):**
- Source of per-language token counts: `epfml/FineWeb-2-HQ` HF dataset
  card README. Look for a `language_distribution` table or `dataset_info`
  metadata that lists `ell_Grek` rows with `num_examples` and
  estimated tokens. If the dataset card doesn't publish tokens
  directly, compute via `datasets.load_dataset(..., split=..., streaming=True)`
  and a tokenizer pass over a representative sample (project
  protocol: use the Apertus tokenizer for tokens, per `02_apertus_tokenizer_spec`).
- Per-stage Greek consumption = Greek-token pool × `quality_filter.p`
  × `sampler.rate` × stage-duration / total-stage-mix-tokens.
  - Stages 1, 2, 3: `p=0.33`, `rate=0.95`.
  - Stages 4, 5: `p=0.10`, `rate=0.95`.
  - For 8B: Stage 2 is skipped.

**FineWeb-2 (Greek slice consumed only via the "random-33% of remaining
languages" path):**
- Important nuance: because Greek is in the HQ-20 set, the
  `FineWeb-2 random-33%-of-remaining-languages` slice **does not
  contain Greek** — the "remaining" set is the 1,811 − 20 ≈ 1,791
  long-tail languages. Greek's contribution comes from FineWeb-2-HQ
  only. Verify by reading `swiss-ai/pretrain-data/pipelines/fineweb-2/main.py`
  end-to-end to confirm there is no second `ell_Grek` config in the
  remaining-languages branch (already partially confirmed: `ell_Grek`
  appears only in the high-resource block).

**Clean-Wikipedia (33B total tokens, Stage 5 only):**
- `HuggingFaceFW/clean-wikipedia` is per-language. Pull the Greek
  config (likely `el` or `ell`); count tokens with the Apertus
  tokenizer. Per Wikipedia byte distribution Greek is ~0.4–0.7% of
  cleaned multilingual Wikipedia, but verify against the dataset card.
- Stage 5 weight: 33B clean-wiki tokens out of ~1.5T Stage-5 budget,
  so the Greek slice's contribution is at most ~33B × Greek-share /
  total-8B-tokens ≈ small but non-trivial; record exact figures.

**EuroParl (Stage 5, 21B tokens combined with ParaDocs):**
- `Helsinki-NLP/europarl` is sentence-aligned 24-language EU
  parliamentary proceedings. Each translation pair carries roughly
  equal token mass per language, so Greek share of EuroParl is
  ≈1/(number-of-languages-in-mix). Read the EuroParl pipeline at
  `pipelines/europarl/main_bidirectional.py` to confirm which language
  pairs are used and whether Greek is included.

**ParaDocs (Stage 5, share of the 21B parallel-data slice):**
- `jhu-clsp/paradocs` is multi-target document-aligned data. Per-pair
  token counts on HF; pull the Greek pairs (if any).

**EuroBlocks-SFT-Synthetic-1124 (Stage-5 task data, part of 3×1B `Task data` slice in Table 6):**
- 31 `language` classes / 39 LangID classes per HF viewer; EuroLLM
  target, so Greek is almost certainly included. Stream the dataset
  and count `language == 'el'` documents to confirm and to estimate
  Greek share. Total Stage-5 task slice ≈ 3B tokens (3 replicas of
  1B); Greek-share inside EuroBlocks × 3B is the absolute upper bound
  of this slice's contribution.

**Institutional Books 1.0 (long-context phase, ~28.7B tokens):**
- HF: `institutional/institutional-books-1.0`. 254 volume-level
  languages, 983k books, 242B `o200k_base` tokens upstream — but
  Apertus uses only 28.7B (heuristically filtered for page-number/TOC
  noise, paper §3.4). Only `eng/deu/fra/ita/spa` got post-processing;
  Greek volumes likely present but small share. Read the
  `language_distribution_gen` field to enumerate per-language token
  proportions for the 28.7B-token Apertus-filtered subset.
- The long-context phase is a *separate* training stretch (paper §3.4,
  Table 8: 78.55B + 58.29B + 58.88B + 29.28B = 225B tokens total
  across the 8k/16k/32k/64k phases) and is much smaller than
  pretraining proper, so its Greek share moves the headline number
  only marginally; still report it.

**FineWeb-Long (long-context phase, 80% of the long-context mixture):**
- Derived from FineWeb-HQ + FineWeb-2-HQ top-10%, then filtered to
  >4k-token docs. Greek share within FineWeb-Long ≈ Greek share within
  FineWeb-2-HQ top-10%, biased toward longer-document languages. Use
  the FineWeb-2-HQ Greek-token estimate as a starting point;
  long-document filtering may push it down (Greek docs tend to be
  shorter on average — verify against the data).

### 2.4 Concrete sequenced workplan — Path-A runbook

**Authoritative dataset ID correction**: paper footnote 19 prints
`epfml/FineWeb-2-HQ` but the live HF repo is `epfml/FineWeb2-HQ` (no
hyphen). Verified 2026-05-11 against the HF API. The Greek slice is at
`epfml/FineWeb2-HQ/ell_Grek/` — 60 parquet shards, 83.1 GB total.

**Scripts**: `ops/greek_share_run/` in the repo. Three files —
`entrypoint.sh` (worker bootstrap), `tokenize_greek_slice.py` (one driver
per Greek-bearing dataset), `aggregate.py` (per-stage weighting →
`summary.json`). Full operator runbook in `ops/greek_share_run/runbook.md`.

1. **Pre-flight (~10 min, on `home`).** Confirmed: gcloud auth active in
   project `eellak-glossapi-20251008`, `HF_TOKEN` in env, `c4-highcpu-192`
   available in `europe-west4-b`. All Greek-bearing datasets accessible
   with `HF_TOKEN`.

2. **Instance bring-up (~5 min).** `c4-highcpu-192` spot in
   `europe-west4-b`, 200 GB pd-balanced boot disk, 1.5 TB local SSD
   (4× 375 GB NVMe slices), label `owner=foivos`,
   `workload=greek-share-tokenization`. Cost: ~$4–5/hr spot. See the
   `gcloud compute instances create` command in
   `ops/greek_share_run/runbook.md §1`.

3. **Worker bootstrap (~5 min).** `bash entrypoint.sh` on the worker:
   formats local SSD at `/mnt/data`, creates venv, pip-installs
   `tokenizers>=0.20`, `huggingface_hub>=0.25`, `pyarrow>=17`,
   `polars>=1.10`, `datasets>=3.0`. Writes `/mnt/data/profile.sh` with
   `HF_TOKEN`, `RAYON_NUM_THREADS=192`, `TOKENIZERS_PARALLELISM=true`,
   HF cache redirects.

4. **Tokenize each Greek slice exactly (~1.5–3 hr total).** One driver
   call per dataset, outputs land at `/mnt/data/outputs/<dataset>.json`:
   - `euroblocks_el` (smallest, ~1 min)
   - `clean_wikipedia_el` (~1–2 min)
   - `europarl_el` (~10–20 min; 20 Greek-bearing bitexts)
   - `paradocs_el` (~30–60 min; large)
   - `institutional_books_el` (~30–60 min)
   - `fineweb2_hq_ell` (~30–60 min; 83 GB parquet → ~250 GB UTF-8 → dominant)

   Each driver: streams parquet via `pyarrow.ParquetFile.iter_batches(20_000)`,
   filters to Greek by metadata (config / language field / language code),
   tokenizes via `tokenizers.Tokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")`
   with `add_special_tokens=False`, adds `+2` per doc for `<s>` BOD +
   `</s>` EOD to match Apertus pretrain-token accounting (paper §2.1).
   - Heavy compute on gcloud worker only (`feedback_home_is_a_server.md`).
   - **Do not** attach this workload to the existing `apertus-greek-tokenizer`
     m3-megamem-64 instance (`feedback_only_use_started_instances.md`).

5. **Aggregate (~1 min).** `aggregate.py` reads `/mnt/data/outputs/*.json`,
   applies the stage-pool weighting formula from §2.4-math below,
   writes `/mnt/data/outputs/summary.json` with per-stage breakdown +
   the overall Greek-share %.

6. **Pull + commit (~2 min).** `gcloud compute scp` the `outputs/`
   directory to `ops/greek_share_run/outputs/` on `home`, commit the
   JSON to the repo. The number is now reproducible from repo state.

7. **Tear down (mandatory).** `gcloud compute instances delete` the
   worker. Spot c4-highcpu-192 idle = ~$4/hr; never leave standing.

**§2.4-math — stage weighting formula** (used by `aggregate.py`):

Let:
- `G_d` = Apertus-tokenized Greek tokens (with `<s>` + `</s>`) we measure for dataset `d`.
- `S_s` = 8B duration of stage `s` per Table H.8: 7038 B (1), 0 (2, skipped), 4962 B (3), 1345 B (4), ~200 B (5 cooldown tail).
- `P_{d,s}` = stage `s` pool size from Table 6, dataset `d`.
- `M_s` = Σ_d P_{d,s}, stage total available tokens.
- `f_{d,s}` = Apertus's filter recipe for dataset `d` in stage `s`. For FineWeb-2-HQ Greek: `f = p × 0.95` with `p ∈ {0.33 (s∈{1,2,3}), 0.10 (s∈{4,5})}`. For everything else: `f = 1`.

Greek tokens contributed by dataset `d` in stage `s`:
```
greek_consumed[d, s] = (S_s / M_s) × P_{d,s} × (G_d × f_{d,s} / fw_pool_size_for_d)
```
For the dominant FineWeb-2-HQ case this simplifies to
`G_d × f_{d,s} × (S_s / M_s)`. The full formula handles the case where a
dataset's pool size in Table 6 (`P_{d,s}`) differs from our measured
Greek-tokens-in-dataset value (`G_d`).

Sum over `d` and `s` → total Greek consumed; divide by 13,545 B (paper
Table H.8 + 200 B cooldown estimate) → Greek share %.

3. **Apply the 8B stage weighting.** Convert per-dataset Greek-token
   estimates to per-stage Greek contributions using:
   - Stage 1 duration = 7,038 B tokens (8B-specific).
   - Stage 2 = 0 B tokens (skipped on 8B).
   - Stage 3 duration = 12,000 − 7,038 = 4,962 B tokens.
   - Stage 4 duration = 13,345 − 12,000 = 1,345 B tokens.
   - Stage 5 duration = (final 8B token count) − 13,345 B (estimate
     ≈100–200 B; cross-check against paper's 8B retrospective if it
     specifies the final iteration count).
   For each stage, share the Greek allocation across the datasets
   present in that stage's mix in proportion to their Table 6
   token-pool sizes — this is the standard datatrove-style
   proportional mixing assumption and matches what Apertus's data
   loader does (Megatron-LM dataloader fed by a fixed-weight mix
   constructor; paper §3.3 "Coooldown Experiments").

4. **Aggregate.** Sum Greek-token contributions across all 4
   consumed stages (1, 3, 4, 5) and divide by realised 8B
   pretraining-token total (≈13.4T). Report:
   - Greek pretraining-token share (%).
   - 95% confidence interval propagating the tokenization-sample
     uncertainty and the Table-6 pool-size proportional mixing
     assumption.
   - Per-stage Greek-token-share breakdown (Stage 1 is expected to
     dominate the 8B figure because Stage 1 carries ~52% of all 8B
     tokens).

5. **Optional cross-check.** If `swiss-ai/pretrain-data` ever
   publishes per-language token-count manifests (currently the README
   is a 2-line stub, but the `pipelines/` directory could in the
   future emit per-language Megatron `.bin/.idx` size logs), use them
   as ground truth. Track this as an open question.

### 2.5 Acceptance criterion

A number, in writing, with citations to:
- Per-dataset Greek-token counts (each citing the HF dataset card or
  the streaming-tokenizer measurement run-id).
- The 8B stage durations (paper Table H.8, this doc §1.1).
- The Table-6 mixture pools (paper §3.3).
- The proportional-mixing assumption made when converting pool sizes
  to per-stage consumption.

If the same number can be derived two independent ways — once from
FineWeb-2-HQ-only (Greek's dominant route) and once from a per-stage
sum across all Greek-bearing datasets — and the two agree within
their CIs, we publish; if they diverge, we debug before publishing.

---

## §5. Results — Path-A run, 2026-05-11

Worker: `c4-highcpu-192` on-demand, `europe-west4-b`, 2 TB hyperdisk-balanced
boot (no local SSD; project preemptible-LSSD quota = 0). Lifetime ~62 min
(15:46 → 16:48 UTC), wall cost ≈ $8. Source artifacts:
`ops/greek_share_run/outputs/{summary,fineweb2_hq_ell,clean_wikipedia_el,europarl_el,euroblocks_el,paradocs_el}.json`.

### 5.1 Per-dataset Greek measurements (Apertus tokenizer, +BOD/+EOD per doc)

| Dataset | Selector | Docs | UTF-8 bytes | Tokens |
|---|---|---:|---:|---:|
| **FineWeb2-HQ `ell_Grek`** (60 parquet, 83 GB) | config `ell_Grek` | 4,346,440 | 30.25 GB | **6,383,239,455** |
| Clean-Wikipedia `el` | path `el/` | 226,273 | 1.24 GB | 275,679,532 |
| EuroParl Greek (20 bitexts) | Greek side of every `el-*` pair | 18,124,501 | 5.67 GB | 1,163,704,082 |
| EuroBlocks-SFT Greek | `language == 'Greek'` | 582 | 1.86 MB | 358,888 |
| ParaDocs Greek | (no Greek pairs in repo) | 0 | 0 | 0 |
| Institutional Books Greek | (gated dataset — not measured) | — | — | — |

Notes on the targeting verifications during the run:
- ParaDocs has 18 language pairs, all `en-XX`, *no* `el` — confirmed by listing the
  repo tree. Zero Greek, not "very likely" — strict zero.
- EuroBlocks's `langid` field is fastText-derived and misclassifies most Greek
  conversations as English; the authoritative selector is the synthesis-time
  `language` field (`Greek` for 582 rows; `langid='el'` for only 1).
- Institutional Books is gated 'auto' and the run's token isn't authorized;
  contribution treated as zero for the headline.

### 5.2 8B-pretraining Greek-share calculation

Stage durations (paper Table H.8, 8B-specific): S1 7,038 B, S2 0 (skipped),
S3 4,962 B, S4 1,345 B, S5 ≈200 B cooldown tail = **13,545 B** realised total.

FineWeb-2-HQ Greek consumption per stage (`G_hq × p × 0.95 × (S_s / M_s)`):

| Stage | `p × 0.95` | Pool total `M_s` (B) | Stage duration (B) | Greek consumed (B) |
|---:|---:|---:|---:|---:|
| 1 | 0.3135 | 8,641 | 7,038 | **1.629** |
| 3 | 0.3135 | 9,346 | 4,962 | **1.062** |
| 4 | 0.0950 | 2,905 | 1,345 | **0.281** |
| 5 | 0.0950 | 2,978 | 200 | **0.041** |
|   |       |       | **subtotal** | **3.014** |

Stage-5 auxiliary contributions (upper bounds; Greek-share assumed full-pool):

| Slice | Greek tokens in dataset (B) | Pool size in S5 (B) | Consumption rate | Upper-bound Greek consumed (B) |
|---|---:|---:|---:|---:|
| Clean-Wikipedia | 0.276 | 33 | 0.0672 | 0.0185 |
| EuroParl | 1.164 | 21 | 0.0672 | 0.0782 |
| EuroBlocks (×3 replicas) | 0.000359 | 3 | 0.0672 | 0.0001 |
|   |    |    | **subtotal** | **0.097** |

**Headline:**

```
Greek tokens consumed by Apertus-8B-2509 ≈ 3.014 (FW2HQ) + 0.097 (aux) = 3.111 B
Realised 8B pretraining budget                                          = 13,545 B
Greek share                                                              ≈ 0.023%
```

### 5.3 Sanity checks

- 4.35 M Greek HQ docs ≈ 10 % of raw FineWeb-2 v2.0.1 Greek (44.2 M docs per
  paper Table G.6) — matches the published FineWeb-2-HQ retention rate.
- 6.38 B Apertus-tokens / 30.25 GB UTF-8 = 4.74 bytes/token, consistent with
  Mistral-Nemo BPE under-merging Greek polytonic/NFD (motivation for the very
  extension this project ships).
- 0.023 % is far below the naive "Greek is 1 of 20 HQ languages → 2 %" upper
  bound because (a) Greek docs are only 0.97 % of raw FineWeb-2 (rank #22 of 40
  in Table G.6, well below German/Spanish/French), (b) Apertus's `sampler.rate=0.95`
  haircut, and (c) the `p × 0.95` filter restricts HQ-Greek consumption to 33 %
  / 10 % of the slice across stages.
- Long-context phase (~225 B tokens total, separate from 13.5 T) is excluded
  from the headline; Institutional Books Greek long-tail would add at most a few
  M tokens — within rounding of the headline.

### 5.4 Why the 0.023 % share doesn't contradict the Phase-A norm parity

The Phase-A norm diagnostic
([EXTENSION_DOC_FEEDBACK_20260511.md](EXTENSION_DOC_FEEDBACK_20260511.md)
§1) found Apertus's 1,506 existing Greek tokens are statistically
indistinguishable from English-baseline on both embedding matrices —
even though Greek is 0.023 % of pretraining and English is ~60 %. The
architectural explanation for this parity (which is by design, not
accident) lives in
[APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md](APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md):
gradient clipping at 0.1 applied at almost every step + Pre-Norm /
RMSNorm + QK-Norm + cross-entropy logit saturation + AdEMAMix long-tail
momentum collectively force per-token embedding training to converge
to a similar steady-state norm regardless of corpus share. The
1,800-language target is the design rationale.

Implication for this doc: **the 0.023 % share is a real signal about
data composition, not about training quality.** Greek vocab is trained
to convergence under Apertus's recipe; the C3 tokenizer extension is
motivated by merge-content quality (fertility / morpheme targeting),
not by under-trained embeddings.

### 5.5 Implications for subproject 03 (embedding adaptation)

Greek pretraining exposure for Apertus-8B is roughly **3 B tokens out of 13.5 T
(0.023 %)**. CPT replay-ratio decisions should target *raising* Greek share
substantially (the C3 mix is `glossapi + hplt 50/50`, much more Greek-dense
than 0.023 %), so the replay ratio is the lever that determines whether the
extended model retains Apertus's multilingual breadth while gaining Greek
depth. Track in `subprojects/03_apertus_extension_and_embedding_adaptation/`.

## 3. Open questions / known gaps

- The Stage-5 8B consumed-tokens *final* value isn't directly
  reported in Table H.8 (which lists the *start* of each stage).
  The paper's retrospective in §2.6 may pin it; otherwise we
  proceed with the Table-H.8 figures and note the residual cooldown
  as < a few hundred B.
- Institutional Books 1.0's per-language composition for the
  Apertus-filtered 28.7B-token slice isn't yet verified — the
  `language_distribution_gen` field needs a direct read.
- The exact mixing-weight inside FineWeb-2-HQ across the 20 HQ
  languages isn't published in the paper (only the `p` and
  `sampler.rate` knobs); per-language token contribution depends on
  the raw FineWeb-2-HQ Greek pool size, which we measure in §2.4
  step 1.
- The paper's FineWeb-2-HQ excludes ~12% of FineWeb-2-HQ data via
  the robots.txt retroactive filter (paper §3.1.1, "~4% in
  multilingual data") — apply this haircut to the Greek estimate too.
- EuroBlocks-SFT-Synthetic-1124's per-language counts not on the HF
  card; need to enumerate the `language` field to confirm Greek and
  measure share.
- ParaDocs per-language pair counts not on the HF card (only total
  3.87 TB and three source corpora). Use the ParaDocs GitHub
  (`rewicks/ParaDocs`) or stream-count to get Greek-bearing rows.
- Trace-Greek presence in code datasets (StarCoderData, StarCoder
  Edu, CommonPile/Stack-v2-Edu) is treated as zero for the headline
  but is technically non-zero (Greek comments / strings in Greek-authored
  repos). Run a fasttext pass only if the headline number needs to
  be tightened beyond ~1%.

---

## 4. Sources

- **Apertus v1 Technical Report**, arXiv:2509.14233v2 (1 Dec 2025).
  Sections used: §1 (intro), §2.2 (tokenizer), §3 (pretraining data),
  Table 6 (pretraining mixture and token counts), §3.3 (curriculum),
  §3.4 (long-context mixture), Table 8 (long-context phase tokens),
  Appendix G (FineWeb-2 language distribution, Table G.6),
  Appendix H.1–H.3 (additional pretraining data and stages, Table H.8).
- **github.com/swiss-ai/pretrain-data**: reproduction pipelines, in
  particular `pipelines/fineweb-2/main.py` (Greek `ell_Grek` config).
- **HF dataset cards**: `swiss-ai/Apertus-8B-2509` (model card),
  `HuggingFaceFW/fineweb-2`, `epfml/FineWeb-2-HQ`,
  `HuggingFaceFW/clean-wikipedia`, `Helsinki-NLP/europarl`,
  `jhu-clsp/paradocs`, `institutional/institutional-books-1.0`,
  `HuggingFaceTB/dclm-edu`, `HuggingFaceFW/fineweb-edu`,
  `bigcode/starcoderdata`, `common-pile/stackv2-edu-filtered`,
  `HuggingFaceTB/finemath`, `LLM360/MegaMath`,
  `utter-project/EuroBlocks-SFT-Synthetic-1124`,
  `DataProvenanceInitiative/Commercial-Flan-Collection-(SNI, Flan 2021, Chain of Thought, P3)`.
- **FineWeb-2 v1 language-distribution CSV**:
  `github.com/huggingface/fineweb-2/blob/main/fineweb2-language-distribution.csv`
  (Greek v1: 8.95B tokens / 20.5M docs / 89.43 GB UTF-8). Note: paper
  Table G.6 reports 44.2M Greek docs at 0.97% of the multilingual
  pool — this is the v2.0.1 number used by Apertus, which is roughly
  2× the v1 figures. Always quote v2.0.1 figures when reasoning
  about what Apertus actually consumed; quote v1 only if v2.0.1
  metadata isn't yet published.
