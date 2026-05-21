# 03.2 Apertus × C3 dedup audit — plan

**Status**: plan. Not yet executed. Date: 2026-05-18.

## 1. Goal

For each (Apertus-Greek source × C3-source) pair, quantify
document-level overlap at three levels of strictness, all driven
by the existing `glossapi_corpus_cli text_dedup` pipeline:
- **strict_exact** (blake3 over NFC + whitespace-collapse normalised text)
- **relaxed_exact** (additional lowercase + strip-punctuation + ZWS removal)
- **near-duplicate** (128-perm MinHash, token 5-shingles, Jaccard ≥ 0.85)

The single deliverable is **a per-pair overlap table** plus three
derived numbers:

1. **CPT-fresh fraction**: of C3 train (14.4 M docs / ~100 B chars),
   what fraction is NOT in any Apertus-Greek source?
2. **Held-out contamination**: of C3 val (7,654 docs) + test (7,282
   docs), what fraction was in Apertus pretraining?
3. **Per-source overlap matrix**: which C3 sub-sources contribute
   most overlap with Apertus, and which are entirely fresh?

## 2. The two corpora to compare

### 2.1 Apertus Greek pretraining sources (per [`../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md) §5.1)

| # | Source | HF id | Docs | UTF-8 |
|---|---|---|---:|---:|
| A1 | FineWeb-2-HQ `ell_Grek` | `epfml/FineWeb2-HQ`, config `ell_Grek` | 4,346,440 | 30.25 GB |
| A2 | Clean-Wikipedia `el` | `HuggingFaceFW/clean-wikipedia`, config `el` | 226,273 | 1.24 GB |
| A3 | EuroParl Greek | `Helsinki-NLP/europarl`, 20 `el-*` bitexts | 18,124,501 (sentences) | 5.67 GB |
| A4 | EuroBlocks-SFT Greek | `utter-project/EuroBlocks-SFT-Synthetic-1124`, `language=='Greek'` | 582 | 1.86 MB |

A1 is overwhelmingly dominant (~97 % of Apertus's Greek mass). A3
is sentence-level not doc-level — treat separately.

**Two-axis reporting on Apertus side** (per review rounds
2026-05-18): for each Apertus source we report overlap against C3
on **both**:
- **source-universe**: the full released slice (e.g. all 4.35 M
  FW2-HQ Greek docs). Answers "could this have been seen by the
  model." Conservative — over-removes C3 docs.
- **consumed_exposure_estimate** (the load-bearing number for CPT
  planning, but interpret with care): the *expected
  probability-weighted* overlap, treating Apertus's sampler as
  stochastic with known marginal rates (per-stage `p × 0.95` ×
  stage-duration; math in
  [`../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md) §5.2).

**Crucial honesty caveat — we do NOT have a deterministic consumed-
document manifest.** Apertus's RNG seed for FW2-HQ subsampling, the
exact document order, and the per-document revision pins are not
published. What we measure is "*if Apertus's reported sampler rates
are accurate, what is the expected fraction of overlapping documents
in the released slice that would have been encountered during
pretraining*?" This carries the same uncertainty as the upstream
0.023 % Greek-share estimate
(see [`../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md)
§3 "Open questions / known gaps").

This is **not** a per-document deterministic claim that document X
was seen. It is a probability-weighted EXPECTATION over an
unmeasured stochastic process. Per-doc rows in
`consumed_exposure_estimate` carry a `consumed_probability` column,
not a binary `seen` flag.

For FW2-HQ ell_Grek the expectation differs from source-universe
by ~3–10× (Apertus consumed ~10–33 % of release across stages). For
CPT planning, the consumed_exposure_estimate is the principled
input; the source-universe number sets the upper bound and is
reported alongside for transparency.

### 2.2 C3 tokenizer-training corpus

Per [`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md)
and (authoritative) the canonical `source_dataset_summary.parquet`
emitted by the C3 train manifest. The audit's source list is derived
from that artifact at execution time, not from prose — the prose
inventory and the manifest have historically drifted (review
2026-05-18 found 19 vs 18 vs 17 discrepancies in earlier docs).

Headline numbers — **two distinct row-count universes**, do not
conflate (per review 2026-05-18):

| Pool | HF id | docs in train split | docs in mix (pre-split) |
|---|---|---:|---:|
| C-G | GlossAPI corpus (17 sub-datasets per `source_dataset_summary.parquet`) | **517,791** | 546,920 |
| C-H | HPLT clean60 Greek (`HPLT/ell_Grek_ge8_no_mt_clean60`) | **13,883,763** | 13,906,493 |

The audit must be **explicit about which universe each
measurement uses**:
- **CPT-planning** measurements (fresh-data budget, replay-ratio)
  use the **train** counts (517,791 + 13,883,763 = 14,401,554).
  This is what feeds the model.
- **Held-out contamination** measurements operate on the val
  (7,654) + test (7,282) splits specifically — these are the
  eval-integrity universes.
- Whole-mix measurements (mix.parquet, 14,453,413 rows) are useful
  only for cross-checking the pre-split distribution; not CPT-actionable.

Total source count for the cross-product: **18** (17 GlossAPI + 1
HPLT). With 4 Apertus sources, the full cross-product is **4 × 18 =
72 pairs** (was reported as 84 in an earlier draft — fixed per review).

Mix artifact details (for cross-reference):
- `/home/foivos/data/glossapi_work/.../mix.parquet`: 14,453,413 rows
- Splits: train (14,401,554), val (7,654), test (7,282)

The 17 GlossAPI sub-datasets vary in expected Apertus-overlap:
- **High-overlap candidates**: `Wikisource_Greek_texts`,
  `HuggingFaceFW/finewiki` (Greek config), `eurlex-greek-legislation`,
  `ellinika_dedomena_europaikou_koinovouliou`,
  `AI-team-UoA/greek_legal_code` — all public-web / EU-legislation
  sources also represented in Apertus's Greek pretraining.
- **Low-overlap candidates**: `Apothetirio_Kallipos`,
  `Apothetirio_Pergamos`, `greek_phd` (didaktorika.gr),
  `openarchives.gr`, `dimodis_logotexnia`, `Ekklisiastika_Keimena`,
  `Sxolika_vivlia`, `1000_prwta_xronia_ellhnikhs`,
  `klasikh_arx_ell_grammateia`, `Ellinika_Keimena_Project_Gutenberg`,
  `openbook_gr`, `opengov.gr-diaboyleuseis` — niche Greek-academic /
  literary sources unlikely to be in CC-derived crawls.

**Source list to actually drive the audit will be read from the
manifest at runtime.** The prose list above is a sanity-check anchor
only.

## 3. Methodology

The audit **drives the existing `glossapi_corpus_cli text_dedup`
pipeline in cross-corpus mode** rather than reimplementing md5 +
MinHash from scratch. That pipeline is versioned code we already use
for C3's internal dedup and the polytonic extension's dedup; using
it here means the audit's notion of "overlap" is the same notion
C3 was internally deduped against.

### 3.1 Implementation reference — pinned, recorded, audited

Canonical code: [`../../../glossapi_corpus_cli/text_dedup.py`](../../../glossapi_corpus_cli/text_dedup.py).

**Pinned at run time** (per review 2026-05-18 round 3) — recorded
in `manifests/run_<ts>/text_dedup_pin.json`:

| Field | Value (as of 2026-05-18) |
|---|---|
| Repo commit | **`9a6b039`** (`Add firing count attribution workflow`) |
| File git-hash-object | **`6b9bfdb0bd9923349c348f80866c472101ab8fcf`** |
| Verify at run start | `git rev-parse HEAD` + `git hash-object glossapi_corpus_cli/text_dedup.py`, halt if either differs from pin |

Defaults adopted from the existing code, verbatim, **with one
deliberate user-chosen override**:

| Parameter | Audit value | Code default | Code constant |
|---|---|---|---|
| Near-dup Jaccard threshold | 0.85 | 0.85 | `DEFAULT_NEAR_THRESHOLD` |
| MinHash permutations | **128** | 128 | `DEFAULT_NUM_PERM` |
| LSH bands | 32 | 32 | `DEFAULT_BANDS` |
| LSH rows per band | 4 | 4 | `DEFAULT_ROWS_PER_BAND` |
| Shingle mode | token | token | `DEFAULT_SHINGLE_MODE` |
| Shingle size | 5 tokens | 5 | `DEFAULT_SHINGLE_SIZE` |
| Short-doc behavior | skip token-shingling if `< 20` tokens | same | `SHORT_DOC_TOKEN_THRESHOLD` |
| Shingle hash | blake3 8-byte | same | `shingle_hashes_from_text` |
| **Greek diacritic policy** | **`preserve`** (USER DECISION 2026-05-18) | `preserve` | `DEFAULT_GREEK_DIACRITIC_POLICY` |
| Pipeline stages | `strict_exact → relaxed_exact → near_signatures → near_candidates → near_clusters` | same | `STRICT_STAGE`, `RELAXED_STAGE`, etc. |

**Greek diacritic policy = `preserve`** is a deliberate user
decision (not the code default — the code defaults to preserve but
the live HF publish bundle was built with `strip`). Rationale:
- Apertus's tokenizer has distinct single-token merges for
  monotonic ά (U+03AC) vs the unaccented α (U+03B1). Per the
  `CLAUDE.md` empirical tokenizer facts: bare α, monotonic ά, and
  final sigma ς are all single-token vocab hits; oxia ά (U+1F71),
  psili+oxia ἄ (U+1F04), and combining acute are NOT.
- Preserving diacritics keeps the audit faithful to the text the
  Apertus model would actually tokenize. Stripping would conflate
  "ά" and "α" in our hashes, marking distinct documents as identical
  even though Apertus would see them differently.
- This means the audit is **NOT directly comparing to C3's
  internally-deduped state** (C3 inherited `strip` from the HF
  publish bundle); the audit produces a fresh measurement at
  `preserve` granularity.

**Manifest contents** (`manifests/run_<ts>/text_dedup_pin.json`),
recorded at run-start, per reviewer guardrail:

```json
{
  "text_dedup_commit": "9a6b039",
  "text_dedup_file_hash": "6b9bfdb0bd9923349c348f80866c472101ab8fcf",
  "greek_diacritic_policy": "preserve",
  "num_perm": 128,
  "shingle_version": "shingle_token_5_v1",
  "exact_strict_version": "exact_strict_norm_v1",
  "exact_relaxed_version": "exact_relaxed_norm_v5_preserve",
  "near_norm_version": "near_norm_v5_preserve",
  "source_universe": {
    "apertus": ["FW2-HQ ell_Grek", "Clean-Wikipedia el", "EuroParl Greek", "EuroBlocks Greek"],
    "c3": "train_split_only (17 GlossAPI sources + HPLT, from source_dataset_summary.parquet)"
  }
}
```

If any pin doesn't match at start, halt with a clear "policy drift
detected — reconcile before running" message rather than silently
producing inconsistent hashes.

### 3.2 Pipeline stages used

The audit runs four detection stages per source-pair, in order of
increasing recall and decreasing strictness. All four are emitted
to artifacts:

1. **`strict_exact`** — case-, whitespace-, NFC-normalized blake3
   hash. Matches docs that are bit-for-bit identical after light
   normalization. (Replaces my earlier "md5" framing — the
   existing pipeline uses blake3 + a stricter normaliser.)
2. **`relaxed_exact`** — additional normalisation: lowercase,
   strip-punctuation, ZWS/ZWJ/ZWNJ/BOM removal, line-wrap dehyphen.
   Catches docs that differ only in punctuation or case.
3. **`near_signatures`** — 128-perm MinHash signatures of token
   5-shingles, blake3-hashed.
4. **`near_candidates` + `near_clusters`** — LSH-banded candidate
   generation + actual Jaccard ≥ 0.85 validation.

### 3.3 Cross-corpus mode

The dedup pipeline is normally run within-corpus (find duplicates
in a single mix). For the audit, we run it in **cross-corpus mode**:
the input is two corpora tagged with `corpus_id ∈ {apertus, c3}`,
and the output cluster table flags whether each cluster spans
corpora.

Outputs per pair (in Parquet, not JSON — see §5):
- `strict_exact_<a_source>_x_<c_source>.parquet`
- `relaxed_exact_<a_source>_x_<c_source>.parquet`
- `near_overlap_<a_source>_x_<c_source>.parquet`
  with columns `(a_doc_id, c_doc_id, jaccard, length_ratio,
  cluster_id)`.

### 3.4 Pair-specific tweaks

- **A3 EuroParl vs C-G `eurlex-greek-legislation`**: EuroParl is
  sentence-aligned; eurlex is doc-level. Compare at the sentence
  level — split eurlex docs into sentences using the same
  segmentation pipeline that EuroParl uses (or NLTK Greek tokenizer
  if EuroParl's segmenter is upstream-fixed). Then run all four
  stages above at sentence granularity. Report % of EuroParl
  sentences contained in eurlex docs.
- **A3 EuroParl vs C-G `ellinika_dedomena_europaikou_koinovouliou`**:
  same. This GlossAPI source is "EU Parliament Greek data" — high
  a priori overlap with EuroParl.
- **A2 Clean-Wikipedia el vs C-G `HuggingFaceFW/finewiki`**: both are
  Greek Wikipedia. **Expected near-100 % overlap** (modulo revision
  drift between snapshots). Report exact-vs-relaxed-vs-near gap; if
  exact is low and relaxed/near is high, that's a Wikipedia-revision
  drift finding.

### 3.5 Held-out contamination check — full ladder

C3 val/test splits are 7,654 + 7,282 docs. **Use the full detection
ladder, not just strict-exact** (per review 2026-05-18):

For every (val_doc OR test_doc) × every Apertus source, run all
four stages — strict_exact, relaxed_exact, near_signatures,
near_clusters. Plus sentence-level matching against EuroParl
specifically (sentence-segment val/test docs, then strict + near
against EuroParl sentences).

Output: `holdout_contamination.parquet` with one row per
(c3_doc_id, a_source, stage, match_strength, a_doc_id). A val/test
doc is flagged contaminated if matched at ANY level. The strictest
level it matches at is its contamination severity.

This is a **gating check** for downstream CPT eval — if non-trivial
contamination is found, val/test get rebuilt by anti-joining the
Apertus side at the strictest-matching level.

### 3.6 Per-doc overlap-ratio + action rule for partial overlap

(Added per review 2026-05-18, refined 2026-05-18 round 2 — the
original plan defined comparisons but not the action rule for
partial overlap. Critical for the CPT recipe.)

For every C3 document with any match against any Apertus source,
compute `overlap_ratio ∈ [0, 1]` using these per-match-type rules:

| Match type | overlap_ratio rule | Reason |
|---|---|---|
| `strict_exact` whole-doc match | **1.0** | Doc is bit-for-bit duplicate after light normalisation |
| `relaxed_exact` whole-doc match | **1.0** | Doc is duplicate after fuller normalisation (case + punctuation) |
| `near` whole-doc match (Jaccard ≥ 0.85) | **1.0** | By the 0.85-Jaccard definition the doc IS already a near-duplicate; partial-credit math would be misleading |
| `sentence-level` matches (EuroParl-style) | **`union(matched_char_spans) / total_doc_chars`** | A long legal doc that quotes one EuroParl sentence has low ratio; one that's mostly EuroParl text has high ratio |

When a doc matches at multiple match types, the final
`overlap_ratio` is `max` across types. A doc that's both a relaxed-
exact whole-doc duplicate AND has scattered sentence-level matches
takes ratio 1.0.

The default 3-tier classification (sensitivity-tunable — see below):

| Tier | overlap_ratio | Action for CPT |
|---|---|---|
| **drop** | ≥ 0.30 | C3 doc is treated as ALREADY-EXPOSED. Drop from the fresh-data budget; goes into the replay pool. |
| **partial** | 0.05 – 0.30 | C3 doc is treated as PARTIALLY-EXPOSED. Counted at 0.5× weight in the fresh budget; flagged for review. |
| **trace** | < 0.05 | C3 doc is treated as effectively FRESH. Full weight in the fresh budget. |

**These thresholds are NOT a recipe truth — they are a default
sensitivity setting.** Per review 2026-05-18 round 2: REPORT.md
must include a sensitivity-analysis table showing fresh-row
totals + fresh-token totals under at least **three** threshold
settings. Suggested grid:

| Setting | drop ≥ | partial ≥ |
|---|---|---|
| Strict | 0.10 | 0.02 |
| Default | 0.30 | 0.05 |
| Lenient | 0.50 | 0.10 |

The user picks the recipe-relevant threshold from this sensitivity
table; the audit does not pre-decide.

**Why thresholds, not strict 0/1 on sentence matches**: a long
legal C3 document that happens to quote one EuroParl sentence is
not "exposed to Apertus" in any meaningful sense. The threshold
treats a doc as exposed only if a meaningful fraction of its
content is in Apertus's released corpus.

**Note on naming**: we use "EXPOSED" rather than "SEEN" throughout
to honour the §2.1 caveat — even consumed-exposure is probabilistic
expectation, not deterministic per-doc evidence.

## 4. Sources of overlap (a priori predictions)

Before measuring, expected hotspots:

| C3 source | Most-overlapping Apertus source | Reason |
|---|---|---|
| HPLT clean60 | FW2-HQ ell_Grek | Both CC-derived web crawl |
| `finewiki el` | Clean-Wikipedia el | Same Wikipedia |
| `Wikisource_Greek_texts` | (none directly) | Wikisource is in finewiki/Wikipedia but not Clean-Wikipedia |
| `eurlex-greek-legislation` | EuroParl Greek | Same EU legal corpus |
| `ellinika_dedomena_europaikou_koinovouliou` | EuroParl Greek | Same EU Parliament corpus |
| `Apothetirio_Kallipos` | (none) | Greek academic textbooks; not in CC |
| `Apothetirio_Pergamos` | (none) | UoA digital library; not in CC |
| `greek_phd` (didaktorika.gr) | (none) | Theses; not in CC |
| `openarchives.gr` | (small via FW2-HQ) | Academic aggregator; possibly partly crawled by CC |
| `dimodis_logotexnia` | (none) | Folk literature; local scrape |
| `Ekklisiastika_Keimena` | (none) | Ecclesiastical texts; not in CC |
| `Sxolika_vivlia` | (none) | School books; local scrape |
| `1000_prwta_xronia_ellhnikhs` | (small via FW2-HQ) | Ancient/historical Greek |

Predicted high-overlap fraction: ~30-40 % of C3 docs (mostly the
HPLT pool). Predicted CPT-fresh fraction: ~60-70 %. To be measured.

## 5. Outputs

All artifacts are **Parquet** (zstd-compressed, matching the existing
text_dedup pipeline's `PARQUET_COMPRESSION` default). No JSON for
bulk-row data.

```
artifacts/
├── sources/                                            — per-source intermediate tables (strict + relaxed exact + MinHash sigs + LSH bands)
│   ├── apertus_fw2hq_ell_grek/                         — sub-dir, multiple shards
│   ├── apertus_cleanwiki_el/
│   ├── apertus_europarl_greek/                         — sentence-level
│   ├── apertus_euroblocks_greek/
│   ├── c3_glossapi_<17 sub-datasets>/                  — names read from source_dataset_summary.parquet
│   └── c3_hplt_clean60/
├── overlap/
│   ├── strict_exact/<a_source>_x_<c_source>.parquet    — full cross-product (72 pairs)
│   ├── relaxed_exact/<a_source>_x_<c_source>.parquet
│   └── near/<a_source>_x_<c_source>.parquet            — Jaccard ≥ 0.85
├── consumed_estimate/                                  — Apertus-sampler-reconstructed overlap (§2.1)
│   └── <a_source>_consumed_universe.parquet            — doc_id list under Apertus's stage-weighted sampler
├── per_c3_doc_overlap.parquet                          — one row per C3 doc with any match
│                                                         columns: c3_doc_id, c3_source, overlap_ratio,
│                                                                  best_match_stage, best_match_a_source,
│                                                                  tier ∈ {drop, partial, trace}
├── holdout_contamination.parquet                       — C3 val/test contamination ladder (§3.5)
├── summary_matrix.parquet                              — per-pair overlap counts (strict + relaxed + near)
├── per_c3_source_actionable.parquet                    — per-C3-source CPT recipe inputs (§9)
└── manifests/                                          — input fingerprints, code commit pins, run-id
REPORT.md                                                — narrative + CPT implications
```

The `per_c3_doc_overlap.parquet` and `per_c3_source_actionable.parquet`
are the load-bearing outputs for the CPT recipe (per review 2026-05-18
"final output should be C3-side actionable").

## 6. Compute — multi-worker gcloud architecture

The hashing + MinHash work is embarrassingly parallel across input
shards. Run it as a **fan-out across 8 × `c4-highcpu-192` spot
workers in `europe-west4-b`**, matching the proven pattern of
`apertus-vocab-attr-w0..w7` (per session memory). Total wall-clock
~45 min; total cost ~$80.

### 6.1 Worker layout

8 workers, named `apertus-dedup-w{0..7}-<YYYYMMDDtHHMMSS>z`,
labelled `owner=foivos`, `workload=apertus-c3-dedup`. Each:
- `c4-highcpu-192`: 192 vCPUs (no GPU)
- 200 GB pd-balanced boot
- 1.5 TB local SSD (`/mnt/data`) — staging for parquet downloads
- Spot pricing in `europe-west4-b` (~$4-5/hr per worker)

`home` is the coordinator: orchestrates start, dispatches per-worker
configs, polls for completion, pulls artifacts, runs the joins,
tears the workers down. **`home` does not handle the bulk
hashing** — heavy I/O routes to the workers per
`feedback_no_heavy_compute_on_laptop.md` and
`feedback_utilize_available_compute.md`.

### 6.2 Storage budget (revised per review 2026-05-18)

Earlier draft underestimated. Realistic budget:

| Artifact | Size estimate |
|---|---:|
| Source parquet downloads (apertus FW2-HQ + cleanwiki + europarl) | ~40 GB compressed |
| C3 mix parquet (already on disk) | ~50 GB compressed |
| MinHash signatures (128 perms × 8 bytes × ~20 M docs across all corpora) | **~20 GB raw** |
| LSH band hashes (32 bands × 8 bytes × 20 M docs) | ~5 GB |
| Strict + relaxed exact hash tables | ~3 GB total |
| Cross-product overlap parquets (72 pairs, mostly small) | ~5 GB |
| `per_c3_doc_overlap.parquet` (rows × 100 B × ~14 M C3 docs with any match) | ~1 GB |
| Logs, manifests, summary matrix | < 100 MB |
| **Total during run** | **~75 GB** |
| **Final after sig cleanup** | **~15 GB** |

**Strategy**: keep full 128-perm sigs on the workers' local SSD
during the run; persist only LSH band hashes + per-cluster
representatives to the coordinator-side artifacts. The full sigs
are reconstructable from the source data, so discarding them
post-join is safe. This cuts final disk footprint by ~20 GB.

**Per-worker `/mnt/data` budget**: each worker handles a
bytes-balanced subset, so any single worker's intermediate state is
≤ 15 GB. Well within the 1.5 TB local SSD.

### 6.3 Partition strategy — shard-uniform with source-tagging

Workers process parquet shards in balanced batches, regardless of
which source the shard belongs to. Each row carries a
`source_id` tag so the coordinator can re-group later. Workload:

| Source | Approx shards | Approx bytes | Workload share |
|---|---:|---:|---:|
| FW2-HQ `ell_Grek` (Apertus) | 60 | 30 GB (compressed) / 83 GB (uncompressed) | dominant |
| Clean-Wikipedia `el` (Apertus) | 3 | 0.6 GB | small |
| EuroParl Greek (Apertus) | 20 bitexts | 5.7 GB | medium |
| EuroBlocks Greek (Apertus) | 1 | 2 MB | trivial |
| GlossAPI corpus (C3) | 19 sub-datasets, ~50 shards | ~2 GB | medium |
| HPLT clean60 (C3 sampled subset) | ~70 of 250 parquets | ~50 GB | dominant |

Shards bucketed into 8 worker groups by bytes-balanced partition
(use a greedy `pyarrow`-based byte-count → bin-pack assignment).
FW2-HQ + HPLT dominate the workload; the small sources can ride
along on any worker.

### 6.4 Per-worker pipeline

Each worker:

1. **Bootstrap** (~3 min): format `/mnt/data`, install
   `pyarrow>=17`, `polars>=1.10`, `huggingface_hub>=0.25`,
   `datasketch` (for MinHash), `unicodedata2`. Set
   `RAYON_NUM_THREADS=192` if any Rust component is used.
2. **Pull assigned shards** (~5-10 min): `huggingface-cli download
   --revision <pinned-rev>` for each assigned dataset shard. Cache
   at `/mnt/data/sources/<dataset>/`.
3. **Hash pass** (~15-20 min): stream each parquet via
   `pyarrow.ParquetFile.iter_batches(20_000)`, in parallel across
   192 cores (one batch per core). Per row, emit (matching
   `text_dedup.py` stage outputs — see §3):
   - `strict_exact_hash` (blake3 8-byte digest of NFC +
     whitespace-collapse normalised text)
   - `relaxed_exact_hash` (blake3 8-byte digest of additionally
     lowercased + punctuation-stripped + ZWS-removed text)
   - `minhash_sig` (128-perm MinHash signature on token 5-shingles,
     blake3-hashed; skip token-shingling if doc has < 20 tokens
     per `SHORT_DOC_TOKEN_THRESHOLD`)
   - `lsh_band_hashes` (32 bands × 4 rows, derived from sig)
   - `char_len`, `byte_len`, `token_count`
   - `source_id`, `doc_id`, `corpus_id ∈ {apertus, c3}`
   Output: `/mnt/data/output/worker_<i>.parquet` (zstd-compressed,
   matching `text_dedup.py`'s `PARQUET_COMPRESSION`).
4. **Upload** (~2-3 min): `gsutil cp` (or `scp -C`) the worker
   output to a coordinator-accessible location:
   `gs://eellak-glossapi-20251008-dedup/run_<ts>/worker_<i>.parquet`
   (or to home via `scp` if no GCS bucket exists yet).
5. **Sentinel**: touch `/mnt/data/output/_done` to signal completion.

### 6.5 Coordinator (home) phases

1. **Pre-flight** (~5 min): confirm `gcloud auth`, generate
   `run_<ts>` directory, write the 8 per-worker configs to
   `manifests/worker_<i>.json` based on the bytes-balanced
   partition.
2. **Spin up** (~5 min): `gcloud compute instances create` for the
   8 workers, in parallel. Use spot mode. Apply labels for
   tear-down.
3. **Dispatch** (~2 min): scp `worker_bootstrap.sh` + per-worker
   config to each worker, run-and-detach the bootstrap.
4. **Poll** (~30 min, mostly waiting): every 60 s, ssh-query each
   worker for `/mnt/data/output/_done`. Track progress via a
   coordinator-side state file. Surface per-worker logs in real
   time.
5. **Pull and concat** (~5 min): once all 8 workers report done,
   pull the 8 output parquets back to home, concatenate into
   per-source tables.
6. **Joins on home** (~10-15 min): three sets of joins, all using
   the existing `text_dedup.py` join logic in cross-corpus mode:
   - `strict_exact_hash` INNER JOINs across (apertus, c3) corpora
     for each source-pair.
   - `relaxed_exact_hash` INNER JOINs similarly.
   - LSH-banded MinHash candidate generation + Jaccard ≥ 0.85
     validation.
   Inputs aggregate to ~3-5 GB per source after sig-cleanup; fits
   in-memory with `polars` lazy/streaming joins. Emit:
   - `artifacts/overlap/strict_exact/<a>_x_<c>.parquet`
   - `artifacts/overlap/relaxed_exact/<a>_x_<c>.parquet`
   - `artifacts/overlap/near/<a>_x_<c>.parquet`
   - `artifacts/holdout_contamination.parquet`
   - `artifacts/per_c3_doc_overlap.parquet`
   - `artifacts/per_c3_source_actionable.parquet`
7. **Synthesis** (~5 min): generate `summary_matrix.parquet` +
   `REPORT.md` from the join outputs.
8. **Tear down** (CRITICAL — ~2 min):
   `gcloud compute instances delete dedup-w{0..7}-<ts> --zone=europe-west4-b`
   for all 8 workers. Compute SA cannot self-delete
   (`gcloud_compute_sa_no_delete.md`); the coordinator MUST do
   this. Verify zero instances remain with the
   `workload=apertus-c3-dedup` label.

### 6.6 Cost and supervision discipline

- Per-worker: ~$4-5/hr spot × ~0.75 hr ≈ $3-4 per worker.
- 8 workers: **~$25-30 active**, plus ~$5 overhead for boot /
  shutdown / SSD storage.
- **Total wall-clock**: ~45 min (5 spin-up + 30 work + 10 joins).
- **Total cost**: ~$30-40.

Per `feedback_supervise_dont_just_act.md` and the gcloud cost
discipline rule in `glossapi_tokenizer_methodology.md` §9: the
coordinator does NOT fire-and-forget. The polling loop surfaces
per-worker stage updates back to the human operator. If any
worker enters an unexpected state, halt and inspect before
proceeding.

### 6.7 Kill-switch

A single command must cleanly tear down all workers at any time:

```bash
gcloud compute instances delete \
  $(gcloud compute instances list --filter="labels.workload=apertus-c3-dedup AND labels.run=<ts>" --format="value(name)") \
  --zone=europe-west4-b --quiet
```

Wired into the coordinator as a SIGINT handler. Tested with a
dry-run before the real run.

### 6.8 What stays on home

After all workers tear down:
- All artifacts under `artifacts/` (a few GB)
- All hash tables + summary matrices
- `REPORT.md`

Total final disk footprint on home: ~5 GB.

### 6.9 Leverage existing dedup metadata — with guardrails

The HF nanochat release bundles dedup metadata that *partially*
overlaps with what the audit needs. Reusing it can cut C3-side
hashing work — but ONLY where the policy + version pin match.

**Live published bundle** at
`/home/foivos/data/glossapi_work/hf_release_publish/dedup_metadata/full_publish_aws_strip_20260328T101312Z/`:

- `decision_rows = 717,265`, `kept_rows = 674,411`,
  `dropped_rows = 42,854` (per `final/run_summary.json`)
- Manifest (per `builder_metadata/manifest.json`):
  - `num_perm = 128`, `shingle_version = shingle_token_5_v1`,
    `bands = 32`, `rows_per_band = 4`, `candidate_score_floor = 0.85`
  - `exact_strict_version = exact_strict_norm_v1`
  - `exact_relaxed_version = exact_relaxed_norm_v5_strip` ⚠
  - `near_norm_version = near_norm_v5_strip` ⚠
  - **`greek_diacritic_policy = strip`** ⚠ — mismatch with audit's `preserve`

**Reuse decision matrix** (per reviewer guardrails, 2026-05-18 r3):

| C3-side stage | Live bundle version | Audit version | Reuse? |
|---|---|---|---|
| `strict_exact` | `exact_strict_norm_v1` | `exact_strict_norm_v1` | **YES** — policy-independent; same hashes. |
| `relaxed_exact` | `exact_relaxed_norm_v5_strip` | `exact_relaxed_norm_v5_preserve` | **NO** — different normalisation, different hashes. Must recompute. |
| `near` (MinHash sigs + LSH bands) | `near_norm_v5_strip` | `near_norm_v5_preserve` | **NO** — same reason. Must recompute. |

So we can save ~25% of the C3-side hashing work by ingesting the
strict_exact memberships from the published bundle and skipping
that one stage on workers. Relaxed + near get fresh computation
under `preserve`.

**Coverage check before any reuse**: the published bundle covers
the GlossAPI nanochat release universe (~717 K decision rows from
the broader 810 K raw input). The C3 train subset (517,791 rows) is
a strict subset, but verify row-level coverage before reuse:

```
required = set(C3-train glossapi doc_ids)
covered  = set(published bundle doc_ids where corpus = glossapi)
gap      = required - covered
```

If `gap` is empty → reuse strict_exact memberships directly.
If `gap` is non-empty → either recompute strict_exact for the gap
docs, or recompute strict_exact entirely (simpler; <30 % of total
hashing cost).

The HPLT side of C3 (13.88 M docs) is NOT covered by the published
bundle (the bundle is GlossAPI nanochat only), so HPLT strict_exact
must be computed fresh regardless.

**Apertus side** has no published bundle. All Apertus-side
hashing is fresh.

**Run-start checklist for reuse**:

1. Verify `git rev-parse HEAD == 9a6b039` and
   `git hash-object glossapi_corpus_cli/text_dedup.py == 6b9bfdb...`.
2. Read `hf_release_publish/dedup_metadata/latest.json` → confirm
   pointer is to `full_publish_aws_strip_20260328T101312Z`.
3. Read that bundle's `builder_metadata/manifest.json` → confirm
   `num_perm`, `shingle_version`, `exact_strict_version`,
   `greek_diacritic_policy = strip`.
4. Apply reuse decision matrix above. Record what was reused and
   what was recomputed in `manifests/run_<ts>/reuse_log.json`.
5. If ANY mismatch surfaces (different `num_perm`, different
   `shingle_version`, missing files), do NOT reuse — recompute
   that stage in full and note the reason.

**What this saves**:
- C3-side strict_exact: ~5 min of hashing eliminated.
- All other stages: unchanged.
- Saves ~10% of total worker wall-clock; minor but worthwhile, and
  more importantly ensures the audit's C3 strict_exact matches
  what the HF release uses (verifiable equivalence).

## 7. Implementation outline — coordinator + worker split

Two execution domains:

### 7.1 Coordinator scripts (run on `home`)

```
scripts/coordinator/
├── 00_pre_flight.py                  — verify gcloud auth, write run_<ts>/
├── 01_bytes_balanced_partition.py    — read parquet metadata, bin-pack
│                                       shards into 8 worker manifests
├── 02_spin_up_workers.sh             — gcloud instances create × 8 in parallel
├── 03_dispatch_bootstrap.sh          — scp + ssh run-and-detach per worker
├── 04_poll_and_collect.py            — supervisor loop: poll, log, fetch on done
├── 05_concat_per_source.py           — group worker outputs by source_id
├── 06_exact_overlap_join.py          — INNER JOIN strict + relaxed exact hashes (blake3, single-machine)
├── 07_minhash_overlap_lsh.py         — LSH band join + Jaccard ≥ 0.85 validation (128 perms)
├── 08_holdout_contamination_check.py — focused on C3 val/test
├── 09_build_summary_report.py        — emits REPORT.md
└── 99_teardown.sh                    — gcloud instances delete × all workers
```

### 7.2 Worker scripts (deployed to each `c4-highcpu-192`)

```
scripts/worker/
├── bootstrap.sh                      — apt deps, venv, pip installs,
│                                       /mnt/data setup, hf_token env
├── pull_assigned_shards.py           — huggingface-cli download per config
├── hash_pass.py                      — parallel strict_exact + relaxed_exact (blake3) + 128-perm
│                                       MinHash sigs + LSH band hashes per parquet, one batch
│                                       per core via concurrent.futures; matches text_dedup.py defaults
├── upload_output.sh                  — gsutil cp or scp to coordinator
└── _done_sentinel.sh                 — touch /mnt/data/output/_done
```

### 7.3 Manifests and state

```
manifests/
├── run_<ts>/
│   ├── partition.json                — bytes-balanced shard assignment
│   ├── worker_<0..7>.json            — per-worker config (shards, source map)
│   ├── workers.list                  — gcloud names + IPs for teardown
│   ├── progress.jsonl                — append-only worker state stream
│   └── teardown_log.txt              — verification of zero remaining instances
```

Every script is single-purpose, deterministic, re-runnable. Worker
outputs are cacheable on the coordinator side — if a single worker
fails, re-dispatch only that one rather than the whole sweep.

### 7.4 Idempotency and re-runs

- Run-id is the timestamp `<ts>`; every artifact is namespaced
  under `run_<ts>/`. Re-runs use a fresh `<ts>`.
- Per-worker outputs are content-addressed by `(source_id, shard_id)`;
  the coordinator can detect partial-write corruption and re-issue
  to a fresh worker.
- The exact + MinHash join steps are pure functions of the
  concatenated per-source tables; they can be re-run independently
  of the worker phase.

## 8. Risks

1. **Normalization drift across pipelines**: FW2-HQ, HPLT clean60,
   and GlossAPI each ran their own cleaning. The same source
   document could have different text in different pipelines after
   whitespace / quote / dash normalization. `strict_exact` will
   under-count overlap, and `relaxed_exact` partially mitigates;
   MinHash near-dup at 0.85 Jaccard catches the rest. **Mitigation**:
   the full three-stage ladder (strict + relaxed + near) is the
   reason this audit doesn't rely on any single match-type. Report
   all three numbers per pair.
2. **EuroParl sentence-level alignment**: EuroParl is parallel
   sentences, not docs. Comparing sentence-level hashes against
   document-level hashes would always miss. **Mitigation**:
   sentence-segment the GlossAPI eurlex / EU-parliament subsets
   before comparing (per §3.4 pair-specific tweaks).
3. **HPLT version drift**: We use `hplt-greek-ge8-no-mt-clean60-wave4`
   which is HPLT 2.0 + our quality filters. FW2-HQ uses FineWeb-2
   v2.0.1, which is its own CC-derived pipeline. Both touch CC, but
   the specific CC snapshots differ. Overlap will be document-level,
   not URL-level. **Mitigation**: use MinHash for fuzzy doc match;
   don't try URL deduplication unless URLs are preserved (they aren't,
   in HPLT clean60 metadata).
4. **`finewiki el` vs Clean-Wikipedia `el` may not be identical**:
   different Wikipedia snapshots (different revision IDs). Expected
   near-100 % near-dup overlap, but `strict_exact` may show diffs.
   **Mitigation**: report all three stages (strict / relaxed / near)
   per pair; if `strict_exact` is low and `near` is high, that's a
   Wikipedia-revision-drift finding worth recording.
5. **Compute scope creep**: full cross-product is 4 × 18 = 72 pairs
   (was reported as 84 in an earlier draft — corrected per review
   2026-05-18; actual C3 source count from `source_dataset_summary.parquet`
   is 17 GlossAPI + 1 HPLT = 18). Many pairs will be near-zero
   (e.g., A4 EuroBlocks Greek 582 docs ∩ anything). **Mitigation**:
   short-circuit pairs with mismatched size by 1000× (if `min(|A|,
   |C|) < max / 1000`, skip MinHash and report only strict_exact).
6. **EuroBlocks is synthetic data**: it's Llama-3.1-70B generated.
   Some "Greek text" may be near-duplicates of real Greek web text
   the generator imitated. Unlikely but possible. Document as
   curiosity if any matches.

## 9. Success criteria

1. **All 72 pairs measured** at strict_exact + relaxed_exact + near
   levels (or short-circuited per Risk 5).
2. **Held-out contamination check passes the full ladder** (strict
   + relaxed + near + EuroParl-sentence) at ≤ 0.5 % per Apertus
   source — or contamination docs are identified for held-out
   rebuild by anti-join at the strictest matching level.
3. **REPORT.md delivers** both numbers per review 2026-05-18:
   - **Source-universe overlap** (against full Apertus released
     slices). Conservative upper bound.
   - **Consumed-estimate overlap** (against the
     Apertus-sampler-reconstructed subset). The CPT-relevant number.
4. **`per_c3_source_actionable.parquet`** delivers, for every C3
   source, the C3-side actionable numbers per review 2026-05-18:
   - `fresh_rows` (count of C3 docs in `trace` tier)
   - `partial_rows` (count in `partial` tier)
   - `seen_rows` (count in `drop` tier)
   - `fresh_chars`, `fresh_apertus_tokens` (Apertus-tokenized)
   - `recommended_action` ∈ {include_full, include_half_weight,
     replay_only}
5. **Reproducible**: every script is deterministic, every artifact
   has a manifest with input hashes, the text_dedup commit pin is
   recorded.
6. **All 8 workers cleanly torn down** after the run — zero
   instances remain with `workload=apertus-c3-dedup AND run=<ts>`
   labels. Verified by `99_teardown.sh` and logged to
   `manifests/run_<ts>/teardown_log.txt`. Per
   `feedback_instance_stop_decision.md`: paid idle compute
   compounds; the coordinator's job is not done until verification.

## 10. Open design choices before execution

- **Per-sub-dataset breakdown granularity within GlossAPI?** The
  audit's natural cross-product is 4 Apertus × 18 C3 sources = 72
  pairs (17 GlossAPI + 1 HPLT, sourced from
  `source_dataset_summary.parquet`). Could collapse to just
  "GlossAPI as one pool" + HPLT = 8 pairs for a faster first pass;
  per-sub-dataset breakdown adds detail but quadruples join count.
  Start with the full 72-pair breakdown since the dominant cost is
  the upstream hashing pass, not the downstream joins.
- **Whether to dedup the 34.7 M HPLT-clean60 docs NOT in the C3 mix
  too**: the broader HPLT release is what Apertus might also have
  overlap with, even though C3 only sampled 28.6 %. For CPT
  planning, only the C3 mix matters (since that's what feeds
  CPT). Skip the unsampled HPLT for now; revisit only if eval
  contamination check requires it.
- **Whether to include Apertus's `apertus-pretrain-romansh` and
  `apertus-pretrain-swiss` released-but-not-used corpora**: these
  are NOT in v1 pretraining (Appendix H.2), so the model did not
  see them. Exclude.
- **Optional secondary measurement — HF-release × Apertus overlap**
  (added per review 2026-05-18 r3). The audit's primary universe is
  C3 train (517,791 + 13,883,763 docs). A secondary measurement
  could dedup the **broader HF nanochat release** (~810,638 raw →
  ~674,411 kept after publish-bundle exact dedup, all GlossAPI
  sources, no HPLT) against Apertus. This tells future consumers
  of the HF release how much of the broader Greek-nanochat corpus
  overlaps with Apertus pretraining — useful for third-party
  tokenizer extensions or other Greek-corpus users, but not
  directly CPT-relevant for this run. **Keep this explicitly
  SECONDARY**: produce the C3-train × Apertus numbers first; only
  run HF-release × Apertus if there's spare worker time. Output
  goes to a separate `artifacts/secondary_hf_release/` directory
  so it doesn't get conflated with the load-bearing CPT inputs.
- **Whether to also audit against Apertus's long-context phase
  (FineWeb-Long + Institutional Books, ~225 B tokens)**: small
  Greek share, Institutional Books is gated. Skip in first pass.
  REPORT.md must then state scope explicitly as **"Apertus
  main-pretraining + cooldown only; long-context phase excluded"**
  — not "everything the model saw." Per
  [`../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md)
  §3.4 / Institutional Books has 254 volume-level languages,
  including some Greek; the long-context phase Greek share is
  non-zero but bounded. Revisit if 03's long-context handling
  becomes a planning issue.

## 11. What happens after this audit

The output `REPORT.md` feeds three downstream decisions in 03:

1. **CPT replay ratio**: how much Apertus-pretraining-style data to
   mix back into CPT to prevent catastrophic forgetting on the
   "already-seen" Greek. The fresh-vs-seen split tells us this.
2. **Eval slice integrity**: if held-out contamination is
   non-trivial, rebuild C3 val/test by anti-joining the Apertus
   side at the strictest matching level (per §3.5 full-ladder
   check — typically `strict_exact` if hits are present there;
   else `relaxed_exact`; else `near` at Jaccard ≥ 0.85).
3. **CPT data volume claim**: when we report "Apertus has now been
   continued-pretrained on +X B Greek tokens," the X needs to be
   the FRESH fraction, not the raw C3 char count.

The audit doesn't pre-decide any of these; it just provides the
quantitative inputs that make the decisions principled.
