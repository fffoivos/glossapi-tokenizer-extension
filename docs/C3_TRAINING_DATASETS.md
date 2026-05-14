# C3 training datasets

What went into the BPE training of
`C3_wave2_broad_glossapi_plus_hplt_50_50`. See
[C3_CONVERGENCE.md](C3_CONVERGENCE.md) for the convergence statement
and [C3_CUTOFF_REPORT.md](C3_CUTOFF_REPORT.md) for the cutoff sweep.

## Headline numbers

- Mix recipe: **`glossapi + hplt`** at **`50 / 50` by training-token mass**
- Training rows: **14,401,554** docs (train split only)
- Training chars: **≈ 100 B** (50 B per pool)
- Cleaner: **wave-2 broad cleaner**
  (`codex/cleaner-audit-counters-20260506` of `eellak/glossAPI`)
- Quality gates applied at mix-build: `greek_badness_score < 60`,
  `mojibake_badness_score ≤ 0.1`, `charset_greek_ratio ≥ 0.5`,
  non-empty post-clean, `openarchives.gr` `needs_ocr == False`
- Output tokenizer: base `131,072` (Apertus) + added `25,600` = total
  `156,672` vocab; trained on the gcloud instance, runtime ≈ 14,446 s
  (4.0 h on 64 vCPUs)

## Pipeline chain

```
upstream source datasets
       │
       ▼  (canonical column normalization, source_dataset / source_doc_id tagging)
GlossAPI canonical source-parquet release
       │
       ▼  (Corpus.clean, wave-2 broad cleaner)
cleaned_canonical/  (per-doc text + quality scores)
       │
       ▼  (selected_input: quality gates + openarchives.gr OCR exclusion)
selected_input.parquet
       │
       ▼  (mix-build, 50/50 by char mass)
mix.parquet  (14.45 M rows, 105 B chars)
       │
       ▼  (export_text_budgeted_splits.py, deterministic stable_key)
train.parquet (14.40 M / 100 B)  +  val.parquet (7,654 / 50 M)  +  test.parquet (7,282 / 50 M)
       │
       ▼  (train_continuous_bpe_tokenizer.py)
C3 tokenizer (156,672 vocab)
```

## Source pool 1 — GlossAPI corpus

Canonical upload:
**[`fffoivos/glossapi-greek-nanochat-pretraining-dataset`](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset)**
on Hugging Face. 19 source datasets, 810,638 raw rows; the wave-2
broad-cleaner + quality filters drop this to **546,920 rows / ≈ 52 B
chars** in the C3 mix.

The corpus combines a local GlossAPI collection with four external HF
sources. Broad subject tags from
`/home/foivos/data/glossapi_work/dataset_broader_subjects.txt`.

### Local GlossAPI sources

| source_dataset | raw rows | broad subject | upstream |
| --- | ---: | --- | --- |
| `1000_prwta_xronia_ellhnikhs` | 1,016 | ancient Greek, historical | local scrape |
| `Apothetirio_Kallipos` | 4,827 | academic textbooks | [kallipos.gr](https://repository.kallipos.gr/) (UoA digital library of OER textbooks) |
| `Apothetirio_Pergamos` | 15,241 | academic | [pergamos.lib.uoa.gr](https://pergamos.lib.uoa.gr/) (UoA Pergamos digital library) |
| `dimodis_logotexnia` | 11 | mediaeval/modern Greek folk literature | local scrape |
| `Ekklisiastika_Keimena` | 675 | ecclesiastical | local scrape |
| `Ellinika_Keimena_Project_Gutenberg` | 214 | modern Greek, katharevousa | [gutenberg.org Greek catalogue](https://www.gutenberg.org/browse/languages/el) |
| `ellinika_dedomena_europaikou_koinovouliou` | 28,723 | Greek-language EU Parliament data | EU Open Data Portal / Europarl |
| `eurlex-greek-legislation` | 22,694 | law | [eur-lex.europa.eu](https://eur-lex.europa.eu/) Greek legislation |
| `greek_phd` | 37,229 | academic theses | [didaktorika.gr](https://www.didaktorika.gr/) (National Documentation Centre, EKT) |
| `klasikh_arx_ell_grammateia` | 815 | classical ancient Greek literature | local scrape (likely [Open Library / Hellenic World](http://www.greek-language.gr/)-style sources) |
| `openarchives.gr` | 46,000 (kept) / 179,845 raw | academic / repository | [openarchives.gr](https://www.openarchives.gr/) (EKT aggregator). Heavily filtered: `needs_ocr == False`, `greek_badness_score < 25` upstream gate |
| `openbook_gr` | 3,719 | modern Greek literature | [openbook.gr](https://www.openbook.gr/) |
| `opengov.gr-diaboyleuseis` | 1,397 | law / public consultation | [opengov.gr/diaboyleyseis](https://www.opengov.gr/home/diavoulefseis) |
| `Sxolika_vivlia` | 123 | textbooks | local scrape (Greek school books, [ebooks.edu.gr](http://ebooks.edu.gr/)-style) |
| `Wikisource_Greek_texts` | 5,394 | modern Greek, katharevousa, literature, historical | [el.wikisource.org](https://el.wikisource.org/) |

### External Hugging Face sources merged in

| source_dataset | raw rows | broad subject | HF link |
| --- | ---: | --- | --- |
| `HuggingFaceFW/finewiki` (`el`) | 242,517 | wiki | [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) |
| `AI-team-UoA/greek_legal_code` | 47,563 | law | [AI-team-UoA/greek_legal_code](https://huggingface.co/datasets/AI-team-UoA/greek_legal_code) |

### Explicit exclusions

- `95k_deigma_ellinikis` — excluded entirely from the canonical
  upload (`CONTENT_METADATA_INVENTORY.md` § Summary Decisions).
- `GFOSS_blog_dataset`, `istorima`, `oldopenbook.gr` — listed in
  `dataset_broader_subjects.txt` as considered but not in the final
  upload.
- `HuggingFaceFW/finepdfs-edu` and `OPUS/OpenSubtitles-el-v2018` —
  intentionally dropped from the wave-3 production reclean (per
  `subprojects/_archive/01_0_cleaning_iteration_and_thresholds/WAVE3_PRODUCTION_PROGRESS_2026-04-28.md`).
  The hardcoded `DROPPED_DATASETS` set in
  `subprojects/_archive/01_0_cleaning_iteration_and_thresholds/scripts/reclean_canonical_to_parquet.py`
  enforces the exclusion. **Neither dataset was in C3's training
  mix.**
  - OPUS was re-cleaned (39-col schema with populated quality scores)
    and uploaded to
    [`fffoivos/glossapi-greek-nanochat-pretraining-dataset`](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset)
    on 2026-05-12. It is part of the published HF dataset going
    forward, but **was not** seen by C3's BPE training.
  - `HuggingFaceFW/finepdfs-edu` remains dropped.

### Canonical schema (per row)

21 columns, see
`/home/foivos/data/glossapi_work/hf_release/README.md` § Canonical
Columns. Key fields:

- identity: `source_dataset`, `source_doc_id`, `title`, `author`,
  `source_metadata_json`
- text: `text`
- quality / content-analysis: `greek_badness_score`,
  `mojibake_badness_score`, `greek_percentage`, `latin_percentage`,
  `polytonic_ratio`, `table_ratio`, `is_historical_or_polytonic`,
  `contains_math`, `contains_latex`
- pipeline: `is_empty`, `filter`, `needs_ocr`, `ocr_success`,
  `quality_method`, `reevaluated_at`

## Source pool 2 — HPLT clean60

The pre-built filtered Greek slice of HPLT 2.0, packaged as
**[`fffoivos/hplt-greek-ge8-no-mt-clean60-wave4`](https://huggingface.co/datasets/fffoivos/hplt-greek-ge8-no-mt-clean60-wave4)**
on Hugging Face. Local copy lived (on the gcloud instance, now
terminated) under
`/home/foivos/data/glossapi_work/hf_release_publish_hplt_clean60/data/*.parquet`
(250 parquets, ≈ 103 GB, 48.7 M rows total).

### Upstream

- [HPLT 2.0 release](https://hplt-project.org/) — High Performance
  Language Technologies multilingual web-crawl, Greek slice
  (`ell_Grek`).

### Filters applied to produce HPLT clean60

1. Language tag = `ell_Grek` (Greek in Greek script).
2. HPLT quality bins `>= 8` (top quality tier from the HPLT 2.0
   per-doc scoring).
3. Drop documents labelled `Machine translated or generated`
   (HPLT 2.0 ML-translation flag).
4. Run upstream `Corpus.clean(..., write_cleaned_files=False,
   drop_bad=False)` on every doc to compute
   `greek_badness_score` + `mojibake_badness_score`.
5. Drop `greek_badness_score > 60` ("clean60" suffix).
6. Normalize to the same 21-column GlossAPI source-parquet schema;
   `source_dataset = "HPLT/ell_Grek_ge8_no_mt_clean60"`,
   `source_doc_id = "hplt::<shard_name>::<id_within_shard>"`.

### How C3 sampled HPLT

From `mix.source_mix_summary.json`:

- HPLT clean60 available: **48,577,489 rows** / 183.45 B chars
- C3 mix selected: **13,906,493 rows** / 52.47 B chars (28.6 % of HPLT
  by rows, ≈ 28.6 % by chars)
- Sampler mode: `fraction_mode: of_total`, `fraction: 0.5` (target
  half of the mix by chars; the sampler picked rows to hit that
  char budget)

The other 34.7 M unsampled HPLT docs are the source pool used to
build the verified-virgin eval slice
(`hplt_virgin_eval_20260511.parquet`); see C3_CONVERGENCE.md §
Held-out integrity.

## Mix recipe

`mix.source_mix_summary.json` (full):

```json
{
  "name": "glossapi_plus_hplt_50_50",
  "components": [
    { "name": "glossapi", "fraction_mode": "of_total", "fraction": 0.5,
      "exclude_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
      "available_rows": 546920, "available_chars": 52467980059,
      "selected_rows":  546920, "selected_chars": 52467980059 },
    { "name": "hplt", "fraction_mode": "of_total", "fraction": 0.5,
      "include_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
      "available_rows": 48577489, "available_chars": 183450301755,
      "selected_rows":  13906493, "selected_chars": 52467977691 }
  ]
}
```

Resulting `mix.parquet`: **14,453,413 rows / 104.94 B chars**.

## Train / val / test split

Built by
[`subprojects/_archive/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py`](../subprojects/_archive/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py)
with seed-salt `c3_20260506_wave2_broad_glossapi_plus_hplt_50_50`,
deterministic stable-key ordering.

| split | rows | chars (budget) |
| --- | ---: | ---: |
| train | 14,401,554 | 100,000,000,000 |
| val | 7,654 | 50,000,000 |
| test | 7,282 | 50,000,000 |

⚠ Known issue: the splitter partitions by row, not by document. Where
the input mix had duplicate texts, the duplicates can land in
different splits. Measured on C3 exports: 30 train ↔ val and 36
train ↔ test exact text-md5 collisions. Documented in
[C3_CONVERGENCE.md](C3_CONVERGENCE.md) § Held-out integrity; cleaned
val/test parquets (`val_clean.parquet`, `test_clean.parquet`)
emit by anti-joining train text-md5 and are what the cutoff sweep
evaluates against.

## What C3 did NOT see

For posterity (and for choosing eval slices):

- 71.4 % of HPLT clean60 (~34.7 M docs) — not sampled into the mix.
- Roughly 32 % of the GlossAPI canonical raw rows (810 K → 546 K
  after quality + dedup + OCR filters).
- `openarchives.gr` rows with `needs_ocr == True` (≈ 22,800 docs).
- Any HPLT doc that failed `quality_bin >= 8`, was tagged
  machine-translated, or had `greek_badness_score > 60`.
- The `95k_deigma_ellinikis` collection (explicitly excluded
  from the canonical upload).

These are the candidate pools for any future evaluation slice that
needs to be guaranteed unseen-by-C3.
