# 02_1_3 Fertility evaluation

Sub-subproject of `02_1_tokenizer_experiments`. **Stage 3**:

```
[02_1_2 cutoff variant builder] → N merged variants per arm
       │
       ▼
[02_1_3 fertility evaluation] → intrinsic + fertility metrics per (variant, slice)
       │
       ▼
[02_1_4 cutoff analysis] → combines into a cutoff recommendation
```

## Goal

For each cutoff variant of a tokenizer arm, measure:

- `chars_per_token` (higher = better compression)
- `tokens_per_byte` / `bytes_per_token` (inverse view)
- `greek_word_space_fertility` (lower = fewer tokens per Greek word;
  the primary Greek-quality metric)
- `single_token_greek_word_share` (higher = more whole-word tokens)
- `added_token_rate` (share of decoded tokens that come from the new
  added units)
- `eval_added_vocab_utilization_rate` (fraction of added tokens that
  appear at least once on the eval slice)
- `eval_unused_added_tokens` (count of added tokens never seen)
- `unk_rate` / `byte_fallback_rate` (compatibility sanity)

Evaluated on **verifiable clean held-out slices** — see "Held-out
integrity" below.

## Inputs

- Cutoff variant dirs from `02_1_2`
- Apertus base tokenizer (for the `apertus_base` reference row)
- Held-out parquet slices (clean — produced by the helper scripts in
  this sub-subproject)

## Outputs

`fertility_<arm>_<scope>_<date>/`:
- `metrics_by_slice.json` — full per-(tokenizer, slice) row records
- `metrics_by_slice.csv` — same, flat csv
- `summary.json` + `SUMMARY.md` — auto-generated rollup
- `tokenizers.json` — tokenizer-set manifest (per-tokenizer sha, vocab,
  added-units count)
- `sample_manifests/<slice>.json` — reservoir sample metadata
- `samples/<slice>.txt.gz` — the actual sampled docs (small)

## Scripts

- `scripts/run_tokenizer_fertility_suite.py` — main driver. Takes
  `--tokenizer name=path` and `--slice name=path` arguments
  (repeatable) and produces the artifacts above. Imports the heavy
  lifting from
  `tokenizer_analysis/run_wave4_fertility_eval.py`.
- `scripts/clean_holdouts.py` — anti-join val/test against train on
  text-md5 to produce verifiable-clean held-out parquets. Required
  because the splitter has a row-vs-doc bug (see
  `docs/C3_CONVERGENCE.md` § Held-out integrity).
- `scripts/build_virgin_hplt_eval.py` — sample 10k HPLT docs whose
  `source_doc_id` is **not in** the training mix. Guaranteed unseen by
  the tokenizer's BPE training. Used for the C3 sweep on
  `virgin_hplt` (the cleanest of the held-out slices we use).

## Held-out integrity (required reading)

The default C3 train/val/test splits emitted by the splitter at
`subprojects/_archive/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py`
partition by row, not by doc/text. Verified on C3 exports: 30 train↔val
+ 36 train↔test exact text-md5 collisions (~0.4–0.5% of held-out).

For the C3 cutoff sweep we evaluated on **three verified-clean slices**:

- `virgin_hplt` (10,000 docs) — HPLT docs whose `source_doc_id` is not
  in the C3 training mix. Built by `build_virgin_hplt_eval.py`.
- `C3_val_clean` (7,624 docs) — C3 val minus 30 train-overlap rows.
  Built by `clean_holdouts.py`.
- `C3_test_clean` (7,246 docs) — C3 test minus 36 train-overlap rows.

Any future modern/polytonic arm should source its own held-outs the same
way, plus add polytonic-eval slices chosen from the dataset review. That
arm and its cutoff grid are not fixed in this document.

## Example invocation (C3 sweep, 25 cutoffs × 3 clean slices)

```bash
source /home/foivos/venvs/glossapi-corpus-clean/bin/activate
TOK_ARGS=""
for n in $(seq 1024 1024 25600); do
  TOK_ARGS="$TOK_ARGS --tokenizer c3_added_${n}=/home/foivos/runs/c3_cutoff_eval_20260511/cutoff_tokenizers/c3_added_${n}"
done
python3 scripts/run_tokenizer_fertility_suite.py \
  --repo-root /home/foivos/Projects/glossapi-tokenizer-extension \
  --output-dir /home/foivos/runs/c3_cutoff_eval_20260511/fertility_c3_full_25_clean \
  --latest-glossapi-limit 0 \
  --tokenizer apertus_base=/home/foivos/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_20260415 \
  $TOK_ARGS \
  --slice virgin_hplt=/home/foivos/runs/c3_cutoff_eval_20260511/virgin_hplt_eval/hplt_virgin_eval_20260511.parquet \
  --slice C3_val_clean=/home/foivos/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/splits/glossapi_plus_hplt_50_50/exports/val_clean.parquet \
  --slice C3_test_clean=/home/foivos/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/splits/glossapi_plus_hplt_50_50/exports/test_clean.parquet
```

Runtime: 78 rows (26 tokenizers × 3 slices) on the gcloud worker
finishes in ~3 minutes.

## Where the C3 outputs live

- Instance: `~/runs/c3_cutoff_eval_20260511/fertility_c3_full_25_clean_20260511/`
- Home copy: `metrics_by_slice.json` pulled to `/tmp/c3_cutoff_metrics.json`
  during the analysis step (`02_1_4`)
