# 03.2 Apertus × C3 dedup audit

## Scope

Quantify the document-level overlap between **Apertus-8B-2509's
Greek pretraining data** and the **C3 tokenizer-training corpus**.

Output is per-source dedup-overlap measurements plus their CPT
implications: how much of our extension corpus is genuinely *new*
to the model, how much of our held-out evaluation slices were
contaminated by Apertus pretraining, and what that means for the
CPT replay ratio.

This is **measurement, not modification**. No corpus is rewritten;
no model is trained. Output is a set of per-pair dedup artifacts
plus a single report.

**Execution model**: 8 × `c4-highcpu-192` spot workers in
`europe-west4-b`, coordinator on `home`. ~45 min wall-clock,
~$30-40 total cost. Follows the proven `apertus-vocab-attr-w0..w7`
fan-out pattern. See [`PLAN.md`](PLAN.md) §6 for architecture and
§7 for the coordinator/worker script split.

## Why this matters for 03

Parent `03/TODO.md` flags two open questions this audit informs:

1. **Exact multilingual replay ratio** — depends on how much *fresh*
   Greek signal CPT actually delivers. If 50 % of C3 overlaps with
   Apertus pretraining, the effective new-data budget is half what
   the C3 numbers suggest.
2. **CPT acceptance criteria** — held-out evaluation slices must be
   verified not-already-seen by Apertus pretraining, or the
   eval numbers are leakage-contaminated.

The Greek-share number we already have (0.023 % of consumed Apertus
pretraining tokens) tells us *total Greek exposure* but not *per-doc
overlap with our extension corpus*. The audit fills that gap.

## Deliverables

All bulk-row artifacts are **Parquet** (zstd-compressed), matching
the existing `glossapi_corpus_cli text_dedup` pipeline's default.

- [`PLAN.md`](PLAN.md) — full experiment plan with sources,
  methodology (drives `glossapi_corpus_cli text_dedup` cross-corpus
  mode at versioned defaults), compute estimate, risks.
- `artifacts/overlap/strict_exact/<a>_x_<c>.parquet` — strictest
  exact-match intersection per pair (blake3 over NFC + whitespace-
  collapse normalised text).
- `artifacts/overlap/relaxed_exact/<a>_x_<c>.parquet` — relaxed
  exact-match (lowercased, strip-punctuation, ZWS removed).
- `artifacts/overlap/near/<a>_x_<c>.parquet` — MinHash near-dup
  (Jaccard ≥ 0.85, 128 perms, token 5-shingles — defaults pinned
  to `text_dedup.py`).
- `artifacts/consumed_estimate/*.parquet` — overlap against the
  Apertus-sampler-reconstructed *consumed* subset, not just the
  released slice.
- `artifacts/per_c3_doc_overlap.parquet` — one row per C3 doc with
  any match: `overlap_ratio`, `tier ∈ {drop, partial, trace}`,
  `best_match_*` columns.
- `artifacts/holdout_contamination.parquet` — full-ladder contamination
  check on C3 val/test (strict + relaxed + near + sentence-level
  EuroParl).
- `artifacts/per_c3_source_actionable.parquet` — C3-side
  actionable per-source CPT inputs (fresh-rows / fresh-chars /
  fresh-tokens / recommended-action).
- `REPORT.md` — synthesised summary table + CPT-planning implications.

## Out of scope

- Re-training the C3 tokenizer.
- Re-running Apertus pretraining.
- Polytonic / Ancient Greek corpora (those live in
  `02_1_polytonic_greek_extension/` and have their own dedup story).
- Apertus's English / multilingual non-Greek pretraining — only
  Greek-bearing slices are audited here.

## See also

- [`PLAN.md`](PLAN.md) — full plan
- [`../README.md`](../README.md) — parent 03 subproject overview
- [`../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md)
  — Apertus Greek pretraining inventory (the 4 datasets to dedup against)
- [`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md)
  — C3 corpus inventory (the GlossAPI + HPLT mix to dedup)
- [`../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)
  — the chosen 17,408 cutoff this audit is being run for
