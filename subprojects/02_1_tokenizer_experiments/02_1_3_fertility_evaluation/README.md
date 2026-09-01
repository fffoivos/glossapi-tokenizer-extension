# 02.1.3 — Fertility Evaluation

> **In one line:** the in-domain metric harness for cutoff variants — and the place where a held-out-integrity bug was found and worked around rather than fixed.
> **Period:** ran for the C3 sweep 2026-05-11; committed here 2026-05-18 (`7deea009`). **Status:** completed; the harness and its clean slices were reused unchanged by `02_1_7` and by the polytonic arm.
> **Came from / led to:** [`../02_1_2_cutoff_variant_builder/`](../02_1_2_cutoff_variant_builder/README.md) → this → [`../02_1_4_cutoff_analysis/`](../02_1_4_cutoff_analysis/README.md) and [`../02_1_7_intrinsic_eval_sweep/`](../02_1_7_intrinsic_eval_sweep/README.md)

## Why this existed

Stage 3: for every (cutoff variant, held-out slice) pair, measure what the added units actually buy. The primary Greek-quality number this whole program tracks — `greek_word_space_fertility`, tokens per Greek word — is defined and computed here, alongside `chars_per_token`, `tokens_per_byte`, `single_token_greek_word_share`, `added_token_rate`, added-vocab utilization, unused added tokens, and unk / byte-fallback rates.

## History

### 2026-05-11 — the splitter bug

Verification of the C3 exports found the train/val/test splits were **not disjoint at the text level**: the splitter partitioned by `source_split_row_id` (row index) rather than by text or document id, so duplicate texts in the input mix were sent independently to different splits. Measured: 29,527 duplicate texts inside train, **30** train∩val and **36** train∩test exact text-md5 collisions — roughly 0.4–0.5 % of each held-out ([`../../../docs/C3_CONVERGENCE.md`](../../../docs/C3_CONVERGENCE.md) § Held-out integrity).

The contamination was below the metric noise floor (~1 leaked doc per 300-doc sample), but the slices were not *verifiable* held-outs, and the project rule was that they had to be. The fix was not back-ported to C3's exports. Instead two helper scripts in this directory built a clean evaluation path:

- `build_virgin_hplt_eval.py` — samples 10,000 HPLT docs whose `source_doc_id` is **not** in the C3 training mix, anti-joined against the `fffoivos/hplt-greek-ge8-no-mt-clean60-wave4` release. Guaranteed unseen by C3's BPE.
- `clean_holdouts.py` — anti-joins val/test against train on text-md5, giving `C3_val_clean` (7,624 docs) and `C3_test_clean` (7,246).

Every fertility number cited anywhere downstream comes from those three slices.

### 2026-05-11 → 2026-05-18 — the sweeps

The C3 cutoff sweep ran 26 tokenizers × 3 slices = **78 rows in about 3 minutes** on the gcloud worker. `02_1_7` later re-ran the same harness on its own four in-house slices (C2_val/test, C3_val/test) plus four latest-GlossAPI datasets, so the report can put in-domain fertility next to TokEval's FLORES+ fertility and document the methodology delta between them.

## Outcome

- Outputs per run: `metrics_by_slice.{json,csv}`, `summary.json` + `SUMMARY.md`, `tokenizers.json` (per-tokenizer sha, vocab, added-unit count), plus reservoir-sample manifests and the sampled docs themselves.
- Fertility on the clean slices, averaged: `apertus_base` **2.413** → **1.471** at 11,264 → **1.345** at the eventually-chosen 17,408 on `C3_val` ([`../02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md) §2; [`../02_1_7_intrinsic_eval_sweep/REPORT.md`](../02_1_7_intrinsic_eval_sweep/REPORT.md)).
- Recommendation left standing for future arms: source held-outs the same way (anti-join at document level), and add register-specific control slices before training — a point the polytonic arm's own to-do list inherited and never closed.
- The underlying splitter defect was documented with a fix path (dedup the mix on `text` before split assignment, or make `stable_key` a content hash) but **not implemented**.

## Where things are

| What | Where |
|---|---|
| Metric driver | `scripts/run_tokenizer_fertility_suite.py` |
| Clean-holdout builder | `scripts/clean_holdouts.py` |
| Virgin-HPLT slice builder | `scripts/build_virgin_hplt_eval.py` |
| Held-out integrity finding | [`../../../docs/C3_CONVERGENCE.md`](../../../docs/C3_CONVERGENCE.md) § Held-out integrity |
| C3 sweep outputs | `runs/c3_cutoff_eval_20260511/fertility_c3_full_25_clean_20260511/` on the gcloud worker (not in git) |

## Working documents

None. Three scripts and this README.
