# Archived subprojects (DONE for the C3 shipping path)

These four subprojects produced inputs that C3 was trained on. They
are settled and should not drive new execution. Each subproject's own
`README.md` still has its `## Status` block marking it DONE.

Live work for the C3 cutoff decision and downstream implementation
lives at `../02_1_tokenizer_experiments/`, `../02_2_tokenizer_implementation/`,
`../03_apertus_extension_and_embedding_adaptation/`. See
[../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md).

## Contents

- `01_hplt_filtering/` — built the HPLT `ell_Grek` ≥8-quality, no-MT,
  `greek_badness ≤ 60` slice ("clean60") that fed the C3 mix.
- `01_1_corpus_dedup/` — dedup contract + repaired execution; produced
  the `dedup_metadata` bundle used by the C3 pipeline.
- `01_2_training_dataset_mix/` — built the `glossapi + hplt 50/50`
  `mix.parquet` that C3 trained from. Known caveat: the splitter
  (`scripts/export_text_budgeted_splits.py`) partitions by row not by
  doc, leaking 30+36 docs across val/test (see
  [../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md)
  § Held-out integrity).
- `01_0_cleaning_iteration_and_thresholds/` — produced the wave-2
  broad cleaner that ran on C3's training inputs. **Note**: the
  `cleaner/per-line-badness-20260504` branch is documented here as a
  future-improvement track decoupled from C3 — if revisited, the plan
  doc to read is `PER_LINE_CLEANER_BRANCH_PLAN_2026-05-04.md`.
