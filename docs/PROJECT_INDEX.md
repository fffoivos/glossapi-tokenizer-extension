# Project Index

This repo is intentionally split into smaller subprojects.

## Current Stage

**C3 cutoff decision.** Tokenizer arm is converged
(`C3_wave2_broad_glossapi_plus_hplt_50_50`). The only open
tokenizer-side decision is which cutoff to ship from the frozen grid
`{10240, 15360, 20480, 25600}`. See
[C3_CONVERGENCE.md](C3_CONVERGENCE.md).

Read order for a fresh agent:
1. [C3_CONVERGENCE.md](C3_CONVERGENCE.md)
2. [GLOBAL_DECISIONS.md](GLOBAL_DECISIONS.md)
3. [CURRENT_STATUS.md](CURRENT_STATUS.md)
4. [ACTIVE_BACKLOG.md](ACTIVE_BACKLOG.md) — §Tokenizer Critical Path

The subprojects below are listed in the order they are traversed; most
of the earlier ones are settled and the live work is in `02_1` (cutoff
sweep) → `02_2` (merge-rule extension) → `03` (embedding adaptation).

## Subprojects

### Live

1. [02_apertus_tokenizer_spec](../subprojects/02_apertus_tokenizer_spec/README.md) — pinning checklist still to lock
2. [02_1_tokenizer_experiments](../subprojects/02_1_tokenizer_experiments/README.md) — **active** (C3 cutoff sweep)
3. [02_2_tokenizer_implementation](../subprojects/02_2_tokenizer_implementation/README.md) — gated on cutoff
4. [03_apertus_extension_and_embedding_adaptation](../subprojects/03_apertus_extension_and_embedding_adaptation/README.md) — gated on tokenizer freeze
   - [03_1_greek_embedding_diagnostic](../subprojects/03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/README.md) — pre-extension diagnostic of how Apertus represents Greek (E/U geometry, hull occupancy, binary classifier, cross-language clusters)

### Archived (DONE for the C3 shipping path)

See [../subprojects/_archive/README.md](../subprojects/_archive/README.md).

- `01_hplt_filtering` — HPLT clean60 slice
- `01_1_corpus_dedup` — dedup contract + repair
- `01_2_training_dataset_mix` — mix.parquet builder + splitter
- `01_0_cleaning_iteration_and_thresholds` — wave-2 broad cleaner (parallel `cleaner/per-line-badness-20260504` branch decoupled from C3)

## Canonical Sources Of Truth

### Live

- **C3 convergence** (read first):
  - [C3_CONVERGENCE.md](C3_CONVERGENCE.md)
- **C3 cutoff sweep results** (1k–25k, plots + tables):
  - [C3_CUTOFF_REPORT.md](C3_CUTOFF_REPORT.md)
- **C3 training datasets** (inventory + source links):
  - [C3_TRAINING_DATASETS.md](C3_TRAINING_DATASETS.md)
- global decisions:
  - [GLOBAL_DECISIONS.md](GLOBAL_DECISIONS.md)
- current status:
  - [CURRENT_STATUS.md](CURRENT_STATUS.md)
- active backlog:
  - [ACTIVE_BACKLOG.md](ACTIVE_BACKLOG.md)
- Apertus pretraining data inventory + plan for estimating Greek share:
  - [APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md](APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md)
- Apertus architecture choices that force cross-language embedding-norm convergence (reconciles Phase-A norm parity with the 0.023 % Greek data share):
  - [APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md](APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md)
- **Embedding diagnostic plan v2** (Greek vs ¬Greek, hull occupancy, infiltrators, clustering, analogies — the live diagnostic):
  - plan: [EMBEDDING_DIAGNOSTIC_PLAN_V2.md](EMBEDDING_DIAGNOSTIC_PLAN_V2.md)
  - sub-subproject (scripts + artifacts + reports): [../subprojects/03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/README.md](../subprojects/03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/README.md)
- **Vocab-language attribution** (1,933 canonical langs × 131,072 Apertus vocab entries; sub-subproject under 02_2):
  - report (with run status + scripts + artifact spec): [../subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/RUN_REPORT.md](../subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/RUN_REPORT.md)
- functional issues TODO:
  - [FUNCTIONAL_ISSUES_TODO.md](FUNCTIONAL_ISSUES_TODO.md)
- machine-readable config:
  - [apertus_greek_extension.yaml](../config/apertus_greek_extension.yaml)

### Archived (settled pre-C3-convergence work — read only for historical reconstruction)

- [_archive/README.md](_archive/README.md) — index of archived docs
- pre-convergence pipeline / dedup planning (E2E, near-dedup, HF dedup,
  builder/tokenizer efficiency, stage-verification checklist) all live
  under [_archive/](_archive/)

### Other

- legacy material (add-tokens baseline, exploratory HPLT):
  - [legacy/README.md](../legacy/README.md)

## Current Status

- C3 (`C3_wave2_broad_glossapi_plus_hplt_50_50`) is the converged
  tokenizer arm; see [C3_CONVERGENCE.md](C3_CONVERGENCE.md)
- the filtered HPLT slice and the four-arm exploration are settled
- the open tokenizer-side decision is C3's cutoff from the frozen grid
  `{10240, 15360, 20480, 25600}`
- no merge-rule Apertus extension has been built yet (gated on cutoff)
- the HF upload is an operational sidetrack, decoupled from the
  tokenizer critical path
