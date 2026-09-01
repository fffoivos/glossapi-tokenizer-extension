# Index of `docs/`

The program's history is told per subproject in `subprojects/*/README.md` and
summarized in the [root README](../README.md). This file only says what each
document in `docs/` is, when it was written, and whether it still governs
anything. Dates are first/last commit dates on the consolidated history.

## The consolidation note

| Document | Dates | What it is |
|---|---|---|
| [CONSOLIDATION_REVIEW_20260901.md](CONSOLIDATION_REVIEW_20260901.md) | 2026-09-01 | What the 2026-09-01 branch consolidation found and did, the decisions left to the owner, and the writers' archive-candidate and contradiction lists. |

## Still in force

| Document | Dates | What it is |
|---|---|---|
| [LOSS_MEASUREMENT_POLICY.md](LOSS_MEASUREMENT_POLICY.md) | 2026-06-11 | Repo-wide rule: never compare arms with different tokenizers on raw Megatron `lm loss`; use held-out tokenizer-fair BPB and downstream benchmarks. Applied from the 03 bakeoff onward. |
| [GLOBAL_DECISIONS.md](GLOBAL_DECISIONS.md) | 2026-04-10 → 2026-08-05 | The hard constraints of the extension (preserve Apertus ids/special tokens/regex, vocab divisible by 128, untied embeddings) and the goal statement. Its "tokenizer arm" section is a May 2026 decision record (C3 at 25,600 added units, cutoff still open); the shipped cutoff and later polytonic extension are in `subprojects/02_1_tokenizer_experiments/README.md`. |
| [../subprojects/CURRENT_HYPERPARAMETERS.md](../subprojects/CURRENT_HYPERPARAMETERS.md) | 2026-07-11 (+ 2026-08-06 RoPE-geometry correction) | The frozen Apertus-8B CPT training configuration (AdEMAMix, WSD, geometry) that the 8B runs used. Lives under `subprojects/`, listed here because it is the one config document that outlived its subproject. |

## Decision records and reports (historical, correct as of their date)

| Document | Dates | What it is |
|---|---|---|
| [C3_CONVERGENCE.md](C3_CONVERGENCE.md) | 2026-05-11 (committed 05-14) | Why the tokenizer track converged on the C3 arm (continuous BPE, GlossAPI+HPLT 50/50) and closed the F1/F2/C1/C2 exploration. |
| [C3_CUTOFF_REPORT.md](C3_CUTOFF_REPORT.md) | 2026-05-14 | The 1k–25k cutoff sweep (fertility, chars/token, added-vocab utilization) with the plots in `figures/`; built by `_scripts/build_c3_cutoff_report.py`. Outcome: 17,408 added units (vocab 148,480). |
| [C3_TRAINING_DATASETS.md](C3_TRAINING_DATASETS.md) | 2026-05-14 → 06-11 | Inventory of the datasets the C3 tokenizer was trained on, with source links. |
| [APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md](APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md) | 2026-05-14 | Reconstruction of how much Greek Apertus saw in pretraining; background for the replay decisions in 03–07. |
| [APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md](APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md) | 2026-05-14 | Architecture facts (untied embeddings, xIELU, QK-norm) relevant to the 03_1 embedding diagnostic. |
| [APERTUS_GREEK_BEHAVIORAL_NLL_PHASE_B.md](APERTUS_GREEK_BEHAVIORAL_NLL_PHASE_B.md) | 2026-05-14 | Phase-B behavioral NLL study of Greek tokens in Apertus (input to the init-method choice in 03_4). |
| [EMBEDDING_DIAGNOSTIC_PLAN_V2.md](EMBEDDING_DIAGNOSTIC_PLAN_V2.md) | 2026-05-14 | The plan the 03_1 diagnostic executed. |
| [APERTUS_EXTENSION_ARTIFACT_MAP_20260525.md](APERTUS_EXTENSION_ARTIFACT_MAP_20260525.md), [..._RELEASE_REORGANIZATION_PLAN_20260525.md](APERTUS_EXTENSION_RELEASE_REORGANIZATION_PLAN_20260525.md), [..._RELEASE_RENAMING_PLAN_20260525.md](APERTUS_EXTENSION_RELEASE_RENAMING_PLAN_20260525.md), [..._RELEASE_FOUR_ACTOR_LAYOUT_20260525.md](APERTUS_EXTENSION_RELEASE_FOUR_ACTOR_LAYOUT_20260525.md), [APERTUS_RELEASE_UPLOAD_VERIFICATION_20260525.md](APERTUS_RELEASE_UPLOAD_VERIFICATION_20260525.md) | 2026-05-25 → 06-11 | How the bakeoff checkpoints, tokenizer and evals were laid out and published to Hugging Face (`fffoivos/apertus-tokenizer-extension`); the local mirror is `../release/`. |
| [AGENT1_POST_NANOCHAT_DATA_REVIEW_AND_DEDUP_STATUS_2026-07-18.md](AGENT1_POST_NANOCHAT_DATA_REVIEW_AND_DEDUP_STATUS_2026-07-18.md), [AGENT1_V5_DATASET_AND_HF_READINESS_AUDIT_2026-07-18.md](AGENT1_V5_DATASET_AND_HF_READINESS_AUDIT_2026-07-18.md), [AGENT1_V5_CSCS_DEDUP_ACCELERATION_PLAN_2026-07-18.md](AGENT1_V5_CSCS_DEDUP_ACCELERATION_PLAN_2026-07-18.md), [AGENT1_V5_DEDUP_ACCELERATION_IMPLEMENTATION_STATUS_2026-07-18.md](AGENT1_V5_DEDUP_ACCELERATION_IMPLEMENTATION_STATUS_2026-07-18.md), [AGENT1_V5_LSH_OVERSIZED_DIAGNOSIS_2026-07-21.md](AGENT1_V5_LSH_OVERSIZED_DIAGNOSIS_2026-07-21.md) | 2026-07-18 → 07-22 | The "Agent 1" v5 corpus build: post-NanoChat data review, HF readiness audit, the CSCS dedup acceleration plan, and the oversized-LSH-group diagnosis that unblocked the final dedup. Companion snapshot: [hf/agent1_v5_pre_dedup_audit_snapshot/](hf/agent1_v5_pre_dedup_audit_snapshot/README.md). The full corpus story is in `subprojects/05_token_distillation_cpt/04_full_corpus_preparation/README.md`. |

## Status snapshots (historical; do not act on them)

| Document | Dates | What it was |
|---|---|---|
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | 2026-04-10 → 2026-08-05 | The "current phase" note of the tokenizer track; last substantively updated for the C3 cutoff phase (May 2026). |
| [ACTIVE_BACKLOG.md](ACTIVE_BACKLOG.md) | 2026-04-10 → 05-14 | The C3 cutoff-decision work list. |
| [FUNCTIONAL_ISSUES_TODO.md](FUNCTIONAL_ISSUES_TODO.md) | 2026-04-14 | Pipeline defects found during the April corpus-cleaning phase (dedup admission gate etc.). |
| [../subprojects/SUBPROJECTS_OVERVIEW.md](../subprojects/SUBPROJECTS_OVERVIEW.md) | 2026-05-18 | One paragraph per sub-subproject of 02–05 as of mid-May; superseded by the per-subproject READMEs. |

## `_archive/` — the April 2026 corpus-pipeline phase

[_archive/README.md](_archive/README.md) indexes the pre-C3 planning docs
(pipeline recovery and scale plan, E2E verification, near-dedup redesign, the
2026-04-15 worker run report, the embedding-init test plan v4, the 2026-05-11
external feedback on the extension doc). They document the work archived under
`subprojects/_archive/01_*`.

## Non-document entries

- `figures/` — the six C3 cutoff plots referenced by `C3_CUTOFF_REPORT.md`.
- `_scripts/build_c3_cutoff_report.py` — regenerates that report from the sweep outputs.
- `hf/agent1_v5_pre_dedup_audit_snapshot/` — README + provenance JSON of the v5 pre-dedup dataset build as uploaded to Hugging Face.
