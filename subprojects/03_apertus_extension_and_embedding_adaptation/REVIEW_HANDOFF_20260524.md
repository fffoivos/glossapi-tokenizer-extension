# Apertus Greek CPT Review Handoff - 2026-05-24

Purpose: give a reviewer a single map of the current docs, scripts, and
evidence artifacts for the Apertus-8B Greek CPT decision. This is not a
replacement for the artifacts below; it tells you what each one is, which ones
are authoritative, and what claims they support.

Current reviewed commit: `29ce766 Add Apertus production review handoff`.

## Executive State

The current production default is **Vanilla Apertus-8B-2509 with the original
131,072-token base tokenizer**.

Why:

- The 2B three-arm bakeoff answered Vanilla vs ReTok vs Centroid.
- The bounded Token Distillation challenger improved the extended-tokenizer
  path, but did not beat Vanilla on the aggregate Greek/downstream preservation
  gate.
- The selected Vanilla path now has a buildable NFC-safe Megatron dataset,
  R17-preserving init evidence, and a dry-run-validated 15B production launcher.

Important number disambiguation:

- **2B tokens** = training budget per bakeoff arm.
- **~6B tokens** = total across three bakeoff arms.
- **~2B tokens** = TD challenger training budget.
- **7B target** = original corpus-build target for the JSONL stream, not per-arm
  training consumption.
- **9.83B base tokens** = size of the final NFC-safe base-tokenized Megatron
  dataset artifact available for production/repetition.

## Recommended Review Order

1. [`PRODUCTION_DECISION_STATE.md`](PRODUCTION_DECISION_STATE.md)
   - Current verdict, final result tables, selected production path, remaining
     boundaries.
2. [`ARTIFACTS_AND_HYDRATION.md`](ARTIFACTS_AND_HYDRATION.md)
   - What belongs in git, what stays on Clariden, and how to verify/hydrate
     production-critical artifacts.
3. [`CLARIDEN_INVENTORY_20260524.md`](CLARIDEN_INVENTORY_20260524.md)
   - Source-of-truth map of the remote checkpoints, datasets, eval outputs,
     code/envs, and intentionally absent artifacts.
4. [`TAKEOVER_LOG_20260521.md`](TAKEOVER_LOG_20260521.md)
   - Chronological operations log, including restarts, failures, fixes,
     training/eval completions, CPU/GPU allocation correction, and production
     launcher validation.
5. [`03_4_implementation_experiments/init_bakeoff/production_cpt/README.md`](03_4_implementation_experiments/init_bakeoff/production_cpt/README.md)
   - Concrete production launch path and dry-run evidence.
6. [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_digest.md`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_digest.md)
   - Final Vanilla/ReTok/Centroid checkpoint digest.
7. [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_digest.md`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_digest.md)
   - Final TD challenger digest.
8. [`03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md`](03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md)
   - Per-checkpoint trajectory analysis and TD-vs-Vanilla crossover projection.
9. [`03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md)
   - Corpus composition, source weights, NFC production data note, xfer-only CPU
     build rule.
10. [`03_4_implementation_experiments/init_bakeoff/megatron_patches/README.md`](03_4_implementation_experiments/init_bakeoff/megatron_patches/README.md)
   - HF <-> Megatron conversion, R17 issue, xIELU/QK-Norm patching, roundtrip
     pass criteria.

## Current Authoritative Docs

| Path | What it is | Review for |
|---|---|---|
| [`PRODUCTION_DECISION_STATE.md`](PRODUCTION_DECISION_STATE.md) | Current production decision overlay | Selected path, result tables, completed gates, production boundary |
| [`ARTIFACTS_AND_HYDRATION.md`](ARTIFACTS_AND_HYDRATION.md) | Repo ownership and hydration policy | What is committed, what stays on Clariden, launch/readiness checks |
| [`CLARIDEN_INVENTORY_20260524.md`](CLARIDEN_INVENTORY_20260524.md) | Remote artifact inventory | Dataset/checkpoint/eval/code paths, sizes, intentionally absent artifacts |
| [`TAKEOVER_LOG_20260521.md`](TAKEOVER_LOG_20260521.md) | Chronological operations log | What was run, what failed, what was changed, what completed |
| [`cpt_plan.md`](cpt_plan.md) | Design-space plan, now with decision overlay | Original objective and assumptions; do not treat older exploratory items as current TODOs without the overlay |
| [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) | Apertus-faithful training recipe | Optimizer, LR, batch shape, Goldfish, architecture, production-vs-bakeoff differences |
| [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md) | TD planning doc | Sensitive points for adapting TD to Apertus/ReTok |
| [`03_4_implementation_experiments/init_bakeoff/README.md`](03_4_implementation_experiments/init_bakeoff/README.md) | Implementation tree overview | Directory layout and execution sequence |
| [`03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md`](03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md) | Original bakeoff plan | The three-arm experiment design; now superseded by completed evidence |
| [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) | Earlier audit findings | Historical blockers and fixes |
| [`RISKS.md`](RISKS.md) | Risk register | Conversion and fidelity risks, especially R17 |
| [`COMPLETENESS_CHECK.md`](COMPLETENESS_CHECK.md) | Earlier completeness review | Gap tracking context |
| [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md) | Fidelity checklist | Apertus-specific constraints the implementation must preserve |

## Corpus And Dataset Artifacts

Authoritative corpus docs:

| Path | What it is | Review for |
|---|---|---|
| [`03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md) | Canonical CPT source build order | Apertus-overlap drop before mix build; internal dedup applied upstream |
| [`03_2_apertus_c3_dedup_audit/REPORT_dedup_20260519T010924Z.md`](03_2_apertus_c3_dedup_audit/REPORT_dedup_20260519T010924Z.md) | Dedup run report | Apertus overlap removal evidence |
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md) | Bulk/anneal mix description | 70/24/4/2 bulk mix, code fallback, math inclusion, NFC production binary |
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json) | Machine-readable bulk recipe | Source weights and local staged parquet paths |
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/anneal.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/anneal.json) | Draft anneal recipe | Design artifact only; not production launcher input |

Corpus scripts:

| Path | What it does |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/prepare_greek_pool.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/prepare_greek_pool.sh) | Builds the selected Greek pool after Apertus-overlap removal and internal dedup |
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py`](03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py) | Streaming, deterministic token-fair interleaver |
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/normalize_jsonl_nfc.py`](03_4_implementation_experiments/init_bakeoff/corpus_build/normalize_jsonl_nfc.py) | Final JSONL NFC normalization |
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/preprocess_hf_jsonl_to_megatron.py`](03_4_implementation_experiments/init_bakeoff/corpus_build/preprocess_hf_jsonl_to_megatron.py) | xfer-native HF-tokenizer -> Megatron indexed-dataset fallback |
| [`03_4_implementation_experiments/init_bakeoff/check_cpu_only_slurm.sh`](03_4_implementation_experiments/init_bakeoff/check_cpu_only_slurm.sh) | Pre-submit audit that CPU-only jobs use `xfer` and no GPU directives |
| [`03_4_implementation_experiments/init_bakeoff/slurm_cpu_only_guard.sh`](03_4_implementation_experiments/init_bakeoff/slurm_cpu_only_guard.sh) | Runtime guard for CPU-only Slurm jobs |

Key local corpus evidence:

| Path | What it proves |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/README.md`](03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/README.md) | Production-safe base tokenizer preprocessing summary |
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/bulk_mix_text_document.manifest.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/bulk_mix_text_document.manifest.json) | `9,831,704,774` base-tokenized tokens, `5,754,172` rows/sequences |
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/validate_hf_preproc-2367575.out`](03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/validate_hf_preproc-2367575.out) | Fallback preprocessor matched canonical Megatron output on first 1000 rows |
| [`03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/preprocess_base_nfc_hf-2367579.out`](03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/preprocess_base_nfc_hf-2367579.out) | Full xfer preprocessing job log |

Remote production data path:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document
```

Remote artifact inventory and hydration policy:
[`CLARIDEN_INVENTORY_20260524.md`](CLARIDEN_INVENTORY_20260524.md) and
[`ARTIFACTS_AND_HYDRATION.md`](ARTIFACTS_AND_HYDRATION.md).

## Initialization And Conversion Artifacts

Docs and scripts:

| Path | What it is |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/arms/README.md`](03_4_implementation_experiments/init_bakeoff/arms/README.md) | Init arm overview |
| [`03_4_implementation_experiments/init_bakeoff/arms/build_init_checkpoints.py`](03_4_implementation_experiments/init_bakeoff/arms/build_init_checkpoints.py) | Builds Vanilla/ReTok/Centroid HF-format init checkpoints |
| [`03_4_implementation_experiments/init_bakeoff/arms/vanilla.py`](03_4_implementation_experiments/init_bakeoff/arms/vanilla.py) | No-extension control |
| [`03_4_implementation_experiments/init_bakeoff/arms/retok.py`](03_4_implementation_experiments/init_bakeoff/arms/retok.py) | Subpiece-mean ReTok initialization |
| [`03_4_implementation_experiments/init_bakeoff/arms/centroid.py`](03_4_implementation_experiments/init_bakeoff/arms/centroid.py) | Greek centroid initialization |
| [`03_4_implementation_experiments/init_bakeoff/megatron_patches/loader_apertus_hf.py`](03_4_implementation_experiments/init_bakeoff/megatron_patches/loader_apertus_hf.py) | Custom Apertus HF -> Megatron loader |
| [`03_4_implementation_experiments/init_bakeoff/megatron_patches/patch_apertus_extras.py`](03_4_implementation_experiments/init_bakeoff/megatron_patches/patch_apertus_extras.py) | Copies xIELU/QK-Norm extras into converted Megatron checkpoints |
| [`03_4_implementation_experiments/init_bakeoff/megatron_patches/verify_hf_roundtrip.py`](03_4_implementation_experiments/init_bakeoff/megatron_patches/verify_hf_roundtrip.py) | Verifies tensor/logit roundtrip safety |
| [`03_4_implementation_experiments/init_bakeoff/megatron_patches/runtime/pretrain_gpt_te_guard.py`](03_4_implementation_experiments/init_bakeoff/megatron_patches/runtime/pretrain_gpt_te_guard.py) | Narrow TE empty-extra-state guard used during Megatron training |

Key local init/conversion evidence:

| Path | What it proves |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/arms/init_modern_only_148480_20260521/init_build_summary.json`](03_4_implementation_experiments/init_bakeoff/arms/init_modern_only_148480_20260521/init_build_summary.json) | Init artifact summary for bakeoff arms |
| [`03_4_implementation_experiments/init_bakeoff/megatron_patches/vanilla_r17_roundtrip_2341182/verification.json`](03_4_implementation_experiments/init_bakeoff/megatron_patches/vanilla_r17_roundtrip_2341182/verification.json) | Selected Vanilla init roundtrip: standard/R17/xIELU/QK/logit diffs are zero |
| [`03_4_implementation_experiments/init_bakeoff/megatron_patches/td_layer11_r17_roundtrip_2357565/verification.json`](03_4_implementation_experiments/init_bakeoff/megatron_patches/td_layer11_r17_roundtrip_2357565/verification.json) | TD layer11 candidate roundtrip safety |

Remote selected init path:

```text
/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched
```

## Bakeoff Training Scripts And Artifacts

Training scripts:

| Path | What it does |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/bakeoff_training/_train_config_common.env`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/_train_config_common.env) | Shared Apertus/Megatron training constants; production overrides supported |
| [`03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch) | Parameterized Megatron training job for arms and production |
| [`03_4_implementation_experiments/init_bakeoff/bakeoff_training/submit_all_arms.sh`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/submit_all_arms.sh) | Three-arm bakeoff submitter |
| [`03_4_implementation_experiments/init_bakeoff/bakeoff_training/submit_td_layer11_2b_chain.sh`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/submit_td_layer11_2b_chain.sh) | TD 2B challenger chain submitter |
| [`03_4_implementation_experiments/init_bakeoff/bakeoff_training/summarize_training_logs.py`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/summarize_training_logs.py) | Training curve summary helper |

Key training evidence:

| Path | What it proves |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_training_summary.md`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_training_summary.md) | Final 2B training completion summary |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_training_curve.csv`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_training_curve.csv) | Per-arm training curve |
| [`03_4_implementation_experiments/init_bakeoff/bakeoff_training/smoke_td_layer11_2357596/README.md`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/smoke_td_layer11_2357596/README.md) | TD layer11 Megatron load/train smoke |
| [`03_4_implementation_experiments/init_bakeoff/bakeoff_training/smoke_td_layer11_2node_2357684/README.md`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/smoke_td_layer11_2node_2357684/README.md) | Failed two-node smoke; reason production stays one-node |

## Evaluation Scripts And Artifacts

Evaluation scripts:

| Path | What it does |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md) | Eval task groups and cadence |
| [`03_4_implementation_experiments/init_bakeoff/eval/run_eval.sbatch`](03_4_implementation_experiments/init_bakeoff/eval/run_eval.sbatch) | Single-model lm-eval job |
| [`03_4_implementation_experiments/init_bakeoff/eval/run_eval_packed_arms.sbatch`](03_4_implementation_experiments/init_bakeoff/eval/run_eval_packed_arms.sbatch) | Packed multi-arm eval on one 4-GPU node |
| [`03_4_implementation_experiments/init_bakeoff/eval/submit_bakeoff_checkpoint_eval_packed.sh`](03_4_implementation_experiments/init_bakeoff/eval/submit_bakeoff_checkpoint_eval_packed.sh) | Conversion + packed eval submitter |
| [`03_4_implementation_experiments/init_bakeoff/eval/compute_tokenizer_fair_metrics.py`](03_4_implementation_experiments/init_bakeoff/eval/compute_tokenizer_fair_metrics.py) | BPC/NLL/tokenizer-fair metrics |
| [`03_4_implementation_experiments/init_bakeoff/eval/compute_new_token_diagnostics.py`](03_4_implementation_experiments/init_bakeoff/eval/compute_new_token_diagnostics.py) | New-token diagnostics |
| [`03_4_implementation_experiments/init_bakeoff/eval/summarize_bakeoff.py`](03_4_implementation_experiments/init_bakeoff/eval/summarize_bakeoff.py) | Builds compact bakeoff summaries |

Baseline and bakeoff evidence:

| Path | What it proves |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_corrected_20260521/README.md`](03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_corrected_20260521/README.md) | Corrected V4-HF Apertus baseline |
| [`03_4_implementation_experiments/init_bakeoff/eval/v4_postconv_retry_20260521/README.md`](03_4_implementation_experiments/init_bakeoff/eval/v4_postconv_retry_20260521/README.md) | Raw post-conversion collapse context |
| [`03_4_implementation_experiments/init_bakeoff/eval/V4_BENCHMARK_COMPARISON.md`](03_4_implementation_experiments/init_bakeoff/eval/V4_BENCHMARK_COMPARISON.md) | V4-HF vs post-conversion comparison |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_summary.md`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_summary.md) | Full final checkpoint table |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_digest.md`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_digest.md) | Human-readable final bakeoff digest |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/*_iter0000476_results.json`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/) | Per-arm final lm-eval JSON outputs |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/*_iter0000476_tokenizer_fair_metrics.json`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/) | Final BPC/NLL/tokenizer-fair metrics |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/*_iter0000476_new_token_diagnostics.json`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/) | Final new-token diagnostics for extended arms |
| [`03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md`](03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md) | Per-checkpoint slope and crossover analysis |

## Token Distillation Artifacts

Docs and scripts:

| Path | What it is |
|---|---|
| [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md) | High-level TD plan and concerns |
| [`03_4_implementation_experiments/init_bakeoff/token_distillation/README.md`](03_4_implementation_experiments/init_bakeoff/token_distillation/README.md) | TD implementation runbook |
| [`03_4_implementation_experiments/init_bakeoff/token_distillation/RUN_LOG_20260523.md`](03_4_implementation_experiments/init_bakeoff/token_distillation/RUN_LOG_20260523.md) | TD chronological run log |
| [`03_4_implementation_experiments/init_bakeoff/token_distillation/td_coverage_prepass.py`](03_4_implementation_experiments/init_bakeoff/token_distillation/td_coverage_prepass.py) | CPU firing/coverage scan over actual extended-token emissions |
| [`03_4_implementation_experiments/init_bakeoff/token_distillation/train_retok_td.py`](03_4_implementation_experiments/init_bakeoff/token_distillation/train_retok_td.py) | TD training adapter for fixed-ID ReTok tokenizer |
| [`03_4_implementation_experiments/init_bakeoff/token_distillation/verify_td_preservation.py`](03_4_implementation_experiments/init_bakeoff/token_distillation/verify_td_preservation.py) | Preservation verifier for rows that should not move |

TD evidence:

| Path | What it proves |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/token_distillation/full_td_20260523T092602Z/README.md`](03_4_implementation_experiments/init_bakeoff/token_distillation/full_td_20260523T092602Z/README.md) | Full TD artifact summary |
| [`03_4_implementation_experiments/init_bakeoff/eval/td_full25_intrinsics_20260523T124000Z/TD_PILOT_INTRINSICS_SUMMARY.md`](03_4_implementation_experiments/init_bakeoff/eval/td_full25_intrinsics_20260523T124000Z/TD_PILOT_INTRINSICS_SUMMARY.md) | TD intrinsic comparison |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_digest.md`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_digest.md) | Final TD 2B downstream + intrinsic digest |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_results.json`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_results.json) | Final TD lm-eval JSON |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_tokenizer_fair_metrics.json`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_tokenizer_fair_metrics.json) | Final TD BPC/NLL/tokenizer-fair metrics |
| [`03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_new_token_diagnostics.json`](03_4_implementation_experiments/init_bakeoff/eval/live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_new_token_diagnostics.json) | Final TD new-token diagnostics |

## Production Launcher Artifacts

| Path | What it is |
|---|---|
| [`03_4_implementation_experiments/init_bakeoff/production_cpt/README.md`](03_4_implementation_experiments/init_bakeoff/production_cpt/README.md) | Production launch runbook |
| [`03_4_implementation_experiments/init_bakeoff/production_cpt/submit_vanilla_base_15b_chain.sh`](03_4_implementation_experiments/init_bakeoff/production_cpt/submit_vanilla_base_15b_chain.sh) | Dry-run-by-default 15B production submitter |
| [`03_4_implementation_experiments/init_bakeoff/production_cpt/dryrun_default_vanilla_base_15b_nfc_20260524T121007/submission_plan.json`](03_4_implementation_experiments/init_bakeoff/production_cpt/dryrun_default_vanilla_base_15b_nfc_20260524T121007/submission_plan.json) | Clariden dry-run plan: 15B, Goldfish, 14-job chain |
| [`03_4_implementation_experiments/init_bakeoff/production_cpt/dryrun_default_vanilla_base_15b_nfc_20260524T121007/submission_chain.tsv`](03_4_implementation_experiments/init_bakeoff/production_cpt/dryrun_default_vanilla_base_15b_nfc_20260524T121007/submission_chain.tsv) | Dry-run chain expansion; no jobs submitted |

Default live launch command, after review:

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/production_cpt
DRY_RUN=0 CONFIRM_PRODUCTION_LAUNCH=1 bash submit_vanilla_base_15b_chain.sh
```

The launcher refuses multi-node production by default because the two-node smoke
failed before iteration 1 with NCCL/OFI `NO_SPACE`.

## Current Remote Paths To Verify On CSCS

```bash
test -e /iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched/release
test -e /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document.bin
test -e /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document.idx
test -e /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/production_cpt/submit_vanilla_base_15b_chain.sh
squeue -u fffoivos
```

As of the production launcher dry run, these paths existed and `squeue` was
empty.

## Non-Authoritative Or Stale Context

- [`REVIEW_PRESENTATION.md`](REVIEW_PRESENTATION.md) is useful presentation
  context, but can lag behind the live implementation/evidence. Cross-check it
  against `PRODUCTION_DECISION_STATE.md` and the live artifacts above.
- [`03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/anneal.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/anneal.json)
  is not a production artifact yet. It still needs a proper xfer build from the
  selected post-dedup Greek parquet and local staged replay/code/math sources.
- The old raw HF -> Megatron -> HF conversion result is a failure-mode baseline,
  not the accepted path. Production uses R17-patched checkpoints.
- The code bucket used `codeparrot/codeparrot-clean-train` as the accessible
  cleaned-code fallback, not exact StarCoder/The Stack.
- `SUGGESTIONS.md` is currently an untracked local note in the worktree. Review
  it as context if present, but it is not part of the committed handoff unless
  intentionally added later.

## Reviewer Checklist

- Confirm the selected path is Vanilla/base tokenizer and that the TD result
  does not clear the gate to displace it.
- Confirm the 2B experiment budget was honored; do not confuse corpus-build
  target/available dataset size with per-arm training consumption.
- Confirm the production data prefix comes from NFC-safe base-tokenized data.
- Confirm xIELU/QK-Norm preservation via the Vanilla R17 verification JSON.
- Confirm CPU-only dataset work has xfer guard coverage.
- Dry-run the production submitter on Clariden and inspect
  `submission_plan.json` before any live launch.
- If an anneal phase is desired, treat it as new CPU dataset-build work on xfer,
  not as already ready.
