# 07 · evaluation — panels, GreekMMLU, per-document scoring and bit-exact exports

> **In one line:** the measurement side of the 8B run — 13 source-conditioned panels every 25 updates, 19 native-GreekMMLU checkpoints, per-document scoring at three milestones, and an export verifier that had to be rewritten because the standard logit check does not fit on one GH200.
> **Period:** 2026-08-05 → 2026-08-09. **Status:** completed; 19 GreekMMLU receipts and 39 document-local receipts bound in the campaign completion receipt.
> **Came from / led to:** the panel design inherited from [`../../06_dataset_scheduling_experiments/evaluation/`](../../06_dataset_scheduling_experiments/evaluation/) → this → [`../presentations/`](../presentations/) and the conclusions in [`../../09_full_8b_cpt_results_analysis/`](../../09_full_8b_cpt_results_analysis/).

## Why this existed

Three measurement regimes had to coexist without any of them stalling training: cheap in-allocation validation on the 16-node training job, expensive checkpoint→HF→GreekMMLU pipelines on separate short allocations, and exact per-document scoring at only three milestones. The `debug-qos` limits on Clariden (one running, two submitted jobs per user) meant the expensive path had to be a *serial self-advancing chain*, not a queued graph.

## The pieces

- **Panels.** [`freeze_validation_manifest.py`](freeze_validation_manifest.py) froze the 12 inherited heldouts plus neutral external Greek. [`finalize_initial_validation.py`](finalize_initial_validation.py) freezes all finite iteration-zero metrics; [`finalize_source_validation_checkpoint.py`](finalize_source_validation_checkpoint.py) freezes one exact-checkpoint panel. Source-conditioned validation itself runs inside `train_segment.sbatch` — every 25 updates in the original recipe, every 238 updates (≈1B token slots) in the derived sanitized one.
- **GreekMMLU.** [`finalize_hf_greekmmlu.py`](finalize_hf_greekmmlu.py) freezes full and decontaminated metrics for an HF checkpoint (16,632 public / 16,159 clean items, `dascim/GreekMMLU@6a03aa06…`). [`run_evaluation_queue.py`](run_evaluation_queue.py) maintains at most four pipelines with bounded retries; [`continue_checkpoint_evaluation.py`](continue_checkpoint_evaluation.py) and [`run_checkpoint_evaluation_debug.py`](run_checkpoint_evaluation_debug.py) implement the serial `afterany` chain each job submits before it starts; [`finalize_split_checkpoint_evaluation.py`](finalize_split_checkpoint_evaluation.py) handles the milestones where GreekMMLU and document scoring run concurrently after conversion, keeping the measured path near an hour instead of an unsafe 88–90 minute sequential tail.
- **Exports.** [`verify_exact_checkpoint_weight_mapping_8b.py`](verify_exact_checkpoint_weight_mapping_8b.py) and [`finalize_checkpoint_export_8b.py`](finalize_checkpoint_export_8b.py) exist because the converter's simultaneous Megatron/HF `--test-logits` diagnostic exceeds one 96 GB GH200. The 8B gate instead proves that **all 323 learned source tensors map bit-exactly into the four HF shards**, including the untied output embedding, and then requires a successful authoritative float32 native-GreekMMLU run of that export (`a0ca3e88`).
- **Per-document scoring.** [`score_documents_hf.py`](score_documents_hf.py) emits `nll_numerator_nats`, `target_tokens`, `utf8_bytes`, `bpb` and base/added-token splits per document, with document-local BOS context and no cross-document carry-over. [`analyze_per_document_endpoints.py`](analyze_per_document_endpoints.py) compares endpoints with paired document/cluster bootstraps — cluster IDs exist only for neutral external Greek, so the other panels' uncertainty must be called a *document* bootstrap until a cluster mapping is frozen.
- **Iteration-zero anchor.** [`materialize_corrected_initial_hf.py`](materialize_corrected_initial_hf.py) froze a corrected-geometry HF view of the zero-drift TD round trip. Four commits on 2026-08-05 were needed to get this right: `151a636a` froze the corrected anchor, `1530ab79` bound it to the canonical tokenizer bytes, `6f7e239a` stopped Slurm export from truncating checkpoint lists, and `7d99513a` (08-08) kept the anchor on its source filesystem.
- **0.5B bridge.** [`build_mini_per_document_manifest.py`](build_mini_per_document_manifest.py) bridges subproject 06's frozen validation manifest to raw per-document inputs — the machinery for the never-executed 0.5B per-document rerun.

## Outcome

- 19 GreekMMLU checkpoints at updates `0, 400, 1192, 2384, 3576, 4768, 5960, 7152, 8344, 9536, 10728, 11920, 13112, 14304, 14627, 15496, 16688, 17880, 18284`; 39 per-document panel receipts across milestones `0`, `14,627` and `18,284`.
- Headline series on the fixed 16,159-question clean subset: 35.782% → 56.810% at update 9,536 → 54.855% terminal; choice NLL 1.45858 → 1.07399 → 1.12213.
- All 13 per-document panels improved from initialization to terminal, and every terminal-minus-cooldown-start delta is negative.
- Measured costs that shaped the schedule: conversion ≈5 minutes, native GreekMMLU ≈54–56 minutes — which is why the inherited 30-minute Mini allocation was replaced by 75 minutes (`6a54d25a`) and why ordinary milestones just fit the 85-minute debug limit.

## Working documents

Fifteen files. Tests: [`../tests/test_full8b_orchestration.py`](../tests/test_full8b_orchestration.py) and [`../tests/test_resource_routing.py`](../tests/test_resource_routing.py).
