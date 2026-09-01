#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
CODE=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T014000Z-hard-h2g-ledger-invariant-v15
SUB="$CODE/subprojects/08_targeted_8b_cpt_experiments"
export H2G_CODE_ROOT="$CODE" H2G_CODE_RECEIPT="$CODE.receipt.json" H2G_STAGE_ROOT="$ST" PYTHONDONTWRITEBYTECODE=1
bash "$ST/control/retries/postprocess_direct_v15_update3456.sh" 8b
/usr/bin/python3.11 "$SUB/scripts/preflight_train_segment.py" \
  --experiment "$SUB/configs/hard_h_to_g_replication_v1.json" \
  --scale 8b --phase 3 --start-update 3456 --exit-update 3457 \
  --load-checkpoint "$ST/runs/direct_v15_8b_s4_3218_3456_3140780/checkpoints" \
  --checkpoint-permit "$ST/receipts/direct_v15_8b_update3456_checkpoint_permit.json" \
  --source-phase-cache-receipt "$ST/receipts/phase_3_blend_cache_v15_postprocess_rebind.json" \
  --phase-data-path-spec "$ST/data/phases/phase3/phase_data_path.json" \
  --phase-data-path "1.0 $ST/megatron/phase3_openarchives_ext_text_document 0.253164557 $ST/megatron/phase3_foreign_ext_text_document 0.012658228 $ST/megatron/phase3_old_greek_ext_text_document" \
  --phase-cache-receipt "$ST/receipts/phase_3_blend_cache.json" \
  --phase-cache-root "$ST/data/phases/phase3/cache" \
  --phase-cache-tree-sha256 6575b7e478a1db225facb2d2c2ea6edc62e7dc7cfb0a6a5d9d7b56698f1d2260 \
  --megatron-root "$ST/tools/megatron_training_c92402e_extra_valid_helpers_v2" \
  --megatron-receipt "$ST/receipts/training_megatron_runtime_helpers_v2.json" \
  --validation-root "$ST/validation/historical_148480_v1" \
  --validation-receipt "$ST/receipts/historical_online_validation_148480_canonical_v2.json" \
  --runtime-compat-dir "$SUB/runtime_compat" --runtime-compat-receipt "$ST/receipts/dcp_metadata_compat_v15.json" \
  --extra-valid-sets "hplt openarchives greek_phd english de ru zh code old_greek" \
  --new-greek-valid-sets "greek_phd hplt openarchives" \
  --training-run-permit "$ST/receipts/training_run_permit_8b_v15.json" \
  --peak-lr 5.5e-5 --floor-lr 5.5e-6 --nodes 16 --tensor-parallel 2 --microbatch 2 \
  --one-update-resume-smoke --preallocation-static \
  --preauthorization-manifest "$ST/receipts/artifact_manifest_pre_extension_authorization.json" \
  --output "$ST/receipts/canonical_static_preflight_v15/8b_smoke_3456.json"
