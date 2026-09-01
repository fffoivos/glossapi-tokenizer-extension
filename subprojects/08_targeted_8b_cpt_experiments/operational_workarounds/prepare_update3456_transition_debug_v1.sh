#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
CODE=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T014000Z-hard-h2g-ledger-invariant-v15
SUB="$CODE/subprojects/08_targeted_8b_cpt_experiments"
COMPAT="$ST/receipts/producer_bundle_compatibility_extension_gate_v15_phase3_ledger.json"
P8="$ST/receipts/direct_v15_8b_update3456_checkpoint_permit.json"
P1="$ST/receipts/direct_v15_1p5b_update3456_checkpoint_permit.json"
export H2G_CODE_ROOT="$CODE" H2G_CODE_RECEIPT="$CODE.receipt.json" H2G_STAGE_ROOT="$ST"
export PYTHONDONTWRITEBYTECODE=1

bash "$ST/control/retries/postprocess_direct_v15_update3456.sh"

H2G_AUTHORITY_KIND=checkpoint_3456 \
H2G_8B_SOURCE_RECEIPT="$P8" H2G_1P5B_SOURCE_RECEIPT="$P1" \
  bash "$SUB/clariden/freeze_post_checkpoint_authorities_debug.sbatch"

static_smoke() {
  local scale=$1 nodes=$2 tp=$3 mb=$4 permit=$5 load_root=$6 output=$7
  if [[ -s "$output" ]]; then
    printf 'static smoke contract already exists: %s\n' "$scale"
    return 0
  fi
  /usr/bin/python3.11 "$SUB/scripts/preflight_train_segment.py" \
    --experiment "$SUB/configs/hard_h_to_g_replication_v1.json" \
    --scale "$scale" --phase 3 --start-update 3456 --exit-update 3457 \
    --load-checkpoint "$load_root" --checkpoint-permit "$permit" \
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
    --runtime-compat-dir "$SUB/runtime_compat" \
    --runtime-compat-receipt "$ST/receipts/dcp_metadata_compat_v15.json" \
    --extra-valid-sets "hplt openarchives greek_phd english de ru zh code old_greek" \
    --new-greek-valid-sets "greek_phd hplt openarchives" \
    --training-run-permit "$ST/receipts/training_run_permit_${scale}_v15.json" \
    --peak-lr 5.5e-5 --floor-lr 5.5e-6 --nodes "$nodes" \
    --tensor-parallel "$tp" --microbatch "$mb" --one-update-resume-smoke \
    --preallocation-static \
    --preauthorization-manifest "$ST/receipts/artifact_manifest_pre_extension_authorization.json" \
    --output "$output"
}

mkdir -p "$ST/receipts/canonical_static_preflight_v15"
static_smoke 8b 16 2 2 "$P8" \
  "$ST/runs/direct_v15_8b_s4_3218_3456_3140780/checkpoints" \
  "$ST/receipts/canonical_static_preflight_v15/8b_smoke_3456.json"
static_smoke 1p5b 2 1 4 "$P1" \
  "$ST/runs/direct_v15_1p5b_s4_3218_3456_3146990/checkpoints" \
  "$ST/receipts/canonical_static_preflight_v15/1p5b_smoke_3456.json"

printf '%s\n' "update-3456 permits, pair authority, and static smoke contracts passed"
