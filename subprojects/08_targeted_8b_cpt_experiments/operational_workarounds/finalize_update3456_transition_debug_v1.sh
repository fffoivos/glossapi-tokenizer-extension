#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
CODE=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T014000Z-hard-h2g-ledger-invariant-v15
SUB="$CODE/subprojects/08_targeted_8b_cpt_experiments"
COMPAT="$ST/receipts/producer_bundle_compatibility_extension_gate_v15_phase3_ledger.json"
SOURCE_CACHE="$ST/receipts/phase_3_blend_cache_v15_postprocess_rebind.json"
export H2G_CODE_ROOT="$CODE" H2G_CODE_RECEIPT="$CODE.receipt.json" H2G_STAGE_ROOT="$ST" PYTHONDONTWRITEBYTECODE=1

rebind_preflight() {
  local original=$1 output=$2
  [[ -s "$output" ]] && return 0
  /usr/bin/python3.11 - "$SUB" "$original" "$SOURCE_CACHE" "$output" <<'PY'
import sys
from pathlib import Path
sub, original, cache, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(sub / "scripts"))
from contract_utils import file_binding, read_json, write_json_atomic
value = read_json(original)
old = value["phase_cache_receipt"]
value["phase_cache_receipt"] = file_binding(cache)
value["segment_contract"]["phase_cache_receipt"] = file_binding(cache)
value["postprocess_rebind"] = {
    "reason": "accept immutable phase-3 cache under the executing V15 audit bundle",
    "original_preflight": file_binding(original),
    "original_phase_cache_receipt": old,
}
write_json_atomic(output, value)
PY
}

finalize_smoke() {
  local scale=$1 job=$2
  local smoke="$ST/runs/direct_v15_${scale}_resume_smoke_3456_${job}"
  local audit="$ST/receipts/direct_v15_${scale}_resume_smoke_3457_checkpoint_audit.json"
  local result="$ST/receipts/direct_v15_${scale}_resume_smoke_3456.json"
  if [[ -s "$audit" && -s "$result" ]]; then
    printf 'resume smoke already finalized: %s\n' "$scale"
    return 0
  fi
  local rebound="$ST/receipts/direct_v15_${scale}_resume_smoke_3456_preflight_phase3_rebind.json"
  rebind_preflight "$smoke/preflight.json" "$rebound"
  uenv run pytorch/v2.9.1:v2 --view=default -- \
    env PYTHONPATH="$SUB/runtime_compat:$ST/tools/megatron_training_c92402e_extra_valid_helpers_v2" RUNTIME_COMPAT_DIR="$SUB/runtime_compat" \
    python3 "$SUB/scripts/audit_training_checkpoint.py" \
      --scale "$scale" --source-phase 3 --update 3457 \
      --checkpoint-root "$smoke/checkpoints/iter_0003457" \
      --source-phase-cache-receipt "$SOURCE_CACHE" \
      --segment-preflight "$rebound" --training-log "$smoke/driver.out" \
      --output "$audit"
  /usr/bin/python3.11 "$SUB/scripts/finalize_phase3_resume_smoke.py" \
    --scale "$scale" --start-update 3456 --floor-lr 5.5e-6 \
    --preflight "$rebound" --checkpoint-audit "$audit" \
    --training-log "$smoke/driver.out" --output "$result"
}

selection=${1:-both}
if [[ "$selection" == both || "$selection" == 8b ]]; then
  finalize_smoke 8b 3140780_retry2
fi
if [[ "$selection" == 8b ]]; then
  printf '%s\n' "8B update-3456 resume smoke finalized"
  exit 0
fi
[[ "$selection" == both ]] || exit 2
finalize_smoke 1p5b 3146990

H2G_AUTHORITY_KIND=resume_3456 \
H2G_8B_SOURCE_RECEIPT="$ST/receipts/direct_v15_8b_resume_smoke_3456.json" \
H2G_1P5B_SOURCE_RECEIPT="$ST/receipts/direct_v15_1p5b_resume_smoke_3456.json" \
  bash "$SUB/clariden/freeze_post_checkpoint_authorities_debug.sbatch"

/usr/bin/python3.11 "$SUB/scripts/freeze_artifact_manifest.py" \
  --gate-stage pre_second_extension --producer-compatibility "$COMPAT" --allow-partial \
  --artifact "both_update_3456_checkpoint_permits=$ST/receipts/checkpoint_pair_3456.json" \
  --artifact "phase_3_3456_to_3457_resume_receipts=$ST/receipts/resume_pair_3456.json" \
  --output "$ST/receipts/artifact_manifest_pre_second_extension_authorization.json"

static_s5() {
  local scale=$1 nodes=$2 tp=$3 mb=$4 permit=$5 load_root=$6 output=$7
  /usr/bin/python3.11 "$SUB/scripts/preflight_train_segment.py" \
    --experiment "$SUB/configs/hard_h_to_g_replication_v1.json" \
    --scale "$scale" --phase 3 --start-update 3456 --exit-update 3694 \
    --load-checkpoint "$load_root" --checkpoint-permit "$permit" \
    --source-phase-cache-receipt "$SOURCE_CACHE" \
    --phase-data-path-spec "$ST/data/phases/phase3/phase_data_path.json" \
    --phase-data-path "1.0 $ST/megatron/phase3_openarchives_ext_text_document 0.253164557 $ST/megatron/phase3_foreign_ext_text_document 0.012658228 $ST/megatron/phase3_old_greek_ext_text_document" \
    --phase-cache-receipt "$SOURCE_CACHE" --phase-cache-root "$ST/data/phases/phase3/cache" \
    --phase-cache-tree-sha256 6575b7e478a1db225facb2d2c2ea6edc62e7dc7cfb0a6a5d9d7b56698f1d2260 \
    --megatron-root "$ST/tools/megatron_training_c92402e_extra_valid_helpers_v2" \
    --megatron-receipt "$ST/receipts/training_megatron_runtime_helpers_v2.json" \
    --validation-root "$ST/validation/historical_148480_v1" \
    --validation-receipt "$ST/receipts/historical_online_validation_148480_canonical_v2.json" \
    --runtime-compat-dir "$SUB/runtime_compat" --runtime-compat-receipt "$ST/receipts/dcp_metadata_compat_v15.json" \
    --extra-valid-sets "hplt openarchives greek_phd english de ru zh code old_greek" \
    --new-greek-valid-sets "greek_phd hplt openarchives" \
    --training-run-permit "$ST/receipts/training_run_permit_${scale}_v15.json" \
    --peak-lr 5.5e-5 --floor-lr 5.5e-6 --nodes "$nodes" --tensor-parallel "$tp" --microbatch "$mb" \
    --canonical-resume --preallocation-static \
    --preauthorization-manifest "$ST/receipts/artifact_manifest_pre_second_extension_authorization.json" \
    --output "$output"
}

static_s5 8b 16 2 2 "$ST/receipts/direct_v15_8b_update3456_checkpoint_permit.json" \
  "$ST/runs/direct_v15_8b_s4_3218_3456_3140780/checkpoints" \
  "$ST/receipts/canonical_static_preflight_v15/8b_s5.json"
static_s5 1p5b 2 1 4 "$ST/receipts/direct_v15_1p5b_update3456_checkpoint_permit.json" \
  "$ST/runs/direct_v15_1p5b_s4_3218_3456_3146990/checkpoints" \
  "$ST/receipts/canonical_static_preflight_v15/1p5b_s5.json"

H2G_GATE_STAGE=pre_second_extension H2G_MANIFEST_MODE=final \
H2G_PRODUCER_COMPATIBILITY="$COMPAT" \
  bash "$SUB/clariden/freeze_extension_artifact_manifest_debug.sbatch"
H2G_GATE_STAGE=pre_second_extension H2G_PRODUCER_COMPATIBILITY="$COMPAT" \
  bash "$SUB/clariden/freeze_extension_launch_gate_debug.sbatch"

printf '%s\n' "update-3456 resume smokes, second-extension gate, and s5 static contracts passed"
