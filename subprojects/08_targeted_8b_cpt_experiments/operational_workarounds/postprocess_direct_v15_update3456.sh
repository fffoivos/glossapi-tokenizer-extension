#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "direct endpoint postprocess requires one debug node" >&2
  exit 2
}
ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
CODE=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T014000Z-hard-h2g-ledger-invariant-v15
SUB="$CODE/subprojects/08_targeted_8b_cpt_experiments"
RUNTIME="$SUB/runtime_compat"
SOURCE_CACHE="$ST/receipts/phase_3_blend_cache_v15_postprocess_rebind.json"
export H2G_CODE_ROOT="$CODE" H2G_CODE_RECEIPT="$CODE.receipt.json" PYTHONDONTWRITEBYTECODE=1

if [[ ! -s "$SOURCE_CACHE" ]]; then
  /usr/bin/python3.11 "$SUB/scripts/freeze_phase_blend_cache.py" \
    --phase 3 --data-path-spec "$ST/data/phases/phase3/phase_data_path.json" \
    --cache-root "$ST/data/phases/phase3/cache" \
    --blend-manifest "$ST/receipts/phase_3_cache_build.json" --output "$SOURCE_CACHE"
fi

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

postprocess() {
  local scale=$1 output_root=$2 preflight=$3 training_log=$4 prefix=$5
  local checkpoint_root="$output_root/checkpoints/iter_0003456"
  local audit="$ST/receipts/${prefix}_checkpoint_audit.json"
  local permit="$ST/receipts/${prefix}_checkpoint_permit.json"
  [[ -f "$output_root/checkpoints/latest_checkpointed_iteration.txt" ]] || return 3
  [[ "$(<"$output_root/checkpoints/latest_checkpointed_iteration.txt")" == 3456 ]] || return 3
  if [[ -s "$audit" && -s "$permit" ]]; then
    printf 'endpoint already postprocessed: %s\n' "$scale"
    return 0
  fi
  [[ ! -e "$audit" && ! -e "$permit" ]] || return 4
  uenv run pytorch/v2.9.1:v2 --view=default -- \
    env PYTHONPATH="$RUNTIME:$ST/tools/megatron_training_c92402e_extra_valid_helpers_v2" RUNTIME_COMPAT_DIR="$RUNTIME" \
    python3 "$SUB/scripts/audit_training_checkpoint.py" \
      --scale "$scale" --source-phase 3 --update 3456 \
      --checkpoint-root "$checkpoint_root" \
      --source-phase-cache-receipt "$SOURCE_CACHE" \
      --segment-preflight "$preflight" --training-log "$training_log" \
      --output "$audit"
  /usr/bin/python3.11 "$SUB/scripts/build_checkpoint_permit.py" \
    --scale "$scale" --source-phase 3 --update 3456 \
    --checkpoint-root "$checkpoint_root" --checkpoint-audit "$audit" \
    --source-phase-cache-receipt "$SOURCE_CACHE" --output "$permit"
}

selection=${1:-both}
if [[ "$selection" == both || "$selection" == 8b ]]; then
rebind_preflight \
  "$ST/receipts/train_preflight/8b_p3_3218_3456_8b_s4_3140780_v15.json" \
  "$ST/receipts/train_preflight/8b_p3_3218_3456_8b_s4_3140780_v15_phase3_rebind.json"
postprocess 8b \
  "$ST/runs/direct_v15_8b_s4_3218_3456_3140780" \
  "$ST/receipts/train_preflight/8b_p3_3218_3456_8b_s4_3140780_v15_phase3_rebind.json" \
  "$ST/control/retries/direct_v15_8b_s4_3140780.launch.log" \
  direct_v15_8b_update3456
fi
if [[ "$selection" == both || "$selection" == 1p5b ]]; then
rebind_preflight \
  "$ST/receipts/train_preflight/1p5b_p3_3218_3456_1p5b_s4_3146990_v15.json" \
  "$ST/receipts/train_preflight/1p5b_p3_3218_3456_1p5b_s4_3146990_v15_phase3_rebind.json"
postprocess 1p5b \
  "$ST/runs/direct_v15_1p5b_s4_3218_3456_3146990" \
  "$ST/receipts/train_preflight/1p5b_p3_3218_3456_1p5b_s4_3146990_v15_phase3_rebind.json" \
  "$ST/control/retries/direct_v15_1p5b_s4_3146990.launch.log" \
  direct_v15_1p5b_update3456
fi
[[ "$selection" == both || "$selection" == 8b || "$selection" == 1p5b ]] || exit 2
