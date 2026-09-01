#!/usr/bin/env bash
set -euo pipefail

ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
V14=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260821T183000Z-hard-h2g-phase3-lineage-v14
SUB="$V14/subprojects/08_targeted_8b_cpt_experiments"
COMP="$ST/receipts/producer_bundle_compatibility_extension_gate_v14_phase3_ledger_v2.json"
RUN8="$ST/runs/efficiency_bound_proven_8b_v112_recovery_codebinding_v3_staticready"
RUN1="$ST/runs/h2g_1p5b_2node_v1_retry14_direct_salloc_uenv_20260821"
P8="$RUN8/segments/s3/attempts/attempt_000003/checkpoint_permit.ca9dd64_postprocess_fix.json"
P1="$RUN1/segments/s3/attempts/attempt_000002/checkpoint_permit.json"
S8="$ST/receipts/phase3_resume_smoke_8b_3218.json"
S1="$ST/receipts/phase3_resume_smoke_1p5b_3218.json"
PRE="$ST/receipts/artifact_manifest_pre_extension_authorization.json"
AUTH="$ST/receipts/owner_extension_authorization.json"
FINAL="$ST/receipts/artifact_manifest_pre_extension.json"
GATE="$ST/receipts/launch_gate_pre_extension.json"

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "joint gate must run inside one held debug node" >&2; exit 2;
}
for path in "$P8" "$P1" "$S8" "$S1" "$COMP"; do
  [[ -s "$path" ]] || { echo "required joint-gate input missing: $path" >&2; exit 2; }
done
for path in \
  "$ST/receipts/cross_scale_realized_ledger_match.json" \
  "$ST/receipts/phase_local_cursor_guard.json" \
  "$ST/receipts/constant_floor_scheduler_resume.json" \
  "$PRE" "$AUTH" "$FINAL" "$GATE"; do
  [[ ! -e "$path" ]] || { echo "immutable joint-gate output already exists: $path" >&2; exit 2; }
done

common=(
  H2G_CODE_ROOT="$V14"
  H2G_CODE_RECEIPT="$V14.receipt.json"
  H2G_STAGE_ROOT="$ST"
  H2G_PRODUCER_COMPATIBILITY="$COMP"
)

env "${common[@]}" \
  H2G_PHASE1_CACHE_RECEIPT="$ST/receipts/phase_1_blend_cache_runtime_exact_v86.json" \
  H2G_PHASE2_CACHE_RECEIPT="$ST/receipts/phase_2_blend_cache_runtime_exact_v89r1.json" \
  H2G_8B_UPDATE_3218_PERMIT="$P8" H2G_1P5B_UPDATE_3218_PERMIT="$P1" \
  bash "$SUB/clariden/freeze_cross_scale_realized_ledger_debug.sbatch"

env "${common[@]}" H2G_AUTHORITY_KIND=phase3_entry \
  H2G_8B_SOURCE_RECEIPT="$S8" H2G_1P5B_SOURCE_RECEIPT="$S1" \
  bash "$SUB/clariden/freeze_post_checkpoint_authorities_debug.sbatch"

env "${common[@]}" H2G_GATE_STAGE=pre_extension \
  H2G_MANIFEST_MODE=pre_authorization \
  H2G_8B_UPDATE_3218_PERMIT="$P8" H2G_1P5B_UPDATE_3218_PERMIT="$P1" \
  bash "$SUB/clariden/freeze_extension_artifact_manifest_debug.sbatch"

env "${common[@]}" H2G_AUTHORIZATION_STAGE=pre_extension \
  H2G_OWNER_CONFIRMATION="The owner explicitly directed us to start the training extension and keep monitoring it through completion." \
  H2G_OWNER_CONFIRMED_AT="2026-08-21T23:59:00+03:00" \
  H2G_PREAUTHORIZATION_MANIFEST="$PRE" \
  H2G_OWNER_AUTHORIZATION_OUTPUT="$AUTH" \
  bash "$SUB/clariden/freeze_owner_authorization_debug.sbatch"

env "${common[@]}" H2G_GATE_STAGE=pre_extension H2G_MANIFEST_MODE=final \
  H2G_8B_UPDATE_3218_PERMIT="$P8" H2G_1P5B_UPDATE_3218_PERMIT="$P1" \
  bash "$SUB/clariden/freeze_extension_artifact_manifest_debug.sbatch"

env "${common[@]}" H2G_GATE_STAGE=pre_extension \
  bash "$SUB/clariden/freeze_extension_launch_gate_debug.sbatch"

/usr/bin/python3.11 - "$GATE" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
assert doc["status"] == "launch_ready", doc
assert doc["gate_stage"] == "pre_extension", doc
print(sys.argv[1], doc["status"], doc["gate_stage"])
PY
