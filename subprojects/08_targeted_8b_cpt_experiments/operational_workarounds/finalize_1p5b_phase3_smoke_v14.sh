#!/usr/bin/env bash
set -euo pipefail

ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
V14=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260821T183000Z-hard-h2g-phase3-lineage-v14
V21=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260819T024500Z-hard-h2g-phase3-capacity-v21
MEGA="$ST/tools/megatron_training_c92402e_extra_valid_helpers_v2"
S="$V14/subprojects/08_targeted_8b_cpt_experiments"
O="$ST/runs/phase3_resume_smoke_1p5b_3218_v14_direct_salloc_3141832_v8"
AUD="$ST/receipts/phase3_smoke_checkpoint_audit_1p5b_3218.json"
REC="$ST/receipts/phase3_resume_smoke_1p5b_3218.json"
RUN="$ST/runs/h2g_1p5b_2node_v1_retry14_direct_salloc_uenv_20260821"
Q="$RUN/qualification/32fd635f36bea663b3a63a8eabb71d013e7a587b533387d92f31481ba48dccd9/promotion_checkpoint_compatible_v105"
RUNNER=/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260818T211000Z-6b7796a-sequence-range

[[ ! -e "$AUD" && ! -e "$REC" ]] || { echo "immutable 1.5B smoke receipts already exist" >&2; exit 2; }
export H2G_CODE_ROOT="$V14"
export H2G_CODE_RECEIPT="$V14.receipt.json"
/usr/bin/python3.11 "$V14/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$V14" --receipt "$V14.receipt.json" --kind scientific

# The phase-3 cache authority was frozen by V21; audit in that accepted producer
# identity, then finalize the trajectory evidence under the joint V14 identity.
export H2G_CODE_ROOT="$V21"
export H2G_CODE_RECEIPT="$V21.receipt.json"
/usr/bin/python3.11 "$V21/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$V21" --receipt "$V21.receipt.json" --kind scientific
"$RUNNER/bin/apertus-campaign-uenv-exec" "$Q/manifest-proven.json" \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$V21/subprojects/08_targeted_8b_cpt_experiments/runtime_compat:$MEGA" \
  python3 "$V21/subprojects/08_targeted_8b_cpt_experiments/scripts/audit_training_checkpoint.py" \
  --scale 1p5b --source-phase 3 --update 3219 \
  --checkpoint-root "$O/checkpoints/iter_0003219" \
  --source-phase-cache-receipt "$ST/receipts/phase_3_blend_cache.json" \
  --segment-preflight "$O/preflight.json" --training-log "$O/driver.out" \
  --output "$AUD"

export H2G_CODE_ROOT="$V14"
export H2G_CODE_RECEIPT="$V14.receipt.json"
/usr/bin/python3.11 "$S/scripts/finalize_phase3_resume_smoke.py" \
  --scale 1p5b --start-update 3218 --floor-lr 5.5e-6 \
  --preflight "$O/preflight.json" --checkpoint-audit "$AUD" \
  --training-log "$O/driver.out" --output "$REC"

/usr/bin/python3.11 - "$AUD" "$REC" <<'PY'
import json, sys
for path in sys.argv[1:]:
    doc = json.load(open(path, encoding="utf-8"))
    print(path, doc.get("status"), doc.get("scale"), doc.get("update") or doc.get("end_update"))
PY
