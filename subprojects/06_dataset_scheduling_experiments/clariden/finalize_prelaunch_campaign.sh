#!/usr/bin/env bash
# Close every semantic gate, freeze the campaign, and exercise submission dry-run.
# This script never submits GPU work.
set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}

required=(
  SCIENTIFIC_BUNDLE EFFICIENCY_BUNDLE FINAL_ROOT EXPERIMENT_MATRIX
  OVERLAY_MANIFEST TD_INITIALIZATION_RECEIPT POOL_CORPUS_RECEIPT
  PACKED_CORPUS_RECEIPT SCHEDULE_MANIFEST SCHEDULE_AUDIT GOLDFISH_UNIFORMITY
  VALIDATION_MANIFEST CHECKPOINT_PLAN GREEKMMLU_RUNTIME_SMOKE
  GREEKMMLU_WAVE_SMOKE B1_RESTART_RECEIPT B2_CONTENTION_RECEIPT
  LR_SELECTION_RECEIPT PRELAUNCH_SMOKE_RECEIPT TOKENIZER_DIR
  INITIAL_CHECKPOINT_ROOT TOKEN_BYTE_LENGTHS_RECEIPT GREEKMMLU_CLEAN_SUBSET
  ENDPOINT_BENCHMARK_CONTRACT LM_EVAL_RUNTIME_RECEIPT LM_EVAL_ROOT
  MEGATRON_DIR MEGATRON_RUNTIME_RECEIPT NATIVE_GREEK_EVAL_ROOT
  PYTHON_COMPAT_DIR RUN_PARENT
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required environment variable: $name" >&2; exit 2; }
done
[[ ! -e "$FINAL_ROOT" ]] || { echo "refusing to replace FINAL_ROOT=$FINAL_ROOT" >&2; exit 3; }
mkdir -p "$FINAL_ROOT"

gate_dir="$FINAL_ROOT/gates"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/finalize_launch_gate_set.py" \
  --experiment-matrix "$EXPERIMENT_MATRIX" \
  --overlay-manifest "$OVERLAY_MANIFEST" \
  --initialization-receipt "$TD_INITIALIZATION_RECEIPT" \
  --pool-corpus-receipt "$POOL_CORPUS_RECEIPT" \
  --packed-corpus-receipt "$PACKED_CORPUS_RECEIPT" \
  --schedule-manifest "$SCHEDULE_MANIFEST" --schedule-audit "$SCHEDULE_AUDIT" \
  --goldfish-uniformity "$GOLDFISH_UNIFORMITY" \
  --validation-manifest "$VALIDATION_MANIFEST" --checkpoint-plan "$CHECKPOINT_PLAN" \
  --greekmmlu-runtime-smoke "$GREEKMMLU_RUNTIME_SMOKE" \
  --greekmmlu-wave-smoke "$GREEKMMLU_WAVE_SMOKE" \
  --b1-restart-receipt "$B1_RESTART_RECEIPT" \
  --b2-contention-receipt "$B2_CONTENTION_RECEIPT" \
  --lr-selection-receipt "$LR_SELECTION_RECEIPT" \
  --prelaunch-smoke-receipt "$PRELAUNCH_SMOKE_RECEIPT" \
  --megatron-dir "$MEGATRON_DIR" \
  --megatron-runtime-receipt "$MEGATRON_RUNTIME_RECEIPT" \
  --output-dir "$gate_dir"

mapfile -t gate_paths < <("$HOST_PYTHON" - "$EXPERIMENT_MATRIX" "$gate_dir" <<'PY'
import json,sys
from pathlib import Path
matrix=json.load(open(sys.argv[1])); root=Path(sys.argv[2])
for gate in matrix["launch_gates"]: print(root / f"{gate}.json")
PY
)
gate_args=()
for path in "${gate_paths[@]}"; do gate_args+=(--gate-receipt "$path"); done

authorized_matrix="$FINAL_ROOT/experiment_matrix.authorized.json"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/authorize_experiment_matrix.py" \
  --experiment-matrix "$EXPERIMENT_MATRIX" "${gate_args[@]}" \
  --output "$authorized_matrix"

scientific_receipt="$FINAL_ROOT/scientific_bundle.json"
efficiency_receipt="$FINAL_ROOT/efficiency_bundle.json"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/freeze_code_bundle.py" \
  --root "$SCIENTIFIC_BUNDLE" --kind scientific --output "$scientific_receipt"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/freeze_code_bundle.py" \
  --root "$EFFICIENCY_BUNDLE" --kind efficiency --output "$efficiency_receipt"

campaign="$FINAL_ROOT/campaign_manifest.json"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/build_campaign_manifest.py" \
  --experiment-matrix "$authorized_matrix" \
  --schedule-manifest "$SCHEDULE_MANIFEST" --checkpoint-plan "$CHECKPOINT_PLAN" \
  --tokenizer-dir "$TOKENIZER_DIR" --initial-checkpoint-root "$INITIAL_CHECKPOINT_ROOT" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --initialization-receipt "$TD_INITIALIZATION_RECEIPT" \
  --lr-selection-receipt "$LR_SELECTION_RECEIPT" \
  --token-byte-lengths-receipt "$TOKEN_BYTE_LENGTHS_RECEIPT" \
  --greekmmlu-clean-subset "$GREEKMMLU_CLEAN_SUBSET" \
  --endpoint-benchmark-contract "$ENDPOINT_BENCHMARK_CONTRACT" \
  --lm-eval-runtime-receipt "$LM_EVAL_RUNTIME_RECEIPT" --lm-eval-root "$LM_EVAL_ROOT" \
  "${gate_args[@]}" --megatron-dir "$MEGATRON_DIR" \
  --megatron-runtime-receipt "$MEGATRON_RUNTIME_RECEIPT" \
  --native-greek-eval-root "$NATIVE_GREEK_EVAL_ROOT" \
  --python-compat-dir "$PYTHON_COMPAT_DIR" \
  --scientific-bundle "$SCIENTIFIC_BUNDLE" --efficiency-bundle "$EFFICIENCY_BUNDLE" \
  --scientific-bundle-receipt "$scientific_receipt" \
  --efficiency-bundle-receipt "$efficiency_receipt" --output "$campaign"

CAMPAIGN_MANIFEST="$campaign" RUN_PARENT="$RUN_PARENT" DRY_RUN=1 \
  bash "$SCIENTIFIC_BUNDLE/clariden/submit_production_campaign.sh" \
  | tee "$FINAL_ROOT/submission_dry_run.log"
"$HOST_PYTHON" - "$FINAL_ROOT/prelaunch_closure.json" "$campaign" "$authorized_matrix" "$gate_dir/gate_set_manifest.json" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
from pathlib import Path
out=Path(sys.argv[1]); files=[Path(value) for value in sys.argv[2:]]
def rec(path): return {"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
payload={"schema_version":"apertus_mini_prelaunch_closure_v1","status":"passed_dry_run_only","created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"gpu_jobs_submitted":0,"evidence":[rec(path) for path in files]}
fd,tmp=tempfile.mkstemp(prefix=".closure.",suffix=".partial",dir=out.parent)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp,out); print(json.dumps(payload,sort_keys=True))
PY
