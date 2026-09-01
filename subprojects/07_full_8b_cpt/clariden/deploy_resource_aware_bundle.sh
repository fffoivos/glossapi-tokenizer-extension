#!/usr/bin/env bash
# Deploy and freeze the thin full-8B resource-routing bundle.
# Usage: deploy_resource_aware_bundle.sh REMOTE_OPS_ROOT
# The target must not exist. This transfers only small orchestration sources.
set -euo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 REMOTE_OPS_ROOT" >&2; exit 2; }
remote_root=$1
case "$remote_root" in
  /iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt-ops/*) ;;
  *) echo "refusing unexpected remote ops root: $remote_root" >&2; exit 2 ;;
esac

subproject_root=$(cd "$(dirname "$0")/.." && pwd -P)
scientific_root=/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt/20260808T023300Z-sanitized-v45
receipt="$remote_root.receipt.json"
files=(
  clariden/submit_production_resource_aware.sh
  clariden/finalize_and_submit_production_resource_aware.sbatch
  clariden/supervise_campaign_resource_aware.sbatch
  clariden/run_prequeued_train_holder.sbatch
  clariden/prequeue_next_segment_debug.sbatch
  clariden/run_checkpoint_evaluation_debug.sbatch
  clariden/run_per_document_group_debug.sbatch
  clariden/finalize_split_checkpoint_evaluation_debug.sbatch
  clariden/continue_checkpoint_evaluation_debug.sbatch
  clariden/prove_resource_aware_routing.sbatch
  clariden/resource_aware_routing_child.sbatch
  clariden/prove_evaluation_overlap.sbatch
  clariden/prove_successor_launch_gate_debug.sbatch
  clariden/run_per_document_group_resource_aware.sh
  clariden/prepare_successor_stage_debug.sbatch
  clariden/prepare_successor_contracts_debug.sbatch
  scripts/supervise_campaign_resource_aware.py
  scripts/prequeue_next_segment.py
  scripts/compare_successor_stage.py
  scripts/rebind_selected_execution_profile.py
  scripts/successor_semantic_identity.py
  scripts/build_successor_launch_gate.py
  scripts/audit_submitted_job_resources.py
  scripts/analyze_retention_snapshot.py
  scripts/transition_pending_supervisor.py
  scripts/reconstruct_legacy_supervisor_receipt.py
  configs/prequeue_schedule_8b.json
  evaluation/run_checkpoint_evaluation_debug.py
  evaluation/finalize_split_checkpoint_evaluation.py
  evaluation/continue_checkpoint_evaluation.py
  tests/test_resource_routing.py
  tests/test_retention_snapshot.py
)

ssh -o BatchMode=yes clariden \
  "test ! -e '$remote_root' && test ! -e '$receipt' && mkdir -p '$remote_root'"
(
  cd "$subproject_root"
  rsync -aR -- "${files[@]}" "clariden:$remote_root/"
)

ssh -o BatchMode=yes clariden /usr/bin/env \
  REMOTE_OPS_ROOT="$remote_root" SCIENTIFIC_ROOT="$scientific_root" \
  bash -s <<'REMOTE'
set -euo pipefail
for file in "$REMOTE_OPS_ROOT"/clariden/*; do bash -n "$file"; done
/usr/bin/python3.11 - "$REMOTE_OPS_ROOT" <<'PY'
import ast,sys
from pathlib import Path
root=Path(sys.argv[1])
paths=sorted(root.rglob("*.py"))
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
print({"ok":True,"python_files":len(paths)})
PY
/usr/bin/python3.11 "$SCIENTIFIC_ROOT/subprojects/06_dataset_scheduling_experiments/production/freeze_code_bundle.py" \
  --root "$REMOTE_OPS_ROOT" --kind efficiency --output "$REMOTE_OPS_ROOT.receipt.json"
/usr/bin/python3.11 "$SCIENTIFIC_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$REMOTE_OPS_ROOT" --receipt "$REMOTE_OPS_ROOT.receipt.json" --kind efficiency
REMOTE
printf '%s\n' "$remote_root" "$receipt"
