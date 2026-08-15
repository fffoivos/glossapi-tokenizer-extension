#!/usr/bin/env bash
# Score all 13 frozen validation panels in four sequential four-GPU groups.
# This reuses the proven group runner while consuming one debug allocation.
#SBATCH --job-name=t8b-docval-all
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_VALIDATION_MANIFEST:?set frozen validation manifest}"
: "${FULL8_HF_MODEL:?set exported HF checkpoint}"
: "${FULL8_HF_TOKENIZER:?set frozen tokenizer}"
: "${FULL8_DOCVAL_OUTPUT:?set final checkpoint document output root}"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "per-document validation must run on debug" >&2
  exit 2
}
[[ "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "per-document validation must use exactly one node" >&2
  exit 2
}
[[ "${SLURM_JOB_NUM_NODES:-0}" == 1 ]] || {
  echo "per-document validation allocation drift" >&2
  exit 2
}
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign the four GPUs explicitly}"

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

case "$FULL8_DOCVAL_OUTPUT" in
  /capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/targeted8b/*/prelaunch/*) ;;
  *) echo "refusing unexpected per-document output: $FULL8_DOCVAL_OUTPUT" >&2; exit 2 ;;
esac
final_output=$FULL8_DOCVAL_OUTPUT
test ! -e "$final_output" || {
  echo "final per-document output already exists: $final_output" >&2
  exit 2
}
output_parent=$(dirname "$final_output")
staging="$final_output.partial.${SLURM_JOB_ID:?missing Slurm job id}"
test ! -e "$staging" || {
  echo "staging per-document output already exists: $staging" >&2
  exit 2
}
mkdir -p "$output_parent"
cleanup() {
  status=$?
  if [[ "$status" -ne 0 && -e "$staging" ]]; then
    rm -rf -- "$staging"
  fi
  exit "$status"
}
trap cleanup EXIT

export FULL8_DOCVAL_OUTPUT="$staging"
for group in 0 1 2 3; do
  export SLURM_ARRAY_TASK_ID=$group
  bash "$FULL8_OPS_ROOT/clariden/run_per_document_group_resource_aware.sh"
done

/usr/bin/python3.11 - "$FULL8_VALIDATION_MANIFEST" "$staging" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
output = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
names = [row["name"] for row in manifest["panels"]]
if len(names) != 13 or len(set(names)) != 13:
    raise SystemExit("frozen validation manifest must contain 13 unique panels")
for name in names:
    receipt_path = output / f"{name}.receipt.json"
    documents_path = output / f"{name}.documents.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "apertus_per_document_validation_v1":
        raise SystemExit(f"receipt schema drift for {name}")
    if str(receipt.get("status", "")).lower() not in {"completed", "frozen", "passed"}:
        raise SystemExit(f"per-document evaluation did not pass for {name}")
    if not documents_path.is_file() or documents_path.stat().st_size <= 0:
        raise SystemExit(f"missing per-document payload for {name}")
receipts = sorted(output.glob("*.receipt.json"))
documents = sorted(output.glob("*.documents.jsonl"))
if len(receipts) != 13 or len(documents) != 13:
    raise SystemExit("per-document output count drift")
print(json.dumps({"status": "passed", "panels": len(names)}, sort_keys=True))
PY

mv "$staging" "$final_output"
trap - EXIT
printf 'published_per_document_output=%s\n' "$final_output"
