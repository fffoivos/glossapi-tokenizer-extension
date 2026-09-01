#!/usr/bin/env bash
# Create only the final release-polytonic pool receipt after all immutable
# processing and token receipts already exist.
#SBATCH --job-name=t8b-finalize-release-poly
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "release-polytonic finalization must run on debug" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

subproject="$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
stage="$TARGET8_STAGE_ROOT/experiment_a"
decontam="$stage/poly_decontaminated"
eligible="$stage/poly_training_eligible"
receipts="$stage/receipts"
output="$receipts/poly_pool_receipt.json"
[[ ! -e "$output" ]] || { echo "immutable poly pool receipt exists: $output" >&2; exit 2; }
for path in \
  "$stage/polytonic_release_source_audit.json" \
  "$stage/poly_source/extraction_manifest.json" \
  "$stage/poly_release_token_receipt.json" \
  "$decontam/decontamination_manifest.json" \
  "$decontam/audit_receipt.json" \
  "$receipts/poly_greekmmlu_removed_tokens.json" \
  "$eligible/exclusion_manifest.json" \
  "$eligible/audit_receipt.json" \
  "$receipts/poly_validation_excluded_tokens.json"; do
  [[ -s "$path" ]] || { echo "required immutable receipt missing: $path" >&2; exit 2; }
done

uenv run pytorch/v2.9.1:v2 --view=default -- \
  /iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python \
  "$subproject/scripts/finalize_poly_pool.py" \
  --release-source-audit "$stage/polytonic_release_source_audit.json" \
  --source-extraction-manifest "$stage/poly_source/extraction_manifest.json" \
  --source-token-receipt "$stage/poly_release_token_receipt.json" \
  --decontamination-manifest "$decontam/decontamination_manifest.json" \
  --decontamination-audit "$decontam/audit_receipt.json" \
  --decontamination-dropped-tokens "$receipts/poly_greekmmlu_removed_tokens.json" \
  --validation-exclusion-manifest "$eligible/exclusion_manifest.json" \
  --validation-exclusion-audit "$eligible/audit_receipt.json" \
  --validation-excluded-tokens "$receipts/poly_validation_excluded_tokens.json" \
  --output "$output"

/usr/bin/python3.11 - "$output" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))
if value.get("status") != "passed":
    raise SystemExit("release-polytonic pool receipt did not pass")
checks = value.get("invariants", {})
if checks.get("pinned_hf_release_only") is not True:
    raise SystemExit("release-polytonic pool is not pinned-release-only")
if checks.get("no_deduplication_performed") is not True:
    raise SystemExit("release-polytonic pool lacks no-dedup proof")
print(json.dumps({"ok": True, "counts": value["counts"], "receipt": sys.argv[1]}, sort_keys=True))
PY
