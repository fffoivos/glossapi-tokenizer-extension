#!/usr/bin/env bash
# Resume release-only polytonic preparation after an already completed,
# immutable decontamination output.  This intentionally does not re-run or
# replace token counting or decontamination.
#SBATCH --job-name=t8b-resume-release-poly
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "release-polytonic resumption must run on debug" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

subproject="$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
stage="$TARGET8_STAGE_ROOT/experiment_a"
[[ -s "$stage/poly_release_token_receipt.json" ]] || {
  echo "release-only token receipt missing" >&2
  exit 2
}
[[ -s "$stage/poly_decontaminated/decontamination_manifest.json" ]] || {
  echo "completed decontamination manifest missing" >&2
  exit 2
}
for path in \
  "$stage/poly_decontaminated/audit_receipt.json" \
  "$stage/poly_training_eligible" \
  "$stage/receipts/poly_pool_receipt.json"; do
  [[ ! -e "$path" ]] || { echo "immutable output already exists: $path" >&2; exit 2; }
done

bash "$subproject/clariden/audit_poly_decontamination_debug.sbatch"
bash "$subproject/clariden/exclude_poly_validation_debug.sbatch"
bash "$subproject/clariden/audit_poly_validation_debug.sbatch"
bash "$subproject/clariden/finalize_poly_pool_debug.sbatch"

/usr/bin/python3.11 - "$stage" <<'PY'
import json
import sys
from pathlib import Path

stage = Path(sys.argv[1])
paths = {
    "source_tokens": stage / "poly_release_token_receipt.json",
    "decontamination_audit": stage / "poly_decontaminated/audit_receipt.json",
    "validation_audit": stage / "poly_training_eligible/audit_receipt.json",
    "pool": stage / "receipts/poly_pool_receipt.json",
}
for name, path in paths.items():
    value = json.load(path.open())
    if value.get("status") != "passed":
        raise SystemExit(f"{name} receipt did not pass: {path}")
pool = json.load(paths["pool"].open())
if pool.get("invariants", {}).get("pinned_hf_release_only") is not True:
    raise SystemExit("final polytonic pool is not release-only")
if pool.get("invariants", {}).get("no_deduplication_performed") is not True:
    raise SystemExit("final polytonic pool lacks no-dedup proof")
print(json.dumps({"ok": True, "receipts": {key: str(path) for key, path in paths.items()}}, sort_keys=True))
PY
