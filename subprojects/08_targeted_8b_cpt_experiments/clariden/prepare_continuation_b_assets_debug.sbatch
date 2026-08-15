#!/usr/bin/env bash
#SBATCH --job-name=t8b-b-assets
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "B asset preparation must run on debug" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

stage="$TARGET8_STAGE_ROOT/experiment_b"
[[ -s "$stage/schedules/schedule_manifest.json" ]] || {
  echo "exact B continuation schedule is not frozen" >&2
  exit 2
}
for path in \
  "$stage/inventory/pool_corpus_receipt.json" \
  "$stage/contracts/training_assets_receipt.json" \
  "$stage/contracts/frozen_experiment_contract.json"; do
  [[ ! -e "$path" ]] || { echo "immutable B asset exists: $path" >&2; exit 2; }
done

export TARGET8_EXPERIMENT=B
bash "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/build_continuation_b_pool_view_debug.sbatch"
bash "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_training_assets_debug.sbatch"
bash "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/freeze_experiment_contract_debug.sbatch"

/usr/bin/python3.11 - "$stage" <<'PY'
import json, sys
from pathlib import Path

stage = Path(sys.argv[1])
checks = {
    "pool": stage / "inventory/pool_corpus_receipt.json",
    "training_assets": stage / "contracts/training_assets_receipt.json",
    "experiment_contract": stage / "contracts/frozen_experiment_contract.json",
}
for name, path in checks.items():
    value = json.load(path.open())
    if str(value.get("status", "")).lower() not in {"passed", "frozen", "completed", "ready", "launch_ready"}:
        raise SystemExit(f"{name} receipt did not pass: {path}")
print(json.dumps({"ok": True, "receipts": {name: str(path) for name, path in checks.items()}}, sort_keys=True))
PY
