#!/usr/bin/env bash
#SBATCH --job-name=early-wsd-freeze
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${EARLY_BASE_BUNDLE:?set hardware-proven base bundle}"
: "${EARLY_STAGING_ROOT:?set staged causal code overlay}"
: "${EARLY_CODE_ROOT:?set new immutable bundle root}"
: "${EARLY_CODE_BUNDLE_RECEIPT:?set new bundle receipt}"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "bundle materialization and hashing must run on debug" >&2
  exit 2
}
[[ -d "$EARLY_BASE_BUNDLE" && -f "$EARLY_BASE_BUNDLE.receipt.json" ]]
[[ -d "$EARLY_STAGING_ROOT/subproject" ]]
[[ -f "$EARLY_STAGING_ROOT/exact_checkpoint_hook.py" ]]
[[ ! -e "$EARLY_CODE_ROOT" && ! -e "$EARLY_CODE_BUNDLE_RECEIPT" ]]

base_production="$EARLY_BASE_BUNDLE/subprojects/06_dataset_scheduling_experiments/production"
/usr/bin/python3.11 "$base_production/verify_code_bundle.py" \
  --root "$EARLY_BASE_BUNDLE" --receipt "$EARLY_BASE_BUNDLE.receipt.json" --kind scientific

mkdir -p "$(dirname "$EARLY_CODE_ROOT")"
cp -a "$EARLY_BASE_BUNDLE" "$EARLY_CODE_ROOT"
chmod -R u+w "$EARLY_CODE_ROOT"
mkdir -p "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment"
cp -a "$EARLY_STAGING_ROOT/subproject/." \
  "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/"
cp "$EARLY_STAGING_ROOT/exact_checkpoint_hook.py" \
  "$EARLY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/training/exact_checkpoint_hook.py"

find "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/clariden" \
  -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0 |
  while IFS= read -r -d '' file; do bash -n "$file"; done
/usr/bin/python3.11 - "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment" <<'PY'
import ast
import sys
from pathlib import Path

paths = sorted(Path(sys.argv[1]).rglob("*.py"))
for path in paths:
    ast.parse(path.read_text(), filename=str(path))
print({"ok": True, "python_files": len(paths)})
PY
/usr/bin/python3.11 \
  "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/scripts/prepare_launch.py" \
  --contract "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/configs/experiment_contract.json" \
  --static-only
/usr/bin/python3.11 "$base_production/freeze_code_bundle.py" \
  --root "$EARLY_CODE_ROOT" --kind scientific --output "$EARLY_CODE_BUNDLE_RECEIPT"
/usr/bin/python3.11 "$base_production/verify_code_bundle.py" \
  --root "$EARLY_CODE_ROOT" --receipt "$EARLY_CODE_BUNDLE_RECEIPT" --kind scientific
chmod -R a-w "$EARLY_CODE_ROOT"
