#!/usr/bin/env bash
# Freeze only the new experiment wrapper and the no-save control hook over the
# already hardware-proven v39 scientific bundle. No data/model artifact moves.
set -euo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 REMOTE_ROOT" >&2; exit 2; }
remote_root=$1
base=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260812T084500Z-targeted8b-v39
case "$remote_root" in /iopsstor/scratch/cscs/fffoivos/orchestration/early-cooldown-8b/*) ;; *) echo "unexpected bundle root" >&2; exit 2;; esac
repo_root=$(cd "$(dirname "$0")/../../.." && pwd -P)
receipt="$remote_root.receipt.json"
ssh clariden /usr/bin/env REMOTE_ROOT="$remote_root" BASE="$base" RECEIPT="$receipt" bash -s <<'REMOTE'
set -euo pipefail
[[ -d "$BASE" && ! -e "$REMOTE_ROOT" && ! -e "$RECEIPT" ]]
mkdir -p "$(dirname "$REMOTE_ROOT")"
cp -a "$BASE" "$REMOTE_ROOT"
chmod -R u+w "$REMOTE_ROOT"
REMOTE
rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' \
  "$repo_root/subprojects/10_early_cooldown_causal_experiment/" \
  "clariden:$remote_root/subprojects/10_early_cooldown_causal_experiment/"
rsync -a "$repo_root/subprojects/06_dataset_scheduling_experiments/training/exact_checkpoint_hook.py" \
  "clariden:$remote_root/subprojects/06_dataset_scheduling_experiments/training/exact_checkpoint_hook.py"
ssh clariden /usr/bin/env REMOTE_ROOT="$remote_root" RECEIPT="$receipt" bash -s <<'REMOTE'
set -euo pipefail
find "$REMOTE_ROOT/subprojects/10_early_cooldown_causal_experiment/clariden" -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0 |
  while IFS= read -r -d '' file; do bash -n "$file"; done
/usr/bin/python3.11 - "$REMOTE_ROOT/subprojects/10_early_cooldown_causal_experiment" <<'PY'
import ast,sys
from pathlib import Path
paths=sorted(Path(sys.argv[1]).rglob('*.py'))
for path in paths: ast.parse(path.read_text(),filename=str(path))
print({'ok':True,'python_files':len(paths)})
PY
/usr/bin/python3.11 "$REMOTE_ROOT/subprojects/10_early_cooldown_causal_experiment/scripts/prepare_launch.py" \
  --contract "$REMOTE_ROOT/subprojects/10_early_cooldown_causal_experiment/configs/experiment_contract.json" --static-only
/usr/bin/python3.11 "$REMOTE_ROOT/subprojects/06_dataset_scheduling_experiments/production/freeze_code_bundle.py" \
  --root "$REMOTE_ROOT" --kind scientific --output "$RECEIPT"
/usr/bin/python3.11 "$REMOTE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$REMOTE_ROOT" --receipt "$RECEIPT" --kind scientific
chmod -R a-w "$REMOTE_ROOT"
REMOTE
printf '%s\n%s\n' "$remote_root" "$receipt"
