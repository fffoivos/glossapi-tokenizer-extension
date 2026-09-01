#!/usr/bin/env bash
# Stage only the causal wrapper and checkpoint hook from the Mac, then submit a
# debug job to copy, validate, hash and freeze the scientific bundle.
set -euo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 REMOTE_ROOT" >&2; exit 2; }
remote_root=$1
base=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260812T084500Z-targeted8b-v39
case "$remote_root" in /iopsstor/scratch/cscs/fffoivos/orchestration/early-cooldown-8b/*) ;; *) echo "unexpected bundle root" >&2; exit 2;; esac
repo_root=$(cd "$(dirname "$0")/../../.." && pwd -P)
receipt="$remote_root.receipt.json"
staging_root="${remote_root}.staging"
ssh clariden /usr/bin/env REMOTE_ROOT="$remote_root" STAGING_ROOT="$staging_root" RECEIPT="$receipt" bash -s <<'REMOTE'
set -euo pipefail
[[ ! -e "$REMOTE_ROOT" && ! -e "$RECEIPT" && ! -e "$STAGING_ROOT" ]]
mkdir -p "$STAGING_ROOT/subproject"
mkdir -p /capstor/scratch/cscs/fffoivos/runs/10_early_cooldown/_bundle_logs
REMOTE
rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyo' \
  "$repo_root/subprojects/10_early_cooldown_causal_experiment/" \
  "clariden:$staging_root/subproject/"
rsync -a "$repo_root/subprojects/06_dataset_scheduling_experiments/training/exact_checkpoint_hook.py" \
  "clariden:$staging_root/exact_checkpoint_hook.py"
job=$(ssh clariden sbatch --parsable --partition=debug --nodes=1 --time=00:20:00 \
  --output="/capstor/scratch/cscs/fffoivos/runs/10_early_cooldown/_bundle_logs/%x-%j.out" \
  --error="/capstor/scratch/cscs/fffoivos/runs/10_early_cooldown/_bundle_logs/%x-%j.err" \
  --export="ALL,EARLY_BASE_BUNDLE=$base,EARLY_STAGING_ROOT=$staging_root,EARLY_CODE_ROOT=$remote_root,EARLY_CODE_BUNDLE_RECEIPT=$receipt" \
  "$staging_root/subproject/clariden/freeze_bundle_debug.sbatch")
printf '%s\n%s\n%s\n' "$job" "$remote_root" "$receipt"
