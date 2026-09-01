#!/usr/bin/env bash
#SBATCH --job-name=h2g-ledger-authority
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_PHASE1_CACHE_RECEIPT:?set exact clean Phase-1 cache receipt}"
: "${H2G_PHASE2_CACHE_RECEIPT:?set exact clean Phase-2 cache receipt}"
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_8B_UPDATE_3218_PERMIT:?set exact canonical 8B update-3218 permit}"
: "${H2G_1P5B_UPDATE_3218_PERMIT:?set exact canonical 1.5B update-3218 permit}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "ledger authority requires one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
out="$H2G_STAGE_ROOT/receipts/cross_scale_realized_ledger_match.json"
[[ ! -e "$out" ]] || { echo "immutable ledger authority exists: $out" >&2; exit 2; }
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/freeze_cross_scale_realized_ledger.py" \
  --phase1-cache-receipt "$H2G_PHASE1_CACHE_RECEIPT" \
  --phase2-cache-receipt "$H2G_PHASE2_CACHE_RECEIPT" \
  --realized-ledger-receipt "$H2G_STAGE_ROOT/data/phase3/realized_documents.receipt.json" \
  --8b-checkpoint-permit "$H2G_8B_UPDATE_3218_PERMIT" \
  --1p5b-checkpoint-permit "$H2G_1P5B_UPDATE_3218_PERMIT" \
  --output "$out"
