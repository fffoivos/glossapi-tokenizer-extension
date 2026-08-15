#!/usr/bin/env bash
#SBATCH --job-name=h2g-datapath
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_PHASE:?set phase 1, 2 or 3}"
: "${H2G_ACTIVE_MODERN_PREFIX:?set exact active-modern binary prefix}"
: "${H2G_ACTIVE_MODERN_RECEIPT:?set active-modern tokenization receipt}"
: "${H2G_FOREIGN_REPLAY_PREFIX:?set exact foreign replay binary prefix}"
: "${H2G_FOREIGN_REPLAY_RECEIPT:?set foreign-replay tokenization receipt}"
: "${H2G_OLD_GREEK_REPLAY_PREFIX:?set exact Old-Greek replay binary prefix}"
: "${H2G_OLD_GREEK_REPLAY_RECEIPT:?set Old-Greek tokenization receipt}"
: "${H2G_PHASE_DATA_PATH_SPEC:?set immutable phase data-path output}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "phase data-path freeze requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
export H2G_CODE_ROOT H2G_CODE_RECEIPT
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_phase_data_path_spec.py" \
  --phase "$H2G_PHASE" --active-modern-prefix "$H2G_ACTIVE_MODERN_PREFIX" \
  --active-modern-receipt "$H2G_ACTIVE_MODERN_RECEIPT" \
  --foreign-replay-prefix "$H2G_FOREIGN_REPLAY_PREFIX" \
  --foreign-replay-receipt "$H2G_FOREIGN_REPLAY_RECEIPT" \
  --old-greek-replay-prefix "$H2G_OLD_GREEK_REPLAY_PREFIX" \
  --old-greek-replay-receipt "$H2G_OLD_GREEK_REPLAY_RECEIPT" \
  --output "$H2G_PHASE_DATA_PATH_SPEC"
