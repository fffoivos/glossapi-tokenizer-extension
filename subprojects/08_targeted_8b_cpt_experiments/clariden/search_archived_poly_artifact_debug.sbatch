#!/usr/bin/env bash
#SBATCH --job-name=t8b-find-archived-poly
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "archived artifact discovery must run on debug" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

output=${TARGET8_POLY_ARCHIVE_SEARCH_OUTPUT:-$TARGET8_STAGE_ROOT/prelaunch/exact_poly_artifact_archive_search_v4.json}
[[ ! -e "$output" ]] || { echo "immutable archive-search receipt exists" >&2; exit 2; }
minimum_bytes=${TARGET8_ARCHIVE_MINIMUM_BYTES:-1000000}
maximum_bytes=${TARGET8_ARCHIVE_MAXIMUM_BYTES:-20000000000}
timeout_seconds=${TARGET8_ARCHIVE_TIMEOUT_SECONDS:-300}

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/search_archived_poly_artifact.py" \
  --root /users/fffoivos \
  --root /iopsstor/scratch/cscs/fffoivos \
  --root /capstor/scratch/cscs/fffoivos \
  --minimum-bytes "$minimum_bytes" \
  --maximum-bytes "$maximum_bytes" \
  --timeout-seconds "$timeout_seconds" \
  --output "$output"
