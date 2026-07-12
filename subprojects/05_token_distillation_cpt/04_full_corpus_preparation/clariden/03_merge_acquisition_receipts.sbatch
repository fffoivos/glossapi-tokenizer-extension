#!/usr/bin/env bash
# Lightweight, receipt-only merge of the independently acquired HF and MDC
# inventories. No corpus payload is copied or transformed by this job.
#SBATCH --job-name=cpt4_merge_acquire
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "HF and MDC acquisition-receipt merge"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

HF_RECEIPT=${HF_ACQUISITION_RECEIPT:?set HF_ACQUISITION_RECEIPT to the passed HF receipt}
MDC_RECEIPT=${MDC_ACQUISITION_RECEIPT:?set MDC_ACQUISITION_RECEIPT to the passed MDC receipt}
OUTPUT=${MERGED_ACQUISITION_RECEIPT:?set an immutable MERGED_ACQUISITION_RECEIPT output path}
phase04_require_file "$HF_RECEIPT"
phase04_require_file "$MDC_RECEIPT"
[[ ! -e "$OUTPUT" ]] || {
    echo "ERROR: refusing to overwrite merged acquisition receipt: $OUTPUT" >&2
    exit 84
}
mkdir -p "$(dirname "$OUTPUT")"
command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 82; }
exec 9>"$(dirname "$OUTPUT")/.phase04-acquisition-receipt-merge.lock"
flock -n 9 || { echo "ERROR: another acquisition-receipt merge is active" >&2; exit 83; }

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/merge_acquisition_receipts.py" \
    --sources "$SOURCE_CONFIG" \
    --hf-receipt "$HF_RECEIPT" \
    --mdc-receipt "$MDC_RECEIPT" \
    --destination-root "$DATA_ROOT" \
    --expected-code-commit "$PHASE04_EXPECTED_COMMIT" \
    --output "$OUTPUT"

echo "ACQUISITION_RECEIPT=$OUTPUT"
