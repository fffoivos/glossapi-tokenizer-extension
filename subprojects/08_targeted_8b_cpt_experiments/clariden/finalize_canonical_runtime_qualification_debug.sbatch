#!/usr/bin/env bash
#SBATCH --job-name=h2g-qual-final
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_SCALE \
  H2G_QUALIFICATION_RUN_ROOT APERTUS_CANONICAL_ROOT \
  APERTUS_CANONICAL_CODE_RECEIPT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "$H2G_SCALE" == 8b || "$H2G_SCALE" == 1p5b ]] || {
  echo "H2G_SCALE must be 8b or 1p5b" >&2; exit 2;
}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "qualification finalization requires one debug node" >&2; exit 2;
}

verify_code="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify_code" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
/usr/bin/python3.11 "$verify_code" \
  --root "$APERTUS_CANONICAL_ROOT" --receipt "$APERTUS_CANONICAL_CODE_RECEIPT" --kind efficiency

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
canonical_data_root=${H2G_CANONICAL_DATA_ROOT:-$H2G_STAGE_ROOT/canonical/pre_main_data_v2}
manifest="$canonical_data_root/compiled_${H2G_SCALE}_candidate.json"
context="$H2G_STAGE_ROOT/receipts/canonical_gate_context_${H2G_SCALE}.json"
gate_set="$H2G_STAGE_ROOT/receipts/canonical_gate_set_${H2G_SCALE}.json"
qualification="$H2G_STAGE_ROOT/receipts/canonical_runtime_qualification_${H2G_SCALE}.json"
proven_runtime="$canonical_data_root/runtime_${H2G_SCALE}_proven.json"

[[ -s "$manifest" && -d "$H2G_QUALIFICATION_RUN_ROOT" ]] || {
  echo "candidate manifest or qualification run root missing" >&2; exit 2;
}
for output in "$context" "$gate_set" "$qualification" "$proven_runtime"; do
  [[ ! -e "$output" ]] || { echo "immutable qualification output exists: $output" >&2; exit 2; }
done

export H2G_CODE_ROOT H2G_CODE_RECEIPT PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3.11 "$subproject/scripts/build_canonical_qualification_context.py" \
  --manifest "$manifest" --run-root "$H2G_QUALIFICATION_RUN_ROOT" \
  --scale "$H2G_SCALE" --canonical-runner-root "$APERTUS_CANONICAL_ROOT" \
  --output "$context"

PYTHONPATH="$APERTUS_CANONICAL_ROOT/src" /usr/bin/python3.11 -m apertus_cscs_campaign \
  verify --manifest "$manifest" --context "$context" --output "$gate_set"
PYTHONPATH="$APERTUS_CANONICAL_ROOT/src" /usr/bin/python3.11 -m apertus_cscs_campaign \
  qualify-runtime --manifest "$manifest" --gate-set "$gate_set" --output "$qualification"

/usr/bin/python3.11 "$subproject/scripts/promote_canonical_runtime.py" \
  --compiled-candidate "$manifest" --qualification-receipt "$qualification" \
  --canonical-runner-root "$APERTUS_CANONICAL_ROOT" --output "$proven_runtime"
