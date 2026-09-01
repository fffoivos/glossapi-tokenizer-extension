#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-full-valid
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=02:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${RUN_ROOT:?set campaign run root}"
: "${CHECKPOINT_RECEIPT:?set final checkpoint receipt}"
OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/full_endpoint_validation}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to replace $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
preflight="$OUTPUT_ROOT/preflight.json"
preflight_args=(
  --campaign-manifest "$CAMPAIGN_MANIFEST" --checkpoint-receipt "$CHECKPOINT_RECEIPT" \
  --output "$preflight"
)
if [[ -n "${RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT:-}" ]]; then
  preflight_args+=(
    --runtime-scientific-bundle "$SCIENTIFIC_BUNDLE"
    --runtime-scientific-bundle-receipt "$RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT"
  )
fi
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/preflight_final_validation.py" \
  "${preflight_args[@]}"

mapfile -t all_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#all_nodes[@]}" -eq 20 ]] || { echo "expected exactly 20 nodes" >&2; exit 2; }
arms=(D0_mixed D1_hard_h_to_g D2_hard_g_to_h D3_gradual_h_to_g D4_gradual_g_to_h)
declare -A pids=()
failure=0
for index in "${!arms[@]}"; do
  arm=${arms[$index]}; offset=$((index * 4))
  node_csv=$(IFS=,; echo "${all_nodes[*]:offset:4}")
  master=${all_nodes[$offset]}; arm_root="$OUTPUT_ROOT/$arm"
  mkdir -p "$arm_root"
  env CAMPAIGN_MANIFEST="$CAMPAIGN_MANIFEST" SEGMENT_PREFLIGHT="$preflight" \
    ARM_ID="$arm" ARM_NODELIST="$node_csv" ARM_OUTPUT_ROOT="$arm_root" \
    MASTER_ADDR="$master" MASTER_PORT="$((29700 + index))" \
    INITIAL_VALIDATION_ONLY=1 EVAL_ITERS=5 \
    bash "$SCIENTIFIC_BUNDLE/clariden/run_production_arm_segment.sh" \
      >"$arm_root/driver.out" 2>"$arm_root/driver.err" &
  pids[$!]=$arm
done
for pid in "${!pids[@]}"; do
  if ! wait "$pid"; then echo "full validation failed: ${pids[$pid]}" >&2; failure=1; fi
done
((failure == 0))
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/finalize_full_endpoint_validation.py" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" --output-root "$OUTPUT_ROOT" \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" \
  --output "$OUTPUT_ROOT/full_endpoint_validation_receipt.json"
