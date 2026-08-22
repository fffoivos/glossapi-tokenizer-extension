#!/usr/bin/env bash
# Launch the two independent endpoint exports/scores concurrently inside the
# released 16-node training allocation. Scientific inputs and scorer stay frozen.
set -euo pipefail

job_id=${H2G_ADOPTED_JOB_ID:-${SLURM_JOB_ID:-}}
[[ -n "$job_id" ]] || { echo "H2G_ADOPTED_JOB_ID or SLURM_JOB_ID is required" >&2; exit 2; }
code=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T200100Z-hard-h2g-full-public-stablelr-b1951246-v4
stage=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
branch=$stage/stable_peak_branch/20260822T202000Z-v1
trajectory=$branch/evaluation/stable_peak_full_public/20260822T225800Z-v1
runner=$code/subprojects/08_targeted_8b_cpt_experiments/operational_workarounds/run_greekmmlu_trajectory_in_allocation.sh
wrapper=/users/fffoivos/run_rank0_hold_peers_20260822.sh
runtime=/iopsstor/scratch/cscs/fffoivos/python_envs/h2g_greekmmlu_eval_runtime_20260817_v2
examples=$stage/evaluation/greekmmlu_full_public/20260822T193500Z-v1/public_examples.json

pids=()
for update in 3094 3218; do
  done_file=$trajectory/controller_score_${update}_parallel.done
  rm -f "$done_file"
  srun --input=none --jobid="$job_id" --overlap --nodes=4 --ntasks=4 \
    --ntasks-per-node=1 --cpus-per-task=8 \
    env \
      H2G_CODE_ROOT="$code" \
      H2G_CODE_RECEIPT="$code.receipt.json" \
      H2G_STAGE_ROOT="$stage" \
      H2G_TRAJECTORY_ROOT="$trajectory" \
      H2G_CHECKPOINT_SOURCES="$trajectory/checkpoint_${update}_stable_peak.tsv" \
      EVAL_VENV="$runtime" \
      H2G_GREEKMMLU_MODE=full_public \
      H2G_GREEKMMLU_PUBLIC_EXAMPLES="$examples" \
      "$wrapper" "$done_file" bash "$runner" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
