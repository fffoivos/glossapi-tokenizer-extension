#!/usr/bin/env bash
set -euo pipefail

updates=(2618 3218 3694)
models=(
  /capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/evaluation/greekmmlu_trajectories/20260822T110500Z-full-clean-v1/exports/8b/iter_0002618/hf
  /capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/evaluation/greekmmlu_trajectories/20260822T110500Z-full-clean-v1/exports/8b/iter_0003218/hf
  /capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/evaluation/final_exports/8b_update3694_v15/hf
)

rank=${SLURM_PROCID:?matrix must run under a three-task srun}
(( rank >= 0 && rank < 3 )) || {
  echo "unexpected matrix rank: $rank" >&2
  exit 2
}

code_root=/iopsstor/scratch/cscs/fffoivos/evals/hard_h2g_legacy_bf16_20260822/code_cfdd0e7b
relative=subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval
script=$code_root/$relative/run_native_greek_mcq_eval.py
registry=$code_root/$relative/native_greek_benchmark_registry.json
output_root=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/evaluation/greekmmlu_legacy_bf16/20260822T224500Z-v1
python=/iopsstor/scratch/cscs/fffoivos/python_envs/h2g_greekmmlu_eval_runtime_20260817_v2/bin/python
cache=/iopsstor/scratch/cscs/fffoivos/native_greek_eval/cache

[[ $(sha256sum "$script" | awk '{print $1}') == b9f75809b6e617cfd419dc5420e480dee72bb3f1df7fa8f82e04793b4dfd19c4 ]]
[[ $(sha256sum "$registry" | awk '{print $1}') == fcf732c142efdd204fe8a64ac4fb1159f47e7b2bac0e947207c2a971329bf508 ]]
[[ -x $python ]]
[[ -d ${models[$rank]} ]]

update=${updates[$rank]}
output=$output_root/iter_$(printf '%07d' "$update")
[[ ! -e $output ]] || {
  echo "immutable legacy output exists: $output" >&2
  exit 2
}
mkdir -p "$output" "$cache"/{hf_home,hf_datasets,xdg,tmp}

export HF_HOME=$cache/hf_home
export HF_DATASETS_CACHE=$cache/hf_datasets
export XDG_CACHE_HOME=$cache/xdg
export TMPDIR=$cache/tmp
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_NO_TF=1 USE_TF=0

exec "$python" "$script" \
  --registry "$registry" \
  --benchmarks greekmmlu \
  --model "hard-h2g-8b-u${update}=${models[$rank]}" \
  --output-dir "$output" \
  --sample-size 0 \
  --random-state 42 \
  --dtype bfloat16 \
  --max-input-tokens 3072 \
  --candidate-batch-size 16 \
  --example-batch-size 16
