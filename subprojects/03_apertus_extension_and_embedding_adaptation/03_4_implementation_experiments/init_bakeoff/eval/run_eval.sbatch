#!/usr/bin/env bash
#SBATCH --job-name=apertus_eval
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err
#
# Compute justification (per [[feedback_compute_sweet_spot_justify]]):
#
# 1. Parallelism strategy.
#    lm-eval-harness runs each task sequentially. Inside a task,
#    loglikelihood batches are passed to HF generate/forward which uses
#    the single GPU for batched matmul. The python wrapper around
#    batch_size=auto picks the largest power-of-two that fits in GPU
#    memory (observed: 32 for Apertus-8B at bf16 on an H100).
#
# 2. GPU saturation.
#    Apertus-8B (8.6 GB bf16 + KV cache for batch≤32 ≈ ~20 GB) fits on
#    one H100 (80 GB). Multi-GPU sharding via accelerate adds NCCL
#    overhead with no throughput gain at this model size. --gpus-per-node=1
#    is the sweet spot. (Switch to --gpus-per-node=2 with
#    `parallelize=True` only for the 70B-scale models, not relevant here.)
#
# 3. CPU + memory ceiling.
#    --cpus-per-task=72 = one socket; HF dataloader workers + tokenizer
#    pre-batching consume ~16 cores during the bigger Greek tasks, so 72
#    is comfortable headroom and leaves the other 3 sockets free for
#    co-tenants. --mem=200G covers cached HF datasets (~5 GB) + model
#    weights (~17 GB bf16) + activations + batched KV cache.
#
# 4. Known efficiency gaps.
#    Per-task sequential. If the wallclock matters for the bakeoff arms
#    (16 task labels × ~30 min each = ~8 h), we'd split task_list across
#    multiple sbatches with --array, each picking up a subset. For V4
#    baseline (one run) the sequential single-GPU path is fine.
#
# Parameterized eval job using the swiss-ai/lm-evaluation-harness clone,
# pre-installed at $LM_EVAL_PYTHONPATH (target-install — uenv site-packages
# are read-only so pip install -e . into the uenv fails).
#
# Required environment variables:
#   MODEL_PATH    — path to HF-format model directory
#   OUTPUT_DIR    — where to drop the eval JSON + per-sample logs
#   TASK_GROUP    — full | retention_only | greek_only | safety_only
#
# Optional:
#   BATCH_SIZE    — default "auto"
#   LIMIT         — optional per-task sample cap
#   DTYPE         — default "bfloat16"
#   UENV_IMAGE    — default pytorch/v2.9.1:v2 (transformers 4.57.0, has ApertusForCausalLM)
#
# Greek task names are the ones that exist NATIVELY in swissai harness as of
# 2026-05-21. Gaps (hellaswag_greek, winogrande_greek, mmlu_pro_greek,
# truthfulqa_greek) are tracked by PF5 — custom YAMLs from ILSP datasets.
# The original V4 set assumed those existed; they did not.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
TASK_GROUP="${TASK_GROUP:-full}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"

LM_EVAL_PYTHONPATH="${LM_EVAL_PYTHONPATH:-/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval}"

# Task lists per group. Names verified to exist in swissai harness at 2026-05-21.
declare -A TASK_LISTS
# Group 1: retention. Apertus pretraining-eval set + multilingual NLI/COPA.
# Table 14 includes both MMLU and Global-MMLU; keep global_mmlu here so the
# V4 baseline can actually fill the documented §5.6 retention gates.
TASK_LISTS[retention]="arc_challenge,arc_easy,hellaswag,winogrande,piqa,mmlu,global_mmlu,xnli,xcopa"
# Group 2: Greek tasks that exist natively in swissai harness.
TASK_LISTS[greek]="arc_challenge_mt_el,xnli_el,xquad_el,belebele_ell_Grek,global_mmlu_full_el,include_base_44_greek_few_shot_en,global_piqa_completions_ell_grek"
# Group 3: safety. Lower priority for bakeoff selection.
TASK_LISTS[safety]="harmbench,polyglotoxicityprompts"

case "$TASK_GROUP" in
    full)            TASK_LIST="${TASK_LISTS[retention]},${TASK_LISTS[greek]}" ;;
    retention_only)  TASK_LIST="${TASK_LISTS[retention]}" ;;
    greek_only)      TASK_LIST="${TASK_LISTS[greek]}" ;;
    safety_only)     TASK_LIST="${TASK_LISTS[safety]}" ;;
    *)               echo "ERROR: TASK_GROUP must be full|retention_only|greek_only|safety_only" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=== run_eval.sbatch ==="; date -u
echo "host: $(hostname)"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TASK_GROUP: $TASK_GROUP -> $TASK_LIST"
echo "DTYPE: $DTYPE  BATCH: $BATCH_SIZE"
echo

UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"

cat > run_metadata.json <<META
{
  "model_path": "$MODEL_PATH",
  "task_group": "$TASK_GROUP",
  "task_list": "$TASK_LIST",
  "batch_size": "$BATCH_SIZE",
  "dtype": "$DTYPE",
  "lm_eval_pythonpath": "$LM_EVAL_PYTHONPATH",
  "lm_eval_source": "swiss-ai/lm-evaluation-harness (target-install)",
  "uenv_image": "$UENV_IMAGE",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "host": "$(hostname)",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

uenv start "$UENV_IMAGE" --view=default --ignore-tty <<INNER
set -euo pipefail
export PYTHONPATH=$LM_EVAL_PYTHONPATH
python3 -m lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=$DTYPE,trust_remote_code=True \
    --tasks $TASK_LIST \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_DIR/results.json \
    --log_samples \
    $LIMIT_ARG
INNER

python3 -c "
import json
from pathlib import Path
md = json.loads(Path('run_metadata.json').read_text())
md['end_time'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
Path('run_metadata.json').write_text(json.dumps(md, indent=2))
print('updated run_metadata.json with end_time')
"

echo
echo "=== eval complete ==="; date -u
# Avoid `ls | head` — pipefail catches SIGPIPE on ls (exit 141 → slurm shows
# exit 13) and marks the whole job FAILED even though eval succeeded.
# (Bit V4 job 2334245 on 2026-05-21.)
ls -la "$OUTPUT_DIR" || true
echo
echo "Next: compute_bootstrap_cis.py $OUTPUT_DIR/samples_*.jsonl"
