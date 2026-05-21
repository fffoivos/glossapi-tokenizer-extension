#!/usr/bin/env bash
#SBATCH --job-name=apertus_eval
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err
#
# Parameterized eval job using swiss-ai/lm-evaluation-harness.
#
# Required environment variables (export via --export or set before sbatch):
#   MODEL_PATH    — path to HF-format model directory (e.g. .../apertus-8b-2509 or .../init_checkpoints/retok)
#   OUTPUT_DIR    — where to drop the eval JSON + per-sample logs
#   TASK_GROUP    — one of: full | retention_only | greek_only | safety_only
#
# Optional:
#   BATCH_SIZE    — default "auto" (lm-eval-harness auto-tunes)
#   LIMIT         — optional sample-count cap per task (for debugging)
#   DTYPE         — default "bfloat16"
#
# Submit:
#   sbatch --export=MODEL_PATH=...,OUTPUT_DIR=...,TASK_GROUP=full run_eval.sbatch
#
# Output (under $OUTPUT_DIR):
#   results.json                  — lm-eval-harness aggregated metrics
#   samples_<task>.jsonl          — per-sample outputs (for bootstrap CI)
#   run_metadata.json             — model path, harness commit, task list, walltime

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
TASK_GROUP="${TASK_GROUP:-full}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"

LM_EVAL_HARNESS_DIR="${LM_EVAL_HARNESS_DIR:-/iopsstor/scratch/cscs/fffoivos/code/eval/lm-evaluation-harness-swissai}"

# Task lists per group. Synced to EVAL_RECIPE.md (Apertus Table-14 pretraining
# evals only; Table-22 post-training tasks excluded — see EVAL_RECIPE.md "Scope
# of tasks: match Apertus's pretraining-eval table, not post-training").
declare -A TASK_LISTS
# Group 1: Apertus pretraining-eval set per Table 14.
# Includes Global-MMLU (multilingual Greek-inclusive); excludes GSM8K, HumanEval
# (Table-22 post-training only, won't fluctuate meaningfully from a CPT init choice).
TASK_LISTS[retention]="arc_challenge,arc_easy,hellaswag,winogrande,piqa,mmlu,global_mmlu,xnli,xcopa"
# Group 2: ILSP Greek suite (dedicated Greek signal beyond multilingual coverage).
# Excludes mgsm_greek + ifeval_greek (post-training-only Greek variants per EVAL_RECIPE.md
# "Greek post-training-only tasks" note).
TASK_LISTS[greek]="arc_greek,hellaswag_greek,winogrande_greek,mmlu_greek,mmlu_pro_greek,truthfulqa_greek,medical_mcqa_greek,belebele_ell_Grek"
# Group 3: safety. Inspect AI suite; lower priority for the bakeoff selection.
TASK_LISTS[safety]="harmbench,polyglotoxicityprompts,attaq_greek"

case "$TASK_GROUP" in
    full)            TASK_LIST="${TASK_LISTS[retention]},${TASK_LISTS[greek]}" ;;
    retention_only)  TASK_LIST="${TASK_LISTS[retention]}" ;;
    greek_only)      TASK_LIST="${TASK_LISTS[greek]}" ;;
    safety_only)     TASK_LIST="${TASK_LISTS[safety]}" ;;
    *)               echo "ERROR: TASK_GROUP must be full|retention_only|greek_only|safety_only" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=== run_eval.sbatch ==="
date -u
echo "host: $(hostname)"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TASK_GROUP: $TASK_GROUP"
echo "TASK_LIST: $TASK_LIST"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "DTYPE: $DTYPE"
echo

# Load the uenv pytorch image (Apertus team's preferred path on Clariden)
# This is the pattern from cscs-clariden-project-understanding/README.md
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.6.0:v1}"
echo "uenv image: $UENV_IMAGE"

# Capture harness commit for the run_metadata
HARNESS_COMMIT=$(cd "$LM_EVAL_HARNESS_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
echo "lm-evaluation-harness commit: $HARNESS_COMMIT"

# Record metadata before the run
cat > run_metadata.json <<META
{
  "model_path": "$MODEL_PATH",
  "task_group": "$TASK_GROUP",
  "task_list": "$TASK_LIST",
  "batch_size": "$BATCH_SIZE",
  "dtype": "$DTYPE",
  "lm_eval_harness_dir": "$LM_EVAL_HARNESS_DIR",
  "lm_eval_harness_commit": "$HARNESS_COMMIT",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "host": "$(hostname)",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

# Run lm-evaluation-harness inside the uenv pytorch image with --view=default
# (which exposes the modern Python + Torch stack per the project brief).
uenv run "$UENV_IMAGE" --view=default -- bash -c "
set -euo pipefail
cd $LM_EVAL_HARNESS_DIR
pip install -e . --quiet 2>&1 | tail -5 || true
which lm_eval || pip install lm-eval --quiet

lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=$DTYPE,trust_remote_code=True \
    --tasks $TASK_LIST \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_DIR/results.json \
    --log_samples \
    $LIMIT_ARG
"

# Append end time + parse results into a flat metric table
python3 -c "
import json
from pathlib import Path
md = json.loads(Path('run_metadata.json').read_text())
md['end_time'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
Path('run_metadata.json').write_text(json.dumps(md, indent=2))
print('updated run_metadata.json with end_time')
"

echo
echo "=== eval complete ==="
date -u
ls -la "$OUTPUT_DIR" | head -20
echo
echo "Next: compute_bootstrap_cis.py $OUTPUT_DIR/samples_*.jsonl"
