#!/usr/bin/env bash
# Run a local Codex adversarial review for one TD checkpoint.
#
# This is intentionally not a Slurm job: Codex auth/tooling lives on home, while
# the checkpoint and eval artifacts live on Clariden. The spawned Codex agent is
# asked to inspect Clariden read-only through ssh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
REMOTE_HOST="${REMOTE_HOST:-clariden}"
CODEX_REVIEW_MODEL="${CODEX_REVIEW_MODEL:-gpt-5.5}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt}"

checkpoint_label=""
iteration=""
token_mark=""
train_run_dir=""
eval_root=""
hf_checkpoint_dir=""
output_dir=""

usage() {
  cat <<'USAGE'
Usage:
  run_td_checkpoint_adversarial_review.sh \
    --checkpoint-label TD-17k-L11-0.5B \
    --iteration 119 \
    --tokens 500000000 \
    --train-run-dir /capstor/.../train_run \
    --eval-root /capstor/.../eval_run

Optional:
  --hf-checkpoint-dir PATH   HF converted checkpoint dir if already known.
  --output-dir PATH          Local output dir for critique artifacts.

Environment:
  CODEX_REVIEW_MODEL         Default: gpt-5.5
  CODEX_REASONING_EFFORT     Default: xhigh
  REMOTE_HOST                Default: clariden
  DRY_RUN=1                  Write prompt/metadata and print command only.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint-label) checkpoint_label="${2:?}"; shift 2 ;;
    --iteration) iteration="${2:?}"; shift 2 ;;
    --tokens) token_mark="${2:?}"; shift 2 ;;
    --train-run-dir) train_run_dir="${2:?}"; shift 2 ;;
    --eval-root) eval_root="${2:?}"; shift 2 ;;
    --hf-checkpoint-dir) hf_checkpoint_dir="${2:?}"; shift 2 ;;
    --output-dir) output_dir="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$checkpoint_label" ] || [ -z "$iteration" ] || [ -z "$token_mark" ] || [ -z "$train_run_dir" ] || [ -z "$eval_root" ]; then
  echo "ERROR: checkpoint label, iteration, token mark, train run dir, and eval root are required." >&2
  usage >&2
  exit 2
fi

if [ -z "$hf_checkpoint_dir" ]; then
  iter_pad="$(printf "%07d" "$iteration")"
  hf_checkpoint_dir="$eval_root/iter_${iter_pad}_hf"
fi

safe_label="$(printf "%s" "$checkpoint_label" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_//; s/_$//')"
output_dir="${output_dir:-$SUBPROJECT_DIR/adversarial_reviews/$safe_label}"
prompt_file="$output_dir/prompt.md"
events_file="$output_dir/codex_events.jsonl"
critique_file="$output_dir/adversarial_critique.md"
metadata_file="$output_dir/review_metadata.env"

mkdir -p "$output_dir"

cat > "$metadata_file" <<META
CHECKPOINT_LABEL="$checkpoint_label"
ITERATION="$iteration"
TOKEN_MARK="$token_mark"
TRAIN_RUN_DIR="$train_run_dir"
EVAL_ROOT="$eval_root"
HF_CHECKPOINT_DIR="$hf_checkpoint_dir"
REMOTE_HOST="$REMOTE_HOST"
CODEX_REVIEW_MODEL="$CODEX_REVIEW_MODEL"
CODEX_REASONING_EFFORT="$CODEX_REASONING_EFFORT"
START_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
META

cat > "$prompt_file" <<PROMPT
You are the adversarial reviewer for the TD-17k Apertus CPT checkpoint
\`$checkpoint_label\` ($token_mark tokens, Megatron iteration $iteration).

Goal: find flaws, hidden assumptions, methodological mistakes, missing evidence,
data/eval leakage, bad comparisons, broken scripts, checkpoint/eval artifact
problems, compute hygiene issues, and ways the current TD interpretation could
be wrong. Be skeptical and concrete. Do not edit files, submit jobs, cancel jobs,
or modify remote state. Use read-only shell commands only.

Read these local files first:
- $SUBPROJECT_DIR/RUNBOOK.md
- $SUBPROJECT_DIR/LOG.md
- $SUBPROJECT_DIR/ARCHIVE.md
- $SUBPROJECT_DIR/03_training_experiments/configs/common_cpt.env
- $SUBPROJECT_DIR/03_training_experiments/configs/arm2_modern_greek.env
- $SUBPROJECT_DIR/scripts/
- $REPO_ROOT/subprojects/04_cpt_training_regime_on_vanilla/reports/5B_REPORT.md
- $REPO_ROOT/subprojects/04_cpt_training_regime_on_vanilla/reports/path_a_probe_results_20260531.md
- $REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/
- $REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training/

Inspect these Clariden artifacts through read-only SSH commands such as
\`ssh $REMOTE_HOST 'ls ...'\`, \`tail\`, \`grep\`, \`sacct\`, and \`find\`:
- Training run dir: $train_run_dir
- Megatron checkpoint iteration: $train_run_dir/checkpoints/iter_$(printf "%07d" "$iteration")
- Eval root: $eval_root
- HF checkpoint dir: $hf_checkpoint_dir

Return a Markdown critique with:

1. Verdict: whether this checkpoint's evidence is trustworthy enough to use.
2. Critical findings: issues that could invalidate the run, checkpoint, evals,
   or comparison.
3. Major findings: issues that materially weaken confidence but may be fixable.
4. Minor findings and hygiene notes.
5. Missing evidence: exact files/commands/results still needed.
6. Recommended next actions before reading or acting on this checkpoint.

Pay particular attention to whether the checkpoint is really using the
extended tokenizer/data path, the Path-A R17 TD init checkpoint, the 256-aligned
148,480-token vocabulary, the latest Goldfish/AdEMAMix training settings, and a
valid Vanilla-Path-A comparison. Flag any accidental dependence on vanilla-only
Task-1 assumptions.

Use file paths, job IDs, metric names, and exact artifact paths wherever
possible. If you cannot access a required artifact, say so directly.
PROMPT

echo "Adversarial review context:"
echo "  prompt:   $prompt_file"
echo "  events:   $events_file"
echo "  critique: $critique_file"
echo "  model:    $CODEX_REVIEW_MODEL"
echo "  effort:   $CODEX_REASONING_EFFORT"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1; not invoking codex exec."
  echo "Command:"
  printf 'codex exec -C %q -m %q -c %q -s danger-full-access --json -o %q - < %q > %q\n' \
    "$REPO_ROOT" \
    "$CODEX_REVIEW_MODEL" \
    "model_reasoning_effort=\"$CODEX_REASONING_EFFORT\"" \
    "$critique_file" \
    "$prompt_file" \
    "$events_file"
  exit 0
fi

command -v codex >/dev/null 2>&1 || {
  echo "ERROR: codex CLI is not on PATH. Run this sidecar from home." >&2
  exit 127
}

codex exec \
  -C "$REPO_ROOT" \
  -m "$CODEX_REVIEW_MODEL" \
  -c "model_reasoning_effort=\"$CODEX_REASONING_EFFORT\"" \
  -s danger-full-access \
  --json \
  -o "$critique_file" \
  - < "$prompt_file" > "$events_file"

cat >> "$metadata_file" <<META
END_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CRITIQUE_FILE="$critique_file"
EVENTS_FILE="$events_file"
PROMPT_FILE="$prompt_file"
META

echo "Adversarial critique written to: $critique_file"
