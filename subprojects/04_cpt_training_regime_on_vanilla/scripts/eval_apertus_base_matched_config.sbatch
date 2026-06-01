#!/usr/bin/env bash
# Submit the three matched-config Apertus-Base eval sidecars.
#
# Three sidecars are submitted in parallel (no dependency between them
# because there is no Megatron->HF conversion step — Apertus-Base is
# already HF):
#   1. native_mcq  -> run_native_greek_mcq_eval.sbatch
#   2. retention   -> run_eval.sbatch (TASK_GROUP=retention_only)
#   3. bpb         -> run_tokenizer_fair_metrics.sbatch (Greek heldout)
#
# All three point at the matched-config staging dir created by
# build_apertus_base_matched_config.sh:
#   /iopsstor/.../models/apertus-8b-2509-matched-rope500k-seq4096/
#
# The matched dir overrides only rope_theta=500000,
# max_position_embeddings=4096, rope_scaling=null. Weights and tokenizer
# are symlinked to the original Apertus-Base snapshot.
#
# IMPORTANT — Slurm --export quoting (post-fix from
# submit_checkpoint_sidecars.sh lines 142-145, iter-238 critical bug):
#   Slurm splits --export=ALL,KEY=VAL,KEY=VAL,... on every top-level
#   comma. A KEY whose VAL itself contains commas (e.g.
#   BENCHMARKS="greekmmlu,ilsp_medical_mcqa,ilsp_mcqa_asep,plutus_qa") is
#   silently truncated to the first segment. The fix is to set the
#   comma-bearing variables as env-var prefixes on the sbatch command
#   line (before `sbatch`), and rely on --export=ALL to carry them through
#   intact. The native job below does exactly that.
#
# Naming hint: this file is named .sbatch to match the user's spec, but
# it is NOT a #SBATCH job itself — it is a fan-out submitter that issues
# three sbatch calls. Run as `bash eval_apertus_base_matched_config.sbatch`
# (or directly: `./eval_apertus_base_matched_config.sbatch`). Do not feed
# it to sbatch.
#
# Compute justification (per [[feedback_compute_sweet_spot_justify]]).
# Per-sidecar resource shape mirrors submit_checkpoint_sidecars.sh, which
# is the audited shape on Clariden GH200 normal partition:
#
# 1. Parallelism strategy.
#    Single-GPU per sidecar, no DP/TP. lm-eval-harness and the native MCQ
#    runner both batch internally on one device. Three sidecars run as
#    independent Slurm jobs that the scheduler may parallelize across
#    nodes; they do not depend on each other.
#
# 2. CPU+GPU saturation.
#    Apertus-8B bf16 weights ~17 GB + KV cache for the candidate-batch =
#    16 setup well under one GH200's 80 GB. CPU saturation: --cpus-per-task
#    matches the upstream sidecar templates (retention 72, native 18,
#    bpb 18 — bpb and native are tokenizer/forward-only and don't profit
#    from extra cores; retention has heavier dataloading across many
#    tasks).
#
# 3. Mem ceiling vs observed peak.
#    --mem=220G covers worst-case retention (lm-eval cached HF datasets
#    plus model + activations + KV). Observed peak from the 04 sidecar
#    runs at iter 238 was well below this; 220G is the safe headroom
#    matching the audited template.
#
# 4. Known efficiency gaps.
#    Slurm at Clariden `normal` partition allocates a full 4-GPU GH200 node
#    even when --gpus-per-node=1 is requested (documented in iter-238
#    critique Major-6). This is a partition policy we cannot change from a
#    submit script; node-billing inefficiency is the cost of using
#    `normal`. The dependency-free three-way fan-out means total wall-clock
#    is max(t_native, t_retention, t_bpb), not the sum.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt}"

# Matched-config staging dir produced by build_apertus_base_matched_config.sh.
MATCHED_MODEL_DIR="${MATCHED_MODEL_DIR:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509-matched-rope500k-seq4096}"

# Greek heldout JSONL — SAME set the 04 CPT sidecars use, so the BPB
# delta is directly comparable.
HELDOUT_JSONL="${HELDOUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl}"

# Output root for this matched-config baseline.
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/eval_apertus_base_matched_rope500k_seq4096}"
NATIVE_OUT_DIR="$EVAL_ROOT/native_mcq"
RETENTION_OUT_DIR="$EVAL_ROOT/retention"
BPB_JSON="$EVAL_ROOT/heldout_greek_bpb.json"
JOBS_TSV="$EVAL_ROOT/sidecar_jobs.tsv"

# Native MCQ benchmark list — full 3-task headline plus Plutus diagnostic.
# MUST be passed as an env-var prefix on the sbatch line, NOT inside
# --export=ALL,...,BENCHMARKS="..." (see Slurm-quoting note above).
NATIVE_BENCHMARKS="${NATIVE_BENCHMARKS:-greekmmlu,ilsp_medical_mcqa,ilsp_mcqa_asep,plutus_qa}"
NATIVE_SAMPLE_SIZE="${NATIVE_SAMPLE_SIZE:-0}"

# Retention task group — matches what 04 CPT sidecars submit so the
# retention delta is task-comparable.
RETENTION_TASK_GROUP="${RETENTION_TASK_GROUP:-retention_only}"

# Model label embedded in MODEL_SPEC for the native MCQ runner's output.
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-Apertus-Base-matched-rope500k-seq4096}"

# Preflight checks.
test -d "$MATCHED_MODEL_DIR" || {
  echo "ERROR: matched model dir missing: $MATCHED_MODEL_DIR" >&2
  echo "       run scripts/build_apertus_base_matched_config.sh first." >&2
  exit 2
}
test -f "$MATCHED_MODEL_DIR/config.json" || {
  echo "ERROR: missing $MATCHED_MODEL_DIR/config.json" >&2
  exit 2
}
test -f "$MATCHED_MODEL_DIR/MANIFEST.json" || {
  echo "ERROR: missing $MATCHED_MODEL_DIR/MANIFEST.json (rerun build script)" >&2
  exit 2
}
test -f "$HELDOUT_JSONL" || {
  echo "ERROR: heldout JSONL not found: $HELDOUT_JSONL" >&2
  exit 2
}
for f in run_native_greek_mcq_eval.sbatch run_eval.sbatch run_tokenizer_fair_metrics.sbatch; do
  test -f "$EVAL_DIR/$f" || { echo "ERROR: missing $EVAL_DIR/$f" >&2; exit 2; }
done

mkdir -p "$EVAL_ROOT" "$NATIVE_OUT_DIR" "$RETENTION_OUT_DIR"

# Confirm the matched dir's config actually carries the three overrides.
python3 - "$MATCHED_MODEL_DIR/config.json" <<'PY' || exit 2
import json, sys
cfg = json.loads(open(sys.argv[1]).read())
checks = [
    ("rope_theta", 500000),
    ("max_position_embeddings", 4096),
]
for k, v in checks:
    assert cfg.get(k) == v, f"matched config {k}={cfg.get(k)!r}, expected {v!r}"
assert cfg.get("rope_scaling") is None, \
    f"matched config rope_scaling={cfg.get('rope_scaling')!r}, expected null"
print("matched config geometry verified: rope_theta=500000, "
      "max_position_embeddings=4096, rope_scaling=null")
PY

echo
echo "=== eval_apertus_base_matched_config.sbatch ==="
date -u
echo "MATCHED_MODEL_DIR:   $MATCHED_MODEL_DIR"
echo "EVAL_ROOT:           $EVAL_ROOT"
echo "NATIVE_BENCHMARKS:   $NATIVE_BENCHMARKS"
echo "RETENTION_TASK_GROUP:$RETENTION_TASK_GROUP"
echo "HELDOUT_JSONL:       $HELDOUT_JSONL"
echo "CHECKPOINT_LABEL:    $CHECKPOINT_LABEL"
echo "JOBS_TSV:            $JOBS_TSV"
echo

printf "kind\tjob_id\toutput\n" > "$JOBS_TSV"

# (1) native Greek MCQ — full 4-benchmark list, env-var-prefix pattern.
native_job="$(BENCHMARKS="$NATIVE_BENCHMARKS" sbatch --parsable \
  --account=a0140 \
  --partition=normal \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=18 \
  --mem=220G \
  --time=10:00:00 \
  --job-name="04base_native_matched" \
  --output="$RUN_ROOT/%x-%j.out" \
  --error="$RUN_ROOT/%x-%j.err" \
  --export=ALL,MODEL_SPEC="$CHECKPOINT_LABEL=$MATCHED_MODEL_DIR",OUTPUT_DIR="$NATIVE_OUT_DIR",SAMPLE_SIZE="$NATIVE_SAMPLE_SIZE",SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
  "$EVAL_DIR/run_native_greek_mcq_eval.sbatch")"
printf "native_mcq\t%s\t%s\n" "$native_job" "$NATIVE_OUT_DIR" >> "$JOBS_TSV"

# (2) retention (lm-eval-harness retention task group).
retention_job="$(sbatch --parsable \
  --account=a0140 \
  --partition=normal \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=72 \
  --mem=220G \
  --time=08:00:00 \
  --job-name="04base_ret_matched" \
  --output="$RUN_ROOT/%x-%j.out" \
  --error="$RUN_ROOT/%x-%j.err" \
  --export=ALL,MODEL_PATH="$MATCHED_MODEL_DIR",OUTPUT_DIR="$RETENTION_OUT_DIR",TASK_GROUP="$RETENTION_TASK_GROUP",SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
  "$EVAL_DIR/run_eval.sbatch")"
printf "retention\t%s\t%s\n" "$retention_job" "$RETENTION_OUT_DIR" >> "$JOBS_TSV"

# (3) Greek BPB on the 500-doc heldout (max_context=4096 by default,
#     matching the 04 CPT sidecars exactly — so prefix-truncation rates
#     are also matched and BPB deltas are directly comparable).
bpb_job="$(sbatch --parsable \
  --account=a0140 \
  --partition=normal \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=18 \
  --mem=220G \
  --time=02:00:00 \
  --job-name="04base_bpb_matched" \
  --output="$RUN_ROOT/%x-%j.out" \
  --error="$RUN_ROOT/%x-%j.err" \
  --export=ALL,MODEL_PATH="$MATCHED_MODEL_DIR",EVAL_JSONL="$HELDOUT_JSONL",OUTPUT_JSON="$BPB_JSON",SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
  "$EVAL_DIR/run_tokenizer_fair_metrics.sbatch")"
printf "heldout_greek_bpb\t%s\t%s\n" "$bpb_job" "$BPB_JSON" >> "$JOBS_TSV"

echo
echo "Submitted matched-config Apertus-Base eval sidecars:"
echo "  native_mcq job: $native_job  -> $NATIVE_OUT_DIR"
echo "  retention  job: $retention_job  -> $RETENTION_OUT_DIR"
echo "  bpb        job: $bpb_job  -> $BPB_JSON"
echo "Jobs TSV: $JOBS_TSV"
echo "Monitor:"
echo "  squeue -j $native_job,$retention_job,$bpb_job -o '%.18i %.32j %.2t %.10M %.10l %D %R'"
