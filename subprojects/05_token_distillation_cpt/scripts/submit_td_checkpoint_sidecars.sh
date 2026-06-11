#!/usr/bin/env bash
# Submit eval sidecars for one TD-17k CPT checkpoint.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt}"
HF_TOKENIZER_DIR="${HF_TOKENIZER_DIR:-}"
HELDOUT_JSONL="${HELDOUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl}"
LOCAL_REPO_ROOT="${LOCAL_REPO_ROOT:-/home/foivos/Projects/glossapi-tokenizer-extension}"
DEFAULT_STARCODER_CODE_HELDOUT_JSONL="${DEFAULT_STARCODER_CODE_HELDOUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2/val_forget_code_starcoder_200_for_bpb.jsonl}"
DEFAULT_LEGACY_CODE_HELDOUT_JSONL="${DEFAULT_LEGACY_CODE_HELDOUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_code_heldout_200_20260528.jsonl}"
DEFAULT_CODE_HELDOUT_JSONL="${DEFAULT_CODE_HELDOUT_JSONL:-}"
DEFAULT_MATH_HELDOUT_JSONL="${DEFAULT_MATH_HELDOUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_math_heldout_200_20260528.jsonl}"
CHECKSUM_SCRIPT="${CHECKSUM_SCRIPT:-$REPO_ROOT/subprojects/05_token_distillation_cpt/scripts/write_checkpoint_checksum_manifest.py}"

ITER=""
TOKEN_MARK=""
TRAIN_RUN_DIR=""
RUN_TAG=""
CHECKPOINT_LABEL=""
EVAL_ARM="${EVAL_ARM:-td}"

SUBMIT_NATIVE="${SUBMIT_NATIVE:-1}"
SUBMIT_GREEK_NLP="${SUBMIT_GREEK_NLP:-1}"
SUBMIT_BPB="${SUBMIT_BPB:-1}"
SUBMIT_RETENTION="${SUBMIT_RETENTION:-1}"
SUBMIT_CHECKSUM="${SUBMIT_CHECKSUM:-1}"
NATIVE_BENCHMARKS="${NATIVE_BENCHMARKS:-greekmmlu,ilsp_medical_mcqa,ilsp_mcqa_asep,plutus_qa}"
NATIVE_SAMPLE_SIZE="${NATIVE_SAMPLE_SIZE:-0}"
GREEK_NLP_SAMPLE_SIZE="${GREEK_NLP_SAMPLE_SIZE:-100}"
RETENTION_TASK_GROUP="${RETENTION_TASK_GROUP:-retention_only}"

CODE_HELDOUT_JSONL="${CODE_HELDOUT_JSONL:-}"
MATH_HELDOUT_JSONL="${MATH_HELDOUT_JSONL:-}"
if [ -z "$DEFAULT_CODE_HELDOUT_JSONL" ]; then
  if [ -s "$DEFAULT_STARCODER_CODE_HELDOUT_JSONL" ]; then
    DEFAULT_CODE_HELDOUT_JSONL="$DEFAULT_STARCODER_CODE_HELDOUT_JSONL"
  else
    DEFAULT_CODE_HELDOUT_JSONL="$DEFAULT_LEGACY_CODE_HELDOUT_JSONL"
  fi
fi
if [ -z "$CODE_HELDOUT_JSONL" ] && [ -f "$DEFAULT_CODE_HELDOUT_JSONL" ]; then
  CODE_HELDOUT_JSONL="$DEFAULT_CODE_HELDOUT_JSONL"
fi
if [ -z "$MATH_HELDOUT_JSONL" ] && [ -f "$DEFAULT_MATH_HELDOUT_JSONL" ]; then
  MATH_HELDOUT_JSONL="$DEFAULT_MATH_HELDOUT_JSONL"
fi

usage() {
  cat <<'USAGE'
Usage:
  submit_td_checkpoint_sidecars.sh \
    --train-run-dir /capstor/.../05_td17k_l11_path_a_goldfish_5b_... \
    --run-tag 05_td17k_l11_path_a_goldfish_5b_... \
    --arm td \
    --iteration 119 \
    --tokens 499122176 \
    --checkpoint-label TD-17k-L11-0.5B

Submits Megatron->HF conversion, native Greek MCQ, greek-nlp/benchmark sample,
heldout Greek BPB, multilingual retention, code/math BPB when heldouts exist,
and a CPU-only checksum manifest job. Also writes the local Codex
adversarial-review command to run after sidecars finish.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --train-run-dir) TRAIN_RUN_DIR="${2:?}"; shift 2 ;;
    --run-tag) RUN_TAG="${2:?}"; shift 2 ;;
    --arm) EVAL_ARM="${2:?}"; shift 2 ;;
    --iteration) ITER="${2:?}"; shift 2 ;;
    --tokens) TOKEN_MARK="${2:?}"; shift 2 ;;
    --checkpoint-label) CHECKPOINT_LABEL="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$TRAIN_RUN_DIR" ] || [ -z "$RUN_TAG" ] || [ -z "$ITER" ] || [ -z "$TOKEN_MARK" ]; then
  echo "ERROR: train run dir, run tag, iteration, and tokens are required." >&2
  usage >&2
  exit 2
fi
if [ -z "$CHECKPOINT_LABEL" ]; then
  case "$EVAL_ARM" in
    vanilla) CHECKPOINT_LABEL="Vanilla-${TOKEN_MARK}" ;;
    td|td_layer11) CHECKPOINT_LABEL="TD-17k-L11-${TOKEN_MARK}" ;;
    extension|retok|centroid) CHECKPOINT_LABEL="${EVAL_ARM}-${TOKEN_MARK}" ;;
    *) echo "ERROR: unsupported eval arm: $EVAL_ARM" >&2; exit 2 ;;
  esac
fi
case "$ITER" in
  *[!0-9]*|"") echo "ERROR: iteration must be an integer: $ITER" >&2; exit 2 ;;
esac
case "$EVAL_ARM" in
  vanilla|retok|centroid|extension|td|td_layer11) ;;
  *) echo "ERROR: --arm must be vanilla|retok|centroid|extension|td|td_layer11, got: $EVAL_ARM" >&2; exit 2 ;;
esac
CONVERT_ARM="$EVAL_ARM"
if [ "$EVAL_ARM" = "td" ]; then
  CONVERT_ARM="td_layer11"
fi
if [ -z "$HF_TOKENIZER_DIR" ]; then
  case "$EVAL_ARM" in
    vanilla)
      HF_TOKENIZER_DIR="/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509"
      ;;
    td|td_layer11)
      HF_TOKENIZER_DIR="/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/hf_roundtrip"
      ;;
    extension|retok|centroid)
      HF_TOKENIZER_DIR="/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/hf_roundtrip"
      ;;
  esac
fi
if [ ! -f "$HF_TOKENIZER_DIR/config.json" ] || [ ! -f "$HF_TOKENIZER_DIR/tokenizer.json" ]; then
  echo "ERROR: HF_TOKENIZER_DIR must contain config.json and tokenizer.json: $HF_TOKENIZER_DIR" >&2
  exit 2
fi

ITER_PAD="$(printf "%07d" "$ITER")"
MEGATRON_CKPT_ROOT="${MEGATRON_CKPT_ROOT:-$TRAIN_RUN_DIR/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/eval_${RUN_TAG}}"
ITER_ROOT="$EVAL_ROOT/iter_${ITER_PAD}"
HF_OUT_DIR="${HF_OUT_DIR:-${ITER_ROOT}_hf}"
NATIVE_OUT_DIR="$ITER_ROOT/native_mcq"
GREEK_NLP_OUT_DIR="$ITER_ROOT/greek_nlp_s${GREEK_NLP_SAMPLE_SIZE}"
RETENTION_OUT_DIR="$ITER_ROOT/retention"
BPB_JSON="$ITER_ROOT/heldout_greek_bpb.json"
CHECKSUM_OUT="$ITER_ROOT/checksums/${CHECKPOINT_LABEL}_iter_${ITER_PAD}_checksum_manifest.json"
JOBS_TSV="$ITER_ROOT/sidecar_jobs.tsv"
CODEX_REVIEW_CMD="$ITER_ROOT/run_adversarial_review_from_home.sh"

test -d "$MEGATRON_CKPT_ROOT/iter_$ITER_PAD"
test -f "$HELDOUT_JSONL"
test -f "$EVAL_DIR/convert_bakeoff_checkpoint_to_hf.sbatch"
test -f "$EVAL_DIR/run_native_greek_mcq_eval.sbatch"
test -f "$EVAL_DIR/run_greek_nlp_benchmark_hf.sbatch"
test -f "$EVAL_DIR/run_tokenizer_fair_metrics.sbatch"
test -f "$EVAL_DIR/run_eval.sbatch"
if [ "$SUBMIT_CHECKSUM" = "1" ]; then
  test -f "$CHECKSUM_SCRIPT"
fi
mkdir -p "$ITER_ROOT" "$NATIVE_OUT_DIR" "$GREEK_NLP_OUT_DIR" "$RETENTION_OUT_DIR"

printf "kind\tjob_id\tdependency\toutput\n" > "$JOBS_TSV"

echo "=== submit_td_checkpoint_sidecars.sh ==="
echo "RUN_TAG:             $RUN_TAG"
echo "EVAL_ARM:            $EVAL_ARM"
echo "CHECKPOINT_LABEL:    $CHECKPOINT_LABEL"
echo "ITER/TOKENS:         $ITER / $TOKEN_MARK"
echo "MEGATRON_CKPT_ROOT:  $MEGATRON_CKPT_ROOT"
echo "EVAL_ROOT:           $EVAL_ROOT"
echo "HF_OUT_DIR:          $HF_OUT_DIR"
echo "HF_TOKENIZER_DIR:    $HF_TOKENIZER_DIR"
echo "NATIVE_BENCHMARKS:   $NATIVE_BENCHMARKS"
echo "SUBMIT_CHECKSUM:     $SUBMIT_CHECKSUM"
echo "CHECKSUM_OUT:        $CHECKSUM_OUT"
echo "JOBS_TSV:            $JOBS_TSV"
echo

convert_job="$(sbatch --parsable \
  --account=a0140 \
  --partition=normal \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=72 \
  --mem=400G \
  --time=01:00:00 \
  --job-name="05tohf_i${ITER}" \
  --output="$RUN_ROOT/%x-%j.out" \
  --error="$RUN_ROOT/%x-%j.err" \
  --export=ALL,RUN_TAG="$RUN_TAG",ARM="$CONVERT_ARM",ITER="$ITER",MEGATRON_CKPT_ROOT="$MEGATRON_CKPT_ROOT",HF_TOKENIZER_DIR="$HF_TOKENIZER_DIR",HF_OUT_DIR="$HF_OUT_DIR",OUT_ROOT="$EVAL_ROOT",SCRIPT_DIR_OVERRIDE="$EVAL_DIR",OVERWRITE=1 \
  "$EVAL_DIR/convert_bakeoff_checkpoint_to_hf.sbatch")"
printf "convert\t%s\tnone\t%s\n" "$convert_job" "$HF_OUT_DIR" >> "$JOBS_TSV"

job_ids=("$convert_job")

if [ "$SUBMIT_NATIVE" = "1" ]; then
  # Slurm splits --export on top-level commas; passing BENCHMARKS="a,b,c"
  # inside --export silently truncates the benchmark list to "a". Set it as
  # an env var on the sbatch line so --export=ALL carries it through intact.
  native_job="$(BENCHMARKS="$NATIVE_BENCHMARKS" sbatch --parsable \
    --dependency="afterok:$convert_job" \
    --account=a0140 \
    --partition=normal \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --gres=gpu:1 \
    --cpus-per-task=18 \
    --mem=220G \
    --time=10:00:00 \
    --job-name="05native_i${ITER}" \
    --output="$RUN_ROOT/%x-%j.out" \
    --error="$RUN_ROOT/%x-%j.err" \
    --export=ALL,MODEL_SPEC="$CHECKPOINT_LABEL=$HF_OUT_DIR",OUTPUT_DIR="$NATIVE_OUT_DIR",SAMPLE_SIZE="$NATIVE_SAMPLE_SIZE",SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
    "$EVAL_DIR/run_native_greek_mcq_eval.sbatch")"
  printf "native_mcq\t%s\t%s\t%s\n" "$native_job" "$convert_job" "$NATIVE_OUT_DIR" >> "$JOBS_TSV"
  job_ids+=("$native_job")
fi

if [ "$SUBMIT_GREEK_NLP" = "1" ]; then
  greek_nlp_job="$(sbatch --parsable \
    --dependency="afterok:$convert_job" \
    --account=a0140 \
    --partition=normal \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --gres=gpu:1 \
    --cpus-per-task=72 \
    --mem=240G \
    --time=08:00:00 \
    --job-name="05gnlp_i${ITER}" \
    --output="$RUN_ROOT/%x-%j.out" \
    --error="$RUN_ROOT/%x-%j.err" \
    --export=ALL,MODEL_SPEC="$CHECKPOINT_LABEL=$HF_OUT_DIR",OUTPUT_DIR="$GREEK_NLP_OUT_DIR",TASK=all,SAMPLE_SIZE="$GREEK_NLP_SAMPLE_SIZE",SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
    "$EVAL_DIR/run_greek_nlp_benchmark_hf.sbatch")"
  printf "greek_nlp\t%s\t%s\t%s\n" "$greek_nlp_job" "$convert_job" "$GREEK_NLP_OUT_DIR" >> "$JOBS_TSV"
  job_ids+=("$greek_nlp_job")
fi

if [ "$SUBMIT_BPB" = "1" ]; then
  bpb_job="$(sbatch --parsable \
    --dependency="afterok:$convert_job" \
    --account=a0140 \
    --partition=normal \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --gres=gpu:1 \
    --cpus-per-task=18 \
    --mem=220G \
    --time=02:00:00 \
    --job-name="05bpb_i${ITER}" \
    --output="$RUN_ROOT/%x-%j.out" \
    --error="$RUN_ROOT/%x-%j.err" \
    --export=ALL,MODEL_PATH="$HF_OUT_DIR",EVAL_JSONL="$HELDOUT_JSONL",OUTPUT_JSON="$BPB_JSON",SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
    "$EVAL_DIR/run_tokenizer_fair_metrics.sbatch")"
  printf "heldout_greek_bpb\t%s\t%s\t%s\n" "$bpb_job" "$convert_job" "$BPB_JSON" >> "$JOBS_TSV"
  job_ids+=("$bpb_job")
fi

if [ "$SUBMIT_RETENTION" = "1" ]; then
  retention_job="$(sbatch --parsable \
    --dependency="afterok:$convert_job" \
    --account=a0140 \
    --partition=normal \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --gres=gpu:1 \
    --cpus-per-task=72 \
    --mem=220G \
    --time=08:00:00 \
    --job-name="05ret_i${ITER}" \
    --output="$RUN_ROOT/%x-%j.out" \
    --error="$RUN_ROOT/%x-%j.err" \
    --export=ALL,MODEL_PATH="$HF_OUT_DIR",OUTPUT_DIR="$RETENTION_OUT_DIR",TASK_GROUP="$RETENTION_TASK_GROUP",SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
    "$EVAL_DIR/run_eval.sbatch")"
  printf "retention\t%s\t%s\t%s\n" "$retention_job" "$convert_job" "$RETENTION_OUT_DIR" >> "$JOBS_TSV"
  job_ids+=("$retention_job")
fi

for optional in code math; do
  var_name="$(printf "%s_HELDOUT_JSONL" "$optional" | tr '[:lower:]' '[:upper:]')"
  heldout_path="${!var_name:-}"
  if [ -n "$heldout_path" ]; then
    test -f "$heldout_path"
    out_json="$ITER_ROOT/heldout_${optional}_bpb.json"
    opt_job="$(sbatch --parsable \
      --dependency="afterok:$convert_job" \
      --account=a0140 \
      --partition=normal \
      --nodes=1 \
      --ntasks-per-node=1 \
      --gpus-per-node=1 \
      --gres=gpu:1 \
      --cpus-per-task=18 \
      --mem=220G \
      --time=02:00:00 \
      --job-name="05${optional}bpb_i${ITER}" \
      --output="$RUN_ROOT/%x-%j.out" \
      --error="$RUN_ROOT/%x-%j.err" \
      --export=ALL,MODEL_PATH="$HF_OUT_DIR",EVAL_JSONL="$heldout_path",OUTPUT_JSON="$out_json",SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
      "$EVAL_DIR/run_tokenizer_fair_metrics.sbatch")"
    printf "%s_bpb\t%s\t%s\t%s\n" "$optional" "$opt_job" "$convert_job" "$out_json" >> "$JOBS_TSV"
    job_ids+=("$opt_job")
  fi
done

if [ "$SUBMIT_CHECKSUM" = "1" ]; then
  checksum_job="$(sbatch --parsable \
    --dependency="afterok:$convert_job" \
    --account=a0140 \
    --partition=xfer \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=8G \
    --time=04:00:00 \
    --job-name="05cksum_i${ITER}" \
    --output="$RUN_ROOT/%x-%j.out" \
    --error="$RUN_ROOT/%x-%j.err" \
    --wrap="python3 '$CHECKSUM_SCRIPT' --checkpoint-label '$CHECKPOINT_LABEL' --iteration '$ITER' --token-mark '$TOKEN_MARK' --megatron-checkpoint-dir '$MEGATRON_CKPT_ROOT/iter_$ITER_PAD' --hf-checkpoint-dir '$HF_OUT_DIR' --output-json '$CHECKSUM_OUT'")"
  printf "checksum\t%s\t%s\t%s\n" "$checksum_job" "$convert_job" "$CHECKSUM_OUT" >> "$JOBS_TSV"
  job_ids+=("$checksum_job")
fi

cat > "$CODEX_REVIEW_CMD" <<CMD
#!/usr/bin/env bash
set -euo pipefail
cd "$LOCAL_REPO_ROOT"
./subprojects/05_token_distillation_cpt/scripts/run_td_checkpoint_adversarial_review.sh \\
  --checkpoint-label "$CHECKPOINT_LABEL" \\
  --iteration "$ITER" \\
  --tokens "$TOKEN_MARK" \\
  --train-run-dir "$TRAIN_RUN_DIR" \\
  --eval-root "$EVAL_ROOT" \\
  --hf-checkpoint-dir "$HF_OUT_DIR"
CMD
chmod +x "$CODEX_REVIEW_CMD"

echo
echo "Submitted jobs: ${job_ids[*]}"
echo "Jobs TSV: $JOBS_TSV"
echo "After eval jobs finish, run from home:"
echo "  $CODEX_REVIEW_CMD"
echo "Monitor:"
echo "  squeue -j $(IFS=,; echo "${job_ids[*]}") -o '%.18i %.32j %.2t %.10M %.10l %D %R'"
