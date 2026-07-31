#!/usr/bin/env bash
# Submit one receipt-bound Megatron->HF conversion and GreekMMLU evaluation.
set -euo pipefail

: "${TRAINING_ASSETS_RECEIPT:?set TRAINING_ASSETS_RECEIPT}"
: "${TRAIN_RUN_DIR:?set TRAIN_RUN_DIR}"
: "${RUN_TAG:?set RUN_TAG}"
: "${ITERATION:?set ITERATION}"
case "$ITERATION" in *[!0-9]*|0|"") echo "ERROR: ITERATION must be positive" >&2; exit 2 ;; esac

read -r REPO_ROOT MEGATRON_DIR HF_TEMPLATE CONVERTER NATIVE_SBATCH FINALIZER < <(
  python3 - "$TRAINING_ASSETS_RECEIPT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve(); a=json.load(open(p,encoding="utf-8"))
if a.get("schema_version") != "greek_cpt_training_assets_receipt_v1" or a.get("status") != "frozen": raise SystemExit("training assets are not frozen")
for row in a["hf_conversion_template"]["files"].values():
 q=Path(row["path"])
 if not q.is_file() or q.stat().st_size != row["bytes"] or hashlib.sha256(q.read_bytes()).hexdigest() != row["sha256"]: raise SystemExit(f"HF conversion template drift: {q}")
d=a["dependencies"]
for name in ("checkpoint_converter","checkpoint_converter_runtime","native_greekmmlu_sbatch","native_greekmmlu_runner","native_greekmmlu_registry","greekmmlu_checkpoint_finalizer"):
 row=d[name]; q=Path(row["path"])
 if not q.is_file() or q.stat().st_size != row["bytes"] or hashlib.sha256(q.read_bytes()).hexdigest() != row["sha256"]: raise SystemExit(f"frozen evaluation dependency drift: {name}")
print(a["repository"]["root"],a["megatron"]["root"],a["hf_conversion_template"]["root"],d["checkpoint_converter"]["path"],d["native_greekmmlu_sbatch"]["path"],d["greekmmlu_checkpoint_finalizer"]["path"])
PY
)
[[ "$(git -C "$MEGATRON_DIR" rev-parse HEAD)" == "c92402e39ef3c8e69ea378a59e79059dc14541f4" ]]
[[ -z "$(git -C "$MEGATRON_DIR" status --porcelain --untracked-files=all)" ]]

padded=$(printf '%07d' "$ITERATION")
checkpoint_root="$TRAIN_RUN_DIR/checkpoints"
checkpoint="$checkpoint_root/iter_$padded"
marker="$checkpoint_root/latest_checkpointed_iteration.txt"
test -s "$checkpoint/.metadata"
test -s "$marker"
latest=$(tr -d '[:space:]' < "$marker")
[[ "$latest" =~ ^[0-9]+$ ]] && (( latest >= ITERATION )) || {
  echo "ERROR: checkpoint $ITERATION is not yet complete (latest=$latest)" >&2
  exit 3
}

tokens=$((ITERATION * 4194304))
eval_root="${EVAL_ROOT:-$TRAIN_RUN_DIR/checkpoint_evaluations}"
iter_root="$eval_root/iter_$padded"
hf_out="$iter_root/hf"
native_out="$iter_root/greekmmlu"
receipt="$iter_root/evaluation_receipt.json"
submission="$iter_root/submission.json"
if [[ -s "$submission" ]]; then
  echo "already submitted: $submission"
  exit 0
fi
[[ ! -e "$iter_root" ]] || { echo "ERROR: partial evaluation output exists: $iter_root" >&2; exit 3; }
mkdir -p "$iter_root" "$native_out"

eval_dir="$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval"
run_root="$TRAIN_RUN_DIR/checkpoint_evaluation_logs"
mkdir -p "$run_root"
convert_job=$(sbatch --parsable --account=a0140 --partition=normal --nodes=1 \
  --ntasks-per-node=1 --gpus-per-node=1 --gres=gpu:1 --cpus-per-task=72 \
  --mem=400G --time=01:00:00 --job-name="cpt25b_hf_i${ITERATION}" \
  --output="$run_root/%x-%j.out" --error="$run_root/%x-%j.err" \
  --export="ALL,RUN_TAG=$RUN_TAG,ARM=td_layer11,ITER=$ITERATION,MEGATRON_CKPT_ROOT=$checkpoint_root,HF_TOKENIZER_DIR=$HF_TEMPLATE,HF_OUT_DIR=$hf_out,OUT_ROOT=$iter_root,SCRIPT_DIR_OVERRIDE=$eval_dir,MEGATRON_DIR_OVERRIDE=$MEGATRON_DIR,OVERWRITE=0" \
  "$CONVERTER")

native_job=$(BENCHMARKS=greekmmlu sbatch --parsable --dependency="afterok:$convert_job" \
  --account=a0140 --partition=normal --nodes=1 --ntasks-per-node=1 \
  --gpus-per-node=1 --gres=gpu:1 --cpus-per-task=18 --mem=220G --time=01:00:00 \
  --job-name="cpt25b_gmmlu_i${ITERATION}" --output="$run_root/%x-%j.out" \
  --error="$run_root/%x-%j.err" \
  --export="ALL,MODEL_SPEC=Greek-CPT25B-L11-${tokens}tok=$hf_out,OUTPUT_DIR=$native_out,SAMPLE_SIZE=0,SCRIPT_DIR_OVERRIDE=$eval_dir" \
  "$NATIVE_SBATCH")

finalize_job=$(sbatch --parsable --dependency="afterok:$native_job" --account=a0140 \
  --partition=xfer --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --job-name="cpt25b_evalrec_i${ITERATION}" --output="$run_root/%x-%j.out" \
  --error="$run_root/%x-%j.err" \
  --wrap="python3 '$FINALIZER' --iteration '$ITERATION' --tokens '$tokens' --checkpoint-dir '$checkpoint' --hf-dir '$hf_out' --eval-dir '$native_out' --training-assets-receipt '$TRAINING_ASSETS_RECEIPT' --output '$receipt'")

python3 - "$submission" "$TRAINING_ASSETS_RECEIPT" "$ITERATION" "$tokens" "$convert_job" "$native_job" "$finalize_job" "$receipt" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
out,assets,iteration,tokens,convert,native,finalize,receipt=sys.argv[1:]
value={"schema_version":"greek_cpt_checkpoint_evaluation_submission_v1","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"training_assets_receipt":{"path":os.path.realpath(assets),"sha256":hashlib.sha256(open(assets,"rb").read()).hexdigest()},"iteration":int(iteration),"tokens":int(tokens),"jobs":{"convert":convert,"greekmmlu":native,"finalize":finalize},"expected_receipt":receipt}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
printf 'iteration=%s convert=%s greekmmlu=%s finalize=%s receipt=%s\n' \
  "$ITERATION" "$convert_job" "$native_job" "$finalize_job" "$receipt"
