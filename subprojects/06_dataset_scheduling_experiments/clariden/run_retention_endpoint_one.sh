#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?set exact HF export}"
: "${MODEL_LABEL:?set model label}"
: "${OUTPUT_DIR:?set unique output directory}"
: "${LM_EVAL_ROOT:?set frozen repaired lm-eval target install}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing to replace $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
tasks=arc_challenge,arc_easy,hellaswag,winogrande,piqa,mmlu,global_mmlu,xnli,xcopa
cat >"$OUTPUT_DIR/run_metadata.json" <<META
{
  "model_label": "$MODEL_LABEL",
  "model_path": "$MODEL_PATH",
  "tasks": "$tasks",
  "batch_size": "auto",
  "dtype": "bfloat16",
  "lm_eval_root": "$LM_EVAL_ROOT"
}
META
export PYTHONPATH="$LM_EVAL_ROOT"
export LD_LIBRARY_PATH="$LM_EVAL_ROOT/scipy.libs:$LM_EVAL_ROOT/numpy.libs:$LM_EVAL_ROOT/scikit_learn.libs:${LD_LIBRARY_PATH:-}"
if [[ -n "${RETENTION_SHARED_CACHE_ROOT:-}" ]]; then
  [[ -d "$RETENTION_SHARED_CACHE_ROOT/hf_home" ]] || { echo "missing shared HF cache" >&2; exit 2; }
  [[ -d "$RETENTION_SHARED_CACHE_ROOT/hf_datasets" ]] || { echo "missing shared datasets cache" >&2; exit 2; }
  export HF_HOME="$RETENTION_SHARED_CACHE_ROOT/hf_home"
  export HF_DATASETS_CACHE="$RETENTION_SHARED_CACHE_ROOT/hf_datasets"
  export XDG_CACHE_HOME="$RETENTION_SHARED_CACHE_ROOT/xdg"
  export TMPDIR="$RETENTION_SHARED_CACHE_ROOT/tmp"
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
else
  mkdir -p "$OUTPUT_DIR/cache"/{hf_home,hf_datasets,xdg,tmp}
  export HF_HOME="$OUTPUT_DIR/cache/hf_home"
  export HF_DATASETS_CACHE="$OUTPUT_DIR/cache/hf_datasets"
  export XDG_CACHE_HOME="$OUTPUT_DIR/cache/xdg"
  export TMPDIR="$OUTPUT_DIR/cache/tmp"
fi
python3 - <<'PY'
import lm_eval
from pytablewriter import LatexTableWriter
print(lm_eval.__file__)
PY
python3 -m lm_eval \
  --model hf --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True" \
  --tasks "$tasks" --batch_size auto --output_path "$OUTPUT_DIR/results.json" --log_samples
python3 - "$OUTPUT_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); results=sorted(root.glob("**/results*.json"))
if not results: raise SystemExit("lm-eval produced no results JSON")
payload={
 "schema_version":"apertus_mini_retention_endpoint_v1","status":"completed",
 "results":[{"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in results],
}
(root/"receipt.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
