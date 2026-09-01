#!/usr/bin/env bash
#SBATCH --job-name=mini-docval-smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${MINI_DOCVAL_MANIFEST:?set frozen per-document validation manifest}"
: "${MINI_SMOKE_MODEL:?set D0 terminal HF model}"
: "${MINI_TOKENIZER:?set Mini extended tokenizer}"
: "${MINI_DOCVAL_ROOT:?set new per-document campaign root}"
read -r input < <(python3 - "$MINI_DOCVAL_MANIFEST" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); rows=[p for p in d["panels"] if p["name"]=="hplt"]
if len(rows)!=1: raise SystemExit("expected one HPLT panel")
print(rows[0]["raw_jsonl"]["path"])
PY
)
mkdir -p "$MINI_DOCVAL_ROOT/smoke"
CUDA_VISIBLE_DEVICES=0 uenv run pytorch/v2.9.1:v2 --view=default -- \
  python3 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/score_documents_hf.py" \
  --model "$MINI_SMOKE_MODEL" --tokenizer "$MINI_TOKENIZER" --input "$input" \
  --output "$MINI_DOCVAL_ROOT/smoke/hplt.documents.jsonl" \
  --receipt "$MINI_DOCVAL_ROOT/smoke/hplt.receipt.json" \
  --device cuda --dtype bfloat16 --trust-remote-code
python3 - "$MINI_DOCVAL_ROOT/smoke/hplt.receipt.json" <<'PY'
import json,math,sys
d=json.load(open(sys.argv[1])); rate=float(d["target_tokens_per_scoring_second"])
if d.get("status")!="completed" or not math.isfinite(rate) or rate<=0:
    raise SystemExit("invalid HPLT throughput calibration")
print(json.dumps({"smoke":"passed","target_tokens_per_scoring_second":rate,"elapsed_seconds_total":d["elapsed_seconds_total"]}))
PY
