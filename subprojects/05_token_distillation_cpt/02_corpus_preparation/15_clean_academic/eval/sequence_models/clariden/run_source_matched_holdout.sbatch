#!/usr/bin/env bash
#SBATCH --job-name=cpt_source_holdout
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

REPO_ROOT=${SOURCE_HOLDOUT_REPO_ROOT:?set exact clean checkout}
EXPECTED_COMMIT=${SOURCE_HOLDOUT_EXPECTED_COMMIT:?set exact commit}
RUN_ID=${SOURCE_HOLDOUT_RUN_ID:?set immutable run ID}
CONFIRM=${CONFIRM_SOURCE_HOLDOUT:-0}
[[ "$CONFIRM" == "1" ]] || { echo "plan-only: set CONFIRM_SOURCE_HOLDOUT=1" >&2; exit 3; }

EVAL_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
MODULE="$EVAL_DIR/sequence_models/source_matched_holdout.py"
NORMALIZATION_ROOT=${SOURCE_HOLDOUT_NORMALIZATION_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/pipeline_runs/full-corpus-v2-20260712-d076a59/stages/10-normalize}
HISTORICAL_ROOT=${SOURCE_HOLDOUT_HISTORICAL_ROOT:-/iopsstor/scratch/cscs/fffoivos/inputs/APERTUS_CLASSIFIER_HANDOFF_20260712/STRUCT_2K}
HISTORICAL_MANIFEST=${SOURCE_HOLDOUT_HISTORICAL_MANIFEST:-$HISTORICAL_ROOT/manifest.jsonl}
C2_MODEL=${SOURCE_HOLDOUT_C2_MODEL:?set immutable C2 model path}
C2_SHA256=${SOURCE_HOLDOUT_C2_SHA256:?set frozen C2 sha256}
C2_BIAS=${SOURCE_HOLDOUT_C2_BIAS:-0.5}
RUNTIME_VENV=${SOURCE_HOLDOUT_RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/full_corpus_v2}
UENV=${SOURCE_HOLDOUT_UENV:-pytorch/v2.9.1:v2}
RUN_ROOT=${SOURCE_HOLDOUT_RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/source_matched_holdouts}
FINAL="$RUN_ROOT/$RUN_ID"
STAGING="$RUN_ROOT/.${RUN_ID}.partial-${SLURM_JOB_ID}"

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "commit mismatch" >&2; exit 4; }
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "checkout is dirty" >&2; exit 5; }
[[ -f "$MODULE" && -f "$HISTORICAL_MANIFEST" && -f "$C2_MODEL" ]] || { echo "required input absent" >&2; exit 6; }
[[ ! -e "$FINAL" && ! -e "$STAGING" ]] || { echo "output already exists" >&2; exit 7; }
mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0

runner=(uenv run "$UENV" --view=default -- "$RUNTIME_VENV/bin/python")
cd "$EVAL_DIR"
"${runner[@]}" -m sequence_models.source_matched_holdout build \
    --normalization-root "$NORMALIZATION_ROOT" \
    --historical-manifest "$HISTORICAL_MANIFEST" \
    --historical-root "$HISTORICAL_ROOT" \
    --quota greek_phd=150 --quota kallipos=150 --quota openarchives=200 \
    --oversample 4 \
    --seed source-matched-holdout-v1-20260713 \
    --near-duplicate-threshold 0.80 \
    --documents-out "$STAGING/documents.jsonl" \
    --manifest-out "$STAGING/selection.manifest.json"

"${runner[@]}" -m sequence_models.source_matched_holdout predict \
    --documents "$STAGING/documents.jsonl" \
    --selection-manifest "$STAGING/selection.manifest.json" \
    --model "$C2_MODEL" \
    --expected-model-sha256 "$C2_SHA256" \
    --expected-deletion-bias "$C2_BIAS" \
    --workers 8 \
    --predictions-out "$STAGING/c2.predictions.jsonl" \
    --receipt-out "$STAGING/c2.prediction.receipt.json"

"${runner[@]}" -m sequence_models.source_matched_holdout build-review \
    --documents "$STAGING/documents.jsonl" \
    --selection-manifest "$STAGING/selection.manifest.json" \
    --predictions "$STAGING/c2.predictions.jsonl" \
    --prediction-receipt "$STAGING/c2.prediction.receipt.json" \
    --per-stratum 50 --context-radius 20 \
    --seed source-matched-holdout-review-v1-20260713 \
    --requests-out "$STAGING/review.requests.jsonl" \
    --key-out "$STAGING/review.key.jsonl" \
    --manifest-out "$STAGING/review.manifest.json"

"${runner[@]}" - "$STAGING" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import hashlib, json, os, pathlib, sys
root=pathlib.Path(sys.argv[1]); commit=sys.argv[2]; job=sys.argv[3]
sel=json.loads((root/'selection.manifest.json').read_text())
pred=json.loads((root/'c2.prediction.receipt.json').read_text())
review=json.loads((root/'review.manifest.json').read_text())
assert sel['document_count']==500 and sel['source_counts']=={'greek_phd':150,'kallipos':150,'openarchives':200}
assert pred['counts']['documents']==500 and pred['execution']['model_fitting_performed'] is False
assert review['request_count']==200 and all(value==50 for value in review['counts'].values())
files=[]
for path in sorted(root.iterdir()):
    if path.name=='run.receipt.json' or not path.is_file(): continue
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({'path':path.name,'bytes':path.stat().st_size,'sha256':h})
receipt={'schema_version':'academic-structure-source-matched-run-v1','status':'passed','code_commit':commit,'slurm_job_id':job,'production_eligible':False,'document_count':500,'review_case_count':200,'artifacts':files,'corpus_mutation_performed':False,'model_fitting_performed':False,'codex_review_performed':False}
payload=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
fd=os.open(root/'run.receipt.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'wb') as f: f.write(payload); f.flush(); os.fsync(f.fileno())
PY

mkdir -p "$RUN_ROOT"
mv "$STAGING" "$FINAL"
trap - EXIT
echo "published=$FINAL"
