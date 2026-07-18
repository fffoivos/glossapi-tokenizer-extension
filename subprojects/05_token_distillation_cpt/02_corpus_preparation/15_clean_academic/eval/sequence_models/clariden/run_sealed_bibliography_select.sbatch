#!/usr/bin/env bash
#SBATCH --job-name=bib_sealed_select
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

REPO_ROOT=${SEALED_BIB_REPO_ROOT:?set the exact clean Clariden checkout}
EXPECTED_COMMIT=${SEALED_BIB_EXPECTED_COMMIT:?set the reviewed commit}
CONFIRM=${CONFIRM_SEALED_BIB_SELECT:-0}
[[ "$CONFIRM" == "1" ]] || { echo "plan-only: set CONFIRM_SEALED_BIB_SELECT=1" >&2; exit 3; }

EVAL_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
NORMALIZATION_ROOT=${SEALED_BIB_NORMALIZATION_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/pipeline_runs/full-corpus-v2-20260712-d076a59/stages/10-normalize}
NORMALIZATION_SHA=${SEALED_BIB_NORMALIZATION_SHA:-ccd6c2f6212ef597a63900b8015b9a432e820fe2fea8a41d7a62ac713470bc45}
STRUCT_ROOT=${SEALED_BIB_STRUCT_ROOT:-/iopsstor/scratch/cscs/fffoivos/inputs/APERTUS_CLASSIFIER_HANDOFF_20260712/STRUCT_2K}
STRUCT_MANIFEST=${SEALED_BIB_STRUCT_MANIFEST:-$STRUCT_ROOT/manifest.jsonl}
STRUCT_MANIFEST_SHA=${SEALED_BIB_STRUCT_MANIFEST_SHA:-c08611352517ff40668eb1a74daf1c5bb645f3acf03eec4c002bb2b3f222621c}
PRIOR500=${SEALED_BIB_PRIOR500:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/source_matched_holdouts/source_matched_c2_20260713_43cf377_a/documents.jsonl}
PRIOR500_SHA=${SEALED_BIB_PRIOR500_SHA:-377b21a1cdc6a41d31264a7ad459d0539d29894285bda79c9f3cb33eb3a0dd25}
RUNTIME_VENV=${SEALED_BIB_RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/full_corpus_v2}
UENV=${SEALED_BIB_UENV:-pytorch/v2.9.1:v2}
RUST_DEP=${SEALED_BIB_RUST_DEP:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/python_deps/glossapi_rs_noise_6f29a2825559c540-py312}
SEALED_ROOT=${SEALED_BIB_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260718}
FINAL="$SEALED_ROOT/00_candidate_selection"
STAGING="$SEALED_ROOT/.00_candidate_selection.partial-${SLURM_JOB_ID}"

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "commit mismatch" >&2; exit 4; }
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "checkout is dirty" >&2; exit 5; }
[[ -f "$EVAL_DIR/sequence_models/sealed_bibliography_test.py" ]] || { echo "module absent" >&2; exit 6; }
[[ "$(sha256sum "$NORMALIZATION_ROOT/normalization_manifest.json" | awk '{print $1}')" == "$NORMALIZATION_SHA" ]] || { echo "normalization hash drift" >&2; exit 7; }
[[ "$(sha256sum "$STRUCT_MANIFEST" | awk '{print $1}')" == "$STRUCT_MANIFEST_SHA" ]] || { echo "STRUCT-2K hash drift" >&2; exit 8; }
[[ "$(sha256sum "$PRIOR500" | awk '{print $1}')" == "$PRIOR500_SHA" ]] || { echo "prior-500 hash drift" >&2; exit 9; }
[[ -d "$RUST_DEP/glossapi_rs_noise" ]] || { echo "GlossAPI Rust dependency absent" >&2; exit 10; }
[[ ! -e "$FINAL" && ! -e "$STAGING" ]] || { echo "candidate stage already exists" >&2; exit 11; }
mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
runner=(uenv run "$UENV" --view=default -- env "PYTHONPATH=$RUST_DEP:$EVAL_DIR" "$RUNTIME_VENV/bin/python")
cd "$EVAL_DIR"
"${runner[@]}" -c 'import glossapi_rs_noise as g; from importlib.metadata import version; assert version("glossapi-rs-noise")=="0.1.0"; assert hasattr(g,"score_markdown_file_detailed"); print(g.__file__)'
"${runner[@]}" -m sequence_models.sealed_bibliography_test select-candidates \
  --normalization-root "$NORMALIZATION_ROOT" \
  --historical-manifest "$STRUCT_MANIFEST" \
  --historical-root "$STRUCT_ROOT" \
  --previous-documents "$PRIOR500" \
  --quota greek_phd=50 --quota kallipos=50 --quota openarchives=50 \
  --oversample 4 \
  --seed bibliography-sealed-150-v1-20260718 \
  --near-duplicate-threshold 0.80 \
  --candidates-out "$STAGING/candidates.private.jsonl" \
  --quality-packet-out "$STAGING/quality.flagged.private.jsonl" \
  --receipt-out "$STAGING/selection.receipt.json"

"${runner[@]}" - "$STAGING" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import hashlib, json, os, pathlib, sys
root=pathlib.Path(sys.argv[1]); commit=sys.argv[2]; job=sys.argv[3]
selection=json.loads((root/'selection.receipt.json').read_text())
assert selection['status']=='candidate_pool_ready_for_prediction_blind_quality_gate'
assert selection['pool_counts']=={'greek_phd':200,'kallipos':200,'openarchives':200}
assert selection['global_deduplication']['threshold']==0.8
assert selection['global_deduplication']['same_source_only'] is False
assert selection['one_representation_per_work'] is True
files=[]
for path in sorted(root.iterdir()):
    if path.name=='run.receipt.json' or not path.is_file(): continue
    files.append({'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
receipt={'schema_version':'bibliography-sealed-selection-job-v1','status':'passed','code_commit':commit,'slurm_job_id':job,'artifacts':files,'corpus_mutation_performed':False,'model_predictions_accessed':False,'gpu_used':False}
payload=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
fd=os.open(root/'run.receipt.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'wb') as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
PY

mkdir -p "$SEALED_ROOT"
mv "$STAGING" "$FINAL"
trap - EXIT
echo "published=$FINAL"
