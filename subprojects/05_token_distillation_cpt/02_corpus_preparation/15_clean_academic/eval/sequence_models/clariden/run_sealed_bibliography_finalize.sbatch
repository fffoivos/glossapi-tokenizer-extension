#!/usr/bin/env bash
#SBATCH --job-name=bib_sealed_packets
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

REPO_ROOT=${SEALED_BIB_REPO_ROOT:?set the exact clean Clariden checkout}
EXPECTED_COMMIT=${SEALED_BIB_EXPECTED_COMMIT:?set the reviewed commit}
QUALITY_CONSENSUS=${SEALED_BIB_QUALITY_CONSENSUS:?set the passed quality consensus path}
QUALITY_CONSENSUS_SHA=${SEALED_BIB_QUALITY_CONSENSUS_SHA:?set its exact sha256}
CONFIRM=${CONFIRM_SEALED_BIB_PACKETS:-0}
[[ "$CONFIRM" == "1" ]] || { echo "plan-only: set CONFIRM_SEALED_BIB_PACKETS=1" >&2; exit 3; }

EVAL_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
SEALED_ROOT=${SEALED_BIB_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260718}
CANDIDATES="$SEALED_ROOT/00_candidate_selection/candidates.private.jsonl"
CANDIDATE_RECEIPT="$SEALED_ROOT/00_candidate_selection/selection.receipt.json"
RUNTIME_VENV=${SEALED_BIB_RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/full_corpus_v2}
UENV=${SEALED_BIB_UENV:-pytorch/v2.9.1:v2}
FINAL="$SEALED_ROOT/10_sealed_inputs"
STAGING="$SEALED_ROOT/.10_sealed_inputs.partial-${SLURM_JOB_ID}"

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "commit mismatch" >&2; exit 4; }
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "checkout is dirty" >&2; exit 5; }
[[ -f "$CANDIDATES" && -f "$CANDIDATE_RECEIPT" && -f "$QUALITY_CONSENSUS" ]] || { echo "required input absent" >&2; exit 6; }
[[ "$(sha256sum "$QUALITY_CONSENSUS" | awk '{print $1}')" == "$QUALITY_CONSENSUS_SHA" ]] || { echo "quality consensus hash drift" >&2; exit 7; }
[[ ! -e "$FINAL" && ! -e "$STAGING" ]] || { echo "sealed-input stage already exists" >&2; exit 8; }
mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0
runner=(uenv run "$UENV" --view=default -- "$RUNTIME_VENV/bin/python")
cd "$EVAL_DIR"
"${runner[@]}" -m sequence_models.sealed_bibliography_test finalize-selection \
  --candidates "$CANDIDATES" \
  --candidate-receipt "$CANDIDATE_RECEIPT" \
  --quality-consensus "$QUALITY_CONSENSUS" \
  --documents-out "$STAGING/documents.private.jsonl" \
  --public-exclusions-out "$STAGING/exclusions.public.json" \
  --receipt-out "$STAGING/selection.final.receipt.json"

"${runner[@]}" -m sequence_models.sealed_bibliography_test create-alias-secret \
  --output "$STAGING/alias.secret"
"${runner[@]}" -m sequence_models.sealed_bibliography_test prepare-annotation \
  --documents "$STAGING/documents.private.jsonl" \
  --selection-receipt "$STAGING/selection.final.receipt.json" \
  --alias-secret "$STAGING/alias.secret" \
  --max-lines 400 --max-chars 80000 --overlap 15 \
  --pass-a-out "$STAGING/pass-a.packet.private.jsonl" \
  --pass-b-out "$STAGING/pass-b.packet.private.jsonl" \
  --line-key-out "$STAGING/line-key.private.jsonl" \
  --receipt-out "$STAGING/annotation-packets.receipt.json"

"${runner[@]}" - "$STAGING" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import hashlib, json, os, pathlib, sys
root=pathlib.Path(sys.argv[1]); commit=sys.argv[2]; job=sys.argv[3]
selection=json.loads((root/'selection.final.receipt.json').read_text())
packets=json.loads((root/'annotation-packets.receipt.json').read_text())
assert selection['status']=='passed_150_document_quality_gated_selection'
assert selection['source_counts']=={'greek_phd':50,'kallipos':50,'openarchives':50}
assert packets['document_count']==150 and packets['line_count']>0
assert packets['chunk_policy']=={'max_lines':400,'max_characters':80000,'max_display_characters_per_line':20000,'overlap_lines':15,'pass_a':'ordinary forward boundaries','pass_b':'half-chunk staggered boundaries, reversed chunk presentation'}
files=[]
for path in sorted(root.iterdir()):
    if path.name=='run.receipt.json' or not path.is_file(): continue
    files.append({'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
receipt={'schema_version':'bibliography-sealed-packets-job-v1','status':'passed','code_commit':commit,'slurm_job_id':job,'document_count':150,'line_count':packets['line_count'],'artifacts':files,'gpu_used':False}
payload=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
fd=os.open(root/'run.receipt.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'wb') as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
PY

mv "$STAGING" "$FINAL"
trap - EXIT
echo "published=$FINAL"
