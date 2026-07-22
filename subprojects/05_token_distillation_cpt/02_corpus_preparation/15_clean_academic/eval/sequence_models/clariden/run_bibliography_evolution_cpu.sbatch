#!/usr/bin/env bash
# Execute one immutable, leak-checked bibliography evolution queue row.
# Submit as --array=0-(N-1)%PARALLEL after freezing QUEUE_JSONL.
#SBATCH --job-name=bib_evolve_cpu
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%A_%a.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle sequence-model parent}
QUEUE_JSONL=${QUEUE_JSONL:?set immutable rendered candidate queue}
QUEUE_SHA256=${QUEUE_SHA256:?set expected queue SHA-256}
LEAKAGE_POLICY=${LEAKAGE_POLICY:?set frozen leakage policy}
CANDIDATE_ROOT=${CANDIDATE_ROOT:?set new candidate-store root}
WORKING_DIR=${WORKING_DIR:-$CODE_ROOT}
UENV=${UENV:-pytorch/v2.9.1:v2}
PYTHON_DEPS=${PYTHON_DEPS:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/python_deps/sklearn-1.9.0-py312}

[[ -f "$QUEUE_JSONL" && ! -L "$QUEUE_JSONL" ]] || exit 90
[[ -f "$LEAKAGE_POLICY" && ! -L "$LEAKAGE_POLICY" ]] || exit 91
[[ -d "$PYTHON_DEPS" && ! -L "$PYTHON_DEPS" ]] || exit 94
[[ "$(sha256sum "$QUEUE_JSONL" | awk '{print $1}')" == "$QUEUE_SHA256" ]] || exit 92
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 93
python3 - "$QUEUE_JSONL" "$LEAKAGE_POLICY" "$SLURM_ARRAY_TASK_ID" <<'PY'
import hashlib, json, sys
rows=[json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
policy=json.load(open(sys.argv[2], encoding="utf-8"))
canonical=(json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
policy_sha=hashlib.sha256(canonical).hexdigest()
if any(row.get("leakage_policy_sha256") != policy_sha for row in rows):
    raise SystemExit("queue is bound to a different leakage policy")
if rows[int(sys.argv[3])]["changed_component"].startswith("signal."):
    raise SystemExit("CPU evolution runner refuses signal-model training")
PY

# Non-signal candidates are CPU work.  Clariden has no separate CPU-only
# compute partition, so accelerator visibility is explicitly disabled.
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUNTIME_PYTHONPATH="$CODE_ROOT:$PYTHON_DEPS"

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$RUNTIME_PYTHONPATH" python3 -c \
  'import numpy, sklearn, torch; assert sklearn.__version__ == "1.9.0"; print(numpy.__version__, sklearn.__version__, torch.__version__)'

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$RUNTIME_PYTHONPATH" python3 -m sequence_models.bibliography_evolution execute-queue-index \
  --queue "$QUEUE_JSONL" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --leakage-policy "$LEAKAGE_POLICY" \
  --candidate-root "$CANDIDATE_ROOT" \
  --cwd "$WORKING_DIR"
