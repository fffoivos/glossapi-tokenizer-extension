#!/usr/bin/env bash
# Signal-model-only queue runner.  Request an accelerator at submission only if
# profiling shows CPU is insufficient; no other generation may use this file.
#SBATCH --job-name=bib_evolve_signal
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%A_%a.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle sequence-model parent}
QUEUE_JSONL=${QUEUE_JSONL:?set a G4 signal-only queue}
QUEUE_SHA256=${QUEUE_SHA256:?set expected queue SHA-256}
LEAKAGE_POLICY=${LEAKAGE_POLICY:?set frozen leakage policy}
CANDIDATE_ROOT=${CANDIDATE_ROOT:?set new candidate-store root}
UENV=${UENV:-pytorch/v2.9.1:v2}
PYTHON_DEPS=${PYTHON_DEPS:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/python_deps/sklearn-1.9.0-py312}

[[ -f "$QUEUE_JSONL" && ! -L "$QUEUE_JSONL" ]] || exit 90
[[ -d "$PYTHON_DEPS" && ! -L "$PYTHON_DEPS" ]] || exit 93
[[ "$(sha256sum "$QUEUE_JSONL" | awk '{print $1}')" == "$QUEUE_SHA256" ]] || exit 91
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || exit 92
# Prevent accidental use for a decoder/header/gap row.
python3 - "$QUEUE_JSONL" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
rows=[json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
row=rows[int(sys.argv[2])]
if row["changed_component"] not in {"signal.architecture", "signal.training", "signal.features"}:
    raise SystemExit("signal runner refuses non-signal candidate")
PY

export PYTHONHASHSEED=0
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
  --cwd "$CODE_ROOT"
