#!/usr/bin/env bash
#SBATCH --job-name=mini_lm_eval_runtime
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${LM_EVAL_RUNTIME_ROOT:?set new immutable target root}"
: "${LM_EVAL_RUNTIME_RECEIPT:?set new runtime receipt path}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
lock="$SCIENTIFIC_BUNDLE/evaluation/lm_eval_runtime_requirements_0_4_11.txt"
partial="${LM_EVAL_RUNTIME_ROOT}.partial.${SLURM_JOB_ID}"

[[ "$UENV_IMAGE" == pytorch/v2.9.1:v2 ]] || { echo "unexpected uenv: $UENV_IMAGE" >&2; exit 2; }
[[ ! -e "$LM_EVAL_RUNTIME_ROOT" && ! -e "$LM_EVAL_RUNTIME_RECEIPT" && ! -e "$partial" ]] || {
  echo "refusing to replace an lm-eval runtime artifact" >&2
  exit 2
}
test -s "$lock"
mkdir -p "$partial"
uenv run "$UENV_IMAGE" --view=default -- python3 -m pip install \
  --disable-pip-version-check --no-cache-dir --no-deps --target "$partial" -r "$lock"
alias_source="$SCIENTIFIC_BUNDLE/evaluation/lm_eval_custom_tasks/global_mmlu.yaml"
alias_target="$partial/lm_eval/tasks/global_mmlu/_global_mmlu.yaml"
test -s "$alias_source"
[[ ! -e "$alias_target" ]] || { echo "official runtime unexpectedly defines global_mmlu alias" >&2; exit 2; }
install -m 0444 "$alias_source" "$alias_target"

uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$partial" python3 - <<'PY'
import importlib.metadata as metadata
from lm_eval.tasks import TaskManager

required = {
    "arc_challenge", "arc_easy", "hellaswag", "winogrande", "piqa",
    "mmlu", "global_mmlu", "xnli", "xcopa",
}
manager = TaskManager()
missing = sorted(required - set(manager.all_tasks))
if missing:
    raise SystemExit(f"required retention tasks unavailable: {missing}")
assert metadata.version("lm-eval") == "0.4.11"
assert metadata.version("accelerate") == "1.13.0"
print({"lm_eval": metadata.version("lm-eval"), "tasks": len(manager.all_tasks)})
PY

mv "$partial" "$LM_EVAL_RUNTIME_ROOT"
uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$LM_EVAL_RUNTIME_ROOT" python3 \
  "$SCIENTIFIC_BUNDLE/evaluation/freeze_lm_eval_runtime.py" \
  --lm-eval-root "$LM_EVAL_RUNTIME_ROOT" --requirements-lock "$lock" \
  --uenv-image "$UENV_IMAGE" --output "$LM_EVAL_RUNTIME_RECEIPT"
