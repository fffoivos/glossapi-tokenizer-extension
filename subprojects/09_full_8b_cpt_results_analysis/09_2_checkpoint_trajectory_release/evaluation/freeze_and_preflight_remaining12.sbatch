#!/usr/bin/env bash
#SBATCH --job-name=full8-r12-preflight
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/iopsstor/scratch/cscs/fffoivos/evals/full8_remaining12_checkpoint_release_20260817/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/fffoivos/evals/full8_remaining12_checkpoint_release_20260817/logs/%x-%j.err
#SBATCH --comment=AUTHORIZED_FULL8_REMAINING12_PREFLIGHT

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${REMAINING12_WRAPPER_ROOT:?set materialized remaining12 wrapper root}"
: "${REMAINING12_ASSETS_ROOT:?set new remaining12 asset root}"
: "${REMAINING12_GIT_COMMIT:?set source commit label}"
allocation_nodes=${SLURM_NNODES:-${SLURM_STEP_NUM_NODES:-}}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "$allocation_nodes" == 1 ]] || {
  echo "remaining12 preflight requires one debug node" >&2
  exit 2
}

SOURCE_ROOT=/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812
EVAL_CODE_ROOT="$SOURCE_ROOT/code/036e1e2e53e13d3cd58cc8cb22b3c38a52881580"
EVAL_VENV=/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026
eval_dir="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation"
freeze="$eval_dir/freeze_eval_bundle.py"
verify="$eval_dir/verify_eval_bundle.py"
binder="$REMAINING12_WRAPPER_ROOT/evaluation/rebind_remaining12_native_suite.py"
bindings="$REMAINING12_WRAPPER_ROOT/evaluation/remaining12_checkpoint_bindings.json"
run_root=/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12
runtime_wheel="$REMAINING12_WRAPPER_ROOT/vendor/accelerate-1.14.0-py3-none-any.whl"

[[ ! -e "$REMAINING12_WRAPPER_ROOT/bundle_receipt.json" ]] || {
  echo "wrapper is already frozen" >&2; exit 2;
}
[[ ! -e "$REMAINING12_ASSETS_ROOT" ]] || {
  echo "refusing to overwrite asset root" >&2; exit 2;
}
for path in "$freeze" "$verify" "$binder" "$bindings" "$runtime_wheel"; do
  [[ -f "$path" ]] || { echo "missing preflight input $path" >&2; exit 2; }
done

uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$runtime_wheel" python3 - <<'PY'
import accelerate, torch, transformers
expected = {"torch": "2.9.1", "transformers": "4.57.0", "accelerate": "1.14.0"}
actual = {"torch": torch.__version__.split("+")[0], "transformers": transformers.__version__, "accelerate": accelerate.__version__}
if actual != expected:
    raise SystemExit(f"runtime drift: {actual}")
print(actual)
PY

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$freeze" --root "$REMAINING12_WRAPPER_ROOT" --git-commit "$REMAINING12_GIT_COMMIT"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$verify" --root "$REMAINING12_WRAPPER_ROOT" \
  --receipt "$REMAINING12_WRAPPER_ROOT/bundle_receipt.json"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$binder" --bindings "$bindings" --run-root "$run_root" \
  --output-dir "$REMAINING12_ASSETS_ROOT"

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$REMAINING12_ASSETS_ROOT/rebind_receipt.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
receipt = json.load(open(sys.argv[1]))
if receipt.get("status") != "passed" or not all(receipt.get("checks", {}).values()):
    raise SystemExit("remaining12 rebind receipt did not pass")
if len(receipt.get("checkpoints", [])) != 12:
    raise SystemExit("expected twelve checkpoint bindings")
subset = receipt["clean_subset"]
if subset["exclusions"]["rows"] != 10076 or sum(subset["retained_by_benchmark"].values()) != 73894:
    raise SystemExit("contamination population drift")
for key in ("contract", "manifest"):
    row = receipt[key]
    actual = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
    if actual != row["sha256"]:
        raise SystemExit(f"{key} receipt hash drift")
print(json.dumps({"ok": True, "checkpoints": 12, "retained": 73894}, sort_keys=True))
PY
