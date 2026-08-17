#!/usr/bin/env bash
# Experiment-owned workaround for fffoivos/apertus-cscs-efficiency#88.
# It creates a new immutable evaluator extension; it never mutates the verified
# base data runtime or a frozen code bundle.
#SBATCH --job-name=h2g-gmmlu-runtime
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_BASE_RUNTIME_ROOT:?set verified base data runtime root}"
: "${H2G_BASE_RUNTIME_RECEIPT:?set verified base data runtime receipt}"
: "${H2G_GREEKMMLU_EVAL_RUNTIME_ROOT:?set new immutable evaluator runtime root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "GreekMMLU evaluator runtime extension must run on debug" >&2
  exit 2
}

verify_bundle="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify_bundle" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

requirements="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/greekmmlu_eval_runtime_extension_requirements_v1.txt"
verifier="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/verify_greekmmlu_eval_runtime_extension.py"
[[ -s "$requirements" ]] || { echo "requirements lock missing: $requirements" >&2; exit 2; }
[[ -f "$verifier" ]] || { echo "runtime verifier missing: $verifier" >&2; exit 2; }
[[ -d "$H2G_BASE_RUNTIME_ROOT" && -f "$H2G_BASE_RUNTIME_RECEIPT" ]] || {
  echo "verified base runtime is missing" >&2
  exit 2
}
[[ "$H2G_BASE_RUNTIME_RECEIPT" == "$H2G_BASE_RUNTIME_ROOT/runtime_receipt.json" ]] || {
  echo "base runtime receipt must be its immutable in-root receipt" >&2
  exit 2
}
[[ ! -e "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT" ]] || {
  echo "immutable evaluator runtime exists" >&2
  exit 2
}

base_code_root=$(python3 - "$H2G_BASE_RUNTIME_RECEIPT" <<'PY'
import json, pathlib, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("schema_version") != "apertus_hard_h_to_g_data_runtime_v1":
    raise SystemExit("base runtime receipt schema drift")
print(receipt["code_bundle"]["root"])
PY
)
base_code_receipt="${base_code_root}.receipt.json"
[[ -d "$base_code_root" && -f "$base_code_receipt" ]] || {
  echo "base runtime code binding is missing" >&2
  exit 2
}
/usr/bin/python3.11 "$verify_bundle" \
  --root "$base_code_root" --receipt "$base_code_receipt" --kind scientific

uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$H2G_BASE_RUNTIME_ROOT/bin/python" \
  "$base_code_root/subprojects/08_targeted_8b_cpt_experiments/scripts/verify_data_runtime.py" \
  --runtime-root "$H2G_BASE_RUNTIME_ROOT" --receipt "$H2G_BASE_RUNTIME_RECEIPT" \
  --requirements "$base_code_root/subprojects/08_targeted_8b_cpt_experiments/configs/data_runtime_requirements_v1.txt" \
  --code-root "$base_code_root" --code-receipt "$base_code_receipt" >/dev/null

partial="${H2G_GREEKMMLU_EVAL_RUNTIME_ROOT}.partial-${SLURM_JOB_ID}"
[[ ! -e "$partial" ]] || { echo "runtime partial exists: $partial" >&2; exit 2; }
mkdir -p "$(dirname "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT")"
cleanup_partial() { rm -rf -- "$partial"; }
trap cleanup_partial EXIT

uenv run pytorch/v2.9.1:v2 --view=default -- bash -lc '
set -euo pipefail
[[ "$(uname -m)" == aarch64 ]]
python3 -m venv --system-site-packages "$1"
site="$1/lib/python3.12/site-packages"
printf "%s\\n" "$2/lib/python3.12/site-packages" > "$site/h2g_base_data_runtime.pth"
"$1/bin/python" -m pip install --disable-pip-version-check --no-deps --requirement "$3"
PYTHONDONTWRITEBYTECODE=1 "$1/bin/python" - "$1" "$2" "$3" "$4" "$5" "$6" <<"PY"
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

root, base, requirements, code_root, code_receipt_path, final_root = map(Path, sys.argv[1:])
code_receipt = json.loads(code_receipt_path.read_text(encoding="utf-8"))
base_receipt = base / "runtime_receipt.json"
if not base_receipt.is_file():
    raise SystemExit("base runtime receipt is not at its immutable root")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema_version": "apertus_hard_h_to_g_greekmmlu_eval_runtime_extension_v1",
    "status": "completed",
    "machine": platform.machine(),
    "python_version": platform.python_version(),
    "uenv_image": "pytorch/v2.9.1:v2",
    "runtime_root": str(final_root),
    "requirements_path": str(requirements),
    "requirements_sha256": sha256(requirements),
    "packages": {"accelerate": importlib.metadata.version("accelerate")},
    "base_runtime": {"root": str(base), "receipt_sha256": sha256(base_receipt)},
    "code_bundle": {"root": str(code_root), "tree_sha256": code_receipt["tree_sha256"]},
}
(root / "runtime_receipt.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
PY
' bash "$partial" "$H2G_BASE_RUNTIME_ROOT" "$requirements" "$H2G_CODE_ROOT" "$H2G_CODE_RECEIPT" "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT"

mv "$partial" "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT"
trap - EXIT
failed_root="${H2G_GREEKMMLU_EVAL_RUNTIME_ROOT}.failed-${SLURM_JOB_ID}"
[[ ! -e "$failed_root" ]] || { echo "failed-runtime quarantine exists: $failed_root" >&2; exit 2; }
quarantine_failed_runtime() { mv -- "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT" "$failed_root"; }
trap quarantine_failed_runtime ERR
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT/bin/python" "$verifier" \
  --runtime-root "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT" \
  --receipt "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT/runtime_receipt.json" \
  --requirements "$requirements" \
  --base-runtime-root "$H2G_BASE_RUNTIME_ROOT" \
  --base-runtime-receipt "$H2G_BASE_RUNTIME_RECEIPT" \
  --code-root "$H2G_CODE_ROOT" --code-receipt "$H2G_CODE_RECEIPT" \
  > "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT/runtime_verification.json"
chmod -R a-w "$H2G_GREEKMMLU_EVAL_RUNTIME_ROOT"
trap - ERR
