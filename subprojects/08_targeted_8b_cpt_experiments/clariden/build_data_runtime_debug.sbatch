#!/usr/bin/env bash
#SBATCH --job-name=h2g-runtime
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_RUNTIME_ROOT:?set a new immutable runtime root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "runtime build must run on debug" >&2; exit 2; }

verify_bundle="$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify_bundle" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

requirements="$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/data_runtime_requirements_v1.txt"
[[ -s "$requirements" ]] || { echo "requirements lock missing: $requirements" >&2; exit 2; }
[[ ! -e "$TARGET8_RUNTIME_ROOT" ]] || { echo "immutable runtime exists: $TARGET8_RUNTIME_ROOT" >&2; exit 2; }
partial="${TARGET8_RUNTIME_ROOT}.partial-${SLURM_JOB_ID}"
[[ ! -e "$partial" ]] || { echo "runtime partial exists: $partial" >&2; exit 2; }
mkdir -p "$(dirname "$TARGET8_RUNTIME_ROOT")"
cleanup_partial() { rm -rf -- "$partial"; }
trap cleanup_partial EXIT

uenv run pytorch/v2.9.1:v2 --view=default -- bash -lc '
set -euo pipefail
[[ "$(uname -m)" == aarch64 ]]
# The historical Megatron preprocessing tool imports torch from the frozen
# PyTorch uenv.  Keep the small data packages pinned in this venv, while making
# the uenv torch package visible to the exact same interpreter used below.
python3 -m venv --system-site-packages "$1"
"$1/bin/python" -m pip install --disable-pip-version-check --requirement "$2"
PYTHONDONTWRITEBYTECODE=1 "$1/bin/python" - "$1" "$2" "$3" "$4" "$5" <<"PY"
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
requirements = Path(sys.argv[2]).resolve()
code_root = Path(sys.argv[3]).resolve()
code_receipt_path = Path(sys.argv[4]).resolve()
final_root = Path(sys.argv[5]).resolve()
code_receipt = json.loads(code_receipt_path.read_text(encoding="utf-8"))
packages = {
    name: importlib.metadata.version(name)
    for name in (
        "blake3", "datasets", "duckdb", "huggingface-hub", "numpy", "pyarrow",
        "pytest", "regex", "tokenizers", "transformers", "zstandard",
    )
}
expected = {
    "blake3": "1.0.9", "datasets": "4.0.0", "duckdb": "1.5.4",
    "huggingface-hub": "0.36.0", "numpy": "2.5.1", "pyarrow": "21.0.0",
    "pytest": "8.4.1", "regex": "2026.7.10", "tokenizers": "0.22.1",
    "transformers": "4.57.0", "zstandard": "0.25.0",
}
if packages != expected:
    raise SystemExit(f"resolved package drift: {packages!r}")
for name in (
    "blake3", "datasets", "duckdb", "huggingface_hub", "numpy", "pyarrow",
    "pyarrow.compute", "pyarrow.dataset", "pyarrow.parquet", "pytest", "regex",
    "tokenizers", "torch", "transformers", "zstandard",
):
    module = importlib.import_module(name)
    if not getattr(module, "__file__", None):
        raise SystemExit(f"incomplete import: {name}")
if importlib.import_module("pyarrow").__version__ != "21.0.0":
    raise SystemExit("live PyArrow version mismatch")
if not importlib.import_module("torch").__version__.startswith("2.9.1"):
    raise SystemExit("live uenv torch version mismatch")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

site = root / "lib" / "python3.12" / "site-packages"
required = (
    "pyarrow/__init__.py",
    "pyarrow/compute.py",
    "pyarrow/dataset.py",
    "pyarrow/parquet/__init__.py",
)
required_files = {
    str((site / relative).relative_to(root)): sha256(site / relative)
    for relative in required
}
payload = {
    "schema_version": "apertus_hard_h_to_g_data_runtime_v1",
    "status": "completed",
    "machine": platform.machine(),
    "python_version": platform.python_version(),
    "uenv_image": "pytorch/v2.9.1:v2",
    "torch_version": importlib.metadata.version("torch"),
    "runtime_root": str(final_root),
    "requirements_path": str(requirements),
    "requirements_sha256": sha256(requirements),
    "packages": packages,
    "required_files": required_files,
    "code_bundle": {
        "root": str(code_root),
        "tree_sha256": code_receipt["tree_sha256"],
    },
}
(root / "runtime_receipt.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
' bash "$partial" "$requirements" "$TARGET8_CODE_ROOT" "$TARGET8_CODE_RECEIPT" "$TARGET8_RUNTIME_ROOT"

mv "$partial" "$TARGET8_RUNTIME_ROOT"
trap - EXIT
failed_root="${TARGET8_RUNTIME_ROOT}.failed-${SLURM_JOB_ID}"
[[ ! -e "$failed_root" ]] || { echo "failed-runtime quarantine exists: $failed_root" >&2; exit 2; }
quarantine_failed_runtime() { mv -- "$TARGET8_RUNTIME_ROOT" "$failed_root"; }
trap quarantine_failed_runtime ERR
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$TARGET8_RUNTIME_ROOT/bin/python" \
    "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/verify_data_runtime.py" \
    --runtime-root "$TARGET8_RUNTIME_ROOT" \
    --receipt "$TARGET8_RUNTIME_ROOT/runtime_receipt.json" \
    --requirements "$requirements" \
    --code-root "$TARGET8_CODE_ROOT" \
    --code-receipt "$TARGET8_CODE_RECEIPT" \
    > "$TARGET8_RUNTIME_ROOT/runtime_verification.json"
chmod -R a-w "$TARGET8_RUNTIME_ROOT"
trap - ERR
