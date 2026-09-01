#!/usr/bin/env bash
#SBATCH --job-name=h2g-td-xfer-env
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_TD_XFER_RUNTIME_ROOT:?set a new immutable xfer runtime root}"
[[ "${SLURM_JOB_PARTITION:-}" == xfer && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" ]] || { echo "TD xfer runtime must be CPU-only" >&2; exit 2; }

verify_bundle="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify_bundle" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
requirements="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/td_xfer_runtime_requirements_v1.txt"
[[ -s "$requirements" ]] || { echo "missing xfer runtime lock: $requirements" >&2; exit 2; }
[[ ! -e "$H2G_TD_XFER_RUNTIME_ROOT" ]] || { echo "immutable runtime exists" >&2; exit 2; }
partial="${H2G_TD_XFER_RUNTIME_ROOT}.partial-${SLURM_JOB_ID}"
[[ ! -e "$partial" ]] || { echo "runtime partial exists" >&2; exit 2; }
cleanup_partial() { rm -rf -- "$partial"; }
trap cleanup_partial EXIT

/usr/bin/python3.11 -m venv "$partial"
"$partial/bin/python" -m pip install --disable-pip-version-check --requirement "$requirements"
"$partial/bin/python" - "$partial" "$requirements" "$H2G_CODE_ROOT" "$H2G_CODE_RECEIPT" "$H2G_TD_XFER_RUNTIME_ROOT" <<'PY'
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
requirements = Path(sys.argv[2]).resolve()
code_root = Path(sys.argv[3]).resolve()
code_receipt = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
final_root = Path(sys.argv[5]).resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

packages = {name: importlib.metadata.version(name) for name in ("numpy", "tokenizers")}
expected = {"numpy": "2.4.6", "tokenizers": "0.22.1"}
if packages != expected:
    raise SystemExit(f"resolved package drift: {packages!r}")
site = root / "lib" / "python3.11" / "site-packages"
required_files = {
    str(path.relative_to(root)): sha256(path)
    for path in (site / "numpy/__init__.py", site / "tokenizers/__init__.py")
}
payload = {
    "schema_version": "apertus_hard_h_to_g_td_xfer_runtime_v1",
    "status": "completed",
    "machine": platform.machine(),
    "python_version": platform.python_version(),
    "runtime_root": str(final_root),
    "requirements_path": str(requirements),
    "requirements_sha256": sha256(requirements),
    "packages": packages,
    "required_files": required_files,
    "code_bundle": {"root": str(code_root), "tree_sha256": code_receipt["tree_sha256"]},
}
(root / "runtime_receipt.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
mv "$partial" "$H2G_TD_XFER_RUNTIME_ROOT"
trap - EXIT

failed_root="${H2G_TD_XFER_RUNTIME_ROOT}.failed-${SLURM_JOB_ID}"
quarantine_failed_runtime() { mv -- "$H2G_TD_XFER_RUNTIME_ROOT" "$failed_root"; }
trap quarantine_failed_runtime ERR
PYTHONDONTWRITEBYTECODE=1 "$H2G_TD_XFER_RUNTIME_ROOT/bin/python" \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/verify_td_xfer_runtime.py" \
  --runtime-root "$H2G_TD_XFER_RUNTIME_ROOT" \
  --receipt "$H2G_TD_XFER_RUNTIME_ROOT/runtime_receipt.json" \
  --requirements "$requirements" \
  --code-root "$H2G_CODE_ROOT" --code-receipt "$H2G_CODE_RECEIPT" \
  > "$H2G_TD_XFER_RUNTIME_ROOT/runtime_verification.json"
chmod -R a-w "$H2G_TD_XFER_RUNTIME_ROOT"
trap - ERR
