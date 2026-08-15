#!/usr/bin/env bash
# Source only after the immutable scientific bundle has been verified.

: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
data_runtime_receipt=${H2G_DATA_RUNTIME_RECEIPT:-$data_runtime_root/runtime_receipt.json}
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"

uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/verify_data_runtime.py" \
    --runtime-root "$data_runtime_root" \
    --receipt "$data_runtime_receipt" \
    --requirements "$subproject/configs/data_runtime_requirements_v1.txt" \
    --code-root "$H2G_CODE_ROOT" \
    --code-receipt "$H2G_CODE_RECEIPT" >/dev/null
