#!/usr/bin/env bash
#SBATCH --job-name=mini_td_conversion_receipt
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_FULL_TD_DIR:?set full tied-TD HF directory}"
: "${TD_CONVERSION_ROOT:?set completed conversion root}"
: "${MEGATRON_COMMIT:?set pinned Megatron commit}"
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}

test -s "$TD_CONVERSION_ROOT/roundtrip_verification.json"
test -d "$TD_CONVERSION_ROOT/distributed/torch_dist/release"
test -d "$TD_CONVERSION_ROOT/hf_roundtrip"
"$HOST_PYTHON" \
  "$MINI_CODE_BUNDLE/initialization/finalize_td_megatron_conversion.py" \
  --hf-reference "$MINI_FULL_TD_DIR" \
  --torch-dist-root "$TD_CONVERSION_ROOT/distributed/torch_dist" \
  --hf-roundtrip "$TD_CONVERSION_ROOT/hf_roundtrip" \
  --verification "$TD_CONVERSION_ROOT/roundtrip_verification.json" \
  --megatron-commit "$MEGATRON_COMMIT" \
  --output "$TD_CONVERSION_ROOT/conversion_receipt.json"
