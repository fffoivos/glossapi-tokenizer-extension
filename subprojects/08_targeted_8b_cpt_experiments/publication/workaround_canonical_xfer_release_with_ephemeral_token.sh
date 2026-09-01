#!/usr/bin/env bash
# Experiment-side token bridge for the unchanged canonical xfer publisher.
#
# The caller injects HF_TOKEN only in the environment of the held xfer step.
# This adapter writes a mode-600, node-local temporary token file required by
# the canonical publisher, removes it on every exit path, and delegates the
# release transaction unchanged.  It is not a checkpoint publisher itself.
set -euo pipefail

: "${HF_TOKEN:?inject per command}"
: "${EFFICIENCY_BUNDLE:?set immutable bundle root}"
: "${EFFICIENCY_BUNDLE_RECEIPT:?set immutable bundle receipt}"
: "${CHECKPOINT_RELEASE_CONTRACT:?set release contract}"
: "${CHECKPOINT_RELEASE_RECEIPT:?set release receipt}"
: "${REPLAY_XFER_RUNTIME_ROOT:?set xfer runtime root}"
: "${REPLAY_XFER_RUNTIME_RECEIPT:?set xfer runtime receipt}"
: "${REPLAY_XFER_PYTHON:?set xfer runtime Python}"

umask 077
token_file=$(mktemp /tmp/apertus-hf-token.XXXXXX)
printf '%s' "${HF_TOKEN}" >"${token_file}"
unset HF_TOKEN
trap 'rm -f "${token_file}"' EXIT

HF_TOKEN_FILE="${token_file}" \
  bash "${EFFICIENCY_BUNDLE}/slurm/checkpoint/publish_model_checkpoint_xfer.sbatch"
