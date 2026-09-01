#!/usr/bin/env bash
# Temporary experiment-side adapter for Apertus CSCS efficiency issue #130.
#
# freeze-bundle currently resolves a virtualenv's bin/python symlink before
# invoking `-m pip`; on Clariden xfer that changes the selected executable to
# the no-pip system interpreter.  This wrapper is deliberately not a launcher
# or publisher: the canonical freeze-bundle command still performs every
# bundle and receipt operation.  It only preserves the explicitly selected
# virtualenv executable for that command.
set -euo pipefail

: "${APERTUS_VENDOR_PYTHON:?set the exact virtualenv interpreter}"
exec "${APERTUS_VENDOR_PYTHON}" "$@"
