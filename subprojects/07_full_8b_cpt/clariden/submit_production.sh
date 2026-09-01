#!/usr/bin/env bash
# Historical six-segment launcher deliberately disabled on 2026-08-09.
#
# It encoded the superseded 19,248-update / six-segment graph and does not
# bind the scientific and operational bundle receipts. Keeping it executable
# could silently submit a scientifically valid-looking but operationally unsafe
# campaign. It is retained only as a named, fail-closed tombstone for old
# documentation and shell history.
set -euo pipefail

cat >&2 <<'EOF'
ERROR: clariden/submit_production.sh is retired and cannot submit jobs.

The active 8B CPT path is receipt-bound and uses five segments:
  clariden/submit_production_resource_aware.sh

Follow FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md. Invoke it only from the
frozen operational bundle after its scientific and operational launch gates
pass; never recreate the legacy six-segment graph from this checkout.
EOF
exit 2
