#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-retention-shards-finalize
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT:?set recovery bundle receipt}"
: "${WAVE_MANIFEST:?set prebuilt wave manifest}"
: "${RETENTION_WAVE_RECEIPT:?set aggregate receipt path}"
: "${SHARD_RECEIPTS:?set colon-separated shard receipt paths}"

"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/verify_code_bundle.py" \
  --root "$SCIENTIFIC_BUNDLE" --receipt "$RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT" \
  --kind scientific
IFS=':' read -ra shards <<<"$SHARD_RECEIPTS"
for shard in "${shards[@]}"; do
  "$HOST_PYTHON" - "$shard" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get("schema_version") != "apertus_mini_retention_endpoint_shard_v1" or p.get("status") != "completed":
    raise SystemExit(f"incomplete shard receipt: {sys.argv[1]}")
PY
done
exec "$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/finalize_retention_endpoint_wave.py" \
  --wave-manifest "$WAVE_MANIFEST" --output "$RETENTION_WAVE_RECEIPT"
