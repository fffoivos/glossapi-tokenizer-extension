#!/usr/bin/env bash
#SBATCH --job-name=mini_static_evidence
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=02:00:00

set -euo pipefail
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${STATIC_EVIDENCE_ROOT:?set new static-evidence root}"
: "${SCHEDULE_MANIFEST:?set frozen five-arm schedule manifest}"
: "${GPT_DATASET_SOURCE:?set pinned patched Megatron gpt_dataset.py}"
: "${GREEKMMLU_QUERIES:?set frozen GreekMMLU query JSONL}"
: "${BRIDGE_LEDGER_ROOT:?set complete receipt-bound bridge-ledger root}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}

[[ ! -e "$STATIC_EVIDENCE_ROOT" ]] || {
  echo "refusing to replace STATIC_EVIDENCE_ROOT=$STATIC_EVIDENCE_ROOT" >&2
  exit 2
}
for path in "$SCIENTIFIC_BUNDLE" "$SCHEDULE_MANIFEST" "$GPT_DATASET_SOURCE" \
  "$GREEKMMLU_QUERIES" "$BRIDGE_LEDGER_ROOT"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
mkdir -p "$STATIC_EVIDENCE_ROOT"

uenv run "$UENV_IMAGE" --view=default -- python3 \
  "$SCIENTIFIC_BUNDLE/dataset/audit_schedule_only_factor.py" \
  --schedule-manifest "$SCHEDULE_MANIFEST" \
  --output "$STATIC_EVIDENCE_ROOT/schedule_only_factor_audit.json"
uenv run "$UENV_IMAGE" --view=default -- python3 \
  "$SCIENTIFIC_BUNDLE/training/audit_goldfish_added_token_uniformity.py" \
  --pinned-gpt-dataset-source "$GPT_DATASET_SOURCE" \
  --output "$STATIC_EVIDENCE_ROOT/goldfish_added_token_uniformity.json"
uenv run "$UENV_IMAGE" --view=default -- python3 \
  "$SCIENTIFIC_BUNDLE/evaluation/build_greekmmlu_clean_subset.py" \
  --queries-jsonl "$GREEKMMLU_QUERIES" --ledger-root "$BRIDGE_LEDGER_ROOT" \
  --output-ids "$STATIC_EVIDENCE_ROOT/greekmmlu_clean_example_ids.txt" \
  --output-manifest "$STATIC_EVIDENCE_ROOT/greekmmlu_clean_subset_manifest.json"

/usr/bin/python3.11 - "$STATIC_EVIDENCE_ROOT" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
names = (
    "schedule_only_factor_audit.json",
    "goldfish_added_token_uniformity.json",
    "greekmmlu_clean_subset_manifest.json",
    "greekmmlu_clean_example_ids.txt",
)
rows = []
for name in names:
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing static evidence: {path}")
    if path.suffix == ".json" and json.loads(path.read_text())["status"] not in {"passed", "frozen"}:
        raise SystemExit(f"non-passing static evidence: {path}")
    rows.append({
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
payload = {
    "schema_version": "apertus_mini_static_prelaunch_evidence_v1",
    "status": "frozen",
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "scientific_bundle": os.environ["SCIENTIFIC_BUNDLE"],
    "artifacts": rows,
}
out = root / "static_prelaunch_evidence_receipt.json"
fd, tmp = tempfile.mkstemp(prefix=".static-evidence.", suffix=".partial", dir=root)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, out)
print(json.dumps(payload, sort_keys=True))
PY
