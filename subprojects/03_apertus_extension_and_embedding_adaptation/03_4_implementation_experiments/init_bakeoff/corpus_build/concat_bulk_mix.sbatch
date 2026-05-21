#!/usr/bin/env bash
#SBATCH --job-name=concat_bulk_mix
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/concat_bulk_mix_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/concat_bulk_mix_%j.err
#
# Concatenate sharded mix_builder_full.sbatch outputs into the canonical
# bulk_mix.jsonl consumed by bakeoff_training/preprocess_data.sbatch.

set -euo pipefail

STAGE_ROOT=/iopsstor/scratch/cscs/fffoivos/cpt_corpus
SHARD_COUNT="${SHARD_COUNT:-7}"
SHARD_PREFIX="${SHARD_PREFIX:-$STAGE_ROOT/bulk_mix_part_}"
OUTPUT="${OUTPUT:-$STAGE_ROOT/bulk_mix.jsonl}"
TMP_OUTPUT="$OUTPUT.tmp.$$"
MANIFEST="${MANIFEST:-${OUTPUT%.jsonl}.manifest.json}"

echo "=== concat_bulk_mix ==="
date -u
echo "SHARD_COUNT: $SHARD_COUNT"
echo "SHARD_PREFIX: $SHARD_PREFIX"
echo "OUTPUT: $OUTPUT"

rm -f "$TMP_OUTPUT"
shards=()
for i in $(seq 0 $((SHARD_COUNT - 1))); do
    shard_id="$(printf "%02d" "$i")"
    shard="${SHARD_PREFIX}${shard_id}.jsonl"
    manifest="${shard%.jsonl}.manifest.json"
    if [ ! -s "$shard" ]; then
        echo "ERROR: missing or empty shard $shard" >&2
        exit 2
    fi
    if [ ! -s "$manifest" ]; then
        echo "ERROR: missing or empty manifest $manifest" >&2
        exit 2
    fi
    shards+=("$shard")
done

cat "${shards[@]}" > "$TMP_OUTPUT"
mv "$TMP_OUTPUT" "$OUTPUT"

python3 - "$OUTPUT" "$MANIFEST" "${shards[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
manifest = Path(sys.argv[2])
shards = [Path(p) for p in sys.argv[3:]]

total_tokens = 0
total_rows = 0
per_source = {}
per_bucket = {}
shard_manifests = []
schedulers = set()
for shard in shards:
    mp = shard.with_suffix(".manifest.json")
    data = json.loads(mp.read_text())
    shard_manifests.append(str(mp))
    if data.get("scheduler"):
        schedulers.add(data["scheduler"])
    total_tokens += int(data["actual_tokens"])
    total_rows += int(data["actual_rows"])
    for name, stats in data["per_source"].items():
        out = per_source.setdefault(name, {"rows": 0, "tokens": 0, "target_weight": stats["target_weight"]})
        out["rows"] += int(stats["rows"])
        out["tokens"] += int(stats["tokens"])
    for name, stats in data.get("per_bucket", {}).items():
        out = per_bucket.setdefault(name, {"rows": 0, "tokens": 0, "target_weight": stats["target_weight"]})
        out["rows"] += int(stats["rows"])
        out["tokens"] += int(stats["tokens"])

for stats in per_source.values():
    stats["effective_weight"] = stats["tokens"] / max(total_tokens, 1)
for stats in per_bucket.values():
    stats["effective_weight"] = stats["tokens"] / max(total_tokens, 1)

manifest.write_text(json.dumps({
    "output": str(output),
    "shards": [str(p) for p in shards],
    "shard_manifests": shard_manifests,
    "schedulers": sorted(schedulers),
    "actual_tokens": total_tokens,
    "actual_rows": total_rows,
    "size_bytes": output.stat().st_size,
    "per_bucket": per_bucket,
    "per_source": per_source,
}, indent=2, ensure_ascii=False))
PY

echo
echo "=== concatenated mix ==="
ls -lh "$OUTPUT" "$MANIFEST"
wc -l "$OUTPUT"
python3 - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
print("actual_tokens", m["actual_tokens"])
print("actual_rows", m["actual_rows"])
if m.get("per_bucket"):
    print("per_bucket")
    for name, stats in m["per_bucket"].items():
        print(f"{name}\t{stats['effective_weight']:.4f}\t{stats['target_weight']:.4f}\t{stats['tokens']}")
print("per_source")
for name, stats in m["per_source"].items():
    print(f"{name}\t{stats['effective_weight']:.4f}\t{stats['target_weight']:.4f}\t{stats['tokens']}")
PY
