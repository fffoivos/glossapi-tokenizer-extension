#!/usr/bin/env bash
set -euo pipefail

source /home/foivos/venvs/glossapi-corpus-clean/bin/activate

export PYTHONPATH=/home/foivos/Projects/glossapi-tokenizer-extension:${PYTHONPATH:-}
export TOKENIZER_REPO_ROOT=/home/foivos/Projects/glossapi-tokenizer-extension
export GLOSSAPI_WORK_ROOT=/home/foivos/data/glossapi_work
export APERTUS_BASE_TOKENIZER_DIR=/home/foivos/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_20260415
export GLOSSAPI_FAST_MIX_SUMMARY=1

RUN=/home/foivos/runs/wave4_20260429/production_strict_v1
REPO=/home/foivos/Projects/glossapi-tokenizer-extension
ORIG=/home/foivos/data/glossapi_work/hf_release_publish_working/data
CLEAN=/home/foivos/data/glossapi_work_wave4_20260429_strict_v1/canonical
DEDUP=/home/foivos/runs/wave2_20260426/dedup_run/builder_metadata

mkdir -p "$RUN" "$RUN/logs" "$CLEAN/data"
exec >> "$RUN/driver.log" 2>&1

echo "START $(date -Is) host=$(hostname)"

if [ ! -f "$RUN/reclean.done" ]; then
  echo "RECLEAN_START $(date -Is)"
  python "$REPO/subprojects/01_0_cleaning_iteration_and_thresholds/scripts/reclean_canonical_to_parquet.py" \
    --input-glob "$ORIG/*.parquet" \
    --output-root "$CLEAN/data" \
    --workers 64 \
    --batch-size 512 \
    --summary-jsonl "$RUN/reclean_summary.jsonl" \
    --score-missing-greek-badness-dataset HPLT/ell_Grek_ge8_no_mt_clean60 \
    --score-threads-per-worker 1 2>&1 | tee "$RUN/reclean.log"

  python - "$RUN" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in (run / "reclean_summary.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
errors = [row for row in rows if row.get("status") == "error"]
outputs = [row for row in rows if row.get("output_path")]
payload = {
    "ok": not errors,
    "tasks": len(rows),
    "errors": len(errors),
    "output_files": len(outputs),
    "rows_total": sum(int(row.get("rows_written") or 0) for row in rows),
    "rows_greek_badness_scored": sum(int(row.get("rows_greek_badness_scored") or 0) for row in rows),
    "rows_greek_badness_missing_after": sum(int(row.get("rows_greek_badness_missing_after") or 0) for row in rows),
    "rows_mojibake_missing_after": sum(int(row.get("rows_mojibake_missing_after") or 0) for row in rows),
    "chars_before_total": sum(int(row.get("chars_before_total") or 0) for row in rows),
    "chars_after_total": sum(int(row.get("chars_after_total") or 0) for row in rows),
}
(run / "reclean_validation.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({"event": "reclean_validation", **payload}, ensure_ascii=False), flush=True)
if errors:
    raise SystemExit(1)
PY

  touch "$RUN/reclean.done"
  echo "RECLEAN_DONE $(date -Is)"
else
  echo "RECLEAN_SKIP $(date -Is)"
fi

python "$REPO/subprojects/01_2_training_dataset_mix/scripts/wave3_orchestrate.py" \
  --input-root "$CLEAN" \
  --run-root "$RUN" \
  --dedup-metadata-root "$DEDUP" \
  --train-chars 100000000000 \
  --val-chars 50000000 \
  --test-chars 50000000 \
  --seed-salt wave4_20260429 \
  --max-workers 64 \
  --row-group-size 2048 \
  --target-extension-units 25600 \
  --delete-mixes-after-split

echo "ALL_DONE $(date -Is)"
