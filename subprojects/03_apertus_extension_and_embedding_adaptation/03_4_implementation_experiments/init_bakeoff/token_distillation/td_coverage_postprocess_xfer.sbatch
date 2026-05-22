#!/usr/bin/env bash
#SBATCH --job-name=td_coverage_postprocess
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.err

set -euo pipefail

COVERAGE_DIR="${COVERAGE_DIR:?COVERAGE_DIR is required}"
SMOKE_SIZE="${SMOKE_SIZE:-512}"
LAYER_PILOT_SIZE="${LAYER_PILOT_SIZE:-2048}"

if [ -n "${SCRIPT_DIR_OVERRIDE:-}" ]; then
    SCRIPT_DIR="$(cd "$SCRIPT_DIR_OVERRIDE" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/summarize_td_coverage.py" ]; then
    SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

COVERAGE_JSONL="$COVERAGE_DIR/td_coverage_prepass.jsonl"
SUMMARY_JSON="$COVERAGE_DIR/td_coverage_summary.json"
SNIPPETS_JSONL="$COVERAGE_DIR/td_snippet_index/snippets.jsonl"
REPORT_MD="$COVERAGE_DIR/TD_COVERAGE_SUMMARY.md"
PILOT_SELECTION_DIR="$COVERAGE_DIR/pilot_selection"
GATE_JSON="$COVERAGE_DIR/TD_GATE_DECISION.json"

echo "=== td_coverage_postprocess_xfer ==="
date -u
echo "host:         $(hostname)"
echo "COVERAGE_DIR: $COVERAGE_DIR"
echo "SCRIPT_DIR:   $SCRIPT_DIR"
echo

for required in "$COVERAGE_JSONL" "$SUMMARY_JSON" "$SNIPPETS_JSONL"; do
    if [ ! -s "$required" ]; then
        echo "ERROR: missing or empty artifact: $required" >&2
        exit 2
    fi
done

python3 "$SCRIPT_DIR/summarize_td_coverage.py" \
    --coverage-jsonl "$COVERAGE_JSONL" \
    --summary-json "$SUMMARY_JSON" \
    --output-md "$REPORT_MD"

RECOMMENDED="$(python3 - "$SUMMARY_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    print(json.load(fh).get("recommended_next_step", ""))
PY
)"
echo "recommended_next_step: $RECOMMENDED"

if [ "$RECOMMENDED" = "run_full_td_100" ] || [ "$RECOMMENDED" = "run_td_25_with_flagged_tail" ]; then
    python3 "$SCRIPT_DIR/select_td_pilot_tokens.py" \
        --coverage-jsonl "$COVERAGE_JSONL" \
        --summary-json "$SUMMARY_JSON" \
        --output-dir "$PILOT_SELECTION_DIR" \
        --smoke-size "$SMOKE_SIZE" \
        --layer-pilot-size "$LAYER_PILOT_SIZE"
    SELECTION_STATUS="created"
else
    SELECTION_STATUS="skipped"
fi

python3 - "$SUMMARY_JSON" "$GATE_JSON" "$REPORT_MD" "$PILOT_SELECTION_DIR" "$SELECTION_STATUS" <<'PY'
import json
import sys
summary_path, gate_path, report_md, pilot_dir, selection_status = sys.argv[1:6]
with open(summary_path, "r", encoding="utf-8") as fh:
    summary = json.load(fh)
decision = {
    "recommended_next_step": summary.get("recommended_next_step"),
    "selection_status": selection_status,
    "coverage_summary": summary_path,
    "coverage_report_md": report_md,
    "pilot_selection_dir": pilot_dir if selection_status == "created" else None,
    "enough_100_fraction": summary.get("enough_100_fraction"),
    "enough_25_fraction": summary.get("enough_25_fraction"),
    "low_lt25_count": summary.get("low_lt25_count"),
    "tokens_scanned": summary.get("tokens_scanned"),
    "target_extended_tokens": summary.get("target_extended_tokens"),
}
with open(gate_path, "w", encoding="utf-8") as fh:
    json.dump(decision, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print(json.dumps(decision, ensure_ascii=False, indent=2))
PY

echo
echo "=== done ==="
date -u
echo "report: $REPORT_MD"
echo "gate:   $GATE_JSON"
