#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizer_finalization/20260729T094000Z-poly512-1024}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$RUN_ROOT/submission"
mkdir -p "$STATE_DIR"

if [ -s "$STATE_DIR/asset_job.id" ] || [ -s "$STATE_DIR/probe_job.id" ]; then
    echo "refusing duplicate submission; existing job IDs are in $STATE_DIR" >&2
    exit 2
fi

asset_job="$(sbatch --parsable --export=ALL,RUN_ROOT="$RUN_ROOT" "$SCRIPT_DIR/prepare_probe_assets.sbatch")"
probe_job="$(sbatch --parsable --dependency="afterok:$asset_job" --export=ALL,RUN_ROOT="$RUN_ROOT" "$SCRIPT_DIR/run_cutoff_probe.sbatch")"

printf '%s\n' "$asset_job" > "$STATE_DIR/asset_job.id"
printf '%s\n' "$probe_job" > "$STATE_DIR/probe_job.id"
cat > "$STATE_DIR/jobs.tsv" <<EOF
stage	job_id	dependency
assets_and_coverage	$asset_job	none
model_probe	$probe_job	afterok:$asset_job
EOF

echo "asset_job=$asset_job"
echo "probe_job=$probe_job"
echo "monitor: squeue -j $asset_job,$probe_job -o '%.18i %.28j %.2t %.10M %.10l %D %R'"
