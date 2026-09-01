#!/usr/bin/env bash
set -euo pipefail

# Non-submitting scheduler-capacity probe. Run on a Clariden login node.
# Positional arguments override the default five-arm campaign sizes.

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is unavailable; run this script on Clariden" >&2
  exit 2
fi

account="${SLURM_ACCOUNT:-a0140}"
partition="${SLURM_PARTITION:-normal}"
time_limit="${SLURM_TIME_LIMIT:-12:00:00}"

if (( $# > 0 )); then
  node_counts=("$@")
else
  node_counts=(20 40 80 160)
fi

date --iso-8601=seconds
scontrol show partition "$partition" \
  | tr ' ' '\n' \
  | grep -E '^(AllowQos|MaxNodes|MaxTime|OverSubscribe|State)=' \
  || true
sinfo -p "$partition" -h -o '%t|%D' \
  | awk -F'|' '{count[$1]+=$2} END {for (state in count) print state "|" count[state]}' \
  | sort

for nodes in "${node_counts[@]}"; do
  if ! [[ "$nodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid node count: $nodes" >&2
    exit 2
  fi
  echo "TEST_ONLY nodes=$nodes time=$time_limit"
  sbatch --test-only \
    --account="$account" \
    --partition="$partition" \
    --nodes="$nodes" \
    --ntasks-per-node=1 \
    --gpus-per-node=4 \
    --time="$time_limit" \
    --wrap='true'
done

