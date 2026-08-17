#!/usr/bin/env bash
# Submit exactly one receipt-gated remaining12 debug job, never a job graph.
# macOS /bin/bash 3.2 compatible by design.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  coordinate_remaining12_debug.sh preflight --remote clariden --wrapper-root PATH --assets-root PATH --commit SHA
  coordinate_remaining12_debug.sh submit-segment --remote clariden --wrapper-root PATH --assets-root PATH --output-root PATH --segment N

The script always performs an sbatch --test-only probe before submitting a
single debug job.  It does not enqueue successors: the next segment becomes
eligible only after the previous segment has written its receipt.
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
command=$1
shift
remote=
wrapper_root=
assets_root=
output_root=
commit=
segment=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) remote=$2; shift 2 ;;
    --wrapper-root) wrapper_root=$2; shift 2 ;;
    --assets-root) assets_root=$2; shift 2 ;;
    --output-root) output_root=$2; shift 2 ;;
    --commit) commit=$2; shift 2 ;;
    --segment) segment=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$remote" && -n "$wrapper_root" && -n "$assets_root" ]] || {
  echo "remote, wrapper-root and assets-root are required" >&2; exit 2;
}
[[ "$wrapper_root" == /iopsstor/* || "$wrapper_root" == /capstor/* ]] || { echo "unsafe wrapper root" >&2; exit 2; }
[[ "$assets_root" == /iopsstor/* || "$assets_root" == /capstor/* ]] || { echo "unsafe assets root" >&2; exit 2; }

case "$command" in
  preflight)
    [[ -n "$commit" ]] || { echo "commit is required for preflight" >&2; exit 2; }
    script="$wrapper_root/evaluation/freeze_and_preflight_remaining12.sbatch"
    remote_command="cd '$wrapper_root' && REMAINING12_WRAPPER_ROOT='$wrapper_root' REMAINING12_ASSETS_ROOT='$assets_root' REMAINING12_GIT_COMMIT='$commit' sbatch --test-only '$script'"
    ;;
  submit-segment)
    [[ -n "$output_root" && -n "$segment" ]] || { echo "output-root and segment are required" >&2; exit 2; }
    [[ "$output_root" == /iopsstor/* || "$output_root" == /capstor/* ]] || { echo "unsafe output root" >&2; exit 2; }
    [[ "$segment" =~ ^([0-9]|1[0-7])$ ]] || { echo "segment must be 0..17" >&2; exit 2; }
    script="$wrapper_root/evaluation/run_remaining12_native_segment.sbatch"
    remote_command="cd '$wrapper_root' && REMAINING12_WRAPPER_ROOT='$wrapper_root' REMAINING12_ASSETS_ROOT='$assets_root' REMAINING12_OUTPUT_ROOT='$output_root' REMAINING12_SEGMENT_INDEX='$segment' REMAINING12_EXPECTED_NNODES=2 sbatch --test-only '$script'"
    ;;
  *) usage >&2; exit 2 ;;
esac

echo "Running required test-only probe: $command" >&2
ssh "$remote" "$remote_command"

if [[ "$command" == preflight ]]; then
  remote_command="cd '$wrapper_root' && REMAINING12_WRAPPER_ROOT='$wrapper_root' REMAINING12_ASSETS_ROOT='$assets_root' REMAINING12_GIT_COMMIT='$commit' sbatch --parsable '$script'"
else
  remote_command="cd '$wrapper_root' && REMAINING12_WRAPPER_ROOT='$wrapper_root' REMAINING12_ASSETS_ROOT='$assets_root' REMAINING12_OUTPUT_ROOT='$output_root' REMAINING12_SEGMENT_INDEX='$segment' REMAINING12_EXPECTED_NNODES=2 sbatch --parsable '$script'"
fi
job_id=$(ssh "$remote" "$remote_command")
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "unexpected sbatch id: $job_id" >&2; exit 1; }

if [[ "$command" == preflight ]]; then
  expected_nodes=1
  expected_time=00:10:00
else
  expected_nodes=2
  expected_time=00:44:00
fi
ssh "$remote" "scontrol show job -o '$job_id'" | \
  awk -v nodes="$expected_nodes" -v time="$expected_time" '
    /Partition=debug/ && $0 ~ "NumNodes=" nodes && $0 ~ "TimeLimit=" time { ok=1 }
    END { exit(ok ? 0 : 1) }
  ' || { echo "submitted job resource audit failed: $job_id" >&2; exit 1; }
printf '%s\n' "$job_id"
