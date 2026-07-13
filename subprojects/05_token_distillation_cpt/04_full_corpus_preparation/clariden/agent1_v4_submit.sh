#!/usr/bin/env bash
# Dry-run-first Clariden submitter for v4 raw-review stages only.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PHASE04_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT=$(git -C "$PHASE04_DIR" rev-parse --show-toplevel)
source "$HERE/agent1_v4_paths.env"

usage() {
    cat >&2 <<'EOF'
usage: agent1_v4_submit.sh <bootstrap-runtime|freeze|sample|validate-responses|build-site|validate-human-gate|profile-fields|materialize-envelope|status> [sbatch args...]

Only CSCS CPU coordination stages are available.  ``sample`` reads the passed
acquisition receipt and produces the 18 x 20 raw-review packet.  It does not
invoke Codex.  Terra reviews run on the authenticated Mac's isolated runner;
``build-site`` accepts only its validated 360-response JSONL result.

Every invocation is a dry run until CONFIRM_LAUNCH=1.  A real submission also
requires CONFIRM_CLARIDEN_CPU_EXCEPTION=REQTRES_NO_GPU because normal Clariden
nodes can report physical GPUs in AllocTRES even when the request is CPU-only.
EOF
}

require_run_id() {
    [[ "${AGENT1_V4_RUN_ID:-}" =~ ^apertus-c3-prep-v4-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{7,40}$ ]] || {
        echo "ERROR: export AGENT1_V4_RUN_ID=apertus-c3-prep-v4-<UTC>-<shortsha>" >&2
        exit 2
    }
}

resources_for() {
    case "$1" in
        bootstrap-runtime) printf '%s\n' '--cpus-per-task=16 --mem=64G --time=01:00:00' ;;
        freeze|validate-responses|build-site|validate-human-gate) printf '%s\n' '--cpus-per-task=8 --mem=32G --time=02:00:00' ;;
        profile-fields) printf '%s\n' '--cpus-per-task=32 --mem=128G --time=12:00:00' ;;
        materialize-envelope) printf '%s\n' '--cpus-per-task=64 --mem=192G --time=12:00:00' ;;
        sample) printf '%s\n' '--cpus-per-task=32 --mem=128G --time=06:00:00' ;;
        *) return 1 ;;
    esac
}

action=${1:-}
shift || true
case "$action" in
    -h|--help|help|'') usage; exit 0 ;;
    status)
        require_run_id
        squeue -u "${USER:?USER is required}" -o '%.18i %.10T %.24j %.10P %.8M %.12R'
        exit 0
        ;;
esac
resources=$(resources_for "$action") || { usage; exit 2; }
require_run_id
if [[ "${CONFIRM_LAUNCH:-0}" != "1" ]]; then
    printf 'DRY RUN: AGENT1_V4_ACTION=%q AGENT1_V4_RUN_ID=%q sbatch %s %q' \
        "$action" "$AGENT1_V4_RUN_ID" "$resources" "$HERE/agent1_v4_stage.sbatch"
    for value in "$@"; do printf ' %q' "$value"; done
    printf '\n'
    exit 0
fi
[[ "${CONFIRM_CLARIDEN_CPU_EXCEPTION:-}" == "REQTRES_NO_GPU" ]] || {
    echo "ERROR: set CONFIRM_CLARIDEN_CPU_EXCEPTION=REQTRES_NO_GPU" >&2
    exit 3
}
if [[ "$action" == "freeze" || "$action" == "sample" || "$action" == "profile-fields" || "$action" == "materialize-envelope" ]]; then
    [[ -n "${AGENT1_V4_ACQUISITION_RECEIPT:-}" ]] || {
        echo "ERROR: AGENT1_V4_ACQUISITION_RECEIPT must name a passed receipt" >&2
        exit 3
    }
fi
if [[ "$action" == "validate-human-gate" ]]; then
    [[ -n "${AGENT1_V4_HUMAN_DECISIONS:-}" ]] || {
        echo "ERROR: AGENT1_V4_HUMAN_DECISIONS must name the downloaded decision bundle" >&2
        exit 3
    }
fi
if [[ "$action" == "materialize-envelope" ]]; then
    [[ -n "${AGENT1_V4_FIELD_MAPPING:-}" ]] || {
        echo "ERROR: AGENT1_V4_FIELD_MAPPING must name the approved mapping" >&2
        exit 3
    }
fi
export REPO_ROOT PHASE04_DIR AGENT1_V4_CLARIDEN_DIR="$HERE"
export AGENT1_V4_EXPECTED_COMMIT
AGENT1_V4_EXPECTED_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
export PHASE04_EXPECTED_COMMIT="$AGENT1_V4_EXPECTED_COMMIT"
read -r -a resources_array <<<"$resources"
sbatch --export="ALL,REPO_ROOT=$REPO_ROOT,PHASE04_DIR=$PHASE04_DIR,PHASE04_CLARIDEN_DIR=$HERE,PHASE04_EXPECTED_COMMIT=$PHASE04_EXPECTED_COMMIT,AGENT1_V4_ACTION=$action,AGENT1_V4_EXPECTED_COMMIT=$AGENT1_V4_EXPECTED_COMMIT" \
    "${resources_array[@]}" "$@" "$HERE/agent1_v4_stage.sbatch"
