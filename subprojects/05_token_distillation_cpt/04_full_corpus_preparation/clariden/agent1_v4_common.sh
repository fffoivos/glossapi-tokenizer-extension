#!/usr/bin/env bash
# Shared preconditions for the isolated v4 raw-review lane.
set -euo pipefail

agent1_v4_init_paths() {
    [[ "$AGENT1_V4_RUN_ID" =~ ^apertus-c3-prep-v4-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{7,40}$ ]] || {
        echo "ERROR: invalid AGENT1_V4_RUN_ID" >&2
        return 2
    }
    local expected_root="$AGENT1_V4_RUNS_ROOT/$AGENT1_V4_RUN_ID"
    [[ -z "$AGENT1_V4_RUN_ROOT" || "$AGENT1_V4_RUN_ROOT" == "$expected_root" ]] || {
        echo "ERROR: AGENT1_V4_RUN_ROOT must equal $expected_root" >&2
        return 2
    }
    export AGENT1_V4_RUN_ROOT="$expected_root"
    export AGENT1_V4_RUNTIME_VENV="${AGENT1_V4_RUNTIME_VENV:-$AGENT1_V4_RUN_ROOT/00_frozen/runtime}"
    export REPO_ROOT PHASE04_DIR AGENT1_V4_CLARIDEN_DIR
    export AGENT1_V4_ACCOUNT AGENT1_V4_PARTITION AGENT1_V4_UENV
    export AGENT1_V4_SOURCES AGENT1_V4_ROSTER AGENT1_V4_POLICY AGENT1_V4_LICENSE_ADJUDICATION
    export AGENT1_V4_NANOCHAT_ROSTER AGENT1_V4_PROMPT AGENT1_V4_RESPONSE_SCHEMA AGENT1_V4_ENVIRONMENT_LOCK
    export AGENT1_V4_FREEZER AGENT1_V4_PACKET_EXPORTER AGENT1_V4_RESPONSE_VALIDATOR AGENT1_V4_SITE_BUILDER AGENT1_V4_HUMAN_GATE AGENT1_V4_FIELD_PROFILER AGENT1_V4_ENVELOPE_MATERIALIZER AGENT1_V4_GLOSSAPI_COMMIT
    export AGENT1_V4_GREEKMMLU_REGISTRY AGENT1_V4_ACQUISITION_RECEIPT AGENT1_V4_HUMAN_DECISIONS AGENT1_V4_FIELD_MAPPING AGENT1_V4_RUNTIME_VENV
}

agent1_v4_require_clean_commit() {
    [[ "$(git -C "$REPO_ROOT" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] || {
        echo "ERROR: REPO_ROOT is not a Git worktree: $REPO_ROOT" >&2
        return 5
    }
    [[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]] || {
        echo "ERROR: v4 requires a clean remote worktree" >&2
        git -C "$REPO_ROOT" status --short >&2
        return 5
    }
    [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "${AGENT1_V4_EXPECTED_COMMIT:?expected commit unset}" ]] || {
        echo "ERROR: remote worktree commit differs from submitted v4 commit" >&2
        return 5
    }
}

agent1_v4_require_compute_cpu() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: v4 work must run inside Slurm" >&2; return 3; }
    [[ "${SLURM_JOB_PARTITION:-}" == "$AGENT1_V4_PARTITION" ]] || {
        echo "ERROR: unsupported v4 partition ${SLURM_JOB_PARTITION:-unset}" >&2
        return 3
    }
    local record requested
    record=$(scontrol show job "$SLURM_JOB_ID" -o)
    [[ "$record" =~ ReqTRES=([^[:space:]]+) ]] || { echo "ERROR: cannot inspect ReqTRES" >&2; return 3; }
    requested=${BASH_REMATCH[1]}
    [[ "$requested" != *"gres/gpu"* && "$requested" != *"gpu="* ]] || {
        echo "ERROR: v4 raw-review CPU job requested GPU resources: $requested" >&2
        return 3
    }
    printf '%s\n' "$record" > "${AGENT1_V4_ALLOCATION_EVIDENCE:?allocation evidence path unset}"
}

agent1_v4_mask_gpu_visibility() {
    export CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES=''
}

agent1_v4_require_runtime() {
    [[ -x "$AGENT1_V4_RUNTIME_VENV/bin/python" || -L "$AGENT1_V4_RUNTIME_VENV/bin/python" ]] || {
        echo "ERROR: v4 runtime is missing: $AGENT1_V4_RUNTIME_VENV" >&2
        return 6
    }
    uenv run "$AGENT1_V4_UENV" --view=default -- \
        "$AGENT1_V4_RUNTIME_VENV/bin/python" -c 'import platform, pyarrow; assert platform.machine() == "aarch64"' >/dev/null
}

agent1_v4_attempt_dir() {
    local stage=$1
    export AGENT1_V4_ATTEMPT_DIR="$AGENT1_V4_RUN_ROOT/receipts/$stage/attempts/${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
    [[ ! -e "$AGENT1_V4_ATTEMPT_DIR" ]] || { echo "ERROR: immutable attempt exists: $AGENT1_V4_ATTEMPT_DIR" >&2; return 7; }
    mkdir -p "$AGENT1_V4_ATTEMPT_DIR"
    export AGENT1_V4_ALLOCATION_EVIDENCE="$AGENT1_V4_ATTEMPT_DIR/slurm_allocation.txt"
}

agent1_v4_run_python() {
    uenv run "$AGENT1_V4_UENV" --view=default -- "$AGENT1_V4_RUNTIME_VENV/bin/python" "$@"
}
