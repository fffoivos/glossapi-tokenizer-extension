#!/usr/bin/env bash
# Shared invariant checks for Agent 1's independent v3 Clariden lane.
set -euo pipefail

agent1_v3_init_paths() {
    [[ -n "${AGENT1_V3_RUN_ID:-}" ]] || {
        echo "ERROR: AGENT1_V3_RUN_ID is required" >&2
        return 2
    }
    [[ "$AGENT1_V3_RUN_ID" =~ ^agent1-full-corpus-v3-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{7,40}$ ]] || {
        echo "ERROR: invalid AGENT1_V3_RUN_ID: $AGENT1_V3_RUN_ID" >&2
        return 2
    }
    local expected_run_root="$AGENT1_V3_RUNS_ROOT/$AGENT1_V3_RUN_ID"
    local expected_data_root="$AGENT1_V3_DATA_ROOT_BASE/$AGENT1_V3_RUN_ID"
    [[ -z "${AGENT1_V3_RUN_ROOT:-}" || "$AGENT1_V3_RUN_ROOT" == "$expected_run_root" ]] || {
        echo "ERROR: AGENT1_V3_RUN_ROOT must equal $expected_run_root" >&2
        return 2
    }
    [[ -z "${AGENT1_V3_DATA_ROOT:-}" || "$AGENT1_V3_DATA_ROOT" == "$expected_data_root" ]] || {
        echo "ERROR: AGENT1_V3_DATA_ROOT must equal $expected_data_root" >&2
        return 2
    }
    export AGENT1_V3_RUN_ROOT="$expected_run_root"
    export AGENT1_V3_DATA_ROOT="$expected_data_root"
    export AGENT1_V3_RUNTIME_VENV="${AGENT1_V3_RUNTIME_VENV:-$AGENT1_V3_RUN_ROOT/phase0/runtime}"

    # The dispatcher `exec`s the action scripts.  Keep every non-secret path
    # and runtime selector they consume in that child environment; sourcing
    # paths.env alone only creates shell-local variables.
    export REPO_ROOT PHASE04_DIR AGENT1_V3_CLARIDEN_DIR AGENT1_V3_CONTRACT_SCRIPT
    export AGENT1_V3_ACCOUNT AGENT1_V3_PARTITION AGENT1_V3_UENV
    export AGENT1_V3_SOURCE_CONFIG AGENT1_V3_SOURCE_ALIASES
    export AGENT1_V3_CANDIDATE_ROSTER AGENT1_V3_POLICY AGENT1_V3_REVIEW_POLICY AGENT1_V3_REVIEW_PROMPT AGENT1_V3_REVIEW_RESPONSE_SCHEMA
    export AGENT1_V3_LICENSE_ADJUDICATION AGENT1_V3_ELIGIBILITY_POLICY
    export AGENT1_V3_HF_EXISTING_DESTINATION AGENT1_V3_RAW_COMMON_ROOT
    export AGENT1_V3_TOKENIZER_REVISION AGENT1_V3_TOKENIZER_JSON
    export AGENT1_V3_GLOSSAPI_COMMIT AGENT1_V3_GLOSSAPI_ROOT
}

agent1_v3_require_compute_cpu() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: v3 work must run inside Slurm" >&2; return 3; }
    case "${SLURM_JOB_PARTITION:-}" in
        normal) ;;
        debug)
            [[ "${AGENT1_V3_ALLOW_DEBUG:-0}" == "1" ]] || {
                echo "ERROR: debug is permitted only for explicitly bounded smoke work" >&2
                return 3
            }
            ;;
        *) echo "ERROR: unsupported CPU partition ${SLURM_JOB_PARTITION:-unset}" >&2; return 3 ;;
    esac
    local record requested
    record=$(scontrol show job "$SLURM_JOB_ID" -o)
    [[ "$record" =~ ReqTRES=([^[:space:]]+) ]] || {
        echo "ERROR: cannot read ReqTRES for CPU safety gate" >&2
        return 3
    }
    requested=${BASH_REMATCH[1]}
    [[ "$requested" != *"gres/gpu"* && "$requested" != *"gpu="* ]] || {
        echo "ERROR: Agent 1 v3 job requested a GPU: $requested" >&2
        return 3
    }
    # Clariden normal nodes are exclusive GH200 nodes.  AllocTRES reports
    # physical gpu:4 despite a CPU-only ReqTRES.  Record it as evidence rather
    # than making a predicate no Clariden CPU job can satisfy.
    printf '%s\n' "$record" > "${AGENT1_V3_ALLOCATION_EVIDENCE:?allocation evidence path is unset}"
}

agent1_v3_mask_gpu_visibility() {
    export CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES=''
    [[ -z "$CUDA_VISIBLE_DEVICES" && -z "$ROCR_VISIBLE_DEVICES" && -z "$HIP_VISIBLE_DEVICES" ]] || {
        echo "ERROR: GPU visibility could not be cleared" >&2
        return 4
    }
}

agent1_v3_require_clean_commit() {
    [[ "$(git -C "$REPO_ROOT" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] || {
        echo "ERROR: REPO_ROOT is not a Git worktree: $REPO_ROOT" >&2
        return 5
    }
    [[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]] || {
        echo "ERROR: Agent 1 v3 requires a clean exact remote worktree" >&2
        git -C "$REPO_ROOT" status --short >&2
        return 5
    }
    [[ -n "${AGENT1_V3_EXPECTED_COMMIT:-}" ]] || {
        echo "ERROR: AGENT1_V3_EXPECTED_COMMIT is unset" >&2
        return 5
    }
    [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$AGENT1_V3_EXPECTED_COMMIT" ]] || {
        echo "ERROR: remote worktree commit differs from submitted v3 commit" >&2
        return 5
    }
}

agent1_v3_require_runtime() {
    # A uenv-created venv deliberately links `bin/python` to the interpreter
    # inside the container image.  From the host that link can look broken;
    # validate it in uenv below instead of rejecting a valid container venv.
    [[ -L "$AGENT1_V3_RUNTIME_VENV/bin/python" || -x "$AGENT1_V3_RUNTIME_VENV/bin/python" ]] || {
        echo "ERROR: v3 runtime is missing: $AGENT1_V3_RUNTIME_VENV" >&2
        return 6
    }
    command -v uenv >/dev/null || { echo "ERROR: uenv is unavailable" >&2; return 6; }
    uenv run "$AGENT1_V3_UENV" --view=default -- \
        "$AGENT1_V3_RUNTIME_VENV/bin/python" -c 'import platform; assert platform.machine() == "aarch64"' \
        >/dev/null
}

agent1_v3_phase0_attempt() {
    export AGENT1_V3_ATTEMPT_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
    export AGENT1_V3_PHASE0_DIR="$AGENT1_V3_RUN_ROOT/phase0"
    export AGENT1_V3_ATTEMPT_DIR="$AGENT1_V3_PHASE0_DIR/attempts/$AGENT1_V3_ATTEMPT_ID"
    [[ ! -e "$AGENT1_V3_ATTEMPT_DIR" ]] || {
        echo "ERROR: phase0 attempt already exists: $AGENT1_V3_ATTEMPT_DIR" >&2
        return 7
    }
    mkdir -p "$AGENT1_V3_ATTEMPT_DIR"
    export AGENT1_V3_ALLOCATION_EVIDENCE="$AGENT1_V3_ATTEMPT_DIR/slurm_allocation.txt"
}

agent1_v3_begin_stage() {
    local stage=$1
    shift
    export AGENT1_V3_ATTEMPT_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
    export AGENT1_V3_ALLOCATION_EVIDENCE="$AGENT1_V3_RUN_ROOT/stages/$stage/attempts/$AGENT1_V3_ATTEMPT_ID/slurm_allocation.txt"
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$AGENT1_V3_CONTRACT_SCRIPT" begin-stage \
        --run-root "$AGENT1_V3_RUN_ROOT" --run-id "$AGENT1_V3_RUN_ID" \
        --stage "$stage" --attempt-id "$AGENT1_V3_ATTEMPT_ID" "$@"
    export AGENT1_V3_STAGE_DIR="$AGENT1_V3_RUN_ROOT/stages/$stage"
    export AGENT1_V3_ATTEMPT_DIR="$AGENT1_V3_STAGE_DIR/attempts/$AGENT1_V3_ATTEMPT_ID"
}

agent1_v3_finish_stage() {
    local stage=$1
    shift
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$AGENT1_V3_CONTRACT_SCRIPT" finish-stage \
        --run-root "$AGENT1_V3_RUN_ROOT" --run-id "$AGENT1_V3_RUN_ID" \
        --stage "$stage" --attempt-id "$AGENT1_V3_ATTEMPT_ID" "$@"
}
