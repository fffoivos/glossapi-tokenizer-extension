#!/usr/bin/env bash
# Shared Phase-04 Clariden contract. Source from sbatch jobs after `set -euo pipefail`.

phase04_require_compute_partition() {
    local purpose=${1:-Phase-04 preprocessing}
    local partition=${SLURM_JOB_PARTITION:-}

    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: $purpose must run inside a Slurm allocation." >&2
        return 80
    fi
    case "$partition" in
        normal) ;;
        debug)
            if [[ "${PHASE04_ALLOW_DEBUG:-0}" != "1" ]]; then
                echo "ERROR: debug is smoke-only; set PHASE04_ALLOW_DEBUG=1 for a genuine bounded smoke." >&2
                return 81
            fi
            ;;
        xfer)
            echo "ERROR: xfer is for internal CSCS transfers, not $purpose." >&2
            return 82
            ;;
        low)
            echo "ERROR: low is currently unavailable to a0140's normal QoS; use normal." >&2
            return 83
            ;;
        *)
            echo "ERROR: unsupported Clariden partition '${partition:-unset}' for $purpose." >&2
            return 84
            ;;
    esac
}

phase04_require_cpu_request() {
    # Clariden normal nodes physically contain GH200 GPUs, but these corpus jobs
    # request no GPU/GRES.  AllocTRES still reports the four physical GPUs on an
    # exclusive node, so inspect ReqTRES rather than SLURM_JOB_GPUS.
    local job_record requested
    job_record=$(scontrol show job "${SLURM_JOB_ID:?SLURM_JOB_ID is unset}" -o)
    if [[ "$job_record" =~ ReqTRES=([^[:space:]]+) ]]; then
        requested=${BASH_REMATCH[1]}
    else
        echo "ERROR: cannot determine requested TRES for CPU-only safety gate." >&2
        return 93
    fi
    if [[ "$requested" == *"gres/gpu"* || "$requested" == *"gpu="* ]]; then
        echo "ERROR: Phase-04 corpus preparation explicitly requested GPU resources: $requested" >&2
        return 93
    fi
}

phase04_require_file() {
    local path=$1
    [[ -s "$path" ]] || { echo "ERROR: required non-empty file missing: $path" >&2; return 85; }
}

phase04_require_uenv_python() {
    local python_path=${1:-${RUNTIME_VENV:?RUNTIME_VENV is unset}/bin/python}
    command -v uenv >/dev/null || { echo "ERROR: uenv not found" >&2; return 88; }
    uenv run "${PHASE04_UENV:?PHASE04_UENV is unset}" --view=default -- \
        "$python_path" -c 'import platform; assert platform.machine() == "aarch64"' \
        >/dev/null || {
        echo "ERROR: Phase-04 Python is not runnable as aarch64 inside $PHASE04_UENV: $python_path" >&2
        return 89
    }
}

phase04_require_clean_git() {
    local repo=$1
    [[ "$(git -C "$repo" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] || {
        echo "ERROR: Phase-04 execution requires a real Git worktree: $repo" >&2
        return 86
    }
    if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=normal)" ]]; then
        echo "ERROR: Phase-04 execution requires a clean Git worktree at an exact commit." >&2
        git -C "$repo" status --short >&2
        return 87
    fi
}

phase04_require_expected_commit() {
    local repo=$1
    local expected=${PHASE04_EXPECTED_COMMIT:-}
    [[ -n "$expected" ]] || {
        echo "ERROR: PHASE04_EXPECTED_COMMIT is unset; submit through clariden/submit.sh." >&2
        return 91
    }
    local actual
    actual=$(git -C "$repo" rev-parse HEAD)
    [[ "$actual" == "$expected" ]] || {
        echo "ERROR: queued Phase-04 commit $expected differs from current checkout $actual." >&2
        return 92
    }
}

phase04_contract_python() {
    uenv run "${PHASE04_UENV:?PHASE04_UENV is unset}" --view=default -- \
        "${RUNTIME_VENV:?RUNTIME_VENV is unset}/bin/python" \
        "${PHASE04_CLARIDEN_DIR:?PHASE04_CLARIDEN_DIR is unset}/stage_contract.py" "$@"
}

phase04_init_pipeline_run() {
    [[ -n "${PIPELINE_RUN_ID:-}" ]] || {
        echo "ERROR: set PIPELINE_RUN_ID to a stable operator-chosen run ID." >&2
        return 94
    }
    [[ "$PIPELINE_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || {
        echo "ERROR: invalid PIPELINE_RUN_ID: $PIPELINE_RUN_ID" >&2
        return 95
    }
    local expected_root="$PIPELINE_RUNS_ROOT/$PIPELINE_RUN_ID"
    if [[ -n "${PIPELINE_RUN_ROOT:-}" && "$PIPELINE_RUN_ROOT" != "$expected_root" ]]; then
        echo "ERROR: PIPELINE_RUN_ROOT must equal $expected_root" >&2
        return 96
    fi
    PIPELINE_RUN_ROOT=$expected_root
    export PIPELINE_RUN_ROOT
    mkdir -p "$PIPELINE_RUNS_ROOT"
    phase04_contract_python init-run \
        --run-root "$PIPELINE_RUN_ROOT" \
        --run-id "$PIPELINE_RUN_ID" \
        --code-commit "${PHASE04_EXPECTED_COMMIT:?PHASE04_EXPECTED_COMMIT is unset}" \
        --sources "$SOURCE_CONFIG" \
        --cleaning-policy "$CLEANING_POLICY" \
        --eligibility-policy "$TRAINING_ELIGIBILITY_POLICY" \
        --source-license-adjudication "$SOURCE_LICENSE_ADJUDICATION" \
        --tokenizer-sha256 "$TOKENIZER_SHA256"
}

phase04_stage_begin() {
    local stage=$1
    [[ "$stage" =~ ^[0-9][0-9]-[a-z0-9-]+$ ]] || {
        echo "ERROR: invalid canonical stage name: $stage" >&2
        return 97
    }
    PHASE04_STAGE=$stage
    PHASE04_STAGE_DIR="$PIPELINE_RUN_ROOT/stages/$stage"
    export PHASE04_STAGE PHASE04_STAGE_DIR
    command -v flock >/dev/null || { echo "ERROR: flock is required for stage serialization" >&2; return 97; }
    mkdir -p "$PIPELINE_RUN_ROOT/stage_locks"
    exec {PHASE04_STAGE_LOCK_FD}>"$PIPELINE_RUN_ROOT/stage_locks/$stage.lock"
    flock -n "$PHASE04_STAGE_LOCK_FD" || {
        echo "ERROR: another job is already executing $PIPELINE_RUN_ID/$stage" >&2
        return 97
    }
    PHASE04_STAGE_ALREADY_DONE=0
    if [[ -s "$PHASE04_STAGE_DIR/stage_receipt.json" && ! -e "$PHASE04_STAGE_DIR/COMPLETED" ]]; then
        [[ "${RESUME_STAGE:-0}" == "1" ]] || {
            echo "ERROR: stage receipt exists without its completion marker; inspect and resume $stage." >&2
            return 98
        }
        phase04_contract_python repair-stage-marker \
            --stage-dir "$PHASE04_STAGE_DIR" \
            --stage "$stage" \
            --run-id "$PIPELINE_RUN_ID" \
            --code-commit "$PHASE04_EXPECTED_COMMIT"
    fi
    if [[ -e "$PHASE04_STAGE_DIR/stage_receipt.json" || -e "$PHASE04_STAGE_DIR/COMPLETED" ]]; then
        phase04_contract_python validate-stage \
            --stage-dir "$PHASE04_STAGE_DIR" \
            --stage "$stage" \
            --run-id "$PIPELINE_RUN_ID" \
            --code-commit "$PHASE04_EXPECTED_COMMIT"
        PHASE04_STAGE_ALREADY_DONE=1
        export PHASE04_STAGE_ALREADY_DONE
        echo "STAGE_ALREADY_COMPLETE=$PHASE04_STAGE_DIR"
        return 0
    fi
    if [[ -e "$PHASE04_STAGE_DIR" ]]; then
        [[ "${RESUME_STAGE:-0}" == "1" ]] || {
            echo "ERROR: incomplete stage exists: $PHASE04_STAGE_DIR" >&2
            echo "Use 'clariden/submit.sh resume $stage' after inspecting it." >&2
            return 98
        }
    else
        [[ "${RESUME_STAGE:-0}" != "1" ]] || {
            echo "ERROR: cannot resume absent stage: $PHASE04_STAGE_DIR" >&2
            return 99
        }
        mkdir -p "$PHASE04_STAGE_DIR"
    fi
    phase04_contract_python begin-stage \
        --stage-dir "$PHASE04_STAGE_DIR" \
        --stage "$stage" \
        --run-id "$PIPELINE_RUN_ID" \
        --code-commit "$PHASE04_EXPECTED_COMMIT" \
        --job-id "${SLURM_JOB_ID:?SLURM_JOB_ID is unset}"
}

phase04_stage_add_input() {
    local name=$1
    local path=$2
    phase04_require_file "$path"
    phase04_contract_python add-input \
        --stage-dir "$PHASE04_STAGE_DIR" --name "$name" --path "$path"
}

phase04_stage_bind_parameter() {
    local name=$1
    local value=$2
    phase04_contract_python bind-parameter \
        --stage-dir "$PHASE04_STAGE_DIR" --name "$name" --value "$value"
}

phase04_stage_require_upstream() {
    local stage=$1
    local receipt="$PIPELINE_RUN_ROOT/stages/$stage/stage_receipt.json"
    phase04_contract_python validate-stage \
        --stage-dir "$PIPELINE_RUN_ROOT/stages/$stage" \
        --stage "$stage" \
        --run-id "$PIPELINE_RUN_ID" \
        --code-commit "$PHASE04_EXPECTED_COMMIT"
    phase04_stage_add_input "upstream:$stage" "$receipt"
}

phase04_stage_finish() {
    local args=()
    local output
    for output in "$@"; do
        args+=(--required-output "$output")
    done
    phase04_contract_python finish-stage \
        --stage-dir "$PHASE04_STAGE_DIR" \
        --stage "$PHASE04_STAGE" \
        --run-id "$PIPELINE_RUN_ID" \
        --code-commit "$PHASE04_EXPECTED_COMMIT" \
        "${args[@]}"
    echo "STAGE_COMPLETE=$PHASE04_STAGE_DIR"
}

phase04_print_runtime() {
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "slurm_partition=${SLURM_JOB_PARTITION:-none}"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-none}"
    echo "slurm_gpus=${SLURM_GPUS:-none}"
    echo "git_commit=${CODE_COMMIT:-unknown}"
    echo "sources_sha256=${SOURCES_SHA256:-unknown}"
    echo "policy_sha256=${POLICY_SHA256:-unknown}"
    echo "tokenizer_sha256=${TOKENIZER_SHA256:-unknown}"
    echo "detector_binary_sha256=${DETECTOR_BINARY_SHA256:-unknown}"
    echo "input_receipt_sha256=${INPUT_RECEIPT_SHA256:-unknown}"
    echo "detector_build_receipt_sha256=${DETECTOR_BUILD_RECEIPT_SHA256:-unknown}"
}
