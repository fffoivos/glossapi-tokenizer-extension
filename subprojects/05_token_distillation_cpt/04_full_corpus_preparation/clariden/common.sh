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
