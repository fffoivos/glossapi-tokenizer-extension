#!/usr/bin/env bash
#SBATCH --job-name=agent1_v3
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
# Generic v3 dispatcher.  Resource overrides are supplied only by the v3
# submit wrapper and all real work remains CPU-only.
set -euo pipefail

HERE=${AGENT1_V3_CLARIDEN_DIR:?submit through agent1_v3_submit.sh}
source "$HERE/agent1_v3_paths.env"
source "$HERE/agent1_v3_common.sh"

agent1_v3_init_paths

action=${AGENT1_V3_ACTION:?AGENT1_V3_ACTION is required}

# Stage handlers create their own contract-first, job-unique attempt through
# agent1_v3_begin_stage.  CPU preflight happens before that point, so retain
# its Slurm allocation evidence only on node-local scratch; it must not create
# a misleading Phase 0 attempt or an unreceipted canonical stage directory.
agent1_v3_stage_preflight() {
    local evidence_root
    evidence_root=${SLURM_TMPDIR:-/tmp}
    mkdir -p "$evidence_root"
    export AGENT1_V3_ALLOCATION_EVIDENCE="$evidence_root/agent1-v3-${SLURM_JOB_ID:?SLURM_JOB_ID is required}.allocation.txt"
    agent1_v3_require_compute_cpu
    agent1_v3_mask_gpu_visibility
    agent1_v3_require_clean_commit
    agent1_v3_require_runtime
}

case "$action" in
    bootstrap-runtime)
        agent1_v3_phase0_attempt
        agent1_v3_require_compute_cpu
        agent1_v3_mask_gpu_visibility
        agent1_v3_require_clean_commit
        exec bash "$HERE/agent1_v3_bootstrap_runtime.sh"
        ;;
    build-quality-runtime)
        agent1_v3_phase0_attempt
        agent1_v3_require_compute_cpu
        agent1_v3_mask_gpu_visibility
        agent1_v3_require_clean_commit
        agent1_v3_require_runtime
        exec bash "$HERE/agent1_v3_build_glossapi_runtime.sh"
        ;;
    acquire-hf-existing)
        agent1_v3_phase0_attempt
        agent1_v3_require_compute_cpu
        agent1_v3_mask_gpu_visibility
        agent1_v3_require_clean_commit
        agent1_v3_require_runtime
        exec bash "$HERE/agent1_v3_acquire_hf_existing.sh"
        ;;
    acquire-mdc)
        agent1_v3_phase0_attempt
        agent1_v3_require_compute_cpu
        agent1_v3_mask_gpu_visibility
        agent1_v3_require_clean_commit
        agent1_v3_require_runtime
        exec bash "$HERE/agent1_v3_acquire_mdc.sh"
        ;;
    merge-acquisition)
        agent1_v3_phase0_attempt
        agent1_v3_require_compute_cpu
        agent1_v3_mask_gpu_visibility
        agent1_v3_require_clean_commit
        agent1_v3_require_runtime
        exec bash "$HERE/agent1_v3_merge_acquisition.sh"
        ;;
    freeze-run)
        agent1_v3_phase0_attempt
        agent1_v3_require_compute_cpu
        agent1_v3_mask_gpu_visibility
        agent1_v3_require_clean_commit
        agent1_v3_require_runtime
        exec bash "$HERE/agent1_v3_freeze_run.sh"
        ;;
    validate-contract)
        agent1_v3_phase0_attempt
        agent1_v3_require_compute_cpu
        agent1_v3_mask_gpu_visibility
        agent1_v3_require_clean_commit
        agent1_v3_require_runtime
        exec uenv run "$AGENT1_V3_UENV" --view=default -- \
            "$AGENT1_V3_RUNTIME_VENV/bin/python" "$AGENT1_V3_CONTRACT_SCRIPT" validate-run \
            --run-root "$AGENT1_V3_RUN_ROOT" --run-id "$AGENT1_V3_RUN_ID"
        ;;
    normalize|lineage|review-packet)
        agent1_v3_stage_preflight
        exec bash "$HERE/agent1_v3_pre_review.sh" "$action"
        ;;
    quality-review-evidence)
        agent1_v3_stage_preflight
        exec bash "$HERE/agent1_v3_quality_review_evidence.sh"
        ;;
    admission|dedup|greekmmlu-freeze|decontamination|anonymization-sanitization|prestructural-freeze)
        agent1_v3_stage_preflight
        exec bash "$HERE/agent1_v3_post_admission.sh" "$action"
        ;;
    *)
        echo "ERROR: unsupported Agent 1 v3 action: $action" >&2
        exit 64
        ;;
esac
