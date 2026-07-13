#!/usr/bin/env bash
#SBATCH --job-name=agent1_v4
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through agent1_v4_submit.sh}
source "$HERE/agent1_v4_paths.env"
source "$HERE/agent1_v4_common.sh"
agent1_v4_init_paths
action=${AGENT1_V4_ACTION:?AGENT1_V4_ACTION is required}
export AGENT1_V4_EXPECTED_COMMIT="${AGENT1_V4_EXPECTED_COMMIT:-${PHASE04_EXPECTED_COMMIT:?expected commit unset}}"

prepare() {
    local stage=$1
    mkdir -p "$AGENT1_V4_RUN_ROOT"
    agent1_v4_attempt_dir "$stage"
    agent1_v4_require_compute_cpu
    agent1_v4_mask_gpu_visibility
    agent1_v4_require_clean_commit
}

case "$action" in
    bootstrap-runtime)
        prepare "00_bootstrap"
        exec bash "$HERE/agent1_v4_bootstrap_runtime.sh"
        ;;
    freeze)
        prepare "00_freeze"
        agent1_v4_require_runtime
        [[ -s "$AGENT1_V4_ACQUISITION_RECEIPT" ]] || { echo "ERROR: passed acquisition receipt is required" >&2; exit 8; }
        frozen="$AGENT1_V4_RUN_ROOT/00_frozen"
        mkdir -p "$frozen"
        seed="$frozen/sampling_seed.txt"
        [[ ! -e "$frozen/00_freeze_receipt.json" ]] || { echo "ERROR: immutable Stage-00 receipt already exists" >&2; exit 8; }
        if [[ ! -e "$seed" ]]; then
            umask 077
            od -An -N32 -tx1 /dev/urandom | tr -d ' \n' > "$seed"
            chmod 600 "$seed"
        fi
        [[ -f "$seed" && ! -L "$seed" && "$(stat -c '%a' "$seed")" == "600" ]] || { echo "ERROR: frozen seed must be a 0600 regular file" >&2; exit 8; }
        agent1_v4_run_python "$AGENT1_V4_FREEZER" \
            --sources "$AGENT1_V4_SOURCES" --acquisition-receipt "$AGENT1_V4_ACQUISITION_RECEIPT" \
            --roster "$AGENT1_V4_ROSTER" --policy "$AGENT1_V4_POLICY" \
            --license-adjudication "$AGENT1_V4_LICENSE_ADJUDICATION" \
            --nanochat-roster "$AGENT1_V4_NANOCHAT_ROSTER" --environment-lock "$AGENT1_V4_ENVIRONMENT_LOCK" \
            --greekmmlu-registry "$AGENT1_V4_GREEKMMLU_REGISTRY" --prompt "$AGENT1_V4_PROMPT" \
            --response-schema "$AGENT1_V4_RESPONSE_SCHEMA" --sampling-seed-file "$seed" \
            --code-commit "$AGENT1_V4_EXPECTED_COMMIT" --glossapi-commit "$AGENT1_V4_GLOSSAPI_COMMIT" \
            --output "$frozen/00_freeze_receipt.json"
        ;;
    sample)
        prepare "05_sample"
        agent1_v4_require_runtime
        frozen="$AGENT1_V4_RUN_ROOT/00_frozen"
        [[ -s "$frozen/00_freeze_receipt.json" && -s "$frozen/sampling_seed.txt" ]] || { echo "ERROR: passed Stage-00 receipt and seed are required" >&2; exit 8; }
        [[ -s "$AGENT1_V4_ACQUISITION_RECEIPT" ]] || { echo "ERROR: passed acquisition receipt is required" >&2; exit 8; }
        agent1_v4_run_python "$AGENT1_V4_PACKET_EXPORTER" \
            --sources "$AGENT1_V4_SOURCES" --acquisition-receipt "$AGENT1_V4_ACQUISITION_RECEIPT" \
            --roster "$AGENT1_V4_ROSTER" --policy "$AGENT1_V4_POLICY" \
            --seed "$(<"$frozen/sampling_seed.txt")" --prompt "$AGENT1_V4_PROMPT" \
            --response-schema "$AGENT1_V4_RESPONSE_SCHEMA" --code-commit "$AGENT1_V4_EXPECTED_COMMIT" \
            --output "$AGENT1_V4_RUN_ROOT/05_raw_review_packet"
        ;;
    validate-responses)
        prepare "10_validate_responses"
        agent1_v4_require_runtime
        packet="$AGENT1_V4_RUN_ROOT/05_raw_review_packet"
        responses="$AGENT1_V4_RUN_ROOT/10_terra_reviews/responses.jsonl"
        [[ -s "$packet/packet_manifest.json" && -s "$responses" ]] || { echo "ERROR: closed packet and returned response JSONL are required" >&2; exit 8; }
        agent1_v4_run_python "$AGENT1_V4_RESPONSE_VALIDATOR" \
            --packet-root "$packet" --packet-manifest "$packet/packet_manifest.json" \
            --responses "$responses" --output "$AGENT1_V4_RUN_ROOT/10_terra_reviews/response_validation.json"
        ;;
    build-site)
        prepare "15_build_site"
        agent1_v4_require_runtime
        packet="$AGENT1_V4_RUN_ROOT/05_raw_review_packet"
        responses="$AGENT1_V4_RUN_ROOT/10_terra_reviews/responses.jsonl"
        [[ -s "$packet/packet_manifest.json" && -s "$responses" && -s "$AGENT1_V4_RUN_ROOT/10_terra_reviews/response_validation.json" ]] || { echo "ERROR: validated packet/response evidence is required" >&2; exit 8; }
        site_root="$AGENT1_V4_RUN_ROOT/15_review_site"
        mkdir -p "$site_root"
        secret="$site_root/site_secret.txt"
        [[ ! -e "$site_root/site" ]] || { echo "ERROR: immutable review site already exists" >&2; exit 8; }
        if [[ ! -e "$secret" ]]; then
            umask 077
            od -An -N32 -tx1 /dev/urandom | tr -d ' \n' > "$secret"
            chmod 600 "$secret"
        fi
        [[ -f "$secret" && ! -L "$secret" && "$(stat -c '%a' "$secret")" == "600" ]] || { echo "ERROR: private site secret must be a 0600 regular file" >&2; exit 8; }
        agent1_v4_run_python "$AGENT1_V4_SITE_BUILDER" build \
            --packet-root "$packet" --packet-manifest "$packet/packet_manifest.json" \
            --requests "$packet/requests.jsonl" --responses "$responses" \
            --site-secret "$(<"$secret")" --output-dir "$site_root/site"
        ;;
    validate-human-gate)
        prepare "20_human_gate"
        agent1_v4_require_runtime
        packet="$AGENT1_V4_RUN_ROOT/05_raw_review_packet"
        [[ -s "$packet/packet_manifest.json" && -s "$AGENT1_V4_RUN_ROOT/00_frozen/00_freeze_receipt.json" ]] || { echo "ERROR: frozen packet evidence is required" >&2; exit 8; }
        [[ -s "$AGENT1_V4_HUMAN_DECISIONS" ]] || { echo "ERROR: AGENT1_V4_HUMAN_DECISIONS must name the returned human decision bundle" >&2; exit 8; }
        agent1_v4_run_python "$AGENT1_V4_HUMAN_GATE" \
            --packet-root "$packet" --packet-manifest "$packet/packet_manifest.json" \
            --freeze-receipt "$AGENT1_V4_RUN_ROOT/00_frozen/00_freeze_receipt.json" \
            --decisions "$AGENT1_V4_HUMAN_DECISIONS" --output "$AGENT1_V4_RUN_ROOT/20_human_gate_receipt.json"
        ;;
    profile-fields)
        prepare "30_field_profile"
        agent1_v4_require_runtime
        [[ -s "$AGENT1_V4_ACQUISITION_RECEIPT" && -s "$AGENT1_V4_RUN_ROOT/20_human_gate_receipt.json" ]] || { echo "ERROR: passed acquisition and human-gate receipts are required" >&2; exit 8; }
        agent1_v4_run_python "$AGENT1_V4_FIELD_PROFILER" \
            --sources "$AGENT1_V4_SOURCES" --acquisition-receipt "$AGENT1_V4_ACQUISITION_RECEIPT" \
            --human-gate-receipt "$AGENT1_V4_RUN_ROOT/20_human_gate_receipt.json" \
            --output "$AGENT1_V4_RUN_ROOT/30_field_profile.json"
        ;;
    materialize-envelope)
        prepare "30_envelope"
        agent1_v4_require_runtime
        [[ -s "$AGENT1_V4_ACQUISITION_RECEIPT" && -s "$AGENT1_V4_RUN_ROOT/20_human_gate_receipt.json" && -s "$AGENT1_V4_RUN_ROOT/30_field_profile.json" ]] || { echo "ERROR: acquisition, human-gate, and field-profile receipts are required" >&2; exit 8; }
        [[ -s "$AGENT1_V4_FIELD_MAPPING" ]] || { echo "ERROR: AGENT1_V4_FIELD_MAPPING must name the explicit approved mapping" >&2; exit 8; }
        agent1_v4_run_python "$AGENT1_V4_ENVELOPE_MATERIALIZER" \
            --sources "$AGENT1_V4_SOURCES" --acquisition-receipt "$AGENT1_V4_ACQUISITION_RECEIPT" \
            --human-gate-receipt "$AGENT1_V4_RUN_ROOT/20_human_gate_receipt.json" \
            --field-profile "$AGENT1_V4_RUN_ROOT/30_field_profile.json" --field-mapping "$AGENT1_V4_FIELD_MAPPING" \
            --output "$AGENT1_V4_RUN_ROOT/30_nanochat_envelope"
        ;;
    *)
        echo "ERROR: unsupported Agent 1 v4 action: $action" >&2
        exit 64
        ;;
esac
