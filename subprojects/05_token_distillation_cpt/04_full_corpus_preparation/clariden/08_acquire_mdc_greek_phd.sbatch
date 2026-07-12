#!/usr/bin/env bash
# CPU-only, receipt-bound acquisition of the raw Mozilla Greek-PhD release used
# for SPAN classifier research. This job is intentionally outside the corpus DAG.
#SBATCH --job-name=cpt4_mdc_greek_phd
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

HERE=${PHASE04_CLARIDEN_DIR:?submit with the exact checked-out Clariden directory}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "MDC Greek-PhD acquisition"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

: "${MOZILLA_DATA_COLLECTIVE_API_KEY:?inject the MDC API key into the Slurm environment}"

DATASET_ID="cmkwvpu7s0032mo07jpk20pj1"
EXPECTED_FILENAME="greek-phd-theses-corpus-v1-0-4bc0fc26.tar.gz"
EXPECTED_ARCHIVE_SHA256="aff1d6c10e03e2b28c96536f7c676022ec91bc079281d51b1119943f6e965b5a"
ROOT="${MDC_GREEK_PHD_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/full_corpus_v2/mdc/greek_phd_v1/$EXPECTED_ARCHIVE_SHA256}"
ARCHIVE="$ROOT/$EXPECTED_FILENAME"
PARTIAL="$ARCHIVE.partial"
EXTRACTED="$ROOT/extracted"
FILE_MANIFEST="$ROOT/extracted.sha256"
RECEIPT="$ROOT/acquisition.receipt.json"

mkdir -p "$ROOT"
if [[ -f "$RECEIPT" ]]; then
    jq -e --arg sha "$EXPECTED_ARCHIVE_SHA256" \
        '.status == "passed" and .archive.sha256 == $sha' "$RECEIPT" >/dev/null
    echo "MDC_GREEK_PHD_ALREADY_COMPLETE=$RECEIPT"
    exit 0
fi

response="$({
    curl -fsS -X POST \
        -H "Authorization: Bearer $MOZILLA_DATA_COLLECTIVE_API_KEY" \
        "https://mozilladatacollective.com/api/datasets/$DATASET_ID/download"
})"
download_url="$(jq -er '.downloadUrl' <<<"$response")"
api_filename="$(jq -er '.filename' <<<"$response")"
api_checksum="$(jq -er '.checksum' <<<"$response")"
api_checksum="${api_checksum#sha256:}"
api_size="$(jq -er '.sizeBytes | tostring' <<<"$response")"

[[ "$api_filename" == "$EXPECTED_FILENAME" ]] || {
    echo "ERROR: MDC filename drift: $api_filename" >&2
    exit 20
}
[[ "$api_checksum" == "$EXPECTED_ARCHIVE_SHA256" ]] || {
    echo "ERROR: MDC archive checksum drift: $api_checksum" >&2
    exit 21
}

if [[ ! -f "$ARCHIVE" ]]; then
    curl -fL --retry 8 --retry-all-errors --continue-at - \
        --output "$PARTIAL" "$download_url"
    actual_sha="$(sha256sum "$PARTIAL" | awk '{print $1}')"
    [[ "$actual_sha" == "$EXPECTED_ARCHIVE_SHA256" ]] || {
        echo "ERROR: downloaded MDC archive checksum mismatch: $actual_sha" >&2
        exit 22
    }
    mv "$PARTIAL" "$ARCHIVE"
else
    actual_sha="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
    [[ "$actual_sha" == "$EXPECTED_ARCHIVE_SHA256" ]] || {
        echo "ERROR: existing MDC archive checksum mismatch: $actual_sha" >&2
        exit 23
    }
fi

if [[ ! -d "$EXTRACTED" ]]; then
    extract_tmp="$(mktemp -d "$ROOT/.extracted.XXXXXX")"
    trap 'rm -rf "$extract_tmp"' EXIT
    tar -xzf "$ARCHIVE" -C "$extract_tmp"
    mv "$extract_tmp" "$EXTRACTED"
    trap - EXIT
fi

manifest_tmp="$FILE_MANIFEST.partial"
(
    cd "$EXTRACTED"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest_tmp"
mv "$manifest_tmp" "$FILE_MANIFEST"
manifest_sha="$(sha256sum "$FILE_MANIFEST" | awk '{print $1}')"
archive_bytes="$(stat -c %s "$ARCHIVE")"
file_count="$(wc -l <"$FILE_MANIFEST" | tr -d ' ')"

receipt_tmp="$RECEIPT.partial"
jq -n \
    --arg dataset_id "$DATASET_ID" \
    --arg filename "$EXPECTED_FILENAME" \
    --arg archive_path "$ARCHIVE" \
    --arg archive_sha "$EXPECTED_ARCHIVE_SHA256" \
    --arg archive_bytes "$archive_bytes" \
    --arg api_size "$api_size" \
    --arg extracted_path "$EXTRACTED" \
    --arg manifest_path "$FILE_MANIFEST" \
    --arg manifest_sha "$manifest_sha" \
    --arg file_count "$file_count" \
    --arg acquired_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{
        schema_version: "mdc_dataset_acquisition_receipt_v1",
        status: "passed",
        dataset_id: $dataset_id,
        source: "Mozilla Data Collective API",
        snapshot_equivalence_to_historical_span_inputs: "unverified",
        acquired_at: $acquired_at,
        archive: {
            filename: $filename,
            path: $archive_path,
            sha256: $archive_sha,
            bytes: ($archive_bytes | tonumber),
            api_size_bytes: ($api_size | tonumber)
        },
        extracted: {
            path: $extracted_path,
            file_count: ($file_count | tonumber),
            sha256_manifest_path: $manifest_path,
            sha256_manifest_sha256: $manifest_sha
        }
    }' >"$receipt_tmp"
mv "$receipt_tmp" "$RECEIPT"

unset response download_url MOZILLA_DATA_COLLECTIVE_API_KEY
echo "MDC_GREEK_PHD_ACQUISITION_COMPLETE=$RECEIPT"
