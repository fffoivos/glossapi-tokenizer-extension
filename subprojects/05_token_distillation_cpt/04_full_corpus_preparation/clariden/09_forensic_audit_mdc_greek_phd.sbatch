#!/usr/bin/env bash
# Forensic-only audit for an MDC object whose bytes do not match the publisher's
# declared checksum. The object remains quarantined even if coordinates align.
#SBATCH --job-name=cpt4_mdc_phd_audit
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:20:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

HERE=${PHASE04_CLARIDEN_DIR:?submit with the exact checked-out Clariden directory}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "MDC Greek-PhD forensic audit"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

PUBLISHER_SHA="aff1d6c10e03e2b28c96536f7c676022ec91bc079281d51b1119943f6e965b5a"
OBSERVED_SHA="5e06395972d5a000ea7a19af54bae72f9df53d5dff13d3a28c0d65b7dbd28922"
FILENAME="greek-phd-theses-corpus-v1-0-4bc0fc26.tar.gz"
MDC_ROOT="/iopsstor/scratch/cscs/fffoivos/cpt_corpus/full_corpus_v2/mdc/greek_phd_v1"
SOURCE_PARTIAL="$MDC_ROOT/$PUBLISHER_SHA/$FILENAME.partial"
ROOT="$MDC_ROOT/observed-$OBSERVED_SHA"
ARCHIVE="$ROOT/$FILENAME"
EXTRACTED="$ROOT/extracted"
EXTRACTION_MANIFEST="$ROOT/extraction.manifest.json"
EXTRACTION_RECEIPT="$ROOT/extraction.receipt.json"
QUARANTINE_RECEIPT="$ROOT/quarantine.v2.receipt.json"
AUDIT_RECEIPT="$ROOT/span-coordinate-audit.v2.json"
CLASSIFIER_REPO_ROOT="${CLASSIFIER_REPO_ROOT:-$REPO_ROOT}"
EVAL_DIR="$CLASSIFIER_REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
AUDIT_TOOL="${MDC_SPAN_AUDIT_TOOL:-$EVAL_DIR/sequence_models/mdc_span_audit.py}"
EXTRACT_TOOL="${MDC_SAFE_EXTRACT_TOOL:-$EVAL_DIR/sequence_models/mdc_safe_extract.py}"
PYTHON="$RUNTIME_VENV/bin/python"

for path in "$AUDIT_TOOL" "$EXTRACT_TOOL"; do
    [[ -f "$path" ]] || {
        echo "ERROR: MDC forensic tool is absent: $path" >&2
        exit 33
    }
done

mkdir -p "$ROOT"
if [[ ! -f "$ARCHIVE" ]]; then
    [[ -f "$SOURCE_PARTIAL" ]] || {
        echo "ERROR: quarantined download is absent: $SOURCE_PARTIAL" >&2
        exit 30
    }
    observed="$(sha256sum "$SOURCE_PARTIAL" | awk '{print $1}')"
    [[ "$observed" == "$OBSERVED_SHA" ]] || {
        echo "ERROR: quarantined object changed: $observed" >&2
        exit 31
    }
    mv "$SOURCE_PARTIAL" "$ARCHIVE"
fi

observed="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
[[ "$observed" == "$OBSERVED_SHA" ]] || {
    echo "ERROR: observed archive hash drift: $observed" >&2
    exit 32
}
gzip -t "$ARCHIVE"
tar -tzf "$ARCHIVE" >/dev/null

uenv run "$PHASE04_UENV" --view=default -- \
    "$PYTHON" "$EXTRACT_TOOL" \
        --archive "$ARCHIVE" \
        --extracted-root "$EXTRACTED" \
        --manifest "$EXTRACTION_MANIFEST" \
        --receipt "$EXTRACTION_RECEIPT"

extraction_receipt_sha="$(sha256sum "$EXTRACTION_RECEIPT" | awk '{print $1}')"
manifest_sha="$(jq -er '.extraction.manifest_sha256' "$EXTRACTION_RECEIPT")"
file_count="$(jq -er '.extraction.file_count | tostring' "$EXTRACTION_RECEIPT")"
archive_bytes="$(stat -c %s "$ARCHIVE")"

receipt_tmp="$QUARANTINE_RECEIPT.partial"
jq -n \
    --arg archive "$ARCHIVE" \
    --arg observed "$OBSERVED_SHA" \
    --arg publisher "$PUBLISHER_SHA" \
    --arg archive_bytes "$archive_bytes" \
    --arg extraction "$EXTRACTION_RECEIPT" \
    --arg extraction_sha "$extraction_receipt_sha" \
    --arg manifest "$EXTRACTION_MANIFEST" \
    --arg manifest_sha "$manifest_sha" \
    --arg extracted "$EXTRACTED" \
    --arg files "$file_count" \
    '{
        schema_version: "mdc_quarantined_object_receipt_v2",
        status: "quarantined_publisher_checksum_mismatch",
        archive: {
            path: $archive,
            bytes: ($archive_bytes | tonumber),
            observed_sha256: $observed,
            publisher_declared_sha256: $publisher,
            gzip_and_tar_integrity: "passed"
        },
        safe_extraction: {
            receipt_path: $extraction,
            receipt_sha256: $extraction_sha,
            status: "passed_fresh_archive_tree_matches"
        },
        extracted: {
            path: $extracted,
            file_count: ($files | tonumber),
            sha256_manifest_path: $manifest,
            sha256_manifest_sha256: $manifest_sha
        }
    }' >"$receipt_tmp"
if [[ -e "$QUARANTINE_RECEIPT" || -L "$QUARANTINE_RECEIPT" ]]; then
    [[ -f "$QUARANTINE_RECEIPT" && ! -L "$QUARANTINE_RECEIPT" ]] || {
        echo "ERROR: immutable quarantine v2 receipt is not a regular file" >&2
        exit 34
    }
    cmp -s "$receipt_tmp" "$QUARANTINE_RECEIPT" || {
        echo "ERROR: immutable quarantine v2 receipt differs" >&2
        exit 35
    }
    rm -f "$receipt_tmp"
else
    ln "$receipt_tmp" "$QUARANTINE_RECEIPT"
    rm -f "$receipt_tmp"
fi

uenv run "$PHASE04_UENV" --view=default -- \
    env PYTHONPATH="$EVAL_DIR" "$PYTHON" "$AUDIT_TOOL" \
        --archive "$ARCHIVE" \
        --quarantine-receipt "$QUARANTINE_RECEIPT" \
        --publisher-declared-sha256 "$PUBLISHER_SHA" \
        --contents-root "$EXTRACTED/phd-theses-corpus/contents" \
        --manifest "$EVAL_DIR/units/SPAN_manifest.jsonl" \
        --annotations "$EVAL_DIR/annotations_span/all.json" \
        --output "$AUDIT_RECEIPT"

echo "MDC_GREEK_PHD_QUARANTINE_RECEIPT=$QUARANTINE_RECEIPT"
echo "MDC_GREEK_PHD_SPAN_AUDIT=$AUDIT_RECEIPT"
