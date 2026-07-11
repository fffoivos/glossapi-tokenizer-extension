#!/usr/bin/env python3
"""Certify zero-error, parity-bound structural detector audit outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def row_uid(source: str, doc_id: str) -> str:
    return hashlib.sha256(f"{source}\0{doc_id}".encode()).hexdigest()


def update_id_digest(digest: hashlib._Hash, doc_id: str) -> None:
    encoded = doc_id.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--stream-manifest", type=Path, required=True)
    parser.add_argument("--counters", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--parity-receipt", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--input-receipt-check", type=Path, required=True)
    parser.add_argument("--detector-build-receipt", type=Path, required=True)
    parser.add_argument("--sources-config", type=Path, required=True)
    parser.add_argument("--cleaning-policy", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stream = load_object(args.stream_manifest)
    parity = load_object(args.parity_receipt)
    input_receipt = load_object(args.input_receipt)
    input_check = load_object(args.input_receipt_check)
    build_receipt = load_object(args.detector_build_receipt)
    errors: list[str] = []
    binary_sha256 = sha256_file(args.binary)
    sources_config_sha256 = sha256_file(args.sources_config)
    if parity.get("schema_version") != "struct_rust_parity_receipt_v1" or parity.get("status") != "passed":
        errors.append("parity receipt is not a passed struct_rust_parity_receipt_v1")
    if parity.get("binary_sha256") != binary_sha256:
        errors.append("detector binary does not match parity receipt")
    if (
        input_receipt.get("schema_version") != "full_cpt_acquisition_receipt_v1"
        or input_receipt.get("status") != "passed"
    ):
        errors.append("input receipt is not a passed full_cpt_acquisition_receipt_v1")
    if input_receipt.get("sources_config_sha256") != sources_config_sha256:
        errors.append("input receipt does not match sources config")
    if input_check.get("schema_version") != "full_cpt_input_receipt_check_v1" or input_check.get("ok") is not True:
        errors.append("input receipt launch check is not passed")
    if input_check.get("input_receipt_sha256") != sha256_file(args.input_receipt):
        errors.append("input receipt launch check does not match input receipt")
    if input_check.get("source") != args.source:
        errors.append("input receipt launch check source mismatch")
    if (
        build_receipt.get("schema_version") != "full_cpt_detector_build_receipt_v1"
        or build_receipt.get("status") != "passed"
    ):
        errors.append("detector-build receipt is not a passed full_cpt_detector_build_receipt_v1")
    if build_receipt.get("code_commit") != args.code_commit:
        errors.append("detector-build receipt does not match execution commit")
    if build_receipt.get("binary", {}).get("sha256") != binary_sha256:
        errors.append("detector-build receipt does not match detector binary")
    if Path(str(build_receipt.get("binary", {}).get("path", ""))).resolve() != args.binary.resolve():
        errors.append("detector-build receipt binary path mismatch")
    if stream.get("source") != args.source:
        errors.append("stream manifest source mismatch")
    if stream.get("schema_version") != "detector_input_stream_v1":
        errors.append("unsupported detector input-stream manifest")
    streamed_inputs = stream.get("inputs", [])
    streamed_inventory = {
        str(Path(str(row.get("path", ""))).resolve()): row.get("bytes")
        for row in streamed_inputs
        if isinstance(row, dict) and row.get("path")
    }
    checked_inputs = input_check.get("inputs", [])
    checked_inventory = {
        str(Path(str(row.get("path", ""))).resolve()): row.get("bytes")
        for row in checked_inputs
        if isinstance(row, dict) and row.get("path")
    }
    if (
        streamed_inventory != checked_inventory
        or len(streamed_inventory) != len(streamed_inputs)
        or len(checked_inventory) != len(checked_inputs)
    ):
        errors.append("streamed per-path byte inventory differs from the passed input-receipt check")
    for field in ("text_column", "id_column", "source_column"):
        if stream.get(field) != input_check.get(field):
            errors.append(f"streamed {field} differs from the input-receipt check")
    expected_rows = int(stream.get("rows_emitted", -1))
    if expected_rows <= 0:
        errors.append("detector input stream emitted no rows")

    counters: dict[str, dict] = {}
    id_sequence = hashlib.sha256()
    overlap_documents = overlap_chars = overlap_lines = 0
    with args.counters.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" in row:
                errors.append(f"counter row {line_number} contains detector error: {row['error']}")
                continue
            doc_id = str(row.get("doc_id", ""))
            if not doc_id:
                errors.append(f"counter row {line_number} missing doc_id")
                continue
            if doc_id in counters:
                errors.append(f"duplicate counter doc_id {doc_id!r}")
                continue
            if row.get("source") != args.source:
                errors.append(f"{doc_id}: counter source mismatch")
            if row.get("row_uid") != row_uid(args.source, doc_id):
                errors.append(f"{doc_id}: counter row_uid mismatch")
            if len(str(row.get("original_sha256", ""))) != 64:
                errors.append(f"{doc_id}: counter missing original_sha256")
            counters[doc_id] = row
            update_id_digest(id_sequence, doc_id)
            if int(row.get("overlap_pairs", 0)):
                overlap_documents += 1
                overlap_chars += int(row.get("overlap_chars", 0))
                overlap_lines += int(row.get("overlap_lines", 0))

    if len(counters) != expected_rows:
        errors.append(f"counter coverage {len(counters)} != streamed rows {expected_rows}")
    if id_sequence.hexdigest() != stream.get("id_sequence_sha256"):
        errors.append("ordered counter document IDs do not match streamed IDs")

    span_count = 0
    with args.spans.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            span_count += 1
            row = json.loads(line)
            doc_id = str(row.get("doc_id", ""))
            counter = counters.get(doc_id)
            if counter is None:
                errors.append(f"span row {line_number}: unknown doc_id {doc_id!r}")
                continue
            for field in ("source", "row_uid", "original_sha256", "original_chars"):
                if row.get(field) != counter.get(field):
                    errors.append(f"{doc_id}: span/counter {field} mismatch")
            if row.get("kind") not in {"bib_span", "toc_span"}:
                errors.append(f"{doc_id}: unexpected structural span kind {row.get('kind')!r}")

    result = {
        "schema_version": "structural_detector_run_v1",
        "status": "passed" if not errors else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "audit_only",
        "source": args.source,
        "code_commit": args.code_commit,
        "binary_sha256": binary_sha256,
        "parity_receipt": str(args.parity_receipt.resolve()),
        "parity_receipt_sha256": sha256_file(args.parity_receipt),
        "input_receipt": str(args.input_receipt.resolve()),
        "input_receipt_sha256": sha256_file(args.input_receipt),
        "input_receipt_check_sha256": sha256_file(args.input_receipt_check),
        "input_acquisition_code_commit": input_receipt.get("code_commit"),
        "detector_build_receipt": str(args.detector_build_receipt.resolve()),
        "detector_build_receipt_sha256": sha256_file(args.detector_build_receipt),
        "stream_manifest": stream,
        "counters": {"path": args.counters.name, "sha256": sha256_file(args.counters)},
        "spans": {"path": args.spans.name, "sha256": sha256_file(args.spans), "rows": span_count},
        "sources_config_sha256": sources_config_sha256,
        "cleaning_policy_sha256": sha256_file(args.cleaning_policy),
        "conflicts": {
            "documents": overlap_documents,
            "unique_overlap_chars": overlap_chars,
            "unique_overlap_lines": overlap_lines,
            "materialization_status": "blocked_pending_review" if overlap_documents else "not_applicable_audit_only",
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(counters), "spans": span_count, "errors": len(errors)}))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
