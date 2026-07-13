#!/usr/bin/env python3
"""Validate a returned 360-response Terra evidence bundle on CSCS."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v4_raw_review import file_binding, validate_packet, write_json_no_replace  # noqa: E402
from run_agent1_v4_terra_reviews import validate_response  # noqa: E402


RECEIPT_SCHEMA = "agent1_v4_terra_response_validation_receipt_v1"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def validate_response_bundle(
    *, packet_root: Path, packet_manifest_path: Path, responses_path: Path, output: Path
) -> dict[str, object]:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"response validation receipt already exists: {output}")
    manifest = validate_packet(packet_root, packet_manifest_path)
    requests = _read_jsonl(packet_root / "requests.jsonl")
    responses = _read_jsonl(responses_path)
    if len(requests) != 360 or len(responses) != 360:
        raise ValueError("Terra evidence bundle must have exactly 360 requests and responses")
    by_request = {str(row.get("request_id")): row for row in requests}
    if len(by_request) != 360:
        raise ValueError("packet request IDs are not unique")
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    for response in responses:
        request_id = str(response.get("request_id") or "")
        request = by_request.get(request_id)
        if request is None or request_id in seen:
            raise ValueError("response does not bind exactly one packet request")
        document = packet_root / str(request["document_path"])
        validate_response(response, request, document)
        seen.add(request_id)
        source_counts[str(request["source_id"])] += 1
    if seen != set(by_request) or source_counts != manifest.get("source_counts"):
        raise ValueError("Terra response closure differs from packet manifest")
    receipt: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "packet_manifest": file_binding(packet_manifest_path),
        "requests": file_binding(packet_root / "requests.jsonl"),
        "responses": file_binding(responses_path),
        "logical_review_count": len(responses),
        "source_counts": dict(sorted(source_counts.items())),
    }
    write_json_no_replace(output, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate_response_bundle(
        packet_root=args.packet_root,
        packet_manifest_path=args.packet_manifest,
        responses_path=args.responses,
        output=args.output,
    )
    print(json.dumps({"ok": True, "logical_review_count": receipt["logical_review_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
