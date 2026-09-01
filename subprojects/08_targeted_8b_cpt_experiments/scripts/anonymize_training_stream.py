#!/usr/bin/env python3
"""Apply or prove no-op for the receipt-pinned Apertus Stage-B PII masker."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


EXPECTED_MASKER_SHA256 = "8f489a175aeb47f2c0996431a9d1c6f93ec03d4f52d9ea33621b76facfc0e83c"


def load_masker(path: Path):
    require(sha256_file(path) == EXPECTED_MASKER_SHA256, "Stage-B masker SHA-256 drift")
    spec = importlib.util.spec_from_file_location("h2g_pinned_pii_masker", path)
    require(spec is not None and spec.loader is not None, f"cannot import masker: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def input_block(receipt: dict) -> dict:
    for name in ("clean", "output"):
        value = receipt.get(name)
        if isinstance(value, dict) and value.get("sha256") and value.get("rows") is not None:
            return value
    raise ValueError("upstream receipt has no row-bound stream output")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--mode", choices=("assert_noop", "apply"), required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--masker-script", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_jsonl, args.output_receipt):
        require(not output.exists(), f"immutable Stage-B output exists: {output}")
    upstream = read_json(args.input_receipt)
    require(upstream.get("status") == "passed", "Stage-B upstream receipt did not pass")
    block = input_block(upstream)
    require(block["sha256"] == sha256_file(args.input_jsonl), "Stage-B input SHA drift")
    expected_rows = int(block["rows"])
    masker = load_masker(args.masker_script)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{args.output_jsonl.name}.", suffix=".partial", dir=args.output_jsonl.parent)
    temporary = Path(name)
    counts: Counter[str] = Counter()
    try:
        with args.input_jsonl.open(encoding="utf-8") as source, os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                require(isinstance(text, str), f"{line_number}: missing text")
                masked, per_type = masker.mask(text)
                second, second_counts = masker.mask(masked)
                require(second == masked and sum(int(value) for value in second_counts.values()) == 0, f"{line_number}: masker is not idempotent")
                changed = masked != text
                counts["input_rows"] += 1
                counts["changed_rows"] += int(changed)
                for key, value in per_type.items():
                    counts[f"{key}_matches"] += int(value)
                if changed:
                    row["text"] = masked
                if args.mode == "apply":
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush(); os.fsync(output.fileno())
        require(counts["input_rows"] == expected_rows, "Stage-B row-count drift")
        if args.mode == "assert_noop":
            require(counts["changed_rows"] == 0, f"{args.stream}: v2 Stage-B was not a byte-preserving no-op")
            temporary.unlink()
            os.link(args.input_jsonl, args.output_jsonl)
        else:
            os.link(temporary, args.output_jsonl); temporary.unlink()
    except BaseException:
        try: os.close(descriptor)
        except OSError: pass
        temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema_version": "apertus_hard_h_to_g_stage_b_stream_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stream": args.stream, "mode": args.mode,
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"), "nodes": int(os.environ.get("SLURM_NNODES", "0"))},
        "executing_code_bundle": executing_code_bundle(),
        "input": file_binding(args.input_jsonl), "input_receipt": file_binding(args.input_receipt),
        "masker": file_binding(args.masker_script), "masker_upstream_commit": "8af990b9401101cf95acd02b066ed0c449789126",
        "counts": dict(counts),
        "output": {**file_binding(args.output_jsonl), "rows": counts["input_rows"]},
        "invariants": {"row_order_preserved": True, "row_multiplicity_preserved": True, "non_text_fields_preserved": True, "additional_deduplication": False, "masker_idempotence_verified_per_row": True, "asserted_byte_noop": args.mode == "assert_noop"},
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
