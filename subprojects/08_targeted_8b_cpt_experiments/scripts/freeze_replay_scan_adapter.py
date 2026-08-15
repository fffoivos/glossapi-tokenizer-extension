#!/usr/bin/env python3
"""Freeze the scanner adapter for the exact rebuilt replay selection."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-jsonl", type=Path, required=True)
    parser.add_argument("--mix-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.replay_jsonl.is_file(), f"rebuilt replay JSONL missing: {args.replay_jsonl}")
    require(not args.output.exists(), f"immutable adapter exists: {args.output}")
    manifest = read_json(args.mix_manifest)
    require(Path(manifest["output"]).resolve() == args.replay_jsonl.resolve(), "replay manifest output drift")
    expected_rows = int(manifest["actual_rows"])
    per_source = manifest.get("per_source")
    require(isinstance(per_source, dict) and per_source, "replay manifest lacks per-source accounting")
    content_counts = {name: int(value["rows"]) for name, value in sorted(per_source.items())}
    require(sum(content_counts.values()) == expected_rows, "replay per-source row accounting drift")
    require(all(value > 0 for value in content_counts.values()), "rebuilt replay has an empty selected source")
    binding = file_binding(args.replay_jsonl)
    payload = {
        "schema_version": "apertus_replay_scan_adapter_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "source_level_disjointness_escape_allowed": False,
        "expected_selected_source_names": ["replay_only_selected_predecontam"],
        "expected_content_source_counts": content_counts,
        "sources": [
            {
                "name": "replay_only_selected_predecontam",
                "path": str(args.replay_jsonl.resolve()),
                "format": "jsonl",
                "expected_rows": expected_rows,
                "expected_sha256": binding["sha256"],
                "mapping": {
                    "text_field": "text",
                    "source_dataset_field": "source",
                    "source_doc_id_field": "doc_id",
                },
            }
        ],
        "mix_manifest": file_binding(args.mix_manifest),
        "selected_replay": binding,
        "scan_scope": ["native_greek_suite_v1", "greekmmlu_regenerated_v1"],
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
