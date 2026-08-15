#!/usr/bin/env python3
"""Run the frozen legacy scorer while replacing only its dataset loader."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--legacy-evaluator", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    known, forwarded = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location("h2g_frozen_legacy_eval", known.legacy_evaluator)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy evaluator: {known.legacy_evaluator}")
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    queries = [json.loads(line) for line in known.snapshot.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(queries) != 16_632:
        raise RuntimeError("legacy GreekMMLU snapshot count drift")
    rows = []
    for query in queries:
        row = {
            "id": str(query["example_id"]), "question": str(query["question"]),
            "choices": list(query["choices"]), "answer": int(query["answer_index"]),
            "subject": str(query.get("subject") or ""),
        }
        metadata = query.get("metadata")
        if isinstance(metadata, dict):
            row.update({key: metadata[key] for key in ("category", "sub_category", "level", "subject") if key in metadata and metadata[key] is not None})
        rows.append(row)

    def snapshot_loader(specification):
        if specification.get("id") != "greekmmlu":
            raise RuntimeError("legacy snapshot adapter permits only GreekMMLU")
        return rows, "test"

    legacy._load_dataset = snapshot_loader
    sys.argv = [str(known.legacy_evaluator), *forwarded]
    legacy.main()


if __name__ == "__main__":
    main()
