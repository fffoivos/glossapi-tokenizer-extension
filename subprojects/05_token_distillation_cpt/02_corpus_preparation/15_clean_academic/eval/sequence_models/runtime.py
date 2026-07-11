#!/usr/bin/env python3
"""CPU-only parity, export-manifest, and lightweight benchmark hooks."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .baseline import _load_json, predict_document
from .contract import canonical_json_sha256, read_gold, sha256_file
from .features import EVAL_DIR


def compare_prediction_files(left: str | Path, right: str | Path) -> dict[str, Any]:
    def rows(path: str | Path) -> dict[str, Any]:
        result = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    result[row["document_id"]] = [
                        (item["line_id"], item["abs_idx"], item["prediction"])
                        for item in row["lines"]
                    ]
        return result

    lrows, rrows = rows(left), rows(right)
    mismatched_documents = sorted(
        document_id for document_id in set(lrows) | set(rrows)
        if lrows.get(document_id) != rrows.get(document_id)
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    return {
        "schema_version": "academic-structure-runtime-parity-v1",
        "status": "pass" if not mismatched_documents else "fail",
        "left_sha256": sha256_file(left),
        "right_sha256": sha256_file(right),
        "document_count": len(set(lrows) | set(rrows)),
        "mismatched_document_count": len(mismatched_documents),
        "first_mismatched_documents": mismatched_documents[:20],
        "peak_rss_bytes": peak_rss_bytes,
    }


def benchmark_baseline(silver_path: str | Path, repeats: int) -> dict[str, Any]:
    documents = read_gold(silver_path)
    bib_path = EVAL_DIR / "span_line_lr_struct_model.json"
    toc_path = EVAL_DIR / "toc_line_lr_model.json"
    decoder_path = EVAL_DIR / "struct_smooth_params.json"
    bib, toc, decoder = map(_load_json, (bib_path, toc_path, decoder_path))
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        for document in documents:
            predict_document(document, bib, toc, decoder)
        durations.append(time.perf_counter() - started)
    duration = min(durations)
    lines = sum(len(document.lines) for document in documents)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    return {
        "schema_version": "academic-structure-cpu-benchmark-v1",
        "runtime": "python-c0-parity",
        "device": "cpu",
        "repeats": repeats,
        "best_seconds": duration,
        "documents_per_second": len(documents) / duration,
        "lines_per_second": lines / duration,
        "peak_rss_bytes": peak_rss_bytes,
        "silver_sha256": sha256_file(silver_path),
        "artifact_inventory_sha256": canonical_json_sha256(
            {path.name: sha256_file(path) for path in (bib_path, toc_path, decoder_path)}
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    parity = subparsers.add_parser("parity")
    parity.add_argument("--left", required=True)
    parity.add_argument("--right", required=True)
    benchmark = subparsers.add_parser("benchmark-c0")
    benchmark.add_argument("--silver", required=True)
    benchmark.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "parity":
        receipt = compare_prediction_files(args.left, args.right)
    else:
        if args.repeats < 1:
            raise ValueError("repeats must be positive")
        receipt = benchmark_baseline(args.silver, args.repeats)
    payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if receipt.get("status", "pass") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
