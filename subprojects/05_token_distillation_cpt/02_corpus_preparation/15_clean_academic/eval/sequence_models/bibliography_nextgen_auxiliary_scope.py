#!/usr/bin/env python3
"""Materialize the frozen deterministic auxiliary-section scope for decoding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_auxiliary_scope_veto import materialize_auxiliary_headings
from .bibliography_entry_models import load_table
from .bibliography_scope_rules import (
    AUXILIARY_SCOPE_HEADINGS,
    AUXILIARY_SCOPE_PREFIXES,
    BODY_CITATION_SCOPE_HEADINGS,
)
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-auxiliary-scope-v1"


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve()
    base_root = Path(args.base_table_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    table = load_table(base_root, expected_split=args.split)
    headings, scope = materialize_auxiliary_headings(
        table, source, expected_split=args.split
    )
    if headings.shape != scope.shape or scope.shape != (len(table.targets),):
        raise RuntimeError("auxiliary scope contract failure")
    output.mkdir(parents=True)
    for name, value in (
        ("auxiliary_scope_heading.npy", headings),
        ("auxiliary_scope_active.npy", scope),
    ):
        with (output / name).open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_frozen_deterministic_auxiliary_scope",
        "validation_opened": False,
        "test_opened": False,
        "line_count": len(scope),
        "heading_line_count": int(np.count_nonzero(headings)),
        "active_scope_line_count": int(np.count_nonzero(scope)),
        "auxiliary_scope_headings": sorted(AUXILIARY_SCOPE_HEADINGS),
        "body_citation_scope_headings": sorted(BODY_CITATION_SCOPE_HEADINGS),
        "auxiliary_scope_prefixes": list(AUXILIARY_SCOPE_PREFIXES),
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "source_sha256": sha256_file(source),
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
        },
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(
        output / "receipt.json",
        {
            **report,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
