#!/usr/bin/env python3
"""Freeze one exact-checkpoint source-conditioned validation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root / "subprojects/06_dataset_scheduling_experiments"))
    from evaluation.collect_validation_trajectory import parse_log

    validation = json.loads(args.validation_manifest.read_text())
    expected = {row["name"] for row in validation["panels"]}
    rows = [row for row in parse_log(args.log) if int(row["iteration"]) == args.iteration]
    by_panel = {row["panel"]: row for row in rows}
    if set(by_panel) != expected:
        raise ValueError(f"checkpoint validation panel drift: {sorted(set(by_panel) ^ expected)}")
    for panel, row in by_panel.items():
        numeric = [
            value for key, value in row.items()
            if key not in {"panel", "iteration"} and isinstance(value, (int, float))
        ]
        if not numeric or not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"non-finite checkpoint validation metrics: {panel}")
    payload = {
        "schema_version": "apertus_full_8b_source_validation_checkpoint_v1",
        "status": "completed",
        "iteration": args.iteration,
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
        "log_sha256": sha256_file(args.log),
        "panels": [by_panel[name] for name in sorted(expected)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "iteration": args.iteration, "panels": len(by_panel)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
