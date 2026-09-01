#!/usr/bin/env python3
"""Freeze all finite iteration-zero source-conditioned validation metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root / "subprojects/06_dataset_scheduling_experiments"))
    from evaluation.collect_validation_trajectory import parse_log

    validation = json.loads(args.validation_manifest.read_text())
    expected = {row["name"] for row in validation["panels"]}
    rows = [row for row in parse_log(args.log) if int(row["iteration"]) == 0]
    by_panel = {row["panel"]: row for row in rows}
    if set(by_panel) != expected:
        raise ValueError(f"initial validation panel drift: {sorted(set(by_panel) ^ expected)}")
    for panel, row in by_panel.items():
        numeric = [value for key, value in row.items() if key not in {"panel", "iteration"} and isinstance(value, (int, float))]
        if not numeric or not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"non-finite initial validation metrics: {panel}")
    payload = {
        "schema_version": "apertus_full_8b_initial_validation_v1",
        "status": "completed",
        "iteration": 0,
        "validation_manifest_sha256": sha(args.validation_manifest),
        "log_sha256": sha(args.log),
        "panels": [by_panel[name] for name in sorted(expected)],
    }
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "panels": len(by_panel)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
