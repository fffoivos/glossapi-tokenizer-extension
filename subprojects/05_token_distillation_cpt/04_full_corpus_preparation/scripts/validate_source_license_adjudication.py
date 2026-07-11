#!/usr/bin/env python3
"""Validate the receipt-bound source-license decision matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source_license import read_json_object, sha256_file, validate_adjudication


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=here / "configs" / "source_license_adjudication.json",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    sources = read_json_object(args.sources)
    adjudication = read_json_object(args.adjudication)
    errors = validate_adjudication(adjudication, sources, registry_path=args.sources)
    result = {
        "ok": not errors,
        "errors": errors,
        "sources_sha256": sha256_file(args.sources),
        "adjudication_sha256": sha256_file(args.adjudication),
        "candidate_sources": len(adjudication.get("sources", [])),
    }
    if args.json or errors:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"OK: {result['candidate_sources']} candidate sources; "
            f"adjudication_sha256={result['adjudication_sha256']}"
        )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
