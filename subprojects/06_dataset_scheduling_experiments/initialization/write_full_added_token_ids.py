#!/usr/bin/env python3
"""Write and receipt the complete ordered 17,920-ID TD request."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("refusing to overwrite full TD inventory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{value}\n" for value in range(131_072, 148_992)))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    payload = {
        "schema_version": "apertus_mini_full_td_token_inventory_v1",
        "status": "frozen",
        "first_id": 131_072,
        "last_id": 148_991,
        "count": 17_920,
        "ordered_and_contiguous": True,
        "path": str(args.output.resolve()),
        "sha256": digest,
        "bytes": args.output.stat().st_size,
    }
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "count": 17_920, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
