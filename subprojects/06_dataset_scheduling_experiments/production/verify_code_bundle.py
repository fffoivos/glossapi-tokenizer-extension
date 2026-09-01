#!/usr/bin/env python3
"""Verify an immutable code bundle against its receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaign_contract import verify_code_bundle_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kind", choices=("scientific", "efficiency"), required=True)
    args = parser.parse_args()
    value = verify_code_bundle_receipt(args.receipt, args.root, args.kind)
    print(
        json.dumps(
            {"ok": True, "kind": args.kind, "tree_sha256": value["tree_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
