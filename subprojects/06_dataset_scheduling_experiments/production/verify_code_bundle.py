#!/usr/bin/env python3
"""Verify an immutable code bundle against its receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The worker invokes this script by absolute path with PYTHONSAFEPATH=1.  In
# that mode Python intentionally omits the script directory, so bind the
# verifier's sibling module explicitly instead of depending on ambient paths.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
