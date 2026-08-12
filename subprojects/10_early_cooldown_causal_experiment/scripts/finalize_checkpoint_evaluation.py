#!/usr/bin/env python3
"""Freeze one complete milestone GreekMMLU plus 13-panel evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import atomic_json, file_binding, read_json, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--greekmmlu-receipt", type=Path, required=True)
    parser.add_argument("--per-document-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    require(args.iteration in contract["evaluation"]["milestone_iterations"], "milestone drift")
    greek = read_json(args.greekmmlu_receipt)
    require(greek.get("status") == "completed" and greek.get("checkpoint", {}).get("iteration") == args.iteration, "GreekMMLU receipt drift")
    documents = []
    for path in sorted(args.per_document_root.glob("*.receipt.json")):
        row = read_json(path)
        require(row.get("status") == "completed" and int(row.get("aggregate", {}).get("documents", 0)) > 0, f"invalid document receipt: {path}")
        documents.append(file_binding(path))
    require(len(documents) == 13, f"expected 13 document receipts, got {len(documents)}")
    payload = {
        "schema_version": "apertus_full8_early_cooldown_checkpoint_evaluation_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": args.iteration,
        "token_slots": args.iteration * 1024 * 4096,
        "greekmmlu": file_binding(args.greekmmlu_receipt),
        "per_document": documents,
        "root": str(args.root.resolve()),
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "iteration": args.iteration, "document_panels": 13}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
