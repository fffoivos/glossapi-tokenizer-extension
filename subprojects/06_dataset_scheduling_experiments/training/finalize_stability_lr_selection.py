#!/usr/bin/env python3
"""Freeze the common LR selected by candidate-first, fallback-only policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--fallback-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate_receipt.read_text())
    fallback = json.loads(args.fallback_receipt.read_text()) if args.fallback_receipt else None
    if float(candidate.get("peak_lr", -1)) != 3e-4:
        raise ValueError("candidate LR drift")
    if candidate.get("status") == "passed":
        if fallback is not None:
            raise ValueError("fallback was run even though the candidate passed")
        selected = candidate
        reason = "3e-4 candidate passed every predeclared stability gate"
    else:
        if fallback is None or float(fallback.get("peak_lr", -1)) != 1.5e-4 or fallback.get("status") != "passed":
            raise ValueError("candidate failed and the one permitted common fallback did not pass")
        selected = fallback
        reason = "3e-4 candidate failed; same smoke passed at the predeclared 1.5e-4 fallback"
    payload = {
        "schema_version": "apertus_mini_common_lr_selection_v1",
        "status": "frozen",
        "selected_peak_lr": selected["peak_lr"],
        "selected_min_lr": selected["peak_lr"] * 0.1,
        "selection_reason": reason,
        "same_lr_for_all_five_arms": True,
        "candidate_receipt": {"path": str(args.candidate_receipt.resolve()), "sha256": sha(args.candidate_receipt)},
        "fallback_receipt": {"path": str(args.fallback_receipt.resolve()), "sha256": sha(args.fallback_receipt)} if args.fallback_receipt else None,
    }
    partial = Path(str(args.output) + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, args.output)
    print(json.dumps({"ok": True, "selected_peak_lr": selected["peak_lr"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
