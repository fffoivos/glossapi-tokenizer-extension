#!/usr/bin/env python3
"""Repair post-save campaign receipts without rerunning completed training."""

from __future__ import annotations

import argparse
from pathlib import Path

from contract_utils import file_binding, read_json, require, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--phase-cache-receipt", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    args = parser.parse_args()

    attempt = args.attempt_root.resolve()
    audit_path = attempt / "checkpoint_audit.json"
    permit_path = attempt / "checkpoint_permit.json"
    reference_path = attempt / "checkpoint_reference.json"
    require(not reference_path.exists(), f"immutable reference exists: {reference_path}")
    audit = read_json(audit_path)
    permit = read_json(permit_path)
    require(audit.get("status") == "passed", "checkpoint audit is not passing")
    require(permit.get("status") == "passed", "checkpoint permit is not passing")
    require(int(audit.get("update", -1)) == args.update, "audit update drift")
    require(int(permit.get("update", -1)) == args.update, "permit update drift")
    checkpoint_root = attempt / "payload" / "checkpoints" / f"iter_{args.update:07d}"
    require(checkpoint_root.is_dir(), "checkpoint root missing")
    write_json_atomic(
        reference_path,
        {
            "schema_version": "apertus_hard_h_to_g_checkpoint_reference_v1",
            "status": "passed",
            "scale": "8b",
            "phase": 2,
            "update": args.update,
            "claimed_end_update": args.update,
            "gracefully_stopped": False,
            "load_root": str(checkpoint_root.parent.resolve()),
            "checkpoint_root": str(checkpoint_root.resolve()),
            "checkpoint_permit": file_binding(permit_path),
            "source_phase_cache_receipt": file_binding(args.phase_cache_receipt),
            "checkpoint_audit": file_binding(audit_path),
        },
    )
    print(reference_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
