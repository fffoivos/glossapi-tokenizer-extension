#!/usr/bin/env python3
"""Freeze the selected-profile checkpoint conversion and GreekMMLU smoke."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--greekmmlu-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = read_json(args.selected_profile)
    export = read_json(args.export_receipt)
    greek = read_json(args.greekmmlu_receipt)
    if selected.get("status") != "frozen":
        raise ValueError("selected profile is not frozen")
    if export.get("schema_version") != "native_greekmmlu_exact_checkpoint_export_v1" or export.get("status") != "completed":
        raise ValueError("checkpoint export smoke failed")
    if greek.get("schema_version") != "exact_checkpoint_native_greekmmlu_receipt_v1" or greek.get("status") != "completed":
        raise ValueError("native GreekMMLU smoke failed")
    payload = {
        "schema_version": "apertus_full_8b_conversion_greekmmlu_smoke_v1",
        "status": "passed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": 160,
        "selected_profile_id": selected["selection"]["profile_id"],
        "selected_profile": file_binding(args.selected_profile),
        "export_receipt": file_binding(args.export_receipt),
        "greekmmlu_receipt": file_binding(args.greekmmlu_receipt),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "profile": payload["selected_profile_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
