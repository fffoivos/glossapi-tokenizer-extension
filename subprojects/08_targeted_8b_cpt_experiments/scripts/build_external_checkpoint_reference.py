#!/usr/bin/env python3
"""Bind an audited externally saved checkpoint to the canonical campaign runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_checkpoint_permit import validate_permit
from contract_utils import file_binding, read_json, require, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--source-phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--load-root", type=Path, required=True)
    parser.add_argument("--checkpoint-permit", type=Path, required=True)
    parser.add_argument("--source-phase-cache-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), "immutable checkpoint reference exists")
    load_root = args.load_root.resolve()
    checkpoint_root = load_root / f"iter_{args.update:07d}"
    permit = read_json(args.checkpoint_permit)
    validate_permit(
        permit,
        scale=args.scale,
        source_phase=args.source_phase,
        update=args.update,
        checkpoint_root=checkpoint_root,
        source_phase_cache_receipt=args.source_phase_cache_receipt,
    )
    payload = {
        "schema_version": "apertus_hard_h_to_g_checkpoint_reference_v1",
        "status": "passed",
        "scale": args.scale,
        "update": args.update,
        "load_root": str(load_root),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_permit": file_binding(args.checkpoint_permit),
        "source_phase_cache_receipt": file_binding(args.source_phase_cache_receipt),
        "gracefully_stopped": False,
        "source_kind": "audited_intermediate_save",
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
