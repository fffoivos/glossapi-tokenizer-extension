#!/usr/bin/env python3
"""Freeze the exact ordered weighted Megatron --data-path for one phase."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_phase_blend_cache import EXPECTED_COMPONENTS, validate_data_path_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--active-modern-prefix", type=Path, required=True)
    parser.add_argument("--active-modern-receipt", type=Path, required=True)
    parser.add_argument("--foreign-replay-prefix", type=Path, required=True)
    parser.add_argument("--foreign-replay-receipt", type=Path, required=True)
    parser.add_argument("--old-greek-replay-prefix", type=Path, required=True)
    parser.add_argument("--old-greek-replay-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable phase data-path spec exists: {args.output}")
    prefixes = (
        args.active_modern_prefix.resolve(),
        args.foreign_replay_prefix.resolve(),
        args.old_greek_replay_prefix.resolve(),
    )
    receipt_paths = (
        args.active_modern_receipt.resolve(),
        args.foreign_replay_receipt.resolve(),
        args.old_greek_replay_receipt.resolve(),
    )
    components = []
    tokens: list[str] = []
    for (role, weight), prefix, receipt_path in zip(EXPECTED_COMPONENTS, prefixes, receipt_paths, strict=True):
        files = [Path(f"{prefix}.bin"), Path(f"{prefix}.idx")]
        require(all(path.is_file() for path in files), f"phase data component missing: {prefix}")
        tokenized = read_json(receipt_path)
        require(tokenized.get("schema_version") == "apertus_hard_h_to_g_tokenized_stream_v1", f"{role}: tokenized receipt schema drift")
        require(tokenized.get("status") == "frozen", f"{role}: tokenized receipt did not freeze")
        expected_stream = (
            ("hplt" if args.phase == 1 else "openarchives")
            if role == "active_modern" and args.phase < 3
            else ("phase3_openarchives" if role == "active_modern" else "phase3_foreign" if role == "foreign_replay" else "phase3_old_greek")
            if args.phase == 3
            else ("foreign" if role == "foreign_replay" else "old_greek")
        )
        require(tokenized.get("stream") == expected_stream, f"{role}: tokenized stream/phase drift")
        require(Path(str(tokenized.get("dataset_prefix", ""))).resolve() == prefix, f"{role}: tokenized prefix drift")
        expected_files = tokenized.get("files")
        require(isinstance(expected_files, dict) and [expected_files.get("bin"), expected_files.get("idx")] == [file_binding(path) for path in files], f"{role}: tokenized payload binding drift")
        row = {
            "role": role,
            "weight": str(weight),
            "prefix": str(prefix),
            "files": [file_binding(path) for path in files],
            "tokenized_receipt": file_binding(receipt_path),
        }
        components.append(row)
        tokens.extend((str(weight), str(prefix)))
    payload: dict[str, object] = {
        "schema_version": "apertus_hard_h_to_g_phase_data_path_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": args.phase,
        "components": components,
        "data_path_tokens": tokens,
        "data_path_shell_string": " ".join(tokens),
        "executing_code_bundle": executing_code_bundle(),
    }
    validate_data_path_spec(payload, args.phase)
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
