#!/usr/bin/env python3
"""Issue a segment checkpoint permit only from a complete state audit."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache
from producer_bundle_compatibility import require_accepted_producer

REQUIRED_CHECKS = (
    "model_state_metadata_complete",
    "optimizer_state_metadata_complete",
    "optimizer_model_space_metadata_complete",
    "rng_state_complete",
    "scheduler_state_complete",
    "data_cursor_verified",
    "training_log_no_nonfinite_updates",
    "checkpoint_storage_inventory_verified",
    "checkpoint_files_read_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--source-phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--source-phase-cache-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_permit(
    value: dict[str, object],
    *,
    scale: str,
    source_phase: int,
    update: int,
    checkpoint_root: Path,
    source_phase_cache_receipt: Path,
    accepted_producers: set[tuple[str, str, str, int, str]] | None = None,
) -> None:
    require(value.get("schema_version") == "apertus_hard_h_to_g_checkpoint_permit_v2", "checkpoint permit schema drift")
    require(value.get("status") == "passed", "checkpoint permit is not passing")
    require(value.get("scale") == scale and value.get("update") == update, "checkpoint permit scale/update drift")
    require(value.get("source_phase") == source_phase, "checkpoint permit source-phase drift")
    require(Path(str(value.get("checkpoint_root", ""))).resolve() == checkpoint_root.resolve(), "checkpoint permit root drift")
    load_root = checkpoint_root.resolve().parent
    require(Path(str(value.get("load_root", ""))).resolve() == load_root, "checkpoint permit load-root drift")
    tracker = load_root / "latest_checkpointed_iteration.txt"
    require(tracker.is_file() and tracker.read_text(encoding="utf-8").strip() == str(update), "checkpoint permit load-root tracker drift")
    require(value.get("load_tracker") == file_binding(tracker), "checkpoint permit load tracker binding drift")
    phase_binding = value.get("source_phase_cache_receipt")
    require(
        isinstance(phase_binding, dict) and phase_binding == file_binding(source_phase_cache_receipt),
        "checkpoint permit source-phase-cache binding drift",
    )
    checks = value.get("checks")
    require(isinstance(checks, dict) and set(checks) == set(REQUIRED_CHECKS) and all(checks[name] is True for name in REQUIRED_CHECKS), "checkpoint permit checks drift")
    code = value.get("executing_code_bundle")
    current = executing_code_bundle()
    exact_code = (
        isinstance(code, dict)
        and code.get("root") == current["root"]
        and code.get("tree_sha256") == current["tree_sha256"]
    )
    if not exact_code:
        require(
            accepted_producers is not None,
            "checkpoint permit code-bundle drift",
        )
        require_accepted_producer(
            value,
            accepted_producers,
            "checkpoint permit",
        )


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable checkpoint permit exists: {args.output}")
    root = args.checkpoint_root.resolve()
    require(root.is_dir(), "checkpoint root missing")
    require(root.name == f"iter_{args.update:07d}", "checkpoint iteration-directory name drift")
    load_root = root.parent
    tracker = load_root / "latest_checkpointed_iteration.txt"
    require(tracker.is_file() and tracker.read_text(encoding="utf-8").strip() == str(args.update), "checkpoint load-root tracker drift")
    source_cache = read_json(args.source_phase_cache_receipt)
    source_data_path_spec = Path(str(source_cache.get("data_path_spec", {}).get("path", "")))
    source_cache_root = Path(str(source_cache.get("cache_root", "")))
    validate_phase_cache(
        source_cache,
        phase=args.source_phase,
        data_path_spec=source_data_path_spec,
        cache_root=source_cache_root,
    )
    audit = read_json(args.checkpoint_audit)
    require(audit.get("schema_version") == "apertus_hard_h_to_g_checkpoint_state_audit_v1", "checkpoint audit schema drift")
    require(audit.get("status") == "passed", "checkpoint state audit is not passing")
    require(audit.get("scale") == args.scale and int(audit.get("update", -1)) == args.update, "checkpoint audit scale/update drift")
    require(audit.get("source_phase") == args.source_phase, "checkpoint audit source-phase drift")
    require(Path(str(audit.get("checkpoint_root", ""))).resolve() == root, "checkpoint audit root drift")
    require(
        audit.get("source_phase_cache_receipt") == file_binding(args.source_phase_cache_receipt),
        "checkpoint audit source-phase-cache binding drift",
    )
    checks = audit.get("checks")
    require(isinstance(checks, dict) and all(checks.get(name) is True for name in REQUIRED_CHECKS), "checkpoint audit is incomplete")
    audit_code = audit.get("executing_code_bundle")
    current_code = executing_code_bundle()
    require(
        isinstance(audit_code, dict)
        and audit_code.get("root") == current_code["root"]
        and audit_code.get("tree_sha256") == current_code["tree_sha256"],
        "checkpoint audit code-bundle drift",
    )
    payload: dict[str, object] = {
        "schema_version": "apertus_hard_h_to_g_checkpoint_permit_v2",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scale": args.scale,
        "source_phase": args.source_phase,
        "update": args.update,
        "checkpoint_root": str(root),
        "load_root": str(load_root),
        "load_tracker": file_binding(tracker),
        "checkpoint_audit": file_binding(args.checkpoint_audit),
        "source_phase_cache_receipt": file_binding(args.source_phase_cache_receipt),
        "checks": {name: True for name in REQUIRED_CHECKS},
        "executing_code_bundle": executing_code_bundle(),
    }
    validate_permit(
        payload,
        scale=args.scale,
        source_phase=args.source_phase,
        update=args.update,
        checkpoint_root=root,
        source_phase_cache_receipt=args.source_phase_cache_receipt,
    )
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "scale": args.scale, "update": args.update}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
