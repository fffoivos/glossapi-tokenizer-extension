#!/usr/bin/env python3
"""Adopt an immutable phase cache or seed a private qualification overlay."""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from freeze_phase_blend_cache import (
    compare_cache_inventory,
)
from freeze_phase_blend_cache import (
    validate_receipt as validate_phase_cache,
)
from producer_bundle_compatibility import load_authority, require_accepted_producer

OVERLAY_SCHEMA = "apertus_hard_h_to_g_phase_cache_overlay_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("canonical", "qualification-overlay"), required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def clone_declared_files(
    source_root: Path,
    destination_root: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    require(not destination_root.exists(), f"cache destination exists: {destination_root}")
    destination_root.mkdir(parents=True)
    methods: dict[str, int] = {"hardlink": 0, "byte_copy": 0}
    for row in rows:
        relative = Path(str(row.get("relative_path", "")))
        require(str(relative) and not relative.is_absolute() and ".." not in relative.parts, "cache row escapes its root")
        source = source_root / relative
        destination = destination_root / relative
        require(source.is_file(), f"declared cache seed missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
            methods["hardlink"] += 1
        except OSError as error:
            require(error.errno == errno.EXDEV, f"cache seed hardlink failed: {source}: {error}")
            shutil.copyfile(source, destination)
            methods["byte_copy"] += 1
        destination.chmod(destination.stat().st_mode & ~0o222)
    drift = compare_cache_inventory(destination_root, rows)
    require(not drift["missing_relative_paths"], "materialized cache is missing declared files")
    require(not drift["changed_relative_paths"], "materialized cache changed declared files")
    require(int(drift["added_file_count"]) == 0, "materialized cache contains undeclared files")
    return {
        "methods": methods,
        "seed_file_count": len(rows),
        "seed_bytes": sum(int(row["bytes"]) for row in rows),
    }


def compact_drift(value: dict[str, object]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "expected_file_count",
            "observed_file_count",
            "expected_tree_sha256",
            "observed_tree_sha256",
            "missing_relative_paths",
            "changed_relative_paths",
            "added_file_count",
            "added_bytes",
            "added_tree_sha256",
            "added_relative_paths",
        )
    }


def validate_overlay_receipt(
    value: dict[str, Any],
    *,
    phase: int,
    overlay_root: Path,
    accepted_code_bundles: set[tuple[str, str]] | None = None,
    require_pristine: bool,
) -> dict[str, object]:
    require(value.get("schema_version") == OVERLAY_SCHEMA, "phase cache overlay schema drift")
    require(value.get("status") == "seeded" and value.get("phase") == phase, "phase cache overlay status/phase drift")
    require(Path(str(value.get("overlay_root", ""))).resolve() == overlay_root.resolve(), "phase cache overlay-root drift")
    source_binding = value.get("source_cache_receipt")
    require(isinstance(source_binding, dict), "phase cache overlay source binding missing")
    source_path = Path(str(source_binding.get("path", "")))
    require(source_path.is_file() and source_binding == file_binding(source_path), "phase cache overlay source binding drift")
    source = read_json(source_path)
    validate_phase_cache(
        source,
        phase=phase,
        data_path_spec=Path(str(source.get("data_path_spec", {}).get("path", ""))),
        cache_root=Path(str(source.get("cache_root", ""))),
        accepted_code_bundles=accepted_code_bundles,
    )
    require(value.get("source_cache_tree_sha256") == source.get("cache_tree_sha256"), "phase cache overlay source tree drift")
    require(value.get("seed_files") == source.get("cache_files"), "phase cache overlay seed inventory drift")
    current = executing_code_bundle()
    bundle = value.get("executing_code_bundle")
    observed_bundle = (
        str(Path(str(bundle.get("root", ""))).resolve()) if isinstance(bundle, dict) else "",
        str(bundle.get("tree_sha256", "")) if isinstance(bundle, dict) else "",
    )
    allowed = accepted_code_bundles or {
        (str(Path(str(current["root"])).resolve()), str(current["tree_sha256"]))
    }
    require(observed_bundle in allowed, "phase cache overlay code-bundle drift")
    rows = source.get("cache_files")
    require(isinstance(rows, list), "phase cache overlay source inventory missing")
    drift = compare_cache_inventory(overlay_root, rows)
    require(not drift["missing_relative_paths"] and not drift["changed_relative_paths"], "phase cache overlay seed drift")
    if require_pristine:
        require(int(drift["added_file_count"]) == 0, "phase cache overlay was used before preflight")
    for row in rows:
        path = overlay_root / str(row["relative_path"])
        require(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0, "phase cache overlay seed is writable")
    require(overlay_root.stat().st_mode & stat.S_IWUSR != 0, "phase cache overlay root is not owner-writable")
    return compact_drift(drift)


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable cache materialization receipt exists: {args.output}")
    source_path = args.source_receipt.resolve()
    source = read_json(source_path)
    current = executing_code_bundle()
    _, accepted_producers = load_authority(args.producer_compatibility, current)
    accepted_code_bundles = {(root, tree) for root, tree, *_ in accepted_producers}
    require_accepted_producer(source, accepted_producers, "phase cache materialization source")
    source_root = Path(str(source.get("cache_root", ""))).resolve()
    source_drift = validate_phase_cache(
        source,
        phase=args.phase,
        data_path_spec=Path(str(source.get("data_path_spec", {}).get("path", ""))),
        cache_root=source_root,
        accepted_code_bundles=accepted_code_bundles,
        allow_undeclared_files=args.role == "canonical",
    )
    rows = source.get("cache_files")
    require(isinstance(rows, list), "phase cache source inventory missing")
    destination = args.destination_root.resolve()
    require(destination != source_root, "cache materialization cannot reuse its source root")
    clone = clone_declared_files(source_root, destination, rows)
    materialization = {
        "role": "canonical_immutable" if args.role == "canonical" else "qualification_overlay",
        "source_cache_receipt": file_binding(source_path),
        "source_live_inventory_at_adoption": compact_drift(source_drift),
        **clone,
    }
    if args.role == "canonical":
        destination.chmod(destination.stat().st_mode & ~0o222)
        payload = dict(source)
        payload.update(
            {
                "status": "frozen",
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "cache_root": str(destination),
                "materialization": materialization,
                "executing_code_bundle": current,
            }
        )
        validate_phase_cache(
            payload,
            phase=args.phase,
            data_path_spec=Path(str(payload.get("data_path_spec", {}).get("path", ""))),
            cache_root=destination,
        )
    else:
        payload = {
            "schema_version": OVERLAY_SCHEMA,
            "status": "seeded",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "phase": args.phase,
            "source_cache_receipt": file_binding(source_path),
            "source_cache_tree_sha256": source.get("cache_tree_sha256"),
            "overlay_root": str(destination),
            "seed_files": rows,
            "materialization": materialization,
            "executing_code_bundle": current,
        }
        validate_overlay_receipt(
            payload,
            phase=args.phase,
            overlay_root=destination,
            require_pristine=True,
        )
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
