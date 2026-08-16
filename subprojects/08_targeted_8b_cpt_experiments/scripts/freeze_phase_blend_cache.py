#!/usr/bin/env python3
"""Freeze a phase-local binary, blend authority and GPTDataset cache inventory."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import stat
from decimal import Decimal
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    sha256_file,
    write_json_atomic,
)

PHASE_START = {1: 0, 2: 2261, 3: 3218}
PHASE_DATASET_SAMPLES = {1: 3_295_898, 2: 3_295_898, 3: 487_424}
RUNTIME_VALIDATION_SAMPLES = 132_096
RUNTIME_TEST_SAMPLES = 1_024
RUNTIME_DATASET_BUILDER_THREADS = 4
PHASE3_COMPONENT_REQUESTED_SAMPLES = {
    "active_modern": 386_991,
    "foreign_replay": 97_973,
    "old_greek_replay": 4_899,
}
EXPECTED_COMPONENTS = (
    ("active_modern", Decimal("1.0")),
    ("foreign_replay", Decimal("0.253164557")),
    ("old_greek_replay", Decimal("0.012658228")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--data-path-spec", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--blend-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tree_inventory(root: Path) -> tuple[list[dict[str, object]], str]:
    require(root.is_dir(), f"cache root missing: {root}")
    rows: list[dict[str, object]] = []
    tree = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        digest = sha256_file(path)
        row = {"relative_path": relative, "bytes": path.stat().st_size, "sha256": digest}
        rows.append(row)
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(row["bytes"]).encode("ascii"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")
    require(bool(rows), "GPTDataset cache is empty")
    return rows, tree.hexdigest()


def inventory_tree_sha256(rows: list[dict[str, object]]) -> str:
    tree = hashlib.sha256()
    for row in rows:
        relative = str(row.get("relative_path", ""))
        require(relative and not Path(relative).is_absolute(), "cache inventory path is not relative")
        require(".." not in Path(relative).parts, "cache inventory path escapes its root")
        size = int(row.get("bytes", -1))
        digest = str(row.get("sha256", ""))
        require(size >= 0 and len(digest) == 64, "cache inventory row is malformed")
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(size).encode("ascii"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")
    require(bool(rows), "GPTDataset cache inventory is empty")
    return tree.hexdigest()


def compare_cache_inventory(
    root: Path,
    expected_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Classify cache drift without deleting or accepting undeclared files."""

    expected_paths = [str(row.get("relative_path", "")) for row in expected_rows]
    require(len(expected_paths) == len(set(expected_paths)), "duplicate cache inventory path")
    expected_tree = inventory_tree_sha256(expected_rows)
    observed_rows, observed_tree = tree_inventory(root)
    expected = {str(row["relative_path"]): row for row in expected_rows}
    observed = {str(row["relative_path"]): row for row in observed_rows}
    missing = sorted(set(expected) - set(observed))
    changed = sorted(
        relative
        for relative in set(expected) & set(observed)
        if expected[relative] != observed[relative]
    )
    added_rows = [row for row in observed_rows if str(row["relative_path"]) not in expected]
    return {
        "expected_file_count": len(expected_rows),
        "observed_file_count": len(observed_rows),
        "expected_tree_sha256": expected_tree,
        "observed_tree_sha256": observed_tree,
        "missing_relative_paths": missing,
        "changed_relative_paths": changed,
        "added_file_count": len(added_rows),
        "added_bytes": sum(int(row["bytes"]) for row in added_rows),
        "added_tree_sha256": inventory_tree_sha256(added_rows) if added_rows else None,
        "added_relative_paths": [str(row["relative_path"]) for row in added_rows],
    }


def require_read_only(path: Path) -> None:
    require(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0, f"asset remains writable: {path}")


def validate_data_path_spec(
    value: dict[str, object],
    phase: int,
    *,
    verify_payload_hashes: bool = True,
) -> tuple[list[str], list[Path]]:
    require(value.get("schema_version") == "apertus_hard_h_to_g_phase_data_path_v1", "phase data-path schema drift")
    require(value.get("status") == "frozen" and value.get("phase") == phase, "phase data-path status/phase drift")
    components = value.get("components")
    require(isinstance(components, list) and len(components) == len(EXPECTED_COMPONENTS), "phase data-path component count drift")
    tokens: list[str] = []
    prefixes: list[Path] = []
    for row, (role, expected_weight) in zip(components, EXPECTED_COMPONENTS):
        require(isinstance(row, dict) and row.get("role") == role, "phase data-path role/order drift")
        weight = str(row.get("weight", ""))
        require(Decimal(weight) == expected_weight, f"phase data-path weight drift for {role}")
        prefix = Path(str(row.get("prefix", ""))).resolve()
        require(Path(f"{prefix}.bin").is_file() and Path(f"{prefix}.idx").is_file(), f"phase component payload missing: {prefix}")
        files = row.get("files")
        require(isinstance(files, list) and len(files) == 2, f"phase component file binding missing for {role}")
        for binding, path in zip(files, (Path(f"{prefix}.bin").resolve(), Path(f"{prefix}.idx").resolve())):
            require(isinstance(binding, dict), f"phase component file binding malformed for {role}")
            require(Path(str(binding.get("path", ""))).resolve() == path, f"phase component file path drift for {role}")
            if verify_payload_hashes:
                require(file_binding(path) == binding, f"phase component payload binding drift for {role}")
            else:
                require(
                    path.is_file() and path.stat().st_size == int(binding.get("bytes", -1)),
                    f"phase component payload size drift for {role}: {path}",
                )
        tokenized_binding = row.get("tokenized_receipt")
        require(isinstance(tokenized_binding, dict), f"phase component tokenized receipt missing for {role}")
        tokenized_path = Path(str(tokenized_binding.get("path", "")))
        require(tokenized_path.is_file() and tokenized_binding == file_binding(tokenized_path), f"phase component tokenized receipt binding drift for {role}")
        tokenized = read_json(tokenized_path)
        require(tokenized.get("schema_version") == "apertus_hard_h_to_g_tokenized_stream_v1" and tokenized.get("status") == "frozen", f"phase component tokenized receipt drift for {role}")
        expected_stream = (
            ("hplt" if phase == 1 else "openarchives")
            if role == "active_modern" and phase < 3
            else ("phase3_openarchives" if role == "active_modern" else "phase3_foreign" if role == "foreign_replay" else "phase3_old_greek")
            if phase == 3
            else ("foreign" if role == "foreign_replay" else "old_greek")
        )
        require(tokenized.get("stream") == expected_stream, f"phase component stream drift for {role}")
        require(Path(str(tokenized.get("dataset_prefix", ""))).resolve() == prefix, f"phase component prefix/receipt drift for {role}")
        require([tokenized.get("files", {}).get("bin"), tokenized.get("files", {}).get("idx")] == files, f"phase component payload/receipt drift for {role}")
        tokens.extend((weight, str(prefix)))
        prefixes.append(prefix)
    require(value.get("data_path_tokens") == tokens, "phase data-path token sequence drift")
    require(value.get("data_path_shell_string") == " ".join(tokens), "phase data-path shell-string drift")
    return tokens, prefixes


def require_compatible_code_bundle(
    value: object,
    accepted_code_bundles: set[tuple[str, str]] | None = None,
) -> None:
    require(isinstance(value, dict), "phase cache code-bundle binding missing")
    observed = (
        str(Path(str(value.get("root", ""))).resolve()),
        str(value.get("tree_sha256", "")),
    )
    if accepted_code_bundles is None:
        current = executing_code_bundle()
        accepted_code_bundles = {
            (str(Path(str(current["root"])).resolve()), str(current["tree_sha256"]))
        }
    require(observed in accepted_code_bundles, "phase cache code-bundle drift")


def validate_receipt(
    value: dict[str, object],
    *,
    phase: int,
    data_path_spec: Path,
    cache_root: Path,
    accepted_code_bundles: set[tuple[str, str]] | None = None,
    allow_undeclared_files: bool = False,
    verify_payload_hashes: bool = True,
) -> dict[str, object]:
    require(value.get("schema_version") == "apertus_hard_h_to_g_phase_blend_cache_v1", "phase cache schema drift")
    require(value.get("status") == "frozen", "phase cache is not frozen")
    require(value.get("phase") == phase and value.get("phase_start_update") == PHASE_START[phase], "phase cache boundary drift")
    require(value.get("data_path_spec") == file_binding(data_path_spec), "phase data-path-spec binding drift")
    require(Path(str(value.get("cache_root", ""))).resolve() == cache_root.resolve(), "phase cache-root binding drift")
    spec = read_json(data_path_spec)
    data_path_tokens, prefixes = validate_data_path_spec(
        spec,
        phase,
        verify_payload_hashes=verify_payload_hashes,
    )
    require(value.get("data_path_tokens") == data_path_tokens, "phase cache data-path token drift")
    megatron_binding = value.get("megatron_receipt")
    require(isinstance(megatron_binding, dict), "phase cache Megatron receipt binding missing")
    megatron_path = Path(str(megatron_binding.get("path", "")))
    require(megatron_path.is_file() and file_binding(megatron_path) == megatron_binding, "phase cache Megatron receipt binding drift")
    megatron = read_json(megatron_path)
    require(
        megatron.get("status") == "frozen"
        and isinstance(megatron.get("dataset_helpers"), dict)
        and megatron["dataset_helpers"].get("import_smoke_passed") is True,
        "phase cache compiled Megatron dataset-helper authority missing",
    )
    require(
        Path(str(value.get("megatron_root", ""))).resolve()
        == Path(str(megatron.get("output_root", ""))).resolve(),
        "phase cache Megatron root drift",
    )
    require(
        value.get("cache_build_process_group")
        == {
            "backend": "gloo",
            "rank": 0,
            "world_size": 1,
            "started_by_builder": True,
            "destroyed_after_build": True,
        },
        "phase cache single-rank process-group proof missing or drifted",
    )
    construction = value.get("gptdataset_construction")
    require(isinstance(construction, dict), "GPTDataset construction contract missing")
    require(
        construction == {
            "data_seed": 20260609,
            "sequence_length": 4096,
            "curriculum_order_mode": "randomized",
            "megatron_gpt_dataset_no_shuffle": 0,
            "phase_local_cursor_samples": 0,
            "dataset_requested_samples": PHASE_DATASET_SAMPLES[phase],
            "runtime_target_sizes": [
                PHASE_DATASET_SAMPLES[phase],
                RUNTIME_VALIDATION_SAMPLES,
                RUNTIME_TEST_SAMPLES,
            ],
            "runtime_dataset_builder_threads": RUNTIME_DATASET_BUILDER_THREADS,
            "dataset_horizon_policy": "historical_full_horizon" if phase < 3 else "phase_local_extension_horizon",
        },
        "GPTDataset construction drift",
    )
    if phase == 3:
        require(
            value.get("phase3_component_requested_samples") == PHASE3_COMPONENT_REQUESTED_SAMPLES
            and value.get("phase3_component_built_samples") == PHASE3_COMPONENT_REQUESTED_SAMPLES,
            "Phase-3 frozen component sample geometry drift",
        )
        no_wrap = value.get("phase3_no_epoch_wrap")
        require(isinstance(no_wrap, dict) and no_wrap.get("passed") is True, "Phase-3 frozen no-wrap proof missing")
    require_compatible_code_bundle(value.get("executing_code_bundle"), accepted_code_bundles)
    data_rows = value.get("data_files")
    require(isinstance(data_rows, list) and len(data_rows) == len(prefixes) * 2, "phase binary inventory drift")
    expected_files = [Path(f"{prefix}{suffix}").resolve() for prefix in prefixes for suffix in (".bin", ".idx")]
    expected_bindings = [binding for component in spec["components"] for binding in component["files"]]
    require(data_rows == expected_bindings, "phase component-file inventory drift")
    for path in expected_files:
        require_read_only(path)
    declared_cache = value.get("cache_files")
    require(isinstance(declared_cache, list), "phase cache file inventory missing")
    if verify_payload_hashes:
        drift = compare_cache_inventory(cache_root, declared_cache)
        require(drift["expected_tree_sha256"] == value.get("cache_tree_sha256"), "phase cache declared tree drift")
        require(not drift["missing_relative_paths"] and not drift["changed_relative_paths"], "live GPTDataset declared cache drift")
        if not allow_undeclared_files:
            require(int(drift["added_file_count"]) == 0, "live GPTDataset cache drift")
            require(drift["observed_tree_sha256"] == value.get("cache_tree_sha256"), "live GPTDataset cache tree drift")
    else:
        require(
            inventory_tree_sha256(declared_cache) == value.get("cache_tree_sha256"),
            "phase cache declared tree drift",
        )
        expected = {
            str(row["relative_path"]): int(row["bytes"])
            for row in declared_cache
        }
        observed = {
            str(path.relative_to(cache_root)): path.stat().st_size
            for path in cache_root.rglob("*")
            if path.is_file()
        }
        missing = sorted(set(expected) - set(observed))
        changed = sorted(path for path in set(expected) & set(observed) if expected[path] != observed[path])
        added = sorted(set(observed) - set(expected))
        require(not missing and not changed, "live GPTDataset declared cache metadata drift")
        if not allow_undeclared_files:
            require(not added, "live GPTDataset undeclared cache-file drift")
        drift = {
            "verification": "metadata_only_against_frozen_sha256_receipt",
            "expected_file_count": len(expected),
            "observed_file_count": len(observed),
            "expected_tree_sha256": value.get("cache_tree_sha256"),
            "observed_tree_sha256": None,
            "missing_relative_paths": missing,
            "changed_relative_paths": changed,
            "added_file_count": len(added),
            "added_bytes": sum(observed[path] for path in added),
            "added_tree_sha256": None,
            "added_relative_paths": added,
        }
    for row in declared_cache:
        require_read_only(cache_root / str(row["relative_path"]))
    materialization = value.get("materialization")
    if isinstance(materialization, dict) and materialization.get("role") == "canonical_immutable":
        require_read_only(cache_root)
    return drift


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable phase-cache receipt exists: {args.output}")
    data_path_spec = args.data_path_spec.resolve()
    cache_root = args.cache_root.resolve()
    spec = read_json(data_path_spec)
    data_path_tokens, prefixes = validate_data_path_spec(spec, args.phase)
    data_files = [Path(f"{prefix}{suffix}").resolve() for prefix in prefixes for suffix in (".bin", ".idx")]
    blend = read_json(args.blend_manifest)
    require(blend.get("schema_version") == "apertus_hard_h_to_g_phase_cache_build_v1", "cache-build receipt schema drift")
    require(blend.get("status") == "passed", "cache-build receipt is not passing")
    require(int(blend.get("phase", -1)) == args.phase, "blend-manifest phase drift")
    require(blend.get("requested_samples") == PHASE_DATASET_SAMPLES[args.phase], "cache-build sample horizon drift")
    require(
        blend.get("runtime_target_sizes")
        == [PHASE_DATASET_SAMPLES[args.phase], RUNTIME_VALIDATION_SAMPLES, RUNTIME_TEST_SAMPLES]
        and blend.get("runtime_dataset_builder_threads") == RUNTIME_DATASET_BUILDER_THREADS,
        "cache-build production target-size/thread geometry drift",
    )
    require(blend.get("data_path_spec") == file_binding(data_path_spec), "cache-build data-path binding drift")
    require(Path(str(blend.get("cache_root", ""))).resolve() == cache_root, "cache-build root drift")
    megatron_receipt = blend.get("megatron_receipt")
    require(isinstance(megatron_receipt, dict), "cache-build Megatron receipt binding missing")
    megatron_receipt_path = Path(str(megatron_receipt.get("path", "")))
    require(
        megatron_receipt_path.is_file() and file_binding(megatron_receipt_path) == megatron_receipt,
        "cache-build Megatron receipt binding drift",
    )
    process_group = blend.get("cache_build_process_group")
    require(
        process_group
        == {
            "backend": "gloo",
            "rank": 0,
            "world_size": 1,
            "started_by_builder": True,
            "destroyed_after_build": True,
        },
        "cache-build single-rank process-group proof missing or drifted",
    )
    if args.phase == 3:
        no_wrap = blend.get("phase3_no_epoch_wrap")
        require(isinstance(no_wrap, dict) and no_wrap.get("passed") is True, "Phase-3 cache wraps/repeats documents")
        require(
            blend.get("component_requested_samples") == PHASE3_COMPONENT_REQUESTED_SAMPLES
            and blend.get("component_built_samples") == PHASE3_COMPONENT_REQUESTED_SAMPLES,
            "Phase-3 component construction margin/sample geometry drift",
        )
    cache_files, cache_tree = tree_inventory(cache_root)
    # The debug build freezes permissions before the receipt is accepted.
    for path in [*data_files, *(value for value in cache_root.rglob("*") if value.is_file())]:
        path.chmod(path.stat().st_mode & ~0o222)
    payload: dict[str, object] = {
        "schema_version": "apertus_hard_h_to_g_phase_blend_cache_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": args.phase,
        "phase_start_update": PHASE_START[args.phase],
        "data_path_spec": file_binding(data_path_spec),
        "data_path_tokens": data_path_tokens,
        "data_files": [binding for component in spec["components"] for binding in component["files"]],
        "blend_manifest": file_binding(args.blend_manifest),
        "megatron_root": blend.get("megatron_root"),
        "megatron_receipt": megatron_receipt,
        "cache_build_process_group": process_group,
        "cache_root": str(cache_root),
        "cache_files": cache_files,
        "cache_tree_sha256": cache_tree,
        "gptdataset_construction": {
            "data_seed": 20260609,
            "sequence_length": 4096,
            "curriculum_order_mode": "randomized",
            "megatron_gpt_dataset_no_shuffle": 0,
            "phase_local_cursor_samples": 0,
            "dataset_requested_samples": PHASE_DATASET_SAMPLES[args.phase],
            "runtime_target_sizes": [
                PHASE_DATASET_SAMPLES[args.phase],
                RUNTIME_VALIDATION_SAMPLES,
                RUNTIME_TEST_SAMPLES,
            ],
            "runtime_dataset_builder_threads": RUNTIME_DATASET_BUILDER_THREADS,
            "dataset_horizon_policy": "historical_full_horizon" if args.phase < 3 else "phase_local_extension_horizon",
        },
        "phase3_component_requested_samples": PHASE3_COMPONENT_REQUESTED_SAMPLES if args.phase == 3 else None,
        "phase3_component_built_samples": blend.get("component_built_samples") if args.phase == 3 else None,
        "phase3_no_epoch_wrap": blend.get("phase3_no_epoch_wrap") if args.phase == 3 else None,
        "executing_code_bundle": executing_code_bundle(),
    }
    validate_receipt(payload, phase=args.phase, data_path_spec=data_path_spec, cache_root=cache_root)
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "phase": args.phase, "cache_tree_sha256": cache_tree}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
