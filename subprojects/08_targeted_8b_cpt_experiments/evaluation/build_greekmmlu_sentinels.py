#!/usr/bin/env python3
"""Build deterministic nested GreekMMLU sentinel panels."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic  # noqa: E402


SALT = "greekmmlu-sentinel-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--binding-examples",
        type=Path,
        help="final immutable examples path to record when outputs are built in staging",
    )
    parser.add_argument(
        "--binding-output-dir",
        type=Path,
        help="final immutable output directory to record when outputs are built in staging",
    )
    parser.add_argument("--sizes", default="4096,8192")
    return parser.parse_args()


def stable_digest(example_id: str) -> str:
    payload = SALT.encode("utf-8") + bytes([0]) + example_id.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_stratum_digest(stratum: str) -> str:
    payload = SALT.encode("utf-8") + b"\0stratum\0" + stratum.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stratum_key(row: dict[str, Any], use_level: bool) -> str:
    subject = str(row["subject"])
    return f"{subject}\u241f{row['educational_level']}" if use_level else subject


def hamilton(capacities: dict[str, int], slots: int) -> dict[str, int]:
    require(slots >= 0, "negative allocation")
    require(slots <= sum(capacities.values()), "allocation exceeds capacity")
    if slots == 0:
        return {key: 0 for key in capacities}
    total = sum(capacities.values())
    exact = {key: slots * value / total for key, value in capacities.items()}
    result = {key: min(capacities[key], int(exact[key])) for key in capacities}
    remaining = slots - sum(result.values())
    order = sorted(
        capacities,
        key=lambda key: (
            -(exact[key] - int(exact[key])),
            stable_stratum_digest(key),
            key,
        ),
    )
    for key in order:
        if remaining == 0:
            break
        if result[key] < capacities[key]:
            result[key] += 1
            remaining -= 1
    require(remaining == 0, "Hamilton allocation did not close")
    return result


def select_nested(
    rows: list[dict[str, Any]], sizes: list[int]
) -> tuple[dict[int, list[dict[str, Any]]], bool]:
    require(sizes == sorted(set(sizes)) and sizes[0] > 0, "sizes must be unique and increasing")
    require(sizes[-1] <= len(rows), "largest sentinel exceeds clean panel")
    ids = [str(row["example_id"]) for row in rows]
    require(len(ids) == len(set(ids)), "duplicate example ids")
    level_count = sum(bool(row.get("educational_level")) for row in rows)
    use_level = level_count == len(rows)
    prepared = [
        {
            **row,
            "selection_sha256": stable_digest(str(row["example_id"])),
            "stratum": stratum_key(row, use_level),
        }
        for row in rows
    ]
    prepared.sort(key=lambda row: (row["selection_sha256"], row["example_id"]))
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        by_subject[str(row["subject"])].append(row)
    subjects = sorted(by_subject)
    require(sizes[0] >= len(subjects), "smallest sentinel cannot represent every subject")

    # A stable per-subject floor is selected once. Each larger panel then adds
    # an allocation from only the unselected rows. This is an incremental
    # Hamilton construction, so an already selected example can never be
    # removed by an apportionment change at a larger target size.
    selected: dict[str, dict[str, Any]] = {
        str(by_subject[subject][0]["example_id"]): by_subject[subject][0]
        for subject in subjects
    }
    outputs: dict[int, list[dict[str, Any]]] = {}
    for size in sizes:
        remaining_rows = [row for row in prepared if str(row["example_id"]) not in selected]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in remaining_rows:
            grouped[str(row["stratum"])].append(row)
        needed = size - len(selected)
        allocation = hamilton({key: len(value) for key, value in grouped.items()}, needed)
        for key in sorted(grouped):
            for row in grouped[key][: allocation[key]]:
                selected[str(row["example_id"])] = row
        require(len(selected) == size, f"sentinel size did not close: {size}")
        output = sorted(selected.values(), key=lambda row: (row["selection_sha256"], row["example_id"]))
        require({row["subject"] for row in output} == set(subjects), "subject coverage drift")
        outputs[size] = output
    for left, right in zip(sizes, sizes[1:]):
        require(
            {row["example_id"] for row in outputs[left]}
            < {row["example_id"] for row in outputs[right]},
            "sentinels are not strictly nested",
        )
    return outputs, use_level


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    require(not path.exists(), f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def file_binding_at(source: Path, declared_path: Path) -> dict[str, Any]:
    """Bind immutable bytes at their final path while they are still staged."""

    observed = file_binding(source)
    return {**observed, "path": str(declared_path.resolve())}


def main() -> int:
    args = parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if value]
    source = read_json(args.examples)
    require(source.get("schema_version") == "apertus_greekmmlu_clean_examples_v1", "example schema drift")
    require(source.get("status") == "frozen", "examples are not frozen")
    current_bundle = executing_code_bundle()
    source_bundle = source.get("executing_code_bundle")
    require(
        isinstance(source_bundle, dict)
        and source_bundle.get("root") == current_bundle["root"]
        and source_bundle.get("tree_sha256") == current_bundle["tree_sha256"],
        "clean-example code-bundle drift",
    )
    rows = source.get("examples")
    require(isinstance(rows, list), "examples missing")
    outputs, use_level = select_nested(rows, sizes)
    binding_examples = args.binding_examples or args.examples
    binding_output_dir = args.binding_output_dir or args.output_dir
    require(
        binding_examples.name == args.examples.name,
        "declared examples filename differs from staged examples filename",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}
    for size in sizes:
        path = args.output_dir / f"greekmmlu_sentinel_{size}.jsonl"
        write_jsonl_atomic(path, outputs[size])
        paths[str(size)] = file_binding_at(
            path, binding_output_dir / path.name
        )
    manifest_path = args.output_dir / "sentinel_manifest.json"
    payload = {
        "schema_version": "apertus_greekmmlu_sentinel_manifest_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": current_bundle,
        "source_examples": file_binding_at(args.examples, binding_examples),
        "salt": SALT,
        "separator": "single_nul_byte",
        "algorithm": "stable_subject_floor_then_incremental_remaining_capacity_hamilton_with_sha256_ties",
        "nesting_rule": "larger_panels_only_add_rows_and_never_reapportion_selected_rows",
        "stratification": "subject_x_educational_level" if use_level else "subject",
        "educational_level_fallback_used": not use_level,
        "sizes": sizes,
        "panels": paths,
        "subject_counts": {
            str(size): dict(sorted(Counter(row["subject"] for row in outputs[size]).items()))
            for size in sizes
        },
        "strictly_nested": True,
        "selection_authorized": False,
        "authorization_requires": "same_stack_calibration_receipt",
    }
    write_json_atomic(manifest_path, payload)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
