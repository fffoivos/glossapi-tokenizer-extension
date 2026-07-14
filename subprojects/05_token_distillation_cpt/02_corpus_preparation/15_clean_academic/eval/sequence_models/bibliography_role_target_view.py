#!/usr/bin/env python3
"""Join a trusted role overlay to the frozen table without rewriting silver."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import load_table
from .bibliography_role_dataset import (
    SCHEMA_VERSION_V2,
    TARGET_ENTRY_ANCHOR,
    TARGET_MASK,
    TARGET_NEGATIVE,
    load_role_contract,
    text_sha256,
)
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-role-target-view-v1"
ROLE_NAMES = (
    "UNKNOWN", "ENTRY_ANCHOR", "CONTINUATION", "FILLER", "HEADER",
    "SUBHEADER", "NON_BIB",
)
STATUS_NAMES = ("UNRESOLVED", "SINGLE_REVIEW", "PROVISIONAL", "AGREED_REVIEW", "ADJUDICATED")
BOUNDARY_NAMES = ("NONE", "SOFT_STOP", "HARD_STOP")
ROLE_TO_ID = {name: index for index, name in enumerate(ROLE_NAMES)}
STATUS_TO_ID = {name: index for index, name in enumerate(STATUS_NAMES)}
BOUNDARY_TO_ID = {name: index for index, name in enumerate(BOUNDARY_NAMES)}
TRUSTED = frozenset({"AGREED_REVIEW", "ADJUDICATED"})


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{row_number}: expected object")
            yield row


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def load_overlay(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row_number, row in enumerate(_iter_jsonl(path), 1):
        if row.get("schema_version") != SCHEMA_VERSION_V2:
            raise ValueError(f"overlay row {row_number}: expected v2")
        key = (str(row.get("document_id", "")), str(row.get("line_id", "")))
        if not all(key) or key in result:
            raise ValueError(f"overlay row {row_number}: repeated/empty identity")
        if row.get("role") not in ROLE_TO_ID or row.get("role_status") not in STATUS_TO_ID:
            raise ValueError(f"overlay row {row_number}: invalid role/status")
        if row.get("boundary_flag") not in BOUNDARY_TO_ID or row.get("boundary_status") not in STATUS_TO_ID:
            raise ValueError(f"overlay row {row_number}: invalid boundary/status")
        result[key] = row
    if not result:
        raise ValueError("role overlay is empty")
    return result


def _source_rows(path: Path, split: str) -> Iterable[Mapping[str, Any]]:
    for row in _iter_jsonl(path):
        if row.get("split") == split:
            yield row


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path, overlay_path = Path(args.input).resolve(), Path(args.overlay).resolve()
    if args.expected_input_sha256 and sha256_file(source_path) != args.expected_input_sha256:
        raise ValueError("source input SHA-256 differs from pin")
    contract = load_role_contract(args.contract)
    overlay = load_overlay(overlay_path)
    base = load_table(args.base_table_dir, expected_split=args.split)
    n = len(base.targets)
    targets = np.full(n, TARGET_MASK, dtype=np.int8)
    targets[np.isin(base.original_labels, (0, 2))] = TARGET_NEGATIVE
    role_ids = np.full(n, ROLE_TO_ID["UNKNOWN"], dtype=np.uint8)
    role_status_ids = np.full(n, STATUS_TO_ID["UNRESOLVED"], dtype=np.uint8)
    boundary_ids = np.full(n, BOUNDARY_TO_ID["NONE"], dtype=np.uint8)
    boundary_status_ids = np.full(n, STATUS_TO_ID["UNRESOLVED"], dtype=np.uint8)
    overlay_present = np.zeros(n, dtype=np.uint8)
    source_rows = list(_source_rows(source_path, args.split))
    if len(source_rows) != len(base.documents):
        raise ValueError("source/base document count mismatch")
    seen: set[tuple[str, str]] = set()
    for document_index, (source, document) in enumerate(zip(source_rows, base.documents, strict=True)):
        document_id, work_id = str(source.get("document_id", "")), str(source.get("work_id", ""))
        lines = source.get("lines")
        if (
            document_id != document["document_id"] or work_id != document["work_id"]
            or not isinstance(lines, list) or len(lines) != int(document["line_count"])
        ):
            raise ValueError(f"source/base alignment mismatch at document {document_index}")
        start = int(document["line_start"])
        for offset, line in enumerate(lines):
            line_id = str(line.get("line_id") or f"{document_id}:{int(line['abs_idx'])}")
            key = (document_id, line_id)
            reviewed = overlay.get(key)
            if reviewed is None:
                continue
            index = start + offset
            expected = {
                "work_id": work_id,
                "abs_idx": int(line["abs_idx"]),
                "text_sha256": text_sha256(str(line["text"])),
                "original_region_label": str(line["label"]),
            }
            for field, value in expected.items():
                if reviewed.get(field) != value:
                    raise ValueError(f"overlay mismatch for {key}: {field}")
            seen.add(key)
            overlay_present[index] = 1
            role, status = str(reviewed["role"]), str(reviewed["role_status"])
            role_ids[index], role_status_ids[index] = ROLE_TO_ID[role], STATUS_TO_ID[status]
            boundary_ids[index] = BOUNDARY_TO_ID[str(reviewed["boundary_flag"])]
            boundary_status_ids[index] = STATUS_TO_ID[str(reviewed["boundary_status"])]
            if status in TRUSTED:
                if role == "ENTRY_ANCHOR":
                    targets[index] = TARGET_ENTRY_ANCHOR
                elif role == "UNKNOWN":
                    targets[index] = TARGET_MASK
                else:
                    targets[index] = TARGET_NEGATIVE
    if seen != set(overlay):
        raise ValueError(f"overlay contains {len(set(overlay) - seen)} lines absent from source")
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    arrays = {
        "entry_targets": targets,
        "role_ids": role_ids,
        "role_status_ids": role_status_ids,
        "boundary_ids": boundary_ids,
        "boundary_status_ids": boundary_status_ids,
        "overlay_present": overlay_present,
    }
    for name, value in arrays.items():
        _save(output / f"{name}.npy", value)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_source_bound_overlay_join",
        "split": args.split,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "line_count": n,
        "overlay_line_count": len(overlay),
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "overlay": {"path": str(overlay_path), "sha256": sha256_file(overlay_path)},
        "base_table_manifest_sha256": sha256_file(Path(args.base_table_dir).resolve() / "manifest.json"),
        "contract_sha256": contract.sha256,
        "role_encoding": ROLE_TO_ID,
        "status_encoding": STATUS_TO_ID,
        "boundary_encoding": BOUNDARY_TO_ID,
        "entry_target_encoding": {"MASK": TARGET_MASK, "NEGATIVE": TARGET_NEGATIVE, "ENTRY_ANCHOR": TARGET_ENTRY_ANCHOR},
        "entry_target_counts": {
            str(key): value for key, value in sorted(collections.Counter(int(x) for x in targets).items())
        },
        "policy": {
            "source_O_TOC": "entry negative unless trusted overlay corrects it",
            "unreviewed_source_BIB_UNKNOWN": "masked",
            "trusted_ENTRY_ANCHOR": "positive",
            "trusted_other_roles": "negative except UNKNOWN masked",
            "untrusted_overlay": "diagnostic role only; does not override entry target",
        },
    }
    _write_json(output / "manifest.json", manifest)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }
    receipt = {**manifest, "outputs": outputs}
    _write_json(output / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--contract", default=str(Path(__file__).with_name("bibliography_role_contract_v1.json")))
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
