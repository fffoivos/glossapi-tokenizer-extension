#!/usr/bin/env python3
"""Export exact documents touched by the frozen Phase-1/2 GPTDataset caches.

The top-level BlendedDataset indices select a component and component sample.
The component GPTDataset shuffle/sample/document indices then identify every
indexed document contributing tokens to that sample.  This follows the pinned
c92402e ``__getitem__`` path; it does not infer consumption from token quotas.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache


PHASE_SAMPLES = {1: 2261 * 1024, 2: (3218 - 2261) * 1024}
PHASE_ACTIVE_STREAM = {1: "hplt", 2: "openarchives"}
ROLE_STREAM = {"foreign_replay": "foreign", "old_greek_replay": "old_greek"}


def description_rows(cache_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for path in sorted(cache_root.glob("*-description.txt")):
        value = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(value, dict), f"cache description is not an object: {path}")
        result.append((path, value))
    require(result, f"cache has no descriptions: {cache_root}")
    return result


def array_for_description(path: Path, suffix: str) -> Path:
    marker = "-description.txt"
    require(path.name.endswith(marker), f"unexpected description name: {path}")
    return path.with_name(path.name[: -len(marker)] + f"-{suffix}.npy")


def find_blended_description(rows: list[tuple[Path, dict[str, Any]]], consumed: int) -> tuple[Path, dict[str, Any]]:
    candidates = [
        (path, value) for path, value in rows
        if value.get("class") == "BlendedDataset"
        and str(value.get("split", "")).lower() == "train"
        and int(value.get("size", -1)) >= consumed
    ]
    require(len(candidates) == 1, f"expected one consumed BlendedDataset description, found {len(candidates)}")
    return candidates[0]


def find_gpt_description(rows: list[tuple[Path, dict[str, Any]]], prefix: str) -> Path:
    resolved = str(Path(prefix).resolve())
    candidates = [
        path for path, value in rows
        if value.get("class") == "GPTDataset"
        and str(value.get("index_split", "")).lower() == "train"
        and str(Path(str(value.get("dataset_path", ""))).resolve()) == resolved
    ]
    require(len(candidates) == 1, f"expected one GPTDataset description for {resolved}, found {len(candidates)}")
    return candidates[0]


def documents_for_component(
    component_sample_ids: np.ndarray,
    description: Path,
) -> set[int]:
    document = np.load(array_for_description(description, "document_index"), mmap_mode="r", allow_pickle=False)
    sample = np.load(array_for_description(description, "sample_index"), mmap_mode="r", allow_pickle=False)
    shuffle = np.load(array_for_description(description, "shuffle_index"), mmap_mode="r", allow_pickle=False)
    require(component_sample_ids.size > 0, f"component has no realized samples: {description}")
    require(int(component_sample_ids.min()) >= 0 and int(component_sample_ids.max()) < len(shuffle), f"component sample index out of range: {description}")
    realized: set[int] = set()
    chunk_size = 250_000
    for start in range(0, component_sample_ids.size, chunk_size):
        ids = np.asarray(component_sample_ids[start : start + chunk_size], dtype=np.int64)
        shuffled = np.asarray(shuffle[ids], dtype=np.int64)
        beginnings = np.asarray(sample[shuffled, 0], dtype=np.int64)
        ends = np.asarray(sample[shuffled + 1, 0], dtype=np.int64)
        require(bool(np.all(ends >= beginnings)), f"sample/document bounds drift: {description}")
        realized.update(int(value) for value in np.asarray(document[beginnings], dtype=np.int64))
        different = ends != beginnings
        if bool(np.any(different)):
            realized.update(int(value) for value in np.asarray(document[ends[different]], dtype=np.int64))
        for beginning, end in zip(beginnings[(ends - beginnings) > 1], ends[(ends - beginnings) > 1], strict=True):
            realized.update(int(value) for value in np.asarray(document[beginning + 1 : end], dtype=np.int64))
    return realized


def validate_catalog(receipt_path: Path, stream: str, prefix: str) -> tuple[dict[str, Any], Path]:
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == "apertus_hard_h_to_g_tokenized_stream_v1", f"{stream}: catalog receipt schema drift")
    require(receipt.get("status") == "frozen" and receipt.get("stream") == stream, f"{stream}: catalog receipt identity drift")
    require(str(Path(str(receipt.get("dataset_prefix", ""))).resolve()) == str(Path(prefix).resolve()), f"{stream}: catalog prefix drift")
    catalog = receipt.get("document_catalog")
    require(isinstance(catalog, dict), f"{stream}: document catalog missing")
    path = Path(str(catalog.get("path", "")))
    expected_catalog = {
        **file_binding(path),
        "rows": catalog.get("rows"),
        "tokens_including_eod": catalog.get("tokens_including_eod"),
    } if path.is_file() else None
    require(expected_catalog == catalog, f"{stream}: document catalog binding drift")
    return receipt, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-cache-receipt", type=Path, required=True)
    parser.add_argument("--phase2-cache-receipt", type=Path, required=True)
    for stream in ("hplt", "openarchives", "foreign", "old_greek"):
        parser.add_argument(f"--{stream.replace('_', '-')}-tokenized-receipt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_jsonl.exists() and not args.output_receipt.exists(), "immutable realized-ledger output exists")

    catalog_paths: dict[str, Path] = {}
    catalog_receipts: dict[str, dict[str, Any]] = {}
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    trajectory = hashlib.sha256(b"apertus-c92402e-realized-sample-trajectory-v1\0")
    cache_bindings = []
    for phase, receipt_path in ((1, args.phase1_cache_receipt), (2, args.phase2_cache_receipt)):
        phase_cache = read_json(receipt_path)
        data_path = Path(str(phase_cache.get("data_path_spec", {}).get("path", "")))
        cache_root = Path(str(phase_cache.get("cache_root", "")))
        validate_phase_cache(phase_cache, phase=phase, data_path_spec=data_path, cache_root=cache_root)
        spec = read_json(data_path)
        rows = description_rows(cache_root)
        consumed = PHASE_SAMPLES[phase]
        top_path, top_description = find_blended_description(rows, consumed)
        dataset_index_path = array_for_description(top_path, "dataset_index")
        dataset_sample_path = array_for_description(top_path, "dataset_sample_index")
        dataset_index = np.load(dataset_index_path, mmap_mode="r", allow_pickle=False)
        dataset_sample = np.load(dataset_sample_path, mmap_mode="r", allow_pickle=False)
        require(len(dataset_index) == len(dataset_sample) and len(dataset_index) >= consumed, f"phase {phase}: top-level index length drift")
        component_descriptions = top_description.get("datasets")
        require(isinstance(component_descriptions, list) and len(component_descriptions) == 3, f"phase {phase}: blended component description drift")
        components = spec.get("components")
        require(isinstance(components, list) and len(components) == 3, f"phase {phase}: data-path component drift")
        top_ids = np.asarray(dataset_index[:consumed], dtype=np.int16)
        top_samples = np.asarray(dataset_sample[:consumed], dtype=np.int64)
        trajectory.update(phase.to_bytes(1, "little"))
        trajectory.update(consumed.to_bytes(8, "little"))
        trajectory.update(top_ids.tobytes(order="C"))
        trajectory.update(top_samples.tobytes(order="C"))
        for dataset_id, component in enumerate(components):
            require(
                Path(str(component_descriptions[dataset_id].get("dataset_path", ""))).resolve()
                == Path(str(component.get("prefix", ""))).resolve(),
                f"phase {phase}: blended/data-path component order drift",
            )
            role = str(component["role"])
            stream = PHASE_ACTIVE_STREAM[phase] if role == "active_modern" else ROLE_STREAM.get(role)
            require(stream is not None, f"phase {phase}: unknown component role {role}")
            prefix = str(component["prefix"])
            receipt_argument = getattr(args, f"{stream}_tokenized_receipt")
            if stream not in catalog_paths:
                catalog_receipts[stream], catalog_paths[stream] = validate_catalog(receipt_argument, stream, prefix)
            else:
                require(catalog_receipts[stream]["document_catalog"] == read_json(receipt_argument)["document_catalog"], f"{stream}: catalog receipt changed between phases")
            mask = top_ids == dataset_id
            realized_samples = top_samples[mask]
            gpt_description = find_gpt_description(rows, prefix)
            document_ids = documents_for_component(realized_samples, gpt_description)
            for document_id in document_ids:
                entry = observed.setdefault((stream, document_id), {"phases": set(), "roles": set()})
                entry["phases"].add(phase)
                entry["roles"].add(role)
        cache_bindings.append({"phase": phase, "receipt": file_binding(receipt_path), "consumed_samples": consumed})

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{args.output_jsonl.name}.", suffix=".partial", dir=args.output_jsonl.parent)
    temporary = Path(name)
    counts: Counter[str] = Counter()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            for stream in ("hplt", "openarchives", "foreign", "old_greek"):
                wanted = {index for candidate_stream, index in observed if candidate_stream == stream}
                seen: set[int] = set()
                with catalog_paths[stream].open(encoding="utf-8") as source:
                    for line in source:
                        row = json.loads(line)
                        document_index = int(row["document_index"])
                        if document_index not in wanted:
                            continue
                        metadata = observed[(stream, document_index)]
                        target.write(json.dumps({
                            **row,
                            "observed_phases": sorted(metadata["phases"]),
                            "observed_roles": sorted(metadata["roles"]),
                        }, ensure_ascii=False, sort_keys=True) + "\n")
                        seen.add(document_index)
                        counts["documents"] += 1
                        counts[f"{stream}_documents"] += 1
                require(seen == wanted, f"{stream}: realized document ids absent from document catalog")
            target.flush(); os.fsync(target.fileno())
        os.link(temporary, args.output_jsonl); temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema_version": "apertus_hard_h_to_g_realized_document_ledger_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "megatron_revision": "c92402e39ef3c8e69ea378a59e79059dc14541f4",
        "cache_receipts": cache_bindings,
        "tokenized_receipts": {
            stream: file_binding(getattr(args, f"{stream}_tokenized_receipt"))
            for stream in ("hplt", "openarchives", "foreign", "old_greek")
        },
        "realized_sample_trajectory_sha256": trajectory.hexdigest(),
        "output": {**file_binding(args.output_jsonl), "rows": counts["documents"]},
        "counts": dict(counts),
        "invariants": {
            "top_level_blend_indices_consumed_exactly": True,
            "component_shuffle_sample_document_indices_resolved_exactly": True,
            "quota_inference_used": False,
        },
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
