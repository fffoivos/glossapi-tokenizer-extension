#!/usr/bin/env python3
"""Prepare 240 additional role blocks and 30 zero-BIB controls."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bibliography_role_profile import (
    PACKET_SCHEMA_VERSION,
    _context_lines,
    _review_chunk_ranges,
    profile_document,
)
from .contract import canonical_json_sha256, sha256_file


SCHEMA_VERSION = "bibliography-role-bootstrap-selection-v1"
SOURCES = ("greek_phd", "kallipos", "openarchives")


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{number}: expected object")
            yield row


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _rank(seed: str, *values: str) -> int:
    return int.from_bytes(hashlib.sha256("\0".join((seed, *values)).encode()).digest(), "big")


def select_additional_blocks(
    inventory: Sequence[Mapping[str, Any]], initial: Mapping[str, Any],
    *, per_source: int, seed: str,
) -> list[dict[str, Any]]:
    if per_source != 80:
        raise ValueError("v1 bootstrap is frozen at 80 additional blocks per source")
    initial_blocks = {str(row["block_id"]) for row in initial["cases"]}
    initial_works = {str(row["work_id"]) for row in initial["cases"]}
    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        pool = [
            dict(row) for row in inventory
            if row.get("source") == source and row.get("coverage") == "full_document"
            and str(row.get("block_id")) not in initial_blocks
            and str(row.get("work_id")) not in initial_works
        ]
        used_works: set[str] = set()
        used_blocks: set[str] = set()
        local: list[dict[str, Any]] = []

        def take(candidates: Sequence[Mapping[str, Any]], count: int, stratum: str) -> None:
            chosen = 0
            for allow_second_block in (False, True):
                for candidate in candidates:
                    block_id, work_id = str(candidate["block_id"]), str(candidate["work_id"])
                    if block_id in used_blocks or (not allow_second_block and work_id in used_works):
                        continue
                    row = dict(candidate)
                    row["bootstrap_stratum"] = stratum
                    row["work_repeated_within_bootstrap"] = work_id in used_works
                    local.append(row)
                    used_blocks.add(block_id)
                    used_works.add(work_id)
                    chosen += 1
                    if chosen == count:
                        return
            raise ValueError(f"{source}: insufficient {stratum} candidates")

        continuation = sorted(
            [row for row in pool if int(row["long_line_count"]) > 0],
            key=lambda row: (-int(row["long_line_count"]), -int(row["physical_span"]), _rank(seed, source, "continuation", str(row["block_id"]))),
        )
        take(continuation, 10, "underrepresented_continuation_proxy")
        filler = sorted(
            [row for row in pool if int(row["zero_semantic_feature_line_count"]) > 0 or int(row["maximum_low_match_run"]) >= 2],
            key=lambda row: (-int(row["zero_semantic_feature_line_count"]), -int(row["maximum_low_match_run"]), -float(row["low_match_fraction"]), _rank(seed, source, "filler", str(row["block_id"]))),
        )
        take(filler, 10, "underrepresented_filler_proxy")
        hard = sorted(
            pool,
            key=lambda row: (
                -int(int(row["maximum_low_match_run"]) >= 2),
                -int(int(row["long_line_count"]) > 0),
                -int(int(row["exact_header_line_count"]) > 0),
                -float(row["low_match_fraction"]),
                -int(row["physical_span"]),
                _rank(seed, source, "hard", str(row["block_id"])),
            ),
        )
        take(hard, 40, "prediction_blind_hard_proxy")
        random_pool = sorted(
            pool, key=lambda row: _rank(seed, source, "random", str(row["block_id"]))
        )
        take(random_pool, 20, "random")
        if len(local) != per_source or len({row["block_id"] for row in local}) != per_source:
            raise AssertionError(f"{source}: bootstrap balance failure")
        selected.extend(local)
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.input).resolve()
    if args.expected_input_sha256 and sha256_file(source_path) != args.expected_input_sha256:
        raise ValueError("source input SHA-256 differs from pin")
    inventory = list(_iter_jsonl(Path(args.block_inventory).resolve()))
    initial = json.loads(Path(args.initial_selection).read_text(encoding="utf-8"))
    if initial.get("schema_version") != "bibliography-role-review-selection-v1":
        raise ValueError("unsupported initial selection")
    selected = select_additional_blocks(
        inventory, initial, per_source=int(args.per_source), seed=args.seed,
    )
    selected_docs = {str(row["document_id"]) for row in selected}
    rows = [
        row for row in _iter_jsonl(source_path)
        if row.get("split") == "train" and row.get("source") in SOURCES
    ]
    rows_by_doc = {str(row["document_id"]): row for row in rows}
    if not selected_docs <= set(rows_by_doc):
        raise ValueError("selected block document is absent from source")
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        profiles = list(executor.map(profile_document, [rows_by_doc[key] for key in sorted(selected_docs)], chunksize=1))
    profiles_by_doc = {str(row["document_id"]): row for row in profiles}
    blind_cases, provenance_cases = [], []
    for block in selected:
        document_id = str(block["document_id"])
        source = rows_by_doc[document_id]
        blind_lines, labels = _context_lines(source, block, context_lines=5)
        block_case_id = hashlib.sha256(
            f"bibliography-role-review-v1\0{block['block_id']}".encode()
        ).hexdigest()[:24]
        ranges = _review_chunk_ranges(
            blind_lines, maximum_lines=80, maximum_characters=20000, overlap_lines=5,
        )
        review_chunks = []
        for chunk_index, (start, end) in enumerate(ranges):
            case_id = hashlib.sha256(
                f"{block_case_id}\0{chunk_index}\0{start}\0{end}".encode()
            ).hexdigest()[:24]
            blind_cases.append(
                {
                    "case_id": case_id, "block_case_id": block_case_id,
                    "chunk_index": chunk_index, "chunk_count": len(ranges),
                    "document_id": document_id, "work_id": str(block["work_id"]),
                    "source": str(block["source"]),
                    "n_physical_lines": int(source.get("n_physical_lines", 0)),
                    "lines": blind_lines[start:end],
                }
            )
            review_chunks.append(
                {"case_id": case_id, "chunk_index": chunk_index, "chunk_start": start,
                 "chunk_end": end, "context_source_labels": labels[start:end]}
            )
        line_lookup = {
            int(line["abs_idx"]): line for line in profiles_by_doc[document_id]["line_profiles"]
        }
        provenance_cases.append(
            {
                "block_case_id": block_case_id, **block, "review_chunks": review_chunks,
                "block_line_profiles": [
                    line_lookup[int(line["abs_idx"])]
                    for line in source["lines"][int(block["start_line_offset"]):int(block["end_line_offset"]) + 1]
                ],
            }
        )
    role_contract = json.loads(Path(args.role_contract).read_text(encoding="utf-8"))
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "status": "ready_for_blind_contextual_review",
        "blinding": {
            "detector_features_hidden": True, "model_predictions_hidden": True,
            "nomination_strata_hidden": True, "original_region_labels_hidden": True,
        },
        "instructions": {
            "task": "Assign one role and one boundary flag to every displayed line using document context.",
            "roles": role_contract["roles"], "boundary_flags": role_contract["boundary_flags"],
            "response_schema": "bibliography_role_review.schema.json",
        },
        "selection": {
            "purpose": "additional source-balanced role bootstrap",
            "block_count": len(selected), "case_count": len(blind_cases),
            "blocks_per_source": int(args.per_source), "review_coverage": "full_document",
            "context_lines_each_side": 5, "maximum_review_lines": 80,
            "maximum_review_characters": 20000, "review_overlap_lines": 5,
            "source_block_counts": dict(collections.Counter(row["source"] for row in selected)),
            "source_chunk_counts": dict(collections.Counter(row["source"] for row in blind_cases)),
            "bootstrap_stratum_counts": dict(collections.Counter(row["bootstrap_stratum"] for row in selected)),
            "seed": args.seed,
        },
        "cases": blind_cases,
    }
    used_works = {str(row["work_id"]) for row in initial["cases"]} | {str(row["work_id"]) for row in selected}
    controls = []
    for source_name in SOURCES:
        candidates = [
            row for row in rows if row.get("source") == source_name
            and row.get("coverage") == "full_document" and str(row.get("work_id")) not in used_works
            and not any(line.get("label") == "BIB" for line in row.get("lines", []))
        ]
        candidates.sort(key=lambda row: _rank(args.seed, source_name, "zero-bib", str(row["document_id"])))
        chosen, works = [], set()
        for row in candidates:
            if row["work_id"] in works:
                continue
            chosen.append(row); works.add(row["work_id"])
            if len(chosen) == 10:
                break
        if len(chosen) != 10:
            raise ValueError(f"{source_name}: fewer than ten zero-BIB controls")
        controls.extend(
            {"source": source_name, "document_id": row["document_id"], "work_id": row["work_id"],
             "line_count": len(row["lines"]), "n_physical_lines": row.get("n_physical_lines")}
            for row in chosen
        )
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _write_json(output / "role_review_packet.blind.json", packet)
    provenance = {
        "schema_version": "bibliography-role-review-selection-v1",
        "warning": "Contains nomination strata and immutable source labels; do not pass to reviewers.",
        "selection_schema_version": SCHEMA_VERSION,
        "cases": provenance_cases,
    }
    _write_json(output / "selection_manifest.provenance.json", provenance)
    _write_json(
        output / "zero_bib_controls.provenance.json",
        {"schema_version": "bibliography-zero-bib-controls-v1", "document_count": 30,
         "source_counts": dict(collections.Counter(row["source"] for row in controls)), "documents": controls},
    )
    summary = {
        "schema_version": SCHEMA_VERSION, "status": "passed_prediction_blind_bootstrap",
        "code_commit": args.code_commit, "slurm_job_id": args.slurm_job_id,
        "source_sha256": sha256_file(source_path), "block_count": len(selected),
        "review_chunk_count": len(blind_cases), "zero_bib_control_count": len(controls),
        "source_block_counts": dict(collections.Counter(row["source"] for row in selected)),
        "bootstrap_stratum_counts": dict(collections.Counter(row["bootstrap_stratum"] for row in selected)),
        "prediction_inputs_loaded": False,
    }
    _write_json(output / "summary.json", summary)
    outputs = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(output.iterdir())}
    receipt = {**summary, "receipt_content_sha256": canonical_json_sha256({"summary": summary, "outputs": outputs}), "outputs": outputs}
    _write_json(output / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--block-inventory", required=True)
    parser.add_argument("--initial-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-source", type=int, default=80)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", default="bibliography-role-bootstrap-v1")
    parser.add_argument("--role-contract", default=str(Path(__file__).with_name("bibliography_role_contract_v1.json")))
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
