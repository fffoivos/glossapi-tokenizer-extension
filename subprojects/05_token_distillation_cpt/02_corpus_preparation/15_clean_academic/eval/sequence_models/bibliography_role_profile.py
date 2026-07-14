#!/usr/bin/env python3
"""Profile STRUCT-2K line roles and build a blinded full-block review packet.

The job is prediction-blind: it uses immutable silver region labels only to
inventory existing blocks and deterministic line features only to stratify the
review sample.  Model probabilities and predicted outcomes are neither loaded
nor emitted into the blinded packet.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bibliography_entry_dataset import MAX_PHYSICAL_GAP, _line_block_indices
from .bibliography_positional_features import (
    FEATURE_NAMES,
    SEMANTIC_UNION_EXCLUDED,
    extract_positional_line,
)
from .contract import canonical_json_sha256, sha256_file
from .deterministic_structure import BibRole, analyze_bib_line


SCHEMA_VERSION = "bibliography-role-profile-v1"
PACKET_SCHEMA_VERSION = "bibliography-role-review-packet-v1"
SOURCES = ("greek_phd", "kallipos", "openarchives")
STRATA = (
    "exact_header",
    "sparse_internal",
    "long_wrapped",
    "conventional_dense",
    "heterogeneous",
)


def _iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"row {row_number} is not an object")
            yield row


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _rank(seed: str, *values: str) -> int:
    return int.from_bytes(
        hashlib.sha256("\0".join((seed, *values)).encode("utf-8")).digest(), "big"
    )


def _line_id(document_id: str, line: Mapping[str, Any]) -> str:
    value = line.get("line_id")
    if isinstance(value, str) and value:
        return value
    return f"{document_id}:{int(line['abs_idx'])}"


def _profile_line(
    document_id: str,
    line: Mapping[str, Any],
    *,
    block_index: int,
    block_offset: int | None,
    block_line_count: int | None,
) -> dict[str, Any]:
    text, abs_idx, label = line.get("text"), line.get("abs_idx"), line.get("label")
    if not isinstance(text, str) or not isinstance(abs_idx, int):
        raise ValueError(f"{document_id}: malformed source line")
    encoding = extract_positional_line(text)
    semantic_indices = [
        index for index, name in enumerate(FEATURE_NAMES) if name not in SEMANTIC_UNION_EXCLUDED
    ]
    semantic_counts = encoding.counts[semantic_indices]
    feature_count = int((encoding.counts > 0).sum())
    semantic_feature_count = int((semantic_counts > 0).sum())
    evidence = analyze_bib_line(encoding.normalized_text, abs_idx)
    matched_fraction = float(1.0 - encoding.gap_summaries[0])
    return {
        "document_id": document_id,
        "line_id": _line_id(document_id, line),
        "abs_idx": abs_idx,
        "original_region_label": str(label),
        "block_index": block_index,
        "block_offset": block_offset,
        "block_line_count": block_line_count,
        "nfkc_length": encoding.nfkc_length,
        "feature_count": feature_count,
        "semantic_feature_count": semantic_feature_count,
        "semantic_match_occurrences": int(semantic_counts.sum()),
        "matched_character_fraction": round(matched_fraction, 6),
        "unmatched_prefix_fraction": round(float(encoding.gap_summaries[1]), 6),
        "unmatched_suffix_fraction": round(float(encoding.gap_summaries[2]), 6),
        "longest_unmatched_fraction": round(float(encoding.gap_summaries[3]), 6),
        "deterministic_role": evidence.role.value,
        "exact_header_kind": (
            "HEADER"
            if evidence.role == BibRole.HEADING
            else "SUBHEADER"
            if evidence.role == BibRole.SUBHEADING
            else None
        ),
        "short_le_110": encoding.nfkc_length <= 110,
        "long_gt_330": encoding.nfkc_length > 330,
        "low_match": semantic_feature_count <= 1,
    }


def _maximum_true_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _block_strata(summary: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    if int(summary["exact_header_line_count"]) > 0:
        flags.append("exact_header")
    if int(summary["maximum_low_match_run"]) >= 2 or float(summary["low_match_fraction"]) >= 0.2:
        flags.append("sparse_internal")
    if int(summary["long_line_count"]) > 0:
        flags.append("long_wrapped")
    if int(summary["dense_line_count"]) >= max(2, int(summary["line_count"]) // 2):
        flags.append("conventional_dense")
    if len(flags) >= 2 or not flags:
        flags.append("heterogeneous")
    return flags


def profile_document(row: Mapping[str, Any]) -> dict[str, Any]:
    document_id, work_id = str(row.get("document_id", "")), str(row.get("work_id", ""))
    source = str(row.get("source", ""))
    raw_lines = row.get("lines")
    if not document_id or not work_id or not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("document identity and non-empty lines are required")
    block_indices = _line_block_indices(raw_lines)
    block_members: dict[int, list[int]] = collections.defaultdict(list)
    for offset, block_index in enumerate(block_indices):
        if int(block_index) >= 0:
            block_members[int(block_index)].append(offset)

    block_positions: dict[int, tuple[int, int]] = {}
    for members in block_members.values():
        for block_offset, line_offset in enumerate(members):
            block_positions[line_offset] = (block_offset, len(members))

    line_profiles: list[dict[str, Any]] = []
    for line_offset, line in enumerate(raw_lines):
        block_index = int(block_indices[line_offset])
        position = block_positions.get(line_offset)
        line_profiles.append(
            _profile_line(
                document_id,
                line,
                block_index=block_index,
                block_offset=position[0] if position else None,
                block_line_count=position[1] if position else None,
            )
        )

    blocks: list[dict[str, Any]] = []
    for block_index, members in sorted(block_members.items()):
        local = [line_profiles[offset] for offset in members]
        line_count = len(local)
        low_flags = [bool(line["low_match"]) for line in local]
        dense_count = sum(int(line["semantic_feature_count"]) >= 3 for line in local)
        summary: dict[str, Any] = {
            "block_id": f"{document_id}:bib:{block_index}",
            "document_id": document_id,
            "work_id": work_id,
            "source": source,
            "block_index": block_index,
            "start_line_offset": members[0],
            "end_line_offset": members[-1],
            "start_abs_idx": int(raw_lines[members[0]]["abs_idx"]),
            "end_abs_idx": int(raw_lines[members[-1]]["abs_idx"]),
            "line_count": line_count,
            "short_line_count": sum(bool(line["short_le_110"]) for line in local),
            "long_line_count": sum(bool(line["long_gt_330"]) for line in local),
            "low_match_line_count": sum(low_flags),
            "low_match_fraction": round(sum(low_flags) / line_count, 6),
            "maximum_low_match_run": _maximum_true_run(low_flags),
            "zero_semantic_feature_line_count": sum(
                int(line["semantic_feature_count"]) == 0 for line in local
            ),
            "dense_line_count": dense_count,
            "exact_header_line_count": sum(line["exact_header_kind"] is not None for line in local),
            "physical_span": int(raw_lines[members[-1]]["abs_idx"])
            - int(raw_lines[members[0]]["abs_idx"])
            + 1,
        }
        summary["strata"] = _block_strata(summary)
        blocks.append(summary)
    return {
        "document_id": document_id,
        "work_id": work_id,
        "source": source,
        "n_physical_lines": int(row.get("n_physical_lines", 0)),
        "line_profiles": line_profiles,
        "blocks": blocks,
    }


def select_review_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    sources: Sequence[str],
    per_source: int,
    seed: str,
) -> list[Mapping[str, Any]]:
    """Select source/stratum-balanced blocks with work diversity."""

    if per_source < len(STRATA):
        raise ValueError(f"per_source must be at least {len(STRATA)}")
    selected: list[Mapping[str, Any]] = []
    for source in sources:
        local = [row for row in blocks if row["source"] == source]
        if len(local) < per_source:
            raise ValueError(f"{source}: only {len(local)} bibliography blocks")
        chosen_ids: set[str] = set()
        chosen_works: set[str] = set()
        target_by_stratum = {
            stratum: per_source // len(STRATA) + int(index < per_source % len(STRATA))
            for index, stratum in enumerate(STRATA)
        }
        for round_index in range(max(target_by_stratum.values())):
            for stratum in STRATA:
                if round_index >= target_by_stratum[stratum]:
                    continue
                candidates = [
                    row
                    for row in local
                    if stratum in row["strata"]
                    and row["block_id"] not in chosen_ids
                    and row["work_id"] not in chosen_works
                ]
                candidates.sort(
                    key=lambda row: _rank(seed, source, stratum, str(row["block_id"]))
                )
                if candidates:
                    chosen = candidates[0]
                    selected.append(chosen)
                    chosen_ids.add(str(chosen["block_id"]))
                    chosen_works.add(str(chosen["work_id"]))
        if len(chosen_ids) < per_source:
            remainder = [
                row
                for row in local
                if row["block_id"] not in chosen_ids and row["work_id"] not in chosen_works
            ]
            remainder.sort(key=lambda row: _rank(seed, source, "remainder", str(row["block_id"])))
            for chosen in remainder[: per_source - len(chosen_ids)]:
                selected.append(chosen)
                chosen_ids.add(str(chosen["block_id"]))
                chosen_works.add(str(chosen["work_id"]))
        if len(chosen_ids) != per_source:
            raise ValueError(f"{source}: could not select {per_source} work-distinct blocks")
    return selected


def _context_lines(
    row: Mapping[str, Any], block: Mapping[str, Any], *, context_lines: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_lines = row["lines"]
    start = max(0, int(block["start_line_offset"]) - context_lines)
    end = min(len(raw_lines), int(block["end_line_offset"]) + context_lines + 1)
    blind: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    n_physical = max(1, int(row.get("n_physical_lines", 1)) - 1)
    for line in raw_lines[start:end]:
        line_id = _line_id(str(row["document_id"]), line)
        blind.append(
            {
                "line_id": line_id,
                "abs_idx": int(line["abs_idx"]),
                "document_position_percent": round(100 * int(line["abs_idx"]) / n_physical, 2),
                "text": str(line["text"]),
            }
        )
        provenance.append(
            {
                "line_id": line_id,
                "abs_idx": int(line["abs_idx"]),
                "original_region_label": str(line["label"]),
            }
        )
    return blind, provenance


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    contract_path = Path(args.role_contract).resolve()
    if not input_path.is_file() or input_path.is_symlink():
        raise ValueError(f"input must be a real file: {input_path}")
    if not contract_path.is_file() or contract_path.is_symlink():
        raise ValueError(f"role contract must be a real file: {contract_path}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    input_sha256 = sha256_file(input_path)
    if args.expected_input_sha256 and input_sha256 != args.expected_input_sha256:
        raise ValueError("input SHA-256 does not match the pinned value")
    role_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if role_contract.get("schema_version") != "bibliography-role-contract-v1":
        raise ValueError("unsupported role contract")
    sources = tuple(value.strip() for value in args.sources.split(",") if value.strip())
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("sources must be a non-empty unique comma-separated list")

    rows = [
        row
        for row in _iter_rows(input_path)
        if row.get("split") == args.split
        and row.get("coverage") == args.coverage
        and row.get("source") in sources
    ]
    if not rows:
        raise ValueError("no source documents match the requested selection")
    workers = max(1, int(args.workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        profiles = list(executor.map(profile_document, rows, chunksize=1))
    block_inventory = [block for profile in profiles for block in profile["blocks"]]
    selected = select_review_blocks(
        block_inventory,
        sources=sources,
        per_source=int(args.blocks_per_source),
        seed=str(args.seed),
    )

    rows_by_document = {str(row["document_id"]): row for row in rows}
    profile_by_document = {str(row["document_id"]): row for row in profiles}
    blind_cases: list[dict[str, Any]] = []
    selection_cases: list[dict[str, Any]] = []
    for block in selected:
        document_id = str(block["document_id"])
        raw = rows_by_document[document_id]
        blind_lines, source_labels = _context_lines(
            raw, block, context_lines=int(args.context_lines)
        )
        case_id = hashlib.sha256(
            f"bibliography-role-review-v1\0{block['block_id']}".encode("utf-8")
        ).hexdigest()[:24]
        blind_cases.append(
            {
                "case_id": case_id,
                "document_id": document_id,
                "work_id": str(block["work_id"]),
                "source": str(block["source"]),
                "n_physical_lines": int(raw.get("n_physical_lines", 0)),
                "lines": blind_lines,
            }
        )
        line_profile_lookup = {
            int(line["abs_idx"]): line
            for line in profile_by_document[document_id]["line_profiles"]
        }
        selection_cases.append(
            {
                "case_id": case_id,
                **dict(block),
                "context_source_labels": source_labels,
                "block_line_profiles": [
                    line_profile_lookup[int(line["abs_idx"])]
                    for line in raw["lines"][
                        int(block["start_line_offset"]) : int(block["end_line_offset"]) + 1
                    ]
                ],
            }
        )

    output_dir.mkdir(parents=True)
    line_profile_rows = (
        {"source": profile["source"], "work_id": profile["work_id"], **line}
        for profile in profiles
        for line in profile["line_profiles"]
    )
    _write_jsonl(output_dir / "line_profile.jsonl", line_profile_rows)
    _write_jsonl(output_dir / "block_inventory.jsonl", block_inventory)
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "status": "ready_for_blind_contextual_review",
        "blinding": {
            "detector_features_hidden": True,
            "model_predictions_hidden": True,
            "nomination_strata_hidden": True,
            "original_region_labels_hidden": True,
        },
        "instructions": {
            "task": "Assign one role and one boundary flag to every displayed line using document context.",
            "roles": role_contract["roles"],
            "boundary_flags": role_contract["boundary_flags"],
            "response_schema": "bibliography_role_review.schema.json",
        },
        "selection": {
            "case_count": len(blind_cases),
            "blocks_per_source": int(args.blocks_per_source),
            "context_lines_each_side": int(args.context_lines),
            "source_counts": dict(sorted(collections.Counter(case["source"] for case in blind_cases).items())),
            "seed": str(args.seed),
        },
        "cases": blind_cases,
    }
    _write_json(output_dir / "role_review_packet.blind.json", packet)
    _write_json(
        output_dir / "selection_manifest.provenance.json",
        {
            "schema_version": "bibliography-role-review-selection-v1",
            "warning": "Contains nomination strata and immutable source labels; do not pass to first-round reviewers.",
            "cases": selection_cases,
        },
    )

    source_document_counts = collections.Counter(profile["source"] for profile in profiles)
    source_block_counts = collections.Counter(block["source"] for block in block_inventory)
    strata_counts = collections.Counter(
        stratum for block in block_inventory for stratum in block["strata"]
    )
    selected_strata_counts = collections.Counter(
        stratum for block in selected for stratum in block["strata"]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_prediction_blind_profile",
        "input": {"path": str(input_path), "sha256": input_sha256},
        "role_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "split": str(args.split),
        "coverage": str(args.coverage),
        "document_count": len(profiles),
        "line_count": sum(len(profile["line_profiles"]) for profile in profiles),
        "block_count": len(block_inventory),
        "source_document_counts": dict(sorted(source_document_counts.items())),
        "source_block_counts": dict(sorted(source_block_counts.items())),
        "block_strata_counts": dict(sorted(strata_counts.items())),
        "selected_block_count": len(selected),
        "selected_block_strata_counts": dict(sorted(selected_strata_counts.items())),
        "max_physical_gap": MAX_PHYSICAL_GAP,
        "prediction_inputs_loaded": False,
        "model_outcomes_used_for_selection": False,
    }
    _write_json(output_dir / "profile.summary.json", summary)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    receipt = {
        **summary,
        "receipt_content_sha256": canonical_json_sha256(
            {"summary": summary, "outputs": outputs}
        ),
        "outputs": outputs,
    }
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--coverage", default="full_document")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--blocks-per-source", type=int, default=20)
    parser.add_argument("--context-lines", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", default="bibliography-role-pilot-v1")
    parser.add_argument(
        "--role-contract",
        default=str(Path(__file__).with_name("bibliography_role_contract_v1.json")),
    )
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
