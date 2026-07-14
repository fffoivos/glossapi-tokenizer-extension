#!/usr/bin/env python3
"""Validate bibliography role overlays and derive model-specific targets.

The role overlay never rewrites the source STRUCT-2K region labels.  It is
joined by line identity, coordinate, and text hash so stale or cross-document
annotations fail closed.  Only agreed/adjudicated labels are trusted targets;
single-review and model-derived rows remain diagnostic.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contract import canonical_json_sha256, sha256_file


SCHEMA_VERSION = "bibliography-role-overlay-v1"
SCHEMA_VERSION_V2 = "bibliography-role-overlay-v2"
REVIEW_SCHEMA_VERSION = "bibliography-role-review-v1"
CONTRACT_SCHEMA_VERSION = "bibliography-role-contract-v1"
TARGET_MASK = -1
TARGET_NEGATIVE = 0
TARGET_ENTRY_ANCHOR = 1


@dataclass(frozen=True)
class RoleContract:
    roles: frozenset[str]
    boundary_flags: frozenset[str]
    label_statuses: frozenset[str]
    trusted_statuses: frozenset[str]
    positive_roles: frozenset[str]
    negative_roles: frozenset[str]
    masked_roles: frozenset[str]
    sha256: str


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_role_contract(path: str | Path) -> RoleContract:
    contract_path = Path(path).resolve()
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported bibliography role contract: {contract_path}")
    roles = raw.get("roles")
    boundaries = raw.get("boundary_flags")
    statuses = raw.get("label_statuses")
    target = raw.get("entry_anchor_target")
    if not all(isinstance(value, Mapping) and value for value in (roles, boundaries, statuses, target)):
        raise ValueError("role contract inventories must be non-empty objects")
    positive = frozenset(str(value) for value in target.get("positive_roles", []))
    negative = frozenset(str(value) for value in target.get("negative_roles", []))
    masked = frozenset(str(value) for value in target.get("masked_roles", []))
    trusted = frozenset(str(value) for value in target.get("trusted_statuses", []))
    role_names = frozenset(str(value) for value in roles)
    if positive & negative or positive & masked or negative & masked:
        raise ValueError("entry target role sets overlap")
    if positive | negative | masked != role_names:
        raise ValueError("entry target does not account for every role")
    if not trusted or not trusted <= frozenset(str(value) for value in statuses):
        raise ValueError("invalid trusted status inventory")
    return RoleContract(
        roles=role_names,
        boundary_flags=frozenset(str(value) for value in boundaries),
        label_statuses=frozenset(str(value) for value in statuses),
        trusted_statuses=trusted,
        positive_roles=positive,
        negative_roles=negative,
        masked_roles=masked,
        sha256=sha256_file(contract_path),
    )


def entry_anchor_target(
    *, role: str, label_status: str | None = None, role_status: str | None = None,
    contract: RoleContract,
    mask_in_block_nonanchors: bool = False,
) -> int:
    """Return the primary one-vs-rest anchor target or its mask ablation."""

    if (label_status is None) == (role_status is None):
        raise ValueError("supply exactly one of label_status or role_status")
    status = role_status if role_status is not None else label_status
    assert status is not None
    if role not in contract.roles:
        raise ValueError(f"unknown bibliography role: {role}")
    if status not in contract.label_statuses:
        raise ValueError(f"unknown bibliography label status: {status}")
    if status not in contract.trusted_statuses or role in contract.masked_roles:
        return TARGET_MASK
    if role in contract.positive_roles:
        return TARGET_ENTRY_ANCHOR
    if mask_in_block_nonanchors and role != "NON_BIB":
        return TARGET_MASK
    if role in contract.negative_roles:
        return TARGET_NEGATIVE
    raise AssertionError(f"unaccounted role {role}")


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}: row {row_number} is not an object")
            yield row_number, row


def _source_line_inventory(source_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    inventory: dict[tuple[str, str], dict[str, Any]] = {}
    for row_number, document in _iter_jsonl(source_path):
        document_id = document.get("document_id")
        work_id = document.get("work_id")
        lines = document.get("lines")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"source row {row_number}: missing document_id")
        if not isinstance(work_id, str) or not work_id:
            raise ValueError(f"source row {row_number}: missing work_id")
        if not isinstance(lines, list) or not lines:
            raise ValueError(f"source row {row_number}: missing lines")
        for offset, line in enumerate(lines):
            if not isinstance(line, Mapping):
                raise ValueError(f"{document_id}: line {offset} is not an object")
            line_id, text, abs_idx = line.get("line_id"), line.get("text"), line.get("abs_idx")
            if not isinstance(line_id, str) or not line_id:
                raise ValueError(f"{document_id}: line {offset} has no line_id")
            if not isinstance(text, str) or not isinstance(abs_idx, int):
                raise ValueError(f"{document_id}: malformed line {line_id}")
            key = (document_id, line_id)
            if key in inventory:
                raise ValueError(f"duplicate source line identity: {key}")
            inventory[key] = {
                "work_id": work_id,
                "abs_idx": abs_idx,
                "text_sha256": text_sha256(text),
                "original_region_label": str(line.get("label", "")),
            }
    return inventory


def validate_overlay(
    *, source_path: Path, overlay_path: Path, contract: RoleContract,
) -> dict[str, Any]:
    source = _source_line_inventory(source_path)
    seen: set[tuple[str, str]] = set()
    role_counts: collections.Counter[str] = collections.Counter()
    status_counts: collections.Counter[str] = collections.Counter()
    boundary_status_counts: collections.Counter[str] = collections.Counter()
    boundary_counts: collections.Counter[str] = collections.Counter()
    target_counts: collections.Counter[int] = collections.Counter()
    target_mask_ablation_counts: collections.Counter[int] = collections.Counter()
    origins: collections.Counter[str] = collections.Counter()

    rows = 0
    overlay_schema: str | None = None
    for row_number, row in _iter_jsonl(overlay_path):
        row_schema = row.get("schema_version")
        if row_schema not in {SCHEMA_VERSION, SCHEMA_VERSION_V2}:
            raise ValueError(f"overlay row {row_number}: unsupported schema_version")
        if overlay_schema is None:
            overlay_schema = str(row_schema)
        elif row_schema != overlay_schema:
            raise ValueError("overlay mixes schema versions")
        document_id, line_id = row.get("document_id"), row.get("line_id")
        if not isinstance(document_id, str) or not isinstance(line_id, str):
            raise ValueError(f"overlay row {row_number}: invalid line identity")
        key = (document_id, line_id)
        if key in seen:
            raise ValueError(f"overlay repeats line identity {key}")
        seen.add(key)
        expected = source.get(key)
        if expected is None:
            raise ValueError(f"overlay line is absent from source: {key}")
        for field in ("work_id", "abs_idx", "text_sha256", "original_region_label"):
            if row.get(field) != expected[field]:
                raise ValueError(f"overlay row {row_number}: stale or mismatched {field}")
        role = str(row.get("role", ""))
        status = str(
            row.get("role_status", "")
            if row_schema == SCHEMA_VERSION_V2
            else row.get("label_status", "")
        )
        boundary = str(row.get("boundary_flag", ""))
        origin = row.get("label_origin")
        confidence = (
            row.get("role_confidence")
            if row_schema == SCHEMA_VERSION_V2
            else row.get("confidence")
        )
        reviewers = row.get("reviewers")
        if role not in contract.roles or status not in contract.label_statuses:
            raise ValueError(f"overlay row {row_number}: invalid role/status")
        if boundary not in contract.boundary_flags:
            raise ValueError(f"overlay row {row_number}: invalid boundary_flag")
        if not isinstance(origin, str) or not origin:
            raise ValueError(f"overlay row {row_number}: label_origin is required")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError(f"overlay row {row_number}: confidence must be in [0,1]")
        if not isinstance(reviewers, list) or not all(isinstance(value, str) and value for value in reviewers):
            raise ValueError(f"overlay row {row_number}: reviewers must be a string list")
        if status == "AGREED_REVIEW" and len(set(reviewers)) < 2:
            raise ValueError(f"overlay row {row_number}: agreed review needs two reviewers")
        if status == "ADJUDICATED" and len(set(reviewers)) < 2:
            raise ValueError(f"overlay row {row_number}: adjudication lacks review provenance")
        if status == "PROVISIONAL" and origin not in {"deterministic", "model_oof", "model_final"}:
            raise ValueError(f"overlay row {row_number}: invalid provisional origin")
        if row_schema == SCHEMA_VERSION_V2:
            boundary_status = str(row.get("boundary_status", ""))
            boundary_confidence = row.get("boundary_confidence")
            if boundary_status not in contract.label_statuses:
                raise ValueError(f"overlay row {row_number}: invalid boundary_status")
            if (
                not isinstance(boundary_confidence, (int, float))
                or isinstance(boundary_confidence, bool)
                or not 0 <= boundary_confidence <= 1
            ):
                raise ValueError(
                    f"overlay row {row_number}: boundary_confidence must be in [0,1]"
                )
            if boundary_status in {"AGREED_REVIEW", "ADJUDICATED"} and len(set(reviewers)) < 2:
                raise ValueError(
                    f"overlay row {row_number}: trusted boundary needs two reviewers"
                )
            for vote_field in ("raw_role_votes", "raw_boundary_votes"):
                votes = row.get(vote_field)
                if not isinstance(votes, Mapping) or set(votes) != set(reviewers):
                    raise ValueError(
                        f"overlay row {row_number}: {vote_field} must cover reviewers"
                    )
                if not all(isinstance(value, list) and value for value in votes.values()):
                    raise ValueError(
                        f"overlay row {row_number}: {vote_field} values must be lists"
                    )
            review_case_ids = row.get("review_case_ids")
            if not isinstance(review_case_ids, list) or not all(
                isinstance(value, str) and value for value in review_case_ids
            ):
                raise ValueError(
                    f"overlay row {row_number}: review_case_ids must be a string list"
                )
            boundary_status_counts[boundary_status] += 1

        rows += 1
        role_counts[role] += 1
        status_counts[status] += 1
        boundary_counts[boundary] += 1
        origins[origin] += 1
        target_counts[entry_anchor_target(role=role, role_status=status, contract=contract)] += 1
        target_mask_ablation_counts[
            entry_anchor_target(
                role=role,
                role_status=status,
                contract=contract,
                mask_in_block_nonanchors=True,
            )
        ] += 1

    if rows == 0:
        raise ValueError("overlay is empty")
    return {
        "schema_version": (
            "bibliography-role-overlay-validation-v2"
            if overlay_schema == SCHEMA_VERSION_V2
            else "bibliography-role-overlay-validation-v1"
        ),
        "status": "passed",
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "overlay": {"path": str(overlay_path), "sha256": sha256_file(overlay_path)},
        "contract_sha256": contract.sha256,
        "row_count": rows,
        "content_sha256": canonical_json_sha256(
            {"source": sha256_file(source_path), "overlay": sha256_file(overlay_path), "rows": rows}
        ),
        "role_counts": dict(sorted(role_counts.items())),
        "label_status_counts": dict(sorted(status_counts.items())),
        "boundary_status_counts": dict(sorted(boundary_status_counts.items())),
        "boundary_flag_counts": dict(sorted(boundary_counts.items())),
        "label_origin_counts": dict(sorted(origins.items())),
        "entry_anchor_target_counts": {str(key): value for key, value in sorted(target_counts.items())},
        "entry_anchor_mask_in_block_ablation_counts": {
            str(key): value for key, value in sorted(target_mask_ablation_counts.items())
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("bibliography_role_contract_v1.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_overlay(
        source_path=args.source.resolve(),
        overlay_path=args.overlay.resolve(),
        contract=load_role_contract(args.contract),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
