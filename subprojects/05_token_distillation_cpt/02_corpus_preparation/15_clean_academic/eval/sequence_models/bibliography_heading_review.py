#!/usr/bin/env python3
"""Mine, independently review, and adjudicate typed bibliography headings."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import load_table
from .bibliography_role_dataset import text_sha256
from .bibliography_role_features import broad_heading_candidate
from .bibliography_role_v2 import OVERLAY_SCHEMA, TRUSTED_STATUSES, migrate_row
from .contract import canonical_json_sha256, sha256_file


PACKET_SCHEMA = "bibliography-heading-review-packet-v1"
PROVENANCE_SCHEMA = "bibliography-heading-review-provenance-v1"
REVIEW_SCHEMA = "bibliography-heading-review-v1"
RUN_SCHEMA = "bibliography-heading-review-run-v1"
ADJUDICATION_SCHEMA = "bibliography-heading-adjudication-v1"
LABELS = frozenset({"BIB_HEADER", "BIB_SUBHEADER", "NON_BIB_HEADER", "NOT_HEADER", "UNKNOWN"})


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield value


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_overlay(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in _iter_jsonl(path):
        migrated = row if row.get("schema_version") == OVERLAY_SCHEMA else migrate_row(row)
        key = (str(migrated.get("document_id", "")), str(migrated.get("line_id", "")))
        if not all(key) or key in result:
            raise ValueError("role overlay has repeated or empty identity")
        result[key] = migrated
    return result


def _candidate_id(document_id: str, line_id: str, text_hash: str) -> str:
    return hashlib.sha256(f"heading-v1\0{document_id}\0{line_id}\0{text_hash}".encode()).hexdigest()


def build_inventory(
    *, source_path: Path, base_table_dir: Path, entry_oof_path: Path,
    overlay_path: Path, split: str, code_commit: str, slurm_job_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    table = load_table(base_table_dir, expected_split=split)
    entry_probability = np.load(entry_oof_path, mmap_mode="r", allow_pickle=False)
    if entry_probability.shape != (len(table.targets),) or not np.isfinite(entry_probability).all():
        raise ValueError("entry OOF probability is malformed")
    overlay = _load_overlay(overlay_path)
    documents = [row for row in _iter_jsonl(source_path) if row.get("split") == split]
    if len(documents) != len(table.documents):
        raise ValueError("source/base document count mismatch")
    cases: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    trusted_headings = recovered_trusted = 0
    source_counts: dict[str, int] = {}
    for document_index, (source, metadata) in enumerate(zip(documents, table.documents, strict=True)):
        if source.get("document_id") != metadata["document_id"] or source.get("work_id") != metadata["work_id"]:
            raise ValueError(f"source/base identity mismatch at document {document_index}")
        lines = source.get("lines")
        if not isinstance(lines, list) or len(lines) != int(metadata["line_count"]):
            raise ValueError(f"source/base line mismatch at document {document_index}")
        start = int(metadata["line_start"])
        for offset, line in enumerate(lines):
            text = str(line.get("text", ""))
            line_id = str(line.get("line_id") or f"{metadata['document_id']}:{line['abs_idx']}")
            key = (str(metadata["document_id"]), line_id)
            previous_blank = offset > 0 and not str(lines[offset - 1].get("text", "")).strip()
            next_blank = offset + 1 < len(lines) and not str(lines[offset + 1].get("text", "")).strip()
            broad = broad_heading_candidate(text, previous_blank=previous_blank, next_blank=next_blank)
            existing = overlay.get(key)
            existing_role = str(existing.get("role", "")) if existing else ""
            trusted_existing = bool(existing and existing.get("role_status") in TRUSTED_STATUSES)
            is_trusted_heading = trusted_existing and existing_role in {
                "BIB_HEADER", "BIB_SUBHEADER", "NON_BIB_HEADER",
            }
            trusted_headings += int(is_trusted_heading)
            recovered_trusted += int(is_trusted_heading and broad)
            if not (broad or is_trusted_heading):
                continue
            absolute = start + offset
            text_hash = text_sha256(text)
            candidate_id = _candidate_id(str(metadata["document_id"]), line_id, text_hash)
            context_start, context_end = max(0, offset - 5), min(len(lines), offset + 6)
            context = [
                {
                    "line_id": str(local.get("line_id") or f"{metadata['document_id']}:{local['abs_idx']}"),
                    "abs_idx": int(local["abs_idx"]), "text": str(local.get("text", "")),
                    "target": local_offset == offset,
                }
                for local_offset, local in enumerate(lines[context_start:context_end], context_start)
            ]
            cases.append({"candidate_id": candidate_id, "context": context})
            provenance.append({
                "candidate_id": candidate_id, "document_id": str(metadata["document_id"]),
                "work_id": str(metadata["work_id"]), "source": str(metadata["source"]),
                "fold": int(table.folds[absolute]), "line_id": line_id,
                "abs_idx": int(line["abs_idx"]), "text": text, "text_sha256": text_hash,
                "original_region_label": str(line.get("label", "")),
                "entry_probability": float(entry_probability[absolute]),
                "existing_trusted_role": existing_role if trusted_existing else None,
            })
            source_counts[str(metadata["source"])] = source_counts.get(str(metadata["source"]), 0) + 1
    packet = {
        "schema_version": PACKET_SCHEMA,
        "blinding": {
            "entry_predictions_hidden": True, "original_region_labels_hidden": True,
            "existing_role_labels_hidden": True,
        },
        "cases": cases,
    }
    provenance_value = {
        "schema_version": PROVENANCE_SCHEMA,
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "base_table_manifest_sha256": sha256_file(base_table_dir / "manifest.json"),
        "entry_oof_sha256": sha256_file(entry_oof_path),
        "overlay_sha256": sha256_file(overlay_path),
        "cases": provenance,
    }
    report = {
        "schema_version": "bibliography-heading-inventory-v1",
        "status": "passed_complete_broad_heading_inventory",
        "candidate_count": len(cases), "candidate_counts_by_source": source_counts,
        "trusted_heading_count": trusted_headings,
        "broad_predicate_trusted_heading_recall": (
            recovered_trusted / trusted_headings if trusted_headings else 1.0
        ),
        "inventory_trusted_heading_recall": 1.0,
        "code_commit": code_commit, "slurm_job_id": slurm_job_id,
        "packet_content_sha256": canonical_json_sha256(packet),
        "provenance_content_sha256": canonical_json_sha256(provenance_value),
    }
    return packet, provenance_value, report


def _ordered_cases(cases: Sequence[Mapping[str, Any]], pass_id: str) -> list[dict[str, Any]]:
    if pass_id == "pass-a":
        return [dict(row) for row in sorted(cases, key=lambda row: str(row["candidate_id"]))]
    if pass_id == "pass-b":
        return [dict(row) for row in sorted(cases, key=lambda row: hashlib.sha256(str(row["candidate_id"]).encode()).digest(), reverse=True)]
    raise ValueError("pass-id must be pass-a or pass-b")


def validate_review(payload: Mapping[str, Any], expected: Mapping[str, Mapping[str, Any]], reviewer: str) -> dict[str, Any]:
    if payload.get("schema_version") != REVIEW_SCHEMA or payload.get("reviewer") != reviewer:
        raise ValueError("heading review schema/reviewer mismatch")
    rows = payload.get("cases")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("heading review omits or invents cases")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"candidate_id", "label", "confidence", "reason"}:
            raise ValueError("malformed heading review row")
        candidate_id, label = str(row.get("candidate_id", "")), str(row.get("label", ""))
        confidence = row.get("confidence")
        if candidate_id not in expected or candidate_id in result or label not in LABELS:
            raise ValueError("invalid/repeated heading review identity or label")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError("invalid heading confidence")
        if not isinstance(row.get("reason"), str) or not str(row["reason"]).strip():
            raise ValueError("heading reason is required")
        result[candidate_id] = dict(row)
    if set(result) != set(expected):
        raise ValueError("heading review case inventory mismatch")
    return {"schema_version": REVIEW_SCHEMA, "reviewer": reviewer, "cases": [result[key] for key in sorted(result)]}


def run_reviews(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packet).resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("schema_version") != PACKET_SCHEMA:
        raise ValueError("unsupported heading packet")
    cases = _ordered_cases(packet["cases"], args.pass_id)
    prompt_path, schema_path = Path(args.prompt).resolve(), Path(args.output_schema).resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    batches = [cases[index : index + args.batch_size] for index in range(0, len(cases), args.batch_size)]
    output_dir = Path(args.batch_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def execute(batch: list[dict[str, Any]]) -> dict[str, Any]:
        contract = {
            "pass_id": args.pass_id, "reviewer": args.reviewer_id, "model": args.model,
            "cases": [[row["candidate_id"], canonical_json_sha256(row)] for row in batch],
            "prompt_sha256": sha256_file(prompt_path), "schema_sha256": sha256_file(schema_path),
        }
        batch_id = canonical_json_sha256(contract)
        final = output_dir / f"{batch_id}.json"
        expected = {str(row["candidate_id"]): row for row in batch}
        if final.exists():
            stored = json.loads(final.read_text(encoding="utf-8"))
            if stored.get("contract") != contract:
                raise ValueError("existing heading batch contract mismatch")
            return validate_review(stored["review"], expected, args.reviewer_id)
        envelope = {
            "reviewer": args.reviewer_id, "independence": "No other pass is supplied.",
            "cases": batch,
        }
        with tempfile.TemporaryDirectory(prefix="bib-heading-review-") as directory:
            response = Path(directory) / "response.json"
            workspace = Path(directory) / "empty"
            workspace.mkdir()
            command = [
                "codex", "exec", "--model", args.model, "--sandbox", "read-only", "--ephemeral",
                "--skip-git-repo-check", "--cd", str(workspace), "--config",
                f'model_reasoning_effort="{args.reasoning_effort}"', "--output-schema", str(schema_path),
                "--output-last-message", str(response), "-",
            ]
            completed = subprocess.run(
                command, input=prompt.rstrip() + "\n\n" + json.dumps(envelope, ensure_ascii=False),
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=args.timeout_seconds, check=False,
            )
            if completed.returncode != 0 or not response.is_file():
                raise RuntimeError(f"heading review batch failed: {completed.stderr[-2000:]}")
            review = validate_review(json.loads(response.read_text(encoding="utf-8")), expected, args.reviewer_id)
        _write_json_new(final, {"schema_version": "bibliography-heading-review-batch-v1", "contract": contract, "review": review})
        return review

    reviews: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        reviews.extend(executor.map(execute, batches))
    rows = [row for review in reviews for row in review["cases"]]
    rows.sort(key=lambda row: row["candidate_id"])
    aggregate = {"schema_version": REVIEW_SCHEMA, "reviewer": args.reviewer_id, "cases": rows}
    _write_json_new(Path(args.responses_out).resolve(), aggregate)
    receipt = {
        "schema_version": RUN_SCHEMA, "status": "passed", "pass_id": args.pass_id,
        "reviewer": args.reviewer_id, "model": args.model, "candidate_count": len(rows),
        "packet_sha256": sha256_file(packet_path),
        "responses_sha256": sha256_file(Path(args.responses_out).resolve()),
    }
    _write_json_new(Path(args.receipt_out).resolve(), receipt)
    return receipt


def adjudicate(
    provenance: Mapping[str, Any], review_a: Mapping[str, Any], review_b: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = {str(row["candidate_id"]): row for row in provenance["cases"]}
    a = {str(row["candidate_id"]): row for row in review_a["cases"]}
    b = {str(row["candidate_id"]): row for row in review_b["cases"]}
    if set(cases) != set(a) or set(cases) != set(b):
        raise ValueError("heading review/provenance inventories differ")
    output = []
    agreement = 0
    for candidate_id in sorted(cases):
        source, left, right = cases[candidate_id], a[candidate_id], b[candidate_id]
        same = left["label"] == right["label"] and left["label"] != "UNKNOWN"
        agreement += int(same)
        role = str(left["label"]) if same else "UNKNOWN"
        if role == "NOT_HEADER":
            existing = source.get("existing_trusted_role")
            role = str(existing) if existing in {
                "ENTRY", "CONTINUATION", "FILLER", "OTHER",
            } else "OTHER"
        status = "AGREED_REVIEW" if same else "UNRESOLVED"
        boundary = "HARD_STOP" if role in {"BIB_HEADER", "NON_BIB_HEADER"} else "NONE"
        left_boundary = "HARD_STOP" if left["label"] in {"BIB_HEADER", "NON_BIB_HEADER"} else "NONE"
        right_boundary = "HARD_STOP" if right["label"] in {"BIB_HEADER", "NON_BIB_HEADER"} else "NONE"
        output.append({
            "schema_version": OVERLAY_SCHEMA,
            "document_id": source["document_id"], "work_id": source["work_id"],
            "line_id": source["line_id"], "abs_idx": source["abs_idx"],
            "text_sha256": source["text_sha256"],
            "original_region_label": source["original_region_label"],
            "role": role, "role_status": status,
            "role_confidence": min(float(left["confidence"]), float(right["confidence"])),
            "boundary_flag": boundary, "boundary_status": status,
            "boundary_confidence": min(float(left["confidence"]), float(right["confidence"])),
            "reviewers": [review_a["reviewer"], review_b["reviewer"]],
            "review_case_ids": [candidate_id],
            "raw_role_votes": {review_a["reviewer"]: [left["label"]], review_b["reviewer"]: [right["label"]]},
            "raw_boundary_votes": {
                review_a["reviewer"]: [left_boundary], review_b["reviewer"]: [right_boundary],
            },
            "label_origin": "dual_heading_review" if same else "dual_heading_review_disagreement",
            "role_contract_schema": "bibliography-role-contract-v2",
        })
    report = {
        "schema_version": ADJUDICATION_SCHEMA,
        "status": "passed_exact_agreement_only",
        "candidate_count": len(output), "agreement_count": agreement,
        "agreement_rate": agreement / len(output) if output else 0.0,
        "trusted_count": agreement, "masked_disagreement_count": len(output) - agreement,
        "overlay_content_sha256": canonical_json_sha256(output),
    }
    return output, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--source", type=Path, required=True)
    inventory.add_argument("--base-table-dir", type=Path, required=True)
    inventory.add_argument("--entry-oof", type=Path, required=True)
    inventory.add_argument("--overlay", type=Path, required=True)
    inventory.add_argument("--split", default="train")
    inventory.add_argument("--packet-out", type=Path, required=True)
    inventory.add_argument("--provenance-out", type=Path, required=True)
    inventory.add_argument("--receipt-out", type=Path, required=True)
    inventory.add_argument("--code-commit", required=True)
    inventory.add_argument("--slurm-job-id", required=True)
    run = sub.add_parser("run")
    run.add_argument("--packet", required=True)
    run.add_argument("--pass-id", choices=("pass-a", "pass-b"), required=True)
    run.add_argument("--reviewer-id", required=True)
    run.add_argument("--model", default="gpt-5.6-luna")
    run.add_argument("--reasoning-effort", default="high")
    run.add_argument("--prompt", default=str(Path(__file__).with_name("bibliography_heading_review_prompt.md")))
    run.add_argument("--output-schema", default=str(Path(__file__).with_name("bibliography_heading_review.schema.json")))
    run.add_argument("--batch-dir", required=True)
    run.add_argument("--responses-out", required=True)
    run.add_argument("--receipt-out", required=True)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--timeout-seconds", type=int, default=1800)
    adj = sub.add_parser("adjudicate")
    adj.add_argument("--provenance", type=Path, required=True)
    adj.add_argument("--review-a", type=Path, required=True)
    adj.add_argument("--review-b", type=Path, required=True)
    adj.add_argument("--overlay-out", type=Path, required=True)
    adj.add_argument("--receipt-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "inventory":
        packet, provenance, receipt = build_inventory(
            source_path=args.source.resolve(), base_table_dir=args.base_table_dir.resolve(),
            entry_oof_path=args.entry_oof.resolve(), overlay_path=args.overlay.resolve(), split=args.split,
            code_commit=args.code_commit, slurm_job_id=args.slurm_job_id,
        )
        _write_json_new(args.packet_out.resolve(), packet)
        _write_json_new(args.provenance_out.resolve(), provenance)
        _write_json_new(args.receipt_out.resolve(), receipt)
    elif args.command == "run":
        run_reviews(args)
    else:
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        review_a = json.loads(args.review_a.read_text(encoding="utf-8"))
        review_b = json.loads(args.review_b.read_text(encoding="utf-8"))
        rows, receipt = adjudicate(provenance, review_a, review_b)
        args.overlay_out.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.overlay_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        receipt["overlay_sha256"] = sha256_file(args.overlay_out)
        _write_json_new(args.receipt_out, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
