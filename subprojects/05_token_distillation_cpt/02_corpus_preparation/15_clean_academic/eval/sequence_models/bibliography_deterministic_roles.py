#!/usr/bin/env python3
"""Materialize deterministic non-bibliography line roles for train only."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .bibliography_v2 import analyze_bibliography_line_v2
from .bibliography_scope_rules import AUXILIARY_SCOPE_HEADINGS
from .deterministic_structure import _heading_key


SCHEMA_VERSION = "bibliography-deterministic-negative-roles-v2"
ROLE_NAMES = (
    "figure_caption",
    "table_or_equation",
    "exact_negative_scope_heading",
    "generic_markdown_heading",
    "footnote",
    "running_or_enumerated_prose",
    "legal_procedure",
    "other_explicit_negative",
)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_index(reason_codes: Sequence[str]) -> int:
    joined = " ".join(reason_codes)
    if "FIGURE_CAPTION" in joined:
        return 0
    if "TABLE" in joined or "EQUATION" in joined:
        return 1
    if any(
        marker in joined
        for marker in (
            "CV_PUBLICATIONS_HEADING",
            "NOTES_HEADING",
            "AUXILIARY_HEADING",
            "BODY_HEADING",
        )
    ):
        return 2
    if "NONSTRUCTURAL_MARKDOWN_HEADING" in joined:
        return 3
    if "FOOTNOTE" in joined:
        return 4
    if any(
        marker in joined
        for marker in (
            "RUNNING_PROSE",
            "INLINE_CITATION_PROSE",
            "NARRATIVE_AUTHOR_YEAR_PROSE",
            "LONG_ENUMERATED_PROSE",
        )
    ):
        return 5
    if "LEGAL_PROCEDURE" in joined:
        return 6
    return 7


def _analyze_document(
    task: tuple[str, list[Mapping[str, Any]]],
) -> tuple[str, np.ndarray, dict[str, int]]:
    document_id, lines = task
    roles = np.zeros((len(lines), len(ROLE_NAMES)), dtype=np.uint8)
    counts = {name: 0 for name in ROLE_NAMES}
    for line_index, line in enumerate(lines):
        text = line.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{document_id}: line {line_index} has invalid text")
        evidence = analyze_bibliography_line_v2(text, line_index)
        if not evidence.hard_negative:
            continue
        role = (
            ROLE_NAMES.index("exact_negative_scope_heading")
            if _heading_key(text) in AUXILIARY_SCOPE_HEADINGS
            else _role_index(evidence.reason_codes)
        )
        roles[line_index, role] = 1
        counts[ROLE_NAMES[role]] += 1
    return document_id, roles, counts


def _iter_train_tasks(
    path: Path, expected_ids: set[str]
) -> Iterable[tuple[str, list[Mapping[str, Any]]]]:
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            # Fail closed before parsing: validation text is never deserialized.
            if '"split": "train"' not in line:
                continue
            row = json.loads(line)
            document_id = str(row.get("document_id"))
            if document_id not in expected_ids:
                continue
            if document_id in seen:
                raise ValueError(f"duplicate train document {document_id}")
            lines = row.get("lines")
            if not isinstance(lines, list):
                raise ValueError(f"input row {row_number} has no line inventory")
            seen.add(document_id)
            yield document_id, lines
            if seen == expected_ids:
                return
    missing = sorted(expected_ids - seen)
    raise ValueError(f"missing train documents: {missing[:5]}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    expected = {
        str(document["document_id"]): (index, int(document["line_start"]), int(document["line_end"]))
        for index, document in enumerate(table.documents)
    }
    if len(expected) != len(table.documents):
        raise ValueError("train table contains duplicate document IDs")
    roles = np.zeros((len(table.targets), len(ROLE_NAMES)), dtype=np.uint8)
    reason_counts = {name: 0 for name in ROLE_NAMES}
    completed: set[str] = set()

    def consume(result: tuple[str, np.ndarray, dict[str, int]]) -> None:
        document_id, document_roles, counts = result
        _, start, end = expected[document_id]
        if len(document_roles) != end - start:
            raise ValueError(f"{document_id}: source/table line alignment failure")
        roles[start:end] = document_roles
        completed.add(document_id)
        for name, count in counts.items():
            reason_counts[name] += int(count)

    tasks = _iter_train_tasks(input_path, set(expected))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        pending: set[concurrent.futures.Future[Any]] = set()
        for task in tasks:
            pending.add(executor.submit(_analyze_document, task))
            if len(pending) < 2 * int(args.workers):
                continue
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                consume(future.result())
        for future in concurrent.futures.as_completed(pending):
            consume(future.result())
    if completed != set(expected):
        raise ValueError("deterministic-role output is incomplete")

    hard_negative = np.any(roles > 0, axis=1)
    gold_bib = table.original_labels == LABEL_TO_ID["BIB"]
    output_dir.mkdir(parents=True)
    _save_array(output_dir / "negative_roles.npy", roles)
    _save_array(output_dir / "hard_negative.npy", hard_negative)
    role_rows = []
    for index, name in enumerate(ROLE_NAMES):
        role = roles[:, index] > 0
        role_rows.append(
            {
                "role": name,
                "line_count": int(np.count_nonzero(role)),
                "silver_bib_line_count": int(np.count_nonzero(role & gold_bib)),
                "silver_non_bib_line_count": int(np.count_nonzero(role & ~gold_bib)),
                "silver_bib_fraction_with_role": float(
                    np.count_nonzero(role & gold_bib) / max(np.count_nonzero(gold_bib), 1)
                ),
                "role_precision_as_non_bib": float(
                    np.count_nonzero(role & ~gold_bib) / max(np.count_nonzero(role), 1)
                ),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_train_only_roles_validation_unopened",
        "role_names": list(ROLE_NAMES),
        "role_reference": {
            "figure_caption": "Line explicitly begins as a numbered figure or image caption.",
            "table_or_equation": "Line is explicitly formatted as a table row or equation rather than a bibliography entry.",
            "exact_negative_scope_heading": "Exact structural heading denotes notes, publications/CV, body content, abbreviations, figure/table lists, related material, or another explicit non-bibliography section.",
            "generic_markdown_heading": "Unknown Markdown heading; retained separately because an unlisted bibliography subheading is not safely negative scope.",
            "footnote": "Line has an explicit footnote shape that is not also a valid numbered bibliography entry.",
            "running_or_enumerated_prose": "Line has an explicit running, narrative-citation, inline-citation, or long enumerated-prose shape.",
            "legal_procedure": "Line begins as legal/procedural body text without a legal bibliography citation.",
            "other_explicit_negative": "A deterministic hard-negative role not assigned to the six named structural groups.",
        },
        "line_count": len(table.targets),
        "hard_negative_line_count": int(np.count_nonzero(hard_negative)),
        "hard_negative_silver_bib_line_count": int(
            np.count_nonzero(hard_negative & gold_bib)
        ),
        "hard_negative_silver_bib_fraction": float(
            np.count_nonzero(hard_negative & gold_bib)
            / max(np.count_nonzero(gold_bib), 1)
        ),
        "role_counts_from_workers": reason_counts,
        "roles": role_rows,
        "input_sha256": _sha256(input_path),
        "table_receipt_sha256": _sha256(Path(args.table_dir) / "receipt.json"),
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "deterministic_roles_report.json", report)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    receipt = {**report, "outputs": outputs}
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
