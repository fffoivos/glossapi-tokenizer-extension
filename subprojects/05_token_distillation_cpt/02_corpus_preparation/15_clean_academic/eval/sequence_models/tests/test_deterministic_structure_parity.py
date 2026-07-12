from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SEQUENCE_MODELS_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = SEQUENCE_MODELS_DIR.parent
ACADEMIC_CLEANING_DIR = EVAL_DIR.parent
RUST_CRATE = ACADEMIC_CLEANING_DIR / "reference_detector"
FIXTURE_PATH = SEQUENCE_MODELS_DIR / "fixtures" / "deterministic_structure_parity.json"
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.deterministic_structure import (  # noqa: E402
    BibRole,
    StructureDecision,
    StructureKind,
    TocRole,
    detect_structure,
)


# The Python research vocabulary is deliberately target-specific; Rust shares
# one serde snake_case enum across targets. Keep this translation explicit so
# adding or renaming a role fails the harness instead of being silently folded.
PYTHON_TOC_ROLE_TO_RUST = {
    "TOC_HEADING": "heading",
    "STRONG_TOC_ENTRY": "strong_entry_start",
    "WEAK_TOC_ENTRY": "weak_entry_start",
    "POSSIBLE_TOC_CONTINUATION": "possible_continuation",
    "HARD_OTHER": "hard_other",
    "OTHER": "other",
}
PYTHON_BIB_ROLE_TO_RUST = {
    "BIB_HEADING": "heading",
    "BIB_SUBHEADING": "subheading",
    "STRONG_ENTRY_START": "strong_entry_start",
    "WEAK_ENTRY_START": "weak_entry_start",
    "POSSIBLE_CONTINUATION": "possible_continuation",
    "HARD_OTHER": "hard_other",
    "OTHER": "other",
}
PYTHON_KIND_TO_RUST = {"TOC": "toc", "BIB": "bibliography"}


def _load_fixtures() -> list[dict[str, Any]]:
    packet = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert packet["schema_version"] == "deterministic-structure-parity-fixtures-v1"
    fixtures = packet["fixtures"]
    assert fixtures and len({fixture["id"] for fixture in fixtures}) == len(fixtures)
    for fixture in fixtures:
        prefix = fixture.get("repeat_prefix")
        fixture["expanded_lines"] = (
            [prefix["line"]] * prefix["count"] if prefix else []
        ) + fixture["lines"]
    return fixtures


def _rust_binary() -> Path:
    configured = os.environ.get("REFERENCE_DETECT_BIN")
    if configured:
        binary = Path(configured).resolve()
        if not binary.is_file():
            raise AssertionError(f"REFERENCE_DETECT_BIN is not a file: {binary}")
        return binary
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "reference_detect"],
        cwd=RUST_CRATE,
        check=True,
    )
    return RUST_CRATE / "target" / "debug" / "reference_detect"


@pytest.fixture(scope="module")
def parity_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    fixtures = _load_fixtures()
    run_dir = tmp_path_factory.mktemp("deterministic-structure-parity")
    input_path = run_dir / "fixtures.jsonl"
    output_path = run_dir / "decisions.jsonl"
    spans_path = run_dir / "spans.jsonl"
    input_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": fixture["id"],
                    "source": fixture.get("source", "python-rust-parity-fixture"),
                    "text": "\n".join(fixture["expanded_lines"]),
                },
                ensure_ascii=False,
            )
            + "\n"
            for fixture in fixtures
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            str(_rust_binary()),
            "--mode",
            "deterministic-structure",
            "--input",
            str(input_path),
            "--out-spans",
            str(spans_path),
            "--out-counters",
            str(output_path),
            "--source",
            "python-rust-parity-fixture",
        ],
        cwd=RUST_CRATE,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    rust_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rust_by_id = {row["doc_id"]: row for row in rust_rows if "doc_id" in row}
    expected_ids = {fixture["id"] for fixture in fixtures}
    assert set(rust_by_id) == expected_ids
    python_by_id = {
        fixture["id"]: detect_structure(
            fixture["expanded_lines"],
            n_physical_lines=len(fixture["expanded_lines"]),
        )
        for fixture in fixtures
    }
    return {
        "fixtures": fixtures,
        "python": python_by_id,
        "rust": rust_by_id,
    }


def _python_spans(decision: StructureDecision) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "kind": PYTHON_KIND_TO_RUST[span.kind.value],
                "line_start": span.start_line_index,
                "line_end": span.end_line_index,
                "supporting_lines": list(span.supporting_line_indices),
                "bridged_lines": list(span.bridged_line_indices),
                "terminated_by": span.terminator_line_index,
            }
            for span in decision.spans
        ],
        key=lambda span: (span["line_start"], span["line_end"], span["kind"]),
    )


def _rust_spans(decision: dict[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "kind",
        "line_start",
        "line_end",
        "supporting_lines",
        "bridged_lines",
        "terminated_by",
    )
    return sorted(
        [{field: span.get(field) for field in fields} for span in decision["spans"]],
        key=lambda span: (span["line_start"], span["line_end"], span["kind"]),
    )


def _python_conflicts(decision: StructureDecision) -> list[dict[str, int]]:
    return sorted(
        [
            {
                "toc_line_start": conflict.toc_span.start_line_index,
                "toc_line_end": conflict.toc_span.end_line_index,
                "bib_line_start": conflict.bib_span.start_line_index,
                "bib_line_end": conflict.bib_span.end_line_index,
            }
            for conflict in decision.conflicts
        ],
        key=lambda conflict: tuple(conflict.values()),
    )


def _rust_conflicts(decision: dict[str, Any]) -> list[dict[str, int]]:
    fields = (
        "toc_line_start",
        "toc_line_end",
        "bib_line_start",
        "bib_line_end",
    )
    return sorted(
        [
            {field: conflict[field] for field in fields}
            for conflict in decision["conflicts"]
        ],
        key=lambda conflict: tuple(conflict.values()),
    )


def _python_roles(
    decision: StructureDecision, target: str
) -> list[dict[str, int | str | bool]]:
    if target == "toc":
        evidence = decision.toc_evidence
        mapping = PYTHON_TOC_ROLE_TO_RUST
    else:
        evidence = decision.bib_evidence
        mapping = PYTHON_BIB_ROLE_TO_RUST
    return [
        {
            "line_index": item.line_index,
            "role": mapping[item.role.value],
            "hard_negative": item.hard_negative,
        }
        for item in evidence
    ]


def _rust_roles(
    decision: dict[str, Any], target: str
) -> list[dict[str, int | str | bool]]:
    role_field = f"{target}_role"
    return [
        {
            "line_index": item["line_index"],
            "role": item[role_field],
            # Rust's serialized `hard_negative` is a cross-target convenience.
            # Behavioral parity is target-specific, so derive it from that
            # target's role rather than leaking the other target's veto.
            "hard_negative": item[role_field] == "hard_other",
        }
        for item in decision["line_evidence"]
    ]


def test_role_mappings_cover_the_complete_python_enums() -> None:
    assert set(PYTHON_TOC_ROLE_TO_RUST) == {role.value for role in TocRole}
    assert set(PYTHON_BIB_ROLE_TO_RUST) == {role.value for role in BibRole}
    assert set(PYTHON_KIND_TO_RUST) == {kind.value for kind in StructureKind}


def test_fixture_decisions_and_target_roles_match(
    parity_results: dict[str, Any],
) -> None:
    for fixture in parity_results["fixtures"]:
        fixture_id = fixture["id"]
        python_decision = parity_results["python"][fixture_id]
        rust_decision = parity_results["rust"][fixture_id]
        python_spans = _python_spans(python_decision)
        rust_spans = _rust_spans(rust_decision)
        python_conflicts = _python_conflicts(python_decision)
        rust_conflicts = _rust_conflicts(rust_decision)
        assert python_spans == rust_spans == fixture["expected"]["spans"], fixture_id
        assert python_conflicts == rust_conflicts == fixture["expected"]["conflicts"], (
            fixture_id
        )
        for target in fixture["role_targets"]:
            assert _python_roles(python_decision, target) == _rust_roles(
                rust_decision, target
            ), (fixture_id, target)


def test_rust_decisions_bind_exact_input_text_identity(
    parity_results: dict[str, Any],
) -> None:
    for fixture in parity_results["fixtures"]:
        fixture_id = fixture["id"]
        text = "\n".join(fixture["expanded_lines"])
        decision = parity_results["rust"][fixture_id]
        source = fixture.get("source", "python-rust-parity-fixture")
        assert decision["source"] == source
        assert decision["original_chars"] == len(text)
        assert decision["original_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        expected_uid = hashlib.sha256(f"{source}\0{fixture_id}".encode()).hexdigest()
        assert decision["row_uid"] == expected_uid


def test_cv_and_notes_suppress_spans_without_mutating_local_roles(
    parity_results: dict[str, Any],
) -> None:
    for implementation in ("python", "rust"):
        decisions = parity_results[implementation]
        headerless = decisions["bib_headerless"]
        base_roles = (
            _python_roles(headerless, "bib")
            if implementation == "python"
            else _rust_roles(headerless, "bib")
        )
        for protected_id in ("cv_publications_scope", "notes_scope"):
            protected = decisions[protected_id]
            protected_roles = (
                _python_roles(protected, "bib")
                if implementation == "python"
                else _rust_roles(protected, "bib")
            )
            assert [item["role"] for item in protected_roles[1:]] == [
                item["role"] for item in base_roles[:4]
            ]
            assert all(
                item["role"] == "strong_entry_start" for item in protected_roles[1:]
            )


def test_atx_body_headings_are_exact_hard_barriers(
    parity_results: dict[str, Any],
) -> None:
    cases = (
        ("toc_plain_heading_arabic", "toc", 3),
        ("bib_continuation_atx_barrier", "bib", 4),
    )
    for fixture_id, target, barrier in cases:
        python_decision = parity_results["python"][fixture_id]
        rust_decision = parity_results["rust"][fixture_id]
        assert (
            _python_roles(python_decision, target)[barrier]
            == _rust_roles(rust_decision, target)[barrier]
            == {
                "line_index": barrier,
                "role": "hard_other",
                "hard_negative": True,
            }
        )
        assert _python_spans(python_decision)[0]["terminated_by"] == barrier
        assert _rust_spans(rust_decision)[0]["terminated_by"] == barrier
