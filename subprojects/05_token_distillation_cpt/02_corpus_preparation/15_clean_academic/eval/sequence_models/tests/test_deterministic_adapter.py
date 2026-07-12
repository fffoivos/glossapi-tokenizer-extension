from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.deterministic_adapter import (  # noqa: E402
    AblationMode,
    AdapterError,
    SealedPartitionError,
    build_prediction_rows,
    predict_document,
    prediction_row,
    read_base_predictions,
    read_prediction_documents,
    write_prediction_rows,
)
from sequence_models.evaluate import read_predictions as read_evaluation_predictions  # noqa: E402


@dataclass(frozen=True)
class Line:
    line_id: str
    abs_idx: int
    text: str


@dataclass(frozen=True)
class Document:
    document_id: str
    work_id: str
    source: str
    split: str
    n_physical_lines: int
    lines: tuple[Line, ...]
    n_present_lines: int | None = None
    coverage: str | None = "full_document"


def document(
    texts: list[str],
    *,
    document_id: str = "doc-1",
    coordinates: list[int] | None = None,
    split: str = "validation",
) -> Document:
    indexes = coordinates or list(range(len(texts)))
    assert len(indexes) == len(texts)
    lines = tuple(
        Line(f"{document_id}:{position}", abs_idx, text)
        for position, (abs_idx, text) in enumerate(zip(indexes, texts))
    )
    return Document(
        document_id=document_id,
        work_id=f"work-{document_id}",
        source="synthetic",
        split=split,
        n_physical_lines=max(indexes) + 2,
        lines=lines,
        n_present_lines=len(lines),
    )


def toc_document(
    *, document_id: str = "toc", coordinates: list[int] | None = None
) -> Document:
    return document(
        ["ΠΕΡΙΕΧΟΜΕΝΑ", "1. Εισαγωγή ........ 7", "2. Μέθοδος ........ 15"],
        document_id=document_id,
        coordinates=coordinates,
    )


def bib_document(*, document_id: str = "bib") -> Document:
    return document(
        [
            "ΒΙΒΛΙΟΓΡΑΦΙΑ",
            "Smith, J. (2019). First title. London: Press.",
            "Brown, K. (2020). Second title. London: Press.",
        ],
        document_id=document_id,
    )


def base_row(doc: Document, labels: list[str]) -> dict[str, object]:
    return {
        "schema_version": "academic-structure-predictions-v1",
        "model_id": "synthetic-base",
        "document_id": doc.document_id,
        "work_id": doc.work_id,
        "source": doc.source,
        "split": doc.split,
        "lines": [
            {"line_id": line.line_id, "abs_idx": line.abs_idx, "prediction": label}
            for line, label in zip(doc.lines, labels)
        ],
    }


class TestRulesOnlyPrediction:
    @pytest.mark.parametrize(
        ("doc", "target"),
        [(toc_document(), "TOC"), (bib_document(), "BIB")],
    )
    def test_rules_only_emits_confirmed_formal_blocks(
        self, doc: Document, target: str
    ) -> None:
        result = predict_document(doc, AblationMode.RULES_ONLY)
        assert result.labels == (target,) * len(doc.lines)
        assert all(
            "DETERMINISTIC_SPAN" in reasons for reasons in result.line_reason_codes
        )

    def test_large_blank_gaps_break_a_rule_span_but_preserve_identities(self) -> None:
        doc = toc_document(coordinates=[2, 5, 9])
        row = prediction_row(doc, "rules-only")
        assert row["schema_version"] == "academic-structure-predictions-v1"
        assert row["model_id"] == "deterministic-hybrid-rules-only"
        assert [line["line_id"] for line in row["lines"]] == [
            line.line_id for line in doc.lines
        ]
        assert [line["abs_idx"] for line in row["lines"]] == [2, 5, 9]
        assert [line["prediction"] for line in row["lines"]] == ["O"] * 3

    def test_two_known_blank_gaps_can_be_bridged(self) -> None:
        doc = toc_document(coordinates=[0, 2, 4])

        assert predict_document(doc, "rules-only").labels == ("TOC",) * 3

    def test_unrepresented_present_line_gap_is_a_hard_barrier(self) -> None:
        doc = replace(
            toc_document(coordinates=[0, 1000, 1001]),
            n_present_lines=100,
        )

        result = predict_document(doc, "rules-only")
        assert result.labels == ("O", "O", "O")
        assert any(
            "TOC_NEGATIVE_UNREPRESENTED_PHYSICAL_GAP" in row.reason_codes
            for row in result.structure_decision.toc_evidence
        )


class TestNamedHybridAblations:
    def test_headerless_bibliography_is_support_only_not_an_independent_addition(
        self,
    ) -> None:
        doc = document(
            [
                "Smith, J. (2018). First title. London: Press.",
                "Brown, K. (2019). Second title. London: Press.",
                "Jones, P. (2020). Third title. London: Press.",
                "White, R. (2021). Fourth title. London: Press.",
            ]
        )

        result = predict_document(doc, "base-plus-rules", ["O"] * 4)
        assert result.labels == ("O",) * 4
        assert all(
            "DETERMINISTIC_HEADERLESS_BIB_SUPPORT_ONLY" in reasons
            for reasons in result.line_reason_codes
        )

    def test_only_plus_modes_add_deterministic_spans(self) -> None:
        doc = toc_document()
        base = ("O", "O", "O")
        expected = {
            AblationMode.BASE_PLUS_RULES: ("TOC", "TOC", "TOC"),
            AblationMode.BASE_RULES_VETO: ("O", "O", "O"),
            AblationMode.BASE_PLUS_RULES_VETO: ("TOC", "TOC", "TOC"),
        }
        for mode, labels in expected.items():
            assert predict_document(doc, mode, base).labels == labels

    def test_target_specific_hard_negatives_do_not_veto_the_other_target(self) -> None:
        toc_negative = document(["Αναφορά 2024"])
        assert predict_document(toc_negative, "base-plus-rules", ["TOC"]).labels == (
            "O",
        )
        assert predict_document(toc_negative, "base-plus-rules", ["BIB"]).labels == (
            "BIB",
        )

        bib_negative = document(["ΚΕΦΑΛΑΙΟ 2 Μεθοδολογία"])
        assert predict_document(bib_negative, "base-plus-rules", ["BIB"]).labels == (
            "O",
        )
        assert predict_document(bib_negative, "base-plus-rules", ["TOC"]).labels == (
            "TOC",
        )

    def test_hard_negative_splits_are_singleton_vetoed_only_in_veto_modes(self) -> None:
        doc = document(["Πρόλογος", "Αναφορά 2024", "Επίλογος"])
        base = ["TOC", "TOC", "TOC"]
        assert predict_document(doc, "base-plus-rules", base).labels == (
            "TOC",
            "O",
            "TOC",
        )
        assert predict_document(doc, "base-rules-veto", base).labels == ("O", "O", "O")
        assert predict_document(doc, "base-plus-rules-veto", base).labels == (
            "O",
            "O",
            "O",
        )

    def test_veto_keeps_a_non_singleton_ml_run(self) -> None:
        doc = document(["Short citation title", "Second citation title"])
        assert predict_document(doc, "base-rules-veto", ["BIB", "BIB"]).labels == (
            "BIB",
            "BIB",
        )

    def test_same_target_rule_support_protects_an_ml_singleton_without_adding(
        self,
    ) -> None:
        doc = toc_document()
        result = predict_document(doc, "base-rules-veto", ["O", "TOC", "O"])
        assert result.labels == ("O", "TOC", "O")
        assert "DETERMINISTIC_SAME_TARGET_SUPPORT" in result.line_reason_codes[1]

    @pytest.mark.parametrize("mode", ["base-plus-rules", "base-rules-veto"])
    def test_rule_base_target_conflict_fails_closed_in_plus_and_veto_modes(
        self, mode: str
    ) -> None:
        doc = toc_document()
        result = predict_document(doc, mode, ["O", "BIB", "O"])
        assert result.labels[1] == "O"
        assert "RULE_BASE_OVERLAP_FAIL_CLOSED" in result.line_reason_codes[1]

    def test_internal_toc_bib_overlap_withholds_rule_predictions(self) -> None:
        doc = document(
            [
                "ΠΕΡΙΕΧΟΜΕΝΑ",
                "[1] Smith (2018). First title ........ 7",
                "[2] Brown (2019). Second title ........ 15",
                "[3] Jones (2020). Third title ........ 29",
                "[4] White (2021). Fourth title ........ 40",
            ]
        )
        rules = predict_document(doc, "rules-only")
        assert rules.labels == ("O",) * 5
        assert len(rules.structure_decision.conflicts) == 1
        base = predict_document(
            doc, "base-plus-rules", ["O", "BIB", "BIB", "BIB", "BIB"]
        )
        assert base.labels == ("O",) * 5
        assert all(
            "RULE_RULE_OVERLAP_FAIL_CLOSED" in base.line_reason_codes[position]
            for position in range(1, 5)
        )

    def test_synthetic_known_blank_inside_overlap_is_skipped_fail_closed(
        self,
    ) -> None:
        doc = document(
            [
                "ΠΕΡΙΕΧΟΜΕΝΑ",
                "[1] Smith (2018). First title ........ 7",
                "[2] Brown (2019). Second title ........ 15",
                "[3] Jones (2020). Third title ........ 29",
                "[4] White (2021). Fourth title ........ 40",
            ],
            coordinates=[0, 1, 3, 4, 5],
        )

        result = predict_document(doc, "rules-only")
        assert result.labels == ("O",) * 5
        assert len(result.structure_decision.conflicts) == 1

    def test_mode_contract_rejects_missing_or_irrelevant_base_predictions(self) -> None:
        doc = toc_document()
        with pytest.raises(AdapterError, match="requires base"):
            predict_document(doc, "base-plus-rules")
        with pytest.raises(AdapterError, match="must not receive"):
            predict_document(doc, "rules-only", ["O", "O", "O"])
        with pytest.raises(AdapterError, match="unknown ablation"):
            predict_document(doc, "not-a-mode", ["O", "O", "O"])


class TestJsonlContracts:
    def test_reader_does_not_retain_annotation_labels(self, tmp_path: Path) -> None:
        row = {
            "schema_version": "academic-structure-gold-v1",
            "document_id": "input-1",
            "work_id": "work-input-1",
            "source": "synthetic",
            "split": "validation",
            "n_physical_lines": 2,
            "annotation": {"status": "LLM_silver"},
            "lines": [
                {
                    "line_id": "input-1:0",
                    "abs_idx": 0,
                    "text": "Ordinary text",
                    "label": "BIB",
                    "token_count": 2,
                }
            ],
        }
        path = tmp_path / "input.jsonl"
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        loaded = read_prediction_documents(path)
        assert len(loaded) == 1
        assert not hasattr(loaded[0].lines[0], "label")
        assert loaded[0].lines[0].text == "Ordinary text"

    @pytest.mark.parametrize("split", ["test", "historical_test", "sealed_test"])
    def test_json_reader_rejects_sealed_split_before_line_materialisation(
        self, tmp_path: Path, split: str
    ) -> None:
        row = {
            "schema_version": "academic-structure-gold-v1",
            "document_id": "sealed",
            "work_id": "sealed",
            "source": "sealed",
            "split": split,
            "n_physical_lines": 1,
            "lines": "this intentionally is not a line list",
        }
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(SealedPartitionError, match="sealed split"):
            read_prediction_documents(path)

    def test_object_entry_point_rejects_sealed_split(self) -> None:
        doc = replace(toc_document(), split="test")
        with pytest.raises(SealedPartitionError, match="sealed split"):
            predict_document(doc, "rules-only")

    def test_base_prediction_join_requires_exact_line_identity(
        self, tmp_path: Path
    ) -> None:
        doc = toc_document()
        row = base_row(doc, ["O", "O", "O"])
        row["lines"][1]["abs_idx"] = 999
        path = tmp_path / "bad-base.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(AdapterError, match="identity/order"):
            read_base_predictions(path, [doc])

    def test_base_prediction_metadata_cannot_smuggle_a_sealed_split(
        self, tmp_path: Path
    ) -> None:
        doc = toc_document()
        row = base_row(doc, ["O", "O", "O"])
        row["split"] = "TEST "
        row["lines"] = "not inspected after the sealed split is observed"
        path = tmp_path / "sealed-base.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(SealedPartitionError, match="sealed split"):
            read_base_predictions(path, [doc])

    def test_base_prediction_document_metadata_must_match(self, tmp_path: Path) -> None:
        doc = toc_document()
        row = base_row(doc, ["O", "O", "O"])
        row["work_id"] = "wrong-work"
        path = tmp_path / "wrong-work.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(AdapterError, match="work_id identity mismatch"):
            read_base_predictions(path, [doc])

    def test_generated_file_round_trips_as_an_exact_base_prediction(
        self, tmp_path: Path
    ) -> None:
        doc = toc_document()
        rows = build_prediction_rows([doc], "rules-only")
        output = tmp_path / "rules.jsonl"
        write_prediction_rows(output, rows)
        loaded = read_base_predictions(output, [doc])
        assert loaded == {doc.document_id: ("TOC", "TOC", "TOC")}
        assert read_evaluation_predictions(output, [doc]) == {
            doc.document_id: ["TOC", "TOC", "TOC"]
        }
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            write_prediction_rows(output, rows)

    def test_base_inventory_must_match_documents_exactly(self) -> None:
        doc = toc_document()
        with pytest.raises(AdapterError, match="omit"):
            build_prediction_rows([doc], "base-plus-rules", {})
        with pytest.raises(AdapterError, match="inventory"):
            build_prediction_rows(
                [doc],
                "base-plus-rules",
                {doc.document_id: ["O", "O", "O"], "extra": ["O"]},
            )
