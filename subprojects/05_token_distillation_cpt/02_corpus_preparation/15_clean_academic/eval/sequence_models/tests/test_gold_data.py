from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.gold_data import (
    HERE,
    PHASE04_DIR,
    _double_ids,
    _expand_judgments,
    _hash_id,
    _open_state,
    _packet,
    _select_candidates,
    _validate_human_file,
    Identity,
)
from sequence_models.contract import build_split_manifest


CONFIG = {
    "packet_schema_version": "academic-structure-human-packet-v1",
    "annotation_schema_version": "academic-structure-human-annotation-v1",
    "adjudication_schema_version": "academic-structure-human-adjudication-v1",
    "human_attestation": "human-attestation",
}


def candidate(source: str, index: int, *, text_hash: str | None = None) -> dict:
    document_id = f"{source}-doc-{index}"
    return {
        "gold_source": source,
        "sample_priority": f"{index:064x}",
        "document_id": document_id,
        "work_id": f"{source}-work-{index}",
        "representation_id": f"{source}-rep-{index}",
        "upstream_document_id": f"upstream-{index}",
        "canonical_work_key": f"{source}\0work-{index}",
        "text_sha256": text_hash or f"{index + 100:064x}",
        "text": "first\n\nlast\n",
        "source_dataset": source,
        "route_id": source,
        "route_revision": "a" * 40,
        "artifact_path": f"data/{source}.parquet",
        "row_group": 0,
        "row_offset": index,
    }


class SamplingTests(unittest.TestCase):
    def test_pipeline_routes_are_exact_phase04_embedded_routes(self) -> None:
        pipeline = json.loads((HERE / "gold_pipeline.json").read_text(encoding="utf-8"))
        sources = json.loads((PHASE04_DIR / "configs" / "sources.json").read_text(encoding="utf-8"))
        tracked = {route["source_id"] for route in sources["embedded_structural_routes"]}
        configured = {route["phase04_route_id"] for route in pipeline["routes"]}
        self.assertEqual(configured, {"greek_phd", "openarchives", "kallipos", "pergamos"})
        self.assertTrue(configured <= tracked)

    def test_packet_preserves_every_physical_line_including_trailing_blank(self) -> None:
        packet = _packet(candidate("greek_phd", 1), CONFIG, "test")
        self.assertEqual([line["text"] for line in packet["physical_lines"]], ["first", "", "last", ""])
        self.assertEqual(packet["n_physical_lines"], 4)
        self.assertEqual(packet["n_present_lines"], 2)
        self.assertNotIn("prediction", json.dumps(packet).lower())

    def test_selection_is_priority_ordered_and_deduplicates_work_and_exact_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = _open_state(Path(directory) / "state.sqlite", {"binding": "one"})
            rows = [candidate("s", index) for index in range(5)]
            rows[1]["work_id"] = rows[0]["work_id"]
            rows[2]["text_sha256"] = rows[0]["text_sha256"]
            columns = list(rows[0])
            for row in rows:
                connection.execute(
                    f"INSERT INTO candidates({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    [row[column] for column in columns],
                )
            connection.commit()
            selected = _select_candidates(connection, ["s"], 3)
            self.assertEqual([row["document_id"] for row in selected], ["s-doc-0", "s-doc-3", "s-doc-4"])
            connection.close()

    def test_double_annotation_assignment_is_deterministic_and_meets_floor(self) -> None:
        rows = [
            {**candidate("s", index), "split": "test" if index < 150 else "train"}
            for index in range(200)
        ]
        first = _double_ids(rows, "s", 0.2, "seed")
        second = _double_ids(list(reversed(rows)), "s", 0.2, "seed")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 30)

    def test_default_split_has_150_locked_test_documents_per_source(self) -> None:
        identities = [Identity(f"doc-{index}", f"work-{index}", "source") for index in range(500)]
        manifest = build_split_manifest(
            identities,
            {"seed": "fixed", "train_fraction": 0.55, "validation_fraction": 0.15, "test_fraction": 0.30},
        )
        counts = {split: list(manifest["assignments"].values()).count(split) for split in ("train", "validation", "test")}
        self.assertEqual(counts, {"train": 275, "validation": 75, "test": 150})


class HumanImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = _packet(candidate("greek_phd", 1), CONFIG, "test")

    def test_judgments_must_cover_physical_lines_and_treat_blanks_safely(self) -> None:
        judgments = [
            {"start_line": 0, "end_line": 0, "label": "O", "is_running_prose": True},
            {"start_line": 1, "end_line": 1, "label": "O", "is_running_prose": False},
            {"start_line": 2, "end_line": 2, "label": "BIB", "is_running_prose": False},
            {"start_line": 3, "end_line": 3, "label": "O", "is_running_prose": False},
        ]
        expanded = _expand_judgments(judgments, self.packet["physical_lines"], label="test")
        self.assertEqual(expanded, [("O", True), ("O", False), ("BIB", False), ("O", False)])
        judgments[1]["label"] = "BIB"
        with self.assertRaisesRegex(ValueError, "blank"):
            _expand_judgments(judgments, self.packet["physical_lines"], label="test")

    def test_human_attestation_and_identity_are_exact(self) -> None:
        value = {
            "schema_version": CONFIG["annotation_schema_version"],
            "annotation_kind": "human_independent",
            "packet_id": self.packet["packet_id"],
            "document_id": self.packet["document_id"],
            "text_sha256": self.packet["text_sha256"],
            "annotator_id": "reviewer-a",
            "human_attestation": CONFIG["human_attestation"],
            "judgments": [
                {"start_line": 0, "end_line": 0, "label": "O", "is_running_prose": True},
                {"start_line": 1, "end_line": 1, "label": "O", "is_running_prose": False},
                {"start_line": 2, "end_line": 2, "label": "TOC", "is_running_prose": False},
                {"start_line": 3, "end_line": 3, "label": "O", "is_running_prose": False},
            ],
        }
        self.assertEqual(len(_validate_human_file(value, self.packet, CONFIG, adjudication=False)), 4)
        value["human_attestation"] = "generated by a model"
        with self.assertRaisesRegex(ValueError, "attestation"):
            _validate_human_file(value, self.packet, CONFIG, adjudication=False)

    def test_identity_hash_is_stable_and_namespace_separated(self) -> None:
        self.assertEqual(_hash_id("a", "b"), _hash_id("a", "b"))
        self.assertNotEqual(_hash_id("a", "b"), _hash_id("ab"))


if __name__ == "__main__":
    unittest.main()
