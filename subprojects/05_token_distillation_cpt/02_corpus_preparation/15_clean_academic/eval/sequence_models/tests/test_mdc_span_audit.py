from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import tarfile
import unittest


EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.mdc_span_audit import audit  # noqa: E402
from sequence_models.mdc_safe_extract import safe_extract, sha256_file  # noqa: E402


class MdcSpanAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "archive.tar.gz"
        payload = (
            json.dumps({"doc_id": "doc-1", "document": "\nalpha\nomega"}) + "\n"
        ).encode()
        with tarfile.open(self.archive, "w:gz") as archive:
            member = tarfile.TarInfo("phd-theses-corpus/contents/part.jsonl")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        extracted = self.root / "extracted"
        extraction_manifest = self.root / "extraction.manifest.json"
        extraction_receipt_path = self.root / "extraction.receipt.json"
        extraction_receipt = safe_extract(
            self.archive, extracted, extraction_manifest, extraction_receipt_path
        )
        self.contents = extracted / "phd-theses-corpus" / "contents"
        self.quarantine_receipt = self.root / "quarantine.receipt.json"
        observed = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.quarantine_receipt.write_text(
            json.dumps(
                {
                    "schema_version": "mdc_quarantined_object_receipt_v2",
                    "status": "quarantined_publisher_checksum_mismatch",
                    "archive": {
                        "path": str(self.archive.resolve()),
                        "bytes": self.archive.stat().st_size,
                        "observed_sha256": observed,
                        "publisher_declared_sha256": "0" * 64,
                        "gzip_and_tar_integrity": "passed",
                    },
                    "safe_extraction": {
                        "receipt_path": str(extraction_receipt_path.resolve()),
                        "receipt_sha256": sha256_file(extraction_receipt_path),
                        "status": "passed_fresh_archive_tree_matches",
                    },
                    "extracted": {
                        "path": str(extracted.resolve()),
                        "file_count": extraction_receipt["extraction"]["file_count"],
                        "sha256_manifest_path": str(extraction_manifest.resolve()),
                        "sha256_manifest_sha256": sha256_file(extraction_manifest),
                    },
                }
            ),
            encoding="utf-8",
        )
        self.manifest = self.root / "SPAN_manifest.jsonl"
        self.annotations = self.root / "annotations.json"
        self.output = self.root / "audit.json"
        self.annotations.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "unit_id": "S00000",
                            "has_bib": True,
                            "spans": [{"start_line": 1, "end_line": 2}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            archive=str(self.archive),
            quarantine_receipt=str(self.quarantine_receipt),
            publisher_declared_sha256="0" * 64,
            contents_root=str(self.contents),
            shard_glob="*.jsonl",
            manifest=str(self.manifest),
            annotations=str(self.annotations),
            output=str(self.output),
        )

    def _write_manifest(self, win_hi: int) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "unit_id": "S00000",
                    "doc_id": "doc-1",
                    "source": "greek_phd",
                    "window": "tail",
                    "win_lo": 0,
                    "win_hi": win_hi,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_exact_raw_coordinates_pass_without_upgrading_snapshot_equivalence(self) -> None:
        self._write_manifest(3)
        result = audit(self._args())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["source_coordinate_integrity"]["status"], "passed")
        self.assertEqual(
            result["historical_document_union_projection"]["status"],
            "exact_on_present_document_union",
        )
        self.assertEqual(
            result["snapshot_equivalence_to_historical_span_inputs"], "unverified"
        )
        self.assertFalse(result["archive"]["publisher_checksum_matches"])
        self.assertEqual(result["counts"]["target_documents"], 1)
        self.assertEqual(result["counts"]["positive_spans_checked"], 1)
        self.assertEqual(result["counts"]["tail_length_mismatches"], 0)
        self.assertEqual(
            result["inputs"]["selection_contract"][
                "selected_document_id_field_counts"
            ],
            {"doc_id": 1},
        )
        self.assertEqual(
            result["inputs"]["selection_contract"]["selected_text_field_counts"],
            {"document": 1},
        )
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(audit(self._args()), result)

    def test_audit_receipt_is_immutable_on_differing_rerun(self) -> None:
        self._write_manifest(3)
        audit(self._args())
        self.annotations.write_text(
            json.dumps(
                {
                    "annotations": [
                        {"unit_id": "S00000", "has_bib": False, "spans": []}
                    ]
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "immutable audit output differs"):
            audit(self._args())

    def test_tail_length_drift_fails(self) -> None:
        self._write_manifest(4)
        result = audit(self._args())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["source_coordinate_integrity"]["status"], "failed")
        self.assertEqual(result["counts"]["tail_length_mismatches"], 1)
        self.assertEqual(result["counts"]["window_overflows"], 1)

    def test_document_union_projection_separates_adjusted_and_zero_effective(self) -> None:
        self.contents.joinpath("part.jsonl").write_text(
            json.dumps({"doc_id": "doc-1", "text": "\nalpha\nbeta\n\nomega"}) + "\n",
            encoding="utf-8",
        )
        rows = [
            {
                "unit_id": "S00000",
                "doc_id": "doc-1",
                "source": "greek_phd",
                "window": "tail",
                "win_lo": 0,
                "win_hi": 5,
            },
            {
                "unit_id": "S00001",
                "doc_id": "doc-1",
                "source": "greek_phd",
                "window": "body",
                "win_lo": 1,
                "win_hi": 4,
            },
            {
                "unit_id": "S00002",
                "doc_id": "doc-1",
                "source": "greek_phd",
                "window": "body",
                "win_lo": 0,
                "win_hi": 1,
            },
        ]
        self.manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.annotations.write_text(
            json.dumps(
                {
                    "annotations": [
                        {"unit_id": "S00000", "has_bib": False, "spans": []},
                        {
                            "unit_id": "S00001",
                            "has_bib": True,
                            "spans": [{"start_line": 1, "end_line": 4}],
                        },
                        {
                            "unit_id": "S00002",
                            "has_bib": True,
                            "spans": [{"start_line": 10, "end_line": 12}],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        result = audit(self._args())
        self.assertEqual(result["status"], "comparison_only_with_silver_anomalies")
        self.assertEqual(result["source_coordinate_integrity"]["status"], "passed")
        counts = result["historical_document_union_projection"]["counts"]
        self.assertEqual(counts["adjusted_nonempty_spans"], 1)
        self.assertEqual(counts["zero_effective_spans"], 1)
        self.assertEqual(counts["unit_window_escape_spans"], 2)
        adjusted = result["projection_details"]["adjusted_nonempty_spans"][0]
        self.assertEqual(adjusted["effective_present_line_count"], 3)
        self.assertEqual(adjusted["effective_last_present_line"], 4)
        zero = result["projection_details"]["zero_effective_spans"][0]
        self.assertEqual(zero["effective_present_line_count"], 0)


if __name__ == "__main__":
    unittest.main()
