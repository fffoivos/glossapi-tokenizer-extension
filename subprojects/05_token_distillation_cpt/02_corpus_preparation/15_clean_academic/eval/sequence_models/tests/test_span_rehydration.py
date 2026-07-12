from __future__ import annotations

import json
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.span_rehydration import (  # noqa: E402
    Artifact,
    ManifestUnit,
    RehydrationError,
    SourceSpec,
    _load_layout,
    _scan_jsonl_source,
    assemble_kallipos_document,
    canonical_json_sha256,
    inspect_span_snapshot,
    load_batch_names,
    load_manifest,
    rehydrate_span_units,
    sha256_file,
    verify_rehydration_receipt,
)
from sequence_models.silver_reconstruct import (  # noqa: E402
    DeclaredSpan,
    SilverDraft,
    _build_span_drafts,
    _project_document_spans,
    _validate_unit_and_annotation,
)


class SpanRehydrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "SPAN_manifest.jsonl"
        self.batchpaths = self.root / "SPAN_batchpaths.json"
        self.layout = self.root / "layout.json"
        self.source_jsonl = self.root / "greek_phd.jsonl"
        self.source_receipt = self.root / "source-artifacts.json"
        self.rows = [
            {
                "unit_id": "U000",
                "doc_id": "doc-a",
                "source": "openarchives",
                "window": "tail",
                "win_lo": 0,
                "win_hi": 4,
            },
            {
                "unit_id": "U001",
                "doc_id": "doc-a",
                "source": "openarchives",
                "window": "body",
                "win_lo": 2,
                "win_hi": 6,
            },
            {
                "unit_id": "U002",
                "doc_id": "doc-b",
                "source": "openarchives",
                "window": "tail",
                "win_lo": 0,
                "win_hi": 2,
            },
        ]
        self.documents = [
            {"doc_id": "doc-a", "text": "zero\n\n two \nthree\n\nfive\nsix"},
            {"doc_id": "not-selected", "text": "an allowed extra source row"},
            {"doc_id": "doc-b", "text": "beta\nlast"},
        ]
        self._write_jsonl(self.manifest, self.rows)
        self.batchpaths.write_text(
            json.dumps(["/historical/batch_0000.json", "/historical/batch_0001.json"]),
            encoding="utf-8",
        )
        self._write_jsonl(self.source_jsonl, self.documents)
        self._write_layout()
        self._write_source_receipt(self.source_jsonl)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_layout(self) -> None:
        phases = []
        for name, start, rows in (
            ("base", "U000", self.rows[:2]),
            ("extension", "U002", self.rows[2:]),
        ):
            phases.append(
                {
                    "name": name,
                    "builder": "",
                    "start_unit_id": start,
                    "expected_unit_count": len(rows),
                    "expected_unit_ids_sha256": canonical_json_sha256(
                        [row["unit_id"] for row in rows]
                    ),
                    "expected_batch_count": 1,
                }
            )
        self.layout.write_text(
            json.dumps(
                {
                    "schema_version": "span-rehydration-layout-v1",
                    "manifest_sha256": sha256_file(self.manifest),
                    "batchpaths_sha256": sha256_file(self.batchpaths),
                    "batch_size": 2,
                    "builders": [],
                    "phases": phases,
                }
            ),
            encoding="utf-8",
        )

    def _write_source_receipt(self, artifact: Path) -> None:
        self.source_receipt.write_text(
            json.dumps(
                {
                    "schema_version": "span-source-artifacts-v1",
                    "sources": {
                        "openarchives": {
                            "repo_type": "dataset",
                            "repo_id": "example/greek-phd",
                            "revision": "a" * 40,
                            "format": "jsonl_documents",
                            "fields": {
                                "document_id": "doc_id",
                                "text_precedence": ["text", "document", "content"],
                            },
                            "artifacts": [
                                {
                                    "path": str(artifact),
                                    "repository_path": artifact.name,
                                    "sha256": sha256_file(artifact),
                                }
                            ],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def _run(
        self,
        name: str,
        *,
        expected: str | None = None,
        reference: Path | None = None,
    ) -> tuple[Path, Path, dict[str, object]]:
        output = self.root / name
        receipt_path = self.root / f"{name}.receipt.json"
        receipt = rehydrate_span_units(
            manifest_path=self.manifest,
            batchpaths_path=self.batchpaths,
            layout_path=self.layout,
            source_receipt_path=self.source_receipt,
            output_dir=output,
            receipt_path=receipt_path,
            expected_artifact_sha256=expected,
            reference_unit_dir=reference,
        )
        return output, receipt_path, receipt

    def test_rehydrates_overlapping_windows_and_historical_phase_batches(self) -> None:
        output, receipt_path, receipt = self._run("units-a")
        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            ["batch_0000.json", "batch_0001.json"],
        )
        first = json.loads((output / "batch_0000.json").read_text(encoding="utf-8"))
        self.assertEqual([unit["unit_id"] for unit in first], ["U000", "U001"])
        self.assertEqual(
            first[0]["text_numbered"],
            "L00000: zero\nL00002:  two \nL00003: three",
        )
        self.assertEqual(
            first[1]["text_numbered"],
            "L00002:  two \nL00003: three\nL00005: five",
        )
        self.assertEqual(
            tuple(first[0]), ("unit_id", "source", "window", "text_numbered")
        )
        self.assertNotIn("label", json.dumps(first))
        self.assertEqual(receipt["snapshot_equivalence_status"], "rehydrated_unverified_snapshot")
        self.assertFalse(receipt["snapshot_equivalence_verified"])
        self.assertTrue(receipt["research_fit_eligible"])
        self.assertFalse(receipt["promotion_eligible"])
        verified = verify_rehydration_receipt(
            output, receipt_path, self.manifest, self.batchpaths, self.layout
        )
        self.assertFalse(verified["snapshot_equivalence_verified"])
        self.assertTrue(verified["research_fit_eligible"])
        self.assertEqual(verified["research_evidence_scope"], "LLM_silver_comparison_only")

    def test_explicit_expected_artifact_sha_is_the_only_equivalence_gate(self) -> None:
        first, _, unverified = self._run("unverified")
        expected = str(unverified["snapshot_artifact_sha256"])
        _, receipt_path, verified = self._run("verified", expected=expected, reference=first)
        self.assertEqual(verified["snapshot_equivalence_status"], "verified_expected_artifact_sha256")
        self.assertTrue(verified["snapshot_equivalence_verified"])
        self.assertTrue(verified["research_fit_eligible"])
        self.assertFalse(verified["promotion_eligible"])
        comparison = verified["comparison_diagnostic"]
        self.assertTrue(comparison["artifact_sha256_equal"])
        self.assertEqual(comparison["mismatched_unit_count"], 0)
        inspection = inspect_span_snapshot(
            unit_dir=self.root / "verified",
            manifest_path=self.manifest,
            batchpaths_path=self.batchpaths,
            layout_path=self.layout,
        )
        self.assertEqual(inspection["snapshot_artifact_sha256"], expected)
        self.assertEqual(
            json.loads(receipt_path.read_text())["snapshot_artifact_sha256"], expected
        )

    def test_reference_match_without_explicit_sha_stays_unverified(self) -> None:
        reference, _, _ = self._run("reference")
        _, _, receipt = self._run("compared", reference=reference)
        self.assertTrue(receipt["comparison_diagnostic"]["artifact_sha256_equal"])
        self.assertEqual(receipt["snapshot_equivalence_status"], "rehydrated_unverified_snapshot")
        self.assertTrue(receipt["research_fit_eligible"])

    def test_expected_sha_mismatch_is_atomic(self) -> None:
        output = self.root / "bad-units"
        receipt = self.root / "bad.receipt.json"
        with self.assertRaisesRegex(RehydrationError, "artifact SHA-256 mismatch"):
            rehydrate_span_units(
                manifest_path=self.manifest,
                batchpaths_path=self.batchpaths,
                layout_path=self.layout,
                source_receipt_path=self.source_receipt,
                output_dir=output,
                receipt_path=receipt,
                expected_artifact_sha256="0" * 64,
            )
        self.assertFalse(output.exists())
        self.assertFalse(receipt.exists())

    def test_missing_requested_document_fails_closed(self) -> None:
        self._write_jsonl(self.source_jsonl, self.documents[:2])
        self._write_source_receipt(self.source_jsonl)
        with self.assertRaisesRegex(RehydrationError, "missing=.*doc-b"):
            self._run("missing")

    @unittest.skipUnless(shutil.which("zstd"), "zstd executable not installed")
    def test_jsonl_zstd_source_artifact(self) -> None:
        compressed = self.root / "greek_phd.jsonl.zst"
        subprocess.run(
            [shutil.which("zstd"), "-q", "-f", str(self.source_jsonl), "-o", str(compressed)],
            check=True,
        )
        self._write_source_receipt(compressed)
        output, _, receipt = self._run("zstd-units")
        self.assertTrue((output / "batch_0001.json").is_file())
        source = receipt["inputs"]["source_artifacts"][0]
        self.assertEqual(source["sha256"], sha256_file(compressed))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow not installed")
    def test_document_parquet_source_with_configurable_id_and_text_precedence(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.rows = [{**row, "source": "kallipos"} for row in self.rows]
        self._write_jsonl(self.manifest, self.rows)
        self._write_layout()

        parquet = self.root / "greek-phd.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_dataset": ["greek_phd", "greek_phd", "other"],
                    "source_doc_id": ["doc-a", "doc-b", "ignored"],
                    "preferred_text": [None, "beta\nlast", "ignored"],
                    "fallback_text": [
                        "zero\n\n two \nthree\n\nfive\nsix",
                        "must not win",
                        "ignored",
                    ],
                }
            ),
            parquet,
        )
        self.source_receipt.write_text(
            json.dumps(
                {
                    "schema_version": "span-source-artifacts-v1",
                    "sources": {
                        "kallipos": {
                            "repo_type": "dataset",
                            "repo_id": "example/nanochat",
                            "revision": "b" * 40,
                            "format": "parquet_documents",
                            "fields": {
                                "document_id": "source_doc_id",
                                "text_precedence": ["preferred_text", "fallback_text"],
                                "row_filter": {
                                    "column": "source_dataset",
                                    "equals": "greek_phd",
                                },
                            },
                            "artifacts": [
                                {
                                    "path": str(parquet),
                                    "repository_path": "data/greek-phd.parquet",
                                    "sha256": sha256_file(parquet),
                                }
                            ],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        output, _, receipt = self._run("parquet-units")
        first = json.loads((output / "batch_0000.json").read_text(encoding="utf-8"))
        self.assertEqual(first[0]["text_numbered"], "L00000: zero\nL00002:  two \nL00003: three")
        self.assertEqual(
            receipt["extraction"][0]["format"], "parquet_documents"
        )
        self.assertEqual(receipt["extraction"][0]["selected_document_count"], 2)

    def test_kallipos_sections_are_ordered_by_id_before_joining(self) -> None:
        text = assemble_kallipos_document(
            [(30, "third"), (10, "first"), (20, None)], "book-1"
        )
        self.assertEqual(text, "first\n\n\n\nthird")
        with self.assertRaisesRegex(RehydrationError, "duplicate section id"):
            assemble_kallipos_document([(1, "a"), (1, "b")], "book-1")

    def test_raw_jsonl_scan_enforces_forensic_text_digest_and_field_counts(self) -> None:
        source = self.root / "raw-mdc.jsonl"
        text = "first\nsecond"
        self._write_jsonl(
            source,
            [{"doc_id": "doc", "text": text, "document": "must not win"}],
        )
        artifact = Artifact(source, str(source), "contents/raw-mdc.jsonl", sha256_file(source))
        manifest = ManifestUnit("U", "doc", "greek_phd", "tail", 0, 2)
        base_provenance = {
            "acquisition_source_id": "mdc_raw_forensic",
            "selected_document_id_field_counts": {"doc_id": 1},
            "selected_text_field_counts": {"text": 1},
        }
        wrong = SourceSpec(
            "greek_phd",
            "mozilla-data-collective/test",
            "archive",
            "a" * 64,
            "jsonl_documents",
            (artifact,),
            {"document_id": "doc_id", "text_precedence": ["text", "document", "content"]},
            {**base_provenance, "selected_documents_text_sha256": "0" * 64},
        )
        with self.assertRaisesRegex(RehydrationError, "text digest differs"):
            _scan_jsonl_source(wrong, {"doc": [manifest]})
        expected_digest = canonical_json_sha256(
            [("doc", hashlib.sha256(text.encode("utf-8")).hexdigest())]
        )
        correct = SourceSpec(
            **{
                **wrong.__dict__,
                "provenance": {
                    **base_provenance,
                    "selected_documents_text_sha256": expected_digest,
                },
            }
        )
        _, extraction = _scan_jsonl_source(correct, {"doc": [manifest]})
        self.assertEqual(extraction["selected_documents_text_sha256"], expected_digest)
        self.assertEqual(extraction["selected_text_field_counts"], {"text": 1})

    def test_tracked_layout_preserves_240_historical_batches(self) -> None:
        manifest = EVAL_DIR / "units" / "SPAN_manifest.jsonl"
        batchpaths = EVAL_DIR / "units" / "SPAN_batchpaths.json"
        layout = EVAL_DIR / "sequence_models" / "span_rehydration_layout.json"
        rows = load_manifest(manifest)
        names = load_batch_names(batchpaths)
        batches, receipt = _load_layout(
            layout, manifest.resolve(), batchpaths.resolve(), rows, names
        )
        self.assertEqual(len(rows), 3340)
        self.assertEqual(len(batches), 240)
        self.assertEqual(
            [(phase["unit_count"], phase["batch_count"]) for phase in receipt["phases"]],
            [(2483, 178), (857, 62)],
        )

    def test_active_execution_config_is_silver_only_with_100_case_safety_audit(self) -> None:
        config = json.loads(
            (EVAL_DIR / "sequence_models" / "config.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("gold_contract", config)
        self.assertEqual(config["silver_contract"]["annotation_status"], "LLM_silver")
        self.assertFalse(config["silver_contract"]["production_eligible"])
        self.assertEqual(config["deployment_safety_audit"]["required_review_count"], 100)
        self.assertEqual(
            config["deployment_safety_audit"]["maximum_catastrophic_false_deletions"], 0
        )

    def test_annotation_shape_and_unit_coordinates_remain_fail_closed(self) -> None:
        meta = {"win_lo": 10, "win_hi": 15}
        unit = {"text_numbered": "L00010: prose\nL00012: citation\nL00014: citation"}
        annotation = {
            "has_bib": True,
            "spans": [{"start_line": 12, "end_line": 14}],
        }
        line_rows, declared = _validate_unit_and_annotation("U", meta, annotation, unit)
        self.assertEqual([index for index, _ in line_rows], [10, 12, 14])
        self.assertEqual(len(declared), 1)
        annotation["spans"][0]["start_line"] = "11"
        with self.assertRaisesRegex(ValueError, "ordered integer coordinates"):
            _validate_unit_and_annotation("U", meta, annotation, unit)

    def test_document_union_projection_matches_historical_present_line_semantics(self) -> None:
        draft = SilverDraft(
            upstream_doc_id="doc",
            source="greek_phd",
            n_physical_lines=30,
            lines={10: "a", 12: "b", 14: "c", 20: "d"},
            bib_lines=set(),
            declared_spans=[
                DeclaredSpan("exact", 0, 12, 14, 10, 15),
                # End escapes this unit but is present through another window.
                DeclaredSpan("overlap", 0, 10, 20, 10, 15),
                # No sampled present coordinate: historical comparison yields no positives.
                DeclaredSpan("zero", 0, 25, 27, 0, 5),
            ],
            sampled_units=["exact", "overlap", "zero"],
            annotation_units=["exact", "overlap", "zero"],
            missing_annotation_units=[],
        )
        counts, anomalies = _project_document_spans(draft)
        self.assertEqual(draft.bib_lines, {10, 12, 14, 20})
        self.assertEqual(counts["exact_nonempty_span_count"], 1)
        self.assertEqual(counts["adjusted_nonempty_span_count"], 1)
        self.assertEqual(counts["zero_effective_span_count"], 1)
        self.assertEqual([row["outcome"] for row in anomalies], [
            "projected_nonempty", "zero_effective"
        ])
        self.assertEqual(anomalies[0]["effective_last_present_line"], 20)
        self.assertEqual(anomalies[1]["effective_present_line_count"], 0)

    def test_missing_annotation_unit_text_is_retained_as_historical_negative(self) -> None:
        manifest = {
            "annotated": {
                "source": "greek_phd", "doc_id": "doc-a", "win_lo": 0, "win_hi": 2,
            },
            "missing": {
                "source": "greek_phd", "doc_id": "doc-b", "win_lo": 4, "win_hi": 6,
            },
        }
        units = {
            "annotated": {"text_numbered": "L00000: prose\nL00001: citation"},
            "missing": {"text_numbered": "L00004: retained\nL00005: negative"},
        }
        annotations = {
            "annotated": {
                "has_bib": True,
                "spans": [{"start_line": 1, "end_line": 1}],
            }
        }

        drafts = _build_span_drafts(manifest, annotations, units)

        missing = drafts[("greek_phd", "doc-b")]
        self.assertEqual(missing.lines, {4: "retained", 5: "negative"})
        self.assertEqual(missing.declared_spans, [])
        self.assertEqual(missing.sampled_units, ["missing"])
        self.assertEqual(missing.annotation_units, [])
        self.assertEqual(missing.missing_annotation_units, ["missing"])
        counts, anomalies = _project_document_spans(missing)
        self.assertEqual(counts, {})
        self.assertEqual(anomalies, [])
        self.assertEqual(missing.bib_lines, set())


if __name__ == "__main__":
    unittest.main()
