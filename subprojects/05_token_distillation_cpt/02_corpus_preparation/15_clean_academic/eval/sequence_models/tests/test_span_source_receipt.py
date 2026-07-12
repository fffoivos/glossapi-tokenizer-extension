from __future__ import annotations

import argparse
import hashlib
import io
import importlib.util
import json
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.build_span_source_receipt import (  # noqa: E402
    _config_sources,
    build_span_source_receipt,
)
from sequence_models.mdc_span_audit import audit as audit_mdc_span  # noqa: E402
from sequence_models.mdc_safe_extract import safe_extract  # noqa: E402
from sequence_models.span_rehydration import (  # noqa: E402
    RehydrationError,
    load_source_specs,
    sha256_file,
)


RUNTIME_AVAILABLE = (
    importlib.util.find_spec("pyarrow") is not None
    and importlib.util.find_spec("zstandard") is not None
)


def test_phase04_config_expands_all_acquisition_registry_sections() -> None:
    config = {
        "base": {"repo_id": "owner/base"},
        "apertus_overlap_overlay": {"repo_id": "owner/overlap"},
        "tokenizer": {"repo_id": "owner/tokenizer"},
        "sources": [{"source_id": "candidate", "repo_id": "owner/candidate"}],
    }
    rows = _config_sources(config)
    assert set(rows) == {
        "nanochat_base",
        "apertus_overlap_overlay",
        "modern_greek_148k_tokenizer",
        "candidate",
    }


@unittest.skipUnless(RUNTIME_AVAILABLE, "pyarrow/zstandard not installed")
class SpanSourceReceiptBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import zstandard

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.destination = self.root / "staged"
        self.sources_path = self.root / "sources.json"
        self.lock_path = self.root / "sources.lock.json"
        self.acquisition_path = self.root / "acquisition.receipt.json"
        self.manifest_path = self.root / "SPAN_manifest.jsonl"
        self.annotations_path = self.root / "annotations.json"
        self.output_path = self.root / "span-source-artifacts.json"
        revisions = {
            "nanochat_base": "1" * 40,
            "openarchives_current": "2" * 40,
            "kallipos_sections": "3" * 40,
            "greek_phd_v2": "4" * 40,
        }
        configs = {
            "nanochat_base": {
                "repo_id": "owner/nanochat",
                "repo_type": "dataset",
                "revision": revisions["nanochat_base"],
                "include_globs": ["data/*.parquet"],
                "text_columns": ["text"],
                "id_columns": ["source_doc_id"],
            },
            "openarchives_current": {
                "source_id": "openarchives_current",
                "repo_id": "owner/openarchives",
                "repo_type": "dataset",
                "revision": revisions["openarchives_current"],
                "include_globs": ["data/openarchives/**/*.jsonl.zst"],
                "text_columns": ["text"],
                "id_columns": ["doc_id"],
            },
            "kallipos_sections": {
                "source_id": "kallipos_sections",
                "repo_id": "owner/kallipos",
                "repo_type": "dataset",
                "revision": revisions["kallipos_sections"],
                "include_globs": ["Dataset_Kallipos.parquet"],
                "text_columns": ["section"],
                "id_columns": ["id"],
                "work_id_columns": ["filename"],
            },
            "greek_phd_v2": {
                "source_id": "greek_phd_v2",
                "repo_id": "owner/greek-v2",
                "repo_type": "dataset",
                "revision": revisions["greek_phd_v2"],
                "include_globs": ["Greek PhD Theses Corpus v2.0.parquet"],
                "text_columns": ["extracted_md"],
                "alternate_text_columns": ["extracted_text_plain"],
                "id_columns": ["handle_url", "doi", "url"],
            },
        }
        source_config = {
            "schema_version": "full_cpt_sources_v1",
            "base": configs["nanochat_base"],
            "sources": [
                configs["openarchives_current"],
                configs["kallipos_sections"],
                configs["greek_phd_v2"],
            ],
        }
        self._write_json(self.sources_path, source_config)

        files: dict[str, list[Path]] = {source_id: [] for source_id in revisions}
        nano_root = self.destination / "nanochat_base" / revisions["nanochat_base"]
        for index in range(2):
            path = nano_root / "data" / f"greek_phd.part-{index:05d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table(
                    {
                        "source_dataset": ["greek_phd"],
                        "source_doc_id": [("a" if index == 0 else "b") * 64],
                        "text": ["Greek PhD text"],
                    }
                ),
                path,
            )
            files["nanochat_base"].append(path)
        unrelated = nano_root / "data" / "unrelated.parquet"
        pq.write_table(
            pa.table({"source_doc_id": ["x"], "text": ["ignore"]}), unrelated
        )
        files["nanochat_base"].append(unrelated)
        nanochat_kallipos = nano_root / "data" / "Apothetirio_Kallipos.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_dataset": ["Apothetirio_Kallipos"],
                    "source_doc_id": ["paper_A_1"],
                    "text": ["Processed Kallipos"],
                }
            ),
            nanochat_kallipos,
        )
        files["nanochat_base"].append(nanochat_kallipos)

        open_root = (
            self.destination
            / "openarchives_current"
            / revisions["openarchives_current"]
        )
        open_path = open_root / "data" / "openarchives" / "shard_001" / "part.jsonl.zst"
        open_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps({"doc_id": "open-1", "text": "Open text"}) + "\n"
        ).encode()
        open_path.write_bytes(zstandard.ZstdCompressor().compress(payload))
        files["openarchives_current"].append(open_path)

        kallipos_root = (
            self.destination / "kallipos_sections" / revisions["kallipos_sections"]
        )
        kallipos_path = kallipos_root / "Dataset_Kallipos.parquet"
        kallipos_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({"filename": ["paper_A_1"], "id": [1], "section": ["Section"]}),
            kallipos_path,
        )
        files["kallipos_sections"].append(kallipos_path)

        v2_root = self.destination / "greek_phd_v2" / revisions["greek_phd_v2"]
        v2_path = v2_root / "Greek PhD Theses Corpus v2.0.parquet"
        v2_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "handle_url": ["https://example/1"],
                    "extracted_md": ["newer text"],
                    "extracted_text_plain": ["plain"],
                }
            ),
            v2_path,
        )
        files["greek_phd_v2"].append(v2_path)

        lock_sources = []
        receipt_sources = []
        for source_id, source_files in files.items():
            config = configs[source_id]
            local_root = self.destination / source_id / revisions[source_id]
            locked_files = []
            receipt_files = []
            for path in source_files:
                relative = str(path.relative_to(local_root))
                stat = path.stat()
                digest = sha256_file(path)
                locked_files.append(
                    {
                        "path": relative,
                        "size": stat.st_size,
                        "lfs_sha256": digest,
                        "lfs_size": stat.st_size,
                        "blob_id": "f" * 40,
                    }
                )
                receipt_files.append(
                    {
                        "path": relative,
                        "local_path": str(path.resolve()),
                        "size": stat.st_size,
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "mtime_ns": stat.st_mtime_ns,
                        "ctime_ns": stat.st_ctime_ns,
                        "hash_kind": "lfs_sha256",
                        "expected_hash": digest,
                    }
                )
            lock_sources.append(
                {
                    "source_id": source_id,
                    "repo_id": config["repo_id"],
                    "repo_type": config["repo_type"],
                    "revision": config["revision"],
                    "selected_files": locked_files,
                    "selected_file_count": len(locked_files),
                    "selected_bytes": sum(row["size"] for row in locked_files),
                }
            )
            receipt_sources.append(
                {
                    "source_id": source_id,
                    "repo_id": config["repo_id"],
                    "repo_type": config["repo_type"],
                    "revision": config["revision"],
                    "local_root": str(local_root.resolve()),
                    "selected_file_count": len(receipt_files),
                    "selected_bytes": sum(row["size"] for row in receipt_files),
                    "files": receipt_files,
                }
            )
        self._write_json(
            self.lock_path,
            {
                "schema_version": "full_cpt_sources_lock_v1",
                "sources_config_sha256": sha256_file(self.sources_path),
                "sources": lock_sources,
            },
        )
        self._write_json(
            self.acquisition_path,
            {
                "schema_version": "full_cpt_acquisition_receipt_v1",
                "status": "passed",
                "code_commit": "c" * 40,
                "source_lock": str(self.lock_path.resolve()),
                "source_lock_sha256": sha256_file(self.lock_path),
                "sources_config_sha256": sha256_file(self.sources_path),
                "sources": receipt_sources,
            },
        )
        self.manifest_path.write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {
                        "unit_id": "S0",
                        "doc_id": "a" * 64,
                        "source": "greek_phd",
                        "window": "tail",
                        "win_lo": 0,
                        "win_hi": 1,
                    },
                    {
                        "unit_id": "S1",
                        "doc_id": "open-1",
                        "source": "openarchives",
                        "window": "tail",
                        "win_lo": 0,
                        "win_hi": 1,
                    },
                    {
                        "unit_id": "S2",
                        "doc_id": "paper_A_1",
                        "source": "kallipos",
                        "window": "tail",
                        "win_lo": 0,
                        "win_hi": 1,
                    },
                )
            ),
            encoding="utf-8",
        )
        self._write_json(
            self.annotations_path,
            {
                "annotations": [
                    {"unit_id": "S0", "has_bib": False, "spans": []},
                    {"unit_id": "S1", "has_bib": False, "spans": []},
                    {"unit_id": "S2", "has_bib": False, "spans": []},
                ]
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def _args(self, **overrides: object) -> argparse.Namespace:
        values = {
            "acquisition_receipt": str(self.acquisition_path),
            "source_lock": str(self.lock_path),
            "sources_config": str(self.sources_path),
            "manifest": str(self.manifest_path),
            "annotations": str(self.annotations_path),
            "greek_phd_route": "nanochat_base",
            "kallipos_route": "kallipos_sections",
            "greek_phd_document_id_column": None,
            "greek_phd_text_column": None,
            "allow_unverified_greek_phd_id_domain": False,
            "mdc_quarantine_receipt": None,
            "mdc_span_audit_receipt": None,
            "mdc_expected_observed_sha256": None,
            "allow_quarantined_mdc_comparison_only": False,
            "output": str(self.output_path),
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _write_mdc_fixture(self) -> tuple[Path, Path, str]:
        import zstandard

        self._write_json(
            self.annotations_path,
            {
                "annotations": [
                    {
                        "unit_id": "S0",
                        "has_bib": True,
                        "spans": [{"start_line": 0, "end_line": 1}],
                    },
                    {"unit_id": "S1", "has_bib": False, "spans": []},
                    {"unit_id": "S2", "has_bib": False, "spans": []},
                ]
            },
        )
        root = self.root / "mdc-observed"
        archive = root / "greek-phd.tar.gz"
        archive.parent.mkdir(parents=True)
        payload = (
            json.dumps({"doc_id": "a" * 64, "text": "Greek PhD text"}) + "\n"
        ).encode()
        compressed = zstandard.ZstdCompressor().compress(payload)
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("phd-theses-corpus/contents/part.jsonl.zst")
            member.size = len(compressed)
            tar.addfile(member, io.BytesIO(compressed))
        observed = sha256_file(archive)
        publisher = "0" * 64
        extracted_root = root / "extracted"
        extracted_manifest = root / "extraction.manifest.json"
        extraction_receipt_path = root / "extraction.receipt.json"
        extraction_receipt = safe_extract(
            archive,
            extracted_root,
            extracted_manifest,
            extraction_receipt_path,
        )
        contents = extracted_root / "phd-theses-corpus" / "contents"
        quarantine = root / "quarantine.receipt.json"
        self._write_json(
            quarantine,
            {
                "schema_version": "mdc_quarantined_object_receipt_v2",
                "status": "quarantined_publisher_checksum_mismatch",
                "archive": {
                    "path": str(archive.resolve()),
                    "bytes": archive.stat().st_size,
                    "observed_sha256": observed,
                    "publisher_declared_sha256": publisher,
                    "gzip_and_tar_integrity": "passed",
                },
                "safe_extraction": {
                    "receipt_path": str(extraction_receipt_path.resolve()),
                    "receipt_sha256": sha256_file(extraction_receipt_path),
                    "status": "passed_fresh_archive_tree_matches",
                },
                "extracted": {
                    "path": str(extracted_root.resolve()),
                    "file_count": extraction_receipt["extraction"]["file_count"],
                    "sha256_manifest_path": str(extracted_manifest.resolve()),
                    "sha256_manifest_sha256": sha256_file(extracted_manifest),
                },
            },
        )
        audit_path = root / "span-coordinate-audit.json"
        audit_mdc_span(
            argparse.Namespace(
                archive=str(archive),
                quarantine_receipt=str(quarantine),
                publisher_declared_sha256=publisher,
                contents_root=str(contents),
                shard_glob="*.jsonl.zst",
                manifest=str(self.manifest_path),
                annotations=str(self.annotations_path),
                output=str(audit_path),
            )
        )
        return quarantine, audit_path, observed

    def test_builds_path_filtered_receipt_from_acquisition_and_lock(self) -> None:
        receipt = build_span_source_receipt(self._args())
        self.assertEqual(
            set(receipt["sources"]), {"greek_phd", "openarchives", "kallipos"}
        )
        greek = receipt["sources"]["greek_phd"]
        self.assertEqual(greek["format"], "parquet_documents")
        self.assertEqual(greek["fields"]["document_id"], "source_doc_id")
        self.assertEqual(len(greek["artifacts"]), 2)
        self.assertTrue(
            all(
                "greek_phd.part-" in row["repository_path"]
                for row in greek["artifacts"]
            )
        )
        self.assertEqual(
            receipt["sources"]["openarchives"]["format"], "jsonl_documents"
        )
        self.assertEqual(receipt["sources"]["kallipos"]["format"], "parquet_sections")
        self.assertEqual(
            receipt["snapshot_equivalence_status"], "rehydrated_unverified_snapshot"
        )
        specs, artifacts = load_source_specs(
            self.output_path, {"greek_phd", "openarchives", "kallipos"}
        )
        self.assertEqual(
            specs["greek_phd"].provenance["acquisition_source_id"], "nanochat_base"
        )
        self.assertEqual(len(artifacts), 4)

    def test_v2_route_requires_explicit_unverified_identifier_override(self) -> None:
        with self.assertRaisesRegex(RehydrationError, "historical hash doc_id"):
            build_span_source_receipt(self._args(greek_phd_route="greek_phd_v2"))
        with self.assertRaisesRegex(
            RehydrationError, "identifier alignment is unverified"
        ):
            build_span_source_receipt(
                self._args(
                    greek_phd_route="greek_phd_v2",
                    greek_phd_document_id_column="handle_url",
                )
            )

    def test_kallipos_nanochat_document_route_is_explicit_and_path_filtered(
        self,
    ) -> None:
        output = self.root / "span-source-artifacts-nanochat-kallipos.json"
        receipt = build_span_source_receipt(
            self._args(kallipos_route="nanochat_base", output=str(output))
        )
        kallipos = receipt["sources"]["kallipos"]
        self.assertEqual(kallipos["format"], "parquet_documents")
        self.assertEqual(kallipos["fields"]["document_id"], "source_doc_id")
        self.assertEqual(
            [row["repository_path"] for row in kallipos["artifacts"]],
            ["data/Apothetirio_Kallipos.parquet"],
        )

    def test_raw_mdc_route_is_quarantine_and_audit_bound(self) -> None:
        quarantine, audit_path, observed = self._write_mdc_fixture()
        output = self.root / "span-source-artifacts-mdc.json"
        receipt = build_span_source_receipt(
            self._args(
                output=str(output),
                greek_phd_route="mdc_raw_forensic",
                mdc_quarantine_receipt=str(quarantine),
                mdc_span_audit_receipt=str(audit_path),
                mdc_expected_observed_sha256=observed,
                allow_quarantined_mdc_comparison_only=True,
            )
        )
        greek = receipt["sources"]["greek_phd"]
        self.assertEqual(greek["repo_type"], "archive")
        self.assertEqual(greek["revision"], observed)
        self.assertEqual(greek["acquisition_source_id"], "mdc_raw_forensic")
        self.assertEqual(len(greek["artifacts"]), 1)
        self.assertEqual(greek["selected_document_id_field_counts"], {"doc_id": 1})
        self.assertEqual(greek["selected_text_field_counts"], {"text": 1})
        self.assertRegex(greek["selected_documents_text_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            receipt["derivation"]["schema_reports"]["greek_phd"]["audit_status"],
            "comparison_only_with_silver_anomalies",
        )
        self.assertEqual(
            receipt["derivation"]["schema_reports"]["greek_phd"][
                "projection_counts"
            ]["adjusted_nonempty_spans"],
            1,
        )
        specs, artifacts = load_source_specs(
            output, {"greek_phd", "openarchives", "kallipos"}
        )
        self.assertEqual(specs["greek_phd"].revision, observed)
        self.assertEqual(len(artifacts), 3)

    def test_raw_mdc_hand_authored_derivation_bypass_is_rejected(self) -> None:
        quarantine, audit_path, observed = self._write_mdc_fixture()
        output = self.root / "span-source-artifacts-mdc-bypass.json"
        build_span_source_receipt(
            self._args(
                output=str(output),
                greek_phd_route="mdc_raw_forensic",
                mdc_quarantine_receipt=str(quarantine),
                mdc_span_audit_receipt=str(audit_path),
                mdc_expected_observed_sha256=observed,
                allow_quarantined_mdc_comparison_only=True,
            )
        )
        receipt = json.loads(output.read_text(encoding="utf-8"))
        del receipt["derivation"]
        bypass = self.root / "hand-authored-mdc.json"
        self._write_json(bypass, receipt)
        with self.assertRaisesRegex(RehydrationError, "hand-authored receipt rejected"):
            load_source_specs(bypass, {"greek_phd", "openarchives", "kallipos"})
        receipt["sources"]["greek_phd"]["repo_type"] = "dataset"
        receipt["sources"]["greek_phd"]["acquisition_source_id"] = "nanochat_base"
        disguised = self.root / "disguised-jsonl-mdc.json"
        self._write_json(disguised, receipt)
        with self.assertRaisesRegex(RehydrationError, "exact mdc_raw_forensic provenance"):
            load_source_specs(disguised, {"greek_phd", "openarchives", "kallipos"})

    def test_raw_mdc_selected_text_digest_tamper_is_rejected_at_consumption(self) -> None:
        quarantine, audit_path, observed = self._write_mdc_fixture()
        output = self.root / "span-source-artifacts-mdc-digest.json"
        build_span_source_receipt(
            self._args(
                output=str(output),
                greek_phd_route="mdc_raw_forensic",
                mdc_quarantine_receipt=str(quarantine),
                mdc_span_audit_receipt=str(audit_path),
                mdc_expected_observed_sha256=observed,
                allow_quarantined_mdc_comparison_only=True,
            )
        )
        receipt = json.loads(output.read_text(encoding="utf-8"))
        receipt["sources"]["greek_phd"]["selected_documents_text_sha256"] = "f" * 64
        tampered = self.root / "tampered-mdc-digest.json"
        self._write_json(tampered, receipt)
        with self.assertRaisesRegex(RehydrationError, "selected document text digest differs"):
            load_source_specs(tampered, {"greek_phd", "openarchives", "kallipos"})

    def test_raw_mdc_artifact_substitution_is_rejected_at_consumption(self) -> None:
        quarantine, audit_path, observed = self._write_mdc_fixture()
        output = self.root / "span-source-artifacts-mdc-artifact.json"
        build_span_source_receipt(
            self._args(
                output=str(output),
                greek_phd_route="mdc_raw_forensic",
                mdc_quarantine_receipt=str(quarantine),
                mdc_span_audit_receipt=str(audit_path),
                mdc_expected_observed_sha256=observed,
                allow_quarantined_mdc_comparison_only=True,
            )
        )
        receipt = json.loads(output.read_text(encoding="utf-8"))
        artifact = receipt["sources"]["greek_phd"]["artifacts"][0]
        substituted = self.root / "substituted" / "part.jsonl.zst"
        substituted.parent.mkdir()
        substituted.write_bytes(Path(artifact["path"]).read_bytes())
        artifact["path"] = str(substituted)
        artifact["bytes"] = substituted.stat().st_size
        artifact["sha256"] = sha256_file(substituted)
        tampered = self.root / "tampered-mdc-artifact.json"
        self._write_json(tampered, receipt)
        with self.assertRaisesRegex(
            RehydrationError, "source artifacts differ from the safe extraction manifest"
        ):
            load_source_specs(tampered, {"greek_phd", "openarchives", "kallipos"})

    def test_raw_mdc_fake_manifest_derivation_is_rejected_at_consumption(self) -> None:
        quarantine, audit_path, observed = self._write_mdc_fixture()
        output = self.root / "span-source-artifacts-mdc-manifest.json"
        build_span_source_receipt(
            self._args(
                output=str(output),
                greek_phd_route="mdc_raw_forensic",
                mdc_quarantine_receipt=str(quarantine),
                mdc_span_audit_receipt=str(audit_path),
                mdc_expected_observed_sha256=observed,
                allow_quarantined_mdc_comparison_only=True,
            )
        )
        receipt = json.loads(output.read_text(encoding="utf-8"))
        external = receipt["derivation"]["external_forensic_inputs"]["greek_phd"]
        original = Path(external["extracted_sha256_manifest"])
        substituted = self.root / "copied-extraction.manifest.json"
        substituted.write_bytes(original.read_bytes())
        external["extracted_sha256_manifest"] = str(substituted)
        external["extracted_sha256_manifest_sha256"] = sha256_file(substituted)
        tampered = self.root / "tampered-mdc-manifest.json"
        self._write_json(tampered, receipt)
        with self.assertRaisesRegex(
            RehydrationError,
            "manifest path/hash differs from the safe extraction receipt",
        ):
            load_source_specs(tampered, {"greek_phd", "openarchives", "kallipos"})

    def test_raw_mdc_route_rejects_audit_binding_drift(self) -> None:
        quarantine, audit_path, observed = self._write_mdc_fixture()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["inputs"]["manifest"]["sha256"] = "f" * 64
        self._write_json(audit_path, audit)
        with self.assertRaisesRegex(RehydrationError, "manifest binding differs"):
            build_span_source_receipt(
                self._args(
                    output=str(self.root / "bad-mdc.json"),
                    greek_phd_route="mdc_raw_forensic",
                    mdc_quarantine_receipt=str(quarantine),
                    mdc_span_audit_receipt=str(audit_path),
                    mdc_expected_observed_sha256=observed,
                    allow_quarantined_mdc_comparison_only=True,
                )
            )

    def test_raw_mdc_route_requires_acknowledgement_and_rejects_shard_drift(self) -> None:
        quarantine, audit_path, observed = self._write_mdc_fixture()
        with self.assertRaisesRegex(RehydrationError, "requires --allow-quarantined"):
            build_span_source_receipt(
                self._args(
                    output=str(self.root / "unacknowledged-mdc.json"),
                    greek_phd_route="mdc_raw_forensic",
                    mdc_quarantine_receipt=str(quarantine),
                    mdc_span_audit_receipt=str(audit_path),
                    mdc_expected_observed_sha256=observed,
                )
            )
        shard = next((quarantine.parent / "extracted").rglob("*.jsonl.zst"))
        shard.write_bytes(shard.read_bytes() + b"drift")
        with self.assertRaisesRegex(RehydrationError, "extracted tree differs"):
            build_span_source_receipt(
                self._args(
                    output=str(self.root / "drifted-mdc.json"),
                    greek_phd_route="mdc_raw_forensic",
                    mdc_quarantine_receipt=str(quarantine),
                    mdc_span_audit_receipt=str(audit_path),
                    mdc_expected_observed_sha256=observed,
                    allow_quarantined_mdc_comparison_only=True,
                )
            )

    def test_rejects_acquisition_lock_hash_drift(self) -> None:
        acquisition = json.loads(self.acquisition_path.read_text(encoding="utf-8"))
        acquisition["source_lock_sha256"] = hashlib.sha256(b"wrong").hexdigest()
        self._write_json(self.acquisition_path, acquisition)
        with self.assertRaisesRegex(
            RehydrationError, "not bound to the supplied source lock"
        ):
            build_span_source_receipt(self._args())


if __name__ == "__main__":
    unittest.main()
