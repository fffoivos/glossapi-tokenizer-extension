from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = HERE / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MDC = load_script("acquire_mdc_sources")
MERGE = load_script("merge_acquisition_receipts")
RESOLVE = load_script("resolve_sources")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_checksum_normalization_and_missing_key() -> None:
    value = "a" * 64
    assert MDC.canonical_checksum(value) == value
    assert MDC.canonical_checksum(f"sha256:{value}") == value
    with pytest.raises(ValueError, match="valid SHA-256"):
        MDC.canonical_checksum("bad")
    with pytest.raises(ValueError, match="API_KEY"):
        MDC.MdcClient("https://example.invalid/api", "")


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    payload = b"forbidden"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe archive member"):
        MDC.safe_extract_tar(archive, tmp_path / "out", maximum_bytes=1000)
    assert not (tmp_path / "escape.txt").exists()


def test_mdc_acquisition_binds_archive_and_never_persists_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_archive = tmp_path / "fixture.tar.gz"
    fixture_parquet = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({"text": ["ένα καθαρό κείμενο"], "id": ["doc-1"]}),
        fixture_parquet,
    )
    with tarfile.open(fixture_archive, "w:gz") as bundle:
        bundle.add(fixture_parquet, arcname="dataset/data.parquet")
    archive_hash = MDC.sha256_file(fixture_archive)
    source = {
        "source_id": "sample_mdc",
        "repo_id": "glossAPI/sample",
        "revision": "1" * 40,
        "acquisition_kind": "mozilla_data_collective",
        "mdc_dataset_id": "dataset-1",
        "mdc_slug": "sample-v1",
        "mdc_name": "Sample",
        "mdc_format": "PARQUET",
        "mdc_expected_filename": "sample.tar.gz",
        "mdc_expected_bytes": fixture_archive.stat().st_size,
        "mdc_expected_sha256": archive_hash,
        "include_globs": ["**/*.parquet"],
        "role": "additive_candidate",
        "text_columns": ["text"],
        "id_columns": ["id"],
    }

    class FakeClient:
        timeout = 5

        def json_request(self, path: str, *, method: str = "GET") -> dict:
            if method == "GET":
                return {
                    "id": "dataset-1",
                    "slug": "sample-v1",
                    "name": "Sample",
                    "format": "PARQUET",
                    "sizeBytes": fixture_archive.stat().st_size,
                    "checksum": f"sha256:{archive_hash}",
                }
            return {
                "filename": "sample.tar.gz",
                "sizeBytes": str(fixture_archive.stat().st_size),
                "checksum": f"sha256:{archive_hash}",
                "contentType": "application/gzip",
                "downloadUrl": "https://storage.example.invalid/secret?signature=never-store",
                "downloadToken": "never-store-token",
            }

    def fake_download(url: str, output: Path, **_: object) -> None:
        assert "signature=" in url
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(fixture_archive.read_bytes())

    monkeypatch.setattr(MDC, "download_storage", fake_download)

    class DriftedMetadataClient(FakeClient):
        def json_request(self, path: str, *, method: str = "GET") -> dict:
            value = super().json_request(path, method=method)
            if method == "GET":
                value["checksum"] = "0" * 64
            return value

    with pytest.raises(ValueError, match="metadata checksum differs from registry"):
        MDC.acquire_source(
            source,
            client=DriftedMetadataClient(),
            destination=tmp_path / "metadata-drift-destination",
            extraction_multiplier=20,
        )

    class DriftedDownloadClient(FakeClient):
        def json_request(self, path: str, *, method: str = "GET") -> dict:
            value = super().json_request(path, method=method)
            if method == "POST":
                value["checksum"] = "0" * 64
            return value

    with pytest.raises(ValueError, match="archive checksum differs from registry"):
        MDC.acquire_source(
            source,
            client=DriftedDownloadClient(),
            destination=tmp_path / "download-drift-destination",
            extraction_multiplier=20,
        )
    receipt = MDC.acquire_source(
        source,
        client=FakeClient(),
        destination=tmp_path / "destination",
        extraction_multiplier=20,
    )
    serialized = json.dumps(receipt, sort_keys=True)
    assert "never-store" not in serialized
    assert receipt["archive"]["sha256"] == archive_hash
    assert receipt["archive"]["registry_sha256"] == archive_hash
    assert receipt["archive"]["metadata_sha256"] == archive_hash
    assert receipt["source_config_sha256"] == MDC.canonical_object_sha256(source)
    assert receipt["selected_file_count"] == 1
    assert receipt["payload_validation"]["status"] == "passed"
    assert receipt["payload_validation"]["total_rows"] == 1
    assert receipt["files"][0]["expected_hash"] == MDC.sha256_file(
        Path(receipt["files"][0]["local_path"])
    )
    resumed = MDC.acquire_source(
        source,
        client=FakeClient(),
        destination=tmp_path / "destination",
        extraction_multiplier=20,
    )
    assert resumed == receipt

    source_receipt = (
        tmp_path
        / "destination"
        / source["source_id"]
        / source["revision"]
        / "source_receipt.json"
    )
    drifted = json.loads(source_receipt.read_text(encoding="utf-8"))
    drifted["payload_validation"]["total_rows"] = 2
    write_json(source_receipt, drifted)
    with pytest.raises(ValueError, match="payload_validation receipt drift"):
        MDC.acquire_source(
            source,
            client=FakeClient(),
            destination=tmp_path / "destination",
            extraction_multiplier=20,
        )
    write_json(source_receipt, receipt)

    changed_source = dict(source)
    changed_source["include_globs"] = ["*.jsonl"]
    with pytest.raises(ValueError, match="source_config_sha256"):
        MDC.acquire_source(
            changed_source,
            client=FakeClient(),
            destination=tmp_path / "destination",
            extraction_multiplier=20,
        )

    Path(receipt["archive"]["local_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="archive drift"):
        MDC.acquire_source(
            source,
            client=FakeClient(),
            destination=tmp_path / "destination",
            extraction_multiplier=20,
        )


def test_mdc_payload_validation_fails_closed_by_format_and_schema(
    tmp_path: Path,
) -> None:
    source = {
        "source_id": "sample",
        "mdc_format": "PARQUET",
        "text_columns": ["text"],
        "id_columns": ["id"],
    }
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"PAR1-fixture-PAR1")
    with pytest.raises(ValueError, match="not readable Parquet"):
        MDC.validate_payload([corrupt], source)

    missing_text = tmp_path / "missing-text.parquet"
    pq.write_table(pa.table({"id": ["doc-1"], "body": ["x"]}), missing_text)
    with pytest.raises(ValueError, match="candidate text columns"):
        MDC.validate_payload([missing_text], source)

    missing_id = tmp_path / "missing-id.parquet"
    pq.write_table(pa.table({"text": ["x"]}), missing_id)
    with pytest.raises(ValueError, match="candidate id columns"):
        MDC.validate_payload([missing_id], source)

    empty = tmp_path / "empty.parquet"
    pq.write_table(
        pa.table(
            {
                "text": pa.array([], type=pa.string()),
                "id": pa.array([], type=pa.string()),
            }
        ),
        empty,
    )
    with pytest.raises(ValueError, match="zero rows"):
        MDC.validate_payload([empty], source)

    unsupported = dict(source, mdc_format="JSONL")
    with pytest.raises(ValueError, match="unsupported MDC payload format"):
        MDC.validate_payload([missing_id], unsupported)


def test_hf_resolver_excludes_external_routes() -> None:
    config = {
        "base": {"repo_id": "a/base"},
        "apertus_overlap_overlay": {"repo_id": "a/overlay"},
        "tokenizer": {"repo_id": "a/tokenizer"},
        "sources": [
            {"source_id": "hf", "repo_id": "a/hf"},
            {
                "source_id": "mdc",
                "repo_id": "a/mdc-pointer",
                "acquisition_kind": "mozilla_data_collective",
            },
        ],
    }
    assert [row["source_id"] for row in RESOLVE.entries(config)] == [
        "nanochat_base",
        "apertus_overlap_overlay",
        "modern_greek_148k_tokenizer",
        "hf",
    ]


def test_merge_requires_every_configured_source(tmp_path: Path) -> None:
    source_file = tmp_path / "payload.parquet"
    source_file.write_bytes(b"data")
    sources = tmp_path / "sources.json"
    write_json(
        sources,
        {
            "base": {"repo_id": "a/base", "revision": "1" * 40},
            "apertus_overlap_overlay": {"repo_id": "a/overlay", "revision": "2" * 40},
            "tokenizer": {"repo_id": "a/tokenizer", "revision": "3" * 40},
            "sources": [
                {
                    "source_id": "external",
                    "repo_id": "a/external",
                    "revision": "4" * 40,
                    "acquisition_kind": "mozilla_data_collective",
                    "mdc_dataset_id": "external-1",
                }
            ],
        },
    )
    config_hash = MERGE.sha256_file(sources)
    file_row = {"local_path": str(source_file), "size": source_file.stat().st_size}
    hf = tmp_path / "hf.json"
    mdc = tmp_path / "mdc.json"
    write_json(
        hf,
        {
            "schema_version": "full_cpt_acquisition_receipt_v1",
            "status": "passed",
            "code_commit": "a" * 40,
            "sources_config_sha256": config_hash,
            "sources": [
                {"source_id": "nanochat_base", "files": [file_row]}
            ],
        },
    )
    write_json(
        mdc,
        {
            "schema_version": "full_cpt_mdc_acquisition_receipt_v1",
            "status": "passed",
            "code_commit": "a" * 40,
            "sources_config_sha256": config_hash,
            "sources": [],
        },
    )
    with pytest.raises(ValueError, match="identities differ from the registry"):
        MERGE.build_receipt(
            sources_path=sources,
            hf_path=hf,
            mdc_path=mdc,
            destination_root=tmp_path,
        )


def test_merge_requires_exact_hf_and_mdc_routes(tmp_path: Path) -> None:
    sources = tmp_path / "sources.json"
    external = {
        "source_id": "external",
        "repo_id": "a/external",
        "revision": "4" * 40,
        "acquisition_kind": "mozilla_data_collective",
        "mdc_dataset_id": "external-1",
        "mdc_format": "PARQUET",
        "mdc_expected_sha256": "f" * 64,
        "text_columns": ["text"],
        "id_columns": ["id"],
    }
    config = {
        "base": {"repo_id": "a/base", "revision": "1" * 40},
        "apertus_overlap_overlay": {"repo_id": "a/overlay", "revision": "2" * 40},
        "tokenizer": {"repo_id": "a/tokenizer", "revision": "3" * 40},
        "sources": [external],
    }
    write_json(sources, config)
    config_hash = MERGE.sha256_file(sources)

    def file_row(name: str, *, parquet: bool = False) -> dict:
        path = tmp_path / name
        if parquet:
            pq.write_table(pa.table({"text": ["κείμενο"], "id": ["1"]}), path)
        else:
            path.write_bytes(name.encode())
        stat = path.stat()
        return {
            "path": name,
            "local_path": str(path),
            "size": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "hash_kind": "sha256",
            "expected_hash": MDC.sha256_file(path),
        }

    hf = tmp_path / "hf.json"
    mdc = tmp_path / "mdc.json"
    write_json(
        hf,
        {
            "schema_version": "full_cpt_acquisition_receipt_v1",
            "status": "passed",
            "code_commit": "a" * 40,
            "sources_config_sha256": config_hash,
            "sources": [
                {
                    "source_id": source_id,
                    "repo_id": row["repo_id"],
                    "revision": row["revision"],
                    "files": [file_row(f"{source_id}.parquet")],
                }
                for source_id, row in (
                    ("nanochat_base", config["base"]),
                    ("apertus_overlap_overlay", config["apertus_overlap_overlay"]),
                    ("modern_greek_148k_tokenizer", config["tokenizer"]),
                )
            ],
        },
    )
    external_file = file_row("external.parquet", parquet=True)
    external_validation = MDC.validate_payload(
        [Path(external_file["local_path"])], external
    )
    write_json(
        mdc,
        {
            "schema_version": "full_cpt_mdc_acquisition_receipt_v1",
            "status": "passed",
            "code_commit": "a" * 40,
            "sources_config_sha256": config_hash,
            "sources": [
                {
                    "source_id": "external",
                    "repo_id": external["repo_id"],
                    "revision": external["revision"],
                    "mdc_dataset_id": external["mdc_dataset_id"],
                    "source_config_sha256": MERGE.canonical_object_sha256(external),
                    "archive": {
                        "sha256": external["mdc_expected_sha256"],
                        "registry_sha256": external["mdc_expected_sha256"],
                        "metadata_sha256": external["mdc_expected_sha256"],
                    },
                    "payload_validation": external_validation,
                    "files": [external_file],
                }
            ],
        },
    )
    receipt = MERGE.build_receipt(
        sources_path=sources,
        hf_path=hf,
        mdc_path=mdc,
        destination_root=tmp_path,
    )
    assert [row["source_id"] for row in receipt["sources"]] == [
        "apertus_overlap_overlay",
        "external",
        "modern_greek_148k_tokenizer",
        "nanochat_base",
    ]

    mdc_value = json.loads(mdc.read_text(encoding="utf-8"))
    del mdc_value["sources"][0]["payload_validation"]
    write_json(mdc, mdc_value)
    with pytest.raises(ValueError, match="payload validation receipt is absent"):
        MERGE.build_receipt(
            sources_path=sources,
            hf_path=hf,
            mdc_path=mdc,
            destination_root=tmp_path,
        )
