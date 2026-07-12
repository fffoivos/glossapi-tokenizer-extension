from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


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
    parquet_bytes = b"PAR1-fixture-PAR1"
    with tarfile.open(fixture_archive, "w:gz") as bundle:
        member = tarfile.TarInfo("dataset/data.parquet")
        member.size = len(parquet_bytes)
        bundle.addfile(member, io.BytesIO(parquet_bytes))
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
    receipt = MDC.acquire_source(
        source,
        client=FakeClient(),
        destination=tmp_path / "destination",
        extraction_multiplier=20,
    )
    serialized = json.dumps(receipt, sort_keys=True)
    assert "never-store" not in serialized
    assert receipt["archive"]["sha256"] == archive_hash
    assert receipt["source_config_sha256"] == MDC.canonical_object_sha256(source)
    assert receipt["selected_file_count"] == 1
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
            "base": {"source_id": "nanochat_base"},
            "sources": [
                {
                    "source_id": "external",
                    "acquisition_kind": "mozilla_data_collective",
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
    with pytest.raises(ValueError, match="missing configured sources"):
        MERGE.build_receipt(
            sources_path=sources,
            hf_path=hf,
            mdc_path=mdc,
            destination_root=tmp_path,
        )
