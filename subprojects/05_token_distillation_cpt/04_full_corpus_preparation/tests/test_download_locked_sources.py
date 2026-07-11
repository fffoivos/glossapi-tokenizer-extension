from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DOWNLOAD = load_module(
    "phase04_download_locked_sources",
    HERE / "scripts" / "download_locked_sources.py",
)
RESOLVE = load_module(
    "phase04_resolve_sources",
    HERE / "scripts" / "resolve_sources.py",
)


class IncompleteRead(Exception):
    pass


def test_resolver_rejects_redacted_lfs_content_identifier() -> None:
    with pytest.raises(ValueError, match="authenticated metadata"):
        RESOLVE.exact_lfs_sha256("*" * 64, repo_id="owner/data", path="part.parquet")
    digest = "a" * 64
    assert (
        RESOLVE.exact_lfs_sha256(digest, repo_id="owner/data", path="part.parquet")
        == digest
    )


def test_snapshot_download_resumes_transient_incomplete_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_download(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["repo_id"] == "owner/data"
        if calls < 3:
            raise IncompleteRead("truncated response")
        return "/staged/snapshot"

    monkeypatch.setattr(DOWNLOAD.time, "sleep", sleeps.append)
    result = DOWNLOAD.snapshot_download_with_retries(
        fake_download,
        attempts=4,
        backoff_seconds=2,
        source_id="fixture",
        repo_id="owner/data",
    )
    assert result == "/staged/snapshot"
    assert calls == 3
    assert sleeps == [2, 4]


def test_snapshot_download_does_not_retry_definite_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    class Unauthorized(Exception):
        response = type("Response", (), {"status_code": 401})()

    def fake_download(**_kwargs):
        nonlocal calls
        calls += 1
        raise Unauthorized("bad token")

    monkeypatch.setattr(DOWNLOAD.time, "sleep", sleeps.append)
    with pytest.raises(Unauthorized, match="bad token"):
        DOWNLOAD.snapshot_download_with_retries(
            fake_download,
            attempts=8,
            backoff_seconds=15,
            source_id="fixture",
        )
    assert calls == 1
    assert sleeps == []


def test_snapshot_download_retry_arguments_are_bounded() -> None:
    with pytest.raises(ValueError, match="attempts"):
        DOWNLOAD.snapshot_download_with_retries(
            lambda: None,
            attempts=0,
            backoff_seconds=1,
            source_id="fixture",
        )
    with pytest.raises(ValueError, match="backoff"):
        DOWNLOAD.snapshot_download_with_retries(
            lambda: None,
            attempts=1,
            backoff_seconds=-1,
            source_id="fixture",
        )


def test_existing_only_rebinds_verified_local_payload_without_token(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "hf"
    payload = destination / "fixture" / "revision-1" / "data" / "part.txt"
    payload.parent.mkdir(parents=True)
    payload.write_text("receipt-bound payload", encoding="utf-8")
    lock = tmp_path / "sources.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_sources_lock_v1",
                "sources": [
                    {
                        "source_id": "fixture",
                        "repo_id": "owner/fixture",
                        "repo_type": "dataset",
                        "revision": "revision-1",
                        "selected_bytes": payload.stat().st_size,
                        "selected_files": [
                            {
                                "path": "data/part.txt",
                                "size": payload.stat().st_size,
                                "blob_id": DOWNLOAD.git_blob_id(payload),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "download.json"
    environment = {key: value for key, value in os.environ.items() if key != "HF_TOKEN"}
    completed = subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "download_locked_sources.py"),
            "--lock",
            str(lock),
            "--destination",
            str(destination),
            "--manifest",
            str(manifest),
            "--existing-only",
        ],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["download_policy"]["existing_only"] is True
    assert value["sources"][0]["git_blob_ids_verified"] == 1
