from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_agent1_v5_release_quality.py"
SPEC = importlib.util.spec_from_file_location("audit_agent1_v5_release_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_preserved_field_hash_distinguishes_null_and_empty() -> None:
    null_hash = hashlib.sha256()
    empty_hash = hashlib.sha256()
    assert (
        AUDIT.update_stream_hash(
            null_hash,
            [
                {
                    "source_dataset": "repo",
                    "source_doc_id": "1",
                    "title": None,
                    "author": None,
                    "source_metadata_json": "{}",
                }
            ],
        )
        == 1
    )
    assert (
        AUDIT.update_stream_hash(
            empty_hash,
            [
                {
                    "source_dataset": "repo",
                    "source_doc_id": "1",
                    "title": "",
                    "author": None,
                    "source_metadata_json": "{}",
                }
            ],
        )
        == 1
    )
    assert null_hash.hexdigest() != empty_hash.hexdigest()


def test_preserved_field_hash_binds_document_identity() -> None:
    left = hashlib.sha256()
    right = hashlib.sha256()
    common = {
        "source_dataset": "repo",
        "title": None,
        "author": None,
        "source_metadata_json": "{}",
    }
    AUDIT.update_stream_hash(left, [{**common, "source_doc_id": "doc-1"}])
    AUDIT.update_stream_hash(right, [{**common, "source_doc_id": "doc-2"}])
    assert left.hexdigest() != right.hexdigest()


def test_resolve_binding_checks_sha_and_containment(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = root / "payload.json"
    payload.write_bytes(b"{}\n")
    binding = {
        "path": "payload.json",
        "bytes": payload.stat().st_size,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    assert AUDIT.resolve_binding(binding, root, containment_root=root) == payload
    payload.write_bytes(b"[]\n")
    with pytest.raises(ValueError, match="SHA-256 drift"):
        AUDIT.resolve_binding(binding, root, containment_root=root)


def test_resolve_binding_rejects_root_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}\n")
    binding = {
        "path": str(outside),
        "bytes": outside.stat().st_size,
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="escapes"):
        AUDIT.resolve_binding(binding, root, containment_root=root)


def test_percentiles_are_deterministic() -> None:
    assert AUDIT.percentiles([float(value) for value in range(1, 11)]) == {
        "p10": 2.0,
        "p50": 5.0,
        "p90": 9.0,
        "p99": 10.0,
    }
    assert AUDIT.percentiles([]) == {"p10": None, "p50": None, "p90": None, "p99": None}


def test_artifact_patterns_cover_release_postconditions() -> None:
    assert AUDIT.GENERATED_IMAGE_RE.search("deadbeef" * 4 + "_3_img.webp")
    assert AUDIT.GENERATED_IMAGE_RE.search("deadbeef" * 4 + "_3_img.avif")
    assert AUDIT.HTML_RE.search("<table><tr><td>x</td></tr></table>")
    assert AUDIT.MOJIBAKE_RE.search("ÎšÎ±Î»Î·Î¼Î­ÏÎ±")
    assert AUDIT.CONTROL_RE.search("bad\x00text")
