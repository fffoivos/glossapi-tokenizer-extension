from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import duckdb


PHASE = Path(__file__).resolve().parents[1]
SCRIPTS = PHASE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from publish_release import (  # noqa: E402
    _build_hardlink_upload_tree,
    require_new_empty_remote,
    validated_token_waterfall,
    validated_public_root,
    verify_local_public_inventory,
    verify_remote_inventory,
)
from materialize_release import build_dataset_card, validate_inputs  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    release = tmp_path / "release"
    root = release / "redistribution" / "data"
    parquet = root / "demo" / "part.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"PAR1 deterministic test bytes")
    manifest: dict[str, object] = {"redistribution_root": "redistribution/data"}
    validation: dict[str, object] = {
        "publication_inventory": {
            "root": str(root.resolve()),
            "bytes": parquet.stat().st_size,
            "rows": 7,
            "files": [
                {
                    "path": "demo/part.parquet",
                    "remote_path": "data/demo/part.parquet",
                    "sha256": _sha(parquet),
                    "bytes": parquet.stat().st_size,
                    "rows": 7,
                }
            ],
        }
    }
    return release, parquet, manifest, validation


def test_publication_local_inventory_is_exact_and_rehashed(tmp_path: Path) -> None:
    release, parquet, manifest, validation = _public_fixture(tmp_path)
    root = validated_public_root(release, manifest, validation)
    verified = verify_local_public_inventory(root, validation)
    assert verified == [
        {
            "path": "demo/part.parquet",
            "remote_path": "data/demo/part.parquet",
            "sha256": _sha(parquet),
            "bytes": parquet.stat().st_size,
            "rows": 7,
        }
    ]
    upload_root = _build_hardlink_upload_tree(
        public_root=root,
        inventory=verified,
        staging_root=tmp_path / "upload-staging",
    )
    staged = upload_root / "data" / "demo" / "part.parquet"
    assert staged.stat().st_ino == parquet.stat().st_ino

    (root / "README.md").write_text("must not leak", encoding="utf-8")
    with pytest.raises(ValueError, match="non-Parquet"):
        verify_local_public_inventory(root, validation)


def test_publication_rejects_symlinked_public_tree(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    release = tmp_path / "release"
    (release / "redistribution").mkdir(parents=True)
    (release / "redistribution" / "data").symlink_to(real, target_is_directory=True)
    manifest = {"redistribution_root": "redistribution/data"}
    validation = {"publication_inventory": {"root": str(real.resolve()), "files": []}}
    with pytest.raises(ValueError, match="symlink"):
        validated_public_root(release, manifest, validation)


def test_dataset_card_is_explicitly_a_redistributable_delta_with_attribution() -> None:
    revision = "a" * 40
    card = build_dataset_card(
        public_sources=[
            {
                "source_id": "eellak_articles",
                "repo_id": "glossAPI/eellak-articles",
                "revision": revision,
                "rows": 123,
            }
        ],
        license_payload={
            "sources": [
                {
                    "source_id": "eellak_articles",
                    "repo_id": "glossAPI/eellak-articles",
                    "revision": revision,
                    "declared_license": "cc-by-sa-4.0",
                    "redistribution": {
                        "eligible": True,
                        "conditions": ["attribution_required", "sharealike_required"],
                    },
                }
            ]
        },
        redistribution_rows=123,
        structural_applied=False,
    )
    assert "license: other" in card
    assert "not** the full private training corpus" in card
    assert "eellak_articles" in card
    assert "cc-by-sa-4.0" in card
    assert "ShareAlike" in card
    assert "structural gate completed as a no-op" in card


class _FakeApi:
    def __init__(self, rows: list[object]):
        self.rows = rows

    def list_repo_tree(self, **_: object) -> list[object]:
        return self.rows


def test_remote_inventory_rejects_stale_files_and_verifies_lfs_sha(tmp_path: Path) -> None:
    expected = [
        {
            "remote_path": "data/demo/part.parquet",
            "sha256": "a" * 64,
            "bytes": 123,
        }
    ]
    valid = SimpleNamespace(
        path="data/demo/part.parquet",
        size=123,
        lfs=SimpleNamespace(sha256="a" * 64),
    )
    result = verify_remote_inventory(
        _FakeApi([valid]),
        repo_id="org/repo",
        commit_sha="abc1234",
        token="not-recorded",
        expected=expected,
    )
    assert result[0]["verification"] == "lfs_sha256"

    stale = SimpleNamespace(path="old.parquet", size=1, lfs=SimpleNamespace(sha256="b" * 64))
    with pytest.raises(RuntimeError, match="remote inventory mismatch"):
        verify_remote_inventory(
            _FakeApi([valid, stale]),
            repo_id="org/repo",
            commit_sha="abc1234",
            token="not-recorded",
            expected=expected,
        )


def test_publisher_rechecks_current_token_waterfall_hash(tmp_path: Path) -> None:
    waterfall = tmp_path / "token-waterfall.json"
    waterfall.write_text('{"reconciled":true}\n', encoding="utf-8")
    manifest = {
        "token_waterfall": str(waterfall),
        "token_waterfall_sha256": _sha(waterfall),
    }
    assert validated_token_waterfall(manifest) == waterfall
    waterfall.write_text('{"reconciled":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="token waterfall drifted"):
        validated_token_waterfall(manifest)


def test_partial_publication_is_refused_with_actionable_cleanup() -> None:
    require_new_empty_remote(_FakeApi([]), repo_id="org/repo")
    system = SimpleNamespace(path=".gitattributes", size=1, lfs=None)
    require_new_empty_remote(_FakeApi([system]), repo_id="org/repo")

    partial = SimpleNamespace(
        path="data/demo/part-0.parquet",
        size=123,
        lfs=SimpleNamespace(sha256="a" * 64),
    )
    with pytest.raises(RuntimeError, match="delete/recreate"):
        require_new_empty_remote(_FakeApi([partial]), repo_id="org/repo")


def test_release_schemas_and_clariden_gates_are_fail_closed() -> None:
    for name in (
        "full_cpt_release_manifest.schema.json",
        "full_cpt_release_validation.schema.json",
        "full_cpt_publication_receipt.schema.json",
    ):
        schema = json.loads((PHASE / "schemas" / name).read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")

    materialize_stage = (PHASE / "clariden" / "90_materialize_validate.sbatch").read_text(encoding="utf-8")
    assert 'value.get("status") != "passed"' in materialize_stage
    assert 'value.get("failed_checks") != []' in materialize_stage
    assert "validation_attempts" in materialize_stage
    assert 'mv "$VALIDATION_ATTEMPT" "$PHASE04_STAGE_DIR/validation.json"' in materialize_stage
    publish_stage = (PHASE / "clariden" / "99_publish_hf.sbatch").read_text(encoding="utf-8")
    assert "--gate-mode manual" in publish_stage
    assert "--gate-mode auto" not in publish_stage
    assert "--remote-mode new-empty" in publish_stage


def test_materialization_rejects_decision_text_hash_drift() -> None:
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE corpus AS SELECT
              'source'::VARCHAR AS source_dataset,
              'upstream-id'::VARCHAR AS source_doc_id,
              'uid'::VARCHAR AS stable_uid,
              'κείμενο'::VARCHAR AS text,
              sha256('κείμενο')::VARCHAR AS cleaned_text_sha256,
              true AS eligible_for_training,
              true AS eligible_for_redistribution
            """
        )
        connection.execute(
            """
            CREATE TABLE decisions AS SELECT
              'source'::VARCHAR AS source_dataset,
              'uid'::VARCHAR AS source_doc_id,
              'uid'::VARCHAR AS stable_uid,
              'keep'::VARCHAR AS decision,
              repeat('0', 64)::VARCHAR AS input_text_sha256
            """
        )
        with pytest.raises(ValueError, match="decision_text_hash_mismatch"):
            validate_inputs(connection)
    finally:
        connection.close()
