from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "phase04_stage_contract_inventory",
        HERE / "clariden" / "stage_contract.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = load_module()


def receipt(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_run_identity_pins_eligibility_and_license_policy(tmp_path: Path) -> None:
    sources = tmp_path / "sources.json"
    cleaning = tmp_path / "cleaning.json"
    eligibility = tmp_path / "eligibility.json"
    license_adjudication = tmp_path / "license.json"
    for path in (sources, cleaning, eligibility, license_adjudication):
        path.write_text(f'{{"name":"{path.stem}"}}\n', encoding="utf-8")
    arguments = argparse.Namespace(
        run_root=tmp_path / "run",
        run_id="fixture",
        code_commit="f" * 40,
        sources=sources,
        cleaning_policy=cleaning,
        eligibility_policy=eligibility,
        source_license_adjudication=license_adjudication,
        tokenizer_sha256="a" * 64,
    )
    CONTRACT.cmd_init_run(arguments)
    manifest = json.loads(
        (arguments.run_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["eligibility_policy_sha256"] == hashlib.sha256(
        eligibility.read_bytes()
    ).hexdigest()
    assert manifest["source_license_adjudication_sha256"] == hashlib.sha256(
        license_adjudication.read_bytes()
    ).hexdigest()

    drifted = tmp_path / "eligibility-drifted.json"
    drifted.write_text('{"name":"drifted"}\n', encoding="utf-8")
    arguments.eligibility_policy = drifted
    with pytest.raises(ValueError, match="immutable run identity drift"):
        CONTRACT.cmd_init_run(arguments)


def normalization_manifest(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "canonical"
    shard = root / "source-a" / "shard-00000" / "part-00000.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"PAR1-test")
    manifest = {
        "schema_version": "full_cpt_normalization_manifest_v1",
        "output": str(root.resolve()),
        "sources": [{"source_id": "source-a", "shards": [receipt(shard)]}],
    }
    path = tmp_path / "normalization_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, root, manifest


def test_exact_normalization_tree_accepts_declared_shards(tmp_path: Path) -> None:
    path, root, manifest = normalization_manifest(tmp_path)
    validated, aggregate, actual_root = CONTRACT.validate_normalization_parquet_tree(
        manifest,
        path=path,
        expected_root=root,
    )
    assert len(validated) == 1
    assert len(aggregate) == 64
    assert actual_root == root.resolve()


def test_exact_normalization_tree_rejects_unreceipted_parquet(tmp_path: Path) -> None:
    path, root, manifest = normalization_manifest(tmp_path)
    rogue = root / "source-a" / "unexpected.parquet"
    rogue.write_bytes(b"PAR1-rogue")
    with pytest.raises(ValueError, match="unexpected"):
        CONTRACT.validate_normalization_parquet_tree(
            manifest,
            path=path,
            expected_root=root,
        )


def test_exact_normalization_tree_rejects_wrong_consumer_root(tmp_path: Path) -> None:
    path, _, manifest = normalization_manifest(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="output root drift"):
        CONTRACT.validate_normalization_parquet_tree(
            manifest,
            path=path,
            expected_root=other,
        )


def test_publication_receipt_validates_local_not_remote_paths(tmp_path: Path) -> None:
    public_root = tmp_path / "redistribution" / "data"
    shard = public_root / "demo" / "part.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"PAR1-public")
    card = tmp_path / "README.md"
    card.write_text("dataset card", encoding="utf-8")
    manifest = {
        "schema_version": "full_cpt_publication_receipt_v1",
        "redistribution_root": str(public_root),
        "local_inventory": [
            {
                **receipt(shard),
                "path": "demo/part.parquet",
                "remote_path": "data/demo/part.parquet",
            },
            {**receipt(card), "remote_path": "README.md"},
        ],
        "remote_inventory": [
            {
                "path": "data/demo/part.parquet",
                "bytes": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    entries = CONTRACT.manifest_inventory_entries(manifest)
    validated, _ = CONTRACT.validate_inventory_entries(entries)
    assert {row["path"] for row in validated} == {
        str(shard.resolve()),
        str(card.resolve()),
    }


def test_stage_parameter_binding_rejects_resume_drift(tmp_path: Path) -> None:
    stage = tmp_path / "20-lineage"
    stage.mkdir()
    CONTRACT.cmd_bind_parameter(
        SimpleNamespace(
            stage_dir=stage,
            name="lineage_debug_exports",
            value="0",
        )
    )
    inputs = json.loads((stage / "stage_inputs.json").read_text(encoding="utf-8"))
    assert inputs["parameters"] == {"lineage_debug_exports": "0"}
    with pytest.raises(ValueError, match="resume parameter drift"):
        CONTRACT.cmd_bind_parameter(
            SimpleNamespace(
                stage_dir=stage,
                name="lineage_debug_exports",
                value="1",
            )
        )
