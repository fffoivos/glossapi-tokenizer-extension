from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bridge_common import (  # noqa: E402
    document_key,
    file_tree_receipt,
    heldout_hash,
    iter_index_lengths,
    selected_by_threshold,
    write_index,
)
from finalize_bridge import (  # noqa: E402
    audit_unique_training_documents,
    capacity_report,
    compute_blend,
)


def shard(
    index: int,
    pool: str,
    source: str,
    tokens: int,
    source_weight: str | None = None,
) -> dict:
    return {
        "kind": "training",
        "task_index": index,
        "task_id": f"task-{index}",
        "pool": pool,
        "source_name": source,
        "source_weight_within_pool": source_weight,
        "counts": {"tokens": tokens, "documents": max(1, tokens // 10)},
        "outputs": {"bin": {"path": f"/tmp/task-{index}.bin"}},
    }


def test_index_roundtrip_exact_counts(tmp_path: Path) -> None:
    index = tmp_path / "sample.idx"
    assert write_index(index, [3, 9, 1]) == (3, 4, 13)
    assert iter_index_lengths(index) == (3, 4, 13)


def test_heldout_selection_is_domain_separated_and_deterministic() -> None:
    first = heldout_hash(7, "hplt", "phase04", "doc-1")
    assert first == heldout_hash(7, "hplt", "phase04", "doc-1")
    assert first != heldout_hash(7, "openarchives", "phase04", "doc-1")
    observed = [
        selected_by_threshold(
            seed=7,
            set_name="hplt",
            source_name="phase04",
            doc_id=f"doc-{index}",
            numerator=1,
            denominator=4,
        )
        for index in range(2000)
    ]
    assert 400 < sum(observed) < 600


def test_document_identity_scopes_old_greek_composite_fields() -> None:
    first = document_key(
        "old_greek",
        "shard-a.parquet",
        1,
        {"source_dataset": "dataset-a", "source_doc_id": "17"},
        identity_scope="global",
    )
    assert first == document_key(
        "old_greek",
        "shard-b.parquet",
        999,
        {"source_dataset": "dataset-a", "source_doc_id": "17"},
        identity_scope="global",
    )
    assert first != document_key(
        "old_greek",
        "shard-a.parquet",
        1,
        {"source_dataset": "dataset-b", "source_doc_id": "17"},
        identity_scope="global",
    )
    assert document_key(
        "fineweb", "shard-a.parquet", 1, {"id": "17"}, identity_scope="file"
    ) != document_key(
        "fineweb", "shard-b.parquet", 1, {"id": "17"}, identity_scope="file"
    )


def test_physical_blend_preserves_exact_logical_79_20_1() -> None:
    shards = [
        shard(0, "new_greek", "phase04_release", 600),
        shard(1, "new_greek", "phase04_release", 400),
        shard(2, "foreign_replay", "source_a", 90, "0.6"),
        shard(3, "foreign_replay", "source_a", 10, "0.6"),
        shard(4, "foreign_replay", "source_b", 50, "0.4"),
        shard(5, "old_greek_replay", "old", 100),
    ]
    blend = compute_blend(
        shards,
        {"new_greek": 79, "foreign_replay": 20, "old_greek_replay": 1},
        100,
    )
    totals: dict[str, Decimal] = {}
    source_totals: dict[str, Decimal] = {}
    for row in blend:
        totals[row["pool"]] = totals.get(row["pool"], Decimal(0)) + Decimal(
            row["weight_exact"]
        )
        source_totals[row["source_name"]] = source_totals.get(
            row["source_name"], Decimal(0)
        ) + Decimal(row["weight_exact"])
    assert totals == {
        "foreign_replay": Decimal("0.20"),
        "new_greek": Decimal("0.79"),
        "old_greek_replay": Decimal("0.01"),
    }
    assert source_totals["source_a"] == Decimal("0.12")
    assert source_totals["source_b"] == Decimal("0.08")
    assert sum(Decimal(row["weight_exact"]) for row in blend) == Decimal(1)


def test_capacity_gate_checks_each_foreign_source() -> None:
    config = {
        "probe": {
            "nominal_tokens": 1000,
            "mix_denominator": 100,
            "mix_numerators": {
                "new_greek": 79,
                "foreign_replay": 20,
                "old_greek_replay": 1,
            },
            "minimum_unique_capacity_ratio": 1.0,
        }
    }
    shards = [
        shard(0, "new_greek", "phase04_release", 790),
        shard(1, "foreign_replay", "source_a", 119, "0.6"),
        shard(2, "foreign_replay", "source_b", 81, "0.4"),
        shard(3, "old_greek_replay", "old", 10),
    ]
    report, failures = capacity_report(shards, config)
    assert any("source_a" in failure for failure in failures)
    assert report["pools"]["foreign_replay"]["passed"] is True
    shards[1]["counts"]["tokens"] = 120
    report, failures = capacity_report(shards, config)
    assert failures == []
    assert all(row["passed"] for row in report["foreign_sources"].values())


def test_capacity_gate_proves_each_physical_prefix_with_boundary() -> None:
    shards = [shard(0, "new_greek", "phase04_release", 10_000)]
    config = {
        "probe": {
            "nominal_tokens": 4_000,
            "effective_training_tokens": 4_000,
            "effective_training_samples": 4,
            "sequence_length": 1_000,
            "mix_denominator": 1,
            "mix_numerators": {"new_greek": 1},
            "minimum_unique_capacity_ratio": 1.005,
            "physical_prefix_sample_capacity_ratio": 1.005,
            "physical_prefix_boundary_samples": 1,
        }
    }
    uniqueness = {
        "pools": [{"pool": "new_greek", "unique_content_tokens": 10_000}],
        "sources": [
            {
                "pool": "new_greek",
                "source_name": "phase04_release",
                "unique_content_tokens": 10_000,
            }
        ],
        "tasks": [{"task_id": "task-0", "unique_content_tokens": 6_001}],
    }
    blend = compute_blend(shards, {"new_greek": 1}, 1)
    report, failures = capacity_report(
        shards, config, uniqueness=uniqueness, blend=blend
    )
    assert failures == []
    prefix = report["physical_prefixes"][0]
    assert prefix["planned_samples"] == 4
    assert prefix["required_samples"] == 6  # ceil(4 * 1.005) + one boundary
    assert prefix["available_nonrepeating_samples"] == 6
    uniqueness["tasks"][0]["unique_content_tokens"] = 6_000
    _, failures = capacity_report(shards, config, uniqueness=uniqueness, blend=blend)
    assert any("prefix/task-0" in failure for failure in failures)


def test_uniqueness_audit_rejects_duplicate_ids_and_discounts_content(
    tmp_path: Path,
) -> None:
    def with_ledger(index: int, rows: list[dict]) -> dict:
        value = shard(
            index, "new_greek", "phase04_release", sum(r["tokens"] for r in rows)
        )
        ledger = tmp_path / f"{index}.jsonl"
        ledger.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        value["outputs"]["retained_ledger"] = {"path": str(ledger)}
        value["counts"]["documents"] = len(rows)
        return value

    content = "a" * 64
    first = with_ledger(
        0,
        [
            {"doc_id": "docv2:" + "1" * 64, "text_sha256": content, "tokens": 7},
            {"doc_id": "docv2:" + "2" * 64, "text_sha256": "b" * 64, "tokens": 11},
        ],
    )
    second = with_ledger(
        1,
        [{"doc_id": "docv2:" + "3" * 64, "text_sha256": content, "tokens": 7}],
    )
    report = audit_unique_training_documents(
        [first, second], tmp_path / "audit.sqlite3"
    )
    assert report["global"]["identity_documents"] == 3
    assert report["global"]["unique_content_documents"] == 2
    assert report["global"]["unique_content_tokens"] == 18
    duplicate = with_ledger(
        2,
        [{"doc_id": "docv2:" + "1" * 64, "text_sha256": "c" * 64, "tokens": 5}],
    )
    with pytest.raises(ValueError, match="duplicate composite document identity"):
        audit_unique_training_documents(
            [first, duplicate], tmp_path / "duplicate.sqlite3"
        )


def test_tree_receipt_rejects_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a").write_text("a", encoding="utf-8")
    receipt = file_tree_receipt(root)
    from bridge_common import validate_file_tree_receipt

    validate_file_tree_receipt(receipt)
    (root / "extra").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory drift"):
        validate_file_tree_receipt(receipt)


def test_frozen_config_and_launcher_are_25b_only() -> None:
    config = json.loads(
        (ROOT / "configs" / "frozen_25b_td.json").read_text(encoding="utf-8")
    )
    assert config["probe"]["nominal_tokens"] == 25_000_000_000
    assert config["probe"]["mix_numerators"] == {
        "new_greek": 79,
        "foreign_replay": 20,
        "old_greek_replay": 1,
    }
    assert config["tokenizer"]["revision"] == "a4826df7f76b54cdd6dc21d09fe97283c466999b"
    launcher = (ROOT / "train" / "submit_25b_probe.sh").read_text(encoding="utf-8")
    assert 'DRY_RUN="${DRY_RUN:-1}"' in launcher
    assert "CONFIRM_GPU_LAUNCH=FULL_CORPUS_TD_25B" in launcher
    assert 'TRAIN_TOKENS="25000000000"' in launcher
    assert "13500000000" not in launcher
    assert "PHASE1_EXIT_ITER" in launcher  # explicitly rejected as stale input
    assert (
        "RESET_DATA_INDEX" in launcher
    )  # explicitly rejected; this is not the old two-phase run
    assert "verify_launch_assets.py" in launcher
    assert "START_ITERATION" in launcher
    assert "while (( current < TOTAL_ITER ))" not in launcher
    assert "24998051840" in launcher


def test_replay_acquisition_pins_adjudicated_historical_revisions() -> None:
    config = json.loads(
        (ROOT / "configs" / "replay_acquisition.json").read_text(encoding="utf-8")
    )
    assert config["status"] == "ready_for_receipt_bound_acquisition"
    pinned = {
        row["phase04_pin"]: row["revision"]
        for row in config["repositories"]
        if row.get("phase04_pin")
    }
    assert pinned == {
        "base": "e1d54136a880ed1df2ed95a5445dabd230453207",
        "apertus_overlap_overlay": "54faa75b5e0b4fad01bf7bf5541210c741cb10b8",
    }
    revisions = {row["repo_id"]: row["revision"] for row in config["repositories"]}
    assert (
        revisions["HuggingFaceFW/fineweb-edu"]
        == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    )
    assert revisions["epfml/FineWeb2-HQ"] == "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e"
    assert (
        revisions["HuggingFaceFW/fineweb-2"]
        == "af9c13333eb981300149d5ca60a8e9d659b276b9"
    )
    assert (
        revisions["HuggingFaceTB/finemath"]
        == "e92b25a616738fe95dc186b64dfb19f9c8525594"
    )
    assert (
        revisions["bigcode/starcoderdata"] == "9fc30b578cedaec69e47302df72cf00feed7c8c4"
    )
    assert all(len(str(row["revision"])) == 40 for row in config["repositories"])
    assert all(
        len(row["observed_head_2026_07_12"]) == 40 for row in config["repositories"]
    )
    evidence = config["historical_revision_adjudication"]
    assert evidence["status"] == "approved_from_retained_build_evidence"
    assert len(evidence["cache_refs"]) == 4
    assert evidence["starcoder_job_evidence"]["selected_link_count"] == 28


def test_cpu_stages_request_no_gpus() -> None:
    for path in sorted((ROOT / "clariden").glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        assert "--gpus" not in text
        assert "--gres=gpu" not in text
        assert "#SBATCH --partition=normal" in text
