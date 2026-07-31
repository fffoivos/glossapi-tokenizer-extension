from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bridge_common import (  # noqa: E402
    document_key,
    build_launch_dependency_receipts,
    file_tree_receipt,
    heldout_hash,
    iter_index_lengths,
    selected_by_threshold,
    tokenizer_tree_receipt,
    validate_frozen_repository,
    validate_launch_dependency_receipts,
    validate_tokenizer_tree_receipt,
    write_index,
)
from finalize_bridge import (  # noqa: E402
    audit_unique_training_documents,
    capacity_report,
    compute_blend,
)
from verify_launch_assets import validate_tokenizer_asset  # noqa: E402
from acquire_replay_sources import _copy_atomic, _select_mapping_files  # noqa: E402


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


def test_replay_atomic_copy_publishes_exact_bytes_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cache" / "source.parquet"
    source.parent.mkdir()
    source.write_bytes((b"receipt-bound-replay\n" * 4096) + b"end")
    destination = tmp_path / "stage" / "part.parquet"
    _copy_atomic(source, destination, replace=False)
    assert destination.read_bytes() == source.read_bytes()
    assert not list(destination.parent.glob(".*.partial"))
    _copy_atomic(source, destination, replace=False)
    assert destination.read_bytes() == source.read_bytes()


def test_replay_atomic_copy_resolves_hf_snapshot_symlink(tmp_path: Path) -> None:
    blob = tmp_path / "cache" / "blobs" / "digest"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"parquet payload")
    snapshot = tmp_path / "cache" / "snapshots" / "commit" / "data" / "part.parquet"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(Path("../../../blobs/digest"))
    destination = tmp_path / "replay" / "data" / "part.parquet"
    _copy_atomic(snapshot, destination, replace=False)
    assert destination.read_bytes() == blob.read_bytes()
    assert not destination.is_symlink()
    assert not list(destination.parent.glob(".*.partial"))


def test_replay_file_cap_is_exact_deterministic_and_domain_separated() -> None:
    files = [f"ell_Grek/{index:03d}.parquet" for index in range(20)]
    mapping = {
        "source_name": "source-a",
        "remote_glob": "ell_Grek/*.parquet",
        "selection": {
            "method": "domain_separated_sha256_rank_v1",
            "seed": 20260609,
            "file_count": 4,
        },
    }
    selected, receipt = _select_mapping_files(
        files,
        mapping,
        repo_id="owner/dataset",
        revision="0" * 40,
    )
    repeated, _ = _select_mapping_files(
        list(reversed(files)),
        mapping,
        repo_id="owner/dataset",
        revision="0" * 40,
    )
    assert selected == repeated
    assert len(selected) == 4
    assert receipt == {
        "method": "domain_separated_sha256_rank_v1",
        "seed": 20260609,
        "matched_file_count": 20,
        "selected_file_count": 4,
        "selection_domain": "full-cpt-replay-file-selection-v1",
    }
    other = dict(mapping, source_name="source-b")
    other_selected, _ = _select_mapping_files(
        files,
        other,
        repo_id="owner/dataset",
        revision="0" * 40,
    )
    assert selected != other_selected


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


def test_tokenizer_tree_receipt_rejects_mutation_extra_and_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tokenizer"
    root.mkdir()
    tokenizer_json = root / "tokenizer.json"
    tokenizer_json.write_text("frozen", encoding="utf-8")
    receipt = tokenizer_tree_receipt(root)
    assert validate_tokenizer_tree_receipt(receipt) == root.resolve()
    input_receipt = {"tokenizer": dict(receipt)}
    assets = {"tokenizer": {"root": str(root.resolve()), "tree": receipt}}
    assert validate_tokenizer_asset(input_receipt, assets, root) == root.resolve()

    tokenizer_json.write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="content drift"):
        validate_tokenizer_tree_receipt(receipt)
    with pytest.raises(ValueError, match="content drift"):
        validate_tokenizer_asset(input_receipt, assets, root)
    tokenizer_json.write_text("frozen", encoding="utf-8")

    extra = root / "extra.json"
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="content drift"):
        validate_tokenizer_tree_receipt(receipt)
    extra.unlink()

    internal_link = root / "linked.json"
    internal_link.symlink_to(tokenizer_json)
    with pytest.raises(ValueError, match="contains a symlink"):
        validate_tokenizer_tree_receipt(receipt)
    internal_link.unlink()

    root_link = tmp_path / "tokenizer-link"
    root_link.symlink_to(root, target_is_directory=True)
    linked_receipt = dict(receipt, root=str(root_link))
    with pytest.raises(ValueError, match="root is a symlink"):
        validate_tokenizer_tree_receipt(linked_receipt)


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
    assert "MEGATRON_LM_SWISSAI_DIR=$MEGATRON_LM_SWISSAI_DIR" in launcher
    assert "MEGATRON_DIR=$MEGATRON_DIR" not in launcher
    common = (ROOT.parent / "03_training_experiments" / "configs" / "common_cpt.env").read_text(
        encoding="utf-8"
    )
    assert 'MEGATRON_LM_SWISSAI_DIR="${MEGATRON_LM_SWISSAI_DIR:-' in common


def test_full_corpus_launch_revalidates_inside_batch_before_config_source() -> None:
    launcher = (ROOT / "train" / "submit_25b_probe.sh").read_text(encoding="utf-8")
    config = (ROOT / "train" / "full_corpus_25b.env").read_text(encoding="utf-8")
    trainer_path = (
        ROOT.parents[1]
        / "03_apertus_extension_and_embedding_adaptation"
        / "03_4_implementation_experiments"
        / "init_bakeoff"
        / "bakeoff_training"
        / "bakeoff_train.sbatch"
    )
    trainer = trainer_path.read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_launch_assets.py").read_text(encoding="utf-8")
    assert 'case "${FULL_CPT_LAUNCH_VERIFY:-0}"' in trainer
    assert trainer.index("$FULL_CPT_VERIFY_SCRIPT") < trainer.index('source "$TRAIN_CONFIG"')
    assert 'uenv run "$FULL_CPT_VERIFY_UENV" --view=default' in trainer
    assert "FULL_CPT_JOB_START_VERIFY_COMPLETED=1" in trainer
    assert 'FULL_CPT_REQUIRE_JOB_START_VERIFY="1"' in config
    assert "full-corpus training config requires the job-start receipt verifier" in trainer

    required_exports = (
        "UENV",
        "PYTHON",
        "SCRIPT",
        "BRIDGE_MANIFEST",
        "TRAINING_DATA_ENV",
        "ASSETS_RECEIPT",
        "REPO_ROOT",
        "TRAINING_ENV",
        "COMMON_TRAINING_ENV",
        "TRAINER",
        "RUNTIME_WRAPPER",
        "LAUNCHER",
        "EXPECTED_LOAD_CHECKPOINT",
        "EXPECTED_MEGATRON_DIR",
        "EXPECTED_TOKENIZER_DIR",
        "START_ITERATION",
        "EXPECTED_EXIT_INTERVAL",
        "PROBE_PLAN",
        "RESUME_RECEIPT",
    )
    for suffix in required_exports:
        variable = f"FULL_CPT_VERIFY_{suffix}"
        assert variable in launcher
        assert variable in trainer
    assert "--expected-load-checkpoint" in trainer
    assert "--expected-megatron-dir" in trainer
    assert "--expected-tokenizer-dir" in trainer
    assert "--expected-load-checkpoint" in launcher
    assert "--expected-megatron-dir" in launcher
    assert "--expected-tokenizer-dir" in launcher
    assert "validate_frozen_repository" in verifier
    assert "validate_launch_dependency_receipts" in verifier
    assert "validate_tokenizer_tree_receipt" in verifier
    freezer = (ROOT / "scripts" / "freeze_training_assets.py").read_text(
        encoding="utf-8"
    )
    assert '"tokenizer": tokenizer' in freezer
    assert "validate_tokenizer_tree_receipt" in freezer
    assert "full_cpt_assert_frozen_recipe" in trainer
    assert "FULL_CPT_FROZEN_RECIPE_CONTRACT" in config


def test_legacy_trainer_without_hook_reaches_legacy_config_unchanged(
    tmp_path: Path,
) -> None:
    trainer = (
        ROOT.parents[1]
        / "03_apertus_extension_and_embedding_adaptation"
        / "03_4_implementation_experiments"
        / "init_bakeoff"
        / "bakeoff_training"
        / "bakeoff_train.sbatch"
    )
    legacy_config = tmp_path / "legacy.env"
    legacy_config.write_text("exit 42\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("FULL_CPT_LAUNCH_VERIFY", None)
    environment.update(
        {
            "SCRIPT_DIR_OVERRIDE": str(trainer.parent),
            "TRAIN_CONFIG_OVERRIDE": str(legacy_config),
        }
    )
    result = subprocess.run(
        ["bash", str(trainer)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 42
    assert "job-start verification" not in result.stderr


def test_full_corpus_config_cannot_bypass_job_start_hook(tmp_path: Path) -> None:
    trainer = (
        ROOT.parents[1]
        / "03_apertus_extension_and_embedding_adaptation"
        / "03_4_implementation_experiments"
        / "init_bakeoff"
        / "bakeoff_training"
        / "bakeoff_train.sbatch"
    )
    bridge_env = tmp_path / "training_data.env"
    bridge_env.write_text(
        "\n".join(
            [
                'FULL_CPT_TOKENIZER_DIR="/tmp/tokenizer"',
                'FULL_CPT_DATA_PREFIX="1 /tmp/data"',
                'FULL_CPT_BRIDGE_MANIFEST="/tmp/bridge.json"',
                'FULL_CPT_INPUT_RECEIPT="/tmp/input.json"',
                'FULL_CPT_HELDOUT_MANIFEST="/tmp/heldout.json"',
                'FULL_CPT_MIX_RECIPE="/tmp/mix.json"',
                'VAL_DATA_DIR="/tmp/heldout"',
                'EXTRA_VALID_SETS="hplt openarchives greek_phd"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("FULL_CPT_LAUNCH_VERIFY", None)
    environment.update(
        {
            "SCRIPT_DIR_OVERRIDE": str(trainer.parent),
            "TRAIN_CONFIG_OVERRIDE": str(ROOT / "train" / "full_corpus_25b.env"),
            "BRIDGE_DATA_ENV": str(bridge_env),
            "INIT_CKPT": "/tmp/init",
        }
    )
    result = subprocess.run(
        ["bash", str(trainer)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 10, result.stderr
    assert "requires the job-start receipt verifier" in result.stderr


def test_full_corpus_config_sanitizes_inherited_semantic_recipe(
    tmp_path: Path,
) -> None:
    bridge_env = tmp_path / "training_data.env"
    bridge_env.write_text(
        "\n".join(
            [
                'FULL_CPT_TOKENIZER_DIR="/frozen/tokenizer"',
                'FULL_CPT_DATA_PREFIX="79 /frozen/new,20 /frozen/foreign,1 /frozen/old"',
                'FULL_CPT_BRIDGE_MANIFEST="/frozen/bridge.json"',
                'FULL_CPT_INPUT_RECEIPT="/frozen/input.json"',
                'FULL_CPT_HELDOUT_MANIFEST="/frozen/heldout.json"',
                'FULL_CPT_MIX_RECIPE="/frozen/mix.json"',
                'VAL_DATA_DIR="/frozen/validation"',
                'EXTRA_VALID_SETS="hplt openarchives greek_phd"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script = tmp_path / "assert-recipe.sh"
    script.write_text(
        "\n".join(
            [
                "set -euo pipefail",
                f'source "{ROOT / "train" / "full_corpus_25b.env"}"',
                "full_cpt_assert_frozen_recipe",
                'printf "%s\\n" "$USE_MOCK_DATA|$UENV_IMAGE|$TENSOR_MODEL_PARALLEL_SIZE|$PIPELINE_MODEL_PARALLEL_SIZE|$LR_WARMUP_INIT|$DATA_SEED|$CURRICULUM_ORDER_MODE|$LOSS_OBJECTIVE|$RESUME_TRAINING|$EXIT_INTERVAL|$BASE_TOKENIZER_DIR|$BASE_DATA_PREFIX"',
                "[[ ! -v DDP_BUCKET_SIZE && ! -v TRAINER_WRAPPER ]]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "BRIDGE_DATA_ENV": str(bridge_env),
            "FULL_CPT_VERIFY_START_ITERATION": "119",
            "FULL_CPT_VERIFY_EXPECTED_EXIT_INTERVAL": "238",
            "FULL_CPT_VERIFY_EXPECTED_LOAD_CHECKPOINT": "/frozen/resume",
            "FULL_CPT_VERIFY_EXPECTED_MEGATRON_DIR": "/frozen/megatron",
            "USE_MOCK_DATA": "1",
            "UENV_IMAGE": "evil/uenv:v0",
            "TENSOR_MODEL_PARALLEL_SIZE": "99",
            "PIPELINE_MODEL_PARALLEL_SIZE": "7",
            "LR_WARMUP_INIT": "9",
            "DATA_SEED": "0",
            "CURRICULUM_ORDER_MODE": "randomized",
            "LOSS_OBJECTIVE": "ntp",
            "RESUME_TRAINING": "0",
            "EXIT_INTERVAL": "999",
            "ENABLE_EXTRA_VALID": "0",
            "USE_DISTRIBUTED_OPTIMIZER": "0",
            "USE_COMM_OVERLAP": "0",
            "DDP_BUCKET_SIZE": "1",
            "TRAINER_WRAPPER": "/evil/wrapper",
        }
    )
    result = subprocess.run(
        ["bash", str(script)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "0|pytorch/v2.9.1:v2|2|1|5.5e-6|20260609|physical_order|goldfish|"
        "1|238|/frozen/tokenizer|79 /frozen/new,20 /frozen/foreign,1 /frozen/old"
    )


@pytest.mark.parametrize(
    ("start", "probe_plan", "resume_receipt", "expects_resume"),
    [(0, "", "", False), (119, "/tmp/plan.json", "/tmp/resume.json", True)],
)
def test_job_start_hook_builds_initial_and_resume_verifier_args(
    tmp_path: Path,
    start: int,
    probe_plan: str,
    resume_receipt: str,
    expects_resume: bool,
) -> None:
    trainer = (
        ROOT.parents[1]
        / "03_apertus_extension_and_embedding_adaptation"
        / "03_4_implementation_experiments"
        / "init_bakeoff"
        / "bakeoff_training"
        / "bakeoff_train.sbatch"
    )
    runtime_wrapper = (
        trainer.parent.parent
        / "megatron_patches"
        / "runtime"
        / "pretrain_gpt_te_guard.py"
    )
    config = tmp_path / "stop-after-hook.env"
    config.write_text("exit 42\n", encoding="utf-8")
    captured = tmp_path / "uenv-args.txt"
    fake_uenv = tmp_path / "uenv"
    fake_uenv.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$HOOK_ARGS_FILE"\n',
        encoding="utf-8",
    )
    fake_uenv.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "HOOK_ARGS_FILE": str(captured),
            "SCRIPT_DIR_OVERRIDE": str(trainer.parent),
            "TRAIN_CONFIG_OVERRIDE": str(config),
            "FULL_CPT_LAUNCH_VERIFY": "1",
            "FULL_CPT_VERIFY_UENV": "pytorch/test:v1",
            "FULL_CPT_VERIFY_PYTHON": "/runtime/python",
            "FULL_CPT_VERIFY_SCRIPT": "/repo/verify_launch_assets.py",
            "FULL_CPT_VERIFY_BRIDGE_MANIFEST": "/receipts/bridge.json",
            "FULL_CPT_VERIFY_TRAINING_DATA_ENV": "/receipts/data.env",
            "FULL_CPT_VERIFY_ASSETS_RECEIPT": "/receipts/assets.json",
            "FULL_CPT_VERIFY_REPO_ROOT": "/repo",
            "FULL_CPT_VERIFY_TRAINING_ENV": str(config),
            "FULL_CPT_VERIFY_COMMON_TRAINING_ENV": "/repo/common.env",
            "FULL_CPT_VERIFY_TRAINER": str(trainer),
            "FULL_CPT_VERIFY_RUNTIME_WRAPPER": str(runtime_wrapper),
            "FULL_CPT_VERIFY_LAUNCHER": "/repo/submit_25b_probe.sh",
            "FULL_CPT_VERIFY_EXPECTED_LOAD_CHECKPOINT": "/checkpoints/load",
            "FULL_CPT_VERIFY_EXPECTED_MEGATRON_DIR": "/megatron",
            "FULL_CPT_VERIFY_EXPECTED_TOKENIZER_DIR": "/tokenizer",
            "FULL_CPT_VERIFY_START_ITERATION": str(start),
            "FULL_CPT_VERIFY_EXPECTED_EXIT_INTERVAL": str(start + 119),
            "FULL_CPT_VERIFY_PROBE_PLAN": probe_plan,
            "FULL_CPT_VERIFY_RESUME_RECEIPT": resume_receipt,
        }
    )
    result = subprocess.run(
        ["bash", str(trainer)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 42, result.stderr
    arguments = captured.read_text(encoding="utf-8").splitlines()
    assert arguments[0:4] == ["run", "pytorch/test:v1", "--view=default", "--"]
    assert arguments[4:6] == ["/runtime/python", "/repo/verify_launch_assets.py"]
    assert arguments[arguments.index("--start-iteration") + 1] == str(start)
    assert arguments[arguments.index("--expected-tokenizer-dir") + 1] == "/tokenizer"
    if expects_resume:
        assert arguments[arguments.index("--probe-plan") + 1] == probe_plan
        assert (
            arguments[arguments.index("--resume-checkpoint-receipt") + 1]
            == resume_receipt
        )
    else:
        assert "--probe-plan" not in arguments
        assert "--resume-checkpoint-receipt" not in arguments


def test_repository_binding_requires_exact_clean_frozen_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "frozen"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    receipt = {"repository": {"root": str(repo), "commit": commit}}
    assert validate_frozen_repository(receipt, repo) == {
        "root": str(repo.resolve()),
        "commit": commit,
        "clean": True,
    }

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty or has untracked"):
        validate_frozen_repository(receipt, repo)
    tracked.write_text("frozen\n", encoding="utf-8")
    untracked = repo / "untracked.txt"
    untracked.write_text("untracked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty or has untracked"):
        validate_frozen_repository(receipt, repo)
    untracked.unlink()

    tracked.write_text("next clean commit\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "next"], cwd=repo, check=True)
    with pytest.raises(ValueError, match="commit drift"):
        validate_frozen_repository(receipt, repo)


@pytest.mark.parametrize(
    "mutated",
    [
        "common_training_environment",
        "launcher",
        "runtime_wrapper",
        "trainer",
        "training_environment",
    ],
)
def test_all_transitive_launch_dependencies_are_receipt_bound(
    tmp_path: Path, mutated: str
) -> None:
    dependencies = {}
    for name in (
        "common_training_environment",
        "launcher",
        "runtime_wrapper",
        "trainer",
        "training_environment",
    ):
        path = tmp_path / name
        path.write_text(f"{name}\n", encoding="utf-8")
        dependencies[name] = path
    receipts = build_launch_dependency_receipts(dependencies)
    assert set(validate_launch_dependency_receipts(receipts, dependencies)) == set(
        dependencies
    )
    dependencies[mutated].write_text(f"{mutated} drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match=mutated):
        validate_launch_dependency_receipts(receipts, dependencies)


def test_clariden_submit_uses_mode_specific_and_uenv_safe_requirements() -> None:
    submit = (ROOT / "clariden" / "submit.sh").read_text(encoding="utf-8")
    paths = (ROOT / "clariden" / "paths.env").read_text(encoding="utf-8")
    assert "restage) bridge_require_base" in submit
    assert "freeze|after-freeze) bridge_require_paths" in submit
    assert "status) bridge_require_run" in submit
    assert 'test -x "$RUNTIME_VENV/bin/python"' not in paths
    assert "bridge_python -c" in paths
    greekmmlu = (
        ROOT.parent / "04_full_corpus_preparation" / "clariden" / "68_freeze_greekmmlu.sbatch"
    ).read_text(encoding="utf-8")
    assert 'uenv run "$PHASE04_UENV" --view=default' in greekmmlu


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
    assert config["selection_policy"] == (
        "all_matching_or_domain_separated_sha256_ranked_capacity_sample_v1"
    )
    assert config["capacity_sampling"]["expected_selected_files"] == 355
    assert (
        config["capacity_sampling"]["expected_selected_remote_bytes"]
        == 250_673_537_368
    )
    repositories = {row["repo_id"]: row for row in config["repositories"]}
    hq_counts = [
        row["selection"]["file_count"]
        for row in repositories["epfml/FineWeb2-HQ"]["mappings"]
    ]
    assert hq_counts == [2] * 12
    fw2_counts = [
        row["selection"]["file_count"]
        for row in repositories["HuggingFaceFW/fineweb-2"]["mappings"]
    ]
    assert fw2_counts == [1] * 11
    assert repositories["HuggingFaceTB/finemath"]["mappings"][0]["selection"][
        "file_count"
    ] == 4


def test_cpu_stages_request_no_gpus() -> None:
    for path in sorted((ROOT / "clariden").glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        assert "--gpus" not in text
        assert "--gres=gpu" not in text
        assert "#SBATCH --partition=normal" in text
