from __future__ import annotations

import importlib
import json
import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

contracts = importlib.import_module(
    "subprojects.05_token_distillation_cpt.02_corpus_preparation."
    "15_clean_academic.production.contracts"
)
run_unit = importlib.import_module(
    "subprojects.05_token_distillation_cpt.02_corpus_preparation."
    "15_clean_academic.production.run_unit"
)
check_qa = importlib.import_module(
    "subprojects.05_token_distillation_cpt.02_corpus_preparation."
    "15_clean_academic.production.check_qa"
)
aggregate = importlib.import_module(
    "subprojects.05_token_distillation_cpt.02_corpus_preparation."
    "15_clean_academic.production.aggregate"
)
build_qa = importlib.import_module(
    "subprojects.05_token_distillation_cpt.02_corpus_preparation."
    "15_clean_academic.production.build_qa"
)
materialize_release = importlib.import_module(
    "subprojects.05_token_distillation_cpt.02_corpus_preparation."
    "15_clean_academic.production.materialize_release"
)
count_tokens = importlib.import_module(
    "subprojects.05_token_distillation_cpt.02_corpus_preparation."
    "15_clean_academic.production.count_tokens"
)
finalize_public_release = importlib.import_module(
    "subprojects.05_token_distillation_cpt.02_corpus_preparation."
    "15_clean_academic.production.finalize_public_release"
)


@dataclass
class FakeSpan:
    line_start: int
    line_end: int
    char_start: int
    char_end: int

    @property
    def line_count(self):
        return self.line_end - self.line_start

    @property
    def char_count(self):
        return self.char_end - self.char_start

    def text_from(self, text):
        return text[self.char_start : self.char_end]


class FakeCleaner:
    def clean_batch(self, texts, num_threads):
        output = []
        for text in texts:
            if text == "Body\nRef 2020":
                output.append(("Body", [FakeSpan(1, 2, 5, len(text))]))
            else:
                output.append((text, []))
        return output


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _fixture(
    tmp_path: Path, *, mode: str = "dry-run", apply: bool = False
) -> tuple[Path, Path, str]:
    release = tmp_path / "release"
    data = release / "data"
    data.mkdir(parents=True)
    source = data / "000002.parquet"
    schema = pa.schema(
        [
            ("text", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("chars", pa.int64()),
            ("non_whitespace_chars", pa.int64()),
            ("utf8_bytes", pa.int64()),
            ("approx_word_count", pa.int64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "text": "Body\nRef 2020",
                    "source_dataset": "Apothetirio_Kallipos",
                    "source_doc_id": "a",
                    "chars": None,
                    "non_whitespace_chars": None,
                    "utf8_bytes": None,
                    "approx_word_count": None,
                },
                {
                    "text": "Untouched",
                    "source_dataset": "Apothetirio_Kallipos",
                    "source_doc_id": "b",
                    "chars": 9,
                    "non_whitespace_chars": 9,
                    "utf8_bytes": 9,
                    "approx_word_count": 1,
                },
            ],
            schema=schema,
        ),
        source,
        row_group_size=2,
    )
    train_archive = tmp_path / "train.tar"
    archive = tmp_path / "gloss.tar"
    wheel = tmp_path / "extension.whl"
    stage = tmp_path / "model.json"
    train_archive.write_bytes(b"train-archive")
    archive.write_bytes(b"archive")
    wheel.write_bytes(b"wheel")
    stage.write_bytes(b'{"folds":[1,2,3,4,5]}')
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "bibliography-line-model-export-v2",
            "stages": {
                "test": {
                    "file": stage.name,
                    "bytes": stage.stat().st_size,
                    "sha256": contracts.sha256_file(stage),
                }
            },
        },
    )
    preflight = tmp_path / "preflight.json"
    parity = tmp_path / "parity.json"
    _write_json(
        preflight,
        {
            "schema_version": contracts.PREFLIGHT_SCHEMA,
            "status": "passed",
            "release": str(release),
        },
    )
    _write_json(
        parity,
        {
            "schema_version": contracts.PARITY_SCHEMA,
            "status": "passed",
            "train_commit": "a" * 40,
            "glossapi_commit": "b" * 40,
            "artifact_manifest_sha256": contracts.sha256_file(manifest),
            "equal_masks": 210704,
            "expected_lines": 210704,
            "candidate_positives": 19117,
            "expected_positives": 19117,
        },
    )
    policy = {
        "analysis_sources": {},
        "expected_analysis_rows": 2,
        "expected_apply_rows": 2 if apply else 0,
        "model_policy": {"character_damage_measure_approved": True},
        "license_overrides": {
            "glossAPI/libduth": {
                "scope": "v2 public release including cleaned libduth",
                "public_redistribution": True,
                "approved_by": "owner",
                "approved_on": "2026-07-28",
                "authorization_basis": (
                    "dataset owner directive; does not represent "
                    "rightsholder permission"
                ),
            }
        },
        "qa_gate": {
            "kallipos_sample_size": 1,
            "kallipos_primarily_bibliography_min": 1,
        },
        "publication_authorized": True,
        "publication_target": {
            "repo_id": "fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2",
            "visibility": "public",
            "gating": "manual",
        },
        "kallipos_apply_authorized": False,
    }
    run_root = tmp_path / "run"
    contract = {
        "schema_version": contracts.CONTRACT_SCHEMA,
        "mode": mode,
        "run_id": "test-run",
        "run_root": str(run_root),
        "release_root": str(release),
        "policy": policy,
        "publication_authorized": False,
        "code": {
            "train_commit": "a" * 40,
            "glossapi_commit": "b" * 40,
            "train_archive": {
                "path": str(train_archive),
                "sha256": contracts.sha256_file(train_archive),
            },
            "glossapi_archive": {
                "path": str(archive),
                "sha256": contracts.sha256_file(archive),
            },
            "glossapi_wheel": {
                "path": str(wheel),
                "sha256": contracts.sha256_file(wheel),
            },
        },
        "model_artifacts": {
            "path": str(manifest),
            "manifest_sha256": contracts.sha256_file(manifest),
            "stages": json.loads(manifest.read_text())["stages"],
        },
        "evidence": {
            "preflight": {
                "path": str(preflight),
                "sha256": contracts.sha256_file(preflight),
            },
            "parity": {
                "path": str(parity),
                "sha256": contracts.sha256_file(parity),
            },
        },
    }
    contract_path = tmp_path / "contract.json"
    contracts.atomic_write_json(contract_path, contract)
    unit_id = contracts.unit_id(2, 0, 1)
    plan = {
        "schema_version": contracts.PLAN_SCHEMA,
        "contract_sha256": contracts.sha256_file(contract_path),
        "rows": 2,
        "apply_rows": 2 if apply else 0,
        "apply_units": 1 if apply else 0,
        "units": [
            {
                "unit_id": unit_id,
                "rank": 2,
                "source_dataset": "Apothetirio_Kallipos",
                "apply": apply,
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": contracts.sha256_file(source),
                "row_group_start": 0,
                "row_group_end": 1,
                "rows": 2,
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    contracts.atomic_write_json(plan_path, plan)
    return contract_path, plan_path, unit_id


def test_stable_unit_id():
    assert contracts.unit_id(2, 0, 5) == "000002-rg0000-0005"


def test_dryrun_writes_exact_document_ledger_and_bound_receipt(tmp_path, monkeypatch):
    contract_path, plan_path, unit_id = _fixture(tmp_path)
    monkeypatch.setattr(run_unit, "_load_cleaner", lambda *args: FakeCleaner())
    args = Namespace(
        contract=str(contract_path),
        plan=str(plan_path),
        unit_id=unit_id,
        mode="dry-run",
        glossapi_src=None,
        threads=8,
        batch_size=2,
    )
    receipt = run_unit.run(args)
    assert receipt["docs"] == 2
    assert receipt["content_chars_removed"] == 8
    assert receipt["separator_chars_removed"] == 1
    assert receipt["total_chars_removed"] == 9
    assert receipt["lines_removed"] == 1
    ledger = pq.read_table(receipt["ledger"]["path"]).to_pylist()
    assert len(ledger) == 2
    assert ledger[0]["chars_before"] - ledger[0]["chars_after"] == 9
    assert not list((tmp_path / "run").rglob("*.partial"))
    assert run_unit.run(args) == receipt


def test_apply_writes_schema_preserving_fragment_and_aggregates_only_apply_units(
    tmp_path, monkeypatch
):
    contract_path, plan_path, unit_id = _fixture(tmp_path, mode="apply", apply=True)
    monkeypatch.setattr(run_unit, "_load_cleaner", lambda *args: FakeCleaner())
    args = Namespace(
        contract=str(contract_path),
        plan=str(plan_path),
        unit_id=unit_id,
        mode="apply",
        glossapi_src=None,
        threads=8,
        batch_size=2,
    )
    receipt = run_unit.run(args)
    output = pq.read_table(receipt["output"]["path"])
    rows = output.to_pylist()
    assert (
        output.schema
        == pq.read_table(tmp_path / "release" / "data" / "000002.parquet").schema
    )
    assert rows[0]["text"] == "Body"
    assert rows[0]["chars"] is None
    assert rows[0]["utf8_bytes"] is None
    assert rows[1]["text"] == "Untouched"
    assert rows[1]["chars"] == 9
    assert run_unit.run(args) == receipt

    summary = aggregate.run(
        Namespace(
            contract=str(contract_path),
            plan=str(plan_path),
            output=None,
        )
    )
    assert summary["schema_version"] == contracts.APPLY_SUMMARY_SCHEMA
    assert summary["mode"] == "apply"
    assert summary["overall"]["docs"] == 2


def test_reconstruct_count_and_finalize_public_release(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    models = pytest.importorskip("tokenizers.models")
    pre_tokenizers = pytest.importorskip("tokenizers.pre_tokenizers")
    contract_path, plan_path, unit_id = _fixture(tmp_path, mode="apply", apply=True)
    release = tmp_path / "release"
    data = release / "data"
    source_rank_two = data / "000002.parquet"
    files = []
    for rank in range(431):
        path = data / f"{rank:06d}.parquet"
        if rank != 2:
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "text": "Untouched",
                            "source_dataset": "other",
                            "source_doc_id": f"other-{rank}",
                            "chars": 9,
                            "non_whitespace_chars": 9,
                            "utf8_bytes": 9,
                            "approx_word_count": 1,
                        }
                    ],
                    schema=pq.read_schema(source_rank_two),
                ),
                path,
            )
        files.append(
            {
                "rank": rank,
                "origin": "candidate",
                "path": f"data/{rank:06d}.parquet",
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "bytes": path.stat().st_size,
                "sha256": contracts.sha256_file(path),
            }
        )
    decision = release / "manifests" / "dedup_decision_ledger.parquet"
    decision.parent.mkdir(exist_ok=True)
    pq.write_table(pa.table({"removed": pa.array([], type=pa.bool_())}), decision)
    source_manifest_path = release / "manifests" / "deduplicated_manifest.json"
    source_manifest = {
        "schema_version": "agent1_v5_deduplicated_release_manifest_v1",
        "status": "passed",
        "root": str(release),
        "repository_id": "owner/old",
        "private_only": True,
        "created_at": "old",
        "run_id": "old",
        "rows": sum(row["rows"] for row in files),
        "input_rows": sum(row["rows"] for row in files),
        "removed_rows": 0,
        "files": files,
        "inventory": {"path": "manifests/old.parquet", "rows": 431},
        "decision_ledger": {
            "path": "manifests/dedup_decision_ledger.parquet",
            "rows": 0,
            "bytes": decision.stat().st_size,
            "sha256": contracts.sha256_file(decision),
        },
    }
    _write_json(source_manifest_path, source_manifest)
    _write_json(
        release / "manifests" / "license_override_receipt.json",
        {"schema_version": "old", "status": "passed"},
    )

    contract = json.loads(contract_path.read_text())
    preflight_path = Path(contract["evidence"]["preflight"]["path"])
    preflight = json.loads(preflight_path.read_text())
    preflight["manifest"] = {
        "path": str(source_manifest_path),
        "sha256": contracts.sha256_file(source_manifest_path),
    }
    _write_json(preflight_path, preflight)
    contract["evidence"]["preflight"]["sha256"] = contracts.sha256_file(preflight_path)
    contracts.atomic_write_json(contract_path, contract)
    plan = json.loads(plan_path.read_text())
    plan["contract_sha256"] = contracts.sha256_file(contract_path)
    contracts.atomic_write_json(plan_path, plan)

    monkeypatch.setattr(run_unit, "_load_cleaner", lambda *args: FakeCleaner())
    run_unit.run(
        Namespace(
            contract=str(contract_path),
            plan=str(plan_path),
            unit_id=unit_id,
            mode="apply",
            glossapi_src=None,
            threads=8,
            batch_size=2,
        )
    )
    summary_path = tmp_path / "run" / "apply" / "summary.json"
    aggregate.run(
        Namespace(
            contract=str(contract_path),
            plan=str(plan_path),
            output=str(summary_path),
        )
    )
    candidate = tmp_path / "candidate"
    materialize_release.run(
        Namespace(
            contract=str(contract_path),
            plan=str(plan_path),
            summary=str(summary_path),
            output_root=str(candidate),
            repo_id="fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2",
            created_at="2026-07-28T00:00:00Z",
        )
    )
    candidate_manifest_path = candidate / "manifests" / "deduplicated_manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    assert candidate_manifest["publication_ready"] is False
    assert candidate_manifest["bibliography_cleaning"]["transformed_ranks"] == [2]
    assert (
        pq.read_table(candidate / "data" / "000002.parquet")["text"].to_pylist()[0]
        == "Body"
    )
    assert (
        os.stat(data / "000000.parquet").st_ino
        == os.stat(candidate / "data" / "000000.parquet").st_ino
    )

    tokenizer = tokenizers.Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "Body": 1, "Ref": 2, "2020": 3, "Untouched": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    token_plan_path = tmp_path / "token-plan.json"
    token_plan = count_tokens.build_plan(
        Namespace(
            base_manifest=str(source_manifest_path),
            cleaned_manifest=str(candidate_manifest_path),
            tokenizer_json=str(tokenizer_json),
            tokenizer_sha256=contracts.sha256_file(tokenizer_json),
            tokenizer_repo_id="owner/tokenizer",
            tokenizer_revision="a" * 40,
            vocab_size=5,
            output=str(token_plan_path),
        )
    )
    receipt_dir = tmp_path / "token-receipts"
    for index in range(len(token_plan["tasks"])):
        count_tokens.run_task(
            Namespace(
                plan=str(token_plan_path),
                task_index=index,
                receipt_dir=str(receipt_dir),
                batch_size=32,
            )
        )
    token_summary_path = tmp_path / "token-summary.json"
    token_summary = count_tokens.aggregate(
        Namespace(
            plan=str(token_plan_path),
            receipt_dir=str(receipt_dir),
            output=str(token_summary_path),
        )
    )
    assert token_summary["cleaned"]["documents"] == source_manifest["rows"]
    assert (
        token_summary["cleaned"]["training_tokens"]
        < token_summary["base"]["training_tokens"]
    )

    final = tmp_path / "final"
    final_manifest = finalize_public_release.run(
        Namespace(
            contract=str(contract_path),
            candidate=str(candidate),
            token_summary=str(token_summary_path),
            repo_id="fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2",
            created_at="2026-07-28T00:00:00Z",
            output_root=str(final),
        )
    )
    assert final_manifest["publication_ready"] is True
    assert final_manifest["private_only"] is False
    assert (
        "does not"
        in (final / "manifests" / "license_override_receipt.json").read_text()
    )
    assert "Bibliography-cleaned v2" in (final / "README.md").read_text()


def test_apply_refuses_unit_outside_frozen_scope(tmp_path, monkeypatch):
    contract_path, plan_path, unit_id = _fixture(tmp_path, mode="apply", apply=False)
    monkeypatch.setattr(run_unit, "_load_cleaner", lambda *args: FakeCleaner())
    with pytest.raises(ValueError, match="outside the frozen apply scope"):
        run_unit.run(
            Namespace(
                contract=str(contract_path),
                plan=str(plan_path),
                unit_id=unit_id,
                mode="apply",
                glossapi_src=None,
                threads=8,
                batch_size=2,
            )
        )


def test_contract_rejects_artifact_tampering(tmp_path):
    contract_path, _, _ = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text())
    Path(contract["model_artifacts"]["path"]).parent.joinpath("model.json").write_text(
        "tampered"
    )
    with pytest.raises(
        ValueError, match="artifact size mismatch|artifact hash mismatch"
    ):
        contracts.validate_contract(contract_path)


def test_qa_packet_rejects_ledger_changed_after_aggregation(tmp_path, monkeypatch):
    contract_path, plan_path, unit_id = _fixture(tmp_path)
    monkeypatch.setattr(run_unit, "_load_cleaner", lambda *args: FakeCleaner())
    receipt = run_unit.run(
        Namespace(
            contract=str(contract_path),
            plan=str(plan_path),
            unit_id=unit_id,
            mode="dry-run",
            glossapi_src=None,
            threads=8,
            batch_size=2,
        )
    )
    summary_path = tmp_path / "run" / "dry-run" / "summary.json"
    aggregate.run(
        Namespace(
            contract=str(contract_path),
            plan=str(plan_path),
            output=str(summary_path),
        )
    )
    ledger_path = Path(receipt["ledger"]["path"])
    pq.write_table(pq.read_table(ledger_path), ledger_path, compression="gzip")
    with pytest.raises(ValueError, match="ledger changed after aggregation"):
        build_qa.run(
            Namespace(
                contract=str(contract_path),
                plan=str(plan_path),
                summary=str(summary_path),
                output=None,
            )
        )


def test_qa_gate_rejects_uncertainty_and_accepts_complete_review(tmp_path):
    packet = {
        "schema_version": contracts.QA_PACKET_SCHEMA,
        "run_id": "x",
        "gate": {
            "kallipos_sample_size": 1,
            "kallipos_primarily_bibliography_min": 1,
        },
        "items": [
            {
                "item_id": "k",
                "reasons": ["kallipos_median", "openarchives_over_50pct"],
            }
        ],
    }
    packet_path = tmp_path / "packet.json"
    contracts.atomic_write_json(packet_path, packet)
    review = {
        "schema_version": contracts.QA_REVIEW_SCHEMA,
        "status": "complete",
        "packet_sha256": contracts.sha256_file(packet_path),
        "reviewer": "Codex",
        "reviewed_utc": "2026-07-27T00:00:00Z",
        "decisions": [
            {
                "item_id": "k",
                "classification": "acceptable",
                "primarily_bibliography": True,
                "rationale": "citation list",
            }
        ],
    }
    review_path = tmp_path / "review.json"
    contracts.atomic_write_json(review_path, review)
    result = check_qa.run(
        Namespace(
            packet=str(packet_path),
            review=str(review_path),
            output=str(tmp_path / "gate.json"),
        )
    )
    assert result["status"] == "passed"
    review["status"] = "incomplete"
    contracts.atomic_write_json(review_path, review)
    result = check_qa.run(
        Namespace(
            packet=str(packet_path),
            review=str(review_path),
            output=str(tmp_path / "gate-incomplete.json"),
        )
    )
    assert result["status"] == "failed"
    review["status"] = "complete"
    review["decisions"][0]["classification"] = "uncertain"
    contracts.atomic_write_json(review_path, review)
    result = check_qa.run(
        Namespace(
            packet=str(packet_path),
            review=str(review_path),
            output=str(tmp_path / "gate-failed.json"),
        )
    )
    assert result["status"] == "failed"
