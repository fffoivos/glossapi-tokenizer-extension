from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
tokenizers = pytest.importorskip("tokenizers")


PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "agent1_v3_release.py"


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    normalized = [{key: row.get(key) for key in columns} for row in rows]
    pq.write_table(pa.Table.from_pylist(normalized), path, compression="zstd")


def _receipt(path: Path, *, relative_to: Path) -> dict[str, object]:
    metadata = pq.ParquetFile(path).metadata
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha_file(path),
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _canonical_sha(value: dict[str, object], *, omitted_key: str) -> str:
    payload = dict(value)
    payload.pop(omitted_key, None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _tokenizer(path: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "email": 1, "pii": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))
    return path


def _source_fields() -> dict[str, object]:
    return {
        "source_id": "demo-source",
        "acquisition_source_id": "demo-source",
        "source_dataset": "demo/dataset",
        "source_repo_id": "demo/dataset",
        "source_revision": "a" * 40,
    }


SITE_EVIDENCE_NAMES = (
    "candidate_roster",
    "review_packet",
    "review_requests",
    "review_responses",
    "response_execution_receipt",
    "adjudication_execution_receipt",
    "stage35_review_closure",
    "review_sample_quality_summary",
    "review_sample_quality_handoff",
    "quality_summary",
    "lineage_summary",
    "source_novelty",
    "license_adjudication",
    "review_aggregate",
    "admission_confirmation",
)


def _release_evidence(tmp_path: Path) -> dict[str, Path]:
    """Create a compact, receipt-bound Phase-10 evidence closure fixture."""

    root = tmp_path / "phase10-evidence"
    paths: dict[str, Path] = {}

    def write(name: str, value: object) -> Path:
        path = root / f"{name}.json"
        _write_json(path, value)
        paths[name] = path
        return path

    write("candidate_roster", {"schema_version": "agent1_full_corpus_v3_candidate_roster_v1"})
    write(
        "review_packet",
        {
            "schema_version": "agent1_v3_review_packet_manifest_v1",
            "status": "materialized_no_model_invocation",
        },
    )
    review_requests = root / "review_requests.jsonl"
    review_requests.parent.mkdir(parents=True, exist_ok=True)
    review_requests.write_text('{"schema_version":"agent1_v3_review_request_v1"}\n', encoding="utf-8")
    paths["review_requests"] = review_requests
    review_responses = root / "review_responses.jsonl"
    review_responses.write_text('{"schema_version":"agent1_v3_review_response_v1"}\n', encoding="utf-8")
    paths["review_responses"] = review_responses
    write(
        "response_execution_receipt",
        {
            "schema_version": "agent1_v3_codex_review_response_execution_receipt_v1",
            "status": "complete",
        },
    )
    write(
        "adjudication_execution_receipt",
        {
            "schema_version": "agent1_v3_codex_review_adjudication_execution_receipt_v1",
            "status": "complete",
            "final_adjudication_manifest": {"status": "complete", "pending_count": 0},
        },
    )
    write(
        "stage35_review_closure",
        {"schema_version": "agent1_v3_quality_review_evidence_closure_v1", "status": "passed"},
    )
    write(
        "review_sample_quality_summary",
        {"schema_version": "agent1_v3_masked_review_sample_quality_summary_v1", "status": "passed"},
    )
    write(
        "review_sample_quality_handoff",
        {"schema_version": "agent1_v3_masked_review_sample_quality_handoff_v1", "status": "passed"},
    )
    write(
        "quality_summary",
        {"schema_version": "dataset_quality_summary_v2", "status": "passed", "scan_mode": "full_scan"},
    )
    write("lineage_summary", {"schema_version": "full_cpt_lineage_summary_v1"})
    write("source_novelty", {"schema_version": "full_cpt_source_novelty_v1"})
    write(
        "license_adjudication",
        {"schema_version": "full_cpt_source_license_adjudication_v1", "status": "technical_audit_complete"},
    )

    aggregate_input_names = tuple(name for name in SITE_EVIDENCE_NAMES if name not in {"review_aggregate", "admission_confirmation"})
    review_aggregate = write(
        "review_aggregate",
        {
            "schema_version": "agent1_full_corpus_v3_source_review_aggregate_v1",
            "status": "passed_review_evidence_no_admission_decision",
            "inputs": {name: _binding(paths[name]) for name in aggregate_input_names},
            "review_closure": {"status": "complete", "pending_count": 0},
        },
    )
    admission_packet = root / "admission-packet.json"
    _write_json(
        admission_packet,
        {
            "schema_version": "agent1_full_corpus_v3_source_admission_packet_v2",
            "status": "pending_user_confirmation",
            "review_aggregate_sha256": _sha_file(review_aggregate),
            "inputs": {"review_aggregate": _binding(review_aggregate)},
        },
    )
    write(
        "admission_confirmation",
        {
            "schema_version": "agent1_full_corpus_v3_source_admission_confirmation_v1",
            "status": "approved",
            "confirmation_mode": "explicit_hash_confirmed_user_confirmation",
            "packet": _binding(admission_packet),
            "user_confirmed_packet_sha256": _sha_file(admission_packet),
        },
    )

    waterfall = write(
        "transformation_waterfall",
        {
            "schema_version": "agent1_full_corpus_v3_token_waterfall_v1",
            "status": "passed_with_independent_semantic_review_pending",
            "anonymization_audit": {
                "schema_version": "agent1_full_corpus_v3_anonymization_audit_v1",
                "status": "automatic_checks_passed_semantic_review_pending",
                "false_positive_audit": {
                    "automatic_policy_lineage_checks": {"status": "passed"},
                    "independent_semantic_review": {
                        "status": "pending",
                        "required_before_any_claim_of_semantic_false_positive_clearance": True,
                        "eligible_rows": 1,
                    },
                },
            },
            "invariants": {
                "dedup_precedes_greekmmlu": True,
                "greekmmlu_precedes_anonymization": True,
                "tokens_are_exact_pinned_tokenizer_counts": True,
                "raw_text_or_pii_in_output": False,
            },
        },
    )
    waterfall_payload = json.loads(waterfall.read_text(encoding="utf-8"))
    clearance_payload: dict[str, object] = {
        "schema_version": "agent1_full_corpus_v3_anonymization_semantic_false_positive_clearance_v1",
        "status": "passed",
        "completed_at": "2026-07-13T12:00:00Z",
        "transformation_waterfall": _binding(waterfall),
        "anonymization_audit_sha256": _canonical_sha(
            waterfall_payload["anonymization_audit"], omitted_key="__never_present__"
        ),
        "independence": {
            "reviewer_is_independent": True,
            "independent_of_automatic_policy_lineage_checks": True,
            "protected_review_environment": True,
            "raw_text_or_pii_in_clearance": False,
        },
        "independent_semantic_review": {
            "status": "cleared",
            "eligible_rows": 1,
            "reviewed_rows": 1,
            "unresolved_rows": 0,
            "false_positive_findings": 0,
        },
    }
    clearance_payload["clearance_sha256"] = _canonical_sha(clearance_payload, omitted_key="clearance_sha256")
    clearance = write("anonymization_semantic_clearance", clearance_payload)
    paths["transformation_waterfall"] = waterfall
    paths["anonymization_semantic_clearance"] = clearance
    return paths


def _fixture(tmp_path: Path, *, applied_structural: bool = False) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    original = "email original@example.gr"
    masked = "email <email-pii>"
    original_hash = _sha_text(original)
    masked_hash = _sha_text(masked)
    uid = "uid-1"
    input_representation = f"normalized-v1:{uid}:{original_hash}"
    masked_representation = f"masked-v1:{uid}:{masked_hash}"

    pool = tmp_path / "dedup-pool"
    _write_parquet(
        pool / "source" / "part.parquet",
        [
            {
                **_source_fields(),
                "stable_uid": uid,
                "text": original,
                "input_representation_id": input_representation,
                "representation_id": input_representation,
                "cleaned_text_sha256": original_hash,
            }
        ],
    )

    dedup_ledger = tmp_path / "dedup-ledger.parquet"
    _write_parquet(
        dedup_ledger,
        [
            {
                "stable_uid": uid,
                "input_representation_id": input_representation,
                "input_text_sha256": original_hash,
                "action": "keep",
            }
        ],
    )
    dedup_manifest = tmp_path / "dedup-manifest.json"
    _write_json(
        dedup_manifest,
        {
            "schema_version": "agent1_full_corpus_v3_dedup_ledger_manifest_v1",
            "status": "passed",
            "ledger": {"path": str(dedup_ledger.resolve()), "sha256": _sha_file(dedup_ledger)},
            "counts": {"ledger_rows": 1, "kept_rows": 1, "dropped_rows": 0},
        },
    )

    decontam_ledger = tmp_path / "decontam-ledger"
    decontam_file = decontam_ledger / "source" / "part.parquet"
    _write_parquet(
        decontam_file,
        [
            {
                "stable_uid": uid,
                "representation_id": input_representation,
                "input_text_sha256": original_hash,
                "action": "keep",
            }
        ],
    )
    decontam_manifest = tmp_path / "decontam-manifest.json"
    _write_json(
        decontam_manifest,
        {
            "schema_version": "agent1_full_corpus_v3_decontamination_manifest_v1",
            "status": "passed",
            "policy": {
                "high_confidence_actions": "drop",
                "ambiguous_match_actions": "quarantine",
                "answer_only_action": "audit_only",
            },
            "counts": {"input": 1, "keep": 1, "drop": 0, "quarantine": 0},
            "files": [{"ledger": _receipt(decontam_file, relative_to=decontam_ledger)}],
        },
    )

    protected_ledger = tmp_path / "protected-ledger"
    protected_file = protected_ledger / "source" / "part.parquet"
    _write_parquet(
        protected_file,
        [
            {
                "stable_uid": uid,
                "input_text_sha256": original_hash,
                "output_text_sha256": masked_hash,
                "parent_representation_id": input_representation,
                "child_representation_id": masked_representation,
                "action": "keep",
                "protected_spans_json": '[{"raw_value":"original@example.gr"}]',
            }
        ],
    )
    anonymized = tmp_path / "anonymized"
    anonymized_file = anonymized / "source" / "part.parquet"
    anonymized_row: dict[str, object] = {
        **_source_fields(),
        "stable_uid": uid,
        "text": masked,
        "input_representation_id": input_representation,
        "representation_id": masked_representation,
        "parent_representation_id": input_representation,
        "parent_text_sha256": original_hash,
        "text_sha256": masked_hash,
        "cleaned_text_sha256": masked_hash,
        "anonymization_action": "keep",
    }
    _write_parquet(anonymized_file, [anonymized_row])
    anonymization_manifest = tmp_path / "anonymization-manifest.json"
    _write_json(
        anonymization_manifest,
        {
            "schema_version": "agent1_full_corpus_v3_anonymization_manifest_v1",
            "status": "completed",
            "output": str(anonymized.resolve()),
            "protected_ledger": {
                "path": str(protected_ledger.resolve()),
                "public_training_output": False,
            },
            "counts": {
                "input_rows": 1,
                "action:keep": 1,
                "action:drop": 0,
                "action:quarantine": 0,
            },
            "files": [
                {
                    "output": _receipt(anonymized_file, relative_to=anonymized),
                    "protected_ledger": _receipt(protected_file, relative_to=protected_ledger),
                }
            ],
        },
    )

    structural_ledger: Path | None = None
    if applied_structural:
        final = tmp_path / "structural-output"
        final_file = final / "source" / "part.parquet"
        structural_text = "email <email-pii> structural"
        structural_hash = _sha_text(structural_text)
        structural_representation = f"structural-v1:{uid}:{structural_hash}"
        _write_parquet(
            final_file,
            [
                {
                    **anonymized_row,
                    "text": structural_text,
                    "representation_id": structural_representation,
                    "parent_representation_id": masked_representation,
                    "parent_text_sha256": masked_hash,
                    "text_sha256": structural_hash,
                    "cleaned_text_sha256": structural_hash,
                }
            ],
        )
        structural_ledger = tmp_path / "structural-ledger.parquet"
        _write_parquet(
            structural_ledger,
            [
                {
                    "stable_uid": uid,
                    "input_representation_id": input_representation,
                    "action": "keep",
                }
            ],
        )
        model_handoff = tmp_path / "model-handoff.json"
        _write_json(model_handoff, {"immutable": True})
        structural_manifest_value: dict[str, object] = {
            "schema_version": "agent1_full_corpus_v3_structural_apply_manifest_v1",
            "status": "passed",
            "mode": "applied",
            "input_root": str(anonymized.resolve()),
            "output_root": str(final.resolve()),
            "model_handoff": {
                "path": str(model_handoff.resolve()),
                "bytes": model_handoff.stat().st_size,
                "sha256": _sha_file(model_handoff),
            },
            "ready_for_application": True,
            "python_rust_probability_parity_passed": True,
            "python_rust_decoded_span_parity_passed": True,
            "source_balanced_safety_metrics_passed": True,
            "false_deletion_audit_passed": True,
        }
    else:
        final = anonymized
        structural_manifest_value = {
            "schema_version": "agent1_full_corpus_v3_structural_apply_manifest_v1",
            "status": "completed",
            "mode": "no_op",
            "input_root": str(anonymized.resolve()),
            "output_root": str(anonymized.resolve()),
            "no_op_reason": "Agent 2 immutable structural handoff is absent",
        }
    structural_manifest = tmp_path / "structural-manifest.json"
    _write_json(structural_manifest, structural_manifest_value)
    tokenizer = _tokenizer(tmp_path / "tokenizer.json")
    evidence = _release_evidence(tmp_path)
    return {
        "pool": pool,
        "dedup_ledger": dedup_ledger,
        "dedup_manifest": dedup_manifest,
        "decontam_ledger": decontam_ledger,
        "decontam_manifest": decontam_manifest,
        "protected_ledger": protected_ledger,
        "anonymized": anonymized,
        "anonymization_manifest": anonymization_manifest,
        "final": final,
        "structural_manifest": structural_manifest,
        "structural_ledger": structural_ledger,
        "tokenizer": tokenizer,
        **{f"evidence_{name}": path for name, path in evidence.items()},
    }


def _command(paths: dict[str, Path], release_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "materialize",
        "--dedup-pool",
        str(paths["pool"]),
        "--dedup-ledger",
        str(paths["dedup_ledger"]),
        "--dedup-manifest",
        str(paths["dedup_manifest"]),
        "--decontamination-ledger",
        str(paths["decontam_ledger"]),
        "--decontamination-manifest",
        str(paths["decontam_manifest"]),
        "--anonymized-root",
        str(paths["anonymized"]),
        "--anonymization-manifest",
        str(paths["anonymization_manifest"]),
        "--anonymization-protected-ledger",
        str(paths["protected_ledger"]),
        "--final-corpus-root",
        str(paths["final"]),
        "--structural-manifest",
        str(paths["structural_manifest"]),
        "--tokenizer-json",
        str(paths["tokenizer"]),
        "--transformation-waterfall",
        str(paths["evidence_transformation_waterfall"]),
        "--anonymization-semantic-clearance",
        str(paths["evidence_anonymization_semantic_clearance"]),
        "--work-database",
        str(release_root.parent / "release-validation.sqlite"),
        "--release-root",
        str(release_root),
        "--batch-rows",
        "1",
    ]
    for name in SITE_EVIDENCE_NAMES:
        command.extend(["--site-input", name, str(paths[f"evidence_{name}"])])
    return command


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def test_materializes_private_noop_release_and_compact_agent3_handoff(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    release_root = tmp_path / "release"
    command = _command(paths, release_root)
    result = _run(command)
    assert json.loads(result.stdout)["ok"] is True

    manifest = json.loads((release_root / "provenance" / "agent1_v3_release_manifest.json").read_text(encoding="utf-8"))
    handoff_path = release_root / "site_handoff" / "dataset_review_site_handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["publish_permitted"] is False
    assert manifest["ordered_transform_contract"]["dedup_before_greekmmlu"] is True
    assert manifest["final_data"]["rows"] == 1
    assert manifest["representation_lineage"]["max_depth"] == 1
    assert handoff["release_is_public_dataset"] is False
    assert "protected_pii_audit_ledgers" in handoff["content_exclusions"]
    assert "original@example.gr" not in handoff_path.read_text(encoding="utf-8")
    assert "protected_spans_json" not in handoff_path.read_text(encoding="utf-8")
    assert (release_root / "training" / "data" / "source" / "part.parquet").is_file()
    assert (release_root / "site_handoff" / "compact" / "quality_summary.json").is_file()
    assert handoff["required_evidence"]["status"] == "passed"
    assert handoff["required_evidence"]["anonymization_semantic_false_positive_clearance"]["status"] == "cleared"
    assert "not_included" not in handoff_path.read_text(encoding="utf-8")

    recheck = _run([sys.executable, str(SCRIPT), "validate", "--release-root", str(release_root)])
    assert json.loads(recheck.stdout)["ok"] is True


def test_applied_structural_child_representation_must_close_its_action_ledger(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, applied_structural=True)
    release_root = tmp_path / "release"
    command = _command(paths, release_root)
    assert paths["structural_ledger"] is not None
    command.extend(["--structural-ledger", str(paths["structural_ledger"])])
    _run(command)
    manifest = json.loads((release_root / "provenance" / "agent1_v3_release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ordered_transform_contract"]["structural_mode"] == "applied"
    assert manifest["representation_lineage"]["max_depth"] == 2


def test_refuses_dedup_ledger_that_is_not_bound_to_pre_mmlu_text(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = pq.read_table(paths["dedup_ledger"]).to_pylist()
    rows[0]["input_text_sha256"] = "0" * 64
    _write_parquet(paths["dedup_ledger"], rows)
    manifest = json.loads(paths["dedup_manifest"].read_text(encoding="utf-8"))
    manifest["ledger"]["sha256"] = _sha_file(paths["dedup_ledger"])
    _write_json(paths["dedup_manifest"], manifest)
    result = _run(_command(paths, tmp_path / "release"), check=False)
    assert result.returncode != 0
    assert "pre-GreekMMLU" in result.stderr


def test_refuses_private_ledger_fields_in_required_agent3_compact_input(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write_json(paths["evidence_review_aggregate"], {"protected_spans_json": "never hand this off"})
    command = _command(paths, tmp_path / "release")
    result = _run(command, check=False)
    assert result.returncode != 0
    assert "protected or benchmark-answer" in result.stderr


def test_refuses_local_release_when_any_mandatory_review_quality_admission_execution_evidence_is_missing(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    release_root = tmp_path / "release"
    command = _command(paths, release_root)
    marker = ["--site-input", "quality_summary", str(paths["evidence_quality_summary"])]
    start = next(index for index in range(len(command)) if command[index : index + 3] == marker)
    del command[start : start + 3]
    result = _run(command, check=False)
    assert result.returncode != 0
    assert "mandatory compact handoff evidence is missing: quality_summary" in result.stderr
    assert not release_root.exists()


def test_refuses_pending_or_not_included_semantic_clearance_and_quality_evidence(tmp_path: Path) -> None:
    pending_paths = _fixture(tmp_path / "pending")
    clearance = json.loads(pending_paths["evidence_anonymization_semantic_clearance"].read_text(encoding="utf-8"))
    clearance["status"] = "pending_independent_review"
    _write_json(pending_paths["evidence_anonymization_semantic_clearance"], clearance)
    pending = _run(_command(pending_paths, tmp_path / "pending-release"), check=False)
    assert pending.returncode != 0
    assert "semantic false-positive clearance: pending/not_included" in pending.stderr

    omitted_paths = _fixture(tmp_path / "omitted")
    quality = json.loads(omitted_paths["evidence_quality_summary"].read_text(encoding="utf-8"))
    quality["status"] = "not_included"
    _write_json(omitted_paths["evidence_quality_summary"], quality)
    omitted = _run(_command(omitted_paths, tmp_path / "omitted-release"), check=False)
    assert omitted.returncode != 0
    assert "full GlossAPI quality summary: pending/not_included" in omitted.stderr
