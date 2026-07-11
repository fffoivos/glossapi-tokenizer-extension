from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
    scripts = str(path.parent)
    sys.path.insert(0, scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)
    return module


CONTRACT = load_module("phase04_stage_contract_structural", HERE / "clariden" / "stage_contract.py")
CLEAN = load_module("phase04_clean_structural_last", HERE / "scripts" / "apply_cleaning_policy.py")
FINAL = load_module(
    "phase04_finalize_structural_last", HERE / "scripts" / "finalize_structural_cleaning.py"
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage50_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "stage50.json"
    write_json(
        path,
        {
            "schema_version": "full_cpt_cleaning_manifest_v1",
            "status": "completed",
            "cleaning_pass": "post_source_post_pii",
            "structural_applied": False,
            "files": [{}],
        },
    )
    return path


def approved_policy(tmp_path: Path, *, approved: bool = True) -> Path:
    path = tmp_path / "policy.json"
    write_json(
        path,
        {
            "schema_version": "full_cpt_cleaning_policy_v1",
            "status": "approved" if approved else "audit_only",
            "validation": {
                "structural_application_receipt_required": True,
                "required_model_evidence": "LLM_silver",
                "required_safety_evidence": "targeted_manual_false_deletion_audit",
            },
            "structural": {
                "toc": {"enabled_for_materialization": approved},
                "bibliography": {"enabled_for_materialization": approved},
                "application_gates": {
                    "minimum_reviewed_deletions": 10,
                    "maximum_running_prose_deletion_rate": 0.001,
                    "minimum_main_text_retention_rate": 0.999,
                    "maximum_catastrophic_document_deletion_rate": 0.0,
                },
            },
        },
    )
    return path


def model_receipt(stage50: Path, *, metrics: dict | None, claims_pass: bool) -> dict:
    zero = "0" * 64
    return {
        "schema_version": "academic_structural_model_receipt_v1",
        "status": "passed" if claims_pass else "no_op",
        "promotion_status": "passed" if claims_pass else "no_op",
        "model_id": "fixture-c2",
        "stage50_cleaning_manifest_sha256": sha(stage50),
        "artifacts": {"code": zero, "config": zero, "checkpoint": zero},
        "evidence": {
            "annotation_status": "LLM_silver",
            "inventory_sha256": zero,
            "task_coverage": ["toc", "bibliography"],
            "work_split": {
                "leak_free": True,
                "work_overlap_count": 0,
                "exact_text_overlap_count": 0,
                "split_manifest_sha256": zero,
            },
        },
        "safety": {
            "status": "passed" if metrics is not None else "unavailable",
            "evidence_status": (
                "targeted_manual_false_deletion_audit" if metrics is not None else "unavailable"
            ),
            "reviewed_deletions": 10 if metrics is not None else 0,
            "audit_receipt_sha256": "0" * 64 if metrics is not None else None,
            "metrics": metrics
            or {
                "running_prose_deletion_rate": None,
                "main_text_retention_rate": None,
                "catastrophic_document_deletion_rate": None,
            },
        },
    }


def validate_receipt(
    tmp_path: Path, receipt: dict, *, approved: bool = True
) -> dict:
    stage50 = tmp_path / "stage50.json"
    receipt_path = tmp_path / "model.json"
    output = tmp_path / "decision.json"
    output.unlink(missing_ok=True)
    write_json(receipt_path, receipt)
    CONTRACT.cmd_validate_structural_model(
        argparse.Namespace(
            receipt=receipt_path,
            requested_mode="apply",
            stage50_cleaning_manifest=stage50,
            cleaning_policy=approved_policy(tmp_path, approved=approved),
            output=output,
        )
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_requested_apply_fails_when_safety_is_unavailable_or_dishonest(
    tmp_path: Path,
) -> None:
    stage50 = stage50_manifest(tmp_path)
    with pytest.raises(
        ValueError, match="requested structural apply is not eligible: safety_metrics_unavailable"
    ):
        validate_receipt(tmp_path, model_receipt(stage50, metrics=None, claims_pass=False))

    dishonest = model_receipt(stage50, metrics=None, claims_pass=True)
    with pytest.raises(ValueError, match="silver-only unavailable"):
        validate_receipt(tmp_path, dishonest)


def test_targeted_manual_safety_can_promote_silver_model(tmp_path: Path) -> None:
    stage50 = stage50_manifest(tmp_path)
    receipt = model_receipt(
        stage50,
        metrics={
            "running_prose_deletion_rate": 0.0,
            "main_text_retention_rate": 1.0,
            "catastrophic_document_deletion_rate": 0.0,
        },
        claims_pass=True,
    )
    decision = validate_receipt(tmp_path, receipt)
    assert decision["status"] == "passed"
    assert decision["apply_structural"] is True
    assert decision["requested_mode"] == "apply"
    assert decision["apply_structural_requested"] is True
    assert decision["model_selection_evidence"] == "LLM_silver"


def test_explicit_noop_and_resume_mode_or_receipt_drift_fail_closed(
    tmp_path: Path,
) -> None:
    stage50 = stage50_manifest(tmp_path)
    policy = approved_policy(tmp_path)
    receipt_path = tmp_path / "model.json"
    output = tmp_path / "decision.json"
    promoted = model_receipt(
        stage50,
        metrics={
            "running_prose_deletion_rate": 0.0,
            "main_text_retention_rate": 1.0,
            "catastrophic_document_deletion_rate": 0.0,
        },
        claims_pass=True,
    )
    write_json(receipt_path, promoted)
    apply_args = argparse.Namespace(
        receipt=receipt_path,
        requested_mode="apply",
        stage50_cleaning_manifest=stage50,
        cleaning_policy=policy,
        output=output,
    )
    CONTRACT.cmd_validate_structural_model(apply_args)
    CONTRACT.cmd_validate_structural_model(apply_args)

    with pytest.raises(ValueError, match="application decision drift on resume"):
        CONTRACT.cmd_validate_structural_model(
            argparse.Namespace(
                receipt=None,
                requested_mode="no_op",
                stage50_cleaning_manifest=stage50,
                cleaning_policy=policy,
                output=output,
            )
        )

    promoted["model_id"] = "different-promoted-model"
    write_json(receipt_path, promoted)
    with pytest.raises(ValueError, match="application decision drift on resume"):
        CONTRACT.cmd_validate_structural_model(apply_args)

    noop_output = tmp_path / "noop-decision.json"
    CONTRACT.cmd_validate_structural_model(
        argparse.Namespace(
            receipt=None,
            requested_mode="no_op",
            stage50_cleaning_manifest=stage50,
            cleaning_policy=policy,
            output=noop_output,
        )
    )
    no_op = json.loads(noop_output.read_text(encoding="utf-8"))
    assert no_op["requested_mode"] == "no_op"
    assert no_op["apply_structural_requested"] is False
    assert no_op["apply_structural"] is False
    assert no_op["reason"] == "operator_selected_no_op"


def test_structural_finalization_request_is_immutable_on_resume(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    noop_args = argparse.Namespace(requested_mode="no_op", receipt=None, output=request)
    CONTRACT.cmd_freeze_structural_request(noop_args)
    CONTRACT.cmd_freeze_structural_request(noop_args)
    receipt = tmp_path / "receipt.json"
    write_json(receipt, {"fixture": True})
    with pytest.raises(ValueError, match="finalization request drift on resume"):
        CONTRACT.cmd_freeze_structural_request(
            argparse.Namespace(requested_mode="apply", receipt=receipt, output=request)
        )


def test_span_index_rejects_overlap_and_wrong_receipt_binding(tmp_path: Path) -> None:
    text_hash = "1" * 64
    model_hash = "2" * 64
    cleaning_hash = "3" * 64
    uid = "4" * 64
    path = tmp_path / "spans.jsonl"
    rows = [
        {
            "stable_uid": uid,
            "input_text_sha256": text_hash,
            "model_receipt_sha256": model_hash,
            "stage50_cleaning_manifest_sha256": cleaning_hash,
            "kind": "toc",
            "char_start": 0,
            "char_end": 5,
            "rule_id": "toc-v1",
        },
        {
            "stable_uid": uid,
            "input_text_sha256": text_hash,
            "model_receipt_sha256": model_hash,
            "stage50_cleaning_manifest_sha256": cleaning_hash,
            "kind": "bibliography",
            "char_start": 4,
            "char_end": 9,
            "rule_id": "bib-v1",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        FINAL._build_span_index(
            [path],
            database=tmp_path / "spans.sqlite",
            model_receipt_sha256=model_hash,
            stage50_manifest_sha256=cleaning_hash,
        )
    rows[1]["char_start"] = 5
    rows[1]["model_receipt_sha256"] = "9" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="model receipt binding"):
        FINAL._build_span_index(
            [path],
            database=tmp_path / "spans.sqlite",
            model_receipt_sha256=model_hash,
            stage50_manifest_sha256=cleaning_hash,
        )


def test_document_actions_are_content_bound_and_conflicts_fail(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    row = {
        "stable_uid": "a" * 64,
        "input_text_sha256": "b" * 64,
        "action": "drop",
        "reason": "base_represented",
    }
    conflict = {**row, "action": "quarantine"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(conflict) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        CLEAN.load_document_actions([path])

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet = tmp_path / "actions.parquet"
    pq.write_table(pa.Table.from_pylist([row, row]), parquet)
    index = CLEAN.build_document_action_index([parquet], tmp_path / "actions.sqlite")
    assert index["input_rows"] == 2
    assert index["distinct_actions"] == 1
    assert index["exact_duplicate_rows"] == 1


def test_completed_inventory_revalidates_attested_shard(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    ledger = tmp_path / "ledger"
    quarantine = tmp_path / "quarantine"
    for directory in (root, ledger, quarantine):
        directory.mkdir()
    shard = root / "part.parquet"
    ledger_shard = ledger / "part.parquet"
    quarantine_shard = quarantine / "part.parquet"
    shard.write_bytes(b"corpus")
    ledger_shard.write_bytes(b"ledger")
    quarantine_shard.write_bytes(b"quarantine")
    manifest = {
        "schema_version": "full_cpt_cleaning_manifest_v1",
        "output": str(root),
        "ledger": str(ledger),
        "quarantine": str(quarantine),
        "files": [
            {
                "output": {"path": "part.parquet", "bytes": 6, "sha256": sha(shard)},
                "ledger": {"path": "part.parquet", "bytes": 6, "sha256": sha(ledger_shard)},
                "quarantine": {
                    "path": "part.parquet",
                    "bytes": 10,
                    "sha256": sha(quarantine_shard),
                },
            }
        ],
    }
    validated, _ = CONTRACT.validate_manifest_inventory(manifest, path=tmp_path / "manifest.json")
    assert len(validated) == 3
    stage = tmp_path / "stage"
    stage.mkdir()
    manifest_path = stage / "manifest.json"
    write_json(manifest_path, manifest)
    receipt_path = stage / "stage_receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": "full_cpt_pipeline_stage_receipt_v1",
            "status": "passed",
            "stage": "50-clean",
            "run_id": "fixture",
            "code_commit": "f" * 40,
            "inputs": {},
            "outputs": [
                {
                    "path": "manifest.json",
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha(manifest_path),
                }
            ],
        },
    )
    (stage / "COMPLETED").write_text(sha(receipt_path) + "  stage_receipt.json\n")
    stage_args = argparse.Namespace(
        stage_dir=stage,
        stage="50-clean",
        run_id="fixture",
        code_commit="f" * 40,
    )
    CONTRACT.cmd_validate_stage(stage_args)
    shard.write_bytes(b"drifted")
    with pytest.raises(ValueError, match="inventory verification failed"):
        CONTRACT.validate_manifest_inventory(manifest, path=tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="attested shard inventory failed"):
        CONTRACT.cmd_validate_stage(stage_args)


def test_two_pass_cli_reuses_stage50_text_for_structural_noop(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizers = pytest.importorskip("tokenizers")
    text = "Καθαρό ακαδημαϊκό κείμενο"
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    row = {
        "source_id": "demo",
        "source_dataset": "demo",
        "source_doc_id": "doc",
        "text": text,
        "title": None,
        "author": None,
        "greek_badness_score": 0.0,
        "mojibake_badness_score": 0.0,
        "needs_ocr": False,
        "is_empty": False,
        "ocr_success": True,
        "is_historical_or_polytonic": False,
        "source_family_id": "demo",
        "acquisition_source_id": "eellak_articles",
        "source_repo_id": "glossAPI/eellak-articles",
        "source_revision": "59fd681c483e6bdcdabe7c1a1f8685c5eebf7883",
        "source_artifact_path": "part.parquet",
        "source_row_id": "part.parquet:0:0",
        "source_text_field": "text",
        "original_text_sha256": text_hash,
        "normalized_text_sha256": text_hash,
        "stable_uid": "2" * 64,
        "work_key": "3" * 64,
        "work_id": "doc",
        "representation_generation": "new_family",
        "lineage_alias_id": "4" * 64,
        "source_metadata_json": "{}",
        "cleaning_profile": "academic_sectioned",
        "structural_policy": "shadow",
        "training_eligibility": "eligible_open",
        "source_role": "additive_candidate",
    }
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pq.write_table(pa.Table.from_pylist([row], schema=CLEAN.canonical_schema()), normalized / "part.parquet")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    admission = tmp_path / "admission.json"
    write_json(
        admission,
        {
            "schema_version": "source_quality_review_admission_v1",
            "pending_adjudications": 0,
            "sources": [{"source_dataset": "demo", "decision": "include"}],
        },
    )
    policy = HERE / "configs" / "cleaning_policy.json"
    stage50 = tmp_path / "stage50"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "apply_cleaning_policy.py"),
            "--input",
            str(normalized),
            "--output",
            str(stage50 / "corpus"),
            "--quarantine",
            str(stage50 / "quarantine"),
            "--ledger",
            str(stage50 / "ledger"),
            "--manifest",
            str(stage50 / "cleaning_manifest.json"),
            "--source-admission",
            str(admission),
            "--cleaning-policy",
            str(policy),
            "--tokenizer-json",
            str(tokenizer_path),
            "--workers",
            "1",
        ],
        check=True,
    )
    stage50_manifest = stage50 / "cleaning_manifest.json"
    decision = tmp_path / "decision.json"
    write_json(
        decision,
        {
            "schema_version": "full_cpt_structural_application_decision_v1",
            "requested_mode": "no_op",
            "apply_structural_requested": False,
            "status": "no_op",
            "apply_structural": False,
            "reason": "fixture_noop",
            "stage50_cleaning_manifest_sha256": sha(stage50_manifest),
            "cleaning_policy_sha256": sha(policy),
        },
    )
    final = tmp_path / "final"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "finalize_structural_cleaning.py"),
            "--input",
            str(stage50 / "corpus"),
            "--input-ledger",
            str(stage50 / "ledger"),
            "--input-quarantine",
            str(stage50 / "quarantine"),
            "--input-cleaning-manifest",
            str(stage50_manifest),
            "--output",
            str(final / "corpus"),
            "--ledger",
            str(final / "ledger"),
            "--quarantine",
            str(final / "quarantine"),
            "--manifest",
            str(final / "cleaning_manifest.json"),
            "--source-admission",
            str(admission),
            "--source-config",
            str(HERE / "configs" / "sources.json"),
            "--license-adjudication",
            str(HERE / "configs" / "source_license_adjudication.json"),
            "--eligibility-policy",
            str(HERE / "configs" / "training_eligibility_policy.json"),
            "--cleaning-policy",
            str(policy),
            "--tokenizer-json",
            str(tokenizer_path),
            "--structural-decision",
            str(decision),
            "--work-dir",
            str(final / "work"),
            "--workers",
            "1",
        ],
        check=True,
    )
    final_row = pq.read_table(final / "corpus" / "part.parquet").to_pylist()[0]
    final_ledger = pq.read_table(final / "ledger" / "part.parquet").to_pylist()[0]
    manifest = json.loads((final / "cleaning_manifest.json").read_text(encoding="utf-8"))
    assert final_row["text"] == text
    assert final_ledger["tokens_toc_removed"] == 0
    assert final_ledger["tokens_bibliography_removed"] == 0
    assert final_ledger["tokens_structural_union_removed"] == 0
    assert final_ledger["final_text_sha256"] == text_hash
    assert manifest["cleaning_pass"] == "structural_last_final"
    assert manifest["structural_semantics"] == "deterministic_no_op"


def test_finalizer_rejects_eligibility_policy_drift_from_stage50(
    tmp_path: Path,
) -> None:
    paths: dict[str, Path] = {}
    for name in (
        "tokenizer",
        "source_config",
        "license_adjudication",
        "eligibility_policy",
        "cleaning_policy",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
        paths[name] = path
    manifest = {
        "tokenizer_sha256": sha(paths["tokenizer"]),
        "source_config_sha256": sha(paths["source_config"]),
        "license_adjudication_sha256": sha(paths["license_adjudication"]),
        "eligibility_policy_sha256": sha(paths["eligibility_policy"]),
        "cleaning_policy_sha256": sha(paths["cleaning_policy"]),
    }
    FINAL._validate_stage50_replay_inputs(
        manifest,
        tokenizer=paths["tokenizer"],
        source_config=paths["source_config"],
        license_adjudication=paths["license_adjudication"],
        eligibility_policy=paths["eligibility_policy"],
        cleaning_policy=paths["cleaning_policy"],
    )
    drifted = tmp_path / "eligibility-policy-drifted.json"
    drifted.write_text('{"name":"changed-policy"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="eligibility policy differs from reviewed Stage 50"):
        FINAL._validate_stage50_replay_inputs(
            manifest,
            tokenizer=paths["tokenizer"],
            source_config=paths["source_config"],
            license_adjudication=paths["license_adjudication"],
            eligibility_policy=drifted,
            cleaning_policy=paths["cleaning_policy"],
        )


def test_stage58_writes_immutable_reviewed_input_binding(tmp_path: Path) -> None:
    names = (
        "tokenizer",
        "eligibility_policy",
        "source_config",
        "source_license_adjudication",
        "cleaning_policy",
    )
    inputs: dict[str, dict[str, object]] = {}
    for name in names:
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        inputs[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        }
    reference = tmp_path / "stage50-receipt.json"
    write_json(
        reference,
        {
            "schema_version": "full_cpt_pipeline_stage_receipt_v1",
            "status": "passed",
            "inputs": inputs,
        },
    )
    current = tmp_path / "stage58-inputs.json"
    write_json(
        current,
        {
            "schema_version": "full_cpt_pipeline_stage_inputs_v1",
            "inputs": {
                **inputs,
                "structural_model_receipt": {
                    "path": str(reference.resolve()),
                    "bytes": reference.stat().st_size,
                    "sha256": sha(reference),
                },
            },
        },
    )
    output = tmp_path / "replay-validation.json"
    arguments = argparse.Namespace(
        reference_receipt=reference,
        current_inputs=current,
        finalizer=True,
        output=output,
    )
    CONTRACT.cmd_validate_cleaning_replay(arguments)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["mode"] == "structural_last_finalizer"
    assert set(receipt["reviewed_inputs"]) == set(names)
    CONTRACT.cmd_validate_cleaning_replay(arguments)

    changed = tmp_path / "changed-eligibility"
    changed.write_text("changed", encoding="utf-8")
    drifted = json.loads(current.read_text(encoding="utf-8"))
    drifted["inputs"]["eligibility_policy"] = {
        "path": str(changed.resolve()),
        "bytes": changed.stat().st_size,
        "sha256": sha(changed),
    }
    write_json(current, drifted)
    with pytest.raises(ValueError, match="differ from the reviewed cleaning pass"):
        CONTRACT.cmd_validate_cleaning_replay(arguments)
