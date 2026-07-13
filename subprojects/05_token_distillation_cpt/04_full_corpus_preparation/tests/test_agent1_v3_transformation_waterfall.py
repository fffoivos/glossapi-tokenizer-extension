from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "agent1_v3_transformation_waterfall.py"


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(path: Path, root: Path | None = None) -> dict[str, object]:
    metadata = pq.ParquetFile(path).metadata
    return {
        "path": str(path.resolve() if root is None else path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha_file(path),
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
    }


def binding(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha_file(path)}


def write_parquet(path: Path, rows: list[dict[str, object]], *, columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    table = (
        pa.Table.from_pylist(normalized)
        if normalized
        else pa.Table.from_arrays([pa.array([], type=pa.string()) for _ in columns], names=columns)
    )
    pq.write_table(table, path, compression="zstd")
    return path


def write_json(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def make_tokenizer(path: Path) -> Path:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path))
    return path


def source_row(
    uid: str,
    source: str,
    route: str,
    text: str,
    *,
    dataset: str | None = None,
) -> dict[str, object]:
    digest = sha_text(text)
    return {
        "stable_uid": uid,
        "text": text,
        "cleaned_text_sha256": digest,
        "acquisition_source_id": source,
        "source_dataset": dataset or source,
        "source_route": route,
        "review_route": route,
        "extraction_route": route,
        "input_representation_id": f"rep:{uid}",
        "representation_id": f"rep:{uid}",
    }


def emitted_row(base: dict[str, object], text: str, action: str) -> dict[str, object]:
    result = dict(base)
    result["text"] = text
    result["cleaned_text_sha256"] = sha_text(text)
    result["anonymization_parent_text_sha256"] = sha_text(str(base["text"]))
    result["anonymization_output_text_sha256"] = sha_text(text)
    result["anonymization_parent_representation_id"] = base["input_representation_id"]
    result["anonymization_child_representation_id"] = f"masked:{base['stable_uid']}"
    result["anonymization_action"] = action
    return result


def make_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "run"
    roots = {
        "pool": root / "pool",
        "materialized": root / "materialized",
        "decontam_output": root / "decontam-output",
        "decontam_drop": root / "decontam-drop",
        "decontam_quarantine": root / "decontam-quarantine",
        "decontam_ledger": root / "decontam-ledger",
        "anonym_output": root / "anonym-output",
        "anonym_drop": root / "anonym-drop",
        "anonym_quarantine": root / "anonym-quarantine",
        "protected_ledger": root / "protected-ledger",
    }
    pool_rows = [
        source_row(
            "uid-keep",
            "amna_press",
            "html_web",
            "άρθρο alice@example.gr",
            dataset="glossAPI/amna-press",
        ),
        source_row("uid-dedup", "psepheda", "pdf_ocr", "διπλό έγγραφο"),
        source_row("uid-mmlu", "psepheda", "pdf_ocr", "ερώτηση GreekMMLU"),
        source_row("uid-private", "diavgeia", "mixed", "ιδιωτικό δελτίο"),
        source_row("uid-quarantine", "diavgeia", "mixed", "πίνακας a@x.gr b@x.gr c@x.gr"),
    ]
    by_uid = {str(row["stable_uid"]): row for row in pool_rows}
    pool_file = write_parquet(roots["pool"] / "part.parquet", pool_rows)
    materialized_rows = [by_uid[uid] for uid in ("uid-keep", "uid-mmlu", "uid-private", "uid-quarantine")]
    materialized_file = write_parquet(roots["materialized"] / "part.parquet", materialized_rows)

    dedup_ledger = root / "dedup-ledger.parquet"
    dedup_rows = []
    for row in pool_rows:
        uid = str(row["stable_uid"])
        action = "drop" if uid == "uid-dedup" else "keep"
        representative = "uid-keep" if uid == "uid-dedup" else uid
        dedup_rows.append(
            {
                "stable_uid": uid,
                "input_representation_id": row["input_representation_id"],
                "input_text_sha256": row["cleaned_text_sha256"],
                "action": action,
                "representative_stable_uid": representative,
                "representative_input_representation_id": f"rep:{representative}",
                "cluster_id": f"cluster:{representative}",
                "method": "exact_then_within_source_then_cross_candidate_then_candidate_to_nanochat_precedence_v1",
                "raw_decision_stage": "identity_closure",
                "reason": "representative" if action == "keep" else "nanochat_license_extraction_quality_provenance_stable_id",
            }
        )
    write_parquet(dedup_ledger, dedup_rows)

    decontam_actions = {"uid-keep": "keep", "uid-mmlu": "drop", "uid-private": "keep", "uid-quarantine": "keep"}
    decontam_reasons = {"uid-mmlu": "greekmmlu_exact_prompt"}
    decontam_ledger_rows = [
        {
            "stable_uid": uid,
            "representation_id": f"rep:{uid}",
            "input_text_sha256": by_uid[uid]["cleaned_text_sha256"],
            "action": action,
            "reason": decontam_reasons.get(uid, "no_high_confidence_match"),
            "benchmark_matches_json": "[]",
        }
        for uid, action in decontam_actions.items()
    ]
    decontam_ledger_file = write_parquet(roots["decontam_ledger"] / "part.parquet", decontam_ledger_rows)
    decontam_output_file = write_parquet(roots["decontam_output"] / "part.parquet", [by_uid[uid] for uid, action in decontam_actions.items() if action == "keep"])
    decontam_drop_file = write_parquet(roots["decontam_drop"] / "part.parquet", [by_uid["uid-mmlu"]])
    decontam_quarantine_file = write_parquet(roots["decontam_quarantine"] / "part.parquet", [], columns=list(pool_rows[0]))

    output_keep = emitted_row(by_uid["uid-keep"], "άρθρο <email-pii>", "keep")
    output_quarantine = emitted_row(by_uid["uid-quarantine"], "πίνακας <email-pii> <email-pii> <email-pii>", "quarantine")
    anonym_output_file = write_parquet(roots["anonym_output"] / "part.parquet", [output_keep])
    anonym_quarantine_file = write_parquet(roots["anonym_quarantine"] / "part.parquet", [output_quarantine])
    private = by_uid["uid-private"]
    anonym_drop_file = write_parquet(
        roots["anonym_drop"] / "part.parquet",
        [
            {
                "stable_uid": "uid-private",
                "acquisition_source_id": "diavgeia",
                "source_dataset": private["source_dataset"],
                "source_doc_id": "private",
                "anonymization_parent_text_sha256": private["cleaned_text_sha256"],
                "anonymization_parent_representation_id": "rep:uid-private",
                "anonymization_action": "drop",
                "anonymization_reasons_json": '["diavgeia_privateData_true"]',
            }
        ],
    )
    protected_rows = [
        {
            "stable_uid": "uid-keep",
            "acquisition_source_id": "amna_press",
            "source_dataset": by_uid["uid-keep"]["source_dataset"],
            "input_text_sha256": by_uid["uid-keep"]["cleaned_text_sha256"],
            "output_text_sha256": output_keep["cleaned_text_sha256"],
            "parent_representation_id": "rep:uid-keep",
            "child_representation_id": "masked:uid-keep",
            "action": "keep",
            "reasons_json": '["approved_high_precision_direct_identifier_masking"]',
            "pii_by_type_json": '{"email":1}',
            "span_count": 1,
            "protected_spans_json": '[{"raw_value":"alice@example.gr"}]',
            "ledger_schema_version": "agent1_full_corpus_v3_protected_anonymization_ledger_v1",
        },
        {
            "stable_uid": "uid-private",
            "acquisition_source_id": "diavgeia",
            "source_dataset": private["source_dataset"],
            "input_text_sha256": private["cleaned_text_sha256"],
            "output_text_sha256": None,
            "parent_representation_id": "rep:uid-private",
            "child_representation_id": None,
            "action": "drop",
            "reasons_json": '["diavgeia_privateData_true"]',
            "pii_by_type_json": "{}",
            "span_count": 0,
            "protected_spans_json": "[]",
            "ledger_schema_version": "agent1_full_corpus_v3_protected_anonymization_ledger_v1",
        },
        {
            "stable_uid": "uid-quarantine",
            "acquisition_source_id": "diavgeia",
            "source_dataset": by_uid["uid-quarantine"]["source_dataset"],
            "input_text_sha256": by_uid["uid-quarantine"]["cleaned_text_sha256"],
            "output_text_sha256": output_quarantine["cleaned_text_sha256"],
            "parent_representation_id": "rep:uid-quarantine",
            "child_representation_id": "masked:uid-quarantine",
            "action": "quarantine",
            "reasons_json": '["approved_high_precision_direct_identifier_masking","diavgeia_pii_heavy_personnel_table"]',
            "pii_by_type_json": '{"email":3}',
            "span_count": 3,
            "protected_spans_json": '[{"raw_value":"a@x.gr"}]',
            "ledger_schema_version": "agent1_full_corpus_v3_protected_anonymization_ledger_v1",
        },
    ]
    protected_file = write_parquet(roots["protected_ledger"] / "part.parquet", protected_rows)
    os.chmod(roots["protected_ledger"], 0o700)
    os.chmod(protected_file, 0o600)

    pool_manifest = write_json(
        root / "pool-manifest.json",
        {"schema_version": "agent1_full_corpus_v3_admitted_pool_manifest_v1", "status": "passed", "files": [receipt(pool_file, roots["pool"])]},
    )
    dedup_ledger_manifest = write_json(
        root / "dedup-ledger-manifest.json",
        {
            "schema_version": "agent1_full_corpus_v3_dedup_ledger_manifest_v1",
            "status": "passed",
            "pool": str(roots["pool"].resolve()),
            "ledger": receipt(dedup_ledger),
            "identity_reconciliation": {
                "selection_order": "exact_identity_then_within_source_then_cross_candidate_then_candidate_to_nanochat_before_representative_precedence",
                "exact_identity_precedes_near_passes": True,
            },
            "ordered_dedup": {
                "schema_version": "agent1_full_corpus_v3_ordered_dedup_composition_v1",
                "pass_order": [
                    "exact_content_work_representation",
                    "within_source_near",
                    "cross_candidate_near",
                    "candidate_to_nanochat_near",
                ],
                "exact_identity_precedes_near_passes": True,
            },
        },
    )
    materialized_manifest = write_json(
        root / "materialized-manifest.json",
        {
            "schema_version": "agent1_full_corpus_v3_dedup_materialization_manifest_v1",
            "status": "passed",
            "ledger": binding(dedup_ledger),
            "files": [receipt(materialized_file, roots["materialized"])],
        },
    )
    decontam_manifest = write_json(
        root / "decontam-manifest.json",
        {
            "schema_version": "agent1_full_corpus_v3_decontamination_manifest_v1",
            "status": "passed",
            "counts": {"input": 4, "keep": 3, "drop": 1},
            "files": [
                {
                    "input": receipt(materialized_file, roots["materialized"]),
                    "output": receipt(decontam_output_file, roots["decontam_output"]),
                    "dropped": receipt(decontam_drop_file, roots["decontam_drop"]),
                    "quarantine": receipt(decontam_quarantine_file, roots["decontam_quarantine"]),
                    "ledger": receipt(decontam_ledger_file, roots["decontam_ledger"]),
                }
            ],
        },
    )
    anonym_manifest = write_json(
        root / "anonym-manifest.json",
        {
            "schema_version": "agent1_full_corpus_v3_anonymization_manifest_v1",
            "status": "completed",
            "input": str(roots["decontam_output"].resolve()),
            "output": str(roots["anonym_output"].resolve()),
            "dropped": str(roots["anonym_drop"].resolve()),
            "quarantine": str(roots["anonym_quarantine"].resolve()),
            "policy": {"mask_types": ["email", "phone", "afm", "amka", "iban", "identity_or_passport", "ip"]},
            "protected_ledger": {
                "path": str(roots["protected_ledger"].resolve()),
                "contains_raw_span_values": True,
                "public_training_output": False,
                "directory_mode": "0700",
                "file_mode": "0600",
            },
            "transform_boundaries": {
                "generic_person_name_ner": False,
                "street_address_masking": False,
                "html_cleaning": False,
                "ocr_cleaning": False,
                "structural_cleaning": False,
                "stable_uid_preserved": True,
                "new_child_representation_ids": True,
            },
            "counts": {"input_rows": 3, "action:keep": 1, "action:drop": 1, "action:quarantine": 1, "protected_ledger_rows": 3},
            "files": [
                {
                    "input": receipt(decontam_output_file, roots["decontam_output"]),
                    "output": receipt(anonym_output_file, roots["anonym_output"]),
                    "dropped": receipt(anonym_drop_file, roots["anonym_drop"]),
                    "quarantine": receipt(anonym_quarantine_file, roots["anonym_quarantine"]),
                    "protected_ledger": receipt(protected_file, roots["protected_ledger"]),
                }
            ],
        },
    )
    closure = write_json(
        root / "ledger-closure.json",
        {
            "schema_version": "agent1_full_corpus_v3_protected_anonymization_ledger_closure_v1",
            "status": "passed",
            "anonymization_manifest": binding(anonym_manifest),
            "protected_ledger": {
                "path": str(roots["protected_ledger"].resolve()),
                "contains_raw_span_values": True,
                "public_training_output": False,
            },
        },
    )
    postmask = write_json(
        root / "postmask.json",
        {
            "schema_version": "agent1_full_corpus_v3_postmask_duplicate_verification_v1",
            "status": "passed",
            "anonymization_manifest": binding(anonym_manifest),
            "source_corpus_root": str(roots["anonym_output"].resolve()),
            "verification_only": True,
            "materialization_performed": False,
            "second_deduplication_applied": False,
            "material_new_duplicate_count": 0,
        },
    )
    prestructural = write_json(
        root / "prestructural.json",
        {
            "schema_version": "agent1_full_corpus_v3_prestructural_manifest_v1",
            "status": "prestructural_frozen",
            "publish_permitted": False,
            "structural_state": "awaiting_agent2_handoff",
            "corpus_root": str(roots["anonym_output"].resolve()),
            "inputs": {
                "dedup_manifest": binding(materialized_manifest),
                "decontamination_manifest": binding(decontam_manifest),
                "anonymization_manifest": binding(anonym_manifest),
                "anonymization_ledger": binding(closure),
                "postmask_duplicate_report": binding(postmask),
            },
        },
    )
    roster = write_json(
        root / "candidate-roster.json",
        {
            "schema_version": "agent1_full_corpus_v3_candidate_roster_v1",
            "base_source_id": "nanochat_base",
            "candidate_source_ids": ["amna_press", "diavgeia", "psepheda"],
            "review_routes": {
                "amna_press": "html_web",
                "diavgeia": "mixed",
                "psepheda": "pdf_ocr",
            },
            "source_routes": {
                "amna_press": "html_web",
                "diavgeia": "mixed",
                "psepheda": "pdf_ocr",
            },
            "extraction_routes": {
                "amna_press": "html_web",
                "diavgeia": "mixed",
                "psepheda": "pdf_ocr",
            },
            "route_policy": {"priority": "logical_source_then_observed_extraction"},
        },
    )
    roots.update(
        {
            "dedup_ledger": dedup_ledger,
            "pool_manifest": pool_manifest,
            "dedup_ledger_manifest": dedup_ledger_manifest,
            "materialized_manifest": materialized_manifest,
            "decontam_manifest": decontam_manifest,
            "anonym_manifest": anonym_manifest,
            "closure": closure,
            "postmask": postmask,
            "prestructural": prestructural,
            "roster": roster,
            "tokenizer": make_tokenizer(root / "tokenizer.json"),
            "work": root / "waterfall.sqlite",
            "waterfall": root / "waterfall.json",
        }
    )
    return roots


def command(paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--dedup-pool", str(paths["pool"]),
        "--dedup-pool-manifest", str(paths["pool_manifest"]),
        "--dedup-ledger", str(paths["dedup_ledger"]),
        "--dedup-ledger-manifest", str(paths["dedup_ledger_manifest"]),
        "--dedup-materialized", str(paths["materialized"]),
        "--dedup-materialization-manifest", str(paths["materialized_manifest"]),
        "--decontamination-input", str(paths["materialized"]),
        "--decontamination-output", str(paths["decontam_output"]),
        "--decontamination-dropped", str(paths["decontam_drop"]),
        "--decontamination-quarantine", str(paths["decontam_quarantine"]),
        "--decontamination-ledger", str(paths["decontam_ledger"]),
        "--decontamination-manifest", str(paths["decontam_manifest"]),
        "--anonymization-input", str(paths["decontam_output"]),
        "--anonymization-output", str(paths["anonym_output"]),
        "--anonymization-dropped", str(paths["anonym_drop"]),
        "--anonymization-quarantine", str(paths["anonym_quarantine"]),
        "--protected-ledger-root", str(paths["protected_ledger"]),
        "--anonymization-manifest", str(paths["anonym_manifest"]),
        "--anonymization-ledger-closure", str(paths["closure"]),
        "--postmask-duplicate-report", str(paths["postmask"]),
        "--prestructural-manifest", str(paths["prestructural"]),
        "--candidate-roster", str(paths["roster"]),
        "--tokenizer-json", str(paths["tokenizer"]),
        "--work-database", str(paths["work"]),
        "--output", str(paths["waterfall"]),
        "--batch-rows", "2",
        "--tokenizer-batch-docs", "2",
    ]


def rebind_anonymization_dependencies(paths: dict[str, Path]) -> None:
    """Model a malicious-but-receipted rewrite to test semantic closure."""

    manifest = json.loads(paths["anonym_manifest"].read_text(encoding="utf-8"))
    files = manifest["files"][0]
    files["output"] = receipt(paths["anonym_output"] / "part.parquet", paths["anonym_output"])
    files["dropped"] = receipt(paths["anonym_drop"] / "part.parquet", paths["anonym_drop"])
    files["quarantine"] = receipt(
        paths["anonym_quarantine"] / "part.parquet", paths["anonym_quarantine"]
    )
    files["protected_ledger"] = receipt(
        paths["protected_ledger"] / "part.parquet", paths["protected_ledger"]
    )
    write_json(paths["anonym_manifest"], manifest)

    closure = json.loads(paths["closure"].read_text(encoding="utf-8"))
    closure["anonymization_manifest"] = binding(paths["anonym_manifest"])
    write_json(paths["closure"], closure)

    postmask = json.loads(paths["postmask"].read_text(encoding="utf-8"))
    postmask["anonymization_manifest"] = binding(paths["anonym_manifest"])
    write_json(paths["postmask"], postmask)

    prestructural = json.loads(paths["prestructural"].read_text(encoding="utf-8"))
    prestructural["inputs"]["anonymization_manifest"] = binding(paths["anonym_manifest"])
    prestructural["inputs"]["anonymization_ledger"] = binding(paths["closure"])
    prestructural["inputs"]["postmask_duplicate_report"] = binding(paths["postmask"])
    write_json(paths["prestructural"], prestructural)


def rebind_dedup_dependencies(paths: dict[str, Path]) -> None:
    ledger_manifest = json.loads(paths["dedup_ledger_manifest"].read_text(encoding="utf-8"))
    ledger_manifest["ledger"] = receipt(paths["dedup_ledger"])
    write_json(paths["dedup_ledger_manifest"], ledger_manifest)

    materialized_manifest = json.loads(paths["materialized_manifest"].read_text(encoding="utf-8"))
    materialized_manifest["ledger"] = binding(paths["dedup_ledger"])
    write_json(paths["materialized_manifest"], materialized_manifest)

    prestructural = json.loads(paths["prestructural"].read_text(encoding="utf-8"))
    prestructural["inputs"]["dedup_manifest"] = binding(paths["materialized_manifest"])
    write_json(paths["prestructural"], prestructural)


def test_receipt_bound_waterfall_reports_routes_masking_deltas_and_pending_semantic_review(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    subprocess.run(command(paths), check=True, text=True, capture_output=True)
    payload = json.loads(paths["waterfall"].read_text(encoding="utf-8"))

    assert payload["schema_version"] == "agent1_full_corpus_v3_token_waterfall_v1"
    assert [row["stage"] for row in payload["stages"]] == [
        "50-dedup-input-pre-greekmmlu",
        "50-dedup-representatives",
        "60-greekmmlu-decontamination",
        "65-anonymization-sanitization",
        "70-prestructural-freeze",
    ]
    assert payload["stages"][0]["documents_after"] == 5
    assert payload["stages"][1]["documents_after"] == 4
    assert payload["stages"][2]["documents_after"] == 3
    assert payload["stages"][3]["documents_after"] == 1
    assert payload["stages"][4]["documents_after"] == 1
    source_routes = {row["acquisition_source_id"]: row["source_route"] for row in payload["source_stage_totals"]}
    assert source_routes == {"amna_press": "html_web", "diavgeia": "mixed", "psepheda": "pdf_ocr"}
    audit = payload["anonymization_audit"]
    assert audit["status"] == "automatic_checks_passed_semantic_review_pending"
    assert audit["false_positive_audit"]["automatic_policy_lineage_checks"]["status"] == "passed"
    assert audit["false_positive_audit"]["independent_semantic_review"]["status"] == "pending"
    assert audit["false_positive_audit"]["independent_semantic_review"]["eligible_rows"] == 3
    assert any(row["action"] == "approved_masking_transform" for row in payload["removal_events"])
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "alice@example.gr" not in serialized
    assert "a@x.gr" not in serialized
    assert "protected_spans_json" not in serialized
    assert payload["inventory_closure"]["pool_rows"] == 5
    assert payload["inventory_closure"]["anonymization_kept_rows"] == 1


def test_rejects_partition_mass_mismatch_even_when_the_ledger_receipt_is_refreshed(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    ledger = paths["decontam_ledger"] / "part.parquet"
    rows = pq.read_table(ledger).to_pylist()
    rows[0]["action"] = "drop"  # output partition still contains this row as a keep
    write_parquet(ledger, rows)
    manifest = json.loads(paths["decontam_manifest"].read_text(encoding="utf-8"))
    manifest["files"][0]["ledger"] = receipt(ledger, paths["decontam_ledger"])
    write_json(paths["decontam_manifest"], manifest)
    # The prestructural manifest is intentionally rebound to the updated
    # immutable-input candidate so failure occurs at partition closure rather
    # than merely at the outer binding check.
    prestructural = json.loads(paths["prestructural"].read_text(encoding="utf-8"))
    prestructural["inputs"]["decontamination_manifest"] = binding(paths["decontam_manifest"])
    write_json(paths["prestructural"], prestructural)

    completed = subprocess.run(command(paths), text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "GreekMMLU partition action drift" in completed.stderr


def test_rejects_receipted_candidate_route_drift(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    pool = paths["pool"] / "part.parquet"
    rows = pq.read_table(pool).to_pylist()
    rows[0]["source_route"] = "pdf_ocr"
    write_parquet(pool, rows)
    pool_manifest = json.loads(paths["pool_manifest"].read_text(encoding="utf-8"))
    pool_manifest["files"] = [receipt(pool, paths["pool"])]
    write_json(paths["pool_manifest"], pool_manifest)

    completed = subprocess.run(command(paths), text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "logical/extraction route differs from the frozen candidate roster" in completed.stderr


@pytest.mark.parametrize("partition", ["anonym_output", "anonym_drop"])
def test_rejects_receipted_anonymization_parent_hash_drift(
    tmp_path: Path, partition: str
) -> None:
    paths = make_fixture(tmp_path)
    path = paths[partition] / "part.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["anonymization_parent_text_sha256"] = sha_text("tampered parent")
    write_parquet(path, rows)
    rebind_anonymization_dependencies(paths)

    completed = subprocess.run(command(paths), text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "anonymization parent content hash drift" in completed.stderr


def test_rejects_receipted_anonymization_representation_drift(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    path = paths["anonym_output"] / "part.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["anonymization_child_representation_id"] = "tampered-child-representation"
    write_parquet(path, rows)
    rebind_anonymization_dependencies(paths)

    completed = subprocess.run(command(paths), text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "anonymization representation lineage drift" in completed.stderr


def test_rejects_receipted_greekmmlu_representation_drift(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    ledger = paths["decontam_ledger"] / "part.parquet"
    rows = pq.read_table(ledger).to_pylist()
    rows[0]["representation_id"] = "tampered-greekmmlu-representation"
    write_parquet(ledger, rows)
    manifest = json.loads(paths["decontam_manifest"].read_text(encoding="utf-8"))
    manifest["files"][0]["ledger"] = receipt(ledger, paths["decontam_ledger"])
    write_json(paths["decontam_manifest"], manifest)
    prestructural = json.loads(paths["prestructural"].read_text(encoding="utf-8"))
    prestructural["inputs"]["decontamination_manifest"] = binding(paths["decontam_manifest"])
    write_json(paths["prestructural"], prestructural)

    completed = subprocess.run(command(paths), text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "GreekMMLU content/representation drift" in completed.stderr


def test_rejects_receipted_dedup_cluster_without_a_representative_keep(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    ledger = paths["dedup_ledger"]
    rows = pq.read_table(ledger).to_pylist()
    for row in rows:
        if row["stable_uid"] == "uid-dedup":
            row["cluster_id"] = "cluster:orphan-drop"
    write_parquet(ledger, rows)
    rebind_dedup_dependencies(paths)

    completed = subprocess.run(command(paths), text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "dedup cluster representative closure drift" in completed.stderr


def test_rejects_receipted_dedup_manifest_when_exact_identity_is_not_first(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    manifest = json.loads(paths["dedup_ledger_manifest"].read_text(encoding="utf-8"))
    manifest["ordered_dedup"]["pass_order"] = [
        "within_source_near",
        "exact_content_work_representation",
        "cross_candidate_near",
        "candidate_to_nanochat_near",
    ]
    write_json(paths["dedup_ledger_manifest"], manifest)

    completed = subprocess.run(command(paths), text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "mandatory ordered pass composition" in completed.stderr
