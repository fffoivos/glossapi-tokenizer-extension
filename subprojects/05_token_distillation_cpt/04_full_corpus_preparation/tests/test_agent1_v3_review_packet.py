from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "scripts" / "agent1_v3_review_packet.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agent1_v3_review_packet_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PACKET = load_module()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_uid(index: int) -> str:
    return digest(f"source-a:{index}")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_fixture(
    tmp_path: Path,
    *,
    count: int,
    opaque_quality_ids: bool,
    source_route: str = "pdf_ocr",
    extraction_route: str | None = None,
) -> dict[str, object]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    canonical_root = tmp_path / "canonical"
    shard = canonical_root / "source-a" / "part-000.parquet"
    shard.parent.mkdir(parents=True)
    texts = {
        stable_uid(index): (
            f"Έγγραφο {index:03d}: γράψε στο person{index}@example.gr για πληροφορίες.\n"
            "Η δεύτερη γραμμή είναι συνεκτικό ελληνικό κείμενο."
        )
        for index in range(count)
    }
    pq.write_table(
        pa.table(
            {
                "stable_uid": list(texts),
                "text": list(texts.values()),
                "source_id": ["source-a"] * count,
                "source_dataset": ["source-a"] * count,
                "source_revision": ["a" * 40] * count,
                "normalized_text_sha256": [digest(text) for text in texts.values()],
                # This is deliberately not part of the packet reader's column
                # set: raw upstream document IDs must never reach reviewers.
                "source_doc_id": [
                    f"s3://restricted/private/opaque-source-record-{index}.pdf"
                    for index in range(count)
                ],
            }
        ),
        shard,
    )
    shard_sha256 = PACKET.sha256_file(shard)
    evidence_rows: list[dict[str, object]] = []
    for index, (uid, text) in enumerate(texts.items()):
        row: dict[str, object] = {
            "source_id": "source-a",
            "source_dataset": "source-a",
            "source_revision": "a" * 40,
            "review_route": source_route,
            "normalized_text_sha256": digest(text),
            "profile_text_variant": "canonical",
            "review_risk_score": float(index),
            "structural_template_id": f"template-{index % 9}",
            "input_shard_path": "source-a/part-000.parquet",
            "input_shard_sha256": shard_sha256,
            "input_row_index": index,
        }
        if opaque_quality_ids:
            row.update(
                {
                    "schema_version": "dataset_quality_document_v1",
                    "document_id": PACKET.quality_document_id(uid),
                }
            )
        else:
            row["stable_uid"] = uid
        evidence_rows.append(row)
    evidence = tmp_path / "full-scan.jsonl"
    evidence.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in evidence_rows),
        encoding="utf-8",
    )

    roster = tmp_path / "roster.json"
    write_json(
        roster,
        {
            "schema_version": "agent1_full_corpus_v3_candidate_roster_v1",
            "candidate_source_ids": ["source-a"],
            "review_routes": {"source-a": source_route},
            "source_routes": {"source-a": source_route},
            "extraction_routes": {"source-a": extraction_route or source_route},
            "route_policy": {"priority": "logical_source_then_observed_extraction"},
            "inventory_only_exclusions": [],
        },
    )
    policy = tmp_path / "policy.json"
    write_json(
        policy,
        {
            "schema_version": "agent1_full_corpus_v3_policy_v1",
            "review": {
                "seed": "frozen-seed",
                "model_environment_variable": "CODEX_REVIEW_MODEL",
                "required_model": "gpt-5.6-luna",
                "reasoning_effort": "low",
                "minimum_documents_per_eligible_source": 100,
                "sample_strata": {"random": 60, "risk": 20, "cluster": 20},
                "large_or_heterogeneous_total": 200,
                "large_or_heterogeneous_source_ids": [],
                "double_review_fraction": 0.1,
                "review_copy": "mask_high_confidence_direct_identifiers_preserve_position_and_original_hash",
                "no_model_fallback": True,
            },
        },
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Rate cleanliness, quality, and diversity.", encoding="utf-8")
    response_schema = tmp_path / "response-schema.json"
    write_json(
        response_schema,
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"const": "agent1_v3_review_response_v1"},
            },
        },
    )
    return {
        "canonical_root": canonical_root,
        "evidence": evidence,
        "roster": roster,
        "policy": policy,
        "prompt": prompt,
        "response_schema": response_schema,
        "texts": texts,
    }


def command(fixture: dict[str, object], *, output: Path, manifest: Path, **overrides: str) -> list[str]:
    values = {
        "seed": "frozen-seed",
        "model": "gpt-5.6-luna",
        "code_commit": "b" * 40,
        **overrides,
    }
    return [
        sys.executable,
        str(SCRIPT),
        "--full-scan-evidence",
        str(fixture["evidence"]),
        "--canonical-root",
        str(fixture["canonical_root"]),
        "--roster",
        str(fixture["roster"]),
        "--policy",
        str(fixture["policy"]),
        "--prompt",
        str(fixture["prompt"]),
        "--response-schema",
        str(fixture["response_schema"]),
        "--seed",
        values["seed"],
        "--model",
        values["model"],
        "--code-commit",
        values["code_commit"],
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    ]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_opaque_full_scan_evidence_materializes_private_100_document_packet(tmp_path: Path) -> None:
    fixture = write_fixture(tmp_path, count=140, opaque_quality_ids=True)
    output = tmp_path / "review-requests.jsonl"
    manifest_path = tmp_path / "review-packet-manifest.json"
    completed = subprocess.run(
        command(fixture, output=output, manifest=manifest_path),
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(completed.stdout)["ok"] is True
    requests = read_jsonl(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary = [request for request in requests if request["reviewer_slot"] == "primary"]
    secondary = [request for request in requests if request["reviewer_slot"] == "secondary"]

    assert len(requests) == 110
    assert Counter(request["sampling_stratum"] for request in primary) == {
        "random": 60,
        "risk": 20,
        "cluster": 20,
    }
    assert Counter(request["sampling_stratum"] for request in secondary) == {
        "random": 6,
        "risk": 2,
        "cluster": 2,
    }
    texts = fixture["texts"]
    assert isinstance(texts, dict)
    for request in requests:
        uid = request["sample_id"]
        assert isinstance(uid, str) and uid in texts
        assert request["model"] == "gpt-5.6-luna"
        assert request["code_commit"] == "b" * 40
        assert PACKET.review._validate_request_binding(request) == []
        assert len(request["review_copy"]) == len(texts[uid])
        assert digest(request["review_copy"]) == request["review_copy_sha256"]
        assert "@example.gr" not in request["review_copy"]
        assert len(request["comparison_bundle"]) == 4
        for comparison in request["comparison_bundle"]:
            assert comparison["sample_id"] != uid
            assert "@example.gr" not in comparison["review_copy"]

    source = manifest["source_review_coverage"][0]
    assert {
        key: source[key]
        for key in ("source_route", "review_route", "extraction_route")
    } == {
        "source_route": "pdf_ocr",
        "review_route": "pdf_ocr",
        "extraction_route": "pdf_ocr",
    }
    assert source["review_denominator"]["eligible_document_count"] == 140
    assert source["review_denominator"]["minimum_requirement_status"] == "met"
    assert source["review_denominator"]["selected_unique_documents"] == 100
    assert source["requested_strata"] == {"random": 60, "risk": 20, "cluster": 20}
    assert source["primary_requests_by_stratum"] == {"random": 60, "risk": 20, "cluster": 20}
    assert source["secondary_requests_by_stratum"] == {"random": 6, "risk": 2, "cluster": 2}
    assert source["eligible_inventory_key_kind"] == "dataset_quality_document_id"
    assert source["selected_evidence_inventory_key_kind"] == "dataset_quality_document_id"
    assert source["selected_inventory_key_kind"] == "canonical_stable_uid"
    assert manifest["review_execution"] == {
        "model_environment_variable": "CODEX_REVIEW_MODEL",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "no_model_fallback": True,
        "model_invocation": "not_run",
        "code_commit": "b" * 40,
        "selection_seed": "frozen-seed",
        "secondary_selection_seed": PACKET.sha256_text(
            "frozen-seed\0agent1-v3-secondary-review-v1"
        ),
        "secondary_fraction": 0.1,
        "prompt_sha256": manifest["inputs"]["prompt"]["sha256"],
        "response_schema_sha256": manifest["inputs"]["response_schema"]["sha256"],
    }
    assert manifest["selection"]["selection_evidence_key_counts"] == {
        "dataset_quality_document_id": 100
    }
    for selected in manifest["selection"]["selected_documents"]:
        assert selected["stable_uid"] in texts
        assert selected["full_scan_selection_evidence_key_kind"] == "dataset_quality_document_id"
        assert selected["full_scan_document_id"] == PACKET.quality_document_id(selected["stable_uid"])
    assert manifest["review_copy_redaction_totals"] == {"email": 100}
    assert manifest["manifest_sha256"] == PACKET._manifest_sha256(manifest)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    serialized = output.read_text(encoding="utf-8") + manifest_path.read_text(encoding="utf-8")
    assert "@example.gr" not in serialized
    assert "opaque-source-record-" not in serialized

    repeat = subprocess.run(
        command(fixture, output=output, manifest=manifest_path), text=True, capture_output=True
    )
    assert repeat.returncode != 0
    assert "immutable packet output already exists" in repeat.stderr


def test_small_source_is_exhaustive_and_native_stable_uid_evidence_is_preserved(tmp_path: Path) -> None:
    fixture = write_fixture(tmp_path, count=17, opaque_quality_ids=False)
    output = tmp_path / "review-requests.jsonl"
    manifest_path = tmp_path / "review-packet-manifest.json"
    subprocess.run(command(fixture, output=output, manifest=manifest_path), check=True, text=True)
    requests = read_jsonl(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary = [request for request in requests if request["reviewer_slot"] == "primary"]
    secondary = [request for request in requests if request["reviewer_slot"] == "secondary"]

    assert Counter(request["sampling_stratum"] for request in primary) == {
        "random": 10,
        "risk": 4,
        "cluster": 3,
    }
    assert Counter(request["sampling_stratum"] for request in secondary) == {
        "random": 1,
        "risk": 1,
        "cluster": 1,
    }
    source = manifest["source_review_coverage"][0]
    assert source["review_denominator"] == {
        "eligible_document_count": 17,
        "minimum_required_documents": 100,
        "configured_review_target": 100,
        "selected_unique_documents": 17,
        "selection_is_exhaustive": True,
        "minimum_requirement_status": "unattainable_exhaustive",
        "target_status": "unattainable_exhaustive",
        "denominator_exception": "eligible_inventory_below_100_all_documents_selected",
    }
    assert source["eligible_inventory_key_kind"] == "canonical_stable_uid"
    assert source["selected_evidence_inventory_key_kind"] == "canonical_stable_uid"
    assert all(
        selected["full_scan_selection_evidence_key"] == selected["stable_uid"]
        and selected["full_scan_selection_evidence_key_kind"] == "canonical_stable_uid"
        and "full_scan_document_id" not in selected
        for selected in manifest["selection"]["selected_documents"]
    )


def test_packet_keeps_observed_extraction_route_as_a_receipt_not_the_primary_review_route(
    tmp_path: Path,
) -> None:
    fixture = write_fixture(
        tmp_path,
        count=17,
        opaque_quality_ids=False,
        source_route="pdf_ocr",
        extraction_route="html_web",
    )
    output = tmp_path / "review-requests.jsonl"
    manifest_path = tmp_path / "review-packet-manifest.json"
    subprocess.run(command(fixture, output=output, manifest=manifest_path), check=True, text=True)
    requests = read_jsonl(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert {request["source_route"] for request in requests} == {"pdf_ocr"}
    source = manifest["source_review_coverage"][0]
    assert source["source_route"] == "pdf_ocr"
    assert source["review_route"] == "pdf_ocr"
    assert source["extraction_route"] == "html_web"
    assert {
        selected["extraction_route"] for selected in manifest["selection"]["selected_documents"]
    } == {"html_web"}


@pytest.mark.parametrize("override", [{"model": "gpt-5.6"}, {"seed": "wrong-seed"}])
def test_packet_rejects_model_or_seed_drift_before_materialization(
    tmp_path: Path, override: dict[str, str]
) -> None:
    fixture = write_fixture(tmp_path, count=17, opaque_quality_ids=False)
    result = subprocess.run(
        command(
            fixture,
            output=tmp_path / "review-requests.jsonl",
            manifest=tmp_path / "review-packet-manifest.json",
            **override,
        ),
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "frozen review policy" in result.stderr
