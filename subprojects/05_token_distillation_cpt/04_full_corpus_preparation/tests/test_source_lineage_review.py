from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_source_reviews as AGGREGATE  # noqa: E402
import source_lineage as LINEAGE  # noqa: E402


def configs() -> tuple[dict, dict, dict, dict]:
    return (
        LINEAGE.load_json(HERE / "configs" / "sources.json"),
        LINEAGE.load_json(HERE / "configs" / "nanochat_initial_roster.json"),
        LINEAGE.load_json(HERE / "configs" / "source_lineage_aliases.json"),
        LINEAGE.load_json(HERE / "configs" / "source_review_policy.json"),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_first_appearance_is_anchored_to_first_data_revision() -> None:
    _, roster, _, _ = configs()
    initial = LINEAGE.first_appearance("greek_phd", roster)
    later = LINEAGE.first_appearance("HPLT/ell_Grek_ge8_no_mt_clean60", roster)
    new = LINEAGE.first_appearance("glossAPI/diavgeia", roster)
    assert initial == {
        "cohort": "nanochat_first_data_revision",
        "revision": "500b8bf577e1e70f4902b77edce2cda02a2559cb",
        "committed_at": "2026-03-16T16:40:36.000Z",
        "anchor_revision": "500b8bf577e1e70f4902b77edce2cda02a2559cb",
    }
    assert later["cohort"] == "nanochat_later_source_name_addition"
    assert new["cohort"] == "not_present_at_nanochat_anchor"


def test_registry_manifest_never_authorizes_blind_append() -> None:
    sources, roster, aliases, _ = configs()
    manifest = LINEAGE.build_registry_manifest(sources, roster, aliases)
    routes = {entry["source_id"]: entry for entry in manifest["candidates"]}
    assert len(routes) == 23
    assert all(route["blind_append_allowed"] is False for route in routes.values())
    assert routes["opengov_deliberations_v2"]["requires_base_identity_audit"] is True
    assert routes["school_books_new_editions"]["reviewed_aliases"][0]["alias_kind"] == "hybrid"
    assert routes["diavgeia"]["fallback_first_appearance"]["cohort"] == (
        "not_present_at_nanochat_anchor"
    )


def test_canonical_lineage_preserves_exact_name_and_stable_identity_across_cleaning() -> None:
    sources, roster, aliases, _ = configs()
    row = {
        "source_id": "archetai",
        "source_dataset": "Exact Upstream Name / Do Not Normalize",
        "source_artifact_path": "part-000.parquet",
        "source_row_id": "17",
        "source_doc_id": "https://EXAMPLE.org/work/17/#fragment",
        "text": "Κείμενο με κενά   \r\n",
    }
    first = LINEAGE.canonicalize_row(
        row, origin="candidate", sources=sources, roster=roster, aliases=aliases
    )
    changed = LINEAGE.canonicalize_row(
        {**row, "text": "Καθαρισμένο κείμενο"},
        origin="candidate",
        sources=sources,
        roster=roster,
        aliases=aliases,
    )
    assert first["source_dataset"] == row["source_dataset"]
    assert first["source_dataset_origin"] == "preserved_upstream_value"
    assert first["stable_uid"] == changed["stable_uid"]
    assert first["work_key"] == changed["work_key"]
    assert first["original_text_sha256"] != changed["original_text_sha256"]
    assert first["first_appearance"]["cohort"] == "not_present_at_nanochat_anchor"


def test_missing_source_name_uses_pinned_repo_and_resegmentation_requires_work_id() -> None:
    sources, roster, aliases, _ = configs()
    fallback = LINEAGE.canonicalize_row(
        {
            "source_id": "archetai",
            "source_artifact_path": "data.parquet",
            "source_row_id": "1",
            "source_doc_id": "doc-1",
            "text": "κείμενο",
        },
        origin="candidate",
        sources=sources,
        roster=roster,
        aliases=aliases,
    )
    assert fallback["source_dataset"] == "glossAPI/archetai"
    assert fallback["source_dataset_origin"] == "pinned_repo_fallback"

    with pytest.raises(ValueError, match="require an explicit work_id"):
        LINEAGE.canonicalize_row(
            {
                "source_id": "pergamos_sections",
                "source_artifact_path": "sections.parquet",
                "source_row_id": "1",
                "source_doc_id": "work-1-section-1",
                "text": "ενότητα",
            },
            origin="candidate",
            sources=sources,
            roster=roster,
            aliases=aliases,
        )


def test_lineage_cli_detects_base_candidate_exact_and_work_relationships(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(
        base,
        [
            {
                "source_dataset": "opengov.gr-diaboyleuseis",
                "source_family_id": "opengov_deliberations",
                "source_artifact_path": "data/opengov.parquet",
                "source_row_id": "base-1",
                "source_doc_id": "consultation-1",
                "work_id": "consultation-1",
                "text": "Ίδιο κείμενο\n",
            }
        ],
    )
    write_jsonl(
        candidate,
        [
            {
                "source_id": "opengov_deliberations_v2",
                "source_dataset": "opengov.gr-diaboyleuseis",
                "source_artifact_path": "opengov_v2.parquet",
                "source_row_id": "candidate-1",
                "source_doc_id": "consultation-1",
                "work_id": "consultation-1",
                "text": "Ίδιο κείμενο",
            }
        ],
    )
    registry = tmp_path / "registry.json"
    rows = tmp_path / "rows.jsonl"
    relationships = tmp_path / "relationships.jsonl"
    summary = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_source_lineage.py"),
            "rows",
            "--base-jsonl",
            str(base),
            "--candidate-jsonl",
            str(candidate),
            "--registry-manifest-out",
            str(registry),
            "--rows-out",
            str(rows),
            "--relationships-out",
            str(relationships),
            "--summary-out",
            str(summary),
        ],
        check=True,
    )
    relations = [json.loads(line) for line in relationships.read_text().splitlines()]
    assert {record["relationship_type"] for record in relations} == {
        "base_candidate_exact_text",
        "base_candidate_same_work_representation",
    }
    candidate_summary = next(
        entry
        for entry in json.loads(summary.read_text())["sources"]
        if entry["source_id"] == "opengov_deliberations_v2"
    )
    assert candidate_summary["base_candidate_exact_clusters"] == 1
    assert candidate_summary["base_candidate_work_clusters"] == 1
    assert "observed_base_candidate_exact_text" in candidate_summary["double_add_hazard_reasons"]


def candidate_rows(source_id: str, source_dataset: str, count: int) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        dirty = index < max(25, count // 4)
        text = (
            f"<div>Άρθρο {index}</div> test{index}@example.gr Ã© "
            if dirty
            else f"Καθαρό και χρήσιμο ελληνικό κείμενο για το έγγραφο {index}. " * 8
        )
        rows.append(
            {
                "source_id": source_id,
                "source_dataset": source_dataset,
                "source_artifact_path": "fixture.parquet",
                "source_row_id": str(index),
                "source_doc_id": f"doc-{index}",
                "review_cluster_id": f"template-{index % 60}",
                "text": text,
            }
        )
    return rows


def run_review_packet(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    requests = output_dir / "requests.jsonl"
    summary = output_dir / "summary.json"
    output_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_source_review_packet.py"),
            "--candidate-jsonl",
            str(input_path),
            "--requests-out",
            str(requests),
            "--summary-out",
            str(summary),
        ],
        check=True,
    )
    return requests, summary


def test_review_packet_uses_exact_100_200_strata_and_is_order_independent(tmp_path: Path) -> None:
    default_rows = candidate_rows("archetai", "Exact Archetai Source", 121)
    default_rows[-1]["privateData"] = True
    large_rows = candidate_rows("diavgeia", "Exact Diavgeia Source", 220)
    combined = default_rows + large_rows
    first_input = tmp_path / "first.jsonl"
    second_input = tmp_path / "second.jsonl"
    write_jsonl(first_input, combined)
    write_jsonl(second_input, list(reversed(combined)))

    first_requests, first_summary = run_review_packet(first_input, tmp_path / "first")
    second_requests, _ = run_review_packet(second_input, tmp_path / "second")
    first_records = [json.loads(line) for line in first_requests.read_text().splitlines()]
    second_records = [json.loads(line) for line in second_requests.read_text().splitlines()]
    assert [record["review_id"] for record in first_records] == [
        record["review_id"] for record in second_records
    ]
    assert "@example.gr" not in first_requests.read_text()
    assert "[REDACTED_EMAIL]" in first_requests.read_text()

    summary = json.loads(first_summary.read_text())
    reports = {entry["source_dataset"]: entry for entry in summary["sources"]}
    default = reports["Exact Archetai Source"]
    assert default["unique_sampled_documents"] == 100
    assert default["sampling_strata"] == {"cluster": 20, "random": 60, "risk": 20}
    assert default["double_review_documents"] == 10
    assert default["request_rows"] == 110
    assert default["private_data_documents_excluded"] == 1
    large = reports["Exact Diavgeia Source"]
    assert large["unique_sampled_documents"] == 200
    assert large["sampling_strata"] == {"cluster": 50, "random": 100, "risk": 50}
    assert large["double_review_documents"] == 20
    assert large["request_rows"] == 220
    assert summary["unique_sampled_documents"] == 300


def review_response(request: dict, action: str = "include", confidence: str = "high") -> dict:
    return {
        "schema_version": "source_quality_review_response_v1",
        "review_id": request["review_id"],
        "sample_id": request["sample_id"],
        "reviewer_slot": request["reviewer_slot"],
        "source_dataset": request["source_dataset"],
        "substantive_training_value": "high",
        "quality_score": 4,
        "language_register": "modern_greek",
        "defects": {key: "none" for key in AGGREGATE.DEFECT_KEYS},
        "variability": {"template_similarity": "low", "substantive_variation": "high"},
        "action": action,
        "defects_deterministically_repairable": action == "include_after_cleaning",
        "safety_or_license_blocker": False,
        "confidence": confidence,
        "evidence": "Substantive, clean Greek prose.",
    }


def test_review_aggregation_requires_adjudication_and_applies_cleaning_gate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, candidate_rows("archetai", "Exact Archetai Source", 120))
    requests_path, summary_path = run_review_packet(input_path, tmp_path / "packet")
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    cleaning_samples = {request["sample_id"] for request in requests if request["reviewer_slot"] == "primary"}
    cleaning_samples = set(sorted(cleaning_samples)[:5])
    responses = [
        review_response(
            request,
            action=(
                "include_after_cleaning" if request["sample_id"] in cleaning_samples else "include"
            ),
        )
        for request in requests
    ]
    reviews_path = tmp_path / "reviews.jsonl"
    write_jsonl(reviews_path, responses)
    output = tmp_path / "admission.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "aggregate_source_reviews.py"),
            "--requests",
            str(requests_path),
            "--packet-summary",
            str(summary_path),
            "--reviews",
            str(reviews_path),
            "--output",
            str(output),
        ]
    )
    assert completed.returncode == 0
    source = json.loads(output.read_text())["sources"][0]
    assert source["decision"] == "include_after_cleaning"
    assert source["post_clean_review_required"] is True

    secondary = next(request for request in requests if request["reviewer_slot"] == "secondary")
    for response in responses:
        if response["review_id"] == secondary["review_id"]:
            response["action"] = "exclude"
            response["defects_deterministically_repairable"] = False
    write_jsonl(reviews_path, responses)
    pending = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "aggregate_source_reviews.py"),
            "--requests",
            str(requests_path),
            "--packet-summary",
            str(summary_path),
            "--reviews",
            str(reviews_path),
            "--output",
            str(output),
        ]
    )
    assert pending.returncode == 2
    result = json.loads(output.read_text())
    assert result["pending_adjudications"] == 1
    assert result["sources"][0]["decision"] == "pending_adjudication"
