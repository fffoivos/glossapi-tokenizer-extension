from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PACKET = load("build_gfm_luna_review_packet")
RUNNER = load("run_gfm_luna_reviews")
AGGREGATE = load("aggregate_gfm_luna_reviews")


def candidate(index: int, *, risk: str = "routine", family: str = "inline") -> dict:
    return {
        "region_id": f"{index:064x}",
        "source_id": f"source-{index % 3}",
        "risk_tier": risk,
        "transformation_family": family,
        "rule_ids": [f"rule-{index % 4}"],
    }


def request(review_id: str, region_id: str, slot: str, risk: str = "routine") -> dict:
    return {
        "review_id": review_id,
        "region_id": region_id,
        "reviewer_slot": slot,
        "source_dataset": "demo",
        "risk_tier": risk,
        "normalizer_sha256": "a" * 64,
    }


def response(req: dict, *, verdict: str = "pass", table: str = "not_applicable") -> dict:
    return {
        "schema_version": "gfm_transformation_review_response_v1",
        "review_id": req["review_id"],
        "region_id": req["region_id"],
        "reviewer_slot": req["reviewer_slot"],
        "source_dataset": req["source_dataset"],
        "text_preservation": "pass" if verdict == "pass" else "fail",
        "artifact_removal": "not_applicable",
        "gfm_validity": "pass",
        "table_outcome": table,
        "unintended_change": "none" if verdict == "pass" else "minor",
        "verdict": verdict,
        "confidence": "high",
        "evidence": "The visible before/after region preserves its readable text.",
    }


def test_selection_keeps_every_high_risk_region_and_fills_target() -> None:
    rows = [candidate(index, risk="high" if index < 7 else "routine", family=f"family-{index % 5}") for index in range(40)]
    selected = PACKET.select_regions(rows, 20)

    assert len(selected) == 20
    assert {row["region_id"] for row in rows[:7]} <= {row["region_id"] for row in selected}
    assert len({row["source_id"] for row in selected}) == 3


def test_revalidation_selects_failures_and_changed_hashes_only() -> None:
    rows = []
    prior = []
    for index in range(3):
        row = {
            **candidate(index),
            "opaque_id": f"opaque-{index}",
            "ordinal": index,
            "before_sha256": f"before-{index}",
            "after_sha256": f"after-{index}",
        }
        rows.append(row)
        prior.append({**row, "validated": index != 1})
    rows[2] = {**rows[2], "after_sha256": "changed"}

    selected, reused = PACKET.select_revalidation_regions(rows, {"regions": prior})

    assert reused == 1
    assert {row["opaque_id"] for row in selected} == {"opaque-1", "opaque-2"}


def test_runner_batches_never_mix_reviewer_slots_or_exceed_character_budget() -> None:
    rows = [
        {**request(f"{index:064x}", f"{index:064x}", "primary" if index % 2 else "secondary"), "before_text": "x" * 120}
        for index in range(8)
    ]
    batches = RUNNER.batch_plan(rows, batch_size=4, max_batch_characters=800)

    assert sum(map(len, batches)) == len(rows)
    assert all(len({row["reviewer_slot"] for row in batch}) == 1 for batch in batches)
    assert all(sum(RUNNER.request_size(row) for row in batch) <= 800 or len(batch) == 1 for batch in batches)


def test_response_validation_rejects_identity_drift_as_retryable_value_error() -> None:
    req = request("1" * 64, "a" * 64, "primary")
    drifted = response(req)
    drifted["review_id"] = "2" * 64

    try:
        RUNNER.validate_response(drifted, req)
    except ValueError as error:
        assert "identity drift" in str(error)
    else:
        raise AssertionError("identity drift was not rejected")


def test_retry_operation_succeeds_after_retryable_failures() -> None:
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("malformed model response")
        return "ok"

    value, attempt = RUNNER.retry_operation(operation)

    assert value == "ok"
    assert attempt == 3


def test_merge_revalidation_replaces_only_targeted_logical_region(tmp_path: Path) -> None:
    base_regions = []
    for index in range(2):
        base_regions.append(
            {
                **candidate(index),
                "opaque_id": f"opaque-{index}",
                "ordinal": index,
                "before_sha256": f"before-{index}",
                "after_sha256": f"after-{index}",
                "validated": index == 0,
                "reviews": [],
                "final_verdict": "pass" if index == 0 else "fail",
            }
        )
    fixed = {**base_regions[1], "validated": True, "final_verdict": "pass"}
    baseline = tmp_path / "baseline.json"
    revalidation = tmp_path / "revalidation.json"
    output = tmp_path / "merged.json"
    baseline.write_text(json.dumps({"regions": base_regions}))
    revalidation.write_text(json.dumps({"regions": [fixed]}))

    summary = AGGREGATE.merge_revalidation(
        baseline_path=baseline, revalidation_path=revalidation, output_path=output
    )

    assert summary["status"] == "passed"
    assert summary["revalidated_regions"] == 1
    assert summary["reused_validated_regions"] == 1


def test_low_confidence_or_disagreement_creates_adjudication(tmp_path: Path) -> None:
    primary = request("1" * 64, "a" * 64, "primary", "high")
    secondary = request("2" * 64, "a" * 64, "secondary", "high")
    requests_path = tmp_path / "requests.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    output_path = tmp_path / "adjudication.jsonl"
    RUNNER.write_jsonl_atomic(requests_path, [primary, secondary])
    RUNNER.write_jsonl_atomic(
        responses_path,
        [response(primary), response(secondary, verdict="needs_human")],
    )

    summary = AGGREGATE.prepare(
        requests_path=requests_path,
        responses_path=responses_path,
        output_path=output_path,
    )
    adjudications = RUNNER.read_jsonl(output_path)

    assert summary["adjudication_requests"] == 1
    assert adjudications[0]["reviewer_slot"] == "adjudicator"
    assert len(adjudications[0]["prior_reviews"]) == 2


def test_response_schema_is_batch_compatible() -> None:
    schema = json.loads((HERE / "schemas/gfm_transformation_review_response.schema.json").read_text())
    batch = RUNNER.make_batch_schema(schema, 3)

    assert batch["properties"]["responses"]["minItems"] == 3
    assert batch["properties"]["responses"]["items"]["properties"]["schema_version"]["type"] == "string"


def test_recorded_repetition_passes_recover_each_coordinate_space() -> None:
    class Normalizer:
        @staticmethod
        def clean_generated_image_artifacts(value: str) -> str:
            return value.replace("(a_img.webp)", "")

    details = [
        {
            "cleaning_stage": "before_generated_image_cleanup",
            "pass_index": 1,
            "spans": [{"start_index": 3, "end_index": 7}],
        },
        {
            "cleaning_stage": "before_generated_image_cleanup",
            "pass_index": 2,
            "spans": [{"start_index": 0, "end_index": 2}],
        },
        {
            "cleaning_stage": "after_generated_image_cleanup",
            "pass_index": 1,
            "spans": [{"start_index": 32, "end_index": 34}],
        },
    ]
    inputs = PACKET.repetition_pass_inputs("abcXXXXdef(a_img.webp)YY", details, normalizer=Normalizer())

    assert inputs[("before_generated_image_cleanup", 1)] == "abcXXXXdef(a_img.webp)YY"
    assert inputs[("before_generated_image_cleanup", 2)] == "abc<!-- repeating-text-removed -->def(a_img.webp)YY"
    assert inputs[("after_generated_image_cleanup", 1)] == "<!-- repeating-text-removed -->c<!-- repeating-text-removed -->defYY"


def test_image_cleanup_input_replays_only_the_pre_image_passes() -> None:
    details = [
        {
            "cleaning_stage": "before_generated_image_cleanup",
            "pass_index": 1,
            "spans": [{"start_index": 3, "end_index": 7}],
        },
        {
            "cleaning_stage": "after_generated_image_cleanup",
            "pass_index": 1,
            "spans": [{"start_index": 0, "end_index": 2}],
        },
    ]

    recovered = PACKET.image_cleanup_input("abcXXXXdef", details)

    assert recovered == "abc<!-- repeating-text-removed -->def"


def test_long_focus_excerpt_is_clipped_on_line_boundaries() -> None:
    source = "\n".join(f"complete-row-{index:03d}" for index in range(100))

    excerpt = PACKET.around(source, 0, len(source), maximum=240)

    assert "[… focus span middle omitted from review excerpt …]" in excerpt
    assert "complete-row-000" in excerpt
    assert "complete-row-099" in excerpt
    assert "complete-row-00\n" not in excerpt


def test_nth_index_localizes_repeated_image_descriptions() -> None:
    value = "before <!-- description-of-removed-image: repeated --> middle " \
        "<!-- description-of-removed-image: repeated --> after"
    needle = "<!-- description-of-removed-image: repeated -->"

    assert PACKET.nth_index(value, needle, 0) == value.index(needle)
    assert PACKET.nth_index(value, needle, 1) == value.rindex(needle)
    assert PACKET.nth_index(value, needle, 2) == -1


def test_request_carries_exact_focus_anchor() -> None:
    region = {
        "region_id": "a" * 64,
        "opaque_id": "b" * 64,
        "source_dataset": "demo",
        "source_doc_id": "doc",
        "document_path": "documents/demo/doc.txt",
        "risk_tier": "high",
        "transformation_family": "table_readable_fallback",
        "rule_ids": ["table_fallback_reason_nested_table"],
        "expected_behavior": "Judge the target table.",
        "focus_anchor": "unique target cell",
        "before_sha256": "c" * 64,
        "after_sha256": "d" * 64,
        "before_text": "before unique target cell",
        "after_text": "after unique target cell",
    }
    built = PACKET.make_request(region, "primary", "e" * 64)

    assert built["focus_anchor"] == "unique target cell"


def test_table_fallback_focus_uses_source_position_and_local_anchor() -> None:
    cleaned = "intro\n<table><tr><th>Max Green</th><th>0.209</th></tr></table>\noutro"
    normalized = "intro\n\nMax Green\n0.209\n\noutro"
    fallback = {
        "source_line": 2,
        "source_column": 0,
        "plain_text_preview": "Max Green\n0.209",
        "anchor_candidates": ["Max Green"],
    }

    before_position, after_position, anchor = PACKET.table_fallback_focus(
        cleaned, normalized, fallback
    )

    assert before_position == cleaned.index("<table>")
    assert after_position == normalized.index("Max Green")
    assert anchor == "Max Green"
