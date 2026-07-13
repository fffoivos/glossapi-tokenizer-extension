from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "scripts" / "agent1_v4_raw_review.py"
POLICY = HERE / "configs" / "agent1_v4_raw_review_policy.json"
PROMPT = HERE / "configs" / "agent1_v4_terra_review_prompt.md"
RESPONSE_SCHEMA = HERE / "schemas" / "agent1_v4_terra_review_response.schema.json"


def load_module():
    spec = importlib.util.spec_from_file_location("agent1_v4_raw_review_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V4 = load_module()


def load_runner():
    script = HERE / "scripts" / "run_agent1_v4_terra_reviews.py"
    spec = importlib.util.spec_from_file_location("agent1_v4_terra_runner_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def load_site_builder():
    script = HERE / "scripts" / "build_agent1_v4_review_site.py"
    spec = importlib.util.spec_from_file_location("agent1_v4_review_site_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SITE = load_site_builder()


def load_freezer():
    script = HERE / "scripts" / "freeze_agent1_v4_review.py"
    spec = importlib.util.spec_from_file_location("agent1_v4_freezer_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FREEZER = load_freezer()


def load_human_gate():
    script = HERE / "scripts" / "validate_agent1_v4_human_decisions.py"
    spec = importlib.util.spec_from_file_location("agent1_v4_human_gate_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HUMAN_GATE = load_human_gate()


def load_response_validator():
    script = HERE / "scripts" / "validate_agent1_v4_terra_responses.py"
    spec = importlib.util.spec_from_file_location("agent1_v4_response_validator_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RESPONSE_VALIDATOR = load_response_validator()


def load_field_profiler():
    script = HERE / "scripts" / "profile_agent1_v4_fields.py"
    spec = importlib.util.spec_from_file_location("agent1_v4_field_profiler_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIELD_PROFILER = load_field_profiler()


def load_envelope_materializer():
    script = HERE / "scripts" / "materialize_agent1_v4_nanochat_envelope.py"
    spec = importlib.util.spec_from_file_location("agent1_v4_envelope_materializer_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENVELOPE = load_envelope_materializer()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def fixture(tmp_path: Path, *, short_source: str | None = None, alternate_only: bool = False) -> dict[str, Path]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    source_ids = policy["source_ids"]
    sources: list[dict[str, object]] = []
    receipt_sources: list[dict[str, object]] = []
    routes: dict[str, str] = {}
    files = tmp_path / "files"
    for number, source_id in enumerate(source_ids):
        count = 19 if source_id == short_source else 23
        values = [f"{source_id} έγγραφο {index:03d}\nΣυνεκτικό ελληνικό κείμενο." for index in range(count)]
        if alternate_only and source_id == source_ids[0]:
            values = ["" for _ in range(count)]
        path = files / f"{source_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "id": [f"{source_id}-{index}" for index in range(count)],
                    "text": values,
                    "alternate_text": [f"alternate {index}" for index in range(count)],
                }
            ),
            path,
        )
        route = "html_web" if number % 3 == 0 else "pdf_ocr" if number % 3 == 1 else "structured"
        routes[source_id] = route
        sources.append(
            {
                "source_id": source_id,
                "repo_id": f"example/{source_id}",
                "revision": f"{number:040x}",
                "role": "additive_candidate",
                "source_family_id": source_id,
                "text_columns": ["text"],
                "alternate_text_columns": ["alternate_text"],
                "id_columns": ["id"],
                "training_eligibility": "eligible_open",
            }
        )
        receipt_sources.append(
            {
                "source_id": source_id,
                "revision": f"{number:040x}",
                "files": [
                    {
                        "path": path.name,
                        "local_path": str(path),
                        "size": path.stat().st_size,
                        "expected_hash": V4.sha256_file(path),
                    }
                ],
            }
        )
    sources_path = tmp_path / "sources.json"
    write_json(
        sources_path,
        {
            "schema_version": "fixture",
            "base": {},
            "apertus_overlap_overlay": {
                "repo_id": "example/apertus-overlay",
                "revision": "f" * 40,
            },
            "tokenizer": {
                "repo_id": "example/tokenizer",
                "revision": "e" * 40,
            },
            "sources": sources,
        },
    )
    receipt_path = tmp_path / "acquisition.json"
    write_json(
        receipt_path,
        {
            "schema_version": "full_cpt_acquisition_receipt_v1",
            "status": "passed",
            "sources_config_sha256": V4.sha256_file(sources_path),
            "sources": receipt_sources,
        },
    )
    roster_path = tmp_path / "roster.json"
    write_json(
        roster_path,
        {
            "candidate_source_ids": source_ids,
            "source_routes": routes,
            "extraction_routes": routes,
        },
    )
    license_path = tmp_path / "license.json"
    write_json(
        license_path,
        {
            "schema_version": "full_cpt_source_license_adjudication_v1",
            "source_registry": {"sha256": V4.sha256_file(sources_path)},
            "sources": [
                {
                    "source_id": source["source_id"],
                    "repo_id": source["repo_id"],
                    "revision": source["revision"],
                    "registry_training_eligibility": source["training_eligibility"],
                    "declared_license": "fixture",
                    "local_training": {"eligible": True, "status": "fixture_allowed"},
                    "redistribution": {"eligible": False, "status": "fixture_denied"},
                }
                for source in sources
            ],
        },
    )
    nanochat_path = tmp_path / "nanochat.json"
    write_json(
        nanochat_path,
        {
            "schema_version": "nanochat_initial_roster_v1",
            "repository": {
                "repo_id": "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
                "first_data_revision": "d" * 40,
            },
        },
    )
    greekmmlu_path = tmp_path / "greekmmlu.json"
    write_json(
        greekmmlu_path,
        {
            "benchmarks": [
                {
                    "id": "greekmmlu",
                    "source": "dascim/GreekMMLU",
                    "revision": "c" * 40,
                    "config": "All",
                    "split": "test",
                }
            ]
        },
    )
    environment_lock = tmp_path / "requirements.lock"
    environment_lock.write_text("fixture-runtime==1\n", encoding="utf-8")
    seed = tmp_path / "sampling-seed.txt"
    seed.write_text("d" * 64 + "\n", encoding="utf-8")
    seed.chmod(0o600)
    return {
        "sources": sources_path,
        "receipt": receipt_path,
        "roster": roster_path,
        "policy": POLICY,
        "prompt": PROMPT,
        "response_schema": RESPONSE_SCHEMA,
        "license": license_path,
        "nanochat": nanochat_path,
        "greekmmlu": greekmmlu_path,
        "environment_lock": environment_lock,
        "seed": seed,
    }


def materialize(inputs: dict[str, Path], output: Path) -> dict[str, object]:
    return V4.materialize_raw_review_packet(
        sources_path=inputs["sources"],
        acquisition_receipt=inputs["receipt"],
        roster_path=inputs["roster"],
        policy_path=inputs["policy"],
        seed_hex="a" * 64,
        prompt_path=inputs["prompt"],
        response_schema_path=inputs["response_schema"],
        code_commit="b" * 40,
        output=output,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def expected_source_counts() -> dict[str, int]:
    return V4.expected_source_counts(json.loads(POLICY.read_text(encoding="utf-8")))


def expected_review_count() -> int:
    return sum(expected_source_counts().values())


def test_materializes_exact_receipt_bound_raw_packet_deterministically(tmp_path: Path) -> None:
    inputs = fixture(tmp_path)
    first = materialize(inputs, tmp_path / "first")
    second = materialize(inputs, tmp_path / "second")

    assert first["logical_review_count"] == expected_review_count()
    assert first["source_counts"] == expected_source_counts()
    assert V4.validate_packet(tmp_path / "first")["status"] == "passed"
    first_requests = read_jsonl(tmp_path / "first" / "requests.jsonl")
    second_requests = read_jsonl(tmp_path / "second" / "requests.jsonl")
    assert first_requests == second_requests
    assert len({request["sample_id"] for request in first_requests}) == expected_review_count()
    assert all((tmp_path / "first" / request["document_path"]).is_file() for request in first_requests)
    assert all("alternate_text" != request["origin_locator"]["text_field"] for request in first_requests)


def test_blocks_when_source_has_fewer_than_twenty_unique_documents(tmp_path: Path) -> None:
    inputs = fixture(tmp_path, short_source="istorima")
    output = tmp_path / "blocked"
    with pytest.raises(V4.PacketBlockedError):
        materialize(inputs, output)
    blocked = json.loads((output / "blocking_issues.json").read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked"
    assert blocked["issues"] == [
        {
            "source_id": "istorima",
            "reason": "fewer_than_required_unique_nonempty_raw_documents",
            "required_documents": 20,
            "eligible_document_units": 19,
            "eligible_unique_documents_at_selection_cutoff": 19,
        }
    ]


def test_does_not_fall_back_to_alternate_text_column(tmp_path: Path) -> None:
    inputs = fixture(tmp_path, alternate_only=True)
    output = tmp_path / "blocked"
    with pytest.raises(V4.PacketBlockedError):
        materialize(inputs, output)
    blocked = json.loads((output / "blocking_issues.json").read_text(encoding="utf-8"))
    assert blocked["issues"][0]["source_id"] == "diavgeia"
    assert blocked["issues"][0]["eligible_document_units"] == 0


def test_seatbelt_profile_denies_file_outside_one_document_root(tmp_path: Path) -> None:
    sandbox_exec = Path("/usr/bin/sandbox-exec")
    if not sandbox_exec.is_file():
        pytest.skip("macOS sandbox-exec unavailable")
    root = tmp_path / "call-root"
    root.mkdir()
    forbidden = tmp_path / "outside.txt"
    forbidden.write_text("must not be readable", encoding="utf-8")
    profile = root / "seatbelt.sb"
    profile.write_text(
        RUNNER.seatbelt_profile(root, Path("/usr/bin/true"), Path.home() / ".codex"),
        encoding="utf-8",
    )
    completed = __import__("subprocess").run(
        [str(sandbox_exec), "-f", str(profile), "/bin/cat", str(forbidden)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0


def test_response_validation_rejects_evidence_outside_cited_lines(tmp_path: Path) -> None:
    document = tmp_path / "document.txt"
    document.write_text("πρώτη γραμμή\nδεύτερη γραμμή", encoding="utf-8")
    request = {
        "request_id": "1" * 64,
        "source_id": "diavgeia",
        "source_doc_id": "doc-1",
        "document_path": "documents/diavgeia/" + "2" * 64 + ".txt",
        "document_sha256": V4.sha256_file(document),
        "prompt_sha256": "3" * 64,
    }
    response = {
        "response_schema_version": "agent1_v4_terra_review_response_v1",
        **request,
        "model_id": "gpt-5.6-terra",
        "cleanliness_score": 4,
        "text_quality_score": 4,
        "confidence": "high",
        "coverage_mode": "full",
        "summary": "Καθαρό κείμενο.",
        "extraction_artifacts": [
            {
                "type": "boilerplate",
                "severity": "minor",
                "line_start": 1,
                "line_end": 1,
                "evidence_excerpt": "δεύτερη γραμμή",
                "explanation": "Λανθασμένο span για δοκιμή.",
                "deterministic_cleaning_possible": True,
                "suggested_cleaning_action": "remove",
            }
        ],
    }
    with pytest.raises(ValueError, match="cited line range"):
        RUNNER.validate_response(response, request, document)


def fake_response(request: dict[str, object]) -> dict[str, object]:
    return {
        "response_schema_version": "agent1_v4_terra_review_response_v1",
        "request_id": request["request_id"],
        "source_id": request["source_id"],
        "source_doc_id": request["source_doc_id"],
        "document_path": request["document_path"],
        "document_sha256": request["document_sha256"],
        "model_id": "gpt-5.6-terra",
        "prompt_sha256": request["prompt_sha256"],
        "cleanliness_score": 4,
        "text_quality_score": 5,
        "confidence": "high",
        "coverage_mode": "full",
        "summary": "Συνθετική απάντηση ελέγχου για τη στατική σελίδα.",
        "extraction_artifacts": [],
    }


def test_builds_private_raw_review_site_with_lazy_raw_documents(tmp_path: Path) -> None:
    inputs = fixture(tmp_path)
    packet_root = tmp_path / "packet"
    materialize(inputs, packet_root)
    requests = read_jsonl(packet_root / "requests.jsonl")
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        "".join(json.dumps(fake_response(request), ensure_ascii=False, sort_keys=True) + "\n" for request in requests),
        encoding="utf-8",
    )

    manifest = SITE.build_site(
        packet_root=packet_root,
        packet_manifest=packet_root / "packet_manifest.json",
        requests_path=packet_root / "requests.jsonl",
        responses_path=responses_path,
        site_secret_hex="c" * 64,
        output_dir=tmp_path / "review-site",
    )

    assert manifest["status"] == "passed"
    assert manifest["source_count"] == 18
    assert manifest["document_count"] == expected_review_count()
    assert manifest["portable_asset_bytes"] <= manifest["max_portable_assets_bytes"]
    assert all(not row["path"].startswith("data/documents/") for row in manifest["portable_assets"])
    index = json.loads((tmp_path / "review-site" / "data" / "index.json").read_text(encoding="utf-8"))
    assert len(index["cards"]) == expected_review_count()
    assert "text" not in index["cards"][0]
    raw_documents = list((tmp_path / "review-site" / "data" / "documents").glob("*.json"))
    assert len(raw_documents) == expected_review_count()
    first_raw = json.loads(raw_documents[0].read_text(encoding="utf-8"))
    assert "Συνεκτικό ελληνικό κείμενο." in first_raw["text"]
    assert "Συνεκτικό ελληνικό κείμενο." not in (tmp_path / "review-site" / "data" / "index.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="127.0.0.1"):
        SITE.serve_site(tmp_path / "review-site", port=8765, bind="0.0.0.0")


def test_freeze_receipt_binds_inputs_but_not_sampling_seed(tmp_path: Path) -> None:
    inputs = fixture(tmp_path)
    output = tmp_path / "00_freeze_receipt.json"
    receipt = FREEZER.freeze_review_inputs(
        sources_path=inputs["sources"],
        acquisition_receipt=inputs["receipt"],
        roster_path=inputs["roster"],
        policy_path=inputs["policy"],
        license_adjudication_path=inputs["license"],
        nanochat_roster_path=inputs["nanochat"],
        environment_lock_path=inputs["environment_lock"],
        greekmmlu_registry_path=inputs["greekmmlu"],
        prompt_path=inputs["prompt"],
        response_schema_path=inputs["response_schema"],
        sampling_seed_path=inputs["seed"],
        code_commit="b" * 40,
        glossapi_commit="a" * 40,
        output=output,
    )

    assert receipt["status"] == "passed"
    assert receipt["review_scope"]["logical_review_count"] == expected_review_count()
    assert len(receipt["sources"]) == 18
    serialized = output.read_text(encoding="utf-8")
    assert "d" * 64 not in serialized
    assert receipt["review_contract"]["sampling_seed_sha256"] == V4.sha256_text("d" * 64)
    assert all(
        source["exact_raw_external_review"]["does_not_change_training_or_redistribution_admission"]
        for source in receipt["sources"]
    )


def test_freeze_rejects_a_group_readable_sampling_seed(tmp_path: Path) -> None:
    inputs = fixture(tmp_path)
    inputs["seed"].chmod(0o644)
    with pytest.raises(ValueError, match="group- or world-readable"):
        FREEZER.freeze_review_inputs(
            sources_path=inputs["sources"],
            acquisition_receipt=inputs["receipt"],
            roster_path=inputs["roster"],
            policy_path=inputs["policy"],
            license_adjudication_path=inputs["license"],
            nanochat_roster_path=inputs["nanochat"],
            environment_lock_path=inputs["environment_lock"],
            greekmmlu_registry_path=inputs["greekmmlu"],
            prompt_path=inputs["prompt"],
            response_schema_path=inputs["response_schema"],
            sampling_seed_path=inputs["seed"],
            code_commit="b" * 40,
            glossapi_commit="a" * 40,
            output=tmp_path / "00_freeze_receipt.json",
        )


def human_decisions(packet_root: Path, *, approve: bool = True) -> dict[str, object]:
    requests = read_jsonl(packet_root / "requests.jsonl")
    source_ids = json.loads(POLICY.read_text(encoding="utf-8"))["source_ids"]
    return {
        "schema_version": "agent1_v4_human_decision_bundle_v1",
        "packet_manifest_sha256": V4.sha256_file(packet_root / "packet_manifest.json"),
        "approval_to_begin_field_discovery": approve,
        "source_status": {source_id: "admit" for source_id in source_ids},
        "source_observations": {source_id: "Reviewed raw extraction." for source_id in source_ids},
        "mapping_questions": {source_id: "" for source_id in source_ids},
        "documents": {
            request["request_id"]: {
                "source_id": request["source_id"],
                "source_doc_id": request["source_doc_id"],
                "disposition": "agree",
                "cleanliness_score_override": None,
                "text_quality_score_override": None,
                "note": "",
            }
            for request in requests
        },
    }


def test_human_gate_requires_all_document_source_and_license_decisions(tmp_path: Path) -> None:
    inputs = fixture(tmp_path)
    packet_root = tmp_path / "packet"
    materialize(inputs, packet_root)
    freeze_receipt = tmp_path / "00_freeze_receipt.json"
    FREEZER.freeze_review_inputs(
        sources_path=inputs["sources"],
        acquisition_receipt=inputs["receipt"],
        roster_path=inputs["roster"],
        policy_path=inputs["policy"],
        license_adjudication_path=inputs["license"],
        nanochat_roster_path=inputs["nanochat"],
        environment_lock_path=inputs["environment_lock"],
        greekmmlu_registry_path=inputs["greekmmlu"],
        prompt_path=inputs["prompt"],
        response_schema_path=inputs["response_schema"],
        sampling_seed_path=inputs["seed"],
        code_commit="b" * 40,
        glossapi_commit="a" * 40,
        output=freeze_receipt,
    )
    decisions_path = tmp_path / "human-decisions.json"
    write_json(decisions_path, human_decisions(packet_root))

    receipt = HUMAN_GATE.validate_human_decisions(
        packet_root=packet_root,
        packet_manifest_path=packet_root / "packet_manifest.json",
        freeze_receipt_path=freeze_receipt,
        decisions_path=decisions_path,
        output=tmp_path / "20_human_gate.json",
    )

    assert receipt["status"] == "passed"
    assert len(receipt["admitted_source_ids"]) == 18
    assert receipt["document_disposition_counts"] == {"agree": expected_review_count()}


def test_human_gate_blocks_without_explicit_field_discovery_approval(tmp_path: Path) -> None:
    inputs = fixture(tmp_path)
    packet_root = tmp_path / "packet"
    materialize(inputs, packet_root)
    freeze_receipt = tmp_path / "00_freeze_receipt.json"
    FREEZER.freeze_review_inputs(
        sources_path=inputs["sources"],
        acquisition_receipt=inputs["receipt"],
        roster_path=inputs["roster"],
        policy_path=inputs["policy"],
        license_adjudication_path=inputs["license"],
        nanochat_roster_path=inputs["nanochat"],
        environment_lock_path=inputs["environment_lock"],
        greekmmlu_registry_path=inputs["greekmmlu"],
        prompt_path=inputs["prompt"],
        response_schema_path=inputs["response_schema"],
        sampling_seed_path=inputs["seed"],
        code_commit="b" * 40,
        glossapi_commit="a" * 40,
        output=freeze_receipt,
    )
    decisions_path = tmp_path / "human-decisions.json"
    write_json(decisions_path, human_decisions(packet_root, approve=False))
    with pytest.raises(ValueError, match="approval to begin field discovery"):
        HUMAN_GATE.validate_human_decisions(
            packet_root=packet_root,
            packet_manifest_path=packet_root / "packet_manifest.json",
            freeze_receipt_path=freeze_receipt,
            decisions_path=decisions_path,
            output=tmp_path / "20_human_gate.json",
        )


def test_response_bundle_validator_closes_the_receipt_bound_response_count(tmp_path: Path) -> None:
    inputs = fixture(tmp_path)
    packet_root = tmp_path / "packet"
    materialize(inputs, packet_root)
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        "".join(
            json.dumps(fake_response(request), ensure_ascii=False, sort_keys=True) + "\n"
            for request in read_jsonl(packet_root / "requests.jsonl")
        ),
        encoding="utf-8",
    )
    receipt = RESPONSE_VALIDATOR.validate_response_bundle(
        packet_root=packet_root,
        packet_manifest_path=packet_root / "packet_manifest.json",
        responses_path=responses,
        output=tmp_path / "response-validation.json",
    )
    assert receipt["status"] == "passed"
    assert receipt["logical_review_count"] == expected_review_count()
    assert receipt["source_counts"] == expected_source_counts()


def test_field_profiler_scans_only_admitted_receipt_bound_sources(tmp_path: Path) -> None:
    inputs = fixture(tmp_path)
    packet_root = tmp_path / "packet"
    materialize(inputs, packet_root)
    freeze_receipt = tmp_path / "00_freeze_receipt.json"
    FREEZER.freeze_review_inputs(
        sources_path=inputs["sources"], acquisition_receipt=inputs["receipt"], roster_path=inputs["roster"],
        policy_path=inputs["policy"], license_adjudication_path=inputs["license"], nanochat_roster_path=inputs["nanochat"],
        environment_lock_path=inputs["environment_lock"], greekmmlu_registry_path=inputs["greekmmlu"],
        prompt_path=inputs["prompt"], response_schema_path=inputs["response_schema"], sampling_seed_path=inputs["seed"],
        code_commit="b" * 40, glossapi_commit="a" * 40, output=freeze_receipt,
    )
    decisions_path = tmp_path / "human-decisions.json"
    write_json(decisions_path, human_decisions(packet_root))
    human_gate = tmp_path / "20_human_gate.json"
    HUMAN_GATE.validate_human_decisions(
        packet_root=packet_root, packet_manifest_path=packet_root / "packet_manifest.json", freeze_receipt_path=freeze_receipt,
        decisions_path=decisions_path, output=human_gate,
    )

    profile = FIELD_PROFILER.profile_fields(
        sources_path=inputs["sources"], acquisition_receipt=inputs["receipt"], human_gate_receipt=human_gate,
        output=tmp_path / "field-profile.json",
    )

    assert profile["status"] == "passed"
    assert len(profile["source_reports"]) == 18
    diavgeia = next(row for row in profile["source_reports"] if row["source_id"] == "diavgeia")
    text_field = next(row for row in diavgeia["fields"] if row["path"] == "text")
    assert text_field["nonblank_count"] == 23
    assert "provisional_text" in text_field["classification"]


def test_materializes_exact_six_column_envelope_from_approved_mapping(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    inputs = fixture(tmp_path)
    packet_root = tmp_path / "packet"
    materialize(inputs, packet_root)
    freeze_receipt = tmp_path / "00_freeze_receipt.json"
    FREEZER.freeze_review_inputs(
        sources_path=inputs["sources"], acquisition_receipt=inputs["receipt"], roster_path=inputs["roster"],
        policy_path=inputs["policy"], license_adjudication_path=inputs["license"], nanochat_roster_path=inputs["nanochat"],
        environment_lock_path=inputs["environment_lock"], greekmmlu_registry_path=inputs["greekmmlu"],
        prompt_path=inputs["prompt"], response_schema_path=inputs["response_schema"], sampling_seed_path=inputs["seed"],
        code_commit="b" * 40, glossapi_commit="a" * 40, output=freeze_receipt,
    )
    decisions_path = tmp_path / "human-decisions.json"
    write_json(decisions_path, human_decisions(packet_root))
    human_gate = tmp_path / "20_human_gate.json"
    HUMAN_GATE.validate_human_decisions(
        packet_root=packet_root, packet_manifest_path=packet_root / "packet_manifest.json", freeze_receipt_path=freeze_receipt,
        decisions_path=decisions_path, output=human_gate,
    )
    profile_path = tmp_path / "field-profile.json"
    FIELD_PROFILER.profile_fields(
        sources_path=inputs["sources"], acquisition_receipt=inputs["receipt"], human_gate_receipt=human_gate, output=profile_path,
    )
    source_ids = json.loads(POLICY.read_text(encoding="utf-8"))["source_ids"]
    mapping_path = tmp_path / "mapping.json"
    write_json(
        mapping_path,
        {
            "schema_version": "agent1_v4_field_mapping_v1",
            "field_profile_sha256": V4.sha256_file(profile_path),
            "human_gate_receipt_sha256": V4.sha256_file(human_gate),
            "approval": {"fixture": "approved"},
            "mappings": {
                source_id: {
                    "text_path": "text", "title_path": None, "author_path": None,
                    "source_dataset_path": None, "source_doc_id_paths": ["id"],
                }
                for source_id in source_ids
            },
        },
    )
    output = tmp_path / "envelope"
    manifest = ENVELOPE.materialize_envelope(
        sources_path=inputs["sources"], acquisition_receipt=inputs["receipt"], human_gate_receipt=human_gate,
        field_profile=profile_path, mapping_path=mapping_path, output=output,
    )

    assert manifest["six_column_schema"] == list(ENVELOPE.SIX_COLUMNS)
    shard = output / "candidates" / "diavgeia.parquet"
    table = pq.read_table(shard)
    assert table.schema.names == list(ENVELOPE.SIX_COLUMNS)
    first = table.to_pylist()[0]
    assert first["title"] is None and first["author"] is None
    assert first["source_dataset"] == "example/diavgeia"
    metadata = json.loads(first["source_metadata_json"])
    assert "text" not in metadata and "alternate_text" not in metadata
