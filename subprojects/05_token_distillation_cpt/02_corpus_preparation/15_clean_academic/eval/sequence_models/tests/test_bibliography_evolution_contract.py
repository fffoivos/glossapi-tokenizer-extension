from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import sequence_models.bibliography_evolution as evolution

from sequence_models.bibliography_evolution import (
    SEALED_REQUEST_SCHEMA,
    _begin_sealed_batch,
    _freeze_manifest,
    _attest_git_checkout,
    _validate_runner,
    _evaluate_acceptance,
)
from sequence_models.bibliography_evolution_contract import (
    BASELINE_SCHEMA,
    FIXED_MODULE_ORDER,
    LEAKAGE_SCHEMA,
    CandidateStore,
    ContractError,
    build_registry,
    expand_template,
    paired_work_bootstrap,
    sha256_directory,
    sha256_file,
    validate_candidate_spec,
    verify_g0,
    with_candidate_id,
)
from sequence_models.bibliography_evolution_core_decode import decoding_document_subset


def _input_receipts() -> dict:
    artifact = Path(__file__).resolve()
    return {
        "dev": {
            "path": str(artifact),
            "sha256": sha256_file(artifact),
            "data_class": "development_labels",
            "split": "validation",
            "document_scope": "prediction_blind_extraction_qualified_268",
            "contains_labels": True,
        }
    }


def _g0_spec() -> dict:
    artifact = Path(__file__).resolve()
    return with_candidate_id(
        {
            "schema_version": "bibliography-evolution-candidate-v1",
            "generation": "G0",
            "parent_candidate_ids": [],
            "hypothesis": "Fixture G0 replays one frozen baseline.",
            "changed_component": "baseline.replay",
            "parameter_family": "byte_identity",
            "changes": {"baseline.replay": {"parameter_family": "byte_identity", "parameters": {"replay": "frozen"}}},
            "fixed_modules": list(FIXED_MODULE_ORDER),
            "sweep_grid": {"replay": ["frozen"]},
            "sweep_point": {"replay": "frozen"},
            "seeds": [20260718],
            "expected_direction": {"token_fp": "nonincrease"},
            "acceptance_rule": {"headline_documents": 268},
            "code_commit": "a" * 40,
            "input_receipts": _input_receipts(),
            "runner": {
                "module": "sequence_models.bibliography_evolution_g0_replay",
                "argv": [
                    "--lock", str(artifact), "--authoritative-root", str(artifact),
                    "--validation-table-dir", str(artifact),
                    "--validation-signal-probability", str(artifact),
                    "--validation-line-probability", str(artifact),
                    "--validation-scope-mask", str(artifact),
                    "--qualified-documents", str(artifact),
                    "--output-dir", "@CANDIDATE_DIR@/backend",
                    "--code-commit", "a" * 40, "--slurm-job-id", "@SLURM_JOB_ID@",
                ],
            },
        }
    )


def _parent_receipt(tmp_path: Path) -> Path:
    spec = _g0_spec()
    store = CandidateStore(tmp_path / "fixture_parent_store")
    receipt = store.root / spec["candidate_id"] / "receipt.json"
    if receipt.is_file():
        return receipt
    candidate = store.create(spec, _policy())
    return store.finalize(candidate, _result(candidate, (10, 20, 0.0, 1.0)))


def _spec(tmp_path: Path, *, generation: str = "G1", value: float = 0.3) -> dict:
    component = "baseline.replay" if generation == "G0" else "decoder.anchor_and_expansion_policy"
    family = "byte_identity" if generation == "G0" else "anchor_threshold"
    parameter = "replay" if generation == "G0" else "anchor_probability"
    point = "frozen" if generation == "G0" else value
    artifact = Path(__file__).resolve()
    inputs = _input_receipts()
    parents: list[str] = []
    if generation != "G0":
        parent = _parent_receipt(tmp_path)
        parent_packet = json.loads(parent.read_text(encoding="utf-8"))
        parents = [parent_packet["candidate_id"]]
        inputs["parent"] = {
            "path": str(parent),
            "sha256": sha256_file(parent),
            "digest_kind": "file_sha256",
            "data_class": "parent_candidate_receipt",
            "split": "development",
            "document_scope": "aggregate_no_rows",
            "contains_labels": False,
            "candidate_id": parent_packet["candidate_id"],
        }
    module = (
        "sequence_models.bibliography_evolution_g0_replay"
        if generation == "G0"
        else "sequence_models.bibliography_evolution_core_decode"
    )
    runner_argv = (
        _g0_spec()["runner"]["argv"]
        if generation == "G0"
        else [
            "--table-dir", str(artifact), "--signal-probability", str(artifact),
            "--line-probability", str(artifact), "--scope-mask", str(artifact),
            "--qualified-documents", str(artifact), "--anchor-probability", str(value),
            "--anchors-required", "2", "--anchor-window", "16",
            "--maximum-bridge-gap", "8", "--inside-probability", "0.05",
            "--adjacent-expansion", "2", "--header-window", "2",
            "--output-dir", "@CANDIDATE_DIR@/backend", "--code-commit", "a" * 40,
            "--slurm-job-id", "@SLURM_JOB_ID@",
        ]
    )
    return with_candidate_id(
        {
            "schema_version": "bibliography-evolution-candidate-v1",
            "generation": generation,
            "parent_candidate_ids": parents,
            "hypothesis": "One isolated parameter family improves one stated objective.",
            "changed_component": component,
            "parameter_family": family,
            "changes": {
                component: {
                    "parameter_family": family,
                    "parameters": {parameter: point},
                }
            },
            "fixed_modules": list(FIXED_MODULE_ORDER),
            "sweep_grid": {parameter: [point]},
            "sweep_point": {parameter: point},
            "seeds": [20260718],
            "expected_direction": {"token_fp": "decrease"},
            "acceptance_rule": {"headline_documents": 268},
            "code_commit": "a" * 40,
            "input_receipts": inputs,
            "runner": {
                "module": module,
                "argv": runner_argv,
            },
        }
    )


def _policy(**updates: object) -> dict:
    value = {
        "schema_version": LEAKAGE_SCHEMA,
        "sealed_test_status": "not_materialized",
        "allowed_development_scopes": [
            "prediction_blind_extraction_qualified_268", "aggregate_no_rows"
        ],
        "forbidden_data_classes": ["sealed_document", "sealed_label", "sealed_prediction"],
        "forbidden_path_tokens": ["sealed_test", "final_test"],
        "sealed_roots": [],
        "sealed_artifact_sha256": [],
    }
    value.update(updates)
    return value


def _result(candidate_dir: Path, metrics: tuple[float, float, float, float]) -> dict:
    rows = candidate_dir / "rows.jsonl"
    rows.write_text("{}\n", encoding="utf-8")
    prediction = candidate_dir / "prediction.npy"
    prediction.write_bytes(b"prediction")
    artifact = candidate_dir / "model.bin"
    artifact.write_bytes(b"model")
    fp, fn, spurious, boundary = metrics
    return {
        "schema_version": "bibliography-evolution-result-v1",
        "status": "passed",
        "all_rows": {"path": rows.name, "sha256": sha256_file(rows)},
        "artifacts": {"model": {"path": artifact.name, "sha256": sha256_file(artifact)}},
        "predictions": {"main": {"path": prediction.name, "sha256": sha256_file(prediction)}},
        "metrics": {
            "document_count": 268,
            "token_fp": fp,
            "token_fn": fn,
            "spurious_blocks_per_zero_block_document": spurious,
            "mean_boundary_error_emitted_lines": boundary,
        },
        "metrics_by_source": {
            "greek_phd": {"document_count": 92},
            "kallipos": {"document_count": 92},
            "openarchives": {"document_count": 84},
        },
        "paired_deltas": {
            "schema_version": "bibliography-evolution-paired-work-bootstrap-v1",
            "work_count": 1,
            "candidate_rows_sha256": sha256_file(rows),
            "baseline_rows_sha256": sha256_file(rows),
            "deltas_candidate_minus_baseline": {
                name: {"point": 0.0, "ci95": [0.0, 0.0], "probability_improved": 0.0}
                for name in ("token_fp", "token_fn", "spurious_blocks_per_zero_block_document", "mean_boundary_error_emitted_lines")
            },
        },
        "runtime": {"wall_seconds": 1.0},
        "job": {"slurm_job_id": "1"},
        "tests": {"status": "passed", "code_commit": "a" * 40},
        "selection": {"eligible_for_pareto": True, "acceptance": {"passed": True, "checks": {}}},
        "rejection": {"reasons": []},
    }


def test_spec_identity_and_one_family_enforcement(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    validate_candidate_spec(spec)
    assert spec["candidate_id"] == _spec(tmp_path)["candidate_id"]
    broken = json.loads(json.dumps(spec))
    broken["changes"]["decoder.gap_connector"] = {
        "parameter_family": "gap",
        "parameters": {"threshold": 0.1},
    }
    with pytest.raises(ContractError, match="exactly the declared component"):
        validate_candidate_spec(broken)
    broken = json.loads(json.dumps(spec))
    broken["changes"]["decoder.anchor_and_expansion_policy"]["parameter_family"] = "other"
    with pytest.raises(ContractError, match="families differ"):
        validate_candidate_spec(broken)


def test_leakage_barrier_and_candidate_store_are_fail_closed(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path / "candidates")
    spec = _spec(tmp_path)
    candidate = store.create(spec, _policy())
    with pytest.raises(FileExistsError):
        store.create(spec, _policy())
    result = _result(candidate, (10, 20, 0.0, 1.0))
    receipt = store.finalize(candidate, result)
    assert receipt.is_file()
    with pytest.raises(FileExistsError):
        store.finalize(candidate, result)

    leaked = json.loads(json.dumps(spec))
    leaked["input_receipts"]["dev"]["split"] = "sealed_test"
    leaked["candidate_id"] = with_candidate_id({k: v for k, v in leaked.items() if k != "candidate_id"})["candidate_id"]
    with pytest.raises(ContractError, match="final-test split"):
        store.create(leaked, _policy())
    wrong_hash = _spec(tmp_path, value=0.77)
    wrong_hash["input_receipts"]["dev"]["sha256"] = "0" * 64
    wrong_hash = with_candidate_id({k: v for k, v in wrong_hash.items() if k != "candidate_id"})
    with pytest.raises(ContractError, match="does not match bytes"):
        store.create(wrong_hash, _policy())
    undeclared = _spec(tmp_path, value=0.78)
    table_value = undeclared["runner"]["argv"].index("--table-dir") + 1
    undeclared["runner"]["argv"][table_value] = "/tmp/not-declared"
    undeclared = with_candidate_id({k: v for k, v in undeclared.items() if k != "candidate_id"})
    with pytest.raises(ContractError, match="undeclared input"):
        store.create(undeclared, _policy())


def test_directory_inputs_bind_every_sibling_and_reject_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "table"
    source.mkdir()
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    (source / "sibling.npy").write_bytes(b"first")
    spec = _spec(tmp_path, value=0.731)
    original = str(Path(__file__).resolve())
    spec["input_receipts"]["dev"] = {
        **spec["input_receipts"]["dev"],
        "path": str(source),
        "sha256": sha256_directory(source),
        "digest_kind": "recursive_tree_sha256_v1",
    }
    spec["runner"]["argv"] = [
        str(source) if value == original else value for value in spec["runner"]["argv"]
    ]
    spec = with_candidate_id({key: value for key, value in spec.items() if key != "candidate_id"})
    CandidateStore(tmp_path / "directory_store").create(spec, _policy())
    (source / "sibling.npy").write_bytes(b"changed")
    drifted = json.loads(json.dumps(spec))
    drifted["sweep_point"]["anchor_probability"] = 0.732
    drifted["sweep_grid"]["anchor_probability"] = [0.732]
    drifted["changes"]["decoder.anchor_and_expansion_policy"]["parameters"]["anchor_probability"] = 0.732
    drifted["runner"]["argv"][drifted["runner"]["argv"].index("--anchor-probability") + 1] = "0.732"
    drifted = with_candidate_id({key: value for key, value in drifted.items() if key != "candidate_id"})
    with pytest.raises(ContractError, match="does not match bytes"):
        CandidateStore(tmp_path / "directory_store").create(drifted, _policy())
    (source / "link").symlink_to(source / "manifest.json")
    with pytest.raises(ContractError, match="symlink"):
        sha256_directory(source)


def test_runner_module_and_unknown_flags_are_rejected(tmp_path: Path) -> None:
    malicious = _spec(tmp_path, value=0.741)
    malicious["runner"]["module"] = "sequence_models.bibliography_entry_models"
    with pytest.raises(ContractError, match="not pinned"):
        with_candidate_id({key: value for key, value in malicious.items() if key != "candidate_id"})
    unknown = _spec(tmp_path, value=0.742)
    unknown["runner"]["argv"].extend(["--secret-path", "/tmp/sealed"])
    with pytest.raises(ContractError, match="unapproved"):
        with_candidate_id({key: value for key, value in unknown.items() if key != "candidate_id"})


def test_pareto_selection_and_exact_tie_break_are_deterministic(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path / "candidates")
    paths = []
    for index, vector in enumerate(((10, 20, 1, 2), (8, 25, 1, 2), (10, 20, 1, 2), (12, 30, 2, 3))):
        spec = _spec(tmp_path, value=0.2 + index / 100)
        candidate = store.create(spec, _policy())
        paths.append(store.finalize(candidate, _result(candidate, vector)))
    left = build_registry(paths)
    right = build_registry(reversed(paths))
    assert left == right
    assert len(left["pareto_candidate_ids"]) == 2
    dominated = [row for row in left["candidates"] if row["objective_vector"] and not row["pareto"]]
    assert any("exact_objective_tie" in row["rejection_reasons"] for row in dominated)


def test_source_stratified_paired_work_bootstrap() -> None:
    baseline = [
        {"work_id": "a", "source": "x", "token_fp": 5, "token_fn": 4, "spurious_zero_blocks": 1, "zero_doc_count": 1, "boundary_error_sum": 2, "boundary_match_count": 1},
        {"work_id": "b", "source": "y", "token_fp": 5, "token_fn": 4, "spurious_zero_blocks": 0, "zero_doc_count": 1, "boundary_error_sum": 4, "boundary_match_count": 2},
    ]
    candidate = [{**row, "token_fp": row["token_fp"] - 1} for row in baseline]
    result = paired_work_bootstrap(candidate, baseline, iterations=100, seed=7)
    assert result["deltas_candidate_minus_baseline"]["token_fp"]["point"] == -2
    assert result["deltas_candidate_minus_baseline"]["token_fp"]["probability_improved"] == 1.0


def test_g3_acceptance_uses_executed_active_trace_not_static_order() -> None:
    names = [
        ("internal_gap_connection", "internal_gap_connection", 0.20, 2),
        ("boundary_trim", "boundary_trim", 0.05, 1),
        ("outward_edge_optional", "outward_edge", 0.40, 1),
        ("weak_unseeded_optional", "weak_unseeded", 0.20, 1),
        ("whole_component_veto", "whole_component_veto", 0.02, 1),
    ]
    trace = [
        {
            "position": index,
            "module": module,
            "operation": operation,
            "status": "enabled_changed_family" if operation == "outward_edge" else "executed_fixed_reference",
            "threshold": threshold,
            "max_lines": max_lines,
        }
        for index, (module, operation, threshold, max_lines) in enumerate(names)
    ]
    spec = {
        "generation": "G3",
        "changed_component": "decoder.outward_edge",
        "acceptance_rule": {"headline_documents": 268, "runs_after_boundary_trim": True},
        "code_commit": "a" * 40,
        "parent_candidate_ids": ["parent"],
    }
    paired = {
        "schema_version": "bibliography-evolution-paired-work-bootstrap-v1",
        "work_count": 1,
        "candidate_rows_sha256": "a" * 64,
        "baseline_rows_sha256": "b" * 64,
        "deltas_candidate_minus_baseline": {
            name: {} for name in (
                "token_fp", "token_fn", "spurious_blocks_per_zero_block_document",
                "mean_boundary_error_emitted_lines",
            )
        },
    }
    acceptance, reasons = _evaluate_acceptance(
        spec, {"module_trace": trace}, {"document_count": 268}, paired,
        {"status": "passed", "code_commit": "a" * 40}, {},
    )
    assert acceptance["passed"] and not reasons
    trace[1]["status"] = "disabled_identity"
    # A forged no-op trace is not the canonical active reference pipeline.
    acceptance, reasons = _evaluate_acceptance(
        spec, {"module_trace": trace}, {"document_count": 268}, paired,
        {"status": "passed", "code_commit": "a" * 40}, {},
    )
    assert not acceptance["passed"]


def test_g0_verification_checks_receipts_artifacts_and_prediction(tmp_path: Path) -> None:
    root = tmp_path / "baseline"
    root.mkdir()
    receipts = {}
    artifacts = {}
    for relative in ("a/receipt.json", "b/receipt.json"):
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        receipts[relative] = sha256_file(path)
    prediction = root / "prediction.npy"
    prediction.write_bytes(b"same")
    artifacts["prediction.npy"] = sha256_file(prediction)
    lock = {
        "schema_version": BASELINE_SCHEMA,
        "authoritative_root": str(root.resolve()),
        "receipt_sha256": receipts,
        "artifact_sha256": artifacts,
        "decoder_config": {"anchor_probability": 0.3},
        "headline_metrics_268": {"document_count": 268},
        "g0_replay": {"prediction_sha256": sha256_file(prediction)},
    }
    assert verify_g0(lock, root=root, replay_prediction=prediction)["status"] == "passed_byte_identical"
    prediction.write_bytes(b"changed")
    with pytest.raises(ContractError):
        verify_g0(lock, root=root, replay_prediction=prediction)


def test_git_commit_cleanliness_and_runtime_job_injection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "tracked.txt").write_text("ok\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert _attest_git_checkout(repo, head)["clean"] is True
    with pytest.raises(ContractError, match="commit drift"):
        _attest_git_checkout(repo, "0" * 40)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ContractError, match="not clean"):
        _attest_git_checkout(repo, head)

    spec = _spec(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    command = _validate_runner(spec, candidate_dir=tmp_path / "candidate")
    assert command[-1] == "98765"


def test_all_templates_expand_to_valid_executable_specs() -> None:
    root = Path(__file__).parents[1]
    packet = json.loads((root / "evolution/experiment_templates.json").read_text(encoding="utf-8"))
    bindings = {key: "/bound/path" for key in (
        "BASELINE_LOCK", "BASELINE_REPLAY_PREDICTION", "BASELINE_ROOT", "BLOCK_OOF_DIR",
        "DETERMINISTIC_HEADER_ROLES", "DETERMINISTIC_ROLES_DIR", "LEARNED_HEADER_ROLE_IDS",
        "LEFT_PREDICTION", "LINE_OOF_DIR", "PARENT_PREDICTION", "QUALIFIED_268_IDS",
        "RIGHT_PREDICTION", "SOURCE", "TRAIN_QUALITY_DECISIONS", "TRAIN_TABLE_DIR",
        "VALIDATION_LINE_PROBABILITY", "VALIDATION_SCOPE_MASK", "VALIDATION_SIGNAL_PROBABILITY",
        "VALIDATION_TABLE_DIR", "PARENT_BARRIER_ARTIFACT", "LEFT_BARRIER_ARTIFACT",
        "RIGHT_BARRIER_ARTIFACT", "TRAIN_RECALL_BLOCK_DIR",
        "VALIDATION_QUALITY_DECISIONS", "VALIDATION_POLICY",
    )}
    def inputs_for(*parent_ids: str, registry: bool = False) -> dict:
        result = _input_receipts()
        for index, parent_id in enumerate(parent_ids):
            result[f"parent_{index}"] = {
                "path": f"/bound/{parent_id}/receipt.json", "sha256": "1" * 64,
                "data_class": "parent_candidate_receipt", "split": "development",
                "document_scope": "aggregate_no_rows", "candidate_id": parent_id,
            }
        if registry:
            result["pareto_registry"] = {
                "path": "/bound/registry.json", "sha256": "2" * 64,
                "data_class": "development_pareto_registry", "split": "development",
                "document_scope": "aggregate_no_rows",
            }
        return result
    bindings.update({
        "CODE_COMMIT": "a" * 40, "CPU_WORKERS": 20, "G0_CANDIDATE_ID": "g0-parent",
        "G1_PARENT_ID": "g1-parent", "G2_PARENT_ID": "g2-parent", "G3_PARENT_ID": "g3-parent",
        "LEFT_PARETO_ID": "g4-left", "RIGHT_PARETO_ID": "g4-right", "MODEL_SEED": 20260718,
        "SLURM_JOB_ID": "dry-run",
        "G0_INPUT_RECEIPTS": inputs_for(),
        "G1_INPUT_RECEIPTS": inputs_for("g0-parent"),
        "G2_INPUT_RECEIPTS": inputs_for("g1-parent"),
        "G3_INPUT_RECEIPTS": inputs_for("g2-parent"),
        "G4_INPUT_RECEIPTS": inputs_for("g0-parent"),
        "G5_INPUT_RECEIPTS": inputs_for("g4-left", "g4-right", registry=True),
    })
    specs = [spec for template in packet["templates"] for spec in expand_template(template, bindings)]
    assert specs
    assert {spec["generation"] for spec in specs} == {"G0", "G1", "G2", "G3", "G4", "G5"}
    assert all("${" not in json.dumps(spec) for spec in specs)
    assert all(spec["runner"]["module"].startswith("sequence_models.bibliography_") for spec in specs)
    g0 = next(spec for spec in specs if spec["generation"] == "G0")
    assert g0["runner"]["module"] == "sequence_models.bibliography_evolution_g0_replay"
    assert "--replay-prediction" not in g0["runner"]["argv"]


def test_g0_decodes_excluded_documents_but_headline_mask_does_not() -> None:
    selected = {0, 2}
    assert decoding_document_subset(3, selected, decode_all_documents=False) == {0, 2}
    # G0 reproduction retains document 1 in the recomputed full prediction;
    # only its reported headline metrics use the separate selected mask.
    assert decoding_document_subset(3, selected, decode_all_documents=True) == {0, 1, 2}
    recomputed = [False, True, True]
    headline_only = [recomputed[index] for index in sorted(selected)]
    assert recomputed[1] is True  # excluded document remains nonzero in the byte replay
    assert headline_only == [False, True]


def test_queue_execution_auto_finalizes_immutable_discoverable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, value=0.619)
    row = {
        "work_id": "work-1", "source": "greek_phd", "token_fp": 1, "token_fn": 2,
        "spurious_zero_blocks": 0, "zero_doc_count": 1, "boundary_error_sum": 1.0,
        "boundary_match_count": 1, "document_count": 1,
    }
    baseline_rows = tmp_path / "baseline_rows.jsonl"
    baseline_rows.write_text(json.dumps(row) + "\n", encoding="utf-8")
    table = tmp_path / "table.stub"
    table.write_text("table\n", encoding="utf-8")
    qualified = tmp_path / "qualified.json"
    qualified.write_text("{}\n", encoding="utf-8")
    tests = tmp_path / "code_tests.json"
    tests.write_text(
        json.dumps(
            {
                "status": "passed", "code_commit": "a" * 40,
                "invariants": {"physical_gap_walls": True, "header_roles_non_seed": True},
            }
        ) + "\n",
        encoding="utf-8",
    )
    for name, path, data_class in (
        ("table", table, "development_table"),
        ("qualified", qualified, "qualified_development_inventory"),
        ("baseline_rows", baseline_rows, "baseline_work_objectives"),
        ("code_tests", tests, "code_test_receipt"),
    ):
        spec["input_receipts"][name] = {
            "path": str(path), "sha256": sha256_file(path), "digest_kind": "file_sha256",
            "data_class": data_class, "split": "development",
            "document_scope": "aggregate_no_rows", "contains_labels": False,
        }
    spec = with_candidate_id({key: value for key, value in spec.items() if key != "candidate_id"})
    queue = tmp_path / "queue.jsonl"
    queue.write_bytes(evolution.canonical_json_bytes(spec))
    candidate_root = tmp_path / "executed"

    def fake_command(_spec_value: dict, *, candidate_dir: Path | None = None) -> list[str]:
        assert candidate_dir is not None
        script = (
            "import json,pathlib,numpy as np;"
            f"b=pathlib.Path({str(candidate_dir / 'backend')!r});b.mkdir();"
            "np.save(b/'prediction.npy',np.array([False,True,False,False],dtype=bool),allow_pickle=False);"
            "np.savez(b/'combined_barriers.npz',hard_wall=np.zeros(4,dtype=bool),"
            "upward_stop=np.zeros(4,dtype=bool),downward_stop=np.zeros(4,dtype=bool));"
            "(b/'receipt.json').write_text(json.dumps({'schema_version':'synthetic-backend-v1',"
            "'status':'passed'})+'\\n')"
        )
        return [sys.executable, "-c", script]

    def fake_metrics(args: object) -> dict:
        Path(args.output_rows).write_text(json.dumps(row) + "\n", encoding="utf-8")
        report = {
            "schema_version": "bibliography-evolution-work-objectives-v1", "status": "passed",
            "document_count": 268, "work_count": 1,
            "metrics": {
                "document_count": 268, "token_fp": 1, "token_fn": 2,
                "spurious_blocks_per_zero_block_document": 0.0,
                "mean_boundary_error_emitted_lines": 1.0,
            },
            "metrics_by_source": {"greek_phd": {"document_count": 268}},
        }
        Path(args.output_report).write_text(json.dumps(report) + "\n", encoding="utf-8")
        return report

    import sequence_models.bibliography_evolution_metrics as metrics_module

    monkeypatch.setattr(evolution, "_validate_runner", fake_command)
    monkeypatch.setattr(
        evolution, "_attest_git_checkout",
        lambda _cwd, expected: {"status": "passed", "head": expected, "clean": True},
    )
    monkeypatch.setattr(metrics_module, "run", fake_metrics)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert evolution.main(
        [
            "execute-queue-index", "--queue", str(queue), "--index", "0",
            "--leakage-policy", str(Path(__file__).parents[1] / "evolution/leakage.policy.json"),
            "--candidate-root", str(candidate_root), "--cwd", str(tmp_path),
        ]
    ) == 0
    receipt = candidate_root / spec["candidate_id"] / "receipt.json"
    assert receipt.is_file() and receipt.stat().st_mode & 0o222 == 0
    registry = build_registry([receipt])
    assert registry["candidate_count"] == 1
    assert registry["candidates"][0]["candidate_id"] == spec["candidate_id"]


def test_sealed_batch_rejects_subset_and_reuse(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path / "candidates")
    paths = []
    for index, vector in enumerate(((10, 20, 1, 2), (8, 25, 1, 2))):
        spec = _spec(tmp_path, value=0.2 + index / 100)
        candidate = store.create(spec, _policy())
        paths.append(store.finalize(candidate, _result(candidate, vector)))
    registry = build_registry(paths)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    inventory_path = tmp_path / "sealed_inventory.json"
    inventory_path.write_text(
        json.dumps({"document_ids": [f"sealed-{index:03d}" for index in range(150)]}),
        encoding="utf-8",
    )
    sealed_table = tmp_path / "sealed_table"
    sealed_table.mkdir()
    (sealed_table / "marker").write_text("frozen\n", encoding="utf-8")
    freeze_receipt = tmp_path / "sealed.FROZEN.json"
    freeze_receipt.write_text(
        json.dumps(
            {
                "schema_version": "bibliography-evolution-sealed-inventory-freeze-v1",
                "status": "frozen",
                "labels_sealed": True,
                "document_count": 150,
                "inventory_path": str(inventory_path.resolve()),
                "inventory_sha256": sha256_file(inventory_path),
                "sealed_table_path": str(sealed_table.resolve()),
                "sealed_table_tree_sha256": sha256_directory(sealed_table),
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    _freeze_manifest(registry_path, manifest_path, inventory_path, freeze_receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_inputs = {}
    for candidate_id in manifest["candidate_ids"]:
        path = tmp_path / f"{candidate_id}.prediction.npy"
        path.write_bytes(candidate_id.encode("utf-8"))
        prediction_inputs[candidate_id] = {"path": str(path), "sha256": sha256_file(path)}
    request = {
        "schema_version": SEALED_REQUEST_SCHEMA,
        "evaluation_mode": "one_simultaneous_batch_all_pareto_candidates",
        "frozen_manifest_id": manifest["frozen_manifest_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "sealed_inventory_sha256": sha256_file(inventory_path),
        "sealed_freeze_receipt_sha256": sha256_file(freeze_receipt),
        "candidate_ids": manifest["candidate_ids"][:1],
        "prediction_inputs": prediction_inputs,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ContractError, match="subset"):
        _begin_sealed_batch(manifest_path, request_path, tmp_path / "batches")
    request["candidate_ids"] = manifest["candidate_ids"]
    request_path.write_text(json.dumps(request), encoding="utf-8")
    _begin_sealed_batch(manifest_path, request_path, tmp_path / "batches")
    with pytest.raises(FileExistsError):
        _begin_sealed_batch(manifest_path, request_path, tmp_path / "batches")
