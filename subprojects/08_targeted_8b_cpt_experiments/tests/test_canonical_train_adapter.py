from __future__ import annotations

# ruff: noqa: E402 -- test imports intentionally follow local path injection.

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EVALUATION))

import bind_native_suite_checkpoint as native_binder
from adopt_canonical_pre_main_data import (
    portable_manifest_gate,
    tokenized_token_count,
    training_phases,
)
from aggregate_frozen_greekmmlu import choice_nll as greekmmlu_choice_nll
from aggregate_frozen_greekmmlu import metrics as greekmmlu_metrics
from audit_training_checkpoint import validate_claim_window
from checkpoint_export_contract import expected_source_keys, validate_geometry
from checkpoint_export_receipt import parse_runtime_parity
from evaluate_td_objective import selected_probe_ids
from finalize_legacy_public_greekmmlu import (
    REFERENCE_ACCURACY,
    equivalence_decision,
    wilson_interval,
)
from materialize_intermediate_checkpoint_branch import hardlink_tree
from preflight_train_segment import validate_segment_boundaries
from run_canonical_train_segment import (
    LATE_BOUND_PHASE_CACHE,
    bridge_triggers,
    load_checkpoint_reference,
    resolve_phase_cache_arguments,
    validate_profile_qualification_contract,
)
from run_checkpoint_export_evaluator import resolve_checkpoint_root
from run_greekmmlu_evaluator import mode_for_iteration
from run_offline_panels_evaluator import resolve_export, validate_panels
from score_frozen_greekmmlu_shard import authority_rows, shard_rows


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def binding(path: Path) -> dict[str, object]:
    import hashlib

    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_intermediate_branch_is_a_no_copy_checkpoint_view(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "nested").mkdir(parents=True)
    payload = source / "nested/state.distcp"
    payload.write_bytes(b"checkpoint")
    count, total = hardlink_tree(source, target)
    linked = target / "nested/state.distcp"
    assert (count, total) == (1, len(b"checkpoint"))
    assert linked.stat().st_ino == payload.stat().st_ino


def test_phase_cache_arguments_are_exact_by_default() -> None:
    args = SimpleNamespace(
        phase=1,
        start_update=0,
        phase_data_path="1.0 /cache/hplt",
        phase_cache_tree_sha256="a" * 64,
    )
    assert resolve_phase_cache_arguments(args) == (
        "1.0 /cache/hplt",
        "a" * 64,
    )


def test_phase3_cache_arguments_are_resolved_from_the_future_receipt(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "phase_3_data_path.json"
    write_json(spec, {"phase": 3})
    cache_root = tmp_path / "phase_3_cache"
    cache_root.mkdir()
    receipt = tmp_path / "phase_3_blend_cache.json"
    write_json(
        receipt,
        {
            "schema_version": "apertus_hard_h_to_g_phase_blend_cache_v1",
            "status": "frozen",
            "phase": 3,
            "cache_root": str(cache_root),
            "data_path_spec": binding(spec),
            "data_path_tokens": ["1.0", "/cache/open archives", "0.25", "/cache/foreign"],
            "cache_tree_sha256": "b" * 64,
        },
    )
    args = SimpleNamespace(
        phase=3,
        start_update=3218,
        phase_data_path=LATE_BOUND_PHASE_CACHE,
        phase_cache_tree_sha256=LATE_BOUND_PHASE_CACHE,
        phase_cache_receipt=receipt,
        phase_cache_root=cache_root,
        phase_data_path_spec=spec,
    )
    data_path, cache_sha256 = resolve_phase_cache_arguments(args)
    assert data_path == "1.0 '/cache/open archives' 0.25 /cache/foreign"
    assert cache_sha256 == "b" * 64


def test_late_bound_phase_cache_is_forbidden_before_phase3(tmp_path: Path) -> None:
    args = SimpleNamespace(
        phase=2,
        start_update=2261,
        phase_data_path=LATE_BOUND_PHASE_CACHE,
        phase_cache_tree_sha256=LATE_BOUND_PHASE_CACHE,
        phase_cache_receipt=tmp_path / "missing.json",
        phase_cache_root=tmp_path,
        phase_data_path_spec=tmp_path / "missing-spec.json",
    )
    with pytest.raises(ValueError, match="only at a Phase-3 segment boundary"):
        resolve_phase_cache_arguments(args)


def test_checkpoint_reference_binds_resume_inputs(tmp_path: Path) -> None:
    load_root = tmp_path / "checkpoints"
    checkpoint = load_root / "iter_0000238"
    checkpoint.mkdir(parents=True)
    permit = tmp_path / "permit.json"
    cache = tmp_path / "cache.json"
    write_json(permit, {"status": "passed"})
    write_json(cache, {"status": "passed"})
    reference = tmp_path / "reference.json"
    write_json(
        reference,
        {
            "schema_version": "apertus_hard_h_to_g_checkpoint_reference_v1",
            "status": "passed",
            "scale": "1p5b",
            "update": 238,
            "load_root": str(load_root),
            "checkpoint_root": str(checkpoint),
            "checkpoint_permit": binding(permit),
            "source_phase_cache_receipt": binding(cache),
        },
    )

    assert (
        load_checkpoint_reference(reference, scale="1p5b", update=238)["update"] == 238
    )
    permit.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_permit drift"):
        load_checkpoint_reference(reference, scale="1p5b", update=238)


def test_adoption_uses_frozen_index_token_field() -> None:
    assert (
        tokenized_token_count({"index": {"documents": 7, "tokens_including_eod": 19}})
        == 19
    )
    with pytest.raises(ValueError, match="token count missing"):
        tokenized_token_count({"index": {"documents": 7}})


def test_canonical_data_manifest_covers_the_full_three_phase_horizon() -> None:
    phases = training_phases()
    assert [row["id"] for row in phases] == [
        "phase_1_hplt",
        "phase_2_openarchives",
        "phase_3_unseen_openarchives",
    ]
    for row in phases:
        assert sum(row["dataset_weights"].values()) == pytest.approx(1.0)
    assert phases[1]["dataset_weights"] == phases[2]["dataset_weights"]


def test_canonical_adoption_wrapper_requires_immutable_runner_receipt_and_v2_root() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "clariden/adopt_canonical_pre_main_data_debug.sbatch"
    ).read_text(encoding="utf-8")
    assert "APERTUS_EFFICIENCY_CODE_RECEIPT:?" in wrapper
    assert '--root "$APERTUS_EFFICIENCY_ROOT"' in wrapper
    assert '--receipt "$APERTUS_EFFICIENCY_CODE_RECEIPT" --kind efficiency' in wrapper
    assert "canonical/pre_main_data_v2" in wrapper
    assert '--canonical-code-receipt "$APERTUS_EFFICIENCY_CODE_RECEIPT"' in wrapper
    assert "APERTUS_EFFICIENCY_COMMIT" not in wrapper
    assert "b940fd135e07f41126fc6ef00c174d42db0d4f5e" not in wrapper


@pytest.mark.parametrize(
    ("scale", "layers", "hidden", "expected_tensors"),
    (("8b", 32, 4096, 323), ("1p5b", 16, 2048, 163)),
)
def test_checkpoint_mapping_geometry_is_scale_aware(
    scale: str, layers: int, hidden: int, expected_tensors: int
) -> None:
    contract = {
        "models": {
            scale: {
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "tie_word_embeddings": False,
            }
        },
        "training": {
            "rope_theta": 500000,
            "max_position_embeddings": 4096,
            "rope_scaling_factor": 8.0,
        },
    }
    config = {
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "tie_word_embeddings": False,
        "vocab_size": 148480,
        "rope_theta": 500000.0,
        "max_position_embeddings": 4096,
        "rope_scaling": {
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "rope_type": "llama3",
        },
    }
    observed = validate_geometry(config, contract, scale=scale, true_vocab_size=148480)
    assert observed["hidden_size"] == hidden
    assert len(expected_source_keys(layers, tied=False)) == expected_tensors
    assert "output_layer.weight" in expected_source_keys(layers, tied=False)


def test_checkpoint_mapping_rejects_wrong_rope_geometry() -> None:
    contract = {
        "models": {
            "1p5b": {
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "tie_word_embeddings": False,
            }
        },
        "training": {
            "rope_theta": 500000,
            "max_position_embeddings": 4096,
            "rope_scaling_factor": 8.0,
        },
    }
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "tie_word_embeddings": False,
        "vocab_size": 148480,
        "rope_theta": 12000000.0,
        "max_position_embeddings": 4096,
        "rope_scaling": {
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "rope_type": "llama3",
        },
    }
    with pytest.raises(ValueError, match="RoPE theta"):
        validate_geometry(config, contract, scale="1p5b", true_vocab_size=148480)


def test_checkpoint_export_runtime_parity_requires_float32_and_thresholds() -> None:
    log = (
        "Converted model agrees on 100.0% of predictions\n"
        "Converted logits are close on 99.0% of values\n"
        "Converted semantic parity dtype: float32"
    )
    assert parse_runtime_parity(log)["runtime_semantic_parity_passed"] is True
    with pytest.raises(ValueError, match="float32"):
        parse_runtime_parity(log.replace("float32", "bfloat16"))


def test_export_evaluator_resolves_one_exact_checkpoint_reference(
    tmp_path: Path,
) -> None:
    load_root = tmp_path / "payload/checkpoints"
    checkpoint = load_root / "iter_0000238"
    checkpoint.mkdir(parents=True)
    permit = tmp_path / "permit.json"
    cache = tmp_path / "cache.json"
    write_json(permit, {"status": "passed"})
    write_json(cache, {"status": "passed"})
    reference = tmp_path / "segments/s0/attempts/0/checkpoint_reference.json"
    reference.parent.mkdir(parents=True)
    write_json(
        reference,
        {
            "schema_version": "apertus_hard_h_to_g_checkpoint_reference_v1",
            "status": "passed",
            "scale": "8b",
            "update": 238,
            "load_root": str(load_root),
            "checkpoint_root": str(checkpoint),
            "checkpoint_permit": binding(permit),
            "source_phase_cache_receipt": binding(cache),
        },
    )
    assert (
        resolve_checkpoint_root(tmp_path, scale="8b", iteration=238)
        == load_root.resolve()
    )


def test_offline_panel_evaluator_resolves_one_verified_export(tmp_path: Path) -> None:
    hf_root = tmp_path / "hf"
    hf_root.mkdir()
    export_receipt = tmp_path / "export/checkpoint_export_receipt.json"
    export_receipt.parent.mkdir()
    write_json(
        export_receipt,
        {
            "schema_version": "apertus_hard_h_to_g_checkpoint_export_v1",
            "status": "completed",
            "ready_for_frozen_evaluators": True,
            "hf_export": {"path": str(hf_root)},
        },
    )
    result = tmp_path / "evaluations/export/result.json"
    result.parent.mkdir(parents=True)
    write_json(
        result,
        {
            "schema_version": "apertus_hard_h_to_g_checkpoint_export_evaluation_v1",
            "status": "completed",
            "scale": "1p5b",
            "iteration": 476,
            "checkpoint_export": binding(export_receipt),
        },
    )
    assert resolve_export(tmp_path, scale="1p5b", iteration=476) == (
        hf_root.resolve(),
        export_receipt.resolve(),
    )


def test_offline_panel_receipts_bind_all_thirteen_inputs_and_outputs(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "panels"
    output.mkdir()
    rows = []
    for index in range(13):
        name = f"panel_{index}"
        raw = tmp_path / f"{name}.jsonl"
        raw.write_text('{"text":"x"}\n', encoding="utf-8")
        raw_binding = binding(raw) | {"rows": 1}
        rows.append({"name": name, "raw_jsonl": raw_binding})
        documents = output / f"{name}.documents.jsonl"
        documents.write_text('{"doc_id":"1"}\n', encoding="utf-8")
        receipt = output / f"{name}.receipt.json"
        write_json(
            receipt,
            {
                "schema_version": "apertus_per_document_validation_v1",
                "status": "completed",
                "model": str(model),
                "input": {key: raw_binding[key] for key in ("path", "bytes", "sha256")},
                "output": binding(documents) | {"rows": 1},
                "aggregate": {
                    "documents": 1,
                    "target_tokens": 1,
                    "utf8_bytes": 1,
                },
            },
        )
    manifest = tmp_path / "validation.json"
    write_json(manifest, {"panels": rows})
    observed = validate_panels(manifest, output, model=model.resolve())
    assert [row["name"] for row in observed] == [
        f"panel_{index}" for index in range(13)
    ]


def test_native_suite_rebind_preserves_examples_and_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = tmp_path / "examples.jsonl"
    examples.write_text('{"benchmark":"demosqa","example_id":"1"}\n', encoding="utf-8")
    source_contract = tmp_path / "source_contract.json"
    write_json(
        source_contract,
        {
            "schema_version": "apertus_full8_native_greek_3cp_contract_v1",
            "checkpoint_scope": [],
            "model_contract": {},
            "scoring": {"method": "frozen"},
            "benchmarks": [{"id": "demosqa"}],
        },
    )
    source_manifest = tmp_path / "source_manifest.json"
    write_json(
        source_manifest,
        {
            "contract": {"sha256": binding(source_contract)["sha256"]},
            "examples": {
                "path": str(examples),
                "sha256": binding(examples)["sha256"],
                "rows": 1,
            },
        },
    )
    code_receipt = tmp_path / "eval_code.json"
    write_json(
        code_receipt,
        {
            "schema_version": "native_greek_eval_code_bundle_v1",
            "status": "frozen",
            "tree_sha256": "a" * 64,
        },
    )
    source_gate = tmp_path / "source_gate.json"
    write_json(
        source_gate,
        {
            "schema_version": "apertus_full8_native_greek_execution_gate_v1",
            "status": "passed",
            "contract_sha256": binding(source_contract)["sha256"],
            "manifest_sha256": binding(source_manifest)["sha256"],
            "code_tree_sha256": "a" * 64,
            "selected": native_binder.SELECTED_PROFILE,
        },
    )
    model = tmp_path / "hf"
    model.mkdir()
    write_json(
        model / "config.json",
        {
            "vocab_size": 148480,
            "rope_theta": 500000.0,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": False,
        },
    )
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        native_binder, "TOKENIZER_HASHES", [binding(model / "tokenizer.json")["sha256"]]
    )
    export = tmp_path / "export.json"
    write_json(
        export,
        {
            "schema_version": "apertus_hard_h_to_g_checkpoint_export_v1",
            "status": "completed",
            "scale": "8b",
            "iteration": 238,
            "ready_for_frozen_evaluators": True,
            "hf_export": {"path": str(model)},
        },
    )
    output = tmp_path / "bound"
    argv = [
        "bind_native_suite_checkpoint.py",
        "--source-contract",
        str(source_contract),
        "--source-manifest",
        str(source_manifest),
        "--source-execution-gate",
        str(source_gate),
        "--eval-code-receipt",
        str(code_receipt),
        "--checkpoint-export",
        str(export),
        "--scale",
        "8b",
        "--iteration",
        "238",
        "--output-dir",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert native_binder.main() == 0
    rebound = json.loads((output / "rebind_receipt.json").read_text())
    assert rebound["status"] == "passed" and all(rebound["checks"].values())


def test_greekmmlu_mode_is_predeclared_and_full_at_calibration_updates() -> None:
    assert mode_for_iteration(0) == "full_clean"
    assert mode_for_iteration(714) == "full_clean"
    assert mode_for_iteration(952) == "sentinel_pair"
    assert mode_for_iteration(3218) == "full_clean"
    assert mode_for_iteration(3456) == "sentinel_pair"
    assert mode_for_iteration(3694) == "full_clean"
    with pytest.raises(ValueError, match="not predeclared"):
        mode_for_iteration(1000)


def test_greekmmlu_shards_partition_without_loss_or_overlap() -> None:
    rows = [{"example_id": str(index)} for index in range(101)]
    partitions = [
        shard_rows(rows, shard_index=index, shard_count=16) for index in range(16)
    ]
    observed = [row["example_id"] for shard in partitions for row in shard]
    assert len(observed) == len(set(observed)) == len(rows)
    assert set(observed) == {row["example_id"] for row in rows}


def test_greekmmlu_metrics_preserve_continuous_and_accuracy_signals() -> None:
    rows = [
        {
            "correct": True,
            "answer_index": 0,
            "choice_scores": [
                {"avg_logprob": -0.1, "sum_logprob": -0.2},
                {"avg_logprob": -1.0, "sum_logprob": -2.0},
            ],
            "correct_answer_utf8_bytes": 2,
        },
        {
            "correct": False,
            "answer_index": 1,
            "choice_scores": [
                {"avg_logprob": -0.2, "sum_logprob": -0.4},
                {"avg_logprob": -0.3, "sum_logprob": -0.6},
            ],
            "correct_answer_utf8_bytes": 3,
        },
    ]
    observed = greekmmlu_metrics(rows)
    assert observed["n"] == 2 and observed["accuracy"] == 0.5
    assert observed["choice_nll"] == pytest.approx(
        sum(greekmmlu_choice_nll(row) for row in rows) / 2
    )
    assert observed["correct_answer_bpb"] > 0


def test_greekmmlu_debug_wrapper_fixes_float32_batch1_and_four_nodes() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "clariden/run_frozen_greekmmlu_4node_debug.sbatch").read_text(
        encoding="utf-8"
    )
    shard = (root / "clariden/run_greekmmlu_shard.sh").read_text(encoding="utf-8")
    assert "#SBATCH --partition=debug" in wrapper
    assert "#SBATCH --nodes=4" in wrapper
    assert "--ntasks=16" in wrapper
    assert "--candidate-batch-size 1" in shard
    assert 'case "${SLURM_JOB_PARTITION:-}" in' in wrapper
    assert "debug|normal" in wrapper


def test_full_public_examples_are_accepted_as_scoring_authority(tmp_path: Path) -> None:
    path = tmp_path / "public.json"
    rows = [
        {
            "example_id": f"greekmmlu:{index}",
            "row_index": index,
            "subject": "subject",
            "educational_level": None,
        }
        for index in range(16_632)
    ]
    write_json(
        path,
        {
            "schema_version": "apertus_greekmmlu_public_examples_v1",
            "status": "frozen",
            "dataset": {
                "repo_id": "dascim/GreekMMLU",
                "revision": "6a03aa06b68beb932fb75edff3a34e50b3674649",
                "config": "All",
                "split": "test",
            },
            "public_count": 16_632,
            "examples": rows,
        },
    )
    assert len(authority_rows(path)) == 16_632


def test_legacy_public_equivalence_gate_is_owner_ratified_one_point_band() -> None:
    assert equivalence_decision(REFERENCE_ACCURACY) == "pass"
    assert equivalence_decision(REFERENCE_ACCURACY - 0.01) == "pass"
    assert equivalence_decision(REFERENCE_ACCURACY + 0.01) == "pass"
    assert equivalence_decision(REFERENCE_ACCURACY - 0.01001) == "fail"


def test_legacy_public_wrapper_is_single_debug_node_and_exact_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "clariden/run_legacy_public_greekmmlu_debug.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=debug" in wrapper
    assert "#SBATCH --nodes=1" in wrapper
    assert 'case "$H2G_ITERATION" in' in wrapper
    assert "2618|3218|3694" in wrapper
    assert "--dtype bfloat16" in wrapper
    assert "--candidate-batch-size 16" in wrapper


def test_adoption_rebinds_manifest_gate_out_of_transactional_root(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "training_data_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    gate = {"manifest": binding(manifest)}
    assert portable_manifest_gate(gate, manifest, tmp_path)["manifest"]["path"] == (
        "training_data_manifest.json"
    )


def test_td_objective_probe_selection_is_hash_stable_and_not_prefix_based() -> None:
    token_ids = list(range(131_072, 131_200))
    first = selected_probe_ids(token_ids, salt="probe-v1", count=32)
    second = selected_probe_ids(list(reversed(token_ids)), salt="probe-v1", count=32)
    assert first == second
    assert first != token_ids[:32]
    assert len(first) == len(set(first)) == 32


def test_trigger_bridge_translates_canonical_stop(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    payload = tmp_path / "payload"
    canonical.mkdir()
    (canonical / "save").touch()
    (canonical / "exit").touch()
    stop = threading.Event()
    thread = threading.Thread(target=bridge_triggers, args=(canonical, payload, stop))
    thread.start()
    for _ in range(50):
        if (payload / "triggers/save").exists() and (
            payload / "triggers/exit"
        ).exists():
            break
        stop.wait(0.02)
    stop.set()
    thread.join(timeout=2)
    assert (payload / "triggers/save").is_file()
    assert (payload / "triggers/exit").is_file()


def test_trigger_bridge_reasserts_stop_after_trainer_startup_clears_it(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    payload = tmp_path / "payload"
    canonical.mkdir()
    (canonical / "save").touch()
    (canonical / "exit").touch()
    stop = threading.Event()
    thread = threading.Thread(target=bridge_triggers, args=(canonical, payload, stop))
    thread.start()
    for _ in range(50):
        if (payload / "triggers/save").exists() and (
            payload / "triggers/exit"
        ).exists():
            break
        stop.wait(0.02)
    (payload / "triggers/save").unlink()
    (payload / "triggers/exit").unlink()
    for _ in range(50):
        if (payload / "triggers/save").exists() and (
            payload / "triggers/exit"
        ).exists():
            break
        stop.wait(0.02)
    stop.set()
    thread.join(timeout=2)
    assert (payload / "triggers/save").is_file()
    assert (payload / "triggers/exit").is_file()


def test_graceful_checkpoint_must_be_strictly_inside_claim() -> None:
    preflight = {"start_update": 0, "exit_update": 952}
    assert validate_claim_window(
        preflight, observed_update=700, allow_graceful_stop=True,
        allow_intermediate_save=False,
    ) == (0, 952)
    with pytest.raises(ValueError, match="outside"):
        validate_claim_window(
            preflight, observed_update=952, allow_graceful_stop=True,
            allow_intermediate_save=False,
        )


def test_intermediate_save_can_anchor_stable_peak_branch() -> None:
    preflight = {"start_update": 2261, "exit_update": 3218}
    assert validate_claim_window(
        preflight, observed_update=2499, allow_graceful_stop=False,
        allow_intermediate_save=True,
    ) == (2261, 3218)
    validate_segment_boundaries(
        phase=2,
        start_update=2499,
        exit_update=3218,
        one_update_resume_smoke=False,
        canonical_resume=False,
        lr_policy="stable_peak",
    )


def test_canonical_resume_keeps_frozen_segment_endpoint() -> None:
    validate_segment_boundaries(
        phase=1,
        start_update=700,
        exit_update=952,
        one_update_resume_smoke=False,
        canonical_resume=True,
        lr_policy="matched_wsd",
    )
    with pytest.raises(ValueError, match="canonical resume"):
        validate_segment_boundaries(
            phase=1,
            start_update=700,
            exit_update=900,
            one_update_resume_smoke=False,
            canonical_resume=True,
            lr_policy="matched_wsd",
        )
    with pytest.raises(ValueError, match="crosses"):
        validate_segment_boundaries(
            phase=2,
            start_update=2200,
            exit_update=2380,
            one_update_resume_smoke=False,
            canonical_resume=True,
            lr_policy="matched_wsd",
        )


def test_runtime_qualification_contract_is_exact_profile_and_data_bound(
    tmp_path: Path,
) -> None:
    initialization = tmp_path / "init"
    cache_root = tmp_path / "cache"
    initialization.mkdir()
    cache_root.mkdir()
    cache_receipt = tmp_path / "cache.json"
    write_json(cache_receipt, {"status": "passed"})
    args = SimpleNamespace(
        phase=1,
        start_update=0,
        end_update=952,
        scale="8b",
        tensor_parallel=2,
        microbatch=2,
        peak_lr="5.5e-5",
        floor_lr="5.5e-6",
        load_checkpoint=initialization,
        phase_cache_receipt=cache_receipt,
        phase_cache_root=cache_root,
        phase_cache_tree_sha256="a" * 64,
        phase_data_path="1.0 /prepared/hplt",
    )
    contract = {
        "schema_version": "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1",
        "status": "frozen",
        "kind": "profile",
        "scale": "8b",
        "updates": 256,
        "nodes": 16,
        "tensor_parallel": 2,
        "microbatch": 2,
        "peak_lr": "5.5e-5",
        "floor_lr": "5.5e-6",
        "initialization_root": str(initialization),
        "phase_cache_receipt": binding(cache_receipt),
        "phase_cache_root": str(cache_root),
        "phase_cache_tree_sha256": "a" * 64,
        "phase_data_path": "1.0 /prepared/hplt",
        "output_root": str(tmp_path / "qualification"),
    }
    assert (
        validate_profile_qualification_contract(args, contract, live_nodes=16)
        == tmp_path / "qualification"
    )
    contract["phase_cache_tree_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="cache hash drift"):
        validate_profile_qualification_contract(args, contract, live_nodes=16)
