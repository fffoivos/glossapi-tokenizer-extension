from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "successor_semantic_identity",
        ROOT / "scripts/successor_semantic_identity.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_gate_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "build_successor_launch_gate",
        ROOT / "scripts/build_successor_launch_gate.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe() -> dict:
    return {
        "recipe_id": "full8b-mixed-79-20-1-wsd10-sanitized-v1",
        "data": {
            "sanitized_source_receipt": {"path": "/old/pool", "sha256": "old"},
            "eligibility_policy": {"openarchives_needs_ocr_true": "excluded", "proof": {"path": "/old/audit"}},
            "sequence_length": 4096,
            "target_mix": {"modern_greek": "0.79", "foreign_replay": "0.20", "old_greek_replay": "0.01"},
        },
        "tokenizer": {"sha256": "tokenizer"},
        "initialization": {"sha256": "initialization"},
        "model": {"rope": {"base": 500000}},
        "optimization": {"peak_lr": 5.5e-5, "schedule": "WSD10"},
        "batch_and_parallelism": {
            "global_batch_sequences": 1024,
            "global_batch_tokens": 4194304,
            "micro_batch_sequences": 2,
            "training_updates": 18284,
            "training_samples": 18722816,
            "tensor_parallel": 2,
            "pipeline_parallel": 1,
            "context_parallel": 1,
        },
        "evaluation": {"greekmmlu": {"checkpoint_updates": [0, 18284]}},
        "software": {"megatron": "pinned"},
    }


def test_receipt_only_rebinding_keeps_semantic_identity() -> None:
    module = load_module()
    source = recipe()
    successor = copy.deepcopy(source)
    successor["data"]["sanitized_source_receipt"] = {"path": "/new/pool", "sha256": "new"}
    successor["data"]["eligibility_policy"]["proof"] = {"path": "/new/audit"}
    assert module.semantic_identity(source) == module.semantic_identity(successor)
    assert module.normalized_recipe_payload(source) == module.normalized_recipe_payload(successor)


def test_learning_rate_or_rope_change_is_not_normalized_away() -> None:
    module = load_module()
    source = recipe()
    changed_lr = copy.deepcopy(source)
    changed_lr["optimization"]["peak_lr"] = 6e-5
    changed_rope = copy.deepcopy(source)
    changed_rope["model"]["rope"]["base"] = 12000000
    assert module.semantic_identity(source) != module.semantic_identity(changed_lr)
    assert module.semantic_identity(source) != module.semantic_identity(changed_rope)


def test_profile_derivation_is_receipt_only_but_geometry_is_not() -> None:
    module = load_module()
    source = {"scientific_invariants": {"training_updates": 18284}, "profiles": {"dp32": {"nodes": 16}}, "derivation": {"path": "/old"}}
    successor = copy.deepcopy(source)
    successor["derivation"] = {"path": "/new"}
    assert module.normalized_profiles(source) == module.normalized_profiles(successor)
    successor["profiles"]["dp32"]["nodes"] = 32
    assert module.normalized_profiles(source) != module.normalized_profiles(successor)


def test_restart_control_is_derived_only_from_a_passing_bound_parity_receipt() -> None:
    module = load_gate_module()
    parity = {
        "schema_version": "apertus_full_8b_checkpoint_parity_smoke_v1",
        "status": "passed",
        "profile_id": "dp32_16node",
        "checks": {
            "first_restart_provenance": True,
            "second_restart_provenance": True,
            "first_restart_numerically_equivalent": True,
            "second_restart_numerically_equivalent": True,
            "independent_restarts_identical": True,
        },
        "restart": {
            "first": {"provenance": {"passed": True}, "numerical": {"passed": True}},
            "second": {"provenance": {"passed": True}, "numerical": {"passed": True}},
            "independent_identity": {"passed": True},
        },
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "parity.json"
        path.write_text(json.dumps(parity), encoding="utf-8")
        promotion = {
            "parity": {
                "receipt": module.file_binding(path),
                "two_independent_restart_allocations_passed": True,
            }
        }
        control, binding = module.restart_control_from_promotion(promotion)
    assert control["two_independent_restart_allocations_passed"] is True
    assert control["provenance"]["passed"] is True
    assert control["independent_repeat"]["numerical"]["passed"] is True
    assert binding["sha256"] == promotion["parity"]["receipt"]["sha256"]
