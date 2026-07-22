from __future__ import annotations

import copy

import pytest

from sequence_models.bibliography_evolution_finalize_g1_full import (
    EXPECTED_PARENT_ID,
    select_full_g1_parent,
)


def _fixtures() -> tuple[dict, list[dict]]:
    g0 = {
        "candidate_id": "g0-control",
        "generation": "G0",
        "objective_vector": {
            "token_fp": 10,
            "token_fn": 20,
            "spurious_blocks_per_zero_block_document": 0.1,
            "mean_boundary_error_emitted_lines": 2.0,
        },
    }
    control = copy.deepcopy(g0)
    control.update(candidate_id=EXPECTED_PARENT_ID, generation="G1")
    candidates = [g0, control]
    queue = [
        {
            "candidate_id": EXPECTED_PARENT_ID,
            "sweep_point": {"anchor_probability": 0.3},
        }
    ]
    for index in range(26):
        candidate_id = f"g1-child-{index:02d}"
        child = copy.deepcopy(control)
        child["candidate_id"] = candidate_id
        child["objective_vector"]["token_fp"] -= index + 1
        child["objective_vector"]["token_fn"] += index + 1
        candidates.append(child)
        queue.append({"candidate_id": candidate_id, "sweep_point": {"x": index}})
    return {"candidates": candidates}, queue


def test_select_full_g1_parent_keeps_control_without_weak_dominator() -> None:
    registry, queue = _fixtures()
    result = select_full_g1_parent(registry, queue)
    assert result["reference_candidate_id"] == EXPECTED_PARENT_ID
    assert result["qualifying_weak_dominators"] == []


def test_select_full_g1_parent_fails_closed_on_weak_dominator() -> None:
    registry, queue = _fixtures()
    registry["candidates"][2]["objective_vector"]["token_fn"] = 19
    with pytest.raises(RuntimeError, match="weakly dominates"):
        select_full_g1_parent(registry, queue)
