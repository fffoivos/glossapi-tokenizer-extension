from __future__ import annotations

import copy

import pytest

from sequence_models.bibliography_evolution_finalize_g2 import (
    EXPECTED_G2_CODE_COMMIT,
    EXPECTED_G2_RECEIPTS,
    EXPECTED_PARENT_ID,
    select_g2_parent,
)


OBJECTIVES = {
    "token_fp": 101547,
    "token_fn": 39053,
    "spurious_blocks_per_zero_block_document": 1 / 27,
    "mean_boundary_error_emitted_lines": 2.570287539936102,
}


def _queue() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate_id,
            "generation": "G2",
            "code_commit": EXPECTED_G2_CODE_COMMIT,
            "parent_candidate_ids": [EXPECTED_PARENT_ID],
            "sweep_point": {"header_window": index},
        }
        for index, candidate_id in enumerate(EXPECTED_G2_RECEIPTS, start=1)
    ]


def _registry() -> dict[str, object]:
    rows = [{"candidate_id": EXPECTED_PARENT_ID, "objective_vector": OBJECTIVES}]
    for index, candidate_id in enumerate(EXPECTED_G2_RECEIPTS, start=1):
        objective = copy.deepcopy(OBJECTIVES)
        objective["token_fp"] -= 4000 + index
        objective["token_fn"] += 113_000 + index
        objective["mean_boundary_error_emitted_lines"] -= 0.1
        rows.append({"candidate_id": candidate_id, "objective_vector": objective})
    return {"candidates": rows}


def test_no_g2_candidate_is_promoted_when_recall_objective_worsens() -> None:
    result = select_g2_parent(_registry(), _queue())
    assert result["qualifying_weak_dominators"] == []
    assert result["promoted_g2_candidate_id"] is None
    assert result["g3_authorized"] is False


def test_finalizer_fails_closed_if_a_g2_candidate_weakly_dominates() -> None:
    registry = _registry()
    registry["candidates"][1]["objective_vector"] = {
        **OBJECTIVES,
        "token_fp": OBJECTIVES["token_fp"] - 1,
    }
    with pytest.raises(RuntimeError, match="weakly dominates"):
        select_g2_parent(registry, _queue())


def test_queue_inventory_and_parent_are_frozen() -> None:
    queue = _queue()
    queue[0]["parent_candidate_ids"] = ["g1-wrong"]
    with pytest.raises(RuntimeError, match="lineage drift"):
        select_g2_parent(_registry(), queue)
