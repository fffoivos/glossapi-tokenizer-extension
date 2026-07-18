from __future__ import annotations

import numpy as np
import pytest

from sequence_models.bibliography_evolution_contract import FIXED_MODULE_ORDER
from sequence_models.bibliography_evolution_composition import (
    combine_parent_barriers,
    enforce_combined_barriers,
)
from sequence_models.bibliography_evolution_headers import (
    ROLE_TO_ID,
    HeaderControllerConfig,
    apply_header_controller,
    assert_header_invariants,
    header_actions,
)
from sequence_models.bibliography_evolution_postprocess import _postprocess_document


def test_agreed_module_order_connects_before_trim_and_attaches_before_veto() -> None:
    assert FIXED_MODULE_ORDER.index("internal_gap_connection") < FIXED_MODULE_ORDER.index("boundary_trim")
    assert FIXED_MODULE_ORDER.index("boundary_trim") < FIXED_MODULE_ORDER.index("bib_header_attachment")
    assert FIXED_MODULE_ORDER.index("bib_header_attachment") < FIXED_MODULE_ORDER.index("whole_component_veto")


def test_heading_roles_have_distinct_directional_non_seed_actions() -> None:
    roles = np.asarray([
        ROLE_TO_ID["BIB_HEADER"], ROLE_TO_ID["BIB_SUBHEADER"], ROLE_TO_ID["NON_BIB_HEADER"]
    ])
    actions = header_actions(roles)
    assert actions.non_seed.tolist() == [True, True, True]
    assert actions.hard_upward_stop.tolist() == [True, False, False]
    assert actions.connector.tolist() == [False, True, False]
    assert actions.hard_downward_stop.tolist() == [False, False, True]
    assert actions.excluded.tolist() == [False, False, True]


def test_headers_cannot_seed_but_main_attaches_and_subheader_connects() -> None:
    absolute = np.arange(8)
    roles = np.zeros(8, dtype=np.uint8)
    roles[1] = ROLE_TO_ID["BIB_HEADER"]
    roles[4] = ROLE_TO_ID["BIB_SUBHEADER"]
    core = np.zeros(8, dtype=bool)
    core[1] = True  # malformed upstream seed is stripped
    core[3] = True
    core[5] = True
    result = apply_header_controller(
        core, roles, absolute, config=HeaderControllerConfig(attachment_window=3, connector_window=2)
    )
    assert result[1:6].all()
    header_only = np.zeros(8, dtype=bool)
    header_only[1] = True
    assert not apply_header_controller(header_only, roles, absolute)[1]


def test_non_bib_header_is_never_swallowed() -> None:
    absolute = np.arange(5)
    roles = np.zeros(5, dtype=np.uint8)
    roles[2] = ROLE_TO_ID["NON_BIB_HEADER"]
    core = np.ones(5, dtype=bool)
    result = apply_header_controller(core, roles, absolute)
    assert not result[2]
    assert_header_invariants(result, roles)
    with pytest.raises(ValueError, match="NON_BIB_HEADER"):
        assert_header_invariants(core, roles)


def test_subheader_does_not_connect_across_physical_gap() -> None:
    absolute = np.asarray([0, 1, 500, 501, 502])
    roles = np.zeros(5, dtype=np.uint8)
    roles[2] = ROLE_TO_ID["BIB_SUBHEADER"]
    core = np.asarray([False, True, False, True, False])
    result = apply_header_controller(core, roles, absolute)
    assert not result[2]


def test_directional_headers_trim_preexisting_overreach() -> None:
    absolute = np.arange(9)
    roles = np.zeros(9, dtype=np.uint8)
    roles[3] = ROLE_TO_ID["BIB_HEADER"]
    core = np.zeros(9, dtype=bool)
    core[1:7] = True
    result = apply_header_controller(core, roles, absolute)
    assert not result[1:3].any()
    assert result[3:7].all()

    roles[:] = 0
    roles[5] = ROLE_TO_ID["NON_BIB_HEADER"]
    result = apply_header_controller(core, roles, absolute)
    assert result[1:5].all()
    assert not result[5:7].any()


def test_bib_header_trims_above_when_block_below_has_one_line_gap() -> None:
    absolute = np.arange(8)
    roles = np.zeros(8, dtype=np.uint8)
    roles[3] = ROLE_TO_ID["BIB_HEADER"]
    core = np.zeros(8, dtype=bool)
    core[1:3] = True
    core[5:7] = True
    result = apply_header_controller(
        core,
        roles,
        absolute,
        config=HeaderControllerConfig(attachment_window=3, connector_window=2),
    )
    assert not result[1:3].any()
    assert result[3:7].all()


def test_exact_wall_is_never_refilled_and_physical_wall_is_preserved() -> None:
    absolute = np.arange(7)
    roles = np.zeros(7, dtype=np.uint8)
    roles[3] = ROLE_TO_ID["BIB_SUBHEADER"]
    core = np.zeros(7, dtype=bool)
    core[1] = core[5] = True
    wall = np.zeros(7, dtype=bool)
    wall[4] = True
    result = apply_header_controller(core, roles, absolute, hard_wall_mask=wall)
    assert not result[3]
    assert not result[4]

    absolute = np.asarray([0, 1, 2, 100, 101, 102, 103])
    wall[:] = False
    result = apply_header_controller(core, roles, absolute, hard_wall_mask=wall)
    assert not result[3]


def test_pairwise_composition_unions_both_parent_barriers() -> None:
    left = {
        "hard_wall": np.array([False, True, False, False]),
        "upward_stop": np.zeros(4, dtype=bool),
        "downward_stop": np.zeros(4, dtype=bool),
    }
    right = {
        "hard_wall": np.zeros(4, dtype=bool),
        "upward_stop": np.array([False, False, False, True]),
        "downward_stop": np.zeros(4, dtype=bool),
    }
    combined = combine_parent_barriers(left, right, (4,))
    result = enforce_combined_barriers(np.ones(4, dtype=bool), combined)
    assert not result[1]
    assert not (result[2] and result[3])


def test_descendant_gap_stage_cannot_refill_persisted_parent_wall() -> None:
    prediction = np.array([True, False, True], dtype=bool)
    signal = np.ones(3, dtype=np.float32)
    absolute = np.arange(3)
    roles = np.zeros(3, dtype=np.uint8)
    hard = np.array([False, True, False])
    result = _postprocess_document(
        prediction, signal, absolute, hard, np.zeros(3, dtype=bool),
        np.zeros(3, dtype=bool), roles,
        operation="internal_gap_connection", threshold=0.0, max_lines=2,
    )
    assert not result[1]

    hard[:] = False
    upward = np.array([False, True, False])
    result = _postprocess_document(
        prediction, signal, absolute, hard, upward, np.zeros(3, dtype=bool), roles,
        operation="internal_gap_connection", threshold=0.0, max_lines=2,
    )
    assert not result[1]
