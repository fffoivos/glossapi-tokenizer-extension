from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "initialization" / "build_td_assets_from_packed_training.py"
SPEC = importlib.util.spec_from_file_location("packed_td_assets", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sequence_record_geometry_matches_frozen_catalogue() -> None:
    assert MODULE.SEQUENCE_DTYPE.itemsize == 18
    assert MODULE.PACKED_WIDTH == 4097


def test_requested_sample_count_supports_exact_all_sentinel() -> None:
    assert MODULE.requested_sample_count(197_148, -1) == 197_148
    assert MODULE.requested_sample_count(500_000, 350_000) == 350_000


def test_exact_subsequence_matching_is_not_set_membership() -> None:
    assert MODULE.contains_subsequence([1, 2, 3, 4], [2, 3])
    assert not MODULE.contains_subsequence([1, 3, 2, 4], [2, 3])
    assert not MODULE.contains_subsequence([1, 2], [])


def test_status_gate_retains_subthreshold_rows_as_fvt() -> None:
    assert MODULE.status_for(25) == ("enough_25", "td_25")
    assert MODULE.status_for(24) == ("low_20_24", "keep_retok")
    assert MODULE.status_for(0) == ("zero", "inspect")


def test_range_constants_cover_both_extension_stages_without_gap() -> None:
    assert MODULE.POLYTONIC_START - MODULE.BASE_VOCAB_SIZE == 17_408
    assert MODULE.TARGET_VOCAB_SIZE - MODULE.POLYTONIC_START == 512
    assert MODULE.TARGET_VOCAB_SIZE % 256 == 0
    assert np.dtype(MODULE.SEQUENCE_DTYPE).names == (
        "sequence_id",
        "packing_task_index",
        "row_index",
        "active_tokens",
    )
