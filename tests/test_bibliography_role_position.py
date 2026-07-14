from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


EVAL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
)
sys.path.insert(0, str(EVAL_ROOT))

from sequence_models.bibliography_positional_features import (  # noqa: E402
    FEATURE_NAMES,
    NONMATCH_CATEGORIES,
    count_gap_scalars,
    extract_positional_line,
    positional_summary_scalars,
    rasterize_position_map,
)
from sequence_models.bibliography_role_dataset import (  # noqa: E402
    TARGET_ENTRY_ANCHOR,
    TARGET_MASK,
    TARGET_NEGATIVE,
    entry_anchor_target,
    load_role_contract,
    text_sha256,
    validate_overlay,
)
from sequence_models.bibliography_role_profile import (  # noqa: E402
    profile_document,
    select_review_blocks,
)


CONTRACT_PATH = EVAL_ROOT / "sequence_models/bibliography_role_contract_v1.json"


def test_positional_encoding_preserves_counts_spans_and_nonmatch_partition() -> None:
    text = "Lewis, M.A. (2006). A title. Nat. Chem. Biol. 2:44-51."
    encoding = extract_positional_line(text)

    observed = np.bincount(encoding.match_feature, minlength=len(FEATURE_NAMES))
    assert np.array_equal(observed.astype(np.uint32), encoding.counts)
    assert np.all(encoding.match_start < encoding.match_end)
    assert np.all(encoding.match_end <= encoding.nfkc_length)
    assert np.all(encoding.nonmatch_start < encoding.nonmatch_end)
    assert np.all(encoding.nonmatch_end <= encoding.nfkc_length)
    assert 0.0 <= encoding.gap_summaries[0] <= 1.0
    assert count_gap_scalars(encoding).shape == (77,)
    assert positional_summary_scalars(encoding).shape == (210,)


def test_position_raster_has_expected_channels_and_explicit_coordinates() -> None:
    encoding = extract_positional_line("Σ. Ν. Νανάς, (1996), ΤΟΜΟΣ 37:298-304")
    plain = rasterize_position_map(encoding, bins=16)
    coordinates = rasterize_position_map(
        encoding, bins=64, include_coordinate_channels=True
    )

    assert plain.shape == (len(FEATURE_NAMES) + len(NONMATCH_CATEGORIES), 16)
    assert coordinates.shape == (
        len(FEATURE_NAMES) + len(NONMATCH_CATEGORIES) + 2,
        64,
    )
    assert np.all((plain >= 0.0) & (plain <= 1.0))
    assert np.all(np.diff(coordinates[-2]) > 0)
    assert np.allclose(coordinates[-2] + coordinates[-1], 1.0)


def test_role_contract_masks_untrusted_and_supports_nonanchor_mask_ablation() -> None:
    contract = load_role_contract(CONTRACT_PATH)

    assert entry_anchor_target(
        role="ENTRY_ANCHOR", label_status="AGREED_REVIEW", contract=contract
    ) == TARGET_ENTRY_ANCHOR
    assert entry_anchor_target(
        role="HEADER", label_status="AGREED_REVIEW", contract=contract
    ) == TARGET_NEGATIVE
    assert entry_anchor_target(
        role="HEADER",
        label_status="AGREED_REVIEW",
        contract=contract,
        mask_in_block_nonanchors=True,
    ) == TARGET_MASK
    assert entry_anchor_target(
        role="ENTRY_ANCHOR", label_status="PROVISIONAL", contract=contract
    ) == TARGET_MASK


def test_overlay_validation_fails_closed_on_source_identity(tmp_path: Path) -> None:
    contract = load_role_contract(CONTRACT_PATH)
    source_path = tmp_path / "silver.jsonl"
    overlay_path = tmp_path / "overlay.jsonl"
    source = {
        "document_id": "doc-1",
        "work_id": "work-1",
        "lines": [
            {
                "line_id": "doc-1:7",
                "abs_idx": 7,
                "text": "Lewis, M.A. (2006). A title.",
                "label": "BIB",
            }
        ],
    }
    source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    overlay = {
        "schema_version": "bibliography-role-overlay-v1",
        "document_id": "doc-1",
        "work_id": "work-1",
        "line_id": "doc-1:7",
        "abs_idx": 7,
        "text_sha256": text_sha256(source["lines"][0]["text"]),
        "original_region_label": "BIB",
        "role": "ENTRY_ANCHOR",
        "boundary_flag": "NONE",
        "label_status": "AGREED_REVIEW",
        "label_origin": "contextual_review",
        "confidence": 0.98,
        "reviewers": ["reviewer-a", "reviewer-b"],
    }
    overlay_path.write_text(json.dumps(overlay) + "\n", encoding="utf-8")

    result = validate_overlay(
        source_path=source_path, overlay_path=overlay_path, contract=contract
    )

    assert result["status"] == "passed"
    assert result["entry_anchor_target_counts"] == {"1": 1}


def _document(document_id: str, work_id: str, source: str) -> dict[str, object]:
    lines = [
        {
            "line_id": f"{document_id}:0",
            "abs_idx": 0,
            "text": "Κεφάλαιο",
            "label": "O",
        },
        {
            "line_id": f"{document_id}:1",
            "abs_idx": 1,
            "text": "ΒΙΒΛΙΟΓΡΑΦΙΑ",
            "label": "BIB",
        },
        {
            "line_id": f"{document_id}:2",
            "abs_idx": 2,
            "text": "Lewis, M.A. (2006). A title. 2:44-51.",
            "label": "BIB",
        },
        {
            "line_id": f"{document_id}:3",
            "abs_idx": 3,
            "text": "Επόμενο κεφάλαιο",
            "label": "O",
        },
    ]
    return {
        "document_id": document_id,
        "work_id": work_id,
        "source": source,
        "split": "train",
        "coverage": "full_document",
        "n_physical_lines": len(lines),
        "lines": lines,
    }


def test_role_profiler_keeps_selection_prediction_blind_and_work_distinct() -> None:
    profiles = [
        profile_document(_document(f"doc-{index}", f"work-{index}", "greek_phd"))
        for index in range(6)
    ]
    blocks = [block for profile in profiles for block in profile["blocks"]]

    selected = select_review_blocks(
        blocks, sources=("greek_phd",), per_source=5, seed="test-seed"
    )

    assert len(selected) == 5
    assert len({row["work_id"] for row in selected}) == 5
    assert all("strata" in row for row in selected)
    assert profiles[0]["line_profiles"][1]["exact_header_kind"] == "HEADER"
