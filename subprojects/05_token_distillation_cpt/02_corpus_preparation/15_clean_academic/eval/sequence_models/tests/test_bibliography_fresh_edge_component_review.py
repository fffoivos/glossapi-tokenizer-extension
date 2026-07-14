import json

import numpy as np

from sequence_models.bibliography_fresh_edge_component_review import (
    REVIEW_HTML,
    _boundary_controls,
    load_prior_exclusions,
    select_component_cases,
)


def test_prior_packet_excludes_documents_and_complete_works(tmp_path):
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "documents": [
                    {"document_id": "doc-a", "work_id": "work-a"},
                    {"document_id": "doc-b", "work_id": "work-b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    documents, works, receipts = load_prior_exclusions([packet])
    assert documents == {"doc-a", "doc-b"}
    assert works == {"work-a", "work-b"}
    assert receipts[0]["document_count"] == 2


def test_boundary_controls_cover_kept_side_of_transitions():
    base = np.ones(8, dtype=bool)
    edge = np.asarray([False, False, True, True, True, True, False, False])
    assert _boundary_controls(base, edge, 0, 7) == {2, 3, 4, 5}


def test_unchanged_block_samples_both_ends():
    base = np.ones(8, dtype=bool)
    assert _boundary_controls(base, base, 0, 7) == {0, 1, 6, 7}


def test_component_selection_is_balanced_and_disjoint():
    rows = []
    for source in ("greek_phd", "kallipos", "openarchives"):
        for index in range(8):
            rows.append(
                {
                    "source": source,
                    "document_id": f"{source}-{index}",
                    "local_start": index * 3,
                    "local_end": index * 3 + 1,
                    "narrative_rank": float(index),
                    "bibliography_rank": float(8 - index),
                    "citation_line_fraction": 0.5,
                }
            )
    selected = select_component_cases(rows, per_stratum_per_source=2)
    assert len(selected) == 12
    identities = {
        (row["document_id"], row["local_start"], row["local_end"])
        for row in selected
    }
    assert len(identities) == len(selected)
    for source in ("greek_phd", "kallipos", "openarchives"):
        local = [row for row in selected if row["source"] == source]
        assert len(local) == 4
        assert {row["selection_stratum"] for row in local} == {
            "citation_dense_narrative_risk",
            "bibliography_like",
        }


def test_review_is_blind_until_decision_and_supports_separate_reviewers():
    assert "Frozen edge decision" in REVIEW_HTML
    assert "reveal.classList.toggle('hidden',!answer)" in REVIEW_HTML
    assert "Reviewer" in REVIEW_HTML
    assert "storageKey" in REVIEW_HTML
    assert "ArrowLeft" in REVIEW_HTML and "ArrowRight" in REVIEW_HTML
    assert "packet_sha256" in REVIEW_HTML
