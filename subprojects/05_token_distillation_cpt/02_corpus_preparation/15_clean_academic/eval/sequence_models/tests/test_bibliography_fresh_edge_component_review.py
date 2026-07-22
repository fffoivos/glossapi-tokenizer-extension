import json

import numpy as np

from sequence_models.bibliography_fresh_edge_component_review import (
    REVIEW_HTML,
    _boundary_controls,
    balance_edge_controls,
    balanced_component_quota,
    build_component_packet,
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


def test_edge_control_sampling_keeps_every_removal_and_balances_controls():
    cases = []
    for source in ("greek_phd", "kallipos", "openarchives"):
        for index in range(5):
            cases.append(
                {
                    "source": source,
                    "document_id": source,
                    "abs_idx": index,
                    "case_id": f"{source}:r{index}",
                    "frozen_action": "remove",
                }
            )
        for index in range(30):
            cases.append(
                {
                    "source": source,
                    "document_id": source,
                    "abs_idx": 100 + index,
                    "case_id": f"{source}:k{index}",
                    "frozen_action": "keep",
                }
            )
    selected = balance_edge_controls(cases, minimum_controls_per_source=7)
    for source in ("greek_phd", "kallipos", "openarchives"):
        local = [row for row in selected if row["source"] == source]
        assert sum(row["frozen_action"] == "remove" for row in local) == 5
        assert sum(row["frozen_action"] == "keep" for row in local) == 7


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


def test_component_quota_uses_largest_exact_source_balance():
    rows = [
        {"source": source}
        for source, count in (("greek_phd", 60), ("kallipos", 45), ("openarchives", 31))
        for _ in range(count)
    ]
    quota, counts = balanced_component_quota(
        rows, requested_per_stratum_per_source=20
    )
    assert quota == 15
    assert counts == {"greek_phd": 60, "kallipos": 45, "openarchives": 31}


def test_long_component_packet_shows_first_middle_last_windows(monkeypatch):
    monkeypatch.setattr(
        "sequence_models.bibliography_fresh_edge_component_review._feature_payload",
        lambda text: (text, {}, {}),
    )
    document = {
        "document_id": "doc",
        "lines": [{"abs_idx": index, "text": f"line {index}"} for index in range(120)],
    }
    selected = [
        {
            "document_id": "doc",
            "work_id": "work",
            "source": "greek_phd",
            "local_start": 5,
            "local_end": 104,
            "abs_start": 5,
            "abs_end": 104,
            "line_count": 100,
            "selection_stratum": "bibliography_like",
            "citation_line_fraction": 1.0,
            "hard_negative_role_fraction": 0.0,
            "entry_positive_fraction": 1.0,
            "signal_median": 1.0,
            "mean_characters": 10.0,
        }
    ]
    case = build_component_packet([document], selected)["cases"][0]
    assert case["displayed_component_line_count"] == 60
    assert sum(row.get("omitted_count", 0) for row in case["context"]) == 40
    assert [row["abs_idx"] for row in case["context"] if row.get("target") and "abs_idx" in row][:2] == [5, 6]


def test_review_is_blind_until_decision_and_supports_separate_reviewers():
    assert "Frozen edge decision" in REVIEW_HTML
    assert "reveal.classList.toggle('hidden',!answer)" in REVIEW_HTML
    assert "Reviewer" in REVIEW_HTML
    assert "storageKey" in REVIEW_HTML
    assert "ArrowLeft" in REVIEW_HTML and "ArrowRight" in REVIEW_HTML
    assert "packet_sha256" in REVIEW_HTML
