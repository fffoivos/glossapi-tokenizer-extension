from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from agent1_v3_decontaminate import action_for_match  # noqa: E402


def test_only_high_confidence_matches_drop() -> None:
    assert action_for_match("drop", "greekmmlu_exact_prompt", [{"item_id": "one"}]) == (
        "drop",
        "greekmmlu_exact_prompt",
    )
    assert action_for_match("keep", "no_high_confidence_match", [{"item_id": "one"}]) == (
        "quarantine",
        "greekmmlu_ambiguous_match_evidence",
    )
    assert action_for_match("keep", "no_high_confidence_match", []) == (
        "keep",
        "no_high_confidence_match",
    )
