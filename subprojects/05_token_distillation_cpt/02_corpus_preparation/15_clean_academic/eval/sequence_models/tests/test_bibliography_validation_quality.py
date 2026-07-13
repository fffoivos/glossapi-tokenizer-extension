from __future__ import annotations

from sequence_models.bibliography_validation_quality import (
    TextQuality,
    analyze_text,
    candidate_reasons,
)


def test_extreme_fragmentation_is_a_candidate() -> None:
    quality = analyze_text([f"word{index}" for index in range(600)])
    assert quality.lines_at_most_one_word_fraction == 1.0
    assert "extreme_line_fragmentation" in candidate_reasons(quality, 0.0)


def test_character_spaced_text_is_a_candidate() -> None:
    quality = analyze_text(["A B C D E F G H I J"] * 100)
    assert quality.lexical_word_count == 1000
    assert "character_spaced_extraction" in candidate_reasons(quality, 0.0)


def test_glyph_placeholders_are_candidates() -> None:
    quality = analyze_text(["GLYPH<12> GLYPH&lt;7&gt;"] * 30)
    assert quality.glyph_placeholder_count == 60
    assert "glyph_placeholder_corruption" in candidate_reasons(quality, 0.0)


def test_readable_prose_is_not_flagged() -> None:
    quality = analyze_text(
        [
            "Παπαδόπουλος, Α. (2020). Ένα κανονικό βιβλιογραφικό λήμμα.",
            "Smith, J. (2018). A readable bibliography entry. Athens: Press.",
        ]
        * 300
    )
    assert candidate_reasons(quality, 10.0) == []


def test_canonical_rust_badness_remains_the_primary_gate() -> None:
    quality = TextQuality(10, 100, 50, 5.0, 0.0, 0.0, 0, 0.0, 0.0, 0)
    assert candidate_reasons(quality, 60.0) == []
    assert candidate_reasons(quality, 60.0001) == ["canonical_greek_badness_gt_60"]
