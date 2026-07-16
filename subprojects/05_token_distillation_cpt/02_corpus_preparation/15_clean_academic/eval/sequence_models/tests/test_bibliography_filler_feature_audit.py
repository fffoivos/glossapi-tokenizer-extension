from sequence_models.bibliography_filler_feature_audit import feature_group


def test_filler_feature_groups_cover_connector_contract() -> None:
    assert feature_group("presence:year_count") == "deterministic_feature_presence"
    assert feature_group("gap:unmatched_fraction") == "unmatched_character_geometry"
    assert feature_group("entry_above_r3_max") == "entry_probability_neighbourhoods"
    assert feature_group("previous_pair:indentation_difference") == "adjacent_line_shape_pairs"
    assert feature_group("joined_next_probability_gain") == "joined_line_entry_gain"
    assert feature_group("bib_header_probability") == "heading_probabilities"
    assert feature_group("inside_anchor_gap") == "block_relative_position"
    assert feature_group("char_length") == "current_line_shape"
