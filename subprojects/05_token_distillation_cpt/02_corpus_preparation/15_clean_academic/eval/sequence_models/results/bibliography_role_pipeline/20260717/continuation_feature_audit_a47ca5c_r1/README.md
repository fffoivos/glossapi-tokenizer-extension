# CONTINUATION feature audit

- trusted subtype rows: 813 (535 CONTINUATION; 278 FILLER)
- OOF fold-weighted CONTINUATION PR-AUC given connector: 0.959594
- frozen entry model and connector models: unchanged
- validation opened: no

## Continuation forms measured from frozen features

| form | lines | share |
|---|---:|---:|
| self_supporting_at_entry_threshold | 40 | 7.5% |
| weak_at_entry_threshold | 495 | 92.5% |
| join_rescued_by_previous | 138 | 25.8% |
| join_rescued_by_next | 94 | 17.6% |
| join_rescued_by_either | 167 | 31.2% |
| join_gain_at_least_0_10 | 122 | 22.8% |
| short_at_most_40_chars | 141 | 26.4% |
| tiny_at_most_3_chars | 41 | 7.7% |
| starts_lowercase | 68 | 12.7% |
| previous_line_ends_open | 81 | 15.1% |
| table_row_feature | 138 | 25.8% |
| url_or_doi_feature | 76 | 14.2% |
| page_or_volume_feature | 185 | 34.6% |
| author_or_name_feature | 346 | 64.7% |

## Most common deterministic features

| feature | continuation | entry | filler | other |
|---|---:|---:|---:|---:|
| `proper_name_word_count` | 62.1% | 97.0% | 5.4% | 51.4% |
| `punctuation_count` | 53.8% | 70.6% | 39.9% | 68.9% |
| `year_count` | 27.1% | 70.7% | 0.0% | 15.3% |
| `table_row_count` | 25.8% | 26.8% | 35.6% | 17.8% |
| `dotted_word_count` | 25.4% | 40.9% | 1.4% | 10.1% |
| `page_range_count` | 21.7% | 42.2% | 0.0% | 5.9% |
| `initial_count` | 16.3% | 66.4% | 1.4% | 14.7% |
| `numbered_entry_count` | 15.3% | 37.0% | 12.2% | 24.1% |
| `page_marker_count` | 13.8% | 25.4% | 0.4% | 1.8% |
| `url_count` | 8.8% | 7.5% | 0.0% | 1.8% |
| `place_name_count` | 8.0% | 10.5% | 0.0% | 1.4% |
| `volume_shape_count` | 7.7% | 11.0% | 0.0% | 0.6% |
| `journal_year_volume_count` | 6.9% | 13.0% | 0.0% | 3.0% |
| `article_page_range_count` | 6.4% | 7.8% | 0.0% | 1.4% |
| `quoted_span_count` | 6.2% | 13.5% | 2.9% | 7.0% |
| `doi_count` | 5.4% | 1.2% | 0.0% | 0.0% |
| `name_initial_pair_count` | 3.7% | 32.9% | 0.0% | 5.7% |
| `volume_marker_count` | 3.7% | 6.5% | 0.0% | 1.2% |
| `publisher_term_count` | 3.2% | 5.7% | 0.0% | 0.4% |
| `access_date_count` | 2.4% | 4.0% | 0.0% | 0.8% |

## Feature-group permutation importance

| group | features | CONTINUATION PR-AUC drop | permutation SD |
|---|---:|---:|---:|
| current_line_shape | 34 | 0.117289 | 0.009812 |
| unmatched_character_geometry | 7 | 0.021406 | 0.003576 |
| adjacent_line_shape_pairs | 18 | 0.005163 | 0.001839 |
| joined_line_entry_gain | 8 | 0.003572 | 0.002314 |
| deterministic_feature_counts | 35 | 0.003224 | 0.001162 |
| block_relative_position | 2 | 0.002519 | 0.001236 |
| entry_probability_neighbourhoods | 30 | 0.001310 | 0.001353 |
| deterministic_feature_presence | 35 | 0.000668 | 0.001052 |
| heading_probabilities | 4 | 0.000024 | 0.000033 |
| nearest_entry_anchor | 4 | -0.002884 | 0.001010 |

## Top individual permutation importances

| feature | group | CONTINUATION PR-AUC drop |
|---|---|---:|
| `token_count` | current_line_shape | 0.018221 |
| `gap:unmatched_prefix_fraction` | unmatched_character_geometry | 0.014374 |
| `whitespace_fraction` | current_line_shape | 0.004609 |
| `symbol_fraction` | current_line_shape | 0.004332 |
| `log1p:punctuation_count` | deterministic_feature_counts | 0.003473 |
| `gap:unmatched_suffix_fraction` | unmatched_character_geometry | 0.003404 |
| `digit_fraction` | current_line_shape | 0.003222 |
| `maximum_token_length` | current_line_shape | 0.002870 |
| `previous_pair:left_ends_opening_terminal` | adjacent_line_shape_pairs | 0.002550 |
| `gap:longest_unmatched_center` | unmatched_character_geometry | 0.002008 |
| `candidate_window_edge_distance` | block_relative_position | 0.001649 |
| `punctuation_fraction` | current_line_shape | 0.001592 |
| `joined_previous_probability_gain` | joined_line_entry_gain | 0.001462 |
| `letter_fraction` | current_line_shape | 0.001367 |
| `joined_previous_distinct_feature_gain` | joined_line_entry_gain | 0.001329 |
| `entry_below_r10_mean` | entry_probability_neighbourhoods | 0.001048 |
| `entry_below_r1_max` | entry_probability_neighbourhoods | 0.000962 |
| `latin_fraction_of_letters` | current_line_shape | 0.000925 |
| `gap:longest_unmatched_fraction` | unmatched_character_geometry | 0.000764 |
| `entry_below_r30_max` | entry_probability_neighbourhoods | 0.000639 |
| `previous_pair:greek_fraction_difference` | adjacent_line_shape_pairs | 0.000535 |
| `entry_above_r30_mean` | entry_probability_neighbourhoods | 0.000451 |
| `entry_above_r5_mean` | entry_probability_neighbourhoods | 0.000384 |
| `nearest_anchor_below_probability` | nearest_entry_anchor | 0.000331 |
| `is_repeated_rule` | current_line_shape | 0.000278 |
| `entry_below_r3_max` | entry_probability_neighbourhoods | 0.000272 |
| `entry_below_r10_max` | entry_probability_neighbourhoods | 0.000247 |
| `joined_next_entry_probability` | joined_line_entry_gain | 0.000192 |
| `presence:punctuation_count` | deterministic_feature_presence | 0.000188 |
| `next_pair:greek_fraction_difference` | adjacent_line_shape_pairs | 0.000169 |
| `entry_above_r5_max` | entry_probability_neighbourhoods | 0.000138 |
| `gap:unmatched_fraction` | unmatched_character_geometry | 0.000122 |
| `char_length` | current_line_shape | 0.000097 |
| `gap:unmatched_run_count` | unmatched_character_geometry | 0.000041 |
| `bib_subheader_probability` | heading_probabilities | 0.000037 |
| `ends_sentence_terminal` | current_line_shape | 0.000031 |
| `previous_pair:right_to_left_length_ratio` | adjacent_line_shape_pairs | 0.000028 |
| `next_pair:right_to_left_length_ratio` | adjacent_line_shape_pairs | 0.000021 |
| `log1p:proper_name_word_count` | deterministic_feature_counts | 0.000013 |
| `ends_opening_terminal` | current_line_shape | 0.000008 |
