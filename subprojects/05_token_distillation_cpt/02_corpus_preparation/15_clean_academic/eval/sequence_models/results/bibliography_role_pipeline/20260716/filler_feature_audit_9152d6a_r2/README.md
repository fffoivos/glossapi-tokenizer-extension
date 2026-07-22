# FILLER feature audit

- trusted subtype rows: 813 (278 FILLER; 535 CONTINUATION)
- OOF fold-weighted FILLER PR-AUC: 0.933635
- validation opened: no

## Feature-group permutation importance

| group | features | PR-AUC drop | permutation SD |
|---|---:|---:|---:|
| current_line_shape | 34 | 0.327554 | 0.026890 |
| unmatched_character_geometry | 7 | 0.038082 | 0.009524 |
| deterministic_feature_counts | 35 | 0.015520 | 0.004307 |
| adjacent_line_shape_pairs | 18 | 0.007780 | 0.004080 |
| entry_probability_neighbourhoods | 30 | 0.006955 | 0.002971 |
| deterministic_feature_presence | 35 | 0.005702 | 0.003968 |
| block_relative_position | 2 | 0.004497 | 0.002593 |
| joined_line_entry_gain | 8 | 0.004450 | 0.002989 |
| heading_probabilities | 4 | -0.000015 | 0.000117 |
| nearest_entry_anchor | 4 | -0.001907 | 0.001684 |

## Top individual permutation importances

| feature | group | PR-AUC drop |
|---|---|---:|
| `token_count` | current_line_shape | 0.051459 |
| `log1p:punctuation_count` | deterministic_feature_counts | 0.013296 |
| `gap:unmatched_prefix_fraction` | unmatched_character_geometry | 0.012505 |
| `previous_pair:left_ends_opening_terminal` | adjacent_line_shape_pairs | 0.008924 |
| `presence:punctuation_count` | deterministic_feature_presence | 0.007538 |
| `gap:longest_unmatched_center` | unmatched_character_geometry | 0.006187 |
| `joined_previous_distinct_feature_gain` | joined_line_entry_gain | 0.004800 |
| `candidate_window_edge_distance` | block_relative_position | 0.003723 |
| `latin_fraction_of_letters` | current_line_shape | 0.003709 |
| `symbol_fraction` | current_line_shape | 0.003628 |
| `punctuation_fraction` | current_line_shape | 0.003363 |
| `digit_fraction` | current_line_shape | 0.003256 |
| `maximum_token_length` | current_line_shape | 0.002222 |
| `whitespace_fraction` | current_line_shape | 0.002174 |
| `entry_below_r3_max` | entry_probability_neighbourhoods | 0.001903 |
| `entry_below_r10_mean` | entry_probability_neighbourhoods | 0.001407 |
| `entry_below_r30_max` | entry_probability_neighbourhoods | 0.001377 |
| `entry_below_r1_max` | entry_probability_neighbourhoods | 0.001363 |
| `gap:unmatched_fraction` | unmatched_character_geometry | 0.001199 |
| `entry_above_r5_max` | entry_probability_neighbourhoods | 0.001164 |
| `entry_above_r30_max` | entry_probability_neighbourhoods | 0.001022 |
| `lowercase_fraction_of_letters` | current_line_shape | 0.000875 |
| `nearest_anchor_below_probability` | nearest_entry_anchor | 0.000796 |
| `entry_below_r10_max` | entry_probability_neighbourhoods | 0.000650 |
| `char_length` | current_line_shape | 0.000529 |
| `nearest_anchor_below_distance` | nearest_entry_anchor | 0.000436 |
| `is_repeated_rule` | current_line_shape | 0.000299 |
| `gap:unmatched_suffix_fraction` | unmatched_character_geometry | 0.000242 |
| `next_pair:greek_fraction_difference` | adjacent_line_shape_pairs | 0.000240 |
| `previous_pair:right_to_left_length_ratio` | adjacent_line_shape_pairs | 0.000238 |
| `entry_above_r10_mean` | entry_probability_neighbourhoods | 0.000205 |
| `entry_above_r30_count025` | entry_probability_neighbourhoods | 0.000193 |
| `gap:longest_unmatched_fraction` | unmatched_character_geometry | 0.000156 |
| `letter_fraction` | current_line_shape | 0.000154 |
| `ends_opening_terminal` | current_line_shape | 0.000093 |
| `next_pair:right_to_left_length_ratio` | adjacent_line_shape_pairs | 0.000082 |
| `joined_previous_probability_gain` | joined_line_entry_gain | 0.000072 |
| `entry_below_r5_max` | entry_probability_neighbourhoods | 0.000056 |
| `ends_sentence_terminal` | current_line_shape | 0.000053 |
| `entry_below_r3_mean` | entry_probability_neighbourhoods | 0.000041 |
