from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_entry_sequence import (
    FEATURE_NAMES_BASE,
    _feature_rows,
    constrained_viterbi,
    make_examples,
)
from sequence_models.feature_crf import LinearChainCRF
from sequence_models.features import TAGS


def test_feature_rows_expose_header_only_in_the_ablation() -> None:
    probability = np.asarray([0.1, 0.9])
    lengths = np.asarray([20, 100])
    absolute = np.asarray([0, 1])
    headings = np.asarray([1, 0])
    baseline = _feature_rows(
        probability,
        lengths,
        absolute,
        2,
        headings,
        seed_length_limit=330,
        include_header=False,
    )
    ablation = _feature_rows(
        probability,
        lengths,
        absolute,
        2,
        headings,
        seed_length_limit=330,
        include_header=True,
    )
    assert len(baseline) == len(ablation) == 2
    assert len(FEATURE_NAMES_BASE) not in baseline[0]
    assert ablation[0][len(FEATURE_NAMES_BASE)] == 1.0
    assert len(FEATURE_NAMES_BASE) not in ablation[1]


def test_feature_rows_can_drop_length_observations_without_dropping_evidence() -> None:
    rows = _feature_rows(
        np.asarray([0.9]),
        np.asarray([900]),
        np.asarray([8]),
        10,
        np.asarray([0]),
        seed_length_limit=330,
        include_header=False,
        dropped_feature_names=("log1p_char_length", "over_seed_length_limit"),
    )
    assert FEATURE_NAMES_BASE.index("entry_probability") in rows[0]
    assert FEATURE_NAMES_BASE.index("physical_position") in rows[0]
    assert FEATURE_NAMES_BASE.index("log1p_char_length") not in rows[0]
    assert FEATURE_NAMES_BASE.index("over_seed_length_limit") not in rows[0]


def test_long_line_cannot_start_a_constrained_crf_block() -> None:
    model = LinearChainCRF(1, active_classes=("BIB",))
    model.emission[:] = 0.0
    model.emission_bias[:] = -10.0
    model.emission_bias[TAGS.index("O")] = 0.0
    model.emission_bias[TAGS.index("S-BIB")] = 20.0
    tags = constrained_viterbi(
        model,
        [{0: 1.0}, {0: 1.0}],
        np.asarray([900, 100]),
        seed_length_limit=330,
        deletion_bias=0.0,
    )
    assert [TAGS[int(tag)] for tag in tags] == ["O", "S-BIB"]


def test_examples_split_on_unknown_and_large_physical_gap() -> None:
    table = SimpleNamespace(
        documents=(
            {
                "line_start": 0,
                "line_end": 5,
                "n_physical_lines": 1000,
            },
        ),
        original_labels=np.asarray([0, 1, 3, 1, 0], dtype=np.uint8),
        abs_indices=np.asarray([0, 1, 2, 900, 901], dtype=np.uint32),
        char_lengths=np.asarray([10, 20, 10, 20, 10], dtype=np.uint32),
        header_kinds=np.zeros(5, dtype=np.uint8),
        token_counts=np.ones(5, dtype=np.uint32),
    )
    examples = make_examples(
        table,
        np.asarray([0.1, 0.9, 0.1, 0.9, 0.1]),
        [0],
        seed_length_limit=330,
        include_header=False,
    )
    assert [example.line_indices for example in examples] == [(0, 1), (3, 4)]
    assert [[TAGS[int(tag)] for tag in example.tags] for example in examples] == [
        ["O", "S-BIB"],
        ["S-BIB", "O"],
    ]
