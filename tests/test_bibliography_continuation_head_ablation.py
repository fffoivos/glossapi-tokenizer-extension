from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
)
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.bibliography_continuation_head_ablation import feature_arms  # noqa: E402


def test_feature_arms_are_nested_and_table_interactions_are_explicit() -> None:
    manifest = json.loads((
        EVAL_DIR / "sequence_models/results/bibliography_role_pipeline/20260715"
        / "connector_table_8f8d6ed_r1/manifest.json"
    ).read_text(encoding="utf-8"))
    names = manifest["feature_names"]
    features = np.zeros((3, len(names)), dtype=np.float32)
    features[0, names.index("presence:table_row_count")] = 1.0
    features[0, names.index("token_count")] = 7.0
    arms, arm_names = feature_arms(features, names)

    assert set(arms) == {
        "all_177", "compact_core", "compact_plus_directional_join",
        "compact_join_table_interactions",
    }
    assert arms["compact_core"].shape[1] < arms["all_177"].shape[1]
    assert arms["compact_plus_directional_join"].shape[1] == arms["compact_core"].shape[1] + 8
    assert arms["compact_join_table_interactions"].shape[1] > arms["compact_plus_directional_join"].shape[1]
    interaction = arm_names["compact_join_table_interactions"].index("table_x:token_count")
    assert arms["compact_join_table_interactions"][0, interaction] == 7.0
    assert arms["compact_join_table_interactions"][1, interaction] == 0.0
    assert all(len(values) == arms[name].shape[1] for name, values in arm_names.items())
