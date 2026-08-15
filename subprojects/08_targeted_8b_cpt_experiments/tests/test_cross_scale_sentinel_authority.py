from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_cross_scale_sentinel_authority import select_cross_scale_trajectory


def test_common_passing_sentinel_uses_largest_shared_panel() -> None:
    decision = select_cross_scale_trajectory(
        {
            "8b": {"jointly_passing_sizes": [4096, 8192]},
            "1p5b": {"jointly_passing_sizes": [8192]},
        }
    )
    assert decision == {
        "mode": "sentinel_pair",
        "selected_size": 8192,
        "common_passing_sizes": [8192],
        "full_fallback_scales": [],
    }


def test_no_common_passing_sentinel_requires_full_both_scales() -> None:
    decision = select_cross_scale_trajectory(
        {
            "8b": {"jointly_passing_sizes": [4096]},
            "1p5b": {"jointly_passing_sizes": [8192]},
        }
    )
    assert decision["mode"] == "full_clean"
    assert decision["full_fallback_scales"] == ["8b", "1p5b"]
