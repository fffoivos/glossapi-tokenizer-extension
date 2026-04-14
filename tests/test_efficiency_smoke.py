from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_efficiency_smoke_harness_runs(tmp_path: Path) -> None:
    work_root = tmp_path / "efficiency"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "ops" / "perf" / "run_efficiency_smoke.py"),
            "--work-root",
            str(work_root),
            "--target",
            "all",
            "--candidate-docs",
            "128",
            "--candidate-group-size",
            "8",
            "--candidate-workers",
            "12",
            "--mix-rows",
            "1000",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(result.stdout)
    assert summary["repo_root"] == str(REPO_ROOT)
    assert summary["mix_streaming"]["streaming_expected"] is True
    assert summary["mix_streaming"]["rows_output"] > 0
    assert summary["near_candidates"]["requested_workers"] == 12
    assert summary["near_candidates"]["capped_workers"] == 8
    assert summary["near_candidates"]["start_method"] in {"spawn", "fork"}
    assert summary["near_candidates"]["summary"]["candidate_pair_rows"] > 0
