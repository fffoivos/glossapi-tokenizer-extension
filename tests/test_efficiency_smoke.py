from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ops.perf.run_efficiency_smoke import LinuxProcessSampler


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_linux_process_sampler_tracks_child_process_memory() -> None:
    code = "import time; payload = bytearray(32 * 1024 * 1024); time.sleep(1.5)"
    with LinuxProcessSampler(pid=os.getpid()) as sampler:
        child = subprocess.Popen([sys.executable, "-c", code])
        time.sleep(0.4)
        child.wait()
        time.sleep(0.2)
    assert sampler.peak_child_count >= 1
    assert sampler.peak_total_child_rss_kb > 0
    assert sampler.peak_total_child_pss_kb > 0


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
    assert summary["near_candidates"]["peak_total_child_pss_mb"] >= 0
    assert summary["near_candidates"]["summary"]["candidate_pair_rows"] > 0
