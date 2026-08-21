from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/workaround_accept_intentional_torchrun_teardown.py"


def fixture(tmp_path: Path, *, fatal: str = "") -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoints"
    metadata = checkpoint / "iter_0000002/.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"complete")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("2\n", encoding="utf-8")
    log = tmp_path / "driver.partial"
    log.write_text(
        "iteration 2/ 3218 | number of skipped iterations: 0 | number of nan iterations: 0 |\n"
        "successfully saved checkpoint from iteration 2 to /tmp/checkpoints\n"
        "RendezvousConnectionError: connection to the C10d store has failed. "
        "Failed to recv, got 0 bytes.\n"
        + fatal,
        encoding="utf-8",
    )
    return log, checkpoint


def run(tmp_path: Path, *, fatal: str = "") -> subprocess.CompletedProcess[str]:
    log, checkpoint = fixture(tmp_path, fatal=fatal)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log",
            str(log),
            "--checkpoint-root",
            str(checkpoint),
            "--expected-update",
            "2",
            "--launcher-returncode",
            "1",
            "--output",
            str(tmp_path / "receipt.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_accepts_only_complete_post_checkpoint_teardown(tmp_path: Path) -> None:
    result = run(tmp_path)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "accepted_post_checkpoint_teardown"
    assert all(receipt["checks"].values())


def test_rejects_worker_failure_even_after_checkpoint(tmp_path: Path) -> None:
    result = run(tmp_path, fatal="ChildFailedError\n")
    assert result.returncode != 0
    assert not (tmp_path / "receipt.json").exists()
