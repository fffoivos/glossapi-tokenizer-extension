from __future__ import annotations

import json
import subprocess
import sys


def test_module_cli_invocation_executes_app() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "glossapi_corpus_cli.cli", "estimate", "--nanochat-depth", "4"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["nanochat_depth"] == 4
    assert "estimated_model_dim" in payload
