from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "agent1_v3_structural_gate.py"


def _file(path: Path, value: str = "{}\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_absent_agent2_handoff_is_explicit_noop(tmp_path: Path) -> None:
    pre = _file(tmp_path / "pre.json", json.dumps({"schema_version": "agent1_full_corpus_v3_prestructural_manifest_v1"}))
    output = tmp_path / "gate.json"
    subprocess.run([sys.executable, str(SCRIPT), "gate", "--prestructural-manifest", str(pre), "--output", str(output)], check=True)
    value = json.loads(output.read_text())
    assert value["mode"] == "no_op"
    assert value["publish_permitted"] is False


def test_agent2_handoff_requires_every_safety_gate(tmp_path: Path) -> None:
    pre = _file(tmp_path / "pre.json", json.dumps({"schema_version": "agent1_full_corpus_v3_prestructural_manifest_v1"}))
    handoff = _file(tmp_path / "handoff.json", json.dumps({"ready_for_corpus_application": True}))
    result = subprocess.run([sys.executable, str(SCRIPT), "gate", "--prestructural-manifest", str(pre), "--model-handoff", str(handoff), "--output", str(tmp_path / "gate.json")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "required gate" in result.stderr
