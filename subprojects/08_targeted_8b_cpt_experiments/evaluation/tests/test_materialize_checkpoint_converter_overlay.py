from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_checkpoint_converter_overlay",
    ROOT / "materialize_checkpoint_converter_overlay.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_builds_a_two_file_overlay_without_changing_the_source(
    tmp_path: Path, monkeypatch
) -> None:
    megatron = tmp_path / "megatron"
    checkpoint = megatron / "tools/checkpoint"
    checkpoint.mkdir(parents=True)
    source_convert = checkpoint / "convert.py"
    source_saver = checkpoint / "saver_swissai_hf.py"
    source_convert.write_text("print('convert')\n", encoding="utf-8")
    source_saver.write_text("VALUE = 'source'\n", encoding="utf-8")
    patch = tmp_path / "overlay.patch"
    patch.write_text(
        """diff --git a/saver_swissai_hf.py b/saver_swissai_hf.py
--- a/saver_swissai_hf.py
+++ b/saver_swissai_hf.py
@@ -1 +1 @@
-VALUE = 'source'
+VALUE = 'overlay'
""",
        encoding="utf-8",
    )
    output = tmp_path / "overlay"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "overlay",
            "--megatron-root",
            str(megatron),
            "--patch",
            str(patch),
            "--output-root",
            str(output),
        ],
    )
    assert MODULE.main() == 0
    assert source_saver.read_text(encoding="utf-8") == "VALUE = 'source'\n"
    assert (output / "convert.py").read_text(encoding="utf-8") == "print('convert')\n"
    assert (output / "saver_swissai_hf.py").read_text(encoding="utf-8") == "VALUE = 'overlay'\n"
    receipt = json.loads((output / "converter_overlay_receipt.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["source_saver"]["sha256"] != receipt["overlay_saver"]["sha256"]
