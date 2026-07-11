from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
HERE = Path(__file__).resolve().parents[1]
ACADEMIC = HERE.parent / "02_corpus_preparation" / "15_clean_academic"
EVAL = ACADEMIC / "eval"
BINARY = ACADEMIC / "reference_detector" / "target" / "debug" / "reference_detect"
sys.path.insert(0, str(EVAL))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LINE_LR = load_module("phase04_line_lr", EVAL / "line_lr.py")
SIGNALS = load_module("phase04_span_signals", EVAL / "span_signals.py")


def python_scores(lines: list[str], *, toc: bool) -> list[float]:
    document = {"lines": list(enumerate(lines)), "N": len(lines)}
    model_path = EVAL / ("toc_line_lr_model.json" if toc else "span_line_lr_struct_model.json")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    rows = []
    for index, features in enumerate(LINE_LR.doc_features(document)):
        values = [features[name] for name in LINE_LR.FEATS]
        if toc:
            own = SIGNALS.toc_signals(lines[index])
            values.extend(own[name] for name in SIGNALS.TOC_KEYS)
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float64)
    mu = np.asarray(model["mu"], dtype=np.float64)
    sd = np.asarray(model["sd"], dtype=np.float64)
    weight = np.asarray(model["weight"], dtype=np.float64)
    bias = float(model["bias"])
    scores = 1.0 / (1.0 + np.exp(-(((matrix - mu) / sd) @ weight + bias)))
    if toc:
        scores *= np.asarray([index < int(0.30 * len(lines)) for index in range(len(lines))])
    return scores.tolist()


@pytest.mark.skipif(not BINARY.is_file(), reason="run cargo build before the Phase-04 tests")
@pytest.mark.parametrize("toc", [False, True])
def test_unicode_edge_feature_parity_for_frozen_heads(tmp_path: Path, toc: bool) -> None:
    lines = [f"a{chr(codepoint)}" for codepoint in range(0x0370, 0x0400)]
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps({"present": lines, "abs_idx": list(range(len(lines))), "n_total": len(lines)}) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(BINARY),
            "--mode",
            "toc-score-lines" if toc else "score-lines",
            "--input",
            str(input_path),
            "--out-spans",
            str(tmp_path / "unused.jsonl"),
            "--out-counters",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rust = json.loads(output_path.read_text(encoding="utf-8"))["p"]
    python = python_scores(lines, toc=toc)
    assert max(abs(left - right) for left, right in zip(python, rust)) < 1e-12
