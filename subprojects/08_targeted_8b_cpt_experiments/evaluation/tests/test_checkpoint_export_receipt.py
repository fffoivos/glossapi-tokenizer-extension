from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "checkpoint_export_receipt", ROOT / "checkpoint_export_receipt.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def log(agreement: float) -> str:
    return f"""
Converted semantic parity dtype: float32
Converted model agrees on {agreement:.2f}% of predictions
Converted logits are close on 95.78% of values
Converted mean KL divergence: 1.000000000000e-06
Converted max KL divergence: 1.000000000000e-03
Converted mean total variation: 1.000000000000e-04
Converted max total variation: 1.000000000000e-02
Converted mean top-token log-prob absolute difference: 1.000000000000e-03
Converted p99 top-token log-prob absolute difference: 1.000000000000e-02
Converted p99.9 top-token log-prob absolute difference: 2.000000000000e-02
Converted max top-token log-prob absolute difference: 3.000000000000e-02
"""


def test_uses_the_pinned_converter_native_argmax_threshold() -> None:
    result = MODULE.parse_runtime_parity(log(99.49))
    assert result["runtime_semantic_parity_passed"] is True
    assert result["minimum_prediction_agreement_percent"] == 99.0
    assert result["native_argmax_threshold_passed"] is True
    assert result["diagnostic_99_5_percent_agreement_passed"] is False


def test_rejects_a_value_below_the_native_converter_threshold() -> None:
    with pytest.raises(ValueError, match="runtime semantic parity"):
        MODULE.parse_runtime_parity(log(98.99))
