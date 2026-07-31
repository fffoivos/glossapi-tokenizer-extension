#!/usr/bin/env python3
"""Freeze the exact GreekMMLU filter implementation, inputs, and policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROBE_ROOT = HERE.parent
SHARED = PROBE_ROOT.parent / "05_training_dataset_bridge" / "scripts"
DEFAULT_IMPLEMENTATION = (
    PROBE_ROOT.parent
    / "04_full_corpus_preparation"
    / "scripts"
    / "decontaminate_full_corpus.py"
)
sys.path.insert(0, str(SHARED))

from bridge_common import read_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


def _receipt(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _load(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("cpt25b_greekmmlu_filter", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, default=DEFAULT_IMPLEMENTATION)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    implementation = args.implementation.resolve()
    queries = args.queries.resolve()
    benchmark_manifest = args.benchmark_manifest.resolve()
    module = _load(implementation)
    policy = {
        "policy_version": module.POLICY_VERSION,
        "k": int(module.DEFAULT_K),
        "min_coverage": float(module.DEFAULT_MIN_COVERAGE),
        "minhash_threshold": float(module.DEFAULT_MINHASH_THRESHOLD),
        "min_matched_grams": int(module.DEFAULT_MIN_MATCHED_GRAMS),
        "max_gap_tokens": int(module.DEFAULT_MAX_GAP),
        "drop_rules": [
            "greekmmlu_exact_prompt",
            "greekmmlu_exact_question_answer",
            "greekmmlu_ngram_minhash_answer",
        ],
        "answer_only_action": "audit_only",
        "missing_correct_answer_action": (
            "retain_item_for_exact_prompt_matching_and_disable_answer_dependent_drops"
        ),
    }
    index, benchmark = module.load_benchmark_index(
        queries,
        benchmark_manifest,
        k=policy["k"],
        min_coverage=policy["min_coverage"],
        minhash_threshold=policy["minhash_threshold"],
        min_matched_grams=policy["min_matched_grams"],
        max_gap_tokens=policy["max_gap_tokens"],
    )
    if not index.items:
        raise ValueError("GreekMMLU benchmark index is empty")
    payload = {
        "schema_version": "greek_cpt_decontamination_binding_v1",
        "status": "frozen",
        "created_at": utc_now(),
        "implementation": _receipt(implementation),
        "queries": _receipt(queries),
        "benchmark_manifest": _receipt(benchmark_manifest),
        "benchmark": benchmark,
        "policy": policy,
    }
    output = args.output.resolve()
    if output.exists():
        existing = read_json(output)
        for key in ("implementation", "queries", "benchmark_manifest", "policy"):
            if existing.get(key) != payload[key]:
                raise ValueError(f"existing decontamination binding drift: {key}")
        print(json.dumps({"ok": True, "resumed": True, "output": str(output)}, sort_keys=True))
        return 0
    write_json_atomic(output, payload)
    print(json.dumps({"ok": True, "output": str(output), "items": len(index.items)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
