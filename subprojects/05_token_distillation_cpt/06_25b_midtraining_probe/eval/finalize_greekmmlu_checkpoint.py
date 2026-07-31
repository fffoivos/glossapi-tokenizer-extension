#!/usr/bin/env python3
"""Validate one completed GreekMMLU checkpoint evaluation and freeze evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROBE_ROOT = HERE.parent
SHARED = PROBE_ROOT.parent / "05_training_dataset_bridge" / "scripts"
sys.path.insert(0, str(SHARED))

from bridge_common import sha256_file, utc_now, write_json_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--training-assets-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    args = parse_args()
    if args.iteration <= 0 or args.tokens != args.iteration * 4_194_304:
        raise ValueError("iteration/token accounting drift")
    checkpoint = args.checkpoint_dir.resolve()
    if not checkpoint.is_dir() or not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(checkpoint)
    hf_dir = args.hf_dir.resolve()
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        if not (hf_dir / name).is_file():
            raise FileNotFoundError(hf_dir / name)
    shards = sorted(hf_dir.glob("model-*-of-*.safetensors"))
    if not shards:
        raise ValueError("converted HF checkpoint has no model shards")

    eval_dir = args.eval_dir.resolve()
    aggregates = sorted(eval_dir.glob("*_native_mcq_aggregate.json"))
    if len(aggregates) != 1:
        raise ValueError("expected exactly one native Greek aggregate")
    aggregate = json.loads(aggregates[0].read_text(encoding="utf-8"))
    headline = aggregate.get("headline", {})
    accuracy = float(headline.get("micro_accuracy", float("nan")))
    if (
        aggregate.get("schema") != "native-greek-mcq-aggregate-v1"
        or headline.get("n_tasks") != 1
        or headline.get("total_n") != 16_632
        or not math.isfinite(accuracy)
        or not 0.0 <= accuracy <= 1.0
    ):
        raise ValueError("GreekMMLU aggregate drift")
    eval_files = [_file(path) for path in sorted(eval_dir.rglob("*")) if path.is_file()]
    if not eval_files:
        raise ValueError("GreekMMLU output directory is empty")
    assets_path = args.training_assets_receipt.resolve()
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    if (
        assets.get("schema_version") != "greek_cpt_training_assets_receipt_v1"
        or assets.get("status") != "frozen"
    ):
        raise ValueError("training assets are not frozen")
    payload = {
        "schema_version": "greek_cpt_checkpoint_greekmmlu_evaluation_v1",
        "status": "passed",
        "completed_at": utc_now(),
        "iteration": args.iteration,
        "tokens": args.tokens,
        "checkpoint": {
            "root": str(checkpoint),
            "metadata": _file(checkpoint / ".metadata"),
        },
        "hf_conversion": {
            "root": str(hf_dir),
            "config": _file(hf_dir / "config.json"),
            "tokenizer": _file(hf_dir / "tokenizer.json"),
            "index": _file(hf_dir / "model.safetensors.index.json"),
            "shard_count": len(shards),
            "total_model_bytes": sum(path.stat().st_size for path in shards),
        },
        "greekmmlu": {
            "aggregate": _file(aggregates[0]),
            "total_n": 16_632,
            "micro_accuracy": accuracy,
            "files": eval_files,
        },
        "training_assets_receipt": _file(assets_path),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "iteration": args.iteration, "accuracy": accuracy}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
