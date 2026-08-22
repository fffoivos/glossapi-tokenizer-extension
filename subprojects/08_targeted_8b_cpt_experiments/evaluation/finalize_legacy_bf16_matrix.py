#!/usr/bin/env python3
"""Freeze the three-checkpoint legacy-BF16 GreekMMLU comparison."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


UPDATES = (2618, 3218, 3694)
SCORER_SHA256 = "b9f75809b6e617cfd419dc5420e480dee72bb3f1df7fa8f82e04793b4dfd19c4"
REGISTRY_SHA256 = "fcf732c142efdd204fe8a64ac4fb1159f47e7b2bac0e947207c2a971329bf508"


def binding(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"immutable matrix receipt exists: {args.output}")
    scorer = binding(args.scorer)
    registry = binding(args.registry)
    if scorer["sha256"] != SCORER_SHA256 or registry["sha256"] != REGISTRY_SHA256:
        raise ValueError("legacy evaluator bytes do not match cfdd0e7b")

    results = []
    for update in UPDATES:
        directory = args.root / f"iter_{update:07d}"
        headlines = list(directory.glob("*_native_mcq_headline.json"))
        if len(headlines) != 1:
            raise ValueError(f"update {update}: expected one headline file")
        rows = json.loads(headlines[0].read_text(encoding="utf-8"))
        if len(rows) != 1 or rows[0].get("benchmark") != "greekmmlu" or int(rows[0].get("n", -1)) != 16632:
            raise ValueError(f"update {update}: GreekMMLU cardinality drift")
        files = [binding(path) for path in sorted(directory.iterdir()) if path.is_file()]
        results.append(
            {
                "update": update,
                "correct": int(rows[0]["correct"]),
                "n": 16632,
                "accuracy": float(rows[0]["accuracy"]),
                "files": files,
            }
        )

    best = max(results, key=lambda row: row["accuracy"])
    target = 9973 / 16632
    delta_pp = (float(best["accuracy"]) - target) * 100
    payload = {
        "schema_version": "apertus_hard_h2g_legacy_bf16_matrix_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluator_revision": "cfdd0e7b",
        "scorer": scorer,
        "registry": registry,
        "settings": {
            "dtype": "bfloat16",
            "max_input_tokens": 3072,
            "candidate_batch_size": 16,
            "example_batch_size": 16,
            "random_state": 42,
            "questions": 16632,
        },
        "results": results,
        "replication_decision": {
            "historical_target_correct": 9973,
            "historical_target_n": 16632,
            "historical_target_accuracy": target,
            "pre_registered_band_percentage_points": 1.0,
            "best_update": best["update"],
            "best_delta_percentage_points": delta_pp,
            "replicated": abs(delta_pp) <= 1.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
