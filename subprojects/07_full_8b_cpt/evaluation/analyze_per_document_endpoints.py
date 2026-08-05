#!/usr/bin/env python3
"""Compare endpoint BPB with paired document/cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arm(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("arms must be NAME=/path/to/model/output")
    return name, Path(raw_path)


def read_units(path: Path, use_cluster: bool) -> dict[str, tuple[float, int]]:
    units: dict[str, list[float | int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            unit = row.get("cluster_id") if use_cluster else row.get("doc_id")
            if unit is None:
                raise ValueError(f"missing bootstrap unit in {path}")
            current = units.setdefault(str(unit), [0.0, 0])
            current[0] += float(row["nll_numerator_nats"])
            current[1] += int(row["utf8_bytes"])
    if not units or any(int(value[1]) <= 0 for value in units.values()):
        raise ValueError(f"empty or zero-byte units in {path}")
    return {key: (float(value[0]), int(value[1])) for key, value in units.items()}


def load_panel(root: Path, panel: dict[str, Any]) -> dict[str, tuple[float, int]]:
    name = panel["name"]
    receipt_path = root / f"{name}.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed":
        raise ValueError(f"incomplete receipt: {receipt_path}")
    if receipt.get("input", {}).get("sha256") != panel["raw_jsonl"]["sha256"]:
        raise ValueError(f"input drift for {name} in {root}")
    output = Path(receipt["output"]["path"])
    if not output.is_file() or sha256_file(output) != receipt["output"]["sha256"]:
        raise ValueError(f"output drift: {output}")
    return read_units(output, panel.get("cluster_field") is not None)


def bpb(nll: np.ndarray | float, byte_count: np.ndarray | float) -> np.ndarray | float:
    return nll / (math.log(2.0) * byte_count)


def paired_bootstrap(
    base: dict[str, tuple[float, int]],
    candidate: dict[str, tuple[float, int]],
    *,
    samples: int,
    seed: int,
    chunk: int = 128,
) -> dict[str, Any]:
    if set(base) != set(candidate):
        only_base = len(set(base) - set(candidate))
        only_candidate = len(set(candidate) - set(base))
        raise ValueError(f"unpaired units: base_only={only_base}, candidate_only={only_candidate}")
    keys = sorted(base)
    base_nll = np.asarray([base[key][0] for key in keys], dtype=np.float64)
    base_bytes = np.asarray([base[key][1] for key in keys], dtype=np.float64)
    cand_nll = np.asarray([candidate[key][0] for key in keys], dtype=np.float64)
    cand_bytes = np.asarray([candidate[key][1] for key in keys], dtype=np.float64)
    point = float(bpb(cand_nll.sum(), cand_bytes.sum()) - bpb(base_nll.sum(), base_bytes.sum()))
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        indices = rng.integers(0, len(keys), size=(stop - start, len(keys)))
        deltas[start:stop] = (
            bpb(cand_nll[indices].sum(axis=1), cand_bytes[indices].sum(axis=1))
            - bpb(base_nll[indices].sum(axis=1), base_bytes[indices].sum(axis=1))
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "units": len(keys),
        "candidate_minus_baseline_bpb": point,
        "percentile_95_ci": [float(low), float(high)],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    arms = dict(args.arm)
    if len(arms) != len(args.arm) or args.baseline not in arms:
        raise ValueError("arm names must be unique and include the baseline")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {
        "apertus_full_8b_validation_manifest_v1",
        "apertus_per_document_validation_manifest_v1",
    }:
        raise ValueError("unsupported validation manifest")
    result: dict[str, Any] = {
        "schema_version": "apertus_per_document_endpoint_comparison_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline": args.baseline,
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256_file(args.manifest)},
        "panels": {},
    }
    for panel_index, panel in enumerate(manifest["panels"]):
        loaded = {name: load_panel(root, panel) for name, root in arms.items()}
        base = loaded[args.baseline]
        aggregate = {
            name: {
                "units": len(rows),
                "bpb": float(bpb(sum(x[0] for x in rows.values()), sum(x[1] for x in rows.values()))),
            }
            for name, rows in loaded.items()
        }
        comparisons = {
            name: paired_bootstrap(
                base,
                rows,
                samples=args.samples,
                seed=args.seed + panel_index * 100 + arm_index,
            )
            for arm_index, (name, rows) in enumerate(loaded.items())
            if name != args.baseline
        }
        result["panels"][panel["name"]] = {
            "bootstrap_unit": panel["bootstrap_unit"],
            "aggregate": aggregate,
            "paired_vs_baseline": comparisons,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "panels": len(result["panels"]), "arms": len(arms)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
