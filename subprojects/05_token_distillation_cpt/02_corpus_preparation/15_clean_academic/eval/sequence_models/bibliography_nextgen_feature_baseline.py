#!/usr/bin/env python3
"""Expose one frozen OOF feature signal as a nextgen decoder baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_nextgen_models import SCHEMA_VERSION as MODEL_SCHEMA
from .bibliography_nextgen_table import SCHEMA_VERSION as TABLE_SCHEMA
from .contract import sha256_file


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_root = Path(args.table_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest = json.loads((table_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TABLE_SCHEMA or manifest.get("test_opened") is not False:
        raise ValueError("feature baseline requires the sealed development table")
    names = tuple(manifest["feature_names"])
    if args.feature not in names or not args.feature.startswith("probability:"):
        raise ValueError("baseline must name a probability feature")
    features = np.load(table_root / "features.npy", mmap_mode="r", allow_pickle=False)
    probability = np.asarray(features[:, names.index(args.feature)], dtype=np.float32)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("baseline feature is not a probability")
    output.mkdir(parents=True)
    with (output / "oof_probability.npy").open("xb") as handle:
        np.save(handle, probability, allow_pickle=False)
    report = {
        "schema_version": MODEL_SCHEMA,
        "status": "passed_frozen_feature_oof_baseline",
        "kind": f"feature_baseline:{args.feature}",
        "feature": args.feature,
        "validation_opened": False,
        "test_opened": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "table_manifest_sha256": sha256_file(table_root / "manifest.json"),
            "table_features_sha256": sha256_file(table_root / "features.npy"),
        },
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(
        output / "receipt.json",
        {
            **report,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
