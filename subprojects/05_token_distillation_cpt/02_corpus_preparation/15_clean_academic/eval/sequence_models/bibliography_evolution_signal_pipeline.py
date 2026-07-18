#!/usr/bin/env python3
"""Train one signal-TCN variant and evaluate its frozen anchored prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train = output / "train"
    validation = output / "validation"
    common = ["--code-commit", args.code_commit, "--slurm-job-id", args.slurm_job_id]
    train_command = [
        sys.executable, "-m", "sequence_models.bibliography_signal_tcn",
        "--input", args.input, "--table-dir", args.train_table_dir,
        "--line-oof-dir", args.line_oof_dir, "--block-oof-dir", args.block_oof_dir,
        "--deterministic-roles-dir", args.deterministic_roles_dir,
        "--quality-decisions", args.train_quality_decisions,
        "--output-dir", str(train), "--line-arm", "D1",
        "--hidden-dim", str(args.hidden_dim), "--dilations", *(str(value) for value in args.dilations),
        "--dropout", str(args.dropout), "--epochs", str(args.epochs),
        "--seed", str(args.seed), "--workers", str(args.workers), "--cpus", str(args.cpus),
        *common,
    ]
    subprocess.run(train_command, check=True)
    validation_command = [
        sys.executable, "-m", "sequence_models.bibliography_signal_validation",
        "--input", args.input, "--validation-table-dir", args.validation_table_dir,
        "--validation-line-probability", args.validation_line_probability,
        "--signal-tcn-dir", str(train), "--train-recall-block-dir", args.train_recall_block_dir,
        "--quality-decisions", args.validation_quality_decisions,
        "--policy", args.validation_policy, "--output-dir", str(validation),
        "--workers", str(args.workers), "--torch-threads", str(args.cpus),
        *common,
    ]
    subprocess.run(validation_command, check=True)
    source_prediction = validation / "recall_first_anchored.prediction.npy"
    prediction = output / "prediction.npy"
    shutil.copyfile(source_prediction, prediction)
    scope = np.load(validation / "validation_auxiliary_scope.npy", allow_pickle=False).astype(bool)
    with (output / "combined_barriers.npz").open("xb") as handle:
        np.savez(
            handle,
            hard_wall=scope,
            upward_stop=np.zeros(len(scope), dtype=bool),
            downward_stop=np.zeros(len(scope), dtype=bool),
        )
    result = {
        "schema_version": "bibliography-evolution-signal-pipeline-v1",
        "status": "passed_train_oof_and_fixed_development_evaluation",
        "validation_opened": True,
        "final_test_opened": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "architecture": {"hidden_dim": args.hidden_dim, "dilations": args.dilations, "dropout": args.dropout},
        "prediction_sha256": _sha256(prediction),
        "train_receipt_sha256": _sha256(train / "receipt.json"),
        "validation_receipt_sha256": _sha256(validation / "receipt.json"),
    }
    _write(output / "report.json", result)
    _write(output / "receipt.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--block-oof-dir", required=True)
    parser.add_argument("--deterministic-roles-dir", required=True)
    parser.add_argument("--train-quality-decisions", required=True)
    parser.add_argument("--validation-table-dir", required=True)
    parser.add_argument("--validation-line-probability", required=True)
    parser.add_argument("--train-recall-block-dir", required=True)
    parser.add_argument("--validation-quality-decisions", required=True)
    parser.add_argument("--validation-policy", required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--dilations", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpus", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
