#!/usr/bin/env python3
"""Recompute G0 from locked inputs, then require byte-identical prediction."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from .bibliography_evolution_contract import load_json, sha256_file, verify_g0
from .bibliography_evolution_core_decode import parse_args as decode_args
from .bibliography_evolution_core_decode import run as run_decode


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_json(Path(args.lock))
    config = lock["decoder_config"]
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    decode = output / "decode"
    run_decode(
        decode_args(
            [
                "--table-dir", args.validation_table_dir,
                "--signal-probability", args.validation_signal_probability,
                "--line-probability", args.validation_line_probability,
                "--scope-mask", args.validation_scope_mask,
                "--qualified-documents", args.qualified_documents,
                "--anchor-probability", str(config["anchor_probability"]),
                "--anchors-required", str(config["anchors_required"]),
                "--anchor-window", str(config["anchor_window"]),
                "--maximum-bridge-gap", str(config["maximum_bridge_gap"]),
                "--inside-probability", str(config["inside_probability"]),
                "--adjacent-expansion", str(config["adjacent_expansion"]),
                "--header-window", str(config["header_window"]),
                "--decode-all-documents",
                "--output-dir", str(decode),
                "--code-commit", args.code_commit,
                "--slurm-job-id", args.slurm_job_id,
            ]
        )
    )
    generated = decode / "prediction.npy"
    verification = verify_g0(
        lock, root=Path(args.authoritative_root), replay_prediction=generated
    )
    prediction = output / "prediction.npy"
    shutil.copyfile(generated, prediction)
    shutil.copyfile(decode / "combined_barriers.npz", output / "combined_barriers.npz")
    result = {
        "schema_version": "bibliography-evolution-g0-executed-replay-v1",
        "status": "passed_executed_byte_identical_replay",
        "generated_prediction_sha256": sha256_file(prediction),
        "decode_receipt_sha256": sha256_file(decode / "receipt.json"),
        "verification": verification,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
    }
    _write(output / "report.json", result)
    _write(output / "receipt.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--authoritative-root", required=True)
    parser.add_argument("--validation-table-dir", required=True)
    parser.add_argument("--validation-signal-probability", required=True)
    parser.add_argument("--validation-line-probability", required=True)
    parser.add_argument("--validation-scope-mask", required=True)
    parser.add_argument("--qualified-documents", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
