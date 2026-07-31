#!/usr/bin/env python3
"""Revalidate all frozen assets and exact segment semantics at Slurm job start."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROBE_ROOT = HERE.parent
SHARED = PROBE_ROOT.parent / "05_training_dataset_bridge" / "scripts"
sys.path.insert(0, str(SHARED))

from bridge_common import (  # noqa: E402
    read_json,
    sha256_file,
    validate_file_tree_receipt,
    validate_tokenizer_tree_receipt,
)


BOUNDARIES = {0: (1, 1785), 1785: (1, 3570), 3570: (2, 5960)}
SMOKE_BOUNDARIES = {0: (1, 1), 1: (2, 2)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--load-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--smoke-verification", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="validate the isolated 0..1..2 smoke chain instead of production boundaries",
    )
    return parser.parse_args()


def _validate_file_binding(value: dict, label: str) -> Path:
    path = Path(str(value.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(value.get("bytes", -1))
        or sha256_file(path) != value.get("sha256")
    ):
        raise ValueError(f"smoke evidence drift: {label}")
    return path


def _validate_smoke_verification(path: Path, assets_path: Path) -> None:
    value = read_json(path.resolve())
    if (
        value.get("schema_version") != "greek_cpt_two_phase_smoke_verification_v1"
        or value.get("status") != "passed"
    ):
        raise ValueError("two-phase GPU smoke has not passed")
    bound_assets = value.get("training_assets_receipt", {})
    if (
        Path(str(bound_assets.get("path", ""))).resolve() != assets_path
        or bound_assets.get("sha256") != sha256_file(assets_path)
    ):
        raise ValueError("GPU smoke is bound to different training assets")
    if value.get("geometry") != {
        "nodes": 1,
        "gpus_per_node": 4,
        "tensor_parallel": 2,
        "global_batch_samples": 8,
        "microbatch_size": 2,
        "iterations": 2,
    }:
        raise ValueError("GPU smoke geometry drift")
    for name, iteration in (("phase1", 1), ("phase2", 2)):
        phase = value.get(name, {})
        receipt_path = _validate_file_binding(
            phase.get("checkpoint_receipt", {}), f"{name} checkpoint receipt"
        )
        _validate_file_binding(phase.get("metadata", {}), f"{name} metadata")
        log = phase.get("log_evidence", {})
        if log.get("iteration") != iteration:
            raise ValueError(f"GPU smoke iteration drift: {name}")
        _validate_file_binding(log.get("stdout", {}), f"{name} stdout")
        _validate_file_binding(log.get("stderr", {}), f"{name} stderr")
        checkpoint = read_json(receipt_path)
        if (
            checkpoint.get("schema_version")
            != "greek_cpt_resume_checkpoint_receipt_v1"
            or checkpoint.get("status") != "frozen"
            or checkpoint.get("smoke") is not True
            or checkpoint.get("iteration") != iteration
            or checkpoint.get("training_assets_receipt", {}).get("sha256")
            != sha256_file(assets_path)
        ):
            raise ValueError(f"GPU smoke checkpoint binding drift: {name}")
    if value.get("phase1", {}).get("data_prefix") == value.get("phase2", {}).get("data_prefix"):
        raise ValueError("GPU smoke did not switch phase blends")
    if value.get("phase2", {}).get("phase_relative_data_index") != "8 -> 0":
        raise ValueError("GPU smoke did not prove the phase-relative data index")


def main() -> int:
    args = parse_args()
    boundaries = SMOKE_BOUNDARIES if args.smoke else BOUNDARIES
    if args.start not in boundaries or boundaries[args.start] != (args.phase, args.end):
        kind = "smoke" if args.smoke else "production"
        raise ValueError(f"segment is not a frozen {kind} boundary")
    assets_path = args.assets.resolve()
    assets = read_json(assets_path)
    if assets.get("schema_version") != "greek_cpt_training_assets_receipt_v1" or assets.get("status") != "frozen":
        raise ValueError("training assets are not frozen")
    validate_file_tree_receipt(assets["megatron"]["tree"])
    validate_tokenizer_tree_receipt(assets["tokenizer"]["tree"])
    for name, receipt in assets["hf_conversion_template"]["files"].items():
        path = Path(receipt["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(receipt["bytes"])
            or sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"frozen HF conversion template drift: {name}")
    for name, receipt in assets["dependencies"].items():
        path = Path(receipt["path"])
        if not path.is_file() or path.stat().st_size != int(receipt["bytes"]) or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"frozen launch dependency drift: {name}")
    if args.smoke:
        if args.smoke_verification is not None:
            raise ValueError("smoke segments cannot claim a pre-existing smoke verification")
    else:
        if args.smoke_verification is None:
            raise ValueError("production segments require a passed two-phase GPU smoke")
        _validate_smoke_verification(args.smoke_verification, assets_path)
    if args.start == 0:
        if args.resume_receipt is not None:
            raise ValueError("initial segment must not bind a resume receipt")
        validate_file_tree_receipt(assets["init_checkpoint"]["tree"])
        if args.load_checkpoint.resolve() != Path(assets["init_checkpoint"]["root"]).resolve():
            raise ValueError("initial checkpoint path differs from frozen assets")
    else:
        if args.resume_receipt is None:
            raise ValueError("resume segment requires a checkpoint receipt")
        resume = read_json(args.resume_receipt.resolve())
        if resume.get("schema_version") != "greek_cpt_resume_checkpoint_receipt_v1" or resume.get("status") != "frozen":
            raise ValueError("resume checkpoint receipt is invalid")
        if bool(resume.get("smoke", False)) != args.smoke:
            raise ValueError("resume checkpoint smoke/production mode drift")
        if int(resume.get("iteration", -1)) != args.start:
            raise ValueError("resume checkpoint iteration drift")
        bound_assets = resume.get("training_assets_receipt", {})
        if (
            Path(bound_assets.get("path", "")).resolve() != assets_path
            or bound_assets.get("sha256") != sha256_file(assets_path)
        ):
            raise ValueError("resume checkpoint is bound to different training assets")
        validate_file_tree_receipt(resume["checkpoint_tree"])
        if args.load_checkpoint.resolve() != Path(resume["checkpoint_tree"]["root"]).resolve():
            raise ValueError("resume checkpoint path differs from receipt")
    print(json.dumps({"ok": True, "smoke": args.smoke, "start": args.start, "end": args.end, "phase": args.phase}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
