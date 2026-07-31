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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--load-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start not in BOUNDARIES or BOUNDARIES[args.start] != (args.phase, args.end):
        raise ValueError("segment is not one of the three frozen boundaries")
    assets = read_json(args.assets.resolve())
    if assets.get("schema_version") != "greek_cpt_training_assets_receipt_v1" or assets.get("status") != "frozen":
        raise ValueError("training assets are not frozen")
    validate_file_tree_receipt(assets["megatron"]["tree"])
    validate_tokenizer_tree_receipt(assets["tokenizer"]["tree"])
    for name, receipt in assets["dependencies"].items():
        path = Path(receipt["path"])
        if not path.is_file() or path.stat().st_size != int(receipt["bytes"]) or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"frozen launch dependency drift: {name}")
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
        if int(resume.get("iteration", -1)) != args.start:
            raise ValueError("resume checkpoint iteration drift")
        validate_file_tree_receipt(resume["checkpoint_tree"])
        if args.load_checkpoint.resolve() != Path(resume["checkpoint_tree"]["root"]).resolve():
            raise ValueError("resume checkpoint path differs from receipt")
    print(json.dumps({"ok": True, "start": args.start, "end": args.end, "phase": args.phase}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
