#!/usr/bin/env python3
"""Fail closed before an LR-floor training segment starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BOUNDARIES = {(0, 2253, 1), (2253, 2574, 2), (2574, 3218, 2)}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(receipt: dict, label: str) -> Path:
    path = Path(receipt["path"]).resolve()
    if not path.is_file() or path.stat().st_size != int(receipt["bytes"]) or sha(path) != receipt["sha256"]:
        raise ValueError(f"frozen file drift: {label}")
    return path


def validate_large_payload(receipt: dict, label: str) -> Path:
    """Check presence/size while trusting the hash bound by a hashed manifest."""
    path = Path(receipt["path"]).resolve()
    expected_sha = str(receipt.get("sha256", ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(receipt["bytes"])
        or len(expected_sha) != 64
    ):
        raise ValueError(f"frozen dataset payload drift: {label}")
    return path


def validate_dataset_inventory(path: Path) -> None:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if (
        dataset.get("schema_version")
        != "apertus8b_lr_floor_dataset_manifest_v1"
        or dataset.get("status") != "completed"
    ):
        raise ValueError("derived dataset manifest drift")
    for phase, phase_value in dataset["phases"].items():
        for row in phase_value["blend"]:
            manifest_path = validate_file(
                row["manifest"], f"phase/{phase}/{row['task_id']}/manifest"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                raise ValueError(f"incomplete shard manifest: {manifest_path}")
            for name in ("bin", "idx"):
                if row["payloads"][name] != manifest["outputs"][name]:
                    raise ValueError(
                        f"dataset payload receipt mismatch: phase/{phase}/{row['task_id']}/{name}"
                    )
                validate_large_payload(
                    row["payloads"][name],
                    f"phase/{phase}/{row['task_id']}/{name}",
                )
    for row in dataset["heldouts"]:
        manifest_path = validate_file(
            row["manifest"], f"heldout/{row['name']}/manifest"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"incomplete heldout manifest: {manifest_path}")
        for name in ("bin", "idx"):
            validate_large_payload(
                manifest["outputs"][name], f"heldout/{row['name']}/{name}"
            )


def validate_tree(receipt: dict) -> None:
    root = Path(receipt["root"]).resolve()
    for row in receipt["files"]:
        validate_file({"path": str(root / row["path"]), "bytes": row["bytes"], "sha256": row["sha256"]}, f"checkpoint/{row['path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--lr-floor-percent", type=int, required=True)
    parser.add_argument("--load-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-receipt", type=Path)
    args = parser.parse_args()
    if (args.start, args.end, args.phase) not in BOUNDARIES or args.lr_floor_percent not in {10, 20, 30}:
        raise ValueError("segment boundary or LR floor is not in the frozen matrix")
    if args.start < 2574 and args.lr_floor_percent != 10:
        raise ValueError("shared prefix must use the T10 placeholder schedule")
    assets_path = args.assets.resolve()
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    if assets.get("schema_version") != "apertus8b_lr_floor_training_assets_v1" or assets.get("status") != "frozen":
        raise ValueError("training assets are not frozen")
    for label in ("recipe", "dataset_manifest", "training_data_env"):
        path = validate_file(assets[label], label)
        if label == "dataset_manifest":
            validate_dataset_inventory(path)
    for label, receipt in assets["dependencies"].items():
        validate_file(receipt, label)
    if args.start == 0:
        if args.resume_receipt is not None:
            raise ValueError("initial segment cannot bind a resume receipt")
        checkpoint = assets["initialization"]["checkpoint"]
        if args.load_checkpoint.resolve() != Path(checkpoint["root"]).resolve():
            raise ValueError("initial checkpoint root drift")
        validate_tree(checkpoint["tree"])
    else:
        if args.resume_receipt is None:
            raise ValueError("resumed segment requires a checkpoint receipt")
        receipt = json.loads(args.resume_receipt.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "apertus8b_lr_floor_checkpoint_v1" or receipt.get("status") != "frozen" or int(receipt.get("iteration", -1)) != args.start:
            raise ValueError("resume checkpoint receipt drift")
        bound = receipt["training_assets_receipt"]
        if Path(bound["path"]).resolve() != assets_path or bound["sha256"] != sha(assets_path):
            raise ValueError("resume checkpoint is bound to other training assets")
        if args.load_checkpoint.resolve() != Path(receipt["checkpoint_root"]).resolve():
            raise ValueError("resume checkpoint root drift")
        validate_tree(receipt["checkpoint_tree"])
        marker = Path(receipt["marker"]["path"])
        if marker.read_text().strip() != str(args.start) or sha(marker) != receipt["marker"]["sha256"]:
            raise ValueError("resume checkpoint marker drift")
    print(json.dumps({"ok": True, "start": args.start, "end": args.end, "phase": args.phase, "lr_floor_percent": args.lr_floor_percent}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
