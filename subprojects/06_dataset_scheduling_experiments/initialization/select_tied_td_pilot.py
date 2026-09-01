#!/usr/bin/env python3
"""Select the one tied-TD recipe with the best joint three-slice BPB."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


CANDIDATES = {
    "layer7_mse": (7, "mse"),
    "layer7_mse_ce_auto": (7, "mse_ce_auto"),
    "last_mse": (-1, "mse"),
    "last_mse_ce_auto": (-1, "mse_ce_auto"),
}
SLICES = ("hplt", "non_hplt", "polytonic")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--pilot-token-selection", type=Path, required=True)
    parser.add_argument("--fvt-baseline-dir", type=Path, required=True)
    parser.add_argument("--fvt-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = read_json(args.pilot_token_selection)
    expected_ids_hash = selection["token_ids_file"]["sha256"]
    if (
        selection.get("schema_version") != "apertus_mini_td_pilot_token_selection_v1"
        or selection.get("status") != "frozen"
        or int(selection.get("selected_count", -1)) != 1_024
        or int(selection.get("modern_selected", -1)) <= 0
        or int(selection.get("polytonic_selected", -1)) <= 0
    ):
        raise ValueError("pilot token selection is not the frozen mixed-stage inventory")

    baseline_bpb = {}
    baseline_receipts = {}
    fvt_model_files = {}
    for name in ("config.json", "model.safetensors", "tied_retok_manifest.json", "tokenizer.json"):
        path = args.fvt_model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        fvt_model_files[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    for slice_name in SLICES:
        path = args.fvt_baseline_dir / "metrics" / slice_name / "tokenizer_fair_metrics.json"
        metrics = read_json(path)
        if Path(str(metrics.get("model_path", ""))).resolve() != args.fvt_model_dir.resolve():
            raise ValueError(f"{slice_name} FVT metrics were produced by a different model")
        value = metrics.get("global", {}).get("bpb_bits_per_byte")
        if not isinstance(value, (int, float)) or not 0.0 < float(value) < 100.0:
            raise ValueError(f"invalid {slice_name} FVT baseline BPB")
        baseline_bpb[slice_name] = float(value)
        baseline_receipts[slice_name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    baseline_macro_bpb = sum(baseline_bpb.values()) / len(SLICES)

    rows = []
    for pilot_id, (target_layer, loss_profile) in CANDIDATES.items():
        root = args.pilot_root / pilot_id
        manifest_path = root / "tied_td_manifest.json"
        verification_path = root / "initialization_verification.json"
        manifest = read_json(manifest_path)
        verification = read_json(verification_path)
        if (
            manifest.get("status") != "completed"
            or manifest.get("scope") != "pilot"
            or int(manifest.get("requested_token_count", -1)) != 1_024
            or int(manifest.get("trained_token_count", -1)) != 1_024
            or float(manifest.get("trained_token_fraction", 0.0)) != 1.0
            or int(manifest.get("target_layer", -99)) != target_layer
            or manifest.get("loss_profile") != loss_profile
            or manifest.get("token_ids_file_sha256") != expected_ids_hash
            or manifest.get("input_output_share_storage") is not True
            or manifest.get("norm_collapse_gate_passed") is not True
        ):
            raise ValueError(f"TD manifest failed the frozen pilot contract: {pilot_id}")
        if verification.get("status") != "pass" or not all(
            verification.get("checks", {}).values()
        ):
            raise ValueError(f"structural/collapse gate failed: {pilot_id}")
        bpb = {}
        metrics_receipts = {}
        for slice_name in SLICES:
            path = root / "metrics" / slice_name / "tokenizer_fair_metrics.json"
            metrics = read_json(path)
            if Path(str(metrics.get("model_path", ""))).resolve() != root.resolve():
                raise ValueError(f"{slice_name} metrics were produced by a different model for {pilot_id}")
            value = metrics.get("global", {}).get("bpb_bits_per_byte")
            if not isinstance(value, (int, float)) or not 0.0 < float(value) < 100.0:
                raise ValueError(f"invalid {slice_name} BPB for {pilot_id}")
            bpb[slice_name] = float(value)
            metrics_receipts[slice_name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        rows.append(
            {
                "pilot_id": pilot_id,
                "target_layer": target_layer,
                "loss_profile": loss_profile,
                "bpb": bpb,
                "macro_bpb": sum(bpb.values()) / len(SLICES),
                "delta_macro_bpb_vs_fvt": (
                    sum(bpb.values()) / len(SLICES) - baseline_macro_bpb
                ),
                "manifest": {
                    "path": str(manifest_path.resolve()),
                    "sha256": sha256_file(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                },
                "verification": {
                    "path": str(verification_path.resolve()),
                    "sha256": sha256_file(verification_path),
                    "bytes": verification_path.stat().st_size,
                },
                "metrics": metrics_receipts,
            }
        )
    winner = min(rows, key=lambda row: (row["macro_bpb"], row["pilot_id"]))
    if winner["macro_bpb"] > baseline_macro_bpb:
        raise ValueError(
            "all tied-TD pilots regress macro BPB versus the frozen tied-FVT baseline: "
            f"best={winner['macro_bpb']:.12g} baseline={baseline_macro_bpb:.12g}"
        )
    payload = {
        "schema_version": "apertus_mini_tied_td_pilot_selection_v1",
        "status": "selected",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_rule": "minimum unweighted macro BPB over fixed HPLT, non-HPLT and polytonic slices among candidates passing every structural and collapse gate; fail closed if the winner regresses macro BPB versus the frozen tied-FVT baseline",
        "checkpoint_averaging": False,
        "fvt_baseline": {
            "model_dir": str(args.fvt_model_dir.resolve()),
            "model_files": fvt_model_files,
            "bpb": baseline_bpb,
            "macro_bpb": baseline_macro_bpb,
            "metrics": baseline_receipts,
        },
        "pilot_token_selection": {
            "path": str(args.pilot_token_selection.resolve()),
            "sha256": sha256_file(args.pilot_token_selection),
        },
        "candidates": rows,
        "selected_pilot_id": winner["pilot_id"],
        "selected_target_layer": winner["target_layer"],
        "selected_loss_profile": winner["loss_profile"],
        "selected_macro_bpb": winner["macro_bpb"],
        "selected_delta_macro_bpb_vs_fvt": winner["delta_macro_bpb_vs_fvt"],
        "fvt_macro_non_regression_gate_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "selected": winner["pilot_id"], "macro_bpb": winner["macro_bpb"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
