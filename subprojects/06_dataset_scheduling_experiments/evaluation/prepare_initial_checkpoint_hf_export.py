#!/usr/bin/env python3
"""Bind iteration zero to the canonical HF source of the training release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_reference(reference: dict[str, Any]) -> Path:
    path = Path(reference["path"]).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(reference["bytes"])
        or sha256_file(path) != reference["sha256"]
    ):
        raise ValueError(f"artifact reference drift: {path}")
    return path


def verify_tree(root: Path, rows: list[dict[str, Any]]) -> None:
    expected = {row["relative_path"] for row in rows}
    observed = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise ValueError(f"artifact tree inventory drift: {root}")
    for row in rows:
        path = root / row["relative_path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"artifact tree file drift: {path}")


def prepare(
    campaign_manifest: Path,
    source_checkpoint_root: Path,
    output_root: Path,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    campaign_manifest = campaign_manifest.resolve()
    campaign = read_json(campaign_manifest)
    if (
        campaign.get("schema_version") != "apertus_mini_campaign_manifest_v1"
        or campaign.get("status") != "frozen"
    ):
        raise ValueError("campaign manifest is not frozen")
    assets = campaign["assets"]
    source_checkpoint_root = source_checkpoint_root.resolve()
    if source_checkpoint_root != Path(assets["initial_checkpoint_root"]).resolve():
        raise ValueError("iteration-zero source root is not the campaign initialization")

    initialization_path = verify_reference(assets["initialization_receipt"])
    initialization = read_json(initialization_path)
    if (
        initialization.get("schema_version")
        != "apertus_mini_tied_td_initialization_v1"
        or initialization.get("status") != "passed"
        or Path(initialization["initial_checkpoint_root"]).resolve()
        != source_checkpoint_root
        or initialization.get("input_output_embeddings_tied") is not True
    ):
        raise ValueError("initialization receipt drift")
    conversion_path = verify_reference(initialization["evidence"]["conversion"])
    conversion = read_json(conversion_path)
    required_checks = {
        "all_apertus_extra_tensors_within_tolerance",
        "all_standard_tensors_within_tolerance",
        "fixed_prompt_top1_logits_match",
        "pipeline_parallel_size_one",
        "tensor_parallel_size_one",
        "torch_dist_release_complete",
    }
    if (
        conversion.get("schema_version") != "apertus_mini_td_megatron_conversion_v2"
        or conversion.get("status") != "passed"
        or Path(conversion["initial_checkpoint_root"]).resolve()
        != source_checkpoint_root
        or any(conversion.get("checks", {}).get(key) is not True for key in required_checks)
    ):
        raise ValueError("TD conversion receipt drift")

    release = source_checkpoint_root / "release"
    release_rows = conversion["release_tree"]
    verify_tree(release, release_rows)
    hf_reference = Path(conversion["hf_reference"]).resolve()
    hf_rows = conversion["hf_reference_tree"]
    verify_tree(hf_reference, hf_rows)
    reference_model = next(
        row for row in hf_rows if row["relative_path"] == "model.safetensors"
    )
    roundtrip_model = next(
        row
        for row in conversion["hf_roundtrip_tree"]
        if row["relative_path"] == "model.safetensors"
    )
    if reference_model["sha256"] != roundtrip_model["sha256"]:
        raise ValueError("HF roundtrip model is not bitwise identical to the TD model")

    tokenizer_json = tokenizer_dir.resolve() / "tokenizer.json"
    reference_tokenizer = hf_reference / "tokenizer.json"
    if sha256_file(reference_tokenizer) != sha256_file(tokenizer_json):
        raise ValueError("canonical TD tokenizer differs from the campaign tokenizer")
    config = read_json(hf_reference / "config.json")
    expected_geometry = {
        "vocab_size": 148992,
        "hidden_size": 1024,
        "intermediate_size": 6144,
        "num_hidden_layers": 20,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "tie_word_embeddings": True,
        "rope_theta": 500000.0,
        # Apertus serializes the unscaled/default RoPE geometry explicitly.
        "rope_scaling": {"rope_type": "default"},
    }
    if any(config.get(key) != value for key, value in expected_geometry.items()):
        raise ValueError("canonical TD HF geometry drift")

    output_root = output_root.resolve()
    if output_root.exists():
        raise ValueError(f"refusing to replace initial export root: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "hf").symlink_to(hf_reference, target_is_directory=True)
    payload = {
        "schema_version": "native_greekmmlu_exact_checkpoint_export_v1",
        "status": "completed",
        "source": {
            "source_checkpoint_root": str(source_checkpoint_root),
            "iteration": 0,
            "source_iteration": str(release.resolve()),
            "source_files": release_rows,
            "source_tree_manifest_sha256": canonical_sha256(release_rows),
        },
        "conversion": {
            "route": "receipt_bound_canonical_td_hf_reference",
            "live_roundtrip_skipped": True,
            "initialization_receipt": {
                "path": str(initialization_path),
                "sha256": sha256_file(initialization_path),
            },
            "conversion_receipt": {
                "path": str(conversion_path),
                "sha256": sha256_file(conversion_path),
            },
            "fixed_prompt_top1_logits_match": True,
            "hf_roundtrip_model_bitwise_identical": True,
        },
        "hf_export": {
            "path": str(hf_reference),
            "files": hf_rows,
            "tree_manifest_sha256": canonical_sha256(hf_rows),
            "tokenizer_json_sha256": sha256_file(reference_tokenizer),
            "frozen_tokenizer_json_sha256": sha256_file(tokenizer_json),
            "tokenizer_semantically_identical_to_frozen_overlay": True,
            "geometry": expected_geometry,
        },
        "ready_for_frozen_native_greekmmlu": True,
    }
    receipt = output_root / "checkpoint_eval_export_receipt.json"
    temporary = Path(str(receipt) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(
        args.campaign_manifest,
        args.source_checkpoint_root,
        args.output_root,
        args.tokenizer_dir,
    )
    print(json.dumps({"ok": True, "iteration": payload["source"]["iteration"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
