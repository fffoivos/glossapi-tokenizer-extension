#!/usr/bin/env python3
"""Freeze a corrected-geometry HF view of the zero-drift TD round trip."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from contract import atomic_write_json, file_binding, sha256_file


EXPECTED_FILES = {
    "config.json",
    "generation_config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
}


def canonical_tree(rows: list[dict]) -> str:
    encoded = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--roundtrip-verification", type=Path, required=True)
    parser.add_argument("--expected-verification-sha256", required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output_root.resolve()
    if output.exists() or args.receipt.exists():
        raise FileExistsError(output if output.exists() else args.receipt)
    source_files = {path.name for path in source.iterdir() if path.is_file()}
    if source_files != EXPECTED_FILES or any(path.is_symlink() for path in source.iterdir()):
        raise ValueError("canonical HF round-trip inventory drift")
    if sha256_file(args.roundtrip_verification) != args.expected_verification_sha256:
        raise ValueError("round-trip verification hash drift")
    verification = json.loads(args.roundtrip_verification.read_text(encoding="utf-8"))
    required_zero = (
        "standard_max_abs_diff", "r17_max_abs_diff", "xielu_max_abs_diff", "qk_norm_max_abs_diff",
    )
    if (
        any(float(verification.get(name, -1)) != 0.0 for name in required_zero)
        or verification.get("orig_only") != []
        or verification.get("trip_only") != []
        or verification.get("shape_mismatches") != []
        or int(verification.get("standard_changed_over_tol_count", -1)) != 0
        or int(verification.get("r17_changed_over_tol_count", -1)) != 0
        or float(verification.get("logits", {}).get("logit_max_abs_diff", -1)) != 0.0
    ):
        raise ValueError("HF round-trip is not zero drift")
    if sha256_file(source / "tokenizer.json") != args.expected_tokenizer_sha256:
        raise ValueError("HF round-trip tokenizer drift")
    original = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (
        float(original.get("rope_theta", -1)) != 12_000_000.0
        or int(original.get("max_position_embeddings", -1)) != 65_536
        or original.get("tie_word_embeddings") is not False
        or int(original.get("vocab_size", -1)) != 148_992
    ):
        raise ValueError("canonical HF source geometry drift")
    corrected = dict(original)
    corrected["rope_theta"] = 500_000.0
    corrected["max_position_embeddings"] = 4_096
    changed = {key for key in set(original) | set(corrected) if original.get(key) != corrected.get(key)}
    if changed != {"rope_theta", "max_position_embeddings"}:
        raise ValueError("corrected HF config changed fields beyond RoPE geometry")

    output.mkdir(parents=True)
    rows = []
    for name in sorted(EXPECTED_FILES - {"config.json"}):
        source_path = source / name
        destination = output / name
        os.link(source_path, destination)
        if source_path.stat().st_ino != destination.stat().st_ino:
            raise ValueError(f"HF anchor is not hard-linked to canonical source: {name}")
        rows.append({"relative_path": name, "bytes": destination.stat().st_size, "sha256": sha256_file(destination)})
    config_path = output / "config.json"
    config_path.write_text(json.dumps(corrected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows.append({"relative_path": "config.json", "bytes": config_path.stat().st_size, "sha256": sha256_file(config_path)})
    rows.sort(key=lambda row: row["relative_path"])
    payload = {
        "schema_version": "apertus_full_8b_corrected_initial_hf_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_root": str(output),
        "source_root": str(source),
        "source_roundtrip_verification": file_binding(args.roundtrip_verification),
        "source_config": file_binding(source / "config.json"),
        "corrected_config": file_binding(config_path),
        "config_changed_fields": sorted(changed),
        "geometry": {"rope_theta": 500_000.0, "max_position_embeddings": 4_096},
        "zero_tensor_and_logit_drift": True,
        "non_config_files_hardlinked_to_source": True,
        "file_count": len(rows),
        "tree_sha256": canonical_tree(rows),
        "files": rows,
    }
    atomic_write_json(args.receipt, payload)
    print(json.dumps({"ok": True, "files": len(rows), "tree_sha256": payload["tree_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
