#!/usr/bin/env python3
"""Create a hard-linked HF view with only the approved CPT geometry changed."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    require,
    require_receipt,
    require_relative_inventory,
    sha256_file,
    write_json_atomic,
)

TARGET = {
    "rope_theta": 500_000.0,
    "max_position_embeddings": 4_096,
    # The matched study runs both scales with the historical 8B command-line
    # geometry: --use-rope-scaling --rope-scaling-factor 8.  The 1.5B parent
    # ships with {"rope_type": "default"}; leaving that value in the HF view
    # would make update-zero evaluation use a different rotary function from
    # every trained checkpoint even though the tensors are identical.
    "rope_scaling": {
        "type": "llama3",
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "rope_type": "llama3",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--expected-vocab-size", type=int, required=True)
    parser.add_argument("--expected-hidden-size", type=int, required=True)
    parser.add_argument("--expected-hidden-layers", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_root.resolve()
    target = args.output_root.resolve()
    require(source.is_dir(), f"source HF root missing: {source}")
    require(
        not target.exists() and not args.output_receipt.exists(),
        "immutable geometry output exists",
    )
    authority = require_receipt(
        args.source_authority,
        schemas={
            "apertus_pinned_hf_model_materialization_v1",
            "apertus_1p5b_td_initialization_verification_v2",
        },
    )
    if authority["schema_version"] == "apertus_pinned_hf_model_materialization_v1":
        authority_root = Path(str(authority.get("output_root", ""))).resolve()
        authority_rows = authority.get("files")
    else:
        authority_root = Path(str(authority.get("td_model_root", ""))).resolve()
        authority_rows = authority.get("td_model_files")
    require(authority_root == source, "geometry source authority root drift")
    source_inventory = require_relative_inventory(root=source, rows=authority_rows)
    source_hashes = {row["path"]: row["sha256"] for row in source_inventory}
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    expected = {
        "vocab_size": args.expected_vocab_size,
        "hidden_size": args.expected_hidden_size,
        "num_hidden_layers": args.expected_hidden_layers,
        "tie_word_embeddings": False,
    }
    require(
        {key: source_config.get(key) for key in expected} == expected,
        "source HF invariant drift",
    )
    require((source / "tokenizer.json").is_file(), "source tokenizer missing")
    corrected = dict(source_config)
    corrected.update(TARGET)
    changed_keys = sorted(
        key for key in TARGET if source_config.get(key) != TARGET[key]
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        target.mkdir()
        created = True
        hardlinks = []
        for source_path in sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.name != "config.json"
        ):
            relative = source_path.relative_to(source)
            target_path = target / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_path, target_path)
            source_stat, target_stat = source_path.stat(), target_path.stat()
            require(
                (source_stat.st_dev, source_stat.st_ino)
                == (target_stat.st_dev, target_stat.st_ino),
                f"hardlink failed: {relative}",
            )
            hardlinks.append(
                {
                    "path": str(relative),
                    "bytes": source_stat.st_size,
                    "sha256": source_hashes[str(relative)],
                    "device": source_stat.st_dev,
                    "inode": source_stat.st_ino,
                }
            )
        write_json_atomic(target / "config.json", corrected)
    except BaseException:
        if created:
            shutil.rmtree(target, ignore_errors=True)
        raise
    require(
        json.loads((target / "config.json").read_text(encoding="utf-8")) == corrected,
        "corrected config drift",
    )
    output_files = sorted(
        hardlinks
        + [
            {
                "path": "config.json",
                "bytes": (target / "config.json").stat().st_size,
                "sha256": sha256_file(target / "config.json"),
            }
        ],
        key=lambda row: row["path"],
    )
    receipt = {
        "schema_version": "apertus_training_geometry_hf_view_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_root": str(source),
        "source_authority": file_binding(args.source_authority),
        "output_root": str(target),
        "source_geometry": {key: source_config.get(key) for key in TARGET},
        "training_geometry": TARGET,
        "changed_config_keys": changed_keys,
        "all_non_config_files_hardlinked": True,
        "hardlinked_file_count": len(hardlinks),
        "hardlinked_files": hardlinks,
        "output_files": output_files,
        "source_config": file_binding(source / "config.json"),
        "output_config": file_binding(target / "config.json"),
        "tokenizer": file_binding(target / "tokenizer.json"),
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output_receipt, receipt)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
