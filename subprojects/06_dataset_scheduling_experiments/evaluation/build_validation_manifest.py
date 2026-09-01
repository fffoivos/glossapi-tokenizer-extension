#!/usr/bin/env python3
"""Freeze the 12 prepared heldouts plus one neutral external Greek panel."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


EXPECTED_PREPARED = {
    "code",
    "de",
    "english",
    "math",
    "ru",
    "zh",
    "greek_phd",
    "historical_polytonic",
    "hplt",
    "non_hplt",
    "openarchives",
    "old_greek",
}
NEUTRAL_NAME = "neutral_external_modern_greek"
FAST_TOKEN_TARGET = 2_097_152
MIN_NEUTRAL_TOKENS = 10_000_000
MAX_NEUTRAL_TOKENS = 20_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def panel_from_manifest(path: Path, expected_name: str | None = None) -> dict:
    value = read_json(path)
    if value.get("schema_version") != "full_cpt_megatron_shard_v1" or value.get("status") != "completed" or value.get("kind") != "heldout":
        raise ValueError(f"invalid heldout binary manifest: {path}")
    name = str(value["heldout_name"])
    if expected_name is not None and name != expected_name:
        raise ValueError(f"neutral panel name drift: {name}")
    prefix = Path(value["output_prefix"])
    outputs = value["outputs"]
    for suffix, key in ((".bin", "bin"), (".idx", "idx")):
        payload = Path(str(prefix) + suffix)
        receipt = outputs[key]
        if not payload.is_file() or payload.stat().st_size != int(receipt["bytes"]) or sha256(payload) != receipt["sha256"]:
            raise ValueError(f"heldout payload drift: {payload}")
    tokens = int(value["counts"]["tokens"])
    return {
        "name": name,
        "megatron_prefix": str(prefix.resolve()),
        "documents": int(value["counts"]["documents"]),
        "tokens": tokens,
        "fast_loss_active_tokens": min(tokens, FAST_TOKEN_TARGET),
        "manifest": str(path.resolve()),
        "manifest_sha256": sha256(path),
        "bin_sha256": outputs["bin"]["sha256"],
        "idx_sha256": outputs["idx"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-corpus-receipt", type=Path, required=True)
    parser.add_argument("--neutral-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pool = read_json(args.pool_corpus_receipt)
    if pool.get("schema_version") != "apertus_mini_schedule_pool_corpus_v1" or pool.get("status") != "completed":
        raise ValueError("pool corpus receipt is incomplete")
    prepared = [panel_from_manifest(Path(row["manifest_path"])) for row in pool["heldouts"]]
    if {row["name"] for row in prepared} != EXPECTED_PREPARED:
        raise ValueError("prepared heldout inventory drift")
    neutral = panel_from_manifest(args.neutral_manifest, NEUTRAL_NAME)
    if not MIN_NEUTRAL_TOKENS <= neutral["tokens"] <= MAX_NEUTRAL_TOKENS:
        raise ValueError("neutral external Greek panel must contain 10M-20M tokens")
    neutral_manifest = read_json(args.neutral_manifest)
    external = neutral_manifest.get("external_validation", {})
    source_separated = (
        external.get("publishers_or_domains_absent_from_training") is True
        or external.get("source_time_window_absent_from_training") is True
    )
    if (
        external.get("document_cluster_split") is not True
        or external.get("global_exact_dedup_against_training") is not True
        or external.get("global_minhash_dedup_against_training") is not True
        or float(external.get("minhash_threshold", 0.0)) != 0.85
        or not source_separated
        or external.get("candidate_documents_never_used_for_training") is not True
        or not external.get("source_snapshot_receipts")
        or not external.get("dedup_receipt")
    ):
        raise ValueError("neutral external Greek provenance/dedup gates failed")
    for receipt in [*external["source_snapshot_receipts"], external["dedup_receipt"]]:
        payload = Path(receipt["path"])
        if (
            not payload.is_file()
            or payload.is_symlink()
            or payload.stat().st_size != int(receipt["bytes"])
            or sha256(payload) != receipt["sha256"]
        ):
            raise ValueError(f"neutral external evidence drift: {payload}")
    panels = sorted([*prepared, neutral], key=lambda row: row["name"])
    payload = {
        "schema_version": "apertus_mini_validation_manifest_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pool_corpus_receipt": {
            "path": str(args.pool_corpus_receipt.resolve()),
            "sha256": sha256(args.pool_corpus_receipt),
        },
        "fast_panel_policy": {
            "same_examples_every_checkpoint": True,
            "loss_active_token_target_per_panel": FAST_TOKEN_TARGET,
            "goldfish_masking": False,
        },
        "panels": panels,
        "panel_count": len(panels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "panels": len(panels), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
