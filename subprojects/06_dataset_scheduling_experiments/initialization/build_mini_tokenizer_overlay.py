#!/usr/bin/env python3
"""Overlay the production Greek merge chain onto the pinned Mini tokenizer.

The Apertus-v1.1-0.5B tokenizer and the Apertus-8B tokenizer used to build the
Greek extension have the same BPE merge prefix but differ in a small set of
reserved/special-token IDs.  This builder preserves every Mini base ID and all
Mini front-end metadata, then appends the exact production token/merge chain at
IDs 131072..148991.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_VOCAB_SIZE = 131_072
TARGET_VOCAB_SIZE = 148_992
ALIGNMENT = 256
EXPECTED_MINI_TOKENIZER_SHA256 = (
    "be12f4375d655cc740864e3a9041bcddd8477942f209d9e7f27f6c8767162638"
)
EXPECTED_PRODUCTION_TOKENIZER_SHA256 = (
    "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b"
)
SIDECARS = (
    "tokenizer_config.json",
    "special_tokens_map.json",
)
PAD_TOKEN = "<pad>"
PAD_TOKEN_ID = 10


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_merge(value: Any) -> tuple[str, str]:
    if isinstance(value, list) and len(value) == 2:
        return str(value[0]), str(value[1])
    if isinstance(value, str):
        parts = value.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    raise ValueError(f"unsupported merge representation: {value!r}")


def by_id(vocab: dict[str, int]) -> dict[int, str]:
    result = {int(token_id): str(token) for token, token_id in vocab.items()}
    if len(result) != len(vocab):
        raise ValueError("vocabulary contains duplicate IDs")
    return result


def build_overlay(
    mini: dict[str, Any],
    production: dict[str, Any],
    *,
    base_vocab_size: int = BASE_VOCAB_SIZE,
    target_vocab_size: int = TARGET_VOCAB_SIZE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mini_model = mini.get("model", {})
    production_model = production.get("model", {})
    mini_vocab = mini_model.get("vocab")
    production_vocab = production_model.get("vocab")
    mini_merges = mini_model.get("merges")
    production_merges = production_model.get("merges")
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (mini_vocab, dict),
            (production_vocab, dict),
            (mini_merges, list),
            (production_merges, list),
        )
    ):
        raise ValueError("both tokenizers must contain BPE vocabularies and merges")

    mini_ids = by_id(mini_vocab)
    production_ids = by_id(production_vocab)
    if sorted(mini_ids) != list(range(base_vocab_size)):
        raise ValueError("Mini vocabulary IDs are not the expected contiguous base range")
    if sorted(production_ids) != list(range(target_vocab_size)):
        raise ValueError("production vocabulary IDs are not the expected contiguous range")
    if production_merges[: len(mini_merges)] != mini_merges:
        raise ValueError("Mini BPE merges are not an exact prefix of production merges")

    appended_merges = production_merges[len(mini_merges) :]
    expected_added = target_vocab_size - base_vocab_size
    if len(appended_merges) != expected_added:
        raise ValueError(
            f"expected {expected_added} appended merges, got {len(appended_merges)}"
        )

    overlay_vocab: dict[str, int] = {
        str(token): int(token_id) for token, token_id in mini_vocab.items()
    }
    live = set(overlay_vocab)
    merge_errors: list[str] = []
    for offset, merge in enumerate(appended_merges):
        left, right = split_merge(merge)
        token = left + right
        token_id = base_vocab_size + offset
        if left not in live:
            merge_errors.append(f"{token_id}: left operand is not live")
        if right not in live:
            merge_errors.append(f"{token_id}: right operand is not live")
        if token in live:
            merge_errors.append(f"{token_id}: merge result already exists")
        if production_ids.get(token_id) != token:
            merge_errors.append(f"{token_id}: production token/merge result mismatch")
        overlay_vocab[token] = token_id
        live.add(token)
    if merge_errors:
        raise ValueError("invalid appended merge chain: " + "; ".join(merge_errors[:20]))

    overlay = copy.deepcopy(mini)
    overlay["model"]["vocab"] = overlay_vocab
    overlay["model"]["merges"] = list(mini_merges) + list(appended_merges)

    overlay_ids = by_id(overlay_vocab)
    if sorted(overlay_ids) != list(range(target_vocab_size)):
        raise AssertionError("overlay vocabulary is not contiguous")
    if any(overlay_ids[token_id] != mini_ids[token_id] for token_id in mini_ids):
        raise AssertionError("overlay changed a Mini base token ID")
    if any(
        overlay_ids[token_id] != production_ids[token_id]
        for token_id in range(base_vocab_size, target_vocab_size)
    ):
        raise AssertionError("overlay changed an appended production token ID")

    mismatches = [
        {
            "token_id": token_id,
            "mini_token": mini_ids[token_id],
            "production_base_token": production_ids[token_id],
        }
        for token_id in range(base_vocab_size)
        if mini_ids[token_id] != production_ids[token_id]
    ]
    summary = {
        "base_vocab_size": base_vocab_size,
        "target_vocab_size": target_vocab_size,
        "appended_token_count": expected_added,
        "appended_merge_count": len(appended_merges),
        "base_merge_count": len(mini_merges),
        "base_id_mismatch_count_vs_production_tokenizer": len(mismatches),
        "base_id_mismatches_vs_production_tokenizer": mismatches,
        "mini_base_ids_preserved": True,
        "production_appended_ids_preserved": True,
        "mini_tokenizer_json_front_end_preserved": all(
            overlay.get(key) == mini.get(key)
            for key in (
                "version",
                "truncation",
                "padding",
                "added_tokens",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
            )
        ),
    }
    return overlay, summary


def write_json(path: Path, value: dict[str, Any], *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def reconcile_pad_metadata(
    tokenizer_config: dict[str, Any], special_tokens_map: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Declare Mini's existing ID-10 `<pad>` token as the actual pad token."""

    tokenizer_config = copy.deepcopy(tokenizer_config)
    special_tokens_map = copy.deepcopy(special_tokens_map)
    tokenizer_config["pad_token"] = PAD_TOKEN
    special_tokens_map["pad_token"] = {
        "content": PAD_TOKEN,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    }
    return tokenizer_config, special_tokens_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--production-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-mini-sha256", default=EXPECTED_MINI_TOKENIZER_SHA256
    )
    parser.add_argument(
        "--expected-production-sha256",
        default=EXPECTED_PRODUCTION_TOKENIZER_SHA256,
    )
    args = parser.parse_args()

    mini_path = args.mini_tokenizer_dir / "tokenizer.json"
    production_path = args.production_tokenizer_dir / "tokenizer.json"
    mini_sha = sha256_file(mini_path)
    production_sha = sha256_file(production_path)
    if mini_sha != args.expected_mini_sha256:
        raise SystemExit(f"Mini tokenizer hash drift: {mini_sha}")
    if production_sha != args.expected_production_sha256:
        raise SystemExit(f"production tokenizer hash drift: {production_sha}")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")

    overlay, summary = build_overlay(read_json(mini_path), read_json(production_path))
    args.output_dir.mkdir(parents=True)
    output_tokenizer = args.output_dir / "tokenizer.json"
    write_json(output_tokenizer, overlay, compact=True)
    source_sidecar_hashes: dict[str, str] = {}
    for name in SIDECARS:
        source = args.mini_tokenizer_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        source_sidecar_hashes[name] = sha256_file(source)
    tokenizer_config, special_tokens_map = reconcile_pad_metadata(
        read_json(args.mini_tokenizer_dir / "tokenizer_config.json"),
        read_json(args.mini_tokenizer_dir / "special_tokens_map.json"),
    )
    write_json(args.output_dir / "tokenizer_config.json", tokenizer_config)
    write_json(args.output_dir / "special_tokens_map.json", special_tokens_map)
    output_sidecar_hashes = {
        name: sha256_file(args.output_dir / name) for name in SIDECARS
    }

    manifest = {
        "schema_version": "apertus_mini_greek_tokenizer_overlay_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "mini_tokenizer": {
            "path": str(args.mini_tokenizer_dir.resolve()),
            "tokenizer_json_sha256": mini_sha,
        },
        "production_extension_tokenizer": {
            "path": str(args.production_tokenizer_dir.resolve()),
            "tokenizer_json_sha256": production_sha,
        },
        "output": {
            "path": str(args.output_dir.resolve()),
            "tokenizer_json_sha256": sha256_file(output_tokenizer),
            "source_sidecar_sha256": source_sidecar_hashes,
            "output_sidecar_sha256": output_sidecar_hashes,
        },
        "pad_metadata_reconciliation": {
            "source_model_config_pad_token_id": 3,
            "source_token_at_id_3": "[INST]",
            "source_tokenizer_declared_pad_token": None,
            "existing_pad_token": PAD_TOKEN,
            "existing_pad_token_id": PAD_TOKEN_ID,
            "output_declared_pad_token": PAD_TOKEN,
            "output_pad_token_id": PAD_TOKEN_ID,
            "changes_token_ids_or_bpe_merges": False,
        },
        "alignment": {
            "divisor": ALIGNMENT,
            "quotient": TARGET_VOCAB_SIZE // ALIGNMENT,
            "remainder": TARGET_VOCAB_SIZE % ALIGNMENT,
            "padding_tokens": 0,
        },
        **summary,
    }
    write_json(args.output_dir / "overlay_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
