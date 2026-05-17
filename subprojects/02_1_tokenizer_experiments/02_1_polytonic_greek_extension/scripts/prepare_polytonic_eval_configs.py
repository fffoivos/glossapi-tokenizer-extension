#!/usr/bin/env python3
"""Prepare TokEval-style configs for the polytonic cutoff variants."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--c3-sweep-dir",
        type=Path,
        default=Path(
            "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
            "02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.variants_manifest.read_text(encoding="utf-8"))

    tokenizers: dict[str, dict[str, str]] = {}
    meta = []
    for item in manifest["variants"]:
        name = item["variant_id"]
        tokenizers[name] = {"class": "hf", "path": item["variant_dir"]}
        meta.append(
            {
                "name": name,
                "polytonic_added_count": int(item["polytonic_added_count"]),
                "added_tokens": int(item["polytonic_added_count"]),
                "final_vocab_size": int(item["final_vocab_size"]),
                "tokenizer_sha256": item.get("tokenizer_sha256"),
                "base_tokenizer_sha256": item.get("base_tokenizer_sha256"),
                "curated": False,
            }
        )

    write_json(args.output_dir / "polytonic_tokenizers.json", tokenizers)
    write_json(args.output_dir / "polytonic_tokenizers_meta.json", {"tokenizers": meta})

    copied = {}
    for name in ("apertus55_lang_config.json", "greek_only_lang_config.json"):
        src = args.c3_sweep_dir / "configs" / name
        dst = args.output_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing C3 sweep config: {src}")
        shutil.copy2(src, dst)
        copied[name] = str(dst)

    write_json(
        args.output_dir / "run_config.json",
        {
            "variants_manifest": str(args.variants_manifest),
            "c3_sweep_dir": str(args.c3_sweep_dir),
            "tokenizer_count": len(tokenizers),
            "tokenizers_config": str(args.output_dir / "polytonic_tokenizers.json"),
            "tokenizers_meta": str(args.output_dir / "polytonic_tokenizers_meta.json"),
            "copied_language_configs": copied,
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "tokenizers": len(tokenizers)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
