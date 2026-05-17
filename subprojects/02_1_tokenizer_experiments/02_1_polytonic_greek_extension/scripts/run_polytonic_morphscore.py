#!/usr/bin/env python3
"""Run the C3 sweep's Greek MorphScore check on polytonic variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def patch_morphscore_encode() -> None:
    from morphscore import morphscore as ms

    def encode_text(tokenizer, text, add_special_tokens=False):
        enc = tokenizer(text, add_special_tokens=add_special_tokens)
        ids = None
        try:
            ids = enc["input_ids"]
        except (KeyError, TypeError):
            pass
        if ids is None and hasattr(enc, "input_ids"):
            ids = enc.input_ids
        if ids is None and hasattr(enc, "ids"):
            ids = enc.ids
        if ids is None and isinstance(enc, list):
            ids = enc
        if ids is None:
            raise ValueError(f"Unexpected token format: {type(enc)} - {enc}")
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return ids[0]
        return list(ids)

    ms.encode_text = encode_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizers-config", type=Path, required=True)
    parser.add_argument("--tokenizers-meta", type=Path, required=True)
    parser.add_argument("--tokeval-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.tokeval_root))
    patch_morphscore_encode()

    from morphscore.morphscore import MorphScore
    from transformers import AutoTokenizer

    paths = json.loads(args.tokenizers_config.read_text(encoding="utf-8"))
    meta = json.loads(args.tokenizers_meta.read_text(encoding="utf-8"))["tokenizers"]
    by_name = {item["name"]: item for item in meta}

    morph = MorphScore(
        language_subset=["ell_Grek"],
        by_split=False,
        freq_scale=True,
        exclude_single_tok=False,
        data_dir=str(args.tokeval_root / "morphscore_data"),
    )

    results = []
    for name in paths:
        tok = AutoTokenizer.from_pretrained(paths[name]["path"])
        metrics = morph.eval(tok)
        row = {
            "variant_id": name,
            "polytonic_added_count": by_name[name]["polytonic_added_count"],
            "final_vocab_size": by_name[name]["final_vocab_size"],
            "tokenizer_sha256": by_name[name].get("tokenizer_sha256"),
            "metrics": metrics if isinstance(metrics, dict) else dict(metrics),
        }
        results.append(row)
        ell = row["metrics"].get("ell_Grek", row["metrics"])
        print(
            f"{name:24s} recall={ell.get('morphscore_recall', 0):.4f} "
            f"precision={ell.get('morphscore_precision', 0):.4f} "
            f"tcr={ell.get('mean_token_char_ratio', 0):.4f}"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "variants": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
