"""Run MorphScore on all 15 cutoff variants for Greek.

Uses the catherinearnett/morphscore Greek dataset (21,428 UD-derived
word/morpheme pairs). For each variant tokenizer, computes:
  - morphscore_recall — fraction of gold morpheme boundaries the
    tokenizer hits
  - morphscore_precision — fraction of tokenizer boundaries that hit a
    gold morpheme boundary
  - mean_token_char_ratio — over-segmentation indicator

Output: artifacts/morphscore_greek_results.json (one record per variant)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SSP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
)
sys.path.insert(0, str(SSP / "vendor/tokenizer-intrinsic-evals"))

# Patch MorphScore's encode_text to handle HF BatchEncoding (the upstream
# helper doesn't recognise the dict-like object that newer transformers
# returns; we just unwrap input_ids).
from morphscore import morphscore as _ms


def _patched_encode_text(tokenizer, text, add_special_tokens=False):
    enc = tokenizer(text, add_special_tokens=add_special_tokens)
    # transformers BatchEncoding supports attribute + dict access
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
    # Unwrap batched [[...]] → [...]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return ids[0]
    return list(ids)


_ms.encode_text = _patched_encode_text
from morphscore.morphscore import MorphScore
from transformers import AutoTokenizer

DATA_DIR = SSP / "vendor/tokenizer-intrinsic-evals/morphscore_data"
TOKENIZERS_META = SSP / "configs/cutoff_sweep_tokenizers_meta.json"
TOKENIZERS_CFG = SSP / "configs/cutoff_sweep_tokenizers.json"
OUT_JSON = SSP / "artifacts/morphscore_greek_results.json"


def main() -> None:
    meta = json.loads(TOKENIZERS_META.read_text())["tokenizers"]
    paths = json.loads(TOKENIZERS_CFG.read_text())

    morph = MorphScore(
        language_subset=["ell_Grek"],
        by_split=False,
        freq_scale=True,
        exclude_single_tok=False,
        data_dir=str(DATA_DIR),
    )

    results = []
    for m in meta:
        name = m["name"]
        tok_path = paths[name]["path"]
        try:
            tok = AutoTokenizer.from_pretrained(tok_path)
        except Exception as e:
            print(f"  [skip {name}] tokenizer load failed: {e}")
            continue
        try:
            result = morph.eval(tok)
            row = {
                "variant_id": name,
                "added_tokens": m["added_tokens"],
                "curated": m["curated"],
                "metrics": result if isinstance(result, dict) else dict(result),
            }
            results.append(row)
            r = row["metrics"].get("ell_Grek", row["metrics"])
            print(f"  ✓ {name:30s}  "
                  f"recall={r.get('morphscore_recall', 'n/a'):.4f}  "
                  f"precision={r.get('morphscore_precision', 'n/a'):.4f}  "
                  f"tcr={r.get('mean_token_char_ratio', 'n/a'):.4f}  "
                  f"n={r.get('num_samples', 'n/a')}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            import traceback; traceback.print_exc()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_JSON}  ({len(results)} variants)")


if __name__ == "__main__":
    main()
