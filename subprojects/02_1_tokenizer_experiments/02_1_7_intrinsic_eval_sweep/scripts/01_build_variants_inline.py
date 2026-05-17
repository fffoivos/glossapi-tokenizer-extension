"""Inline cutoff variant builder.

Replacement for `02_1_2_cutoff_variant_builder/scripts/build_cutoff_variants.py`,
which depends on a `build_continuous_cutoff` helper from a deleted script
(`tokenizer_analysis/run_wave4_fertility_eval.py`). The truncation logic is
small enough to reimplement directly:

  Given Apertus base (131,072 vocab) and a "full" continuous-BPE
  tokenizer (Apertus + 25,600 added), build a tokenizer at vocab =
  131,072 + N by:
    1. keeping every vocab entry with id < 131,072 + N
    2. keeping the first (n_base_merges + N) merges
    3. preserving normalizer / pre_tokenizer / post_processor / model
       config from the full tokenizer

The output tokenizer.json is loadable with
`transformers.AutoTokenizer.from_pretrained`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

SSP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
)
BASE_DIR = SSP / "tokenizers_local/apertus_base"
FULL_DIR = SSP / "tokenizers_local/c3_full"
OUT_DIR = SSP / "variants"

BASE_VOCAB = 131_072
CUTOFFS = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192,
           9216, 10240, 11264, 12288,
           # Extended sweep to full C3 vocab
           13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480,
           21504, 22528, 23552, 24576, 25600]


def build_cutoff(full_tok: dict, base_n_merges: int, cutoff: int,
                 out_dir: Path, src_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_vocab_size = BASE_VOCAB + cutoff
    # Truncate vocab
    new_vocab = {tok: tid for tok, tid in full_tok["model"]["vocab"].items()
                 if tid < target_vocab_size}
    assert len(new_vocab) == target_vocab_size, \
        f"cutoff {cutoff}: got {len(new_vocab)}, want {target_vocab_size}"
    full_tok["model"]["vocab"] = new_vocab
    # Truncate merges
    full_tok["model"]["merges"] = full_tok["model"]["merges"][:base_n_merges + cutoff]
    # Write
    (out_dir / "tokenizer.json").write_text(
        json.dumps(full_tok, ensure_ascii=False)
    )
    # Copy non-tokenizer.json siblings from the FULL dir (config + special tokens)
    for sib in src_dir.iterdir():
        if sib.name == "tokenizer.json":
            continue
        if sib.is_file():
            shutil.copy2(sib, out_dir / sib.name)
    print(f"  ✓ {out_dir.name}  (vocab {len(new_vocab):,}, "
          f"merges {len(full_tok['model']['merges']):,})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read once
    base_tok = json.loads((BASE_DIR / "tokenizer.json").read_text())
    base_n_merges = len(base_tok["model"]["merges"])
    full_tok_text = (FULL_DIR / "tokenizer.json").read_text()

    print(f"Apertus base merges: {base_n_merges}")
    print(f"C3 full merges     : {len(json.loads(full_tok_text)['model']['merges'])}")
    print(f"C3 full vocab      : {len(json.loads(full_tok_text)['model']['vocab'])}")

    # Apertus base as variant #0
    base_out = OUT_DIR / "apertus_base"
    if not base_out.exists():
        shutil.copytree(BASE_DIR, base_out)
        print(f"  ✓ apertus_base (vocab {len(base_tok['model']['vocab']):,})")

    # Each cutoff — load fresh full tokenizer so the truncation isn't cumulative
    for cutoff in CUTOFFS:
        out = OUT_DIR / f"c3_added_{cutoff}"
        if (out / "tokenizer.json").exists():
            print(f"  [cached] c3_added_{cutoff}")
            continue
        full_tok = json.loads(full_tok_text)
        build_cutoff(full_tok, base_n_merges, cutoff, out, FULL_DIR)


if __name__ == "__main__":
    main()
