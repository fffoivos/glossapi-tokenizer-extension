"""Build Apertus-compatible merged tokenizer variants at a series of
added-unit cutoffs.

Given:
  - the Apertus base tokenizer (131,072 vocab, frozen)
  - a "full" continuous-BPE tokenizer trained from Apertus (e.g. C3 at
    156,672 vocab)
  - a list of cutoffs N (number of added units to keep)

For each N, produce a tokenizer of total vocab = 131,072 + N by:
  1. keeping every vocab entry with id < (131,072 + N)
  2. keeping the first (base_merges + N) merges
  3. copying tokenizer_config.json + special_tokens_map.json unchanged

The output is loadable with `transformers.AutoTokenizer.from_pretrained`
and preserves Apertus front-end behavior exactly (regex split, ByteLevel,
special tokens, first 1000 ids).

Reuses `build_continuous_cutoff` from
`tokenizer_analysis/run_wave4_fertility_eval.py` (the original C1-cutoff
builder). For arms whose output dir naming differs (e.g. C3), we rename
the dir after build.

Usage:
  python3 build_cutoff_variants.py \\
    --arm-name C3_wave2_broad_glossapi_plus_hplt_50_50 \\
    --arm-prefix c3 \\
    --base-dir /home/foivos/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_20260415 \\
    --full-dir /home/foivos/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/tokenizers/C3_wave2_broad_glossapi_plus_hplt_50_50/tokenizer \\
    --out-dir /home/foivos/runs/c3_cutoff_eval_20260511/cutoff_tokenizers \\
    --cutoffs 1024 2048 ... 25600

For C3 specifically, the standard sweep is 25 cutoffs at 1024 step from
1024 to 25600.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FERTILITY_SCRIPT = REPO_ROOT / "tokenizer_analysis" / "run_wave4_fertility_eval.py"


def load_build_continuous_cutoff(script_path: Path):
    spec = importlib.util.spec_from_file_location("fertility_helpers", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.build_continuous_cutoff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-name", required=True,
                        help="full arm name, e.g. C3_wave2_broad_glossapi_plus_hplt_50_50")
    parser.add_argument("--arm-prefix", required=True,
                        help="short prefix used in output dir names, e.g. c3")
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="Apertus base tokenizer dir (vocab=131,072)")
    parser.add_argument("--full-dir", type=Path, required=True,
                        help="full continuous-BPE tokenizer dir (vocab=156,672 for C3)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="output dir where <prefix>_added_<N>/ subdirs will land")
    parser.add_argument("--cutoffs", nargs="+", type=int, required=True,
                        help="added-unit cutoffs (must be 128-aligned to preserve Apertus divisibility)")
    parser.add_argument("--fertility-script", type=Path, default=DEFAULT_FERTILITY_SCRIPT,
                        help="source of build_continuous_cutoff (default: tokenizer_analysis/run_wave4_fertility_eval.py)")
    args = parser.parse_args()

    build_fn = load_build_continuous_cutoff(args.fertility_script)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for n in sorted(set(args.cutoffs)):
        if n % 128 != 0:
            print(f"SKIP {n} — not 128-aligned, would break Apertus front-end divisibility")
            continue
        dst = args.out_dir / f"{args.arm_prefix}_added_{n}"
        if (dst / "tokenizer.json").exists():
            print(f"skip (exists): {dst}")
            continue
        # build_continuous_cutoff hardcodes a `c1_added_<N>` dir name.
        tmp = build_fn(
            base_dir=args.base_dir,
            full_dir=args.full_dir,
            out_dir=args.out_dir,
            added_units=n,
        )
        if tmp != dst:
            shutil.move(str(tmp), str(dst))
        print(f"built: {dst}")


if __name__ == "__main__":
    main()
