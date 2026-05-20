"""Production driver: build the three init-arm checkpoints on Clariden.

This script loads the Apertus-8B-2509 model, applies one or more of
the three init methods (Vanilla / ReTok / Centroid), and writes the
resulting HF-format checkpoints. The output checkpoints feed the
Megatron-LM-Swiss-AI training jobs (via `swiss-ai/hfconverter` at
staging time).

Designed to run on a Clariden `debug` allocation (~30 min wall) since
the full Apertus model load (~16 GB) needs more RAM than home has.

Usage:
    # Build all three arms
    python3 build_init_checkpoints.py \\
        --apertus-base /iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509 \\
        --extended-tokenizer /iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_extended_153600 \\
        --out-root /iopsstor/scratch/cscs/fffoivos/init_checkpoints \\
        --arms vanilla retok centroid

    # Just ReTok (e.g., after a config tweak)
    python3 build_init_checkpoints.py ... --arms retok

Each arm writes to `<out-root>/<arm>/` and the result is a standard HF
model directory (config.json, model-*.safetensors, tokenizer files
mirrored from the extended bundle).

Hardware:
    - ~32 GB RAM during peak (full model in fp32 momentarily; bf16
      load if available is preferred)
    - ~32 GB disk per arm checkpoint
    - GPU not strictly needed (init is CPU work), but useful if we
      want to run a forward-pass sanity check after init.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from retok import compute_retok_init  # noqa: E402
from centroid import compute_centroid_init  # noqa: E402


def build_vanilla(*, apertus_base: Path, out_dir: Path) -> None:
    """Vanilla arm: copy the base Apertus checkpoint with no modification."""
    print(f"\n=== Vanilla arm ===")
    print(f"  Vanilla = unmodified Apertus-8B-2509 + original tokenizer (vocab 131,072).")
    print(f"  Strategy: symlink the base, don't duplicate 16 GB of safetensors.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in apertus_base.iterdir():
        dst = out_dir / src.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            # Filesystem doesn't support symlinks: hardlink, then fallback to copy
            try:
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
    print(f"  ✓ wrote {out_dir}")


def build_extension_arm(
    *,
    arm: str,
    apertus_base: Path,
    extended_tokenizer_dir: Path,
    out_dir: Path,
    new_vocab_size: int = 153_600,
) -> dict:
    """Build a ReTok or Centroid checkpoint. Returns a stats dict.

    Steps:
    1. Load tokenizers (base + extended).
    2. Load Apertus model.
    3. Extract base E and U as numpy.
    4. Apply the chosen init function → new E + U row sets.
    5. resize_token_embeddings(new_vocab_size).
    6. Write the new rows into model.get_input_embeddings() and
       model.get_output_embeddings().
    7. save_pretrained(out_dir).
    8. Copy extended-tokenizer files into out_dir so the saved model
       directory is a stand-alone HF-loadable bundle.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore

    print(f"\n=== {arm.title()} arm ===")
    t0 = time.time()

    base_tk = AutoTokenizer.from_pretrained(str(apertus_base))
    ext_tk = AutoTokenizer.from_pretrained(str(extended_tokenizer_dir))
    print(f"  base vocab: {base_tk.vocab_size:,}")
    print(f"  extended vocab: {ext_tk.vocab_size:,}  (target: {new_vocab_size:,})")
    assert ext_tk.vocab_size == new_vocab_size, "extended tokenizer must match new_vocab_size"

    print(f"  loading Apertus base from {apertus_base} (bf16 if available)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(apertus_base),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    D = model.get_input_embeddings().weight.shape[1]
    print(f"  model loaded: vocab={base_vocab_size:,}, hidden={D}, took {time.time()-t0:.1f}s")
    assert base_vocab_size == 131_072, f"base vocab {base_vocab_size} != 131,072"

    # Extract base E and U into numpy for the init algorithms
    print(f"  extracting E and U to numpy...")
    base_E = model.get_input_embeddings().weight.detach().cpu().to(torch.float32).numpy()
    base_U = model.get_output_embeddings().weight.detach().cpu().to(torch.float32).numpy()
    print(f"  base_E: {base_E.shape}, dtype: {base_E.dtype}")
    print(f"  base_U: {base_U.shape}, dtype: {base_U.dtype}")

    # Apply init algorithm
    stats = {"arm": arm}
    if arm == "retok":
        new_E, new_U = compute_retok_init(
            base_E=base_E, base_U=base_U,
            base_tokenizer=base_tk, extended_tokenizer=ext_tk,
            new_id_range=(base_vocab_size, new_vocab_size),
            verbose=True,
        )
    elif arm == "centroid":
        new_E, new_U, cent_stats = compute_centroid_init(
            base_E=base_E, base_U=base_U,
            base_tokenizer=base_tk, extended_tokenizer=ext_tk,
            new_id_range=(base_vocab_size, new_vocab_size),
            base_vocab_size=base_vocab_size,
            verbose=True,
        )
        stats.update(cent_stats)
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    # Free base_E / base_U numpy memory before doing the resize (avoid 8 GB transient)
    del base_E, base_U

    print(f"  resizing model: {base_vocab_size:,} → {new_vocab_size:,}")
    model.resize_token_embeddings(new_vocab_size)
    new_E_size, new_U_size = model.get_input_embeddings().weight.shape, model.get_output_embeddings().weight.shape
    print(f"  post-resize E: {new_E_size}  U: {new_U_size}")

    # Write the new rows
    print(f"  writing new rows {base_vocab_size}..{new_vocab_size}")
    new_E_tensor = torch.from_numpy(new_E).to(model.get_input_embeddings().weight.dtype)
    new_U_tensor = torch.from_numpy(new_U).to(model.get_output_embeddings().weight.dtype)
    with torch.no_grad():
        model.get_input_embeddings().weight[base_vocab_size:new_vocab_size] = new_E_tensor
        model.get_output_embeddings().weight[base_vocab_size:new_vocab_size] = new_U_tensor

    # Sanity: forward pass on a polytonic sample
    print(f"  sanity: forward pass on a Greek sample...")
    sample_ids = ext_tk.encode("Η ελληνική γλώσσα Ἐν ἀρχῇ ἦν ὁ Λόγος", add_special_tokens=False)
    with torch.no_grad():
        out = model(torch.tensor([sample_ids]))
    logits = out.logits  # [1, T, V]
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    print(f"  logits shape: {tuple(logits.shape)}  has_nan={has_nan}  has_inf={has_inf}")
    stats["sanity_logits_shape"] = list(logits.shape)
    stats["sanity_has_nan"] = bool(has_nan)
    stats["sanity_has_inf"] = bool(has_inf)
    assert not has_nan and not has_inf, "forward pass produced nan or inf — init or resize is broken"
    assert logits.shape[-1] == new_vocab_size, f"logit vocab {logits.shape[-1]} != {new_vocab_size}"

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  saving to {out_dir}")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    # Save tokenizer files from the extended bundle alongside
    ext_tk.save_pretrained(str(out_dir))

    elapsed = time.time() - t0
    print(f"  ✓ {arm} arm built in {elapsed:.1f}s")
    stats["wall_seconds"] = elapsed
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apertus-base", type=Path, required=True,
                    help="path to local Apertus-8B-2509 HF checkpoint dir")
    ap.add_argument("--extended-tokenizer", type=Path, required=True,
                    help="path to the composite 153,600 ship-bundle tokenizer dir")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="output root; per-arm directories created underneath")
    ap.add_argument("--arms", nargs="+", choices=["vanilla", "retok", "centroid"],
                    default=["vanilla", "retok", "centroid"])
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for arm in args.arms:
        out_dir = args.out_root / arm
        if arm == "vanilla":
            build_vanilla(apertus_base=args.apertus_base, out_dir=out_dir)
            all_stats[arm] = {"arm": "vanilla", "strategy": "symlink to base"}
        else:
            stats = build_extension_arm(
                arm=arm,
                apertus_base=args.apertus_base,
                extended_tokenizer_dir=args.extended_tokenizer,
                out_dir=out_dir,
            )
            all_stats[arm] = stats

    summary_path = args.out_root / "init_build_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2, default=str))
    print(f"\n✓ all arms built. Summary at {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
