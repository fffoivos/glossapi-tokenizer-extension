"""Production driver: build the three init-arm checkpoints on Clariden.

This script loads the Apertus-8B-2509 model, applies one or more of
the three init methods (Vanilla / ReTok / Centroid), and writes the
resulting HF-format checkpoints. The output checkpoints feed the
Megatron-LM-Swiss-AI training jobs after conversion with
`tools/checkpoint/convert.py --loader apertus_hf --saver core`.

Designed to run on a Clariden `debug` allocation (~30 min wall) since
the full Apertus model load (~16 GB) needs more RAM than home has.

Usage:
    # Build all three arms
    python3 build_init_checkpoints.py \\
        --apertus-base /iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509 \\
        --extended-tokenizer /iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480 \\
        --out-root /iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480 \\
        --vocab-size 148480 \\
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


def _assert_special_tokens_preserved(base_tk, ext_tk) -> None:
    """Fail before building if the extension shifted document-boundary IDs."""
    if dict(base_tk.special_tokens_map) != dict(ext_tk.special_tokens_map):
        raise AssertionError(
            "extended tokenizer changed the special token map: "
            f"base={base_tk.special_tokens_map!r} ext={ext_tk.special_tokens_map!r}"
        )

    token_attrs = (
        "bos_token_id",
        "eos_token_id",
        "unk_token_id",
        "pad_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    )
    shifted = {}
    for attr in token_attrs:
        base_id = getattr(base_tk, attr, None)
        ext_id = getattr(ext_tk, attr, None)
        if base_id != ext_id:
            shifted[attr] = (base_id, ext_id)
    if shifted:
        raise AssertionError(f"extended tokenizer shifted special token IDs: {shifted!r}")

    if list(base_tk.all_special_ids) != list(ext_tk.all_special_ids):
        raise AssertionError(
            "extended tokenizer changed all_special_ids order/content: "
            f"base={base_tk.all_special_ids!r} ext={ext_tk.all_special_ids!r}"
        )

    print("  special tokens: preserved (map + IDs)")


def _snapshot_apertus_extra_params(model):
    """Capture xIELU + QK-Norm tensors that must survive embedding resize."""
    import re

    extra = re.compile(r"(act_fn\.(alpha_p|alpha_n|beta|eps)|self_attn\.(q_norm|k_norm)\.weight)")
    tensors = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if extra.search(name)
    }
    if not tensors:
        raise AssertionError("found no Apertus xIELU/QK-Norm tensors to protect")
    return tensors


def _assert_apertus_extra_params_unchanged(model, before) -> None:
    after = dict(model.named_parameters())
    missing = sorted(set(before) - set(after))
    if missing:
        raise AssertionError(f"Apertus extra tensors disappeared after resize: {missing[:5]!r}")
    changed = []
    for name, old in before.items():
        new = after[name].detach().cpu()
        if old.shape != new.shape or not new.equal(old):
            changed.append(name)
    if changed:
        raise AssertionError(
            "resize_token_embeddings changed Apertus xIELU/QK-Norm tensors: "
            f"{changed[:10]!r}"
        )
    print(f"  Apertus extras: preserved across resize ({len(before)} tensors)")


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
    _assert_special_tokens_preserved(base_tk, ext_tk)

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
    resize_protected = _snapshot_apertus_extra_params(model)

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
    try:
        # We overwrite every new row immediately below; skip Transformers'
        # covariance-based initializer where the installed version supports it.
        model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(new_vocab_size)
    new_E_size, new_U_size = model.get_input_embeddings().weight.shape, model.get_output_embeddings().weight.shape
    print(f"  post-resize E: {new_E_size}  U: {new_U_size}")
    _assert_apertus_extra_params_unchanged(model, resize_protected)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"  moving model to {torch.cuda.get_device_name(0)!r} for sanity forward...")
        model.to(device)
    with torch.no_grad():
        out = model(torch.tensor([sample_ids], device=device))
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
                    help="path to the extended ship-bundle tokenizer dir")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="output root; per-arm directories created underneath")
    ap.add_argument("--arms", nargs="+", choices=["vanilla", "retok", "centroid"],
                    default=["vanilla", "retok", "centroid"])
    ap.add_argument("--vocab-size", type=int, default=148_480,
                    help=("target vocab size after extension. 148,480 = modern-only "
                          "(default; bakeoff scope). 153,600 = composite (modern + "
                          "polytonic; future production scope)."))
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
                new_vocab_size=args.vocab_size,
            )
            all_stats[arm] = stats

    summary_path = args.out_root / "init_build_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2, default=str))
    print(f"\n✓ all arms built. Summary at {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
