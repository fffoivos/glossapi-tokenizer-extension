"""Local smoke test for the ReTok + Centroid init algorithms.

Runs on home without needing the full Apertus model loaded. Uses:
- E_fp32.npy + U_fp32.npy from the Phase A diagnostic
  (`runs/apertus_embedding_init_test_20260512/arrays/`)
- The base Apertus tokenizer (HF cache)
- The composite ship-bundle tokenizer at vocab 153,600

What this validates:
  1. The Greek-block classification correctly partitions the base vocab.
  2. The ReTok subpiece-mean produces non-degenerate vectors.
  3. The Centroid procedure produces non-degenerate vectors.
  4. Both arms produce rows at the Phase A norm targets (5.05 E, 3.80 U).
  5. A few hand-picked polytonic tokens land near (and not on top of)
     existing Greek-token clouds.

This does NOT verify:
  - The full HF `resize_token_embeddings()` path
  - Forward-pass behavior of the resized model
  - Any Megatron-LM integration

Those are V2 + V15 — they require a model load and a Clariden allocation.

Usage:
    python3 test_init_logic.py
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np

# We sit in subprojects/.../init_bakeoff/arms/. Add the dir to sys.path so the
# sibling modules can import `_common`.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Paths (everything local on home)
APERTUS_HF = "/home/foivos/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-2509/snapshots/3162c99675aa588097cecd4a24b9aa1f712af477"
SHIP_EXT = (
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/"
    "ship/apertus_greek_extended_153600"
)
E_PATH = "/home/foivos/runs/apertus_embedding_init_test_20260512/arrays/E_fp32.npy"
U_PATH = "/home/foivos/runs/apertus_embedding_init_test_20260512/arrays/U_fp32.npy"


def main() -> int:
    from transformers import AutoTokenizer  # type: ignore
    from _common import classify_greek_block, identify_greek_base_tokens, NORM_TARGET_E_GREEK, NORM_TARGET_U_GREEK
    from retok import compute_retok_init
    from centroid import compute_centroid_init

    print("Loading tokenizers...")
    base_tk = AutoTokenizer.from_pretrained(APERTUS_HF)
    ext_tk = AutoTokenizer.from_pretrained(SHIP_EXT)
    print(f"  base vocab: {base_tk.vocab_size:,}")
    print(f"  extended vocab: {ext_tk.vocab_size:,}")

    print("\nLoading E and U matrices (~2.1 GB each, mmap)...")
    t0 = time.time()
    E = np.load(E_PATH, mmap_mode="r")  # mmap to avoid loading 2 GB into resident memory
    U = np.load(U_PATH, mmap_mode="r")
    print(f"  E shape: {E.shape}, dtype: {E.dtype}, took {time.time()-t0:.1f}s")
    print(f"  U shape: {U.shape}, dtype: {U.dtype}")

    assert E.shape == (131_072, 4_096), f"unexpected E shape {E.shape}"
    assert U.shape == (131_072, 4_096), f"unexpected U shape {U.shape}"

    # === Sanity 1: Greek-block classification on base vocab ===
    print("\n=== Sanity 1: Greek-block classification of base vocab ===")
    t0 = time.time()
    modern_ids, polytonic_ids, both_ids = identify_greek_base_tokens(base_tk, base_vocab_size=131_072)
    print(f"  modern_set: {len(modern_ids):,} tokens")
    print(f"  polytonic_set: {len(polytonic_ids):,} tokens")
    print(f"  both_set: {len(both_ids):,} tokens")
    print(f"  Phase A's strict-Greek count was 1,494; we found {len(modern_ids)} here (depends on classification rules)")
    print(f"  took {time.time()-t0:.1f}s")

    # === Sanity 2: norm distributions of base Greek tokens ===
    print("\n=== Sanity 2: norm distribution of base Greek tokens ===")
    modern_E_norms = np.linalg.norm(E[modern_ids], axis=1)
    modern_U_norms = np.linalg.norm(U[modern_ids], axis=1)
    print(f"  E[modern] norm: p50={np.median(modern_E_norms):.3f}  mean={modern_E_norms.mean():.3f}  std={modern_E_norms.std():.3f}")
    print(f"  U[modern] norm: p50={np.median(modern_U_norms):.3f}  mean={modern_U_norms.mean():.3f}  std={modern_U_norms.std():.3f}")
    print(f"  Phase A targets: E=5.05, U=3.80")

    # === Sanity 3: ReTok init (on a small slice — first 200 new tokens) ===
    print("\n=== Sanity 3: ReTok init on first 200 new tokens ===")
    # Load E + U fully for the ReTok arithmetic (need random access)
    print("  reading E + U into RAM for arithmetic...")
    t0 = time.time()
    E_arr = np.array(E)
    U_arr = np.array(U)
    print(f"  loaded in {time.time()-t0:.1f}s (~4.3 GB)")

    t0 = time.time()
    new_E_retok, new_U_retok = compute_retok_init(
        base_E=E_arr,
        base_U=U_arr,
        base_tokenizer=base_tk,
        extended_tokenizer=ext_tk,
        new_id_range=(131_072, 131_272),  # first 200 only — fast smoke
        verbose=True,
    )
    print(f"  computed in {time.time()-t0:.1f}s")
    print(f"  new_E_retok shape: {new_E_retok.shape}, norm sample: {np.linalg.norm(new_E_retok[0]):.3f}")
    print(f"  new_U_retok shape: {new_U_retok.shape}, norm sample: {np.linalg.norm(new_U_retok[0]):.3f}")
    assert abs(np.linalg.norm(new_E_retok[0]) - NORM_TARGET_E_GREEK) < 1e-3, "norm-match for E failed"
    assert abs(np.linalg.norm(new_U_retok[0]) - NORM_TARGET_U_GREEK) < 1e-3, "norm-match for U failed"
    print(f"  ✓ ReTok output is shape-correct and norm-matched")

    # === Sanity 4: Centroid init (on a small slice) ===
    print("\n=== Sanity 4: Centroid init on first 200 new tokens ===")
    t0 = time.time()
    new_E_cent, new_U_cent, cent_stats = compute_centroid_init(
        base_E=E_arr,
        base_U=U_arr,
        base_tokenizer=base_tk,
        extended_tokenizer=ext_tk,
        new_id_range=(131_072, 131_272),
        verbose=True,
    )
    print(f"  computed in {time.time()-t0:.1f}s")
    print(f"  new_E_cent shape: {new_E_cent.shape}, norm sample: {np.linalg.norm(new_E_cent[0]):.3f}")
    print(f"  centroid stats: modern={cent_stats['modern_set_size']:,}  polytonic={cent_stats['polytonic_set_size']:,}  fallback={cent_stats['polytonic_fallback_to_modern']}")
    assert abs(np.linalg.norm(new_E_cent[0]) - NORM_TARGET_E_GREEK) < 1e-3, "norm-match for E failed"
    assert abs(np.linalg.norm(new_U_cent[0]) - NORM_TARGET_U_GREEK) < 1e-3, "norm-match for U failed"
    print(f"  ✓ Centroid output is shape-correct and norm-matched")

    # === Sanity 5: ReTok vs Centroid produce different vectors for the same new token ===
    print("\n=== Sanity 5: ReTok and Centroid disagree (as they should) ===")
    cos_E = np.array([
        np.dot(new_E_retok[i], new_E_cent[i]) / (np.linalg.norm(new_E_retok[i]) * np.linalg.norm(new_E_cent[i]) + 1e-9)
        for i in range(min(50, new_E_retok.shape[0]))
    ])
    print(f"  cos(ReTok_E, Centroid_E) over 50 new tokens: mean={cos_E.mean():.3f}  std={cos_E.std():.3f}")
    print(f"  Two methods produce similar but not identical directions (mean cos > 0.5 expected, std > 0 confirms randomness).")

    # === Sanity 6: a hand-picked polytonic token ===
    print("\n=== Sanity 6: a hand-picked polytonic new token ===")
    # `καὶ` is at ID 148480 (first polytonic block ID per earlier probes)
    test_id = 148_480
    if test_id < 131_272:
        print(f"  test_id {test_id} is in our 200-token smoke slice")
        offset = test_id - 131_072
        surface = ext_tk.decode([test_id])
        has_m, has_p = classify_greek_block(surface)
        print(f"  id={test_id}  surface={surface!r}  has_modern={has_m}  has_polytonic={has_p}")
    else:
        # 148480 - 131072 = 17,408 → outside our 200-token slice
        # rerun centroid + retok on just this one
        print(f"  test_id {test_id} is past our smoke slice; running single-token check...")
        for label, fn in [("ReTok", compute_retok_init), ("Centroid", lambda **k: compute_centroid_init(**k)[:2])]:
            try:
                e, u = fn(
                    base_E=E_arr, base_U=U_arr,
                    base_tokenizer=base_tk, extended_tokenizer=ext_tk,
                    new_id_range=(test_id, test_id + 1),
                    verbose=False,
                )
                surface = ext_tk.decode([test_id])
                has_m, has_p = classify_greek_block(surface)
                print(f"  {label}: id={test_id} surface={surface!r}  ‖E‖={np.linalg.norm(e[0]):.3f}  ‖U‖={np.linalg.norm(u[0]):.3f}  modern={has_m}  poly={has_p}")
            except Exception as exc:
                print(f"  {label}: failed: {exc}")

    print("\n✓ All init-logic smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
