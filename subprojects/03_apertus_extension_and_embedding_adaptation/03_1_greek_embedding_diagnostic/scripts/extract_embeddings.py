"""Download Apertus-8B-2509 and save E (input) + U (output) embedding rows.

Output:
  arrays/E_fp32.npy       (131072, 4096) float32
  arrays/U_fp32.npy       (131072, 4096) float32
  arrays/E_norms.npy      (131072,) float32
  arrays/U_norms.npy      (131072,) float32
  arrays/extract_meta.json

Why fp32: per the test plan §2.8 pitfall #7, bf16's 7-bit mantissa
makes low-magnitude singular values unreliable; SVD/cov/PCA must run
in fp32.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("/home/foivos/runs/apertus_embedding_init_test_20260512/arrays")
MODEL_ID = "swiss-ai/Apertus-8B-2509"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Pre-download with explicit retry-friendly snapshot_download so the
    # subsequent .from_pretrained() is cache-only.
    print(f"[download] {MODEL_ID} via snapshot_download", flush=True)
    from huggingface_hub import snapshot_download
    snap = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                        "tokenizer*", "special_tokens_map.json"],
        max_workers=4,
    )
    print(f"[download] cache at {snap} ({time.time()-t0:.1f}s)", flush=True)

    print(f"[load] {MODEL_ID} (CPU, bf16)", flush=True)
    from transformers import AutoModelForCausalLM, AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    print(f"[load] vocab_size={cfg.vocab_size}, hidden_size={cfg.hidden_size}, "
          f"tie_word_embeddings={cfg.tie_word_embeddings}", flush=True)

    # Load just the model with bf16 weights; we'll extract E + U and discard.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    E = model.get_input_embeddings().weight.detach()
    U = model.get_output_embeddings().weight.detach()
    print(f"[shape] E={tuple(E.shape)} dtype={E.dtype}", flush=True)
    print(f"[shape] U={tuple(U.shape)} dtype={U.dtype}", flush=True)

    # Cast to fp32 and save as numpy.
    E_fp32 = E.to(torch.float32).numpy()
    U_fp32 = U.to(torch.float32).numpy()

    np.save(OUT_DIR / "E_fp32.npy", E_fp32)
    np.save(OUT_DIR / "U_fp32.npy", U_fp32)
    print(f"[save] E_fp32.npy {E_fp32.nbytes/1e9:.2f} GB", flush=True)
    print(f"[save] U_fp32.npy {U_fp32.nbytes/1e9:.2f} GB", flush=True)

    # Per-row L2 norms — used everywhere downstream.
    E_norms = np.linalg.norm(E_fp32, axis=1)
    U_norms = np.linalg.norm(U_fp32, axis=1)
    np.save(OUT_DIR / "E_norms.npy", E_norms)
    np.save(OUT_DIR / "U_norms.npy", U_norms)
    print(f"[norms] E median={np.median(E_norms):.4f}, mean={E_norms.mean():.4f}", flush=True)
    print(f"[norms] U median={np.median(U_norms):.4f}, mean={U_norms.mean():.4f}", flush=True)

    meta = {
        "model_id": MODEL_ID,
        "vocab_size": int(cfg.vocab_size),
        "hidden_size": int(cfg.hidden_size),
        "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        "E_shape": list(E_fp32.shape),
        "U_shape": list(U_fp32.shape),
        "wall_seconds": int(time.time() - t0),
    }
    (OUT_DIR / "extract_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {time.time()-t0:.1f}s total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
