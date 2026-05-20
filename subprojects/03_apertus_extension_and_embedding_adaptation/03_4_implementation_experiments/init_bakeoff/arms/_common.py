"""Shared utilities for the three init arms.

All three arms (Vanilla / ReTok / Centroid) need:
- a way to classify any token surface form by Greek Unicode block
- a way to norm-match a vector to Phase A targets (5.05 for E, 3.80 for U)
- consistent type/shape conventions

These functions are pure NumPy — they don't depend on the Apertus model
being loaded. That lets `test_init_logic.py` exercise them against the
2.1 GB E/U matrices on home without an HF model load.
"""
from __future__ import annotations
from typing import Iterable

import numpy as np

# Phase A norm targets (from `old_experiments_plan.md` § 8.2 + the diagnostic at
# `runs/apertus_greek_diagnostic_20260511_v2/`). Greek-content rows want p50 5.05
# in E and 3.80 in U. These match English-baseline within 1 %.
NORM_TARGET_E_GREEK = 5.05
NORM_TARGET_U_GREEK = 3.80

# Unicode blocks that flag a surface form as Greek.
GREEK_MODERN_START, GREEK_MODERN_END = 0x0370, 0x03FF       # Modern Greek + Coptic
GREEK_POLYTONIC_START, GREEK_POLYTONIC_END = 0x1F00, 0x1FFF  # Greek Extended (polytonic)

# Combining diacritic block — present means the surface form might be NFD-decomposed.
COMBINING_START, COMBINING_END = 0x0300, 0x036F

# Centroid-reliability threshold per v0.7 § 5: if the polytonic set in the base
# vocab is smaller than this, the polytonic centroid is unreliable and we
# fall back to the modern centroid.
POLYTONIC_FALLBACK_MIN_TOKENS = 50


def classify_greek_block(surface: str) -> tuple[bool, bool]:
    """Return (has_modern, has_polytonic) for a surface form.

    A token is `has_modern` if it contains any codepoint in U+0370-U+03FF.
    A token is `has_polytonic` if it contains any codepoint in U+1F00-U+1FFF.
    A token can be in both sets simultaneously (rare; if so the centroid procedure
    averages the two centroids — see v0.7 § 5 step 3).
    """
    has_modern = False
    has_polytonic = False
    for ch in surface:
        c = ord(ch)
        if GREEK_MODERN_START <= c <= GREEK_MODERN_END:
            has_modern = True
        elif GREEK_POLYTONIC_START <= c <= GREEK_POLYTONIC_END:
            has_polytonic = True
        if has_modern and has_polytonic:
            break
    return has_modern, has_polytonic


def norm_match(vec: np.ndarray, target_norm: float) -> np.ndarray:
    """Rescale `vec` (shape [D] or [N, D]) to have L2 norm `target_norm`.

    Per Phase A § 8.2, Greek-content target is 5.05 for E and 3.80 for U.
    Direction is preserved; only the magnitude is adjusted.

    Returns a new array; does not mutate input.
    """
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        cur = np.linalg.norm(arr)
        if cur < 1e-12:
            return arr.copy()
        return (arr * (target_norm / cur)).astype(np.float32)
    elif arr.ndim == 2:
        cur = np.linalg.norm(arr, axis=1, keepdims=True)
        # avoid division by zero
        cur = np.where(cur < 1e-12, 1.0, cur)
        return (arr * (target_norm / cur)).astype(np.float32)
    else:
        raise ValueError(f"norm_match expects 1-D or 2-D, got shape {arr.shape}")


def identify_greek_base_tokens(
    base_tokenizer,
    base_vocab_size: int = 131_072,
) -> tuple[list[int], list[int], list[int]]:
    """Walk the base Apertus vocab (IDs 0..base_vocab_size) and return:
        (modern_set, polytonic_set, both_set)
    Each is a list of token IDs. `both_set` is the subset that has BOTH a
    modern-block char AND a polytonic-block char in its surface form (rare).

    Note: surface forms come from `base_tokenizer.decode([id])` which strips
    the ByteLevel prefix-space marker; that's fine for block classification.
    """
    modern_ids: list[int] = []
    polytonic_ids: list[int] = []
    both_ids: list[int] = []
    for tid in range(base_vocab_size):
        surface = base_tokenizer.decode([tid])
        has_m, has_p = classify_greek_block(surface)
        if has_m and has_p:
            both_ids.append(tid)
            modern_ids.append(tid)
            polytonic_ids.append(tid)
        elif has_m:
            modern_ids.append(tid)
        elif has_p:
            polytonic_ids.append(tid)
    return modern_ids, polytonic_ids, both_ids


def compute_centroid_and_std(
    matrix: np.ndarray,
    ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute centroid + per-dimension std of `matrix[ids, :]`.

    Returns (centroid, sigma), each of shape [D]. `sigma` is the per-dim
    std, equivalent to Mundra et al. 2024's "Univariate" baseline.

    DEPRECATED for the centroid init. Kept for backward compat with the
    smoke test. Use `compute_centroid_and_cov` for the production path
    per the audit at `../audits/INIT_PAPERS_AUDIT.md` Q6.
    """
    ids_arr = np.asarray(list(ids), dtype=np.int64)
    if ids_arr.size == 0:
        D = matrix.shape[1]
        return np.zeros(D, dtype=np.float32), np.zeros(D, dtype=np.float32)
    sub = matrix[ids_arr].astype(np.float32)  # [N, D]
    centroid = sub.mean(axis=0)
    sigma = sub.std(axis=0)  # per-dim std
    return centroid.astype(np.float32), sigma.astype(np.float32)


def compute_centroid_and_cov(
    matrix: np.ndarray,
    ids: Iterable[int],
    reg_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute centroid + full covariance of `matrix[ids, :]`.

    Returns (centroid, cov), shapes [D] and [D, D]. Matches Hewitt 2021's
    multivariate-normal recipe: `μ = mean`, `Σ = (E-μ).T @ (E-μ) / n`.
    Adds a small ridge `reg_eps * I` to Σ for numerical stability when
    sampling.

    [Cite: references/papers/hewitt_vocab_expansion.html lines 230-237;
     references/papers/mundra_2407.05841.pdf §3.3 Theorems 1-2 p.3,
     §4.4 "Multivariate" baseline p.5, Table 2 p.6]
    """
    ids_arr = np.asarray(list(ids), dtype=np.int64)
    if ids_arr.size == 0:
        D = matrix.shape[1]
        return np.zeros(D, dtype=np.float32), np.eye(D, dtype=np.float32) * reg_eps
    sub = matrix[ids_arr].astype(np.float32)  # [N, D]
    centroid = sub.mean(axis=0)
    centered = sub - centroid[None, :]
    n = sub.shape[0]
    # Σ = (E - μ)^T (E - μ) / n  per Hewitt
    cov = (centered.T @ centered) / max(n, 1)
    # Ridge for numerical stability — np.random.multivariate_normal otherwise
    # may complain about a singular covariance matrix.
    D = cov.shape[0]
    cov = cov + reg_eps * np.eye(D, dtype=np.float32)
    return centroid.astype(np.float32), cov.astype(np.float32)
