"""ReTok init arm.

Per v0.7 § 5 + `old_experiments_plan.md` § 5 Experiment 2:

For each new token T (IDs 131,072 ... 153,599 in the extended tokenizer):
  1. Decode T to its surface form `s`.
  2. Tokenize `s` with the BASE Apertus tokenizer (vocab 131,072) →
     subpiece IDs p1, p2, ..., pk.
  3. E[T] = mean(base_E[p_i] for i in 1..k), then Phase A norm-match
     to 5.05.
  4. U[T] = mean(base_U[p_i] for i in 1..k), then Phase A norm-match
     to 3.80.

This is the "base-piece retokenization" variant of ReTok (per
`old_experiments_plan.md` § 5: avoids the merge-order chaining of
the strict ReTok paper, which propagates errors through dependent
merges). It's what production extensions like EEVE-Korean and
Chinese LLaMA actually do.

This module is pure NumPy (no HF model load). The driver
`build_init_checkpoints.py` is responsible for actually writing the
result rows into the resized model and calling `model.save_pretrained`.

Usage:
    new_E_rows, new_U_rows = compute_retok_init(
        base_E=numpy_E_matrix,           # [131072, 4096]
        base_U=numpy_U_matrix,           # [131072, 4096]
        base_tokenizer=base_tk,          # HF AutoTokenizer for Apertus base
        extended_tokenizer=ext_tk,       # HF AutoTokenizer for the ship bundle
        new_id_range=(131072, 153600),
    )
    # new_E_rows.shape == (22528, 4096), norm-matched to 5.05
    # new_U_rows.shape == (22528, 4096), norm-matched to 3.80
"""
from __future__ import annotations
from typing import Iterable

import numpy as np

from _common import (
    NORM_TARGET_E_GREEK,
    NORM_TARGET_U_GREEK,
    norm_match,
)


def compute_retok_init(
    *,
    base_E: np.ndarray,
    base_U: np.ndarray,
    base_tokenizer,
    extended_tokenizer,
    new_id_range: tuple[int, int] = (131_072, 153_600),
    norm_target_E: float = NORM_TARGET_E_GREEK,
    norm_target_U: float = NORM_TARGET_U_GREEK,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (new_E_rows, new_U_rows) for IDs in `new_id_range`.

    Both returned arrays are [N_new, D]. Rows are L2-norm-matched to
    `norm_target_*` (5.05 for E, 3.80 for U by default).
    """
    assert base_E.shape == base_U.shape, f"E/U shape mismatch: {base_E.shape} vs {base_U.shape}"
    D = base_E.shape[1]
    new_start, new_end = new_id_range
    N_new = new_end - new_start
    new_E = np.zeros((N_new, D), dtype=np.float32)
    new_U = np.zeros((N_new, D), dtype=np.float32)

    n_decoded_empty = 0
    n_subpieces_total = 0
    n_max_subpieces = 0
    for offset in range(N_new):
        new_id = new_start + offset
        # Use the extended tokenizer to recover the surface form for this new ID
        surface = extended_tokenizer.decode([new_id], skip_special_tokens=False)
        if not surface:
            # Should not happen for a well-formed extension, but be defensive
            n_decoded_empty += 1
            continue
        # Re-tokenize the surface form with the BASE tokenizer (vocab 131,072).
        # We use add_special_tokens=False so we get only the content subpieces.
        subpiece_ids = base_tokenizer.encode(surface, add_special_tokens=False)
        if not subpiece_ids:
            # Defensive — base tokenizer should always produce at least 1 piece
            continue
        n_subpieces_total += len(subpiece_ids)
        n_max_subpieces = max(n_max_subpieces, len(subpiece_ids))
        # Average the base-vocab subpiece embeddings
        sub_E = base_E[subpiece_ids].astype(np.float32).mean(axis=0)  # [D]
        sub_U = base_U[subpiece_ids].astype(np.float32).mean(axis=0)  # [D]
        new_E[offset] = sub_E
        new_U[offset] = sub_U
        if verbose and offset < 5:
            print(f"  ReTok new_id={new_id} surface={surface!r:<30} → {len(subpiece_ids)} subpieces {subpiece_ids[:6]}")

    # Norm-match each row to Phase A target
    new_E = norm_match(new_E, target_norm=norm_target_E)
    new_U = norm_match(new_U, target_norm=norm_target_U)

    if verbose:
        avg_subpieces = n_subpieces_total / max(N_new - n_decoded_empty, 1)
        print(f"ReTok: {N_new:,} new rows, mean subpieces per new token = {avg_subpieces:.2f}, max = {n_max_subpieces}")
        print(f"ReTok: empty-decode count = {n_decoded_empty}")
        print(f"ReTok: E norm sample = {np.linalg.norm(new_E[0]):.3f}, target = {norm_target_E}")
        print(f"ReTok: U norm sample = {np.linalg.norm(new_U[0]):.3f}, target = {norm_target_U}")
    return new_E, new_U
