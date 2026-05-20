"""Centroid init arm.

Per v0.7 § 5:

For each new token T:
  1. Decode T to its surface form.
  2. Classify by Unicode block: modern (U+0370–U+03FF), polytonic
     (U+1F00–U+1FFF), or both.
  3. Look up the appropriate centroid (per-script mean of existing
     Greek tokens in the base Apertus vocab). If T has both tags,
     average the modern + polytonic centroids.
  4. Fallback: if the polytonic set is smaller than 50 base tokens,
     the polytonic centroid is unreliable and we fall back to the
     modern centroid for polytonic new tokens.
  5. Sample noise ε ~ N(0, diag(σ)) where σ is the per-dim std of the
     chosen set around its centroid.
  6. new_row = centroid + ε, then norm-match to Phase A target.

Same procedure independently for E and U.

Unlike ReTok, Centroid doesn't use the new token's specific subpiece
decomposition — it uses a script-level distributional prior. The
bakeoff tests whether the per-token subpiece info (ReTok) is worth
the computation vs the simpler script-level prior (Centroid).
"""
from __future__ import annotations

import numpy as np

from _common import (
    NORM_TARGET_E_GREEK,
    NORM_TARGET_U_GREEK,
    POLYTONIC_FALLBACK_MIN_TOKENS,
    classify_greek_block,
    compute_centroid_and_std,
    identify_greek_base_tokens,
    norm_match,
)


def compute_centroid_init(
    *,
    base_E: np.ndarray,
    base_U: np.ndarray,
    base_tokenizer,
    extended_tokenizer,
    new_id_range: tuple[int, int] = (131_072, 153_600),
    base_vocab_size: int = 131_072,
    norm_target_E: float = NORM_TARGET_E_GREEK,
    norm_target_U: float = NORM_TARGET_U_GREEK,
    seed: int = 20_260_520,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (new_E_rows, new_U_rows, stats_dict).

    new_E_rows, new_U_rows are [N_new, D] and norm-matched.
    stats_dict contains: modern_set_size, polytonic_set_size, both_set_size,
    polytonic_fallback_to_modern (bool), and per-new-token classification
    histogram.
    """
    assert base_E.shape == base_U.shape, f"E/U shape mismatch"
    D = base_E.shape[1]
    new_start, new_end = new_id_range
    N_new = new_end - new_start

    # Step 1+2: identify Greek base tokens and compute per-script centroids
    if verbose:
        print(f"Centroid: walking {base_vocab_size:,} base tokens to find Greek-block members...")
    modern_ids, polytonic_ids, both_ids = identify_greek_base_tokens(
        base_tokenizer, base_vocab_size=base_vocab_size
    )

    polytonic_fallback = len(polytonic_ids) < POLYTONIC_FALLBACK_MIN_TOKENS
    if verbose:
        print(f"Centroid: modern set = {len(modern_ids):,}, polytonic set = {len(polytonic_ids):,}, both = {len(both_ids):,}")
        if polytonic_fallback:
            print(f"Centroid: polytonic set < {POLYTONIC_FALLBACK_MIN_TOKENS} → polytonic new tokens will use modern centroid (fallback)")

    E_modern_mu, E_modern_sigma = compute_centroid_and_std(base_E, modern_ids)
    U_modern_mu, U_modern_sigma = compute_centroid_and_std(base_U, modern_ids)
    if polytonic_fallback:
        E_poly_mu, E_poly_sigma = E_modern_mu, E_modern_sigma
        U_poly_mu, U_poly_sigma = U_modern_mu, U_modern_sigma
    else:
        E_poly_mu, E_poly_sigma = compute_centroid_and_std(base_E, polytonic_ids)
        U_poly_mu, U_poly_sigma = compute_centroid_and_std(base_U, polytonic_ids)

    if verbose:
        print(f"Centroid: E_modern_mu norm = {np.linalg.norm(E_modern_mu):.3f}")
        print(f"Centroid: E_modern_sigma mean = {E_modern_sigma.mean():.4f}")
        print(f"Centroid: U_modern_mu norm = {np.linalg.norm(U_modern_mu):.3f}")

    # Step 3-5: for each new token, classify + sample
    rng = np.random.default_rng(seed)
    new_E = np.zeros((N_new, D), dtype=np.float32)
    new_U = np.zeros((N_new, D), dtype=np.float32)
    hist = {"modern_only": 0, "polytonic_only": 0, "both": 0, "neither": 0}

    for offset in range(N_new):
        new_id = new_start + offset
        surface = extended_tokenizer.decode([new_id], skip_special_tokens=False)
        has_modern, has_polytonic = classify_greek_block(surface)
        if has_modern and has_polytonic:
            hist["both"] += 1
            E_mu = 0.5 * (E_modern_mu + E_poly_mu)
            U_mu = 0.5 * (U_modern_mu + U_poly_mu)
            E_sigma = 0.5 * (E_modern_sigma + E_poly_sigma)
            U_sigma = 0.5 * (U_modern_sigma + U_poly_sigma)
        elif has_polytonic:
            hist["polytonic_only"] += 1
            E_mu, E_sigma = E_poly_mu, E_poly_sigma
            U_mu, U_sigma = U_poly_mu, U_poly_sigma
        elif has_modern:
            hist["modern_only"] += 1
            E_mu, E_sigma = E_modern_mu, E_modern_sigma
            U_mu, U_sigma = U_modern_mu, U_modern_sigma
        else:
            hist["neither"] += 1
            # Defensive fallback: a Greek-extension token with no Greek-block
            # codepoint shouldn't exist, but if it does, use modern centroid.
            E_mu, E_sigma = E_modern_mu, E_modern_sigma
            U_mu, U_sigma = U_modern_mu, U_modern_sigma
            if verbose and hist["neither"] <= 5:
                print(f"  Centroid: WARN id={new_id} surface={surface!r} has no Greek-block codepoint")

        # Sample noise per-dim from N(0, sigma_d)
        eps_E = rng.standard_normal(D).astype(np.float32) * E_sigma
        eps_U = rng.standard_normal(D).astype(np.float32) * U_sigma
        new_E[offset] = E_mu + eps_E
        new_U[offset] = U_mu + eps_U

    # Step 6: norm-match
    new_E = norm_match(new_E, target_norm=norm_target_E)
    new_U = norm_match(new_U, target_norm=norm_target_U)

    stats = {
        "modern_set_size": len(modern_ids),
        "polytonic_set_size": len(polytonic_ids),
        "both_set_size": len(both_ids),
        "polytonic_fallback_to_modern": polytonic_fallback,
        "new_token_classification": hist,
    }
    if verbose:
        print(f"Centroid: new-token classification: {hist}")
        print(f"Centroid: E norm sample = {np.linalg.norm(new_E[0]):.3f}, target = {norm_target_E}")
        print(f"Centroid: U norm sample = {np.linalg.norm(new_U[0]):.3f}, target = {norm_target_U}")
    return new_E, new_U, stats
