"""Centroid init arm.

Per v0.7 § 5, refined by the 2026-05-21 audit (see `../../AUDIT_FINDINGS.md`):

For each new token T:
  1. Decode T to its surface form.
  2. Classify by Unicode block: modern (U+0370–U+03FF), polytonic
     (U+1F00–U+1FFF), or both.
  3. Look up the appropriate (centroid, covariance) pair (per-script
     statistics over existing Greek tokens in the base Apertus vocab).
     If T has both tags, average the modern + polytonic centroids
     half-and-half (papers silent — Q9 in audit).
  4. Fallback: if the polytonic set is smaller than 50 base tokens,
     the polytonic statistics are unreliable and we fall back to the
     modern centroid for polytonic new tokens.
  5. Sample ε ~ N(0, Σ) — **full multivariate normal** per Hewitt 2021,
     not the diagonal-σ "Univariate" baseline Mundra calls inadequate
     (audit Q6).
  6. new_row = centroid + ε, then norm-match to Phase A target.

Same procedure independently for E and U.

[Cite: references/papers/hewitt_vocab_expansion.html (multivariate-normal
        recipe); references/papers/mundra_2407.05841.pdf §5.1 + Table 2 p.6
        (Univariate is inadequate)]

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
    compute_centroid_and_cov,
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

    # Use FULL covariance per Hewitt 2021 — diagonal-only is Mundra's "Univariate"
    # baseline which the paper explicitly calls inadequate (Mundra §5.1, p.5).
    # [Cite: references/papers/hewitt_vocab_expansion.html lines 230-237;
    #  references/papers/mundra_2407.05841.pdf §5.1 + Table 2 p.6]
    # Precompute Cholesky L once per script class so per-token sampling is
    # O(D²) instead of O(D³) (full multivariate_normal redoes the Cholesky on
    # every call).
    E_modern_mu, E_modern_cov = compute_centroid_and_cov(base_E, modern_ids)
    U_modern_mu, U_modern_cov = compute_centroid_and_cov(base_U, modern_ids)
    E_modern_L = np.linalg.cholesky(E_modern_cov)
    U_modern_L = np.linalg.cholesky(U_modern_cov)
    if polytonic_fallback:
        E_poly_mu, E_poly_L = E_modern_mu, E_modern_L
        U_poly_mu, U_poly_L = U_modern_mu, U_modern_L
        E_both_mu, E_both_L = E_modern_mu, E_modern_L
        U_both_mu, U_both_L = U_modern_mu, U_modern_L
    else:
        E_poly_mu, E_poly_cov = compute_centroid_and_cov(base_E, polytonic_ids)
        U_poly_mu, U_poly_cov = compute_centroid_and_cov(base_U, polytonic_ids)
        E_poly_L = np.linalg.cholesky(E_poly_cov)
        U_poly_L = np.linalg.cholesky(U_poly_cov)
        # Both-block: half-and-half of modern + polytonic (Q9 in audit; papers silent)
        E_both_mu = 0.5 * (E_modern_mu + E_poly_mu)
        U_both_mu = 0.5 * (U_modern_mu + U_poly_mu)
        E_both_L = np.linalg.cholesky(0.5 * (E_modern_cov + E_poly_cov))
        U_both_L = np.linalg.cholesky(0.5 * (U_modern_cov + U_poly_cov))

    if verbose:
        print(f"Centroid: E_modern_mu norm = {np.linalg.norm(E_modern_mu):.3f}")
        print(f"Centroid: E_modern_cov diagonal mean = {np.diag(E_modern_cov).mean():.4f}")
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
            E_mu, E_L = E_both_mu, E_both_L
            U_mu, U_L = U_both_mu, U_both_L
        elif has_polytonic:
            hist["polytonic_only"] += 1
            E_mu, E_L = E_poly_mu, E_poly_L
            U_mu, U_L = U_poly_mu, U_poly_L
        elif has_modern:
            hist["modern_only"] += 1
            E_mu, E_L = E_modern_mu, E_modern_L
            U_mu, U_L = U_modern_mu, U_modern_L
        else:
            hist["neither"] += 1
            E_mu, E_L = E_modern_mu, E_modern_L
            U_mu, U_L = U_modern_mu, U_modern_L
            if verbose and hist["neither"] <= 5:
                print(f"  Centroid: WARN id={new_id} surface={surface!r} has no Greek-block codepoint")

        # Sample N(μ, Σ) via precomputed Cholesky: x = μ + L @ z, z ~ N(0, I).
        # Mathematically identical to `rng.multivariate_normal(μ, Σ)` but
        # O(D²) per sample instead of O(D³) (the Cholesky was paid once above).
        z_E = rng.standard_normal(D).astype(np.float32)
        z_U = rng.standard_normal(D).astype(np.float32)
        new_E[offset] = E_mu + (E_L @ z_E)
        new_U[offset] = U_mu + (U_L @ z_U)

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
