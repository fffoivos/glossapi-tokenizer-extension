# 03_1 Greek (and pan-language) embedding diagnostic

## Scope

A diagnostic characterisation of how Apertus-8B-2509 represents languages
on its input (E) and output (U) embedding matrices. Runs entirely on
existing E and U rows of the base model — no new-token init, no CPT,
no LOO.

This **precedes** the embedding-extension and CPT work that 03 will
eventually drive. The pipeline went through three iterations:

- **v2 series** (single-anchor): Greek vs ¬Greek, byte-level classifier
  groups, hull/infiltrator/family analysis. Headline finding: Greek is a
  coherent geometric subspace with morphologically meaningful clustering;
  the Mahalanobis-into-hull "infiltrators" finding is dominated by a
  projection-asymmetry geometric artefact, not semantic Greek-overlap.
- **v3 series** (11 PMI-attributed languages): proper per-language token
  sets from `02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution`.
  Centroid + spectrum + subspace overlap + shared dimensions + variance
  capture + L-discriminant directions + pair-specific shared subspaces.
  **v3-corrected** (2026-05-15) fixes a Marchenko-Pastur edge bug and a
  truncated-covariance bug in pair-specificity; numbers materially shifted.
- **v4 series** (all 75 well-sampled languages): same pipeline scaled to
  the 75 PMI-attributed languages with non-empty masked sets. Drops 12
  empty-masked-set languages (Amharic, Khmer, Sinhala, Lao, Tibetan,
  Oriya, Dhivehi — Apertus's tokenizer byte-fragments those scripts —
  plus Middle High German and 4 undetermined-language sets).

## Canonical plan

[../../../docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md](../../../docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md)
— the design framing. v3 and v4 implementations followed that plan with
methodology refinements documented inline in the reports.

## Results

- [artifacts/results/report_v2.md](artifacts/results/report_v2.md) —
  v2 series report (Greek vs ¬Greek). Headline contrast table, binary
  classifier, hull occupancy, clustering, families, semantic clusters.
- [artifacts/results/report_v3_subspace_meaning.md](artifacts/results/report_v3_subspace_meaning.md)
  — v3 series report (11 languages, **v3-corrected**). Generalised
  eigendecomposition for language-discriminant directions + principal-
  angle pair-specific shared subspaces with row-based specificity check.
- [artifacts/results/report_v4_full_panel.md](artifacts/results/report_v4_full_panel.md)
  — v4 series report (all 75 well-sampled PMI-attributed languages).
  Covers canonical steps 1–3 (geometry, subspace overlap, shared dims).
  Steps 4 + 5 not run at this scale (see report §6).
- [artifacts/results/report_v4_vocab_and_training_inference.md](artifacts/results/report_v4_vocab_and_training_inference.md)
  — analytical follow-up to the v4 panel. Vocab-allocation distribution,
  language-family patterns, Heaps-law relative training-mass estimates.
- [artifacts/results/report_phase2_preliminary.md](artifacts/results/report_phase2_preliminary.md)
  — **archived** Phase 2 LOO benchmark report; results were
  methodologically contaminated. Kept for traceability only.
- [artifacts/results/REVIEW.md](artifacts/results/REVIEW.md) — session
  review across all phases (what was tried, what was archived, cloud
  instance history).

## Scripts

Live scripts at [scripts/](scripts/), organised by series.

### v1-v2 series (single-anchor Greek vs ¬Greek)

| script | purpose |
|---|---|
| `extract_embeddings.py` | Download `swiss-ai/Apertus-8B-2509`; save E, U as fp32 |
| `build_token_freq.py` | Tokenise el HPLT + GlossAPI slices; per-id frequency counts |
| `phase0_centroids_and_pcs.py` | v1 multi-group centroids + top-K SVD per group |
| `recompute_strict_greek.py` | Replace broad-Greek (1,506) with strict 1,494; refit classifier |
| `phase0_2_7_10_linear_classifier.py` | v1 7-class linear classifier |
| `build_groups_greek_vs_not.py` | v2 canonical groups file |
| `phase0_greek_vs_not_geometry.py` | v2 main pipeline (§3.1–§3.6 + §3.9 + §3.10) |
| `phase0_full_negreek_spectrum.py` | v2.3 fix — full d=4,096 eigendecomp of ¬Greek covariance |
| `phase0_binary_classifier.py` | v2.3 binary Greek vs ¬Greek logistic regression |
| `phase0_direction_cosines.py` | v2.3 per-token unit-direction projected onto top-K PCs |
| `phase0_greek_clustering.py` | v2 §3.11 k-means k∈{8,16,32} |
| `phase0_greek_families_and_analogies.py` | v2 §3.12 morphology families + Mikolov analogies |
| `phase0_infiltrators_filtered.py` | v2.1 floor-filtered infiltrators + quantile hull |
| `phase0_semantic_cluster_compare.py` | Phase E cross-language semantic cluster cosines |
| `render_diagnostic_plots_greek.py`, `render_diagnostic_plots_v2_1.py`, `render_diagnostic_plots_v2_3.py` | v2.x figure suites |

### v3 series (11 PMI-attributed languages)

| script | purpose |
|---|---|
| `build_groups_perlang_v3.py` | 11-language canonical groups file |
| `phase0_perlang_geometry_v3.py` | per-language centroid + full spectrum + MP edge + K_sig + PR + κ + in-group hull + direction cosines |
| `phase0_perlang_subspace_overlap_v3.py` | pairwise principal-angle subspace overlap + top-1 PC alignment |
| `phase0_perlang_shared_dims_v3.py` | excess-shared-dims matrix + variance capture matrix |
| `phase0_pair_specific_shared_v3.py` | canonical-direction extraction + row-based specificity check |
| `phase0_perlang_discriminant_v3.py` | generalised eigendecomposition: L-discriminant directions |
| `render_perlang_plots_v3.py` | figure suite |

**v3-corrected (2026-05-15)** in-place patches in
`phase0_perlang_geometry_v3.py` (MP edge formula: `c = d/n`, not `min/max`;
K_floor = 32 for rank-deficient groups) and
`phase0_pair_specific_shared_v3.py` (`var_C(d)` computed from centred rows
directly, not from truncated K_sig PC reconstruction).

### v4 series (75 well-sampled PMI-attributed languages)

`*_v4.py` versions of the v3 scripts, paths re-routed to
`artifacts/geometry/v4_perlang/` and `artifacts/figures/v4_perlang/`.
Plus:

| script | purpose |
|---|---|
| `build_groups_88lang_v4.py` | All 75 well-sampled languages canonical groups file (12 skipped: 7 empty-masked-set scripts + 4 `und_*` + 1 historical) |

## Artifacts

`artifacts/` ≈ 4.1 GB total (geometry NPZ/JSON for v3 + v4 series — the
v4 75-language per-language artefacts dominate the size — plus figures,
reports, frequency counts, and the preliminary Phase 2 results).
Local-only (`.gitignore`'d at the project level); regenerable by
re-running the scripts.

**NOT mirrored** (~4 GB, regenerable by `extract_embeddings.py`):
- `arrays/E_fp32.npy`, `U_fp32.npy` — Apertus input + output embeddings, fp32 cast
- `arrays/{E,U}_norms.npy` — per-row L2 norms

These live in the canonical run-dir (see below).

## Data location

The canonical run-dir is at:

```
/home/foivos/runs/apertus_embedding_init_test_20260512/
```

It contains the foundation arrays `arrays/E_fp32.npy`, `U_fp32.npy`,
plus the v2-series geometry/, figures/, and reports. The v3 and v4
series write to the sub-subproject's own `artifacts/geometry/` and
`artifacts/figures/` paths. Scripts hard-code these paths; to relocate,
edit `ROOT = Path(...)` (run-dir) and `SP = Path(...)` (sub-subproject)
at the top of each script.

## Methodology evolution

- **v2 → v2.3** (within Greek vs ¬Greek):
  - v2.1 filter the untrained-floor before computing Mahalanobis-to-Greek hull
  - v2.3 fix the truncated-SVD bug that biased ¬Greek's K_sig low + top-PC shares high
  - v2.3 implement binary classifier + direction-cosine artefacts promised by the plan
- **v3 → v3-corrected** (11 PMI languages):
  - MP edge formula was using `q = min(d,n)/max(d,n)` (always ≤ 1). Correct
    aspect ratio is `c = d/n` (can exceed 1 for n < d). This dropped K_sig
    substantially for rank-deficient languages (Greek E 619 → 123; Georgian
    218 → 0).
  - `var_C(d)` was computed from each language's truncated K_sig PC
    reconstruction, assigning zero variance to anything in C's sub-K_sig
    tail and inflating pair-specificity scores. Now computed from centred
    rows directly. Pair-specific direction counts dropped by orders of
    magnitude and reordered: tight-script cousins now dominate Greek's
    specificity rankings (Hindi, Armenian, Hebrew, Georgian, Thai), the
    wide-Latin partners collapse to specificity ≤ 1.5.
- **v3 → v4** (11 → 75 languages):
  - Same methodology, scaled to all well-sampled PMI-attributed languages.
  - Plot strategy adjusts for the 75-row pairwise matrices (small labels,
    hierarchical-cluster ordering recommended).

## Status

- **v2 series**: complete and live.
- **v3 series**: complete and live (corrected 2026-05-15). Includes
  pair-specific shared subspace (step 4) and language-discriminant
  directions (step 5).
- **v4 series**: **steps 1–3 complete** (geometry, subspace overlap,
  shared dims). Steps 4 and 5 deliberately **not run** at 75-lang
  scale per user scope guidance — they were added during methodological
  discussion later in v3 and aren't part of the originally-established +
  reviewed canonical pipeline at the time the 75-language run was
  requested. v3-11-lang results for steps 4 and 5 remain available in
  `report_v3_subspace_meaning.md`. See `report_v4_full_panel.md`.

## Key headline numbers (v3-corrected, 11 languages)

- Greek (n=1,479) centroid displaced **0.676 (E)** / **0.733 (U)** from
  classified-global; ¬Greek (n=126,990) sits at 0.008 / 0.009 (¬Greek
  IS the global by mass).
- K_significant (above MP edge): Greek **123 / 83** on E/U (down from
  v3-original 619 / 608 with the formula fix).
- Greek's strongest pair-specific direction by specificity score:
  Thai (2.23), Hindi (1.96), Georgian (1.93), Armenian (1.85). The wide-
  Latin partners drop to specificity ≤ 1.5 — Greek↔Korean specifically
  has **zero** pair-specific directions (all candidate-shared directions
  are also used by at least one other language).
- Greek tokens cluster by morphology: `μέν*`, `ματ*`, `μεγ-`, `αυτ-`,
  `συν-` families 5–9× tighter than random.
- en↔el cosine for Greek-origin concepts (democracy / philosophy /
  mathematics) averages **+0.05** in the static input-embedding view
  — no etymology-bridge advantage over non-Greek-origin pairs (+0.04).
- Mikolov word-level analogies don't work on byte-level BPE (7/8
  candidates skipped because key Greek words aren't single tokens).
