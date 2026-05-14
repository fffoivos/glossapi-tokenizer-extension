# 03_1 Greek embedding diagnostic

## Scope

A diagnostic characterisation of how Apertus-8B-2509 represents Greek
on its input (E) and output (U) embedding matrices, plus a binary
Greek-vs-¬Greek separability check and a cross-language semantic
cluster comparison. Runs entirely on existing E and U rows of the
base model — no new-token init, no CPT, no LOO.

This **precedes** the embedding-extension and CPT work that 03 will
eventually drive. It is a snapshot of what the base model knows about
Greek before we change anything.

## Canonical plan

- [../../../docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md](../../../docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md)
  — the live plan (v2 + v2.1 floor filter + v2.3 spectrum/classifier corrections).

## Results

- [artifacts/results/report_v2.md](artifacts/results/report_v2.md) — the
  results report; the canonical write-up. Contains the headline contrast
  table, the binary classifier results, the §3 hull occupancy + §3.9–§3.10
  infiltrators check, §4 clustering, §5 long-token families + analogy
  arithmetic, §7b v2.1 follow-ups, §8 Phase E cross-language clusters.
- [artifacts/results/REVIEW.md](artifacts/results/REVIEW.md) — a session
  review across all phases (what was tried, what was archived, what
  cloud instances were used).
- [artifacts/results/report_phase2_preliminary.md](artifacts/results/report_phase2_preliminary.md)
  — **archived** Phase 2 LOO benchmark report; results were methodologically
  contaminated (LOO targets included in the centroid used to seed the inits,
  and the test reconstructs already-trained rows rather than test new-token
  init). Kept for traceability. The plan-v2 framing supersedes this.

## Scripts

All 17 live scripts at [scripts/](scripts/). Canonical execution order:

| # | script | purpose |
|---:|---|---|
| 1 | `extract_embeddings.py` | Download `swiss-ai/Apertus-8B-2509`; save E, U as fp32 |
| 2 | `build_token_freq.py` | Tokenise el HPLT + GlossAPI slices; per-id frequency counts |
| 3 | `phase0_centroids_and_pcs.py` | v1 multi-group centroids + top-K SVD per group |
| 4 | `recompute_strict_greek.py` | Replace broad-Greek (1,506) with strict 1,494; refit classifier |
| 5 | `phase0_2_7_10_linear_classifier.py` | v1 7-class linear classifier (kept for cross-comparison) |
| 6 | `build_groups_greek_vs_not.py` | v2 canonical groups file (Greek + ¬Greek) |
| 7 | `phase0_greek_vs_not_geometry.py` | v2 main pipeline: §3.1–§3.6 + §3.9 + §3.10 |
| 8 | `phase0_full_negreek_spectrum.py` | **v2.3 fix** — full d=4,096 eigendecomposition of ¬Greek covariance. **Replaces** the truncated top-500 randomized SVD that biased K_sig low and top-PC shares high. |
| 9 | `phase0_binary_classifier.py` | **v2.3** binary Greek vs ¬Greek logistic regression (the v2 plan promised this) |
| 10 | `phase0_direction_cosines.py` | **v2.3** per-token unit-direction projected onto top-K PCs (Fig 4 + Fig 5 ingredients) |
| 11 | `phase0_greek_clustering.py` | v2 §3.11 k-means k∈{8,16,32} on Greek subspace |
| 12 | `phase0_greek_families_and_analogies.py` | v2 §3.12 long-token families + Mikolov analogies |
| 13 | `render_diagnostic_plots_greek.py` | v2 figure suite |
| 14 | `phase0_infiltrators_filtered.py` | v2.1 floor-filtered infiltrators + quantile hull |
| 15 | `render_diagnostic_plots_v2_1.py` | v2.1 refreshed figures |
| 16 | `phase0_semantic_cluster_compare.py` | Phase E cross-language semantic cluster cosines |
| 17 | `render_diagnostic_plots_v2_3.py` | **v2.3** Fig 4 + Fig 5 + Fig 10 + refreshed Fig 7/8/9 |

Conventions: scripts read from `arrays/` (foundation arrays) and
`geometry/` (per-script intermediates) under the canonical run-dir
location (see "Data location"). Outputs go into the same paths.
Scripts are idempotent — running them again overwrites the
artefacts; no destructive cleanup needed.

## Artifacts

All live small artefacts mirrored under [artifacts/](artifacts/) — geometry
JSONs/NPZ/NPY, all figures, reports, frequency counts, the preliminary
Phase 2 results.

`artifacts/` ≈ 168 MB total. **NOT mirrored** (because they're 4 GB and
trivially regenerable by running `extract_embeddings.py`):

- `arrays/E_fp32.npy` (2.15 GB) — Apertus input embeddings, fp32 cast
- `arrays/U_fp32.npy` (2.15 GB) — Apertus output embeddings, fp32 cast
- `arrays/{E,U}_norms.npy` — per-row L2 norms

These live in the canonical run-dir (see below).

## Data location

The canonical run-dir is at:

```
/home/foivos/runs/apertus_embedding_init_test_20260512/
```

It contains the foundation arrays `arrays/E_fp32.npy`, `U_fp32.npy`,
plus the same per-pipeline geometry/, figures/, and reports that are
mirrored here. Scripts in this sub-subproject hard-code those paths;
to relocate, edit `ROOT = Path(...)` at the top of each script.

## v2 → v2.1 → v2.3 lineage

This sub-subproject went through three rounds of correction:

- **v2** (initial Greek-vs-¬Greek pipeline). Headline finding: Greek
  centroid is geometrically displaced from global; Greek hull is
  permeable to many ¬Greek tokens.
- **v2.1** (floor filter + quantile hull). The naive top-1000
  infiltrators were dominated by Mistral-inherited untrained-floor
  rows (`<|fim_begin|>`, `[AVAILABLE_TOOLS]`, etc.). Filtering by
  `‖row‖ ≤ p1(classified)` revealed the underlying distribution.
- **v2.3** (spectrum fix + missing plan pieces). The ¬Greek spectrum
  in v2 was computed via truncated randomized SVD (top-500), which
  capped K_significant near 500 by construction and biased PR / κ /
  top-PC shares. v2.3 supplies the full d=4,096 covariance
  eigendecomposition, plus the binary Greek-vs-¬Greek classifier
  and per-token direction-cosine artefacts that the plan promised
  but v2 omitted.

The `report_v2.md` reflects the v2.3 corrected numbers and acknowledges
the v2 numbers were biased in the headline table.

## Status

**Diagnostic complete and live.** Awaiting:
- The user's per-language attribution (per-language `apertus-vocab-attr-w[0-7]`
  worker outputs) before reopening the per-language version of this
  diagnostic for non-Greek script-aggregates (Cyrillic / CJK /
  Latin-script).
- The strict-Greek U-side linear classifier recompute (legacy 7-class,
  `geometry/linear_classifier_U.json`) is still finishing in a
  background process (PID 5339). v2/v2.3 outputs are unaffected — the
  v2.3 binary classifier under `artifacts/geometry/v2/linear_classifier_binary_U.json`
  already used the strict-Greek set.

## Key headline numbers (from `report_v2.md` §1 + §2.4)

- Greek (n=1,494) centroid displaced **0.676 (E)** / **0.733 (U)** from global; ¬Greek (n=126,990) sits at 0.008 / 0.009.
- K_significant (above Marchenko-Pastur edge): Greek **397 / 361**; ¬Greek **2,139 / 1,662** (full spectrum).
- Binary Greek-vs-¬Greek classifier macro F1: **0.99 (E)** / **1.00 (U)**. `cos(weight, μ_Greek − μ_¬Greek)` = **0.85 / 0.92**.
- Greek tokens cluster by morphology: `μέν*`, `ματ*`, `μεγ-`, `αυτ-`, `συν-` families 5–9× tighter than random.
- en↔el cosine for Greek-origin concepts (democracy / philosophy / mathematics): **+0.05** average — no etymology-bridge advantage over non-Greek-origin pairs (**+0.04**).
- Mikolov word-level analogies don't work because byte-level BPE splits most Greek words.
