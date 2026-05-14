# Embedding diagnostic plan v2 — per-language manifold

Date: 2026-05-13. Supersedes
[`_archive/APERTUS_EMBEDDING_INIT_TEST_PLAN_v4.md`](_archive/APERTUS_EMBEDDING_INIT_TEST_PLAN_v4.md)
(v4); the init-benchmark layer of v4 is archived (see §1 below).

This plan re-anchors the embedding work as a **pure diagnostic** —
characterising how Apertus-8B-2509 represents each target language
on its input (E) and output (U) embedding manifolds. No init
benchmark, no LOO-on-existing-tokens, no contrast-subset framing.

**Implementation lives under**:
[`../subprojects/03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/`](../subprojects/03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/README.md)
— 17 scripts, ~168 MB of small artefacts (geometry JSONs/NPZ, figures,
reports), plus a pointer to the 4 GB E/U arrays in the canonical
run-dir. Read the sub-subproject's README first, then this plan.

Headline results (from the sub-subproject's `report_v2.md`):

- Greek centroid is displaced 0.676 (E) / 0.733 (U) from the
  classified-global centroid; ¬Greek's centroid IS the global by mass.
- Binary Greek-vs-¬Greek classifier macro F1 = 0.99 (E) / 1.00 (U);
  `cos(weight, μ_Greek − μ_¬Greek)` = 0.85 / 0.92.
- Greek tokens cluster meaningfully by morphology (`μέν*`, `ματ*`,
  `συν-`, `αυτ-`, `πολ-` families 5–9× tighter than random).
- Greek's hull is permeable to ¬Greek tokens at low Mahalanobis, but
  the closest are under-trained-language content (E) and predictable-
  output sentence-boundary patterns (U), not semantically Greek-like.
- No strong static-embedding evidence of a Greek↔English etymology
  bridge in input embeddings (en↔el cosine ≈ +0.05 for Greek-origin
  concepts vs +0.04 for non-Greek-origin).

---

## 0. Why v2 (what changed from v4)

Three things forced a rewrite:

1. **The LOO-on-existing-tokens benchmark tested the wrong thing.**
   The framing in v4 was "for each existing Greek token t, predict
   what the trained embedding looks like using only knowledge of
   the rest of Greek". But the goal Apertus's extension actually
   needs answered is: "for a **new** Greek-script token (not in
   Apertus's vocab yet), what init will let CPT converge to a
   well-placed embedding?". Those are different problems:
   - LOO reconstructs an **already-trained** row from aggregate
     geometry of other already-trained rows. The aggregate captures
     the model's *current* compression of Greek; the test token's
     trained embedding was itself part of that compression.
   - A new-token init has nothing in the model yet — the question is
     where to **place** it so that gradient descent during CPT moves
     it usefully.
   The technical self-inclusion of the LOO target in the centroid
   (each at 0.066% weight) is a secondary issue; the primary issue
   is that LOO reconstruction quality is a poor proxy for CPT-init
   quality, which is what new-token addition actually depends on.
   Either question is interesting; v4's results spoke to the first
   but were sold as if they answered the second.
2. **The "Greek-clean vs Greek-script-misclassified" contrast is
   no longer wanted.** Pure-Greek-only (the strict 1,494 set from
   `base_greek_tokens.jsonl`) is the canonical definition; the 13
   mixed-script tokens (μm, μg, -α, Aβ, NBSP-μ, etc.) are simply
   excluded from "Greek".
3. **The diagnostic ideas now centre on per-language manifold
   geometry**, not on init candidates. Each language gets the same
   measurements; the comparison is between languages, not between
   inits.

What survives from v4:
- The per-group centroid + top-K SVD machinery (Phase 0 §2.1–§2.3
  in v4 → §3.1 here).
- The linear language-classifier as a separability summary
  (§2.7.10 in v4 → §3.7 here).
- The pure-Greek strict definition (1,494 tokens) introduced in
  the cleanup pass.

What's dropped:
- All §3–§5 (init-method catalog, LOO, geometry-fit).
- §6.2 init-recommendation half.
- §6.5 mini-CPT.
- §2.7.4 Greek-clean vs Greek-script-misclassified subcontrast.

---

## 1. Cleanup proposal (requires user OK before any moves)

Two-step cleanup: archive the v4-only artefacts, harmonise the
live docs to v2's framing.

### 1.1 Docs to archive (move to `docs/_archive/`)

| current path | reason | move to |
|---|---|---|
| `docs/APERTUS_EMBEDDING_INIT_TEST_PLAN.md` | replaced by this doc; the init-benchmark + diagnostic-layer split is no longer the framing | `docs/_archive/APERTUS_EMBEDDING_INIT_TEST_PLAN_v4.md` |
| `docs/PHASE2_MULTI_GPU_EXECUTION_PLAN.md` | the multi-GPU operational doc — was for an init benchmark that won't be re-run | `docs/_archive/PHASE2_MULTI_GPU_EXECUTION_PLAN.md` |
| `docs/EXTENSION_DOC_FEEDBACK_20260511.md` | feedback notes that fed into v3 of the init plan; no longer load-bearing | `docs/_archive/EXTENSION_DOC_FEEDBACK_20260511.md` |
| `docs/APERTUS_GREEK_BEHAVIORAL_NLL_PHASE_B.md` | Phase B v5 NLL triangulation report; the diagnostic findings are preserved in the new plan, but the report itself was a discrete deliverable | keep in place, add archive note at the top |

### 1.2 Run-dir artefacts to archive

In `/home/foivos/runs/apertus_embedding_init_test_20260512/`:

| path | action | reason |
|---|---|---|
| `scripts/build_init_catalog.py` | archive → `scripts/_archive/` | init-benchmark only |
| `scripts/build_loo_target_ids.py` | archive | LOO-only |
| `scripts/build_phase2_assignments.py` | archive | LOO-only |
| `scripts/run_phase2_worker.py` | archive | LOO-only |
| `scripts/run_phase2_orchestrator.py` | archive | LOO-only |
| `scripts/aggregate_phase2.py` | archive | LOO-only |
| `scripts/preflight_swap_round_trip.py` | archive | LOO-only |
| `scripts/gcloud_phase2_setup.sh` | archive | LOO-only |
| `scripts/run_home_prereqs.sh` | archive | chains LOO-only scripts |
| `loo_inputs/init_candidates/` | archive → `loo_inputs/_archive/` | LOO inits |
| `loo_inputs/assignments.json` | archive | LOO-only |
| `loo_inputs/loo_target_ids.json` | archive | LOO-only |
| `loo_inputs/loo_target_meta.json` | archive | LOO-only |
| `phase2_results/` | **keep** | preliminary biased benchmark, retained for traceability with an explicit caveat note |
| `report.md` | rename → `report_phase2_preliminary.md`, add stale-banner | Phase 2 reflects contaminated centroid; new diagnostic report will live separately |
| `arrays/E_fp32.npy`, `U_fp32.npy`, `*_norms.npy` | **keep** | foundational |
| `arrays/token_freq_*.npy` | **keep** | foundational |
| `geometry/centroids_*.npy`, `pc_basis_*.npy`, `anisotropy_*.json`, `linear_classifier_*.json` | **keep** but re-run after strict-Greek fix | already partially regenerated by `recompute_strict_greek.py` |
| `scripts/extract_embeddings.py` | **keep** | reused by v2 |
| `scripts/build_token_freq.py` | **keep** | reused |
| `scripts/phase0_centroids_and_pcs.py` | **keep, refactor** to use strict groups | reused, needs strict-Greek + new-language additions |
| `scripts/phase0_2_7_10_linear_classifier.py` | **keep, refactor** | reused |
| `scripts/recompute_strict_greek.py` | **keep** for the strict-Greek replay | reused |

### 1.3 Live docs to harmonise

- `docs/PROJECT_INDEX.md`: drop the
  `APERTUS_EMBEDDING_INIT_TEST_PLAN` + `PHASE2_MULTI_GPU_EXECUTION_PLAN`
  references, add this v2 doc as the new diagnostic plan source of truth.
- `docs/CURRENT_STATUS.md`: update "current stage" wording from
  "embedding-init benchmark scaffolded" to "per-language manifold
  diagnostic active".
- `docs/ACTIVE_BACKLOG.md`: drop init-benchmark items, add the
  v2 per-language diagnostic items (one per language × matrix).
- `docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md`:
  keep, still load-bearing for the per-language interpretation.
- `docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`: keep,
  still load-bearing for cross-language context.

### 1.4 Sequence

The cleanup can happen as one atomic commit. Proposed order:
1. Add v2 doc (this file).
2. Move docs in §1.1 to `docs/_archive/`.
3. Move run-dir scripts + LOO inputs in §1.2 to their `_archive/`
   subdirs.
4. Rename `report.md` → `report_phase2_preliminary.md` with banner.
5. Harmonise links in §1.3.
6. Run `recompute_strict_greek.py` (in progress; check log) to
   replace the broad-Greek centroid/PCs/classifier with strict-Greek
   versions.

---

## 2. Scope — Greek vs ¬Greek only (this iteration)

The byte-level Phase A v2 classifier gives us script-level groups,
not language-level groups. Latin-script (the "English-baseline"
label of 73,910 tokens) is actually used across English / German /
French / Spanish / Italian / Portuguese / Dutch / Polish / Czech /
… — so any per-language statement based on it would be misleading.
Similarly Cyrillic conflates Russian / Ukrainian / Bulgarian / …,
CJK conflates Chinese + Japanese, Arabic-script conflates Arabic /
Persian / Urdu.

Only **Greek** is reasonably 1:1 with its script (Greek script ≈
Modern Greek). So for this iteration of the diagnostic we restrict
to two groups:

1. **Greek** (strict, n = 1,494) — from `base_greek_tokens.jsonl`.
2. **¬Greek** = the **classified subset minus Greek**
   (n = 128,484 − 1,494 = **126,990**).
   "Classified" = excludes `special`, `byte_fragment`,
   `whitespace_only`, `digits_only` buckets per the Phase A v2
   invariant.

The other script-level groups (English-baseline, Cyrillic, German,
French, CJK, structural) **collapse into ¬Greek** for now. The
proper per-language split will happen in a later iteration once a
corpus-based token-language attribution (built by tokenising the
training data per language) is available.

Languages explicitly **out of scope** for this iteration:
Hebrew, Arabic, Hindi, Russian, Korean — they need either dedicated
script-level classification (Hebrew, Arabic, Devanagari) or the
deferred corpus-based attribution.

---

## 3. Diagnostic specification — Greek + ¬Greek, per matrix

Run on each (group g ∈ {Greek, ¬Greek}, matrix M ∈ {E, U})
independently, plus one **cross-group** check (§3.9 infiltrators)
where we project ¬Greek tokens onto Greek's hull.

### 3.1 Centroids

- `μ_g_M = mean(M[g])` — vector of shape (4096,).
- `μ_global_M = mean(M[classified])` — global reference, classified
  subset only (excludes special / byte_fragment / whitespace_only /
  digits_only buckets per the Phase A v2 invariant).
- Save: `geometry/per_lang/{g}_{M}_centroid.npy`,
  `geometry/per_lang/{g}_{M}_centroid_to_mu_global.json` (norm + cosine).

### 3.2 Distance distribution to centroid

For each token t ∈ g, compute:
- **Euclidean**: `d_t = ‖M[t] − μ_g‖₂`
- **Mahalanobis** (in the per-language eigenbasis):
  `m_t = √Σ_k z_t,k² / λ_g,k`
  where `z_t,k = (M[t] − μ_g) · v_g,k` and `λ_g,k`, `v_g,k` are the
  per-language eigenvalues + eigenvectors.

Save:
- `geometry/per_lang/{g}_{M}_distance.npz` with arrays
  `ids` (token-id), `euclid`, `mahalanobis`.

### 3.3 Direction distribution to centroid

For each token t ∈ g:
- `v_t = (M[t] − μ_g) / ‖M[t] − μ_g‖`  — unit vector.
- Project onto top-K principal axes of the per-group covariance:
  `c_t,k = v_t · e_g,k` for k = 1..min(K, n_g).
- The per-token cosines `c_t,k` are the "directional signature".

Save:
- `geometry/per_lang/{g}_{M}_direction_cosines.npz` with arrays
  `ids`, `cos_topK` shape `(n_g, K)`.

K choice: K = min(64, n_g − 1) by default. K is increased to the
post-MP-floor count K_sig (§3.5) for visualisation.

### 3.4 Hull occupancy (Mahalanobis-σ thresholds)

For each language:
- Compute the cumulative distribution of Mahalanobis distance m_t.
- Report **fraction of tokens within 0.5σ / 1σ / 2σ / 3σ** of the
  centroid, where σ means **standard deviation of m_t within the
  group** (not σ of the underlying Gaussian assumption — we don't
  assume Gaussian).
- Top-20 outliers: tokens with the **largest m_t**. Save:
  - token-id
  - surface form (raw_token + decoded_text from Phase A v2's
    classification jsonl)
  - m_t value
  - per-axis breakdown (which top-K PCs they're most off in)

Save:
- `geometry/per_lang/{g}_{M}_hull.json`:
  ```
  {
    "mahalanobis_std": float,
    "frac_within_0_5_sigma": float,
    "frac_within_1_sigma": float,
    "frac_within_2_sigma": float,
    "frac_within_3_sigma": float,
    "top_20_outliers": [
      {"id": int, "raw_token": str, "decoded_text": str,
       "mahalanobis": float, "top3_pc_contribution": [int]},
      ...
    ]
  }
  ```

### 3.5 Marchenko-Pastur noise floor + significant-PC count

For each language:
- Centre the rows: `X = M[g] − μ_g`, shape (n_g, d) with d = 4096.
- Compute the **full spectrum** of singular values of `X / √n_g`,
  i.e. eigenvalues of `Cov_g = X^T X / n_g`. For n_g < d, there are
  only n_g nonzero eigenvalues (rank-deficient).
- Estimate the per-dim noise variance:
  `σ² = median(λ_lower_half)` — median of the bottom half of the
  spectrum, where the MP bulk dominates. Validated against
  `trace(Cov) / d` for sanity.
- **MP upper edge**:
  `λ_+ = σ² × (1 + √(min(d, n_g) / max(d, n_g)))²`
- **K_sig** = number of eigenvalues above `λ_+`. This is the
  per-language signal-bearing PC count.

Save:
- `geometry/per_lang/{g}_{M}_spectrum.npz` with arrays
  `eigenvalues` (length min(n_g, d)).
- `geometry/per_lang/{g}_{M}_mp.json`:
  ```
  {
    "n_g": int, "d": 4096,
    "sigma_sq_estimate": float,
    "mp_upper_edge": float,
    "k_significant": int,
    "k_significant_at_95pct_var": int  // alternative count via cumulative-var threshold
  }
  ```

### 3.6 Participation ratio + shape anisotropy

For each language, from the per-language spectrum {λᵢ}:
- **Variance share per PC**: `s_i = λᵢ / Σλⱼ`.
- **Cumulative variance**: `S_k = Σᵢ≤k sᵢ`.
- **Participation ratio**:
  `PR = (Σλᵢ)² / (n_eig × Σλᵢ²)` where `n_eig = min(n_g, d)`.
  PR ∈ (1/n_eig, 1]: PR = 1 means uniform spectrum (isotropic);
  PR → 1/n_eig means rank-1 collapse.
- **Shape anisotropy κ** (relative shape anisotropy from the
  gyration-tensor literature):
  `κ = ((n_eig × Σλᵢ²) / (Σλᵢ)² − 1) / (n_eig − 1)`.
  κ = 0 for isotropic, κ = 1 for rank-1.
  This is `1 − PR` rescaled to [0, 1]; reported alongside PR for
  convenience.
- **Top-1 PC variance share**: `s_1` (single-number anisotropy
  summary from v4).

Save:
- `geometry/per_lang/{g}_{M}_shape.json` with all of the above
  plus `cumulative_variance` array.

### 3.7 Linear classifier — binary Greek vs ¬Greek

Logistic regression on the combined classified set, binary label
(Greek = 1, ¬Greek = 0). Reports test accuracy, Greek F1 /
precision / recall, and the Greek weight vector. Compares the
weight vector to:
- `μ_Greek − μ_¬Greek` (Greek-vs-not centroid displacement)
- Top-1 PC of the per-Greek covariance
- Top-1 PC of the per-¬Greek covariance

Saves:
- `geometry/linear_classifier_binary_E.json`, `_U.json`

### 3.8 Cross-group summary table

`geometry/per_lang/cross_group_summary.csv` — one row per
(group, matrix) plus one "infiltrator-summary" row per matrix:

Columns:
- group (Greek / ¬Greek / infiltrators-into-Greek)
- matrix (E or U)
- n_tokens
- median ‖row‖
- ‖μ_g − μ_classified-global‖
- median Euclidean distance to own centroid
- median Mahalanobis distance to own centroid
- mahalanobis-σ std (within-group)
- frac within {0.5, 1, 2, 3} σ
- σ² noise estimate
- MP upper edge
- K_significant
- top-1 PC variance share
- top-10 PC variance share
- cumulative variance at K_significant
- PR (participation ratio)
- shape anisotropy κ
- linear-classifier Greek F1
- linear-classifier Greek precision
- linear-classifier Greek recall
- cos(classifier weight, μ_Greek − μ_¬Greek)

### 3.9 NEW — ¬Greek tokens inside Greek's hull ("infiltrators")

Symmetric to §3.4 (Greek outliers): instead of Greek tokens that
fall outside the Greek hull, find **¬Greek tokens that fall inside
the Greek hull**. Directly visualises how "private" Apertus's
Greek subspace is — if many non-Greek tokens land inside, Greek is
not geometrically isolated.

For each token t in ¬Greek (n = 126,990):
- Compute Mahalanobis distance to Greek centroid using **Greek's
  MP-significant PC subspace** (Σ_Greek is rank-deficient since
  n_Greek = 1,494 < d = 4,096; use the K_sig leading eigenvectors
  of Greek's covariance for the metric):
  `m_Greek(t) = √Σ_{k=1..K_sig} z_t,k² / λ_Greek,k`
  where `z_t,k = (M[t] − μ_Greek) · e_Greek,k`.
- σ from §3.4 is the std of Greek's *own* in-group Mahalanobis
  distances (i.e. distributed under the same metric as §3.4's
  Greek-outlier analysis). Same axis ⇒ §3.4 and §3.9 plots are
  directly comparable.

Report:
- **Fraction of ¬Greek tokens with m_Greek ≤ kσ** for
  k = 0.5, 1, 2, 3.
- **Top-20 infiltrators**: ¬Greek tokens with **smallest**
  m_Greek (most "Greek-like" by the metric). For each:
  - token-id
  - raw_token (byte-level surface form)
  - decoded_text
  - Phase A v2 group assignment (so we can see whether infiltrators
    are predominantly Latin, Cyrillic, structural, etc.)
  - m_Greek value
  - top-3 PCs contributing most to the proximity
- **Breakdown of infiltrators-within-1σ by their original group**:
  e.g. "of the 1,247 ¬Greek tokens within 1σ of μ_Greek, 836 are
  English-baseline, 192 are Cyrillic, 87 are structural, …".

Save:
- `geometry/per_lang/infiltrators_into_greek_{E,U}.json`:
  ```
  {
    "matrix": "E",
    "k_sig_used": int,
    "mahalanobis_std_greek_in_group": float,
    "frac_negreek_within_0_5_sigma": float,
    "frac_negreek_within_1_sigma": float,
    "frac_negreek_within_2_sigma": float,
    "frac_negreek_within_3_sigma": float,
    "count_negreek_within_1_sigma_by_source_group": {
      "English-baseline": int, "Cyrillic": int, ...
    },
    "top_20_infiltrators": [
      {"id": int, "raw_token": str, "decoded_text": str,
       "source_group": str, "mahalanobis_to_greek": float,
       "top3_pc_contribution": [int]},
      ...
    ]
  }
  ```

This is the diagnostic answer to "is Greek a private subspace?".
Companion plot in §4.

### 3.10 NEW — Top-1000 ¬Greek nearest the Greek centroid

Extension of §3.9. Take the top-1,000 ¬Greek tokens by smallest
`m_Greek` (Mahalanobis in Greek's K_sig subspace).

Reports:
- **Distance comparison**: empirical CDF of `m_Greek` for the
  1,000 ¬Greek tokens, plotted on the **same axis** as Greek's
  own in-group Mahalanobis CDF. Lets you read directly: "the
  median of these 1,000 ¬Greek tokens sits at the X-th percentile
  of Greek tokens' own distance distribution".
- **Percentile mapping**: for each of the 1,000, its percentile in
  Greek's own distance distribution. Histogram of these percentiles.
- **Source-group composition** of the top-1,000: how many from
  English-baseline / Cyrillic / German / French / CJK / structural.
  Useful for understanding which non-Greek populations cross
  furthest into Greek.

Save:
- `geometry/top1000_negreek_near_greek_{E,U}.json`
- `geometry/top1000_negreek_near_greek_{E,U}.csv` — full table
  for spreadsheet review.

### 3.11 NEW — Local clustering within Greek (qualitative)

Sub-cluster structure inside the Greek subspace. Looking for
"do Greek tokens form meaningful local clusters?"

Method:
- Project Greek's 1,494 rows onto Greek's top-K_sig PCs.
- Cluster in that K-dim space with **HDBSCAN** (`min_cluster_size=15`)
  and with **k-means** (k = 8, 16, 32). Two methods so the user
  can compare a density-based vs centroid-based decomposition.
- For each cluster: compute the cluster centroid, then list the
  20 nearest Greek tokens by cosine to that centroid.

Output: human-readable cluster gallery — per cluster, list of 20
member surface forms decoded back to readable Greek. The user
inspects manually for semantic / morphological / register meaning.

Save:
- `geometry/greek_clusters_E.json` — `{method, cluster_id: [member_ids, ...], cluster_centroid, top_20_by_closeness}`
- `geometry/greek_clusters_E.md` — pre-rendered markdown with
  decoded surface forms per cluster, ready to read.
- Same for U.

### 3.12 NEW — Long-token families + analogy arithmetic

#### 3.12.1 Long-token families

Filter Greek tokens by `len(decoded_text) ≥ 4` (drops single
characters and 2-3-letter fragments). Expected count: 400-600
tokens.

Family-discovery heuristic:
- For each long token, take the **first 3 letters** of decoded
  text (after stripping any leading space marker) as a "root key".
- Group tokens by root key. Families with ≥ 4 members are kept.
- For each family:
  - Mean intra-family cosine similarity (between members).
  - Compare to a size-matched random-subset baseline (10 bootstrap
    resamples from the broader 1,494 set).
  - Family-tightness ratio = `intra_family_similarity / random_baseline_similarity`.

Output: gallery of families with intra-family cosine similarity,
random baseline, tightness ratio, and 10-token preview per family.

Save:
- `geometry/greek_long_token_families_E.json`
- `geometry/greek_long_token_families_E.md`
- Same for U.

#### 3.12.2 Analogy arithmetic

Test the classic Mikolov-style `v(a) − v(b) + v(c) ≈ v(d)`
arithmetic on hand-picked Greek pairs. Candidates (subject to
tokenisation existence — many will require finding whole-word
tokens in Apertus's vocab):

- **Article gender**: ο / η / το
- **Singular → plural**: παιδί / παιδιά, βιβλίο / βιβλία, κόρη / κόρες
- **Masculine → feminine**: γιος / κόρη, αδελφός / αδελφή
- **Verb person**: γράφω / γράφεις, γράφω / γράφει
- **Verb tense**: γράφω / έγραψα (likely not single tokens —
  expect this to fail)

Method:
- For each candidate `(a, b, c, expected_d)`, compute the
  difference vector `Δ = E[a] − E[b]`.
- Apply: `target = E[c] + Δ`.
- Find the 5 nearest existing tokens by cosine to `target`,
  excluding `a`, `b`, `c` themselves.
- Report whether `expected_d` is in top-5, and the actual top-5
  with cosines.

Save:
- `geometry/greek_analogy_tests_E.json` per analogy:
  `{a, b, c, expected_d, ranks_of_expected, top5_results}`.
- Same for U.

This is exploratory — Apertus's byte-level BPE may split many
Greek words across multiple tokens, in which case the analogy
arithmetic isn't even well-defined for those forms. Report which
candidates have all four parts as single whole-word tokens, and
which had to be skipped because of tokenisation.

---

## 4. Visualisation specification

All plots in **matplotlib**, saved as `figures/*.png` +
`figures/*.pdf`. Two groups (Greek, ¬Greek) → fixed two-colour
palette across all plots: **Greek = blue**, **¬Greek = orange**.
Infiltrators visually inherit Greek's panel position but use a
distinct hatch / line style.

### 4.1 Distance distribution

**Figure 1: distance KDE, faceted by matrix**
- 2-panel figure (E, U).
- x-axis = Euclidean distance to *own* centroid; y-axis = density.
- Two curves overlaid per panel: Greek (centred on μ_Greek),
  ¬Greek (centred on μ_¬Greek). Semi-transparent fill.
- Median + 90th percentile annotated per curve.

**Figure 2: Mahalanobis distance KDE, faceted by matrix**
- Same shape as Fig 1 but x = Mahalanobis distance to *own*
  centroid using *own* MP-significant subspace.
- Vertical reference lines at 0.5σ / 1σ / 2σ / 3σ (where σ is the
  within-group std of the empirical Mahalanobis distance).

**Figure 2b: Infiltrators — ¬Greek tokens scored against Greek's
hull (new, the §3.9 plot)**
- 2-panel figure (E, U).
- Overlaid KDE of Mahalanobis distance **to μ_Greek using Greek's
  MP-significant subspace**:
  - Curve A: Greek tokens (already in Figure 2's Greek curve;
    repeated here for reference).
  - Curve B: ¬Greek tokens — same metric, different distribution.
- Same σ reference lines as Fig 2 (defined on Greek's in-group
  Mahalanobis).
- The overlap region (where curve B's mass is below curve A's
  3σ line) is the headline finding: "the fraction of ¬Greek
  tokens that look Greek-like geometrically".

### 4.2 Direction distribution

**Figure 3: top-2-PC scatter density, faceted by group × matrix**
- 4-panel figure (Greek×E, Greek×U, ¬Greek×E, ¬Greek×U).
- For each, project the centred rows onto the *group's own*
  top-2 PCs. Hexbin density.
- Axis z-scored per panel by λ₁, λ₂ (so circles ↔ Mahalanobis-unit).
- For Greek panels: overlay the top-20 infiltrators from §3.9 as
  red dots, labelled by their surface form.

**Figure 4: cosine-to-top-1-PC distribution**
- KDE of `cos(v_t, e_g,1)` per token, per group, per matrix.
- 2 panels (E, U), 2 curves per panel.

**Figure 5: top-K PC angular heatmap**
- For each group, mean `|cos(v_t, e_g,k)|` for k = 1..K_sig.
- 2 rows (Greek, ¬Greek) × 2 columns (E, U) heatmaps.

### 4.3 Hull occupancy + infiltrators

**Figure 6: hull-occupancy stacked bars, faceted by matrix**
- Two groups of bars: Greek-vs-own-hull, ¬Greek-vs-Greek-hull
  (the infiltrators).
- Stacked: [≤0.5σ, 0.5–1σ, 1–2σ, 2–3σ, >3σ].
- 2 panels (E, U).
- Annotate the absolute count within 1σ on top of each bar
  (Greek: out of 1,494; ¬Greek-into-Greek: out of 126,990).

**Figure 6b: infiltrator-source breakdown**
- For the ¬Greek-into-Greek-1σ count, stacked bar showing source
  group composition (English-baseline / Cyrillic / German /
  French / CJK / structural). Tells us *which* non-Greek
  populations cross into Greek's hull most.

### 4.4 Spectrum (Marchenko-Pastur)

**Figure 7: per-group scree on log-y, MP edge overlaid**
- 4-panel figure (Greek×E, Greek×U, ¬Greek×E, ¬Greek×U).
- x = PC index, y = eigenvalue (log scale).
- Horizontal line at `λ_+` (MP upper edge). Annotate K_sig.

**Figure 8: cumulative variance per group**
- Two curves overlaid (Greek, ¬Greek), 2 panels (E, U).
- Reference lines at 50% / 90% / 95% variance.
- K_sig marker dot on each curve.

### 4.5 Shape summary

**Figure 9: PR + κ comparison bar chart**
- 2-panel figure: PR (left), κ (right).
- 4 bars each: Greek/E, Greek/U, ¬Greek/E, ¬Greek/U.

### 4.6 Linear classifier summary (binary)

**Figure 10: binary Greek-vs-¬Greek classifier summary**
- 2 panels.
- Left: Greek F1 / precision / recall on E and on U (grouped bars).
- Right: `cos(classifier weight, μ_Greek − μ_¬Greek)` and
  `cos(classifier weight, top-1 PC of Greek)` on E and on U.

### 4.7 Infiltrator gallery (NEW — companion to §3.9)

**Figure 11: top-20 infiltrators table-as-figure**
- Two panels (E, U), each a rendered table.
- Columns: rank, token-id, raw_token, decoded_text, source group
  (English-baseline / Cyrillic / …), m_Greek, dominant PC.
- Render with matplotlib's `table()` for vector output, or
  emit as a markdown table + image.

### 4.8 Cross-plot consistency

- Palette: Greek = `#1f77b4` (blue), ¬Greek = `#ff7f0e` (orange).
  Source-group sub-shades for Figure 6b use `tab10`.
- Same x-axis ranges for matching Greek-side / ¬Greek-side
  distance plots.
- All plots saved at 150 dpi PNG + PDF.

---

## 5. Implementation plan

### 5.1 New scripts

- `scripts/build_groups_greek_vs_not.py`
  - Loads `base_greek_tokens.jsonl` (strict 1,494) + the Phase A v2
    classification (for the "classified subset" mask and the per-
    source-group labels used in §3.9 infiltrator breakdown).
  - Emits `geometry/groups_greek_vs_not.json`:
    `{"Greek": [...], "not_Greek": [...], "source_group_of_negreek": {...}}`.
- `scripts/phase0_greek_vs_not_geometry.py`
  - Computes §3.1–§3.6 for both groups + §3.9 infiltrators.
  - Single pass over (E, U) — no per-language fanout needed.
- `scripts/render_diagnostic_plots_greek.py`
  - Reads the artefacts + cross-group summary, produces
    Figures 1–11 in `figures/`.

### 5.2 Refactored scripts

- `scripts/phase0_2_7_10_linear_classifier.py` — repurpose as
  **binary Greek vs ¬Greek** classifier; keep the saga setup, drop
  the 7-class multinomial labelling.

### 5.3 Run order

1. (Once) `extract_embeddings.py` — already done.
2. (Once) `recompute_strict_greek.py` — running; produces the
   strict-Greek centroid + PCs the new script needs. E side done;
   U side fitting (~30 min remaining).
3. `build_groups_greek_vs_not.py` — emits the canonical groups file.
4. `phase0_greek_vs_not_geometry.py` — produces all §3.1–§3.6 + §3.9
   artefacts. Critical bits:
   - Computes the **full** spectrum for Greek (n=1,494 ≪ d so the
     spectrum has rank ≤ 1,493 — fast, ~30 s per matrix).
   - For ¬Greek (n=126,990 ≫ d): the **full** d=4,096 spectrum via
     eigendecomposition of `Cov = X^T X / n` (~30 s per matrix).
     **DO NOT** use a truncated randomized_svd for the spectrum
     statistics — it biases sigma_sq high, MP_edge high, K_significant
     low, and total_var low; PR and κ depend on Σλ and Σλ² which
     need the full spectrum. v2.3 supplies this via
     `phase0_full_negreek_spectrum.py`; the original
     `phase0_greek_vs_not_geometry.py` truncated to top-500 and
     produced biased stats that were later overwritten.
   - For §3.9 infiltrators: project all 126,990 ¬Greek rows onto
     Greek's K_sig leading eigenvectors, compute Mahalanobis,
     sort, take top-20. ~5 s per matrix.
5. `phase0_2_7_10_linear_classifier.py` — binary; ~30 min per
   matrix on saga.
6. `render_diagnostic_plots_greek.py` — produces all figures.
7. Write `report_v2.md` consuming the cross-group summary +
   figures.

### 5.4 Cost

All home CPU. Estimated wall-clock with the reduced scope:

| step | wall | notes |
|---|---|---|
| build_groups_greek_vs_not | <30 s | trivial |
| phase0_greek_vs_not_geometry (E + U) | ~10–15 min total | Greek spectrum is fast (small n); ¬Greek SVD is the longest piece |
| linear_classifier binary (E + U) | ~60 min | binary is faster than 7-class on saga |
| render_diagnostic_plots_greek | ~5 min | matplotlib only |
| report writeup | ~1 h | manual |

Total: **~1.5–2 h on home CPU + ~1 h writeup. $0 GPU.**

Roughly **3× faster** than the original 7-language version, because
we skip 5 of the 7 per-group spectrum computations and drop a 7-way
classifier to binary.

### 5.5 Sanity checks before publishing

- For each language, the centroid + per-token distances should
  satisfy: `median(d_t) ≈ √(2 × trace(Cov_g) × n_g / (n_g + 1))`
  (Gaussian-cloud sanity).
- MP edge for English-baseline (n = 73,910 ≫ d = 4096) should be
  the classical square-edge formula; for Greek (n = 1,494 < d)
  should use the rectangular MP with n_g < d branch.
- Linear classifier F1 should be reproducible against the existing
  v4 run within ±0.005 macro F1 (no methodology change beyond
  strict-Greek).
- All figures must use the same per-language colour mapping
  (literal palette dict at the top of `render_diagnostic_plots.py`).

---

## 6. Out of scope for v2

- Init benchmark of any kind.
- Mini-CPT / CPT-validation.
- Cross-script NN map / Greek-clean vs Greek-mixed contrast.
- Per-layer hidden-state probe (§2.7.9 in v4).
- Per-dimension specialisation / per-token specialist-dim ranking
  (§2.7.7 in v4).
- Untrained-floor landmark (§2.7.8 in v4) — kept as a possible
  follow-up but not in v2 scope.

If init choice is needed later, the natural path is to start from
the strict-Greek centroid and the per-language MP-significant
subspace, and either (a) freeze-and-eval, or (b) skip directly to
a small CPT, rather than re-running an LOO benchmark.

---

## 7. Open questions for you

Two decisions left:

1. **Cleanup approval** — sign off on §1.1, §1.2, §1.3 moves, or
   call out anything you'd rather keep in-place? (Independent of the
   diagnostic — can do this before or after the run.)
2. **σ definition for hull occupancy** — confirm "1σ" means the
   standard deviation of the empirical Mahalanobis distances
   *within the Greek group* (not the chi-squared-based Gaussian
   threshold). Same σ is then used for §3.4 (Greek outliers) and
   §3.9 (¬Greek infiltrators) so the two plots are on the same axis.

Target-language set is settled (Greek + ¬Greek) — the per-language
split waits for your corpus-tokenisation classifier.

Once σ is confirmed, the v2 pipeline runs end-to-end in ~1.5-2 h
home CPU + ~1 h writeup.
