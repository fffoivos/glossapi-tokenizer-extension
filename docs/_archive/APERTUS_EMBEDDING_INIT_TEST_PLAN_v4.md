# Apertus embedding-initialization test plan

Date: 2026-05-12.

## 0. What this plan answers — two layers

This plan has two distinct layers with different scopes, different
costs, and different audiences. Keep them mentally separate.

### 0.1 Diagnostic layer (current goal, ~$1, ~2.5 h)

Phase 0 + §2.7 additions: **characterise Apertus-8B-2509's embedding
manifold via a small set of named contrasts** — Greek vs English-
baseline, Greek vs Cyrillic, Greek vs vocab-minus-Greek, Greek-clean
vs Greek-script-misclassified, plus within-Greek frequency
stratification (full list in §2.0.5). The deliverable is **a single
contrast table** (§6.1.2) where rows = geometric properties (tightness,
centroid distance, NN share, E↔U coupling, linear-classifier F1,
Bhattacharyya separation, etc.) and columns = groups. The table is
what answers "how is Greek different from other languages in
Apertus." Per-group descriptive blocks are drill-down for individual
surprising rows.

Reusable artefacts (per-group centroids, covariances, top-K PCs,
linear-classifier weight vectors, per-layer separation curve) live
under `geometry/` and feed any future language-extension work, not
just the C3 extension.

This is the current goal. Everything from §1 through §2.8 +
the §6.1 diagnostic write-up serves it. Cost: ~1 hour CPU on
`home` for §2.1–§2.8 + §2.7.10, plus ~20 min GPU (~$1) for the
§2.7.9 per-layer probe, plus ~1 hour for the §6.1 write-up. Total
~$1, ~2.5 h wall. See §7.1 for the detailed table.

### 0.2 Init-benchmark layer (scaffolded for later, ~$22, ~7 h)

§3 through §6.2: **rank candidate init methods for the C3 extension
on a behavioural NLL benchmark, plus a mini-CPT validation.** Useful
when the project commits to building the actual extension and needs
to defend the init choice. Reuses the diagnostic-layer artefacts
(centroids, covariances) but adds the LOO behavioural test (Phase
2), geometry-fit correlation (Phase 3), and mini-CPT validation
(§6.5).

This is scaffolded — written down so the methodology is reproducible
when the time comes — but **is not part of the current goal**. Every
section from §3 onward is marked accordingly. Skip those sections
unless you're planning to commit to running them. Cost if run: ~$22,
~7 h wall. See §7.2 for the detailed table.

### 0.3 Why both are in one doc

The two layers share infrastructure (the same model, the same
group classification, the same per-group centroids). Splitting the
diagnostic findings into one doc and the init benchmark into another
would force a future reader to read both anyway. Keeping them
together with a clear scope marker on each section is the right
trade-off — but a reader expecting a benchmark plan or a reader
expecting a diagnostic plan must know which layer they're reading.

The user-asked-for piece — testing centroid-family inits + the
anisotropic-distribution characterisation — splits across both
layers. The **diagnostic layer** computes the per-group centroids
(§2.2) and characterises their anisotropy + tightness (§2.7.5)
+ frequency stratification (§2.7.1) + cross-matrix structure
(§2.7.3); this answers "where would different centroid inits land
in Apertus's manifold". The **benchmark layer** then takes those
centroid candidates and ranks them on behavioural NLL (§3.2 → §4).
Both are real questions; only the first (diagnostic) is in the
current scope.

---

## 1. Architectural facts we're working with

- `vocab_size = 131,072`; `hidden_size = 4,096`.
- `tie_word_embeddings = False`. `E` and `U` are independent matrices.
  Their norm distributions, centroids, and PC structure differ
  empirically (Phase A v2): U medians sit ~25% below E medians, so
  per-matrix init is mandatory.
- The new tokens that C3 will add are predominantly Greek-script
  (see `C3_CUTOFF_REPORT.md`); a smaller fraction are
  `structural_non_linguistic` (table separators, punctuation runs,
  math symbols, etc.). Different new-token sub-populations almost
  certainly want different inits — Greek-script new tokens should be
  initialised to look like Greek-group existing tokens; structural
  new tokens should be initialised to look like structural existing
  tokens.

The plan therefore tracks **per-group** statistics throughout, not
just global ones.

### 1.5 Apertus design choices that shape interpretation

Three Apertus-specific facts are load-bearing for reading the
diagnostic outputs. Without each, a specific Phase 0 finding would
have a different meaning.

| Apertus choice | What it changes about how we read Phase 0 |
|---|---|
| `tie_word_embeddings = False` | E and U are independent learned matrices, not constrained to agree. The whole §2.7.3 cross-matrix block (per-token cosine, per-group centroid cosine, subspace angle) is meaningful *because* of this — on a tied-embedding model it would be trivially uninformative. |
| `qk_norm = True` (RMS norm on Q/K) | Attention reads E direction-sensitively but norm-insensitively. U is not normed before softmax, so U-norm directly affects logit magnitude. **Practical reading**: per-frequency-bin U-norm signal in §2.7.1 will be stronger than per-frequency-bin E-norm signal; treat U as "the confidence side", E as "the meaning side". |
| Tokenizer = Mistral-Nemo `tekken v3`, **inherited not co-trained** | Vocab structure was set by Mistral, not by Apertus's training data. Some slots are slack (`<SPECIAL_NNN>` reserved + byte-fallback rows). This is why §2.7.8 (untrained-floor landmark) is informative for *Apertus specifically* — it characterises how Apertus's training touched the inherited slack rows. Same diagnostic on a model with a co-trained tokenizer would be meaningless. |

Context (not load-bearing for the diagnostic layer, but worth knowing
when reading the cross-references): Apertus consumed ~3.1 B Greek
pretraining tokens (`APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`)
across 1,496 Greek-script vocab slots ⇒ ~2 M occurrences per slot;
pretraining was ~60% English + 40% multilingual; the optimizer is
AdEMAMix (matters only for the scaffolded §6.5 mini-CPT, not for
the diagnostic layer); Goldfish Loss masked ~0.3 T pretrain tokens.

---

## 2. Phase 0 — characterise the existing embedding manifold

### 2.0 What existing science predicts the diagnostic layer will find

Three published findings about decoder-LM embedding geometry are
directly testable by Phase 0 + §2.7. Each Phase 0 output should be
annotated **CONFIRMS** or **DIVERGES** vs the prediction. Divergences
are the Apertus-specific interpretation; confirmations are
calibration.

| literature claim | testable prediction in our data | which §2.7 subsection tests it |
|---|---|---|
| Mu & Viswanath 2018, *All-But-the-Top*: LM embeddings have a near-constant PC1 capturing no semantics (frequency artefact). | E and U each have a dominant PC1 (> 20% of total variance) that mostly disappears after subtracting `mu_global` row-wise. PC1 share post-centring < 10%. | §2.2 PC computation + §2.7.2 inline pitfall (force pre-centring) |
| Ethayarajh 2019, *How Contextual Are Contextualised Word Reps*: static embeddings are highly anisotropic (random-pair cosine 0.3–0.5). | Pre-centring random-pair cosine on classified tokens is in that range; post-centring drops to 0.05–0.15. | §2.1 anisotropy index, pre vs post centring |
| Gao et al. 2019, *Representation Degeneration*: low-frequency tokens drift into a narrow cone during training. | Within each group, low-freq tokens have HIGHER cosine to the group centroid (tighter cone) and lower U-norms; high-freq tokens are more spread. | §2.7.1 frequency-stratified diagnostics |

Other predictions from the multilingual / init-method literature
(Wendler 2024 middle-layer separation, ReTok / TokenAdapt init
rankings, Singh / Conneau cross-lingual centroid alignment) are
addressed by **the init-benchmark layer** (§3 onward) and the
optional per-layer probe (§2.7.9), not by the core diagnostic layer.
Listed there.

### 2.0.5 Primary contrasts the diagnostics feed

The numbers most §2.7.x diagnostics produce are not self-interpretable.
They acquire meaning only through specific comparisons. The plan's
primary deliverable is **a contrast table** (§6.1.2), not per-group
descriptive blocks. The contrasts that drive every §2.7.x diagnostic
are:

| contrast | what it isolates | what each direction supports |
|---|---|---|
| **Greek vs English-baseline** | cross-script effects (Greek = contrastive case; English-baseline = the high-resource reference) | Greek tighter/more-anisotropic ⇒ a script-specific representation; Greek matches English at matched frequency ⇒ no script-specific signature |
| **Greek vs Cyrillic** | language-specific effects net of script-family-general effects — Cyrillic is the closest analog to Greek in Apertus (similar non-English-alphabetic status, similar pretraining-share order of magnitude, similar morphological richness) | Greek differs from Cyrillic ⇒ language-specific representation, not just "non-Latin-script generic"; Greek ≈ Cyrillic ⇒ the model groups Greek-and-Cyrillic together as a non-Latin cluster |
| **Greek vs vocab-minus-Greek** | subspace coherence — is Greek geometrically distinguishable from the rest of the cloud? | clear separation ⇒ Apertus has learned a Greek subspace; Greek interleaved into the rest ⇒ Apertus represents Greek compositionally without a dedicated subspace |
| **Greek-clean vs Greek-script-misclassified** | script-boundary rigidity — does Apertus encode loanwords / mixed-script Greek differently from clean Greek? | clean-Greek separates from mixed-script-Greek ⇒ rigid script boundaries; they overlap ⇒ Apertus's "Greekness" is permissive at the script boundary |
| **Greek vs frequency-matched random subset** | size baseline — does any Greek-specific signature survive size-matching? (already in §2.7.5) | Greek tighter than random-matched ⇒ a real cluster; Greek ≈ random-matched ⇒ no group signature |
| **Within-Greek Q1 vs Q4 (frequency quartiles)** | frequency confound *within* Greek (rare-token cone effect) | Q4 tokens narrower / closer to centroid than Q1 ⇒ Gao 2019 degeneration effect is active in Greek as in English |

Each §2.7.x diagnostic below names which of these contrasts its
numbers feed. §6.1.2 publishes the contrast table as the
diagnostic-report's primary artifact; per-group descriptive findings
are drill-down for individual surprising rows.

A few subsections (§2.7.6 GMM, §2.7.7 per-dim participation, §2.7.8
untrained-floor) characterise Apertus on its own terms rather than via
contrast. They are kept because they answer Apertus-specific
questions (sub-structure within groups, specialist dimensions,
training-touch on inherited slack) that no inter-group comparison
exposes. They are flagged as "characterisation, not contrast" below.

CPU-only, runs on `home`. Reuses the saved `E_norms_all.npy` and
`U_norms_all.npy` from Phase A v2 plus the `token_classification.jsonl`
group labels. Need to additionally load `E` and `U` themselves
(4 × 131,072 × 4,096 bytes = ~2 GB each in fp32, ~4 GB peak). The
script can be the same `phase_a_diagnostic.py` skeleton extended.

**Note (v4 of this plan, after the contrast-framing critique):**
§2.1–§2.6 builds the per-group summary statistics (centroids, top-K
PCs, anisotropy) that every later contrast consumes. §2.7 reorganises
those statistics into the **contrasts named in §2.0.5** — Greek vs
English-baseline, Greek vs Cyrillic, Greek vs vocab-minus-Greek,
Greek-clean vs Greek-script-misclassified, within-Greek frequency
quartiles, plus characterisation-only diagnostics (§2.7.6 bimodality,
§2.7.7 per-dim participation, §2.7.8 untrained-floor) that don't fit
the contrast frame but answer Apertus-specific questions. §2.7.10
adds a linear language-classifier probe that gives a global-decision-
boundary view of Greek separability to complement §2.7.4's local-
neighbourhood view. §2.8 lists the methodological invariants that
must hold for any of the contrasts to be valid.

### 2.1 Per-matrix global summary

For each of E and U:
- `mu_global` = mean of all 131,072 rows. Shape `(4096,)`.
- `||mu_global||` (how far is the centroid from zero?).
- Mean per-row norm; std of per-row norm (already in Phase A v2 stats).
- Mean cosine between random row pairs (10k pair sample). If this is
  > 0.3, the cloud is strongly anisotropic / off-zero — common for
  trained transformer embeddings.

### 2.2 Per-group centroids and covariance

For each analyzed group from Phase A v2 — Greek, CJK, Cyrillic,
German, French, English-baseline, structural_non_linguistic — plus
"all-vocab":

- `mu_<group>` = mean of rows in that group (per matrix). Save as
  `arrays/centroids_E.npy` shape `(n_groups+1, 4096)` and
  `arrays/centroids_U.npy`.
- Top-`K=64` principal components of `(rows - mu_<group>)`, computed
  via randomised SVD (e.g. `sklearn.decomposition.TruncatedSVD` or
  `torch.svd_lowrank`). Save the `(K, 4096)` PC basis and the K
  singular values. K=64 keeps each per-group blob to ~1 MB.
- Trace of group covariance (`sum(s_i^2)`) — the within-group
  variance budget.
- `mu_<group>` distance to `mu_global` (the per-group displacement
  from the cloud's centre).
- Top-1 PC variance share = `s_1^2 / sum(s_i^2)` (a single-number
  anisotropy measure per group).

### 2.3 Cross-group geometry

- Pairwise cosine between group centroids — populates a
  `(n_groups, n_groups)` matrix per matrix. Tells us whether
  Greek-group and structural-group inits would have lived in similar
  or different parts of the manifold (likely different — that's the
  whole point of doing per-group centroids instead of one global one).
- Pairwise distance ratio: `||mu_<gA> - mu_<gB>|| / sqrt((trace_A +
  trace_B) / 2)`. Tells us how far apart group centroids are relative
  to within-group spread.

### 2.4 Manifold-distance scaffolding (used in Phase 3)

For each existing-vocab row `r` (used as test-target ground truth):
- Distance to its group centroid: `||r - mu_<r's group>||`.
- Mahalanobis distance under group covariance (computed in top-K PC
  basis to avoid inverting 4096×4096).
- 1-NN distance to nearest other existing embedding (10-NN search is
  cheap with FAISS or scikit-learn BallTree on a 131k × 4096 cloud).

These are baseline "manifold-fit" numbers — every init candidate is
later compared to these distributions.

### 2.5 Outputs

```
runs/apertus_embedding_init_test_20260512/
  geometry/
    centroids_E.npy           (n_groups+1, 4096)
    centroids_U.npy
    pc_basis_E_<group>.npy    (K, 4096) per group
    pc_singvals_E_<group>.npy (K,) per group
    same for U
    pair_cos_E.npy            (n_groups, n_groups)
    pair_cos_U.npy
    anisotropy_E.json         per-group top-1 PC share + trace + ||mu||
    anisotropy_U.json
  figures/
    pc_spectrum_<group>.png   bar plot, log-y, per group
    centroid_cosine_heatmap_E.png
    centroid_cosine_heatmap_U.png
    norm_vs_distance_to_group_centroid.png
```

### 2.6 Cost

~30 min on `home` (CPU); zero $.

### 2.7 Contrast-driven diagnostics

Each subsection below feeds one or more contrasts named in §2.0.5.
The numbers they produce are not meaningful as per-group values; they
become meaningful as Greek-vs-X deltas, frequency-quartile deltas, or
geometry-direction alignments. §6.1.2 is the publishing format — a
single table where rows are properties and columns are groups, so
every Greek-vs-X contrast is read off as a row, not assembled by the
reader from a per-group dump.

The diagnostics add roughly ~30 min of CPU work to Phase 0 plus the
§2.7.9 ~20 min GPU pass and the §2.7.10 ~5 min sklearn pass.
Methodological invariants from the old §2.8 are folded into each
relevant subsection where they bite specifically.

#### 2.7.1 Frequency stratification (the master confound — do this first)

**Primary contrasts this feeds:** (a) Within-Greek Q1 vs Q4
(frequency confound within group); (b) makes every other contrast
in §2.0.5 conditionable on matched frequency, so "Greek vs English"
becomes "Greek Q1 vs English Q1", etc.

Token frequency is the single strongest confound in any LM-embedding
analysis. High-freq vs low-freq tokens have systematically different
norms and directions for reasons unrelated to language identity (Gao
2019). Greek is systematically rarer than English-baseline (~3.1 B
vs ~8 T pretraining tokens), so a naïve "Greek vs English" geometric
comparison partly measures "rare vs frequent". **Compute frequency
quartiles first; condition every other diagnostic in §2.7.2 onward
on them.**

Recipe:
- Frequency proxy = per-token total `count` summed across the v4
  Phase B slices (already on disk; corpus-derived, but rank-order
  is stable). Tokens not appearing in any slice get count = 0 and
  group into Q4.
- Per group, partition tokens into 4 quartiles Q1..Q4 by frequency
  (Q1 = most-frequent within the group, Q4 = least).
- Re-run §2.2 stats PER QUARTILE: `mu_<group>_Qk`,
  `trace(Cov_<group>_Qk)`, top-K PCs per quartile.
- Tightness vs frequency-matched random subset: for each
  `<group>_Qk`, draw a random size-matched subset of classified
  tokens with frequency rank in the same global decile (not
  uniformly across all tokens — that would re-introduce the
  frequency confound on the comparison side). Compute its trace
  over 10 bootstrap resamples; report median.

**Pitfall (was §2.8 #4): norm/direction entanglement.** Whenever a
later §2.7.x subsection finds a directional pattern (cosine, PC
projection), re-run the diagnostic per-norm-quartile to check the
pattern survives. If it disappears, it was norm-confounded.

Output: `centroids_by_group_quartile_{E,U}.npy` shape
`(n_groups, 4, 4096)`; `tightness_index_by_group_quartile.json`.
Adds ~10 minutes.

#### 2.7.2 Cloud visualisation

**Primary contrasts this feeds:** Greek vs vocab-minus-Greek
(subspace coherence — visible directly in the PCA/UMAP plots);
Greek vs English-baseline (do the two groups occupy different
regions of the cloud?); Greek vs Cyrillic (do the two non-Latin
groups co-locate or separate?).

For E and U separately:

- **2D and 3D PCA projection of all 131,072 rows**, colored by Phase
  A v2 group + frequency quartile from §2.7.1. Save as
  `figures/pca2d_E_by_group.png`, `pca3d_E_by_group.png`, same for U.
  **Pitfall (was §2.8 #3): the PC1 of raw LM embeddings is a near-
  constant frequency artefact — subtract `mu_global` row-wise before
  computing the PC basis.** Otherwise PCA-2D collapses to a band along
  the artefact direction. The shape of the picture answers: does the
  model have a geometrically separated "Greek subspace", or is Greek
  interleaved into a more isotropic English-baseline blob?
- **UMAP projection** with the same colouring (5–10 min on 131k ×
  4096 in fp32 with `umap-learn`). Catches non-linear structure that
  PCA flattens.
- **Scatter `||E[t]||` vs `||U[t]||`** per token, colored by group.
  Save as `figures/norm_E_vs_norm_U_by_group.png`. A clean line ⇒ E
  and U norms are tightly coupled; a cloud ⇒ independent; per-group
  slopes ⇒ training imprinted group-specific norm relationships.
- **Cosine-to-group-centroid histograms** per group, separately for
  E and U. Save as `figures/cos_to_mu_<group>_E.png` and `_U.png`.
  Narrow unimodal ⇒ tight cluster; broad ⇒ fuzzy region; bimodal ⇒
  sub-structure worth investigating in §2.7.6.

#### 2.7.3 Cross-matrix (E↔U) comparison

**Primary contrast this feeds:** Greek vs English-baseline + Greek
vs Cyrillic, on a *new axis* — does Apertus encode Greek with
similar E↔U coupling to other languages, or is the input/output
binding different for Greek? Per-token cosines and per-group
centroid cosines are aggregated as Greek-vs-X deltas in the §6.1.2
contrast table.

Apertus's untied design means E and U are independent learned
representations of the same vocab. With `tie_word_embeddings = False`
they are not constrained to agree. Things worth measuring:

- **Per-token cosine `cos(E[t], U[t])`** for every classified token.
  Histogram per group. High ⇒ E and U encode similar information per
  token; low ⇒ they encode complementary aspects. Output:
  `e_vs_u_cosine_per_token.npy` (length 131072 in token-id order),
  plus per-group histogram figures.
- **Per-group centroid cosine `cos(mu_E_<group>, mu_U_<group>)`.**
  Save as `e_vs_u_centroid_cosine_per_group.json`. Tells us whether
  "Greek direction" is a coherent concept across both matrices.
- **Subspace angle between E and U top-K PC bases**, per group.
  Principal-angles vector (length K) — small angles ⇒ E and U
  manifolds span nearly the same subspace; large angles ⇒ they live
  in geometrically different subspaces. Standard `numpy.linalg.svd`
  on `U_basis.T @ E_basis` returns these. Save as
  `e_u_subspace_angles_<group>.npy`.

#### 2.7.4 Cross-script nearest-neighbour map

**Primary contrasts this feeds:** Greek vs English-baseline + Greek
vs Cyrillic on in-group-NN-share — the headline number is "Greek
has N% in-group nearest neighbours vs English-baseline M%"; **Greek
vs vocab-minus-Greek** (do Greek tokens predominantly retrieve other
Greek tokens, or do they retrieve compositionally similar tokens
from anywhere in the cloud?); **Greek-clean vs Greek-script-
misclassified** (do mixed-script / loanword tokens retrieve clean
Greek as neighbours, or do they retrieve their other-script company?
the contrast measures Apertus's script-boundary rigidity).

The pairwise-centroid-cosine heatmap (§2.3) shows centroid-level
similarity. The token-level question is whether individual tokens
have within-group or cross-group neighbours.

- For each group, sample 20 random members. For each, find the 10
  nearest E-neighbours among the full vocab (cosine or Euclidean —
  report both to be safe). Output:
  `knn_neighbourhood_E_<group>.json` with the within-group / out-of-
  group share per sampled token.
- Repeat for U.
- **In-group-NN-share contrast table**: collapse the per-token NN
  shares into per-group medians and publish as a Greek-vs-X column
  in the §6.1.2 contrast table. The exact deliverable is the median
  in-group-NN share for E and U separately, per group, across the
  20-token samples.
- **Greek-clean vs Greek-script-misclassified subcontrast**:
  identify tokens whose surface form contains at least one Greek
  codepoint but whose classifier label is *not* `greek` (mixed-script
  fragments, numeric-with-Greek, loanwords retokenised by the
  byte-level BPE). Call this the `greek_mixed` set; compute its in-
  group-NN share separately. The contrast is `greek` vs `greek_mixed`:
  if they share neighbours, script boundaries in Apertus are
  permissive; if not, Apertus rigidly compartmentalises Greek-only
  tokens from Greek-containing-but-mixed tokens. Output:
  `script_boundary_rigidity_greek.json` with the two NN-share
  distributions and a KS-test statistic.
- **Asymmetry detector**: tokens whose E-neighbours are mostly Greek
  but U-neighbours are mostly English (or vice versa). Output:
  `asymmetric_e_vs_u_membership_tokens.json`. The interesting cases
  for individual inspection; often reveals where the model encodes a
  token differently as "input" vs "what it predicts".

#### 2.7.5 Group tightness vs random-subset baseline

**Primary contrasts this feeds:** Greek vs frequency-matched random
subset (the size baseline — does Greek tighten compared to "any
1,500 tokens"?); Greek vs English-baseline at matched frequency
(comparable tightness indices, condition on Q1..Q4); Greek vs
Cyrillic at matched frequency (the language-specific contrast);
**Greek vs vocab-minus-Greek** (the cluster-coherence question
below).

`trace(Cov_<group>)` by itself is hard to interpret. Compare to:

- A random subset of `|<group>|` tokens drawn uniformly from the
  full classified vocab. Compute its `trace(Cov_random)` over 10
  bootstrap resamples; take the median. **Pitfall (was §2.8 #6):
  group sizes are wildly unequal (English-baseline ~74k, Greek
  ~1.5k); use pooled-σ-normalised distances when comparing across
  groups, not raw Euclidean.**
- **Tightness index** = `trace(Cov_<group>) / median(trace(Cov_random))`.
  < 1 ⇒ this group is a tight cluster (the model compressed it).
  ≈ 1 ⇒ as spread as a random size-matched subset (no group
  signature). > 1 ⇒ over-spread (the model *separated* group
  members).
- Re-run per frequency quartile from §2.7.1 to disentangle "Greek
  is tight" from "rare tokens are tight".
- **Greek vs vocab-minus-Greek (subspace coherence)**: compute the
  Bhattacharyya distance (Gaussian approximation in the top-K PC
  basis) between Greek and the rest of the classified vocab. Then
  compute the same Bhattacharyya distance between Cyrillic and the
  rest of the classified vocab, and between English-baseline and
  the rest. Three numbers; the Greek number is interpretable only
  by comparison to the other two. Headline: "Apertus separates
  Greek from non-Greek by Bhattacharyya D = X; for comparison, it
  separates Cyrillic from non-Cyrillic by D = Y, English-baseline
  from non-English-baseline by D = Z." Output: `subspace_coherence_per_group.json`.
- Output: `tightness_index_per_group.json` and
  `tightness_index_per_group_quartile.json` — one number per
  (group, matrix, [quartile]).

#### 2.7.6 Subgroup detection — bimodality check (exploratory)

**Characterisation, not contrast** — this asks "does Greek (or any
group) have hidden sub-structure that a single centroid would
misrepresent?" rather than a cross-group comparison. The output
feeds the §6.1.2 contrast table only as a per-group BIC delta
column ("how unimodal is each group's cloud?"), not as a Greek-vs-X
delta.

For each group, fit a 1-component and a 2-component Gaussian mixture
in the per-group top-K PC subspace and compare BIC. Where 2-component
wins by ≥ 10 BIC, the group has real sub-structure: the §2.2 centroid
sits in a hole between two modes and is a poor single representative.

**Methodology caveat**: fitting the GMM in the top-K PC subspace
(not full 4096-D) is a tractability choice. K = 64 captures most of
the within-group variance, but a bimodal direction that lies
orthogonal to the top-K subspace will be missed by this test.
Expected likely positives: English-baseline (function words vs
content words separable) and structural_non_linguistic (punctuation
runs vs single symbols). Other groups: less expected; if a 2-mode
split shows up there, treat it as a real finding worth examining
individually.

This is **exploratory** — flag where the centroid is unreliable as a
group summary, document the modes if found. The diagnostic layer
stops there; what to do with the finding (e.g. switch from group
centroid to per-mode centroid as an init reference) is a benchmark-
layer question and is out of scope here.

Output: `gmm_bic_per_group.json` with BIC delta + per-mode means
and weights where applicable.

#### 2.7.7 Per-dimension participation

**Characterisation, not contrast** — asks "does Apertus have
specialist dimensions, and what does each one encode?" rather than
a Greek-vs-X comparison. Feeds the §6.1.2 contrast table only if
specialist dimensions turn out to be group-specific (e.g. a "Greek
dimension"); in that case the dimension's top tokens become an
exhibit alongside the table.

For each of the 4096 embedding dimensions:

- Fraction of classified tokens with `|E[t, d]| > τ · median(|E[:, d]|)`
  for `τ ∈ {1, 3, 10}`. Per-dim "participation" share.
- Histogram of these participation shares across all 4096 dims.
  Save `figures/per_dim_participation_E.png`. A uniform distribution
  ⇒ dimensions are evenly used; a heavy tail of low-participation
  dims ⇒ specialist dimensions exist.
- For the bottom-10 lowest-participation dims, list the top-50 tokens
  by `|E[t, d]|` — does each "specialist dim" correspond to a
  recognisable subcategory (e.g. "Greek dimension", "math dimension",
  "URL dimension")? Output: `specialist_dims_E_tokens.json`.
- Repeat for U.

#### 2.7.8 Untrained-floor landmark — how Apertus touched the inherited-tokenizer slack rows

**Characterisation, not contrast** — Apertus-specific question
about what training did to the inherited-Mistral-tokenizer slack
rows. Not part of the Greek-vs-X axis but is a known landmark
direction that any §2.7.x contrast pattern (Greek centroid, Cyrillic
centroid, vocab-minus-Greek mean) can be compared against to check
whether a "Greek direction" is in fact a "drift-away-from-untrained
direction" in disguise.

Apertus's tokenizer was inherited from Mistral-Nemo `tekken v3` (see
§1.5), so not every vocab slot was actually exercised by Apertus's
training data. The bottom-100 rows by `‖U‖` (Phase A measured median
0.4566) are mostly `<SPECIAL_NNN>` reserved slots + mojibake. Where
they sit *directionally* is Apertus-specific: it characterises how
Apertus's training treated the inherited slack — stayed at the init,
drifted toward the cloud centre, or biased toward a particular
group.

- `‖mu_untrained_E‖` and `‖mu_untrained_U‖` — if near zero,
  untrained rows are effectively untouched; if large, training
  drifted them.
- `cos(mu_untrained, mu_global)` — drift toward the centre, or
  orthogonal (held at random init)?
- `cos(mu_untrained, mu_<group>)` per group — drift bias toward
  any particular language?

Output: `untrained_floor_geometry.json`. ~5 minutes; included
because it characterises an Apertus-specific aspect (inherited-
tokenizer slack) that nothing else in the plan covers.

#### 2.7.9 Per-layer Greek-subspace probe (committed in v3 — was optional)

**Primary contrast this feeds:** Greek vs English-baseline as a
function of depth — the contrast is between two language groups
measured at every transformer layer, not just at layer 0. The
deliverable is one curve in `figures/per_layer_group_separation.png`
plus the Wendler-2024 prediction overlaid; the "where does
Greek-ness live" answer is the layer where the curve peaks.

Embeddings are at layer 0; LM head reads layer 32. Where does
"Greek-ness" live in between? Run a forward pass on a small Greek
sample (~200 KB text) and capture hidden states at every layer for
every position. Per layer:

- Compute the per-group centroid of hidden states at positions where
  the target token belongs to that group.
- Pooled-σ-normalised distance between `mu_Greek_layer_l` and
  `mu_English-baseline_layer_l`. Plot as function of `l`.

Literature prediction (Wendler et al. 2024, *Do Llamas Work in
English*): per-layer language separation peaks in early-to-middle
layers and merges toward the top — the model maintains a language-
specific surface representation early and converges to a language-
agnostic semantic representation near the LM head. **For Apertus
specifically** the prediction is that peak Greek/English separation
sits around layer 10–20 of 32.

The diagnostic value: it tells us at which depth Apertus's "language
identity" representation lives, which is the substrate any
language-extension work has to interact with.

- Cost: ~20 min on the GPU instance (small text + hidden-state
  capture). **Committed in v3** (was optional in v2) because the
  literature prediction is sharp and the cost is low.
- Output: `figures/per_layer_group_separation.png` +
  `per_layer_group_distances.json`.

#### 2.7.10 Linear language-classifier probe

**Primary contrast this feeds:** Greek vs every-other-group at the
**linear-separability** level — does a single hyperplane in E (or
U) cleanly cut Greek tokens away from the rest, or is the boundary
non-linear / fuzzy? Complements §2.7.4's NN-share (local-neighbourhood
view) with a global-decision-boundary view.

Cheap, contrastive by construction, and isolates a "language vector"
subspace directly:

- Fit a multinomial logistic regression on E (and separately on U):
  features = the per-token embedding row, label = the Phase-A v2
  group classification (`greek`, `english_baseline`, `cyrillic`,
  `german`, `french`, `cjk`, `structural_non_linguistic`).
  Stratified train/test split (80/20). sklearn
  `LogisticRegression(multi_class='multinomial', max_iter=2000)`.
- **Per-group classifier accuracy and F1** — a Greek-row is the
  headline; high Greek-F1 ⇒ Greek is linearly separable in E (or U);
  low Greek-F1 ⇒ Greek is non-linearly distributed.
- **Per-group confusion matrix** — what does Greek get *confused with*?
  Cyrillic confusion is the diagnostic case (script-family-general
  effect); English-baseline confusion would be surprising and
  warrant individual-token inspection. Output: `linear_classifier_confusion.json`.
- **"Greek direction"**: the Greek logistic-regression weight vector,
  L2-normalised. Compare via cosine to: `mu_<greek> - mu_<global>`
  (the centroid-displacement direction); the top-K PC directions of
  Greek; the "drift-away-from-untrained" direction from §2.7.8. If
  the classifier's Greek direction aligns with `mu_<greek> -
  mu_<global>`, then the centroid displacement *is* the language
  identity direction; if not, language identity is encoded somewhere
  else and the centroid is misleading. Output:
  `language_direction_alignment_greek.json`.
- Run also on Cyrillic and on English-baseline for the Greek-vs-X
  comparison; the contrast statement in §6.1.2 is "Greek-F1 = X vs
  Cyrillic-F1 = Y vs English-baseline-F1 = Z; Greek confuses most
  with W".

Cost: ~5 min sklearn on `home` CPU. Output: `linear_classifier_E.json`
and `_U.json` with per-group accuracy/F1/precision/recall + confusion
matrix + weight-vector alignment metrics.

### 2.8 Remaining invariants (not folded inline)

Most pitfalls are now folded into the §2.7.x subsection they affect —
PC1 frequency artefact into §2.7.2 (cloud visualisation), norm/direction
entanglement into §2.7.1 (frequency stratification), group-size
inequality into §2.7.5 (tightness baseline), and centroid-in-a-hole
into §2.7.6 (GMM bimodality). What remains are three numerics-level
invariants that apply to **every** §2.7.x subsection rather than to
any specific one, so they live here instead of being repeated three
times:

1. **Cosine sample size + statistic.** When summarising "is the cloud
   anisotropic" via random-pair cosine, sample **10k pairs uniformly
   across the classified-token subset, not across the full 131,072
   rows**, and **report the median, not the mean** (cosine
   distributions are right-skewed). Re-sample 10 times and check the
   estimate is stable; if not, bump to 100k pairs.
2. **`<SPECIAL_NNN>` exclusion at the global-stats level.** Phase A
   v2's regex catches these in the per-group classifier, but
   `mu_global`, `trace(Cov_global)`, and random-pair cosine should
   be computed on the **classified-token subset only** (drop
   `special`, `byte_fragment`, `whitespace_only`, `digits_only`).
   Otherwise the untrained tail pulls `mu_global` toward 0 and
   inflates global variance with near-zero noise rows.
3. **bf16 → fp32 cast before SVD / covariance.** Apertus ships in
   bf16. **Cast `E` and `U` to fp32 before any SVD / covariance /
   PCA**. bf16's 7-bit mantissa makes low-magnitude singular values
   unreliable, which would in turn corrupt the top-K PC basis and
   every Mahalanobis distance downstream.

These need to be implemented as explicit invariants in the Phase 0
script, not "things to remember". Each one should fail loudly (or
emit a warning to the report) if the upstream data violates them.

---

> ⚠️ **§3 through §6.2 are the init-benchmark layer — scaffolded for
> later, not the current goal.** Per §0.2: the current goal is the
> diagnostic layer (§1–§2.8 + §6.1 diagnostic write-up). The
> sections below define the LOO behavioural benchmark, the geometry-
> fit correlation, the method catalog, and the mini-CPT validation
> that together rank candidate inits for the C3 extension. They are
> written down here so the methodology is reproducible when the
> project commits to building the actual extension, but they are
> **not part of the work being executed now**. Skip these sections
> unless you're planning to commit to running them; come back when
> the diagnostic-layer findings motivate the benchmark.

## 3. Phase 1 — define the candidate init methods (init-benchmark layer)

A small, defensible catalog. Each method is a function `init(token_id,
surface_text, base_E, base_U, geometry) → (e_init, u_init)` returning
the two vectors. All methods get a uniform interface so they can be
plugged into Phase 2.

### 3.1 Baselines

- **Z**: zero init. `e_init = 0; u_init = 0`. Sanity floor.
- **N(0,σ²)**: isotropic Gaussian with σ matched to existing-vocab
  per-dim std. Random-init reference.
- **Norm-only scrambled**: random unit vector scaled to existing-group
  median norm. Norm right, direction wrong.

### 3.2 Centroid family (the user's "mean init" question)

- **C-global**: `e_init = mu_global_E`, `u_init = mu_global_U`. One
  vector for every new token. Tests whether the global cloud centre
  alone is enough.
- **C-group**: per-token, pick the group it belongs to (Greek for
  Greek-script tokens, structural for punct/symbol-only tokens, etc.).
  `e_init = mu_<group>_E`. Tests whether group-level centroids beat
  the global one.
- **C-group + norm-match**: same direction as C-group, but rescale
  to the per-group median norm. Tests whether direction (centroid)
  alone suffices once magnitudes are corrected.
- **C-group + Gaussian noise**: `mu_<group>` plus small noise drawn
  from `N(0, ε² · Cov_<group>_topK)` for small ε. Breaks the
  degeneracy that every new token in a group would otherwise share an
  identical init.

### 3.3 Anisotropy-matched random

- **A-aniso**: sample from `N(mu_<group>, Cov_<group>_topK)` per
  token. Matches first and second moments of the existing group
  cloud. Each new token gets a unique vector that respects the
  cloud's shape.
- **A-PCs-only**: sample in the K-dim PC subspace of the group,
  project back to 4096-D. Forces the init to live in the
  high-variance subspace.

### 3.4 Constituent-based (existing literature reference candidates)

- **R1**: ReTok merge-order averaging (Option A1 in the feedback doc).
  For each new token `T = merge(A, B)`, `e_init = norm_match((E[A] +
  E[B]) / 2)`, where A or B may already be a previously-initialised
  new token. Chained.
- **R2**: base-piece retokenisation averaging (Option A2). For each
  new token `T` with surface text `s`, retokenise `s` with the **base
  Apertus tokenizer**, average those base embeddings, norm-match.
  Avoids R1's error compounding.
- **R2 + group-norm-match**: R2 directional, but rescale to the per-
  group median norm rather than the global one.

### 3.5 Model-internal contextual (Option C)

- **CTX**: run base Apertus over Greek text; for each new token's
  surface, find positions where the base tokenizer would have
  produced the constituent base-pieces; average the **last-layer
  hidden state** of those constituents in context; use that as the
  init (plus a norm match to the target group). Expensive (one
  base-Apertus forward pass over ~10 MB Greek text, cacheable across
  candidate sets).

### 3.6 Hybrid

- **R2-CG**: take R2's direction (base-piece average) and shrink it
  toward the group centroid by some factor `α ∈ [0, 1]`. `α = 0`
  recovers R2; `α = 1` recovers C-group. Tests whether a blend wins.

### 3.7 Cost

All algebra on home; per-method computation for the LOO set runs in
seconds. CTX is the exception — it needs a single GPU forward pass.

---

## 4. Phase 2 — leave-one-out behavioural evaluation (init-benchmark layer)

This is the load-bearing test. Embedding inits are a means to an end:
they should produce a token that, when slotted into Apertus's
otherwise-unchanged model, behaves close to a real trained token on
the LM-head + input-embed side.

### 4.1 Test set: held-out Greek tokens with ground truth

We don't have ground-truth embeddings for the *new* C3 tokens (they
don't exist yet in Apertus). But we **do** have ground-truth
embeddings for the existing 1,506 Greek tokens. **Treat each existing
Greek token as if it were new**: replace its embedding rows with a
candidate init, measure the behavioural NLL gap vs. the original
trained embedding. The init method that minimises this gap is the
one closest to "what trained-from-scratch would look like".

Selection criterion for the LOO set:
- Pick 100 existing Greek tokens with `count ≥ 50` in the Phase B v4
  `hplt_el` slice (so each has enough target positions to measure
  NLL on stably).
- Stratify by token-id frequency rank (so we test high-, mid-, and
  low-frequency Greek tokens).
- Save the chosen token-id list as `loo_target_ids.json` so all
  candidates are evaluated on the exact same tokens.

### 4.2 The eval corpus

Re-use `runs/apertus_greek_phase_b_v4_20260512/hplt_el.parquet`
(5,262 diversified Greek HPLT docs, ~3.85 M Apertus tokens). Already
known to be representative and held-out from any pretraining-related
contamination.

### 4.3 Two evaluation modes

**Mode A — U-only swap** (isolates LM-head side):
- For each candidate method `m`:
  - Construct `U' = U`, then for each LOO token `t`: `U'[t] =
    u_init_m(t)`.
  - Keep `E` unchanged.
  - Forward pass: input the corpus using *original* `E` (so context
    flow is unmolested); use `U'` to compute logits.
  - Per-position cross-entropy at positions where the target is one
    of the LOO tokens.
  - NLL_delta_m(t) = NLL_under_U'[t] − NLL_under_original_U[t].
    Lower is better; 0 means "init is indistinguishable from trained
    embedding in LM-head terms".

**Mode B — E + U swap** (deployment-realistic):
- Same as Mode A but also replace `E[t]` with `e_init_m(t)` for each
  LOO token. Now both the context-flowing and the prediction
  paths are using the init. NLL is measured at LOO-target positions.
- Mode B is what production deployment looks like (we replace both
  matrices for new tokens). Mode A isolates the LM-head signal so we
  can interpret deviations.

Mode A is mandatory; Mode B is the headline.

### 4.4 Forward-pass strategy — data-parallel across N A100s

Per `(m, mode)` pair (24 pairs total = 12 methods × 2 modes):
- Edit the model in-place:
  `model.get_output_embeddings().weight.data[loo_ids] = U'_swap_rows`.
  Same for `E`.
- Run the per-doc forward pass over the eval corpus
  (`hplt_el.parquet`).
- For each LOO token, accumulate `sum_loss` and `count` at positions
  where the target id is in `loo_target_ids`.
- Restore the original rows; emit the per-token NLL_delta dict.

**Parallelism**: each `(method, mode)` pair is independent of every
other pair. We distribute the 24 pairs across `N` A100-40GB GPUs on
a single `a2-highgpu-Ng` instance:

- Each worker process loads Apertus once (~16 GB bf16, fits on a
  40 GB A100).
- Each worker is assigned ⌈24/N⌉ pairs and runs them sequentially
  on its GPU (in-place weight edit + restore per pair).
- All workers read the same eval corpus from local disk read-only.
- Each worker writes `phase2_results/worker_<rank>.json`; the
  aggregator (§4.5) collects them.

Wall-time scales near-linearly because the only serial cost is one
~30 s model load per GPU and there's no inter-worker comm during
the run.

| GPUs | wall | $ (eur-w4 a2-highgpu-Ng) |
|---|---|---|
| 1 | ~3 h | ~$11 |
| 2 | ~90 min | ~$11 |
| 4 | ~45 min | ~$11 |

**Default for the run: `a2-highgpu-4g`** (4 × A100-40GB). Operational
recipe lives in [PHASE2_MULTI_GPU_EXECUTION_PLAN.md](PHASE2_MULTI_GPU_EXECUTION_PLAN.md)
— SKU procurement, file layout, orchestrator design, pre-flight
checks, stop-and-cleanup.

If cost is the constraint over wall-time: only run the top 4–6
most-promising methods in Mode B once Mode A has ranked them.

### 4.5 Aggregation

For each method `m` and mode:
- Median NLL_delta across the 100 LOO tokens (headline number).
- p25 / p75 / p95 of NLL_delta (variance across tokens).
- Per-token-frequency breakdown: does method m do better on
  high-frequency or low-frequency Greek tokens?
- Fraction of LOO tokens where method m beats the C-group baseline.

Save as `phase2_loo_results.json`.

### 4.6 Outputs

```
runs/apertus_embedding_init_test_20260512/
  loo/
    loo_target_ids.json
    nll_delta_per_method.json   {method: {mode: [list of 100 NLL_delta values]}}
    nll_delta_summary.json      {method: {mode: {median, p25, p75, p95, frac_better_than_C_group}}}
    figures/
      nll_delta_box_per_method.png
      nll_delta_vs_token_frequency.png
```

---

## 5. Phase 3 — embedding-geometry sanity checks (init-benchmark layer)

Cheap, deterministic, runs on home. For each candidate init for each
LOO token, compute:

- **Norm**: `||u_init|| / median_||U[Greek]||` — should be ≈ 1.
- **Cosine to group centroid**: `cos(u_init, mu_<Greek>)` — should be
  high (≥ 0.5) for any init that respects the group.
- **k-NN distance**: distance to the nearest existing-vocab embedding
  (excluding the LOO token itself). Tests whether the init lands on
  the manifold or off it.
- **Mahalanobis distance under group covariance** (top-K PC basis):
  the "shape-aware" distance to the group centre. Low = on manifold;
  high = outside the cloud.
- **Top-K PC variance fit**: project `u_init - mu_<group>` onto the
  group's top-K PC basis and compute the fraction of `||u_init -
  mu_<group>||²` captured. Should match the group's own top-K fit
  for ground-truth embeddings.

These metrics are computed for both `e_init` and `u_init`
independently, and saved as a `(n_methods, n_loo_tokens, n_metrics)`
array. The relationship between geometry (Phase 3) and behaviour
(Phase 2) is the diagnostic question: does a method that fits the
manifold also produce small NLL_delta? If yes, we can use cheap
geometric metrics to extend the analysis to *new* C3 tokens for which
we have no ground truth.

### 5.1 Outputs

```
  geometry_fit/
    per_token_metrics.json       {method: {token_id: {norm_ratio, cos_to_mu, knn_dist, mahalanobis, pc_fit_frac}}}
    correlation_phase2_vs_phase3.json   correlations between NLL_delta and each geometric metric
    figures/
      geom_vs_nll_scatter_<metric>.png
```

---

## 6. Phase 4 — write up

The report is split into **two halves that can be published
independently** — matching the §0 two-layer split. The diagnostic
half (§6.1) is the current goal and shippable on its own. The
init-recommendation half (§6.2) is scaffolded for later and is only
filled in if §3–§5 + §6.5 actually run.

Single markdown report at
`runs/apertus_embedding_init_test_20260512/report.md`. The report's
job is to make the result usable by a non-author — explicitly
prioritise interpretation aids over data dumps.

### 6.1 Diagnostic report (current-goal half — runnable independently)

This half publishes after Phase 0 + §2.7 alone. It describes what
Apertus's embedding manifold looks like; it makes **no recommendation
about init methods**. A reader interested only in characterising
Apertus can stop here.

#### 6.1.1 Executive summary (diagnostic)

One page max, at the top. **Every bullet is a contrast statement**
— a Greek-vs-X claim, not a per-group description. The reader
walks away with the model-understanding answer from the contrasts;
per-group numbers in isolation do not deliver that.

Template (target ~6 bullets, filled in after Phase 0 + §2.7 run):

- **Greek vs English-baseline (cross-script effect):** "Greek is N%
  [tighter / more anisotropic / less linearly separable] than
  English-baseline at matched frequency. The headline difference
  is [the specific property, e.g. in-group NN share]."
- **Greek vs Cyrillic (language-specific net of script-family):**
  "Greek's [property X] matches Cyrillic's within ε ⇒ this is a
  non-Latin-script-general property" OR "Greek's [property X]
  diverges from Cyrillic's by Δ ⇒ this is language-specific to
  Greek."
- **Greek vs vocab-minus-Greek (subspace coherence):** "Apertus
  separates Greek from non-Greek by Bhattacharyya D = X; for
  comparison, Cyrillic-vs-non-Cyrillic separates by Y, English-vs-
  non-English by Z. ⇒ Apertus [does / does not] have a dedicated
  Greek subspace."
- **Greek-clean vs Greek-script-misclassified (boundary rigidity):**
  "Clean-Greek and mixed-script-Greek [share / do not share]
  nearest-neighbour pools; KS = K. ⇒ Apertus's Greek boundary is
  [permissive / rigid] at the script edge."
- **E↔U coupling for Greek vs other groups:** "Greek's per-token
  E↔U cosine median is X, English-baseline's is Y, Cyrillic's is Z.
  ⇒ [interpret]."
- **Per-layer separation peak (Greek vs English):** "Greek/English
  separation in hidden space peaks at layer L (out of 32);
  Wendler 2024 predicted early-to-middle. ⇒ [confirms / diverges]."

Plus:
- **One picture** — the embedding-cloud PCA-2D coloured by group,
  centroids marked, frequency-quartile contours overlaid; Greek,
  Cyrillic, English-baseline highlighted; vocab-minus-Greek shown
  as a backdrop.
- **One paragraph** — what these contrasts collectively say about
  how Apertus encodes Greek relative to other languages and to its
  full vocabulary, without committing to a specific init method.

If a §6.1.1 bullet *can't* be phrased as a contrast statement, it
doesn't belong in the exec summary — move it to §6.1.2 drill-down
or to §6.1.4 token examples.

#### 6.1.2 The contrast table (primary artifact)

The primary deliverable of the diagnostic layer is **a single
contrast table**: rows = geometric properties, columns = groups.
Greek-vs-X is read off horizontally; the table is what answers "how
is Greek different from other languages in Apertus."

| property (row) | English-baseline | Greek | Cyrillic | German | French | CJK | structural |
|---|---|---|---|---|---|---|---|
| n_tokens | | | | | | | |
| tightness index vs random-matched (§2.7.5) | | | | | | | |
| tightness index vs random-matched, Q1 (freq-matched §2.7.1) | | | | | | | |
| tightness index vs random-matched, Q4 (freq-matched §2.7.1) | | | | | | | |
| centroid distance from `mu_global` (pooled-σ, §2.2 + §2.7.5 pitfall) | | | | | | | |
| median in-group NN share, E (§2.7.4) | | | | | | | |
| median in-group NN share, U (§2.7.4) | | | | | | | |
| median per-token E↔U cosine (§2.7.3) | | | | | | | |
| per-group centroid E↔U cosine (§2.7.3) | | | | | | | |
| top-1 PC variance share, post-centring (§2.2 + §2.7.2 pitfall) | | | | | | | |
| bimodality BIC delta, 2-comp − 1-comp (§2.7.6) | | | | | | | |
| Bhattacharyya D vs vocab-minus-`<group>` (§2.7.5) | | | | | | | |
| Q1 vs Q4 centroid distance (within-group, pooled-σ, §2.7.1) | | | | | | | |
| linear-classifier F1, E (§2.7.10) | | | | | | | |
| linear-classifier F1, U (§2.7.10) | | | | | | | |
| top confusion target (§2.7.10) | | | | | | | |

(Same table also produced for U where the property differs by matrix.)

The Greek column is the case of interest; the **English-baseline,
Cyrillic, German, French, CJK, structural** columns exist to make
the Greek number interpretable. The script-misclassified-Greek
contrast and the per-layer separation curve don't fit a single
column and live below the table as standalone exhibits.

After the table:

- **Annotation**: every Greek-row number that materially differs
  from English-baseline AND from Cyrillic gets a one-sentence
  "implication" footnote — what the difference says about how
  Apertus encodes Greek.
- **CONFIRMS / DIVERGES vs §2.0 literature predictions**: each row
  that maps onto a Mu/Viswanath, Ethayarajh, Gao, or Wendler
  prediction gets a marker. The §2.0 literature claims were
  per-group predictions, not contrast predictions, so this marker
  applies to whichever column the literature spoke about (PC1
  share is global; anisotropy is global-pre-centring; degeneration
  is within-group; per-layer separation is Greek-vs-English in
  §2.7.9).
- **Drill-down plots** (after the table, not before): 4–6 plots
  that visualise the most-surprising rows — PCA-2D coloured by
  group, UMAP, ‖E‖-vs-‖U‖ scatter, cosine-to-centroid histograms
  per group, per-layer separation curve from §2.7.9, linear-
  classifier confusion heatmap, script-boundary-rigidity histogram.
  Per-group descriptive blocks become drill-down for individual
  surprising rows in the table, not the main delivery format.

#### 6.1.3 What contradicts intuition

Whenever Phase 0 + §2.7 findings differ from the §2.0 literature
predictions, call it out in a dedicated subsection:

> **Expected:** PC1 dominates per-group variance even after
> centring on `mu_global`.
> **Observed:** PC1 share after centring is only 7% — Apertus's
> embedding cloud is more isotropic than typical decoder LMs in
> the literature. **Implication:** the all-but-the-top frequency-
> artefact is weaker here, perhaps because of Apertus's
> qk_norm-on-attention design. (Implication for init choice is
> deferred to §6.2 if that half is ever written.)

These divergences are the bits that make Apertus *specifically*
make sense to a reader who already knows the general literature.

#### 6.1.4 Token-level annotated examples

For each major Phase 0 + §2.7 claim, show 5–10 specific Greek (or
other-group) tokens with their text, frequency, group, position in
geometry (centroid distance, top-3 PC coords, nearest-neighbour
list). Show them as marked points on the PCA-2D plot. Concrete
examples > abstract claims; a reader who can name 10 specific
tokens that behave a specific way is more equipped than one given
only percentile tables.

#### 6.1.5 Honest limitations (diagnostic scope)

What the diagnostic-layer artefacts do NOT tell us:

**Architectural limits:**
- They describe E and U at the **input/output boundary**, not what
  happens to a representation inside the 32 transformer layers
  (the §2.7.9 probe is one small slice of that, not a full picture).
- They are a static snapshot; they say nothing about gradient
  landscape during CPT (which is what the §6.5 mini-CPT validation
  is for in the init-benchmark layer, if it ever runs).
- The group classification is byte-level surface-form heuristic; a
  token like `Α` (Greek capital alpha) appearing in formulae may be
  ambiguous between "Greek letter" and "math variable" and a single
  centroid hides that (the §2.7.4 Greek-clean vs Greek-script-
  misclassified contrast captures *some* of this but doesn't resolve
  the underlying ambiguity).

**Predictive-validity limits — what contrasts can and cannot predict:**
The contrasts in §2.0.5 measure **structural properties** of how
Apertus encodes Greek:
- *How distinct* — cluster separability (§2.7.5 Bhattacharyya,
  §2.7.10 linear-classifier F1).
- *How compact* — tightness vs neighbours (§2.7.5 tightness index).
- *How individuated* — within-group spread, E↔U coupling (§2.7.3,
  §2.7.7).
- *How generalisable* — frequency-conditioned geometry (§2.7.1).

These predict structural properties of Greek encoding in the model.
They **do not** predict downstream task accuracy on Greek
benchmarks. A model with a tight, separable Greek subspace can
still be bad at Greek QA; a model with Greek interleaved
compositionally can still be good at it. The contrasts are
about *representation geometry*, not *capability*. The diagnostic
layer's audience is "people who want to understand Apertus's
internal Greek representation"; it is **not** a model-quality
report.

#### 6.1.6 Reproducibility checklist (diagnostic)

- Model SHA: `swiss-ai/Apertus-8B-2509@<commit>`
- Dataset SHAs for the frequency-proxy inputs (v4 Phase B slices on disk).
- Random seeds (§2.7.6 GMM init, §2.7.1 frequency-quartile cut,
  §2.7.2 PCA / UMAP seeds).
- Library versions (torch, transformers, scikit-learn, umap-learn).
- Cost incurred (~$1 GPU for §2.7.9 if run; otherwise zero).
- Scripts checked in to `runs/apertus_embedding_init_test_20260512/`
  with their executed commands.

---

### 6.2 Init recommendation (init-benchmark half — scaffolded for later)

> ⚠️ **This half is only written if §3–§5 + §6.5 actually ran.** If
> the project stopped at the diagnostic layer, §6.2 is omitted and
> the report ships as a diagnostic-only document. The structure
> below documents what §6.2 should look like *when* the benchmark
> runs.

#### 6.2.1 Executive ranking (benchmark)

- **Headline ranking** — which init wins, by how much, with what 95% CI.
- **One picture** — the embedding cloud PCA-2D coloured by group,
  with the winning init's placement of representative new tokens
  marked.
- **One paragraph** — what this means for the C3 extension's
  shipping decision.

#### 6.2.2 Decision cards per method

One per method, uniform format. The reader scans these to pick:

```
NAME: <e.g. "R2 base-piece retokenisation average">
ALGORITHM: <one sentence>
GEOMETRIC SIGNATURE: <where it places new tokens — direction,
  norm, manifold position — in 2 sentences>
LOO NLL_delta (Mode B, median): <number ± p25–p75 range>
GEOMETRY FIT (median Mahalanobis to group): <number>
STRENGTHS: <bullet list, 3–5 items>
WEAKNESSES: <bullet list, 3–5 items>
USE FOR: <which token sub-populations — Greek-script / structural / both>
IMPLEMENTATION COMPLEXITY: <small / medium / large>
PRODUCTION RISK IF WRONG: <what fails if this is the chosen method
  and the test was misleading>
```

#### 6.2.3 LOO NLL results

Ranking table of median NLL_delta per method, Mode A and Mode B.
Per-frequency-quartile breakdowns (do methods differ in how well
they handle high-freq vs low-freq Greek tokens?). Best and worst
LOO tokens for the winning method, with surface text + group + freq
rank.

#### 6.2.4 Geometry-fit correlation

Scatter plots correlating each geometric metric (norm ratio,
cosine-to-centroid, k-NN distance, Mahalanobis, top-K PC fit) with
Phase 2's NLL_delta. If Spearman ρ > 0.6, the geometric metric
becomes a cheap surrogate we can use to extrapolate to actual C3
new tokens that don't have ground truth.

#### 6.2.5 Mini-CPT validation result

NLL-vs-step trajectory plot per top-3 init. Reads off: static-init
ranking (step 0) vs CPT-friendly ranking (step 500). If they
disagree, mini-CPT wins.

#### 6.2.6 Recommendation

Which method(s) to ship for the C3 extension, **per new-token
group** (Greek-script vs structural vs anything else C3 added).
With:
- Fallback method (in case the winner is hard to implement at
  25k-token scale).
- Hyperparameter notes (e.g. norm-match target, noise σ, blend α
  for hybrid methods).
- Pre-flight invariants the implementation must satisfy (norm in
  expected range, group-cosine ≥ X, etc.) so a downstream bug
  doesn't silently swap in a bad init.

#### 6.2.7 The honest limitation (benchmark scope)

LOO-on-existing-tokens measures "how close can each method get to a
*trained* embedding without seeing the training data". Real new C3
tokens won't have ground-truth embeddings — they'll start from the
chosen init and move during CPT. The init's job is to give CPT a
good starting point, not to nail the final embedding. So Phase 2's
"smallest NLL_delta" wins are an approximation of "best CPT starting
point" — not a proof.

Mitigation: §6.2.4's geometry-fit correlation is one independent
check; §6.2.5's mini-CPT is the gold-standard direct check. If all
three agree, ship. If they disagree, document and choose the most
behavioural (mini-CPT > LOO NLL > geometry-fit).

#### 6.2.8 Reproducibility checklist (benchmark)

Everything in §6.1.6 plus:
- LOO target-id list (`loo_target_ids.json`).
- Random seed for the §6.5 mini-CPT data ordering.
- GPU cost incurred per phase (for budgeting future repeats).
- AdEMAMix optimiser configuration used in §6.5.

---

## 6.5 [Recommended] Mini-CPT validation for the top-3 init methods (init-benchmark layer)

Phase 2's LOO measures **static** init quality — how close to a
trained embedding can each candidate get *without* moving. The
actual production question is **dynamic**: which init lets CPT
converge fastest to a usable embedding. A method that places new
tokens in a sub-optimal but recoverable location may beat one that
nails the static target if it leaves CPT a friendly gradient
landscape.

Bridging that gap with the gold-standard test:

1. Pick the **top 3 init methods** from Phase 2's LOO ranking (§6.2.3).
2. Pick **256 actual C3 added tokens** (random sample from the
   C3 glossary — `runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/`).
3. Construct the **256-token mini-extension**: resize Apertus's
   `E` and `U` by 256 rows; populate the new rows with each
   candidate method's init.
4. Run a **tiny CPT**: 500 optimisation steps on a small Greek
   mix (v4 `hplt_el` + a small `glossapi_el_modern` slice; mix
   it 70/30 web/academic). Optimizer = **AdEMAMix** to match
   Apertus's pretraining (not Adam — see §1.5). LR = Apertus's
   final-stage rate, then linear decay. Only the 256 new rows
   are trained; everything else is frozen.
5. Measure **per-token NLL on a held-out Greek slice every 50
   steps**. Plot NLL trajectory per init method.
6. Read off:
   - **NLL at step 0** = static init quality (re-validates Phase 2 LOO)
   - **NLL at step 500** = where CPT lands from this init
   - **Slope of NLL decay over steps 0..500** = how gradient-
     friendly the init was

The init that wins on **NLL at step 500** is the production choice.
If it matches the §6.2.3 LOO winner, the static ranking generalises
and we're done. If they disagree, the mini-CPT result wins and we
recommend it. Either way the doc records the test — the disagreement
case is more informative.

Cost: 500 steps × 256 new params × 3 methods on an A100 ≈ 2 hours
GPU + setup. Total ≈ $10. Wall ~2 h.

Optional in the headline plan; **mandatory before committing to a
production init choice for the C3 extension**.

---

## 7. Concrete workplan & cost

Two blocks, matching the §0 split. The diagnostic-layer block is the
current goal and stands on its own; the benchmark-layer block is
scaffolded and runs only if the project commits to it.

### 7.1 Diagnostic-layer cost (current goal)

| phase | work | hardware | wall | $ |
|---|---|---|---|---|
| Phase 0 (§2.1–§2.6) | characterise manifold (per-group centroids + top-K PCs + anisotropy) | home CPU | ~30 min | 0 |
| Phase 0 (§2.7.1–§2.7.8) | contrast-driven diagnostics (freq stratification + visualisation + E↔U + neighbourhoods + tightness + GMM-BIC + per-dim + untrained-floor) | home CPU | ~30 min | 0 |
| Phase 0 (§2.7.9, committed in v3) | per-layer Greek-subspace probe (Greek-vs-English by depth) | A100-40GB (already-stopped instance) | ~20 min | ~$1 |
| Phase 0 (§2.7.10) | linear language-classifier probe (Greek-vs-X linear separability) | home CPU | ~5 min | 0 |
| Diagnostic write-up (§6.1) | manifold characterisation report; CONFIRMS/DIVERGES annotation against §2.0 predictions | home | ~1 h | 0 |
| **diagnostic-layer total** | | | **~2.5 h** | **~$1** |

### 7.2 Init-benchmark-layer cost (scaffolded — only if committed)

| phase | work | hardware | wall | $ |
|---|---|---|---|---|
| Phase 1 | implement init catalog (≤ 12 methods); compute candidate vectors for 100 LOO tokens | home CPU | ~30 min | 0 |
| Phase 1.5 | CTX method one GPU forward pass to cache contextual hidden states for the LOO surface forms | `a2-highgpu-1g` (1× A100-40GB) | ~20 min | ~$1 |
| Phase 2 | LOO behavioural NLL eval; 12 methods × 2 modes = 24 independent forward passes, data-parallel across 4 GPUs | **`a2-highgpu-4g` (4× A100-40GB)** | ~45 min | ~$11 |
| Phase 3 | embedding-geometry sanity checks per (method, token) pair | home CPU | ~10 min | 0 |
| Phase 6.5 (mandatory before shipping) | mini-CPT validation: 500 steps × 3 top methods × 256 new tokens, AdEMAMix on a small Greek mix | `a2-highgpu-1g` (1× A100-40GB) — no multi-GPU benefit (sequential CPT) | ~2 h | ~$10 |
| Benchmark write-up (§6.2) | decision cards + ranking + recommendation | home | ~1 h | 0 |
| **init-benchmark-layer total** | | | **~7 h** | **~$22** |

### 7.3 Combined-run note

If both layers are run end-to-end, total is ~7.5 h / ~$23. The
GPU work runs across **two distinct gcloud SKUs**, not one warmed-up
instance:

- `a2-highgpu-1g` (1× A100) for §2.7.9, Phase 1.5, and Phase 6.5.
  These are single-pass / sequential and gain nothing from multi-GPU.
- `a2-highgpu-4g` (4× A100) for Phase 2 only. Spun up just for
  the ~45 min data-parallel run, then stopped.

Detailed multi-GPU recipe (instance procurement, orchestration,
file layout, pre-flight, cleanup) in
[PHASE2_MULTI_GPU_EXECUTION_PLAN.md](PHASE2_MULTI_GPU_EXECUTION_PLAN.md).

### 7.4 Sequencing

1. **Diagnostic layer (current):** Phase 0 §2.1–§2.6 on home →
   §2.7.1–§2.7.8 on home → §2.7.10 linear classifier on home →
   §2.7.9 per-layer probe on `a2-highgpu-1g` → §6.1 write-up
   (contrast table + drill-down plots). Stop here unless project
   commits to the benchmark.
2. **Init-benchmark layer (if committed):**
   1. Phase 1 init-catalog computation on home (uses Phase 0
      centroids / PCs / classifier weights).
   2. Phase 1.5 CTX-cache forward pass on `a2-highgpu-1g`
      (reuse the diagnostic-layer instance if still up; otherwise
      restart it). Stop the 1g instance when done.
   3. **Spin up `a2-highgpu-4g` → run Phase 2 (~45 min data-parallel)
      → stop and delete the 4g instance.** Cost ~$11 for the
      session.
   4. Restart `a2-highgpu-1g` → run Phase 6.5 mini-CPT → stop.
   5. Phase 3 geometry-fit checks on home.
   6. §6.2 write-up on home.

### 7.5 Pre-flight checks before kicking off the GPU work (benchmark layer)

- Confirm `loo_target_ids` is stable and stratified.
- Spot-check 2-3 candidate inits for each method by eyeballing
  (`||u_init|| in expected range, cos to mu_Greek > 0.3, etc.`) —
  catch implementation bugs before the GPU run.
- Confirm the in-place model.weight.data swap works without
  corrupting the model (round-trip: swap, swap back, NLL matches
  baseline). Quick CPU test on a 1-doc corpus.
- **Multi-GPU-specific pre-flight**: dry-run the orchestrator with
  N=4 workers on 2 (method × mode) pairs and 1 doc to confirm the
  per-worker NLL aggregation matches a single-worker reference.
  See [PHASE2_MULTI_GPU_EXECUTION_PLAN.md](PHASE2_MULTI_GPU_EXECUTION_PLAN.md)
  §6 for the exact dry-run command.

---

## 8. Open questions / extensions (init-benchmark layer)

These are NOT in the headline plan; they live in the
init-benchmark layer (§3 onward) and are worth flagging only when
the project commits to that layer:

- **Multi-token surface forms**: most C3 new tokens are 3-8 base-piece
  surface forms. The candidate-init machinery in Phase 1 handles this
  uniformly, but the LOO set should include tokens of varying length
  (already implicit in the frequency-stratified pick).
- **Group assignment for new structural tokens**: C3 adds ~150
  structural tokens (`table_separator`, `punctuation_run`, etc.).
  Phase 0 already builds the `structural_non_linguistic` centroid;
  this naturally extends.
- **E vs U asymmetry**: Phase 2 Mode A tells us only about U; Mode B
  conflates E and U. If we want to disentangle further: also run
  "E-only swap" — keep U at original, replace E[t]. Then NLL at
  positions *after* t reveals the input-side effect of the init.
  Mode A + Mode E together would isolate the two halves. Not in the
  default plan but a cheap follow-up if Mode B is ambiguous.
- **Full CPT convergence test**: after §6.5 picks a winner, do a
  larger CPT (1k–5k steps) starting from that init and from the
  next-best method, and compare the per-token NLL trajectories.
  Strongest test but its own subproject; outside this plan's scope.

---

## 9. Artifacts produced

Split by layer to match §0.

### 9.1 Diagnostic layer (current goal)

- A **manifold characterisation** under
  `runs/apertus_embedding_init_test_20260512/geometry/` —
  per-group centroids, top-K PC bases, anisotropy stats,
  E↔U cross-matrix metrics, frequency-stratified versions,
  cross-script neighbourhood maps, Greek-clean vs script-
  misclassified-Greek NN distributions, Bhattacharyya separations
  vs vocab-minus-`<group>`, GMM-BIC subgroup detection, per-dim
  participation, untrained-floor geometry, linear language-
  classifier confusion matrices and weight-vector alignment,
  per-layer Greek-subspace separation. Reusable for any future
  init / adaptation work — not just the C3 extension.
- **The contrast table** (`figures/contrast_table.{csv,md}` plus
  the rendered version in §6.1.2 of the report) — the primary
  diagnostic-layer deliverable: rows = geometric properties,
  columns = groups, every Greek-vs-X cell is interpretable
  horizontally.
- A **diagnostic report** (§6.1) that publishes alone: what
  Apertus's embedding manifold looks like, with CONFIRMS /
  DIVERGES annotations against the §2.0 literature predictions,
  built around the contrast table.

### 9.2 Init-benchmark layer (only if committed)

- A **candidate-init library** (`init_methods.py`) with all the
  methods in §3, plug-in interface, unit tests.
- A **LOO benchmark** (`phase2_loo_results.json`) ranking the 12
  methods on real behavioural NLL.
- A **geometry-fit benchmark** correlating cheap manifold metrics
  with the behavioural ranking.
- A **mini-CPT validation** (`phase65_minicpt_trajectories.json`)
  comparing the top-3 methods at step 0 vs step 500 of a small CPT.
- An **init recommendation** (§6.2) for the C3 extension's init
  method, per new-token group (Greek vs structural), with a
  documented fallback path if the winning method turns out to be
  hard to implement at scale (~25 k tokens).

---

## 10. What I'd ask before kicking off the diagnostic layer

These are the layer-relevant decisions for the current goal (§0.1).
Benchmark-layer questions (LOO set size, method count, CTX inclusion)
are deferred to whenever §3–§6.2 actually gets committed to.

- **Per-layer probe (§2.7.9) — run it?** v3 commits to running it
  because the literature prediction is sharp (Wendler 2024) and the
  cost is low (~20 min GPU, ~$1). The alternative is to keep it as
  optional and rely on E/U-only geometry from layer 0. Run by
  default unless you'd rather defer.
- **GMM-BIC subgroup detection (§2.7.6) — include in the headline
  diagnostic, or treat as exploratory only?** Current v4 framing is
  exploratory (flag if found, don't act). Doesn't fit the contrast
  table as a Greek-vs-X delta (it's characterisation, not contrast),
  but does appear as a per-group column. If you want a bimodality
  flag to appear in the §6.1.1 exec summary, say so and it gets
  promoted from "where the centroid is unreliable" footnote to a
  top-line claim.
- **Script-boundary contrast (§2.7.4 Greek-clean vs Greek-script-
  misclassified) — headline or footnote?** Default in v4 is one
  exec-summary bullet (script-boundary rigidity). If you'd rather
  treat it as a drill-down only and keep the §6.1.1 to language-
  level contrasts (Greek vs English, Greek vs Cyrillic, Greek vs
  vocab-minus-Greek), say so.
- **Linear classifier (§2.7.10) weight-vector visualisation —
  publish?** The classifier produces a "Greek direction" weight
  vector; visualising which embedding dimensions carry the most
  Greek-vs-rest weight is interpretable but adds ~10 min of plot
  work and a §6.1.4 token-examples exhibit. Skip unless you want
  the per-dimension interpretation in the report.
- **Diagnostic-report shape — include a glossary + takeaways list
  in §6.1.1?** A short "what these terms mean" block (anisotropy,
  PC1 artefact, pooled-σ, tightness index, Bhattacharyya
  separation, linear-classifier F1) plus a 5-bullet takeaways list
  would make §6.1 readable by someone who hasn't read the
  literature. Adds ~30 min of writing.

Default if you say "go" on the diagnostic layer: run §2.1–§2.8 +
§2.7.10 end-to-end on home + §2.7.9 on the already-stopped GPU
instance + §6.1 contrast-table-first write-up. ~2.5 h wall, ~$1
GPU.
