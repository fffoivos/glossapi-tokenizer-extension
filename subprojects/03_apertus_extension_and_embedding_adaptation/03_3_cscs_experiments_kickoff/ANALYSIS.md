# CSCS Experiments Kickoff — State Audit and Path Forward

*Drafted 2026-05-20. Author: Claude session driven by Foivos. Updated 2026-05-20 with the cutoff-decision artifact, the polytonic arm state, the sub-1B vocab-budget pattern, and the CSCS-clariden onboarding context.*

This is the analysis-before-rewrite. `experiments_plan.md` is **not
edited** here. Everything that needs to be reconciled into the plan
is parked in [§7 Review checkpoints](#7-review-checkpoints--what-still-needs-your-explicit-sign-off).

> **Update 2026-05-20**: four findings folded in after the first draft missed them:
> - **Cutoff is committed**: 17,408 (curated + backfilled) → total vocab 148,480 (256-aligned). See [`02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md). Review checkpoint A is **resolved**.
> - **Polytonic arm is a separate downstream track** with its own run already complete (+5,120 added → total 153,600 = 256 × 600). See [POLYTONIC_VOCAB_BUDGET_CHECK.md](POLYTONIC_VOCAB_BUDGET_CHECK.md) for the sub-1B-language-pattern verification of that budget.
> - **CSCS is already onboarded** for the user — account `a0140`, ssh aliases configured, 2-node 8-GPU DDP smoke test ran successfully 2026-03-28. See [`/home/foivos/Projects/cscs-clariden-project-understanding/README.md`](file:///home/foivos/Projects/cscs-clariden-project-understanding/README.md) for what's known about Clariden + the working launch pattern.
> - **CSCS daily key refresh is automated** via the new `cscs-key` Rust tool (the old `sshservice-cli/cscs-keygen.{sh,py}` is deprecated): `cscs-key sign --headless --duration 1d` (OIDC device-code flow; v1.1.0 of `cscs-key` caps duration at 1 d). Auth fully verified end-to-end 2026-05-20 (cert valid, `ssh ela` + `ssh clariden` both reachable) — see [CSCS_AUTH_WORKFLOW.md](CSCS_AUTH_WORKFLOW.md).

> **Constraint update 2026-05-20 (later)**: **we no longer have GCloud access.**
> Anything in earlier drafts that said "GCP scratch VM" or "the active
> gcloud worker" or "restart the terminated gcloud instance" needs an
> on-CSCS or on-home substitute. The two work items that materially
> depend on this are: (a) the CPT corpus build per
> [`CPT_DATASET_BUILD_RUNBOOK.md`](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md)
> — moves to Clariden `xfer` partition (256 vCPU, 500 GB RAM, 24 h —
> ample for the 14.4 M-doc pool); (b) the held-out contamination
> rerun ([Review checkpoint B](#7-review-checkpoints--what-still-needs-your-explicit-sign-off))
> — must either be reconstructed from the published nanochat corpus on
> HF by re-running the splitter with the original seed, or be left as
> a documented gap. See §1.4 finding 6 + §6 for per-item revisions.

---

## 1. Δ since the plan (2026-05-12 → 2026-05-20)

The plan's six decision nodes (§3) and ten open questions (§10) at
their current resolution status:

### 1.1 Resolved since 2026-05-12

| Plan reference | Status now | Source of resolution |
|---|---|---|
| §10 Q2 — "Phase 1 grounding step?" | already resolved at v0.10 (no) | Phase A norm diagnostic |
| §10 Q7 — Phase B behavioral cross-check | already resolved at v0.10 (done) | Phase B v4 NLL |
| **§2.6 / Constraint 2 — "Greek tokens are well-trained"** | **strengthened from "well-trained on average" to "Greek is a coherent geometric subspace with learned morphological structure"** | [03_1 diagnostic v2 report](../03_1_greek_embedding_diagnostic/artifacts/results/report_v2.md), 2026-05-13 |
| §6.4.4 — "HPLT MT / register concerns — partial" | corpus-side now closed for C3 train; eval-side measured | `hplt-greek-ge8-no-mt-clean60-wave4` snapshot; verified-virgin slice exists |
| §10 Q4 — CPT *corpus build path* | **resolved as a runbook** (corpus volumes are TBD; corpus *identity* and dedup are spec'd) | [CPT_DATASET_BUILD_RUNBOOK.md](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md), dedup audit r1–r4 |
| §3 Node 2 input — Apertus-overlap budgeting | **measured** (per-source overlap matrix at strict / relaxed / near MinHash) | [REPORT_dedup_20260519T010924Z.md](../03_2_apertus_c3_dedup_audit/REPORT_dedup_20260519T010924Z.md) |

### 1.2 Partial — informed by new evidence, not yet committed in writing

| Plan reference | New evidence | Why still partial |
|---|---|---|
| ~~§3 Node 1 / §10 Q1 — BPE cutoff~~ | **RESOLVED 2026-05-18**: cutoff = **17,408** (curated + backfilled), total vocab **148,480 = 256 × 580**. Picked from a 25-point sweep `{1024…25600}`, not from the older `{10240, 15360, 20480, 25600}` grid (which is therefore stale framing). 69 noise tokens structurally removed at build time; 69 valid merges backfilled from positions 17,409..17,477 to preserve alignment + append-only ids. 0 cascade-skips. Direct measurement showed the backfilled variant is marginally better than the raw cutoff. Ship artifact at [`variants/c3_added_17408_curated_padded/tokenizer.json`](../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/variants/c3_added_17408_curated_padded/tokenizer.json) (sha256 `358ae3f2…6adfc394`). Full decision write-up at [`CHOSEN_CUTOFF.md`](../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md). | `experiments_plan.md` v0.12 still describes the older grid and the 8K Pareto-lean argument. Plan diff needed: §2.4, §3 Node 1, §10 Q1, §11 worst-50 absorption check — all should be folded to reference 17,408 + the curation mechanism + the backfilled-padded variant. |
| §3 Node 2 / §8.5 / §10 Q5 — replay ratio | Apertus saw 2.22M / 98.2M HF-pool docs (2.27 %). Per-source fresh-share now quantified. Suggests "Apertus-Greek replay" is a tiny tail — most of the C3 train pool is fresh. | The 10–15 % non-Greek replay number in the plan is a different question (catastrophic-forgetting prophylaxis vs Apertus-overlap dedup). Both still need pre-commit. |
| §10 Q6 — eval suite slices | `hplt_el` + `glossapi_el_modern` exist (Phase B v4). `virgin_hplt` exists for tokenizer-side cutoff sweep. Cross-language regression slices (en/fr/de/ru) do not exist yet. Native Greek benchmarks (GreekMMLU, Belebele, Medical MCQA, OYXOY) are listed but not wired into a harness. | Has to be locked before CPT comparison fires. |
| **§3 Node 4 / §10 Q8 — pre-commit decision rule** | No new evidence shifts this, but the asymmetric M_ext / M_van structure is plausibly sharper now that we expect quality ties: §2.7's NLL ≈ 0.95 finding plus the diagnostic v2 finding (Greek is well-structured *and* per-token-well-predicted) together imply the comparison will be efficiency-dominated. | Values for X / M_progress / M_ext / M_van / T still suggested-only. Must lock before any arm completes CPT, per plan §3 Node 4 hard temporal constraint. → [Review checkpoint C](#7-review-checkpoints--what-still-needs-your-explicit-sign-off) |

### 1.3 Still open exactly as the plan stated

- §10 Q3 — HPLT polytonic/Katharevousa register supplementation. Confirmed out-of-scope for the *deployment register* but the question of whether to add a polytonic-aware track later is unresolved.
- §3 Node 6 — Krikri positioning. Still optional, late.
- §11 — worst-50 modern-Greek tokens vs C3 absorption check. Cheap script, hasn't been run.

### 1.4 New directions opened since the plan was written

These are findings that didn't exist when v0.12 was committed and would change wording in the plan if folded back in:

1. **Apertus has learned a Greek-axis** — direction of the centroid
   displacement `μ_Greek − μ_¬Greek` correlates at cos 0.85 (E) /
   0.92 (U) with the binary linear classifier's separating
   hyperplane. The "Greek-ness axis" is fully recoverable from the
   centroid offset. Macro F1 = 1.00 on U.
   Diagnostic v2 §2.4. **Implication for the plan:** the §2.6
   single-line "no representational gap" claim is now backed by a
   strong, multi-axis structural finding — the language axis is
   present and clean. Phase 1 grounding wasn't needed anyway, but
   this also rules out the "maybe there's a half-formed Greek
   subspace" hedge that some skeptics could have raised.

2. **Greek tokens cluster meaningfully by morphology** — `μέν*`
   passive participles 9.2× tighter than random Greek pairs; `ματ*`
   abstract noun endings 9.0×; `μεγ*` 7.1×; `αυτ-` 6.3×; `συν-` /
   `αρχ-` clusters visible at k-means k=8.
   Diagnostic v2 §5. **Implication:** Apertus already encodes Greek
   morphology compositionally in its existing 1,494 Greek tokens.
   This *softens* the ReTok-vs-Distillation case (Distillation's
   marginal value is largest when the base model's morphology
   knowledge is poor; here it's not) and *strengthens* the §2.7
   inference that the case for extension is purely compression
   economy, not quality lift.

3. **No Greek↔English etymological bridge in static embeddings.**
   `δημοκρατία ↔ democracy` cosine = +0.050; matches `freedom ↔
   ελευθερία` (+0.033). Greek-loanword cluster tightness is carried
   by the Latin-script-to-Latin-script pairs (en/fr/de/es/it), not
   by en↔el. Diagnostic v2 §8. **Implication:** any framing that
   says "Apertus already has Greek-via-English bridge so CPT just
   needs to fine-tune the surface" is wrong. The CPT corpus has to
   teach genuine Greek-internal structure (which is consistent with
   the plan's Constraint 1).

4. **Infiltrators result reframed.** What looked like "non-Greek
   tokens that look Greek-like to the model" turned out to be a
   high-D-shell + projection-asymmetry geometric effect; after
   floor-filtering, top "infiltrators" are under-trained-language
   content tokens (Telugu, Devanagari, Korean) or
   predictable-output sentence-boundary patterns. Diagnostic v2 §7b.
   **Implication:** clears a potential confound on whether Apertus
   has Greek-mixing in its existing embeddings. It doesn't.

5. **Per-source overlap with Apertus pretraining is quantified.**
   18 of 20 GlossAPI/HPLT-pool sources are `include_full` (fresh
   share ≥ 0.985). The two exceptions: `HuggingFaceFW/finewiki` at
   0.484 fresh (Wikipedia revision drift vs Apertus's Clean-Wiki
   slice → `include_half_weight` recommended), and HPLT clean60 at
   0.957 fresh (its absolute volume — 48.7M docs — pulls 2.10M
   strict-exact matches with FW2-HQ ell_Grek). Per the runbook:
   build the CPT pool from the published nanochat corpus minus
   `apertus_overlap_drop_docs.parquet`, then replay nanochat's
   internal dedup. **Implication for the plan:** §3 Node 2's
   training-mix design is now ~80 % concrete.

6. **Held-out contamination check is SKIPPED in the current dedup
   run.** The C3 mix manifest with val/test doc-id lists lives on
   the now-unreachable `apertus-greek-tokenizer-20260408t160000z`
   gcloud instance — and **GCloud access was lost 2026-05-20**, so
   the previously-suggested "restart the instance" alternative is
   gone. Remaining options: re-derive the val/test partition by
   re-running the splitter (`subprojects/_archive/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py`)
   from the published nanochat corpus on HF with the original seed,
   or accept that the val/test integrity claim has a hole and
   document it. → [Review checkpoint B](#7-review-checkpoints--what-still-needs-your-explicit-sign-off)

7. **Greek diacritic policy = `preserve` is now committed** (user
   decision 2026-05-18, recorded in the dedup-audit `text_dedup_pin.json`).
   This wasn't in the plan; if there's ever a tokenizer-side dedup
   refresh, that policy decision needs to propagate.

8. **Polytonic / Ancient Greek extension exists as a separate downstream
   arm** (subproject [`02_1_polytonic_greek_extension/`](../../02_1_tokenizer_experiments/02_1_polytonic_greek_extension/), 2026-05-18). It is a continuous-BPE
   continuation of the C3-17,408 ship variant, adding **+5,120** polytonic
   tokens → final vocab **153,600 = 256 × 600**. The full 11-step sweep
   {0…5120 by 512} ran and chose +5,120 as the best grid point on
   `poly_val_balanced` (Greek-word fertility 3.00 → 1.96; distinctive-
   polytonic fertility 3.00 → 1.79; modern-Greek regression negligible).
   The plan's §1 / §2.7 / §10 Q3 already say polytonic is *out of scope*
   for the C3 deployment register — this arm is what fills that gap as
   a separate downstream layer, not part of the three-arm C3 comparison.
   **Implication for the plan**: §10 Q3 should be reframed from "open"
   to "addressed by a separate stacked extension; deployment decision
   = whether to ship 148,480 or 153,600 as the base for CPT." See
   [POLYTONIC_VOCAB_BUDGET_CHECK.md](POLYTONIC_VOCAB_BUDGET_CHECK.md) for the verification of the +5,120 budget
   against the sub-1B-language vocab-vs-tokens pattern.

9. **CSCS Clariden is already onboarded** for this user. Account
   `a0140` is live (user association observed via `sacctmgr`), ssh
   aliases `ela / clariden / bristen` work end-to-end, and a real
   2-node 8-GPU DDP smoke test ran successfully 2026-03-28 (jobid
   1743975, hosts `nid006238 / nid006247`, `all_reduce_value == 28.0
   == expected`). The PyTorch `uenv` image `pytorch/v2.6.0:v1` is
   verified working with `--view=default`. See
   [`/home/foivos/Projects/cscs-clariden-project-understanding/README.md`](file:///home/foivos/Projects/cscs-clariden-project-understanding/README.md).
   **Implication**: the [§5](#5-cscs-auth--execution-workflow) "what's already in place" claim is significantly stronger than originally drafted — there's a working slurm launch pattern and a known scratch layout, not just configured ssh.

---

## 2. What "Greek embeddings are as well-initialized as all others" actually says now

The phrasing in the kickoff message — "Greek embeddings are as well
initialized as all others" — is stronger than what was already in
the plan, and the diagnostic v2 supports the stronger reading. The
plan v0.12 has:

- **Phase A** (§2.6): Greek tokens are statistically
  indistinguishable from English-baseline in *norm distribution* on
  E and U. "Greek tokens are well-trained on average."
- **Phase B v4** (§2.7): Greek tokens predict modern Greek at
  median NLL ≈ 0.95, ~3× lower than English-on-English. "Greek
  tokens are well-predicted in context."

Diagnostic v2 adds, on top of these:

| Axis | Finding | Reading |
|---|---|---|
| Geometric coherence | Greek is a real subspace with `K_significant = 397` (E) / 361 (U) above the Marchenko-Pastur noise floor. Centroid is 84× more displaced from the global mean than ¬Greek's centroid is. | Apertus has not "smeared" Greek into the surrounding cloud — it has a Greek-shaped cloud-shaped-by-Greek-data sitting inside the embedding manifold. |
| Language-axis | A binary Greek-vs-¬Greek logistic classifier on raw E or U scores macro F1 = 0.99 (E) / 1.00 (U). The classifier's weight is essentially `(μ_Greek − μ_¬Greek)` (cos 0.85 / 0.92). | The model knows what "Greek" is as a direction in embedding space, and that direction is exactly where the Greek centroid is. |
| Within-Greek morphology | 5–9× cosine tightness on common-root families (`μέν*`, `ματ*`, `μεγ*`, `αυτ-`, `συν-`). | Apertus has learned Greek-internal compositional structure, not just "Greek-vs-rest." |
| Cross-language semantic bridge | No Greek↔English etymological bridge in static input embeddings — en↔el cosine is ~+0.05 regardless of Greek-origin status. | The model's Greek competence is Greek-internal, not English-mediated. Aligns with plan Constraint 1's case for no-translation-mediation. |

**This is what the kickoff statement is referring to.** The
implication that propagates downstream is the **same one already in
the plan**, just sharper:

- Phase B v4 already framed extension as *purely compression
  economy* (plan §2.7 / §10 Q1).
- Diagnostic v2 takes the **"no quality gap to close"** statement
  from "no average gap" → "no geometric gap, no morphological gap,
  no language-axis gap, no semantic-bridge gap."

Where this should land in `experiments_plan.md`:

- §2.6 conclusion ("Greek tokens are well-trained") gets the
  diagnostic-v2 backing as a second paragraph or footnote.
- §2.7 *Implications* (3 bullets) gets a fourth bullet: the
  three-arm comparison now expects all three arms to clear the
  Greek progress threshold easily, and the deciding axis is
  efficiency-on-extension vs preservation-on-multilingual. This is
  already stated in plan §3 Node 4 *(weight)* and §10 Q8c *(note on
  M_ext vs M_van asymmetry)* but could be sharper.
- §6.4.4 (cleanup): no longer needs the hedge — Greek embedding
  posture is now characterized at four levels.

→ All concrete edits parked in `PLAN_DIFF.md` after review.

---

## 3. CPT corpus — current shape

Per the runbook:

```
final CPT pool
  = ( nanochat-published corpus )
  − ( apertus_overlap_drop_docs.parquet )       # hard-exclude
  − ( internal dedup, drop_intra_and_inter )    # replay
```

The order matters and is captured in the runbook. The pool sits at
~95.98M fresh docs (of 98.20M HF pool) before internal dedup; after
the published nanochat internal-dedup overlay, it drops to the
~14.4M-doc range that C3 also drew from. Concretely:

- Source identity → **decided** (HF nanochat + dedup overlays).
- Source weighting → **partially decided** — `per_c3_source_actionable.parquet` recommends `include_full` / `include_half_weight` / `replay_only` per source. The actual mix ratios for CPT (Greek source weights, plus the non-Greek replay ratio that plan §8.5 / §10 Q5 still owes) are not yet locked.
- Apertus overlap → **measured**, hard-drop overlay published at `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z`.
- Held-out integrity check → **skipped this run** (manifest hostage on terminated instance).
- Greek diacritic policy → **`preserve`** (committed).

What's still missing on the CPT-corpus side, before training fires:

1. **Decide whether to recompute the held-out contamination check.**
   Restarting the terminated gcloud instance once is cheap; without
   it we can't verify val/test integrity. → [Review checkpoint B](#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
2. **Pick CPT mix ratios.** Plan §3 Node 2 + §10 Q5. The dedup
   audit gives per-source actionable counts; what's still needed is
   "X % Greek (with HPLT half-weighted, finewiki half-weighted), Y %
   non-Greek replay (drawn from Apertus-style FineWeb-Edu /
   FineWeb-HQ / FineWeb-2 for forgetting prophylaxis)." Suggest a
   first cut: 85 % Greek / 15 % non-Greek replay, with the Greek
   sub-mix following the `recommended_action` column verbatim.
3. **Build the CPT mix on Clariden's `xfer` partition and stage it on `/iopsstor/scratch/cscs/fffoivos/cpt_corpus_v1/`.** The [runbook](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md) was originally written for a GCP scratch VM that uploaded to HF, but GCloud access was lost 2026-05-20. The same build steps (download nanochat from HF, hard-exclude Apertus overlap, replay internal dedup, write final pool) run cleanly on a single `xfer` allocation (256 vCPU + 500 GB RAM + 24 h walltime, ample for the ~14.4 M-doc pool). Output stays on iopsstor for the next training step — no HF round-trip needed. Single biggest gate before the three arms fire.

---

## 4. What still has to happen before the three arms fire on CSCS

Ordered roughly by their dependency graph. **Bold = blocking the
arms.**

### 4.1 Tokenizer-side (chained, in repo subprojects 02_1 / 02_2)

1. **Commit the C3 cutoff in writing** (plan §3 Node 1).
2. Build Apertus-compatible merged variants at the chosen cutoff
   ([`ACTIVE_BACKLOG.md`](../../../docs/ACTIVE_BACKLOG.md) item 3) —
   this is mostly a script that takes the C3 `tokenizer.json` and
   truncates `model.merges` to the first N additions.
3. Run the intrinsic + fertility bundle on the chosen merged
   variant across `virgin_hplt`, `C3_val_clean`, `C3_test_clean`,
   `modern_greek_eval` ([`ACTIVE_BACKLOG.md`](../../../docs/ACTIVE_BACKLOG.md) items 4–5)
   — for the *chosen* cutoff only, not the full sweep again. Already
   have the sweep data in [C3_CUTOFF_REPORT.md](../../../docs/C3_CUTOFF_REPORT.md);
   we just need the focused single-row read.
4. **Implement merge-rule extension** in `subprojects/02_2_tokenizer_implementation/`
   ([`ACTIVE_BACKLOG.md`](../../../docs/ACTIVE_BACKLOG.md) item 8) — first 1000 ids preserved, special-token behavior preserved, regex split + byte-level regime preserved.
5. Worst-50 modern-Greek tokens absorption check at the chosen
   cutoff (plan §11). Cheap, sharp signal.

### 4.2 Embedding-init side (new code in 03)

6. **ReTok initializer** — base-piece retokenization → mean →
   per-side norm matching against the Phase A targets in plan §8.2
   (Greek-content E p50 = 5.05, U p50 = 3.80; structural E p50 =
   4.91, U p50 = 3.58). ~50 lines per the plan.
7. **Token-distillation refinement** — port `konstantinjdobler/token-distillation`
   for the E side; pattern (2) NTP-only objective on `U` rows during
   training. Plan §5 Experiment 3.
8. **Resize-token-embeddings glue** — `model.resize_token_embeddings()`
   in HF transformers handles both `E` and `U` since `tie_word_embeddings = False`.

### 4.3 Training harness on CSCS (the load-bearing gate)

9. **Pick the framework.** Apertus was pretrained on
   `swiss-ai/Megatron-LM-Swiss-AI` (the Megatron fork the Swiss-AI
   team maintains; verifiable at the Swiss-AI org). Most ergonomic
   options for CPT, in increasing complexity:
   - HF transformers + `accelerate` / `torchtitan` — easiest, won't
     match Apertus's training-recipe details (AdEMAMix, WSD, 0.1
     clip).
   - `nanotron` (HF Megatron-style) — closer to Apertus's regime;
     supports the staged unfreezing in plan §8.3.
   - `swiss-ai/Megatron-LM-Swiss-AI` (Apertus's own harness) —
     exactly the recipe in `APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md` §1.1–§1.5.
     Highest setup cost, lowest training-dynamics-drift cost.
   → [Review checkpoint D](#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
10. **Provision Alps storage and compute.** Workspace under
    `/capstor/scratch/cscs/fffoivos` (or `/iopsstor`). Stage the CPT
    parquet pool, the Apertus checkpoint, the chosen tokenizer
    artifact. Slurm partition + GPU budget request.
11. **Write the three sbatch scripts** — Vanilla, ReTok,
    Distillation, sharing the same data loader, schedule, eval
    cadence; only the init step + tokenizer artifact differ.

### 4.4 Evaluation suite assembly

12. **Lock the eval slices to materialize** — `virgin_hplt`,
    `C3_val_clean`, `C3_test_clean` already exist; the cross-language
    regression slices (en / fr / de / ru, plan §10 Q6) do not.
    Build them by anti-joining the corresponding FineWeb-2 slices
    against any Apertus-overlapping doc-ids (so we measure on
    *fresh* content for each language).
13. **Wire one native-Greek benchmark first** — GreekMMLU's public
    split (16,857 questions) is methodologically strongest and
    contamination-resistant via the private leaderboard split. Get
    a single eval harness invocation working before training fires.
    The rest (Belebele, Medical MCQA, OYXOY) can come online during
    the CPT pilot.

### 4.5 Decision-rule pre-commitment (plan §3 Node 4)

14. **Lock X, M_progress, M_ext, M_van, T, and the gate-language
    list — in writing — before any arm completes CPT.** This is the
    only hard temporal constraint in the plan and is currently the
    project's biggest exposed surface. → [Review checkpoint C](#7-review-checkpoints--what-still-needs-your-explicit-sign-off).

---

## 5. CSCS auth + execution workflow

**Canonical doc:** [`CSCS_AUTH_WORKFLOW.md`](CSCS_AUTH_WORKFLOW.md).
The summary below captures only the current-state snapshot relevant
to this analysis; the workflow doc is the single source of truth.

### 5.1 Current state (verified 2026-05-20)

- Tool: `cscs-key` v1.1.0 installed at `~/.local/bin/cscs-key`. The old `sshservice-cli` (portal-side username/password/OTP) is **deprecated**.
- Daily refresh: `cscs-key sign --headless --duration 1d` (1 d is the only valid duration in this build; OIDC device-code flow). The OIDC refresh token is cached locally so subsequent days usually skip the device-code dance.
- Local keypair (ed25519) at `~/.ssh/cscs-key{,.pub}` with comment `cscs-key fffoivos@home 2026-05-20`. Private 432 B chmod 600, public 115 B chmod 644.
- Cert at `~/.ssh/cscs-key-cert.pub`, valid 2026-05-20T15:03:46 → 2026-05-21T15:03:46 (24 h).
- ssh aliases in `~/.ssh/config`: `ela` / `clariden` / `bristen`, all pointing at the key + cert.
- **End-to-end exercised**: `ssh ela 'hostname'` → `ela5`; `ssh clariden 'hostname'` → `clariden-ln001`; `sinfo`, `squeue`, `sbatch --test-only` all returned cleanly. See [`AUTH_AND_NODE_FINDING.md`](../03_4_implementation_experiments/AUTH_AND_NODE_FINDING.md) for the live probe transcripts.

### 5.2 Daily handoff

1. You run `cscs-key sign --headless --duration 1d`, complete the OIDC device-code prompt on any browser, and the cert lands at `~/.ssh/cscs-key-cert.pub`.
2. You tell me "cert refreshed."
3. I run the cert health check below before any CSCS-touching tool call.

### 5.3 Pre-operation cert health check (my side)

Before any `ssh ela …` / `ssh clariden …` / `ssh bristen …`, I run:

```bash
ssh-keygen -L -f ~/.ssh/cscs-key-cert.pub 2>/dev/null | awk '
  /Valid:/ {print}
  /Key ID:/ {print}
'
```

and parse the `Valid: from … to …` line against `date -u +%Y-%m-%dT%H:%M:%S`. If the cert is expired (or within ~30 min of expiring during a long-running session), I stop and ask you to refresh, rather than silently failing somewhere in the middle of an sbatch operation. **No "try the ssh and see what happens" pattern** — that wastes cycles and surfaces opaque errors.

### 5.4 What I can do once the cert is fresh

Read/inspect, anytime:

```bash
ssh ela 'squeue -u $USER'                # job queue
ssh ela 'sacct -X --starttime $(date -d "1 day ago" +%F)'   # recent runs
ssh ela 'ls -la /capstor/scratch/cscs/fffoivos/'  # workspace inventory
ssh clariden 'sinfo -o "%P %t %D %C %m %G"'   # partition / GPU state
```

Workflow-side, with explicit confirmation per
[`feedback_supervise_dont_just_act.md`](memory) and the
"executing actions with care" guidance:

```bash
# 1. Stage artifacts (rsync, hf_transfer pull, etc.) onto Alps storage.
# 2. Submit a sized-and-bounded sbatch — never an unbounded run.
# 3. Tail the log via ssh ela 'tail -F /capstor/scratch/cscs/fffoivos/runs/<job>/log'.
# 4. Cancel via scancel only after surfacing the state to you.
```

### 5.5 Light persistence — recording each job in the subproject

To avoid the "what's running on Alps right now?" mystery (the same
pattern we already hit with `apertus-greek-tokenizer-20260408t160000z`),
every cluster job I submit drops a one-line record into
[`03_3_cscs_experiments_kickoff/job_log.jsonl`](.) at submit time:

```json
{"ts": "...", "host": "clariden", "job_id": 12345, "sbatch": "vanilla_pilot_v1.sbatch", "arm": "vanilla", "workspace": "/capstor/...", "expected_wall_s": 86400}
```

That gives both of us a single place to grep for "what did I leave
running over the weekend?" without `ssh`-ing in.

---

## 6. Recommended next-action sequence

If we want the three arms firing on Alps in **two to three weeks**,
ordered by criticality:

| Order | Action | Owner | Blocking? |
|---|---|---|---|
| 1 | Commit C3 cutoff in writing → update [GLOBAL_DECISIONS.md](../../../docs/GLOBAL_DECISIONS.md) Open Decisions block | you (or you sign-off, I edit) | yes |
| 2 | Decide held-out contamination action ([Review checkpoint B](#b-held-out-contamination-check-rerun) — three options after GCloud loss) | you decide; I drive | no but informs eval-integrity claim |
| 3 | Confirm we adopt p-skarvelis's HF-Trainer pipeline scaffold as the trunk (see [`../03_4_implementation_experiments/STORAGE_AND_EXISTING_WORK.md`](../03_4_implementation_experiments/STORAGE_AND_EXISTING_WORK.md) §3) | you decide | yes |
| 4 | Lock X / M_progress / M_ext / M_van / T + gate-language list, in writing | you set values; I park them in `experiments_plan.md` §10 Q8f | yes (must be before any arm completes) |
| 5 | Build CPT mix on **Clariden `xfer` partition** (was: GCP scratch VM — no longer available); upload to private HF dataset or stage on `/iopsstor/scratch/cscs/fffoivos/cpt_corpus_v1/` directly | I drive on Clariden | yes |
| 6 | Build Apertus-compatible merged tokenizer variant at chosen cutoff — **DONE locally** at [`ship/apertus_greek_modern_only_148480/`](ship/apertus_greek_modern_only_148480/) + [`ship/apertus_greek_extended_153600/`](ship/apertus_greek_extended_153600/); intrinsic + fertility check on chosen one ✓ | ✅ done | — |
| 7 | Implement merge-rule extension code in `subprojects/02_2_tokenizer_implementation/` — partially deferred; the ship bundle obviates the need for separate merge-rule machinery for the comparison runs | I drive when needed | no for pilots, yes for shipping |
| 8 | Write ReTok init + Distillation refinement code (this subproject area or a new one under 03) | I drive | yes |
| 9 | Cross-language regression eval slices (en/fr/de/ru) — build by anti-joining FW2 against Apertus-overlap | I drive (also fits in a Clariden `xfer` slot) | yes |
| 10 | Wire GreekMMLU eval harness on Alps | I drive once cert is fresh | partially (can be parallel with first pilot) |
| 11 | Submit Vanilla pilot first (smallest setup change, calibrates Alps throughput + storage) | I drive on Alps, you cert-refresh daily | — |
| 12 | Submit ReTok pilot in parallel once Vanilla is mid-run and we have throughput numbers | I drive | — |
| 13 | Submit Distillation pilot once ReTok init code is verified end-to-end on a 10 B-token toy run | I drive | — |

Items 1–4 don't require CSCS or compute. Items 5 and 9 now run on
Clariden's `xfer` partition (the former GCP-scratch-VM path is
closed). Item 6 is **done** — the ship bundles are local and
verified. Items 8 onwards are training-side work that hits Clariden
GPU partitions.

---

## 7. Review checkpoints — what still needs your explicit sign-off

These are the items where I don't have the authority to make the
reasonable call without you naming the value:

### A. ~~C3 cutoff — exact pick from `{10240, 15360, 20480, 25600}`~~ — **RESOLVED**

**Decided 2026-05-18: cutoff = 17,408 (curated + backfilled).**
Total vocab **148,480 = 256 × 580**. Curation removes 69 noise tokens
structurally at build time; 69 valid merges from positions
17,409..17,477 backfill the slots; 0 cascade-skips. The earlier frozen
grid `{10240, 15360, 20480, 25600}` in [GLOBAL_DECISIONS.md](../../../docs/GLOBAL_DECISIONS.md) is therefore
**stale** — the actual sweep was finer (every 1024 from 1024 to
25600) and the chosen value 17,408 is not in either grid version.

Justification (from [`CHOSEN_CUTOFF.md`](../../02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)):
- Greek fertility on C3_val held-out: 1.345 (−44.2 % vs apertus_base 2.41)
- 82.4 % of theoretical-max fertility improvement captured (asymptote 1.118; full sweep at 25,600 captures 88.4 %)
- First cutoff where the next 1k buys < 1 % marginal improvement
- 148,480 = 128 × 1160 = 256 × 580 (aligned)
- TFG @ Apertus-55 = 0.11613 (vs base 0.11606; basis-points regression)

→ Action: fold this into `experiments_plan.md` §2.4 / §3 Node 1 / §10 Q1 / §11. Update [`GLOBAL_DECISIONS.md`](../../../docs/GLOBAL_DECISIONS.md) to remove the old grid framing.

### B. Held-out contamination check rerun

**Updated 2026-05-20**: the previously-recommended option (restart
the terminated gcloud instance, extract the C3 mix manifest, rerun
step 08) is **no longer available** — GCloud access was lost the
same day. Three remaining options, ordered by effort:

1. **Document the gap, ship without the integrity check.** Honest framing: "C3 val/test integrity is verified only as far as the train side; ~0.4–0.5 % cross-split contamination is expected from the splitter's row-level partitioning (per `C3_CONVERGENCE.md` § Held-out integrity), and we did not separately measure Apertus pretraining overlap with C3 val/test." Cheapest.
2. **Re-derive the val/test partition** by re-running the splitter (`subprojects/_archive/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py`) on a Clariden `xfer` allocation, with the same seed and input manifest from the published nanochat corpus. Output is the C3 val/test doc-id list, identical to what was on the terminated instance. Then rerun step 08 of the dedup audit against that. ~1–2 h on `xfer`. Most defensible.
3. **Use the existing `virgin_hplt` slice** (already on home, built by anti-joining HPLT clean60 against C3 train text-md5) as the held-out throughout, accepting that it's an HPLT-only slice and skipping the GlossAPI-side eval. Cheap but narrower.

Decide which option; if (2), I drive once you say go.

### C. Decision-rule numbers (§10 Q8f)

Plan suggestions:

- X = 5 % (preservation gate per gate-language)
- M_progress = 3–5 % (Greek improvement floor)
- M_ext = 1–2 % (extension beats Vanilla)
- M_van = 3–5 % (Vanilla beats extension)
- T = 2–3 % (Distillation beats ReTok)
- Gate languages: English, French, Russian, German (the v0.12 list)

Sign off literal values + the gate-language list, or amend.

### D. CSCS training framework choice

`swiss-ai/Megatron-LM-Swiss-AI` (closest to Apertus's training
dynamics, highest setup cost) vs `nanotron` (closer to Apertus than
HF but lower setup cost) vs HF transformers + accelerate (lowest
setup cost, biggest training-dynamics drift). My read: nanotron is
the right balance for a 3-way pilot comparison where the *internal*
delta matters more than the *absolute* match to Apertus's recipe —
all three arms drift together. But Megatron-LM-Swiss-AI is the
defensible choice for the "main CPT" run after the winning arm is
picked.

### E. CPT mix ratios + non-Greek replay

The `recommended_action` column from the dedup audit gives the
Greek-side weights. The non-Greek replay ratio (plan §8.5 / §10 Q5)
is still a 10–15 % range. Pick a number, or commit to running it
as a 1-D sweep at the smallest model scale we can afford.

### F. Do we add a polytonic / Katharevousa track later? (Plan §10 Q3)

Confirmed out of scope for the current deployment register. Leaving
it open is fine. Sign-off here is just whether to formally close
the question or keep the door ajar in the writeup.

---

## 8. What I am *not* doing in this subproject

- Not editing `experiments_plan.md`. All proposed edits live in §1.4
  / §7 above and will land in `PLAN_DIFF.md` after review.
- Not running gcloud workloads — **GCloud access lost 2026-05-20**.
  Anything that previously routed there now goes to Clariden's
  `xfer` partition (CPU-heavy) or to home (small-scale).
- Not submitting any Alps job (auth + non-job read-only probes were
  exercised — see §5.1; no `sbatch` has been submitted).
- Not picking decision-rule numbers, framework, or cutoff on your
  behalf. Those are review checkpoints, not auto-mode defaults.

---

## 9. Pointers I needed and might need again

| Need | Path |
|---|---|
| Plan as of 2026-05-12 | [`../experiments_plan.md`](../experiments_plan.md) |
| Greek embedding diagnostic v2 results | [`../03_1_greek_embedding_diagnostic/artifacts/results/report_v2.md`](../03_1_greek_embedding_diagnostic/artifacts/results/report_v2.md) |
| Dedup audit (May 19 run) | [`../03_2_apertus_c3_dedup_audit/REPORT_dedup_20260519T010924Z.md`](../03_2_apertus_c3_dedup_audit/REPORT_dedup_20260519T010924Z.md) |
| CPT corpus build steps | [`../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md) |
| C3 cutoff sweep tables | [`../../../docs/C3_CUTOFF_REPORT.md`](../../../docs/C3_CUTOFF_REPORT.md) |
| Apertus training-dynamics architecture | [`../../../docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md`](../../../docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md) |
| Apertus pretraining inventory + Greek share | [`../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md) |
| CSCS SSH config | `~/.ssh/config` Host blocks `ela`, `clariden`, `bristen` |
| CSCS cert health check | `ssh-keygen -L -f ~/.ssh/cscs-key-cert.pub` |
