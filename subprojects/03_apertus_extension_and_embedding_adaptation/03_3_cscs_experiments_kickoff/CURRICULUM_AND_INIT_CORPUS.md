# CPT Curriculum + Init-Corpus Decision

> **v0.7 supersedes this doc as canonical.** The plan at
> [`../cpt_plan.md`](../cpt_plan.md) v0.7 §2 + §3 + §4 sets the live
> curriculum and replay policy. This doc was written under v0.5 / v0.6
> framing and is retained as design rationale. Specific propagations
> from v0.7 to flag inline below:
>
> - **Replay split**: v0.7 §4.1 default is **70 / 30 Greek / non-Greek**, not the 85 / 15 this doc argued. The 70 / 30 figure is the working default; final value per Q B1.
> - **Curriculum structure**: v0.7 §2 specifies a **single shuffled-mixture bulk + an annealing tail in the final 10–20 %** — *not* the four-phase HPLT-broad → register → academic+legal → dictionary structure this doc proposed. Replay is present from token 0 (not Phase-0-only), and Phase 3 dictionary handling is incorporated into the anneal mixture rather than a distinct trailing phase.
> - **Init-pilot corpus**: still **fresh-only** via the Apertus-overlap-drop overlay — v0.7 §2: "Old Apertus Greek pretraining data is not replayed." Consistent with §1 of this doc.
> - **Tokenizer scope**: extended ship bundle (vocab **153,600** = modern +17,408 + polytonic +5,120). v0.7 §1's "148,480" wording is a typo; the param math (184.5 M = 22,528 × 4,096 × 2) and §3.1 polytonic-exposure metrics imply 153,600.

*2026-05-20. Reconciles three inputs into a concrete CPT plan:*
- *colleague's PPL/quality/novelty ranking ([`collegues_Apertus_plan.md`](../collegues_Apertus_plan.md))*
- *your "HPLT foundation → OpenSubtitles + openbook + openarchives" suggestion*
- *the dedup audit's per-source actionable recommendations ([REPORT_dedup_20260519T010924Z.md](../03_2_apertus_c3_dedup_audit/REPORT_dedup_20260519T010924Z.md))*

## 1. Should init experiments use the mixed set or the Apertus-fresh-only set?

**Recommendation: fresh-only for the three-arm init comparison; mixed for the main CPT after a winner is picked.**

### Why fresh-only for the comparison phase

**Reading the numbers in this section honestly requires holding three
dedup states distinct:** (a) raw HF pool, ~98.2 M docs; (b) after
Apertus-overlap drop, ~95.98 M docs (lose 2.27 %); (c) after
internal `drop_intra_and_inter` dedup of the remaining pool, ~14.4 M
docs (the C3 train scale per [C3_TRAINING_DATASETS.md](../../../docs/C3_TRAINING_DATASETS.md)). The
trainable pool is (c); the 2.27 % overlap figure applies to (a→b),
not to (c).

| Argument | Weight |
|---|---|
| The dedup audit was *built* to enable a clean fresh-only comparison. The Apertus-overlap-drop overlay at `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z` is exactly the artifact this experiment needs. | high |
| Only 2.27 % of the 98.2 M HF-pool docs overlap with Apertus (2.22 M docs), so going fresh-only at the pool stage loses ~2 % of doc count; the loss after internal dedup is smaller because most overlap docs come from clusters that internal dedup would also collapse. The trade is clearly favorable. | high |
| Including the Apertus-overlap docs blurs **"what does init method X teach the model"** with **"what is the model already responding to from its base pretraining"** — precisely the confound the three-arm comparison is supposed to resolve. | very high |
| Asymmetric M_ext / M_van thresholds in plan §10 Q8 are tightest when the comparison is on novel data. Replay should be its own variable, not baked into the comparison corpus. | high |
| **Token budget against the post-internal-dedup pool**: 14.4 M docs × ~100 B chars total (C3 training scale per [C3_TRAINING_DATASETS.md](../../../docs/C3_TRAINING_DATASETS.md)) = ~6,900 chars/doc avg. At the C3-extended 148,480 tokenizer's ~3.9 chars/token: **~26 B tokens trainable**. At base Apertus tokenizer's ~2.6 chars/token: **~38 B tokens trainable**. For 3 arms × 10 B = 30 B total, that's ~80 % of one epoch at the extended tokenizer — tight but viable for pilots; main-CPT extension would need either repeat passes or an HPLT-broad augmentation. | medium |
| The Apertus-overlap docs are still useful: they go into a *named* replay pool (or are dropped entirely), not into the silent middle of the training mix. | medium |

### Why mixed (cautious) is OK for the main CPT post-winner

Once the winning arm is locked, the second pass — the 20–40 B-token main run — is about lifting Greek quality, not about distinguishing init methods. At that point:
- The Apertus-overlap docs are still valid Greek content. Removing them just for cleanliness costs token mass without a benefit.
- Periodic replay (per Krikri / Claude-review note) wants English + Greek-already-known segments anyway. Apertus-overlap Greek docs naturally fit the "in-distribution Greek the model has seen" role for those mini-replay batches.

### Concrete corpus IDs

For the **three-arm init pilots**:

```
init_corpus = mix_prepare_selected_input(
    source = fffoivos/glossapi-greek-nanochat-pretraining-dataset,
    exclude_doc_keys = fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z/
                       artifacts/dedup_20260519T010924Z/cpt_final_overlay/
                       apertus_overlap_drop_docs.parquet,
    dedup = drop_intra_and_inter,
)
```

This is exactly what [`CPT_DATASET_BUILD_RUNBOOK.md`](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md) builds. The output is the post-internal-dedup, Apertus-fresh pool — the ~14.4 M doc / ~26-38 B token universe described above. **It is the fresh-only pool by construction.**

For the **main CPT** (winning arm only):

Same source pool with `--exclude-doc-keys-path` **omitted**, plus periodic English/code replay segments. The "main CPT corpus" thus differs from the "init pilot corpus" only by inclusion of the ~2.27 % Apertus-overlap docs. No re-build needed; just two materializations of the same nanochat-source corpus, one with the overlay and one without.

## 2. Curriculum — three inputs reconciled

### 2.1 What each input says

**Colleague's `Apertus_plan.md`** ranks 24 GlossAPI sub-datasets by `priority = 0.55·gap + 0.30·novelty + 0.15·quality − noise_penalty`. The HP tier (priority ≥ 1.4) ordered by score:

| rank | dataset | gap | novelty | quality | role |
|---:|---|---:|---:|---:|---|
| 1 | modern-greek-dictionary | 1.00 | 0.86 | 0.64 | gap restoration (last) |
| 2 | artos-zois | 0.22 | 0.65 | 1.00 | foundation, high-quality literary |
| 3 | Ellinika_Keimena_Project_Gutenberg | 0.27 | 0.55 | 1.00 | foundation, literary |
| 4 | klasikh_arx_ell_grammateia | 0.11 | 0.65 | 1.00 | foundation, ancient/historical |
| 5 | eurlex-greek-legislation | 0.00 | 0.75 | 0.99 | legal specialization |
| 6 | Wikisource_Greek_texts | 0.22 | 0.56 | 0.97 | foundation, literary |
| 7 | Ekklisiastika_Keimena | 0.08 | 0.66 | 0.99 | foundation, religious register |
| 8 | archetai | 0.24 | 0.55 | 0.97 | foundation |
| 9 | dimodis_logotexnia | 0.08 | 0.64 | 1.00 | foundation, folk literature |
| 10 | 1000_prwta_xronia_ellhnikhs | 0.09 | 0.65 | 0.98 | foundation, historical |
| 11 | opengov-deliberations-v2 | 0.03 | 0.78 | 0.71 | civic discourse |
| 12 | openarchives.gr | 0.22 | 0.59 | 0.68 | academic aggregator |
| 13 | Sxolika_vivlia | 0.05 | 0.43 | 1.00 | textbooks |
| 14 | ert-press | 0.09 | 0.63 | 0.76 | journalism |
| 15 | 95k_deigma_ellinikis | 0.22 | 0.56 | 0.65 | misc |
| 16 | istorima | 0.12 | 0.63 | 0.68 | historical narrative |

Colleague's proposed phases:
- **Phase 1 (foundation)**: artoszois + Project_Gutenberg + Ekklisiastika + 1000_prwta — high-quality, low-novelty, register-stable.
- **Phase 2 (domain broadening)**: openarchives.gr + eurlex — adds academic and legal specialization.
- **Phase 3 (gap restoration)**: modern-greek-dictionary — the biggest gap signal but structurally weird (definition format).

Note: colleague's table **does not score HPLT or OpenSubtitles** — they're listed at the top of the priority table as placeholder rows ("HPLT dedublicate", "OPUS__OpenSubtitles") without numbers. The plan explicitly states this is "από test χωρίς προσθήκη νέας γνώσης" — a test without HPLT/OpenSubtitles measurement.

**Your suggestion**: "start with deduped HPLT to set a basic foundation and then shift opensubs + openbook + openarchives as next place."

**Dedup audit per-source recommendation** (from `per_c3_source_actionable.parquet`):

| Source | total | fresh share | recommendation |
|---|---:|---:|---|
| HPLT/ell_Grek_ge8_no_mt_clean60 | 48,728,774 | 95.7 % | include_full |
| OPUS/OpenSubtitles-el-v2018 | 143,441 | 100 % | include_full |
| openbook_gr | 3,719 | 98.5 % | include_full |
| openarchives.gr | 153,215 | 100 % | include_full |
| Apothetirio_Pergamos | 15,240 | 100 % | include_full |
| Apothetirio_Kallipos | 4,827 | 99.9 % | include_full |
| greek_phd | 37,217 | 100 % | include_full |
| eurlex-greek-legislation | 22,694 | 99.3 % | include_full |
| ellinika_dedomena_europaikou_koinovouliou | 28,723 | 99.7 % | include_full |
| Ekklisiastika_Keimena | 675 | 100 % | include_full |
| Wikisource_Greek_texts | 5,377 | 99.5 % | include_full |
| dimodis_logotexnia | 11 | 100 % | include_full |
| klasikh_arx_ell_grammateia | 815 | 100 % | include_full |
| Sxolika_vivlia | 123 | 100 % | include_full |
| AI-team-UoA/greek_legal_code | 47,563 | 99.9 % | include_full |
| 1000_prwta_xronia_ellhnikhs | 1,015 | 99.9 % | include_full |
| opengov.gr-diaboyleuseis | 1,394 | 99.9 % | include_full |
| Ellinika_Keimena_Project_Gutenberg | 214 | 95.8 % | include_full |
| **HuggingFaceFW/finewiki** | 239,695 | **48.4 %** | **include_half_weight** |

Only one source (Greek Wikipedia via finewiki) is recommended at half-weight, due to Wikipedia-revision overlap with Apertus's Clean-Wikipedia slice.

### 2.2 Reconciliation: three-stage curriculum, all three inputs honored

Your suggestion is consistent with the dedup audit and adds the
register-coverage layer the colleague's plan is missing. The
colleague's PPL ranking still informs *what specifically* to put in
each phase. Concretely:

#### Phase 0 — broad-foundation pass *(your suggestion)*

| | |
|---|---|
| **Goal** | Stabilize the model on broad modern web Greek so that later register-specific phases land on a good baseline. |
| **Datasets** | HPLT clean60 (Apertus-deduped) — 46.6 M fresh docs. Single dataset, large. |
| **English anchor** | 10 % FineWeb-HQ English (colleague's default; can tune to 15 % if reasoning regressions appear). |
| **Token budget** | 3 – 5 B tokens (≈ 5 % of HPLT-fresh mass; we don't need to consume the whole pool here). |
| **LR profile** | constant LR or short warmup; full-parameter training but smaller LR on base, higher LR on the new-token rows (per plan §8.4). |
| **Why HPLT first** | Largest, broadest, deduped, web-distribution-stable. Gets the new 22,528 token rows out of their initialization basin with a representative-Greek signal before we start pushing register-specific content. |
| **Risk** | Web Greek has more noise than literary Greek. Mitigate with the wave-2-broad cleaner's `greek_badness_score ≤ 60` filter that's already applied. |

#### Phase 1 — register diversity *(your "opensubs + openbook + openarchives" expanded with colleague's HP literary tier)*

| | |
|---|---|
| **Goal** | Cover the registers that downstream Greek deployment needs: dialogue, didactic prose, literary, academic. |
| **Datasets (weighted by token mass)** | • OPUS/OpenSubtitles-el-v2018 (dialogue) ~12 % • openbook_gr (textbook) ~8 % • openarchives.gr (academic aggregator) ~15 % • artos-zois (high-quality literary, colleague's #2) ~12 % • Ellinika_Keimena_Project_Gutenberg (literary) ~10 % • Ekklisiastika_Keimena (religious register) ~8 % • dimodis_logotexnia (folk lit) ~5 % • Wikisource_Greek_texts ~8 % • 1000_prwta_xronia_ellhnikhs (historical) ~10 % • Sxolika_vivlia (textbooks) ~5 % • ert-press (journalism) ~5 % • istorima (history narrative) ~2 % |
| **English anchor** | 10 % FineWeb-HQ, **periodic** (Krikri-style mini-replay segments, not flat per-batch) per the Claude-review note in `Apertus_plan.md`. |
| **Token budget** | 5 – 8 B tokens. |
| **Why this order** | OpenSubtitles is colloquial register (covers spoken-style Greek the academic register lacks). openbook is structured didactic prose (a useful complement to OpenSubtitles' colloquial flavor). openarchives adds academic-aggregator material. The literary tier (artos-zois / Gutenberg / Ekklisiastika / Wikisource) provides high-quality register depth without dictionary-shaped content. |
| **Note on `openbook`** | Colleague's plan puts openbook in LP tier due to 0.423 noise penalty. The dedup audit says fresh share 98.5 %, recommend include_full. **My read**: the noise penalty likely reflects table-of-contents / boilerplate artifacts. Reasonable to include with a 0.5× weight relative to artos-zois until the cleaned subset is verified. |

#### Phase 2 — academic + legal specialization

| | |
|---|---|
| **Goal** | Inject the academic + legal vocabulary that's the project's stated quality target (per `experiments_plan.md` §1: "academic Greek texts, philosophical, polytonic Katharevousa material"). |
| **Datasets (weighted)** | • Apothetirio_Pergamos (academic theses) ~12 % • greek_phd (Greek PhD theses corpus, **promoted from NR** per Claude review) ~15 % • Apothetirio_Kallipos (academic textbooks) ~12 % • eurlex-greek-legislation ~15 % • AI-team-UoA/greek_legal_code ~10 % • ellinika_dedomena_europaikou_koinovouliou (EU parl Greek) ~12 % • klasikh_arx_ell_grammateia (classical, plays well with the polytonic tokenizer layer) ~10 % • opengov.gr-diaboyleuseis (civic discourse) ~8 % • archetai ~6 % |
| **English anchor** | 10 % periodic replay. |
| **Token budget** | 4 – 6 B tokens. |
| **Why this is Phase 2 not Phase 1** | Specialized vocabulary needs the model to first have a stable general-Greek footing. Putting eurlex right at the start would over-fit the model to legal register. |

#### Phase 3 — gap restoration *(colleague's plan, kept verbatim)*

| | |
|---|---|
| **Goal** | Address the highest measured perplexity gap (modern-greek-dictionary, PPL 21.7 vs corpus median ~5–7) without letting structurally unusual format dominate the trajectory. |
| **Datasets** | modern-greek-dictionary (capped at 18 % per colleague), optionally with synthetic Q/A transformation per Claude review §1. |
| **English anchor** | 15 % (raised from 10 % because dictionary data is the most structurally alien content; matches Krikri-style "increase replay during alien phases"). |
| **Token budget** | 1 – 2 B tokens, reduced LR (5e-6 to 1e-5 vs 1e-5 to 2e-5 in earlier phases). |
| **Risk** | Dictionary entries train the model to complete definitions, not use words in context. The synthetic Q/A transformation step (Claude review §1) is worth running before this phase fires. |

#### Cross-phase non-Greek replay

Per plan §8.5 + Claude-review §4 in colleague's plan: instead of a flat 10 % English-in-every-batch, use **periodic mini-replay segments**:
- 95 % of batches: phase-specific Greek mix.
- Every N steps (e.g. N=20): one batch from FineWeb-HQ English + StarCoder code, drawn from Apertus's actual stage-4/stage-5 mix.
- During Phase 3 (alien-format dictionary), bump replay frequency to every 10 steps.

This matches Krikri's reported approach (paper §4.2) and the Claude-review recommendation.

## 3. Decisions still on the user

| Decision | Default I'd commit to if you don't redirect |
|---|---|
| HPLT clean60 subsampling for Phase 0 | sample 5 % of fresh HPLT (~2.3 M docs ≈ 4-5 B tokens), seed=20260520, deterministic. Repeatable by anyone with the audit overlay + the HF release. |
| openbook weight in Phase 1 | include at 0.5× relative to artos-zois, revisit after first eval pass. |
| Greek_PhD_Theses_Corpus in Phase 2 | include at colleague's expected HP-tier weight (15 %) **once we've measured its PPL** — that's a 30-min job and is the cheapest blocker to clear. |
| modern-greek-dictionary synthetic-Q/A pass | yes — Claude-review §1 is right that dictionary format is too alien; the Q/A transformation makes it train context-properly. |
| Cross-phase replay cadence | start at every-20-steps for Phases 0–2, every-10-steps for Phase 3. Adjust if a hold-out English benchmark regresses. |
| Finewiki at half-weight (dedup audit's only `include_half_weight` flag) | yes — dedup audit recommends it, no reason to override. |

## 4. What this implies for the three-arm init pilots

**Each pilot arm** (Vanilla, ReTok, Distillation) runs on the
**Phase 0 + Phase 1 mix**, not the full curriculum:

- Total budget: 10 B tokens per arm
- Mix: ~40 % HPLT (Phase 0 broad) + ~50 % register-diverse GlossAPI (Phase 1 condensed) + ~10 % English periodic replay
- Same data loader, same schedule, same eval cadence across arms
- The only difference is the init code (Vanilla = no extension; ReTok = mean-of-subtoken; Distillation = ReTok + attention refinement) and the tokenizer artifact (Vanilla uses the original 131,072; ReTok and Distillation use the rebuilt **modern-only 148,480** ship variant at [`ship/apertus_greek_modern_only_148480/`](ship/apertus_greek_modern_only_148480/) — **not** the on-disk `c3_added_17408_curated_padded/` directory, which has the same `TokenizersBackend` wrapper bug as the polytonic-builder output and is not HF-loadable; **polytonic +5,120** is held out for a separate downstream specialization arm using [`ship/apertus_greek_extended_153600/`](ship/apertus_greek_extended_153600/))

**Why Phases 0+1 condensed for the pilot, not 0+1+2+3**:
- The full curriculum is what we'd want after the winner is locked.
- The pilot needs to *distinguish init methods*, not to maximize final Greek quality.
- Phases 2 and 3 are register-specialization + gap restoration — both happen with the same data regardless of init method, so they add work without sharpening the comparison.

If a single specific arm shows a register weakness in the pilot (e.g., Vanilla regresses on legal because it has no eurlex-friendly vocab), that's a finding the comparison should surface — but it would show up in Phase 0+1 perplexity on a held-out legal slice, before we burn the budget on Phase 2.

## 5. Open items to land before pilot kickoff

1. Pre-register the [§10 Q8 decision rule](../experiments_plan.md) numbers (X / M_progress / M_ext / M_van / T) — [review checkpoint C in ANALYSIS.md](ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
2. Measure PPL on Greek_PhD_Theses_Corpus (the one NR-tier dataset that's almost certainly HP-tier in disguise — cheapest unblock).
3. Decide whether to ship the modern-only 148,480 tokenizer for the comparison or the modern+polytonic 153,600. **My read**: comparison runs on **148,480** (matches the plan's three-arm question — "does extension help modern Greek"); polytonic is a *separate* downstream layer with its own CPT after the modern-tokenizer winner is picked.
4. Authorize building the CPT pool per the [runbook](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md). **As of 2026-05-20 the GCloud path described in that runbook is no longer available** — the build needs to move to a Clariden `xfer`-partition allocation (256 vCPU / 500 GB RAM / 24 h, ample for the 14.4 M-doc pool). See [`STORAGE_AND_EXISTING_WORK.md` § 2](../03_4_implementation_experiments/STORAGE_AND_EXISTING_WORK.md#2-cpu-options-the-users-question) for the substitution.
