# CPT Plan v0.6 — Delta vs Prior Planning

> **HISTORICAL.** v0.7 supersedes v0.6, and the user has explicitly chosen "closest to Apertus original process" as the canonical guiding principle (2026-05-20). Of the §5 "Recommended posture" table in this doc, the framework row ("pragmatic split: HF Trainer for bakeoff, Megatron-LM for production") is **withdrawn** — the canonical position is **Megatron-LM-Swiss-AI throughout**. The other recommended-posture rows mostly hold but should be read in v0.7's language (Centroid not Distillation; 70/30 not 80/20; bakeoff at 2 B per arm; vocab scope = 153,600 composite). The live answers doc is [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md).

*2026-05-20. Companion to [`cpt_plan.md`](cpt_plan.md) (v0.6). Maps
v0.6 against the body of prior planning under
`03_apertus_extension_and_embedding_adaptation/`. Flags hard
contradictions, soft divergences, new requirements v0.6 surfaces
that prior work missed, and one internal inconsistency in v0.6
itself.*

> **Premise**: the user has named `cpt_plan.md` v0.6 as canonical.
> Where v0.6 takes a different position than prior docs, **the prior
> docs need to track it.** This doc lists the changes that need to
> propagate, with the reasoning for whether to accept them as-is or
> push back.

## Prior planning artifacts being compared against

- [`experiments_plan.md`](experiments_plan.md) v0.12 (parent plan; pre-existing)
- [`Apertus_plan.md`](Apertus_plan.md) (colleague's PPL-priority curriculum)
- [`03_3_cscs_experiments_kickoff/ANALYSIS.md`](03_3_cscs_experiments_kickoff/ANALYSIS.md)
- [`03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md`](03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md)
- [`03_3_cscs_experiments_kickoff/SHIP_TOKENIZER_RECONSTRUCTION.md`](03_3_cscs_experiments_kickoff/SHIP_TOKENIZER_RECONSTRUCTION.md)
- [`03_3_cscs_experiments_kickoff/REPLAY_LANGUAGE_SELECTION.md`](03_3_cscs_experiments_kickoff/REPLAY_LANGUAGE_SELECTION.md)
- [`03_4_implementation_experiments/STORAGE_AND_EXISTING_WORK.md`](03_4_implementation_experiments/STORAGE_AND_EXISTING_WORK.md)
- [`03_4_implementation_experiments/ENVIRONMENT_AND_BENCHMARKS.md`](03_4_implementation_experiments/ENVIRONMENT_AND_BENCHMARKS.md)

---

## 1. Hard contradictions (v0.6 wins by being canonical, prior docs need updating)

### 1.1 Init experiments: Distillation **bracketed**, replaced by Centroid

| | Position |
|---|---|
| Prior `experiments_plan.md` §5 | Vanilla / ReTok / **Distillation** — the complexity ladder. Distillation = ReTok + gradient descent on attention behavior (Dobler 2025). |
| Prior `ANALYSIS.md`, `REVIEW_PACKET.md`, `ENVIRONMENT_AND_BENCHMARKS.md` | All assumed the three arms include Distillation. `konstantinjdobler/token-distillation` listed as a code repo to port. |
| **v0.6 §5 + §13** | **Distillation bracketed.** Three arms = Vanilla / ReTok / **Centroid**. Centroid = per-script-centroid + Gaussian-noise init, closed-form, <1 min CPU. |

**v0.6's reasoning** (§13): Apertus-specific adaptation cost is ~3 weeks vs ~1 week naive. Untied E/U requires separate LM-head calibration; QK-Norm interaction requires careful attention-target extraction; xIELU's gradient characteristics need validation; layer-choice sweep adds compute. Uncertain benefit at 1.5–2 B bakeoff scale.

**My read**: this is a reasonable call. The plan's logic — "all three arms closed-form" — gives a cleaner three-way comparison (Vanilla vs simple-extension vs subpiece-informed-extension) than mixing closed-form with gradient-descent init. The Centroid arm tests a different hypothesis (script-level distributional prior vs per-token subpiece info) that's genuinely informative.

**Action**: update `experiments_plan.md` §5 (or supersede with v0.6); update `ENVIRONMENT_AND_BENCHMARKS.md` §1.3 to drop `konstantinjdobler/token-distillation` from the "must clone" list (move to "keep on radar per §13 conditions"); update review checkpoints in `ANALYSIS.md`.

### 1.2 Outer Greek/non-Greek replay split: **70/30** vs prior 85/15 – 90/10

| | Replay share |
|---|---|
| `Apertus_plan.md` (colleague) | 10 % English anchor (so 90/10 Greek/non-Greek) |
| `p-skarvelis` actual `run_config.json` | `greek_probability: 0.9` → 90/10 |
| `CURRICULUM_AND_INIT_CORPUS.md` §1 default | 85/15 |
| **v0.6 §4.1 default** | **70/30** |

**v0.6's reasoning** (§4.1): published range across CPT recipes is 44-80 % target / 56-20 % replay (Sailor2, AMD Finnish, SEA-LION v3, EstLLM, Racka). 65/35 to 75/25 is the middle.

**Caveat in those references**: those recipes target languages with much smaller Apertus pretraining shares than Greek's 0.023 %. Lower-resource targets justify higher replay because catastrophic forgetting risk scales with how much the model has to learn. Greek's 0.023 % means we have a *lot* to teach — but Apertus is also already broadly multilingual, so forgetting risk on other languages may be lower than the Sailor2 / SEA-LION case.

**My read**: this is the **biggest numerical divergence** in v0.6 vs prior. It's defensible but not obvious. Recommend starting at **80/20** (compromise) and using v0.6's §6.4 stability diagnostics — particularly **language-ID drift on Tier 1 languages** — to decide whether to push toward 70/30 (if drift appears) or hold at 80/20 (if not).

**Action**: `CURRICULUM_AND_INIT_CORPUS.md` and `Apertus_plan.md` percentages need to be re-aligned with the v0.6 70/30 default, or with a chosen compromise.

### 1.3 Init experiment budget per arm: **2 B** vs prior 10 B

| | Tokens per arm |
|---|---|
| Prior `experiments_plan.md` §8.7 (pilot phase) | 10 B |
| Prior `AUTH_AND_NODE_FINDING.md` § 6.1 sizing table | 10 B |
| **v0.6 §5.4** | **1.5–2 B (default 2 B)** |

This is **not a contradiction in substance** but in framing.

- Prior plan: 10 B/arm was *the pilot* — sized for both init-method discrimination AND quality-trajectory discrimination.
- v0.6: 2 B/arm is *bakeoff only* — sized for init-method discrimination. The actual quality pilot happens post-winner (the "production CPT" of 10-20 B in §9).

v0.6's framing is more honest about what the experiment tests. **Accept.**

Practical compute impact: 3 arms × 2 B = 6 B for the bakeoff (vs the prior 30 B). At p-skarvelis's measured 107 k tok/s on 4 nodes, **the entire bakeoff fits in ~16 h serial** or ~5.2 h with three arms parallel on 4 nodes each (= 12 nodes peak). Far cheaper than the prior plan.

**Action**: update `AUTH_AND_NODE_FINDING.md` § 6.3 — the three real pilots described there are actually a **bakeoff** at 2 B/arm + a **production CPT** at 15-20 B on the winner.

### 1.4 Training framework: **Megatron-LM** (v0.6) vs HF Trainer (our recent lean)

| | Framework |
|---|---|
| Prior `STORAGE_AND_EXISTING_WORK.md` §3.4 implication | HF Trainer + `ApertusForCausalLM` (matches p-skarvelis) — "lower setup cost" |
| Prior `ENVIRONMENT_AND_BENCHMARKS.md` §1.1 recommendation | "Start with `swiss-ai/apertus-finetuning-recipes`" (HF Trainer) |
| Prior `ANALYSIS.md` Review Checkpoint D | "My read: nanotron for the three-arm pilot; Megatron-LM-Swiss-AI for the post-winner main CPT" |
| **v0.6 §7.1** | **"Primary: Megatron-LM."** Apertus was trained on Swiss-AI fork supporting xIELU + AdEMAMix. TRL + Accelerate listed as **alternative for smaller experiments**. |

**v0.6's reasoning** (implicit): Apertus pretraining recipe = Megatron-LM-Swiss-AI; matching the recipe needs Megatron-LM. AdEMAMix, xIELU, QK-Norm, Goldfish, the cross-doc attention masking (V12), EoD loss masking (V13), xIELU scalar handling (V15) — all of these are first-class in Megatron-LM-Swiss-AI but not in HF Trainer.

**Tension with p-skarvelis's existing work**: p-skarvelis uses HF Trainer + transformers 4.57.6 + SDPA, NOT Megatron. Their pipeline works and produced a usable Apertus-Greek SFT checkpoint (`apertus-greek-sft/`, 16 GB safetensors, eval_loss 0.654). Following v0.6 means **forking from their pipeline** rather than continuing it.

**My read**: v0.6 is right that Megatron-LM is the recipe-faithful choice. But the cost is high (re-engineering p-skarvelis's pipeline). Two options:
- **(a) Accept v0.6**: adopt Megatron-LM-Swiss-AI; coordinate with p-skarvelis on what to keep from their HF-Trainer work (treat as analyzed baselines only, like the F1/F2/C1/C2 tokenizer arms).
- **(b) Pragmatic compromise**: run the **bakeoff** on HF Trainer (cheap, matches p-skarvelis, tests init-method differences in isolation), run the **production CPT** on Megatron-LM (recipe-faithful, lets us match Apertus's hyperparams).

Recommend (b) unless you have strong reason to commit to Megatron throughout. (b) also matches v0.6's own positioning of TRL + Accelerate "for smaller experiments" — the bakeoff IS a smaller experiment.

**Action**: clarify Review Checkpoint D in `ANALYSIS.md` and reconcile with v0.6 §7.1. If (b), say so explicitly.

### 1.5 Tokenizer scope in CPT: **modern-only 148,480 (148,480)** or **composite 153,600?**

**This is an internal inconsistency within v0.6 itself.**

v0.6 §1: *"Tokenizer extension is already complete: +17,408 modern Greek tokens and +5,120 ancient/polytonic Greek tokens (vocabulary 131,072 → 148,480). The new embedding rows add roughly **184.5M parameters** (22,528 × 4,096 × 2 for untied input + output) before optimizer state, or ~2.2% of the 8B base."*

Arithmetic check:
- 131,072 + 17,408 = **148,480** (modern-only)
- 131,072 + 22,528 = **153,600** (modern + polytonic)
- 22,528 × 4,096 × 2 = **184,549,376** ≈ 184.5 M parameters → consistent with **153,600** total vocab (both extensions)

So the vocab number quoted in v0.6 §1 (148,480) is inconsistent with the parameter count quoted in the same paragraph (184.5 M assumes both extensions).

Prior `SHIP_TOKENIZER_RECONSTRUCTION.md`: we built TWO ship bundles — `apertus_greek_modern_only_148480/` (148,480 = 256 × 580) and `apertus_greek_extended_153600/` (153,600 = 256 × 600). The modern-only is for the three-arm comparison, the composite is for the polytonic downstream specialization.

**Two possible reconciliations** for v0.6:

1. **v0.6 has a typo**: the vocab is actually 153,600 (matching the param count). The polytonic +5,120 is in scope of CPT from the start.
2. **v0.6 intends modern-only 148,480 for CPT**: the param count is wrong; polytonic stays out of scope of CPT (consistent with our prior framing where polytonic is a stacked downstream layer).

The §3.1 polytonic exposure metrics, §3.3 Goldfish hash interaction with the **extended vocabulary 131,072 → 148,480** wording, and §5.7 caveat on polytonic signal at 2 B all read like the +5,120 polytonic IS in scope. So I lean (1) — v0.6 intends 153,600.

**Action**: **the user / plan author needs to disambiguate.** Probably the simplest fix: in v0.6 §1 change "148,480" to "153,600" and confirm the polytonic +5,120 is in scope of CPT (not deferred). If polytonic is out of scope, then the §3.1 polytonic exposure metrics need rewording, and the active ship bundle is the modern-only 148,480.

---

## 2. Soft divergences (v0.6 frames differently but doesn't conflict materially)

### 2.1 Apertus Greek pretraining data not replayed (v0.6 explicit)

v0.6 §2: *"Old Apertus Greek pretraining data is not replayed."*

Prior `CURRICULUM_AND_INIT_CORPUS.md` argued for **fresh-only for init pilots, mixed for main CPT** (where "mixed" = including the ~2.27 % Apertus-overlap docs from the dedup audit).

These are **compatible**:
- v0.6's "not replayed" = don't *deliberately add* old Apertus Greek data as replay (= the FW2-HQ-Greek slice the model already saw).
- Our "mixed" = whatever Apertus-overlap content happened to land in our nanochat-source pool stays in (= mostly different from FW2-HQ).

No conflict. The Apertus-overlap-drop overlay from 03_2 implements the v0.6 position: removes the actual overlap docs but doesn't constrain mixed-vs-fresh ratios.

### 2.2 LR schedule: WSD (v0.6) vs cosine (prior)

`Apertus_plan.md` and p-skarvelis's runs both use cosine LR.
v0.6 §3.3 specifies WSD with brief re-warmup, per "How LR Decay Wastes Your Best Data" (2025).

WSD is more current best practice for CPT specifically (cosine wastes the highest-quality anneal tokens on the early plateau and the lowest-quality late tokens on the steep-decay endpoint, where WSD shifts the decay to align with the anneal mixture). Accept.

### 2.3 Anneal phase: v0.6 explicit, prior implicit

v0.6 has an explicit anneal phase (final 10-20 % with mixture shift + LR decay). `Apertus_plan.md` doesn't (Claude review at the bottom flagged this as missing, citing Krikri).

v0.6 implements the Claude review's recommendation. Accept.

### 2.4 Replay language list: 24 (v0.6) vs 34 (`REPLAY_LANGUAGE_SELECTION.md`)

Overlap:
- v0.6 list (24): Tier 1 (8) + Tier 2 (11) + Tier 3 (5)
- Our list (34): Tier A (20) + Tier B (11) + Tier C (3)

Languages in v0.6 not in ours: **none** (all 24 are in our 34 in some tier).

Languages in ours not in v0.6: Hungarian, Swedish, Danish, Vietnamese, Indonesian, Croatian, Slovenian, Romansh, Korean, Hindi, Bengali (11).

Two material differences in v0.6's framing:

- **v0.6 Tier 3 framing is over-pessimistic**: v0.6 calls Tier 3 (Latin, Armenian, Georgian, Albanian, Macedonian) "preservation aspiration — Apertus had near-zero exposure." Our [audit](cpt_plan_v0.6_answers.md#q-c3-apertus-per-language-token-shares--partial-we-have-an-audit) shows **all five hit the 1 B sample cap in FineWeb-2**. Their Apertus exposure is real (FW2 is in Apertus's mix, just not at HQ tier). So Tier 3 should be treated as actual replay, not a token gesture.
- **v0.6 excludes Hungarian / Swedish / Danish / Vietnamese / Indonesian**: all four are FineWeb-2-HQ languages that Apertus is strong on. Our justification was "Apertus knows them, replay preserves capability." v0.6's apparent justification is "not enough criterion convergence." Both are defensible; resolution depends on whether you want broader Apertus-baseline preservation (favor our 34) or sharper focus on Greek-related languages (favor v0.6's 24).

Recommend: meet at **~30 languages** by adding Hungarian, Swedish, Danish, Romansh from our list (rationale per criteria 1+2 / 1+2 / 1+2 / Swiss-official) and adding Vietnamese, Indonesian if you want Asian global-script coverage beyond Chinese+Japanese. Korean/Hindi/Bengali defer.

### 2.5 Evaluation metric upgrade: BPC primary

v0.6 §5.1 correctly notes per-token NLL is NOT comparable across tokenizer variants (because tokenizers represent different amounts of text per token). It introduces BPC, NLL-per-Unicode-char, NLL-per-word as the primary tokenizer-agnostic suite.

Our Phase B v4 work used per-token NLL — which is fine for *within* one tokenizer (e.g., the modern-only 148,480 across the bakeoff) but breaks down for cross-arm comparison with Vanilla (131,072).

**This is an upgrade to our planning, not a contradiction.** Adopt.

---

## 3. New requirements v0.6 surfaces that prior planning missed

These are technically-real concerns the prior docs didn't raise:

| v0.6 ref | Concern | Effort | Why we missed it |
|---|---|---|---|
| V12 | Cross-document attention masking preserved in CPT dataloader (Apertus trained with strict document separation; if CPT doesn't, sequences from different docs bleed via attention) | ~1 h | We focused on data pipeline; didn't audit Megatron config flags |
| V13 | EoD token loss masking preserved (Apertus masks loss on EoD positions) | ~1 h | Same |
| V14 | BoD/EoD special tokens preserved through tokenizer extension | ~30 min | **Partially done**: our ship-bundle verification shows the 1000 `added_tokens` (which include BoD/EoD) are byte-identical to Apertus. HF↔Megatron roundtrip not yet checked. |
| V15 | xIELU trainable scalars (αp, αn per layer) in optimizer's parameter list after `resize_token_embeddings` | ~30 min | Easy to miss; xIELU's per-layer trainable scalars are not standard activation behavior |
| V16 | Tokenizer byte-fallback sanity check for new polytonic tokens (don't collide with byte-fallback sequences for the same chars) | ~1 h | We checked structural integrity of the tokenizer but not byte-fallback collisions specifically |
| V8 + G1 | Goldfish hash uniformity with extended vocab (if hash is vocab-aware, extension changes per-token masking distribution) | ~1 day | We didn't know Apertus's Goldfish hash specifics |
| V1 + K1 | Decontamination as gating (NeMo Curator pipeline against all eval sets, with multiple Greek normalization views) | 3-5 days | We mentioned eval contamination in `experiments_plan.md` §10 Q6 but didn't make it gating |
| §6.1 | Bootstrap CIs over evaluation samples + checkpoint-window averaging (Park et al. Oct 2025 instability) | embedded in eval methodology | We treated checkpoints as point estimates; v0.6 is statistically more careful |
| §6.5 | Inspect AI for custom evals (polytonic continuation, language-ID drift, accent accuracy) | 1-2 weeks construction | We listed candidate benchmarks but didn't propose custom evals at this granularity |

**Net assessment**: v0.6 surfaces 7 verification items + 2 methodology upgrades that should have been on our list. None are deal-breakers; all are bounded-effort engineering items.

---

## 4. Internal inconsistency within v0.6 itself

Already flagged in §1.5 above: the vocab number (148,480) doesn't match the param count (184.5 M ⇒ 153,600 vocab). One of the two is a typo. The plan author should clarify which.

---

## 5. Recommended posture (which side wins on each item)

| Item | Side to keep | Rationale |
|---|---|---|
| Init arms (Distillation vs Centroid) | **v0.6** (Centroid) | Cleaner three-way comparison; adaptation cost defensible |
| Replay split | **80/20 compromise** | v0.6's 70/30 is conservative for Greek's 0.023 % baseline; our 85/15 is aggressive; 80/20 balances with V4-baseline calibration |
| Init budget per arm (2 B vs 10 B) | **v0.6 (2 B)** with prior 10 B reinterpreted as "production CPT on winner" | More honest framing |
| Framework (Megatron-LM vs HF Trainer) | **Pragmatic split (b)**: HF Trainer for bakeoff, Megatron-LM for production | Saves engineering cost on the bakeoff; preserves recipe fidelity for the actual production run |
| Tokenizer scope (148,480 vs 153,600) | **Needs clarification** | Internal inconsistency in v0.6 §1 — author should disambiguate. My lean: 153,600 (param count is the truth, vocab number is the typo). |
| LR schedule (WSD vs cosine) | **v0.6 (WSD)** | Current best practice for CPT |
| Anneal phase | **v0.6 (explicit)** | Already recommended in Claude-review of `Apertus_plan.md` |
| Replay language list | **Compromise ~30**: v0.6's 24 + Hungarian / Swedish / Danish / Romansh from ours | Adds 4 FW2-HQ languages and the Swiss-official Romansh; defers global-script-diversity additions to post-winner |
| Tier 3 framing | **Our audit wins** (Tier 3 has real corpus, ≥1 B sample tokens each) | v0.6's "preservation aspiration" framing is over-pessimistic |
| Eval metric primary | **v0.6 (BPC)** | Upgrades our per-token NLL approach |
| Apertus Greek replayed? | **No (v0.6)** | Compatible with our position |
| New V12-V16 verifications | **v0.6 (adopt all)** | Real engineering concerns we missed |
| Decontamination as gating | **v0.6 (adopt as V1)** | Necessary for honest eval reporting |
| Bootstrap CIs + checkpoint averaging | **v0.6 (adopt)** | Statistically more careful |

## 6. What needs to happen to reconcile

If you confirm the recommended posture above, the prior docs need the following edits — none of which I'll make until you OK:

1. **Add a top-of-file pointer** in each prior doc to `cpt_plan.md` v0.6 as the canonical superseding plan.
2. **`experiments_plan.md` §5**: bracket Distillation, add Centroid, update implementation §8.
3. **`CURRICULUM_AND_INIT_CORPUS.md`**: update the 85/15 default to 80/20 (or whatever you pick); explicitly point to v0.6 §4.1 for the canonical framing.
4. **`REPLAY_LANGUAGE_SELECTION.md`**: reconcile with v0.6 24-language list; note the Tier 3 audit correction.
5. **`SHIP_TOKENIZER_RECONSTRUCTION.md`**: stay as-is — both bundles are valid; the v0.6 question is which is in the CPT scope. Wait for v0.6 clarification.
6. **`ENVIRONMENT_AND_BENCHMARKS.md`**: drop Distillation; add Centroid; add Inspect AI; add the V12-V16 verifications to the §3.3 slurm-required tasks; add NeMo Curator for decontamination.
7. **`AUTH_AND_NODE_FINDING.md`** §6: reinterpret 10 B/arm as production CPT (post-winner), not bakeoff; bakeoff = 2 B/arm × 3 arms.
8. **`ANALYSIS.md` Review Checkpoint D**: rewrite to reflect the pragmatic-split position (HF Trainer for bakeoff, Megatron-LM for production), or commit to one and supersede the other.

Once the user signs off on the posture, these edits are mechanical.
