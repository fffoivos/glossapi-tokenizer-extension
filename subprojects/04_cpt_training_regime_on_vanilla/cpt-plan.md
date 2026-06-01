# Apertus Greek CPT — Experimental Plan v1.0

**Status:** Working draft for collaborative iteration. Three distinct tasks are scoped: a regime diagnostic experiment (§2), a production extension experiment (§3), and the data mix design (§4). Open questions throughout are explicitly framed as research prompts for follow-up agents.

**Scope:** Plan the next round of CPT experimentation on Apertus-8B-2509 for Greek vocabulary extension. Anchors on the four-arm bakeoff (2026-05-12 → 2026-05-26), the native Greek benchmark suite (2026-05-26), the Apertus technical report (arXiv 2509.14233), the Krikri paper (arXiv 2505.13772), and the v0.12 project plan.

---

## (1) Background — what we did and what we learned

### (1.1) The bakeoff (BAKEOFF_FINAL_RESULTS_20260526)

Four arms ran at 2–5B CPT tokens each on Apertus-8B-2509, testing init methods for vocabulary extension:

- **Vanilla** — no extension; baseline continuation.
- **ReTok** — base-piece retokenization averaging for the 17K new tokens.
- **Centroid** — Hewitt-2021 full-Σ Gaussian init.
- **TokenDistil (TD) layer11** — Token Distillation init at hidden layer 11.

Total compute across arms: ~8B training tokens.

**Initial reading (before native Greek suite):** TD won the bakeoff's downstream Greek aggregate at 5B by +0.69 pp over Vanilla, with Vanilla winning BPC and TD winning English/multilingual retention. The +0.69 pp TD-over-Vanilla margin was driven by +7.57 pp on xquad_el alone — strip xquad_el and Vanilla wins on 6 of 7 remaining Greek tasks.

**Revised reading (with native Greek suite, 2026-05-26):** The previous Greek aggregate was MT-derived. On native Greek benchmarks (GreekMMLU, ILSP Medical MCQA, ILSP ASEP MCQA), the picture inverts:

| Model | Native MCQ general | MCQ + Plutus |
|---|---:|---:|
| **Apertus-Base** | **0.4817** | **0.4902** |
| Vanilla-3.5B | 0.4370 | 0.4333 |
| Vanilla-2B | 0.4327 | 0.4256 |
| Vanilla-5B | 0.4305 | 0.4329 |
| TokenDistil-5B | 0.4109 | 0.4160 |
| TokenDistil-3.5B | 0.4028 | 0.4121 |
| TokenDistil-2B | 0.3961 | 0.4049 |
| ReTok-3.5B | 0.3770 | 0.3772 |
| ReTok-2B | 0.3685 | 0.3731 |
| TokenDistil-Init | 0.2939 | 0.2915 |
| Centroid-2B | 0.2824 | 0.2796 |

Three load-bearing findings:

1. **Apertus-Base is the native-Greek ceiling.** No continued arm recovers to it within tested CPT budgets. Best continued arm (Vanilla-3.5B) sits 4.47 pp below Apertus-Base.
2. **Vanilla is the strongest continued arm on native Greek**, not TD. The TD-wins-Greek headline from the bakeoff was an MT-derived artifact, exactly the concern v0.12 §10 Q6 flagged when specifying native-sourced benchmarks for the headline.
3. **TD has a positive slope on native MCQ; Vanilla does not.** TD goes 0.2939 (init) → 0.3961 (2B) → 0.4028 (3.5B) → 0.4109 (5B), monotonic. Vanilla goes 0.4327 (2B) → 0.4370 (3.5B) → 0.4305 (5B), non-monotonic with a peak at 3.5B. **Vanilla appears to "peak early" then drift; TD recovers slowly from a lower start.** Linear-slope extrapolation: TD would need ~3.6B more tokens to catch Vanilla-5B on native MCQ.

### (1.2) What we got wrong (or didn't apply) from the v0.12 plan

**The static-LOO embedding test didn't generalize to CPT.** The earlier LOO benchmark picked C-group-normmatch by 3×. The bakeoff tested a related Centroid method at mass-replacement scale and it failed completely. The proxy ("closeness to a trained embedding under single-token swap") was too far from the production setup.

**v0.12 §8.3 staged training and §8.4 differential LR weren't applied.** Both were named explicitly for extension arms in the parent plan. The bakeoff ran uniform LR, full-parameter from iter 1.

**Apertus's documented continuation recipe wasn't followed exactly.**
- LR: 1.5e-5 (bakeoff) vs 1.1e-5 (Apertus's WSD-designed continuation LR).
- AdEMAMix β3: 0.9999 (bakeoff) vs the paper's explicit short-run recommendation of β3=0.99 (used by Apertus for SFT).
- Loss: NTP (bakeoff) vs Goldfish k=50/h=50 (Apertus pretraining).

**The CPT budget was below pilot scale.** v0.12 §8.7 specified 10B per arm; bakeoff ran 2–5B. Rankings were still shifting at 3.5B.

**Eval suite drift was real.** The bakeoff's original Greek aggregate used MT-derived tasks; the native suite (2026-05-26) is the corrected version. Native Greek MCQ is now the headline metric. MT-derived Greek is demoted to diagnostic.

### (1.3) Broadly what we have learned about Apertus

**Greek is well-trained in Apertus.** Phase A: Greek E/U row norms statistically indistinguishable from English-baseline. Phase B v4: modern Greek predicted at median NLL ~0.95, ~3× better per-token than English-on-English. Apertus saw ~3.1B Greek pretraining tokens / 1,506 Greek vocab slots ≈ 2M occurrences per slot.

**Greek occupies a linearly-separable region of embedding space.** Linear classifier F1 = 0.973 on E, 0.997 on U. The classifier's Greek-direction weight vector aligns with the centroid-offset direction at cos 0.90–0.93. *Caveat:* partially confounded by surface byte features and norm differences.

**Apertus's embedding cloud is more isotropic than older decoder LMs.** Random-pair cosine median 0.0025 on E, 0.009 on U (vs Ethayarajh 2019's predicted 0.3–0.5). Likely architectural cause: qk_norm.

**Untied embeddings matter.** E and U have different norm distributions (U medians ~25% below E) and different anisotropy. TD's `learn_output_with_ce=True` handled this without degenerate-norm failure.

**Apertus is well-conditioned for vocabulary extension in principle.** The bakeoff's degradation across all arms is more likely a regime issue than a fundamental limitation.

### (1.4) Corpus and resources

**Base model:** `swiss-ai/Apertus-8B-2509` (September 2025 release, base). Architecture: 32 layers, hidden 4096, MLP 21504, 32/8 Q/KV heads (GQA), xIELU activation, RMSNorm with qk_norm, no biases, untied embeddings, RoPE θ=500K, max position 65,536, vocab 131,072 (Mistral-Nemo tekken v3).

**Greek CPT corpus available for experiments and production:** ~61.6B Greek tokens total (counted with the extended Apertus tokenizer, ~150K vocab after the 19K Greek extension). Cross-paper comparisons should treat this as approximately equivalent to Krikri's 56.7B (also counted with their extended tokenizer at +20,992 tokens) but byte-normalized comparison is needed for precise parity.

Full source breakdown:

| Source | Tokens | Category |
|---|---:|---|
| HPLT/ell_Grek_ge8_no_mt_clean60 | 44,195,950,025 | Web (filtered) |
| openarchives.gr | 9,084,040,381 | Institutional repository |
| greek_phd / didaktorika | 5,072,421,684 | High-register curated (PhD theses) |
| OPUS/OpenSubtitles-el-v2018 | 1,169,129,913 | Colloquial / dialogue |
| Apothetirio_Pergamos | 818,114,467 | High-register curated (university) |
| HuggingFaceFW/finewiki | 252,454,834 | High-register curated (filtered wiki) |
| eurlex-greek-legislation | 234,316,334 | Legal |
| ellinika_dedomena_europaikou_koinovouliou | 193,859,676 | Parliamentary |
| Apothetirio_Kallipos | 191,417,236 | High-register curated (university) |
| openbook_gr | 149,993,664 | Educational textbooks |
| AI-team-UoA/greek_legal_code | 56,940,266 | Legal |
| Wikisource_Greek_texts | 47,959,191 | Literary / historical |
| 1000_prwta_xronia_ellhnikhs | 47,824,252 | Literary / historical |
| opengov.gr-diaboyleuseis | 32,626,843 | Parliamentary |
| klasikh_arx_ell_grammateia | 26,710,207 | Classical Greek |
| Ellinika_Keimena_Project_Gutenberg | 13,997,757 | Literary / historical |
| Ekklisiastika_Keimena | 12,066,371 | Ecclesiastical |
| Sxolika_vivlia | 10,331,124 | Educational textbooks |
| dimodis_logotexnia | 109,283 | Folk literature |
| **TOTAL** | **61,610,263,508** | |

Aggregated by register / role:

| Category | Tokens | Share | Function |
|---|---:|---:|---|
| HPLT (web, filtered) | 44.20B | 71.7% | Bulk volume; extends what Apertus already saw of web-register Greek |
| openarchives.gr | 9.08B | 14.7% | Institutional content; register/quality TBD (see Q4.4.7) |
| High-register curated | ~6.50B | 10.6% | didaktorika 5.07B + Pergamos 0.82B + finewiki 0.25B + Kallipos 0.19B + openbook 0.15B + Sxolika 0.01B. **didaktorika alone is ~78% of this category.** Register-novel content largely absent from Apertus pretraining. |
| OpenSubtitles (colloquial) | 1.17B | 1.9% | Dialogue / informal register. Register-orthogonal to the curated/web split. |
| Legal & parliamentary | 0.51B | 0.8% | Formal register; eurlex + EU parliament + greek_legal_code + opengov |
| Literary / historical | 0.15B | 0.2% | Wikisource + 1000_prwta_xronia + classical + Gutenberg + ecclesiastical + folk |

This is roughly Krikri-comparable in scale: Krikri used 56.7B monolingual Greek + 21B English + 5.5B Greek-English parallel ≈ 83B CPT on Llama-3.1-8B-Base. With our planned 24% replay share, total CPT volume is ~81–88B, structurally similar to Krikri.

The 17K new vocab slots get roughly 3.6M exposures each from HPLT alone (44.2B / 17K × the Greek fraction of training), well above Phase B v4's ~100K convergence soft-constraint.

**Replay corpus structure for the 24% non-Greek share:** inherited from the bakeoff. See `cpt_plan.md` v0.7 for sources. Composition is revisited as part of §4.

**Compute environment:** Clariden GH200 nodes via CSCS Alps infrastructure. Slurm-managed.

**Pre-existing artifacts the plan depends on:** see §8 References.

---

## (2) Task 1: Regime diagnostic experiment

### (2.1) Full parameter specification

The recipe is the Apertus-faithful continuation regime. Every parameter is listed below — changed and unchanged — to make the assumption set fully explicit.

#### Model and architecture (base, as released)

These are the actual values shipped in `swiss-ai/Apertus-8B-2509`'s
`config.json`. The released base carries the long-context extension
geometry (Apertus paper §2.5), not the initial-pretraining geometry
(§2.3). For Task-1 we train under a *Path-B override* of these values —
see the subsection immediately below.

| Parameter | Value (base, as released) | Source |
|---|---|---|
| Base checkpoint | `swiss-ai/Apertus-8B-2509` | HF model card |
| Layers | 32 | Apertus paper §2.3 |
| Hidden size | 4096 | Apertus paper §2.3 |
| MLP intermediate | 21504 | Apertus paper §2.3 |
| Attention heads (Q / KV) | 32 / 8 (GQA) | Apertus paper §2.3 |
| Head dim | 128 | derived |
| Activation | xIELU | Apertus paper §2.3 |
| Normalization | RMSNorm | Apertus paper §2.3 |
| qk_norm | True | Apertus paper §2.3 |
| Biases | None | Apertus paper §2.3 |
| tie_word_embeddings | False | Apertus paper §2.3 |
| **RoPE θ** | **12,000,000** | **Apertus paper §2.5 long-context extension (shipped value)** |
| **RoPE scaling** | **llama3 (factor=8.0, original_max_position_embeddings=8192, low_freq_factor=1.0, high_freq_factor=4.0)** | **Apertus paper §2.5** |
| **Max position** | **65,536** | **Apertus paper §2.5** |
| Vocabulary | 131,072 (Mistral-Nemo tekken v3) | Apertus paper §2.3 |

#### Training-time positional geometry override (Path B)

The Task-1 Vanilla CPT run does **not** train under the base's released
geometry. It trains under the Apertus paper §2.3 *initial-pretraining*
geometry (rope_theta=500K, max_position=4096, no scaling — what we call
"Path B" below). The base's Path-A geometry (rope_theta=12M,
max_position=65536, llama3 scaling) is therefore *overridden* at training
time:

| Parameter | Training value (Path B) | Source / rationale |
|---|---|---|
| RoPE θ | 500,000 | Apertus paper §2.3 initial pretraining; matches bakeoff training command |
| Max position | 4,096 | Apertus paper §2.3 initial pretraining; matches bakeoff training command |
| RoPE scaling | None (null) | No scaling needed at 4096 sequence length |
| Sequence length | 4,096 | See "Loss and tokenization" table below |

**Why this run is on Path B and not Path A.** The 4-arm bakeoff trained
under Path B (`--max-position-embeddings 4096 --rotary-base 500000`). The
Task-1 regime-diagnostic question is "did the bakeoff regime degrade
Vanilla?" — keeping the same positional geometry preserves the bakeoff
comparison as the cleanest cross-arm signal (iter 477 vs bakeoff
Vanilla-2B = +4.65 pp outside V4 CI is the load-bearing regime evidence).
Switching mid-run would invalidate that comparison.

**Cost of Path B.** Initializing from a Path-A checkpoint and training
under Path B forces the model to re-adapt its positional encoding away
from rope_theta=12M toward rope_theta=500K. Empirical evidence at iter
477 (`reports/decisions_matrix_20260529.md` Decision C +
`adversarial_reviews/Vanilla-2B/adversarial_critique.md`):

- A static "matched-config" Apertus-Base eval at Path B
  (`/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509-matched-rope500k-seq4096/`,
  produced by `scripts/build_apertus_base_matched_config.sh`) shows
  Greek BPB = 1.22 and native MCQ headline = 0.4272 — a 5.5 pp drop on
  MCQ and ~3× degradation on BPB vs the Path-A baseline.
- iter 477 (post-warmup, 2 B training tokens) recovers Greek BPB to
  ~0.43 and headline MCQ to 0.4792 — well above the perturbed
  matched-config baseline and at Apertus-Base Path-A point estimate
  (0.4817, CI [0.4629, 0.4997]).
- Conclusion: Path-B Vanilla CPT *does* re-adapt; the first ~1 B tokens
  carry the rope-adaptation cost, and post-warmup the model produces
  useful Greek-side improvement.

**Path-A revisit is a Task-2 design decision** — see §3.4 Q3.4.10.

#### Optimizer (changed: β3, α/β3 warmup)

| Parameter | Value | Source / change from bakeoff |
|---|---|---|
| Optimizer | AdEMAMix | Apertus paper §2.3 (unchanged) |
| β1 | 0.9 | Apertus pretraining default (unchanged) |
| β2 | 0.999 | Apertus pretraining default (unchanged) |
| **β3** | **0.99** | **Apertus paper Appendix C short-run recommendation; bakeoff used 0.9999** |
| α | 8 | Apertus pretraining default (unchanged) |
| α / β3 warmup | **287 steps** (= first **1.2B tokens** at 4.194M tokens/step) | Mirrors Apertus long-context continuation warmup window; Appendix C says α and β3 require warmup and can be scheduled independently of LR. |
| Weight decay | 0.1 | Apertus pretraining default (unchanged) |
| Gradient clipping | 0.1 | Apertus pretraining default (unchanged; aggressive — designed for AdEMAMix sensitivity) |

#### LR schedule (changed: base value)

| Parameter | Value | Source / change from bakeoff |
|---|---|---|
| **Base LR** | **1.1e-5** | **Apertus paper §2.5 long-context continuation LR; bakeoff used 1.5e-5** |
| LR schedule shape | **linear warmup, then constant** | Apertus long-context continuation sets LR to the final pretraining cooldown value and warms up at the start of each continuation stage. No final cooldown in Task 1. |
| LR warmup | **1.2B tokens** (= 292,969 samples; 287 optimizer steps at global batch 1024) | Apertus paper §2.5 long-context continuation warmup. |

#### Loss and tokenization (changed: loss)

| Parameter | Value | Source / change from bakeoff |
|---|---|---|
| **Loss** | **Goldfish (k=50, h=50)** | **Apertus pretraining default; bakeoff used NTP** |
| Sequence length | 4096 | Apertus pretraining default (unchanged) |
| Cross-document attention | Disabled (packed seqs masked) | Apertus pretraining default (unchanged) |
| EoD token loss masking | Enabled (EoD positions not backpropped) | Apertus pretraining default (unchanged) |
| Tokenizer | Apertus base (no extension in this task) | Vanilla has no new tokens |

#### Batch and precision

| Parameter | Value | Source / notes |
|---|---|---|
| Global batch size | 4.2M tokens (Apertus 8B initial) | Apertus paper §2.3. The 4.2M → 8.4M doubling Apertus did at 8T pretraining tokens is not relevant for a 7–10B CPT. |
| Microbatch / grad accumulation | **microbatch = 2 samples/GPU; TP=2; PP=1; global batch = 1024 samples**. Default 1 GH200 node / 4 GPUs gives DP=2 and **256** grad-accum microbatches. | Proven bakeoff setting; preserves Apertus global batch while avoiding GH200 mb=4 fragmentation/OOM. If node count changes, grad accumulation is derived as `1024 / (2 × DP)`. |
| Mixed precision | bf16 | Apertus pretraining default. Confirm matches bakeoff. |

#### Data (changed: Greek source narrowed; top-level shares held)

| Parameter | Value | Source / notes |
|---|---|---|
| Training mix | 70% Greek / 24% replay / 4% code / 2% math | Bakeoff B1 top-level shares (unchanged — see §4 for justification of holding this fixed) |
| Greek source for Task 1 | HPLT clean60 wave4 only | Narrowed to avoid mixing in GlossAPI sources before the HPLT-first diagnostic is read |
| Replay composition | Bakeoff default (see `cpt_plan.md`) | Held constant to isolate the regime question |

#### Eval

| Parameter | Value | Source / notes |
|---|---|---|
| Headline Greek metric | Native MCQ aggregate (GreekMMLU + ILSP Medical MCQA + ILSP ASEP MCQA) | Native suite report 2026-05-26 |
| Diagnostic Greek metrics | Plutus QA (reported separately); greek-nlp/benchmark sample-100 supporting mean; BPC | Native suite report |
| Multilingual retention | English / French / Russian / German aggregates (existing lm-eval bundles) | Bakeoff infrastructure |
| Evaluation checkpoints | 0.5B, 1B, 2B, 3.5B, 5B, 7B, 10B | See Q2.4.1 |
| Adversarial checkpoint review | After each checkpoint's eval sidecars finish, run local `codex exec` with model `gpt-5.5` and `model_reasoning_effort="xhigh"` to critique scripts, logs, checkpoint health, eval artifacts, and interpretation | Required skepticism sidecar; report must address critical findings before conclusions are drawn |

### (2.2) Primary question — framed as diagnostic, not "corrected"

**Did continued training under the bakeoff regime hurt a model that already knew Greek well, and if so, what was the cause?**

The bakeoff result is unusual: every CPT arm — *including Vanilla, which has no vocab surgery* — degraded native Greek MCQ relative to Apertus-Base. This rules out "init method is bad" as a complete explanation, since Vanilla degraded too. It implicates either (a) the CPT regime itself (LR / β3 / loss), (b) the data curriculum (replay insufficient, distribution shock, mix wrong), or (c) intrinsic CPT interference on a base model that's already well-trained on the target language.

Experiment (2) tests hypothesis (a) by running Vanilla under an Apertus-faithful regime. If Vanilla under the new regime recovers (matches or approaches Apertus-Base on native MCQ), regime mismatch was the dominant cause. If it still degrades, hypotheses (b) and (c) become the next investigation targets — and we should not run another extension experiment until they're addressed.

Sub-questions worth tracking through the run:

- Does the 3.5B → 5B Vanilla decline reproduce under the new regime? (If yes, drift is not regime-caused.)
- Does BPC improve relative to bakeoff Vanilla in the first 0.5–1B tokens? (Early loss signal — fastest diagnostic.)
- Do BPC and native MCQ move in the same direction? (The bakeoff showed them diverging.)
- Does multilingual retention (English / French / Russian / German) hold?
- Does native MCQ approach Apertus-Base by 7–10B? (The aspirational target.)

### (2.3) Diagnostic mapping — failure pattern → suspected cause

If the experiment results don't cleanly succeed, the diagnostic logic below is the agent's first reading. This is *not* a set of decision thresholds (those are deferred); it's a mapping from observed signal pattern to suspected cause, used to choose the next investigation.

| Observed pattern | Suspected cause | Next investigation |
|---|---|---|
| BPC improves vs bakeoff Vanilla early (≤1B); native MCQ trajectory recovers | Regime hypothesis confirmed | Proceed to Task 2 (extension experiment) |
| BPC improves; native MCQ still peaks early then drifts | Drift is likely intrinsic to CPT on this base, not regime-caused | Investigate curriculum (Task 3) and replay composition before more compute |
| BPC improves; native MCQ stays flat below Apertus-Base | Language modeling is fine, but benchmark-relevant knowledge / reasoning format / calibration is lost | Investigate forgetting via KL-to-base on fixed probes; consider higher replay share |
| BPC fails to improve relative to bakeoff Vanilla even by 1–2B | The three-change bundle doesn't address the bottleneck | Run targeted ablation (Q2.4.5) to isolate which of LR / β3 / Goldfish is responsible, or look outside the bundle |
| Multilingual retention (English / French / Russian / German) degrades | Replay too low or LR too high relative to Greek-mix volume | Increase replay share; revisit replay composition (Task 3) |
| New Vanilla beats bakeoff Vanilla but neither approaches Apertus-Base | Apertus-Base may be unrecoverable with this much CPT regardless of regime; the small Greek pretraining share (3.1B / 13.5T) may already be "saturated" for our budget | Investigate whether CPT can improve over base at all on a well-trained-Greek model; possibly the wrong base for this objective |

### (2.4) Open questions for agents to address

The following are research prompts. Each has a working answer where one's available, and a specific task framed for the next iteration.

**Q2.4.1 — Token budget and checkpoint schedule.**

Working answer: **7–10B tokens, checkpoints at 0.5B, 1B, 2B, 3.5B, 5B, 7B, 10B.** Rationale: 0.5B catches the early-shock signal; 1B and 2B catch initial trajectory; 3.5B and 5B match the bakeoff's peak-then-decline pattern; 7B and 10B are new territory. The budget consumes ~13–18% of total Greek corpus, leaving ~46–49B for production / extension follow-up.

Agent task: validate or revise. If Q3.4.3 (TD slope extrapolation) and Q2.4.2 (Vanilla peak-early mechanism) suggest different trajectories take longer to stabilize, the 10B figure may need to grow.

**Q2.4.2 — Why does Vanilla peak early under the bakeoff regime?**

The native suite shows Vanilla goes 0.4327 → 0.4370 → 0.4305 across 2B → 3.5B → 5B. Non-monotonic.

Candidate mechanisms: (a) catastrophic forgetting overcoming gains; (b) replay-share insufficiency at sustained scale; (c) LR too high causing late-stage divergence; (d) AdEMAMix slow gradient memory (β3=0.9999 in bakeoff) carrying stale momentum that misdirects late-stage updates; (e) data-distribution shock if the bakeoff Greek mix's character changed during the run.

Agent task: literature search on non-monotonic CPT trajectories. The "Reuse, Don't Retrain" framework (arXiv 2407.07263) and the Swallow Japanese CPT report (OpenReview TQdd1VhWbe) are starting points. If experiment (2) shows the new regime fixes Vanilla's trajectory, mechanism (c) or (d) is supported. If the new regime doesn't fix it, (a) and (b) become primary suspects and Task 3 becomes more urgent.

**Q2.4.3 — α / β3 warmup duration for CPT. RESOLVED.**

The Apertus paper says α/β3 warmup is needed for stable AdEMAMix training, and that the α/β3 schedules can be independent of the LR schedule. For main pretraining, Apertus used 100K steps. For long-context continuation, Apertus used a 1.2B-token LR warmup at the start of each continuation stage.

**Task 1 setting:** use the same continuation warmup window for α and β3 as for LR: **1.2B tokens = 287 optimizer steps** at 4.194M tokens/step. This avoids the bakeoff's heavy 50%-of-run warmup while still giving the cold optimizer state a controlled start.

No separate ablation is required before sbatch generation.

**Q2.4.4 — LR schedule shape and warmup. RESOLVED.**

The Apertus paper documents WSD with 1-sqrt cooldown for full pretraining and a 1.2B-token warmup for long-context continuation. Task 1 is continuation, not from-scratch pretraining, so it follows the continuation pattern.

**Task 1 setting:** LR warms linearly for **1.2B tokens** to **1.1e-5**, then stays constant. No final cooldown is used in Task 1 because the point is to read the HPLT+replay/code/math response curve, not to anneal a production endpoint.

Production CPT can revisit WSD cooldown once the regime diagnostic has been read.

**Q2.4.5 — Optional ablation: isolate which of LR / β3 / Goldfish is load-bearing.**

If experiment (2) succeeds (corrected Vanilla recovers), we won't know which of the three coupled changes did it. A small ablation would isolate this:

- Arm A: bakeoff regime (LR 1.5e-5, β3 0.9999, NTP) — control via bakeoff data
- Arm B: LR only (LR 1.1e-5, β3 0.9999, NTP)
- Arm C: LR + β3 (LR 1.1e-5, β3 0.99, NTP)
- Arm D: full Apertus-faithful (LR 1.1e-5, β3 0.99, Goldfish) — same as the experiment (2) main arm

Truncated to 1–2B tokens per arm. Total cost: ~3–6B additional Greek tokens (5–10% of total Greek budget).

Working position for v1.0: **defer this ablation; run experiment (2) main arm only.** Rationale: the three changes are well-motivated as a coupled Apertus-faithful recipe, decomposing them is a second-order question, and the budget is better spent on Task 2 extension experiments after the regime question is answered.

Agent task: validate or revise — if scientific clarity about which change mattered is higher-priority than originally framed, the ablation moves into v1.1 commitments.

**Q2.4.6 — Krikri Vanilla counterfactual.**

Krikri used 56.7B Greek + 21B English + 5.5B Greek-English parallel = ~83B CPT with vocabulary extension. Reported +10.8% average Greek improvement over Llama-3.1-8B-Base. **The Krikri paper does not isolate the vocabulary-extension contribution from the raw-CPT contribution** — the +10.8% is a joint effect of tokenizer extension, 83B CPT, curriculum, parallel data, annealing, and synthetic QA.

Agent task: read the Krikri paper sections on ablations, vocab extension contribution, and CPT scale effects. Report what's known and what's not. Particularly: did they run any Vanilla-only counterfactual at any scale?

**Q2.4.7 — Literature backing for the regime intervention.**

What does published work say about LR / β / loss regime sensitivity in CPT specifically? The Apertus paper notes (Section 2.6) the team is unsure their own pretraining LR was optimal. "Reuse, Don't Retrain" (arXiv 2407.07263) argues LR schedule and data distribution are the first-order CPT levers.

Agent task: literature pass on CPT studies that explicitly vary LR / β / loss regime and report effect sizes. Anchor citations needed.

**Q2.4.8 — Goldfish for CPT specifically.**

Apertus's Goldfish ablations are for *pretraining* at 8B scale (Appendix F.3, Table F.5). No published CPT comparison of Goldfish vs NTP exists that we're aware of.

Agent task: literature search; if nothing exists, frame this as an exploratory choice we're making with paper-backed pretraining evidence but no direct CPT precedent.

---

## (3) Task 2: Production extension experiment

This task is the actual production-relevant question: does Apertus + extended Greek vocabulary outperform Vanilla Apertus on Greek capability under a corrected regime, and does it cross within available budget?

Vocabulary extension's primary value proposition is **inference efficiency** (better Greek compression, lower fertility, faster generation per Greek byte). The capability question is secondary: extension arms need to match or beat Vanilla on Greek downstream while preserving multilingual capability, and they need to spend compute training 17K new tokens to do so. If extension never beats Vanilla within budget, Vanilla is the production choice and extension is a research artifact.

**Task 2 cannot be designed in detail until Task 1 produces its result.** What follows is the framing and the sub-tasks that should be in place by the time Task 1's outcome is known.

### (3.1) Regime carryover from Task 1

The most important Task-1 result is the corrected hyperparameter regime. Task
2 should treat it as the default experimental center, not as background
detail: LR 1.1e-5, 1.2 B-token warmup then constant schedule, AdEMAMix
β3=0.99, Goldfish k=h=50, and the proven batch/parallelism/eval-sidecar
shape. Path A is the geometry cleanup around that regime; TD layer, cutoff,
and stabilization duration are the next variables to test. Any run that
changes the corrected hyperparameters should be a named hyperparameter
ablation.

**Expected:** same Apertus-faithful regime as §2.1 (LR 1.1e-5, β3=0.99, Goldfish, full parameter table inherited), *plus two interventions specifically for extension arms*:

**Embedding-only stabilization** (v0.12 §8.3; EEVE-style). First N tokens train only the new E and U rows; everything else frozen. Then unfreeze for full-parameter training.

The right value of N is underspecified. Krikri used 5B tokens of embedding-only training before unfreezing. EEVE used staged freezing across multiple phases (smaller per-phase). My earlier v0.12 working answer was 0.5–1B. **Given Krikri's 5B precedent and EEVE's staged structure, 0.5–1B is a lower bound, not a recommendation. The plausible range is 0.5B to ~5B, with the upper end matching Krikri.** Open question Q3.4.1 below.

**Differential LR on new-token rows** (v0.12 §8.4). New rows of E and U get a multiplier on the base LR; everything else stays at base LR (1.1e-5).

Working starting multiplier: **5× base LR (i.e., 5.5e-5 on new rows)**, with 10× contingent on diagnostic signals (new-row norms healthy, output logit calibration stable, update norms not spiking). This is the reviewer-corrected version of v0.12's 5–10× recommendation: start conservative, escalate if safe. Apertus's untied embeddings mean E and U behave differently; the multiplier may need to be set separately for the two — see Q3.4.2.

### (3.2) Sub-task: TD layer choice

Layer 11 was the bakeoff's TD choice. The TD paper itself reports that the optimal target layer is *not necessarily the final layer* and that performance can degrade near the final layers. **Layer 11 was a hypothesis in the bakeoff, not a settled fact.**

Before any TD CPT rerun, do a cheap TD layer sweep:

- Layers 4, 8, 11, 16, 20 (suggested set; agent can refine).
- Evaluate on: static embedding geometry (row norms, neighbor sanity), held-out Greek NLL after only the embedding-row training phase, early stabilization curve at 100M–300M tokens.
- Cost: each is initialization + brief training; well under 1B tokens total.

Pick best layer before launching the full TD extension run.

#### (3.2.1) Layer-11 evidence audit (added 2026-06-01)

A retrospective audit of *why* layer 11 was the bakeoff choice. Recording
this because it surfaced as a planning-agent question and the answer is
weaker than the name "layer 11" suggests.

**Origin: heuristic, not empirical.** Layer 11 = `ceil(num_hidden_layers / 3) = ceil(32 / 3) = 11`.
This "one-third depth" is the **TD package's README suggestion**, not
the TD paper's recommendation. The TD paper's actual default is
`target_layer = -1` (last layer = layer 32). Paper §5.3 explicitly says
last-layer is "a principled choice, as it guarantees that no subtoken
interactions that are only modeled in later layers are excluded from
the objective." We deviated from the paper default based on the package
README's parenthetical hint.

**What was planned but not executed.** `subprojects/03_apertus_extension_and_embedding_adaptation/TOKEN_DISTILLATION_PLAN.md §16, §6.1`:

- Candidate A: `target_layer = -1` (paper default).
- Candidate B: `target_layer = 11` (package README).
- Candidate C (optional): `L*` from a logit-lens/tuned-lens probe.

Only Candidate B (layer 11) was actually trained. Candidate A was
never run; the logit-lens / tuned-lens probe was never run; only a
1,966.9 s timing measurement on layer 11 was recorded, as a compute-
cost pilot, not a quality comparison.

**Implication for Task 2.**

- Run the deferred layer sweep as the first GPU step in Task 2.
- 5 candidate layers × ~0.5 B tokens each (Path-A geometry) ≈
  ~125 GPU-h total. Methodology: same as the Path-A probe — single
  checkpoint at iter 119, single sidecar fan-out, paired CI vs
  Vanilla-Path-A baseline.
- Add the logit-lens probe as a cheap diagnostic (runs in minutes per
  `TOKEN_DISTILLATION_PLAN.md §6.1`); if `L*` clusters around a layer
  not in the standard sweep, include it as a sixth candidate.
- Document the chosen layer with full provenance in Task-2 v1.x.

### (3.3) Sub-task: embedding-only stabilization duration

Per §3.1, the right N for embedding-only training is between 0.5B (my earlier guess) and 5B (Krikri's precedent). This is a real research question, not a defaulted decision.

**Plausible decision approach:** track the new-row metrics (norm distribution, pairwise distances, output logit calibration) during embedding-only training and unfreeze when they stabilize — rather than fix a token count in advance. This is more diagnostic than thresholded.

Agent task: design the stabilization detection criterion. See also Q3.4.1.

### (3.4) Open questions for agents to address

**Q3.4.1 — Embedding-only training duration.**

Krikri used 5B tokens of embedding-only training before unfreezing. EEVE used staged freezing (input embeddings → output embeddings → adapters → full) without a single duration value. My v0.12 plan suggested 0.5–1B.

Agent task: read Krikri (arXiv 2505.13772) and EEVE (Kim et al. 2024) for the rationale behind their stabilization durations. Specifically: did they ablate the duration, or did 5B / staged just work? Develop a stabilization criterion (norm convergence, logit calibration, distance stability) that doesn't require pre-committing to a token count.

**Q3.4.2 — Differential LR multiplier for E vs U rows separately.**

Apertus's untied embeddings have different statistics for E and U (U norms ~25% below E). A single multiplier for both may be wrong. The output head U is also more sensitive — bad U row values can destabilize logits even if E rows look geometrically reasonable.

Agent task: determine whether to set separate multipliers for E and U. Working defaults if separation is needed: 5× on E, 3× on U (lower because more sensitive). Validate against the TD paper's output-row handling and the reviewer's E-vs-U separation argument.

**Q3.4.3 — TD slope extrapolation.**

TD's native MCQ trajectory at 5B is +0.0081/1.5B (3.5B → 5B). Three data points are too few for reliable extrapolation. *Does TD's slope flatten further past 5B, or stay roughly linear?*

Agent task: look for TD or Token Distillation paper trajectories at >5B tokens. If TD slopes typically flatten by some characteristic budget, extending to 10B+ may not catch Vanilla. If slopes stay linear, the extrapolation has more weight.

**Q3.4.4 — Vanilla-vs-extension crossover at 8B scale: literature.**

To my knowledge, no clean published study measures the trajectory crossover between Vanilla CPT and CPT-with-vocabulary-extension on a target language, at 8B model scale, as a function of CPT token count. EEVE reports static endpoints. Krikri did both but doesn't isolate. Chinese-LLaMA scaled CPT but didn't run Vanilla counterfactual.

Agent task: literature pass; report what's known about crossover trajectories specifically.

**Q3.4.5 — Theoretical case for crossover.**

Why expect vocabulary extension to produce better downstream Greek (not just efficiency) under sufficient CPT? Arguments for: more efficient tokenization = more "real Greek content" per training step; per-token information density closer to model's optimization target; reduced fertility = longer effective Greek context within 4096-token window. Counter-argument: new tokens need to acquire trained-quality representations from scratch, consuming CPT tokens that aren't available for refining existing capability. 17K new tokens × ~100K firing exposures = 1.7B token-firings of opportunity cost.

Agent task: which arguments have empirical support in the literature?

**Q3.4.6 — Comparison structure under finite compute.**

Two options:
(a) **Equal-token.** Vanilla and Extension at the same total CPT tokens; compare at matched checkpoints. Clean methodology but ignores extension's efficiency value.
(b) **Equal-capability.** Vanilla to the budget that reaches Apertus-Base on native MCQ (if reachable). Extension to more tokens. Compare capability-per-inference-cost. More honest about purpose, harder to run cleanly.

Working default: **(a) equal-token**, because it's clean and inference-efficiency can be quantified post-hoc from fertility numbers. Agent should validate.

**Q3.4.7 — BPC vs native MCQ divergence.**

The bakeoff showed BPC and native MCQ moving in different directions across arms. Does this persist under the corrected regime (Task 1 will partially answer this)? If they continue to diverge, we need to decide which metric is primary for the production decision.

Agent task: empirical (will be answered by Task 1's data).

**Q3.4.8 — Diagnostic tools for trajectory analysis.**

What should be measured to understand extension trajectories? Candidate diagnostics (some from the reviewer):

- Per-token NLL over training time (available).
- Native MCQ with bootstrap CIs — needed to know whether the 4.47 pp Apertus-Base vs Vanilla-3.5B gap is statistically meaningful.
- KL-to-base on fixed probes (Greek, English, replay, code, math).
- New-token firing count histograms — mean exposure isn't enough; rare new tokens may stay dead.
- E and U row norm trajectories *separately* — Apertus has untied embeddings.
- Pairwise distances among new-token rows during training — catches "all new rows stay similar" failure.
- Output-logit calibration for new tokens — track whether new tokens are over/under-produced.
- Per-task native MCQ breakdown (GreekMMLU vs Medical vs ASEP vs Plutus) — different tasks reveal different failure modes.
- Per-language retention (English / French / German / Russian) + code/math.
- Goldfish hash uniformity over extended vocabulary (production-blocking if Goldfish + extension).

Agent task: select 5–7 diagnostics that should be standard for trajectory analysis. Build infrastructure where not already present.

**Q3.4.9 — Apertus paper does not address vocabulary extension.**

All extension-specific design choices (per-row LR, stabilization, init method, vocab size) lack paper backing. We are extending the paper, not following it.

Agent task: for each extension-specific choice, identify the strongest published reference and the strongest counter-argument. Krikri (the most-similar published precedent) is partially documented; EEVE's staged-freezing rationale is well-documented; TD's E-vs-U handling is documented in its own paper. Synthesize these references for v1.1.

**Q3.4.10 — Positional geometry: continue Path B for bakeoff comparability, or switch to Path A for clean Apertus-Base comparison?**

The bakeoff and the 04 Task-1 Vanilla CPT both train on Path B
(`rope_theta=500000`, `max_position=4096`, no scaling — Apertus paper
§2.3 initial-pretraining geometry). The released `swiss-ai/Apertus-8B-2509`
ships on Path A (`rope_theta=12000000`, `max_position=65536`, llama3
scaling — Apertus paper §2.5 long-context extension). Empirical evidence
from Task-1 iter 477 (see §2.1 "Training-time positional geometry
override (Path B)") shows:

- Path-B inference on Path-A weights perturbs the base (matched-config
  Apertus-Base BPB = 1.22, native MCQ headline = 0.4272 — 5.5 pp drop
  on MCQ).
- After ~1 B tokens of Path-B CPT, the model re-adapts and produces
  useful Greek-side improvement; iter 477 (2 B) sits at Apertus-Base
  Path-A point estimate on MCQ.
- The cleanest cross-arm signal for Task 1 is Path-B vs Path-B
  (iter 477 vs bakeoff Vanilla-2B = +4.65 pp outside V4 CI). There is
  no bakeoff-Path-A baseline.

**Working position for Task 2: switch to Path A.** Rationale:

- Path A is the inherited geometry — no rope re-adaptation cost in the
  first ~1 B tokens. Free signal-to-noise improvement at small budgets.
- Path A enables clean apples-to-apples comparison vs Apertus-Base
  without the matched-config workaround (which is itself perturbed —
  not a valid baseline).
- Cost of switching: there is no bakeoff-Path-A counterpart, so the
  bakeoff-comparability argument that justified Path B for Task 1 does
  not apply to Task 2's extension question. Task 2's primary comparison
  is *extension vs Vanilla under the same regime + geometry*, not
  *extension vs bakeoff arms*.
- Cost of keeping Path B: ~1 B tokens of rope re-adaptation per arm
  in every Task-2 experiment, plus the geometry-confound caveat for
  any Apertus-Base comparison.

Concrete Path-A training settings for Task 2 (override of §2.1's Path-B
table):

| Parameter | Task 2 Path-A value | Source |
|---|---|---|
| RoPE θ | 12,000,000 | Apertus paper §2.5 + released base `config.json` |
| Max position | 65,536 | Apertus paper §2.5 + released base `config.json` |
| RoPE scaling | llama3 (factor=8.0, original_max_position_embeddings=8192, low_freq_factor=1.0, high_freq_factor=4.0) | Apertus paper §2.5 + released base `config.json` |
| Sequence length | 4,096 (training only; geometry supports longer) | Tractability; matches Task 1's training-time sequence |

Agent task for Task 2 v1.1: confirm or revise this Path-A switch. If
confirming, lock the Path-A geometry block above as the Task 2 baseline
and remove the matched-config baseline from the 5 B report's primary
comparison (it remains as a diagnostic, not a baseline). If revising
back to Path B, document why bakeoff-comparability matters more than
Apertus-Base comparability for the extension question — and note that
no published bakeoff-Path-A baseline exists either way.

---

## (4) Task 3: Data mix design

**Status: working draft, updated with actual dataset counts.** Specific schedule percentages still need final decisions, but the structural framework is now anchored on real numbers.

### (4.1) Greek staging structure

The Greek corpus categories from §1.4 — repeated here for convenience:

| Category | Tokens | Share |
|---|---:|---:|
| HPLT (web, filtered) | 44.20B | 71.7% |
| openarchives.gr | 9.08B | 14.7% |
| High-register curated | ~6.50B | 10.6% |
| OpenSubtitles | 1.17B | 1.9% |
| Legal & parliamentary | 0.51B | 0.8% |
| Literary / historical | 0.15B | 0.2% |

Working sketch for the staging (production CPT, not Task 1):

**The 6.5B high-register pool is the cooldown anchor.** didaktorika (5.07B) is ~78% of this pool; Pergamos (0.82B), finewiki (0.25B), Kallipos (0.19B), openbook_gr (0.15B), and Sxolika_vivlia (0.01B) are the minor additions. Whatever cooldown structure we use, "where does didaktorika go" is essentially the central staging decision.

**The 44.2B HPLT is the bulk-phase workhorse** by an order of magnitude over anything else.

**openarchives.gr (9.08B) is the genuine unknown.** It's 14.7% of the corpus — too large to ignore — and its register/quality determines whether it joins the bulk phase, the cooldown phase, or sits between. See Q4.4.7.

**OpenSubtitles (1.17B) is register-orthogonal** to the curated/web distinction. Probably belongs in the bulk phase rather than cooldown.

**Legal/parliamentary (0.51B) and literary/historical (0.15B) are too small to stage separately.** Mix into bulk phase uniformly.

Two plausible production-CPT staging structures:

**Structure A — narrow cooldown (~10% of training):**

1. **Bulk phase (~90% of CPT):** HPLT + openarchives.gr + OpenSubtitles + legal + literary, uniform mix.
2. **Cooldown phase (~10%, final ~6–8B tokens):** the 6.5B high-register pool, didaktorika-dominant.

Sized so each high-register token is seen approximately once during cooldown. Tight but feasible.

**Structure B — wider cooldown with upsampling (~15–20% of training):**

1. **Bulk phase (~80–85% of CPT):** HPLT + OpenSubtitles + legal + literary.
2. **Cooldown phase (~15–20%, ~12–18B tokens):** high-register pool (6.5B) plus openarchives.gr (9.08B), with high-register tokens upsampled 1.5–2× to keep them dominant.

This matches Apertus's own cooldown approach (data upsampling during cooldown is documented in paper §3.3). Requires deciding openarchives.gr is cooldown-quality content.

Both structures put the high-register acquisition under decaying LR — which matches Apertus's high-quality-data-last pattern and naturally absorbs the AdEMAMix slow-gradient transition shock (see §4.2).

**Within-cooldown ordering matters.** didaktorika is so large relative to the other high-register sources that the cooldown's character is mostly determined by didaktorika's properties. Pergamos / Kallipos / finewiki could be saved for the very end (final 10–20% of cooldown) as a quality-peak, or mixed uniformly with didaktorika throughout cooldown — open question.

This staging is NOT applied in Task 1 (the regime experiment); Task 1 uses uniform mixing to isolate the regime question. The staged-mix design is for production CPT and possibly for Task 2's extension experiments.

### (4.2) Optimizer interaction with stage transitions — AdEMAMix slow gradient

AdEMAMix maintains two momentum terms:

- Fast EMA with β1 = 0.9: effective memory ≈ 10 steps.
- **Slow EMA with β3 = 0.99: effective memory ≈ 100 steps.** (At β3=0.9999, ~10,000 steps; we're using 0.99.)

At our batch size of 4.2M tokens / step:

- Fast EMA memory ≈ 42M tokens.
- **Slow EMA memory ≈ 420M tokens.**

**Implication for staged data transitions:** if the within-Greek mix shifts abruptly at some token count (e.g., HPLT-dominant → GlossAPI-dominant at 80% of budget), the slow EMA momentum spends ~420M tokens still reflecting HPLT gradient statistics while the fast EMA and the loss signal switch to GlossAPI. This is a transient distribution-shift shock that can:

1. Mute the cooldown stage's effective LR (slow EMA momentum points in the "wrong" direction).
2. Produce a brief loss spike or instability.
3. In the worst case, undo some of the cooldown stage's intended adaptation.

The mitigations that exist in the literature:

**(a) Gradual mixing ramp** — smooth the HPLT → GlossAPI transition over several hundred million tokens instead of switching at a step. The slow EMA adapts continuously rather than chasing a discontinuity. This is Apertus's own approach in their staged pretraining (per the paper, stages have data composition changes but typically with rebalancing rather than hard switches).

**(b) Concurrent LR decay** — combine the data transition with WSD's 1-sqrt cooldown. The slow EMA's lingering momentum has less effect because every momentum term is being multiplied by a decaying LR. This is the standard "do everything at cooldown" approach.

**(c) β3 reset at transition** — at the stage boundary, reset slow EMA state (or briefly increase β3 warmup again). Loses some of AdEMAMix's benefit but produces a cleaner adaptation. The AdEMAMix paper mentions α/β3 warmup at training start but doesn't discuss mid-training resets; this would be a deviation from documented practice.

**(d) Larger batch at transition** — temporarily increase batch size during the transition window so per-batch gradient noise is reduced and the slow EMA's contribution to total update is dampened. Apertus paper §2.3 notes batch-size increase acts similarly to LR decrease.

Working default: **combine (a) and (b)** — gradual mix ramp concurrent with WSD cooldown. This matches Apertus's own documented practice. (c) and (d) are escape hatches if (a)+(b) prove insufficient.

This is a real consideration that the bakeoff didn't face (uniform mix, no stage transitions). It only becomes relevant when staging is introduced.

### (4.3) Replay composition

The 24% non-Greek share is currently a single number from the bakeoff. The bakeoff's multilingual retention loss (varied by arm; TD held English better than Vanilla, all arms lost some) suggests **replay composition matters**, not just replay share.

Open questions raised by the bakeoff data:

- Is 24% replay sufficient for our Greek volume (~70% of training), or does sustained Greek emphasis at this share force forgetting? Higher replay share (30-40%) is plausible if retention loss is observed.
- What's the right composition within the 24%? Krikri used a heavy English replay (21B vs 56.7B Greek) plus 5.5B Greek-English parallel for catastrophic-forgetting mitigation. Our 24% likely doesn't include parallel data.
- Should replay composition be staged alongside the Greek mix, or held constant? If GlossAPI is at cooldown, should replay similarly shift toward high-quality content?

**Working position:** hold replay at 24% with bakeoff composition for Task 1 (isolate the regime question); revisit replay composition as part of Task 2 / production design once Task 1 shows whether the new regime preserves retention.

### (4.4) Open questions for agents to address

**Q4.4.1 — Exact within-Greek schedule.**

Dataset counts are now known (see §1.4 and §4.1). The remaining decisions:

1. Choose between Structure A (~10% cooldown, ~6B) and Structure B (~15–20% cooldown, ~12–18B with upsampling). Depends on Q4.4.7 (openarchives.gr placement) and on whether upsampling high-register tokens during cooldown is preferred over diluting them with openarchives.gr.
2. Decide bulk-phase mixing proportions within HPLT + openarchives.gr + OpenSubtitles + legal + literary. Uniform across the bulk phase is the simplest default.
3. Decide within-cooldown ordering: didaktorika throughout vs Pergamos/Kallipos/finewiki saved for the very end as a quality-peak.

Agent task: design the specific mix percentage schedule for both Structure A and Structure B, presenting them as alternatives for a final choice. Validate against Apertus's stage-composition pattern (§3.3 of paper).

**Q4.4.2 — Stage transition mechanism.**

The §4.2 working default is gradual mix ramp + WSD cooldown. The exact ramp duration is unspecified.

Agent task: how long should the ramp be? The slow EMA memory is ~420M tokens at β3=0.99; the ramp should plausibly be ≥ 2-3× that to avoid lingering momentum effects. Working answer: **ramp over ~1-1.5B tokens centered on the start of cooldown.** Validate against literature on optimizer-data interaction during distribution shifts.

**Q4.4.3 — Replay composition and share for production CPT.**

Bakeoff retention data + Task 1 retention data jointly inform this. With 61.6B Greek = 70% of training, replay at 24% = ~21B non-Greek tokens — structurally comparable to Krikri's 21B English replay.

Agent task: after Task 1 results, evaluate whether 24% is sufficient, what composition should be, and whether replay should be staged.

**Q4.4.4 — Parallel data inclusion.**

Krikri included 5.5B Greek-English parallel data specifically for cross-lingual retention. Our replay doesn't currently include this. Agent task: evaluate whether parallel data is worth ~5% of budget — if so, where does it come from (OPUS? curated?), and at what stage should it appear?

**Q4.4.5 — Dedup against Apertus pretraining.**

The 44B HPLT-Greek may overlap with Apertus's pretraining Greek content (FineWeb-2-HQ ell_Grek, ~6.4B tokens — same kind of crawl, similar filters). Document-level dedup against the pretraining corpus is computationally expensive at 44B-token scale. **Working position from earlier project discussion: scope dedup to Greek-against-Greek only (deduplicate the 61.6B Greek corpus internally; do not dedup against the multilingual pretraining corpus).** Agent should confirm this is still the right scope.

**Q4.4.6 — Within-high-register quality stratification.**

The 6.5B high-register pool has its own internal structure. didaktorika (5.07B) is ~78% of the pool. Pergamos (0.82B), finewiki (0.25B), Kallipos (0.19B), openbook_gr (0.15B), Sxolika_vivlia (0.01B) are the smaller additions.

The didaktorika subset itself is heterogeneous — PhD theses span many domains and quality levels. Within the cooldown, is there value in stratifying further (e.g., by department, by year, by length, by metadata-derived quality signal)?

Agent task: design within-high-register quality stratification for the cooldown. Default working answer: **treat didaktorika as a uniform block; place Pergamos + Kallipos + finewiki + openbook + Sxolika at the very end of cooldown (final ~1.5B tokens) as a curated quality-peak.** Validate or revise.

**Q4.4.7 — openarchives.gr placement (NEW).**

openarchives.gr at 9.08B is the largest single source after HPLT and was not in our previous mental model. It's institutional repository content from Greek universities and research bodies. The placement decision depends on its register / quality, which we haven't sampled in detail:

- If it's roughly thesis-adjacent quality (academic, formal register): include it in the cooldown pool, growing it from 6.5B to ~15.6B. This gives Structure B more breathing room.
- If it's roughly web-adjacent (variable quality, less curated): keep it in the bulk phase with HPLT.
- If it's genuinely mid-register (institutional content but not all academic): consider a transitional stage between bulk and cooldown — first half of training is HPLT-only, middle is HPLT+openarchives, final cooldown is high-register-only.

Agent task: sample openarchives.gr content (a few hundred documents across different sub-collections if there are any) and characterize its register/quality. Recommend placement based on the sampling. Until this is done, plans should treat openarchives.gr's placement as the largest single open variable in the corpus design.

---

## (5) Commitments

1. **Task 1 (regime diagnostic experiment)** is the next experiment. Single-arm Vanilla, full parameter spec in §2.1, headline metric native MCQ (§1.4 eval), checkpoints at 0.5B, 1B, 2B, 3.5B, 5B, 7B, 10B (Q2.4.1), and a Codex adversarial critique at each checkpoint after eval artifacts exist.

2. The question Task 1 answers is **whether the bakeoff's regime caused the Greek degradation across all arms, including Vanilla** (§2.2 diagnostic framing).

3. **Native Greek MCQ aggregate** is the headline Greek metric. MT-derived Greek tasks are demoted to diagnostics.

4. **Task 2 (extension experiment)** is deferred until Task 1's result is in hand. Its design depends on what Task 1 reveals.

5. **Task 3 (data mix design)** is a parallel work item. Dataset counts now known (§1.4, §4.1). Working sketch: bulk phase = HPLT + openarchives.gr + OpenSubtitles + minor sources; cooldown = 6.5B high-register pool with didaktorika as the anchor. Gradual ramp + WSD cooldown to absorb stage transitions safely under AdEMAMix.

6. Open questions Q2.4.1–Q2.4.8, Q3.4.1–Q3.4.9, Q4.4.1–Q4.4.7 are **passed to follow-up iterations and other agents**. Q4.4.7 (openarchives.gr placement) is the largest single open variable in the corpus design.

---

## (6) Non-commitments — what v1.0 deliberately does not address

- **Decision rule thresholds** (specific token-conditional stop conditions). Per Fivos's explicit guidance, these are decided post-result, not pre-committed.
- **Eval suite expansion** beyond the current native suite (OYXOY, GreekBarBench, civics, lyceum math are cached but not yet scored — see native suite report's "Remaining caveats").
- **Krikri positioning narrative** (the v0.12 Node 6 question; writeup-time decision).
- **Whether to revisit BPE cutoff** (the C3 production-side sweep on `{10240, 15360, 20480, 25600}` is a separate decision tracked elsewhere).
- **Production CPT itself.** This plan is the experimental precursor; production CPT design depends on outcomes here.
- **Production-blocking verifications.** Several pre-existing open items from `cpt_plan.md` v0.7 remain unresolved and gate production CPT (not Task 1):
  - **V1 — decontamination** of the Greek CPT corpus against native Greek benchmark prompts.
  - **V4 — baseline variance** with bootstrap CIs on the native MCQ aggregate. Needed to know whether the 4.47 pp Apertus-Base vs Vanilla-3.5B gap is meaningful or noise-dominated.
  - **V8 — Goldfish hash uniformity** check on the extended vocabulary. Required if Goldfish is used in extension arms.
  - **R17 — production patch** via `patch_apertus_extras.py` for extended-tokenizer artifact handling.

---

## (7) How to read this document

If you are an agent picking this up:

- **§1** is settled history. Treat the native Greek suite as the corrected headline.
- **§2 (Task 1)** is the next concrete experiment. The full parameter spec is in §2.1. The diagnostic framing is in §2.2 and the failure-mode mapping in §2.3.
- **§3 (Task 2)** is the production extension question, framed but not designed. Sub-tasks (TD layer sweep, stabilization criterion) and open questions are scoped.
- **§4 (Task 3)** is parallel work on the data mix. Schedule specifics pending dataset counts; structural framework and the AdEMAMix slow-gradient interaction are scoped.
- **§5** lists commitments. **§6** lists non-commitments.

Each Q* is a research prompt. Treat them as the work list, not as rhetorical hedges. **v1.1** should answer the Q2.4.x questions (regime experiment readiness). **v1.2** should answer Q3.4.x questions or substantially narrow them. **Q4.4.x answers depend on dataset counts**, which Fivos is producing.

The doc is structured to be appended to, not rewritten. When you answer a Q*, leave the question framing visible and add an "Answer (as of [date]):" block underneath. This preserves the history of what changed when.

---

## (8) References and artifacts

### Published papers

- **Apertus technical report:** *Apertus: Democratizing Open and Compliant LLMs for Global Language Environments* — arXiv 2509.14233. Primary reference for architecture, AdEMAMix optimizer, Goldfish loss, WSD schedule, long-context extension recipe used as our CPT analog.
- **Krikri:** *Krikri: Advancing Open Large Language Models for Greek* (ILSP) — arXiv 2505.13772. Closest published analog: Greek CPT on Llama-3.1-8B with vocabulary extension, ~83B CPT tokens, +10.8% Greek improvement. Used 5B-token embedding-only training before unfreezing.
- **Token Distillation:** Dobler & de Melo — arXiv 2505.20133, repo at https://github.com/konstantinjdobler/token-distillation. TD paper notes optimal target layer is not necessarily the last layer.
- **EEVE:** Kim et al. 2024 — staged Korean adaptation of Llama-2 (input embeddings → output embeddings → adapters → full); source for v0.12 §8.3 staged training. Explicitly argues input + output trained simultaneously can complicate convergence.
- **Swallow:** *Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities* — OpenReview TQdd1VhWbe. Reports monotonic improvement up to 100B Japanese CPT tokens on Llama-2; vocabulary expansion improved token efficiency without broad degradation except for summarization.
- **Reuse, Don't Retrain:** *A Recipe for Continued Pretraining of Language Models* — arXiv 2407.07263. Argues LR schedule and data distribution are first-order CPT levers; recommends beginning closer to original pretraining distribution and shifting later toward target abilities.
- **AdEMAMix:** Pagliardini et al. 2025. The optimizer used by Apertus. Documents α/β3 warmup and the sensitivity to long-momentum decay.

### Model checkpoints

- `swiss-ai/Apertus-8B-2509` — base model.
- `ilsp/Llama-Krikri-8B-Base` / `-Instruct` — for capability comparison at Greek-CPT-comparable scale.

### Internal artifacts

**Diagnostic and analysis runs:**
- `runs/apertus_greek_diagnostic_20260511_v2/` — Phase A norm diagnostic.
- `runs/apertus_greek_phase_b_v4_20260512/` — Phase B v4 behavioral NLL.
- `runs/apertus_embedding_init_test_20260512/` — embedding init LOO test.

**Bakeoff results:**
- `03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md`

**Native Greek suite (2026-05-26, the corrected headline):**
- `/capstor/scratch/cscs/fffoivos/runs/eval/native_greek_suite_20260526/` — root.
- `summary/NATIVE_GREEK_SUITE_SUMMARY.md` — corrected Greek headline report.
- `summary/native_mcq_aggregate.csv` — §1.1 table source.
- `summary/native_mcq_per_task.csv` — per-task breakdown.

**Plan documents:**
- `apertus_greek_tokenizer_extension.md` v0.12 — parent project plan; this experimental plan operationalizes its §3 decision nodes.
- `cpt_plan.md` v0.7 — operational CPT plan with V1–V16 verifications, replay corpus structure, B1 training mix definition.
- `apertus_fidelity_checklist.md` — 21 items matching Apertus pretraining; documents current deviations.
- `TOKEN_DISTILLATION_PLAN.md`, `RISKS.md`, `AUDIT_FINDINGS.md` — supporting material.

**Tokenizer / dataset documentation:**
- `docs/C3_CONVERGENCE.md`, `docs/C3_CUTOFF_REPORT.md`, `docs/C3_TRAINING_DATASETS.md` — BPE cutoff analysis and training dataset reports.

### Read order if you're new to the project

1. `BAKEOFF_FINAL_RESULTS_20260526.md` — what was run, with the original (now-superseded) reading.
2. `NATIVE_GREEK_SUITE_SUMMARY.md` — the corrected Greek-side reading.
3. This document — what we're doing next.
4. `apertus_greek_tokenizer_extension.md` v0.12 §3 (decision nodes) — for the project-level decision structure this plan operationalizes.
5. Apertus paper §2.3, §2.5, and Appendix C — for the hyperparameter recipe being applied and the long-context continuation precedent.
