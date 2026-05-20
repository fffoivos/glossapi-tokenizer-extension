# CPT Plan v0.6 — Answers to the Open Questions

*2026-05-20. Companion to [`cpt_plan.md`](cpt_plan.md) (v0.6). Works
through every question in §10 (decisions pending from Fivos), §11
(lookups), and §12 (verifications), answering what can be answered
from existing artifacts, flagging what genuinely needs user input,
and pointing at what's still a fetchable lookup.*

> Status legend per question:
> - **ANSWERED** — answer derived here from on-disk artifacts.
> - **PARTIAL** — partial answer based on what's known; rest is a fetchable lookup or a value judgment.
> - **NEEDS USER** — value judgment / project-direction call only Fivos can make.
> - **NEEDS LOOKUP** — concrete fact, fetchable from the Apertus tech report or HF.

## §10 — Decisions pending from Fivos

### Q A1. Capability targets — **NEEDS USER**

The plan lists 5 options: (a) balanced register-aware Greek assistant, (b) academic/digital-humanities Greek, (c) polytonic-strong classical generator, (d) modern Greek conversational, (e) other.

**Prior framing in `experiments_plan.md` v0.12 §1** *(now superseded by `cpt_plan.md`)* explicitly said: *"This project does not currently have a specific deployment target. The goal is broadly 'improve Greek' ... with one hard constraint: preserve Apertus's multilingual character."*

If you keep that framing, the closest of the v0.6 options is **(a) balanced register-aware Greek assistant** + non-regression on multilingual capability. If the polytonic +5,120 extension is intended to be a first-class capability rather than a stacked downstream specialization, that pushes toward **(c) polytonic-strong** — but that conflicts with the current bracketing of polytonic from the bakeoff.

Recommendation: **(a) balanced register-aware**, with polytonic as a *secondary objective* exercised via the §6.2 polytonic continuation eval rather than a primary capability target. This matches the spirit of `experiments_plan.md` and is consistent with how `cpt_plan.md` actually treats polytonic in §1 (mentioned in the param math but flagged only via the §3.1 polytonic exposure metrics, not in the main capability list).

### Q A2. Total token budget for CPT (post-init) — **NEEDS USER** (default already reasonable)

Plan working range: **10–20 B tokens**. Prior `experiments_plan.md` §8.7 had: pilot 10 B / arm, main 20–40 B for the winner. v0.6's 10–20 B post-init is slightly conservative relative to that. Either is defensible.

Constraint to weigh: per [`CURRICULUM_AND_INIT_CORPUS.md § 1`](03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md#1-should-init-experiments-use-the-mixed-set-or-the-apertus-fresh-only-set), the post-internal-dedup Greek pool is **~26-38 B tokens** at our extended-tokenizer fertility. A 20 B budget at 70 % Greek = 14 B Greek tokens consumed — well under one epoch. A 30 B budget at 70 % Greek = 21 B Greek — about one epoch. 40 B would require repeat passes.

Recommendation: **start at 15 B; expand to 20 B if early eval looks under-saturated; reserve 30+ B for after the winner is locked.**

### Q A3. Compute timeline / deadline — **NEEDS USER**

No prior artifact constrains this. Plan implies "no specific deadline."

### Q A4. Stakeholders / downstream consumers — **NEEDS USER**

Same as A1 — `experiments_plan.md` framing was "no specific deployment target." If that's still true, A4 is "the Swiss-AI Apertus team + the GlossAPI / EELLAK community." If anyone else's downstream depends on this, name them.

### Q A5. Colleague sign-off on shuffled-bulk + annealing — **NEEDS USER**

Specifically `Apertus_plan.md` (Xronopoulos → Petros Stefaneas) and `p-skarvelis` (the existing CPT-runner on a0140). Their current pipeline is **closer to the v0.6 shape** than v0.5 was: they use cosine LR (≠ WSD but mid-training arXiv ≈ Oct 2025 supports WSD over cosine for CPT), 90/10 Greek/non-Greek (vs v0.6's 70/30), and don't anneal explicitly. The shape v0.6 proposes is a fork from their pipeline, not a continuation. **Worth a sync meeting before launch.**

### Q A6. Specific downstream tasks — **NEEDS USER**

Translation, OCR post-correction, summarization, dialect detection, polytonic generation. The 03_3 `CURRICULUM_AND_INIT_CORPUS.md` evaluation list adds: GreekMMLU, Belebele, Medical MCQA, OYXOY, GreekBarBench, GreekSUM. These map to: (1) translation = covered by `ilsp/flores200_*`; (2) OCR post-correction = NOT covered, would need custom eval; (3) summarization = `IMISLab/GreekSUM` (gated); (4) dialect detection = NOT covered; (5) polytonic generation = covered by v0.6 §6.2 custom eval.

Recommendation: confirm OCR-post-correction is in or out of scope. If in, we need an Anemi-style eval slice — that's a 1-week construction job.

### Q A7. Team structure — **NEEDS USER**

### Q B1. Outer target/replay split — **NEEDS USER**, *but our prior planning diverges from v0.6 default here*

- v0.6 default: **70/30** (Greek/non-Greek)
- `Apertus_plan.md`: **90/10** (more aggressive Greek)
- `p-skarvelis` actual runs: **90/10** (`greek_probability: 0.9` in `run_config.json`)
- `CURRICULUM_AND_INIT_CORPUS.md`: **85/15** (cautious middle)

The v0.6 70/30 is **more replay-heavy** than any prior position. Justification in v0.6 §4.1 cites Sailor2, AMD Finnish, SEA-LION v3, EstLLM, Racka. Those are post-CPT-stability-aware recipes for languages with much smaller Apertus pretraining shares than Greek's 0.023 % — they're correct that more replay is safer in those cases.

**My read:** start at **80/20** as a compromise that respects v0.6's caution about catastrophic forgetting without going as aggressive on Greek as p-skarvelis's 90/10. If V4 baselines show the 80/20 ratio causing retention issues, drop to 70/30; if not, hold or push higher Greek.

### Q B2. Code share — **NEEDS USER** (default (b) ≈ 4 % is reasonable)

v0.6's range: 0 % (Sailor2) to 20 % (SEA-LION v3). Apertus pretraining had ~2 % code share. Default (b) = 4 % is mid-range and roughly doubles Apertus's baseline.

Recommendation: **(b) 4 %**, drawn from `bigcode/starcoderdata` to match Apertus's source.

### Q B3. Anneal composition priority — **NEEDS USER** (default (d) balanced is good)

Tied to A1. Under "balanced register-aware Greek assistant" the v0.6 default (d) Balanced fits. Under "polytonic-strong" it'd shift to (c).

### Q B4. Loss objective for init bakeoff — **PARTIAL / RECOMMEND**

v0.6 default: NTP for bakeoff, Goldfish for production. **Recommendation: accept the default.** Reasoning:
- NTP for bakeoff gives a cleaner signal because Goldfish masking would interact with vocab extension (V8) and confound init-method comparison.
- Goldfish for production matches Apertus's pretraining recipe — memorization suppression is part of Apertus's compliance posture.

Caveat: V8 (Goldfish hash uniformity with extended vocab) needs to be verified before production. If V8 reveals a non-uniform mask distribution, fix or fall back to NTP-only for production.

### Q B5. Init experiment budget per variant — **PARTIAL / RECOMMEND 2 B**

v0.6 options: 1.5 B or 2 B. Default 2 B.

p-skarvelis's existing 1 B-token CPT runs ([STORAGE_AND_EXISTING_WORK.md §3](03_4_implementation_experiments/STORAGE_AND_EXISTING_WORK.md#3-existing-apertus-greek-work-in-a0140-p-skarvelis)) show loss curve still descending at 700 steps in the cosine schedule; the curve hasn't bottomed out, so 1.5 B may still be in transient noise. **2 B is the better default** for clean ReTok-vs-Centroid discrimination, at the cost of 50 % more compute.

This is **smaller than the 10 B/arm pilot** in earlier `experiments_plan.md` §8.7 — that's because the earlier 10 B was sized for **quality discrimination**, while v0.6's 2 B is sized purely for **init-method discrimination**. The new framing is the right shape for what the bakeoff actually tests.

Total bakeoff: 3 × 2 B = **6 B tokens across all three arms**. At 107 k tok/s on 4 nodes (per p-skarvelis's measured throughput), each arm = ~5.2 h wall; three arms in series ≈ 16 h; in parallel ≈ 5.2 h. Fits inside a 12 h `normal` allocation if run on 4 nodes per arm with 3 arms parallel.

### Q B6. Adaptation work prioritization — **PARTIAL / RECOMMEND ACCEPT**

v0.6's split (must-have vs deferrable) is sound. The must-have list:
- **G1** (Goldfish hash uniformity) — only if running Goldfish during bakeoff; if NTP-only per B4, can defer to production
- **H1** (BPC unit choice) — yes, 1 hour of utility code
- **I1** (NFC normalization on training text) — yes, ~2 hours
- **I2** (resize_token_embeddings with untied E/U) — yes, ~4 hours; **already partly verified locally** via the `build_and_verify_ship_tokenizer.py` script in 03_3
- **K1** (decontamination via NeMo Curator) — yes, gating; 3-5 days

The deferrable:
- **B** (FOCUS for polytonic) — skip per v0.6's own default (use pure subpiece mean)
- **E1** (density warmup) — skip for pilot; revisit if Phase 0 stress probe (5.5) shows the density mismatch is biting
- **J1** (vLLM/SGLang compatibility) — defer to post-winner-shipping

## §11 — Lookups pending

### Q C1. Apertus pretraining peak LR — **ANSWERED in v0.6**

v0.6 resolved this. Apertus 8B peak LR = 1.1e-4 (tech report Table 2). CPT peak range 1.1e-5 to 2.2e-5; default 1.5e-5.

### Q C2. Apertus optimizer hyperparameters — **NEEDS LOOKUP** (tech report §2.3 / B.4)

Specifically need: AdEMAMix β1, β2, α (EMA decay), weight decay. The Apertus tech report (arXiv:2509.14233 v2) §2.3 + Appendix B.4 has these. **Fetch as part of pre-launch checklist.**

### Q C3. Apertus per-language token shares — **PARTIAL** (we have an audit)

We have the May-13 vocab-attribution audit ([03_3 POLYTONIC_VOCAB_BUDGET_CHECK.md](03_3_cscs_experiments_kickoff/POLYTONIC_VOCAB_BUDGET_CHECK.md)) which captures per-language token counts as sampled at 1 B-cap from FineWeb-2 / FineWeb-2-HQ / Clean-Wikipedia / EuroParl. For the 24 v0.6 plan-languages:

| ISO + script | tokens (sampled) | source | tier in v0.6 |
|---|---:|---|---|
| `eng_Latn` | 1.005 B ★ | clean_wikipedia | T1 |
| `fra_Latn` | 1.007 B ★ | fineweb_2_hq | T1 |
| `deu_Latn` | 1.006 B ★ | fineweb_2_hq | T1 |
| `ita_Latn` | 1.005 B ★ | fineweb_2_hq | T1 |
| `spa_Latn` | 1.003 B ★ | fineweb_2_hq | T1 |
| `rus_Cyrl` | 1.004 B ★ | fineweb_2_hq | T1 |
| `arb_Arab` | 1.005 B ★ | fineweb_2_hq | T1 |
| `cmn_Hani` | 1.008 B ★ | fineweb_2_hq | T1 |
| `tur_Latn` | 1.006 B ★ | fineweb_2_hq | T2 |
| `bul_Cyrl` | 1.003 B ★ | fineweb_2 | T2 |
| `srp_Cyrl` | 1.008 B ★ | fineweb_2 | T2 |
| `ron_Latn` | 1.009 B ★ | fineweb_2 | T2 |
| `heb_Hebr` | 1.008 B ★ | fineweb_2 | T2 |
| `por_Latn` | 1.003 B ★ | fineweb_2_hq | T2 |
| `pol_Latn` | 1.005 B ★ | fineweb_2_hq | T2 |
| `nld_Latn` | 1.002 B ★ | fineweb_2_hq | T2 |
| `pes_Arab` | NOT FOUND | — | T2 (try `fas_Arab`) |
| `ukr_Cyrl` | 1.007 B ★ | fineweb_2 | T2 |
| `jpn_Jpan` | 1.003 B ★ | fineweb_2_hq | T2 |
| `lat_Latn` | 1.007 B ★ | fineweb_2 | T3 |
| `hye_Armn` | 1.006 B ★ | fineweb_2 | T3 |
| `kat_Geor` | 1.011 B ★ | fineweb_2 | T3 |
| `als_Latn` | 1.004 B ★ | fineweb_2 | T3 (Albanian) |
| `sqi_Latn` | NOT FOUND | — | T3 (alt code; same lang as `als_Latn`) |
| `mkd_Cyrl` | 1.002 B ★ | fineweb_2 | T3 |

★ = sample hit the 1 B cap → **actual corpus has more.**

**Note caveat**: Apertus's per-language pretraining share is *not* the same as our FW2 sample sizes. Apertus used FineWeb-2-HQ for 20 high-resource languages plus FineWeb-2 for everything else, with stage-specific sampling. The full per-stage breakdown is in tech report §3 / Appendix G — fetching that is the actual answer to Q C3.

**For the v0.6 Tier 3 weighting**: the plan's worry ("Tier 3 ≈ preservation aspiration, near-zero base exposure") is **less true than v0.6 assumes**. Latin, Armenian, Georgian, Albanian, Macedonian all have ≥1 B sample tokens in FineWeb-2. They're not zero-exposure; Apertus has at least some real coverage. Recommend treating Tier 3 as actual replay (not floor weight only).

### Q C4. Apertus Goldfish loss configuration — **NEEDS LOOKUP** (tech report §3.3)

Token masking rate, hash function. Needed for V8 verification.

### Q C5. Apertus tokenizer config — **ANSWERED in v0.6**

Mistral-Nemo `tekken` v3. v0.6 confirms.

### Q D1. Apertus Megatron-LM fork — **ANSWERED**

`swiss-ai/Megatron-LM` (github.com/swiss-ai/Megatron-LM), default branch. Updated 2026-05-18 (3 days before this writing). Companion: `swiss-ai/pretrain-code` describes itself as *"Pretraining codebase for Apertus models, based on Megatron-LM"*. The pair is the canonical Apertus training stack. Both already inventoried in [`03_4 ENVIRONMENT_AND_BENCHMARKS.md § 1.1`](03_4_implementation_experiments/ENVIRONMENT_AND_BENCHMARKS.md#11-training-infrastructure-pick-one-trunk).

### Q D2. FineWeb-2 Tier 3 language audit — **ANSWERED** (audit above for Q C3)

All five Tier 3 languages have ≥1 B sample tokens. **Their corpus is meaningful**, contradicting v0.6's "preservation aspiration" framing. Recommend giving Tier 3 actual share, not just floor weight.

### Q D3. Apertus intermediate checkpoints — **NEEDS LOOKUP** (HF)

Per v0.6 §11 Q D3, these would be on different HF branches of `swiss-ai/Apertus-8B-2509`. Worth checking with `huggingface_hub` API for the list of available revisions. Useful for the Llama-3-style annealing-as-quality-meter pattern.

## §12 — Verifications pending

| V# | Item | Status | Notes |
|---|---|---|---|
| V1 | Decontamination (gating) | **NOT DONE** | Highest priority; NeMo Curator pipeline. 3-5 days work. |
| V2 | Tokenizer extension forward pass test | **PARTIALLY DONE** | `build_and_verify_ship_tokenizer.py` in 03_3 already verifies `AutoTokenizer.from_pretrained()` loads both ship bundles cleanly. Full E + U resize forward pass not yet tested with a real model. |
| V3 | Dataloader state preservation probe | **NOT DONE** | Stop/resume at 100M; confirm next batch is 100M+1. ~1 hour on debug partition. |
| V4 | Run-to-run variance baseline + bootstrap CI calibration | **NOT DONE** | Run full eval suite on Apertus-8B base, bootstrap variance (1000 resamples). Sets §5.6 thresholds. ~4-8 h GPU + analysis. |
| V5 | Polytonic token concentration audit | **PARTIALLY DONE** | The `POLYTONIC_VOCAB_BUDGET_CHECK.md` analyses the budget against the sub-1B-language pattern. v0.6's expanded metric list (Goldfish-masked target occurrences, frequency quantiles, update/weight-norm ratio) needs separate implementation. |
| V6 | Dedup re-verification with accent-normalized hashing | **NOT DONE** | Polytonic vs monotonic of same passage. Doable on `xfer`. |
| V7 | Replay dataset acquisition test | **PARTIALLY DONE** | Per [`03_4 ENVIRONMENT_AND_BENCHMARKS.md`](03_4_implementation_experiments/ENVIRONMENT_AND_BENCHMARKS.md), staging plan exists; not yet executed. |
| V8 | Goldfish hash uniformity check | **NOT DONE** | Gated by Q C4 lookup. |
| V9 | NFC normalization probe | **NOT DONE** | ~2 hours. |
| V10 | vLLM/SGLang compatibility | **NOT DONE** | Deferrable post-pilot. |
| ~~V11~~ | LM-head calibration | **REMOVED** in v0.6 (specific to bracketed Distillation) | — |
| V12 | Cross-document attention masking preserved | **NOT DONE** | Needs Megatron config check. ~1 hour. |
| V13 | EoD token loss masking preserved | **NOT DONE** | ~1 hour. |
| V14 | BoD/EoD special tokens preserved | **DONE LOCALLY** | The ship-bundle verification confirms the 1000 `added_tokens` (which include BoD/EoD) are byte-identical to Apertus base. Re-confirm in HF↔Megatron roundtrip. |
| V15 | xIELU trainable scalars in optimizer param list | **NOT DONE** | ~30 min. Critical: easy to miss. |
| V16 | Tokenizer byte-fallback sanity check | **NOT DONE** | ~1 hour. |

## Summary: what's ready vs what's blocking

**Ready to do today (no slurm needed):**
- C2, C4 lookups (tech-report §2.3 / §3.3 via WebFetch)
- D3 (HF API for `swiss-ai/Apertus-8B-2509` revisions)
- The user-input decisions in §10 once the team aligns

**Ready to do on `xfer` (next slurm slot):**
- V1 decontamination pipeline
- V6 dedup re-verification with accent normalization
- V9 NFC normalization
- V12, V13 Megatron config checks (could also be on `debug`)

**Ready to do on `debug` (GPU, 1.5 h):**
- V3 dataloader state preservation
- V14 ship-bundle Megatron roundtrip
- V15 xIELU scalar verification
- V16 tokenizer byte-fallback
- V2 full tokenizer-extension forward pass

**Ready to do on `normal` (GPU, 12 h):**
- V4 variance baseline + bootstrap calibration
- The init bakeoff itself (3 × 2 B tokens ≈ 16 h serial / 5.2 h parallel on 4 nodes/arm)

**Genuinely needs user input** (cannot answer without project direction):
- Q A1, A3, A4, A5, A6, A7
- Q B1, B2, B3 (defaults exist but they're value judgments)
