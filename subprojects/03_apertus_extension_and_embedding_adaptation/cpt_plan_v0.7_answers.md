# cpt_plan v0.7 — Answers (2026-05-21)

*Self-contained response covering every question in `cpt_plan.md` v0.7 §10 (decisions), §11 (lookups), §12 (verifications). Every answer is inline; cross-references to other artifacts in this subproject are FYI only.*

## State of play

Implementation pass complete + two audit passes (locally-pinned-source + colleague reviewer). Recipe + sbatch + eval tooling ready at the level of paper / sbatch-line / code-line fidelity. Audit passes surfaced **9 real issues across two rounds** — 4 in our self-audit (all patched) + 5 in colleague reviewer round-2 (all 5 fixed: B1 HF→Megatron loader emitting unsupported keys, B2 Vanilla arm tokenizer mismatch, B3 corpus dedup path skipping the runbook flow, H4 eval task list contradicting EVAL_RECIPE scope, H5 BPC bias on long docs). Silent-failure risk inventory: now 17 risks in 3 tiers (R17 new — xIELU + QK-Norm trained values reset to defaults through the HF→Megatron path; acceptable for bakeoff, gating for production CPT). Two pre-submit blockers remain on Clariden (HF→Megatron loader roundtrip on Apertus-8B-2509; held-out eval slice reconstruction). No CSCS jobs have been submitted; local smoke tests are green.

Status legend: **RESOLVED** = answer known + cited · **LOCKED** = working default applied in code · **PENDING** = needs Fivos input · **DEFERRED** = explicitly out of scope for v0.7 · **NOT POSSIBLE** = answer doesn't exist in available sources.

---

## Q A — Decisions from Fivos (§10)

These need Fivos's input; they are not technical lookups. v0.7's framing already marks A1 as deferred.

| # | Question | Status | Working assumption + impact if changed |
|---|---|---|---|
| A1 | Capability targets | **DEFERRED** | None set; placeholder defaults flow downstream into §5.6 selection weights. |
| A2 | Total token budget for CPT post-init | **PENDING** | Working assumption: 15-20 B (v0.7 §3 + §9). Affects anneal-decay span, save-interval cadence, total cost. |
| A3 | Compute timeline / deadline | **PENDING** | Bakeoff assumes ~12 h-per-arm Clariden `normal` budget (1 node × 4 × GH200). Production budget gated on A2. No external deadline currently driving the schedule. |
| A4 | Stakeholders / downstream consumers | **PENDING** | Determines decontamination scope (V1). Working assumption: ILSP suite + Global-MMLU are comparison-grade benchmarks that need item-level dedup against training data. |
| A5 | Colleague sign-off on shuffled-bulk + annealing | **PENDING** | Both encoded in our recipes (`bulk.json` 32 sources weighted 70/24/4/2; `anneal.json` 14 sources 85/12/3). No objections received from p-skarvelis; no explicit sign-off either. |
| A6 | Specific downstream tasks | **PENDING** | Affects eval-suite emphasis (§5.6 weighted score). Using v0.7 §5.6 range-midpoint weights (35 % Greek BPC / 30 % Greek benchmarks / 12.5 % polytonic / 20 % retention / 7.5 % efficiency) until set. |
| A7 | Team structure | **PENDING** | Soft dependency; doesn't gate the bakeoff itself. |

---

## Q B — Design decisions (§10)

Defaults locked into code. None explicitly confirmed by Fivos; flag any you disagree with.

### B1 — Outer Greek/non-Greek split

**LOCKED at 70 % Greek / 24 % replay / 4 % code / 2 % math.**

Was 70/30 split originally (v0.7 §4.1 default). Rebalanced 2026-05-21 to make room for FineMath (Apertus stage-1 source). The 2 % math came from English replay (FW-Edu trimmed 3.9 % → 1.9 %); other 23 replay-language weights unchanged.

Per-bucket detail:
- **70 % Greek** — HPLT clean60 (35 %) + GlossAPI literary (18.2 %) + dialogue/textbooks (6.3 %) + academic (5.6 %) + legal/civic (3.5 %) + dictionary/misc (1.4 %, capped). All post-Apertus-overlap-drop, post-internal-dedup.
- **24 % replay** — 24 languages across 3 tiers. T1 (8 langs, 11.0 % total): English 1.9 % (FineWeb-Edu Score-3) + 7 others (fra/deu/ita/spa/rus/arb/cmn, 1.3 % each, FineWeb2-HQ). T2 (11 langs, 9.88 % total, 0.898 % each, FineWeb-2 standard): tur/bul/srp/ron/heb/por/pol/nld/pes/ukr/jpn. T3 (5 langs, 3.12 % total, 0.624 % each, FineWeb-2): lat/hye/kat/sqi/mkd.
- **4 % code** — StarCoderData (= Stack v1.2 subset; Apertus footnote 23).
- **2 % math** — FineMath-3plus (HuggingFaceTB/finemath; Apertus stage-1 source per `submit_apertus_8b.sh:L29`).

**Caveat:** The "24 replay languages" framing is our derivation. Apertus tech report Appendix G Table G.6 p.88 enumerates **20 high-resource languages** that receive quality+toxicity filtering; we added 4 (T3: lat/hye/kat/sqi/mkd) for region-specific coverage relevant to Greek. Token-share targets per language are our derivation too — Apertus publishes only document counts (Appendix G), not training-mix token shares. See Q C3 for an empirical-derivation proposal.

### B2 — Code share

**LOCKED at 4 %.** v0.7 §4.3 range 0-20 %; we picked the lower end consistent with Apertus's actual pretraining mix (FineMath + StarCoder both present from stage 1). Source: `bigcode/starcoderdata` = Stack v1.2; Li et al. arXiv:2305.06161.

### B3 — Anneal composition priority

**LOCKED at (d) balanced.** Anneal mixture: 85 % Greek / 12 % replay / 3 % code, with shift to highest-quality Greek subsets (literary 30 %, academic+legal 22 %, dictionary uncapped at 15 %, dialogue+textbooks 8 %, HPLT broad reduced to 10 %). Replay drops from 24 % → 12 % per v0.7 §3.2 "30 % → 15 %" pattern. Tier 1 ≥ 70 % within replay during anneal. Encoded in `anneal.json` (14 sources, weights sum to 1.0). Not used in the bakeoff — production-only.

### B4 — Loss objective for init bakeoff

**LOCKED at NTP for bakeoff, Goldfish for production.**

The bakeoff measures init quality across three arms; loss is held constant across arms so the comparison isolates init. NTP is the deterministic, well-tested default.

Production CPT restores Goldfish with Apertus's exact configuration (k = 50, h = 50, hash table 1,000,003, seed `2971215073`, deterministic prod-mod hash — see Q C4).

### B5 — Init experiment budget per variant

**LOCKED at 2 B tokens per arm.** v0.7 §5.4 says "1.5-2 B per variant"; we picked the upper end. Three arms × 2 B = 6 B total bakeoff budget. At Apertus's initial global-batch shape (1024 samples × 4096 seq-length = 4.19 M tokens/step), each arm runs ~477 iterations.

### B6 — Adaptation work prioritization (§8 items relevant before kickoff)

Status after the audit pass:

- **G1 Goldfish hash uniformity** (V8): READY now that Q C4 is resolved; production-only (bakeoff is NTP). Not blocking.
- **K1 Decontamination scope** (V1): PENDING on Q A4. Gating bakeoff if we want clean external comparison.
- **H1 BPC unit choice** (§5.1): RESOLVED in our `compute_tokenizer_fair_metrics.py` (Unicode characters via `len(text)` after upstream NFC; bytes via `text.encode("utf-8")`).
- **I1 NFC normalization** (V9): RESOLVED + operationally enforced via `normalize_nfc.sh` wrapper.
- **I2 `resize_token_embeddings` with untied E and U** (V2): mechanism audited; runtime check still pending Clariden debug slot.
- **J1 vLLM/SGLang compat** (V10): DEFERRED to post-pilot per v0.7.
- **B FOCUS for polytonic**: NOT in scope; v0.7 §8 default (b) holds — skip FOCUS, use baseline subpiece mean.
- **E1 Density warmup**: NOT in scope; v0.7 §8 default holds — skip.

---

## Q C — Apertus lookups (§11)

### C1 — Apertus pretraining peak LR — **RESOLVED**

Apertus pretrain peak LR = **1.1e-4** (paper Table 2 p.10; `submit_apertus_8b.sh:L245` `--lr 0.00011`).

CPT divergence: we use **1.5e-5** per v0.7 §3.3 (CPT operates near-converged; ~14 % of pretrain peak). Min LR = 1.5e-6 = 0.1 × peak.

### C2 — AdEMAMix optimizer hyperparameters — **RESOLVED**

All values from Apertus tech report Table C.4 p.82 + `submit_apertus_8b.sh:L208-219`:

| Param | Apertus pretrain value | Bakeoff value |
|---|---|---|
| β1 | 0.9 | 0.9 |
| β2 | 0.999 | 0.999 |
| β3_end | 0.9999 | 0.9999 |
| α (slow-EMA weight) | 8.0 | 8.0 |
| weight_decay | 0.1 | 0.1 |
| Gradient clip (global-norm) | **0.1** | 0.1 |
| α / β3 warmup | **100,000 steps** ("before the first checkpoint of WSD"; ~2.8 % of run) | **238 steps** (50 % of ~477-step bakeoff horizon — Apertus's 2.8 % policy would collapse to ~14 steps at our scale; we use the AdEMAMix paper's conservative cold-restart default) |
| init_method_std | 0.008944 | 0.008944 |

Paper citation: Pagliardini, Ablin, Grangier, *The AdEMAMix Optimizer: Better, Faster, Older*, ICLR 2025 (arXiv:2409.03137). Code path: `swiss-ai/Megatron-LM/megatron/core/optimizer/ademamix.py` (audited against the paper — exp_avg_slow not bias-corrected, β3 warmup uses log-form half-life-linear schedule).

### C3 — Per-language token shares — **PARTIAL (empirical-derivation path proposed)**

Apertus tech report publishes **FineWeb-2 document counts per language** (Appendix G Table G.6 p.88) — not training-mix token shares. The training data is built from public datasets (FW2, FW2-HQ, FW-Edu, StarCoderData, FineMath) with stage-specific iteration counts published in paper Table 6 p.24.

**Empirical-derivation path** (~4-6 h on Clariden xfer; not yet run):

1. For each (stage × dataset × language-config) Apertus declares, stream-read with `datasets.load_dataset(..., streaming=True)`.
2. Tokenize each doc with Apertus's tokenizer (or use char count as a cheap proxy).
3. Sum tokens per language across all stages, weighted by the published per-stage iteration count.
4. Divide by total tokens summed across stages.

Same shape as our vocab-extension token-attribution audit (already done for the 17,408 modern Greek tokens against the GlossAPI corpus). Result: a per-language token-share table covering all ~24 Apertus replay languages.

The 24 high-resource languages with their document counts from Apertus's Appendix G Table G.6 (in descending order by % of FW2 docs):

Russian (13.26 %), Mandarin (12.66 %), German (9.36 %), Spanish (8.88 %), Japanese (8.23 %), French (7.28 %), Italian (4.80 %), Portuguese (4.16 %), Polish (3.03 %), Dutch (2.93 %), Indonesian (2.04 %), Turkish (1.94 %), Czech (1.37 %), Korean (1.27 %), Arabic (1.26 %), Romanian (1.19 %), Persian (1.12 %), Ukrainian (1.04 %), Hungarian (1.03 %), Swedish (0.99 %), **Greek 0.97 %**, Danish (0.94 %), Vietnamese (0.89 %).

(These are FW2 *document* shares; token shares may differ because per-doc token counts vary by language and by Apertus's per-stage quality filtering.)

### C4 — Apertus Goldfish loss configuration — **RESOLVED**

From Apertus tech report §2.3 p.11 + Algorithm 1 p.86 + Megatron implementation:

| Param | Value | Source |
|---|---|---|
| Mask frequency `k` | 50 (= 2 % of tokens masked) | paper §2.3; `submit_apertus_8b.sh:L291` `--goldfish-k 50` |
| Hash context window `h` | 50 preceding tokens | paper §2.3; sbatch L292 |
| Hash function | Deterministic: `prod(last h tokens) mod table_size` | paper Algorithm 1; code `apply_goldfish` |
| Hash table size | **1,000,003** | code `_HASH_TABLE_SIZE` constant in `megatron/core/datasets/gpt_dataset.py` |
| Hash seed | **`2971215073`** | code `_create_hash_table(device)` |
| Implementation | **Front-loaded during data loading**, not at training time | paper §2.3 p.11, §F p.85 |
| Goldfish token id constant | `-2` (masked positions get this id pre-loss) | code `_GOLDFISH_TOKEN_ID` |

Paper citation: Hans, Wen, Jain et al., *Goldfish Loss: Mitigating Memorization in Generative LLMs* (arXiv:2406.10209).

### C5 — Apertus tokenizer config — **RESOLVED**

**Base**: Mistral-Nemo tekken v3 byte-level BPE, vocab 131,072. Inherited unchanged from Mistral-Nemo (paper §2.2 p.10).

**Tokenizer properties** (verified locally):
- `normalizer: null` — no Unicode normalization at tokenizer level → pre-tokenization NFC required at corpus level (V9).
- Pre-tokenizer: GPT-2-style regex split → ByteLevel.
- Model: BPE, vocab 131,072.
- Single-token vocab hits: bare α, monotonic ά, final sigma ς, en-dash, em-dash, ellipsis, smart quotes, NBSP.
- Special tokens preserved at IDs 0-999 (unk=0, bos=1, eos=2, pad=3, plus Mistral reserved).

**Extension** (modern-only bakeoff target = 148,480 vocab):
- +17,408 modern Greek tokens curated from C3 BPE training on GlossAPI + HPLT 50/50.
- All special-token IDs preserved (V14 verified).
- Polytonic NT-style phrase compresses from 19 tokens (base) to 9 tokens (148,480 modern-only) to 5 tokens (composite 153,600 with polytonic extension — parked for future polytonic specialization run).
- No byte-fallback corruption on extended polytonic IDs (V16 verified — all decode to clean polytonic Unicode codepoints).

---

## Q D — Engineering lookups (§11)

### D1 — Megatron-LM fork branch / commit — **RESOLVED**

- **Repo**: `swiss-ai/Megatron-LM` (GitHub).
- **Branch**: `main`. Apertus production does not pin a specific tag (paper footnote 4 "https://github.com/swiss-ai/Megatron-LM" with no commit hash).
- **Our pinned commit**: `c92402e39ef3c8e69ea378a59e79059dc14541f4` (HEAD as of 2026-05-20).
- **Apertus-specific paths** (all verified to exist):
  - AdEMAMix optimizer: `megatron/core/optimizer/ademamix.py`
  - xIELU activation: `megatron/training/activations.py` (classes `XIELU`, `XIPReLU`)
  - QK-Norm wiring: `megatron/core/transformer/attention.py` L652-656 (norm before RoPE at L432)
  - Goldfish loss: `megatron/core/datasets/gpt_dataset.py` (`apply_goldfish`, `_create_hash_table`, `_HASH_TABLE_SIZE`, `_GOLDFISH_TOKEN_ID`)
  - Training entry: `pretrain_gpt.py`
  - HF→Megatron loader (Apertus-specific): **does not exist upstream** — we wrote `loader_apertus_hf.py` (see "What we built" below).
  - Megatron→HF saver: `tools/checkpoint/saver_swissai_hf.py` (handles Apertus's `ApertusConfig` / `ApertusForCausalLM`).
- **Production launch script** (the canonical "what was run"): `swiss-ai/pretrain-code/pretraining/submit_apertus_8b.sh` at commit `531cc8be2f76064127cad99a61019f985a7c7ee2` (HEAD 2026-05-20). Our `bakeoff_train.sbatch` mirrors this flag-set with the three CPT-specific deviations called out below.

### D2 — FineWeb-2 Tier 3 language audit — **PENDING**

Need token counts (or document counts as a cheap proxy) for `lat_Latn` / `hye_Armn` / `kat_Geor` / `sqi_Latn` (or `als_Latn`) / `mkd_Cyrl`. v0.7 §4.2 framing: under ~100 M tokens → treat as "preservation aspiration" not "active maintenance".

This is a cheap `datasets.load_dataset(..., split_info=True)` call per language. ~30 min when run. Not bakeoff-gating.

### D3 — Apertus intermediate checkpoints — **PENDING**

v0.7 §11 suggests they're "available on HF branches" and "useful for annealing-as-quality-meter". Not enumerated yet; not bakeoff-gating.

---

## V1-V16 — Verifications (§12)

Status at 2026-05-21 after the implementation pass. Distribution: DONE 5 · PARTIAL 3 · READY 1 · NOT DONE 5 · DEFERRED 1. (Was 2/1/0/8/3 in the 2026-05-20 snapshot.)

| # | Status | What we have / what's pending |
|---|---|---|
| **V1** Decontamination scope (item-level dedup of clean-measurement benchmarks against training data) | **NOT DONE** | Gated on Q A4 (which benchmarks need to be clean for external comparison). Approach when run: NeMo Curator downstream-task decontamination workflow on Clariden `xfer`, against the chosen benchmarks' literal test items (not blanket Greek removal). ~1-3 days. |
| **V2** Tokenizer extension forward pass (model produces vocab-148,480 logits, no nan/inf) | **PARTIAL** | Tokenizer side ✓ (`build_and_verify_ship_tokenizer.py` confirms both ship bundles load via `AutoTokenizer`, 1,000 added_tokens byte-identical to base, special-token IDs preserved). Model-side smoke (instantiate `ApertusForCausalLM`, call `resize_token_embeddings(148480)`, forward pass on Greek, check `[B, T, 148480]` logit shape no nan/inf) scheduled for Clariden debug slot. ~30 min once that slot is available. |
| **V3** Dataloader-state preservation across checkpoint boundary | **NOT DONE** | Megatron-LM default behavior. Smoke verify on first Clariden debug submit: stop at 100 M tokens, resume from checkpoint, verify next batch index is correct. |
| **V4** Run-to-run variance baseline on unmodified Apertus-8B-2509 (full eval suite + bootstrap CIs) | **NOT DONE** | **Gating for §5.6 hard-gate thresholds.** Threshold setting is deferred to post-V4: we can't say "more than 3 p.p. drop is a failure" if the run-to-run variance of HellaSwag is itself 2 p.p. ~3-4 h on Clariden `normal`. |
| **V5** Polytonic-token concentration audit under planned Greek mixture (input/target occurrences, register distribution, frequency quantiles, update norms) | **NOT DONE** | Gates on CPT corpus build (V7). ~2 h analysis after the bulk_mix.jsonl is produced. For our modern-only bakeoff this is less critical (no polytonic in bakeoff scope per v0.7 §5.8); becomes critical for the future polytonic specialization run. |
| **V6** Accent-normalized dedup re-verification | **NOT DONE** | Existing dedup audit ran with `greek_diacritic_policy = preserve`. To catch polytonic/monotonic variants of the same passage, re-run with `policy = strip` on Clariden `xfer`. ~2-3 h. |
| **V7** Replay dataset acquisition (all §4.4 sources accessible + tokenizable) | **PARTIAL** | Pull scripts (`pull_greek_corpus.sh`, `pull_replay_datasets.sh`) written and bash-syntax-clean. Cover GlossAPI nanochat + Apertus-overlap drop overlay + FineWeb-Edu Score-3 + FineWeb2-HQ (7 T1 langs) + FineWeb-2 (T2/T3) + StarCoderData + FineMath-3plus. Clariden login-node pulls not yet executed; ~30-60 min wall, ~25-50 GB on iopsstor. |
| **V8** Goldfish hash uniformity across new tokens | **READY** | Was gated on Q C4. With Q C4 resolved (k=h=50, hash 1,000,003, seed 2971215073), V8 is now runnable: tokenize sample corpus with extended tokenizer, apply hash, count masked positions per new ID, verify uniformity within ±2σ across 17,408 (modern-only) or 22,528 (composite future-polytonic) new IDs. **Production-only** — the bakeoff itself is NTP. ~1 h once CPT corpus is built. |
| **V9** NFC normalization of training corpus | **DONE** | Multiple converging mechanisms: (a) `text_dedup.py` line 883 NFC-normalizes before hashing (so NFD-leakage doesn't pass dedup); (b) HPLT upstream delivers ~100 % NFC (500/500 sample-verified); (c) finepdfs-edu has ~0.07 % NFD leakage but our pipeline now operationally enforces NFC via the `normalize_nfc.sh` wrapper, which runs `verify_and_normalize_nfc.py normalize` (idempotent in-place) over every parquet shard between corpus pull and `mix_builder.py`. The end-to-end sequence places it explicitly as a Clariden xfer step. |
| **V10** vLLM / SGLang compatibility with vocab 148,480 | **DEFERRED** | v0.7 §8 allows this to defer if production serving isn't immediate. Vocab 148,480 = 256 × 580 is 256-aligned (kernel-friendly); should work but unverified. ~2-4 h per system when needed. |
| **V12** Cross-document attention masking (matching Apertus pretraining) | **DONE (config)** | Apertus pretraining uses `--reset-position-ids --reset-attention-mask` (`submit_apertus_8b.sh:L286-289`). Our bakeoff sbatch mirrors these. Runtime verification on first Clariden submit: 2-doc test batch with positions reset at the EoD boundary. |
| **V13** EoD token loss masking | **DONE (config)** | Apertus uses `--eod-mask-loss` (`submit_apertus_8b.sh:L288`). Our bakeoff sbatch mirrors. **Caveat**: the EoD token ID in the extended-vocab ship bundle must match the base Apertus EoD ID. V14 confirmed special-token ID preservation but `build_init_checkpoints.py` doesn't yet have an explicit equality assertion — flagged as R8 in our risk inventory; 15-min cheap mitigation pending. |
| **V14** BoD/EoD special token preservation in extended tokenizer | **DONE** | Verified by `build_and_verify_ship_tokenizer.py` on every run: all 1,000 added_tokens byte-identical to Apertus base (unk=0, bos=1, eos=2, pad=3, plus Mistral-reserved IDs 4-999). Both 148,480 modern-only and 153,600 composite ship bundles pass this check. |
| **V15** xIELU per-layer αp / αn scalars survive vocab extension | **PARTIAL** | Mechanism audited LOW RISK: `XIELU.__init__` in `megatron/training/activations.py` lines 33-46 registers `alpha_p` and `alpha_n` as `nn.Parameter(...)` instances. PyTorch's `nn.Module.parameters()` auto-walks all `nn.Parameter` children, so they're in the optimizer's param list. `transformers.PreTrainedModel.resize_token_embeddings()` only mutates `embed_tokens.weight` and `lm_head.weight` — it does NOT touch other modules; XIELU instances are untouched; their αp/αn `nn.Parameter` references stay alive; their optimizer-state entries persist. **Open**: explicit before/after-resize equality assertion in `build_init_checkpoints.py` flagged as R9; 15-min cheap mitigation pending. |
| **V16** Tokenizer byte-fallback for new polytonic tokens | **DONE** | Verified: 10 distinct polytonic IDs hit across 10 NT-style test samples; every polytonic ID decodes back to clean Greek with proper polytonic Unicode codepoints (U+1F00–U+1FFF). No byte-fallback corruption. Apertus base needs 19 tokens for a phrase; composite 153,600 bundle needs 5 — 3.8× compression all routed through the polytonic block. |

---

## Three CPT-specific deviations from Apertus pretraining

These are the only places the bakeoff knowingly diverges from Apertus's exact pretraining recipe. Each has explicit reason and citation.

| # | Deviation | Reason | Apertus value | Bakeoff value |
|---|---|---|---|---|
| 1 | **LR peak** | CPT operates near-converged; standard practice (Llama-3 CPT, Aya) is ~10-20 % of pretrain peak | 1.1e-4 | **1.5e-5** (≈ 14 %) |
| 2 | **α / β3 warmup** | Apertus's 100k-step policy is ~2.8 % of pretrain run (15 T tokens). At our 2 B / 477-step bakeoff, 2.8 % collapses to ~14 steps — too short. Per AdEMAMix paper §"Switching optimizers" cold-restart guidance, use a conservative ramp over a meaningful fraction of the new run | 100,000 steps | **238 steps** (50 % of bakeoff horizon) |
| 3 | **Loss objective** | Bakeoff measures init quality across three arms; loss is held constant across arms to isolate the init variable | Goldfish (k=h=50) | **NTP** (bakeoff only; production restores Goldfish with Apertus's exact config) |

Everything else mirrors Apertus exactly: AdEMAMix optimizer + all hyperparams, 0.1 gradient clip, WSD with 1-sqrt cooldown, xIELU activation with αp=αn=0.8 init + β=0.5, QK-Norm per-head RMSNorm before RoPE, cross-doc attention mask + EoD loss mask both ON, untied E/U embeddings, bias-free linear layers, bf16 + fp32 master grads (FP8 explicitly rolled back per paper Appendix D), sequence length 4,096, RoPE θ=500,000, `make-vocab-size-divisible-by 128`.

---

## What we built (summary for context)

Recipe + tooling implemented as 13 commits since 2026-05-20:

| Layer | What |
|---|---|
| **Dataset assembly** | `corpus_build/` — streaming HF interleaver (`mix_builder.py`); recipes `bulk.json` (32 sources, 70/24/4/2) + `anneal.json` (14 sources, 85/12/3); pull scripts for Greek + replay + math + code; `normalize_nfc.sh` wrapper for V9 enforcement. |
| **Training recipe** | `TRAINING_RECIPE.md` — every numeric value cites paper §/page/Table or sbatch line or code path. |
| **Three init arms** | `arms/{vanilla,retok,centroid}.py` + `arms/_common.py` + `arms/build_init_checkpoints.py` + `arms/test_init_logic.py` (local smoke). |
| **Training sbatch** | `bakeoff_training/_train_config_common.env` + `bakeoff_train.sbatch` (parameterized, mirrors `submit_apertus_8b.sh` flag-for-flag) + `preprocess_data.sbatch` + `submit_all_arms.sh`. |
| **HF→Megatron loader (custom)** | `megatron_patches/loader_apertus_hf.py` — first-party HF→Megatron path for Apertus didn't exist upstream (`loader_llama_mistral.py` covers llama/mistral/qwen but not Apertus's xIELU + QK-Norm + GQA + bias-free arch). Inverse of `saver_swissai_hf.py`. Roundtrip-validation procedure documented. |
| **Eval tooling** | `eval/` — lm-eval-harness wrappers (V4 baseline + per-arm), bootstrap CIs, **`compute_tokenizer_fair_metrics.py`** (§5.1 BPC + NLL/char + STRR + tokens/word), **`compute_new_token_diagnostics.py`** (all 7 §5.3 diagnostics), `summarize_bakeoff.py` (markdown aggregator for manual §5.6 selection). |
| **References** | `references/` — 8 cloned repos at pinned commits (swiss-ai's Megatron-LM, pretrain-code, pretrain-data, lm-evaluation-harness, apertus-finetuning-recipes, apertus-tech-report; apple/ml-ademamix; EleutherAI/lm-evaluation-harness) + 15 papers (HTML preferred; ~15 MB total). |
| **Documentation** | `REVIEW_PRESENTATION.md` (reviewer entry), `AUDIT_FINDINGS.md` (source audit + 4 patches applied), `RISKS.md` (16 silent-failure risks), `COMPLETENESS_CHECK.md` (script-coverage gap analysis), this doc. |

---

## Audit findings (source-vs-implementation cross-check)

### Round-1: self-audit against locally-pinned primary sources (8 repos + 15 papers) — 4 real bugs patched

1. **Sbatch flag-name typos**: `--xielu-activation` → `--xielu`; `--ademamix-{beta3,alpha}-warmup-steps` → `--ademamix-{beta3,alpha}-warmup`. Would have failed first submission.
2. **Missing `--dist-ckpt-strictness assume_ok_unexpected`**: this is what allows Megatron to load a checkpoint whose embedding shape was resized 131,072 → 148,480 by our init builder. Without it Megatron refuses the shape mismatch.
3. **`retok.py` zero-row failure mode**: silent `continue` on empty decode left the new-token row at zero. After `norm_match` (which leaves zero alone), the resulting softmax logit at that position is 1 while real logits are large-negative → the new token would dominate the softmax. Hewitt 2021's diagnosed "zero-init disaster" case. **Patched**: hard fallback to global base-vocab mean.
4. **`centroid.py` was using diagonal-σ**: that's Mundra 2024's "Univariate" baseline which the paper explicitly calls inadequate (§5.1 + Table 2 p.6). Hewitt 2021 uses full Σ multivariate normal. **Patched**: switched to full Σ with 1e-8 ridge for numerical stability + precomputed Cholesky for O(D²) per-sample efficiency. Smoke test now runs in 4.1 s.

9 additional Apertus throughput/memory/correctness flags now mirrored: `--cross-entropy-loss-fusion`, `--manual-gc --manual-gc-interval 500`, `--overlap-grad-reduce --overlap-param-gather`, `--no-check-for-nan-in-loss-and-grad`, `--make-vocab-size-divisible-by 128`, `--ckpt-format torch_dist`, `--attention-dropout 0.0 --hidden-dropout 0.0`, `--split 100,0,0`, full network-arch declaration block (32 layers / 4096 hidden / 21504 FFN / 32 heads / 8 KV-groups GQA / etc.).

### Round-2: colleague reviewer audit — 5 real issues, all fixed

1. **B1 — HF→Megatron loader emitted Apertus-specific keys not in saver_core's protocol.** `saver_core.py:L357-443` only consumes standard transformer-protocol message keys; `check_message()` rejects extras (or with `--no-checking` silently drops them). Our previous loader sent `mlp xielu alpha p/n` and `q/k norm weight` → conversion would fail or silently drop those values. The documented command also missed required `--model-type GPT`. **Fixed**: loader now emits only saver_core-consumable keys; README + install.sh updated with `--model-type GPT`. **New risk R17 surfaced** (xIELU + QK-Norm reset to defaults through this path) — acceptable for bakeoff (same defaults across all 3 arms; comparison still valid), gating before production CPT. Post-conversion patcher scaffold at `megatron_patches/patch_apertus_extras.py`.

2. **B2 — Vanilla arm tokenizer mismatch.** Vanilla uses Apertus's base 131,072 vocab (control arm); but `preprocess_data.sbatch` + `bakeoff_train.sbatch` hardcoded the 148,480 extended tokenizer for all three arms. Vanilla trained on 148,480-tokenized data would either crash on out-of-range token IDs or silently corrupt the control arm. **Fixed**: `_train_config_common.env` now defines `BASE_TOKENIZER_DIR` + `EXT_TOKENIZER_DIR` + `BASE_DATA_PREFIX` + `EXT_DATA_PREFIX`. Run `preprocess_data.sbatch` twice (once per tokenizer family) to build two byte-identical-document-stream Megatron binaries that differ only in tokenization. `bakeoff_train.sbatch` switches per-arm; `submit_all_arms.sh` sanity-checks both prefixes exist.

3. **B3 — Corpus dedup path skipped the runbook flow.** `CPT_DATASET_BUILD_RUNBOOK.md` mandates: hydrate nanochat → hard-exclude Apertus-overlap → replay nanochat internal dedup with `drop_intra_and_inter` → build the mix. `mix_builder.py` streamed HF directly with no selected-pool input; the Apertus drop was applied only to the HPLT source in `bulk.json:23` (not the other 5 Greek sources). Internal-dedup replay was missing entirely. Order-wrong AND incomplete. **Fixed**: new `prepare_greek_pool.sh` wrapper invokes `glossapi_corpus_cli mix-prepare-selected-input` to produce the `$SELECTED` parquet with all three runbook steps. `mix_builder.py` now supports `local_parquet` with `${VAR}` env-var expansion; all 6 Greek sources in `bulk.json` read from `${SELECTED}` (Apertus drop is upstream, uniform across all sources). `pull_greek_corpus.sh` extended to pull wave2 builder_metadata needed by the internal-dedup replay.

4. **H4 — Eval task list contradicted EVAL_RECIPE.md scope.** EVAL_RECIPE.md says Table-14 pretraining evals only; `run_eval.sbatch` included GSM8K, HumanEval, ifeval_greek (all Apertus Table 22 post-training tasks) and omitted Global-MMLU (Table 14). **Fixed**: synced task lists to EVAL_RECIPE.md. Removed `gsm8k`, `humaneval`, `mgsm_greek`, `ifeval_greek`; added `global_mmlu`.

5. **H5 — BPC bias on long documents in `compute_tokenizer_fair_metrics.py`.** Script counted chars/bytes/words on the FULL text but truncated token IDs to `max_context` before forward pass → BPC and NLL/char divided prefix-only loss by full-document denominators → artificially low. **Fixed**: when truncated, decode the kept ID list back to text and compute char/byte/word counts on that scored prefix. Added truncation counter in the output JSON.

---

## Silent-failure risk inventory

17 risks documented in 3 tiers — places where the bakeoff could be silently wrong despite passing local smoke tests. R17 added 2026-05-21 in response to reviewer round-2: xIELU + QK-Norm trained values reset to defaults through the HF→Megatron conversion path. Acceptable for bakeoff (all 3 arms inherit the same defaults); gating before production CPT.

**Tier 1 (could invalidate the bakeoff entirely):**

| # | Risk | What goes wrong silently | Mitigation status |
|---|---|---|---|
| R1 | HF→Megatron QKV interleaving in our `loader_apertus_hf.py` untested | A wrong stride in the GQA interleave (num_heads=32, num_kv=8, heads_per_group=4) produces a tensor of the same shape with Q heads bound to wrong K/V groups. Converted model loads, forward passes succeed numerically, but attention is computed against permuted keys/values — outputs are noise. | Roundtrip validation procedure documented; must run on Apertus-8B-2509 (Clariden GPU + weights) before first sbatch. |
| R2 | Token-stream determinism across the three arms unverified | `datasets.interleave_datasets` with HF streaming isn't strictly deterministic if HF library version, cache state, or shard counts differ across runs. If arm A and arm B see slightly different streams, we're comparing init + data drift. | 15-min addition: MD5-of-output-JSONL written to mix_builder manifest; re-runs that don't match the hash fail loud. Not done. |
| R3 | Held-out Greek eval slice doesn't exist + cleanliness vs Apertus pretraining unverified | If the slice contains anything Apertus saw, Apertus-base looks artificially strong from memorization → wrong arm wins selection. | Reconstruct dedup-audit val/test partition on Clariden xfer (gcloud-loss-affected artifact; reconstruction path documented). |
| R4 | ReTok / Centroid surface-form decode has a leading-space artifact | Apertus uses ByteLevel BPE; word-initial tokens prefix `Ġ`. `extended_tokenizer.decode([new_id])` returns surface with or without leading space. `base_tokenizer.encode(surface)` then produces different subpiece chains depending. ReTok inits would be systematically wrong for word-initial new tokens. | 30-min unit test: decode 100 random new tokens, encode each with base, assert subpiece IDs match manual computation. Not done. |

**Tier 2 (subtle bias)**: R5 stale norm targets (5.05/3.80 from May-11 diagnostic; may not match Apertus-8B-2509), R6 silent Apertus-overlap-drop failure on schema mismatch, R7 lm-eval-harness commit not pinned, R8 EoD special-token ID equality not asserted, R9 xIELU survival across resize_token_embeddings not asserted.

**Tier 3 (would surface eventually)**: R10-R16 — `--dist-ckpt-strictness assume_ok_unexpected` lenience; centroid Cholesky rank-deficiency edge case; STRR regex edge cases; NFD-decomposed token classification edge case; TP=1 numerical drift vs Apertus TP=2; FineMath share 2 % is a guess; OPUS deferred.

**Cheap mitigations available** (~2 h total for all 7, not yet landed; awaiting decision):
- R2: MD5-of-output-JSONL in `mix_builder.py`
- R4: ReTok subpiece-decode unit test
- R5: norm-drift check in `build_init_checkpoints.py`
- R6: drop-rate sanity assertion in `mix_builder.py`
- R7: pin lm-eval-harness commit + assert in `run_eval.sbatch`
- R8: special-token equality check in `build_init_checkpoints.py`
- R9: xIELU αp/αn equality assertion around `resize_token_embeddings`

**Hard mitigations** (require Clariden / external setup): R1 roundtrip on Apertus weights; R3 held-out slice reconstruction.

---

## What's still gating bakeoff submission

Five items, ordered by hardness:

1. **R1 roundtrip validation** on unmodified Apertus-8B-2509 — needs Clariden GPU + Apertus weights. Procedure documented (HF → Megatron via our loader → HF via existing saver → diff against original; pass condition `max abs diff < 1e-3` on bf16-quantised weights).
2. **V4 baseline** — full eval suite × unmodified Apertus-8B-2509 on Clariden `normal`, ~3-4 h. Sets §5.6 hard-gate thresholds. The "more than X p.p. drop is a failure" rule has no value of X until run-to-run variance from V4 is known.
3. **R3 held-out eval slice** — reconstruct the dedup-audit val/test partition on Clariden xfer (option B per the original audit's review-checkpoint list, since the gcloud-stored partition is no longer accessible).
4. **V7 Clariden pull** — 30-60 min login-node `huggingface-cli download` execution of our existing pull scripts.
5. **ILSP harness task YAMLs** — staging-time merge from Meltemi (`LeonVouk/lighteval`) or Krikri (`ilsp/lm-evaluation-harness-greek`) forks. The swiss-ai harness fork has the infrastructure but not the Greek `*_greek` task definitions; needs ~1 h merge before V4 baseline can include Greek tasks.

Items 1-3 are real verifications. Items 4-5 are mechanical pulls.

---

## Open Fivos-decisions blocking nothing in the short term

Listed here for visibility, even though they don't gate bakeoff submission:

- A1 capability targets (deferred per v0.7)
- A2 total CPT token budget (working assumption 15-20 B)
- A3 compute timeline (no external deadline)
- A4 stakeholders / consumers (gates V1 decontamination scope only)
- A5 colleague sign-off
- A6 specific downstream tasks (affects §5.6 weighted-score weights)
- A7 team structure
- D2 FineWeb-2 Tier 3 audit (cheap; non-blocking)
- D3 Apertus intermediate checkpoints (non-blocking)

If decisions on A2/A4/A6 land before we run V4, we can fold them into threshold-setting immediately. Otherwise we proceed with the working assumptions above.
