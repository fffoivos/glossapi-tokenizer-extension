# SUGGESTIONS — actionable signals from the 2026-05-21 session

Sibling to [`SESSION_LOG_20260521.md`](SESSION_LOG_20260521.md) (audit trail) and [`CSCS_OVERNIGHT_STATE.md`](CSCS_OVERNIGHT_STATE.md) (current operating state). This doc is the **forward-looking** counterpart: things we should do, change, or watch based on what we observed. Open to ongoing extension.

Each entry tagged with priority:
- **[BLOCKER]** — must do before the relevant next step
- **[PRIMARY]** — substantive improvement to results / interpretation quality
- **[NICE]** — engineering hygiene; matters for future sessions
- **[WATCH]** — open risk to track; not actionable yet
- **[DONE — date]** — item closed; kept for historical context

---

## 0. Update from systematic plan cross-check (2026-05-22)

After a systematic read of `BAKEOFF_PLAN.md`, `cpt_plan.md v0.7` §3-6, `TRAINING_RECIPE.md` §11-13, `apertus_fidelity_checklist.md` §10 (kickoff gates):

**Newly identified gaps that the initial review missed:**

- **V15** (`apertus_fidelity_checklist.md` §2.1) — verify xIELU `αp/αn` scalars are in the optimizer parameter list **after** `resize_token_embeddings`. R17 patcher landed the *values*; V15 is about whether they get *trained* during CPT. This is independent — fixing one doesn't fix the other. **[BLOCKER pre-bakeoff]** — see new entry §1.6.
- **V1** (apertus_fidelity_checklist.md §8.1 + §10) — eval-set decontamination via NeMo Curator. Original plan budgeted 3-5 days. No evidence this was performed; if eval-set rows leaked into the corpus, V4 (and per-arm) numbers are inflated. **[BLOCKER for credible production CPT, not for the bakeoff comparison]** — see new entry §3.5.
- **V14** (apertus_fidelity_checklist.md §3.4) — HF↔Megatron special-token roundtrip. `build_init_checkpoints.py` checks at build time via `_assert_special_tokens_preserved`. We haven't seen the same check run after Megatron conversion → confirm BoD/EoD IDs aren't shifted by the Megatron `tokenizer-type HuggingFaceTokenizer` adapter. **[PRIMARY]** — see new entry §1.7.
- **Math 2% bucket** — `bulk.json` has Greek 70 / replay 24 / code 4 / math 2. `cpt_plan.md v0.7` §4 specifies replay + code (24 langs + 4% code). Math is mentioned in v0.7 §6.3 retention but not in the §4 replay design. The 2% math addition is defensible but appears to be a 2026-05-21 implementation decision, not a plan-rooted choice. **[WATCH]** — see new entry §4.5.

**Plan items now confirmed done from this side:**

- `cpt_plan.md v0.7` §2 settled shape — every item matches `_train_config_common.env` and `bakeoff_train.sbatch`.
- `apertus_fidelity_checklist.md` §10 gates 1-4, 6, 7, 11 — RESOLVED (per `TRAINING_RECIPE.md §13` table + bakeoff_train.sbatch). Gates 5 (V1), 8 (V14), 9 (V15), 10 (V16) are still open or unverified.
- The R17 patcher is **bit-perfect**: `verify_hf_roundtrip.py` on jobs 2341182/2341239/2341241 reports `standard_max_abs_diff=0.0`, `r17_max_abs_diff=0.0`, `xielu_max_abs_diff=0.0`, `qk_norm_max_abs_diff=0.0`, `logit_max_abs_diff=0.0`, with top-id match on all three test prompts (2 Greek + 1 English).
- `submit_all_arms.sh` already defaults to `INIT_CKPT_SUBDIR=megatron_tp2_r17patched`. So the next launch correctly uses the patched checkpoints.

**Major implication for §2:** the bakeoff is now back to **retention** (against V4-HF) not **recovery** (against V4-postconv), because the patched init starts essentially at V4-HF performance. The V4-postconv framing in `V4_BENCHMARK_COMPARISON.md` is the *un-patched ablation question*, not the canonical bakeoff framing. SUGGESTIONS §2.1 is rewritten below.

---

## 1. Bakeoff execution (immediate)

### 1.1 [BLOCKER] Smoke one arm for an hour before firing all three
The cancelled `bakeoff_tp2_mb2_bucketfix_20260521_2204` (jobs 2340682/3/4) reached the `training ...` line but was halted before iteration 1 logs appeared. We still don't have empirical evidence that `mb=2` + `expandable_segments:True` clears the first forward pass on GH200. Submitting 3 × 12h jobs blind risks 36 node-hours on a likely re-fail.

**Do:** submit Vanilla alone with `--exit-interval 10` (or rely on the existing `triggers/exit` mechanism) at `time=01:00:00`, no save, measure tokens/sec, verify first 10 iterations log cleanly. Cost: ~1 node-hour. Then submit_all_arms.sh.

### 1.2 [PRIMARY] Validate the 2B-token / 12h walltime budget against measured throughput
With TP=2, mb=2, 256 grad-accum steps per global batch, on GH200, the iteration time is unknown. 477 iterations × ETA-per-iter must fit in 12h (43200s) per arm. The smoke from 1.1 will measure this. Target: ≥ ~5500 tokens/sec/GPU sustained for the budget to fit comfortably; below 3000 means we need either more walltime (12h is the partition cap) or a smaller token budget.

### 1.3 [PRIMARY] Add a TE-guard skip counter so the reviewer can see how many empty `_extra_state` tensors were no-op'd
`megatron_patches/runtime/pretrain_gpt_te_guard.py` silently `return None`s on every empty-state EOFError. Hardening: increment a module-level counter, print it once at end-of-training-init. Quietly skipping 200 tensors is qualitatively different from quietly skipping 5; reviewer should see the count.

### 1.4 [PRIMARY] Document the deliberate mb=4 → mb=2 fidelity drift in `_train_config_common.env`
The microbatch deviation from Apertus's `4` is loss-equivalent (global batch tokens 4.19M preserved; just 128 vs 256 grad-accum steps), but the comment block should explicitly state that loss-equivalence so the reviewer doesn't have to derive it. Suggested text:

> Global batch tokens preserved at 4.19M; Apertus mb=4 → 128 microbatches, us mb=2 → 256 microbatches. The bf16 accumulation order differs but the optimizer step sees the same total gradient. All three bakeoff arms use the same mb=2 schedule so cross-arm comparison stays internally consistent.

### 1.6 [BLOCKER pre-bakeoff] Verify V15 — xIELU `αp/αn` are in the optimizer parameter list after resize
`apertus_fidelity_checklist.md` §2.1: "After calling `resize_token_embeddings()`, the optimizer's parameter list is rebuilt. **Verify the xIELU scalars are still in the optimizer's parameter list.** Easy to miss; resize_token_embeddings is documented for the embedding tables, not for everything else."

The R17 patcher confirmed the *values* are present at load time. V15 is about whether they *get gradients* during training. Independent issue.

**Cheap check:** during the §1.1 smoke (one-arm 1h smoke), grep the iteration-1 log for the optimizer's parameter-group sizes. If `αp/αn` (32 layers × 2 scalars = 64 params) are not in the optimizer state, the smoke will pass (the model still runs forward) but the bakeoff will silently train with xIELU frozen at the R17-patched values. That would null the entire R17 patcher benefit during training.

If found missing: pretrain-code already includes xIELU in the param list (Apertus's own training works); investigate whether Megatron's `--use-distributed-optimizer` + our embedding-resized init checkpoint is the failure path.

### 1.7 [PRIMARY] Confirm V14 special-token roundtrip survives Megatron preprocess + training
`build_init_checkpoints.py` already runs `_assert_special_tokens_preserved` at build time. Two later stages can still shift IDs and aren't checked:
- `preprocess_data.sbatch` invokes Megatron's `tools/preprocess_data.py --tokenizer-type HuggingFaceTokenizer`; the adapter could re-map BoD/EoD if it interprets `special_tokens_map.json` differently.
- The Megatron training-time `--tokenizer-model` path uses the same adapter — if there's a mismatch with the converted checkpoint's tokenizer-id assumptions, BoD/EoD become garbage.

**Cheap check:** decode the first 100 token IDs from the Megatron `.bin` file with both `BASE_TOKENIZER_DIR` and the swissai-Megatron `HuggingFaceTokenizer` adapter; confirm the BoD/EoD positions align with where the JSONL has document boundaries. Add as a §1.1 smoke companion.

### 1.5 [WATCH] AdEMAMix β3/α warmup at 50% of the bakeoff means real signal is concentrated in iterations 238-477
We set `ADEMA_BETA3_WARMUP_STEPS=238` and `ADEMA_ALPHA_WARMUP_STEPS=238` (out of ~477 total iterations) because Apertus's 100k-step warmup, naively scaled to bakeoff, would have been ~14 steps. The current value is defensible per the AdEMAMix paper's "switch optimizers" guidance, but it compresses the post-warmup signal window to ~half the run. Plan eval cadence around this:

- Trajectory metrics (loss / PPL / BPC) every 100M tokens = every ~25 steps — so ~10 checkpoints during warmup vs ~10 post-warmup.
- Downstream eval every 500M tokens = ~4 checkpoints total; only ~2 are post-warmup.
- The §5.6 windowed selection ("last 3-5 checkpoints in 80-100% of budget") is *entirely* post-warmup, which is what matters. Good.
- But the *trajectory* plots will show a misleading "no learning" plateau during steps 0-238 — make sure the per-checkpoint eval doc explicitly notes this.

---

## 2. Bakeoff evaluation (how to read the results)

### 2.1 [PRIMARY] §5.6 hard gates stay anchored on V4-HF (retention), now that the R17 patch lands the arms near V4-HF
**Updated 2026-05-22 after R17 patcher landed bit-perfect.** With `megatron_tp2_r17patched` init checkpoints, the bakeoff arms start essentially at V4-HF logits (verified: `logit_max_abs_diff = 0.0` on three test prompts across all three arms). §5.6's original framing — "regression vs V4-HF retention threshold" — is the correct one.

The V4-postconv framing developed in [`V4_BENCHMARK_COMPARISON.md`](03_4_implementation_experiments/init_bakeoff/eval/V4_BENCHMARK_COMPARISON.md) is now the **un-patched ablation question**, useful for:
- (a) Documenting what R17 costs on a raw HF→Megatron conversion (the empirical case for the patcher);
- (b) Sanity check: if anyone re-runs without the patcher by accident, the per-task baseline comparison set up here is the right floor.

**Concrete §5.6 thresholds (retention against V4-HF):**

- HG1 (English / core retention): per-task drop > Y p.p. on `mmlu` / `arc_challenge` / `hellaswag` / `winogrande` / `piqa`. Y = `max(3 × bootstrap_stderr_V4_HF, 3 p.p.)`. Bootstrap CIs are §2.4 below.
- HG3 (new-token row collapse): same as planned — cosine clustering + near-zero usage from §5.3 diagnostics.
- HG4 (polytonic): planned, but the bakeoff is modern-only per `BAKEOFF_PLAN.md` 2026-05-20 scope update; not testable in this bakeoff.
- HG5 (efficiency / compression ratio): planned — only fires for ReTok/Centroid (extended-vocab).
- HG6 (language-ID drift): still PENDING(custom-eval-construction).

The arms should be expected to *retain* V4-HF performance and *gain* on Greek, not climb from a chance floor.

### 2.2 [PRIMARY] Use the high-headroom / high-baseline tasks as primary bakeoff signals
From [`V4_BENCHMARK_COMPARISON.md`](03_4_implementation_experiments/init_bakeoff/eval/V4_BENCHMARK_COMPARISON.md):

**Primary Greek signals (high V4-HF + large headroom + tight stderr):**
- `belebele_ell_Grek` — headroom 0.41
- `include_base_44_greek_few_shot_en` (use the arts_humanities + business_commerce sub-scores; both Δ > 0.45)
- `global_mmlu_full_el` — headroom 0.29
- `arc_challenge_mt_el` (acc_norm) — headroom 0.22

**Deprioritize:** `xnli_el` (low headroom + near-chance baseline), `global_piqa_completions_ell_grek` (wide stderr), `include_base_44_greek` {`professional_certification`, `health_oriented_education`} (low baselines + small headroom).

### 2.3 [PRIMARY] Add EL/EN cross-lingual ratio columns to `summarize_bakeoff.py`
V4-HF EL/EN ratios are diagnostic:
- ARC: `arc_challenge_mt_el / arc_challenge` = 0.82
- MMLU: `global_mmlu_full_el / mmlu` = 0.87

A successful arm should *raise* these ratios — Greek up without symmetric English uplift. Single-number summary of "did the bakeoff work in the right direction." Cheaper than the full weighted §5.6 score for triage. Worth a column in the summary table.

### 2.4 [PRIMARY] Run `compute_bootstrap_cis.py` on V4-HF and V4-postconv per-sample jsonls
Both V4 runs produced `samples_<task>.jsonl` per task. v0.7 §6.1 prescribes bootstrap-over-samples (not bootstrap-over-runs) for CIs because most benchmark items are deterministic. The script exists at `eval/compute_bootstrap_cis.py` but neither V4 baseline has been processed. CI numbers feed the §5.6 noise floor.

Output target: a `bootstrap_cis.json` in each `v4_*_20260521/` dir; cross-reference from `EVAL_RECIPE.md` so the `PENDING(V4)` cells get concrete numbers.

### 2.5 [PRIMARY] Drop STEM-MMLU as a retention signal
V4-HF `mmlu_stem = 0.4903` — already near chance for some subjects. Doesn't have the headroom to distinguish arms. The English MMLU retention triplet for §5.6 should be `mmlu_humanities` (0.5362), `mmlu_social_sciences` (0.7026), `mmlu_other` (0.6713) — these have signal.

### 2.6 [PRIMARY] Treat survivor tasks differently from collapsed tasks during training
Postconv tasks that retained signal (`winogrande`, `piqa`, `xcopa`, `xnli` aggregate) are 2-3-way MC with strong format priors; they kept ~chance behavior because the R17-broken model still emits *some* output. The collapsed tasks (`mmlu`, ARC, `hellaswag`, `belebele`, `global_mmlu_full_el`) need information retrieval, not just format coherence — these are where arm differentiation will emerge.

**Prediction:** survivor tasks will recover within tens of millions of tokens and then plateau; the collapsed tasks are the bakeoff signal. Don't pull the plug early because survivor numbers look flat — the action is in the 4-way MC tasks.

### 2.7 [PRIMARY] Use `acc_norm` for length-normalized Greek tasks
For `arc_challenge_mt_el` and `global_piqa_completions_ell_grek`, acc_norm beats acc by 0.03-0.04 — Greek tokens are longer and raw acc is length-biased. Use `acc_norm` as the canonical metric for these in `summarize_bakeoff.py`.

### 2.8 [PRIMARY] xquad_el is a binary sanity gate, not a numeric one
V4-postconv f1=0 means the model produces zero token overlap with gold spans — broken generation, not bad generation. Treat xquad_el during training as: did the arm reach f1 > 0.05 at all? If yes, generation is alive. The score is too floor-bound for fine-grained arm comparison until late training.

### 2.9 [NICE] Per-checkpoint eval should split survivor vs. collapsed tasks visually
Survivor-task numbers will look flat (recovery is fast then plateaus); collapsed-task numbers will climb slowly. In one plot it looks like "nothing's happening on survivors"; split them in `run_bakeoff_arm_eval.sh`'s output table so the reviewer can read what's actually moving.

### 2.10 [PRIMARY] Adopt tokenizer-fair training-loss logging
Raw Megatron `lm loss` is per-target-token CE. It is not comparable across
Vanilla's 131,072-token vocab and the 148,480-token extended arms because the
tokenizers have different compression rates and output softmax sizes. Treat raw
`lm loss` as health-only telemetry unless all compared runs use the same
tokenizer.

For current runs, use heldout checkpoint BPC/BPB from
`compute_tokenizer_fair_metrics.py` as the cross-tokenizer loss signal. For
future Megatron runs, patch stdout logging to add measurement-only fields:

```text
lm loss: ... | bpb: ... | bpt: ... | base_loss: ... | new_loss: ... | n_new: ... |
```

Implementation constraints:

- Compute all fields over the same loss-mask positions as `lm loss`, including
  EOD/padding masks and Goldfish masks.
- Reduce numerator/denominator pairs across CP/DP ranks the same way `lm loss`
  is reduced.
- Use `base_vocab_size=131072` for the base/new split.
- Do not change the optimizer loss.

Verification before production: short Vanilla + TD smokes; confirm
`lm_loss / ln(2) / bpt ~= bpb`, Vanilla has `n_new=0` and
`base_loss ~= lm_loss`, and extended arms have nonzero `n_new`.

Documented in
[`LOSS_MEASUREMENT_POLICY.md`](03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md).

---

## 3. Production CPT prerequisites (gating the next phase)

### 3.1 [DONE — 2026-05-21] Finish `patch_apertus_extras.py`
The bakeoff is acceptable with R17 because all three arms inherit the same xIELU + QK-Norm reset. **Production CPT is not acceptable with R17** — recovering 30-50 percentage points across MMLU / HellaSwag / ARC inside a 15-20B-token production budget is unrealistic (the model is at chance level on most info-rich tasks). Empirical evidence in [`V4_BENCHMARK_COMPARISON.md`](03_4_implementation_experiments/init_bakeoff/eval/V4_BENCHMARK_COMPARISON.md).

The scaffold at `init_bakeoff/megatron_patches/patch_apertus_extras.py` needs to be completed:
1. Open the Megatron `torch_dist` shards under `release/mp_rank_*/model_optim_rng.pt`.
2. Walk the per-layer xIELU `act_fn.{alpha_p, alpha_n, beta, eps}` tensors.
3. Overwrite from the HF source's matching keys.
4. Also overwrite `self_attn.{q_norm, k_norm}.weight` for safety, though our R1 diff shows those were within bf16 noise.
5. Re-run V4-postconv after patching → confirm the diff vs V4-HF closes to standard tensor noise floor.

This is **independent of bakeoff arm selection** (all three arms use the same converted checkpoint; patching is a pre-conversion step). It must be validated before the production-CPT submission.

### 3.2 [DONE — 2026-05-21] Validate patcher
**Validated by `verify_hf_roundtrip.py` on all three arms** (jobs 2341182 vanilla / 2341239 retok / 2341241 centroid):

- `standard_max_abs_diff = 0.0`
- `r17_max_abs_diff = 0.0` (xIELU + QK-Norm both at 0)
- `logit_max_abs_diff = 0.0` on 3 prompts (2 Greek + 1 English); top-id match on all.

The patcher passes the "patched HF roundtrip is bit-identical to HF source" test directly — stronger than the originally-proposed indirect test (running V4-postconv on the patched checkpoint). No further validation needed for the bakeoff. For production-CPT release, a full V4-eval on a patched roundtrip is optional confirmation (cheap; identical numbers to V4-HF expected).

### 3.3 [DONE — 2026-05-21] xIELU vs QK-Norm priority moot
The patcher restores **both** xIELU `αp/αn/β/ε` and QK-Norm `q_norm/k_norm` in one pass; the prioritization conversation is closed.

### 3.4 [DONE — 2026-05-21] R17 is no longer a bakeoff confound
With patched init, the arms start near V4-HF performance, not near chance. The "if 2B tokens aren't enough" scenario is no longer driven by R17 recovery — if the bakeoff doesn't differentiate, that's a real signal about the embedding-init effect being small, not about R17.

### 3.5 [BLOCKER for credible production CPT] V1 — eval-set decontamination
`apertus_fidelity_checklist.md` §8.1 calls this "must-do gating" before kickoff and budgets 3-5 days via NeMo Curator on Clariden xfer. I see no evidence this was executed.

If our Greek corpus contains eval-set rows (e.g., Belebele Greek, GreekMMLU passages, INCLUDE-44 leaked into the academic/literary buckets), V4-HF benchmark numbers are inflated and the per-arm Δ becomes incomparable to Apertus-paper baselines.

**For the bakeoff comparison itself** this is partially OK because all three arms see the same corpus, so the *relative* arm comparison is robust to contamination. But the **absolute** numbers and the §5.6 retention thresholds tied to V4-HF would be wrong, and any "this beats Apertus on benchmark X" claim becomes unsafe.

**Action:** run NeMo Curator's eval-set decontamination against `selected_after_apertus_and_internal_dedup.parquet` for at least Belebele Greek + INCLUDE-44 Greek + Global-MMLU Greek pages. Quantify how many rows would have been removed; if < 0.1% of tokens, treat the V4-HF baselines as trustworthy; if > 1%, re-pull the corpus with the contamination overlay applied and re-bake.

---

## 4. Corpus design implications (for production)

### 4.1 [PRIMARY] greek_literary is exhausted at the current dedup_action
The bucket-preserving scheduler caught `greek_literary` exhausting at 122M unique tokens (vs 1.27B target after the 70% Greek allocation). This is what triggered the global-redistribution bug; we fixed the scheduler.

**For production at 15-20B tokens:** Greek will exhaust *more* sources at the same dedup aggressiveness (`drop_intra_and_inter` + `dedup_similarity_threshold=0.85`). Options:
- (a) Relax to `drop_intra` only (intra-source dedup, allow inter-source duplicates) — preserves more tokens at the cost of repetition across sources.
- (b) Lower the dedup_similarity_threshold to 0.92 — keeps more near-duplicates.
- (c) Pull additional Greek sources before production — but the corpus build is already C3-converged per the tokenizer-side memory; this is upstream work.

Decide before production-CPT corpus build.

### 4.2 [PRIMARY] Code source deviation (codeparrot vs StarCoder) needs to be either accepted or reverted
The 2026-05-21 fallback was `codeparrot/codeparrot-clean-train` because BigCode StarCoder / TheStack were gated under our auth. For production:
- (a) Accept the fallback — codeparrot is a reasonable proxy but the §5.6 HG2 ("code retention") thresholds were originally calibrated against StarCoder-style code; revise.
- (b) Authenticate against BigCode and re-pull StarCoder for production — the cleaner choice but requires the auth flow.

Decide before production.

### 4.3 [PRIMARY] The bucket-preserving scheduler is now necessary, not nice-to-have
We caught the `65.18/27.47/4.90/2.45` drift only because we read the manifest before preprocessing. The token-fair scheduler we had before the bucket-preserving fix would have silently let Greek drift to 65% on production-scale runs too — the bug was a feature of the source-level token-fair allocation when a finite source exhausts. The patched `mix_builder.py` should stay as the canonical mix builder; the un-patched version should be tagged as deprecated.

### 4.5 [WATCH] Math 2% bucket isn't in `cpt_plan.md v0.7` §4 design
`bulk.json` allocates `70 / 24 / 4 / 2` to Greek / replay / code / math. `cpt_plan.md v0.7` §4 ("Replay: design space") specifies 24 languages + 4% code share within the 30% non-Greek bucket — math is mentioned in §6.3 (retention benchmarks) but not in the §4 mixture design.

Math at 2% is defensible (Apertus's stage-5 cooldown uses FineMath; we mirror conservatively), but it's an implementation decision made 2026-05-21, not plan-rooted. Two follow-ups:

- (a) Decide whether math stays at 2% for production CPT or is dropped in favour of more replay (24+4 → 30+0).
- (b) Document the decision in `cpt_plan.md` v0.8 (or an addendum) so the bakeoff recipe and the plan don't drift further.

### 4.4 [NICE] Per-bucket exhaustion telemetry in the manifest
The bucket-preserving manifest already reports per-bucket shares. Add to the manifest a per-source "exhausted? yes/no" boolean + actual-tokens-vs-target so the next operator can see at a glance which sources hit their floor. We have the data already (we read it to debug greek_literary); just surface it.

---

## 5. Infrastructure & engineering hygiene

### 5.1 [BLOCKER for next session] Pin `pytorch/v2.9.1:v2` everywhere; don't trust `pytorch/v2.6.0:v1`
`v2.6.0:v1` ships transformers 4.48.3 which lacks `ApertusForCausalLM` — every checkpoint conversion / HF load fails on it. We fell into this trap twice (R1, then init checkpoint conversion). Update all sbatches' `UENV_IMAGE` defaults to `pytorch/v2.9.1:v2`; remove `v2.6.0:v1` references from `_train_config_common.env` and any other docs that still mention it.

### 5.2 [NICE] Document the lm-eval install recipe in a versioned doc
The install pattern is currently:
```
pip install --target=/iopsstor/.../python_envs/lm_eval --quiet .
pip install --target=/iopsstor/.../python_envs/lm_eval --no-deps --quiet accelerate typer duckdb blake3 zstandard polars
rm -rf TARGET/huggingface_hub TARGET/huggingface_hub-*.dist-info
mkdir TARGET/glossapi_rs_noise && copy in our pure-Python stub
```
This is mentioned in `EVAL_RECIPE.md` but not in a script. Create `eval/install_lm_eval.sh` that performs this idempotently; reference from EVAL_RECIPE.md. Today this recipe lives in our heads + a few doc paragraphs — fragile.

### 5.3 [NICE] Document the duckdb `preserve_insertion_order=false` fix as upstreamable
`glossapi_corpus_cli/pipeline.py` now routes every `duckdb.connect()` through `_duckdb_connect_streaming()`. This is a correctness fix for any large-corpus run, not just CSCS. Worth either:
- (a) opening a PR against the canonical `glossapi_corpus_cli` upstream
- (b) noting in `glossAPI`'s shared docs that any `COPY ... ORDER BY` against a >100 GB parquet WILL OOM unless this PRAGMA is set.

### 5.4 [NICE] xfer-partition routing
Memory: xfer is out until 2026-06-11. All CPU-heavy jobs go to `normal` with explicit `--cpus-per-task` / `--mem` / `--gpus-per-node=0`. After 2026-06-11, xfer is preferable (24h walltime vs normal's 12h) for the corpus chain.

### 5.5 [NICE] xIELU isn't CUDA-fused on aarch64
Stderr from every R1/V4-postconv run: `CUDA-fused xIELU not available (No module named 'xielu') – falling back to a Python version`. The Python xIELU is slower than the CUDA kernel. For production CPT at 15-20B tokens this is a non-trivial throughput loss. Either:
- (a) Build `xielu` for aarch64 on Clariden.
- (b) Measure the slowdown on the bakeoff smoke from §1.1 and accept it.

Don't decide until the smoke gives us a tokens/sec baseline.

### 5.6 [NICE] `glossapi_rs_noise` aarch64 wheel
We stubbed it out (the production code path doesn't need `score_markdown_directory_detailed`), but **any future use of the Rust cleaner on Clariden** (cleanup-branch, normalization re-runs) will fail at import. Either:
- (a) Build the aarch64 wheel during normal partition CPU-time and stash it on iopsstor.
- (b) Document that the cleaner code path is x86-only and corpus cleaning must happen on `home`.

We're using (b) implicitly; consider making it explicit in the project README.

### 5.7 [NICE] SIGPIPE / `ls | head` pattern
We hit this in `run_eval.sbatch` once and patched it. There are probably similar `set -o pipefail` + `<cmd> | head` patterns elsewhere in our sbatches. Audit `corpus_build/*.sbatch` and `bakeoff_training/*.sbatch` for this; replace with `<cmd> || true` (when it's tail/post-info, not result-bearing).

### 5.8 [NICE] Sbatch quoting around `srun bash -c "..."` is dangerous
We hit the nested-quoting trap once (`bakeoff_bucketfix_20260521_2133` failure). The current fix uses `printf %q` to quote the command array. Document this pattern as the canonical way to launch under `uenv run ... srun ... bash -c` in our sbatches, so future sbatches don't reinvent it.

---

## 6. Open questions / risks to track

### 6.1 [WATCH] We don't yet have a tokens/sec measurement for TP=2 + mb=2 + xIELU-py on GH200
This is the most load-bearing unknown for the bakeoff budget. Smoke from §1.1 resolves it.

### 6.2 [WATCH] We haven't seen iteration 1 logs on the corrected (TP=2 + mb=2 + expandable_segments) config
The cancelled run got to `training ...` but not iteration 1. Could still OOM. The smoke from §1.1 settles this too.

### 6.3 [WATCH] R17 might not be uniform — the QK-Norm drift was below 1e-3 in this checkpoint
We measured R17 against *one* Apertus checkpoint (the 2509 release). If we re-run R1 on a different Apertus checkpoint (e.g., a future release or a different stage of pretraining), QK-Norm drift might exceed 1e-3 and become a second load-bearing R17 component. Worth a note in the patcher design: handle both xIELU and QK-Norm robustly even if QK-Norm appeared small here.

### 6.4 [WATCH] PF5 — ILSP YAMLs still pending
V4 used 7 Greek tasks; the ILSP suite would add `hellaswag_greek`, `winogrande_greek`, `mmlu_pro_greek`, `truthfulqa_greek`, `medical_mcqa_greek` from the Meltemi/Krikri team's harness fork. Without those:
- Greek signal coverage is narrower than the original plan (7 tasks vs 12).
- Specifically, we don't have HellaSwag Greek (commonsense) or WinoGrande Greek (coreference) — both would diversify our signal beyond MMLU-style multiple choice.

The bakeoff can proceed without PF5, but landing it before per-arm eval makes the §5.6 weighted score more defensible.

### 6.5 [WATCH] cp_comm_type='p2p' and tensor-parallelism comms
The Megatron config printed `cp_comm_type='p2p'` in the bakeoff stdout. With TP=2 across 4 GPUs (DP=2), inter-rank comm is over NVLink within node — fine. If we ever expand to multi-node, this becomes a network-fabric question.

### 6.6 [WATCH] AdEMAMix's two EMA buffers double the optimizer state vs AdamW
That's part of why TP=1 OOMed. For production CPT at 8B model, AdEMAMix has roughly 16 bytes/param × 8B = 128 GB optimizer state (in fp32), vs AdamW's 8 bytes/param × 8B = 64 GB. TP=2 splits this; multi-node TP=2/DP=4 is roughly equivalent at 4 nodes. Tracking for production sizing.

### 6.7 [WATCH] The 2B-token bakeoff budget might be undersized for arm differentiation
If R17 dominates the first ~1B tokens of recovery (all arms relearning xIELU), the embedding-init differences (the actual bakeoff variable) might not surface until tokens 1.5B-2B — leaving only 500M tokens of post-warmup, post-recovery signal. If this is what we see, the conclusion is "bakeoff needs the patcher to be meaningful" (§3.4), not "bakeoff needs more tokens."

---

## 7. Token Distillation — revisit (added 2026-05-22)

### 7.1 [PRIMARY] Scope TD now (code only, no compute) so it can fire fast if the bakeoff is inconclusive
`cpt_plan.md v0.7 §13` brackets Token Distillation with explicit revisit condition: *"if ReTok-vs-Centroid bakeoff is inconclusive"*. The bakeoff results land in 12-24h; if we're going to act on them quickly, the implementation should already exist.

Standalone plan created at [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md). Treat that file as the execution plan; this section is the short suggestion log.

**Several of the v0.6 §13 cost arguments are weaker than v0.6 estimated:**

- *"Untied E/U requires separate LM-head calibration"* — paper handles this trivially: NTP on U rows alongside MSE-on-hidden-states for E rows. Not a separate calibration phase, just a second loss term.
- *"QK-Norm interaction"* — only blocks attention-pattern matching, not hidden-state matching. `cpt_plan.md v0.7 §13` itself notes: *"Use `model.forward()` outputs rather than reimplemented attention math."* Hidden-state-MSE-at-last-layer satisfies this exactly.
- *"xIELU validation"* — same as above; both teacher and student emit hidden states through the same xIELU stack; activation differences cancel.
- *"Layer-choice sweep"* — the Token Distillation reference code defaults to the final hidden layer, but the authors note that roughly one-third-depth layers can work better. No broad sweep is needed, but the TD plan should include a cheap two-layer pilot: final layer vs one-third-depth.

**What's still real:**

- TD adds gradient descent to init → breaks the "all arms are closed-form" property the bakeoff relies on. Add TD as a **4th arm** (`distill_retok` = ReTok-initialized + TD post-pass), not a replacement for one of the three. The other three remain comparable.
- TD needs ~10-50M Greek tokens of source data + ~1-3h on one GH200 per arm. Real but cheap relative to the 12h training arm.
- The reference implementation already freezes the transformer and trains embedding rows, with gradient surgery to preserve original rows. For Apertus, the right factorization is **new-row E + new-row U only**: hidden-state MSE for input rows, and next-token CE for the untied output rows. That isolates "what does TD give us *on top of* ReTok init?" cleanly.

### 7.2 [PRIMARY] Pre-decide the bakeoff→TD trigger
So that when bakeoff results land, the decision is procedural rather than re-litigated. Proposed trigger using the §5.3 diagnostic suite:

| §5.3 diagnostic | Bakeoff "differentiates" floor | Bakeoff "behaviorally dead" floor |
|---|---|---|
| D5 (greedy-gen new-token utilization rate) | best arm > 0.20 AND ≥ 2× worst arm | all arms < 0.05 |
| D2 (prob mass on new tokens) | best arm > 0.10 AND ≥ 2× worst arm | all arms < 0.02 |
| D7 (cosine off-diagonal mean) | best arm < 0.3 AND ≤ 0.5× worst arm | all arms > 0.6 (collapse) |

If at least two of these three favor "differentiates", TD is optional. If at least two favor "behaviorally dead", TD becomes the next experiment.

### 7.3 [PRIMARY] Cheapest TD variant given everything we already have
Implementation outline that reuses existing artifacts (no new corpus or checkpoint builds):

- **Teacher:** the unmodified Apertus-8B-2509 HF base (`/iopsstor/.../models/apertus-8b-2509/`). No copy needed.
- **Student:** the ReTok-initialized HF checkpoint (`init_checkpoints/modern_only_148480/retok/`) — already has resized E/U with ReTok values.
- **Source data:** ~10-50M Greek tokens from `bulk_mix.jsonl`, tokenized two ways:
  - teacher path uses BASE 131,072 tokenizer (new Greek words fragment into subpieces)
  - student path uses EXTENDED 148,480 tokenizer (new tokens whole)
- **Loss:** MSE between last-layer hidden states at the position of the LAST subtoken (teacher) and the position of the SINGLE new token (student). Aggregate over the 17,408 new tokens.
- **Only train new-row E and new-row U**, freeze everything else. Hidden-state MSE updates E; next-token CE updates U. This avoids touching xIELU / QK-Norm and keeps the run as an initialization refinement, not CPT.
- **Compute budget:** ~1-3 hours on one GH200 per arm. Cheaper than R1, R17-patch, or V4-HF eval — fits in a `debug` 1.5h slot if we tune iterations.

### 7.4 [WATCH] Verify the agent's "Kaplan et al. 2025 — Tokens to Words" citation before relying on the 5-12-layer detokenization claim
The agent's argument that early layers (5-12) are "where detokenization completes" for Llama-class 32-layer models is conceptually consistent with logit-lens / tuned-lens results, but the specific paper citation and layer range should be checked before any of our docs leans on it. Doesn't change the bakeoff plan — we'd use last-layer TD anyway per the original paper's principled default. Just don't quote "Kaplan 2025" as an authority without confirming.

### 7.5 [WATCH] If TD-on-top-of-ReTok beats vanilla ReTok meaningfully, full TD becomes the production path
And then the production CPT needs:
- TD on ReTok init (or whichever arm wins on §5.3) before the 15-20B-token CPT fires.
- The xIELU patcher (`patch_apertus_extras.py`, already done) so the student's downstream activations match the teacher's.
- A larger source-data slice (~100-200M Greek tokens) for the production TD run.

These three together close the loop: closed-form init for scaffolding → TD for manifold placement → patcher for activation consistency → CPT for full task transfer.

---

## How to keep this doc current

Add new entries under the appropriate section as findings come in. Existing entries stay (this is a forward-looking log, not a state snapshot). When an entry's recommended action is taken, mark it `[DONE — <date>]` rather than deleting it — the historical context matters for the reviewer.
