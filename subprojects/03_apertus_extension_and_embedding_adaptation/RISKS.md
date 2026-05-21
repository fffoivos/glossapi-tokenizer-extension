# Silent-failure risks (2026-05-21)

*Things that could be wrong in our bakeoff implementation that our current tests / controls **wouldn't reliably catch**. The bakeoff would run, produce numbers that look fine, and we'd conclude the wrong thing.*

Maintained as a living issues file. Each entry: what goes wrong silently + why current controls don't catch it + a candidate mitigation. Mitigations are not implemented unless explicitly flagged.

## Tier 1 — could invalidate the bakeoff entirely

### R1. HF→Megatron QKV interleaving in `loader_apertus_hf.py` is untested

**Mechanism.** Apertus uses GQA with `num_heads=32`, `num_kv_heads=8`, `heads_per_group=4`. `saver_swissai_hf.py` builds `q_slice / k_slice / v_slice` to gather from Megatron's interleaved `qkv_weight[(num_heads+2·num_kv_heads), head_dim, hidden]` into HF's separate `q_proj / k_proj / v_proj`. Our loader implements the **inverse** of that pattern in `_interleave_qkv`.

**What goes wrong silently.** A wrong stride (off-by-one in `(heads_per_group + 2)*g + heads_per_group`, or wrong nesting of Q/K/V into a group) produces a tensor of the **same shape** with Q heads bound to wrong K/V groups. The converted model loads cleanly, forward passes succeed numerically, but attention is computed against permuted keys/values — outputs are noise. The model fine-tunes from that state into something coherent but unrelated to Apertus-base.

**Why current controls don't catch it.** The AST-parse check in `install.sh` only confirms the file is syntactically valid. The roundtrip validation procedure in `megatron_patches/README.md` would catch this exactly (HF → Megatron → HF should byte-equal the original) **but has not been run yet** — it needs Clariden + Apertus-8B-2509 weights.

**Mitigation.** Run the documented roundtrip on Apertus-8B-2509 before the first bakeoff sbatch. Hard gate: `max abs diff < 1e-3` on bf16-quantised weights. The procedure is in [`init_bakeoff/megatron_patches/README.md`](03_4_implementation_experiments/init_bakeoff/megatron_patches/README.md) §"Roundtrip validation procedure".

---

### R2. Token streams across the three arms may not be byte-identical

**Mechanism.** v0.7 §5 + `BAKEOFF_PLAN.md` claim the three arms share a seed and therefore see identical token streams; the only differential is init. We rely on `datasets.interleave_datasets(..., seed=DATA_SEED)` for this property.

**What goes wrong silently.** `datasets.interleave_datasets` is **not strictly deterministic** in streaming mode if:
- HF library version differs between arm submissions (`datasets` 2.x vs 3.x changed shuffling internals)
- HF cache state differs (some shards prefetched, others not, on different runs)
- Number of dataset shards visible at download time differs (HF Hub may add shards mid-experiment)
- `pyarrow` version differs (affects shard-iteration order)

If arm A and arm B see even slightly different streams, our "apples-to-apples" claim is broken — we're comparing init + data drift.

**Why current controls don't catch it.** We compute an MD5 / SHA nowhere. `mix_builder.py` writes a manifest with per-source token counts but not a stream-identity hash. Two independent runs with the same seed could produce slightly different JSONLs and we wouldn't know.

**Mitigation.** Add MD5-of-output-JSONL to `mix_builder.py` manifest. Re-runs that don't match the hash fail loud. ~15 min of work; not done.

---

### R3. Held-out Greek eval slice doesn't exist + cleanliness is unverified

**Mechanism.** `compute_tokenizer_fair_metrics.py` and `compute_new_token_diagnostics.py` consume a held-out JSONL of Greek docs. v0.7 §5.1 + §6.1 require this slice to be (a) disjoint from the bakeoff training mix, (b) disjoint from Apertus's pretraining corpus, (c) representative across registers.

**What goes wrong silently.** If the held-out slice contains anything Apertus saw during pretraining (e.g., we slice from HPLT-clean60 without applying the Apertus-overlap drop), Apertus-base looks artificially strong from memorization. The bakeoff arms' actual capability gains are masked by the base's inflated baseline; the wrong arm wins selection.

**Why current controls don't catch it.** The slice doesn't exist yet. When constructed, it would need item-level dedup against (a) our bakeoff training mix manifest and (b) Apertus's pretraining-corpus signatures. We have machinery for (a) (the dedup audit's overlap-drop overlay) but not for (b) at item granularity.

**Mitigation.** The natural source is the **dedup-audit val/test partition** built in 2026-04 (the gcloud-loss-affected artifact). Reconstruction path is documented at [`03_3_cscs_experiments_kickoff/ANALYSIS.md`](03_3_cscs_experiments_kickoff/ANALYSIS.md) "Review checkpoint B" with three options ranked by defensibility. Option B (re-run the splitter on Clariden xfer with the original seed) is the most defensible. Decision pending.

---

### R4. ReTok / Centroid surface-form decode has a leading-space artifact

**Mechanism.** Apertus uses ByteLevel BPE (Mistral-Nemo tekken v3 inheritance). Word-initial tokens are prefixed with `Ġ` (the byte-level marker for ASCII space). `extended_tokenizer.decode([new_id])` may return surface with or without a leading space depending on whether the new token is a word-initial merge:
- New ID = "Ġ Ελλάδα" → decoded as " Ελλάδα" (with leading space)
- New ID = "α" (word-internal) → decoded as "α" (no leading space)

`base_tokenizer.encode(surface, add_special_tokens=False)` then produces **different subpiece chains** depending on whether the input starts with a space.

**What goes wrong silently.** For word-initial new tokens (the majority of the 17,408 modern Greek extensions, since BPE merges tend to absorb word-initial space), ReTok's subpiece-mean is computed from a slightly different decomposition than the new token would actually appear in during training. Centroid is unaffected (it doesn't depend on the new token's own subpieces). So ReTok systematically diverges from the prescription it claims to implement.

**Why current controls don't catch it.** `test_init_logic.py` validates shape + norm + that ReTok != Centroid output. It doesn't sample any new token, manually compute the expected ReTok mean, and assert match.

**Mitigation.** A 30-line addition to `test_init_logic.py`: pick 10 new tokens with surface forms starting and not starting with Greek characters, manually decompose, compute the expected mean, assert. ~30 min; not done.

---

## Tier 2 — could subtly bias results

### R5. Norm targets E = 5.05 / U = 3.80 measured on an older Apertus snapshot

**Mechanism.** Phase A norm targets in `arms/_common.py` are `NORM_TARGET_E_GREEK = 5.05` and `NORM_TARGET_U_GREEK = 3.80`. These came from `runs/apertus_greek_diagnostic_20260511_v2/` — a May-11 measurement on an Apertus snapshot. Both ReTok and Centroid post-pass-norm-match to these values.

**What goes wrong silently.** If `swiss-ai/Apertus-8B-2509` has a different Greek-row norm distribution than the May-11 diagnostic measured (different checkpoint sub-version, different layer-norm scaling, model card update we missed, …), our new rows are systematically mis-scaled vs. existing Greek rows. The bakeoff selects an arm that "wins" because its mis-scaled rows happen to play better with the test-time logit distribution, not because its init is genuinely better.

**Why current controls don't catch it.** We never recompute the norm distribution at build time and compare to the hardcoded constants.

**Mitigation.** In `build_init_checkpoints.py`, compute base Greek-row p50 norms on the loaded checkpoint; warn (or fail) if they differ > 5 % from the hardcoded targets. ~30 min; not done.

---

### R6. Apertus-overlap drop silently fails if `doc_key` schema mismatch

**Mechanism.** `pull_greek_corpus.sh` pulls `fffoivos/apertus-c3-dedup-audit-dedup-...` overlay parquet. `mix_builder.py:_load_drop_keys` reads it, expects `doc_key` column (with `hf_pool_doc_key` / `doc_id` fallbacks), and filters the nanochat stream by membership. The dedup audit reports a ~15 % drop rate of nanochat docs against this overlay.

**What goes wrong silently.** If the overlay was built against an older nanochat version whose `doc_key` field has different formatting (e.g., trailing whitespace, normalized vs raw form, different prefixes), the membership check returns False for every key. The filter drops zero docs. We silently train on data Apertus has already memorized → the bakeoff measures memorization vs init, not init quality alone.

**Why current controls don't catch it.** `mix_builder.py` logs the number of loaded drop keys and produces a per-source token count in the manifest, but doesn't assert "non-trivial drop rate" against the input.

**Mitigation.** Sanity assertion in `_build_source_stream`: after first batch of N docs, if drop rate < 1 % when overlay is configured, log a loud warning. Optionally fail. ~10 min; not done.

---

### R7. lm-eval-harness commit not pinned

**Mechanism.** `eval/pull_benchmarks.sh` does `git clone https://github.com/swiss-ai/lm-evaluation-harness.git` without checking out a specific commit. `run_eval.sbatch` captures the commit at runtime in `run_metadata.json` for the audit trail but doesn't assert against a pinned value.

**What goes wrong silently.** V4 baseline runs at harness commit X. Two weeks later, bakeoff per-arm evals run at commit Y after a `git pull`. Task definitions can change between X and Y — e.g., HellaSwag's gold-key extraction, XNLI's per-language splits, MMLU's prompt format. The reported deltas between V4 baseline and per-arm results are then a mix of (real bakeoff effect) + (harness drift).

**Why current controls don't catch it.** We record the commit but don't compare across runs. Cross-checkpoint comparison is downstream — by the time we'd notice the inconsistency the eval runs have already cost compute.

**Mitigation.** Pin a specific commit in `pull_benchmarks.sh` (`git checkout <pin>` after clone). In `run_eval.sbatch`, assert `git rev-parse HEAD == $PINNED_COMMIT`. ~15 min; not done.

---

### R8. EoD token ID in extended vocab not explicitly verified to match base (V14)

**Mechanism.** Megatron-LM-Swiss-AI's cross-doc attention plumbing (`--reset-attention-mask --reset-position-ids --eod-mask-loss`) marks document boundaries by looking for the **EoD token ID** in the tokenized data. The EoD ID is read from the tokenizer at preprocess-data time.

**What goes wrong silently.** If our tokenizer extension shifted special-token IDs in any way — e.g., if `add_tokens` placed new Greek tokens BEFORE the EoD specials instead of after — then the EoD ID baked into our binary dataset doesn't match the EoD ID Megatron derives from the loaded model. Cross-doc attention bleeds → bakeoff trains on effectively-concatenated multi-document context. The model still learns; the LR / norm dynamics shift; the comparison to Apertus-base is contaminated.

**Why current controls don't catch it.** V14 in `cpt_plan_v0.7_status.md` says "likely already verified during extension work" but is marked `unconfirmed`. `build_init_checkpoints.py` doesn't perform an explicit special-token-ID equality check between base and extended tokenizers.

**Mitigation.** Add a one-liner in `build_init_checkpoints.py`: assert `base_tokenizer.special_tokens_map == extended_tokenizer.special_tokens_map` and `base.eos_token_id == extended.eos_token_id`. ~15 min; not done.

---

### R9. `build_init_checkpoints.py` doesn't assert xIELU αp/αn survive `resize_token_embeddings` (Q12)

**Mechanism.** Apertus's xIELU activation has per-layer trainable `alpha_p` and `alpha_n` scalars (and optional `beta`, `eps`). Audit Q12 confirmed they're registered as `nn.Parameter` children of each XIELU module, so `model.parameters()` includes them and the optimizer picks them up. `resize_token_embeddings` should only touch the embedding tensor + LM head, leaving XIELU modules intact.

**What goes wrong silently.** If a future `transformers` release changes `resize_token_embeddings` to rebuild more than just the embedding (e.g., creates a fresh model and copies forward — happens for some HF surgery patterns), the XIELU modules could be replaced with fresh-init versions. The per-layer `αp = αn = 0.8` defaults would be RE-INSTALLED but any **trained** values from Apertus pretraining would be lost. The bakeoff then trains from xIELU-at-init rather than xIELU-as-Apertus-trained — fidelity drift.

**Why current controls don't catch it.** The audit recommended an assertion (`before == after` for all `alpha_*` params before/after resize). Not yet added to `build_init_checkpoints.py`.

**Mitigation.** ~15 min: snapshot all `*.alpha_p`, `*.alpha_n`, `*.beta`, `*.eps` parameter tensors before resize, snapshot after, `torch.equal()` for each pair, raise on mismatch. Not done.

---

## Tier 3 — would be caught eventually

| # | Risk | Why lower priority |
|---|---|---|
| R10 | `--dist-ckpt-strictness assume_ok_unexpected` is broadly lenient — accepts any shape mismatch, not just the intended embedding resize | First training step would NaN if the mismatch is real (forward pass fails immediately) |
| R11 | Centroid full-Σ Cholesky rank-deficiency on the 1,507-modern-Greek base subset (rank ≪ 4096) | 1e-8 ridge handles numerically; D7 (cos-similarity / effective-rank) diagnostic catches collapse downstream |
| R12 | STRR word-splitter regex edge cases (Greek apostrophes, άνω-τελεία `·`, polytonic combining marks) | Affects absolute STRR but per-arm comparison stays consistent (same splitter) |
| R13 | `classify_greek_block` doesn't tag NFD-decomposed Greek tokens fully (`α` + combining-acute counted as modern-only, not polytonic-flagged) | Modern-only bakeoff makes this near-moot; only matters for the future polytonic specialization run |
| R14 | TP=1 numerical drift vs Apertus's pretraining TP=2 (different all-reduce order in bf16) | Sub-bf16 numerical drift; not catastrophic; affects the absolute Apertus-base score but the per-arm comparison is internally consistent (all three arms use TP=1) |
| R15 | FineMath share at 2 % is a guess (Apertus's per-source pretraining shares aren't published) | Wrong number is recoverable post-V4 if math retention regresses on lm-eval-harness; doesn't invalidate the init comparison |
| R16 | OPUS Greek-English parallel data excluded (schema-incompat with mix_builder) | v0.7 §4.4 marks it optional; absence is documented |

---

## Hard prerequisites (gated on Clariden / external setup, not our code)

These aren't risks per se — they're items that must be settled before the bakeoff can submit:

1. **R1's roundtrip validation** on Apertus-8B-2509 (Clariden GPU + weights)
2. **R3's held-out eval slice construction** (the gcloud-loss val/test partition decision)
3. **ILSP harness task YAMLs merge** from Meltemi/Krikri forks (already documented as a staging-time step)

---

## Cheap mitigations available (not yet implemented)

The R-numbered mitigations above that are tagged "~15-30 min" are independent of Clariden — they're additions to local Python / shell. Bundle:

| Risk | Where | Effort |
|---|---|---|
| R2 | MD5-of-JSONL in `mix_builder.py` | 15 min |
| R4 | ReTok subpiece-decode unit test | 30 min |
| R5 | Norm-drift check in `build_init_checkpoints.py` | 30 min |
| R6 | Drop-rate sanity assertion in `mix_builder.py` | 10 min |
| R7 | Pin lm-eval-harness commit in `pull_benchmarks.sh` + assertion in `run_eval.sbatch` | 15 min |
| R8 | Special-token equality check in `build_init_checkpoints.py` | 15 min |
| R9 | xIELU scalar-survival assertion in `build_init_checkpoints.py` | 15 min |

Total: ~2 hours for all 7. Decision on whether to land these as a bundle is open.

---

## Why this matters

Most of these risks share a structural property: **they're silent because we don't have the test, not because the test is hard to write**. R2, R5, R6, R7, R8, R9 are all single-digit lines of assertion code that we never wrote.

The exception is R1 + R3: those require Clariden access — a roundtrip on Apertus weights, and a clean held-out partition. Those are the ones to flag to the reviewer as gating the bakeoff.

R4 is in between: the test is cheap (~30 min) but it could reveal a real algorithmic flaw in ReTok that would invalidate one whole arm of the bakeoff. Worth running soon.
