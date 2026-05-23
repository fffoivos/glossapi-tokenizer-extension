# Token Distillation Parallel Plan

Status: draft execution plan, 2026-05-22

Purpose: prepare a bounded Token Distillation challenger that can start quickly if
the current Vanilla / ReTok / Centroid bakeoff leaves ReTok promising but
under-initialized. This is a companion plan, not a change to the running
three-arm bakeoff.

Primary external references:

- Paper: https://arxiv.org/abs/2505.20133
- Official implementation: https://github.com/konstantinjdobler/token-distillation

## 1. Decision Summary

Token Distillation is feasible for Apertus without inventing a new method. The
earlier "bracketed" status was too conservative if it was based on tied-vs-untied
embedding concerns. The official implementation learns input embeddings by
hidden-state distillation and has explicit untied-output handling through a
next-token CE path.

What is still real is integration risk:

- Apertus uses untied `embed_tokens` and `lm_head`, so `E` and `U` must be
  checked independently.
- Our tokenizer is an extended BPE tokenizer with fixed production IDs, not a
  base tokenizer plus `add_tokens(...)`.
- The distillation run must preserve xIELU and QK-Norm values and must survive
  the same HF -> Megatron -> HF roundtrip gate as the other arms.
- Layer choice is not a free constant: the paper defaults to the final hidden
  layer, while the package README says earlier target layers around the
  one-third mark can work better. We will run a small layer pilot that tests
  both: final layer plus the package-suggested one-third layer. If the
  logit-lens/tuned-lens probe in Section 6.1 suggests a different layer, test
  that as well.

## 2. When To Start

Do not interrupt the running three-arm bakeoff. Token Distillation can begin
once all of these are true:

1. The current bakeoff has reached the final scheduled checkpoint, or the
   final checkpoint is safely saved and the remaining work is only evaluation.
2. ReTok is not already decisively beaten by Vanilla on both Greek BPC and the
   new-token diagnostics.
3. One spare GH200 node can be used without delaying final bakeoff evals.
4. The implementation runs first on a small token subset and passes all
   preservation gates in Section 8.

Recommended trigger:

- Run the Token Distillation pilot if ReTok has healthy new-token behavior
  but still trails Vanilla on Greek BPC or downstream Greek tasks.
- Do not run it if Vanilla is clearly best and ReTok has weak new-token use;
  in that case the tokenizer extension itself is probably not helping enough.

## 3. Experiment Shape

Name: `retok_td`

The experiment is not "TD from scratch." It is ReTok plus a TD refinement:

- Logical teacher: the original-tokenization path through the same frozen
  Apertus weights. In implementation this does not need to be a second 8B model
  in memory: the extended student can run the unmerged/base-tokenized sequence
  under `torch.no_grad()` because all original rows and transformer weights are
  frozen and identical to the base model.
- Student: ReTok-initialized HF checkpoint with the exact modern-only extended
  tokenizer used by the current bakeoff.
- Input objective: hidden-state MSE between teacher runs using base-tokenized
  spans and student runs using the corresponding single new token.
- Output objective: next-token CE on the merged sequence, with gradients allowed
  to update only the new `lm_head` rows.
- Trainable parameters: new input rows and new output rows only.
- Frozen parameters: all transformer blocks, all original embedding/output
  rows, xIELU scalars, QK-Norm weights, RMSNorm weights, and all other base
  model parameters.

This isolates the question: "Does TD improve ReTok's placement of the new rows?"

## 4. Model Format Plan

Token Distillation should run in HF format, not Megatron format.

Inputs:

- Teacher/reference HF model: the unmodified Apertus HF base, used for validation
  and optional row-diff checks. Do not require it to stay resident during TD
  training unless an implementation choice truly needs it.
- Teacher tokenizer: base 131,072-token Apertus tokenizer.
- Physical training model: ReTok HF init checkpoint for the modern-only
  148,480-token tokenizer. This model serves both the merged/student path and
  the unmerged/logical-teacher path.
- Student tokenizer: exact shipped extended tokenizer from
  `03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_only_148480/`.

Outputs:

- HF checkpoint for `retok_td`.
- Exact student tokenizer copied through unchanged.
- A manifest with model paths, tokenizer hash, base vocab size, extended vocab
  size, trainable row ranges, layer choice, data slice, and source commit of
  the Token Distillation code.
- Megatron TP=2 `torch_dist` checkpoint produced with the same R17-preserving
  conversion path used by the current bakeoff.

Sensitive point: do not let the reference implementation call
`target_tokenizer.add_tokens(...)` for production. That would create added-token
behavior and may change IDs/segmentation relative to our BPE extension. Patch or
wrap the implementation so the student tokenizer is loaded directly and treated
as authoritative.

Second sensitive point: do not load two 8B models just because the method is
described as teacher/student. The official training loop uses one extended model
for both the merged and unmerged forwards; preserve that memory-efficient shape
unless a later validation test proves it insufficient.

## 5. Token Mapping And Data

For each new token ID in the extended tokenizer:

1. Decode the token string using the same ByteLevel-aware readable-token logic
   used in the tokenizer audits.
2. Tokenize the same surface string with the base tokenizer to get the teacher
   phrase IDs.
3. Tokenize it with the extended tokenizer and verify it maps to exactly the
   intended new token ID in the relevant context.
4. Mine snippets from the CPT Greek corpus where that surface string occurs.

Use real corpus snippets, not generated snippets, for the main run. Generated
snippets are acceptable only as a fallback for very low-coverage tokens in a
separate flagged bucket.

Initial data budgets:

- Smoke: 256 to 512 high-frequency new tokens, 20 to 50 snippets per token.
- Layer pilot: 1,024 to 2,048 tokens, 50 snippets per token, comparing
  last-layer TD against the package-suggested one-third-depth layer and, if
  available, the Section 6.1 probe-suggested layer.
- Full modern run: all 17,408 modern new tokens, target 100 snippets per token
  when available.
- Production polytonic extension: only after the modern path passes; then repeat
  for the additional +5,120 polytonic tokens.

Coverage policy:

- Do not silently skip tokens. Emit `coverage.json` with requested snippets,
  found snippets, used snippets, and fallback action per token.
- Split results into enough-coverage and low-coverage buckets during analysis.
- If more than 10 percent of new tokens have fewer than 20 real snippets, stop
  before full TD and inspect the token inventory/corpus mismatch.

## 5.1 Firing Prepass On The Exact 2B Dataset

Before any GPU TD run, count actual new-token firings and usable snippets on the
exact normalized text/parquet slice that produced the 2B-token bakeoff dataset.
Do this on the pre-packed source text, not only on Megatron `.bin`, because TD
needs text spans and 50-token contexts.

This is a CPU/I/O job and should run as a sidecar only after the current bakeoff
final checkpoints are safely saved, or while final downstream evals are running.
Do not let it compete with training checkpoint saves or packed eval cache reads.

Output artifact:

- `td_coverage_prepass.jsonl`: one row per new token.
- `td_coverage_summary.json`: aggregate thresholds and recommended action.
- `td_snippet_index/`: sampled snippet references or extracted tokenized snippets
  keyed by `new_token_id`.

Required row fields:

- `new_token_id`
- `token_string`
- `base_subtoken_ids`
- `base_subtoken_len`
- `extended_firings`
- `raw_surface_occurrences` where cheap to compute
- `usable_snippets_25`
- `usable_snippets_100`
- `docs_with_firing`
- `example_snippet_refs`
- `status`: `enough_100`, `enough_25`, `low_20_24`, `low_lt20`,
  `zero`, or `mismatch`
- `recommended_action`: `td_100`, `td_25`, `keep_retok`,
  `generated_fallback_candidate`, or `inspect`

Counting rule:

- A firing means the extended tokenizer actually emits that new token ID.
- Merge ancestry does not count.
- A surface substring does not count unless it corresponds to the emitted
  new-token span under the extended tokenizer.
- For each firing span, tokenize the same surface span with the base tokenizer
  to build the logical-teacher sequence.

Decision gate:

- If at least 90 percent of new tokens have `usable_snippets_100 >= 100`, run
  full TD at the 100-snippet setting.
- If at least 90 percent have `usable_snippets_25 >= 25`, run TD at the paper's
  fast 25-snippet setting and keep a flagged tail strategy.
- If more than 10 percent have `usable_snippets_25 < 25`, do not launch full TD.
  Inspect whether the problem is dataset coverage, tokenizer/string decoding, or
  low-value tail tokens.
- Tokens with `usable_snippets < 20` keep ReTok by default. Generated snippets
  are allowed only as a separately reported fallback bucket.
- Tokens with `status = mismatch` block the TD run until the tokenizer/span
  mapping is fixed.

## 6. Layer Choice

**Default candidate for Apertus 8B: last layer (`target_layer = -1`). Required comparison candidate: package-suggested one-third-depth layer. No broad sweep.**

The paper argues for last layer as the conservative, portable default. Direct quotes:

> *"In practice, we simply use the last layer's hidden state but analyze this choice in Section 5.3."* — paper §3.2, p.4.

> *"For our main experiments, we choose to keep the last layer because this choice does not necessitate a model-specific sweep over target layers. Also, the last layer is a principled choice, as it guarantees that no subtoken interactions that are only modeled in later layers are excluded from the objective."* — paper §5.3 (per a second-pass review of the paper's layer-choice analysis).

The argument for the last layer is **conservative-and-correct**: any way the original subtoken sequence influenced downstream computation must, by construction, have shown up by the final hidden state. Matching there upper-bounds the objective. But the package README is explicit that earlier target layers around the one-third mark can yield better results in many cases. That means we should not pick last-layer only without checking the package-suggested alternative.

Implication for Apertus 8B specifically: run a **small** layer pilot, not a sweep.

- Candidate A: `target_layer = -1`, the paper-default last layer.
- Candidate B: `target_layer = ceil(num_hidden_layers / 3)` in HF hidden-state tuple indexing after confirming Section 16 Q11. For Apertus-8B's 32 transformer layers, expected candidate is `target_layer = 11` (roughly after the first third of blocks).
- Candidate C: if the Section 6.1 logit-lens/tuned-lens probe suggests a materially different detokenization layer `L*`, test that layer too. If `L*` is within +/-1 of Candidate B, do not add a third candidate.
- Implementation note from the first live smoke: the vendored package does **not** ship a separate layer-suggestion tool. Its actionable upstream guidance is the README note that one-third-depth target layers can work better than the last layer. Therefore the first layer pilot should compare Candidate A and Candidate B only, unless we have a well-defined Apertus-specific probe result.

Selection rule:

- Choose the candidate with better heldout Greek BPC/NLL and D1/D2/D4/D5 on the layer-pilot subset.
- If differences are within noise, choose last layer for portability and to avoid model-specific overfitting.
- Do not test more than three candidates unless the first TD run fails in a way that is clearly layer-specific.

For polytonic Phase 2 specifically (the +5,120 polytonic tokens, deferred per §5): **also default to last layer**. The detokenization machinery for unfamiliar diacritic-heavy sequences plausibly uses more late-layer compute than for typical modern words — the model has to reassemble diacritic + base-char interactions it saw less cleanly during pretraining. The conservative default earns its keep here.

### 6.1 Layer-suggestion probe to find the "detokenization layer" L*

Before the layer pilot, run a cheap logit-lens probe to get our best Apertus-specific guess for where the residual stream finishes resolving multi-token Greek words into a single contextual representation:

1. Take a held-out set of multi-token Greek words from our corpus (~200-500 words that the base tokenizer fragments into 2-5 subpieces).
2. Run them through the **original Apertus base** (no TD), recording the residual stream at every layer at the position of the LAST subtoken.
3. Project each layer's residual through Apertus's `lm_head` (logit lens) OR a learned per-layer linear probe (tuned lens — Belrose et al. 2023, *Eliciting Latent Predictions from Transformers with the Tuned Lens*, arXiv:2303.08112).
4. Find the layer index `L*` where the top-1 projection first reliably matches the whole-word identity (i.e., where the model has "decided" what whole word it's processing).

For Llama-class 32-layer 8B models, `L*` is reportedly in the first third of layers (per a Kaplan et al. 2025 "Tokens to Words" line of work — **citation needs verification before being load-bearing**). For Apertus, plausibly similar (`L* ≈ 8-12` of 32), but Apertus's xIELU + QK-Norm could shift this; the only way to know is to run the probe.

**Cost:** one forward pass per probe sample × 32 layers of projection ≈ trivial on one GH200 (< 30 min for 500 words).

**What it gives us:** an *informative diagnostic* for the paper-quality writeup ("here's where Apertus does detokenization"), and a sanity check that last-layer TD isn't wasted compute (if `L* = 5` and we train at layer 32, we're propagating identical signal through 27 layers per backward pass). If we find `L* = 25+`, that's also informative — Apertus's late layers may be more LM-head-staging than Llama's, in which case last-layer is doubly the right choice.

**Action:** run the cheap logit-lens version after the first successful TD smoke
and before the layer pilot. If it suggests a clear `L*` that differs from the
one-third-depth layer, include `L*` as Candidate C in the layer pilot. The
heavier tuned-lens version remains optional and should not block the modern TD
path. Tracked as task TD7 in §9.

## 7. Output Embeddings

Apertus has untied embeddings, so `lm_head` rows must be handled deliberately.

Primary output strategy:

- Start from the ReTok `lm_head` rows already present in the student checkpoint.
- During TD, train only the new output rows with next-token CE.
- Keep original output rows frozen and verify they are unchanged exactly.
- Keep the input hidden-state MSE and output CE accounting separate in logs.

Fallbacks:

- If CE makes output norms unstable, train only input rows and keep ReTok output
  rows fixed.
- If ReTok output rows look bad before TD, use subtoken-mean output initialization
  from the base output matrix, then CE-train only new output rows.

Avoid using a tied-embedding shortcut. It does not apply to Apertus.

## 8. Verification Gates

The TD run is not usable until all of these pass.

Tokenizer gates:

- Extended tokenizer files are byte-identical before and after TD.
- Base special token IDs and extended special token IDs match the current
  bakeoff tokenizer manifests.
- New token IDs equal the planned contiguous range.
- A fixed sample of Greek strings tokenizes exactly as in the ReTok arm.

Model gates:

- Original input rows unchanged exactly.
- Original output rows unchanged exactly.
- New input/output rows finite, nonzero, and within documented norm envelopes.
- xIELU scalar tensors unchanged exactly.
- QK-Norm tensors unchanged exactly.
- Logits on prompts that do not use new tokens remain unchanged or within a
  documented numerical tolerance.

Roundtrip gates:

- Convert HF `retok_td` -> Megatron TP=2 with the R17-preserving path.
- Convert back or run the existing roundtrip verifier.
- Require zero or near-zero diff for xIELU, QK-Norm, and unchanged base rows.
- Run a short Megatron load/train smoke before queueing any CPT-scale arm.

Quality gates:

- Pre-CPT BPC should improve over ReTok on the heldout Greek slice.
- D1/D2/D4/D5 should improve or remain healthy versus ReTok.
- No obvious English retention regression on the small prompt/eval sanity set.
- If pre-CPT gates are neutral, do not spend a full 2B-token arm on TD.

## 9. Implementation Tasks

Task TD0 - vendor or pin reference code:

- Pin the official Token Distillation repo commit in a manifest.
- Do not rely on a moving GitHub checkout from inside a Slurm job.

Task TD1 - exact-tokenizer adapter:

- Add a wrapper that accepts separate teacher and student tokenizers.
- Disable `add_tokens(...)` in the production path.
- Build explicit mapping:
  `new_token_id -> token_string -> base_phrase_ids -> student_token_id`.

Task TD2 - local corpus snippet source:

- Implement a local JSONL/parquet snippet miner over the existing CPT corpus
  slice.
- Cache snippets in a reusable artifact keyed by tokenizer hash and corpus hash.
- First mode is the Section 5.1 firing prepass: count actual extended-token
  firings and usable snippets on the exact 2B bakeoff text slice, then produce
  `td_coverage_prepass.jsonl`, `td_coverage_summary.json`, and
  `td_snippet_index/`.

Task TD3 - embedding-only trainer:

- Load the ReTok HF student once and use it for both the merged/student forward
  and the unmerged/logical-teacher forward.
- Freeze all non-new rows and all transformer parameters.
- Backprop hidden-state MSE to new input rows.
- Backprop CE to new output rows only.
- Emit loss, norm, gradient, and coverage telemetry.

Task TD4 - pilot runs:

- Run smoke on 256 to 512 high-frequency new tokens.
- Only choose smoke tokens from `status in {enough_100, enough_25}` in the
  Section 5.1 prepass.
- Run the Section 6.1 logit-lens probe to get a candidate detokenization layer
  `L*`.
- Run a layer pilot on 1,024 to 2,048 tokens comparing:
  - `target_layer=-1` (paper default),
  - `target_layer=ceil(num_hidden_layers / 3)` (package README suggestion;
    expected Apertus-8B value: `11` after Q11 indexing confirmation),
  - `target_layer=L*` if the probe suggests a materially different layer.
- Pick the output strategy before full TD.

Task TD5 - full modern TD:

- Run all 17,408 modern new tokens.
- Save HF checkpoint, manifest, coverage, and diagnostics.
- Run verification gates.

Task TD6 - optional fourth CPT arm:

- Only after TD5 passes, convert to Megatron and run a short CPT smoke.
- If smoke passes and pre-CPT diagnostics beat ReTok, submit `retok_td` as a
  separate fourth arm or a shorter challenger run.

## 10. Risks And Mitigations

Tokenizer mismatch:

- Highest-risk integration point. Mitigation: never regenerate the tokenizer;
  load the exact extended tokenizer and assert ID/sample-tokenization equality.

Layer mis-selection:

- Mitigation: test the paper-default last layer against the package-suggested
  one-third-depth layer and, if available, the tool/probe-suggested layer.
  Keep the candidate set tiny to avoid turning TD into a layer-sweep project.

Output-row instability:

- Mitigation: keep ReTok `U` as the starting point, CE-train only new rows, and
  fall back to fixed ReTok `U` if norms or logits become unstable.

Silent base-model drift:

- Mitigation: exact row-diff checks after every pilot and full run.

xIELU/QK-Norm drift:

- Mitigation: freeze all non-embedding parameters and reuse the R17 roundtrip
  verifier.

Coverage skew:

- Mitigation: coverage manifest, frequency buckets, and a hard stop if too many
  tokens lack real snippets.

Resource overuse:

- Mitigation: one-node pilots first; do not queue a full fourth arm before the
  current bakeoff result is interpretable.

## 11. Expected Cost

Order-of-magnitude expectation:

- smoke: less than 1 node-hour,
- layer pilot: 1 to 2 node-hours,
- full modern TD: a few node-hours,
- optional CPT challenger: same training cost shape as one current bakeoff arm.

The TD preparation is worth doing now because implementation and verification
can be made ready without spending the CPT-scale cost event.

> Compute reality check (paper, page 4): *"These restrictions ensure that our method is quick to run, initializing 2,500 new tokens on a single GPU in under 10 minutes."* (1× H100 80GB, AdamW, 25 snippets × 50 ctx tokens.) Our larger modern setting is 17,408 tokens × 100 snippets × 50 ctx, roughly **28×** the paper's fast setting before hardware and batching differences. Conservative expectation: **4–6 hours on one GH200**, or **1.5–2 hours** only if a 4-GPU implementation scales well. Plan walltime as 6h single-GPU or 4h one-node to avoid brittle queue churn.

---

## 12. Apertus-specific considerations

These are questions about how Apertus's architectural quirks interact with the official TD implementation. Each is tagged with priority and a `Resolution` line that gets filled in once verified.

### A1. Position-encoding asymmetry between teacher and student sequences (RoPE phase shift)

**The mechanism.** When the teacher reads the unmerged sequence `[ctx_L, s1, ..., sk, ctx_R]` and the student reads the merged sequence `[ctx_L, T, ctx_R, PAD × (k-1)]`, the RoPE phase at every position *after* the merged-phrase position differs by `(k-1)` between teacher and student. Apertus uses RoPE with llama3-style scaling factor 8.0, so the phase difference manifests in the q/k rotations the model applies before the dot product.

**Resolution (from official code, train_loop.py:67-89):** the implementation **does** pad the merged sequence with PAD tokens at the end to match the unmerged length. Both teacher and student therefore see the same total sequence length, but the meaningful tokens occupy different positions:
- Teacher: real tokens at positions `[0, L+k+R-1]`, no padding.
- Student: real tokens at positions `[0, L+R-1]`, PAD at positions `[L+R, L+k+R-1]`.

The MSE loss is taken over positions selected by `unmerged_to_merged_mask == 1` (teacher) and `merged_seq != pad_id` (student). These two selections produce the same number of positions. The compare position for the assigned-phrase's last subtoken (teacher position `L+k-1`) is matched to the student position `L` (the single new token).

**Implication:** the RoPE phases at the compare positions for `ctx_R` tokens *differ between teacher and student* by `(k-1)`. The implementation accepts this and trusts that the model's downstream layers still produce comparable hidden states. The paper does the same. For modern Greek where `k` is typically 2-5, this is mild; for polytonic where `k` can be larger (more aggressive subword fragmentation), it's more pronounced. **Status: accepted as the paper's design**; we don't need to change it, but it's worth flagging as a known approximation.

**Action:** none for modern bakeoff. For the polytonic Phase 2 sub-plan (§9.5 below), consider whether `k > 8` tokens should be excluded from the TD set.

### A2. Cached teacher hidden states — optional optimization, not first implementation

The official `train_loop.py:282-285` runs the teacher forward fresh on every epoch under `torch.no_grad()`. The teacher is frozen, so its hidden states for each `(snippet, target_position)` pair are **fixed across all epochs**. The plan should add caching:

- After dataloader construction, do one teacher forward pass over the whole dataset, save `og_hiddens[mask==1]` to disk keyed by snippet hash + target layer.
- During the training loop, skip the teacher forward entirely; read cached hiddens.

**Cache footprint:** ~17,408 tokens × 100 snippets × `~50` positions × 4096 dim × bf16 ≈ **700 GB per target layer**. Too big for one node's RAM and large enough that random per-batch reads could become an I/O problem if the dataloader shuffles globally.

If we want a tighter cache: store only the positions used by the loss (mask=1 positions), which is at most `~50` per snippet. Same order of magnitude.

**Cache key must include:** teacher checkpoint hash, dtype (bf16/fp32), `attn_implementation` (sdpa/flash-attn-2), `target_layer`, RoPE config, NFC version of the snippet text. Invalidation must trigger a fresh cache.

**Speedup:** potentially useful when repeating the same data/layer across several
LR or batch-size probes. For the first one-epoch run, caching may not beat the
simplicity of recomputing the teacher forward, especially if cache reads become
random.

**Action:** do **not** make caching a blocker for TD1-TD5. Implement uncached
first, then add a cache only if the layer pilot repeats the same target layer or
the teacher forward is empirically dominating wall time.

### A3. NFC normalization on snippets — must match the corpus we trained on

`apertus_fidelity_checklist.md` V9: the CPT corpus is NFC-normalized in place by `normalize_nfc.sh`. The TD snippet miner must read from the **post-normalize parquet** (`cpt/selected_after_apertus_and_internal_dedup.parquet` after job 2335826), not the pre-normalize version. Any TD that conflates the two will see surface-string mismatches (e.g., NFD-decomposed Greek diacritics will not match new-token strings whose tokenizer was trained on NFC).

**Action:** assert in the snippet miner that the corpus was last modified after the normalize_nfc job's recorded timestamp; refuse to run if not.

### A4. Mixed-precision recipe for TD

The official code uses `torch.autocast("cuda", dtype=torch.bfloat16)` with `mixed_precision=True`. This matches Apertus's bf16-main / fp32-master-grads pretraining recipe. **Recommendation:** keep the default. AdamW moment buffers are fp32 by default in PyTorch; with `weight_decay=0` (paper default) this is identical to Apertus's optimizer state for the trainable rows.

### A5. Apertus-overlap-drop applied to the CPT corpus — soft consideration

The CPT corpus has `apertus_overlap_drop_docs.parquet` rows removed (the docs that overlap with Apertus's pretraining set). The TEACHER (Apertus base) has *seen* those dropped docs and presumably has sharper hidden states on them. Snippets drawn from the post-drop CPT corpus give a slightly noisier teacher.

For the modern-Greek TD, this is a minor effect — the corpus is still ~5B Greek tokens, plenty for snippets. But for the polytonic Phase 2 (where the corpus is thinner anyway), drawing some snippets from the pre-drop set could help. The pre-drop corpus exists at `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/nanochat/data/*.parquet` (before the apertus-overlap drop overlay was applied).

**Action:** for modern TD, ignore. For polytonic TD, consider adding ~10-20% snippets from the pre-drop corpus to the pool.

### A6. Polytonic snippet sourcing — a separate Phase 2 data problem

The plan's §5 mentions polytonic TD as a follow-on. The modern-only CPT corpus has almost no polytonic content (the corpus was Greek-modern-targeted). For the +5,120 polytonic tokens, snippets must come from elsewhere:

- The polytonic-marked subsets of the original GlossAPI corpus (pre-Apertus-overlap-drop).
- External polytonic sources: ancient/koine Wikipedia, Project Gutenberg Greek, the patristic / classical Greek corpus subsets.
- The `legacy/corpus_clean_normalization/` polytonic-flagged documents (per `glossapi_corpus_cli/pipeline.py`'s `derive_historical_flag`).

**Action:** Phase 2 TD requires a polytonic snippet pull as a prerequisite. Add as a separate work item; do not block modern TD on it.

---

## 13. Paper and repo specifics — verifications from the research pass

Each row records what the plan was assuming, what the paper / repo actually says, and the resulting action.

### P1. The implementation uses `model.forward()` outputs, not custom attention hooks

**Plan assumption:** "Use `model.forward()` rather than reimplemented attention math" (per `cpt_plan.md v0.7 §13`'s general advice).

**Verified:** `train_loop.py:278` calls `model(merged_seq, output_hidden_states=True)`, `train_loop.py:285` calls `model(unmerged_seq, output_hidden_states=True)` under `torch.no_grad()`. The hidden states are read from `outputs["hidden_states"][target_layer]`. Standard HuggingFace API. **Compatible with Apertus's xIELU + QK-Norm** because both happen inside `model.forward()`.

**Action:** none. The QK-Norm worry from v0.6 §13 is genuinely moot for hidden-state TD.

### P2. Learning rate, optimizer, scheduler, step count

**Plan assumption:** "LR ~1e-3 to 1e-4 AdamW, ~500-2000 steps".

**Verified from paper (p.4) + code (`train_loop.py:250-259`):**
- Optimizer: AdamW, `weight_decay=0.0`.
- Scheduler: **constant LR** (code comment: *"paper used linear warmup + decay but constant works fine and might even converge faster"*). The paper itself says *"linear warmup, constant rate"*.
- Default LR in code: `1e-4`. README: *"learning rates around 1e-4 work well for many models"*.
- Paper: *"For fair comparison, we sweep for the best learning rate for all methods that require a learning rate."* — they sweep per method.
- Step count: not directly named; controlled by `epochs * len(dataloader)`. Default `epochs=1`. With 17,408 tokens × 100 snippets / batch_size 16 ≈ 109,000 batches per epoch.

**Action:** start with `lr=1e-4, epochs=1, weight_decay=0, scheduler=constant` (paper + repo defaults). LR sweep is a Phase 2 optimization if results are off; do not sweep on the first run.

### P3. Snippet count + length recommendations

**Plan assumption:** "100 snippets per token, snippet length not specified."

**Verified from paper (p.4) + README:**
- **Fast setting (used for main results):** `snippets_per_token=25, snippet_len=50`. *"2,500 new tokens on a single GPU in under 10 minutes"*.
- **Larger compute budget (paper §5.2):** `snippets_per_token=100, snippet_len=50`.
- README: *"We observe diminishing returns for scaling `snippet_len` beyond 50 and `snippets_per_token` beyond 100."*

**Recommendation:** start at the larger budget (`100, 50`) per the plan's §5. If wall is too long, the fast setting (`25, 50`) still gives strong results in the paper's Table 1.

**Action:** set `snippets_per_token=100, snippet_len=50` for the modern run. Add `snippets_per_token=25` as the smoke-pilot config (matches paper's main-results setting).

### P4. Snippet sampling and batch composition

**Verified from code (`tokdist.py:191-278`):**
- **Pattern matching:** uses Aho-Corasick (`ahocorasick.py`) to find all occurrences of new-token surface strings in a tokenized corpus, in a single linear pass.
- **Snippet size:** `offset_before = snippet_len*2`, `offset_after = snippet_len*2` (so candidates have up to ~4× snippet_len + pattern_len tokens), then truncated to exactly `snippet_len` with the pattern roughly centered.
- **Filtering:** if fewer than `snippets_per_token` usable snippets are found, the token is **skipped** (printed to stdout, returned in the `skipped` list). The plan's §5 coverage policy already mirrors this.
- **Shuffling:** `random.shuffle(truncated)` before truncation to `snippets_per_token`, to avoid sort-order bias.
- **BOS prefix:** `bos_prefix = [bos_token_id]` is prepended to every snippet.
- **Batches:** dataset is flat (one entry per (snippet, assigned_phrase) pair), dataloader uses `shuffle=True` so batches mix tokens.

**Action:** mirror this snippet-mining logic in our adapter; replace `HFDataSource.load_dataset(...)` with PyArrow streaming over our local CPT corpus parquets (the official code only knows how to read HF datasets).

### P5. Loss normalization

**Verified from code (`train_loop.py:288-300`):**
- `torch.nn.functional.mse_loss(...)` with default `reduction='mean'` over `(num_selected_positions × hidden_dim)`.
- Teacher selection: `og_hiddens[unmerged_to_merged_mask == 1]` — selects positions of unmerged tokens NOT in the merged phrase, plus the LAST subtoken position of the merged phrase. Sum equals length of merged seq (before padding).
- Student selection: `token_distillation_hiddens[merged_seq != pad_id]` — non-pad positions. Same length.

**Implication:** the loss is averaged over **every position that the model produces in the merged sequence**, not only the new-token position. This means the gradient signal includes "your downstream context also has to match the teacher's downstream context after the substitution." Subtle but important: it pushes the new embedding to be compatible with *what the teacher does next*, not just to match the teacher *at the new token position*.

This is consistent with the paper's intuition (§3.1): *"the multiple subtokens t1, ..., tn have on other tokens attending to them in succeeding positions after seeing t*"*.

**Action:** preserve this behaviour in our adapter; do not "fix" it to be new-token-position-only.

### P6. Output embeddings for untied models

**Plan assumption:** train new U rows via NTP on the same data.

**Verified from paper (p.4) + code (`train_loop.py:339-348`):**
- Paper: *"Since our method backpropagates gradients back from the hidden states, we do not learn output embeddings with our distillation-based objective."*
- Paper: *"In practice, for learning output embeddings, we can simply add a next-token prediction objective just for the output embeddings at a minimal computational overhead."*
- Code: separate `torch.autograd.backward([ce_loss], inputs=[lm_head.weight], retain_graph=True)` for output, and `loss.backward(inputs=[input_embeddings.weight])` for input.
- The original-row gradients are zeroed for both input and output (`train_loop.py:354-359`) before the optimizer step, so base rows never drift.

**Important nuance from paper (p.7):** *"a subtoken attention distillation objective is superior to next-token prediction for learning new token embeddings, the combination via a sum of the two objectives (Token Distillation + NTP) generally yields worse results than Token Distillation alone."* — so combining input MSE + input NTP on the SAME parameter is worse. The split (MSE on E, CE on U) is the right pattern.

**Action:** use `learn_output_with_ce=True` for the untied Apertus path. Train new U rows via the CE-on-merged-logits-of-next-token path. The CE loss is only computed on the merged sequence, not the unmerged.

### P7. The "earlier target layer" claim — verified, but paper still defaults to LAST

**Plan assumption (§6):** *"the authors note that roughly one-third-depth layers often work better."*

**Verified from paper (p.4, §3.2 last paragraph):** *"In practice, we simply use the last layer's hidden state but analyze this choice in Section 5.3."*

**Verified from code (`tokdist.py:67`):** `target_layer: int = -1` default. Comment: *"or e.g. an early layer such as 6 or 7"* — a hint, not a claim of superiority.

**Verified from paper §5.3:** the authors report a slight downward trend as the
target approaches the last layer, and say earlier layers can improve results and
speed. They still use the last layer for main experiments because it avoids a
model-specific sweep and is principled: later subtoken interactions cannot be
excluded.

**Action:** run the small layer pilot in §6 before full modern TD: final layer
vs the one-third-depth layer, plus the probe-suggested layer if different.
Candidate selection uses the cheap gates in §6, not the full downstream suite.

---

## 14. Implementation specifics for our setup

### 14.1 We must NOT use the high-level `TokenDistillation.run(...)` entry point

The official `run()` method calls `target_tokenizer.add_tokens(new_tokens)` (`tokdist.py:97, 523`) which **appends** new tokens to the source tokenizer. This is incompatible with our setup because:

- Our extended tokenizer (`apertus_greek_modern_only_148480/`) was built by **merge-rule extension** (`model.vocab` + `model.merges`), not by `add_tokens(...)`.
- The new-token IDs in our tokenizer occupy the range `[131072, 148480)` in a specific order determined by our BPE merge order.
- If `add_tokens(...)` is called on the original base tokenizer with our new-token strings, the resulting IDs will match our planned range *only if* the order matches — which is fragile and unverified.

**Action:** drop into the lower-level `train_embeddings()` function directly, with:
- Student `model = AutoModelForCausalLM.from_pretrained(retok_init_checkpoint)` — already has the 148,480 vocab.
- Student tokenizer loaded directly from our shipped extended bundle.
- Skip `_build_target_tokenizer`; skip `extend_pretrained_with_tokens_and_embeddings`.
- Construct `phrase_to_new_id` and `assigned_new_phrases` from our **known** new-token IDs.

### 14.2 Snippet miner: PyArrow batch-streaming over local parquet, not HF datasets

The official `build_snippets_for_tokens_from_hf` uses `datasets.load_dataset(streaming=True)` over an HF dataset. Our corpus is local parquet on iopsstor. Replacement plan:

- Read `cpt/selected_after_apertus_and_internal_dedup.parquet` via `pyarrow.parquet.ParquetFile(...).iter_batches(batch_size=N)`.
- Tokenize each batch with the **base 131,072 tokenizer** (same as paper's "source tokenizer").
- Run Aho-Corasick (use the official `ahocorasick.py`, no need to reimplement) over the tokenized stream.
- Reuse the truncation + filtering logic from `build_snippets_for_tokens_from_hf` (offset_before, offset_after, snippet_len, snippets_per_token).
- Cache the tokenized parquet on iopsstor scratch keyed by `tokenizer.name_or_path + parquet hash`.

The tokenized-corpus cache (from the paper's `_tokenize_dataset_if_needed`) is **a real time-saver**: tokenizing the 47M-row Greek pool once with the base tokenizer takes ~30-60 min on one node. Don't re-tokenize per TD run.

### 14.3 Frozen-parameter table (replaces vague "freeze everything else" in §3)

| Tensor | Trained? | Notes |
|---|---|---|
| `model.embed_tokens.weight[131072:148480, :]` | **YES** (MSE-on-hiddens loss) | new input rows |
| `lm_head.weight[131072:148480, :]` | **YES** (CE loss on merged seq) | new output rows |
| `model.embed_tokens.weight[:131072, :]` | NO (grad zeroed after backward) | base input rows, verified unchanged at end |
| `lm_head.weight[:131072, :]` | NO (grad zeroed after backward) | base output rows, verified unchanged at end |
| `model.layers.*.self_attn.{q,k,v,o}_proj.weight` | NO | attention QKV + output |
| `model.layers.*.self_attn.{q_norm, k_norm}.weight` | NO | QK-Norm (R17 sensitive — DO NOT TOUCH) |
| `model.layers.*.mlp.{up_proj, down_proj}.weight` | NO | MLP (xIELU lives between these) |
| `model.layers.*.mlp.act_fn.{alpha_p, alpha_n, beta, eps}` | NO | xIELU scalars (R17 sensitive — DO NOT TOUCH) |
| `model.layers.*.input_layernorm.weight` | NO | RMSNorm input |
| `model.layers.*.post_attention_layernorm.weight` | NO | RMSNorm post-attn |
| `model.norm.weight` | NO | final RMSNorm |

The official `train_embeddings` already implements this freeze pattern correctly via `param.requires_grad=False` for everything except E/U, and zeros gradients on the original rows after backward. **Verify post-TD that R17-sensitive tensors are unchanged** (use `verify_hf_roundtrip.py`'s R17/xIELU/QK-Norm diff functions).

### 14.4 Padding direction — RIGHT-pad, not left

Code (`train_loop.py:106-140`) right-pads the merged sequence to the unmerged length. The loss explicitly excludes pad positions on both teacher and student sides. This is correct *for this objective* — the trailing PAD positions are not used by the loss, and causal attention from non-pad positions doesn't attend forward to pad.

**Action:** when porting to our adapter, keep right-padding. Don't try to "fix" it to left-padding (which would break the position-mapping logic).

### 14.5 Single-GPU vs DDP

Paper experiments are on 1× H100. Our full-modern TD at the larger compute budget is expected to be roughly 4-6 hours single-GPU on GH200. **Recommendation: single-GPU for smoke, layer pilot, and the first full modern run unless walltime proves too long.** This keeps us close to the official implementation and avoids turning TD into an infrastructure project. Add DDP only after a single-GPU profile shows a clear need.

DDP correctness check for TD: with batches shuffled across tokens (§P4), each DDP rank sees a random sample of tokens per batch, so gradients average correctly over the embedding matrix. Verify the gradient zeroing-for-base-rows happens *after* gradient reduction (otherwise the AllReduce would distribute the zeroed grads, which is fine but the assertion at end-of-training would need adjustment).

### 14.6 Compute budget — concrete numbers

Using paper numbers + Apertus scaling:
- Paper: 2,500 tokens × 25 snippets × 50 ctx = 3.125M snippet position-tokens, ~10 min on 1× H100.
- Our larger budget: 17,408 tokens × 100 snippets × 50 ctx = ~87M position-tokens.
- Naive scaling: 87M / 3.125M × 10 min ≈ **~280 minutes (4.7h) on 1× H100** at the paper's fast recipe.
- GH200 should be faster than H100 for this workload, but the Apertus HF path uses Python xIELU fallback and our adapter adds verification/telemetry. Expect **4-6h single-GH200** as the planning number.
- 4-way DDP, if implemented and scaling well: **1.5-2h** planning number.

The plan's §11 "few node-hours" is consistent with this. **Recommend setting walltime to 6h for single-GPU full TD** or **4h for one-node DDP full TD**.

---

## 15. Tools and libraries we will use

| Concern | Tool | Notes |
|---|---|---|
| Model loading | `transformers` (pinned to uenv's 4.57.0) | has `ApertusForCausalLM` |
| TD library code | clone of `konstantinjdobler/token-distillation` at a pinned commit | use `train_embeddings()`, `ahocorasick.py`, and `utils.py`; **skip** `tokdist.TokenDistillation` |
| DDP / mixed precision | `accelerate` (already in our lm_eval target install) | wrap the train loop |
| Corpus streaming | `pyarrow.parquet.iter_batches` | already battle-tested by `mix_builder.py` |
| Tokenizer cache | `datasets.Dataset.save_to_disk` (already pulled into the uenv) | reuse the paper's caching pattern |
| Checkpoint surgery / verification | `safetensors` + our existing `verify_hf_roundtrip.py` | mandatory post-TD check (gate §8) |
| HF → Megatron conversion | `tools/checkpoint/convert.py` with our `loader_apertus_hf.py` | proven on R17 patch jobs 2341182/2341239/2341241 |
| Telemetry / loss curves | TensorBoard (already wired in bakeoff_train.sbatch) | optional but cheap |
| **Layer-suggestion probe (§6.1)** | [`AlignmentResearch/tuned-lens`](https://github.com/AlignmentResearch/tuned-lens) — Belrose et al. 2023, arXiv:2303.08112 | logit-lens is free (project residual through Apertus's existing `lm_head`); tuned-lens trains a small per-layer affine probe (~minutes per layer on one GH200). For finding a candidate `L*`, logit-lens is sufficient on first pass. If it suggests a layer materially different from the one-third-depth package suggestion, include that layer in the pilot. |
| Multi-token Greek word inventory (for the §6.1 probe) | one-pass scan of the corpus + base tokenizer | reuse the Aho-Corasick driver from TD2; just emit the set of `(word, k)` where `k > 1` is the base-tokenizer fragment count |

**Pinning the TD repo:** the official repo is small (~10 Python files) and stable. Vendor at a pinned commit hash into `subprojects/.../init_bakeoff/td/external/token_distillation/` so a moving GitHub HEAD can't break our run. Pin should land before TD0.

---

## 16. Open questions that still need answers

Carried forward; will be resolved before TD1 lands.

| # | Question | Source | How to answer |
|---|---|---|---|
| Q1 | Does the paper / package actually suggest testing an earlier layer? | paper §5.3 + package README | RESOLVED: yes. Paper keeps last layer as the default to avoid model-specific sweeps; package README says earlier target layers around the one-third mark can yield better results in many cases. Plan tests both. |
| Q2 | What's the empirical coverage for our 17,408 new tokens against the 5B-Greek-token CPT corpus? | local Aho-Corasick prepass | run a one-pass coverage check before launching TD; gate as per §5 of the plan |
| Q3 | How many tokens fragment into k > 8 base subtokens (where RoPE phase shift may be too large)? | base tokenizer applied to all 17,408 new-token surface strings | trivial offline check; report distribution of k |
| Q4 | Does the paper sweep batch_size, or is `batch_size=1` (repo default) actually their setting? | paper appendix C / repo example.py | check arxiv:2505.20133 appendix C; example.py uses default which is `batch_size=1` — but the paper says *"We set the batch size to optimize throughput"* (p.4), so they presumably batch in practice |
| Q5 | Does the bakeoff differentiate ReTok vs Centroid enough on §5.3 D5/D2/D7 diagnostics to gate TD on (per `SUGGESTIONS.md §7.2`)? | bakeoff arm eval | answered by reading per-arm §5.3 diagnostics after each arm's mid-training checkpoint |
| Q6 | For polytonic TD (Phase 2), what snippet sources have enough polytonic content? | survey of available polytonic corpora | external work; descope from modern-only TD plan |
| Q7 | Does the TD-trained checkpoint round-trip through `tools/checkpoint/convert.py` with **zero R17 drift** (since we never touched xIELU/QK-Norm during TD)? | run R17 verifier post-TD | mechanical; reuse `verify_hf_roundtrip.py` |
| Q8 | When we combine TD (input MSE) with NTP (output CE) on Apertus's untied head, does the NTP loss act as a regularizer (paper p.7 says yes for tied) or as a confound (paper p.7 says yes for untied combination on same param)? | the paper warns about TD+NTP on the SAME param being worse; our case is TD on E + NTP on U, which is the paper's recommended pattern | implementation Q only — code already does this split correctly per `learn_output_with_ce=True` |
| Q9 | Should the TD eval gate (§8 "Pre-CPT BPC should improve over ReTok") use a numerical threshold (e.g. >2× bootstrap stderr of ReTok)? | calibrate from V4-HF bootstrap CIs | depends on `compute_bootstrap_cis.py` results on V4 |
| Q10 | Verify the "Kaplan et al. 2025 *Tokens to Words*" citation referenced in §6.1 before relying on the "L* ≈ first-third of layers" empirical claim | arXiv / Google Scholar search | one-off literature check; if the citation doesn't pan out, the §6.1 probe still works as a self-contained logit-lens diagnostic — we just lose the prior expectation about where L* should land |
| Q11 | Confirm the HF hidden-state indexing convention for `target_layer = -1` on Apertus. HF returns `outputs.hidden_states[0] = embedding output` and `outputs.hidden_states[N] = output of transformer block N`. For Apertus (32 layers), `hidden_states[-1]` should be the final transformer block's output BEFORE final RMSNorm and `lm_head`. Confirm this matches what the paper means by "last layer." | quick `model.forward(..., output_hidden_states=True)` shape check on Apertus + cross-check vs `tokdist.py` reading `hidden_states[target_layer]` | mechanical; trivial offline check. If the paper means "post-final-norm representation" we'd need `target_layer = -1` to be the post-norm layer or apply the final norm manually before comparing. |

---

## 17. Notes from the research pass (paper + repo, 2026-05-22)

Direct quotes from arxiv:2505.20133 v3 + the official repo, kept here so we don't re-derive them.

**Method (paper §3.2, p.4):**

> *"We learn e* by minimizing the mean-squared error between hidden states for a given target layer l: min_{e*} E_{s∼S} [ 1/|M(s_τ, s_τ*)| · Σ_{(i,j)∈M(s_τ,s_τ*)} || H_{e*}^{(l)}(s_τ*)_i − H^{(l)}(s_τ)_j ||²₂ ]. In practice, we simply use the last layer's hidden state but analyze this choice in Section 5.3."*

**Hyperparameters (paper §3.3, p.4):**

> *"We employ a simple setup and use the AdamW (Kingma & Ba 2017; Loshchilov & Hutter 2019) optimizer for all trainable parameters. We set the batch size to optimize throughput and run all experiments on Nvidia H100 80GB GPUs. We do not use weight decay and maintain a constant learning rate with a linear warmup. For fair comparison, we sweep for the best learning rate for all methods that require a learning rate. Since our method is aimed to serve as an initialization rather than as full-scale further training, we restrict the number of example sequences to a maximum of 25 per target token and truncate to a context length of 50 tokens. These restrictions ensure that our method is quick to run, initializing 2,500 new tokens on a single GPU in under 10 minutes."*

**Output embeddings (paper §3.3, p.4):**

> *"Since our method backpropagates gradients back from the hidden states, we do not learn output embeddings with our distillation-based objective. In fact, this is not possible, as our new tokens are not part of the original model that serves as the "teacher". In practice, for learning output embeddings, we can simply add a next-token prediction objective just for the output embeddings at a minimal computational overhead or freely combine our method with any other method for initializing output embeddings."*

**Snippet retrieval (paper §3.3, p.4):**

> *"Our main approach is to simply retrieve snippets that contain our target tokens from a domain-specific or general corpus. This can be implemented efficiently using the algorithm proposed by Aho & Corasick (1975). Then we can truncate the snippets to a small window around our target token to optimize computational efficiency."*

**Freezing original embeddings (paper §5.1, p.7):**

> *"Note that for Token Distillation, we only optimize new token embeddings. Therefore, we also compare against the NTP baseline, where we similarly optimize only the new token embeddings."*

**Combining TD + NTP on the same parameter is worse than TD alone (paper §5.1, p.7):**

> *"In support of our argument that a subtoken attention distillation objective is superior to next-token prediction for learning new token embeddings, the combination via a sum of the two objectives (Token Distillation + NTP) generally yields worse results than Token Distillation alone."*

> *"However, we can add a dynamic downweighting factor (see Section 4.2) to the next-token prediction objective (Token Distillation + αNTP), which mostly alleviates the negative interference while keeping the regularizing effect."*

(Implication for us: the α-NTP "regularizer" pattern is only beneficial when applied to the **input** embedding via a tied head — i.e., adds an output-loss component that backprops through the tied input/output matrix. For Apertus's *untied* setup, the equivalent is the separate CE-on-U pass we already use; we should NOT additionally add α-NTP on the input E.)

**Tied-vs-untied (paper §5.1, p.7):**

> *"Note that in Table 2, only for Llama-3.2-3B, Token Distillation obtains results that are on par with random initialization. Llama-3.2-3B(-i) are the only models in our lineup with tied embedding weights. Our objective does not explicitly enforce a bound on the norm of the new embedding, which in this case led to a failure mode of always generating a specific new embedding with very large norm."*

(Apertus is **untied**, so this failure mode doesn't apply. Confirms the bracketing was over-cautious on the tied/untied dimension.)

**Layer choice rationale (paper §5.3) — second pass:**

> *"For our main experiments, we choose to keep the last layer because this choice does not necessitate a model-specific sweep over target layers. Also, the last layer is a principled choice, as it guarantees that no subtoken interactions that are only modeled in later layers are excluded from the objective."*

> *"[Earlier target layers] can be much faster if we select early target layers. We leave further exploration of this as an exciting direction for future work."*

(§5.3 also reports a slight downward trend in quality as the target approaches the last layer — i.e., earlier layers can marginally improve. The paper still defaults to last because the marginal gain doesn't justify a broad per-model sweep. For our one-shot Apertus adaptation, §6 now runs the smallest useful check: last layer vs the package-suggested one-third-depth layer, plus the probe-suggested `L*` only if it is materially different.)
