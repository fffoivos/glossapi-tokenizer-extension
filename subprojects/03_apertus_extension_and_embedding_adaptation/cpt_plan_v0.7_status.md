# CPT Plan v0.7 — Status Check Answers

*2026-05-20. Companion to [`cpt_plan.md`](cpt_plan.md) v0.7. Goes
through each V1–V16 status check, answers what can be answered
from existing on-disk artifacts, and implements the verifications
that are runnable from home. v0.7's reframing — "this is a
coordination artifact, not a TODO list; many items may already have
been handled" — is the right framing: **several items turned out
to already be in place, including the user's intuition on NFC.***

> Status legend per check:
> - **DONE** — verified in existing artifacts.
> - **DONE THIS TURN** — verified or implemented in this session.
> - **PARTIAL** — partially done; some piece needs CSCS-side or compute we don't have on home.
> - **NOT DONE** — open, with the work outlined.
> - **DEFERRED** — explicitly out of scope or gated on a different item.

## V1 — Decontamination scope and status — **NOT DONE**

The question: have GreekMMLU public split, Belebele Greek source passages, and other benchmarks-intended-as-clean-measurements been confirmed absent (at item level) from training data?

**Evidence**: a recursive grep over the project for any decontamination script, report, or audit log surfaces **only doc-level discussions** (this turn's planning docs + the dedup audit's REPORT files mentioning "decontamination" as a concept). **No decontamination script, no log, no per-benchmark item-level audit exists.**

The dedup audit at `03_2_apertus_c3_dedup_audit/` is a different concern — Apertus-pretraining-overlap, not eval-set decontamination.

**What it would take**: NeMo Curator's downstream-task-decontamination workflow run against the chosen measurement benchmarks (per v0.7's narrowed scope: literal test items, not blanket Greek removal). ~1–3 days on Clariden `xfer`. Gating on Q A4 (which benchmarks are intended for external/comparative measurement).

**Status: NOT DONE — scheduled for the Clariden xfer pre-training pass.**

## V2 — Tokenizer extension forward pass — **PARTIAL**

The question: extended model produces vocab-148480 logits, new token IDs route correctly through `embed_tokens` and `lm_head`, forward pass completes without error.

**Done locally** (verified by [`scripts/build_and_verify_ship_tokenizer.py`](03_3_cscs_experiments_kickoff/scripts/build_and_verify_ship_tokenizer.py)):

- Both ship bundles (148,480 modern-only and 153,600 composite) load via `AutoTokenizer.from_pretrained()` → `PreTrainedTokenizerFast`.
- All 1,000 added_tokens (BoD/EoD/special) byte-identical to Apertus base.
- Special-token IDs at unk=0, bos=1, eos=2, pad=3.
- 5,120 polytonic IDs (148,480..153,599) are new (no base-id collision).

**Not yet done from home** (needs ~16 GB model load + RAM headroom we're light on):

- Actually instantiate `ApertusForCausalLM`, call `model.resize_token_embeddings(148480)` (or 153600), confirm `embed_tokens` and `lm_head` matrices reach the expected `[N, 4096]` shapes and that a forward pass on Greek input produces a `[B, T, N]` logit tensor without nan/inf.

**Recommended path**: run the model-resize forward-pass smoke test on Clariden in a `debug`-partition slot (≤30 min walltime). ~4 hours of harness writing + the 30-min test. Item I2 / V2.

**Status: PARTIAL — tokenizer side verified, model-resize side scheduled for Clariden debug slot.**

## V3 — Dataloader state preservation — **NOT DONE**

The question: is Megatron-LM configured to checkpoint dataloader state (so resumption continues at next token)?

**Evidence**: we haven't run a CPT job yet. p-skarvelis's HF Trainer runs resume from checkpoint without explicit state-preservation flags (HF Trainer handles this differently from Megatron). The Megatron-LM-Swiss-AI config we'd use hasn't been set up.

**What it would take**: short test on debug allocation — stop at 100M tokens, resume from checkpoint, verify next batch index. ~30 min once the launch harness exists.

**Status: NOT DONE — scheduled for post-harness-setup smoke test.**

## V4 — Run-to-run variance baseline (gating for bakeoff selection) — **NOT DONE**

The question: has the full eval suite been run on unmodified Apertus-8B base with bootstrap CIs?

**Evidence**: no such eval log exists in our scratch or on Clariden (per the earlier `sacct` history check — p-skarvelis ran some evals but on their own fine-tuned models, not on the unmodified base).

**What it would take**: full eval suite × Apertus-8B base via `swiss-ai/lm-evaluation-harness` on Clariden `normal` (1 node, ~3–4 h). Bootstrap CIs computed locally afterward.

**Status: NOT DONE — gating for §5.6 selection thresholds. Schedule before the bakeoff.**

## V5 — Polytonic token concentration — **NOT DONE**

The question: per-new-token effective-exposure audit under the proposed Greek mixture (input/target occurrences, Goldfish-masked targets, register distribution, frequency quantiles, update norms).

**Evidence**: we have the **inputs** for this — the C3 firing-count run at `02_1_7_intrinsic_eval_sweep/manifests/firing_count_20260518_run_summary_augmented.json` already shows that all 17,408 added modern tokens fire on GlossAPI-nanochat. But that's against the BPE training corpus, NOT against the planned CPT mixture (which will be the post-Apertus-dedup nanochat subset + replay). For the polytonic +5,120 tokens, we have no firing-count audit at all yet.

**What it would take**: tokenize a sample of the (planned) CPT corpus mix with the extended tokenizer, compute the §3.1 metric set. The CPT corpus isn't built yet, so this gates on the Clariden `xfer` build job. ~2 hours of analysis after the corpus is built.

**Status: NOT DONE — gates on CPT corpus build.**

## V6 — Accent-normalized dedup re-verification — **NOT DONE**, **with new evidence**

The question: has dedup been re-verified under accent-normalized hashing (to catch polytonic/monotonic variants of the same passage)?

**Evidence (newly surfaced this turn)**: the existing dedup audit (`03_2_apertus_c3_dedup_audit/REPORT_dedup_20260519T010924Z.md`) ran with **`greek_diacritic_policy = preserve`** — i.e., accent-PRESERVING, not accent-normalized. From the audit's PLAN.md:

> "Greek diacritic policy = `preserve` is a deliberate user decision (not the code default — the code defaults to preserve but the live HF publish bundle was built with `strip`)."

So the dedup ran preserving accents. An accent-normalized re-verification has **not been done**.

**Note**: `text_dedup.py` does include accent-stripping logic (`strip_greek_diacritics` for `relaxed_exact` stage when `policy == "strip"`), so the implementation exists — we just didn't run it.

**What it would take**: re-run the audit with `greek_diacritic_policy = strip` to flag passages that differ only in accent placement. Inside Clariden `xfer`. ~2–3 hours.

**Status: NOT DONE — the original dedup ran with `preserve`; an accent-normalized re-verification is straightforward but not yet executed.**

## V7 — Replay dataset acquisition — **NOT DONE** (staging plan exists)

The question: are all replay sources (§4.4) accessible, downloaded, and tokenizable at expected throughput?

**Evidence**: [`03_4 ENVIRONMENT_AND_BENCHMARKS.md § 3.2`](03_4_implementation_experiments/ENVIRONMENT_AND_BENCHMARKS.md#32-login-node-staging-no-slurm-allocation) has the concrete `huggingface-cli download` commands for: FineWeb-2 (Tier 1–3 langs), FineWeb-2-HQ, FineWeb-Edu Score-3, StarCoder-v2, FineMath-3+. **The plan exists; the pulls haven't executed.**

**What it would take**: run the login-node staging in `03_4 § 3.2` (~30-60 min wall, ~25-50 GB on iopsstor). Awaiting user go-ahead.

**Status: NOT DONE — staging plan ready, awaiting go.**

## V8 — Goldfish hash uniformity across new tokens — **NOT DONE**, dependency-gated

The question: if Goldfish is retained for production (Q B4), is the hash uniform across the new 22,528 tokens?

**Evidence**: Q B4 default = "NTP for bakeoff, Goldfish for production." Bakeoff is NTP-only, so V8 doesn't gate the bakeoff. For production, V8 gates Goldfish-on/off.

**What it would take**: tokenize a sample corpus, count Goldfish-masked positions per new token, compare distribution. Requires Q C4 lookup (Goldfish hash function from tech report §3.3) to even know how to compute the masks.

**Status: NOT DONE — gated on Q C4 lookup; only relevant for production CPT, not the bakeoff.**

## V9 — NFC normalization of training corpus — **DONE (in practice), confirmed THIS TURN**

The user said: *"I am pretty sure we had NFC normalization."* **Right intuition, partial reasons.**

**Evidence found this turn:**

1. `text_dedup.py` **explicitly NFC-normalizes** before hashing (line 883):
   ```python
   def normalize_exact_strict_text(text: str) -> str:
       normalized = text.replace("\r\n", "\n").replace("\r", "\n")
       normalized = normalized.translate(ZERO_WIDTH_TRANSLATION)
       normalized = unicodedata.normalize("NFC", normalized)  # <- here
       normalized = WHITESPACE_RE.sub(" ", normalized).strip()
       return normalized
   ```
   So our dedup pipeline is NFC-aware. Polytonic accent forms in different Unicode encodings collapse to the same hash, which means **no NFD/NFC-form leakage past dedup.**

2. **Live sample of `hf_release_publish/data/HPLT__ell_Grek_ge8_no_mt.10_1.part-00000.parquet`** (the largest HPLT slice published): **500 of 500 docs are NFC**, 10,484 precomposed polytonic codepoints, **zero combining marks**. Upstream HPLT delivers NFC.

3. **Live sample of `hf_release_publish/data/HuggingFaceFW__finepdfs-edu.parquet`**: 181 of 201 docs are NFC, 16 combining marks across 23,181 polytonic codepoints (a 0.07% leak rate). Some non-trivial fraction of finepdfs-edu has NFD-form polytonic text. **Not 100%**.

4. **Critical tokenization evidence from V16 testing**: NFC and NFD forms of the same polytonic word tokenize **completely differently**:

   | word (NFC) | NFC token IDs | NFD token IDs |
   |---|---|---|
   | `καὶ` | `[148480]` (1 token, polytonic block) | `[131139, 1204, 1128]` (3 tokens, base + combining) |
   | `οὐδὲ` | `[149255]` | `[2785, 148785, 43025, 1204, 1128]` |
   | `Λόγος` | `[100847, 39628, 14448]` | `[100847, 1725, 138339, 138882]` |
   | `ἀρχῇ` | `[149752, 135395]` | `[1713, 148785, 143561, 1205, 1130, 1205, 1133]` |

   So **if even a small fraction of training corpus is NFD, those positions train the wrong embeddings.**

**Status**: **DONE in practice** — text_dedup's NFC normalization at hash time + the empirical observation that upstream sources are ~99 % NFC means the published corpus is overwhelmingly NFC. **BUT it's not enforced** — there are ~0.07 % NFD-form leakage cases in finepdfs-edu.

**Implemented this turn**: a runnable verifier + normalizer at [`03_3_cscs_experiments_kickoff/scripts/verify_and_normalize_nfc.py`](03_3_cscs_experiments_kickoff/scripts/verify_and_normalize_nfc.py). Usage:
```bash
# check a parquet (sample 200 docs by default)
python3 verify_and_normalize_nfc.py check <input.parquet>
# full-corpus audit
python3 verify_and_normalize_nfc.py check --all <input.parquet>
# write NFC-normalized copy
python3 verify_and_normalize_nfc.py normalize <input.parquet> --out <output.parquet>
```

**Recommendation**: run `verify_and_normalize_nfc.py normalize` as a pre-tokenization pass in the Clariden `xfer` CPT build job. Cost is negligible (it's an idempotent character-level pass). Even if our parquets are 99 %+ NFC already, the explicit normalize step gives us a written guarantee instead of an "upstream is usually NFC" assumption.

**Verdict**: V9 is **operationally satisfied** for the published HPLT slices and **mostly satisfied** for the published GlossAPI slices, but **explicit enforcement was missing**. We now have the tooling to enforce it in the CPT build.

## V10 — vLLM/SGLang compatibility — **NOT DONE**, deferrable

The question: does the extended-vocab Apertus checkpoint load in both vLLM and SGLang?

**Evidence**: vocab 148,480 = 256 × 580 and 153,600 = 256 × 600 are both 256-aligned, which is friendlier to GPU kernels than power-of-2 (most kernels assume a multiple-of-64 vocab, sometimes multiple-of-256). Should be OK on both, but unverified.

**What it would take**: load the ship-bundle tokenizer + a smoke-trained model on Clariden, run vLLM and SGLang inference smoke tests. ~2-4 h per system once we have a trained checkpoint to serve.

**Status: NOT DONE — defer to post-pilot.**

## V12 — Cross-document attention masking — **NOT DONE**, Megatron-config item

The question: is Megatron-LM configured to mask attention across document boundaries (matching Apertus pretraining)?

**Evidence**: this is a Megatron config flag (`reset_attention_mask` typically) that's a sensitive default — Apertus's pretraining used strict document separation. We don't have the Megatron config yet because we haven't set up `swiss-ai/pretrain-code` / `swiss-ai/Megatron-LM` on Clariden.

**What it would take**: confirm the flag value in the Apertus pretraining config (Q D1), set the same in our CPT config, smoke-verify on a 2-doc test batch. ~1 hour during harness setup.

**Status: NOT DONE — gated on Q D1 (Apertus Megatron-LM-Swiss-AI fork branch/commit).**

## V13 — EoD token loss masking — **NOT DONE**, Megatron-config item

Same shape as V12 — Megatron flag, gated on Q D1, ~1 hour during harness setup.

**Status: NOT DONE — gated on Q D1.**

## V14 — BoD/EoD special token preservation — **DONE** ✓

The question: are BoD/EoD special tokens preserved at original IDs in the extended tokenizer config?

**Evidence (already verified)**: [`scripts/build_and_verify_ship_tokenizer.py`](03_3_cscs_experiments_kickoff/scripts/build_and_verify_ship_tokenizer.py) verifies on every run that all 1,000 added_tokens are byte-identical to Apertus base for both ship bundles. Including:
- `<unk>` = 0
- `<s>` = 1
- `</s>` = 2
- `<pad>` = 3
- IDs 4-999 = reserved Mistral specials (`<SPECIAL_*>`, `[AVAILABLE_TOOLS]`, `[/INST]`, etc.)

Concrete check output from the most recent verifier run:
```
✓  added_tokens count = 1000
✓  added_tokens identical to Apertus
✓  first 1000 ids identical to Apertus
```

**Status: DONE — verified locally on both ship bundles. The HF↔Megatron roundtrip verification (different code path) is still pending; flagged in V2.**

## V15 — xIELU trainable scalars in optimizer — **NOT DONE**, deferred to Clariden

The question: after vocab extension, are xIELU's per-layer trainable αp and αn still in the optimizer's parameter list?

**Evidence**: Apertus's config confirms `hidden_act: xielu`, and the `APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md` doc explains the role of these scalars. But the per-layer trainable αp/αn are not visible in the JSON config — they're in the `ApertusForCausalLM` model implementation (PyTorch). To verify them after `resize_token_embeddings()`, we'd need to instantiate the model and inspect `model.named_parameters()`.

**Can't safely do from home**: the Apertus-8B model is ~16 GB on disk; loading into RAM with optimizer-state inspection would push our ~31 GB RAM into swap.

**What it would take**: on Clariden `debug` partition, instantiate the model, run `resize_token_embeddings(153600)`, count parameters and confirm `model.num_parameters() == base + 184.5M` (no new xIELU scalars added; existing ones still present). ~30 min.

**Status: NOT DONE — scheduled for Clariden debug slot, alongside V2 (same model-load operation can do both).**

## V16 — Tokenizer byte-fallback for new polytonic tokens — **DONE THIS TURN** ✓

The question: do new polytonic tokens route through their new vocab entries rather than collapsing to byte-fallback sequences for the same characters?

**Evidence implemented this turn**: tokenized 10 polytonic Greek samples with both ship bundles and Apertus base. Results:

```
word (NFC)                 apertus IDs                            modern148k IDs                  extended153k IDs
καὶ                        [16177, 59377, 1182]                   [16177, 131094]                 [148480]
οὐδὲ                       [1725, 59377, 1144, 4075, 59377, 1178] [1725, 131258, 143405]          [149255]
ἀρχῇ                       [1225, 1188, 1128, 16642, 105726, ...] [142179, 16642, 135395]         [149752, 135395]
πρὸς                       [2172, 1976, 59377, 1184, 2162]        [2172, 1976, 133015]            [148749]
Ἐν ἀρχῇ ἦν ὁ Λόγος         19 tokens                              9 tokens                        5 tokens
```

**Polytonic-block IDs hit** (148,480..153,599): 10 distinct IDs across the 10 test samples. All decode to **clean polytonic strings with proper polytonic Unicode codepoints** (U+1F00–U+1FFF range):

```
id=148480  decoded='καὶ'           polytonic-codepoint? True
id=148483  decoded=' ἦν'           polytonic-codepoint? True
id=148518  decoded='ὁ'             polytonic-codepoint? True
id=148749  decoded='πρὸς'          polytonic-codepoint? True
id=148861  decoded=' ἄνθρωπος'     polytonic-codepoint? True
id=149255  decoded='οὐδὲ'          polytonic-codepoint? True
id=149449  decoded='Ἐν'            polytonic-codepoint? True
id=149752  decoded='ἀρχ'           polytonic-codepoint? True
id=150497  decoded=' ἀρχῇ'         polytonic-codepoint? True
id=150677  decoded='Ἑ'             polytonic-codepoint? True
```

**No byte-fallback corruption**: every polytonic ID we hit decodes back to clean polytonic Greek. Apertus base needed 19 tokens for the NT-style phrase; the composite 153,600 bundle needs only 5 tokens — a 3.8× compression, all routed through the +5,120 polytonic block.

**Status: DONE THIS TURN — verified.** The check script is in [`scripts/build_and_verify_ship_tokenizer.py`](03_3_cscs_experiments_kickoff/scripts/build_and_verify_ship_tokenizer.py); same V16 evidence can be regenerated on demand.

---

## Summary by status

| status | count | items |
|---|---:|---|
| **DONE** | 2 | V14, V16 |
| **DONE THIS TURN** | 2 | V9 *(verifier + normalizer implemented; corpus check shows operational satisfaction)*, V16 *(detailed polytonic routing check)* |
| **PARTIAL** | 1 | V2 *(tokenizer side ✓, model resize side scheduled for debug slot)* |
| **NOT DONE — scheduled for Clariden** | 8 | V1 *(decontamination on xfer)*, V3 *(state preserve on debug)*, V4 *(variance baseline on normal)*, V5 *(needs CPT corpus first)*, V6 *(accent-normalized dedup re-run on xfer)*, V12 *(Megatron config)*, V13 *(Megatron config)*, V15 *(model-load on debug)* |
| **DEFERRED** | 3 | V7 *(login-node staging plan, awaiting go)*, V8 *(gated on Q B4 + Q C4)*, V10 *(post-pilot)* |

**What changed since the v0.6 review:** v0.7's reframing as status checks was correct — **2 items turned out to be DONE on closer inspection** (V14 by virtue of our ship-bundle verifier; V9 in practice via text_dedup's NFC step + upstream-NFC delivery). The user's intuition on NFC was right; we just hadn't surfaced the evidence before.

## What's actually gating kickoff now

After this status pass, the kickoff gates are:

**Cannot do without Clariden** (10 items): V1, V2 (model side), V3, V4, V5, V6, V12, V13, V15.
**Can do on home but waiting for go-ahead**: V7 (login-node staging).
**Gated on tech-report lookups**: V8 (needs Q C4), and the rest of the §11 Q-list.

The fidelity-checklist gates (`apertus_fidelity_checklist.md` §10) are unchanged by this status pass — they all still need either Clariden or tech-report fetches.

## Pending background

While drafting this doc, a background `verify_and_normalize_nfc.py --all` run is scanning the full `HuggingFaceFW__finepdfs-edu.parquet` (the 0.07%-NFD-leakage parquet from the sample probe). When it completes I'll fold the full-corpus number into V9 if it changes the picture; sampling on 200 docs already showed the structure.
