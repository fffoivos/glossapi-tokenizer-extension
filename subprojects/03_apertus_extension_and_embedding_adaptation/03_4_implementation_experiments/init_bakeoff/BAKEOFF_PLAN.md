# Init Bakeoff Setup

*2026-05-20. Scope: three closed-form init experiments per
[`cpt_plan.md`](../../cpt_plan.md) v0.7 §5. Vanilla / ReTok / Centroid,
2 B tokens per arm, identical conditions otherwise. Winner becomes
the production CPT starting point; not in scope here.*

> Downstream of [`cpt_plan.md`](../../cpt_plan.md) v0.7 + the
> [`apertus_fidelity_checklist.md`](../../apertus_fidelity_checklist.md).
> Resolved positions baked in: vocab 153,600 (modern + polytonic
> active), Megatron-LM-Swiss-AI trunk, AdEMAMix + 0.1 grad clip,
> seq=4096, NTP loss during bakeoff (Goldfish deferred to production),
> 70/30 Greek/non-Greek replay, B=2 B tokens per arm.

## 1. The three arms in concrete terms

All three arms apply their init procedure to **both `E` (input
embedding) and `U` (LM head)** matrices independently, since Apertus
is `tie_word_embeddings: False`. All three then train under
identical Megatron-LM-Swiss-AI conditions for 2 B tokens.

| | Vanilla | ReTok | Centroid |
|---|---|---|---|
| **Vocab** | 131,072 (original Apertus base) | 153,600 (composite) | 153,600 (composite) |
| **What changes vs base** | Nothing | E and U get 22,528 new rows | E and U get 22,528 new rows |
| **Init rule for new row T** | n/a | `mean(base_E[p] for p in base_tokenizer.encode(decode(T)))` + Phase A norm match | per-script centroid of base Greek tokens + Gaussian noise + Phase A norm match |
| **Code (this dir)** | [`arms/vanilla.py`](arms/vanilla.py) | [`arms/retok.py`](arms/retok.py) | [`arms/centroid.py`](arms/centroid.py) |
| **Init compute** | none | ~1 min CPU | <1 min CPU |
| **Per-arm extra params** | 0 | ~184.5 M (22,528 × 4,096 × 2) | ~184.5 M |
| **Per-arm wall (estim.)** | ~11 h on 1 node @ seq=4096 (≈ 50 k tok/s effective for 2 B tokens) | ~11 h | ~11 h |

The bakeoff is **clean** because all three arms are closed-form. No
gradient descent on init (Distillation bracketed in v0.7 §13), so
the variance source is purely training-time, not init-time.

## 2. What the bakeoff is testing

- **Vanilla vs (ReTok or Centroid)**: does vocab extension justify its 184.5 M-parameter overhead? Hard-tested via §5.6 hard gates (English / code / multilingual retention; polytonic character-NLL; throughput) + the §5.6 selection score on non-failing candidates.
- **ReTok vs Centroid**: does the per-token-specific subpiece info (ReTok) beat the script-level distributional prior (Centroid)? If ReTok wins, subpiece info matters. If Centroid wins or ties, the simpler recipe is cheaper.

Per v0.7 §5.8: at 2 B tokens, polytonic embeddings may still be undertrained. **Honest comparison is on modern Greek**; polytonic signal is secondary.

## 3. Apertus-fidelity constraints (from `apertus_fidelity_checklist.md`)

The bakeoff config **must** match Apertus pretraining on:

- **Optimizer**: AdEMAMix (β1, β2, α, weight decay — pending Q C2 tech-report lookup)
- **Gradient clipping**: 0.1 global-norm
- **Activation**: xIELU with trainable per-layer αp, αn scalars (V15: must verify still in optimizer param list after `resize_token_embeddings`)
- **Attention**: QK-Norm
- **LR schedule**: WSD with brief re-warmup (1-2 % of CPT tokens) from low → CPT peak (1.5 e-5 default per v0.7 §3.3) → plateau → linear decay aligned with anneal
- **Loss**: NTP for the bakeoff (Goldfish deferred — see §10 Q B4 + V8)
- **Document separation**: cross-document attention mask ON, EoD loss mask ON (V12, V13)
- **Sequence length**: 4,096 (Apertus pretraining default; not 2,048 as p-skarvelis used)
- **Mixed precision**: bf16
- **Batch size**: target Apertus's 4.2–8.4 M tokens/step; for the bakeoff, smaller is acceptable to fit a single 12 h slot (see §5)

## 4. Data: bakeoff corpus

Per [`../../03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md`](../../03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md) §1
+ v0.7 §2: **fresh-only post-internal-dedup pool** (Apertus-overlap dropped
via the [`03_2 dedup audit`](../../03_2_apertus_c3_dedup_audit/) overlay).
Pre-tokenization: NFC normalization via [`verify_and_normalize_nfc.py`](../../03_3_cscs_experiments_kickoff/scripts/verify_and_normalize_nfc.py)
(V9).

Mix per v0.7 §2 + §4:
- 70 % Greek (the dedup'd pool)
- 30 % non-Greek replay across v0.7 §4.2's 24 languages (8 T1 + 11 T2 + 5 T3) + 4 % code share (Q B2)

The same mix and the same shuffled stream are presented to all three
arms — that's what makes the comparison apples-to-apples.

**Total bakeoff tokens consumed**: 3 × 2 B = 6 B. At our ~14.4 M-doc / ~26–38 B-token post-dedup pool, this is well inside one epoch. Each arm sees 2 B tokens; the three arms share a deterministic dataloader-seed so token streams across arms are *identical up to the init point*.

## 5. Slurm shape

Per [`../AUTH_AND_NODE_FINDING.md`](../AUTH_AND_NODE_FINDING.md) § 6:

| field | value | rationale |
|---|---|---|
| partition | `normal` | 12 h cap, expected idle nodes |
| account | `a0140` | only project allocation |
| nodes per arm | **1** (4 × GH200) | smallest unit, fits 12 h cap at seq=4096 |
| GPUs per arm | 4 | `--nproc_per_node=4`, FSDP or DDP per Megatron config |
| time | `--time=12:00:00` | full partition cap; 2 B at ~50 k tok/s ≈ 11.1 h |
| seq length | 4,096 | Apertus pretraining default |
| microbatch | tune at calibration | start 1-2 / GPU |
| global step size | ~4 M tokens (Apertus pretrain default) via grad-accum | matches recipe |
| save_steps | every ~250 M tokens (≈ every ~65 steps at 4 M global batch) | enough resolution for §5.3 diagnostic suite + checkpoint windowing per §5.6 |
| eval cadence | every 100 M tokens (~25 steps) on the trajectory metrics; every 500 M on benchmarks | per v0.7 §6.1 |
| eval target | last 3-5 checkpoints in 80-100 % of budget; bootstrap CI | per v0.7 §5.6 |

Three arms in parallel = 12 GPUs (3 × 4) peak, all on `normal`. QoS `normal` has no concurrent-job gate per the [`ANALYSIS.md`](../../03_3_cscs_experiments_kickoff/ANALYSIS.md) probe. End-to-end wall ≈ 12 h for the bakeoff if all three arms run in parallel.

## 6. Pre-Clariden checklist (status)

Things to settle on home before submitting to Clariden:

- [ ] **Q C2 lookup**: AdEMAMix hyperparams (β1, β2, α, weight decay). From tech report §2.3 / Appendix B.4. Needed for `arms/build_init_checkpoints.py` optimizer config metadata + Megatron config.
- [ ] **Q D1 lookup**: `swiss-ai/Megatron-LM` fork branch / commit. Needed before installing Megatron on Clariden.
- [x] **Init scripts** local-tested: see [`arms/test_init_logic.py`](arms/test_init_logic.py) — verifies Centroid + ReTok produce sensible vectors using E/U matrices already on home, without needing the full Apertus model load.
- [ ] **NFC enforcement**: `verify_and_normalize_nfc.py normalize` runs over the CPT corpus parquets during the xfer build pass.
- [ ] **CPT corpus build on Clariden xfer**: per [`../../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](../../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md). Need to authorise.

On-Clariden gates (verifications):
- [ ] **V2** — `model.resize_token_embeddings(153_600)` produces correct E + U shapes; forward pass no nan/inf
- [ ] **V3** — Megatron dataloader-state preservation across checkpoint boundary
- [ ] **V4** — variance baseline on Apertus-8B base sets thresholds for §5.6
- [ ] **V12 + V13** — cross-document attention mask, EoD loss mask
- [ ] **V14** — HF↔Megatron special-token roundtrip
- [ ] **V15** — xIELU αp/αn still in optimizer param list

## 7. Sbatch templates (forthcoming)

The four sbatch shapes we'll need:

- `apertus_base_eval_baseline.sbatch` — V4 baseline eval. `-N 1 -p normal -t 12:00:00`. Runs lm-eval-harness + Inspect AI evals against unmodified Apertus-8B-2509. Produces variance + bootstrap CIs for §5.6 thresholds.
- `vanilla_bakeoff.sbatch` — Vanilla arm. `-N 1 -p normal -t 12:00:00`, base Apertus checkpoint, NTP loss, no resize, 2 B tokens of Greek+replay mix.
- `retok_bakeoff.sbatch` — ReTok arm. `-N 1 -p normal -t 12:00:00`, ReTok-initialized 153,600-vocab checkpoint, same training config.
- `centroid_bakeoff.sbatch` — Centroid arm. Same as ReTok, Centroid-initialized checkpoint.

All three bakeoff sbatches use a **shared seed for the dataloader** so token streams are identical across arms up to the init differential.

## 8. What's in this directory

- `BAKEOFF_PLAN.md` — this doc
- `arms/vanilla.py` — Vanilla arm: a no-op stub that just produces a verification config (no resize needed)
- `arms/retok.py` — ReTok init: per-new-token subpiece-mean + Phase A norm-matching, for both E and U
- `arms/centroid.py` — Centroid init: per-script (modern / polytonic / both) centroid of base Greek tokens + Gaussian noise + norm-matching
- `arms/build_init_checkpoints.py` — driver: load Apertus base, apply each arm's init, save resized HF-format model checkpoints
- `arms/test_init_logic.py` — local smoke test: validates Centroid + ReTok algorithms against the E/U matrices we have on home, *without needing the full model load*

The scripts produce HF-format checkpoints. Conversion to Megatron-LM-Swiss-AI format happens at staging-time on Clariden via `swiss-ai/hfconverter` (one-time per arm).
