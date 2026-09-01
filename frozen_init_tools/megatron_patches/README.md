# Megatron-LM-Swiss-AI patches

In-repo patches we maintain against the pinned `swiss-ai/Megatron-LM` clone. The core pieces are the missing HF→Megatron checkpoint loader for Apertus, the post-conversion Apertus-extras patcher, and the roundtrip verifier.

## Why

`swiss-ai/Megatron-LM/tools/checkpoint/` ships:
- `saver_swissai_hf.py` — **Megatron → HF** (works; Apertus-aware: `ApertusConfig` / `ApertusForCausalLM` imports, xIELU + QK-Norm + GQA interleaving)
- `loader_llama_mistral.py` — **HF → Megatron** for llama2 / llama3 / mistral / yi-34B / qwen2.5 — **no Apertus support**
- `loader_mixtral_hf.py` — Mixtral only
- `loader_core.py`, `loader_legacy.py` — for resuming Megatron-format checkpoints, not HF

Our bakeoff needs HF → Megatron for Apertus (the only public release format is HF, but we train in Megatron). [`AUDIT_FINDINGS.md`](../../../AUDIT_FINDINGS.md) §G flagged this as a pre-submit blocker.

## What

`loader_apertus_hf.py` — implements the inverse of `saver_swissai_hf.py`. Tensor-name mapping is taken line-for-line from `saver_swissai_hf.py` L237-345 at the pinned `swiss-ai/Megatron-LM` commit `c92402e39ef3c8e69ea378a59e79059dc14541f4`. Apertus-specific bits handled:

- **xIELU MLP** (single `up_proj`, no SwiGLU gate; `act_fn.alpha_p / alpha_n / beta / eps` per layer)
- **QK-Norm** (`q_norm.weight` + `k_norm.weight` per layer, RMSNorm per-head)
- **GQA QKV interleaving** (num_heads = 32 → num_kv_heads = 8, heads_per_group = 4): HF's separate `q_proj` / `k_proj` / `v_proj` are interleaved into Megatron's `qkv_weight` of shape `(num_heads + 2·num_kv_heads, head_dim, hidden)`
- **Untied E / U** (`tie_word_embeddings=False`)
- **Bias-free** (`--disable-bias-linear`; no bias on linear layers; Apertus removes all bias terms)
- **RoPE θ = 500,000**, llama3-style scaling factor 8 if `rope_scaling` present in the HF config

## Install

At Clariden setup time, after cloning `swiss-ai/Megatron-LM`:

```bash
bash install.sh /path/to/swiss-ai/Megatron-LM
```

The install is a `ln -sf` into `$MEGATRON_DIR/tools/checkpoint/loader_apertus_hf.py`, so it follows our local edits live. Re-run after pulling Megatron updates.

## Use

```bash
cd /path/to/swiss-ai/Megatron-LM
python3 tools/checkpoint/convert.py \
    --model-type GPT \
    --loader apertus_hf \
    --saver core \
    --load-dir   /path/to/Apertus-8B-2509-hf-or-resized-hf \
    --save-dir   /path/to/Apertus-8B-2509-megatron \
    --tokenizer-model /path/to/Apertus-8B-2509-hf-or-resized-hf \
    --bf16 \
    --loader-transformer-impl transformer_engine
```

`--model-type GPT` is required (convert.py:114). `--saver core` writes the standard Megatron distributed-checkpoint format that `bakeoff_train.sbatch --load $INIT_CKPT --ckpt-format torch_dist` reads.

### Empirically-required CLI knobs (added 2026-05-21 during R1)

- **`--loader-transformer-impl transformer_engine`** — without this, `saver_core.validate_args(margs)` asserts `args.transformer_impl == "transformer_engine"` because the Apertus checkpoint propagates `qknorm_impl=apex` (megatron/training/arguments.py:811). The assertion message — "OP arguments are only checked with the TE transformer implementation" — is misleading; the real fix is to flip `loader_transformer_impl` away from its default `"local"`. The saver still writes whatever `--saver-transformer-impl` says (default `local`).
- **`--bf16` is on the loader, not convert.py**'s top-level parser — `loader_apertus_hf.add_arguments` registers `--bf16` / `--fp16` itself (convert.py at pinned commit `c92402e3` does not). Verified empirically; "unrecognized argument: --bf16" if you omit `--bf16` from the loader's group.
- **`saver_swissai_hf` does NOT accept `--bf16`** — only the loader. Don't pass `--bf16` on the back-leg `convert.py`.
- **uenv image must contain `ApertusForCausalLM`** — `pytorch/v2.9.1:v2` ships transformers 4.57.0 and works; the older `pytorch/v2.6.0:v1` ships transformers 4.48.3 and fails with `ImportError: cannot import name 'ApertusForCausalLM'`. The loader falls back to `AutoModelForCausalLM` + `trust_remote_code=True`, but that still hits `ValueError: model_type apertus not recognized` on the 4.48 path. Use 2.9.1:v2 for both legs.

## Raw-conversion caveat and R17 patcher

`saver_core.py`'s `check_message()` at L357-443 only consumes the standard transformer protocol keys (input/post norm, qkv, dense, mlp l0/l1, optional biases, optional router). It does NOT accept Apertus-specific keys like `mlp xielu alpha p`, `q norm weight`, `k norm weight` — they'd either be rejected (default checking) or silently dropped (`--no-checking`). Either way, those parameters in the saved Megatron checkpoint land at their `XIELU.__init__` / `RMSNorm.__init__` defaults (αp = αn = 0.8, β = 0.5; q/k_norm = ones-vector), not Apertus's pretraining-trained values.

That raw conversion is tracked as **R17** in [`../../../RISKS.md`](../../../RISKS.md). It is now fixed by [`patch_apertus_extras.py`](patch_apertus_extras.py), which copies these tensors from the source HF checkpoint into every converted Megatron TP rank:

- `model.layers.*.mlp.act_fn.alpha_p` -> `decoder.layers.*.mlp.activation_func.alpha_p`
- `model.layers.*.mlp.act_fn.alpha_n` -> `decoder.layers.*.mlp.activation_func.alpha_n`
- `model.layers.*.self_attn.q_norm.weight` -> `decoder.layers.*.self_attention.q_layernorm.weight`
- `model.layers.*.self_attn.k_norm.weight` -> `decoder.layers.*.self_attention.k_layernorm.weight`

Megatron does not serialize xIELU `beta` / `eps` in these checkpoints; the patcher verifies the HF source values match Megatron defaults before accepting their absence.

Use [`r17_patch_roundtrip.sbatch`](r17_patch_roundtrip.sbatch) to patch and prove a checkpoint:

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches
sbatch --export=ALL,ARM=vanilla,OVERWRITE=1,LOGITS=1 r17_patch_roundtrip.sbatch
```

Validated patched TP=2 init checkpoints on 2026-05-21 UTC:

| Arm | Job | Patched Megatron dir | Tensor diff | Logit diff |
|---|---:|---|---:|---:|
| `vanilla` | `2341182` | `vanilla/megatron_tp2_r17patched` | `0.0` | `0.0` |
| `retok` | `2341239` | `retok/megatron_tp2_r17patched` | `0.0` | `0.0` |
| `centroid` | `2341241` | `centroid/megatron_tp2_r17patched` | `0.0` | `0.0` |

`submit_all_arms.sh` now defaults to `INIT_CKPT_SUBDIR=megatron_tp2_r17patched`; override it only for an intentional raw-conversion ablation.

## Raw R1 result (2026-05-21, Apertus-8B-2509, job 2333864)

| Metric | Result |
|---|---|
| Standard-tensor max abs diff | **`0.0`** (bit-perfect through bf16 cast) |
| R17 keys changed | **`128`** — exactly `32 layers × 4 xIELU params` (`alpha_p`, `alpha_n`, `beta`, `eps`) per layer |
| Shape mismatches | none |
| Keys present in orig but not roundtrip | none |
| Keys present in roundtrip but not orig | none |
| Pass criterion | **RAW-CONVERSION ONLY** (standard max abs diff < 1e-3, R17 drift detected) |

**Reading:** the raw loader + saver carry every non-R17 weight bit-exactly, but raw conversion is not acceptable for the live bakeoff because it drops Apertus extras. The 128 raw R17 deltas are exactly the per-layer xIELU parameters. Later R17-patched roundtrips for all three arms verify `standard_max_abs_diff=0.0`, `r17_max_abs_diff=0.0`, `xielu_max_abs_diff=0.0`, `qk_norm_max_abs_diff=0.0`, and zero logit drift on smoke prompts. The live bakeoff uses the patched `megatron_tp2_r17patched` checkpoints, not the raw R17-reset checkpoints.

R1 sbatch lives at [`r1_roundtrip.sbatch`](r1_roundtrip.sbatch). It applies two extra fixes empirically required by saver_core:

1. **Between legs, mark the Megatron checkpoint as `release`** — `saver_core` writes `iter_0000000/` + `latest_checkpointed_iteration.txt='0'`, but `loader_core.read_metadata` asserts `iteration > 0 OR file=='release'` (megatron/training/checkpointing.py:242). Rename `iter_0000000/` → `release/` and overwrite the iteration file with the literal string `release`.
2. **Pass `--loader-transformer-impl transformer_engine` on BOTH legs** — both `saver_core` (leg 1) and `loader_core` (leg 2) call `validate_args(margs)` which triggers the OP-args assertion when `qknorm_impl=apex` + `transformer_impl != transformer_engine`.

## Roundtrip validation procedure (must run before first sbatch)

The raw HF → Megatron conversion is not zero-drift for Apertus extras because `saver_core` has no protocol slots for xIELU αp/αn and QK-Norm q/k_norm. The accepted procedure is therefore:

1. Convert HF → Megatron with `loader_apertus_hf.py`.
2. Rename `iter_0000000` to `release`.
3. Run `patch_apertus_extras.py` to copy xIELU/QK tensors from the source HF checkpoint into every Megatron TP rank.
4. Convert Megatron → HF with `saver_swissai_hf.py`.
5. Run `verify_hf_roundtrip.py` and require zero standard, R17, xIELU, QK, and smoke-logit drift.

Use `r17_patch_roundtrip.sbatch`; it performs the full sequence:

```bash
sbatch --export=ALL,ARM=vanilla,OVERWRITE=1,LOGITS=1 r17_patch_roundtrip.sbatch
sbatch --export=ALL,ARM=retok,OVERWRITE=1,LOGITS=1 r17_patch_roundtrip.sbatch
sbatch --export=ALL,ARM=centroid,OVERWRITE=1,LOGITS=1 r17_patch_roundtrip.sbatch
```

**Pass criteria:**

- **Standard tensors** (embeddings, attn, mlp, norms, LM head) — `max abs diff == 0.0` on the verified bf16 roundtrip.
- **R17 tensors** (xIELU αp/αn, q_norm/k_norm) — `max abs diff == 0.0`. Any reset to defaults is a blocker for the accepted bakeoff path.
- **Logits** on the smoke prompts — `max_abs == 0.0` and top-id matches.

The live `bakeoff_1node_chain_20260522_005620` run satisfies this gate by loading `.../{vanilla,retok,centroid}/megatron_tp2_r17patched`.

## Open question

If swiss-ai already has an internal HF → Megatron loader (which they must, to validate `saver_swissai_hf.py`), this loader becomes redundant. Worth filing an issue / discussion on [`swiss-ai/Megatron-LM`](https://github.com/swiss-ai/Megatron-LM) asking before relying on our version long-term.

## Files in this dir

- [`README.md`](README.md) — this doc
- [`loader_apertus_hf.py`](loader_apertus_hf.py) — the loader
- [`install.sh`](install.sh) — symlink it into a Megatron-LM clone
- [`r1_roundtrip.sbatch`](r1_roundtrip.sbatch) — Clariden sbatch that runs the full R1 procedure end-to-end; passed 2026-05-21 (job 2333864)
- [`patch_apertus_extras.py`](patch_apertus_extras.py) — copies xIELU/QK tensors from the source HF checkpoint into converted Megatron TP ranks
- [`verify_hf_roundtrip.py`](verify_hf_roundtrip.py) — verifies standard tensors, R17 tensors, and optional smoke logits after Megatron→HF conversion
- [`r17_patch_roundtrip.sbatch`](r17_patch_roundtrip.sbatch) — Clariden sbatch that patches and verifies a checkpoint end-to-end; passed for all three bakeoff arms on 2026-05-22 UTC
