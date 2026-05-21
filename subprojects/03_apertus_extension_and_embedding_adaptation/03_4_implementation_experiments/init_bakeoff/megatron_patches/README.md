# Megatron-LM-Swiss-AI patches

In-repo patches we maintain against the pinned `swiss-ai/Megatron-LM` clone. Currently one file: a missing HF→Megatron checkpoint loader for Apertus.

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
    --bf16
```

`--model-type GPT` is required (convert.py:114). `--saver core` writes the standard Megatron distributed-checkpoint format that `bakeoff_train.sbatch --load $INIT_CKPT --ckpt-format torch_dist` reads.

## Caveat: xIELU + QK-Norm trained values are NOT preserved through this path

`saver_core.py`'s `check_message()` at L357-443 only consumes the standard transformer protocol keys (input/post norm, qkv, dense, mlp l0/l1, optional biases, optional router). It does NOT accept Apertus-specific keys like `mlp xielu alpha p`, `q norm weight`, `k norm weight` — they'd either be rejected (default checking) or silently dropped (`--no-checking`). Either way, those parameters in the saved Megatron checkpoint land at their `XIELU.__init__` / `RMSNorm.__init__` defaults (αp = αn = 0.8, β = 0.5; q/k_norm = ones-vector), not Apertus's pretraining-trained values.

**For the modern-only bakeoff this is acceptable** — all three arms inherit the same defaults, so the cross-arm comparison stays valid. Absolute scores will be lower than running unmodified Apertus (which has its trained xIELU + QK-Norm state). Tracked as **R17** in [`../../../RISKS.md`](../../../RISKS.md).

**For production CPT** a post-conversion patcher is needed: open the saved Megatron `torch_dist` checkpoint, walk the `*.distcp` shards, and overwrite the per-layer xIELU αp / αn / β / ε and QK-Norm q_norm / k_norm tensors from the HF source. Pseudocode in [`patch_apertus_extras.py`](patch_apertus_extras.py) (scaffold; needs Megatron checkpoint-format knowledge to complete). This is OUT OF SCOPE for the bakeoff and IN SCOPE for the production-CPT pre-submit checklist.

## Roundtrip validation procedure (must run before first sbatch)

Confirm HF → Megatron (this loader) → HF (`saver_swissai_hf.py`) round-trips with zero numerical drift on a small Apertus checkpoint, before trusting it on the 8B production base.

```bash
# 1. Get a small Apertus-shaped test model (or use full 8B)
APERTUS_HF=/path/to/Apertus-8B-2509-hf

# 2. HF → Megatron
python3 tools/checkpoint/convert.py \
    --loader apertus_hf --saver core \
    --load-dir $APERTUS_HF --save-dir /tmp/apertus_megatron \
    --tokenizer-model $APERTUS_HF --bf16

# 3. Megatron → HF (back)
python3 tools/checkpoint/convert.py \
    --loader core --saver swissai_hf \
    --load-dir /tmp/apertus_megatron --save-dir /tmp/apertus_hf_roundtrip \
    --hf-tokenizer $APERTUS_HF --bf16

# 4. Diff the round-trip output against the original
python3 - <<'PY'
import torch
from transformers import AutoModelForCausalLM
orig  = AutoModelForCausalLM.from_pretrained("/path/to/Apertus-8B-2509-hf", torch_dtype=torch.bfloat16)
trip  = AutoModelForCausalLM.from_pretrained("/tmp/apertus_hf_roundtrip",  torch_dtype=torch.bfloat16)
keys_orig = set(orig.state_dict().keys())
keys_trip = set(trip.state_dict().keys())
print(f"keys in orig only: {len(keys_orig - keys_trip)}")
print(f"keys in trip only: {len(keys_trip - keys_orig)}")
max_abs = max_rel = 0.0
for k in sorted(keys_orig & keys_trip):
    a, b = orig.state_dict()[k].float(), trip.state_dict()[k].float()
    abs_diff = (a - b).abs().max().item()
    rel_diff = (abs_diff / (a.abs().max().item() + 1e-12))
    max_abs = max(max_abs, abs_diff)
    max_rel = max(max_rel, rel_diff)
    if abs_diff > 1e-3:
        print(f"  DRIFT: {k}: abs={abs_diff:.6f} rel={rel_diff:.6f}")
print(f"\nmax abs diff: {max_abs:.6f}   max rel diff: {max_rel:.6f}")
assert max_abs < 1e-3, "round-trip exceeded bf16 tolerance — investigate before training"
PY
```

Expected: bf16 round-trip should be byte-equal or within ~1e-3 on bf16-quantised weights. Any larger drift means our loader has a bug.

## Open question

If swiss-ai already has an internal HF → Megatron loader (which they must, to validate `saver_swissai_hf.py`), this loader becomes redundant. Worth filing an issue / discussion on [`swiss-ai/Megatron-LM`](https://github.com/swiss-ai/Megatron-LM) asking before relying on our version long-term.

## Files in this dir

- [`README.md`](README.md) — this doc
- [`loader_apertus_hf.py`](loader_apertus_hf.py) — the loader
- [`install.sh`](install.sh) — symlink it into a Megatron-LM clone
