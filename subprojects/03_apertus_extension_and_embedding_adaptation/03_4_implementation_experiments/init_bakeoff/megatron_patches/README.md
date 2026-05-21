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

## R1 result (2026-05-21, Apertus-8B-2509, job 2333864)

| Metric | Result |
|---|---|
| Standard-tensor max abs diff | **`0.0`** (bit-perfect through bf16 cast) |
| R17 keys changed | **`128`** — exactly `32 layers × 4 xIELU params` (`alpha_p`, `alpha_n`, `beta`, `eps`) per layer |
| Shape mismatches | none |
| Keys present in orig but not roundtrip | none |
| Keys present in roundtrip but not orig | none |
| Pass criterion | **PASS** (standard max abs diff < 1e-3, R17 drift expected) |

**Reading:** the loader + saver carry every non-R17 weight bit-exactly. The 128 R17 deltas are exactly the per-layer xIELU parameters. Q-Norm / K-Norm are still expected to reset by mechanism, but the original R1 script only counted R17 keys whose absolute diff exceeded 1e-3, so q/k norms did not contribute to that 128 count. Future reruns of `r1_roundtrip.sbatch` print a separate q/k max-diff. This is the empirical justification for the bakeoff's two-V4-run plan: V4-HF (unmodified Apertus) and V4-post-conversion (Apertus → Megatron → HF) span the R17 risk so the bakeoff arms are compared on the same R17-reset footing.

R1 sbatch lives at [`r1_roundtrip.sbatch`](r1_roundtrip.sbatch). It applies two extra fixes empirically required by saver_core:

1. **Between legs, mark the Megatron checkpoint as `release`** — `saver_core` writes `iter_0000000/` + `latest_checkpointed_iteration.txt='0'`, but `loader_core.read_metadata` asserts `iteration > 0 OR file=='release'` (megatron/training/checkpointing.py:242). Rename `iter_0000000/` → `release/` and overwrite the iteration file with the literal string `release`.
2. **Pass `--loader-transformer-impl transformer_engine` on BOTH legs** — both `saver_core` (leg 1) and `loader_core` (leg 2) call `validate_args(margs)` which triggers the OP-args assertion when `qknorm_impl=apex` + `transformer_impl != transformer_engine`.

## Roundtrip validation procedure (must run before first sbatch)

The HF → Megatron → HF roundtrip is **NOT zero-drift** with the current loader, because xIELU αp/αn/β/ε and QK-Norm q/k_norm aren't carried through the saver_core protocol (R17 in `RISKS.md`). The roundtrip is **zero-drift on the standard tensors** (embeddings, attention QKV/dense, MLP up/down, layer norms, final norm, LM head) — **and resets xIELU + QK-Norm tensors to `__init__` defaults**.

The validation procedure splits the diff into two sets and applies different pass criteria to each:

```bash
APERTUS_HF=/path/to/Apertus-8B-2509-hf

# 1. HF → Megatron (drops xIELU + QK-Norm extras; standard tensors round-trip)
python3 tools/checkpoint/convert.py \
    --model-type GPT \
    --loader apertus_hf --saver core \
    --load-dir $APERTUS_HF --save-dir /tmp/apertus_megatron \
    --tokenizer-model $APERTUS_HF --bf16 \
    --loader-transformer-impl transformer_engine

# 1b. saver_core writes iter_0000000 + latest_checkpointed_iteration.txt='0',
#     but loader_core asserts iteration > 0 OR file=='release'. Convert to release.
mv /tmp/apertus_megatron/iter_0000000 /tmp/apertus_megatron/release
echo release > /tmp/apertus_megatron/latest_checkpointed_iteration.txt

# 2. Megatron → HF (back). xIELU + QK-Norm tensors land at __init__ defaults.
#    NOTE: saver_swissai_hf does NOT accept --bf16; only the loader does.
python3 tools/checkpoint/convert.py \
    --model-type GPT \
    --loader core --saver swissai_hf \
    --load-dir /tmp/apertus_megatron --save-dir /tmp/apertus_hf_roundtrip \
    --hf-tokenizer $APERTUS_HF \
    --loader-transformer-impl transformer_engine

# 3. Diff with two-tier pass criterion
python3 - <<'PY'
import torch, re
from transformers import AutoModelForCausalLM

orig  = AutoModelForCausalLM.from_pretrained("/path/to/Apertus-8B-2509-hf", torch_dtype=torch.bfloat16)
trip  = AutoModelForCausalLM.from_pretrained("/tmp/apertus_hf_roundtrip",  torch_dtype=torch.bfloat16)

# Keys we expect to reset to defaults per R17 — diff is informational only here.
R17_PATTERN = re.compile(r"(act_fn\.(alpha_p|alpha_n|beta|eps)|self_attn\.(q_norm|k_norm)\.weight)")

keys_orig = set(orig.state_dict().keys())
keys_trip = set(trip.state_dict().keys())
print(f"keys in orig only: {sorted(keys_orig - keys_trip)}")
print(f"keys in trip only: {sorted(keys_trip - keys_orig)}")

standard_max_abs = 0.0
r17_keys_changed = 0
for k in sorted(keys_orig & keys_trip):
    a, b = orig.state_dict()[k].float(), trip.state_dict()[k].float()
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH: {k}: {a.shape} vs {b.shape}")
        continue
    abs_diff = (a - b).abs().max().item()
    if R17_PATTERN.search(k):
        if abs_diff > 1e-3:
            r17_keys_changed += 1   # expected — reset to defaults
    else:
        if abs_diff > standard_max_abs:
            standard_max_abs = abs_diff
        if abs_diff > 1e-3:
            print(f"  DRIFT (standard tensor — NOT expected): {k}: abs={abs_diff:.6f}")

print(f"\nstandard tensors max abs diff: {standard_max_abs:.6f}")
print(f"R17 tensors changed (expected per R17): {r17_keys_changed}")
assert standard_max_abs < 1e-3, "standard tensors drifted — loader has a bug"
PY
```

**Pass criteria:**

- **Standard tensors** (embeddings, attn, mlp, norms, LM head) — `max abs diff < 1e-3` on bf16-quantised weights. Larger drift = loader bug.
- **R17 tensors** (xIELU αp/αn/β/ε, q_norm/k_norm) — **expected** to differ from original (reset to `__init__` defaults). The count is informational; we just want to see them mentioned so we don't pretend nothing was dropped.

This is a deliberate weaker pass criterion than a true zero-drift roundtrip. To get true zero-drift, implement `patch_apertus_extras.py` (currently a scaffold) to restore xIELU + QK-Norm tensors post-conversion. That's IN SCOPE for production CPT and OUT OF SCOPE for the modern-only bakeoff (which accepts R17 because all three arms inherit the same reset).

## Open question

If swiss-ai already has an internal HF → Megatron loader (which they must, to validate `saver_swissai_hf.py`), this loader becomes redundant. Worth filing an issue / discussion on [`swiss-ai/Megatron-LM`](https://github.com/swiss-ai/Megatron-LM) asking before relying on our version long-term.

## Files in this dir

- [`README.md`](README.md) — this doc
- [`loader_apertus_hf.py`](loader_apertus_hf.py) — the loader
- [`install.sh`](install.sh) — symlink it into a Megatron-LM clone
- [`r1_roundtrip.sbatch`](r1_roundtrip.sbatch) — Clariden sbatch that runs the full R1 procedure end-to-end; passed 2026-05-21 (job 2333864)
