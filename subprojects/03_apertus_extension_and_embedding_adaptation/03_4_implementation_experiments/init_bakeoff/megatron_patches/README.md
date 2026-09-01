# megatron_patches — the HF→Megatron bridge and the R17 fix

> **In one line:** `swiss-ai/Megatron-LM` ships a Megatron→HF saver for Apertus but no loader in the other direction, so one was written — and building it exposed **R17**, a silent reset of Apertus's xIELU and QK-Norm parameters that would have invalidated the whole bakeoff.
> **Period:** 2026-05-21 (`3cc6b45f`, loader) → 2026-05-23 (TD arm's roundtrip gate). **Status:** completed; every checkpoint the bakeoff trained from is R17-patched and verified at zero drift.
> **Feeds:** [`../arms/`](../arms/README.md) → this → [`../bakeoff_training/`](../bakeoff_training/README.md).

## Why this existed

Apertus is released only in HF format, but the bakeoff trains in Megatron. Upstream `tools/checkpoint/` has loaders for llama2/3, mistral, yi and qwen2.5 — none for Apertus. The 2026-05-21 recipe audit flagged this as a pre-submit blocker.

## History

### The loader — 2026-05-21

[`loader_apertus_hf.py`](loader_apertus_hf.py) implements the exact inverse of `saver_swissai_hf.py` (tensor mapping taken line-for-line from L237–345 at pinned commit `c92402e3`), handling xIELU MLP (single `up_proj`, no SwiGLU gate), per-layer QK-Norm, GQA QKV interleaving (32 heads → 8 KV groups, 4 heads per group), untied `E`/`U`, bias-free linears, and RoPE θ = 500,000 with llama3 scaling.

Four CLI knobs turned out to be empirically required and are documented in-file: `--loader-transformer-impl transformer_engine` on **both** legs (otherwise `validate_args` asserts because the checkpoint carries `qknorm_impl=apex`, with a misleading error message); `--bf16` registered on the loader, not on `convert.py`; `--bf16` **not** accepted by `saver_swissai_hf`; and the uenv must contain `ApertusForCausalLM`, so `pytorch/v2.9.1:v2` (transformers 4.57.0) rather than `v2.6.0:v1` (4.48.3).

Two structural fixes were also needed: mark the converted checkpoint as `release` (Megatron's `loader_core` asserts `iteration > 0 OR file == 'release'`, but `saver_core` writes `iter_0000000`), and emit only `saver_core` protocol keys (reviewer round-2 fix B1, `4f6bd388`).

### R17 — found 2026-05-21, fixed 2026-05-22

The R1 roundtrip (job `2333864`) reported:

| Metric | Result |
|---|---|
| Standard-tensor max abs diff | **0.0** (bit-perfect through the bf16 cast) |
| R17 keys changed | **128** = 32 layers × 4 xIELU params |
| Shape mismatches / missing / extra keys | none |

`saver_core.check_message()` only consumes the standard transformer protocol keys, so Apertus-specific tensors (`mlp xielu alpha_p/alpha_n`, `q_norm.weight`, `k_norm.weight`) are either rejected or silently dropped — landing at `XIELU.__init__` / `RMSNorm.__init__` defaults (αp = αn = 0.8, β = 0.5, q/k_norm = ones) instead of Apertus's trained values.

**How bad:** the post-conversion V4 eval measured it directly — `arc_easy` 0.8363 → 0.2614, `hellaswag` 0.7884 → 0.2675, `mmlu` 0.5923 → 0.2295, i.e. at or below chance ([`../eval/V4_BENCHMARK_COMPARISON.md`](../eval/V4_BENCHMARK_COMPARISON.md)). An unpatched bakeoff would have compared four differently-initialised copies of a destroyed model.

[`patch_apertus_extras.py`](patch_apertus_extras.py) copies the four tensor families from the source HF checkpoint into every Megatron TP rank; xIELU `beta`/`eps` are not serialised by Megatron, so the patcher verifies the HF values match Megatron defaults before accepting their absence.

### The gate — 2026-05-22 → 2026-05-23

[`r17_patch_roundtrip.sbatch`](r17_patch_roundtrip.sbatch) runs convert → rename → patch → convert back → verify, with pass criteria of **0.0** max abs diff on standard tensors, R17 tensors, and smoke-prompt logits.

| Arm | Job | Patched dir | Tensor diff | Logit diff |
|---|---:|---|---:|---:|
| `vanilla` | `2341182` | `vanilla/megatron_tp2_r17patched` | 0.0 | 0.0 |
| `retok` | `2341239` | `retok/megatron_tp2_r17patched` | 0.0 | 0.0 |
| `centroid` | `2341241` | `centroid/megatron_tp2_r17patched` | 0.0 | 0.0 |
| `td_full25_layer11` | `2357565` | `td_full25_layer11_r17_roundtrip_2357565` | 0.0 | 0.0 |

`submit_all_arms.sh` was changed to default to `INIT_CKPT_SUBDIR=megatron_tp2_r17patched` (`f13f6567`); the raw conversion is reachable only as a deliberate ablation.

## Outcome

- All four bakeoff arms trained from R17-preserved checkpoints; V15 (xIELU scalars survive resize) is confirmed empirically by both the roundtrip and 5 B of stable training.
- R17 remains listed as an **open production gate** in [`../../../CPT_MASTER_20260526.md`](../../../CPT_MASTER_20260526.md) §3.4 — the patcher exists and is proven, but had not been applied to a production initial checkpoint when the subproject closed.
- Open question left in place: swiss-ai must have an internal HF→Megatron loader to validate their saver; nobody asked them, so this loader may be redundant.

## Where things are

| What | Where |
|---|---|
| Loader + install | [`loader_apertus_hf.py`](loader_apertus_hf.py), [`install.sh`](install.sh) (symlinks into a Megatron clone, so local edits are live) |
| R17 patcher + verifier | [`patch_apertus_extras.py`](patch_apertus_extras.py), [`verify_hf_roundtrip.py`](verify_hf_roundtrip.py) |
| Gate jobs | [`r1_roundtrip.sbatch`](r1_roundtrip.sbatch) (raw), [`r17_patch_roundtrip.sbatch`](r17_patch_roundtrip.sbatch) (patched), [`td_layer11_r17_roundtrip.sbatch`](td_layer11_r17_roundtrip.sbatch) (TD arm) |
| Vendored upstream pieces for reference | [`saver_core.py`](saver_core.py), [`saver_swissai_hf.py`](saver_swissai_hf.py), [`schema_base.py`](schema_base.py), [`schema_core.py`](schema_core.py), [`utils.py`](utils.py) |
| Risk entry | `R17` in [`../../../RISKS.md`](../../../RISKS.md) |

## Working documents

[`vanilla_r17_roundtrip_2341182/`](vanilla_r17_roundtrip_2341182/README.md) and [`td_layer11_r17_roundtrip_2357565/`](td_layer11_r17_roundtrip_2357565/README.md) — per-run audit copies (job id, verification JSON, pinned Megatron commit). Historical receipts.
