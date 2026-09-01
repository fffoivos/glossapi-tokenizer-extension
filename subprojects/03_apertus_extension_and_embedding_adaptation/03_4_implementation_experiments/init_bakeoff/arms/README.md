# arms — the four initialisation methods

> **In one line:** the code that decides what the 17,408 new rows of `E` and `U` contain before training starts, plus the Clariden pipeline that materialises and converts the resulting checkpoints.
> **Period:** 2026-05-20 (`af438d4d`) → 2026-05-21 (build + convert run). **Status:** completed.
> **Feeds:** [`../bakeoff_training/`](../bakeoff_training/README.md) via [`../megatron_patches/`](../megatron_patches/README.md).

## Why this existed

Apertus has `tie_word_embeddings=false`, so the input embedding `E` and the LM head `U` are independent matrices and each needs its own initialisation for every new token ID. Three closed-form hypotheses were coded so the comparison would have **no gradient descent at init time** — the only variance source is training, not initialisation. (Token Distillation, the fourth arm, breaks that property deliberately and lives in [`../token_distillation/`](../token_distillation/README.md).)

## The arms

| Arm | Vocab | Init rule for a new row `T` | Extra params (E + U) |
|---|---:|---|---:|
| `vanilla.py` | 131,072 | none — symlinks the base checkpoint | 0 |
| `retok.py` | 148,480 | `mean(base_E[p] for p in base_tokenizer.encode(decode(T)))`, then norm-match | ~142.6 M (17,408 × 4,096 × 2) |
| `centroid.py` | 148,480 | per-script centroid of base Greek tokens + Gaussian noise, then norm-match | ~142.6 M |

Both extension arms norm-match to the targets measured in [`../../../03_1_greek_embedding_diagnostic/`](../../../03_1_greek_embedding_diagnostic/README.md): **E = 5.05, U = 3.80**.

## History

| Date | What happened | Result | Evidence |
|---|---|---|---|
| 2026-05-20 | Three arms + shared helpers + local smoke test written | Smoke green: both extension arms produce norm-matched `[200, 4096]` rows in ~10–15 s without loading the full model, reproducing the diagnostic's medians to within 1 % (`E[modern].p50 = 5.047`, `U[modern].p50 = 3.797`). Critically, ReTok and Centroid rows for the same token are **near-orthogonal (mean cos ≈ 0.03)** — confirming the two arms test genuinely different hypotheses | [`test_init_logic.py`](test_init_logic.py), `af438d4d` |
| 2026-05-21 | Recipe audit patched the arm scripts against pinned primary sources | see `_archive/2026-05-24_2B_bakeoff_review/AUDIT_FINDINGS.md` | `fde4146d` |
| 2026-05-21 | Clariden build + convert pipeline run | Jobs `2335382` (build, 2m43s) and `2335384` (convert, 1m41s) both `0:0`. Produced `vanilla/` (symlink), `retok/` and `centroid/` (~16 GB each) plus Megatron `release` checkpoints | [`init_modern_only_148480_20260521/`](init_modern_only_148480_20260521/README.md) |
| 2026-05-21 | Two environment failures fixed along the way | `2335353` failed on Slurm spool-path handling (fixed with `SLURM_SUBMIT_DIR`); `2335371` failed because `pytorch/v2.6.0:v1` ships transformers 4.48.3 without `ApertusForCausalLM` — init jobs moved to `pytorch/v2.9.1:v2` while the training jobs stayed on 2.6.0 | `_archive/2026-05-21_overnight_session/CSCS_OVERNIGHT_STATE.md` |
| 2026-05-21 | Default init root switched to the modern-only tree | `dfc7320e` |

## Outcome

- Three init checkpoints in both HF and Megatron form, with `init_build_summary.json` recording per-arm stats and the V2 sanity check (correct shapes, no NaN/inf on the forward pass).
- The composite 153,600 path survives behind `--vocab-size 153600` in `build_init_checkpoints.py` but was never used — the 2026-05-20 scope decision took the bakeoff modern-only, and the eventual production polytonic tokenizer was a different, smaller extension (148,992).
- The checkpoints the training jobs actually loaded are the **R17-patched TP=2** variants, not these raw conversions — see [`../megatron_patches/README.md`](../megatron_patches/README.md).

## Where things are

| What | Where |
|---|---|
| Init logic | [`vanilla.py`](vanilla.py), [`retok.py`](retok.py), [`centroid.py`](centroid.py), [`_common.py`](_common.py) |
| Clariden driver + jobs | [`build_init_checkpoints.py`](build_init_checkpoints.py), [`build_init_checkpoints.sbatch`](build_init_checkpoints.sbatch), [`convert_init_checkpoints.sbatch`](convert_init_checkpoints.sbatch), [`submit_init_pipeline.sh`](submit_init_pipeline.sh) |
| Home-side smoke (no GPU, no model load) | [`test_init_logic.py`](test_init_logic.py) |
| Weights | Clariden `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/` |

## Working documents

[`init_modern_only_148480_20260521/`](init_modern_only_148480_20260521/README.md) — audit copy of the one build/convert run: job ids, elapsed times, `init_build_summary.json`, and the Slurm `.out`/`.err` logs. Historical receipt.
