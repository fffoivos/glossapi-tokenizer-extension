# References manifest

Pinned external sources that justify the training recipe. Every concrete
claim in [`../TRAINING_RECIPE.md`](../TRAINING_RECIPE.md),
[`../REVIEW_PRESENTATION.md`](../REVIEW_PRESENTATION.md), and the
sbatch / Python under [`../03_4_implementation_experiments/init_bakeoff/`](../03_4_implementation_experiments/init_bakeoff/)
should cite either a `papers/` PDF section or a `repos/<repo>/<path>:<line>`
at the pinned commit listed below.

Repos are **not committed** (large; `references/repos/` is gitignored). They
are reproducible via [`clone_references.sh`](clone_references.sh). Papers
(small PDFs) are committed where their license allows it; otherwise the
script downloads them on demand.

## Repos (cloned at pinned commits)

| Repo | URL | Pinned commit | Purpose |
|---|---|---|---|
| `swiss-ai_Megatron-LM` | https://github.com/swiss-ai/Megatron-LM | `c92402e39ef3c8e69ea378a59e79059dc14541f4` (main HEAD 2026-05-20) | Apertus training engine — AdEMAMix optimizer, xIELU activation, QK-Norm, Goldfish loss, cross-doc / EoD plumbing |
| `swiss-ai_pretrain-code` | https://github.com/swiss-ai/pretrain-code | `531cc8be2f76064127cad99a61019f985a7c7ee2` (main HEAD 2026-05-20) | The canonical Apertus pretrain launch scripts — `pretraining/submit_apertus_8b.sh` is the authoritative source for "what flags were actually run" |
| `swiss-ai_pretrain-data` | https://github.com/swiss-ai/pretrain-data | HEAD | Apertus's DataTrove-based preprocessing — `examples/tokenize_megatron/preprocess_megatron.py` |
| `swiss-ai_lm-evaluation-harness` | https://github.com/swiss-ai/lm-evaluation-harness | HEAD | Apertus team's fork of EleutherAI's harness — cited in tech report §5.1 footnote 45 |
| `swiss-ai_apertus-finetuning-recipes` | https://github.com/swiss-ai/apertus-finetuning-recipes | HEAD | (SFT only; no published CPT recipe — verified by Agent A) |
| `swiss-ai_apertus-tech-report` | https://github.com/swiss-ai/apertus-tech-report | HEAD | Source of `papers/apertus_2509.14233.pdf` |
| `apple_ml-ademamix` | https://github.com/apple/ml-ademamix | HEAD | Author's reference PyTorch implementation of AdEMAMix |
| `EleutherAI_lm-evaluation-harness` | https://github.com/EleutherAI/lm-evaluation-harness | HEAD | Upstream lm-eval-harness (for task-config diffs vs swiss-ai fork) |

## Papers (PDFs)

| File | arXiv ID / DOI | Authors | Year | Use |
|---|---|---|---|---|
| `papers/apertus_2509.14233.pdf` | 2509.14233 v2 | Swiss AI consortium | 2025 | Every Apertus hyperparameter |
| `papers/ademamix_2409.03137.pdf` | 2409.03137 | Pagliardini, Ablin, Grangier | ICLR 2025 | Optimizer update rule, β3 warmup schedule, cold-restart guidance |
| `papers/goldfish_2406.10209.pdf` | 2406.10209 | Hans, Wen, Jain et al. | 2024 | Goldfish loss (production CPT only) |
| `papers/retok_2410.04335.pdf` | 2410.04335 | Gu, Zhao et al. | 2024 | ReTok init for new vocab tokens (E + U) |
| `papers/fvt_emnlp2022_industry_41.pdf` | EMNLP 2022 Industry Track | Gee, Zugarini, Rigutini, Torroni | 2022 | Origin of subpiece-mean init (FVT) |
| `papers/hewitt_vocab_expansion.html` | (technical note) | Hewitt | 2021 | Centroid init recipe |
| `papers/mundra_2407.05841.pdf` | 2407.05841 | Mundra et al. | 2024 | Empirical comparison of vocab-expansion init |
| `papers/qknorm_2010.04245.pdf` | 2010.04245 | Henry et al. | 2020 | QK-Norm |
| `papers/wsd_minicpm_2404.06395.pdf` | 2404.06395 | Hu et al. | 2024 | WSD LR schedule |
| `papers/fineweb_2406.17557.pdf` | 2406.17557 | Penedo et al. | 2024 | FineWeb-Edu / FineWeb-2 |
| `papers/finewebhq_2502.10361.pdf` | 2502.10361 | Messmer, Sabolčec, Jaggi | 2025 | FineWeb2-HQ |
| `papers/starcoder_2305.06161.pdf` | 2305.06161 | Li et al. | 2023 | StarCoderData v1.2 |
| `papers/megatron_1909.08053.pdf` | 1909.08053 | Shoeybi et al. | 2019 | Megatron-LM base engine |
| `papers/meltemi_2407.20743.pdf` | 2407.20743 | Voukoutis et al. | 2024 | Origin of ILSP Greek test sets (core 6) |
| `papers/krikri_2505.13772.pdf` | 2505.13772 | – | 2025 | ILSP Greek suite extension |

## Citation convention

In our docs, after this pass:

- **Paper claim:** `[Cite: references/papers/apertus_2509.14233.pdf §C Table C.4 p.82]` (note: § + table + page; the path is reproducible).
- **Code claim:** `[Cite: references/repos/swiss-ai_Megatron-LM/megatron/core/optimizer/ademamix.py:L<N>@<pinned-commit>]` (path : line @ commit).
- **Repo-pattern claim:** `[Cite: references/repos/swiss-ai_pretrain-code/pretraining/submit_apertus_8b.sh:L207]`.

Audit pass: every citation in `TRAINING_RECIPE.md`, `_train_config_common.env`,
`bakeoff_train.sbatch`, `REVIEW_PRESENTATION.md`, `EVAL_RECIPE.md`, and the
arm scripts will be relinked to the local references in this pass.
