# references — pinned primary sources

> **In one line:** the local corpus of primary sources against which the training recipe was audited on 2026-05-21 — 8 repositories at pinned commits and 15 papers — so that every hyperparameter in the recipe cites a file and line rather than a memory.
> **Period:** 2026-05-21 (`fde4146d`, then `dd612462`). **Status:** frozen; the pins are still what [`../TRAINING_RECIPE.md`](../TRAINING_RECIPE.md) and the sbatch files cite.

## Why this existed

The bakeoff claimed Apertus fidelity: same optimizer, same activation, same normalisation, same document handling. That claim is only checkable if the sources are local and version-pinned. The audit pass that used this directory found real errors — three wrong Megatron flag names, and the absence of an HF→Megatron Apertus loader — which is the argument for having done it.

## What is here

- [`MANIFEST.md`](MANIFEST.md) — the catalogue: repo URLs with pinned commits, the paper table with arXiv IDs and what each is cited for, and the citation convention (`[Cite: references/papers/<file> §<section> p<page>]` for papers, `path:line@commit` for code).
- [`clone_references.sh`](clone_references.sh) — rebuilds `references/repos/` at the pinned commits. The repos themselves are **gitignored** (large) and are not in this tree.
- [`download_papers.sh`](download_papers.sh) — fetches the papers. Committed where the licence allows; downloaded on demand otherwise.

The two pins that matter most downstream: `swiss-ai/Megatron-LM` at `c92402e39ef3c8e69ea378a59e79059dc14541f4` (the training engine) and `swiss-ai/pretrain-code` at `531cc8be2f76064127cad99a61019f985a7c7ee2`, whose `pretraining/submit_apertus_8b.sh` is the authoritative record of what flags Apertus actually ran with.

## History

| Date | What happened | Evidence |
|---|---|---|
| 2026-05-21 | 8 repos cloned at pinned commits, 15 papers downloaded, recipe audited against them | `fde4146d` |
| 2026-05-21 | Most papers switched from PDF to arXiv HTML — **~37 MB → ~15 MB**, and easier to cite by section id. PDFs kept where no HTML exists (QK-Norm, StarCoder, Megatron, Krikri, FVT) and for the Apertus tech report, which is cited by page number | `dd612462`, [`MANIFEST.md`](MANIFEST.md) |

## Working documents

`MANIFEST.md` also names two things that were *not* found: `swiss-ai/apertus-finetuning-recipes` contains SFT recipes only, with **no published CPT recipe**; and the ILSP Greek harness task configs live in the Meltemi/Krikri forks rather than in the swiss-ai harness. Both negatives shaped what had to be written from scratch in [`../03_4_implementation_experiments/init_bakeoff/`](../03_4_implementation_experiments/init_bakeoff/README.md).
