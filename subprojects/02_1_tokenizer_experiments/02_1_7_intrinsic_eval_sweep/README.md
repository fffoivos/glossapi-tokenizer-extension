# 02.1.7 — Intrinsic Eval Sweep

> **In one line:** the stage that actually decided the tokenizer — a 0 → 25,600 sweep on Apertus's own evaluation suite that fixed the cutoff at **17,408 added units, vocab 148,480**, built the curated+backfilled ship artifact, and then counted every token's firings over the full training corpus.
> **Period:** 2026-05-17 → 2026-05-18. Decision frozen 2026-05-18. **Status:** completed; its artifact is the modern-Greek half of every tokenizer used since.
> **Came from / led to:** [`../02_1_4_cutoff_analysis/`](../02_1_4_cutoff_analysis/README.md) + [`../02_1_5_added_token_curation/`](../02_1_5_added_token_curation/README.md) → this → [`../02_1_polytonic_greek_extension/`](../02_1_polytonic_greek_extension/README.md) and [`../../02_2_tokenizer_implementation/`](../../02_2_tokenizer_implementation/README.md)

## Why this existed

`02_1_4` had recommended 11,264 from an argued cap; `02_1_6` had failed to derive a principled budget. What was missing was measurement on the same metric surface Apertus itself used to select its tokenizer (paper §2.2: fertility, compression, vocab utilization, Gini/TFG), at fine enough resolution to see a knee. `02_1_6` had located the implementation — the swiss-ai TokEval fork — so this stage wired it in and ran the grid.

## History

### 2026-05-17 → 2026-05-18 — the sweep

33 tokenizers: `apertus_base`, 25 raw cutoff variants at 1k step to 25,600, 6 curated twins, and the curated+backfilled ship variant. Four sources — TokEval lines and words configs on Apertus-55 FLORES+, the `02_1_3` in-domain harness on gcloud, and MorphScore Greek (UD-derived, n=693). 178,658 merged rows. TokEval pinned at commit `0c4a9c641e78c8243ac753976267fd50675197cb`. Full tables in [`REPORT.md`](REPORT.md).

### 2026-05-18 — the criterion, and the number

The 13k English-unique cap from `02_1_4` was dropped. The decision rule became: **pick the first cutoff where the next 1k of vocab buys less than 1 % additional Greek-fertility improvement.** That is **17,408** ([`CHOSEN_CUTOFF.md`](CHOSEN_CUTOFF.md), decided by the user on this evidence):

- Greek fertility on `C3_val` **1.345**, −44.2 % against `apertus_base` 2.41.
- **82.4 %** of the theoretical maximum fertility gain captured (fitted asymptote 1.118); the full 25,600 captures 88.4 %.
- Vocab **148,480 = 128 × 1160 = 256 × 580**; Apertus ids `0..131,071` verbatim.
- Embedding budget: 136 MiB new embedding rows + 136 MiB new LM-head rows (untied) = **272 MiB**, ~1.7 % of the 8B model.

Two supporting metrics were explicitly demoted rather than leaned on. MorphScore recall moves 0.689 → 0.694 at this cutoff but is "essentially flat" (0.686–0.695) across the whole sweep, so it is colour, not a pillar. TFG on Apertus-55 rises 0.06 % — basis points — and the report argues the metric is structurally biased against script-isolated extensions. The load-bearing argument is the in-domain fertility knee.

### 2026-05-18 — two reviewer rounds and the backfill

Applying `02_1_5`'s curation naively broke the contract. Deleting the 69 in-cutoff noise ids gives 148,411: not 128-aligned, ids renumbered, append-only violated — **rejected in reviewer round 2**. The accepted construction walks C3's merge sequence in order, **skips** the 69, and **accepts the first 17,408 survivors**, renumbering them to `131,072..148,479`:

- 17,477 merges walked = 17,408 accepted + 69 skipped;
- **0 cascade-skips** — no kept merge depended on a skipped one;
- vocab back at 148,480, alignment and append-only intact.

Curation is therefore structural, not a runtime mask. Verification measured raw vs. padded directly on FLORES+ (TFG, Rényi, UTF-8 completeness, fertility, compression, vocab utilization), MorphScore, and all four in-domain Greek slices: every number is flat or **marginally better** for the backfilled build, with the fertility direction consistent across all four Greek slices. The 69 swapped-in merges fire slightly more usefully than the noise tokens they replaced.

### 2026-05-18 — firing-count attribution (`9a6b0392`)

After the decision, the canonical tokenizer was run over the exact C3 BPE training split (`train.parquet` + row-aligned `train_manifest.parquet`, not the upstream mix fallback) on an 8-shard `c4-highcpu-32` fleet, run id `20260518t044858`, ~3,193 s end to end.

- **14,401,554 rows / 99.257 B chars / 24.892 B tokens.**
- GlossAPI-nanochat **49.79 %** of token mass, HPLT **50.21 %** — the "50/50" mix is balanced by token mass, not row count.
- All 17,408 added tokens fire in GlossAPI-nanochat; **27** are zero in HPLT alone; **0** are zero in the combined corpus.
- Component and per-source count invariants passed exactly. The worker VMs had to be deleted manually (the default service account lacked self-delete), which is recorded in the run summary.

## Outcome

- **Ship artifact**: `variants/c3_added_17408_curated_padded/tokenizer.json`, sha `358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`, vocab 148,480. Not in git (large); published to `fffoivos/apertus-tokenizer-extension` and later released as **ModernGreek-148k**.
- **Handoff contract**: consume the file verbatim; no runtime curation logic is needed; the removal mask ships for audit only.
- **Ablation artifacts on disk, explicitly not for shipping**: the pruned `c3_added_17408_curated` (148,411, alignment- and append-only-broken) and the raw `c3_added_17408` (148,480 but carrying the 69 noise tokens).
- **Fed forward**: this tokenizer is the base the polytonic arm continued from, and its 17,408 modern ids are ids `131,072..148,479` of the 148,992 production tokenizer.

## Where things are

| What | Where |
|---|---|
| The decision contract | [`CHOSEN_CUTOFF.md`](CHOSEN_CUTOFF.md) |
| The evidence | [`REPORT.md`](REPORT.md), plots in `artifacts/plots/` (`knee_analysis.png` is the one the decision cites) |
| Ship-artifact builder | `scripts/01c_build_curated_backfilled.py` |
| Build manifest | [`manifests/curated_padded_at_17408_manifest.json`](manifests/curated_padded_at_17408_manifest.json) |
| The 69 filtered ids | [`manifests/removal_mask_at_17408.jsonl`](manifests/removal_mask_at_17408.jsonl) |
| TokEval pin | [`manifests/tokeval_commit.txt`](manifests/tokeval_commit.txt) |
| Firing-count run provenance | [`FIRING_COUNT_RUN_20260518.md`](FIRING_COUNT_RUN_20260518.md), [`manifests/firing_count_20260518_run_summary_augmented.json`](manifests/firing_count_20260518_run_summary_augmented.json) |
| Full pipeline | `scripts/run_all.sh` (build → configs → TokEval → MorphScore → merge → render) |

## Working documents

- [`PLAN.md`](PLAN.md) — the original plan, banner-marked "IMPLEMENTED and shipped"; useful for the table of what was consumed from upstream stages rather than rebuilt.
- [`FIRING_COUNT_PLAN.md`](FIRING_COUNT_PLAN.md) and [`FIRING_COUNT_README.md`](FIRING_COUNT_README.md) — plan and runbook for the cloud firing-count workflow; the completed-run record is `FIRING_COUNT_RUN_20260518.md`.

**Note on two "curated" numbers.** [`REPORT.md`](REPORT.md) § Curated-arm delta reports the *pruned* twins (39 removals at 11,264, 44 at 12,288) and calls their metric deltas "essentially zero" while its own table shows fertility −0.013 and chars/token +0.053. [`CHOSEN_CUTOFF.md`](CHOSEN_CUTOFF.md) reports the *backfilled* ship variant at 17,408, whose deltas are an order of magnitude smaller (fertility −0.0005). These are different constructions; only the backfilled one shipped.
