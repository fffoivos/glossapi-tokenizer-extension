# 02.1.1 — Tokenizer Training

> **In one line:** the continuous-BPE trainer that produced every tokenizer arm in this program, plus the four-arm plan it was originally built to execute — a plan that was overtaken before it finished.
> **Period:** 2026-04-13 (`ca6f6a36`, first orchestration script) → 2026-05-18 (`7deea009`, moved into this directory). **Status:** completed; the scripts are the reusable stage-1 tooling, the plan documents are historical.
> **Came from / led to:** [`../../02_apertus_tokenizer_spec/`](../../02_apertus_tokenizer_spec/README.md) → this → [`../02_1_2_cutoff_variant_builder/`](../02_1_2_cutoff_variant_builder/README.md)

## Why this existed

Stage 1 of the pipeline: turn a training mix into a full BPE tokenizer at the maximum target vocab, while keeping Apertus's front end bit-identical. The hard part is not BPE — it is proving after the fact that the trained artifact still has Apertus's special-token block, regex split, ByteLevel regime and first 1,000 ids untouched, so that every later truncation of it is automatically Apertus-compatible.

## History

| Date | What happened | Result | Evidence |
|---|---|---|---|
| 2026-04-13 | `wait_for_tokenizer_mixes_and_launch_training.sh` added — watcher that launches training when the upstream `mix.parquet` lands | Training could be queued behind the corpus pipeline | `ca6f6a36` |
| 2026-04-14 | `train_discovery_tokenizer.py` added in the repo canonicalization | Fresh-discovery arms F1/F2 buildable | `a062d0aa` |
| 2026-04-26 | Low-level `train_bpe_from_text_shards.py` and `inspect_bpe_vocab_denoising.py` moved in from the glossapi codebase | Shared BPE core for both arm families | `2f6b6b89` |
| 2026-04-29 | `train_continuous_bpe_tokenizer.py`, the publish/watch scripts, and [`CONTINUOUS_BPE_EXTENSION_PLAN.md`](CONTINUOUS_BPE_EXTENSION_PLAN.md) + [`CONTINUOUS_BPE_EXTENSION_TODO.md`](CONTINUOUS_BPE_EXTENSION_TODO.md) landed | Continuous arms buildable; plan froze the ceiling at `+25,600` → total `156,672` and the cutoff grid at `{10240, 15360, 20480, 25600}`. PLAN §2.3 recorded `F1` and `F2` as existing, `C1` and `C2` as **not yet built** | `edb98d6b` |
| 2026-05-11 | C3 (`GlossAPI + HPLT 50/50`, wave-2 broad cleaner, 156,672) was declared the arm; the four-arm comparison was closed without completing | The plan's §2–§7 became dead text; an archive banner was later added to the top of the plan | [`../../../docs/C3_CONVERGENCE.md`](../../../docs/C3_CONVERGENCE.md) |
| 2026-05-18 | Scripts and plans moved from the parent directory into this sub-subproject in the pipeline reorg | This README written | `7deea009` |

## Outcome

- **What actually ran here**: the C3 arm at 156,672 total vocab (`c3_driver.sh` on the gcloud worker, recorded in [`../../../docs/C3_CONVERGENCE.md`](../../../docs/C3_CONVERGENCE.md)), plus the earlier F1/F2 discovery arms. `C1`/`C2` exist only as 156,672 snapshots used as analyzed baselines, and no evidence in this directory shows the planned four-arm head-to-head being run to completion.
- **Per-run contract artifacts**: `replication_check.json` and `front_end_contract_check.json` alongside `tokenizer.json` and `training_summary.json`. Because these are verified on the *full* arm, every cutoff variant downstream inherits the guarantee by copying the front-end JSON verbatim.
- **Left as reusable tooling**: the trainer is parameterized by mix and target vocab, and was reused unchanged by the polytonic arm to continue BPE on top of the frozen 148,480 tokenizer.
- Raw archived arm artifacts (F1, F2, C1 from the wave-3 strict run) live outside git under `runs/_archive/production_strict_v2/tokenizers/`.

## Where things are

| What | Where |
|---|---|
| Continuous-BPE trainer (the one that made C3) | `scripts/train_continuous_bpe_tokenizer.py` |
| Fresh-discovery trainer (F1/F2, archived arms) | `scripts/train_discovery_tokenizer.py` |
| Shared BPE core | `scripts/train_bpe_from_text_shards.py` |
| Post-training added-unit inspection | `scripts/inspect_bpe_vocab_denoising.py` |
| HF upload | `scripts/publish_tokenizer_extension_repo.py` |
| Orchestration (wait-for-mix, watch-and-publish) | `scripts/wait_for_tokenizer_mixes_and_launch_training.sh`, `scripts/watch_continuous_runs_and_publish.py` |

## Working documents

- [`CONTINUOUS_BPE_EXTENSION_PLAN.md`](CONTINUOUS_BPE_EXTENSION_PLAN.md) — historical. Carries its own archive banner: §1 (cutoff grid), §8 (evaluation shape), §9–§10 (publication and acceptance) survived into the C3 work; §2–§7 describe the four-arm comparison that was closed by fiat.
- [`CONTINUOUS_BPE_EXTENSION_TODO.md`](CONTINUOUS_BPE_EXTENSION_TODO.md) — historical work list under the same framing; §1.4 is where the original four-candidate cutoff grid was frozen.

**Discrepancy:** the previous README described the trainer as running "in 4 phases" and then listed six (identity check → count segments → build sequence shards → aggregate sequences → merge loop → write tokenizer); [`../../SUBPROJECTS_OVERVIEW.md`](../../SUBPROJECTS_OVERVIEW.md) says six. Six is the list both documents actually enumerate.
