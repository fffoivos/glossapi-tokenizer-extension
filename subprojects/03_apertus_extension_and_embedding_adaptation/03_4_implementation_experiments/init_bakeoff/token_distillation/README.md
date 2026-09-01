# token_distillation — the fourth arm

> **In one line:** the only non-closed-form arm — instead of a formula, the new rows are *learned* by distilling the base model's hidden states over real corpus snippets containing each new token; it ran as a four-gate ladder in a single day and produced `td_full25_layer11`, the challenger that beat ReTok and Centroid and then nearly beat Vanilla.
> **Period:** 2026-05-22 (plan) → 2026-05-23 (all gates passed). **Status:** completed; the arm was trained and evaluated by [`../bakeoff_training/`](../bakeoff_training/README.md) and [`../eval/`](../eval/README.md).
> **Led to:** [`../../../../05_token_distillation_cpt/`](../../../../05_token_distillation_cpt/), which took TD forward, and the 2026-07-29 production init pipeline in [`../../polytonic_cutoff_probe/`](../../polytonic_cutoff_probe/README.md).

## Why this existed

cpt_plan v0.7 §13 had bracketed distillation as too risky. [`../../../TOKEN_DISTILLATION_PLAN.md`](../../../TOKEN_DISTILLATION_PLAN.md) (2026-05-22) argued the bracketing was based on a tied-vs-untied embedding worry that the official implementation already handles, and that the real risk is integration: Apertus has untied `E`/`U`, our tokenizer is merge-extended with fixed production IDs (not `add_tokens(...)`), and the result must survive the same R17 roundtrip gate as every other arm. The plan set explicit start conditions — do not interrupt the running bakeoff; run only if ReTok has healthy new-token behaviour but still trails Vanilla.

## History — the four gates, all on 2026-05-23

**Gate 1 — CPU coverage prepass.** Scan the mixed JSONL in order until 2 B extended-token emissions and count, per new token ID in `[131072, 148480)`, how many usable snippets exist. Eleven jobs were burned discovering that `xfer` compute nodes are x86_64, expose no `uenv`, and ship Python 3.6 — resolved by an xfer-built Python 3.11 venv with `tokenizers==0.22.1`. A twelfth (`2349342`) scanned the full 2 B and then **refused to emit**, because 9 documents were not NFC; a CPU-only JSONL normaliser was added and the prepass rerun on `bulk_mix.nfc.jsonl`.

Result (job `2351374`, 4h09m): 2,000,000,000 tokens scanned over 1,645,852 docs, `non_nfc_docs = 0`, **17,377 of 17,408 tokens (99.82 %) had ≥100 usable snippets**, 15 more had ≥25, and only 5 fell below 20. Gate decision `run_full_td_100`.

**Gate 2 — bounded smoke.** A small token subset at `target_layer=-1`, 25 snippets, verifying that every base row and every unselected new row stays gradient-zeroed and exact-checked by the vendored training loop.

**Gate 3 — packed layer pilot.** The upstream README suggests target layers around one-third depth can beat the paper-default last layer; for Apertus-8B's 32 layers that is layer 11. Both candidates were packed into one 4-GPU allocation rather than launched separately.

| Arm | BPB | Δ vs ReTok | D1 mean rank | D1 top-1 |
|---|---:|---:|---:|---:|
| ReTok | 2.9503 | 0.0000 | 3868.27 | 0.0065 |
| TD last (-1) | 2.7830 | −0.1673 | 3294.05 | 0.0146 |
| **TD layer 11** | **2.7753** | **−0.1750** | **3263.69** | **0.0148** |

**layer 11 selected.** The pilot also produced the budget calibration that set the production setting: a 100-snippet all-token run would take 18–19 GPU-hours, past `normal`'s 12 h cap, so the full run used the paper-fast **25-snippet** setting.

**Gate 4 — full-token TD + preservation + R17.** `retok_td_full25_layers_20260523T092602Z/layer11` completed in 16,254.8 s over 54,303 dataloader steps: **17,377 tokens trained, 15 skipped**, `target_layer=11`, `snippets_per_token=25`, batch 8. A preservation verifier confirmed every pre-existing row was untouched, and the R17 roundtrip gate passed as job `2357565`.

## Outcome

- **`td_full25_layer11`** — the challenger checkpoint, R17-preserved and Megatron-loadable.
- **At its 2 B checkpoint (iter 476)** it beat ReTok on heldout BPB (0.5311 vs 0.5739) and Centroid (0.8994), and on new-target top-1/top-10 (0.3864/0.6191 vs ReTok's 0.3497/0.5772). ReTok kept a higher five-prompt greedy new-token utilization (0.3580 vs 0.2080) — flagged in the log as a weak diagnostic. All three extended arms preserved the Greek compression gain: 3.973 chars/token and 1.735 tokens/word vs Vanilla's 2.557 and 2.693.
- **It did not beat Vanilla at 2 B** on the aggregate Greek/preservation criteria, which is why the production decision stayed Vanilla — and then the 3.5 B/5 B continuations partially reversed that, and the native-Greek suite reversed it back. See [`../eval/README.md`](../eval/README.md).
- **Integration rule that carried forward:** never call the package's high-level `TokenDistillation.run(...)`, because it appends tokens with `add_tokens(...)`; call the lower-level training loop with an explicit `base_phrase_ids → new_token_id` mapping. Subproject 05's production init pipeline follows exactly this.

## Where things are

| What | Where |
|---|---|
| Coverage gate | [`td_coverage_prepass.py`](td_coverage_prepass.py) + `td_coverage_prepass_xfer.sbatch`, [`summarize_td_coverage.py`](summarize_td_coverage.py), `td_coverage_postprocess_xfer.sbatch` |
| Token selection | [`select_td_pilot_tokens.py`](select_td_pilot_tokens.py) |
| Training | [`train_retok_td.py`](train_retok_td.py), [`train_retok_td.sbatch`](train_retok_td.sbatch), [`train_retok_td_layer_pilot_packed.sbatch`](train_retok_td_layer_pilot_packed.sbatch) |
| Preservation check | [`verify_td_preservation.py`](verify_td_preservation.py) + xfer sbatch |
| Pinned upstream | [`external/token-distillation/`](external/token-distillation/PINNED_UPSTREAM.md) — `konstantinjdobler/token-distillation` at `35702b5809599ecd68b7845eca27a0d7b7cec0da` |
| Plan | [`../../../TOKEN_DISTILLATION_PLAN.md`](../../../TOKEN_DISTILLATION_PLAN.md) |
| Artifacts | Clariden `/iopsstor/scratch/cscs/fffoivos/token_distillation/` (~125 GB) |

## Working documents

- [`RUN_LOG_20260523.md`](RUN_LOG_20260523.md) — 90 KB append-only log covering the whole day plus the 2 B arm's per-checkpoint evals through 2026-05-24. It is the primary evidence for every number above, and also the best record of the `xfer` environment failures. Its early entries say `BPC` where current docs say `BPB`; the header explains the alias.
- [`full_td_20260523T092602Z/`](full_td_20260523T092602Z/README.md) — per-run audit copy for the full-token training job.
