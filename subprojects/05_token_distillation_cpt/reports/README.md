# reports — the 13.5 B two-arm pilot result

> **In one line:** the one published result of this subproject — a self-contained HTML report
> comparing the vanilla and Token-Distillation arms of the 13.5 B CPT pilot.
> **Period:** 2026-06-10 → 2026-06-11 (`dd2776b7` … `47601092`). **Status:** completed;
> the run it describes is the pilot, superseded as a *design* by the sweeps in
> [`../03_training_experiments`](../03_training_experiments/README.md).

## History

The directory started on 2026-06-10 as a home for launch diagnostics — a Megatron/NCCL
`NO_SPACE` write-up and a 16-node CXI timing note — both of which were folded into
[`../ARCHIVE.md`](../ARCHIVE.md) the same day (`3bbe36ba`) and deleted. What remains is the
result package built on 2026-06-11, the day both arms finished, plus one edit (`47601092`)
that added the randomized-data-order caveat to the report's methodology section.

## What is here

| File | What it is |
|---|---|
| [`cpt_2arm_performance.html`](cpt_2arm_performance.html) | The report. Plots are inlined as base64, so it opens standalone. |
| [`build_report.py`](build_report.py) | Regenerates the plots and HTML from the summary JSON. |
| [`cpt_2arm_summary.json`](cpt_2arm_summary.json) | The data: per-arm training curves (3,218 rows), per-set held-out loss every 25 iters, 14 benchmark checkpoints of native MCQ / retention / Greek-NLP, base-model reference values, and BPB byte/token inputs. |
| `assets/*.png` | The six figures, also embedded in the HTML. |

## The result, as reported

- **Native Greek MCQ** (GreekMMLU + ILSP medical MCQA + ILSP ASEP, 18,489 questions,
  micro-accuracy): base **48.3 %** → vanilla **55.3 %** → TD **58.7 %**; macro 42.7 → 51.1 → 53.0.
- **Retention:** English MMLU 56.2 → 59.5 (up), Global-MMLU 49.2 → 50.3, PIQA flat;
  ARC-Challenge −4.8 and ARC-Easy −4.3 are the visible dips.
- **Tokenizer efficiency:** the TD tokenizer needs 1,583 M tokens where the base needs
  2,309 M on the same held-out text — **−31 %** (HPLT −38 %, OpenArchives −30 %, PhD −26 %).
- **Language modeling:** essentially tied in bits/byte, which is the only cross-tokenizer-fair
  loss comparison; per-token loss is not comparable across arms.
- **Run health:** 0 NaN and 0 skipped iterations over 3,218 steps × 2 arms, ~8.6 s/iter at
  64 GPUs (~394 TFLOP/s/GPU), 8.1–8.2 h compute per arm.

## Caveats the report itself states

Equal token budget, not equal epochs (TD ≈ 1.0 epoch of its 13.5 B binary, vanilla ≈ 0.69 of
its 19.5 B one); bits/byte carries ~1 % JSON overhead in absolute terms but cancels across
arms; and **the HPLT→OpenArchives ordering was prepared but did not take effect** — Megatron's
dataloader shuffle was left on, so both arms trained on shuffled data. The arm comparison is
unaffected; any curriculum claim is not supported by this run.

Note that [`../ROADMAP_20260611.md`](../ROADMAP_20260611.md) quotes a different headline
(GreekMMLU-only, 48.8 → 55.6 → 59.3) that no file in this directory reproduces.
