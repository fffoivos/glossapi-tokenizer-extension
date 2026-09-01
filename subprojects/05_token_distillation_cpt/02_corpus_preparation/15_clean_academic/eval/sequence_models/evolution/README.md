# evolution — controlled model-evolution harness

> **In one line:** a receipt-bound generational search around the frozen bibliography decoder that ran two full generations and promoted nothing — a valid negative result, recorded as one.
> **Period:** 2026-07-18 (`c51f34d3` … `931a56d1`) → 2026-07-19 (`74dd32e7` … `4ea5b978`); launch utilities consolidated 2026-07-22 (`0d5c9c72`). **Status:** completed; `g3_authorized: false`.
> **Came from / led to:** [`../README.md`](../README.md) (the frozen D1 → signal-TCN → anchored-decoder baseline) → this → nothing; the model decision was made by the [cohort-2 bake-off](../../BIBLIOGRAPHY_NEXTGEN_COHORT2_BAKEOFF_20260723.md) instead.

## What it was

A generational search (G0–G5) around a **frozen** baseline, scored on a fixed qualified
268-document inventory, with typed input receipts and `recursive_tree_sha256_v1` directory
digests. The registry minimizes four objectives simultaneously — **token false positives, token
false negatives, spurious blocks per zero-block document, and mean emitted-line boundary error**
— and keeps the Pareto set rather than a single winner
([`BIB_EVOLUTION_RUNBOOK.md`](BIB_EVOLUTION_RUNBOOK.md)).

The sealed-evaluation discipline is the point of the design: `freeze-pareto` →
`prepare-sealed-inference` (which never opens labels) → **one** canonical `evaluate-sealed-batch`
with a source-stratified work bootstrap, Bonferroni correction and 10,000 iterations at seed
20260718. The fuse is written *before* labels are read.

## History

### 2026-07-18 — the learned heading models are made deployable

[`LEARNED_HEADING_DEPLOYMENT_20260718.md`](LEARNED_HEADING_DEPLOYMENT_20260718.md) established
that the five fold pickles can be used without retraining: a fold replay reproduced all
**24,616 × 4 stored float32 probabilities exactly** (max absolute difference `0.0`, zero differing
cells, sklearn 1.9.0). Two disjoint modes were defined — receipt-bound grouped-OOF replay
(verification only) and unseen ensemble-mean (operational, requiring **zero work-ID overlap**
against the 1,100 training works). The validation table was confirmed to share no document or work
ID with heading training.

### 2026-07-19 — G1: 33 candidates, nothing better than the parent

Slurm array `2793581`, tasks 0–26, all COMPLETED; 459 owned files verified. Registry
`g0-g1-full-audited-2793581-da7dfcf`: **33 candidates, 33 eligible, 12 on the Pareto front.**
The finalizer retained the parent `g1-1909806a497053bb7ac4c964` — *"no child weakly dominated the
0.30 control."*

### 2026-07-19 — G2: a large, clearly-signed loss

The deterministic G2 preparation counted 144 `BIB_HEADER` / 55 `BIB_SUBHEADER` /
196 `NON_BIB_HEADER` roles and ran four header windows (array `2793872`). Against the parent's
objective vector (token FP 101,547 / token FN 39,053 / spurious 0.037037 / boundary 2.570288),
every window traded a small false-positive gain for a very large false-negative loss:

| Window | Δ token FP | Δ token FN | Δ boundary |
|---|---:|---:|---:|
| 1 | −4,020 | **+113,212** | −0.166693 |
| 2–4 | −4,316 | **+113,208** | −0.199373 |

Final registry: 37 candidates, 37 eligible, 12 Pareto, **none of them G2**. The handoff
`g0-g2-audited-2793872-7df2344` records `promoted G2 candidate: none`, retains the parent, and
sets `g3_authorized: false`. The run document closes with the right sentence:
**"This is a valid negative result."**

## Outcome

- **Zero promotions across two full generations.** G1 could not beat its own control; G2 lost
  ~113 k tokens of recall to buy ~4 k of precision.
- The harness itself worked as designed: the authorization chain (full G1 audit → G2 preparation
  bound to it → frozen no-admission handoff) held, and G3 was never authorized rather than being
  run on hope.
- The learned heading models were proved byte-exactly replayable, which is what let them be used
  later without a retrain.

## Contents

| Path | What |
|---|---|
| [`BIB_EVOLUTION_RUNBOOK.md`](BIB_EVOLUTION_RUNBOOK.md) | The harness: generations, four objectives, receipt types, the G3 canonical pipeline, and the sealed-evaluation protocol. |
| [`BIB_EVOLUTION_G1_G2_RUN_20260719.md`](BIB_EVOLUTION_G1_G2_RUN_20260719.md) | The run record — jobs, registries, objective vectors, and the no-promotion decision. |
| [`LEARNED_HEADING_DEPLOYMENT_20260718.md`](LEARNED_HEADING_DEPLOYMENT_20260718.md) | Byte-exact fold replay, the two operating modes, and the work-ID overlap gate. |
| [`legacy_launch_20260718/`](legacy_launch_20260718/README.md) | The former top-level `train-apertus-toc-bib-evolution-launch` scripts, consolidated here on 2026-07-22 and pinned to commit `931a56d1…`. Retained for replay only. |
