# Full Apertus-8B mixed CPT

**Status (2026-08-05): owner-authorized and receipt-gated.** D0 selection,
`libduth` risk acceptance and the production launch are recorded in
`configs/owner_decisions_20260805.json`. Data freeze, the DP32/DP64 benchmark,
prelaunch evaluation and production submission still have to pass their live
receipts; authorization never bypasses those technical gates.

This subproject turns the completed 0.5B scheduling result into one normal
full-corpus Apertus-8B CPT run. It uses stationary D0 mixing, the complete
eligible Modern-Greek pass, 20% foreign source-family replay, 1% Old-Greek
replay, the production 148,992-token tokenizer, and the verified untied
layer-11 Token-Distillation initialization. Checkpoint averaging is disabled.

The public-facing training contract, including the 79/20/1 training mix, is
merged in [`eellak/greek-apertus` PR #1](https://github.com/eellak/greek-apertus/pull/1)
at merge commit `c1cb85106d44639a0eb5b2bc659333a5241598a0`. The executable,
receipt-gated orchestration remains in this repository and pins that public
contract in the machine-readable recipe.

The machine authority is
[`configs/recipe_8b_full_mixed.json`](configs/recipe_8b_full_mixed.json).
The resource and provenance map remains
[`CPT_LAUNCH_RESOURCE_SPEC_20260801.md`](../05_token_distillation_cpt/CPT_LAUNCH_RESOURCE_SPEC_20260801.md).

## What is frozen in the draft recipe

- 63,776,651,867 production-tokenizer Modern-Greek source tokens before the
  final global exact-content duplicate exclusion;
- enough already-vetted binaries for 45,299,005,189 foreign replay tokens and
  2,666,110,500 Old-Greek replay tokens;
- an expected 80,729,939,067 active-token 79/20/1 stream, padded only with
  loss-inactive tail slots to 19,248 updates at 4,194,304 token slots/update;
- Apertus-8B architecture, corrected main-pretraining RoPE geometry
  (`4096`, base `500000`, scaling factor `8.0`), TP=2 and untied embeddings;
- AdEMAMix `(0.9, 0.999, 0.999, 4.0)`, weight decay `0.1`, clip `0.1`, bf16
  parameters and fp32 main gradients;
- peak LR `5.5e-5`, 400-update warmup from `5.5e-6`, stable phase, and a final
  20% `1-sqrt` cooldown to the WSD-10 floor;
- NaN/Inf checks enabled—the legacy `--no-check-for-nan-in-loss-and-grad`
  escape hatch is not present in this launcher;
- Swiss-AI Megatron upstream `c92402e39ef3c8e69ea378a59e79059dc14541f4`
  plus the hash-pinned extra-validation and exact-evaluation patch set. The
  former `f8d8a30...` value was a patch-header identifier, not a commit present
  in the runtime checkout, and is no longer accepted by the recipe validator;
- the existing patched-runtime receipt at
  `/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1.receipt.json`
  (SHA-256 `99b9ecbd...`) is revalidated against the live tree before every
  training segment;
- six safe 3,208-update Clariden segments on 16 nodes / 64 GH200 GPUs;
- 13 fixed source-conditioned validation panels every 25 updates;
- 20 native GreekMMLU measurements: initialization, post-warmup, about every
  5B tokens, cooldown start and terminal checkpoint;
- per-document validation at initialization, cooldown start and final model.

WSD-10 is the settled baseline, not a new LR winner claim. The T10/T20/T30
experiment did not yet produce per-document BPB uncertainty. If that rerun
selects a different floor, change only the final LR and regenerate the recipe
ID before launch.

## Reusing data without re-tokenizing

`dataset/freeze_source_inventory.py` reads the completed 8B-tokenizer retained
ledgers in indexed-document order, validates every source manifest, freezes
stable pool catalogs and excludes repeated Modern-Greek exact content. The
shared packing code was generalized to take the receipt's pad token (`3` here,
`10` in the Mini overlay) and global batch (`1024` here, `512` in the Mini
study). It then emits all five lightweight order schedules; this run consumes
only `D0_mixed`.

The neutral external Greek cluster-heldout is rebuilt with the production
tokenizer. The other 12 heldouts already exist under the same production
tokenizer as the training binaries.

Dry-run the data graph on Clariden:

```bash
export FULL8_CODE_ROOT=/iopsstor/scratch/cscs/fffoivos/orchestration/full8-cpt/<immutable-bundle>
export FULL8_STAGE_ROOT=/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed/<run-id>
DRY_RUN=1 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/submit_data_pipeline.sh"
```

The real data graph is metadata/packing work only; it does not train a model.
It still requires an explicit `DRY_RUN=0` invocation after reviewing the paths.

## Per-document validation rerun

The five completed 0.5B endpoints require 13 panels, about 113.9M validation
tokens/model and 569.4M tokens in total. The prepared execution shape uses four
one-node/four-GPU panel groups per model; all five models can therefore run on
20 nodes concurrently.

The current planning estimate is:

- remaining one-panel smoke and throughput calibration: 1–3 hours;
- full five-model execution after allocation: 0.5–1.5 hours;
- scheduler queue: separate and not promised.

The range is intentionally marked planning-only until the representative HPLT
smoke supplies measured tokens/s. The scorer emits a numerator and denominator
for every document, plus base/added-token splits. Neutral external Greek has an
explicit `cluster_id`; the inherited panels currently have only `doc_id`, so
their uncertainty must be called a document bootstrap unless a cluster mapping
is frozen later. See
[`configs/per_document_validation_estimate.json`](configs/per_document_validation_estimate.json)
and [`evaluation/score_documents_hf.py`](evaluation/score_documents_hf.py).
The five terminal HF exports are still present under the authoritative
`evaluations_fp32_v1/iteration_0038496/attempt_3` tree, so this estimate does
not require another checkpoint conversion. The complete dry-run-by-default
graph is `clariden/submit_mini_per_document_rerun.sh`; it freezes the raw-input
manifest, requires one D0/HPLT throughput smoke, runs the five four-node arrays,
and produces paired 10,000-sample document/cluster-bootstrap intervals with
`evaluation/analyze_per_document_endpoints.py`.

## ETA and the Sunday question

The estimate uses completed 16-node 8B jobs `2972672`, `2972674`, `2975267`,
`2975269` and `2975271`, not the 0.5B scaling ratio. Their median update time
was 8.67–8.73 seconds and observed wall time was 10.49–11.05 seconds/update
with the actual validation/save regime.

From 2026-08-05 11:34 Europe/Athens to Sunday 2026-08-09 23:59, the calibrated
critical-path forecast is:

| Clock | Low | Central | Conservative |
|---|---:|---:|---:|
| Compute only | 46.7h median | — | 46.7h p90 |
| Training complete, including data prep, queues, gates and recovery | 62.1h | 75.5h | 107.3h |
| Evidence complete, including GreekMMLU and final validation | 64.1h | 81.5h | 119.3h |

The central evidence ETA is Saturday around 21:00 Athens. The conservative
evidence ETA is Monday around 10:55. Therefore **finishing by Sunday is
plausible but not safe to promise on the proven 16-node path**. Training alone
barely fits the conservative bound; complete evidence does not.

For the central evidence path, the effective latest campaign start is Thursday
2026-08-06 around 14:30 Athens. The conservative evidence path would already
have needed to begin before this preparation window. That is why “Sunday is
plausible” is a capacity-dependent forecast, not a commitment.

At 2026-08-05 08:42 UTC, `normal` reported 260 idle nodes. Fresh 16-, 20- and
32-node `sbatch --test-only` probes predicted starts around 12:55 UTC. Those
test-only IDs (`3004934`–`3004936`) are predictions, not reservations. A
32-node TP2/DP64 run is batch-divisible and could create more deadline margin,
but it is not throughput/restart proven and must not replace the 16-node
recipe without an exact-shape benchmark and a regenerated forecast.

The forecast input is
[`configs/eta_16node_to_20260809.json`](configs/eta_16node_to_20260809.json).

## Launch order

1. Freeze an immutable code bundle and data receipts.
2. Run `scripts/validate_recipe.py` against the pool and schedule receipts.
3. Run `clariden/submit_prelaunch_evaluations.sh`: initial 13-panel loss,
   initial full/clean GreekMMLU, and all 13 initial per-document panels.
4. Review finite metrics, exact checkpoint conversion and available storage
   (minimum 6TB for checkpoints plus conversion/evaluation artifacts).
5. Run the matched 288-update DP32/DP64 benchmark and freeze the promoted or
   fallback execution profile.
6. Run the checkpoint-conversion/native-GreekMMLU smoke, then build the single
   authoritative launch-gate receipt.
7. Dry-run `clariden/submit_production.sh`, refresh scheduler/storage evidence,
   then set `CONFIRM_GPU_LAUNCH=APERTUS8B_FULL_MIXED_CPT` and `DRY_RUN=0`.

During execution, run `scripts/campaign_status.py` after every segment gate,
failure, or material evaluation-backlog change. It reports the exact completed
update/token boundary, Slurm states, required/completed GreekMMLU and terminal
per-document receipts, and a remaining training-core range. That range is not
an end-to-end ETA; regenerate the three-clock forecast whenever queue or
recovery assumptions change.

The selected profile uses either three DP64 segments or the proven six DP32
segments. Each segment is advanced by an `afterany` supervisor that hashes the
checkpoint, blocks scientific failures, retries only infrastructure failures
at most twice and submits the next segment. GreekMMLU is kept off the training
critical path by a four-pipeline bounded queue. Completion requires separate
training and evidence receipts: 20 GreekMMLU results, 39 per-document panel
receipts and the terminal model export.

## Not yet done

- no full-8B pool/packing/schedule receipt has been generated by this new code;
- no production-tokenizer neutral heldout has been generated;
- no initial validation, GreekMMLU or full per-document panel has run;
- no two-update train/resume plus Megatron-to-HF conversion smoke has run;
- no 32-node scaling benchmark has run;
- no production Slurm job has been submitted.

These are launch gates, not hidden assumptions.
