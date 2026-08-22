# Hard HPLT → OpenArchives cross-scale experiment handoff

Date: 2026-08-22

Status: **training complete; 34/34 clean GreekMMLU checkpoint evaluations complete; report QA passed**

Worktree: `/Users/foivoskarounos-zamparloukos/Projects/.codex-worktrees/h2g-greekmmlu-trajectories`

Branch: `agent/h2g-greekmmlu-trajectories`

## 1. Research questions and result

This experiment had three goals:

1. reproduce the earlier hard HPLT→OpenArchives 8B behavior on a rebuilt, benchmark-clean corpus;
2. test whether a matched Apertus 1.5B run follows the same GreekMMLU trajectory as 8B;
3. extend OpenArchives training by two approximately 1B-token-slot checkpoints.

The cross-scale answer is clear: **1.5B does not mirror 8B on GreekMMLU**.

| Quantity | Apertus 1.5B | Apertus 8B |
|---|---:|---:|
| First measured, update 238 / 0.998B slots | 42.33% | 49.65% |
| Best accuracy | 43.55% at update 1,904 / 7.986B | 58.31% at update 2,618 / 10.981B |
| HPLT boundary, update 2,261 / 9.483B | 42.97% | 55.96% |
| OpenArchives endpoint, update 3,218 / 13.497B | 39.88% | 57.23% |
| Final, update 3,694 / 15.494B | 39.79% | 57.13% |
| HPLT-boundary → OA-endpoint | **−3.09 pp** | **+1.27 pp** |
| First measured → final | **−2.54 pp** | **+7.48 pp** |
| Final below own accuracy peak | −3.76 pp | −1.18 pp |

Trajectory similarity is poor rather than merely vertically offset:

- accuracy level Pearson correlation: `−0.6698`;
- adjacent-checkpoint accuracy-change Pearson correlation: `−0.0581`;
- adjacent-change Spearman correlation: `0.1265`.

The category evidence points the same way. Of 31 subject trajectories, 1.5B reaches its best checkpoint during HPLT for 29 subjects and during OpenArchives for only one. For 8B, 22 subjects peak during OpenArchives and nine during HPLT.

The continuous metrics add an important qualification. The 1.5B final choice NLL (`1.34670`) is lower than at its first measured checkpoint (`1.38264`) despite lower final accuracy. Thus the run improved the average probability assigned to correct choices while changing enough argmax rankings in the wrong direction to reduce accuracy. Accuracy and probability quality are not interchangeable here.

## 2. Frozen scientific contract

The authoritative experiment contract is [hard_h_to_g_replication_v1.json](configs/hard_h_to_g_replication_v1.json). Its historical `status` string predates execution and is not the completion authority; completed checkpoint, evaluation, and aggregate receipts are the execution evidence listed below.

### Models and tokenizer

| Property | 1.5B | 8B |
|---|---:|---:|
| Parent | `swiss-ai/Apertus-v1.1-1.5B@dbe8919...` | `swiss-ai/Apertus-8B-2509@3162c996...` |
| Layers | 16 | 32 |
| Hidden width | 2,048 | 4,096 |
| Attention heads / KV heads | 32 / 8 | 32 / 8 |
| RoPE | theta 500,000; factor 8; max positions 4,096 | same |
| Embeddings | untied | untied |

Both use `fffoivos/apertus-tokenizer-extension` revision `fcd33ec09fb7d86bc072b3a4b3e890efa6473b66`, vocabulary 148,480, tokenizer JSON SHA-256 `358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`, divisible by 256 without padding.

The 8B initialization authority is `experiment-checkpoints/TokenDistil-Init` at that HF revision, target layer 11. The 1.5B Token-Distillation adaptation uses target layer 6 and the predeclared row-norm/coverage gates. The shared TD recipe is one epoch, 25 snippets/token, batch 8, LR `1e-4`, bf16, hidden-state MSE plus output CE, while preserving base rows exactly. See [1P5B_TD_ROW_NORM_DIAGNOSTIC_20260815.md](1P5B_TD_ROW_NORM_DIAGNOSTIC_20260815.md) and [ULTRACODE_R2_REMEDIATION_20260814.md](ULTRACODE_R2_REMEDIATION_20260814.md).

### Data identity and schedule

Source dataset: `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2` revision `987b8955fcd395c6219e39df9e64715457f69065`, 51,839,746 rows. The contract declares it already anonymized and explicitly forbids additional global deduplication.

The rebuilt streams applied the frozen native-suite overlap table, regenerated GreekMMLU decontamination queries, exact validation-text exclusions, and final zero scans. The reconstruction preserves the historical mix algorithm but does **not** claim historical document identity because the deleted historical Parquet selections were rebuilt from the pinned v2 views.

| Phase | Updates | Token slots | Active modern pool | Replay |
|---|---:|---:|---|---|
| HPLT | 1–2,261 | 0–9.483B | HPLT | foreign + Old Greek at frozen blend weights |
| OpenArchives | 2,262–3,218 | 9.483–13.497B | OpenArchives | same replay policy |
| Unseen extension | 3,219–3,694 | 13.497–15.494B | unseen OpenArchives | unseen replay; no repeated main-trajectory documents |

Training uses sequence length 4,096; global batch 1,024 sequences / 4,194,304 token slots; bf16 model precision with FP32 main gradients; AdEMAMix (`β1=.9`, `β2=.999`, `β3=.999`, `α=4`); weight decay `.1`; gradient clip `.1`; Goldfish loss `k=h=50`; cross-document attention disabled with attention, position, and loss masks reset at document boundaries. Both scales use peak LR `5.5e-5`, 400-update warmup, WSD one-minus-square-root decay to 10% of peak through update 3,218, then constant terminal LR for the extension. No checkpoint averaging was used.

## 3. Checkpoint and evaluation contract

Saved nonzero updates evaluated for both scales:

`238, 476, 714, 952, 1190, 1428, 1666, 1904, 2142, 2261, 2380, 2618, 2856, 3094, 3218, 3456, 3694`.

GreekMMLU authority:

- dataset: `dascim/GreekMMLU@6a03aa06b68beb932fb75edff3a34e50b3674649`;
- frozen clean panel: 16,159 questions (public panel: 16,632);
- score: mean continuation log probability per answer choice;
- outputs: accuracy, choice NLL, correct-answer BPB, subject and educational-level breakdowns, and paired per-question correctness.

This produces `17 × 2 × 16,159 = 549,406` checkpoint-question observations. Update 0 was not evaluated on this exact frozen trajectory panel.

## 4. Execution record

Remote stage root:

`/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14`

Trajectory root:

`/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14/evaluation/greekmmlu_trajectories/20260822T110500Z-full-clean-v1`

Frozen evaluator bundle:

`/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T125500Z-hard-h2g-greekmmlu-trajectories-8e49f095-v11`

Bundle tree SHA-256: `b7a9e1446ccf40cdad894738bf4cd20fd83c7ae065c2fdb347071c0c9523ba59`.

Frozen Python runtime:

`/iopsstor/scratch/cscs/fffoivos/python_envs/h2g_greekmmlu_eval_runtime_20260817_v2`

Scoring allocation: Slurm `3148664`, 4 Clariden nodes / 16 GH200 GPUs, state `COMPLETED`, elapsed `03:05:13`. This is approximately 49.4 allocated GPU-hours. It produced 32 new full-clean results and reused the two receipt-bound final results. Checkpoint preparation produced 31 new HF exports and reused three existing receipt-bound exports.

Remote aggregate:

`.../20260822T110500Z-full-clean-v1/trajectory_aggregate.json`

SHA-256: `49c43d10cfc28067461c538c001e3f0c1cd99a3d2b4d577be8f58c50566839b6`.

The exact checkpoint-source matrix is copied to [checkpoint_sources.tsv](presentations/greekmmlu_trajectory_20260822/evidence/checkpoint_sources.tsv). The aggregate embeds SHA-256 bindings to every result receipt, summary, and predictions file.

## 5. Conversion and evaluator parity

All 34 checkpoints pass exact tensor mapping: every source parameter is covered, every mapped parameter tensor is bit-exact, and every HF tensor is accounted for.

- 29/34 exports pass the canonical frozen-evaluator export gate;
- five exports are explicitly scoped to the common HF trajectory evaluator: 1.5B updates 238, 952, 1,428, 2,856 and 8B update 714;
- conversion-probe prediction agreement spans 98.63–100.00%;
- three trajectory-scoped conversions record incomplete optional diagnostics.

This does not invalidate the common-evaluator trajectory, but it limits interpretation of very small checkpoint differences. Exact details and receipt hashes are in [export_parity_audit.json](presentations/greekmmlu_trajectory_20260822/evidence/export_parity_audit.json).

## 6. Learning and forgetting evidence

The dense online validation trajectories contain 148 measurements per panel from update 25 through 3,675.

| Panel | 1.5B final loss | 1.5B loss above historical minimum | 8B final loss | 8B loss above historical minimum |
|---|---:|---:|---:|---:|
| HPLT | 2.3048 | +0.0241 | 1.9953 | +0.0000 |
| OpenArchives | 1.6590 | +0.0000 | 1.4509 | +0.0000 |
| Greek PhD | 1.8879 | +0.0000 | 1.6597 | +0.0000 |
| English | 2.3709 | +0.0336 | 2.0051 | +0.0447 |
| Code | 0.9188 | +0.0367 | 0.7609 | +0.0188 |

Both scales adapt strongly to OpenArchives and Greek PhD. Foreign/code forgetting is real. The extension partially reverses it. For 8B, HPLT loss rises by `+0.0180` during the OpenArchives phase and then falls by `−0.0199` in the extension, reaching a new measured minimum. For 1.5B, the corresponding changes are `+0.0260` and `−0.0019`, leaving residual HPLT forgetting.

The inherited online Old-Greek panel is omitted from retention claims because it is not a reliable independent panel. The report retains the measured data but does not use it as evidence of Old-Greek retention.

## 7. Interpretation boundary

The experiment establishes a scale-dependent response under this exact recipe. It does not identify a single causal mechanism. The models differ in parameter count, depth, width, parent pretraining, and Token-Distillation target layer. With one model and one source-order seed per scale, capacity cannot be separated from those architectural and initialization differences.

The most defensible interpretation is:

- the 8B model has sufficient capacity/optimization headroom to convert OpenArchives exposure into both lower NLL and higher GreekMMLU accuracy;
- the 1.5B model continues improving average correct-choice probability overall, but OpenArchives exposure changes answer rankings in a way that lowers argmax accuracy;
- therefore 1.5B is not a reliable proxy for selecting this 8B curriculum from GreekMMLU trajectory shape alone.

The earlier 8B predecessor used the public 16,632-question evaluator and a different reconstructed corpus realization. It is shown as a separate historical curve, never spliced numerically into the clean 16,159-question trajectory.

## 8. Failures encountered and resolved

### Evaluator runtime was incomplete

The initially selected venv exposed incomplete namespace-only `datasets`/`dill` packages. Scoring failed before producing scientific output. The run was moved to the already frozen project evaluator runtime listed above. This recurrence was added to canonical issue [#88](https://github.com/fffoivos/apertus-cscs-efficiency/issues/88#issuecomment-5380953273), with an acceptance requirement to import actual callable symbols under the exact uenv/venv tuple before allocation.

### Child `srun` consumed the checkpoint manifest

The TSV driver used a piped `while read` loop. The first child `srun` inherited stdin, consumed the remaining rows, and allowed the loop to return success after only one evaluation. Every child export/evaluation invocation was changed to read from `/dev/null`, and a regression assertion was added. The reusable canonical failure is filed as [apertus-cscs-efficiency #136](https://github.com/fffoivos/apertus-cscs-efficiency/issues/136).

### Runtime-logit parity missed threshold for five exports

The checkpoint parameters themselves are exact, but five exports missed the stricter cross-runtime logit/prediction threshold. They were not mislabeled as canonical-ready. A distinct receipt schema limits them to the matched HF trajectory evaluator. Commits `134a5ffa` and `3817baa6` implement that honest evidence boundary.

Key implementation commits:

- `a9ffa35b` — full-panel cross-scale evaluator;
- `db518bfb` — restart-safe checkpoint evaluation;
- `7f476bb9` — matched trajectory aggregation;
- `134a5ffa` — trajectory-only parity warning schema;
- `3817baa6` — incomplete parity-diagnostic accounting;
- `8e49f095` — preserve the source stream across child Slurm steps.

## 9. Local outputs

- Final presentation: [GREEKMMLU_H2G_CROSS_SCALE_TRAJECTORIES_20260822.html](presentations/greekmmlu_trajectory_20260822/GREEKMMLU_H2G_CROSS_SCALE_TRAJECTORIES_20260822.html)
- Presentation builder: [build_report.py](presentations/greekmmlu_trajectory_20260822/build_report.py)
- Analysis generator: [analyze_report_evidence.py](presentations/greekmmlu_trajectory_20260822/analyze_report_evidence.py)
- Analysis summary: [analysis_summary.json](presentations/greekmmlu_trajectory_20260822/evidence/analysis_summary.json)
- Frozen trajectory aggregate: [trajectory_aggregate.json](presentations/greekmmlu_trajectory_20260822/evidence/trajectory_aggregate.json)
- Export audit: [export_parity_audit.json](presentations/greekmmlu_trajectory_20260822/evidence/export_parity_audit.json)
- QA verifier: [verify_report.py](presentations/greekmmlu_trajectory_20260822/verify_report.py)
- QA receipt: [qa_receipt.json](presentations/greekmmlu_trajectory_20260822/qa/qa_receipt.json)

## 10. Rebuild and verify

From the worktree root:

```bash
python3 subprojects/08_targeted_8b_cpt_experiments/presentations/greekmmlu_trajectory_20260822/analyze_report_evidence.py
python3 subprojects/08_targeted_8b_cpt_experiments/presentations/greekmmlu_trajectory_20260822/build_report.py
```

Render both layouts:

```bash
cd subprojects/08_targeted_8b_cpt_experiments/presentations/greekmmlu_trajectory_20260822
python3 -m http.server 8877 --bind 127.0.0.1
playwright screenshot --browser chromium --channel chrome --viewport-size '1440,1000' --full-page --wait-for-timeout 1500 http://127.0.0.1:8877/GREEKMMLU_H2G_CROSS_SCALE_TRAJECTORIES_20260822.html qa/desktop-1440.png
playwright screenshot --browser chromium --channel chrome --viewport-size '430,932' --full-page --wait-for-timeout 1500 http://127.0.0.1:8877/GREEKMMLU_H2G_CROSS_SCALE_TRAJECTORIES_20260822.html qa/narrow-430.png
python3 verify_report.py --visual-inspection-passed
```

Open the final artifact visibly in Firefox:

```bash
/Users/foivoskarounos-zamparloukos/.codex/skills/academic-html-report/scripts/open_final_in_firefox.sh \
  /Users/foivoskarounos-zamparloukos/Projects/.codex-worktrees/h2g-greekmmlu-trajectories/subprojects/08_targeted_8b_cpt_experiments/presentations/greekmmlu_trajectory_20260822/GREEKMMLU_H2G_CROSS_SCALE_TRAJECTORIES_20260822.html
```

## 11. Remaining scientific work

No required work remains for this matched GreekMMLU trajectory experiment. Optional follow-ons are:

1. evaluate update 0 on the same 16,159-question panel;
2. repeat one or both scales with another data-order seed;
3. evaluate selected peak and final checkpoints on the frozen native-Greek suite;
4. investigate why 1.5B NLL improves while argmax accuracy falls, using per-question transition and calibration analysis;
5. repeat only if a concrete new hypothesis justifies the GPU cost.

These are new experiments or analyses, not blockers for this handoff.
