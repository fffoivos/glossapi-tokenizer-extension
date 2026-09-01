# 09 — Full 8B CPT results analysis

> **In one line:** the results authority for the completed sanitized Apertus-8B Greek CPT run (76.685 B active tokens) — it showed that GreekMMLU peaks mid-run at update 9,536 (56.81 %) while Greek language-model loss keeps improving, then closed the evidence by scoring all 19 saved checkpoints, measuring retention and checkpoint instability, and building a five-checkpoint weight average that matches the peak while forgetting less.
> **Period:** 2026-08-12 → 2026-08-23 (committed work runs `2a413042` … `ae2acd30`, ending 08-19; the 08-19 → 08-23 results existed only as working-tree files until they were recovered on 2026-09-01 in `2aec4a66`). **Status:** completed.
> **Came from / led to:** [`07_full_8b_cpt`](../07_full_8b_cpt/) (the run itself) + [`06_dataset_scheduling_experiments`](../06_dataset_scheduling_experiments/) (the five 0.5 B arms) → this → [`10_early_cooldown_causal_experiment`](../10_early_cooldown_causal_experiment/) (causal test of the 9,536 peak) and [`08_targeted_8b_cpt_experiments`](../08_targeted_8b_cpt_experiments/) (follow-ups).

## Why this existed

The 8B CPT run finished on 2026-08-11 (`completion.status = "completed"`, `2026-08-11T07:49:24Z`, in [`presentations/FULL8_RESULTS.data.json`](presentations/FULL8_RESULTS.data.json)), leaving a large working directory of receipts, progress snapshots and exploratory reports in subproject 07. This subproject was created to answer three questions from a small package that can be verified byte-for-byte:

1. **Adaptation and retention** — what did the model learn and what did it forget over 76.685 B active tokens?
2. **Capability timing** — when did native-Greek benchmark performance peak, and did that agree with held-out loss?
3. **Proxy validity** — did the five 0.5 B data-order arms predict the 8 B trajectory?

The ownership boundary was deliberate: 07 kept the recipe, bundles, Slurm orchestration and raw receipts; 09 kept post-hoc conclusions, compact analysis code and the canonical presentations. Synthetic drift simulations, alternative displacement statistics and progress snapshots were intentionally *not* copied, on the stated grounds that they did not change any of the three decisions. Once the mid-run GreekMMLU peak turned out to be real, three further questions were opened — is the peak an artefact of checkpoint *instability* (09_1), what does the trajectory cost on non-Greek downstream tasks (`evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md`), and did the added vocabulary adapt (09_3).

## History

### 2026-08-12 — the package is cut (`2a413042`)

Eleven artifacts were copied out of the completed run root and pinned by SHA-256 in [`evidence/ARTIFACT_MANIFEST.json`](evidence/ARTIFACT_MANIFEST.json): three canonical HTML reports with their `.data.json` payloads and five analysis scripts. [`verify_bundle.py`](verify_bundle.py) fails closed if any of them drifts; [`test_results_contract.py`](test_results_contract.py) is the regression check on the package itself. The three prose documents written the same day are still the conclusions: [`RESULTS.md`](RESULTS.md), [`DATA_AND_LIMITATIONS.md`](DATA_AND_LIMITATIONS.md) and the evaluation-expansion plan [`NATIVE_GREEK_BENCHMARKS.md`](NATIVE_GREEK_BENCHMARKS.md).

`RESULTS.md` recorded the headline immediately: GreekMMLU 35.78 % at initialization → **56.81 % at update 9,536 (39.997 B token slots)** → 54.85 % terminal, on the frozen decontaminated 16,159-question subset, while source-conditioned Greek BPB kept improving to the end. `DATA_AND_LIMITATIONS.md` recorded the awkward part: the executed dataset had a *second* global exact-content deduplication after PII masking (2,378,595 duplicate + 8,081 validation-collision documents removed from 97,136,622), which the owner had not asked for — so the run is internally valid but is not a data-identical replication of the 0.5 B study, and the second dedup must not become release policy.

### 2026-08-12 — building an evaluation that would survive scrutiny (about 20 commits in one day)

`NATIVE_GREEK_BENCHMARKS.md` argued that one exam benchmark cannot decide whether 9,536 is a real capability peak, and proposed a native-Greek panel (native = prompts authored in Greek, not machine-translated). The first screen was built under [`evaluation/`](evaluation/) as a frozen three-checkpoint matrix (initialization / 9,536 / terminal 18,284) over DemosQA, Medical MCQA, ASEP MCQA, GPCR and four OYXOY task views — 83,970 scored examples per checkpoint.

Most of that day's commits are the harness refusing shortcuts, and they read as decisions rather than churn:

| Commit | Attempted speed-up | Outcome |
| --- | --- | --- |
| `be1e1be4`, `635532a1` | BF16 scoring instead of FP32 | **rejected** by a frozen parity gate — it changed answer rankings |
| `688aa43f`, `f118b5b1` | suffix-only logits instead of the legacy full-logit scorer | gated against the legacy scorer on a shared sample |
| `f89d1985`, `05d1a724` | FP32 candidate batch 4 instead of batch 1 | **rejected** — all sampled predictions were preserved but the predeclared raw-score tolerance was missed; production kept batch 1 |
| `606507fe` → `74219b37` | fitting the exact FP32 matrix into Clariden `debug` | resolved by sharding only: 63 independent shards over chained 22-minute four-node jobs inside the 90 node-minute QoS cap |
| `e63d1b79`, `981b7c28` | — | bundle freezer + fail-closed verifier so the executing tree must match its receipt |

The stated principle held: speed came only from independent-example and independent-checkpoint parallelism, never from changing arithmetic inside a scored example.

### 2026-08-12 — contamination: measured, not assumed (`efd9eb8a`, `4ad2f763`, `7ca429d8`)

A reusable audit scanned all 431 published Parquet shards and all 51,839,746 rows of the training corpus and emitted 18,166,197 trace rows. [`evaluation/CONTAMINATION_DROP_DECISION_20260812.md`](evaluation/CONTAMINATION_DROP_DECISION_20260812.md) froze the rule: exclude an evaluation item only when a *label-bearing second surface* co-occurs with the question surface in the same training document (question + correct answer, premise + hypothesis, usage + definition…). Question-only hits were published as candidates but not excluded.

Result: **10,076 of 83,970 scored examples excluded, 73,894 retained** (10,048 strict units out of 80,446 audited). The damage is concentrated in OYXOY's lexical panels because their source dictionary (`glossAPI/modern-greek-dictionary`) is in the corpus — 4,614 of 58,831 WiC items, 4,399 of 14,398 WSD items, 973 of 3,015 metaphor items. The decision was to report **both** full and filtered scores, never to edit benchmark source data, and to leave the exclusion list as a reporting filter with published IDs and evidence.

### 2026-08-12 — the three-checkpoint result (`081510bb`, `f4855dfa`)

[`NATIVE_GREEK_3CP_RESULTS_20260812.md`](NATIVE_GREEK_3CP_RESULTS_20260812.md) confirmed the peak but complicated it. The 39.997 B checkpoint is best on GreekMMLU, Medical MCQA, OYXOY NLI and WSD; the terminal checkpoint is best on GPCR and effectively tied on ASEP and DemosQA. Strict filtering did not reverse the ordering. OYXOY metaphor and WiC "win" at initialization only through class imbalance — their balanced accuracies sit near chance and their choice NLL worsens, so they were labelled label-bias diagnostics, not evidence for the base model. No macro-average was declared, on the explicit grounds that the tasks differ too much in size, balance and contamination rate.

Greek Protipa Exams was contract-pinned but never scored: its authenticated download returned HTTP 403 on 2026-08-12 pending a manual Hugging Face gate. It stayed unscored for the whole subproject.

### 2026-08-13 → 08-14 — the 0.5 B proxy is tested and does not hold up (`1cd667cf` … `2a7eb9d8`)

The same frozen examples, FP32 legacy scorer, prompts and exclusions were rebound to three token-aligned D0 0.5 B checkpoints (init / update 18,944 = 39.728 B / final 38,496 = 80.732 B). Job `3079741` preserved 39 completed shards at its planned wall-time exit and resume job `3079936` finished the remaining 24, aggregated all 63 and completed in 13:06 — the resume path itself took four commits to get right (`b3f26426`, `cf619189`, `8986c606`, `8dc2e769`).

[`D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md`](D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md): strict-filtered choice-NLL **direction agrees on 6/9 benchmarks from init to ~40 B, but only 3/9 from ~40 B to the endpoint, and the exact best-checkpoint identity agrees on 2/9**. Early capability emergence partially transfers; late checkpoint timing does not. `RESULTS.md` was updated the same day to say so, and to note the confound list that forbids a causal scale claim (tied vs untied embeddings, layer-7 vs layer-11 TD, peak LR 1.5e-4 vs 5.5e-5, RoPE scaling 1 vs 8, and 80.730 B pre-sanitation vs 76.685 B post-dedup tokens).

### 2026-08-17 — zooming in on the peak ([`09_1_downstream_task_instability/`](09_1_downstream_task_instability/))

Two things started the same day. The instability metrics of Nishida, Isonuma & Oda ([arXiv:2510.04848](https://arxiv.org/abs/2510.04848)) were implemented against the stored prediction artifacts, and the four saved checkpoints bracketing 9,536 (7,152 / 8,344 / 10,728 / 11,920) were scored on the clean 73,894-example subset. [`09_1_downstream_task_instability/evaluation/PEAK_WINDOW_CLEAN_RESULTS_20260817.md`](09_1_downstream_task_instability/evaluation/PEAK_WINDOW_CLEAN_RESULTS_20260817.md) reached the blunt conclusion: **there is no universal best checkpoint**, and the GreekMMLU-selected 40 B checkpoint is not the optimum for most of these tasks.

### 2026-08-17 → 08-19 — closing the matrix and the release ([`09_2_checkpoint_trajectory_release/`](09_2_checkpoint_trajectory_release/))

The remaining twelve saved 8 B exports were scored so that all 19 checkpoints sit on the same strict subset, and the Hugging Face release was staged. Four commits (`d411cb01`, `9b17b6c5`, `73f527d9`, `b18930bf`) were spent on Clariden allocation shapes rather than science; a nested `srun` attach step exposed only one CPU to the segment wrapper, which is why [`09_2_checkpoint_trajectory_release/evaluation/workaround_resume_remaining12_normal.py`](09_2_checkpoint_trajectory_release/evaluation/workaround_resume_remaining12_normal.py) exists. On 2026-08-19 (`ae2acd30`) the complete 19-point report was built (252/252 shards verified) and the ordered checkpoint metadata was published to the **private** model repo. That is the last committed change in this subproject.

### 2026-08-19 → 08-20 — instability, retention, and a weight average that wins (uncommitted at the time)

Three results landed in two days and none of them was committed; they were recovered on 2026-09-01 and all three carry an explicit "**not independently reviewed**" status line.

- [`09_1_downstream_task_instability/evaluation/OFFLINE_ENSEMBLE_RESULTS_20260819.md`](09_1_downstream_task_instability/evaluation/OFFLINE_ENSEMBLE_RESULTS_20260819.md) — zero-inference checkpoint ensembles over the stored predictions, gate-verified to 1e-9 against the 19-checkpoint payload. Rolling five-checkpoint ensembles cut example-level MTV **2–3× on every benchmark**, and mean accuracy *rose* exactly on the label-bias-unstable tasks (WiC +10.6, NLI +3.8, metaphor +6.0 pp) while staying within ±0.3 pp elsewhere. It also predicted that a peak-window average is the high-value arm and that a cooldown-window average would collapse on NLI.
- [`evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md`](evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md) — the Apertus Table-14 suite over five trajectory points. The **10 B checkpoint (iter 2,384) is the retention optimum**, matching or beating base Apertus-8B on 5 of 9 tasks; from 10 B to terminal MMLU falls 4.0 pp and ARC-challenge 5.0 pp. Greek is the only language slice that rises. This confirmed the replay-BPB forgetting behaviourally and added a third, conflicting selection axis.
- [`09_1_downstream_task_instability/CHECKPOINT_AVERAGE_RESULTS_20260819.md`](09_1_downstream_task_instability/CHECKPOINT_AVERAGE_RESULTS_20260819.md) — two uniform five-checkpoint weight averages were built and evaluated. `avg_peak5` (iters 7,152–11,920) ties the run's GreekMMLU peak (56.78 % vs 56.81 %, stderr ≈ 0.39 pp), sets an all-time ASEP best (56.19 %), and beats the 40 B single on **all nine** retention tasks (macro 64.58 vs 63.68; terminal 62.95). `avg_cooldown5` behaved exactly as forecast — terminal-parity stabilization with a real NLI collapse (43.34 %). A correction is recorded in the same file on 2026-08-20: the terminal column had borrowed 50 B values, and against the true terminal numbers `avg_cooldown5` beats terminal broadly (WiC +28.0). On 2026-08-20 both averages were published as private branches `18-avg-uniform5-tokens30B-50B` and `19-avg-uniform5-tokens61B-77B`, and the repository `default_revision` moved from `08-step9536-tokens40B` to the peak-window average.

### 2026-08-23 — was it the vocabulary? ([`09_3_added_token_adaptation_audit/`](09_3_added_token_adaptation_audit/))

The last question was whether the 17,920 added vocabulary entries explain the peak. A held-out, paired three-test audit (merged-vs-split likelihood, hidden-state agreement at the TD layer, echo probe) over 2,348,881 scored occurrences per checkpoint answered **yes they adapted, no it does not explain the peak**: from update 9,536 to the terminal, 79.8 % of the 17,171 measurable modern tokens *improve*. See [`09_3_added_token_adaptation_audit/RESULTS.md`](09_3_added_token_adaptation_audit/RESULTS.md).

## Outcome

- **Adaptation succeeded; forgetting is real but bounded.** All six Greek learning panels and all 13 exact per-document panels improved from initialization to the terminal checkpoint — e.g. HPLT 1.3762 → 0.3426 BPB, historical/polytonic 2.0678 → 0.5072 ([`presentations/FULL8_RESULTS.data.json`](presentations/FULL8_RESULTS.data.json), `per_document_validation`). English, code, maths, German, Russian and Chinese ended *above* their own best earlier BPB but far below initialization. Training completed with 0 skipped and 0 non-finite updates.
- **GreekMMLU peaks at update 9,536 = 56.81 %** (init 35.78 %, terminal 54.85 %; choice NLL 1.4586 → 1.0740 → 1.1221), confirmed independently on all 19 checkpoints. The benchmark reached 53.57 % by update 400, so most of the gain is very early and the rest is fluctuation.
- **The winning checkpoint depends on the capability.** Best strict-filtered choice NLL falls at update 1,192 for OYXOY WiC, 2,384 for ASEP and Medical MCQA, 3,576 for DemosQA and OYXOY metaphor, 8,344 for OYXOY NLI, 9,536 for GreekMMLU and OYXOY WSD, and 11,920 for GPCR (computed from [`presentations/FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.data.json`](presentations/FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.data.json)). Non-Greek retention adds a fourth optimum at 10 B. No macro-average was ever declared.
- **Much of the checkpoint-to-checkpoint movement is instability, not capability** — five-checkpoint aggregation removes two thirds of it, and a *weight* average of the peak window is the strongest all-round artifact the run produced: peak-level GreekMMLU with materially better retention.
- **The 0.5 B screen was not validated as a predictor.** 0.5 B D0 reached 42.25 % GreekMMLU at its endpoint against 54.85 % for 8 B D0; late-trajectory direction agreement was 3/9 and best-checkpoint agreement 2/9.
- **The vocabulary extension is exonerated.** Added-token behaviour improves monotonically to the terminal checkpoint, so the GreekMMLU regression after 9,536 must not be attributed to it.
- **Decisions recorded** (`RESULTS.md`): preserve the update-9,536 checkpoint; never select a checkpoint from GreekMMLU accuracy alone; keep source-conditioned Greek and replay loss as continuous diagnostics; treat the second post-mask deduplication as a run-specific difference and not a default. The de-facto selection decision of 2026-08-20 was to make `avg_peak5` the repository default revision.
- **Carried forward:** the mid-run peak became the hypothesis under test in [`10_early_cooldown_causal_experiment`](../10_early_cooldown_causal_experiment/), which branches from that exact update-9,536 checkpoint. The frozen scorer, prompts and strict exclusion subset were reused unchanged by 09_1, 09_2 and 10.
- **Left open:** the three post-08-19 result sets were never independently reviewed and were never committed by their author; the cooldown's apparent benefit on every per-document panel still has no matched no-cooldown control; Greek Protipa Exams was never scored (access gate); the Tier-2 generation/critic panel in `NATIVE_GREEK_BENCHMARKS.md` was designed and never run; the lm-eval base-model anchor was inherited from a since-destroyed May install and was not re-anchored; greedy-soup / interpolation arms and the layer-30 divergence of §3 in `09_3/RESULTS.md` were both named as next threads; the model repository stayed private (`status: "complete_private_metadata_release"`).

## Sub-subprojects

| Dir | Role | Period | Status | Result |
| --- | --- | --- | --- | --- |
| [`09_1_downstream_task_instability/`](09_1_downstream_task_instability/) | instability metrics, peak-window scoring, ensembles and weight averaging | 2026-08-17 → 08-20 | completed, unreviewed | no universal best checkpoint; aggregation cuts instability 2–3×; `avg_peak5` ties the peak and forgets less |
| [`09_2_checkpoint_trajectory_release/`](09_2_checkpoint_trajectory_release/) | score the remaining 12 checkpoints; stage the HF release | 2026-08-17 → 08-20 | completed (private) | 19/19 checkpoints scored, 252/252 shards; 18 ordered private branches + 2 averaged branches + 2 frozen datasets |
| [`09_3_added_token_adaptation_audit/`](09_3_added_token_adaptation_audit/) | did the 17,920 added tokens adapt, and do they explain the peak? | 2026-08-23 | completed, unreviewed | adapted monotonically; 79.8 % improve after the peak → not the cause |
| [`analysis/`](analysis/) | the analysis scripts copied from the completed run | 2026-08-12 → 08-14 | completed | learning/forgetting, per-document endpoints, answer drift, source exposure, 0.5 B-vs-8 B comparison |
| [`evaluation/`](evaluation/) | the frozen native-Greek scorer, its parity gates, the contamination decision, the retention suite | 2026-08-12 → 08-19 | completed | 83,970 examples/checkpoint; FP32 batch-1 legacy scorer; 10,076 strict exclusions; retention optimum at 10 B |
| [`presentations/`](presentations/) | the canonical self-contained HTML reports | 2026-08-12 → 08-19 | completed | five reports, each with its exact `.data.json` payload |
| [`evidence/`](evidence/) | hash manifest, strict-filtered metric CSVs, raw retention logs | 2026-08-12 → 08-19 | completed | 11 pinned artifacts; 3 × 8 B and 3 × 0.5 B filtered metric sets; 5 lm-eval result files |

## Where things are

| What | Path |
| --- | --- |
| The three conclusions | [`RESULTS.md`](RESULTS.md) |
| Dataset boundaries and the unrequested second dedup | [`DATA_AND_LIMITATIONS.md`](DATA_AND_LIMITATIONS.md) |
| Three-checkpoint native-Greek results (full and filtered) | [`NATIVE_GREEK_3CP_RESULTS_20260812.md`](NATIVE_GREEK_3CP_RESULTS_20260812.md) |
| 0.5 B-vs-8 B replication verdict | [`D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md`](D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md) (+ adjacent `.data.json`) |
| Contamination exclusion authority | [`evaluation/CONTAMINATION_DROP_DECISION_20260812.md`](evaluation/CONTAMINATION_DROP_DECISION_20260812.md) |
| Non-Greek retention across the trajectory | [`evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md`](evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md) |
| Complete 19-checkpoint report | [`presentations/FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.html`](presentations/FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.html) |
| Checkpoint-average and ensemble results | [`09_1_downstream_task_instability/CHECKPOINT_AVERAGE_RESULTS_20260819.md`](09_1_downstream_task_instability/CHECKPOINT_AVERAGE_RESULTS_20260819.md) |
| Package integrity check | `python3 verify_bundle.py` ([`verify_bundle.py`](verify_bundle.py), manifest in [`evidence/ARTIFACT_MANIFEST.json`](evidence/ARTIFACT_MANIFEST.json)) |
| Evaluation contract (checkpoints, prompts, dtype, batch) | [`evaluation/native_greek_3cp_contract.json`](evaluation/native_greek_3cp_contract.json) |
| Source run root, bundles, raw receipts | subproject [`07_full_8b_cpt`](../07_full_8b_cpt/) |

## Working documents

- **Plan, never fully executed:** [`NATIVE_GREEK_BENCHMARKS.md`](NATIVE_GREEK_BENCHMARKS.md) — the staged native-Greek expansion. Tier 1 (DemosQA, Medical, ASEP, GPCR, OYXOY) ran; Protipa was blocked by access; the Tier-2 generation/critic panel (GreekBarBench, Plutus-ben, GreekSum, Greek Civics QA, GEAR, Greek LLM Arena) was designed but never run. Historical.
- **Superseded reporting:** the three-checkpoint tables in `NATIVE_GREEK_3CP_RESULTS_20260812.md` and the report [`presentations/NATIVE_GREEK_3CP_BENCHMARKS.html`](presentations/NATIVE_GREEK_3CP_BENCHMARKS.html) are subsumed by the 19-checkpoint matrix of 2026-08-19, which uses the same strict subset. They are kept because they carry the full-vs-filtered comparison the later report drops.
- **Recovered, unreviewed:** everything dated 2026-08-19 or later under `09_1_*`, `09_3_*`, `evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md` and `evidence/retention_lm_eval_20260819/` was uncommitted working-tree material until `2aec4a66` (2026-09-01). Treat its numbers as first-pass results with named provenance, not as reviewed conclusions.
- **Directory READMEs** under each sub-subproject describe what that part did; all are histories of completed work.
