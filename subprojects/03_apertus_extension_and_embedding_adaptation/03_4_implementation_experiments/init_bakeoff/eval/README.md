# eval — baselines, tokenizer-fair metrics, and the Greek headline

> **In one line:** the measurement half of the bakeoff — a base-model baseline, a cross-tokenizer-fair loss metric, seven new-token diagnostics, a per-checkpoint eval bridge, and finally a purpose-built native-Greek suite that reversed the conclusion the rest of the stack had produced.
> **Period:** 2026-05-21 (`11b5ba00`) → 2026-08-07 (`5b6dd260`, registry extensions for later subprojects). **Status:** completed for the bakeoff.
> **Reads:** [`../bakeoff_training/`](../bakeoff_training/README.md). **Feeds:** [`../../../CPT_MASTER_20260526.md`](../../../CPT_MASTER_20260526.md) and subprojects 04–09.

## Why this existed

Two arms use a 131,072-token vocabulary and two use 148,480, so the training loss cannot rank them: raw Megatron `lm loss` is per-target-token cross-entropy and both the softmax size and the tokens-per-text change between families. Everything in this directory exists to produce comparisons that survive that: byte-normalised loss, accuracy-based downstream tasks, and diagnostics that ask specifically whether the *new* rows are being used.

## History

| Date | What happened | Result | Evidence |
|---|---|---|---|
| 2026-05-21 | V4-HF baseline on unmodified Apertus (job `2334245`, 1h11m54s) | Valid but **partial** — the task list omitted `global_mmlu`. `mmlu = 0.5923` matches the Apertus tech report's ≈0.59, confirming the harness is wired correctly. Corrected rerun `2335100` completed in 1h10m29s | [`v4_baseline_20260521/V4_RESULTS.md`](v4_baseline_20260521/V4_RESULTS.md), [`v4_baseline_corrected_20260521/`](v4_baseline_corrected_20260521/README.md) |
| 2026-05-21 | V4 post-conversion eval, the empirical R17 test | Round-tripping Apertus through an **unpatched** HF→Megatron→HF conversion collapses it to chance: `arc_easy` 0.8363 → 0.2614, `hellaswag` 0.7884 → 0.2675, `mmlu` 0.5923 → 0.2295. Took three attempts — `2335101` died on a datasets/filelock `No locks available` (fixed with per-job cache isolation), `2335196` timed out at 4 h, then split into `2338020` retention + `2338021` Greek at 8 h | [`V4_BENCHMARK_COMPARISON.md`](V4_BENCHMARK_COMPARISON.md), [`v4_postconv_retry_20260521/`](v4_postconv_retry_20260521/README.md) |
| 2026-05-21 | Tokenizer-fair metrics + new-token diagnostics implemented (v0.7 §5.1 and §5.3) | `compute_tokenizer_fair_metrics.py` emits BPB, NLL/char, NLL/word, tokens/word, chars/token, STRR; `compute_new_token_diagnostics.py` runs all seven D-diagnostics in one pass | `d5bc6c06`, `28b71f24` |
| 2026-05-22 | Held-out slice built and accepted | 500 Greek documents, doc-id uniqueness enforced | `c65e6d9a`, `7dd139d9`, `9eaf2733` |
| 2026-05-22 | Checkpoint→HF→eval bridge and watcher built, then hardened repeatedly | Six consecutive fixes (uenv export, converter plugin path, Megatron group init, distributed shim, DP world-size override, partition) before conversion worked; then packed multi-arm eval jobs with guarded submission and per-job cache isolation | `e35deaea` … `06c2430e`, `f91f2d57`, `4d50a4fc` |
| 2026-05-22 → 05-26 | Per-checkpoint evals at iters 130 / 260 / 325 / 390 / 455 / 476 / 585 / 715 / 834 / 1013 / 1192 | The trajectory data set; digests copied locally as they landed | [`live_summaries/`](live_summaries/README.md) |
| 2026-05-24 → 05-26 | Trajectory analysis, three generations | 2 B analysis → 3.5 B continuation → **5 B canonical result**. Each supersedes the last | [`trajectory_analysis_20260524/README.md`](trajectory_analysis_20260524/README.md) |
| 2026-05-25 → 05-26 | Loss-measurement policy written and propagated | Raw `lm loss` demoted to health telemetry; heldout **BPB** made the selection anchor; historical `BPC` / `bpc_bits_per_byte` declared a bits-per-**byte** alias, not bits per character | [`LOSS_MEASUREMENT_POLICY.md`](LOSS_MEASUREMENT_POLICY.md), `47f42dc2`, `504e5d38` |
| 2026-05-26 | **Native-Greek suite** built and run on all 11 checkpoints | Headline restricted to vetted native datasets (GreekMMLU, ILSP Medical MCQA, ILSP ASEP), MT-derived tasks demoted to diagnostics. Six jobs `2396931`–`2396933` and `2396991`–`2396993` completed; a first `greek-nlp` attempt was invalidated by an upstream GEC temp-directory race and rerun | [`NATIVE_GREEK_EVAL_SUITE_20260526.md`](NATIVE_GREEK_EVAL_SUITE_20260526.md), [`NATIVE_GREEK_SUITE_RESULTS_20260526.md`](NATIVE_GREEK_SUITE_RESULTS_20260526.md), [`GREEK_NLP_BENCHMARK_20260526.md`](GREEK_NLP_BENCHMARK_20260526.md) |
| 2026-06-16 | Peer-model GreekMMLU baseline prepared (Krikri-8B-Base, Gemma-2-9B, Qwen3.5-9B-Base, Gemma-3-12B) | Handoff doc marked "ready to submit"; **no results doc exists in this tree**, so treat it as planned-not-completed here | [`PEER_GREEKMMLU_BASELINE.md`](PEER_GREEKMMLU_BASELINE.md), `42c57a4d` |
| 2026-07-12 → 2026-08-07 | Native-MCQ runner and benchmark registry extended for later subprojects | `01cba0ee`, `ba80bb0c`, `5b6dd260` |

## Outcome

- **The 5 B scoreboard** (Vanilla / TD at iter 1192): Greek no-MT 0.4076 / **0.4204**, EN retention 0.6799 / **0.6903**, multilingual 0.4936 / **0.4976**, BPB **0.4602** / 0.4872. TD's Greek lead is carried by `xquad_el` (+7.57 pp); without it Vanilla is narrowly ahead on the remaining four Greek tasks.
- **The native-Greek reversal:** native MCQ aggregate — Apertus-Base **0.4817**, Vanilla-3.5B 0.4370, Vanilla-5B 0.4305, TD-5B 0.4109, ReTok-3.5B 0.3770, Centroid-2B 0.2824. The recorded decision: *do not call TokenDistil the Greek winner*; Vanilla is the safer Greek-native continued arm, and base Apertus is a ceiling none of the arms recovered.
- **New-token rows plateau by 2 B.** TD's D2 (probability mass on new-token targets) is flat at ~0.342 from iter 476 to 1192; D5 (greedy new-token utilization) is noisy in 0.21–0.30 with no trend and a small sample.
- **Explicit gaps left open:** no per-task confidence intervals on the aggregates (`*_stderr` fields exist but were never propagated); no V4 run-to-run variance baseline, so "X beats Y by N pp" has no noise floor; BPB truncation bias unmeasured (Vanilla truncates 29.2 % of heldout docs vs TD's 24.8 %); OYXOY, GreekBarBench and several other native tasks cached but never scored; gated ILSP datasets unavailable.

## Where things are

| What | Where |
|---|---|
| Canonical 4-arm result | [`trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md`](trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md) |
| Greek-headline correction | [`NATIVE_GREEK_SUITE_RESULTS_20260526.md`](NATIVE_GREEK_SUITE_RESULTS_20260526.md) |
| The loss rule | [`LOSS_MEASUREMENT_POLICY.md`](LOSS_MEASUREMENT_POLICY.md) |
| Task lists, cadence, statistics | [`EVAL_RECIPE.md`](EVAL_RECIPE.md) |
| Metric computation | [`compute_tokenizer_fair_metrics.py`](compute_tokenizer_fair_metrics.py), [`compute_new_token_diagnostics.py`](compute_new_token_diagnostics.py), [`compute_bootstrap_cis.py`](compute_bootstrap_cis.py) |
| Native-Greek runner + registry | [`run_native_greek_mcq_eval.py`](run_native_greek_mcq_eval.py), [`native_greek_benchmark_registry.json`](native_greek_benchmark_registry.json), [`run_greek_nlp_benchmark_hf.py`](run_greek_nlp_benchmark_hf.py) |
| Checkpoint→eval automation | [`convert_bakeoff_checkpoint_to_hf.sbatch`](convert_bakeoff_checkpoint_to_hf.sbatch), `submit_*`/`watch_*` scripts |
| Held-out slice | [`build_cpt_heldout_jsonl.py`](build_cpt_heldout_jsonl.py); data at Clariden `/iopsstor/.../cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl` |
| Full eval outputs | Clariden `/capstor/scratch/cscs/fffoivos/runs/eval/` (~480 GB) |

## Working documents

- **Superseded results:** `trajectory_analysis_20260524/BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md` (2 B only) and `CONTINUATION_3P5B_RESULTS_20260525.md` (3.5 B) — accurate at their own token scope, but not the headline. `BAKEOFF_FINAL_RESULTS_20260526.md` itself carries a post-native-suite banner correcting its Greek conclusion.
- **Baseline run dirs:** [`v4_baseline_20260521/`](v4_baseline_20260521/), [`v4_baseline_corrected_20260521/`](v4_baseline_corrected_20260521/), [`v4_postconv_retry_20260521/`](v4_postconv_retry_20260521/) — use the *corrected* baseline; the first is missing `global_mmlu`.
- **Live snapshots:** [`live_summaries/`](live_summaries/README.md) — ~20 per-checkpoint digests copied off Clariden while runs were in flight. Their Markdown was relabelled BPC → BPB; the underlying JSONs still carry `bpc_bits_per_byte`.
- **Intrinsic pilots:** `td_pilot_intrinsics_20260523T091637Z/`, `td_full25_intrinsics_20260523T124000Z/` — TD layer-pilot and full-token intrinsic summaries.
- **Plans:** [`NATIVE_GREEK_EVAL_SUITE_20260526.md`](NATIVE_GREEK_EVAL_SUITE_20260526.md) and [`GREEK_NLP_BENCHMARK_20260526.md`](GREEK_NLP_BENCHMARK_20260526.md) (executed), [`PEER_GREEKMMLU_BASELINE.md`](PEER_GREEKMMLU_BASELINE.md) (handoff, no results recorded here).
