# Native Greek evaluation suite - 2026-05-26

Purpose: make the Greek decision score native-first and auditable. The
headline Greek aggregate must include only vetted native/authentic Greek
benchmarks with documented provenance, access/license status, data format,
labels, and scoring method.

Machine-translated Greek benchmarks remain diagnostics only. Non-Greek
retention benchmarks remain regression checks only.

## Checkpoint Scope

Evaluate Apertus-Base plus every HF-format experiment checkpoint in
`release/apertus-tokenizer-extension/manifest.json`:

| Checkpoint | HF path on Clariden |
|---|---|
| Apertus-Base | `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509` |
| TokenDistil-Init | `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/hf_roundtrip` |
| TokenDistil-2B | `/capstor/scratch/cscs/fffoivos/runs/eval/td_full25_layer11_2b_20260523T165038Z/iter_0000476_hf` |
| TokenDistil-3.5B | `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf` |
| TokenDistil-5B | `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_td_layer11/iter_0001192_hf` |
| Vanilla-2B | `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_vanilla/iter_0000476_hf` |
| Vanilla-3.5B | `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_vanilla/iter_0000834_hf` |
| Vanilla-5B | `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_vanilla/iter_0001192_hf` |
| ReTok-2B | `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_retok/iter_0000476_hf` |
| ReTok-3.5B | `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_retok/iter_0000834_hf` |
| Centroid-2B | `/capstor/scratch/cscs/fffoivos/runs/eval/bakeoff_1node_chain_20260522_005620_centroid/iter_0000476_hf` |

## Vetted Native Benchmarks

Registry: `native_greek_benchmark_registry.json`.

Local cache manifest:
`artifacts/native_greek_eval_cache/native_greek_benchmark_cache_manifest.json`
generated at `2026-05-26T13:56:13Z`.

| Benchmark | Status | Rows | Scoring status | Headline role |
|---|---:|---:|---|---|
| `greek-nlp/benchmark` | cached + HF runner smoke-passed | upstream task-dependent | generation/tagging metrics from upstream suite | native supporting suite; keep `machine_translation` separate |
| `dascim/GreekMMLU` | cached | 16,632 | deterministic MCQ log-prob runner | native headline MCQ |
| `ilsp/medical_mcqa_greek` | cached | 432 | deterministic MCQ log-prob runner | native medical-domain MCQ |
| `ilsp/mcqa_greek_asep` | cached | 1,200 | deterministic MCQ log-prob runner | native local-exam MCQ |
| `TheFinAI/plutus-QA` | cached | 225 | deterministic MCQ log-prob runner | native finance-domain MCQ, report separately from general Greek |
| OYXOY v1.0 | cached from Zenodo | file-based | adapter pending | native linguistic suite once adapter lands |
| `AUEB-NLP/greek-bar-bench` | cached | 284 | judge/free-text scoring pending | native legal free-text diagnostic only for base CPT |
| `ilsp/greek_civics_qa` | cached | 407 | open-QA scoring pending | native diagnostic until scoring is fixed |
| `ilsp/greek_lyceum_mathematics` | cached | 465 | exact/symbolic scoring pending | native diagnostic until scoring is fixed |

## Unavailable Or Blocked

| Benchmark | Blocker |
|---|---|
| `ilsp/greek-protipa-exams` | gated HF dataset; access request required |
| `ilsp/greek-modern-history-qa` | gated HF dataset; access request required |
| `ilsp/greek-history-trapeza-thematon-co-qa` | gated HF dataset; access request required |
| `ilsp/greek_pcr` | gated HF dataset; access request required |

## MT Diagnostics Excluded From Headline

These are not native-Greek headline evidence: `ilsp/arc_greek`,
`ilsp/hellaswag_greek`, `ilsp/winogrande_greek`, `ilsp/mmlu_greek`,
`ilsp/MMLU-Pro_greek`, `ilsp/truthful_qa_greek`,
`arc_challenge_mt_el`, and `global_piqa_completions_ell_grek`.

They can be reported as comparability diagnostics only.

## Scripts

| Script | Purpose |
|---|---|
| `cache_native_greek_benchmarks.py` | CPU/network cache and availability verifier |
| `cache_native_greek_benchmarks.sbatch` | Clariden `xfer` cache job; no GPUs |
| `run_native_greek_mcq_eval.py` | deterministic MCQ log-prob scoring |
| `run_native_greek_mcq_eval.sbatch` | one checkpoint, one GPU |
| `run_native_greek_mcq_eval_packed.sbatch` | up to four checkpoints packed on one 4-GPU node |
| `submit_native_greek_all_checkpoint_evals.sh` | submit Apertus-Base plus all release checkpoints |
| `run_greek_nlp_benchmark_hf.py` | HF backend for upstream `greek-nlp/benchmark` |
| `run_greek_nlp_benchmark_hf.sbatch` | one-checkpoint upstream benchmark wrapper |
| `run_greek_nlp_benchmark_hf_packed.sbatch` | up to four upstream benchmark checkpoint jobs packed on one 4-GPU node |
| `submit_greek_nlp_all_checkpoint_evals.sh` | submit packed `greek-nlp/benchmark` runs for all checkpoints |
| `summarize_native_greek_suite.py` | collect per-checkpoint outputs into native Greek review tables |

## Scoring Policy

MCQ tasks use deterministic causal-LM log-likelihood scoring. The prompt shows
the Greek question and answer options; each candidate answer text is scored as
the continuation after `Σωστή απάντηση:`. The selected answer is the candidate
with highest average token log-probability. No generated text is sampled.

For decision reporting:

1. native MCQ aggregate: GreekMMLU + ILSP Medical MCQA + ILSP ASEP MCQA,
   with Plutus reported both standalone and as a domain-specific optional add;
2. native supporting tasks: `greek-nlp/benchmark` non-translation tasks;
3. pending native tasks: OYXOY, GreekBarBench, civics, lyceum math after their
   scoring adapters are explicit;
4. MT diagnostics: never averaged into the native Greek headline.

## Current Run State

- Local cache verification succeeded for public/cached datasets listed above.
- `greek-nlp/benchmark` sample-100 Vanilla/TD 5B jobs `2396595` and `2396596`
  were cancelled intentionally: the task-specific generation caps had not
  taken effect, so label-style tasks were wasting GPU time.
- Clariden cache job `2396887` completed on `xfer`; manifest:
  `/iopsstor/scratch/cscs/fffoivos/native_greek_eval/cache/native_greek_benchmark_cache_manifest.json`.
- MCQ smoke job `2396914` passed on Apertus-Base with `SAMPLE_SIZE=2`.
- Full native MCQ packed jobs submitted: `2396931`, `2396932`, `2396933`.
- First packed `greek-nlp/benchmark` sample-100 all-checkpoint jobs
  `2396935`, `2396936`, `2396937` were invalidated. The upstream GEC task
  creates a fixed `repo_244` checkout under the process working directory, so
  the packed workers raced on the same temp repo. Job `2396935` was cancelled
  after sibling workers failed; jobs `2396936` and `2396937` failed quickly.
- `run_greek_nlp_benchmark_hf.py` now changes into each model's resolved
  output directory before calling upstream task code, isolating GEC temp
  checkouts per model. Retry packed GreekNLP jobs submitted: `2396991`,
  `2396992`, `2396993`.
- Final run completed:
  - MCQ jobs `2396931`, `2396932`, `2396933` completed with exit code `0:0`.
  - GreekNLP retry jobs `2396991`, `2396992`, `2396993` completed with exit
    code `0:0`.
  - Final summary artifacts are in
    `native_greek_suite_20260526/summary/`.
  - Decision report:
    `NATIVE_GREEK_SUITE_RESULTS_20260526.md`.
