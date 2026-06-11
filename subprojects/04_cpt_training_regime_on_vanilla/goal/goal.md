# Goal - Vanilla CPT regime experiment

Run the Vanilla Apertus-8B continuation experiment with base 131,072-token
Apertus tokenizer, HPLT-only Greek, and B1 replay/code/math. Test whether the
corrected Apertus-style CPT regime improves Vanilla without tokenizer-extension
confound.

Hyperparam docs: `goal/hyperparameters.json` is the authoritative
machine-readable source for sbatch generation. The human-readable table is in
`../_archive/superseded_drafts/task1_20260601/cpt-plan.md` section 2.1, with decision rationale in sections 2.4-2.5.
Smoke evidence lives in `../RUN_LOG_20260528.md`.

Future-experiment note: the main result to carry forward is this corrected
hyperparameter regime. Task-2 and later runs should use this Vanilla
`goal/hyperparameters.json` as the fixed regime baseline, while their active
goals and launch scripts live in their own subproject.

## Fixed Scope

- Model/tokenizer: `swiss-ai/Apertus-8B-2509`, base vocab 131,072.
- Greek data: `HPLT/ell_Grek_ge8_no_mt_clean60`, with the existing Apertus
  overlap drop overlay.
- Other data: B1 replay/code/math.
- Mix: 70% HPLT Greek / 24% replay / 4% code / 2% math.
- Excluded: GlossAPI Greek sources, OpenArchives, didaktorika, OpenSubtitles,
  high-register cooldown pools, tokenizer extension, staged Greek curriculum.
- First target: train to 5B tokens. Continue to 10B only after the 5B report is
  reviewed.

HPLT has enough base-tokenizer tokens for the 10B cap; exact total is metadata,
not a launch blocker.

## Locked Training Settings

Use `goal/hyperparameters.json` for sbatch generation; do not infer from prose.

- Loss: Goldfish, `k=50`, `h=50`, hash seed `2971215073`.
- LR: warm up to `1.1e-5` over 1.2B tokens, then constant.
- Optimizer: AdEMAMix, beta1 `0.9`, beta2 `0.999`, beta3 `0.99`, alpha `8`.
- Alpha/beta3 warmup: 287 optimizer steps.
- Sequence length 4096; global batch 1024 samples = 4,194,304 tokens.
- Parallelism: TP=2, PP=1, default 1 Clariden GH200 node / 4 GPUs, DP=2.
- Microbatch: 2 samples/GPU; grad accumulation: 256.
- Precision: bf16 with fp32 master gradients.

## Execution

1. Build the final dataset on CPU-only resources. On Clariden, use `xfer` for
   tokenization/filtering/packing/indexing; do not allocate GPUs for CPU work.
   Save manifest, source weights, tokenizer path, and Megatron data prefix.

2. Launch the 5B Vanilla CPT run from the R17-patched TP=2 Vanilla init
   checkpoint, using the smoke-tested scripts and config. Save checkpoints at
   0.5B, 1B, 2B, 3.5B, and 5B.

3. Keep evaluations as sidecars. Training must continue while checkpoint
   conversion/evaluation jobs run independently.

4. Evaluate each required checkpoint:
   - Convert Megatron checkpoint to HF.
   - Run headline native Greek MCQ: GreekMMLU, ILSP Medical MCQA, ILSP ASEP.
   - Run diagnostics: Plutus QA, greek-nlp/benchmark sample, heldout Greek BPB.
   - Run multilingual retention for English, French, German, and Russian.
   - Run code/math heldout BPB or loss if the existing infrastructure supports
     it.
   - Run a local `codex exec` adversarial review with model `gpt-5.5` and
     `model_reasoning_effort="xhigh"` against the checkpoint, scripts, logs,
     evals, and Clariden artifacts.
   - Keep MT-derived Greek tasks out of the headline.

5. Write the 5B report with exact job IDs, commands, dataset prefix,
   checkpoints, eval paths, restarts, throughput/loss health, and comparisons
   to Apertus-Base plus prior bakeoff Vanilla at matched token marks.

6. Continue to 7B/10B only if the 5B report justifies it, using the same
   checkpoint/eval sidecar pattern.

## Required Artifacts

- Goal and hyperparameter source docs: `goal/goal.md` and
  `goal/hyperparameters.json`; cite both in the 5B report.
- Dataset manifest and Megatron prefix.
- Final training sbatch script, resolved command, and run metadata.
- Checkpoints at all planned token marks.
- HF-converted eval checkpoints.
- Per-checkpoint native Greek, BPB, retention, and code/math outputs.
- Per-checkpoint adversarial critique prompt, raw Codex JSONL, and final report.
- Final 5B report, plus 10B report if launched.

## Stop Conditions

Stop before or during the long run if dataset validation fails, model load
fails, training shows NaNs, repeated skipped iterations, OOMs, checkpoint save
failures, corrupted conversion, broken eval loading, or accidental GPU use for
CPU-only dataset work.
