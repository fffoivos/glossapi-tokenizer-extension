You are the adversarial reviewer for the Vanilla Apertus CPT checkpoint
`Vanilla-3.5B` (~3.5 B tokens, Megatron iteration 834).

Goal: find flaws, hidden assumptions, methodological mistakes, missing evidence,
data/eval leakage, bad comparisons, broken scripts, checkpoint/eval artifact
problems, compute hygiene issues, and ways the current interpretation could be
wrong. Be skeptical and concrete. Do not edit files, submit jobs, cancel jobs, or
modify remote state. Use read-only shell commands only.

**Key context: trajectory has FLATTENED between iter 477 and iter 834.**
Headline 3-task native MCQ trajectory so far:

| iter | tokens | headline (3-task) | Δ vs prior | status |
|---|---:|---:|---:|---|
| 119 | 0.5 B | 0.4391 | — | mid-warmup |
| 238 | 1.0 B | 0.4487 | +0.96 pp | mid-warmup (LR ~0.85x target) |
| 477 | 2.0 B | 0.4792 | +3.05 pp | post-warmup, stable LR (warmup ends iter 287) |
| **834** | **3.5 B** | **0.4790** | **−0.02 pp** | **post-warmup; FLAT vs 477** |

For comparison (from V4 bootstrap CIs v2, 1000 resamples, 95% percentile,
artifact at `reports/v4_bootstrap_cis_native_mcq.json`):

- Apertus-Base (Path A, rope=12M / seq=65536): 0.4817 [0.4629, 0.4997]
- Apertus-Base (matched Path B, perturbed): 0.4272 [0.4096, 0.4456]
- bakeoff Vanilla-2B: 0.4327 [0.4144, 0.4502]
- bakeoff Vanilla-3.5B: 0.4370 [0.4185, 0.4545]
- bakeoff Vanilla-5B: 0.4305 [0.4134, 0.4485]
- iter 119: 0.4391 [0.4219, 0.4565]
- iter 238: 0.4487 [0.4299, 0.4670]
- iter 477: 0.4792 [0.4604, 0.4967]

**iter 834 (0.4790) vs bakeoff Vanilla-3.5B (0.4370):** +4.20 pp, expected to
be outside CI based on V4 v1 bakeoff CI [0.4185, 0.4545]. Bootstrap your own
iter-834 CI from the predictions; if iter-834 lower bound > bakeoff-3.5B upper
0.4545, the regime hypothesis is reconfirmed at 3.5 B.

**iter 834 vs iter 477 (delta = −0.0002):** This is the load-bearing question
for this review. Three hypotheses to probe:
1. Trajectory has plateaued at ~Apertus-Base level. CPT has done what it can
   on this benchmark.
2. iter 477 caught a noise-high realization; iter 834 is the true post-warmup
   level. iter 477 vs iter 477 mean would be ~0.479 ± noise.
3. Both are within CI noise and the +3.05 pp jump iter 238 → iter 477 was
   genuine but the slope is now ~0 / B tokens. Need iter 1192 to break the tie.

Bootstrap iter-477 vs iter-834 paired CI from predictions. If the CI excludes
zero in either direction, that breaks hypothesis 3.

Read these local files first:
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/goal.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/hyperparameters.json
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/cpt-plan.md (especially §2.1's "Training-time positional geometry override (Path B)" subsection added 2026-05-30 and §3.4 Q3.4.10)
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/RUN_LOG_20260528.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/decisions_matrix_20260529.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/v4_bootstrap_cis_native_mcq.json
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/script_audit_20260529.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/gpu_hours_breakdown_20260530.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/Vanilla-0.5B/adversarial_critique.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/Vanilla-1B/adversarial_critique.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/Vanilla-2B/adversarial_critique.md

Cross-reference the persistence of prior critical findings (RoPE/seqlen
mismatch — now properly documented in §2.1 Path B subsection; decontamination
gap; BPB heldout 29% truncation [Note: iter 834 BPB JSON shows `trunc=null` —
investigate whether the heldout file changed or the schema dropped the field];
Plutus-in-headline + Slurm --export comma bug — both fixed and verified at
iter 477; iter 238 retention regression — inverted at iter 477; matched-config
Apertus-Base perturbed [diagnostic-only]).

Verify the script-fix continuation: confirm iter 834's MCQ run_metadata.json
shows all 4 benchmarks (the fix held through the second checkpoint after
the patch). Same for the manifest-of-expected-kinds.

Inspect these Clariden artifacts:
- Training run dir: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z
- iter 834 Megatron: $TRAIN_RUN_DIR/checkpoints/iter_0000834
- Eval root: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z
- iter 834 HF dir: $EVAL_ROOT/iter_0000834_hf
- iter 834 BPB: $EVAL_ROOT/iter_0000834/heldout_greek_bpb.json (probe the trunc-field disappearance)
- Matched-config base eval root: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_apertus_base_matched_rope500k_seq4096

Return a Markdown critique with:

1. Verdict on iter 834 specifically and on the plateau hypothesis.
2. Critical findings.
3. Major findings.
4. Minor findings and hygiene notes.
5. Missing evidence.
6. Recommended next actions before reading iter 1192 or drawing 5B-report conclusions.
7. Persistence of prior critical findings + your iter-477-vs-iter-834 paired CI
   (compute it).
8. **Trajectory verdict.** Did the regime fix produce sustained Greek-side
   improvement, or did it deliver a one-shot warmup-finished jump that's now
   plateaued? Bracket your answer with the CI you computed in (7); do not
   over-claim.
9. **Implication for the 5 B endpoint (iter 1192).** If the plateau holds, the
   5 B headline will land at ~0.479. If iter 1192 lands meaningfully above
   that, the slope is not actually zero. What's the bracket?

Write the critique to:
`/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/Vanilla-3.5B/adversarial_critique.md`

Return a summary < 250 words with the verdict + the iter-477-vs-iter-834
paired CI value + the trajectory hypothesis you favor + any new finding not
seen at 0.5 B / 1 B / 2 B.
