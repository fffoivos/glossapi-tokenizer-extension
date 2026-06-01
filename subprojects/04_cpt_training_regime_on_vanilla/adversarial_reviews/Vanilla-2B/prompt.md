You are the adversarial reviewer for the Vanilla Apertus CPT checkpoint
`Vanilla-2B` (2,000,683,008 tokens, Megatron iteration 477).

Goal: find flaws, hidden assumptions, methodological mistakes, missing evidence,
data/eval leakage, bad comparisons, broken scripts, checkpoint/eval artifact
problems, compute hygiene issues, and ways the current interpretation could be
wrong. Be skeptical and concrete. Do not edit files, submit jobs, cancel jobs, or
modify remote state. Use read-only shell commands only.

**Note: iter 477 is the FIRST stable-LR checkpoint.** Warmup ends at iter 287
(1.2B-token warmup window). iter 119 (0.5B) and iter 238 (1B) are mid-warmup
probes; iter 477 is post-warmup. Native-MCQ headline trajectory so far:

| iter | tokens | headline (3-task) | status |
|---|---:|---:|---|
| 119 | 0.5B | 0.4391 | mid-warmup |
| 238 | 1.0B | 0.4487 | mid-warmup (LR ~0.85x target) |
| 477 | 2.0B | **0.4792** | post-warmup, stable LR |

For comparison (from V4 bootstrap CIs, 1000 resamples, 95% percentile,
artifact at `reports/v4_bootstrap_cis_native_mcq.json`):

- Apertus-Base (Path A, rope=12M / seq=65536): 0.4817 [0.4629, 0.4997]
- bakeoff Vanilla-2B (Path B, rope=500K / seq=4096): 0.4327 [0.4144, 0.4502]
- bakeoff Vanilla-3.5B: 0.4370 [0.4185, 0.4545]
- bakeoff Vanilla-5B: 0.4305 [0.4134, 0.4485]
- iter 119: 0.4391 [0.4219, 0.4565]

iter 477 (0.4792) is **+4.65 pp above bakeoff Vanilla-2B**, with the
bakeoff CI upper bound at 0.4502; the delta is outside the bakeoff CI.
The point estimate is inside Apertus-Base's CI [0.4629, 0.4997].

But: the Apertus-Base CI uses the published checkpoint at rope=12M /
seq=65536 (Path A), while our run trains at rope=500K / seq=4096
(Path B). A matched-config Apertus-Base eval was submitted as jobs
2422890 (native MCQ), 2422891 (retention), 2422892 (BPB) against
`/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509-matched-rope500k-seq4096/`.
These were running at the time you started this review. Check their state
via `ssh clariden 'sacct -j 2422890,2422891,2422892 ...'` and include the
matched-config results in your reading if they are COMPLETED by then; if
PENDING/RUNNING, flag the conclusion as awaiting matched-config baseline.

Read these local files first:
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/goal.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/hyperparameters.json
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/cpt-plan.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/RUN_LOG_20260528.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/decisions_matrix_20260529.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/v4_bootstrap_cis_native_mcq.json
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/reports/script_audit_20260529.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/scripts/
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/Vanilla-0.5B/adversarial_critique.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/Vanilla-1B/adversarial_critique.md

Cross-reference the four critical findings from Vanilla-0.5B that persisted
at Vanilla-1B:

1. RoPE/seqlen mismatch (HF config 4096 / rope_theta 500K vs base 65536 / 12M).
2. Plutus-in-headline-JSON (structurally fixed by aggregate's `headline_policy`
   block; operationally re-broken at iter 238 by Slurm `--export` comma bug;
   NOW FIXED — the iter 238 + 477 native_mcq aggregates were re-issued via
   jobs 2422769 + 2422770 with `BENCHMARKS=` set as env-var prefix to bypass
   the comma split).
3. No decontamination evidence for public Greek MCQ benchmarks.
4. BPB heldout 29.2% prefix-truncated.

Plus the new finding from Vanilla-1B:

5. Slurm `--export` comma-split bug in `submit_checkpoint_sidecars.sh` L156
   silently truncated `BENCHMARKS` to "greekmmlu" only at iter 238 + iter 477.
   Now fixed (hash 7eb4667e -> e865c65a; mirror in sync). Resubmits 2422769
   (iter 238) and 2422770 (iter 477) both COMPLETED with corrected 4-benchmark
   set.

Test whether each of 1, 3, 4 persists at iter 477. Verify 2 and 5 are
operationally clean (i.e. iter 477's `Vanilla-2B_native_mcq_aggregate.json`
shows `n_tasks=3` headline + `n_tasks=1` diagnostic, run_metadata.json shows
all four benchmarks).

Inspect these Clariden artifacts through read-only SSH commands such as
`ssh clariden 'ls ...'`, `tail`, `grep`, `sacct`, `jq`, and `find`:
- Training run dir: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z
- Megatron checkpoint iteration: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z/checkpoints/iter_0000477
- Eval root: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z
- HF checkpoint dir: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z/iter_0000477_hf
- Matched-config Apertus-Base eval root: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_apertus_base_matched_rope500k_seq4096
- Matched-config Apertus-Base model: /iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509-matched-rope500k-seq4096

Return a Markdown critique with:

1. Verdict: whether this checkpoint's evidence is trustworthy enough to use,
   given that it's the first stable-LR snapshot and shows +4.65 pp over bakeoff
   Vanilla-2B (point estimate above bakeoff CI upper bound).
2. Critical findings.
3. Major findings.
4. Minor findings and hygiene notes.
5. Missing evidence.
6. Recommended next actions before reading or acting on this checkpoint.
7. Persistence of prior critical findings: explicit one-line verdict on each
   of findings 1, 3, 4, 5 (plus the iter-238 retention regression Major 2 —
   does it persist or invert at iter 477?).
8. **Regime hypothesis verdict.** Per cpt-plan.md §2.2, this checkpoint's job
   is to answer "did the bakeoff regime cause the native Greek MCQ degradation,
   and does the Apertus-faithful regime fix it?" Give your honest read.
   Bracket with V4 CI tolerances; do not over-claim.

Use file paths, job IDs, metric names, and exact artifact paths wherever
possible. If you cannot access a required artifact, say so directly.
