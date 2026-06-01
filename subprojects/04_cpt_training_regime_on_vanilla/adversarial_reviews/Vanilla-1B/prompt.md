You are the adversarial reviewer for the Vanilla Apertus CPT checkpoint
`Vanilla-1B` (998244352 tokens, Megatron iteration 238).

Goal: find flaws, hidden assumptions, methodological mistakes, missing evidence,
data/eval leakage, bad comparisons, broken scripts, checkpoint/eval artifact
problems, compute hygiene issues, and ways the current interpretation could be
wrong. Be skeptical and concrete. Do not edit files, submit jobs, cancel jobs, or
modify remote state. Use read-only shell commands only.

Read these local files first:
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/goal.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/goal/hyperparameters.json
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/cpt-plan.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/RUN_LOG_20260528.md
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/scripts/
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training/

Cross-reference the prior Vanilla-0.5B critique:
- /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/Vanilla-0.5B/adversarial_critique.md
  Its critical findings were: (1) RoPE/seqlen mismatch (training uses
  4096 / rope_theta 500K vs Apertus base config 65536 / rope_theta 12M),
  (2) native MCQ headline JSON includes Plutus (the headline aggregate
  reported elsewhere is 3-task mean ~0.439, including-Plutus is ~0.429),
  (3) no visible decontamination evidence for public Greek MCQ benchmarks,
  (4) BPB heldout heavily prefix-truncated (~29% trunc rate on Greek).
  Explicitly check whether each persists at iter 238 or has changed.

Inspect these Clariden artifacts through read-only SSH commands such as
`ssh clariden 'ls ...'`, `tail`, `grep`, `sacct`, and `find`:
- Training run dir: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z
- Megatron checkpoint iteration: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z/checkpoints/iter_0000238
- Eval root: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z
- HF checkpoint dir: /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z/iter_0000238_hf

Return a Markdown critique with:

1. Verdict: whether this checkpoint's evidence is trustworthy enough to use.
2. Critical findings: issues that could invalidate the run, checkpoint, evals,
   or comparison.
3. Major findings: issues that materially weaken confidence but may be fixable.
4. Minor findings and hygiene notes.
5. Missing evidence: exact files/commands/results still needed.
6. Recommended next actions before reading or acting on this checkpoint.
7. Persistence-of-prior-findings: explicit one-line verdict on each of the four
   Vanilla-0.5B critical findings — still present / fixed / now mitigated /
   unknown.

Use file paths, job IDs, metric names, and exact artifact paths wherever
possible. If you cannot access a required artifact, say so directly.
