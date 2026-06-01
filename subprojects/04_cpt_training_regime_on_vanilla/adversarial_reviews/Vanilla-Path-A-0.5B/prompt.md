You are the adversarial reviewer for the Vanilla-Path-A-0.5B checkpoint (Megatron iter 119, 499,122,176 tokens) trained under Apertus paper §2.5 long-context geometry (rope_theta=12,000,000, max_position_embeddings=65,536, llama3 rope_scaling factor 8.0).

This is the **Path A geometry probe** from `PATH_A_GEOMETRY_PROBE_PLAN.md`. The hypothesis being tested:

> Forcing rope_theta=500K on weights pretrained at rope_theta=12M phase-shifts Q·K rotations and perturbs the model rather than re-anchoring it. The first ~1 B tokens of Task 1's Path B CPT carry the rope re-adaptation cost.

Equivalent operational statement: if we train under Path A (rope=12M, the released base's native geometry), the trajectory should NOT show the rope re-adaptation dip in the first 0.5 B tokens.

## Headline results (verify, do not just quote)

iter 119 evals at `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04a_vanilla_path_a_probe_20260531T103924Z/iter_0000119/`:

- Native MCQ 3-task headline = **0.4942** (greekmmlu 0.5427 / ilsp_medical_mcqa 0.4074 / ilsp_mcqa_asep 0.5325)
- Plutus QA diagnostic = 0.5289
- Greek BPB = 0.4365 (heldout: cpt_greek_heldout_500_20260522.jsonl)
- Code BPB = 0.2725
- Math BPB = 0.5459
- retention key tasks: global_mmlu_en=0.6425, _fr=0.6100, _de=0.5975, mmlu=0.5977, xnli_en=0.5414, _fr=0.4920, _de=0.5060, _ru=0.4912, arc_challenge=0.5452, arc_easy=0.8295, piqa=0.7916, hellaswag=0.6008

## Comparisons (point estimates — paired CIs are being computed in parallel)

| Metric | Path A iter 119 | Path B iter 119 (Task 1) | Apertus-Base Path A | matched-config Path-B init |
|---|---:|---:|---:|---:|
| MCQ headline 3-task | **0.4942** | 0.4391 | 0.4817 | 0.4272 |
| Plutus | 0.5289 | 0.4000 | 0.5156 | 0.3467 |
| Greek BPB | 0.4365 | 0.6049 | n/a | 1.2216 |
| global_mmlu_en | 0.6425 | 0.5974 | n/a | 0.6025 |
| global_mmlu_fr | 0.6100 | 0.5200 | n/a | 0.5425 |
| global_mmlu_de | 0.5975 | 0.5188 | n/a | 0.5725 |
| mmlu | 0.5977 | 0.5674 | n/a | 0.5624 |
| xnli_ru | 0.4912 | 0.4880 | n/a | 0.4884 |
| arc_challenge | 0.5452 | 0.5247 | n/a | 0.5384 |
| arc_easy | 0.8295 | 0.8035 | n/a | 0.8279 |

## Your job — 7 sections

1. **Verdict on the checkpoint**: trustworthy? mechanical issues? converted HF clean? sidecars all green?
2. **Critical findings**: anything that invalidates the run, evals, or comparison.
3. **Major findings**: persistence-of-prior-issues check (the 13 errors in TASK2_HANDOFF.md §2; especially decontamination, BPB truncation, Plutus n=225).
4. **Minor findings + hygiene notes**.
5. **Missing evidence**.
6. **Recommended next actions** before declaring the probe verdict on TASK2_HANDOFF.md §(3.1).
7. **Hypothesis verdict** (load-bearing): CONFIRMED / REFUTED / INTERMEDIATE per `PATH_A_GEOMETRY_PROBE_PLAN.md` §7 decision rule. Bracket your verdict with the comparison point estimates above; the paired CIs are being computed by a separate subagent and you don't need to wait for them.

## Read first

- `PATH_A_GEOMETRY_PROBE_PLAN.md` (the plan + §7 decision rule)
- `TASK2_HANDOFF.md` (especially §2.3 matched-config perturbation finding, §3.1 Path A recommendation)
- `cpt-plan.md` §2.1 (Path A vs Path B description), §3.4 Q3.4.10 (Task 2 recommendation)
- `reports/decisions_matrix_20260529.md` (Decision C: RoPE/seqlen mismatch)
- `goal/hyperparameters.json` (training_geometry block)
- Vanilla-0.5B + Vanilla-1B + Vanilla-2B critiques in `adversarial_reviews/` for persistence-of-findings baseline

## Inspect on Clariden (read-only)

- Train run dir: `/capstor/scratch/cscs/fffoivos/runs/04a_vanilla_path_a_probe/04a_vanilla_path_a_probe_20260531T103924Z`
- iter 119 Megatron checkpoint: `…/checkpoints/iter_0000119`
- HF dir: `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04a_vanilla_path_a_probe_20260531T103924Z/iter_0000119_hf` — verify config.json shows Path A (rope_theta=12M, max_position=65536, rope_scaling=llama3)
- Eval root: `…/eval_04a_vanilla_path_a_probe_20260531T103924Z/iter_0000119/`
- Training log: `/capstor/scratch/cscs/fffoivos/runs/04a_vanilla_path_a_probe/04a_path_a_probe_i119-2437909.out`

## Constraints

ssh clariden read-only. No Slurm. No modifications other than the critique file itself.

## Output

Write the critique to:
```
/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/adversarial_reviews/Vanilla-Path-A-0.5B/adversarial_critique.md
```

## Return

<250 words: verdict + counts per severity + hypothesis verdict (CONFIRMED / REFUTED / INTERMEDIATE) + key warnings or anomalies. Do NOT paste the full critique — it goes on disk.
