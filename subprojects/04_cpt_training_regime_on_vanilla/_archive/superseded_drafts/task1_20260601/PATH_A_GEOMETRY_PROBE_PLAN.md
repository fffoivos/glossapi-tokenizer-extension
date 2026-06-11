# Path A geometry probe — Task 1 follow-up

A small, focused CPT run to test the geometry-perturbation hypothesis
that explains why Task 1's matched-config Apertus-Base baseline started
5.5 pp below the released Apertus-Base on the headline 3-task native
Greek MCQ aggregate (0.4272 vs 0.4817), and why Task 1's iter 119
(0.4391) sits below the released Apertus-Base too.

**Status.** Draft, ready to launch when authorized.
**Authorization required.** Submission affects Slurm queue + costs ~25 GPU-h.

---

## (1) Hypothesis under test

From `TASK2_HANDOFF.md` §2.3 + the Vanilla-2B adversarial critique:

> Forcing `rope_theta=500K` on weights pretrained at `rope_theta=12M`
> phase-shifts every Q·K rotation against the values the model learned
> during pretraining. The matched-config Apertus-Base eval is therefore
> not a clean re-anchor of the base under Path-B geometry — it is itself
> a *perturbation*. The first ~1 B tokens of Task 1's Path-B CPT carry
> the rope re-adaptation cost; iter 119 (0.4391) and the matched-config
> baseline (0.4272) are both perturbed-state readings, not free-state
> readings of the base.

**Equivalent operational statement.** If we run the same Apertus-faithful
CPT regime under the released base's Path-A geometry (rope_theta=12M,
max_position_embeddings=65536, llama3 rope_scaling, sequence_length=4096),
the trajectory should NOT show the rope re-adaptation dip in the first
0.5–1 B tokens, because the model is being trained under the same
positional geometry it was pretrained on.

## (2) What we want to learn

At 0.5 B tokens (matched to Task 1's iter 119), the geometry probe
discriminates between two hypotheses:

| State | Path-A 0.5 B headline | Greek BPB | Reading |
|---|---|---|---|
| Hypothesis CONFIRMED | ≳ Apertus-Base Path-A (0.4817) | clearly below Path-B iter 119 (0.6049) | Geometry was the cause of Task 1's "starts low" pattern. Path-A is the right Task-2 default. Rope re-adaptation is a real ~1 B-token cost in Path-B continuation. |
| Hypothesis REFUTED | ≲ Apertus-Base Path-A (0.4817) | similar to Path-B iter 119 (0.6049) | Geometry was not the dominant cause. Task 1's "starts low" pattern has some other explanation (data quirk, optimizer dynamics, regime change interaction). Re-examine the regime hypothesis itself. |
| Intermediate | At Apertus-Base Path-A within CI | between matched-config and Path-B iter 119 | Geometry contributes part of the cost but isn't the whole story. Use BPB direction + per-task structure to disentangle further. |

The 0.5 B checkpoint also gives us **per-task comparison vs Task 1's iter
119**, which discriminates which native MCQ benchmarks are most
rope-sensitive (greekmmlu vs medical vs ASEP).

## (3) Run settings

**Identical to Task 1 except positional geometry and the explicit "no
extension" arm signal.** Apertus-faithful regime preserved (LR + warmup
+ optimizer + loss + batch + dataset all unchanged) so the only
intentional difference is geometry.

### Geometry (the change being tested)

| Parameter | Path-A probe value | Source |
|---|---:|---|
| `rope_theta` | 12000000 | Apertus paper §2.5 long-context extension + released base `config.json` |
| `max_position_embeddings` | 65536 | Apertus paper §2.5 + released base |
| `rope_scaling` | `llama3` (factor=8.0, original_max_position_embeddings=8192, low_freq_factor=1.0, high_freq_factor=4.0) | Apertus paper §2.5 + released base |
| `sequence_length` | 4096 | Training only; geometry supports longer. Matches Task 1 |

### Regime + everything else (unchanged from Task 1)

| Parameter | Value | Source |
|---|---|---|
| Base checkpoint | `swiss-ai/Apertus-8B-2509` (R17-patched, TP=2 Megatron init from `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched`, re-converted under Path-A geometry — see §4 launch prep) | Task 1 baseline init |
| Optimizer | AdEMAMix, β1=0.9, β2=0.999, β3=0.99, α=8, weight_decay=0.1, gradient_clipping=0.1 | `hyperparameters.json[optimizer]` |
| α/β3 warmup | 287 steps | `hyperparameters.json[optimizer.alpha_beta3_warmup_steps]` |
| LR | 1.1e-5 peak | `hyperparameters.json[lr_schedule.base_lr]` |
| LR schedule | linear warmup 287 steps (= 1.2 B tokens) → constant | Apertus paper §2.5 long-context continuation pattern |
| Loss | Goldfish, k=50, h=50, hash_seed=2971215073, hash_table_size=1000003, prod_mod hash | `hyperparameters.json[loss]` |
| Global batch | 1024 samples = 4,194,304 tokens | Task 1 |
| TP / PP / DP | 2 / 1 / 2 | Task 1 |
| Microbatch | 2 samples/GPU | Task 1 |
| Precision | bf16 | Task 1 |
| Cross-document attention | disabled | Task 1 |
| EoD loss masking | enabled | Task 1 |
| Mix | 70% Greek HPLT clean60 wave4 (Apertus-pretrain dedup overlay) / 24% replay / 4% code / 2% math | `hyperparameters.json[training_mix]` |
| Within-Greek mix | uniform | Task 1 |
| Tokenizer | Apertus base 131,072 (Mistral-Nemo tekken v3); no extension | Task 1 |

### Training schedule

| Parameter | Value | Note |
|---|---|---|
| Target tokens | **5e8 (0.5 B)** | Matched to Task 1's iter 119 checkpoint mark; the minimum that gives a comparable headline reading |
| Iter target | 119 | Equivalent token count at our global batch size |
| Save at | iter 119 (0.5 B) | Single endpoint; no intermediate saves needed for this scope |
| Warmup window | 287 steps (= 1.2 B tokens equivalent) | Identical to Task 1; at iter 119 LR will be ~41 % of peak — mid-warmup, matching Task 1's state at the same iter |
| Walltime budget | 5 hours on `normal` (single node, 4 GPUs) | ~4.3 h expected wall-clock + buffer; 12-hour partition limit gives plenty of margin |

## (4) Launch prep

1. **Re-convert the Apertus-Base init checkpoint to Megatron under Path-A
   geometry.** Existing init at
   `/iopsstor/.../init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched`
   was converted under Path-B config (rope=500K, max_pos=4096). Need a
   sibling at `…/megatron_tp2_r17patched_path_a` with rope=12M,
   max_pos=65536, llama3 scaling baked into the Megatron args.
2. **Mirror updated training config to Clariden.** Create
   `scripts/train_config_04a_vanilla_path_a.env` derived from the Task-1
   `train_config_04_vanilla.env` with the geometry args changed.
3. **Sbatch script.** Reuse `scripts/submit_training_5b_chain.sh` machinery
   but with new run-tag (suggested: `04a_vanilla_path_a_probe_<timestamp>Z`)
   and the iter-119-only schedule. One segment, no chaining.
4. **Sidecar fan-out.** Reuse `scripts/submit_checkpoint_sidecars.sh`
   (post-fix hash `e865c65a…`). Submits convert + native_mcq + greek_nlp +
   greek BPB + retention + code/math BPB + checksum. Same canonical eval
   task list as Task 1 (`goal/canonical_eval_tasks.json` lockdown).
5. **Adversarial review.** After handoff_ready=true, fire a Claude Code
   subagent with a focused prompt comparing Path-A iter 119 to:
   - Task 1 Path-B iter 119 (0.4391 headline, 0.6049 Greek BPB)
   - Matched-config Apertus-Base Path-B-perturbed (0.4272 headline, 1.2216 Greek BPB)
   - Apertus-Base Path A native MCQ (0.4817)

## (5) Cost

| Step | Wall-clock | GPU-h |
|---|---:|---:|
| Re-convert init checkpoint to Path A | ~10 min on a single GPU | ~0.7 |
| Training to iter 119 | ~4.3 h on 4 GPUs | ~17.2 |
| Sidecar fan-out (convert + 6 evals + checksum) | ~1 h dominant by retention + greek_nlp | ~5–7 |
| Adversarial review (Claude subagent, home-side) | ~10 min | 0 |
| **Total** | ~5–6 h wall-clock | **~23–25 GPU-h** |

About 11 % of Task 1's 217 GPU-h. Stops with one explicit user decision
to launch.

## (6) Expected outputs

Artifacts land at:

```
/capstor/scratch/cscs/fffoivos/runs/04a_vanilla_path_a_probe/
  04a_vanilla_path_a_probe_<TS>Z/
    checkpoints/iter_0000119
  eval_04a_vanilla_path_a_probe_<TS>Z/
    iter_0000119/
      native_mcq/   greek_nlp_s100/   heldout_*.json   retention/   checksums/

subprojects/04_cpt_training_regime_on_vanilla/
  adversarial_reviews/Vanilla-Path-A-0.5B/
    prompt.md  adversarial_critique.md  review_metadata.env
  reports/path_a_probe_results_<TS>Z.md   # comparison + verdict
  reports/v4_workspace_path_a/             # bootstrap CIs vs Task 1 + Apertus-Base
```

## (7) Decision rule

After iter 119 sidecars + adversarial review complete:

**Compute bootstrap CIs (same methodology as V4 v3):**
- Path-A iter 119 marginal headline + per-task CIs.
- Path-A iter 119 vs Path-B iter 119 paired CI on headline + per-task.
- Path-A iter 119 vs Apertus-Base Path A paired CI on headline + per-task.
- Path-A iter 119 vs matched-config Apertus-Base Path-B-perturbed paired CI on headline + per-task.

**Hypothesis adjudication:**

- **CONFIRMED** if both:
  - Path-A iter 119 vs Path-B iter 119 paired Δ outside zero positive on headline (Path A at least 2 pp above Path B at matched tokens), AND
  - Path-A iter 119 vs Apertus-Base Path A paired Δ inside or above zero (Path A stays at or above base — no rope re-adaptation dip).

- **REFUTED** if both:
  - Path-A iter 119 vs Path-B iter 119 paired Δ inside zero (Path A and Path B produce comparable results at matched tokens), AND
  - Path-A iter 119 also < Apertus-Base Path A by ≥ 2 pp (something other than geometry is causing the dip).

- **INTERMEDIATE** otherwise. Report direction + magnitude + per-task structure; do not redesign on this alone.

## (8) Decision documentation

Verdict lands in:
- `reports/path_a_probe_results_<TS>Z.md` — the comparison + verdict + per-task CIs.
- `RUN_LOG_20260528.md` — append-only pulse log entry summarising the result.
- `TASK2_HANDOFF.md` §(3.1) — IF the hypothesis is CONFIRMED, this strengthens the Path-A recommendation; if REFUTED, this complicates the Path-A recommendation and the planning agent should reconsider the geometry choice for Task 2 v1.x.

## (9) Optional extensions (not in this plan)

If the user wants to spend more for a stronger reading:

| Extension | Marginal cost | What it adds |
|---|---:|---|
| Train to 1.2 B (= warmup end, iter 287) instead of 0.5 B | +~25 GPU-h (total ~50 GPU-h) | Path-A warmup-end state, comparable to a Path-A "iter 287" if Task 2 had run a longer probe |
| Train to 2 B (= full post-warmup) | +~60 GPU-h (total ~85 GPU-h) | Direct Path-A vs Path-B comparison at the first stable-LR snapshot mark, matched to Task 1 iter 477 |
| Add a second arm: Path-A with B1-mix but no extension (this plan) AND a third arm comparing rope_theta sensitivity by training another arm at rope_theta=2M (intermediate) | +1× to +2× this plan's cost | Discriminates whether the recovery is binary (rope-matches-pretrain or not) or graded |

**Recommendation.** Start with the 0.5 B probe in this plan. If the
result is CONFIRMED, no further geometry probe is needed before Task 2.
If INTERMEDIATE, extend to 1.2 B. If REFUTED, escalate the
investigation methodology before committing more compute.

## (10) Non-commitments

- No threshold rule on the headline number is pre-committed (per
  `cpt-plan.md` §6 standing rule). Outcome read post-result.
- This probe does NOT validate Task 2 fully — it only validates the
  Path-A geometry choice. Other Task-2 questions (extension method, BPE
  cutoff, embedding stabilization) remain open per `cpt-plan.md` §3.4.
- A REFUTED outcome does NOT invalidate Task 1's regime hypothesis
  conclusion (regime supported vs bakeoff, +6.69 pp, CI excludes zero).
  Task 1's bakeoff-vs-this comparison is Path-B-vs-Path-B and is
  unaffected by what happens under Path A.

## (11) Out of scope for this probe

- No vocabulary extension (Vanilla only; same as Task 1 design choice).
- No data mix variations (B1 70/24/4/2 unchanged).
- No optimizer variations.
- No LR schedule variations.
- No mid-training geometry switches.

If any of those become relevant, they need their own probe plan.

---

**End of plan.** Ready to launch when authorized.
