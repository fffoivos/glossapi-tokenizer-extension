# Native Greek suite results - 2026-05-26

Generated after the full native-Greek pass completed on Clariden.

This report supersedes the older fallback-Greek interpretation for
Greek-specific model selection. The earlier lm-eval bakeoff remains valid for
English retention, multilingual retention, BPB, and the historical fallback
Greek tasks, but those Greek tasks were not the vetted native suite.

## Artifacts

Remote full outputs:

- MCQ all-checkpoint root:
  `/capstor/scratch/cscs/fffoivos/runs/eval/native_greek_suite_20260526/mcq_all_checkpoints`
- `greek-nlp/benchmark` all-checkpoint root:
  `/capstor/scratch/cscs/fffoivos/runs/eval/native_greek_suite_20260526/greek_nlp_all_checkpoints_s100`
- Final summary:
  `/capstor/scratch/cscs/fffoivos/runs/eval/native_greek_suite_20260526/summary`

Repo summary copy:

- `native_greek_suite_20260526/summary/NATIVE_GREEK_SUITE_SUMMARY.md`
- `native_greek_suite_20260526/summary/native_mcq_aggregate.csv`
- `native_greek_suite_20260526/summary/native_mcq_per_task.csv`
- `native_greek_suite_20260526/summary/greek_nlp_supporting_aggregate.csv`
- `native_greek_suite_20260526/summary/greek_nlp_per_task.csv`

Slurm verification:

| Job | Scope | State | Elapsed | Exit |
|---:|---|---|---:|---:|
| 2396931 | native MCQ chunk 1 | COMPLETED | 00:15:49 | 0:0 |
| 2396932 | native MCQ chunk 2 | COMPLETED | 00:15:49 | 0:0 |
| 2396933 | native MCQ chunk 3 | COMPLETED | 00:10:01 | 0:0 |
| 2396991 | greek-nlp chunk 1 | COMPLETED | 00:39:08 | 0:0 |
| 2396992 | greek-nlp chunk 2 | COMPLETED | 00:42:58 | 0:0 |
| 2396993 | greek-nlp chunk 3 | COMPLETED | 00:43:25 | 0:0 |

The first packed `greek-nlp` attempt (`2396935`/`2396936`/`2396937`) was
invalidated by an upstream GEC temp-directory race on `repo_244`; the runner
was patched to execute upstream tasks from each model's output directory and
the retry jobs above completed.

## Headline policy

The headline Greek score uses vetted native/authentic Greek datasets only.
Explicit machine-translated Greek tasks remain diagnostics and are excluded
from headline aggregation.

Headline native MCQ aggregate:

- GreekMMLU
- ILSP Medical MCQA Greek
- ILSP ASEP MCQA

Domain add-on:

- Plutus QA, reported separately as `MCQ + Plutus`

Supporting native suite:

- `greek-nlp/benchmark` sample-100 per task, excluding its Greek-source
  `machine_translation` task from the supporting mean.

## Native MCQ aggregate

| Model | Native MCQ general | MCQ + Plutus |
|---|---:|---:|
| Apertus-Base | 0.4817 | 0.4902 |
| Vanilla-3.5B | 0.4370 | 0.4333 |
| Vanilla-2B | 0.4327 | 0.4256 |
| Vanilla-5B | 0.4305 | 0.4329 |
| TokenDistil-5B | 0.4109 | 0.4160 |
| TokenDistil-3.5B | 0.4028 | 0.4121 |
| TokenDistil-2B | 0.3961 | 0.4049 |
| ReTok-3.5B | 0.3770 | 0.3772 |
| ReTok-2B | 0.3685 | 0.3731 |
| TokenDistil-Init | 0.2939 | 0.2915 |
| Centroid-2B | 0.2824 | 0.2796 |

Native MCQ is the cleanest Greek-specific signal in this suite. It says:

- Apertus-Base is still highest overall.
- Vanilla is the strongest continued arm at every matched continuation point
  we tested.
- TokenDistil improves steadily from init -> 2B -> 3.5B -> 5B, but at 5B it
  remains below Vanilla-5B by 1.96 pp and below Vanilla-3.5B by 2.61 pp.
- ReTok trails TD at shared checkpoints; Centroid is not competitive.

## Native MCQ per task

| Task | Best overall | Best continued arm | TD-5B | Vanilla-5B | Reading |
|---|---:|---:|---:|---:|---|
| GreekMMLU | Apertus-Base 0.5280 | Vanilla-5B 0.4747 | 0.4693 | 0.4747 | Vanilla-5B +0.55 pp over TD-5B |
| ILSP Medical MCQA | Apertus-Base 0.4097 | Vanilla-2B/3.5B 0.3472 | 0.3009 | 0.3333 | Vanilla materially ahead |
| ILSP ASEP MCQA | Apertus-Base 0.5075 | Vanilla-3.5B 0.4967 | 0.4625 | 0.4833 | Vanilla materially ahead |
| Plutus QA | Apertus-Base 0.5156 | TD-3.5B / Vanilla-5B 0.4400 | 0.4311 | 0.4400 | TD is competitive, not ahead at 5B |

## greek-nlp supporting aggregate

| Model | greek-nlp supporting mean |
|---|---:|
| Apertus-Base | 0.2150 |
| Vanilla-2B | 0.1978 |
| Vanilla-3.5B | 0.1952 |
| TokenDistil-3.5B | 0.1838 |
| TokenDistil-2B | 0.1750 |
| TokenDistil-5B | 0.1733 |
| Vanilla-5B | 0.1679 |
| TokenDistil-Init | 0.1664 |
| ReTok-2B | 0.1577 |
| ReTok-3.5B | 0.1537 |
| Centroid-2B | 0.1388 |

This aggregate is supporting evidence, not the main selection score. It mixes
heterogeneous generation/tagging metrics and uses sample-100 tasks. It is still
useful as a sanity check:

- Apertus-Base is again highest.
- Vanilla-2B/3.5B are the strongest continued arms.
- TD-5B does not dominate TD-3.5B or Vanilla; the supporting score weakens
  from TD-3.5B to TD-5B.
- Some upstream tasks are not discriminative under this base-model prompting:
  intent classification is 0 for all models; legal classification is 0 for all
  continued models. Treat those task rows as diagnostic failures of the current
  generative prompt/scoring setup, not strong model-ranking evidence.

## Non-Greek regression context

The existing 5B lm-eval results still matter for retention:

- TD-5B beats Vanilla-5B on English retention aggregate by +1.04 pp.
- TD-5B beats Vanilla-5B on multilingual aggregate by +0.40 pp.
- TD-5B wins 5/6 English retention tasks, losing only PIQA.
- Vanilla-5B keeps the better heldout BPB: 0.4602 vs TD-5B 0.4872.

So the updated picture is not "TD is bad." It is:

- TD is better than Vanilla for the old fallback downstream/retention bundle.
- Vanilla is better than TD on the new vetted native-Greek MCQ headline.
- Apertus-Base remains above both continued arms on native Greek tasks.

## Decision

For a Greek-native continuation decision, the native suite changes the
conclusion:

**Do not call TokenDistil the Greek winner. Vanilla is the safer Greek-native
choice among continued arms, and Apertus-Base remains the native-Greek ceiling
we have not recovered.**

TD remains interesting because its native MCQ trajectory is positive:

| TD point | Native MCQ general |
|---|---:|
| Init | 0.2939 |
| 2B | 0.3961 |
| 3.5B | 0.4028 |
| 5B | 0.4109 |

But the slope is not enough to justify declaring a near-term crossover. From
3.5B to 5B, TD gains +0.81 pp on native MCQ while Vanilla drops -0.65 pp. If
that local slope continued and Vanilla were frozen at 5B, TD would need roughly
3.6B more tokens to catch Vanilla-5B on the MCQ aggregate; catching Apertus-Base
would require far more and is not a realistic short continuation target.

Practical recommendation:

- Rule out Centroid and ReTok for this decision.
- Treat Vanilla as the native-Greek-safe continuation arm.
- Keep TD as a secondary candidate only if English/multilingual retention and
  the old fallback downstream aggregate are weighted above the native-Greek MCQ
  headline.
- Do not spend more compute on TD solely to prove a Greek-native win unless we
  first decide that an additional multi-billion-token continuation is worth
  the cost and the target is explicitly "can TD catch Vanilla on native MCQ,"
  not "which arm is best now."

## Remaining caveats

- OYXOY, GreekBarBench, civics, lyceum math, and other open/judge/exact-answer
  native tasks are cached or identified but not yet scored. They should not
  enter the headline until the adapters and scoring protocols are explicit.
- Gated ILSP datasets remain unavailable pending manual access.
- The `greek-nlp/benchmark` supporting mean is sample-100 and mixes metrics; use
  per-task rows for interpretation.
- The older fallback Greek lm-eval aggregate was useful historically, but it is
  no longer the Greek headline once this native suite is available.
