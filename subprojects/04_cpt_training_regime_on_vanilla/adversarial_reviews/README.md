# 04 / adversarial_reviews — one hostile read per checkpoint

> **In one line:** every checkpoint of the Task-1 run was handed to an adversarial reviewer before its numbers were allowed into a report, and three of the run's biggest corrections came out of these six critiques.
> **Period:** 2026-05-28 → 2026-05-31 (committed in `37888147`). **Status:** complete; six reviews, all closed.
> **Parent:** [`../README.md`](../README.md)

## Why this existed

`../goal/goal.md` step 4 made a per-checkpoint adversarial review a *required* sidecar, on the same footing as the evals: run `codex exec` with `gpt-5.5` at `xhigh` reasoning against the checkpoint, its scripts, its logs, its eval outputs and the Clariden artifacts, and look for flaws in methodology, data, comparison claims and compute hygiene. Each review was fired only after the verifier reported `handoff_ready=true` for that checkpoint.

## History

| Date (UTC) | Review | Backend | What it found |
|---|---|---|---|
| 2026-05-28 20:38–20:49 | [`Vanilla-0.5B/`](Vanilla-0.5B) (iter 119) | `codex exec`, `gpt-5.5`, `xhigh` — the only codex-run review | Critical-1: the checkpoint's positional geometry (`rope_theta=500K` / `max_pos=4096`) is not Apertus-Base's (`12M` / `65536` / llama3). Also: no MCQ decontamination, 29.2 % BPB truncation, Plutus mixed into the "headline" JSON. Verdict: not yet trustworthy for interpretation. |
| 2026-05-29 00:48 | [`Vanilla-1B/`](Vanilla-1B) (iter 238) | Claude Code subagent (codex offline from `00:24Z`) | **The comma bug.** `submit_checkpoint_sidecars.sh:156` passed a comma-bearing `BENCHMARKS=` inside `--export=ALL,…`; Slurm split it, so iter 238 evaluated GreekMMLU only while reporting a 3-task headline policy. iter 119 had escaped only because it was resubmitted by hand. |
| 2026-05-29 11:42 | [`Vanilla-2B/`](Vanilla-2B) (iter 477) | Claude Code subagent | **The matched-config baseline is not a baseline.** Forcing `rope_theta=500K` onto Path-A-trained weights perturbs rather than re-anchors — Greek BPB 1.2216 against ~0.43 for the CPT checkpoints. Cleared iter 477 as the first stable-LR readout (+4.65 pp over bakeoff-Vanilla-2B). |
| 2026-05-29 23:46 | [`Vanilla-3.5B/`](Vanilla-3.5B) (iter 834) | Claude Code subagent | Confirmed the aggregate plateau (paired iter-477 vs iter-834 Δ = −0.000186, CI [−0.0123, +0.0114]) but showed it was task-level cancellation: GreekMMLU +2.09 pp and ASEP −1.92 pp, both outside zero. Reconfirmed the regime gain at a second token mark (+4.20 pp). |
| 2026-05-30 | [`Vanilla-5B/`](Vanilla-5B) (iter 1192) | Claude Code subagent | The endpoint review; computed the five load-bearing CIs from prediction JSONLs. Added the finding that `xnli_ru` (−1.85 pp) and `xnli_el` (−1.20 pp) cross below the iter-119 reading for the first time. |
| 2026-05-31 20:00 | [`Vanilla-Path-A-0.5B/`](Vanilla-Path-A-0.5B) | Claude Code subagent | Cleared the Path-A probe as a real Path-A measurement, with C1: the probe loaded the Path-B-converted Megatron init and applied Path-A geometry as runtime flags instead of converting a Path-A sibling — empirically clean here, but a real trap once vocabulary extension changes embedding shape. |

The plateau reading from the 3.5 B review was later superseded by the endpoint (iter 1192 − iter 834 = +1.84 pp, CI clear of zero), and the 3.5 B critique's own ASEP regression finding was reversed by the endpoint recovery — both recorded in [`../reports/5B_REPORT.md`](../reports/5B_REPORT.md) §5.2 and `../_archive/superseded_drafts/task1_20260601/TASK2_HANDOFF.md` §2.5.

## Outcome

- The comma bug, the matched-config downgrade and the geometry confound — the three findings that shaped the final report — all originated here, not in the run's own analysis.
- The codex → Claude Code substitution is recorded per review in `review_metadata.env` (`BACKEND="claude-code-subagent"`, `SUBAGENT_TYPE="general-purpose"`), with the same prompt template and scope; Decisions Matrix row S and 5B report §10.6 treat it as a documented substitution, not an equivalence claim.
- Reviewer coverage is uneven by design: severity counts rise from 4 criticals at 0.5 B to 3 critical / 11 major at the Path-A probe, and the critiques grow from 8 KB to 42 KB as more baselines exist to compare against.

## Where things are

| Path | What it is |
|---|---|
| `Vanilla-<label>/adversarial_critique.md` | The critique itself — verdict, criticals, majors, minors, missing evidence. |
| `Vanilla-<label>/prompt.md` | The exact prompt the reviewer was given, including the cross-reference list of prior checkpoints' unresolved findings. |
| `Vanilla-<label>/review_metadata.env` | Checkpoint label, iteration, token mark, Clariden run/eval/HF paths, backend, handoff timestamp. Note `Vanilla-5B/` has **only** the critique — no `prompt.md`, no metadata file. |
| `04_vanilla_goldfish_5b_20260528T112539Z_watch_state/` | Watcher state for the review chain: `iter_119.review_done` (a timestamp) and four `iter_<N>.pre_review_verify.json` snapshots. These are *in-flight* snapshots — e.g. the iter-1192 file was written at `2026-05-29T00:18Z`, when that checkpoint did not yet exist, and records `exists: false` throughout. |

## Working documents

Everything in this directory is historical. The `*_watch_state/` files are machine state from the local review watcher, not results; the `prompt.md` files matter only for reproducing a review. The file paths recorded inside `Vanilla-0.5B/review_metadata.env` are absolute paths from the original workstation checkout and do not resolve in this repo.
