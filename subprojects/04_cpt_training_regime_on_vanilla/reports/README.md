# 04 / reports — results, audits and the corrections behind them

> **In one line:** every analytical output of the Task-1 Vanilla 5 B run, written in four waves as the run progressed — and two of the waves exist because an earlier one was found wrong.
> **Period:** 2026-05-28 → 2026-06-01 (committed in `37888147`; `PLANNING_AGENT_REPLY` revised and `TD_LAYER_11_…` added in `a19c136f`). **Status:** complete; nothing here was updated after 2026-06-11.
> **Parent:** [`../README.md`](../README.md)

## Why this existed

The run's rule was that no headline number entered a report without a bootstrap CI behind it and an adversarial critique against it. That produced a directory where the reports and the artifacts they cite sit side by side, including the artifacts that disproved earlier reports.

## History

| Date | What landed | Why / result |
|---|---|---|
| 2026-05-28 | [`config_geometry_audit_iter_0000119.md`](config_geometry_audit_iter_0000119.md) + `.json` | Written 21:01 UTC in response to the Vanilla-0.5B critique: the run trains at `rope_theta=500K` / `max_pos=4096` while Apertus-Base ships `12M` / `65536`. Records the *then*-state of `goal/hyperparameters.json`, which itself still carried the contradiction. |
| 2026-05-28 → 05-30 | `latest_5b_report_state.json`, [`latest_5b_report_status.md`](latest_5b_report_status.md), `iter_*_checkpoint_sidecar_*.json` | Renderer and verifier output produced live during the run. The status snapshot is frozen at iter 300 / 2026-05-30T16:03Z — a mid-run state, not the result. |
| 2026-05-29 | [`decisions_matrix_20260529.md`](decisions_matrix_20260529.md), [`script_audit_20260529.md`](script_audit_20260529.md), `v4_bootstrap_cis_native_mcq.json` (v1→v2) | Triggered by the Slurm `--export` comma bug that silently reduced the iter-238/477 native MCQ to GreekMMLU only. 24 decision rows A–X; 3 critical / 7 major / 9 minor script findings. V4 v2 gave the first significant result: iter 477 **+4.65 pp** over bakeoff-Vanilla-2B, CI [+0.0299, +0.0629]. |
| 2026-05-30 | [`5B_REPORT.md`](5B_REPORT.md), [`plutus_investigation_20260530.md`](plutus_investigation_20260530.md), [`gpu_hours_breakdown_20260530.md`](gpu_hours_breakdown_20260530.md), `v4_bootstrap_cis_native_mcq.json` (v3), plots, `eval_data_cache_5b/` | The endpoint wave. V4 v3 = 10 models / 83 delta rows. The 5 B report supersedes `../_archive/superseded_drafts/5B_REPORT_DRAFT.md`. |
| 2026-05-30 (same day) | §7 of `5B_REPORT.md` rewritten | The retention table had used iter 119 as the baseline; iter 119 is 0.5 B tokens *into* training. Rebuilt against the matched-config Path-B init (true iter 0); `plot_retention_per_language.py` got the same fix. |
| 2026-05-31 | [`path_a_probe_results_20260531.md`](path_a_probe_results_20260531.md), `v4_workspace_path_a/` | Verdict on the 0.5 B Path-A probe: **CONFIRMED** — 0.4942 [0.4747, 0.5133], +5.51 pp over Path B at matched tokens. |
| 2026-06-01 | [`PLANNING_AGENT_REPLY_20260601.md`](PLANNING_AGENT_REPLY_20260601.md), [`TD_LAYER_11_SELECTION_PROVENANCE_20260601.md`](TD_LAYER_11_SELECTION_PROVENANCE_20260601.md) | The reply's §4 first claimed TD layer 11 was a heuristic pick; the provenance report **retracts that** — both `target_layer=-1` and `=11` were trained at two scales and layer 11 won on heldout BPB (0.008 pilot, 0.040 full). |

## Outcome

- Headline: iter 1192 3-task native Greek MCQ **0.4973 [0.4779, 0.5156]**, checked directly against `v4_bootstrap_cis_native_mcq.json` (`models["iter-1192-Vanilla-5B"].headline_3task`).
- Load-bearing paired deltas, all CI-outside-zero: +6.69 pp vs bakeoff-Vanilla-5B, +7.01 pp vs matched-config init, +1.56 pp vs Apertus-Base Path A, +1.82 pp vs iter 477, +1.84 pp vs iter 834; the iter-834 − iter-477 plateau (−0.0002) straddles zero.
- Plutus QA fell 0.4889 → 0.4356 at the endpoint; paired vs iter 834 CI [−0.1067, +0.0000], McNemar p = 0.081, but paired vs Apertus-Base [−0.1422, −0.0133] is outside zero. Verdict recorded as *ambiguous, leaning small-real*.
- Two claims here are **not** clean baselines by their own authors' verdict: the matched-config Apertus-Base (perturbed by the rope override — Greek BPB 1.2216) and any Apertus-Base comparison at all (Path-A base vs Path-B run).

## Where things are

| Path | What it is |
|---|---|
| [`5B_REPORT.md`](5B_REPORT.md) | The report. 14 sections; §4 trajectory CIs, §6 Plutus, §7 retention (corrected), §10 caveats, §11 Task-2 implications, §13 what was left undone. |
| `v4_bootstrap_cis_native_mcq.json` | V4 revision v3 — 1000 resamples, 95 % percentile, `rng_seed=20260529`, per-task item-level with paired indices. Authoritative for every cross-arm number in this directory. |
| [`decisions_matrix_20260529.md`](decisions_matrix_20260529.md) | Rows A–X with severity, plan ref, Apertus-paper ref, action, status. Row C/H = geometry, row D = BPB truncation, row E = decontamination, row I = the unapplied `xfer`→`normal` re-route. |
| [`path_a_probe_results_20260531.md`](path_a_probe_results_20260531.md) | The probe verdict that locked Task 2's geometry. |
| [`plutus_investigation_20260530.md`](plutus_investigation_20260530.md) | Item-level transition matrix (26 regressions / 14 gains), 2-choice 52 % flip rate, low-confidence margins. |
| [`script_audit_20260529.md`](script_audit_20260529.md) | The full quoting/Slurm audit; C1 is the comma bug (fixed `7eb4667e…` → `e865c65a…`), C2/C3 unquoted heredocs left deferred. |
| `plot_*.py` / `*.png` | Trajectory, GreekMMLU-vs-English-MMLU, lm-loss and per-language retention plots. `plot_lm_loss_trajectory.py` reads a `train_logs_cache_5b/` directory that was never committed, so it will not re-run here as written. |
| `eval_data_cache_5b/` | 2.1 MB of per-checkpoint native-MCQ + retention JSON/CSV, kept so the plots can be regenerated without pulling from Clariden. |

## Working documents

Historical; kept for provenance.

- **Status snapshots:** `latest_5b_report_status.md` + `latest_5b_report_state.json` — renderer output frozen mid-run (iter 300).
- **Verifier trail:** `iter_<N>_checkpoint_sidecar_handoff_pass.json` (moment `handoff_ready=true` was reached) and `iter_<N>_checkpoint_sidecar_verify_latest.json` (last verifier output); `iter_0000119_checkpoint_sidecar_precheck.json` is a one-off from the iter-119 conversion repair.
- **Superseded audit state:** `config_geometry_audit_iter_0000119.{md,json}` deliberately preserves the pre-fix contradiction in `goal/hyperparameters.json`; the file itself was corrected on 2026-05-29.
- **Bootstrap scratch:** `v4_workspace_path_a/` here, and the five earlier workspaces in [`../_archive/v4_workspaces/`](../_archive/v4_workspaces).
