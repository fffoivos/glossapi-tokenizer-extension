# Annotation + retrain runbook

> **STATUS 2026-06-19: annotation COMPLETE.** 2000/2000 done via gpt-5.5 (steps 1–3 below). Gold at
> `units/STRUCT_2K_gold.jsonl` (train 1392 / test 608; other 78% / bib 15% / toc 6%). Weekly budget ended
> 57% used (43% left, floor protected). **Next = step 4 (retrain), not started.**


**Goal:** **2000 documents, balanced across the 3 datasets** (greek_phd / kallipos / openarchives, ~667
each), each annotated for **table_of_contents + bibliography** only (front+tail windowing on long docs);
front-matter / main-text / appendix are derived deterministically from those two anchors.

Two annotation engines:
- **Opus** workflow (`wf_struct_annotate.js`) — preferred when Opus usage allows.
- **gpt-5.5 via codex** (`run_codex_annotate.py`) — **Plan B**, used because Opus weekly usage was ~82% on
  2026-06-18 and won't finish 2000 before the Jun 22 reset. Gated by a bake-off vs Opus (step 2b).

Read the safety rules FIRST — they exist because an unattended self-firing `/loop` spawned ~1,872 agents
over a day and burned millions of tokens (memory `feedback_no_autonomous_agent_loops`).

## ⛔ Safety rules (non-negotiable — these caused the incident)
1. **NEVER run annotation as a loop / schedule / cloud routine.** No `/loop`, `ScheduleWakeup`,
   `CronCreate`, `/schedule`. Both engines are one-shot, manually-launched, foreground-tracked. (Confirm
   `CronList` empty + no cloud routine before starting.)
2. **ONE run at a time.** Launch, WAIT for completion, verify outputs, only then launch the next.
3. **Opus path:** set a turn budget (`+30000k`); the workflow's `RESERVE` budget-guard stops before it's hit.
   **gpt-5.5 path:** the runner backs off on the subscription's own limit (`fetch_status()` →
   `reset_after_seconds`) — same lesson, different provider. It is ONE sequential resumable process, not fan-out.
4. **Always resumable** — `pending_args.py` (Opus) / the runner's own skip-existing (gpt-5.5) so stop+restart
   re-spends nothing.
5. **Tight inputs only** (front+tail, char-bounded ≤320k). The old chunked path caused agent retry-storms
   (one run hit 178 agents). Do not reintroduce chunking.
6. **If a run stalls, STOP it** (`TaskStop` / kill) — never abandon it.

## Pipeline

### 1. Build units (local CPU, free) — `build_annotation_units.py`
Representative balanced sampling (seeded reservoir over the FULL stream of each source), badness>60 dropped,
leak-free split, token/cost estimate:
```
python build_annotation_units.py --out STRUCT_2K --total 2000        # balanced ~667/source (representative)
```
Knobs: `--total`, `--pool-mult 3`, `--seed`, `--front/--tail/--whole`. Run on the `.venv-hplt-review`
interpreter (it has `glossapi_rs_noise` + `pyarrow`). Prints EST TOTAL tokens — **size to budget/limits**
before annotating. To shrink ~30%: `--front 80000 --tail 140000 --whole 220000`.
(Legacy first-N-per-source mode still available via `--limit` instead of `--total`.)

### 2a. Annotate with Opus — `wf_struct_annotate.js`  (when Opus usage allows)
```
python pending_args.py units/STRUCT_2K       # -> _pending.json (only un-done jobs)
# launch ONE Workflow with wf_struct_annotate.js, args = the _pending list, after starting the msg with +Nk
```
Sequential, budget-guarded, resumable. Prompt is canonically `STRUCT_PROMPT.md` (the JS embeds a synced copy).

### 2b. Annotate with gpt-5.5 (Plan B) — `run_codex_annotate.py` + bake-off
**Precondition:** `!codex login` (the usage-endpoint token expires even while `codex login status` says
"Logged in" — see `CODEX_LIMITS.md`).
First prove gpt-5.5 matches Opus on the 10 docs we already have Opus annotations for:
```
python run_codex_annotate.py --in-dir units/STRUCT2_FT --out-dir units/BAKEOFF_gpt55 --effort medium
python score_engine_agreement.py                      # κ + bib/toc precision + boundary Δ vs Opus
# also eyeball a broader sample: annotate ~25 from STRUCT_2K → build_struct_viz.py → presentations hub
```
**Go/no-go:** proceed only if Cohen's κ is high and **bib/toc precision** is high (low prose-eaten) — recall
and boundary slack are recoverable, eating main text is not. If go:
```
python run_codex_annotate.py --in-dir units/STRUCT_2K --horizon-hours 18 --effort low
```
ONE sequential, resumable, paced, limit-aware process. Rerun the same line to continue after any backoff/stop.
Verify: `ls units/STRUCT_2K/ann_*.json | wc -l == 2000`.

### 3. Build gold (local, free) — `build_gold_from_ann.py`
```
python build_gold_from_ann.py units/STRUCT_2K        # -> units/STRUCT_2K_gold.jsonl
```
Per-line labels {0 other, 1 bibliography, 2 table_of_contents} + the doc-grouped split. Works on either
engine's `ann_*.json`.

### 4. Retrain + evaluate (local, free)
Our own model (not GlossAPI's SVM). Adapt `line_lr.py` to multi-class {0,1,2}:
- features per line from `span_signals.line_signals`; targets = step-3 labels;
- `decode_spans.py` hysteresis per class → spans; `score_span_models.py` on the test split.
- At INFERENCE gate ToC with GlossAPI's tuned limit: ToC only within `min(300 lines, 30% of doc)` (annotation
  windows are intentionally looser — see ANNOTATION_SPEC_v2 Decisions log). Full modeling design = the
  approved plan in `~/.claude/plans/declarative-bouncing-thimble.md`.

## Files
- **Build/sample:** `build_annotation_units.py` (representative units+cost), `badness_filter.py` (gate),
  `pending_args.py` (Opus resume).
- **Prompt (canonical, shared):** `STRUCT_PROMPT.md` — edit here; `wf_struct_annotate.js` embeds a synced copy.
- **Opus engine:** `wf_struct_annotate.js` (budget-guarded workflow).
- **gpt-5.5 engine (Plan B):** `run_codex_annotate.py` (reuses forum `codex_runner.py`), `CODEX_LIMITS.md`
  (auth + windows + pacing), `score_engine_agreement.py` (bake-off vs Opus).
- **Gold + viz:** `build_gold_from_ann.py`, `build_struct_viz.py`.
- **Superseded (ignore):** `build_struct_experiment.py`, `sample_new_docs.py`, `build_fronttail.py`,
  `rechunk_full.py`, `merge_chunks.py` (the chunking path).
- **Cleanup (optional, ask first):** `~/.claude/.../subagents/workflows/` holds ~354MB of stale agent
  transcripts from the incident — deletable once anything wanted is saved.
