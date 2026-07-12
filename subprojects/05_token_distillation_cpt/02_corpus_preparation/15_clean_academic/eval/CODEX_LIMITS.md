# gpt-5.5-via-codex: historical LLM-silver annotation notes

The completed labels described below are LLM silver, not human gold. This is an
archived historical record: do not execute the annotation commands below. The
raw joint artifact was later recovered in a checksum-locked external handoff;
use its importer rather than this historical generation workflow. No replacement
2,000-item annotation run is requested or planned.

Plan B annotates with **gpt-5.5 through the codex CLI** (Opus weekly usage was ~82% on 2026-06-18 and
won't finish 2000 before the Jun 22 reset). This is the operating note: how to read the subscription's
limits, how the runner paces, and the measured per-call cost. The runner is `run_codex_annotate.py`; it
reuses the forum wrapper `…/greek-forum-reasoning-traces/scripts/synthesis/codex_runner.py`.

## 0. Auth precondition (DO THIS FIRST)
`codex login status` can say *"Logged in using ChatGPT"* while the **usage endpoint token is expired**.
Observed 2026-06-18:
```
fetch_status() → HTTP 401 {"code":"token_expired"}
```
So before anything, in the session prompt run:
```
!codex login
```
Then confirm the limits read works (no 401):
```
python -c "import sys; sys.path.insert(0,'/home/foivos/Projects/greek-forum-reasoning-traces/scripts/synthesis'); \
import codex_runner as C, json; print(json.dumps(C.slim_status(C.fetch_status()), indent=2))"
```
You want a dict with `primary` (the 5-hour window) and `secondary` (the 7-day window), each carrying
`used_pct` / `reset_after_seconds` / `reset_at`. If it still 401s, the login didn't refresh the API token.

## 1. The two windows
`codex_runner.fetch_status()` hits `https://chatgpt.com/backend-api/codex/usage` (same data the TUI
`/status` shows) and returns:
- **primary_window** — the rolling **5-hour** budget (`limit_window_seconds≈18000`).
- **secondary_window** — the rolling **7-day** budget (`limit_window_seconds≈604800`).

There is no separate per-call quota query — the only per-call signal is the `tokens used N` line codex
writes to stderr, which the wrapper parses into the ledger. So we **anchor on the windows** at batch
boundaries and **detect limit errors per call** from stderr patterns (`rate limit`, `usage limit reached`,
`UsageLimitReached`, `too many requests`, …).

## 2. How the runner stays inside the limits (the safety contract)
`run_codex_annotate.py` is ONE sequential, resumable, foreground process (never a loop / cron / fan-out):
- **Resumable** — skips any doc whose `ann_<i>.json` exists; stop+restart re-spends nothing.
- **Limit backoff** — on `res.limit_exceeded` it calls `fetch_status()`, sleeps `primary.reset_after_seconds
  + 30s` (or 300s fallback), and **retries the same doc** (up to `--max-backoffs`, default 6, then aborts
  cleanly → rerun later).
- **Pacing** — `--horizon-hours H` spreads N pending docs at `~H·3600/N` s/doc (sleeps the remainder after
  each call), so we trickle load across the 5-hour windows instead of bursting.
- **Audit** — `record_batch_status("pre"/"post")` + a `mid_*` snapshot every `--status-every` docs write the
  window state into the local ledger `units/<run>/codex_usage.jsonl`.

## 3. Pacing math (fill the measured numbers from the bake-off)
After the 10-doc bake-off, read the ledger for per-call `wall_ms` and `tokens_used`:
```
python -c "import json; rows=[json.loads(l) for l in open('units/BAKEOFF_gpt55/codex_usage.jsonl')]; \
import statistics as s; t=[r['tokens_used'] for r in rows if r.get('tokens_used')]; \
w=[r['wall_ms'] for r in rows if r.get('wall_ms')]; \
print('median tok', int(s.median(t)), 'median wall_s', round(s.median(w)/1000,1), 'n', len(t))"
```
Then:
- **tokens for 2000** ≈ `median_tok × 2000`. Compare to the weekly (secondary) budget headroom shown in
  `fetch_status()`. If 2000 at the **full** front+tail (~118k tok/doc → ~180M total) overruns the weekly
  window, rebuild the units shrunk: `build_annotation_units.py --out STRUCT_2K --total 2000 --front 80000
  --tail 140000 --whole 220000` (~80k tok/doc, ~30% less) — one cheap local rerun.
- **wall-clock floor** ≈ `median_wall_s × 2000` (sequential). If that already exceeds 12–24h, raise effort
  efficiency (`--effort low`) and/or shrink windows; pacing only adds idle time, never removes it.
- **horizon** — set `--horizon-hours` to the target (12–24). If the wall-clock floor is *below* the horizon,
  pacing spreads the load (kinder to the 5h window); if *above*, drop pacing (`--horizon-hours 0`) and let it
  run flat-out, limited only by backoff.

### Measured 2026-06-18 (10-doc bake-off, effort=medium, full 120k/200k/320k windows)
| measure | value | source |
|---|---|---|
| tokens / doc | median **118.8k**, mean 117.6k, range 11k–184k | bake-off ledger (10 calls = 1.18M tok) |
| wall-clock / doc | median **20.0s**, mean 23.5s | bake-off ledger |
| **time for 2000** (sequential) | **~11–13h** (mean 23.5s × 2000) | fits the 12–24h window with NO pacing |
| 5h (primary) window moved | **+2 pts** (7→9%) over 10 docs — but resets every 5h, not the binding cap | pre/post status |
| 7d (secondary) window moved | **+1 pt** (17→18%) over 10 docs → ~**1000 docs per full weekly window** | pre/post status |
| weekly headroom at start | ~82% free (18% used) → **~800 docs fit now** at full size | `fetch_status().secondary` |

**The binding constraint is the WEEKLY budget, not time or quality.** At full window size only ~800 of the
2000 fit in the current weekly window. To do all 2000, either (a) **shrink windows** (`--front 80000
--tail 140000 --whole 220000`, ~30% cheaper — bib/toc sit in front+tail so recall is preserved) and/or
`--effort low`, IF the weekly meter is token-based; or (b) **spread across weekly resets** (~800/week; the
runner is resumable, so it annotates until the cap stops it cleanly, then resume after reset) — the only
option if the meter is request-based. A 50–100-doc pilot disambiguates token- vs request-based and pins the
exact per-doc weekly cost. Wall-clock is NOT the limiter: 2000 ≈ 11–13h.

**Quality verdict:** gpt-5.5 vs Opus on the 10 — accuracy 0.984, Cohen κ=0.937, **perfect recall, exact
boundaries (Δ=0, 0 prose-eaten on matched spans)**, and it recovered 7 real bibliographies (3 author-CV +
4 chapter bibs) that Opus had MISSED. gpt-5.5 is validated as ≥ Opus for this task.

### Measured 2026-06-18 (100-doc representative tranche, effort=medium, full windows) — RESOLVES the budget Q
| measure | value |
|---|---|
| time | **100 docs in 36m16s**, avg 21.7s/doc → **2000 ≈ 12h compute** |
| tokens / doc | median **85.5k**, mean 77.7k (7.77M for 100) → 2000 ≈ **~155M tok** |
| **7-day (weekly) window** | 18% → 20% over 100 docs = **~+2 pts / 100** → 2000 ≈ **+40 pts → ~58% of weekly** |
| **5-hour window** | 15% → 28% over 100 docs = **~+13 pts / 100** (≈ +0.36 pt/min) |

**Conclusion: all 2000 FIT in the current weekly budget** (~58% projected, leaving ~40% headroom for your
own codex use) — the weekly cap is NOT the blocker. The **5-hour window is what paces the run**: at +0.36
pt/min it fills in ~4.6h of work (~770 docs), so the runner will hit the 5h cap, auto-back-off until it
resets (rolling 5h), and resume — no manual pacing needed. **Total wall-clock for 2000 ≈ 13–18h** (12h
compute + 5h-window backoff gaps), inside the 12–24h target. No window-shrink or multi-week spreading
needed. The transient 403s on `/usage` are telemetry-only; the runner reads per-call `tokens used` + stderr
limit patterns regardless, and `record_batch_status` already degrades gracefully on a 403.

## 4. Run command (after go/no-go)
```
# bake-off (separate out-dir so Opus annotations in STRUCT2_FT are untouched):
python run_codex_annotate.py --in-dir units/STRUCT2_FT --out-dir units/BAKEOFF_gpt55 --effort medium
python score_engine_agreement.py            # → go/no-go

# full run, resumable + self-protecting (rerun the same line to continue after any stop/backoff/floor-stop):
python run_codex_annotate.py --in-dir units/STRUCT_2K --shuffle-seed 20260618 --effort medium --weekly-floor 30
```
Two guards are built in: (1) **5h rate-limit** → backoff (capped 6h via `backoff_seconds`) + auto-resume;
(2) **`--weekly-floor 30`** → every `--status-every` docs it reads BOTH windows and **stops when weekly(7d)
used ≥ (100−floor−margin)** = "do as many as fit, keep ≥30%, resume the rest after the weekly reset"
(NOT a projection-stop — that wrongly halts on resume). `--shuffle-seed` keeps any interrupted prefix a
balanced source mix.

### TWO independent windows — track BOTH (corrected 2026-06-19)
- **5h (primary)**: ~**0.13 pts/doc** → the one you hit FASTEST. ~770 docs fills it; sustainable rate ≈23s/doc.
  Use **`--min-interval-s 25`** to pace to it (don't burst → no backoff; smoother, same throughput).
- **7d (weekly/secondary)**: ~**0.026 pts/doc** (measured, upper-bound — includes the user's own codex use).
  This is the budget the `--weekly-floor` protects. The guard watches the SHARED meter, so the user's own
  codex use just stops our run sooner.
Full current run command (resumable; rerun verbatim to continue after any stop/backoff/floor-stop/reset):
```
python run_codex_annotate.py --in-dir units/STRUCT_2K --shuffle-seed 20260618 --effort medium \
  --weekly-floor 30 --weekly-margin 2 --status-every 15 --min-interval-s 25
```
Babysit externally with a heartbeat loop tailing the `┊ 5h=…% · weekly(7d)=…%` line from the run log.
Verify on completion: `ls units/STRUCT_2K/ann_*.json | wc -l == 2000`, then `build_gold_from_ann.py
units/STRUCT_2K`.
