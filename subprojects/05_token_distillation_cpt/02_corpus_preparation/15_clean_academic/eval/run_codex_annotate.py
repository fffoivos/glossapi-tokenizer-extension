#!/usr/bin/env python3
"""Annotate front+tail units with gpt-5.5 via the codex CLI — the Plan-B engine (Opus usage is exhausted).

This is the SAFE pattern (see memory feedback_no_autonomous_agent_loops): ONE sequential, resumable,
self-paced, limit-aware foreground process. NOT a /loop, NOT cron, NOT agent fan-out. It reuses the
battle-tested forum wrapper `codex_runner.run_codex` (token parsing + rate-limit detection + ledger).

What it does, per batch_*.json in --in-dir:
  prompt = output-format spec + the doc's numbered text;  rules = STRUCT_PROMPT.md (codex model_instructions
  file, with --ignore-user-config so it replaces the codex agent prompt + ~/.codex/AGENTS.md).
  model = gpt-5.5, sandbox = read-only. stdout JSON → ann_<i>.json in --out-dir (identical schema to Opus).

Safety / pacing:
  • Resumable: skips any doc whose ann_<i>.json already exists (stop+restart never re-spends).
  • Paced: spreads the run over --horizon-hours (sleep between calls); the ChatGPT subscription has its own
    weekly window, so on res.limit_exceeded it reads fetch_status() and BACKS OFF until reset, then resumes.
  • Records fetch_status() at batch boundaries (pre/post) + every --status-every docs into the local ledger.

Usage:
  # smoke (1 doc):
  run_codex_annotate.py --in-dir units/STRUCT2_FT --out-dir units/SMOKE_gpt55 --limit 1
  # bake-off (the 10 docs we have Opus annotations for, written to a separate dir):
  run_codex_annotate.py --in-dir units/STRUCT2_FT --out-dir units/BAKEOFF_gpt55
  # full run (2000), paced over 18h, resumable:
  run_codex_annotate.py --in-dir units/STRUCT_2K --horizon-hours 18 --effort low
"""
import argparse, glob, json, os, random, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT_FILE = HERE / "STRUCT_PROMPT.md"
FORUM = "/home/foivos/Projects/greek-forum-reasoning-traces/scripts/synthesis"
sys.path.insert(0, FORUM)
import codex_runner as CR  # noqa: E402


def run_codex_stdin(*, model_id, prompt, caller, instructions_file, reasoning_effort,
                    sandbox="read-only", timeout=None, poll_status=False):
    """Like CR.run_codex but pipes the prompt via STDIN (codex exec reads stdin when no PROMPT arg).
    Our front+tail docs reach 320 KB — far past Linux's 128 KB single-CLI-arg cap (Errno 7), so the
    prompt CANNOT go on argv. Reuses CR's cmd builder + _finalize (token parse, limit detect, ledger)."""
    cmd = CR._build_cmd(model_id=model_id, instructions_file=instructions_file,
                        reasoning_effort=reasoning_effort, sandbox=sandbox, extra_args=None)
    status_before = CR.slim_status(CR.fetch_status()) if poll_status else None
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[timeout after {timeout}s]"
        exit_code = -1
    wall_ms = int((time.monotonic() - t0) * 1000)
    status_after = CR.slim_status(CR.fetch_status()) if poll_status else None
    return CR._finalize(caller=caller, model_id=model_id, reasoning_effort=reasoning_effort,
                        prompt=prompt, stdout=stdout, stderr=stderr, exit_code=exit_code,
                        wall_ms=wall_ms, status_before=status_before, status_after=status_after,
                        schedule_followup=False)

OUTPUT_SPEC = """OUTPUT FORMAT — reply with ONLY this JSON object (valid JSON, no markdown fences, no commentary,
no preamble). Mark ONLY table_of_contents and bibliography sections; everything else is main text.
{ "doc_id": <from input>, "source": <from input>, "n_lines": <from input>, "mode": <from input>,
  "doc_type": "article" | "book" | "unknown",
  "sections": [ { "kind": "table_of_contents" | "bibliography",
                  "start_line": <int, true L#####>, "end_line": <int, true L#####>,
                  "is_chapter_bibliography": <bool>, "is_authors_own_works": <bool>,
                  "n_entries": <int, reference entries; 0 for ToC>, "has_header": <bool>,
                  "confidence": "high" | "medium" | "low" } ] }
If the document has no ToC and no bibliography, return "sections": []."""


def build_prompt(unit):
    return (
        f"{OUTPUT_SPEC}\n\n"
        f"doc_id={unit['doc_id']}  source={unit['source']}  n_lines={unit['n_lines']}  mode={unit['mode']}\n\n"
        "DOCUMENT (each non-blank line is prefixed \"L#####: \" with the document's TRUE line numbers; blank\n"
        "lines are omitted; if mode is front+tail the middle main-text was elided and marked with a\n"
        "\"[ ... N main-text lines elided ... ]\" line — front and tail are enough to find the ToC and bibliography):\n\n"
        f"{unit['text_numbered']}"
    )


def extract_json(stdout):
    """Pull the JSON object out of codex stdout (tolerate fences / stray prose)."""
    s = (stdout or "").strip()
    if "```" in s:  # strip a ```json … ``` fence if present
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
        if m:
            s = m.group(1).strip()
    i = s.find("{")
    if i < 0:
        return None
    depth, instr, esc = 0, False, False
    for j in range(i, len(s)):
        c = s[j]
        if instr:
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                instr = False
        else:
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[i:j + 1])
                    except Exception:
                        return None
    return None


def jobs_from(in_dir, out_dir):
    out = []
    for bp in sorted(glob.glob(f"{in_dir}/batch_*.json")):
        idx = re.search(r"batch_(\d+)\.json$", bp).group(1)
        out.append({"batch": bp, "idx": idx, "out": f"{out_dir}/ann_{idx}.json"})
    return out


def backoff_seconds(default=300, cap=21600):
    """How long to wait after a limit hit — prefer the real reset window from fetch_status().
    Capped at `cap` (6h) so a weekly-cap hit (reset ~days away) can never sleep for days; the runner
    aborts cleanly after --max-backoffs instead (resumable)."""
    st = CR.slim_status(CR.fetch_status())
    if st and "error" not in st:
        pw = (st.get("primary") or {})
        r = pw.get("reset_after_seconds")
        if isinstance(r, (int, float)) and r > 0:
            return min(int(r) + 30, cap)
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir with batch_*.json")
    ap.add_argument("--out-dir", default=None, help="where ann_*.json + ledger go (default: --in-dir)")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=0, help="cap docs this run (0 = all pending) — for smoke/bake-off")
    ap.add_argument("--horizon-hours", type=float, default=0.0, help="spread the run over N hours (0 = no pacing)")
    ap.add_argument("--shuffle-seed", type=int, default=0, help="fixed shuffle of all jobs (0=index order) → any prefix is a balanced source mix")
    ap.add_argument("--status-every", type=int, default=25)
    ap.add_argument("--weekly-floor", type=float, default=30.0, help="stop cleanly if PROJECTED end-of-run weekly(7d) headroom would dip below this %% (0=disable)")
    ap.add_argument("--weekly-burn", type=float, default=0.026, help="conservative weekly(7d) pts consumed per doc (measured 2026-06-18 ≈0.026); shown as info only")
    ap.add_argument("--weekly-margin", type=float, default=2.0, help="extra %% below the ceiling to absorb the status-every granularity + meter lag")
    ap.add_argument("--min-interval-s", type=float, default=0.0, help="floor on seconds-per-doc cycle: pace to the 5h-window sustainable rate (~25s) so we never burst the 5h budget and trigger backoff")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--max-backoffs", type=int, default=6)
    a = ap.parse_args()
    in_dir = a.in_dir if os.path.isabs(a.in_dir) else str(HERE / a.in_dir)
    out_dir = (a.out_dir if a.out_dir else in_dir)
    out_dir = out_dir if os.path.isabs(out_dir) else str(HERE / out_dir)
    os.makedirs(out_dir, exist_ok=True)
    CR.USAGE_LOG = Path(out_dir) / "codex_usage.jsonl"   # local ledger for this run

    all_jobs = jobs_from(in_dir, out_dir)
    if a.shuffle_seed:   # fixed deterministic order so every prefix is a balanced source mix + reruns continue it
        random.Random(a.shuffle_seed).shuffle(all_jobs)
    pending = [j for j in all_jobs if not os.path.exists(j["out"])]
    if a.limit:
        pending = pending[:a.limit]
    done = len(all_jobs) - len([j for j in all_jobs if not os.path.exists(j["out"])])
    print(f"⛔ NOT-A-LOOP: one sequential resumable process. model={a.model} effort={a.effort}")
    print(f"{len(all_jobs)} batches · {done} already annotated · {len(pending)} to do this run → {out_dir}")
    if not pending:
        print("nothing pending — done."); return
    if not PROMPT_FILE.exists():
        sys.exit(f"missing rules file {PROMPT_FILE}")

    interval = (a.horizon_hours * 3600 / len(pending)) if a.horizon_hours else 0.0
    interval = max(interval, a.min_interval_s)   # 5h-sustainable floor: don't go faster than the 5h window
    if interval:
        print(f"pacing: ≥{interval:.0f}s/doc (5h-sustainable ≈25s avoids bursting the 5h window) "
              f"for {len(pending)} pending")

    caller = f"struct_annotate_{Path(out_dir).name}"
    pre = CR.record_batch_status(caller, "pre")
    print(f"limits PRE: {json.dumps(pre)}")

    def win_used(st, which):
        try:
            return (st or {}).get(which, {}).get("used_pct")
        except Exception:
            return None
    def sec_used(st):
        return win_used(st, "secondary")   # 7-day / weekly window
    run_start_used = sec_used(pre)
    if a.weekly_floor:
        ceiling0 = 100 - a.weekly_floor - a.weekly_margin
        print(f"weekly-floor guard ON: do as many as fit, STOP when weekly(7d) used ≥ {ceiling0:.0f}% "
              f"(keeps ≥{a.weekly_floor:.0f}% reserve + {a.weekly_margin:.0f}% margin); rest resume after the weekly reset. "
              f"start 5h={win_used(pre,'primary')}% 7d={run_start_used}% · burn~{a.weekly_burn:.3f}/doc")

    n_ok = n_fail = 0
    total_call_ms = 0
    run_t0 = time.time()
    for k, j in enumerate(pending):
        unit = json.load(open(j["batch"], encoding="utf-8"))[0]
        prompt = build_prompt(unit)
        backoffs = 0
        while True:
            t0 = time.time()
            res = run_codex_stdin(
                model_id=a.model, prompt=prompt, caller=caller,
                instructions_file=str(PROMPT_FILE), reasoning_effort=a.effort,
                sandbox="read-only", timeout=a.timeout, poll_status=False,
            )
            if res.limit_exceeded:
                backoffs += 1
                if backoffs > a.max_backoffs:
                    print(f"\nABORT: limit still hit after {a.max_backoffs} backoffs — rerun later (resumable). "
                          f"excerpt: {res.limit_excerpt}")
                    CR.record_batch_status(caller, "post_abort")
                    sys.exit(2)
                wait = backoff_seconds()
                print(f"  [{j['idx']}] limit hit ({res.limit_excerpt!r}); backoff {backoffs}/{a.max_backoffs}, "
                      f"sleeping {wait}s …", flush=True)
                time.sleep(wait)
                continue
            break

        total_call_ms += (res.wall_ms or 0)
        obj = extract_json(res.stdout) if res.ok else None
        if obj is None:
            n_fail += 1
            print(f"  [{j['idx']}] FAIL (exit={res.exit_code} ok={res.ok} tok={res.tokens_used}) — "
                  f"no JSON; left unwritten (will retry on rerun). stderr: {res.stderr_tail[:160]}", flush=True)
        else:
            # enforce identity fields from the unit (don't trust the model to echo them)
            obj.setdefault("sections", [])
            obj["doc_id"], obj["source"] = unit["doc_id"], unit["source"]
            obj["n_lines"], obj["mode"] = unit["n_lines"], unit["mode"]
            obj["split"] = unit.get("split")
            obj["_engine"] = {"model": a.model, "effort": a.effort, "tokens_used": res.tokens_used,
                              "wall_ms": res.wall_ms}
            json.dump(obj, open(j["out"], "w", encoding="utf-8"), ensure_ascii=False)
            n_ok += 1
            nt = sum(1 for s in obj["sections"] if s.get("kind") == "table_of_contents")
            nb = sum(1 for s in obj["sections"] if s.get("kind") == "bibliography")
            print(f"  [{j['idx']}] ok {unit['source']:<12} {obj.get('doc_type','?'):<7} "
                  f"toc={nt} bib={nb}  ({res.tokens_used} tok, {res.wall_ms}ms)", flush=True)

        if (k + 1) % a.status_every == 0:
            st = CR.record_batch_status(caller, f"mid_{k+1}")
            cur = sec_used(st); prim = win_used(st, "primary")
            rem = len([jj for jj in all_jobs if not os.path.exists(jj["out"])])
            if cur is not None and a.weekly_floor:
                ceiling = 100 - a.weekly_floor - a.weekly_margin
                live = ((cur - run_start_used) / (k + 1)) if (run_start_used is not None) else 0.0
                burn = max(live, a.weekly_burn)
                proj_rem = 100 - (cur + burn * rem)   # info only: where we'd land if we did ALL remaining
                print(f"  ┊ 5h={prim}% · weekly(7d)={cur}% used (stop≥{ceiling:.0f}%) · {rem} left · "
                      f"if-all-continue→{proj_rem:.0f}% left", flush=True)
                if cur >= ceiling:   # hard ceiling: do as many as fit, then stop (rest after reset)
                    print(f"\n🛑 WEEKLY-FLOOR STOP: weekly(7d) used {cur}% ≥ {ceiling:.0f}% — keeps ≥{a.weekly_floor:.0f}% "
                          f"reserve (+{a.weekly_margin:.0f}% margin). {rem} docs left for after the weekly reset. "
                          f"Stopping cleanly (resumable: rerun to continue).", flush=True)
                    break
        if interval:
            time.sleep(max(0.0, interval - (time.time() - t0)))

    post = CR.record_batch_status(caller, "post")
    elapsed = time.time() - run_t0
    n_done = n_ok + n_fail

    def hms(s):
        s = int(s); return f"{s//3600}h{(s%3600)//60:02d}m{s%60:02d}s"
    print(f"\nDONE this run: {n_ok} ok / {n_fail} fail in {hms(elapsed)} "
          f"(wall {hms(elapsed)}, sum-of-calls {hms(total_call_ms/1000)}, "
          f"avg {total_call_ms/1000/max(n_done,1):.1f}s/doc).")
    remaining = len([j for j in all_jobs if not os.path.exists(j["out"])])
    if n_done:
        # project the remaining run from this run's observed per-doc wall-clock
        per = elapsed / n_done
        print(f"projection: ~{per:.1f}s/doc → {remaining} remaining ≈ {hms(per*remaining)} "
              f"(all {len(all_jobs)} ≈ {hms(per*len(all_jobs))}) at this rate/effort.")
    print(f"limits POST: {json.dumps(post)}")
    print(f"ledger: {CR.USAGE_LOG}")
    if remaining:
        print(f"{remaining} still pending — rerun the same command to continue (resumable).")


if __name__ == "__main__":
    main()
