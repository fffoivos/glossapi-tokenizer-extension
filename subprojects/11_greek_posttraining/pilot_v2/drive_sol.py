#!/usr/bin/env python3
"""Drive both stages of both samples through Sol (codex, gpt-5.6-sol, medium)."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from runner import run_sol, load_json, stage2_input

WORKERS = 4
MODEL = 'sol'


def batches(sample):
    return json.load(open(f'{D}/v2/batches/{sample}_manifest.json'))


def out_path(sample, stage, tag):
    return f'{D}/v2/{sample}/{MODEL}/stage{stage}_{tag}.json'


def do(sample, stage, tag):
    op = out_path(sample, stage, tag)
    if os.path.exists(op):
        try:
            n = len(load_json(op))
            return tag, stage, 'cached', n, ''
        except Exception:
            pass  # unparseable -> redo
    txt = (open(f'{D}/v2/batches/{tag}_s1.txt').read() if stage == 1
           else stage2_input(tag, MODEL, sample))
    t0 = time.time()
    try:
        rc, err = run_sol(txt, op, timeout=2400)
    except Exception as e:
        return tag, stage, 'timeout', 0, str(e)[:200]
    if rc != 0:
        return tag, stage, f'rc={rc}', 0, err[-200:]
    try:
        n = len(load_json(op))
    except Exception as e:
        return tag, stage, 'unparseable', 0, str(e)[:200]
    return tag, stage, f'{time.time()-t0:.0f}s', n, ''


def run_stage(sample, stage):
    tags = [b['tag'] for b in batches(sample)]
    print(f'\n=== sample {sample} stage {stage}: {len(tags)} batches', flush=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(do, sample, stage, t): t for t in tags}
        for f in as_completed(futs):
            tag, st, status, n, err = f.result()
            mark = 'OK ' if n else 'FAIL'
            if n:
                ok += 1
            print(f'  {mark} {tag:24s} s{st} {status:12s} rows={n} {err}', flush=True)
    print(f'=== sample {sample} stage {stage}: {ok}/{len(tags)} batches produced rows', flush=True)
    return ok, len(tags)


if __name__ == '__main__':
    samples = sys.argv[1:] or ['A', 'B']
    for s in samples:
        os.makedirs(f'{D}/v2/{s}/{MODEL}', exist_ok=True)
        run_stage(s, 1)
        run_stage(s, 2)
    print('\nSOL DRIVER DONE', flush=True)
