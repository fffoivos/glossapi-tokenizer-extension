#!/usr/bin/env python3
"""ONE command to stage the front+tail annotation units for the next run. Local CPU only — NO Opus/agents.
Streams the academic sources, drops greek_badness_score>60, builds ONE char-bounded front+tail unit per doc
(whole when small), doc-grouped leak-free train/test split, and writes batch_*.json + _args.json +
manifest.jsonl + a TOKEN/COST estimate so the run can be sized to budget.

Supersedes build_struct_experiment.py / sample_new_docs.py / build_fronttail.py / rechunk_full.py.

Two sampling modes:
  --total N   REPRESENTATIVE (preferred): balanced across the sources (N//k each), seeded reservoir over the
              FULL stream of every source so the sample is not biased to the first shards. Two-stage: a cheap
              raw reservoir (no badness scoring) → a candidate pool, then badness-score only the pool and keep
              the quota with greek_badness_score<=60. greek_phd carries the badness column so it is filtered
              for free during the reservoir.
  --limit M   LEGACY: first-M-per-source (kept for back-compat; biased to early shards).

Usage:
  build_annotation_units.py --out STRUCT_2K --total 2000             # 2000 docs, balanced thirds (representative)
  build_annotation_units.py --out STRUCT_RUN --limit 150             # legacy first-150/source
  [--front 120000 --tail 200000 --whole 320000]                      # char windows (tune for cost)
  [--pool-mult 3 --seed 20260618]                                    # pool oversample + reproducibility
"""
import json, os, glob, io, subprocess, hashlib, argparse, random
import build_span_units as B
import badness_filter as BF
HERE = os.path.dirname(os.path.abspath(__file__))
TOK_PER_CHAR = 1 / 2.7          # rough Greek→Claude token rate (for budgeting only)
OUT_TOK_PER_DOC = 1500          # typical structured-output size


def split_of(doc_id):           # matches span_seq_data.split_of (leak-free, md5%10<3 → test)
    return "test" if int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 10 < 3 else "train"


def numbered(text):
    lines = text.split("\n")
    return [f"L{i:05d}: {lines[i]}" for i in range(len(lines)) if lines[i].strip()], len(lines)


def take_chars(seq, budget):
    out, c = [], 0
    for s in seq:
        out.append(s); c += len(s) + 1
        if c >= budget:
            break
    return out


def front_tail(text, front_c, tail_c, whole_c):
    nl, N = numbered(text)
    total = sum(len(s) + 1 for s in nl)
    if total <= whole_c:
        return "\n".join(nl), "whole", N, total
    fr = take_chars(nl, front_c)
    tl = list(reversed(take_chars(list(reversed(nl)), tail_c)))
    elided = len(nl) - len(fr) - len(tl)
    if elided < 0:
        return "\n".join(nl), "whole", N, total
    body = "\n".join(fr) + f"\n\n[ ... {elided} main-text lines elided ... ]\n\n" + "\n".join(tl)
    return body, f"front{len(fr)}+tail{len(tl)}", N, len(body)


# ───────────────────────── representative sampling (--total) ─────────────────────────
def iter_shard_docs(source, scan_limit=0):
    """Yield (doc_id, text, badness_or_None) for every doc in a jsonl-shard source.
    greek_phd carries greek_badness_score (read for free); openarchives → None (scored later)."""
    has_col = source == "greek_phd"
    n = 0
    for fp in sorted(glob.glob(B.SHARDS[source])):
        pr = subprocess.Popen(["zstd", "-dc", fp], stdout=subprocess.PIPE)
        for raw in io.TextIOWrapper(pr.stdout, encoding="utf-8"):
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except Exception:
                continue
            did = B.pick(d, B.IDK)
            text = B.pick(d, B.TXK) or ""
            if not did or not text.strip():
                continue
            bad = BF.record_badness(d, text) if has_col else None
            yield did, text, bad
            n += 1
            if scan_limit and n >= scan_limit:
                pr.kill(); return
        pr.wait()


def iter_kallipos_docs(scan_limit=0):
    """Yield (filename, reconstructed_text, None) — sections grouped by filename."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(B.KALLIPOS)
    cols = [c for c in ["filename", "section"] if c in [f.name for f in pf.schema_arrow]]
    cur, buf, n = None, [], 0
    for batch in pf.iter_batches(batch_size=20000, columns=cols):
        dd = batch.to_pydict(); fns = dd["filename"]
        for i in range(len(fns)):
            if fns[i] != cur:
                if cur is not None and buf:
                    t = "\n\n".join(buf)
                    if t.strip():
                        yield str(cur), t, None
                        n += 1
                        if scan_limit and n >= scan_limit:
                            return
                cur = fns[i]; buf = []
            buf.append((dd.get("section", [""] * len(fns))[i]) or "")
    if cur is not None and buf:
        t = "\n\n".join(buf)
        if t.strip():
            yield str(cur), t, None


def reservoir(it, k, rng):
    """Algorithm R — uniform sample of size k from a stream of unknown length (seeded rng)."""
    res = []
    for i, x in enumerate(it):
        if i < k:
            res.append(x)
        else:
            j = rng.randint(0, i)
            if j < k:
                res[j] = x
    return res


def sample_source(source, quota, pool_mult, rng, scan_limit=0):
    """Representative, badness<=60 sample of up to `quota` docs from one source.
    Returns (kept, n_pool). kept = list of (doc_id, source, text, badness)."""
    it = iter_kallipos_docs(scan_limit) if source == "kallipos" else iter_shard_docs(source, scan_limit)
    if source == "greek_phd":
        # badness column is free → filter during the reservoir, sample exactly `quota`
        good = ((d, t, b) for d, t, b in it if not BF.is_bad(b))
        pool = reservoir(good, quota, rng)
        kept = [(d, source, t, round(b or 0, 1)) for d, t, b in pool]
        return kept, len(pool)
    # openarchives / kallipos: reservoir a raw pool (no scoring), then badness-score the pool
    pool = reservoir(it, quota * pool_mult, rng)
    rng.shuffle(pool)
    kept = []
    for d, t, _ in pool:
        b = BF.score_text(t)
        if BF.is_bad(b):
            continue
        kept.append((d, source, t, round(b or 0, 1)))
        if len(kept) >= quota:
            break
    return kept, len(pool)


def alloc(total, n):
    base = total // n
    rem = total - base * n
    return [base + (1 if i < rem else 0) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="STRUCT_RUN")
    ap.add_argument("--total", type=int, default=0)        # representative balanced sampling; 0 → use --limit
    ap.add_argument("--limit", default="150")              # legacy first-N per source, or "all"
    ap.add_argument("--pool-mult", type=int, default=3)    # candidate pool = quota × this (openarchives/kallipos)
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--scan-limit", type=int, default=0)   # DEBUG: cap docs scanned/source — BIASES the sample
    ap.add_argument("--front", type=int, default=120_000)
    ap.add_argument("--tail", type=int, default=200_000)
    ap.add_argument("--whole", type=int, default=320_000)
    ap.add_argument("--sources", default="greek_phd openarchives kallipos")
    a = ap.parse_args()
    ODIR = f"{HERE}/units/{a.out}"
    os.makedirs(ODIR, exist_ok=True)
    for f in glob.glob(f"{ODIR}/*"):
        os.remove(f)

    args, manifest = [], []
    counter = [0]

    def on_doc(did, source, text, bad):
        i = counter[0]; counter[0] += 1
        body, mode, N, chars = front_tail(text, a.front, a.tail, a.whole)
        unit = {"doc_id": did, "source": source, "n_lines": N, "mode": mode, "badness": bad,
                "split": split_of(did), "text_numbered": body}
        bp = f"{ODIR}/batch_{i:05d}.json"
        json.dump([unit], open(bp, "w", encoding="utf-8"), ensure_ascii=False)
        args.append({"batch": bp, "out": f"{ODIR}/ann_{i:05d}.json"})
        manifest.append({"i": i, "doc_id": did, "source": source, "split": unit["split"],
                         "n_lines": N, "mode": mode, "chars": chars, "badness": bad})

    srcs = a.sources.split()
    if a.total:
        quotas = alloc(a.total, len(srcs))
        rng = random.Random(a.seed)
        if a.scan_limit:
            print(f"WARNING: --scan-limit={a.scan_limit} set → sample is BIASED to early docs (testing only)")
        for s, q in zip(srcs, quotas):
            print(f"sampling {s}: target {q}  (pool ×{a.pool_mult}, seed {a.seed}) ...", flush=True)
            kept, npool = sample_source(s, q, a.pool_mult, rng, a.scan_limit)
            if len(kept) < q:
                print(f"  WARNING: {s} kept only {len(kept)}/{q} after badness filter (pool {npool}) — raise --pool-mult")
            else:
                print(f"  {s}: kept {len(kept)} (pool {npool})", flush=True)
            for did, src, text, bad in kept:
                on_doc(did, src, text, bad)
    else:
        limit = None if a.limit == "all" else int(a.limit)
        for s in srcs:
            print(f"streaming {s} (legacy first-{limit}) ...", flush=True)
            (stream_kallipos if s == "kallipos" else lambda lim, cb: stream_shards(s, lim, cb))(limit, on_doc)

    json.dump(args, open(f"{ODIR}/_args.json", "w"))
    open(f"{ODIR}/manifest.jsonl", "w").write("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest))
    # cost estimate
    tot_chars = sum(m["chars"] for m in manifest)
    in_tok = int(tot_chars * TOK_PER_CHAR)
    out_tok = len(manifest) * OUT_TOK_PER_DOC
    by_src = {}
    for m in manifest:
        by_src.setdefault(m["source"], [0, 0])
        by_src[m["source"]][0] += 1; by_src[m["source"]][1] += m["chars"]
    print(f"\n{len(manifest)} docs → {len(args)} annotation units  ({sum(1 for m in manifest if m['split']=='test')} test / {sum(1 for m in manifest if m['split']=='train')} train)")
    for s, (c, ch) in by_src.items():
        print(f"  {s:<12} {c:>5} docs  ~{int(ch*TOK_PER_CHAR)//1000:>6}k input tokens")
    print(f"\n  EST INPUT  ~{in_tok//1_000_000:.2f}M tokens   ({in_tok//1000}k)")
    print(f"  EST OUTPUT ~{out_tok//1000}k tokens")
    print(f"  EST TOTAL  ~{(in_tok+out_tok)//1_000_000:.2f}M tokens   (front={a.front} tail={a.tail} whole={a.whole})")
    print(f"\n  units → {ODIR}   (annotate with: see ANNOTATION_RUNBOOK.md)")


# ───────────────────────── legacy first-N path (--limit) ─────────────────────────
def stream_shards(source, limit, on_doc):
    has_col = source == "greek_phd"
    n = 0
    for fp in sorted(glob.glob(B.SHARDS[source])):
        pr = subprocess.Popen(["zstd", "-dc", fp], stdout=subprocess.PIPE)
        for raw in io.TextIOWrapper(pr.stdout, encoding="utf-8"):
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except Exception:
                continue
            did = B.pick(d, B.IDK)
            text = B.pick(d, B.TXK) or ""
            if not did or not text.strip():
                continue
            bad = BF.record_badness(d, text) if has_col else BF.score_text(text)
            if BF.is_bad(bad):
                continue
            on_doc(did, source, text, round(bad or 0, 1)); n += 1
            if limit and n >= limit:
                pr.kill(); return
        pr.wait()


def stream_kallipos(limit, on_doc):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(B.KALLIPOS)
    cols = [c for c in ["filename", "section"] if c in [f.name for f in pf.schema_arrow]]
    cur, buf, n = None, [], 0

    def flush(fn, b):
        nonlocal n
        if fn is None or not b:
            return
        text = "\n\n".join(b)
        if not text.strip():
            return
        bad = BF.score_text(text)
        if BF.is_bad(bad):
            return
        on_doc(str(fn), "kallipos", text, round(bad or 0, 1)); n += 1

    for batch in pf.iter_batches(batch_size=20000, columns=cols):
        dd = batch.to_pydict(); fns = dd["filename"]
        for i in range(len(fns)):
            if fns[i] != cur:
                flush(cur, buf); cur = fns[i]; buf = []
                if limit and n >= limit:
                    return
            buf.append((dd.get("section", [""] * len(fns))[i]) or "")
    flush(cur, buf)


if __name__ == "__main__":
    main()
