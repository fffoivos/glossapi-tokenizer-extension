#!/usr/bin/env python3
"""Append ~5h of UNSEEN documents to the span-annotation queue.

Same windowing as build_span_units, but samples documents DISJOINT from the existing manifest:
greek_phd/openarchives via a fresh stride over the not-yet-sampled doc ids; Kallipos via a different
hash residue. Continues unit-id + batch numbering and writes new batch files, then APPENDS atomically
(temp + os.replace) to SPAN_manifest.jsonl + SPAN_batchpaths.json so the running loop flows straight
into them without ever reading a half-written file.
"""
import argparse, glob, io, json, os, sys, subprocess, collections, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_span_units as B   # reuse SHARDS, KALLIPOS, IDK, TXK, pick, windows_for, is_entry, emit


def atomic_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=160)
    ap.add_argument("--tail", type=int, default=420)
    ap.add_argument("--body", type=int, default=320)
    ap.add_argument("--batch", type=int, default=14)
    a = ap.parse_args()

    man_lines = [json.loads(l) for l in open(f"{HERE}/units/SPAN_manifest.jsonl") if l.strip()]
    seen = collections.defaultdict(set)
    maxctr = -1
    for m in man_lines:
        seen[m["source"]].add(m["doc_id"])
        maxctr = max(maxctr, int(m["unit_id"][1:]))
    ctr = [maxctr + 1]
    batchpaths = json.load(open(f"{HERE}/units/SPAN_batchpaths.json"))
    next_batch = len(batchpaths)
    units, man = [], []

    # greek_phd + openarchives: fresh stride over the UNSEEN doc ids
    for src in ("greek_phd", "openarchives"):
        ids = sorted({json.loads(l)["doc_id"] for l in open(f"{os.path.dirname(HERE)}/out/{src}_full/{src}.counters.jsonl") if l.strip()})
        fresh = [i for i in ids if i not in seen[src]]
        step = max(1, len(fresh) // a.per_source)
        want = set(fresh[::step][:a.per_source])
        got = {}
        for fp in sorted(glob.glob(B.SHARDS[src])):
            pr = subprocess.Popen(["zstd", "-dc", fp], stdout=subprocess.PIPE)
            for raw in io.TextIOWrapper(pr.stdout, encoding="utf-8"):
                if not raw.strip():
                    continue
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                did = B.pick(d, B.IDK)
                if did in want and did not in got:
                    got[did] = B.pick(d, B.TXK) or ""
            pr.wait()
            if len(got) >= len(want):
                break
        for did, t in got.items():
            B.emit(units, man, ctr, src, did, t, a.tail, a.body)
        print(f"{src}: +{len(got)} unseen docs", flush=True)

    # Kallipos: different residue (md5 % 30 == 7), excluding already-sampled filenames
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(B.KALLIPOS)
    cols = [c for c in ["filename", "id", "section"] if c in [f.name for f in pf.schema_arrow]]
    cur = None; buf = []; ndoc = 0

    def keep(fn):
        return int(hashlib.md5(str(fn).encode()).hexdigest(), 16) % 30 == 7 and str(fn) not in seen["kallipos"]

    for b in pf.iter_batches(batch_size=20000, columns=cols):
        d = b.to_pydict(); fns = d["filename"]
        for i in range(len(fns)):
            fn = fns[i]
            if fn != cur:
                if cur is not None and keep(cur) and buf:
                    B.emit(units, man, ctr, "kallipos", str(cur), "\n\n".join(buf), a.tail, a.body); ndoc += 1
                cur = fn; buf = []
            buf.append(d.get("section", [""] * len(fns))[i] or "")
    if cur is not None and keep(cur) and buf:
        B.emit(units, man, ctr, "kallipos", str(cur), "\n\n".join(buf), a.tail, a.body); ndoc += 1
    print(f"kallipos: +{ndoc} unseen docs", flush=True)

    # write new batch files (continue numbering), then APPEND atomically to manifest + batchpaths
    udir = f"{HERE}/units/SPAN"
    newpaths = []
    for bi in range(0, len(units), a.batch):
        idx = next_batch + bi // a.batch
        p = f"{udir}/batch_{idx:04d}.json"
        json.dump(units[bi:bi + a.batch], open(p, "w", encoding="utf-8"), ensure_ascii=False)
        newpaths.append(p)
    old_man = open(f"{HERE}/units/SPAN_manifest.jsonl").read().rstrip("\n")
    atomic_write(f"{HERE}/units/SPAN_manifest.jsonl",
                 old_man + "\n" + "\n".join(json.dumps(m, ensure_ascii=False) for m in man))
    atomic_write(f"{HERE}/units/SPAN_batchpaths.json", json.dumps(batchpaths + newpaths, indent=1))
    print(f"\nEXTENSION: +{len(units)} windows in {len(newpaths)} batches "
          f"(now {len(batchpaths) + len(newpaths)} total); by source={dict(collections.Counter(m['source'] for m in man))}")


if __name__ == "__main__":
    main()
