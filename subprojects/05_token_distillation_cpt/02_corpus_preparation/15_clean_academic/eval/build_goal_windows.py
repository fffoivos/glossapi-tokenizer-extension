#!/usr/bin/env python3
"""Goal-eval, whole-doc sources (greek_phd/openarchives): sample TAIL + BODY windows per doc,
line-numbered with TRUE offsets. Opus marks every bibliographic-LIST span; we compare to the
detector's flagged bibliography lines (endmatter_bib spans) within each window. Body windows are
what catch end-of-CHAPTER lists the end-matter detector structurally misses.
"""
import argparse, glob, io, json, os, subprocess, collections, hashlib

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SHARDS = {"greek_phd": "/home/foivos/data/glossapi_raw/mozilla/greek_phd/phd-theses-corpus/contents/*.jsonl.zst",
          "openarchives": "/home/foivos/data/glossapi_raw/hf/openarchives.gr/data/openarchives/*/*.jsonl.zst"}
IDK = ["id", "source_doc_id", "doc_id"]; TXK = ["text", "document", "content"]


def pick(d, keys):
    for k in keys:
        if isinstance(d.get(k), str):
            return d[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=90); ap.add_argument("--tail", type=int, default=550)
    ap.add_argument("--body", type=int, default=450); ap.add_argument("--batch", type=int, default=12)
    a = ap.parse_args()

    # detector endmatter_bib span (line_start..line_end) per doc
    det = collections.defaultdict(list)
    for src in ("greek_phd", "openarchives"):
        for l in open(f"{ROOT}/out/{src}_full/refspans/{src}.spans.jsonl"):
            d = json.loads(l)
            if d["kind"] == "endmatter_bib":
                det[d["doc_id"]].append((d["line_start"], d["line_end"]))

    # sample docs deterministically per source
    want = collections.defaultdict(list)
    for src in ("greek_phd", "openarchives"):
        ids = sorted({json.loads(l)["doc_id"] for l in open(f"{ROOT}/out/{src}_full/{src}.counters.jsonl") if l.strip()})
        step = max(1, len(ids) // a.per_source)
        for did in ids[::step][:a.per_source]:
            want[src].append(did)

    units = []; man = []
    for src, ids in want.items():
        idset = set(ids); texts = {}
        for fp in sorted(glob.glob(SHARDS[src])):
            proc = subprocess.Popen(["zstd", "-dc", fp], stdout=subprocess.PIPE)
            for raw in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
                if not raw.strip():
                    continue
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                did = pick(d, IDK)
                if did in idset and did not in texts:
                    texts[did] = pick(d, TXK) or ""
            proc.wait()
            if len([d for d in idset if d in texts]) >= len(idset):
                break
        for did in ids:
            if did not in texts:
                continue
            lines = texts[did].split("\n"); N = len(lines)
            detlines = set()
            for (s, e) in det.get(did, []):
                detlines |= set(range(s, min(e, N) + 1))
            # tail window
            for wtype, lo, hi in [("tail", max(0, N - a.tail), N),
                                  ("body", *(lambda b0: (b0, min(N, b0 + a.body)))(int((int(hashlib.md5(did.encode()).hexdigest(), 16) % 1000) / 1000 * max(1, int(N * 0.65))))) ]:
                seg = [(idx, lines[idx]) for idx in range(lo, hi) if lines[idx].strip()]
                if len(seg) < 20:
                    continue
                uid = f"W{len(units):04d}"
                numbered = "\n".join(f"L{idx:05d}: {ln}" for idx, ln in seg)
                units.append({"unit_id": uid, "source": src, "window": wtype, "text_numbered": numbered})
                man.append(dict(unit_id=uid, doc_id=did, source=src, window=wtype, win_lo=lo, win_hi=hi,
                                detector_bib_lines_in_window=sorted(l for l in detlines if lo <= l < hi)))

    udir = f"{HERE}/units/W_windows"; os.makedirs(udir, exist_ok=True)
    for f in os.listdir(udir):
        os.remove(os.path.join(udir, f))
    open(f"{HERE}/units/W_windows_manifest.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(m, ensure_ascii=False) for m in man))
    paths = []
    for bi in range(0, len(units), a.batch):
        p = f"{udir}/batch_{bi // a.batch:03d}.json"; json.dump(units[bi:bi + a.batch], open(p, "w", encoding="utf-8"), ensure_ascii=False); paths.append(p)
    json.dump(paths, open(f"{HERE}/units/W_windows_batchpaths.json", "w"), indent=1)
    print(f"{len(units)} windows in {len(paths)} batches; types={collections.Counter(m['window'] for m in man)}; "
          f"by source={collections.Counter(m['source'] for m in man)}")


if __name__ == "__main__":
    main()
