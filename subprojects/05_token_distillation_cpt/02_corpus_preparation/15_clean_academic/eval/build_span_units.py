#!/usr/bin/env python3
"""Unified bibliography-SPAN annotation sampler — greek_phd + openarchives + Kallipos.

Targets ~2000 bibliography spans for an Opus span+metadata dataset. Each unit is a line-numbered
document window (TRUE offsets). Per document: a TAIL window (catches the end-matter bibliography) +
BODY windows around entry-density spikes (catch end-of-chapter bibliographies → multi-span). Kallipos
is reconstructed from its section rows (concatenate per filename in id order) into a document, matching
the production selected-parquet which is doc-level. Annotation is blind; the detector hypothesis is
NOT shown (manifest keeps it for scoring).
"""
import argparse, glob, io, json, os, re, subprocess, collections, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
SHARDS = {"greek_phd": "/home/foivos/data/glossapi_raw/mozilla/greek_phd/phd-theses-corpus/contents/*.jsonl.zst",
          "openarchives": "/home/foivos/data/glossapi_raw/hf/openarchives.gr/data/openarchives/*/*.jsonl.zst"}
KALLIPOS = "/home/foivos/data/glossapi_raw/hf/Apothetirio_Kallipos/Dataset_Kallipos.parquet"
IDK = ["id", "source_doc_id", "doc_id"]; TXK = ["text", "document", "content"]
YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b"); PAGE = re.compile(r"\b\d+\s*[-–]\s*\d+\b|σσ?\.\s*\d|pp?\.\s*\d")
NAME = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]{2,}"); AUTH = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,\s*[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ.]+")
PLACE = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]"); LOC = ["σσ.", "σ.", "τ.", "εκδ.", "επιμ.", "vol.", "pp.", "eds."]


def is_entry(l):
    return bool(NAME.search(l) and (YEAR.search(l) or PAGE.search(l) or AUTH.search(l) or PLACE.search(l) or any(x in l.lower() for x in LOC)))


def pick(d, ks):
    for k in ks:
        if isinstance(d.get(k), str):
            return d[k]
    return None


def windows_for(text, tail, body, max_body=2):
    lines = text.split("\n"); N = len(lines)
    ne = [i for i in range(N) if lines[i].strip()]
    out = []
    if not ne:
        return out
    # tail
    tail_lo = ne[max(0, len(ne) - tail)]
    out.append(("tail", tail_lo, N))
    # body entry-density spikes in the first 80% (non-overlapping tail)
    body_end = int(N * 0.8)
    dens = []
    blk = 60
    for b in range(0, min(body_end, tail_lo), blk):
        seg = [lines[i] for i in range(b, min(b + blk, N)) if lines[i].strip()]
        if len(seg) >= 8:
            d = sum(1 for l in seg if is_entry(l)) / len(seg)
            if d >= 0.4:
                dens.append((d, b))
    for _, b in sorted(dens, reverse=True)[:max_body]:
        lo = max(0, b - 20); hi = min(N, b + body)
        if lo < tail_lo:
            out.append(("body", lo, hi))
    return out


def emit(units, man, uid_ctr, src, doc_id, text, tail, body):
    lines = text.split("\n")
    for wtype, lo, hi in windows_for(text, tail, body):
        seg = [(i, lines[i]) for i in range(lo, hi) if lines[i].strip()]
        if len(seg) < 15:
            continue
        uid = f"S{uid_ctr[0]:05d}"; uid_ctr[0] += 1
        numbered = "\n".join(f"L{i:05d}: {ln}" for i, ln in seg)
        units.append({"unit_id": uid, "source": src, "window": wtype, "text_numbered": numbered})
        man.append({"unit_id": uid, "doc_id": doc_id, "source": src, "window": wtype, "win_lo": lo, "win_hi": hi})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=480); ap.add_argument("--tail", type=int, default=420)
    ap.add_argument("--body", type=int, default=320); ap.add_argument("--batch", type=int, default=14)
    a = ap.parse_args()
    units, man = [], []; ctr = [0]

    # greek_phd + openarchives from shards
    for src in ("greek_phd", "openarchives"):
        ids = sorted({json.loads(l)["doc_id"] for l in open(f"{os.path.dirname(HERE)}/out/{src}_full/{src}.counters.jsonl") if l.strip()})
        step = max(1, len(ids) // a.per_source); want = set(ids[::step][:a.per_source]); got = {}
        for fp in sorted(glob.glob(SHARDS[src])):
            pr = subprocess.Popen(["zstd", "-dc", fp], stdout=subprocess.PIPE)
            for raw in io.TextIOWrapper(pr.stdout, encoding="utf-8"):
                if not raw.strip():
                    continue
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                did = pick(d, IDK)
                if did in want and did not in got:
                    got[did] = pick(d, TXK) or ""
            pr.wait()
            if len(got) >= len(want):
                break
        for did, t in got.items():
            emit(units, man, ctr, src, did, t, a.tail, a.body)
        print(f"{src}: {len(got)} docs", flush=True)

    # Kallipos: reconstruct docs from sections (concatenate per filename in id order)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(KALLIPOS)
    cols = [c for c in ["filename", "id", "section"] if c in [f.name for f in pf.schema_arrow]]
    cur = None; buf = []; ndoc = 0
    # sample ~per_source filenames deterministically by hashing filename
    def keep(fn):
        return int(hashlib.md5(str(fn).encode()).hexdigest(), 16) % 11 == 0  # ~1/11 of ~4800 ≈ 440
    for b in pf.iter_batches(batch_size=20000, columns=cols):
        d = b.to_pydict(); fns = d["filename"]
        for i in range(len(fns)):
            fn = fns[i]
            if fn != cur:
                if cur is not None and keep(cur) and buf:
                    emit(units, man, ctr, "kallipos", str(cur), "\n\n".join(buf), a.tail, a.body); ndoc += 1
                cur = fn; buf = []
            buf.append(d.get("section", [""] * len(fns))[i] or "")
    if cur is not None and keep(cur) and buf:
        emit(units, man, ctr, "kallipos", str(cur), "\n\n".join(buf), a.tail, a.body); ndoc += 1
    print(f"kallipos: {ndoc} reconstructed docs", flush=True)

    udir = f"{HERE}/units/SPAN"; os.makedirs(udir, exist_ok=True)
    for f in os.listdir(udir):
        os.remove(os.path.join(udir, f))
    open(f"{HERE}/units/SPAN_manifest.jsonl", "w").write("\n".join(json.dumps(m, ensure_ascii=False) for m in man))
    paths = []
    for bi in range(0, len(units), a.batch):
        p = f"{udir}/batch_{bi // a.batch:04d}.json"; json.dump(units[bi:bi + a.batch], open(p, "w", encoding="utf-8"), ensure_ascii=False); paths.append(p)
    json.dump(paths, open(f"{HERE}/units/SPAN_batchpaths.json", "w"), indent=1)
    print(f"\nTOTAL {len(units)} windows in {len(paths)} batches; by source={collections.Counter(m['source'] for m in man)}; "
          f"by window={collections.Counter(m['window'] for m in man)}")


if __name__ == "__main__":
    main()
