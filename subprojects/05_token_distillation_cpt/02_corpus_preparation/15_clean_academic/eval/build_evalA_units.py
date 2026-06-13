#!/usr/bin/env python3
"""Eval A — build bibliography-BOUNDARY annotation units for the whole-doc sources
(greek_phd, openarchives). Unit = a document's TAIL (last ~700 non-empty lines, line-numbered with
TRUE line offsets). The annotator, BLIND to the detector, independently locates the end-matter
bibliography start line (if any), what follows it (appendix/index/eof), and sub-lists — so we can
score the detector's `endmatter_bib` boundary (found/not + start-line error) for both precision AND
recall (incl. the ~20-34% header-not-found docs).

Stratifies by detector `endmatter_header_found` (Y/N) × bib-char mass, oversampling the NOT-found
docs (where the false negatives hide). Writes batch files + a hidden manifest (detector's boundary).
"""
import argparse, glob, io, json, os, re, subprocess, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHARDS = {
    "greek_phd": "/home/foivos/data/glossapi_raw/mozilla/greek_phd/phd-theses-corpus/contents/*.jsonl.zst",
    "openarchives": "/home/foivos/data/glossapi_raw/hf/openarchives.gr/data/openarchives/*/*.jsonl.zst",
}
IDK = ["id", "source_doc_id", "doc_id"]
TXK = ["text", "document", "content"]


def pick(d, keys):
    for k in keys:
        if isinstance(d.get(k), str):
            return d[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=250)
    ap.add_argument("--tail-lines", type=int, default=700)
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()

    # 1. stratified doc sample per source from the detector counters + spans
    want = {}                      # doc_id -> {source, header_found, bib_start_line, bib_chars}
    for src in ("greek_phd", "openarchives"):
        counters = [json.loads(l) for l in open(f"{ROOT}/out/{src}_full/{src}.counters.jsonl") if l.strip()]
        bibstart = {}
        for l in open(f"{ROOT}/out/{src}_full/refspans/{src}.spans.jsonl"):
            d = json.loads(l)
            if d["kind"] == "endmatter_bib":
                bibstart[d["doc_id"]] = d["line_start"]
        # strata: found-high-mass, found-low-mass, NOT-found (oversample NOT-found)
        strata = collections.defaultdict(list)
        for c in counters:
            f = c["endmatter_header_found"]; bc = c.get("endmatter_bib_chars", 0)
            key = "notfound" if not f else ("found_hi" if bc > 20000 else "found_lo")
            strata[key].append(c["doc_id"])
        alloc = {"notfound": a.per_source // 2, "found_hi": a.per_source // 4, "found_lo": a.per_source // 4}
        for key, ids in strata.items():
            ids = sorted(ids)
            n = min(alloc.get(key, 60), len(ids))
            step = max(1, len(ids) // n)
            for did in ids[::step][:n]:
                want[did] = {"source": src, "header_found": next(c["endmatter_header_found"] for c in counters if c["doc_id"] == did),
                             "bib_start_line": bibstart.get(did), "stratum": key}

    # 2. fetch tail text for sampled docs from the shards
    by_src = collections.defaultdict(set)
    for did, m in want.items():
        by_src[m["source"]].add(did)
    texts = {}
    for src, ids in by_src.items():
        files = sorted(glob.glob(SHARDS[src]))
        for fp in files:
            proc = subprocess.Popen(["zstd", "-dc", fp], stdout=subprocess.PIPE)
            for raw in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
                if not raw.strip():
                    continue
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                did = pick(d, IDK)
                if did in ids and did not in texts:
                    texts[did] = pick(d, TXK) or ""
            proc.wait()
            if len([d for d in ids if d in texts]) >= len(ids):
                break

    # 3. write line-numbered tails (TRUE offsets) as batch units + hidden manifest
    udir = f"{HERE}/units/A_boundary"
    os.makedirs(udir, exist_ok=True)
    for f in os.listdir(udir):
        os.remove(os.path.join(udir, f))
    man = open(f"{HERE}/units/A_boundary_manifest.jsonl", "w", encoding="utf-8")
    units = []
    for i, (did, m) in enumerate(sorted(want.items())):
        if did not in texts:
            continue
        lines = texts[did].split("\n")
        ne = [(idx, ln) for idx, ln in enumerate(lines) if ln.strip()]
        tail = ne[-a.tail_lines:] if len(ne) > a.tail_lines else ne
        start_global = tail[0][0] if tail else 0
        numbered = "\n".join(f"L{idx:05d}: {ln}" for idx, ln in tail)
        uid = f"A{i:04d}"
        units.append({"unit_id": uid, "source": m["source"], "tail_starts_at_line": start_global,
                      "total_lines": len(lines), "text_numbered": numbered})
        man.write(json.dumps({"unit_id": uid, "doc_id": did, "source": m["source"], "stratum": m["stratum"],
                              "detector_header_found": m["header_found"], "detector_bib_start_line": m["bib_start_line"],
                              "tail_starts_at_line": start_global}, ensure_ascii=False) + "\n")
    man.close()

    paths = []
    for bi in range(0, len(units), a.batch):
        p = f"{udir}/batch_{bi // a.batch:03d}.json"
        json.dump(units[bi:bi + a.batch], open(p, "w", encoding="utf-8"), ensure_ascii=False)
        paths.append(p)
    json.dump(paths, open(f"{HERE}/units/A_boundary_batchpaths.json", "w"), indent=1)
    print(f"{len(units)} boundary units in {len(paths)} batches; "
          f"strata={collections.Counter(m['stratum'] for m in want.values() if m['source'])}")


if __name__ == "__main__":
    main()
