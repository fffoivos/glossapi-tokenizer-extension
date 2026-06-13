#!/usr/bin/env python3
"""Build the SCALED Eval-B annotation set (~2000 β sections).

Stratified sampling with recorded inclusion WEIGHTS (so corpus-level precision/recall are
unbiased via reweighting), oversampling the decision boundary (bib_weak / kept_hasyear) where
errors live, with a FROZEN 30% held-out TEST split (stratified) so tuning never touches the test.

Writes:
  units/B_scale/batch_NNN.json   — [{unit_id, header_line, section_text}]  (≤BATCH per file)
  units/B_scale_manifest.jsonl   — {unit_id, source, row_id, gate_decision, stratum, weight, split}
  units/B_scale_batchpaths.json  — ["…/batch_000.json", …]   (passed to the annotation workflow as args)
"""
import argparse, json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARQUET = {
    "kallipos": "/home/foivos/data/glossapi_raw/hf/Apothetirio_Kallipos/Dataset_Kallipos.parquet",
    "pergamos": "/home/foivos/data/glossapi_raw/hf/Apothetirio_Pergamos/sections_with_annotation_n_metadata.parquet",
}
TRIG = re.compile(r"yc=(\d+)\s+yd=([\d.]+)\s+latin=([\d.]+)\s+dash=([\d.]+)\s+pos=([\d.]+)")
# target allocation per stratum (oversamples the boundary strata bib_weak / kept_hasyear)
ALLOC = {"bib_strong": 700, "bib_weak": 400, "kept_hasyear": 500, "kept_noyear": 400}


def stratum(gate, f):
    bib = gate.startswith("bib"); strong = (f["yc"] > 0 and (f["latin"] > 0.15 or f["dash"] > 0.3))
    if bib and strong: return "bib_strong"
    if bib: return "bib_weak"
    if f["yc"] > 0: return "kept_hasyear"
    return "kept_noyear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--test-frac", type=float, default=0.30)
    a = ap.parse_args()

    pool = collections.defaultdict(list)
    for src in ("kallipos", "pergamos"):
        for line in open(f"{ROOT}/out/{src}_full/refspans/{src}.spans.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d.get("kind") != "beta_section":
                continue
            m = TRIG.search(d.get("trigger", ""))
            if not m:
                continue
            f = dict(yc=int(m.group(1)), yd=float(m.group(2)), latin=float(m.group(3)),
                     dash=float(m.group(4)), pos=float(m.group(5)))
            pool[stratum(d["gated_by"], f)].append(
                dict(source=src, row_id=d["row_id"], gate_decision=d["gated_by"], features=f))

    chosen = []
    for st, recs in pool.items():
        recs = sorted(recs, key=lambda r: (r["source"], r["row_id"]))
        want = min(ALLOC.get(st, 200), len(recs))
        step = max(1, len(recs) // want)
        pick = recs[::step][:want]
        weight = len(recs) / len(pick)
        for k, r in enumerate(pick):
            r["stratum"] = st
            r["weight"] = round(weight, 3)
            r["split"] = "test" if (k % 10) < int(a.test_frac * 10) else "train"
        chosen += pick
    for i, r in enumerate(sorted(chosen, key=lambda r: (r["source"], r["row_id"]))):
        r["unit_id"] = f"S{i:05d}"

    # fetch section text by row_id
    import pyarrow.parquet as pq
    want = collections.defaultdict(dict)
    for r in chosen:
        want[r["source"]][r["row_id"]] = r
    for src, rid_map in want.items():
        pf = pq.ParquetFile(PARQUET[src])
        cols = [c for c in ["row_id", "header", "section"] if c in [f.name for f in pf.schema_arrow]]
        found = 0
        for b in pf.iter_batches(batch_size=20000, columns=cols):
            d = b.to_pydict(); rids = d["row_id"]
            for i, rid in enumerate(rids):
                if rid in rid_map:
                    rid_map[rid]["header"] = d.get("header", [""] * len(rids))[i] or ""
                    rid_map[rid]["section"] = d.get("section", [""] * len(rids))[i] or ""
                    found += 1
            if found >= len(rid_map):
                break

    chosen = [r for r in chosen if "section" in r]
    udir = f"{HERE}/units/B_scale"
    os.makedirs(udir, exist_ok=True)
    for f in os.listdir(udir):
        os.remove(os.path.join(udir, f))
    man = open(f"{HERE}/units/B_scale_manifest.jsonl", "w", encoding="utf-8")
    for r in chosen:
        man.write(json.dumps({k: r[k] for k in ("unit_id", "source", "row_id", "gate_decision",
                                                "stratum", "weight", "split")}, ensure_ascii=False) + "\n")
    man.close()

    batchpaths = []
    for bi in range(0, len(chosen), a.batch):
        grp = chosen[bi:bi + a.batch]
        recs = [{"unit_id": r["unit_id"], "header_line": r["header"],
                 "section_text": r["section"][:4000]} for r in grp]
        p = f"{udir}/batch_{bi // a.batch:03d}.json"
        json.dump(recs, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        batchpaths.append(p)
    json.dump(batchpaths, open(f"{HERE}/units/B_scale_batchpaths.json", "w"), indent=1)

    bys = collections.Counter(r["stratum"] for r in chosen)
    splits = collections.Counter(r["split"] for r in chosen)
    print(f"{len(chosen)} units in {len(batchpaths)} batches of {a.batch}")
    print("by stratum:", dict(bys), " | weights:", {st: round(pool_w, 1) for st, pool_w in
          {st: len(pool[st]) / max(1, bys[st]) for st in bys}.items()})
    print("split:", dict(splits))


if __name__ == "__main__":
    main()
