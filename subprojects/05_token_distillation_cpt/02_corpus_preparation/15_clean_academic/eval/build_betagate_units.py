#!/usr/bin/env python3
"""Build Eval-B (β-gate) annotation units — stratified, blind to the gate decision.

Reads the detector's β spans (kallipos+pergamos), parses the per-section features from the
trigger, stratifies to OVERSAMPLE the decision boundary (low-signal `bib`, high-signal `kept`),
fetches the section text by row_id from the section parquets, and writes:
  units/B_betagate/unit_<id>.json   — {unit_id, source, header, section_lines} (NO gate decision)
  B_betagate_manifest.jsonl         — hidden truth-join: {unit_id, row_id, gate_decision, features}
  B_betagate_batches.json           — agent batches (list of unit_id lists) + the schema reminder

Deterministic: fixed seed via sorted row_id; no randomness.
"""
import argparse, json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARQUET = {
    "kallipos": "/home/foivos/data/glossapi_raw/hf/Apothetirio_Kallipos/Dataset_Kallipos.parquet",
    "pergamos": "/home/foivos/data/glossapi_raw/hf/Apothetirio_Pergamos/sections_with_annotation_n_metadata.parquet",
}
TRIG = re.compile(r"yc=(\d+)\s+yd=([\d.]+)\s+latin=([\d.]+)\s+dash=([\d.]+)\s+pos=([\d.]+)")


def parse_trigger(t):
    m = TRIG.search(t)
    if not m:
        return None
    yc, yd, latin, dash, pos = m.groups()
    return dict(yc=int(yc), yd=float(yd), latin=float(latin), dash=float(dash), pos=float(pos))


def stratum(gate, f):
    bib = gate.startswith("bib")
    strong = (f["yc"] > 0 and (f["latin"] > 0.15 or f["dash"] > 0.3))
    if bib and strong:        return "bib_strong"      # likely TP
    if bib and not strong:    return "bib_weak"        # likely FP — oversample
    if not bib and f["yc"] > 0: return "kept_hasyear"  # likely FN — oversample
    return "kept_noyear"                                # likely TN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=16)
    ap.add_argument("--double-frac", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    # 1. collect β spans with parsed features, per source
    pool = collections.defaultdict(list)  # stratum -> [rec]
    for src in ("kallipos", "pergamos"):
        sp = f"{ROOT}/out/{src}_full/refspans/{src}.spans.jsonl"
        for line in open(sp, encoding="utf-8"):
            d = json.loads(line)
            if d.get("kind") != "beta_section":
                continue
            f = parse_trigger(d.get("trigger", ""))
            if not f:
                continue
            st = stratum(d["gated_by"], f)
            pool[st].append({"source": src, "row_id": d["row_id"], "gate_decision": d["gated_by"],
                             "features": f, "header_hint": d["trigger"].split("|", 1)[0].strip()})

    # 2. deterministic stratified pick (sort by row_id, take evenly)
    chosen = []
    for st in ("bib_weak", "kept_hasyear", "bib_strong", "kept_noyear"):
        recs = sorted(pool.get(st, []), key=lambda r: (r["source"], r["row_id"]))
        if not recs:
            continue
        step = max(1, len(recs) // a.per_stratum)
        pick = recs[::step][: a.per_stratum]
        for r in pick:
            r["stratum"] = st
        chosen += pick
    # assign unit ids
    for i, r in enumerate(sorted(chosen, key=lambda r: (r["source"], r["row_id"]))):
        r["unit_id"] = f"B{i:04d}"

    # 3. fetch section text by row_id from parquets (stream, keep wanted)
    import pyarrow.parquet as pq
    want = collections.defaultdict(dict)  # source -> row_id -> rec
    for r in chosen:
        want[r["source"]][r["row_id"]] = r
    for src, rid_map in want.items():
        pf = pq.ParquetFile(PARQUET[src])
        cols = [c for c in ["row_id", "header", "section"] if c in [f.name for f in pf.schema_arrow]]
        found = 0
        for b in pf.iter_batches(batch_size=20000, columns=cols):
            d = b.to_pydict()
            rids = d["row_id"]
            for i, rid in enumerate(rids):
                if rid in rid_map:
                    rid_map[rid]["header"] = d.get("header", [""] * len(rids))[i] or ""
                    rid_map[rid]["section"] = d.get("section", [""] * len(rids))[i] or ""
                    found += 1
            if found >= len(rid_map):
                break

    # 4. write units (blind) + hidden manifest + batches
    udir = f"{HERE}/units/B_betagate"
    os.makedirs(udir, exist_ok=True)
    man = open(f"{HERE}/units/B_betagate_manifest.jsonl", "w", encoding="utf-8")
    unit_ids = []
    for r in chosen:
        if "section" not in r:
            continue
        lines = (r["header"] + "\n" + r["section"]).split("\n")
        numbered = "\n".join(f"L{j:04d}: {ln}" for j, ln in enumerate(lines))
        unit = {"unit_id": r["unit_id"], "eval": "B_betagate", "source": r["source"],
                "instruction": "Classify this single document SECTION per ANNOTATION_PROTOCOL.md Eval B. "
                               "It is one section of a Greek academic work. Decide is_reference_list + all typed fields. "
                               "Return ONLY the Eval-B JSON. Quote evidence verbatim from the text.",
                "header_line": r["header"], "section_numbered": numbered}
        json.dump(unit, open(f"{udir}/unit_{r['unit_id']}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        man.write(json.dumps({"unit_id": r["unit_id"], "source": r["source"], "row_id": r["row_id"],
                              "gate_decision": r["gate_decision"], "stratum": r["stratum"],
                              "features": r["features"]}, ensure_ascii=False) + "\n")
        unit_ids.append(r["unit_id"])
    man.close()

    # batches (+ a double-annotation subset for κ)
    bs = a.batch_size
    batches = [unit_ids[i:i + bs] for i in range(0, len(unit_ids), bs)]
    ndouble = int(len(unit_ids) * a.double_frac)
    double = unit_ids[::max(1, len(unit_ids) // max(1, ndouble))][:ndouble]
    json.dump({"batches": batches, "double_annotate": double, "n_units": len(unit_ids),
               "by_stratum": dict(collections.Counter(r["stratum"] for r in chosen if "section" in r))},
              open(f"{HERE}/units/B_betagate_batches.json", "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(unit_ids)} units → {udir}")
    print("by stratum:", dict(collections.Counter(r["stratum"] for r in chosen if "section" in r)))
    print(f"{len(batches)} batches of {bs}, double-annotate {len(double)}")


if __name__ == "__main__":
    main()
