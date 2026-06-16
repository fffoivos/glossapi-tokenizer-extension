#!/usr/bin/env python3
"""Goal-eval, section sources (Kallipos/Pergamos): sample sections from ALL predicted_section
classes — not just β — so we can measure the STACKED pipeline's total-bibliography recall, including
bibliographies the upstream section classifier mislabels as κ (which the β-gate never sees).

Strata (with inclusion weights):
  beta        : predicted_section==β  (detector positives → precision + recall-given-β)
  nonbeta_hi  : predicted_section!=β AND high entry-density (candidate classifier-MISSES → the blind spot)
  nonbeta_lo  : predicted_section!=β AND low entry-density  (base rate / negatives)
detector_flagged = (predicted_section==β AND β-gate said bib). Opus labels is_bibliographic_list.
"""
import argparse, glob, json, math, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PARQUET = {"kallipos": "/home/foivos/data/glossapi_raw/hf/Apothetirio_Kallipos/Dataset_Kallipos.parquet",
           "pergamos": "/home/foivos/data/glossapi_raw/hf/Apothetirio_Pergamos/sections_with_annotation_n_metadata.parquet"}
YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b"); PAGE = re.compile(r"\b\d+\s*[-–]\s*\d+\b|σσ?\.\s*\d|pp?\.\s*\d")
NAME = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]{2,}"); AUTH = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,\s*[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ.]+")
PLACE = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]"); LOC = ["σσ.", "σ.", "τ.", "εκδ.", "επιμ.", "vol.", "pp.", "eds."]


def entry_density(t):
    ls = [l for l in t.split("\n") if l.strip()]
    if not ls:
        return 0.0
    e = sum(1 for l in ls if NAME.search(l) and (YEAR.search(l) or PAGE.search(l) or AUTH.search(l) or PLACE.search(l) or any(x in l.lower() for x in LOC)))
    return e / len(ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=int, default=90); ap.add_argument("--nonbeta-hi", type=int, default=140)
    ap.add_argument("--nonbeta-lo", type=int, default=70); ap.add_argument("--batch", type=int, default=30)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    # β-gate decision per row_id (from the deployed-LR re-run)
    gate = {}
    for src in ("kallipos", "pergamos"):
        p = f"{ROOT}/out/{src}_lr/refspans/{src}.spans.jsonl"
        if os.path.exists(p):
            for l in open(p):
                d = json.loads(l)
                if d.get("kind") == "beta_section" and d.get("row_id"):
                    gate[d["row_id"]] = d["gated_by"].startswith("bib")

    pool = collections.defaultdict(list)
    for src in ("kallipos", "pergamos"):
        pf = pq.ParquetFile(PARQUET[src])
        cols = [c for c in ["row_id", "header", "section", "predicted_section"] if c in [f.name for f in pf.schema_arrow]]
        n = 0
        for b in pf.iter_batches(batch_size=20000, columns=cols):
            d = b.to_pydict()
            for i in range(len(d["row_id"])):
                ps = (d.get("predicted_section", [""] * len(d["row_id"]))[i] or "").strip()
                sec = d.get("section", [""] * len(d["row_id"]))[i] or ""
                if len(sec) < 40:
                    continue
                ed = entry_density(sec)
                rec = dict(source=src, row_id=d["row_id"][i], header=d.get("header", [""] * len(d["row_id"]))[i] or "",
                           section=sec, predicted_section=ps, entry_density=round(ed, 3))
                if ps == "β":
                    pool["beta"].append(rec)
                elif ed >= 0.3:
                    pool["nonbeta_hi"].append(rec)
                else:
                    pool["nonbeta_lo"].append(rec)
            n += len(d["row_id"])
            if n >= 700000:
                break

    chosen = []
    for st, target in (("beta", a.beta), ("nonbeta_hi", a.nonbeta_hi), ("nonbeta_lo", a.nonbeta_lo)):
        recs = sorted(pool.get(st, []), key=lambda r: (r["source"], r["row_id"]))
        want = min(target, len(recs)); step = max(1, len(recs) // want)
        pick = recs[::step][:want]; w = len(recs) / max(1, len(pick))
        for r in pick:
            r["stratum"] = st; r["weight"] = round(w, 2)
        chosen += pick
    for i, r in enumerate(sorted(chosen, key=lambda r: (r["source"], r["row_id"]))):
        r["unit_id"] = f"G{i:04d}"

    udir = f"{HERE}/units/G_sections"; os.makedirs(udir, exist_ok=True)
    for f in os.listdir(udir):
        os.remove(os.path.join(udir, f))
    man = open(f"{HERE}/units/G_sections_manifest.jsonl", "w", encoding="utf-8")
    for r in chosen:
        det = (r["predicted_section"] == "β" and gate.get(r["row_id"], False))
        man.write(json.dumps(dict(unit_id=r["unit_id"], source=r["source"], row_id=r["row_id"], stratum=r["stratum"],
                                  weight=r["weight"], predicted_section=r["predicted_section"], entry_density=r["entry_density"],
                                  detector_flagged_bib=bool(det)), ensure_ascii=False) + "\n")
    man.close()
    units = [{"unit_id": r["unit_id"], "header_line": r["header"], "section_text": r["section"][:4000]} for r in chosen]
    paths = []
    for bi in range(0, len(units), a.batch):
        p = f"{udir}/batch_{bi // a.batch:03d}.json"; json.dump(units[bi:bi + a.batch], open(p, "w", encoding="utf-8"), ensure_ascii=False); paths.append(p)
    json.dump(paths, open(f"{HERE}/units/G_sections_batchpaths.json", "w"), indent=1)
    print(f"{len(units)} section units in {len(paths)} batches; strata={collections.Counter(r['stratum'] for r in chosen)}")
    print(f"detector-flagged-bib among sampled: {sum(1 for r in chosen if r['predicted_section']=='β' and gate.get(r['row_id'],False))}")


if __name__ == "__main__":
    main()
