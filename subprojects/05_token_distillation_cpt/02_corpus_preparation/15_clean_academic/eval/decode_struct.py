#!/usr/bin/env python3
"""Per-class decode + tune for the two-head structural tagger. Reuses decode_spans.hysteresis/tune/line_pr
unchanged by pointing each doc's d['spans'] at the per-class gold runs (struct_lines.set_class). The ToC
head's probabilities are front-gated first. Reports SMOOTHED line precision/recall on the doc-grouped TEST
(the real precision-first metric — raw per-line under-counts because the smoother fills a contiguous block's
interior). Freezes per-class hysteresis to struct_smooth_params.json {"bib":{…},"toc":{…}}.

  python decode_struct.py [bib_target=0.97] [toc_target=0.97]
"""
import json, os, sys, collections
import numpy as np
import decode_spans as DS
import struct_lines as SL
import train_struct as T
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    bib_t = float(sys.argv[1]) if len(sys.argv) > 1 else 0.97
    toc_t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.97
    data = SL.load()
    bundle = SL.build_matrix(data)
    Pb, _ = T.train_head(bundle, SL.BIB_FEATS, bundle["y_bib"], T.BIB_COST)
    Pt, _ = T.train_head(bundle, SL.TOC_FEATS, bundle["y_toc"], T.TOC_COST)
    gate = SL.toc_gate(data)
    pl = {"bib": T.pline_of(Pb, bundle["rows"]),
          "toc": SL.apply_toc_gate(T.pline_of(Pt, bundle["rows"]), gate)}
    targets = {"bib": bib_t, "toc": toc_t}

    out = {}
    for cls in ("bib", "toc"):
        SL.set_class(data, cls)
        best = DS.tune(data, pl[cls], target=targets[cls])
        params = best[0]
        out[cls] = params
        trp, trr, *_ = DS.line_pr(data, pl[cls], params, "train")
        tep, ter, tp, fp, fn = DS.line_pr(data, pl[cls], params, "test")
        # multi-span distribution on test
        nsp = collections.Counter()
        for doc_id, d in data.docs.items():
            if d["split"] == "test":
                nsp[min(len(DS.decode_doc(d, pl[cls][doc_id], params)), 6)] += 1
        print(f"\n[{cls.upper()}]  target line-prec ≥ {targets[cls]}   tuned {params}")
        print(f"  TRAIN  line-prec {trp:.3f}  recall {trr:.3f}")
        print(f"  TEST   line-prec {tep:.3f}  recall {ter:.3f}   (TP {tp:,} FP {fp:,} FN {fn:,})  "
              f"prose-amputation(FP)={1-tep:.3f}")
        print(f"  decoded spans/test-doc: {dict(sorted(nsp.items()))}")

    json.dump(out, open(f"{HERE}/struct_smooth_params.json", "w"), indent=1)
    print(f"\nfroze per-class hysteresis → struct_smooth_params.json  {json.dumps(out)}")


if __name__ == "__main__":
    main()
