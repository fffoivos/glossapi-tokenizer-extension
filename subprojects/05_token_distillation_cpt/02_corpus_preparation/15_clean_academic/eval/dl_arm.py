#!/usr/bin/env python3
"""DL-augmented line arm (frozen-first): line LR over [handcrafted line features ; frozen line
embedding]. Reuses the cached feature matrix + the contiguity smoother + the scorer, so the only new
signal is the embedding. Answers the bench question — do frozen embeddings lift recall at the
precision-first operating point? Dumps decoded spans to units/SPAN_pred_dl.json (picked up by
score_span_models.py as the `dl_crf` arm) and prints line-level precision/recall.

Run: python dl_arm.py [encoder]
"""
import os, sys, json, collections
import numpy as np
import span_seq_data as D
import line_lr as L
import decode_spans as DS
import embed_lines as E
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    enc = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-small"
    data = D.load()
    bundle = L.build_matrix(data)
    emb, docids, counts = E.load_cached(enc)
    nrow = sum(len(d["lines"]) for d in data.docs.values())
    assert emb.shape[0] == nrow == len(bundle["rows"]), (emb.shape[0], nrow, len(bundle["rows"]))
    print(f"emb {emb.shape}; training line+emb LR ...")
    pline, model = L.train_pline(data, bundle, extra_mat=emb, iters=800)
    np.savez(DS.PCACHE.replace(".npz", "_dl.npz"), **pline)
    # tune the smoother precision-first on train, score
    params, _, _ = DS.tune(data, pline, target=0.97)
    json.dump(params, open(f"{HERE}/span_smooth_params_dl.json", "w"), indent=1)
    tr_p, tr_r, *_ = DS.line_pr(data, pline, params, "train")
    te_p, te_r, tp, fp, fn = DS.line_pr(data, pline, params, "test")
    print(f"tuned hysteresis: {params}")
    print(f"TRAIN line-precision {tr_p:.3f} recall {tr_r:.3f}")
    print(f"TEST  line-precision {te_p:.3f} recall {te_r:.3f}  (FP-rate {1-te_p:.3f})")
    preds = {doc: [list(s) for s in DS.decode_doc(d, pline[doc], params)] for doc, d in data.docs.items()}
    json.dump(preds, open(f"{HERE}/units/SPAN_pred_dl.json", "w"))
    print("wrote units/SPAN_pred_dl.json (scored by score_span_models.py as dl_crf)")


if __name__ == "__main__":
    main()
