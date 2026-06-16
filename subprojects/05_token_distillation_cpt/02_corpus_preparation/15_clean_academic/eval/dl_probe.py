#!/usr/bin/env python3
"""Fast scoped DL probe (home CPU is too slow to embed all 840k lines ~3h). Embeds a STRATIFIED
~25k-line sample — oversampling the Greek-script recall holes — and asks the bench question directly:
does adding a frozen line embedding to the interpretable line features lift line-level recall at matched
precision, especially on monotonic/polytonic Greek? Reuses the cached feature matrix (context features
already baked in), so sampling isolated lines is valid. Result drives the ship-DL / Clariden-GPU /
stay-interpretable decision in MODEL_TRADEOFF.md."""
import os, sys, json, collections
import numpy as np
import span_seq_data as D
import line_lr as L
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.RandomState(0)


def main():
    enc = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-small"
    data = D.load()
    bundle = L.build_matrix(data)
    rows = bundle["rows"]; y = bundle["y"]; tr = bundle["tr"]
    Xfeat = bundle["X"][:, [bundle["col"][k] for k in L.FEATS]].astype(np.float64)
    # script per bib row (for slicing)
    smap = {}
    for doc_id, d in data.docs.items():
        for s in d["spans"]:
            a, b = s.get("start_line"), s.get("end_line")
            if isinstance(a, int) and isinstance(b, int):
                for li in range(a, b + 1):
                    smap.setdefault((doc_id, li), s.get("script"))
    script = np.array([smap.get((rows[i][0], data.docs[rows[i][0]]["lines"][rows[i][1]][0]), "")
                       for i in range(len(rows))], dtype=object)
    greek = np.isin(script, ["monotonic_greek", "polytonic_greek"])

    tri = np.where(tr)[0]; tei = np.where(~tr)[0]
    # train sample: 12k random + oversample Greek-script bib
    s_tr = rng.choice(tri, 12000, replace=False)
    g_tr = tri[(y[tri] == 1) & greek[tri]]
    s_tr = np.unique(np.concatenate([s_tr, rng.choice(g_tr, min(4000, len(g_tr)), replace=False)]))
    # test sample: 10k random + ALL Greek-script bib test rows
    s_te = rng.choice(tei, 10000, replace=False)
    g_te = tei[(y[tei] == 1) & greek[tei]]
    s_te = np.unique(np.concatenate([s_te, g_te]))
    emb_idx = np.unique(np.concatenate([s_tr, s_te]))
    print(f"sample: train {len(s_tr):,}  test {len(s_te):,}  (Greek bib: tr {len(g_tr)} te {len(g_te)})")
    print(f"embedding {len(emb_idx):,} lines with {enc} ...")

    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(8)
    model = SentenceTransformer(enc, device="cpu"); model.max_seq_length = 96
    texts = ["passage: " + data.docs[rows[i][0]]["lines"][rows[i][1]][1][:300] for i in emb_idx]
    E = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float64)
    embmap = {int(gi): E[j] for j, gi in enumerate(emb_idx)}

    def emb_of(idxs):
        return np.array([embmap[int(i)] for i in idxs])

    def fit_eval(use_emb):
        Xtr = Xfeat[s_tr]; Xte = Xfeat[s_te]
        if use_emb:
            Xtr = np.column_stack([Xtr, emb_of(s_tr)]); Xte = np.column_stack([Xte, emb_of(s_te)])
        mu, sd, W, b = L.fit_lr(Xtr, y[s_tr], iters=1200)
        Pte = L._prob(Xte, mu, sd, W, b)
        return Pte

    def recall_at_prec(P, yy, target=0.94):
        o = np.argsort(-P); ct = cf = 0; tot = int(yy.sum()); best = 0.0; thr = 1.0
        for j in o:
            if yy[j] == 1: ct += 1
            else: cf += 1
            if ct / (ct + cf) >= target:
                best = ct / tot; thr = float(P[j])
        return best, thr

    yte = y[s_te]; gte = greek[s_te]
    out = {}
    for name, ue in (("feat_only", False), ("feat+emb", True)):
        P = fit_eval(ue)
        r, thr = recall_at_prec(P, yte, 0.94)
        # greek-bib recall at that thr
        pred = P >= thr
        gbib = (yte == 1) & gte
        grec = (pred & gbib).sum() / max(1, gbib.sum())
        out[name] = dict(recall_at_p94=round(r, 3), thr=round(thr, 3), greek_bib_recall=round(float(grec), 3))
        print(f"  {name:<10} recall@prec0.94 = {r:.3f}  | Greek-bib recall@thr = {grec:.3f}  (thr {thr:.3f})")
    dlift = out["feat+emb"]["recall_at_p94"] - out["feat_only"]["recall_at_p94"]
    glift = out["feat+emb"]["greek_bib_recall"] - out["feat_only"]["greek_bib_recall"]
    print(f"\nEMBEDDING LIFT: overall recall@p94 {dlift:+.3f} | Greek-bib recall {glift:+.3f}")
    print("VERDICT:", "embeddings help (worth the Clariden GPU full bench)" if (dlift > 0.02 or glift > 0.05)
          else "embeddings add little on this sample -> ship interpretable; DL as oracle only")
    json.dump({"sample_lift_recall_p94": round(dlift, 3), "greek_bib_lift": round(glift, 3), **out},
              open(f"{HERE}/results_dl_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
