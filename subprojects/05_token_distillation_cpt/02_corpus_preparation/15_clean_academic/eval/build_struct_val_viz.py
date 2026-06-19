#!/usr/bin/env python3
"""Held-out review presentation: 50 random VALIDATION (test-split) docs from STRUCT_2K, each rendered line
by line with the agreed colour scheme — FILL = gpt-5.5 gold structure (white main / blue ToC / red bib /
orange chapter-bib / purple author-CV / grey derived front+appendix); LEFT-BORDER = the trained model's
PREDICTION (blue ToC / red bib / none other). Where fill and border disagree, the line is marked — a visual
error analysis to complement the prose-protection numbers. Multi-page (index + one page per doc) to avoid
the giant-single-page bug. Publishes under ~/presentations/glossapi-tokenizer-extension/struct-val-50/.

  python build_struct_val_viz.py [N=50] [seed=20260619]
"""
import json, os, sys, random, html
import struct_lines as SL
import train_struct as T
import decode_spans as DS
import build_struct_viz as BV
HERE = os.path.dirname(os.path.abspath(__file__))
SDIR = f"{HERE}/units/STRUCT_2K"
SLUG = "struct-val-50"
OUT = os.path.expanduser(f"~/presentations/glossapi-tokenizer-extension/{SLUG}")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20260619

GOLD2CLS = {"toc": "toc", "bib": "bib", "chbib": "bib", "cv": "bib"}   # collapse for agreement vs the 2-class model


def predictions(data, bundle):
    bibm = json.load(open(f"{HERE}/span_line_lr_struct_model.json"))
    tocm = json.load(open(f"{HERE}/toc_line_lr_model.json"))
    pl_bib = T.pline_of(T.apply_model(bundle, bibm), bundle["rows"])
    pl_toc = SL.apply_toc_gate(T.pline_of(T.apply_model(bundle, tocm), bundle["rows"]), SL.toc_gate(data))
    sp = json.load(open(f"{HERE}/struct_smooth_params.json"))
    bibp = {k: sp["bib"][k] for k in ("theta_hi", "theta_lo", "gap", "lmin")}
    tocp = {k: sp["toc"][k] for k in ("theta_hi", "theta_lo", "gap", "lmin")}
    pb, pt = {}, {}
    for doc_id, d in data.docs.items():
        pb[doc_id] = {n for a, b in DS.decode_doc(d, pl_bib[doc_id], bibp) for n in range(a, b + 1)}
        pt[doc_id] = {n for a, b in DS.decode_doc(d, pl_toc[doc_id], tocp) for n in range(a, b + 1)}
    return pb, pt


CSS = """
:root{--ink:#1a1a1a;--muted:#5b5b5b;--line:#d9d4c8;--bg:#faf8f3;--accent:#6b1f1f;
 --c-main:#ffffff;--c-front:#eceae4;--c-toc:#cfe3ff;--c-bib:#fcd2cf;--c-chbib:#ffe2b8;--c-cv:#e6d8fb;
 --b-toc:#2563eb;--b-bib:#dc2626;}
*{box-sizing:border-box}html{font-size:16px}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Iowan Old Style",Palatino,Georgia,serif;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:30px 22px 80px}
h1{font-size:1.5rem;margin:0 0 .15em;font-weight:600}
.sub{color:var(--muted);font-style:italic;margin:0 0 1em}
a{color:var(--accent)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:.7em 0 1em;font-size:.84rem;align-items:center}
.sw{display:inline-block;width:15px;height:15px;border-radius:3px;vertical-align:-2px;margin-right:5px;border:1px solid #0002}
.bd{display:inline-block;width:0;height:15px;border-left:6px solid;vertical-align:-2px;margin-right:6px}
.doc{background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px 4px;overflow:hidden;margin-top:10px}
.ln{display:flex;font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.42;
 padding:0 8px 0 0;border-left:6px solid transparent;white-space:pre-wrap;word-break:break-word}
.ln .num{flex:0 0 52px;color:#aab;text-align:right;padding-right:9px;user-select:none}
.ln .tx{flex:1}
.c-main{background:var(--c-main)}.c-front{background:var(--c-front);color:#888}.c-appx{background:var(--c-front);color:#888}
.c-toc{background:var(--c-toc)}.c-bib{background:var(--c-bib)}.c-chbib{background:var(--c-chbib)}.c-cv{background:var(--c-cv)}
.p-toc{border-left-color:var(--b-toc)}.p-bib{border-left-color:var(--b-bib)}.p-other{border-left-color:transparent}
.dis{outline:2px dotted #b45309;outline-offset:-2px}
.divider{margin:6px 14px;padding:4px 10px;border-top:1px dashed #bbb;border-bottom:1px dashed #bbb;color:#999;font-style:italic;font-size:.78rem;text-align:center;background:#f4f2ec}
.meta{font-size:.86rem;color:var(--muted);margin:.3em 0 .2em}
.nav{display:flex;justify-content:space-between;margin:.4em 0 .2em;font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin-top:.6em}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:left}
th{font-weight:600;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
td.n{font-variant-numeric:tabular-nums;text-align:right}
tr:hover{background:#fff}
.pill{font-size:.74rem;padding:1px 6px;border-radius:9px;background:#eee}
.good{color:#15803d}.bad{color:#b91c1c}
"""

LEGEND = ("<div class=legend>"
          "<b>fill = gpt-5.5 gold:</b>"
          "<span><span class='sw' style='background:var(--c-main)'></span>main (keep)</span>"
          "<span><span class='sw' style='background:var(--c-toc)'></span>ToC</span>"
          "<span><span class='sw' style='background:var(--c-bib)'></span>bibliography</span>"
          "<span><span class='sw' style='background:var(--c-chbib)'></span>chapter-bib</span>"
          "<span><span class='sw' style='background:var(--c-cv)'></span>author CV</span>"
          "<span><span class='sw' style='background:var(--c-front)'></span>front/appendix (derived)</span>"
          "&nbsp;&nbsp;<b>left border = model prediction:</b>"
          "<span><span class='bd' style='border-color:var(--b-toc)'></span>ToC</span>"
          "<span><span class='bd' style='border-color:var(--b-bib)'></span>bib</span>"
          "<span><span class='dis' style='padding:0 6px'>dotted = gold≠pred</span></span>"
          "</div>")


def render_doc(p, doc_id, pb, pt, idx, total):
    rows = []
    agree = tot = 0
    for l in p["lines"]:
        if "div" in l:
            rows.append(f"<div class='divider'>{html.escape(l['div'])}</div>"); continue
        n = l["n"]
        c = l["c"]                                   # gold-derived code: main/front/appx/toc/bib/chbib/cv
        pred = "toc" if n in pt else ("bib" if n in pb else "other")
        gold_cls = GOLD2CLS.get(c, "other")
        dis = " dis" if gold_cls != pred else ""
        if c not in ("front", "appx", "main"):       # only structural lines count toward agreement
            tot += 1; agree += int(gold_cls == pred)
        rows.append(f"<div class='ln c-{c} p-{pred}{dis}'><span class='num'>{n}</span>"
                    f"<span class='tx'>{html.escape(l['t'])}</span></div>")
    npb = len(pt) > 0; agp = (100 * agree // tot) if tot else 100
    gold_spans = ", ".join(f"{s['code']}[{s['a']}-{s['b']}]" for s in p["spans"]) or "—"
    prev = f"<a href='doc_{idx-1:02d}.html'>← prev</a>" if idx > 0 else "<span></span>"
    nxt = f"<a href='doc_{idx+1:02d}.html'>next →</a>" if idx < total - 1 else "<span></span>"
    body = (f"<div class=wrap><div class=nav><a href='index.html'>↑ all {total}</a>"
            f"<span>{idx+1} / {total}</span></div>"
            f"<h1>{p['source']} · {doc_id[:16]}</h1>"
            f"<div class=meta>doc_type <b>{p['doc_type']}</b> · {p['n_lines']} lines ({p['mode']}) · "
            f"badness {p['badness']} · gold spans: {html.escape(gold_spans)} · "
            f"<b>structural-line agreement {agp}%</b> ({agree}/{tot})</div>"
            f"{LEGEND}<div class=doc>{''.join(rows)}</div>"
            f"<div class=nav style='margin-top:10px'>{prev}{nxt}</div></div>")
    return body, agp, tot


def page(title, body):
    return ("<!DOCTYPE html><html lang=el><head><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width, initial-scale=1'>"
            f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>")


def main():
    data = SL.load()
    bundle = SL.build_matrix(data)
    pb_all, pt_all = predictions(data, bundle)
    man = {json.loads(l)["doc_id"]: json.loads(l)["i"]
           for l in open(f"{SDIR}/manifest.jsonl") if l.strip()}
    test = [d for d in data.docs if data.docs[d]["split"] == "test" and d in man]
    rng = random.Random(SEED); rng.shuffle(test)
    sel = test[:N]
    os.makedirs(OUT, exist_ok=True)

    cards = []
    for idx, doc_id in enumerate(sel):
        p = BV.build_doc(f"{SDIR}/ann_{man[doc_id]:05d}.json")
        body, agp, tot = render_doc(p, doc_id, pb_all[doc_id], pt_all[doc_id], idx, len(sel))
        open(f"{OUT}/doc_{idx:02d}.html", "w", encoding="utf-8").write(page(f"{p['source']} {doc_id[:10]}", body))
        gt = sum(1 for s in p["spans"] if s["code"] == "toc"); gb = len(p["spans"]) - gt
        pt_n = 1 if pt_all[doc_id] else 0
        cards.append((idx, p, doc_id, agp, tot, gt, gb, len(pb_all[doc_id]), len(pt_all[doc_id])))

    mean_ag = round(sum(c[3] for c in cards if c[4]) / max(sum(1 for c in cards if c[4]), 1))
    trows = []
    for idx, p, doc_id, agp, tot, gt, gb, npb, npt in cards:
        cls = "good" if agp >= 90 else ("bad" if agp < 70 else "")
        trows.append(f"<tr><td class=n>{idx+1}</td><td><a href='doc_{idx:02d}.html'>{p['source']}</a></td>"
                     f"<td style='font-family:monospace;font-size:.8rem'>{doc_id[:14]}</td>"
                     f"<td>{p['doc_type']}</td><td class=n>{p['badness']}</td>"
                     f"<td class=n>{gt}/{gb}</td><td class=n>{'toc' if npt else '—'}/{('%d'%npb) if npb else '0'} ln</td>"
                     f"<td class='n {cls}'>{agp if tot else '—'}%</td></tr>")
    idxbody = (f"<div class=wrap><p style='margin:0 0 1em'><a href='/'>← presentations</a></p>"
               f"<h1>Held-out structure review — {len(sel)} validation docs</h1>"
               f"<p class=sub>Random sample from the {sum(1 for d in data.docs if data.docs[d]['split']=='test')}-doc "
               f"test split (seed {SEED}). Fill = gpt-5.5 gold structure; left border = the trained two-head "
               f"model's prediction. Mean structural-line agreement <b>{mean_ag}%</b>. Click a row.</p>"
               f"{LEGEND}"
               f"<table><thead><tr><th>#</th><th>source</th><th>doc_id</th><th>type</th><th>badness</th>"
               f"<th>gold toc/bib</th><th>pred</th><th>agree</th></tr></thead><tbody>{''.join(trows)}</tbody></table></div>")
    open(f"{OUT}/index.html", "w", encoding="utf-8").write(page("Held-out structure review — 50 docs", idxbody))
    print(f"wrote {len(sel)} docs → {OUT}/index.html  (mean agreement {mean_ag}%)")
    print(f"URL: http://presentations.localhost:8080/pres/glossapi-tokenizer-extension/{SLUG}/")


if __name__ == "__main__":
    main()
