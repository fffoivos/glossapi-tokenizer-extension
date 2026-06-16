#!/usr/bin/env python3
"""Per-line span visualization for the results hub: 10 held-out documents with bibliographies, each line
coloured by the confusion between the Opus gold annotation and the classifier — green = match (both say
bibliography), red = Opus bibliography the classifier missed, amber = classifier removed but not in Opus.
An indigo left-bar marks the original Opus annotation span. Hovering a line updates a sticky panel to the
right showing its probability against the hysteresis thresholds (θ_lo=0.6 open-extend, θ_hi=0.8 open)."""
import json, os, collections
import span_seq_data as D
import decode_spans as DS
HERE = os.path.dirname(os.path.abspath(__file__))


def pick_docs(data, pline, params, k=10):
    cand = collections.defaultdict(list)
    for doc, d in data.docs.items():
        if d["split"] != "test":
            continue
        gold = set()
        for s in d["spans"]:
            a, b = s.get("start_line"), s.get("end_line")
            if isinstance(a, int) and isinstance(b, int) and b >= a:
                gold.update(range(a, b + 1))
        present = len(d["lines"])
        goldp = sum(1 for a, _ in d["lines"] if a in gold)
        if not (goldp >= 25 and 60 <= present <= 430):
            continue
        predset = set()
        for s, e in DS.decode_doc(d, pline[doc], params):
            predset.update(range(s, e + 1))
        tp = sum(1 for a, _ in d["lines"] if a in gold and a in predset)
        match = tp / goldp
        # keep docs with a real match-rate spread; drop pathological all-miss OCR-shatter cases
        if 0.40 <= match:
            cand[d["source"]].append((goldp, doc))
    for s in cand:
        cand[s].sort(reverse=True)
    srcs = list(cand); out = []; i = 0
    while len(out) < k and any(cand.values()):
        s = srcs[i % len(srcs)]
        if cand[s]:
            out.append(cand[s].pop(0)[1])
        i += 1
    return out


def doc_payload(data, pline, params, doc):
    d = data.docs[doc]
    gold = set()
    kinds = collections.Counter()
    for s in d["spans"]:
        a, b = s.get("start_line"), s.get("end_line")
        if isinstance(a, int) and isinstance(b, int) and b >= a:
            gold.update(range(a, b + 1)); kinds[s.get("kind", "?")] += 1
    predset = set()
    for s, e in DS.decode_doc(d, pline[doc], params):
        predset.update(range(s, e + 1))
    lines = []
    tp = fn = fp = 0
    for i, (a, t) in enumerate(d["lines"]):
        g = 1 if a in gold else 0
        p = 1 if a in predset else 0
        tp += g and p; fn += g and not p; fp += (not g) and p
        lines.append({"n": a, "t": t, "g": g, "p": p, "pr": round(float(pline[doc][i]), 3)})
    return {"id": doc[:12], "source": d["source"], "lines": lines,
            "kinds": ", ".join(f"{k}×{v}" for k, v in kinds.most_common()),
            "tp": tp, "fn": fn, "fp": fp, "gold": tp + fn}


CSS = """
:root{--ink:#1a1a1a;--muted:#5b5b5b;--line:#d9d4c8;--bg:#faf8f3;--accent:#6b1f1f;
 --tp:#34d399;--fn:#f87171;--fp:#fbbf24;--opus:#6366f1;}
*{box-sizing:border-box}html{font-size:16px}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Iowan Old Style",Palatino,Georgia,serif;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:34px 22px 80px}
h1{font-size:1.55rem;margin:0 0 .15em;font-weight:600}
.sub{color:var(--muted);font-style:italic;margin:0 0 1.1em}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:.8em 0 1.2em;font-size:.86rem;align-items:center}
.sw{display:inline-block;width:15px;height:15px;border-radius:3px;vertical-align:-2px;margin-right:5px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 14px}
.tab{border:1px solid var(--line);background:#fff;border-radius:7px;padding:5px 11px;cursor:pointer;font:inherit;font-size:.82rem}
.tab.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tab .b{font-variant-numeric:tabular-nums;opacity:.8;font-size:.76rem}
.grid{display:grid;grid-template-columns:1fr 290px;gap:20px;align-items:start}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.doc{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 4px;overflow:hidden}
.ln{display:flex;font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;line-height:1.42;
 padding:1px 8px 1px 0;border-left:5px solid transparent;cursor:default;white-space:pre-wrap;word-break:break-word}
.ln:hover{outline:2px solid var(--accent);outline-offset:-2px;filter:saturate(1.4)}
.ln .num{flex:0 0 48px;color:#9aa;text-align:right;padding-right:10px;user-select:none}
.ln .tx{flex:1}
.ln.tp{background:var(--tp)}.ln.fn{background:var(--fn)}.ln.fp{background:var(--fp)}
.ln.opus{border-left-color:var(--opus)}
.panel{position:sticky;top:14px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px 18px;font-size:.9rem}
.panel h3{margin:.1em 0 .7em;font-size:1rem}
.pbar{position:relative;height:26px;background:#eee;border-radius:6px;margin:.5em 0 .2em;overflow:hidden}
.pfill{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#93c5fd,#2563eb)}
.thr{position:absolute;top:-3px;bottom:-3px;width:2px;background:#111}.thr.lo{left:60%}.thr.hi{left:80%}
.thr span{position:absolute;top:-15px;left:-7px;font-size:9px;color:#111}
.kv{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px dashed var(--line)}
.kv b{font-variant-numeric:tabular-nums}
.verdict{margin-top:.7em;font-weight:600;padding:6px 9px;border-radius:6px;text-align:center}
.docmeta{font-size:.8rem;color:var(--muted);margin:.2em 0 .8em}
.hint{color:var(--muted);font-size:.82rem;margin-top:.6em}
a{color:var(--accent)}
"""

JS = """
const thr_hi=0.8, thr_lo=0.6;
const root=document.getElementById('doc'), tabs=document.getElementById('tabs'), panel=document.getElementById('panel');
function cls(l){ if(l.g&&l.p)return'tp'; if(l.g&&!l.p)return'fn'; if(!l.g&&l.p)return'fp'; return''; }
function render(k){
  const d=DOCS[k];
  [...tabs.children].forEach((t,i)=>t.classList.toggle('on',i===k));
  let h='<div class="docmeta">'+d.source+' · '+d.id+' · '+d.kinds+' · gold '+d.gold+
        ' lines — <b style="color:#16a34a">'+d.tp+' matched</b>, <b style="color:#dc2626">'+d.fn+' missed</b>, <b style="color:#b45309">'+d.fp+' extra</b></div>';
  h+='<div class="doc">';
  d.lines.forEach((l,i)=>{
    const c=cls(l), op=l.g?' opus':'';
    h+='<div class="ln '+c+op+'" data-k="'+k+'" data-i="'+i+'"><span class="num">'+l.n+'</span><span class="tx">'+esc(l.t)+'</span></div>';
  });
  h+='</div>';
  root.innerHTML=h;
  root.querySelectorAll('.ln').forEach(e=>e.addEventListener('mouseenter',()=>show(DOCS[k].lines[+e.dataset.i])));
}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function show(l){
  const passed = l.pr>=thr_hi?'opens a span (p ≥ 0.80)':(l.pr>=thr_lo?'extends a span (0.60 ≤ p < 0.80)':'below threshold (p < 0.60)');
  const verdict = l.g&&l.p?['MATCH — both say bibliography','#16a34a','#dcfce7']
    : l.g&&!l.p?['MISSED — Opus bibliography, classifier kept it','#dc2626','#fee2e2']
    : !l.g&&l.p?['EXTRA — classifier removed, not in Opus','#b45309','#fef3c7']
    : ['agree — neither marks bibliography','#555','#eee'];
  panel.innerHTML='<h3>Line '+l.n+'</h3>'+
    '<div class="pbar"><div class="pfill" style="width:'+(l.pr*100)+'%"></div>'+
      '<div class="thr lo"><span>θlo</span></div><div class="thr hi"><span>θhi</span></div></div>'+
    '<div class="kv"><span>probability</span><b>'+l.pr.toFixed(3)+'</b></div>'+
    '<div class="kv"><span>threshold</span><b>'+passed+'</b></div>'+
    '<div class="kv"><span>Opus annotation</span><b>'+(l.g?'bibliography':'—')+'</b></div>'+
    '<div class="kv"><span>classifier</span><b>'+(l.p?'removed (bib)':'kept')+'</b></div>'+
    '<div class="verdict" style="color:'+verdict[1]+';background:'+verdict[2]+'">'+verdict[0]+'</div>';
}
DOCS.forEach((d,k)=>{const b=document.createElement('button');b.className='tab';
  b.innerHTML=d.source.slice(0,2).toUpperCase()+' <span class="b">'+(d.tp*100/Math.max(1,d.gold)|0)+'% hit</span>';
  b.onclick=()=>render(k);tabs.appendChild(b);});
render(0);
panel.innerHTML='<h3>Hover a line →</h3><p class="hint">Move over any line to see its probability and whether it passed the span threshold.</p>';
"""


def main():
    data = D.load()
    pline = DS.get_pline(data)
    params = json.load(open(f"{HERE}/span_smooth_params.json"))
    docs = pick_docs(data, pline, params, 10)
    payload = [doc_payload(data, pline, params, doc) for doc in docs]
    html = (
        "<!DOCTYPE html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1'>"
        "<title>Bibliography span classifier — per-line view</title><style>" + CSS + "</style></head><body><div class=wrap>"
        "<p style='margin:0 0 1em'><a href='/'>← presentations</a></p>"
        "<h1>What the classifier removes, line by line</h1>"
        "<p class=sub>10 held-out documents · Opus annotation vs the line-LR span classifier · hover any line for its probability</p>"
        "<div class=legend>"
        "<span><span class='sw' style='background:var(--tp)'></span>match (both = bibliography)</span>"
        "<span><span class='sw' style='background:var(--fn)'></span>Opus bib, classifier missed</span>"
        "<span><span class='sw' style='background:var(--fp)'></span>classifier removed, not in Opus</span>"
        "<span><span class='sw' style='background:var(--opus);border-radius:1px;width:6px'></span>indigo left-bar = original Opus annotation</span>"
        "</div><div class=tabs id=tabs></div><div class=grid><div id=doc></div><div class=panel id=panel></div></div>"
        "</div><script>const DOCS=" + json.dumps(payload, ensure_ascii=False) + ";\n" + JS + "</script></body></html>"
    )
    out = os.path.expanduser("~/presentations/glossapi-tokenizer-extension/span-viz/index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out}  ({len(payload)} docs, {sum(len(p['lines']) for p in payload)} lines)")
    for p in payload:
        print(f"  {p['source']:<12} {p['id']}  gold {p['gold']:>3}  match {100*p['tp']//max(1,p['gold'])}%  ({p['kinds']})")


if __name__ == "__main__":
    main()
