#!/usr/bin/env python3
"""Review presentation for the v2 structure-annotation experiment. Each of the 10 documents is rendered
line by line, coloured by its DERIVED structural section: Opus marks only the table-of-contents and the
bibliographies (incl. chapter bibs + the author's-own-publications/CV list); everything else is derived
deterministically from those anchors — front-matter before the ToC, MAIN TEXT (what we keep) between the
ToC and the first bibliography, appendix after the last. Hover any line for its section + the Opus metadata
(entries, confidence, KEEP vs REMOVE)."""
import json, os, re, glob, sys
import badness_filter as BF
HERE = os.path.dirname(os.path.abspath(__file__))
SDIR = f"{HERE}/units/{sys.argv[1] if len(sys.argv) > 1 else 'STRUCT'}"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "struct-viz"
LINE_RE = re.compile(r"^L(\d+):\s?(.*)$")


def span_code(s):
    if s["kind"] == "table_of_contents":
        return "toc"
    if s.get("is_authors_own_works"):
        return "cv"
    if s.get("is_chapter_bibliography"):
        return "chbib"
    return "bib"


def build_doc(ann_path):
    A = json.load(open(ann_path))
    U = json.load(open(ann_path.replace("ann_", "batch_")))[0]
    # parse lines in DOCUMENT ORDER; detect the front+tail elided divider by its text
    raw = []
    for ln in U["text_numbered"].split("\n"):
        g = LINE_RE.match(ln)
        if not g:
            continue
        n, t = int(g.group(1)), g.group(2)
        if not t.strip():
            continue
        if "main-text lines elided" in t:
            raw.append((None, t.strip(), True)); continue
        raw.append((n, t, False))
    spans = []
    for s in A.get("sections", []):
        spans.append({"code": span_code(s), "a": s["start_line"], "b": s["end_line"],
                      "kind": s["kind"], "ne": s.get("n_entries", 0), "conf": s.get("confidence", "?"),
                      "cv": bool(s.get("is_authors_own_works")), "ch": bool(s.get("is_chapter_bibliography"))})
    first_toc = min([s["a"] for s in spans if s["code"] == "toc"], default=None)
    bib_ends = [s["b"] for s in spans if s["code"] in ("bib", "chbib", "cv")]
    last_bib = max(bib_ends) if bib_ends else None
    lines = []
    for item in raw:
        if item[2]:
            lines.append({"div": item[1]}); continue
        n, t, _ = item
        mi = next((i for i, s in enumerate(spans) if s["a"] <= n <= s["b"]), -1)
        if mi >= 0:
            code = spans[mi]["code"]
        elif first_toc is not None and n < first_toc:
            code = "front"
        elif last_bib is not None and n > last_bib:
            code = "appx"
        else:
            code = "main"
        lines.append({"n": n, "t": t, "c": code, "m": mi})
    badness = BF.score_text("\n".join(item[1] for item in raw if not item[2]))
    return {"id": A["doc_id"][:14], "source": A.get("source", "?"), "doc_type": A.get("doc_type", "?"),
            "n_lines": A.get("n_lines"), "mode": U.get("mode"), "spans": spans, "lines": lines,
            "badness": round(badness, 1) if badness is not None else None}


CSS = """
:root{--ink:#1a1a1a;--muted:#5b5b5b;--line:#d9d4c8;--bg:#faf8f3;--accent:#6b1f1f;
 --c-main:#ffffff;--c-front:#eceae4;--c-appx:#eceae4;--c-toc:#cfe3ff;--c-bib:#fcd2cf;--c-chbib:#ffe2b8;--c-cv:#e6d8fb;
 --b-toc:#2563eb;--b-bib:#dc2626;--b-chbib:#d97706;--b-cv:#7c3aed;--b-front:#b9b4a8;}
*{box-sizing:border-box}html{font-size:16px}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Iowan Old Style",Palatino,Georgia,serif;line-height:1.5}
.wrap{max-width:1240px;margin:0 auto;padding:34px 22px 80px}
h1{font-size:1.55rem;margin:0 0 .15em;font-weight:600}
.sub{color:var(--muted);font-style:italic;margin:0 0 1.1em}
.legend{display:flex;flex-wrap:wrap;gap:13px;margin:.8em 0 1.1em;font-size:.85rem;align-items:center}
.sw{display:inline-block;width:15px;height:15px;border-radius:3px;vertical-align:-2px;margin-right:5px;border:1px solid #0002}
.tag{font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;padding:1px 5px;border-radius:4px;color:#fff;font-family:ui-sans-serif,system-ui}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 14px}
.tab{border:1px solid var(--line);background:#fff;border-radius:7px;padding:5px 10px;cursor:pointer;font:inherit;font-size:.8rem}
.tab.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tab .b{font-variant-numeric:tabular-nums;opacity:.75;font-size:.72rem}
.grid{display:grid;grid-template-columns:1fr 300px;gap:20px;align-items:start}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.doc{background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px 4px;overflow:hidden}
.ln{display:flex;font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.4;
 padding:0 8px 0 0;border-left:6px solid transparent;cursor:default;white-space:pre-wrap;word-break:break-word}
.ln:hover{outline:2px solid var(--accent);outline-offset:-2px;filter:saturate(1.35) brightness(.99)}
.ln .num{flex:0 0 46px;color:#aab;text-align:right;padding-right:9px;user-select:none}
.ln .tx{flex:1}
.ln.main{background:var(--c-main)}
.ln.front{background:var(--c-front);color:#777;border-left-color:var(--b-front)}
.ln.appx{background:var(--c-appx);color:#777;border-left-color:var(--b-front)}
.ln.toc{background:var(--c-toc);border-left-color:var(--b-toc)}
.ln.bib{background:var(--c-bib);border-left-color:var(--b-bib)}
.ln.chbib{background:var(--c-chbib);border-left-color:var(--b-chbib)}
.ln.cv{background:var(--c-cv);border-left-color:var(--b-cv)}
.divider{margin:6px 14px;padding:4px 10px;border-top:1px dashed #bbb;border-bottom:1px dashed #bbb;
 color:#999;font-style:italic;font-size:.78rem;text-align:center;background:#f4f2ec}
.panel{position:sticky;top:14px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px 18px;font-size:.9rem}
.panel h3{margin:.1em 0 .55em;font-size:1rem}
.kv{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px dashed var(--line);gap:10px}
.kv b{font-variant-numeric:tabular-nums;text-align:right}
.verdict{margin-top:.7em;font-weight:600;padding:6px 9px;border-radius:6px;text-align:center}
.docmeta{font-size:.82rem;color:var(--muted);margin:.2em 0 .7em}
.seclist{margin:.4em 0 .2em;font-size:.8rem}
.seclist .row{display:flex;gap:8px;padding:2px 0;align-items:center}
.hint{color:var(--muted);font-size:.82rem;margin-top:.6em}
a{color:var(--accent)}
"""

JS = """
const NAME={main:'MAIN TEXT (keep)',front:'front-matter (derived)',appx:'appendix (derived)',
 toc:'table of contents',bib:'bibliography',chbib:'chapter bibliography',cv:"author's publications (CV)"};
const REMOVE={front:1,appx:1,toc:1,bib:1,chbib:1,cv:1};
const root=document.getElementById('doc'), tabs=document.getElementById('tabs'), panel=document.getElementById('panel');
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function render(k){
  const d=DOCS[k];
  [...tabs.children].forEach((t,i)=>t.classList.toggle('on',i===k));
  const cnt={};d.lines.forEach(l=>{if(l.c)cnt[l.c]=(cnt[l.c]||0)+1;});
  let h='<div class="docmeta"><b>'+d.source+'</b> · '+d.id+' · doc_type=<b>'+d.doc_type+'</b> · '+d.n_lines+' lines ('+d.mode+') · badness '+(d.badness==null?'?':d.badness)+'</div>';
  h+='<div class="seclist">';
  d.spans.forEach(s=>{h+='<div class="row"><span class="tag" style="background:var(--b-'+(s.code==='cv'?'cv':s.code==='chbib'?'chbib':s.code==='toc'?'toc':'bib')+')">'+NAME[s.code]+'</span> L'+s.a+'–'+s.b+(s.ne?' · '+s.ne+' entries':'')+' · '+s.conf+'</div>';});
  h+='</div><div class="doc">';
  d.lines.forEach((l,i)=>{
    if(l.div!==undefined){h+='<div class="divider">'+esc(l.div)+'</div>';return;}
    h+='<div class="ln '+l.c+'" data-k="'+k+'" data-i="'+i+'"><span class="num">'+l.n+'</span><span class="tx">'+esc(l.t)+'</span></div>';
  });
  h+='</div>';
  root.innerHTML=h;
  root.querySelectorAll('.ln').forEach(e=>e.addEventListener('mouseenter',()=>show(k,+e.dataset.i)));
}
function show(k,i){
  const d=DOCS[k], l=d.lines[i];
  const rem=REMOVE[l.c]?['REMOVE — scaffolding','#b91c1c','#fee2e2']:['KEEP — main text','#15803d','#dcfce7'];
  let h='<h3>Line '+l.n+'</h3>'+
    '<div class="kv"><span>section</span><b>'+NAME[l.c]+'</b></div>';
  if(l.m>=0){const s=d.spans[l.m];
    h+='<div class="kv"><span>Opus marked</span><b>'+s.kind+'</b></div>'+
       '<div class="kv"><span>span</span><b>L'+s.a+'–'+s.b+'</b></div>'+
       (s.ne?'<div class="kv"><span>entries</span><b>'+s.ne+'</b></div>':'')+
       '<div class="kv"><span>flags</span><b>'+([s.ch?'chapter':'',s.cv?"author CV":''].filter(Boolean).join(', ')||'—')+'</b></div>'+
       '<div class="kv"><span>confidence</span><b>'+s.conf+'</b></div>';
  }else{
    h+='<div class="kv"><span>derived</span><b>from ToC / bib anchors</b></div>';
  }
  h+='<div class="verdict" style="color:'+rem[1]+';background:'+rem[2]+'">'+rem[0]+'</div>';
  panel.innerHTML=h;
}
DOCS.forEach((d,k)=>{const b=document.createElement('button');b.className='tab';
  const nb=d.spans.filter(s=>s.code!=='toc').length, nt=d.spans.filter(s=>s.code==='toc').length;
  b.innerHTML=d.source.slice(0,2).toUpperCase()+' '+d.id.slice(0,6)+' <span class="b">'+nt+'toc/'+nb+'bib</span>';
  b.onclick=()=>render(k);tabs.appendChild(b);});
render(0);
panel.innerHTML='<h3>Hover a line →</h3><p class="hint">Move over any line to see which structural section it belongs to and whether the cleaner keeps or removes it.</p>';
"""


def main():
    anns = sorted(glob.glob(f"{SDIR}/ann_*.json"))
    allp = [build_doc(a) for a in anns]
    payload = [p for p in allp if not BF.is_bad(p["badness"])]
    dropped = [p for p in allp if BF.is_bad(p["badness"])]
    if dropped:
        print("EXCLUDED by greek_badness_score>60:", ", ".join(f"{p['id']}({p['badness']})" for p in dropped))
    html = (
        "<!DOCTYPE html><html lang=el><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1'>"
        "<title>Document structure — ToC + bibliography review</title><style>" + CSS + "</style></head><body><div class=wrap>"
        "<p style='margin:0 0 1em'><a href='/'>← presentations</a></p>"
        "<h1>Document structure segmentation — v2 annotation review</h1>"
        "<p class=sub>10 documents · Opus marks <b>ToC</b> + <b>bibliographies</b>; front-matter / main-text / appendix are derived from those anchors · hover any line</p>"
        "<div class=legend>"
        "<span><span class='sw' style='background:var(--c-main)'></span><b>main text — KEEP</b></span>"
        "<span><span class='sw' style='background:var(--c-toc);border-color:var(--b-toc)'></span>table of contents</span>"
        "<span><span class='sw' style='background:var(--c-bib);border-color:var(--b-bib)'></span>bibliography</span>"
        "<span><span class='sw' style='background:var(--c-chbib);border-color:var(--b-chbib)'></span>chapter bibliography</span>"
        "<span><span class='sw' style='background:var(--c-cv);border-color:var(--b-cv)'></span>author's publications</span>"
        "<span><span class='sw' style='background:var(--c-front)'></span>front-matter / appendix (derived)</span>"
        "</div><div class=tabs id=tabs></div><div class=grid><div id=doc></div><div class=panel id=panel></div></div>"
        "</div><script>const DOCS=" + json.dumps(payload, ensure_ascii=False) + ";\n" + JS + "</script></body></html>"
    )
    out = os.path.expanduser(f"~/presentations/glossapi-tokenizer-extension/{OUT_NAME}/index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out}  ({len(payload)} docs, {sum(len(p['lines']) for p in payload)} rendered lines)")
    for p in payload:
        ns = ", ".join(f"{s['code']}[{s['a']}-{s['b']}]" for s in p["spans"])
        print(f"  {p['source']:<12} {p['id']:<14} {p['doc_type']:<6} {ns}")


if __name__ == "__main__":
    main()
