#!/usr/bin/env python3
"""Build the bibliography-span dataset presentation (figure + academic-serif HTML) and write it to
~/presentations/glossapi-tokenizer-extension/bibliography-span-dataset/ for the results hub."""
import json, os, collections, base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser("~/presentations/glossapi-tokenizer-extension/bibliography-span-dataset")
os.makedirs(OUT, exist_ok=True)
INK, MUTED, LINE, BG, ACCENT = "#1a1a1a", "#5b5b5b", "#d9d4c8", "#faf8f3", "#6b1f1f"
PAL = ["#6b1f1f", "#1f6b34", "#9a6a00", "#2f5d8a", "#6b4f8a", "#8a6a2f", "#5b5b5b"]

rows = [json.loads(l) for l in open(f"{HERE}/span_dataset.jsonl") if l.strip()]
N = len(rows)
bydoc = collections.defaultdict(list)
for r in rows:
    bydoc[r["doc_id"]].append(r)
ndoc = len(bydoc)


def count(k):
    return collections.Counter(r[k] for r in rows)


def panel(ax, title, counter, order=None):
    items = [(k, counter[k]) for k in (order or [k for k, _ in counter.most_common()])]
    items = [(k, v) for k, v in items if v > 0]
    labels = [k.replace("_", " ") for k, _ in items]
    vals = [v for _, v in items]
    ax.barh(range(len(vals)), vals, color=[PAL[i % len(PAL)] for i in range(len(vals))], height=.72)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * .01, i, str(v), va="center", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)
    ax.set_xlim(0, max(vals) * 1.16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0); ax.set_xticks([])


fig, axs = plt.subplots(2, 3, figsize=(12, 6.8), facecolor="white")
plt.subplots_adjust(wspace=.55, hspace=.32, left=.10, right=.985, top=.93, bottom=.04)
panel(axs[0][0], "Source", count("source"))
panel(axs[0][1], "Span kind", count("kind"))
panel(axs[0][2], "Subject register", count("subject_register"))
panel(axs[1][0], "Script", count("script"))
panel(axs[1][1], "Citation style", count("citation_style"))
# spans-per-doc multiplicity
perdoc = collections.Counter(min(len(v), 5) for v in bydoc.values())
mult = collections.Counter()
for k in sorted(perdoc):
    mult[f"{k}{'+' if k == 5 else ''} span" + ("s" if k > 1 else "")] = perdoc[k]
panel(axs[1][2], "Spans per document", mult, order=list(mult.keys()))
fig.suptitle("", fontsize=1)
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, facecolor="white"); buf.seek(0)
b64 = base64.b64encode(buf.read()).decode()
fig.savefig(f"{OUT}/distributions.png", dpi=140, facecolor="white")

multi = sum(v for k, v in perdoc.items() if k >= 2)
ne = [r["n_entries"] for r in rows if isinstance(r.get("n_entries"), int)]
src = count("source"); kind = count("kind"); script = count("script")
lang = count("language"); subj = count("subject_register"); noise = count("noise_level")


def trow(counter, total):
    return "".join(f"<tr><td>{k.replace('_',' ')}</td><td class=n>{v}</td>"
                   f"<td class=n>{v*100//total}%</td></tr>" for k, v in counter.most_common())


HTML = f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Greek Bibliography-Span Dataset</title>
<meta name=description content="{N} Opus-annotated bibliography spans across greek_phd, openarchives and Kallipos — start/end lines plus rich metadata for a multi-span detector.">
<style>
:root{{--ink:{INK};--muted:{MUTED};--line:{LINE};--bg:{BG};--accent:{ACCENT};--good:#1f6b34}}
*{{box-sizing:border-box}}html{{font-size:17px}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Iowan Old Style","Palatino Linotype",Palatino,Charter,Georgia,"Times New Roman",serif;line-height:1.55}}
.wrap{{max-width:860px;margin:0 auto;padding:54px 26px 90px}}
h1{{font-size:1.85rem;line-height:1.2;margin:0 0 .2em;font-weight:600}}
.sub{{color:var(--muted);font-style:italic;margin:0 0 1.6em}}
h2{{font-size:1.18rem;margin:2.1em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.25em}}
h2 .n{{color:var(--accent);font-variant-numeric:tabular-nums;margin-right:.5em}}
p{{margin:.6em 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:1.2em 0}}
.card{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 14px}}
.card .k{{font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}}
.card .v{{font-size:1.5rem;font-weight:600;font-variant-numeric:tabular-nums}}
.card .d{{font-size:.8rem;color:var(--muted)}}
table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.9rem}}
th,td{{border:1px solid var(--line);padding:5px 10px;text-align:left}}
th{{background:#f1ece1;font-weight:600}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
@media(max-width:680px){{.two{{grid-template-columns:1fr}}}}
figure{{margin:1.6em 0;text-align:center}}
figure img{{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}}
figcaption{{font-size:.86rem;color:var(--muted);margin-top:.6em;text-align:left}}
figcaption b{{color:var(--ink)}}
.good{{color:var(--good)}}.bad{{color:var(--accent)}}
code{{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.85em;background:#efeadf;padding:1px 5px;border-radius:4px}}
a{{color:var(--accent)}}
.foot{{margin-top:3em;color:var(--muted);font-size:.82rem;border-top:1px solid var(--line);padding-top:1em}}
</style></head><body><div class=wrap>
<p style="margin:0 0 1em"><a href="/">← presentations</a></p>
<h1>A bibliography-span dataset for the Greek academic CPT corpus</h1>
<p class=sub>{N:,} Opus-4.8-annotated spans · greek_phd · openarchives · Kallipos · start/end lines + rich metadata</p>

<div class=cards>
<div class=card><div class=k>spans</div><div class=v>{N:,}</div><div class=d>contiguous bibliographic lists</div></div>
<div class=card><div class=k>documents</div><div class=v>{ndoc:,}</div><div class=d>across 3 sources</div></div>
<div class=card><div class=k>multi-span docs</div><div class=v class=bad>{multi*100//ndoc}%</div><div class=d>≥2 bibliographies in one doc</div></div>
<div class=card><div class=k>reference entries</div><div class=v>{sum(ne)//1000}k</div><div class=d>median {int(sorted(ne)[len(ne)//2])}/span</div></div>
</div>

<h2><span class=n>1</span>What this is</h2>
<p>A labelled set of <b>every bibliographic-list span</b> in the Greek academic continued-pretraining
sources, with its <b>start and end line</b> — wherever the list sits: end-of-document, end-of-chapter,
or a subdivided sub-list. Each span carries rich metadata (kind, citation style, language, script,
subject register, noise level, entry count, header presence). It trains and evaluates a
<b>multi-span</b> bibliography detector (not a single end-boundary), and powers the Δstart/Δend
boundary eval at real scale. Annotation is blind Opus 4.8 over Docling-extracted, GlossAPI-cleaned
line-numbered windows.</p>

<figure><img src="data:image/png;base64,{b64}" alt="distributions">
<figcaption><b>Figure 1.</b> Span distribution across the six metadata axes. Sources are balanced
(greek_phd {src['greek_phd']}, Kallipos {src['kallipos']}, openarchives {src['openarchives']});
<b>end-of-chapter ({kind['end_of_chapter']}) slightly outnumbers end-of-document ({kind['end_of_document']})</b>,
and {multi*100//ndoc}% of documents carry more than one span — both arguing a single-boundary cut is
structurally insufficient.</figcaption></figure>

<h2><span class=n>2</span>Why multi-span matters</h2>
<p>The headline structural fact: <b class=bad>{multi}/{ndoc} documents ({multi*100//ndoc}%) contain
two or more distinct bibliography spans</b> — chapter reference lists, subdivided
Ελληνόγλωσση/Ξενόγλωσση sub-lists, archival-sources and web-sources blocks. A detector that finds a
single end-of-document boundary misses all of these. The near-even split between end-of-chapter and
end-of-document spans (Figure 1, top-middle) is the same story from the kind axis.</p>

<div class=two>
<div><table><tr><th>Span kind</th><th class=n>n</th><th class=n>%</th></tr>{trow(kind, N)}</table></div>
<div><table><tr><th>Subject register</th><th class=n>n</th><th class=n>%</th></tr>{trow(subj, N)}</table></div>
</div>

<h2><span class=n>3</span>A +5h unseen extension bought the tail</h2>
<p>The base sample (178 batches) was greek_phd-heavy and thin on several classes. A second pass over
<b>unseen</b> documents — disjoint from the base, drawn from all three sources — was queued behind it
specifically to fill the tail. It worked: every previously-sparse class grew several-fold.</p>
<table><tr><th>Class</th><th class=n>base only</th><th class=n>final</th><th>note</th></tr>
<tr><td>theology</td><td class=n>12</td><td class=n>{subj['theology']}</td><td>Kallipos religious-studies books</td></tr>
<tr><td>law</td><td class=n>22</td><td class=n>{subj['law']}</td><td>archival + footnote-humanities lists</td></tr>
<tr><td>polytonic Greek script</td><td class=n>12</td><td class=n>{script['polytonic_greek']}</td><td>genuinely scarce; modern corpus is monotonic</td></tr>
<tr><td>Greek-language spans</td><td class=n>69</td><td class=n>{lang['greek']}</td><td>humanities/social-science end-matter</td></tr>
<tr><td>iso690 style</td><td class=n>8</td><td class=n>{[c for k,c in count('citation_style').items() if k=='iso690'][0] if 'iso690' in count('citation_style') else 0}</td><td>STEM theses</td></tr>
</table>

<h2><span class=n>4</span>Noise &amp; format spread</h2>
<div class=two>
<div><table><tr><th>Citation style</th><th class=n>n</th><th class=n>%</th></tr>{trow(count('citation_style'), N)}</table></div>
<div><table><tr><th>Script</th><th class=n>n</th><th class=n>%</th></tr>{trow(script, N)}</table>
<table><tr><th>Noise level</th><th class=n>n</th><th class=n>%</th></tr>{trow(noise, N)}</table></div>
</div>
<p>Noise is labelled inside the span (Docling/OCR breakage): {noise['light']*100//N}% light,
{noise['clean']*100//N}% clean, {noise['heavy']*100//N}% heavy — enough heavy-noise spans
({noise['heavy']}) to test whether extraction noise predicts boundary error.</p>

<h2><span class=n>5</span>How it was built</h2>
<p>Sequential Opus annotation, paced one agent at a time to stay under a server tokens-per-minute
throttle that mass-failed the concurrent run, driven by a merge→next-chunk loop. {N:,} spans landed
across {ndoc:,} documents; one batch tripped the content filter on a single doc and was recovered
per-unit. Artefacts: <code>eval/span_dataset.jsonl</code> (one row per span) and
<code>eval/annotations_span/all.json</code> (raw per-window).</p>

<p class=foot>Greek academic reference-cleaning · stage <code>15_clean_academic</code> ·
blind Opus 4.8 annotation · {N:,} spans / {ndoc:,} docs · generated for the results hub.</p>
</div></body></html>"""

open(f"{OUT}/index.html", "w", encoding="utf-8").write(HTML)
print(f"wrote {OUT}/index.html ({len(HTML)} bytes) + distributions.png")
print(f"spans={N} docs={ndoc} multi%={multi*100//ndoc}")
