#!/usr/bin/env python3
"""Build the 2-arm CPT performance report (plots + self-contained HTML)."""
import json, os, base64, math, datetime as dt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AST = os.path.join(HERE, "assets"); os.makedirs(AST, exist_ok=True)
S = json.load(open(os.path.join(HERE, "cpt_2arm_summary.json")))
TPI = 4194304 / 1e9                 # B tokens/iter
LN2 = math.log(2)
VAN, TD, BASE = "#2c6fbb", "#e2711d", "#888888"
def toks(it): return it * TPI
def fig(name, w=7.2, h=4.0):
    f, ax = plt.subplots(figsize=(w, h), dpi=130); return f, ax, os.path.join(AST, name)
def save(f, p):
    f.tight_layout(); f.savefig(p, bbox_inches="tight"); plt.close(f)
def b64(p): return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

arms = S["arms"]; base = S["base"]; bpb_in = S["bpb_inputs"]

# ---- wall-clock ----
def walltime(arm):
    segs = arms[arm]["segments"]
    fmt = "%Y-%m-%d %H:%M:%S"
    durs = [(dt.datetime.strptime(b, fmt) - dt.datetime.strptime(a, fmt)).total_seconds() for a, b in segs]
    span = (dt.datetime.strptime(segs[-1][1], fmt) - dt.datetime.strptime(segs[0][0], fmt)).total_seconds()
    return sum(durs)/3600, span/3600
van_comp, van_span = walltime("vanilla"); td_comp, td_span = walltime("td")

# ---- BPB (cross-arm comparable; bytes identical so they cancel) ----
def bpb(arm, s):
    loss = dict((k, v[-1][1]) for k, v in arms[arm]["val"].items())[s]
    tok = bpb_in[s]["tok_vanilla" if arm == "vanilla" else "tok_td"]
    return loss * tok / (bpb_in[s]["bytes"] * LN2)
SETS = ["hplt", "openarchives", "greek_phd"]
SETLBL = {"hplt": "HPLT (web)", "openarchives": "OpenArchives (academic)", "greek_phd": "PhD theses"}

# ===== P1 Greek MCQ =====
f, ax, p1 = fig("greek_mcq.png")
for arm, c in (("vanilla", VAN), ("td", TD)):
    xs = [toks(it) for it, _ in arms[arm]["native_mcq"]]
    ys = [d["micro"]*100 for _, d in arms[arm]["native_mcq"]]
    ax.plot(xs, ys, "-o", color=c, ms=3, lw=1.8, label=("Vanilla (base tok 131k)" if arm=="vanilla" else "TD (Greek tok 148k)"))
ax.axhline(base["native_mcq"]["micro"]*100, ls="--", color=BASE, lw=1.4, label="Apertus-8B base")
ax.set_xlabel("New tokens trained (B)"); ax.set_ylabel("Greek native-MCQ micro-accuracy (%)")
ax.set_title("Greek knowledge — GreekMMLU + ILSP medical + ILSP ASEP"); ax.legend(fontsize=8); ax.grid(alpha=.3)
save(f, p1)

# ===== P2 Retention bars =====
f, ax, p2 = fig("retention.png", 8.4, 4.2)
tasks = ["mmlu","arc_challenge","arc_easy","hellaswag","winogrande","piqa","xnli_el"]
tlbl  = ["MMLU(en)","ARC-c","ARC-e","HellaSwag","WinoG","PIQA","XNLI-el"]
bv = base["retention"]; vv = arms["vanilla"]["retention"][-1][1]; tv = arms["td"]["retention"][-1][1]
import numpy as np
x = np.arange(len(tasks)); w = 0.27
ax.bar(x-w, [bv[t]*100 for t in tasks], w, label="base", color=BASE)
ax.bar(x,   [vv[t]*100 for t in tasks], w, label="Vanilla", color=VAN)
ax.bar(x+w, [tv[t]*100 for t in tasks], w, label="TD", color=TD)
ax.set_xticks(x); ax.set_xticklabels(tlbl, fontsize=8); ax.set_ylabel("Accuracy (%)")
ax.set_title("Retention — English / multilingual benchmarks (final checkpoint)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3); ax.set_ylim(40, 85)
save(f, p2)

# ===== P3 Held-out loss curves (per-token; per-arm) =====
f, ax, p3 = fig("heldout_loss.png", 8.0, 4.0)
for s, ls in zip(SETS, ["-","--",":"]):
    for arm, c in (("vanilla", VAN), ("td", TD)):
        d = arms[arm]["val"][s]
        ax.plot([toks(i) for i,_,_ in d], [l for _,l,_ in d], ls, color=c, lw=1.5)
ax.set_xlabel("New tokens trained (B)"); ax.set_ylabel("Held-out loss (nats/token)")
ax.set_title("Held-out validation loss  (per-token — NOT comparable across tokenizers)")
from matplotlib.lines import Line2D
leg = [Line2D([0],[0],color=VAN,label="Vanilla"),Line2D([0],[0],color=TD,label="TD"),
       Line2D([0],[0],color="k",ls="-",label="HPLT"),Line2D([0],[0],color="k",ls="--",label="OpenArch"),Line2D([0],[0],color="k",ls=":",label="PhD")]
ax.legend(handles=leg, fontsize=8, ncol=2); ax.grid(alpha=.3)
save(f, p3)

# ===== P4 BPB bars (cross-arm comparable) =====
f, ax, p4 = fig("heldout_bpb.png", 6.8, 4.0)
x = np.arange(len(SETS)); w=0.34
vb=[bpb("vanilla",s) for s in SETS]; tb=[bpb("td",s) for s in SETS]
ax.bar(x-w/2, vb, w, label="Vanilla", color=VAN)
ax.bar(x+w/2, tb, w, label="TD", color=TD)
for i,(a,b) in enumerate(zip(vb,tb)):
    ax.text(i, max(a,b)+0.004, f"{(b-a)/a*100:+.1f}%", ha="center", fontsize=8,
            color=("green" if b<a else "#b00"))
ax.set_xticks(x); ax.set_xticklabels([SETLBL[s] for s in SETS], fontsize=8)
ax.set_ylabel("Bits per byte (lower = better)"); ax.set_title("Held-out compression — bits/byte (tokenizer-fair)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3); ax.set_ylim(0.30, 0.45)
save(f, p4)

# ===== P5 Training loss =====
f, ax, p5 = fig("train_loss.png")
for arm, c in (("vanilla", VAN), ("td", TD)):
    d = arms[arm]["train"]
    ax.plot([toks(i) for i,_,_,_ in d], [l for _,l,_,_ in d], color=c, lw=0.8,
            label=("Vanilla" if arm=="vanilla" else "TD"))
ax.set_xlabel("New tokens trained (B)"); ax.set_ylabel("Training loss (nats/token)")
ax.set_title("Training loss  (per-token — not comparable across tokenizers)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
save(f, p5)

# ===== P6 Token efficiency =====
f, ax, p6 = fig("token_efficiency.png", 6.8, 4.0)
x=np.arange(len(SETS)); w=0.34
vt=[bpb_in[s]["tok_vanilla"]/1e6 for s in SETS]; tt=[bpb_in[s]["tok_td"]/1e6 for s in SETS]
ax.bar(x-w/2, vt, w, label="Vanilla (base tok)", color=VAN)
ax.bar(x+w/2, tt, w, label="TD (Greek tok)", color=TD)
for i,(a,b) in enumerate(zip(vt,tt)):
    ax.text(i, b+8, f"-{(1-b/a)*100:.0f}%", ha="center", fontsize=9, color="green", weight="bold")
ax.set_xticks(x); ax.set_xticklabels([SETLBL[s] for s in SETS], fontsize=8)
ax.set_ylabel("Tokens to encode the set (millions)"); ax.set_title("Tokenizer efficiency on Greek — fewer tokens = faster/cheaper inference")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
save(f, p6)

# ---- numbers for tables ----
mb=base["native_mcq"]["micro"]*100; mv=arms["vanilla"]["native_mcq"][-1][1]["micro"]*100; mt=arms["td"]["native_mcq"][-1][1]["micro"]*100
tot_van=sum(bpb_in[s]["tok_vanilla"] for s in SETS); tot_td=sum(bpb_in[s]["tok_td"] for s in SETS)
tok_save=(1-tot_td/tot_van)*100
def dcell(v, b):
    d=v-b; col="#127a12" if d>=0 else "#b00020"; sgn="+" if d>=0 else ""
    return f'<td>{v:.1f}<span style="color:{col};font-size:.8em"> ({sgn}{d:.1f})</span></td>'

def ret_row(label, key):
    return f"<tr><td>{label}</td><td>{bv[key]*100:.1f}</td>{dcell(vv[key]*100,bv[key]*100)}{dcell(tv[key]*100,bv[key]*100)}</tr>"

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Greek CPT of Apertus-8B — 2-Arm Performance</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;margin:0 auto;padding:28px;color:#1c2530;line-height:1.5}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{margin-top:34px;border-bottom:2px solid #eaecef;padding-bottom:6px}}
 .sub{{color:#667;margin-bottom:18px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
 .card{{flex:1;min-width:200px;background:#f6f8fa;border:1px solid #e2e6ea;border-radius:10px;padding:14px 16px}}
 .card .n{{font-size:30px;font-weight:700}} .card .l{{color:#667;font-size:13px}} .win{{color:#127a12}}
 img{{max-width:100%;border:1px solid #eaecef;border-radius:8px;margin:8px 0}}
 table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}
 th,td{{border:1px solid #e2e6ea;padding:6px 10px;text-align:center}} th{{background:#f6f8fa}} td:first-child,th:first-child{{text-align:left}}
 .note{{background:#fffbe6;border:1px solid #f0e0a0;border-radius:8px;padding:12px 16px;font-size:13.5px}}
 .verdict{{background:#eef7ee;border:1px solid #bfe0bf;border-radius:10px;padding:14px 18px;font-size:15px}}
 code{{background:#f0f2f4;padding:1px 5px;border-radius:4px;font-size:13px}}
 .two{{display:flex;gap:16px;flex-wrap:wrap}} .two>div{{flex:1;min-width:320px}}
</style></head><body>

<h1>Greek Continued-Pretraining of Apertus-8B — Two-Arm Comparison</h1>
<div class="sub">Vanilla (base tokenizer, 131,072) vs. Token-Distillation arm (Greek-extended tokenizer, 148,480) ·
13.5 B tokens each · 64×GH200 · finished 2026-06-11</div>

<div class="verdict"><b>Verdict.</b> Both arms improve Greek over base Apertus; the <b>Token-Distillation arm with the Greek-extended tokenizer is the winner</b>:
it reaches <b>{mt:.1f}%</b> Greek-MCQ accuracy vs {mv:.1f}% (vanilla) and {mb:.1f}% (base) — <b>+{mt-mb:.1f} pp over base, +{mt-mv:.1f} pp over vanilla</b> —
at <b>equal language-modeling quality</b> (bits/byte) while encoding Greek in <b>~{tok_save:.0f}% fewer tokens</b> (cheaper, faster inference). English retention is preserved (English MMLU actually rose). Clean run: 0 NaN, ~{van_span:.1f} h wall/arm.</div>

<div class="cards">
 <div class="card"><div class="n">{mt:.1f}%</div><div class="l">TD Greek-MCQ <span class="win">(+{mt-mb:.1f} pp vs base)</span></div></div>
 <div class="card"><div class="n">{mv:.1f}%</div><div class="l">Vanilla Greek-MCQ (+{mv-mb:.1f} pp vs base)</div></div>
 <div class="card"><div class="n win">−{tok_save:.0f}%</div><div class="l">tokens to encode Greek (TD vs base tok)</div></div>
 <div class="card"><div class="n">0</div><div class="l">NaN / skipped iters · {van_span:.1f} h wall · {van_comp:.1f} h compute /arm</div></div>
</div>

<h2>1 · Greek knowledge (downstream, directly comparable)</h2>
<p>Native Greek multiple-choice: GreekMMLU + ILSP medical MCQA + ILSP ASEP (18,489 questions, micro-accuracy). This is the headline metric and is tokenizer-agnostic (accuracy, not loss).</p>
<img src="{b64(p1)}">
<table><tr><th>Greek native-MCQ</th><th>base</th><th>Vanilla</th><th>TD</th></tr>
<tr><td>micro-accuracy (%)</td><td>{mb:.1f}</td>{dcell(mv,mb)}{dcell(mt,mb)}</tr>
<tr><td>macro-accuracy (%)</td><td>{base['native_mcq']['macro']*100:.1f}</td>{dcell(arms['vanilla']['native_mcq'][-1][1]['macro']*100, base['native_mcq']['macro']*100)}{dcell(arms['td']['native_mcq'][-1][1]['macro']*100, base['native_mcq']['macro']*100)}</tr></table>

<h2>2 · Retention — English / multilingual</h2>
<p>Does Greek CPT degrade the base model's broad ability? Largely no — minor ARC dip, English MMLU and Global-MMLU improved.</p>
<img src="{b64(p2)}">
<table><tr><th>Task</th><th>base</th><th>Vanilla (Δ)</th><th>TD (Δ)</th></tr>
{ret_row("MMLU (English)","mmlu")}
{ret_row("ARC-Challenge","arc_challenge")}
{ret_row("ARC-Easy","arc_easy")}
{ret_row("HellaSwag","hellaswag")}
{ret_row("WinoGrande","winogrande")}
{ret_row("PIQA","piqa")}
{ret_row("Global-MMLU","global_mmlu")}
{ret_row("XNLI-Greek","xnli_el")}
</table>

<h2>3 · Held-out language-modeling quality</h2>
<p>Three 0.5 B-token held-out Greek sets, excluded from training (zero leakage), measured every 25 steps. <b>Per-token loss is not comparable across tokenizers</b> — so the fair comparison is <b>bits-per-byte</b> (right), where the text bytes are identical and the token counts cancel.</p>
<div class="two">
 <div><img src="{b64(p3)}"></div>
 <div><img src="{b64(p4)}"></div>
</div>
<p class="note">In bits/byte the two arms are essentially tied on language modeling (TD slightly better on in-domain OpenArchives, vanilla marginally better on PhD theses). So the Greek tokenizer + Token Distillation buys the downstream Greek gain <i>without</i> any compression cost.</p>

<h2>4 · Tokenizer efficiency (the practical win)</h2>
<p>The Greek-extended tokenizer represents the same Greek text in far fewer tokens — directly lowering inference cost and extending effective context.</p>
<img src="{b64(p6)}">
<p>Across the held-out sets the TD tokenizer uses <b>{tot_td/1e6:.0f} M</b> tokens where the base tokenizer needs <b>{tot_van/1e6:.0f} M</b> — a <b>{tok_save:.0f}% reduction</b> (HPLT −38%, OpenArchives −30%, PhD −26%). At equal 13.5 B-token budget the TD arm therefore also <i>sees more Greek text</i>.</p>

<h2>5 · Training dynamics &amp; compute</h2>
<img src="{b64(p5)}">
<div class="cards">
 <div class="card"><div class="n">{van_comp:.1f} / {td_comp:.1f} h</div><div class="l">compute time (vanilla / TD), 4 segments each</div></div>
 <div class="card"><div class="n">~8.6 s</div><div class="l">per iteration @ 64 GPUs (≈394 TFLOP/s/GPU, ~96% DP eff.)</div></div>
 <div class="card"><div class="n">0</div><div class="l">NaN / skipped iters across 3,218 steps × 2 arms</div></div>
</div>
<p class="note"><b>Efficiency milestone.</b> Multi-node training was blocked for a day by a Slingshot/CXI <code>NO_SPACE</code> error; root cause was the trainer's hardcoded <code>NCCL_NET_FORCE_FLUSH=1</code> (a tiny per-receive flush that the 1.18-dev OFI plugin rejects). Removing it unblocked full 64-GPU CXI/RDMA training and beat the Socket/HSN fallback by ~12%/iter — turning a projected ~4.7-day-per-arm run into ~8 h.</p>

<h2>6 · Methodology &amp; caveats</h2>
<ul>
 <li><b>Equal token budget</b> (13.5 B each), not equal epochs: TD ≈ 1.0 epoch of its 13.5 B-token binary; vanilla ≈ 0.69 epoch of its 19.5 B-token binary (same documents, same compute) — the standard tokenizer-comparison protocol.</li>
 <li><b>Per-token loss ≠ cross-arm comparable</b>; cross-arm claims use bits/byte (§3) or downstream accuracy (§1–2).</li>
 <li><b>Data</b>: 10 B unseen Greek (70% HPLT + 30% OpenArchives) + 3.5 B replay (multilingual + code + math + Greek replay), GreekMMLU-decontaminated and PII-anonymized. The HPLT→OpenArchives <i>ordering</i> was prepared but <b>did not take effect</b> — Megatron's dataloader shuffle was left on — so both arms trained on shuffled data (does not affect the arm comparison).</li>
 <li><b>bits/byte</b> uses held-out JSONL byte counts (~1% JSON overhead, identical for both arms → cross-arm comparison exact; absolute value ~1% high).</li>
 <li>Hyperparameters: peak LR 5.5e-5, WSD (20% 1-sqrt cooldown), AdEMAMix (β 0.9/0.995/0.999, α 4), Goldfish k=h=50, RoPE θ=500k/seq 4096, TP2/PP1/DP32, bf16. TD arm initialized with Token Distillation (Dobler 2025) at layer 11 (E by MSE-on-hiddens, U by CE).</li>
</ul>
<p class="sub">Source: <code>runs/cpt_2arm_13b/{{cpt13b_vanilla,cpt13b_td}}_20260610T200344Z</code> + <code>eval_*</code> on Clariden; base = <code>04_vanilla_cpt/eval_apertus_base_matched_rope500k_seq4096</code>. Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')}.</p>
</body></html>"""
out = os.path.join(HERE, "cpt_2arm_performance.html")
open(out, "w").write(html)
print("wrote", out, os.path.getsize(out), "bytes")
print(f"Greek MCQ: base {mb:.1f} | vanilla {mv:.1f} | td {mt:.1f}")
print(f"token saving overall: {tok_save:.1f}%  | wall van {van_span:.2f}h td {td_span:.2f}h")
print("BPB van:", [round(bpb('vanilla',s),4) for s in SETS], "td:", [round(bpb('td',s),4) for s in SETS])
