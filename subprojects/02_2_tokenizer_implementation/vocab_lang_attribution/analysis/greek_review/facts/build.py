"""Bare-facts Greek frequency plots — light theme, no decoration.

Produces:
  plots/01_greek_freq.png             — all Greek-firing tokens, rank vs count log-log,
                                        coloured Greek-script vs not.
  plots/02_non_greek_in_greek.png     — top 50 non-Greek-script tokens in Greek
                                        as a horizontal bar chart with decoded labels.
  plots/03_categories.png             — non-Greek-script tokens in Greek, grouped by category,
                                        stacked-bar of token count vs Greek-mass.
  tables/top50_non_greek_in_greek.tsv — same data as plot 2 in TSV form.
  tables/top50_greek_script.tsv       — top Greek-script tokens, for reference.
  index.html                          — minimal page that ties them together.
"""

from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent     # vocab_lang_attribution/
PLOTS = HERE / "plots"; TABLES = HERE / "tables"
PLOTS.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor": "white",
    "axes.edgecolor":   "#888",
    "axes.labelcolor":  "#222",
    "axes.titlecolor":  "#222",
    "xtick.color":      "#333",
    "ytick.color":      "#333",
    "text.color":       "#222",
    "grid.color":       "#dddddd",
    "grid.alpha":       0.7,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "normal",
})

GREEK_C  = "#a82020"
OTHER_C  = "#444444"
SUBTLE   = "#777"

# --- Load -------------------------------------------------------------------
z = np.load(ROOT / "outputs/histogram_matrix.npz", allow_pickle=True)
H  = z["H"].astype(np.int64)
ck = list(z["canonical_keys"])
tok = pl.read_parquet(ROOT / "outputs/token_metadata.parquet")
ell = ck.index("ell_Grek")
greek = H[ell]                                # (131072,)
total_greek = int(greek.sum())
is_gs = (tok["has_greek_mono"] | tok["has_greek_poly"]).to_numpy()
decoded = tok["decoded_string"].to_list()
has_latin   = (tok["has_latin_basic"] | tok["has_latin_extended"]).to_numpy()
is_struct   = tok["is_structural_only"].to_numpy()
is_digit    = tok["is_pure_digits"].to_numpy()
is_ws       = tok["is_pure_whitespace"].to_numpy()
is_special  = tok["is_special"].to_numpy()
is_byte     = tok["is_byte_fragment"].to_numpy()

fired = greek > 0
n_fired = int(fired.sum())
n_gs_fired = int((is_gs & fired).sum())
n_non_gs_fired = int((~is_gs & fired).sum())


def safe_label(s: str, n: int = 14) -> str:
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if s == "": return "(empty)"
    if s.isspace(): return f"(ws×{len(s)})"
    return s[:n] + ("…" if len(s) > n else "")


# --- Plot 1: all Greek-firing tokens, log-log, by Greek-script vs not -------
print("plot 1 — all Greek-firing tokens")
order = np.argsort(-greek)
greek_sorted = greek[order]
gs_sorted = is_gs[order]
rank = np.arange(1, len(greek_sorted) + 1)

fig, ax = plt.subplots(figsize=(10, 5.2))
m_other = (~gs_sorted) & (greek_sorted > 0)
m_gs    = (gs_sorted)  & (greek_sorted > 0)
ax.scatter(rank[m_other], greek_sorted[m_other], s=2, c=OTHER_C, alpha=0.55,
           label=f"not Greek-script  ({n_non_gs_fired:,})")
ax.scatter(rank[m_gs], greek_sorted[m_gs], s=4, c=GREEK_C, alpha=0.9,
           label=f"Greek-script  ({n_gs_fired:,})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("rank by count in Greek sample")
ax.set_ylabel("count in Greek sample")
ax.set_title(f"Token counts in the Greek (ell_Grek) sample · "
             f"{n_fired:,} of 131,072 vocab entries fired · sample = {total_greek/1e9:.2f} B")
ax.grid(True, which="both", alpha=0.4)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#bbb")
fig.tight_layout()
fig.savefig(PLOTS / "01_greek_freq.png", dpi=140)
plt.close(fig)


# --- Plot 2: top 50 non-Greek-script tokens in Greek ------------------------
print("plot 2 — top 50 non-Greek-script tokens")
non_gs_idx = np.where(~is_gs & fired)[0]
top50_idx = non_gs_idx[np.argsort(-greek[non_gs_idx])][:50]
labels = [safe_label(decoded[i]) for i in top50_idx]
counts = greek[top50_idx]

fig, ax = plt.subplots(figsize=(10, 13))
y = np.arange(len(top50_idx))[::-1]
ax.barh(y, counts, color="#555", edgecolor="white", linewidth=0.3)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10, fontfamily="DejaVu Sans Mono")
ax.set_xlabel("count in Greek sample")
ax.set_title(f"Top 50 non-Greek-script tokens that fired in Greek\n"
             f"(of {n_non_gs_fired:,} total non-Greek-script tokens that fired ≥ 1 time)",
             loc="left")
ax.grid(True, axis="x", alpha=0.3)
ax.set_axisbelow(True)
# Annotate counts
for yi, c in zip(y, counts):
    ax.text(c + counts.max() * 0.01, yi, f"{int(c):,}", va="center", fontsize=9,
            color="#333", fontfamily="DejaVu Sans Mono")
ax.set_xlim(0, counts.max() * 1.18)
fig.tight_layout()
fig.savefig(PLOTS / "02_non_greek_in_greek.png", dpi=140)
plt.close(fig)


# --- Plot 3: non-Greek-script tokens by category ----------------------------
print("plot 3 — non-Greek-script tokens grouped by category")
# Mutually-exclusive categorisation, priority order
used = np.zeros(len(decoded), dtype=bool)
def mask_and_use(m):
    out = m & (~is_gs) & fired & (~used)
    used[:] = used | out
    return out

cats = [
    ("Structural only (punctuation)", mask_and_use(is_struct), "#456f8a"),
    ("Pure digits",                   mask_and_use(is_digit),  "#9d7a3d"),
    ("Whitespace only",               mask_and_use(is_ws),     "#7a7a7a"),
    ("Special tokens",                mask_and_use(is_special),"#a04a72"),
    ("Byte fragments",                mask_and_use(is_byte),   "#7a4a30"),
    ("Latin letters (text-like)",     mask_and_use(has_latin), "#3a7a4a"),
    ("Other",                         mask_and_use(np.ones_like(used)), "#888888"),
]

# Compute token-count and mass per category
cat_names = [c[0] for c in cats]
cat_tokens = [int(c[1].sum()) for c in cats]
cat_mass   = [int(greek[c[1]].sum()) for c in cats]
cat_colors = [c[2] for c in cats]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
y = np.arange(len(cat_names))[::-1]
ax1.barh(y, cat_tokens, color=cat_colors, edgecolor="white")
for yi, n, name in zip(y, cat_tokens, cat_names):
    ax1.text(n + max(cat_tokens) * 0.01, yi, f"{n:,}", va="center", fontsize=10, fontfamily="DejaVu Sans Mono")
ax1.set_yticks(y); ax1.set_yticklabels(cat_names, fontsize=10)
ax1.set_xlabel("number of distinct vocab entries")
ax1.set_title("How many distinct tokens", loc="left")
ax1.grid(True, axis="x", alpha=0.3); ax1.set_axisbelow(True)
ax1.set_xlim(0, max(cat_tokens) * 1.18)

ax2.barh(y, [m / 1e6 for m in cat_mass], color=cat_colors, edgecolor="white")
for yi, m in zip(y, cat_mass):
    ax2.text(m/1e6 + max(cat_mass)/1e6 * 0.01, yi,
             f"{m/1e6:,.1f} M  ({100*m/total_greek:.2f}%)",
             va="center", fontsize=10, fontfamily="DejaVu Sans Mono")
ax2.set_yticks(y); ax2.set_yticklabels([])
ax2.set_xlabel("total count in Greek sample (millions)")
ax2.set_title("Greek-mass carried", loc="left")
ax2.grid(True, axis="x", alpha=0.3); ax2.set_axisbelow(True)
ax2.set_xlim(0, max(cat_mass)/1e6 * 1.30)

fig.suptitle(f"Non-Greek-script tokens that fired in Greek — by category\n"
             f"({n_non_gs_fired:,} distinct vocab entries, "
             f"{(total_greek - int(greek[is_gs].sum()))/1e6:.1f} M tokens = "
             f"{100*(1 - greek[is_gs].sum()/total_greek):.2f}% of Greek-sample mass)",
             y=1.0, fontsize=12)
fig.tight_layout()
fig.savefig(PLOTS / "03_categories.png", dpi=140)
plt.close(fig)


# --- Tables -----------------------------------------------------------------
print("tables — top 50 non-Greek-script and top 50 Greek-script")

def write_table(path, idxs):
    with open(path, "w") as f:
        f.write("rank\ttoken_id\tdecoded\tcount\tshare_of_greek_mass_pct\tcategory\n")
        for rank, i in enumerate(idxs, 1):
            cat = "structural" if is_struct[i] else \
                  "digit"      if is_digit[i] else \
                  "whitespace" if is_ws[i] else \
                  "special"    if is_special[i] else \
                  "byte_frag"  if is_byte[i] else \
                  "latin"      if has_latin[i] else \
                  "greek"      if is_gs[i] else "other"
            f.write(f"{rank}\t{i}\t{decoded[i]!r}\t{int(greek[i])}\t"
                    f"{100*float(greek[i])/total_greek:.4f}\t{cat}\n")

write_table(TABLES / "top50_non_greek_in_greek.tsv", top50_idx)

# Top 50 Greek-script
gs_idx_all = np.where(is_gs & fired)[0]
top50_gs = gs_idx_all[np.argsort(-greek[gs_idx_all])][:50]
write_table(TABLES / "top50_greek_script.tsv", top50_gs)


# --- HTML -------------------------------------------------------------------
print("writing index.html")

def html_table_rows(idxs, limit=50):
    rows = []
    for rank, i in enumerate(idxs[:limit], 1):
        dec = decoded[i]
        # render safe
        disp = html.escape(dec.replace("\n", "↵").replace("\t", "→"))
        if disp == "": disp = "<i>(empty)</i>"
        cat = ("structural" if is_struct[i] else
               "digit"      if is_digit[i] else
               "whitespace" if is_ws[i] else
               "special"    if is_special[i] else
               "byte_frag"  if is_byte[i] else
               "latin"      if has_latin[i] else
               "greek"      if is_gs[i] else "other")
        rows.append(
            f"<tr><td class='num'>{rank}</td>"
            f"<td><code>{disp}</code></td>"
            f"<td class='num'>{int(greek[i]):,}</td>"
            f"<td class='num'>{100*float(greek[i])/total_greek:.3f}%</td>"
            f"<td class='cat cat-{cat}'>{cat}</td></tr>"
        )
    return "\n".join(rows)

non_gs_rows = html_table_rows(top50_idx)
gs_rows     = html_table_rows(top50_gs)

INDEX_HTML = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>Token counts in the Greek sample</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          color: #222; background: #fff; max-width: 1080px; margin: 32px auto;
          padding: 0 24px; line-height: 1.5; }}
  h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 6px; }}
  h2 {{ font-size: 16px; font-weight: 600; margin: 44px 0 8px; padding-top: 18px;
        border-top: 1px solid #ddd; }}
  p.meta {{ color: #555; font-size: 14px; margin: 0 0 8px; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 18px 0;
         border: 1px solid #ddd; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0 28px; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f7f7f7; font-weight: 600; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.04em; color: #444; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums;
            font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
  code {{ background: #f0f0f0; padding: 1px 6px; border-radius: 2px;
          font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
          font-size: 13px; color: #b03020; }}
  .cat {{ font-size: 11px; padding: 2px 6px; border-radius: 2px;
          font-family: ui-monospace, monospace; }}
  .cat-structural {{ background: #e8effa; color: #2a4a6a; }}
  .cat-digit {{ background: #f7eedc; color: #6e4f1f; }}
  .cat-whitespace {{ background: #ededed; color: #555; }}
  .cat-latin {{ background: #e6f0e6; color: #2a5a3a; }}
  .cat-byte_frag {{ background: #f0e0d0; color: #6b4222; }}
  .cat-special {{ background: #f3e0ec; color: #663049; }}
  .cat-other {{ background: #f0f0f0; color: #555; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px;
                margin: 16px 0 8px; }}
  .stat-grid div {{ }}
  .stat-grid dt {{ font-size: 11px; color: #666; text-transform: uppercase;
                   letter-spacing: 0.05em; margin-bottom: 2px; }}
  .stat-grid dd {{ font-size: 18px; font-weight: 500; color: #111;
                   font-variant-numeric: tabular-nums; margin: 0; }}
</style></head><body>

<h1>Token counts in the Greek (<code>ell_Grek</code>) sample</h1>
<p class="meta">Sample: FineWeb2-HQ <code>ell_Grek</code>, tokenised by Apertus-8B-2509,
   {total_greek:,} content tokens. 1,933 canonical languages probed at 1 B each.</p>

<dl class="stat-grid">
  <div><dt>tokens fired in Greek</dt><dd>{n_fired:,} / 131,072</dd></div>
  <div><dt>Greek-script tokens fired</dt><dd>{n_gs_fired:,} / 1,507</dd></div>
  <div><dt>non-Greek-script fired</dt><dd>{n_non_gs_fired:,}</dd></div>
  <div><dt>total tokens in Greek sample</dt><dd>{total_greek/1e9:.2f} B</dd></div>
</dl>

<h2>1 · Frequency of every token that fired in Greek</h2>
<p class="meta">All {n_fired:,} vocab entries that fired at least once, sorted by Greek count.
Log-log axes. Red = contains a Greek codepoint; grey = does not.</p>
<img src="plots/01_greek_freq.png" alt="Greek token frequency log-log">

<h2>2 · Non-Greek-script tokens that fired in Greek — top 50</h2>
<p class="meta">{n_non_gs_fired:,} distinct vocab entries fired in Greek without
containing a Greek codepoint. These are dominantly punctuation, digits, and
the byte-level structural tokens shared across all languages. Decoded forms
shown; <code>↵</code> = newline.</p>
<img src="plots/02_non_greek_in_greek.png" alt="Top 50 non-Greek-script tokens">

<table>
  <thead><tr><th>#</th><th>token (decoded)</th><th>count</th><th>share</th><th>category</th></tr></thead>
  <tbody>
  {non_gs_rows}
  </tbody>
</table>

<h2>3 · Non-Greek-script tokens, grouped by category</h2>
<p class="meta">Same {n_non_gs_fired:,} non-Greek-script firings, partitioned into
mutually-exclusive categories. Left bar: distinct tokens; right bar: Greek-mass carried.</p>
<img src="plots/03_categories.png" alt="Categories of non-Greek-script tokens">

<h2>4 · Greek-script tokens — top 50 (reference)</h2>
<p class="meta">Top 50 of the 1,504 Greek-script tokens that fired.</p>
<table>
  <thead><tr><th>#</th><th>token</th><th>count</th><th>share</th><th>category</th></tr></thead>
  <tbody>
  {gs_rows}
  </tbody>
</table>

<p class="meta" style="margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd;">
Raw tables: <code>tables/top50_non_greek_in_greek.tsv</code>,
<code>tables/top50_greek_script.tsv</code>.
Underlying matrix: <code>outputs/histogram_matrix.npz</code>.</p>

</body></html>
"""

(HERE / "index.html").write_text(INDEX_HTML)
print("DONE")
print(f"  tokens fired in Greek:           {n_fired:,}")
print(f"  Greek-script tokens that fired:  {n_gs_fired:,} / 1,507")
print(f"  non-Greek-script tokens fired:   {n_non_gs_fired:,}")
