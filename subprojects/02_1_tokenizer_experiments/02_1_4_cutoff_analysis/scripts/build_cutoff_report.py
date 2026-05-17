"""Build C3_CUTOFF_REPORT.md + PNG plots.

Inputs:
  - metrics_by_slice.json from the gcloud fertility suite (pulled to
    /tmp/c3_cutoff_metrics.json on home)
  - cutoff_grid distributions at
    /home/foivos/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/cutoff_grid/distribution_at_<n>.json
Outputs:
  - docs/figures/c3_cutoff_*.png  (one PNG per metric)
  - docs/C3_CUTOFF_REPORT.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os

REPO = Path(__file__).resolve().parents[4]
_HERE = Path(__file__).resolve().parent
FIG_DIR = REPO / "docs" / "figures"
REPORT_PATH = REPO / "docs" / "C3_CUTOFF_REPORT.md"

# Default locations are now local to this sub-subproject. Override with
# env vars for one-off re-runs against fresh data.
DEFAULT_METRICS = (_HERE / "../artifacts/c3_cutoff_metrics.json").resolve()
DEFAULT_CUTOFF_GRID = (_HERE / "../artifacts/cutoff_grid").resolve()
METRICS_JSON = Path(os.environ.get("CUTOFF_METRICS_JSON", str(DEFAULT_METRICS)))
CUTOFF_GRID_DIR = Path(os.environ.get("CUTOFF_GRID_DIR", str(DEFAULT_CUTOFF_GRID)))

# Backwards-compatible fallback to the legacy /tmp + ~/runs paths so a
# fresh run that hasn't snapshotted yet still works.
if not METRICS_JSON.exists():
    legacy = Path("/tmp/c3_cutoff_metrics.json")
    if legacy.exists():
        METRICS_JSON = legacy
if not CUTOFF_GRID_DIR.exists():
    legacy_grid = Path(
        "/home/foivos/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/cutoff_grid"
    )
    if legacy_grid.exists():
        CUTOFF_GRID_DIR = legacy_grid

CUTOFFS = [n * 1024 for n in range(1, 26)]
SLICES_ORDER = ["virgin_hplt", "C3_val_clean", "C3_test_clean"]


def load_metrics() -> list[dict]:
    return json.loads(METRICS_JSON.read_text())


def cutoff_from_name(name: str) -> int | None:
    if name == "apertus_base":
        return 0
    if name.startswith("c3_added_"):
        return int(name.rsplit("_", 1)[1])
    return None


def plot_metric(rows: list[dict], metric: str, ylabel: str, fname: str, title: str, lower_better: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for slice_name in SLICES_ORDER:
        xs, ys = [], []
        for n in [0] + CUTOFFS:
            for r in rows:
                if r["slice"] != slice_name:
                    continue
                if cutoff_from_name(r["tokenizer"]) != n:
                    continue
                v = r.get(metric)
                if v is None:
                    continue
                xs.append(n)
                ys.append(v)
                break
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, label=slice_name)
    ax.set_xlabel("added units on top of Apertus base (= cutoff)")
    ax.set_ylabel(ylabel + ("  (↓ better)" if lower_better else "  (↑ better)"))
    ax.set_title(title)
    ax.set_xticks([0] + CUTOFFS[::2])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_single_metric_one_line(rows: list[dict], metric: str, ylabel: str, fname: str, title: str, agg_slices: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    xs, ys = [], []
    for n in [0] + CUTOFFS:
        vals = []
        for r in rows:
            if r["slice"] not in agg_slices:
                continue
            if cutoff_from_name(r["tokenizer"]) != n:
                continue
            v = r.get(metric)
            if v is not None:
                vals.append(v)
        if vals:
            xs.append(n)
            ys.append(sum(vals) / len(vals))
    ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, color="C2")
    ax.set_xlabel("added units on top of Apertus base (= cutoff)")
    ax.set_ylabel(ylabel)
    ax.set_title(title + f"  (avg over {len(agg_slices)} clean slices)")
    ax.set_xticks([0] + CUTOFFS[::2])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def md_table_per_slice(rows: list[dict], metric: str, header_label: str, fmt: str = "{:.4f}") -> str:
    """Markdown table: cutoff rows × slice cols."""
    lines = [f"| cutoff | " + " | ".join(SLICES_ORDER) + " |"]
    lines.append("| ---: |" + "".join(" ---: |" for _ in SLICES_ORDER))
    for n in [0] + CUTOFFS:
        cells = [str(n)]
        for s in SLICES_ORDER:
            v = next(
                (r.get(metric) for r in rows if r["slice"] == s and cutoff_from_name(r["tokenizer"]) == n),
                None,
            )
            cells.append(fmt.format(v) if v is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_cutoff_grid() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for n in CUTOFFS:
        p = CUTOFF_GRID_DIR / f"distribution_at_{n}.json"
        out[n] = json.loads(p.read_text())
    return out


def md_table_category_x_cutoff(grid: dict[int, dict]) -> str:
    cats = sorted({k for c in grid.values() for k in c["by_category"]})
    cols = CUTOFFS
    lines = ["| category | " + " | ".join(f"{c//1024}k" for c in cols) + " |"]
    lines.append("| --- |" + "".join(" ---: |" for _ in cols))
    for cat in cats:
        cells = [f"`{cat}`"]
        for c in cols:
            cells.append(str(grid[c]["by_category"].get(cat, 0)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_table_structure_x_cutoff(grid: dict[int, dict]) -> str:
    structs = sorted({k for c in grid.values() for k in c["by_greek_structure"]})
    cols = CUTOFFS
    lines = ["| structure | " + " | ".join(f"{c//1024}k" for c in cols) + " |"]
    lines.append("| --- |" + "".join(" ---: |" for _ in cols))
    for s in structs:
        cells = [f"`{s}`"]
        for c in cols:
            cells.append(str(grid[c]["by_greek_structure"].get(s, 0)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_table_lex_x_cutoff(grid: dict[int, dict]) -> str:
    lexs = sorted({k for c in grid.values() for k in c["by_greek_lexical"]})
    cols = CUTOFFS
    lines = ["| lexical | " + " | ".join(f"{c//1024}k" for c in cols) + " |"]
    lines.append("| --- |" + "".join(" ---: |" for _ in cols))
    for s in lexs:
        cells = [f"`{s}`"]
        for c in cols:
            cells.append(str(grid[c]["by_greek_lexical"].get(s, 0)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_table_conf_x_cutoff(grid: dict[int, dict]) -> str:
    order = [">=0.9", "0.7-0.9", "0.5-0.7", "<0.5"]
    cols = CUTOFFS
    lines = ["| confidence | " + " | ".join(f"{c//1024}k" for c in cols) + " |"]
    lines.append("| --- |" + "".join(" ---: |" for _ in cols))
    for b in order:
        cells = [b]
        for c in cols:
            cells.append(str(grid[c]["confidence_buckets"].get(b, 0)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    grid = load_cutoff_grid()

    # Plots — one PNG per metric, lines = slices
    plot_metric(rows, "greek_word_space_fertility", "Greek word fertility", "c3_cutoff_fertility.png",
                "C3 cutoff sweep — Greek word fertility", lower_better=True)
    plot_metric(rows, "chars_per_token", "chars / token", "c3_cutoff_chars_per_token.png",
                "C3 cutoff sweep — chars per token", lower_better=False)
    plot_metric(rows, "single_token_greek_word_share", "single-token Greek-word share",
                "c3_cutoff_single_word_share.png",
                "C3 cutoff sweep — single-token Greek-word share", lower_better=False)
    plot_metric(rows, "tokens_per_byte", "tokens / byte", "c3_cutoff_tokens_per_byte.png",
                "C3 cutoff sweep — tokens per byte", lower_better=True)
    plot_single_metric_one_line(rows, "eval_added_vocab_utilization_rate",
                                "added-vocab utilization rate",
                                "c3_cutoff_added_vocab_utilization.png",
                                "C3 cutoff sweep — fraction of added tokens that the eval slices use",
                                SLICES_ORDER)
    plot_single_metric_one_line(rows, "eval_unused_added_tokens",
                                "unused added tokens (count)",
                                "c3_cutoff_unused_added_tokens.png",
                                "C3 cutoff sweep — unused added tokens on eval",
                                SLICES_ORDER)

    # Markdown
    fert_tab = md_table_per_slice(rows, "greek_word_space_fertility", "fert")
    cpt_tab = md_table_per_slice(rows, "chars_per_token", "chars/tok")
    sws_tab = md_table_per_slice(rows, "single_token_greek_word_share", "single")
    tpb_tab = md_table_per_slice(rows, "tokens_per_byte", "tok/byte", fmt="{:.5f}")
    cat_tab = md_table_category_x_cutoff(grid)
    struct_tab = md_table_structure_x_cutoff(grid)
    lex_tab = md_table_lex_x_cutoff(grid)
    conf_tab = md_table_conf_x_cutoff(grid)

    body = f"""# C3 cutoff report

Date: 2026-05-11.

Tokenizer arm: **C3** (`C3_wave2_broad_glossapi_plus_hplt_50_50`, total
vocab 156,672). See [C3_CONVERGENCE.md](C3_CONVERGENCE.md).

This report sweeps the C3 cutoff at every multiple of 1024 from 1024
to 25600 (25 points, all 128-aligned). At each cutoff we built an
Apertus-compatible merged variant (just `model.vocab` + `model.merges`
truncated to `131072 + N`) and evaluated:

- intrinsic + fertility metrics on three **clean held-out** slices
- categorical composition of the kept added units, from the corrected
  25,600-token C3 glossary

## Held-out slices

All three are verified non-overlapping with C3 training (see
[C3_CONVERGENCE.md](C3_CONVERGENCE.md) § Held-out integrity).

| slice | docs | how it's clean |
| --- | ---: | --- |
| `virgin_hplt` | 10,000 | sampled from HPLT clean60 docs whose `source_doc_id` is **not in** the C3 training mix; guaranteed unseen by C3 BPE |
| `C3_val_clean` | 7,624 | C3 val with the 30 train-overlap text-md5 rows removed |
| `C3_test_clean` | 7,246 | C3 test with the 36 train-overlap text-md5 rows removed |

## Fertility & intrinsic metrics — per cutoff, per slice

The full numerical tables follow; PNG plots are linked first for the
shape.

### Plots

![Greek word fertility](figures/c3_cutoff_fertility.png)

![Chars per token](figures/c3_cutoff_chars_per_token.png)

![Single-token Greek-word share](figures/c3_cutoff_single_word_share.png)

![Tokens per byte](figures/c3_cutoff_tokens_per_byte.png)

![Added-vocab utilization rate](figures/c3_cutoff_added_vocab_utilization.png)

![Unused added tokens](figures/c3_cutoff_unused_added_tokens.png)

### Greek word fertility (lower is better)

{fert_tab}

### Chars per token (higher is better)

{cpt_tab}

### Single-token Greek-word share (higher is better)

{sws_tab}

### Tokens per byte (lower is better)

{tpb_tab}

## Composition of added units at each cutoff

Derived from the **corrected 25,600-token C3 glossary** (single Gemini
pass with morphological decomposition; see
`~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/`).
Each cutoff column counts how many of the first N added units land in
each category / structure / lexical / confidence bucket. By
construction the rows in column N are a strict prefix of the rows in
column N+1024.

### Category × cutoff

{cat_tab}

### Greek morphological structure × cutoff

Note: structure is defined only on Greek tokens (greek_word /
greek_fragment / greek_morpheme / proper_noun / greek_acronym). Non-Greek
categories contribute zero to this view.

{struct_tab}

### Greek lexical role × cutoff

Lexical is defined only for Greek tokens that fit one of the four
named roles; `none` is the residual greek-but-not-tagged set.

{lex_tab}

### Glossary confidence × cutoff

Confidence is the Gemini-pass per-token confidence from the corrected
glossary.

{conf_tab}

## Reading guide

- **Diminishing fertility returns**: the fertility curve typically has
  its sharpest drop between cutoff 0 (apertus_base) and the first 1–3k
  added units; thereafter the marginal fertility gain per added unit
  shrinks. Use the plot to find the elbow on `virgin_hplt` (the most
  trustworthy clean slice).
- **Added-vocab utilization** falls monotonically with cutoff because
  the long tail of merges is rarer in any held-out sample. Below a
  certain cutoff almost every added unit is exercised; above it, you're
  paying embedding rows for tokens that don't appear.
- **Greek-word vs greek-fragment growth**: at small cutoffs the
  composition is fragment-heavy; whole-word inflected Greek forms
  appear later in the merge order. The category × cutoff table shows
  where each category accelerates.
- **Shipping size must remain 128-divisible**. All cutoffs in this
  sweep already satisfy that.
"""
    REPORT_PATH.write_text(body)
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
