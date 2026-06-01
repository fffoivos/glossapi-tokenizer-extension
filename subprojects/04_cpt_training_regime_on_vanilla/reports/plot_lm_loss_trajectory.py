"""Plot training lm loss vs consumed tokens for the 04 Vanilla CPT 5 B run.

Data sources
------------
- Training log iteration lines (one file per chain segment):
    reports/train_logs_cache_5b/i300-2417446.log   (iter 1   -> 300,  0 -> 1.258 B)
    reports/train_logs_cache_5b/i477-2417447.log   (iter 301 -> 477,  1.262 -> 2.001 B)
    reports/train_logs_cache_5b/i596-2417448.log   (iter 478 -> 596,  ~2.005 -> 2.500 B)
    reports/train_logs_cache_5b/i715-2417449.log   (iter 597 -> 715,  ~2.504 -> 2.999 B)
    reports/train_logs_cache_5b/i834-2417450.log   (iter 716 -> 834,  ~3.003 -> 3.498 B)
    reports/train_logs_cache_5b/i953-2417451.log   (iter 835 -> 953,  ~3.502 -> 3.997 B)
    reports/train_logs_cache_5b/i1072-2417452.log  (iter 954 -> 1072, ~4.001 -> 4.496 B)
    reports/train_logs_cache_5b/i1192-2417453.log  (iter 1073 -> 1192, ~4.501 -> 5.000 B)

  These were extracted (read-only) from the corresponding Slurm .out files at
    /capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04van5b_<seg>.out
  via 'grep -E "iteration\\s+[0-9]+/" ...'.

Each iteration line carries 'consumed tokens: X.XXXB' and 'lm loss: Y.YYYE+00'.

Output: plot_lm_loss_trajectory.png next to this script.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS = Path(__file__).resolve().parent
LOG_DIR = REPORTS / "train_logs_cache_5b"
OUT = REPORTS / "plot_lm_loss_trajectory.png"

# Chain order (consumes iters 1..1192 in order).
SEGMENTS = [
    "i300-2417446.log",
    "i477-2417447.log",
    "i596-2417448.log",
    "i715-2417449.log",
    "i834-2417450.log",
    "i953-2417451.log",
    "i1072-2417452.log",
    "i1192-2417453.log",
]

CHECKPOINT_ITERS = [119, 238, 477, 834, 1192]
CHECKPOINT_LABELS = {119: "0.5 B", 238: "1 B", 477: "2 B", 834: "3.5 B", 1192: "5 B"}
WARMUP_END_ITER = 287
WARMUP_END_B = 1.204  # consumed tokens at iter 287

LINE_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*consumed tokens:\s*([0-9.]+)B.*lm loss:\s*([0-9.eE+\-]+)"
)


def parse_segment(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    with path.open() as f:
        for ln in f:
            m = LINE_RE.search(ln)
            if not m:
                continue
            it = int(m.group(1))
            tok = float(m.group(2))
            loss = float(m.group(3))
            rows.append((it, tok, loss))
    return rows


def ema(values: list[float], alpha: float) -> list[float]:
    out: list[float] = []
    s = None
    for v in values:
        s = v if s is None else (alpha * v + (1 - alpha) * s)
        out.append(s)
    return out


def main() -> None:
    iters: list[int] = []
    tokens: list[float] = []
    losses: list[float] = []
    seg_boundaries: list[tuple[str, int, float]] = []
    for seg in SEGMENTS:
        rows = parse_segment(LOG_DIR / seg)
        if rows:
            seg_boundaries.append((seg, rows[0][0], rows[0][1]))
        for it, tok, loss in rows:
            iters.append(it)
            tokens.append(tok)
            losses.append(loss)

    # EMA window ~ 20 iterations. alpha = 2 / (N+1).
    smoothed = ema(losses, alpha=2.0 / (20 + 1))

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Raw points (faint).
    ax.scatter(
        tokens,
        losses,
        s=8,
        color="C0",
        alpha=0.20,
        label="raw lm loss (per iter)",
    )
    # Smoothed line.
    ax.plot(
        tokens,
        smoothed,
        color="C0",
        lw=2.2,
        label="EMA-smoothed lm loss (window ~ 20 iters)",
    )

    # Warmup-end vertical line.
    ax.axvline(WARMUP_END_B, ls="--", color="orange", lw=1.6, alpha=0.85)
    ax.text(
        WARMUP_END_B + 0.04,
        max(losses) * 0.93,
        f"warmup end\n(iter {WARMUP_END_ITER}, {WARMUP_END_B:.2f} B)",
        fontsize=9,
        color="darkorange",
    )

    # Checkpoint markers (vertical lines + annotations).
    iter_to_tok = {it: tok for it, tok in zip(iters, tokens)}
    iter_to_loss = {it: loss for it, loss in zip(iters, losses)}
    for cit in CHECKPOINT_ITERS:
        ck_tok = iter_to_tok.get(cit)
        ck_loss = iter_to_loss.get(cit)
        if ck_tok is None:
            continue
        ax.axvline(ck_tok, ls=":", color="black", lw=1.0, alpha=0.45)
        ax.plot([ck_tok], [ck_loss], marker="o", ms=10,
                color="C3", zorder=5, markeredgecolor="white",
                markeredgewidth=1.5)
        ax.annotate(
            f"iter {cit}\n({CHECKPOINT_LABELS[cit]}, {ck_tok:.3f} B)\nloss {ck_loss:.3f}",
            (ck_tok, ck_loss),
            textcoords="offset points",
            xytext=(8, 22),
            fontsize=8.5,
            color="C3",
            ha="left",
        )

    # Chain-segment boundary tick marks (lightweight, top axis).
    for seg, start_it, start_tok in seg_boundaries:
        ax.axvline(start_tok, ls="-", color="gray", lw=0.4, alpha=0.25)

    ax.set_xlabel("consumed tokens (B)")
    ax.set_ylabel("training lm loss")
    ax.set_title(
        "04 Vanilla CPT 5 B run — training lm loss vs consumed tokens\n"
        "(Goldfish loss k=h=50, LR 1.1e-5, AdEMAMix beta3=0.99, "
        "8 chain segments)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 5.05)
    ax.set_ylim(min(losses) - 0.05, max(losses) + 0.05)
    ax.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(
        f"saved: {OUT}  (n_iter_lines={len(losses)}  "
        f"final_loss={losses[-1]:.4f}  final_tokens={tokens[-1]:.3f}B)"
    )


if __name__ == "__main__":
    main()
