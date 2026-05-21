"""Aggregate per-arm bakeoff JSONs → one markdown table for visual review.

Each "result dir" should contain some subset of:
  - tokenizer_fair_metrics.json     (output of compute_tokenizer_fair_metrics.py)
  - new_token_diagnostics.json      (output of compute_new_token_diagnostics.py)
  - results.json                    (lm-eval-harness aggregated metrics)
  - bootstrap_cis.json              (output of compute_bootstrap_cis.py)
  - run_metadata.json               (any of the wrappers writes this)

We deliberately do NOT compute a weighted selection score — see EVAL_RECIPE.md
§"§5.6 hard gates". This script's job is to present the signals; the
selection call is manual against the V4-derived thresholds.

Usage:
    python3 summarize_bakeoff.py [--out report.md] <result-dir> [<result-dir> ...]

Example:
    python3 summarize_bakeoff.py \\
        /capstor/.../runs/bakeoff/vanilla_ckpt-500M \\
        /capstor/.../runs/bakeoff/retok_ckpt-500M \\
        /capstor/.../runs/bakeoff/centroid_ckpt-500M \\
        --out /capstor/.../runs/bakeoff/summary_500M.md
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


# Columns to extract — (header, dotted-path, fmt)
COLUMNS = [
    ("dir",                 None,                                        "{}"),
    ("vocab",               "tokenizer_fair_metrics.tokenizer_vocab_size", "{:,}"),
    ("BPC (b/byte)",        "tokenizer_fair_metrics.global.bpc_bits_per_byte", "{:.4f}"),
    ("NLL/char",            "tokenizer_fair_metrics.global.nll_per_char",      "{:.4f}"),
    ("NLL/word",            "tokenizer_fair_metrics.global.nll_per_word",      "{:.4f}"),
    ("tok/word",            "tokenizer_fair_metrics.global.tokens_per_word",   "{:.3f}"),
    ("chars/tok",           "tokenizer_fair_metrics.global.chars_per_token",   "{:.3f}"),
    ("STRR",                "tokenizer_fair_metrics.strr.rate",                "{:.3f}"),
    ("D1.top1",             "new_token_diagnostics.forward.d1_rank_of_new_target.top1_rate", "{:.3f}"),
    ("D2.mass_new",         "new_token_diagnostics.forward.d2_avg_prob_mass_new_per_pos",    "{:.4f}"),
    ("D4.top1_new",         "new_token_diagnostics.forward.d4_top1_at_new_target.new_rate",  "{:.3f}"),
    ("D5.util",             "new_token_diagnostics.greedy.utilization_rate",  "{:.3f}"),
    ("D6.E_new/exist",      "new_token_diagnostics.embedding.E_norm.new_to_existing_mean_ratio", "{:.3f}"),
    ("D7.cos_off",          "new_token_diagnostics.embedding.new_E_cos.mean_off_diag",  "{:.4f}"),
    ("D7.eff_rank",         "new_token_diagnostics.embedding.new_E_effective_rank.participation_ratio", "{:.1f}"),
    ("arc_chal",            "results.results.arc_challenge.acc_norm,none",    "{:.3f}"),
    ("hellaswag",           "results.results.hellaswag.acc_norm,none",        "{:.3f}"),
    ("winogrande",          "results.results.winogrande.acc,none",            "{:.3f}"),
    ("piqa",                "results.results.piqa.acc_norm,none",             "{:.3f}"),
    ("mmlu",                "results.results.mmlu.acc,none",                  "{:.3f}"),
]


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _get(d: dict, path: str):
    """Dotted-path lookup with comma fallback for keys like 'acc,none'."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        # Some lm-eval keys contain commas (e.g. 'acc,none') — match exact first
        if part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _gather(result_dir: Path) -> dict:
    """Read whatever JSONs are present in a result dir; return a flat dict
    keyed as 'filestem.dotted.path' so COLUMNS can lookup with one syntax."""
    return {
        "tokenizer_fair_metrics": _load_json(result_dir / "tokenizer_fair_metrics.json"),
        "new_token_diagnostics":  _load_json(result_dir / "new_token_diagnostics.json"),
        "results":                _load_json(result_dir / "results.json"),
        "bootstrap_cis":          _load_json(result_dir / "bootstrap_cis.json"),
        "run_metadata":           _load_json(result_dir / "run_metadata.json"),
    }


def _format_cell(value, fmt: str) -> str:
    if value is None:
        return "—"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _row(result_dir: Path) -> list[str]:
    data = _gather(result_dir)
    out = [result_dir.name]
    for (_header, path, fmt) in COLUMNS[1:]:
        if path is None:
            out.append("")
            continue
        bucket, _, dotted = path.partition(".")
        val = _get(data.get(bucket) or {}, dotted) if bucket in data else None
        out.append(_format_cell(val, fmt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="one or more result dirs (each containing some subset of the known JSONs)")
    ap.add_argument("--out", type=Path, default=None,
                    help="if given, write markdown to file; otherwise print to stdout")
    args = ap.parse_args()

    rows = [_row(d) for d in args.dirs]
    headers = [c[0] for c in COLUMNS]

    # Markdown table
    lines = []
    lines.append("# Bakeoff summary\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Generated by `summarize_bakeoff.py`. See `EVAL_RECIPE.md` §\"§5.6 hard gates\" for the manual-review rubric.")

    body = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
