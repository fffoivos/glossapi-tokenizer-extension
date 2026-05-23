#!/usr/bin/env python3
"""Summarize packed Token Distillation pilot intrinsic evals.

Input layout is produced by run_td_pilot_intrinsics_packed.sbatch:

  OUTPUT_ROOT/
    retok/tokenizer_fair_metrics.json
    retok/new_token_diagnostics.json
    td_last/...
    td_layer11/...

The summary intentionally focuses on the gates we need for the TD layer choice:
heldout Greek BPC/NLL, tokenizer compression/STRR, and D1/D2/D4/D5 new-token
behavior. It does not try to replace the full downstream benchmark suite.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _get(obj: Any, *keys: str) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _read_arm(arm_dir: Path) -> Dict[str, Any]:
    metrics = _load_json(arm_dir / "tokenizer_fair_metrics.json")
    diagnostics = _load_json(arm_dir / "new_token_diagnostics.json")
    run_metadata = _load_json(arm_dir / "run_metadata.json") or {}
    errors = []
    if metrics is None:
        errors.append("missing tokenizer_fair_metrics.json")
    if diagnostics is None:
        errors.append("missing new_token_diagnostics.json")

    g = metrics.get("global", {}) if metrics else {}
    d1 = _get(diagnostics, "forward", "d1_rank_of_new_target") if diagnostics else None
    d4 = _get(diagnostics, "forward", "d4_top1_at_new_target") if diagnostics else None
    greedy = diagnostics.get("greedy") if diagnostics else None
    embedding = diagnostics.get("embedding") if diagnostics else None

    return {
        "arm": arm_dir.name,
        "arm_dir": str(arm_dir),
        "model_path": run_metadata.get("model_path") or (metrics or {}).get("model_path") or (diagnostics or {}).get("model_path"),
        "errors": errors,
        "tokenizer": {
            "bpc_bits_per_byte": g.get("bpc_bits_per_byte"),
            "nll_per_char": g.get("nll_per_char"),
            "nll_per_word": g.get("nll_per_word"),
            "nll_per_token": g.get("nll_per_token"),
            "tokens_per_word": g.get("tokens_per_word"),
            "chars_per_token": g.get("chars_per_token"),
            "strr": _get(metrics, "strr", "rate") if metrics else None,
            "n_docs": g.get("n_docs"),
            "n_tokens": g.get("n_tokens"),
            "wall_seconds": metrics.get("wall_seconds") if metrics else None,
        },
        "new_token": {
            "n_positions_total": _get(diagnostics, "forward", "n_positions_total") if diagnostics else None,
            "n_positions_new_target": _get(diagnostics, "forward", "n_positions_new_target") if diagnostics else None,
            "d1_mean_rank": (d1 or {}).get("mean_rank"),
            "d1_top1_rate": (d1 or {}).get("top1_rate"),
            "d1_top5_rate": (d1 or {}).get("top5_rate"),
            "d1_top10_rate": (d1 or {}).get("top10_rate"),
            "d1_top50_rate": (d1 or {}).get("top50_rate"),
            "d2_avg_prob_mass_new_per_pos": _get(diagnostics, "forward", "d2_avg_prob_mass_new_per_pos") if diagnostics else None,
            "d4_new_rate": (d4 or {}).get("new_rate"),
            "d5_greedy_utilization_rate": (greedy or {}).get("utilization_rate") if greedy else None,
            "d5_n_new_gen": (greedy or {}).get("n_new_gen") if greedy else None,
            "d5_n_total_gen": (greedy or {}).get("n_total_gen") if greedy else None,
            "E_new_to_existing_norm_ratio": _get(embedding, "E_norm", "new_to_existing_mean_ratio") if embedding else None,
            "U_new_to_existing_norm_ratio": _get(embedding, "U_norm", "new_to_existing_mean_ratio") if embedding else None,
            "new_E_cos_p95": _get(embedding, "new_E_cos", "p95_off_diag") if embedding else None,
            "new_E_participation_ratio": _get(embedding, "new_E_effective_rank", "participation_ratio") if embedding else None,
            "wall_seconds": diagnostics.get("wall_seconds") if diagnostics else None,
        },
    }


def _delta(value: Optional[float], base: Optional[float]) -> Optional[float]:
    if value is None or base is None:
        return None
    return value - base


def _write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    arms = summary["arms"]
    base = next((a for a in arms if a["arm"] == "retok"), None)

    lines: List[str] = []
    lines.append("# TD Pilot Intrinsic Eval Summary")
    lines.append("")
    lines.append(f"- Output root: `{summary['output_root']}`")
    lines.append(f"- Eval JSONL: `{summary.get('eval_jsonl') or 'unknown'}`")
    lines.append(f"- Best BPC arm: `{summary.get('best_bpc_arm') or 'n/a'}`")
    lines.append("")
    lines.append("## Heldout Greek Metrics")
    lines.append("")
    lines.append("| arm | BPC | delta vs ReTok | NLL/char | tokens/word | STRR | docs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        tok = arm["tokenizer"]
        delta_bpc = _delta(tok.get("bpc_bits_per_byte"), _get(base, "tokenizer", "bpc_bits_per_byte") if base else None)
        lines.append(
            "| {arm} | {bpc} | {dbpc} | {nllc} | {tpw} | {strr} | {docs} |".format(
                arm=arm["arm"],
                bpc=_fmt(tok.get("bpc_bits_per_byte")),
                dbpc=_fmt(delta_bpc),
                nllc=_fmt(tok.get("nll_per_char")),
                tpw=_fmt(tok.get("tokens_per_word")),
                strr=_fmt(tok.get("strr")),
                docs=_fmt(tok.get("n_docs"), 0),
            )
        )
    lines.append("")
    lines.append("## New-Token Diagnostics")
    lines.append("")
    lines.append("| arm | new targets | D1 mean rank | D1 top1 | D1 top5 | D2 new mass | D4 top1-new | D5 greedy-new | E norm ratio | U norm ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        nt = arm["new_token"]
        lines.append(
            "| {arm} | {npos} | {rank} | {top1} | {top5} | {mass} | {d4} | {d5} | {er} | {ur} |".format(
                arm=arm["arm"],
                npos=_fmt(nt.get("n_positions_new_target"), 0),
                rank=_fmt(nt.get("d1_mean_rank"), 2),
                top1=_fmt(nt.get("d1_top1_rate")),
                top5=_fmt(nt.get("d1_top5_rate")),
                mass=_fmt(nt.get("d2_avg_prob_mass_new_per_pos")),
                d4=_fmt(nt.get("d4_new_rate")),
                d5=_fmt(nt.get("d5_greedy_utilization_rate")),
                er=_fmt(nt.get("E_new_to_existing_norm_ratio")),
                ur=_fmt(nt.get("U_new_to_existing_norm_ratio")),
            )
        )
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- Lower BPC/NLL is better.")
    lines.append("- For D1, lower mean rank and higher top-k rates are better.")
    lines.append("- D2/D4/D5 should move toward healthy use of new IDs without exploding relative to ReTok.")
    lines.append("- If layer-11 improves BPC but shows unstable D-rank or output-norm behavior, prefer last-layer TD for the full run.")
    lines.append("")
    for arm in arms:
        if arm["errors"]:
            lines.append(f"- `{arm['arm']}` errors: {', '.join(arm['errors'])}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--eval-jsonl", type=str, default=None)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--output-md", type=Path, default=None)
    args = ap.parse_args()

    arms = []
    for child in sorted(args.output_root.iterdir()):
        if child.is_dir() and (child / "run_metadata.json").exists():
            arms.append(_read_arm(child))

    best_bpc = None
    for arm in arms:
        bpc = arm["tokenizer"].get("bpc_bits_per_byte")
        if bpc is None:
            continue
        if best_bpc is None or bpc < best_bpc[1]:
            best_bpc = (arm["arm"], bpc)

    summary = {
        "output_root": str(args.output_root),
        "eval_jsonl": args.eval_jsonl,
        "best_bpc_arm": best_bpc[0] if best_bpc else None,
        "arms": arms,
    }
    out_json = args.output_json or (args.output_root / "td_pilot_intrinsics_summary.json")
    out_md = args.output_md or (args.output_root / "TD_PILOT_INTRINSICS_SUMMARY.md")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_markdown(out_md, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote: {out_json}")
    print(f"wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
