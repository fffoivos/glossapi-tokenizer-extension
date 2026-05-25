"""Summarize the Vanilla/ReTok/TD 3.5B continuation results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "per_iter_results"
OUT_MD = ROOT / "CONTINUATION_3P5B_RESULTS_20260525.md"
OUT_JSON = ROOT / "continuation_3p5b_summary.json"
TOK_PER_ITER = 1024 * 4096
ITERS = [476, 585, 715, 834]
ARMS = ["vanilla", "retok", "td"]
ARM_LABEL = {
    "vanilla": "Vanilla",
    "retok": "ReTok",
    "td": "TD layer11",
}


TASKS = [
    ("mmlu", False, "EN retention", "MMLU"),
    ("hellaswag", True, "EN retention", "HellaSwag"),
    ("arc_easy", True, "EN retention", "ARC Easy"),
    ("arc_challenge", True, "EN retention", "ARC Challenge"),
    ("piqa", True, "EN retention", "PIQA"),
    ("winogrande", False, "EN retention", "Winogrande"),
    ("global_mmlu", False, "Multilingual", "Global MMLU"),
    ("xcopa", False, "Multilingual", "XCOPA"),
    ("xnli", False, "Multilingual", "XNLI"),
    ("global_mmlu_full_el", False, "Greek", "Greek MMLU"),
    ("include_base_44_greek_few_shot_en", False, "Greek", "INCLUDE-44 Greek"),
    ("belebele_ell_Grek", False, "Greek", "Belebele Greek"),
    ("arc_challenge_mt_el", True, "Greek", "ARC Challenge MT-el"),
    ("xnli_el", False, "Greek", "XNLI Greek"),
    ("xquad_el", False, "Greek", "XQuAD Greek F1"),
    ("global_piqa_completions_ell_grek", True, "Greek", "PIQA Greek"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def metric(results: dict, task: str, prefer_norm: bool) -> float | None:
    value = results.get(task)
    if value is None:
        return None
    if task == "xquad_el":
        return value.get("f1,none")
    if prefer_norm and "acc_norm,none" in value:
        return value.get("acc_norm,none")
    return value.get("acc,none")


def result_blob(arm: str, iteration: int) -> dict:
    return read_json(RESULTS / f"{arm}_iter{iteration}.json")["results"]


def fair_blob(arm: str, iteration: int) -> dict | None:
    path = RESULTS / "intrinsic" / f"{arm}_iter{iteration:03d}_fair.json"
    if not path.exists():
        return None
    return read_json(path)


def diag_blob(arm: str, iteration: int) -> dict | None:
    path = RESULTS / "diagnostics" / f"{arm}_iter{iteration:03d}_new_token_diagnostics.json"
    if not path.exists():
        return None
    return read_json(path)


def group_average(arm: str, iteration: int, group: str) -> float:
    blob = result_blob(arm, iteration)
    vals = [
        metric(blob, task, prefer_norm)
        for task, prefer_norm, task_group, _ in TASKS
        if task_group == group
    ]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_pp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.2f} pp"


def fmt_signed(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def best_arm(values: dict[str, float]) -> str:
    return max(values, key=values.get)


def make_summary() -> dict:
    groups = ["Greek", "EN retention", "Multilingual"]
    group_rows = []
    for arm in ARMS:
        row = {"arm": arm}
        for group in groups:
            final = group_average(arm, 834, group)
            start = group_average(arm, 476, group)
            row[group] = {
                "iter476": start,
                "iter834": final,
                "delta": final - start,
            }
        fair = fair_blob(arm, 834)
        fair_start = fair_blob(arm, 476)
        row["bpc"] = {
            "iter476": fair_start["global"]["bpc_bits_per_byte"] if fair_start else None,
            "iter834": fair["global"]["bpc_bits_per_byte"] if fair else None,
            "delta": (
                fair["global"]["bpc_bits_per_byte"]
                - fair_start["global"]["bpc_bits_per_byte"]
                if fair and fair_start
                else None
            ),
        }
        group_rows.append(row)

    task_rows = []
    for task, prefer_norm, group, label in TASKS:
        vals = {
            arm: metric(result_blob(arm, 834), task, prefer_norm)
            for arm in ARMS
        }
        starts = {
            arm: metric(result_blob(arm, 476), task, prefer_norm)
            for arm in ARMS
        }
        task_rows.append(
            {
                "task": task,
                "label": label,
                "group": group,
                "values": vals,
                "deltas": {arm: vals[arm] - starts[arm] for arm in ARMS},
                "winner": best_arm(vals),
            }
        )

    diag_rows = []
    for arm in ["retok", "td"]:
        diag = diag_blob(arm, 834)
        start = diag_blob(arm, 585) or diag_blob(arm, 476)
        if not diag:
            continue
        rank = diag["forward"]["d1_rank_of_new_target"]
        top1 = rank["top1_rate"]
        top5 = rank["top5_rate"]
        mean_rank = rank["mean_rank"]
        start_rank = start["forward"]["d1_rank_of_new_target"] if start else None
        diag_rows.append(
            {
                "arm": arm,
                "top1": top1,
                "top5": top5,
                "mean_rank": mean_rank,
                "top1_delta_from_585": (
                    top1 - start_rank["top1_rate"] if start_rank else None
                ),
                "top5_delta_from_585": (
                    top5 - start_rank["top5_rate"] if start_rank else None
                ),
                "mean_rank_delta_from_585": (
                    mean_rank - start_rank["mean_rank"] if start_rank else None
                ),
            }
        )

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "iterations": ITERS,
        "tokens_b": {str(i): i * TOK_PER_ITER / 1e9 for i in ITERS},
        "group_rows": group_rows,
        "task_rows": task_rows,
        "diag_rows": diag_rows,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# 3.5B continuation results - Vanilla vs ReTok vs TD layer11",
        "",
        f"Generated UTC: `{summary['generated_utc']}`.",
        "",
        "This summarizes the continuation run `continuation_3p5b_20260524T143012Z`,",
        "which extended Vanilla, ReTok, and TD layer11 from iter 476 (~2.0B tokens)",
        "to iter 834 (~3.5B tokens). Local JSON snapshots live under",
        "`per_iter_results/`; remote full artifacts remain on Clariden under",
        "`/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_*`.",
        "",
        "Loss-reading rule: raw Megatron `lm loss` is per-token CE and is not",
        "tokenizer-fair across Vanilla vs the 148,480-vocab arms. This report therefore",
        "uses heldout BPC/BPB and downstream evals for cross-arm conclusions; raw",
        "training loss plots are diagnostic-only.",
        "",
        "## Bottom line",
        "",
        "- TD layer11 is the best final benchmark arm overall: it is first on English",
        "  retention and multilingual aggregates, and narrowly first on the Greek",
        "  aggregate at iter 834.",
        "- Vanilla still has the best tokenizer-fair heldout Greek BPC, but its",
        "  downstream Greek aggregate declined during the 2.0B -> 3.5B continuation.",
        "- ReTok improves fastest on BPC and wins Greek MMLU / INCLUDE-44 Greek at",
        "  iter 834, but it remains behind TD and Vanilla on the Greek aggregate.",
        "- If selecting for the actual downstream bakeoff objective, TD layer11 is now",
        "  the leading candidate. If selecting only for heldout BPC, Vanilla remains",
        "  ahead.",
        "",
        "## Aggregate scoreboard at iter 834",
        "",
        "| Arm | Greek agg | Delta vs 476 | EN retention | Delta vs 476 | Multilingual | Delta vs 476 | BPC lower better | BPC delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["group_rows"]:
        lines.append(
            "| {arm} | {gr} | {gr_d} | {en} | {en_d} | {multi} | {multi_d} | {bpc} | {bpc_d} |".format(
                arm=ARM_LABEL[row["arm"]],
                gr=fmt(row["Greek"]["iter834"]),
                gr_d=fmt_pp(row["Greek"]["delta"]),
                en=fmt(row["EN retention"]["iter834"]),
                en_d=fmt_pp(row["EN retention"]["delta"]),
                multi=fmt(row["Multilingual"]["iter834"]),
                multi_d=fmt_pp(row["Multilingual"]["delta"]),
                bpc=fmt(row["bpc"]["iter834"]),
                bpc_d=fmt_signed(row["bpc"]["delta"]),
            )
        )

    lines += [
        "",
        "## Per-task winners at iter 834",
        "",
        "| Group | Task | Vanilla | ReTok | TD layer11 | Winner |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary["task_rows"]:
        vals = row["values"]
        lines.append(
            "| {group} | `{task}` | {van} | {retok} | {td} | {winner} |".format(
                group=row["group"],
                task=row["task"],
                van=fmt(vals["vanilla"]),
                retok=fmt(vals["retok"]),
                td=fmt(vals["td"]),
                winner=ARM_LABEL[row["winner"]],
            )
        )

    lines += [
        "",
        "## Change from iter 476 to iter 834",
        "",
        "| Group | Task | Vanilla delta | ReTok delta | TD layer11 delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["task_rows"]:
        deltas = row["deltas"]
        lines.append(
            "| {group} | `{task}` | {van} | {retok} | {td} |".format(
                group=row["group"],
                task=row["task"],
                van=fmt_pp(deltas["vanilla"]),
                retok=fmt_pp(deltas["retok"]),
                td=fmt_pp(deltas["td"]),
            )
        )

    lines += [
        "",
        "## New-token diagnostics",
        "",
        "Measured on the 500-document heldout slice. Top-k is the fraction of",
        "positions whose correct new token appears in the model's top-k predictions;",
        "lower mean rank is better.",
        "",
        "| Arm | Top-1 at new target | Delta from 585 | Top-5 at new target | Delta from 585 | Mean rank | Delta from 585 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["diag_rows"]:
        lines.append(
            "| {arm} | {top1} | {top1_d} | {top5} | {top5_d} | {rank} | {rank_d} |".format(
                arm=ARM_LABEL[row["arm"]],
                top1=fmt(row["top1"]),
                top1_d=fmt_pp(row["top1_delta_from_585"]),
                top5=fmt(row["top5"]),
                top5_d=fmt_pp(row["top5_delta_from_585"]),
                rank=fmt(row["mean_rank"], 1),
                rank_d=fmt(row["mean_rank_delta_from_585"], 1),
            )
        )

    lines += [
        "",
        "## Artifact checklist",
        "",
        "- Local packed-eval snapshots: `per_iter_results/{vanilla,retok,td}_iter{585,715,834}.json`.",
        "- Local BPC snapshots: `per_iter_results/intrinsic/*_iter{585,715,834}_fair.json`.",
        "- Local new-token diagnostics: `per_iter_results/diagnostics/{retok,td}_iter{585,715,834}_new_token_diagnostics.json`.",
        "- Regenerated plots are written to `plots/`.",
        "- Final remote packed eval job: `2376082`, state `COMPLETED`, exit `0:0`, elapsed `00:59:51`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    summary = make_summary()
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_markdown(summary))
    print(OUT_MD)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
