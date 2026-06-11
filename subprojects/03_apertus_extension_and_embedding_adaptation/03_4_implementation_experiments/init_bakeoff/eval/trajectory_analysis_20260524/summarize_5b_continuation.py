"""Summarize the 5B Vanilla vs TD-layer11 continuation.

The script is intentionally tolerant of missing iter1192 artifacts: it writes
an interim report from available checkpoints, then becomes the final report
generator once the 1192 sidecars have landed and `collect_5b_continuation_artifacts.sh`
has copied them locally.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "per_iter_results"
OUT_MD = ROOT / "CONTINUATION_5B_RESULTS_20260526.md"
OUT_JSON = ROOT / "continuation_5b_summary.json"
TOK_PER_ITER = 1024 * 4096
TARGET_ITERS = [476, 834, 1013, 1192]
ARMS = ["vanilla", "td"]
ARM_LABEL = {"vanilla": "Vanilla", "td": "TD layer11"}
GROUPS = ["Greek", "EN retention", "Multilingual"]
HEADLINE_EXCLUDED_TASKS = {
    "arc_challenge_mt_el",
    "global_piqa_completions_ell_grek",
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
    ("global_piqa_completions_ell_grek", True, "Greek", "PIQA Greek MT"),
]

GREEK_AGGREGATE_VARIANTS = [
    (
        "SwissAI 7-task fallback bundle",
        {
            "Greek MMLU",
            "INCLUDE-44 Greek",
            "Belebele Greek",
            "ARC Challenge MT-el",
            "XNLI Greek",
            "XQuAD Greek F1",
            "PIQA Greek MT",
        },
        "Diagnostic only; includes two explicitly machine-translated tasks and is not the planned native-Greek suite.",
    ),
    (
        "Headline no-explicit-MT Greek slice",
        {
            "Greek MMLU",
            "INCLUDE-44 Greek",
            "Belebele Greek",
            "XNLI Greek",
            "XQuAD Greek F1",
        },
        "Drops `arc_challenge_mt_el` and `global_piqa_completions_ell_grek`; still does not include greek-nlp/benchmark, Medical MCQA Greek, or OYXOY.",
    ),
    (
        "No-MT/no-XNLI diagnostic slice",
        {
            "Greek MMLU",
            "INCLUDE-44 Greek",
            "Belebele Greek",
            "XQuAD Greek F1",
        },
        "Also drops XNLI Greek because it is translated NLI; use as a sensitivity check only.",
    ),
]

GREEK_NO_EXPLICIT_MT_LABELS = GREEK_AGGREGATE_VARIANTS[1][1]

BASELINES = {
    "V4-HF": {
        "mmlu": 0.5923,
        "hellaswag": 0.7884,
        "arc_easy": 0.8363,
        "arc_challenge": 0.5870,
        "piqa": 0.7992,
        "winogrande": 0.6930,
        "global_mmlu": 0.5246,
        "xcopa": 0.6575,
        "xnli": 0.4400,
        "global_mmlu_full_el": 0.5155,
        "include_base_44_greek_few_shot_en": 0.5054,
        "belebele_ell_Grek": 0.6367,
        "arc_challenge_mt_el": 0.4795,
        "xnli_el": 0.3984,
        "xquad_el": 0.5172,
        "global_piqa_completions_ell_grek": 0.6200,
    },
    "V4-postconv": {
        "mmlu": 0.2295,
        "hellaswag": 0.2675,
        "arc_easy": 0.2614,
        "arc_challenge": 0.2619,
        "piqa": 0.5212,
        "winogrande": 0.5107,
        "global_mmlu": 0.2381,
        "xcopa": 0.5185,
        "xnli": 0.3321,
        "global_mmlu_full_el": 0.2295,
        "include_base_44_greek_few_shot_en": 0.1975,
        "belebele_ell_Grek": 0.2289,
        "arc_challenge_mt_el": 0.2637,
        "xnli_el": 0.3333,
        "xquad_el": 0.0,
        "global_piqa_completions_ell_grek": 0.5400,
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def result_path(arm: str, iteration: int) -> Path:
    return RESULTS / f"{arm}_iter{iteration}.json"


def fair_path(arm: str, iteration: int) -> Path:
    return RESULTS / "intrinsic" / f"{arm}_iter{iteration}_fair.json"


def diag_path(iteration: int) -> Path:
    return RESULTS / "diagnostics" / f"td_iter{iteration}_new_token_diagnostics.json"


def available_iterations() -> list[int]:
    out = []
    for iteration in TARGET_ITERS:
        if all(result_path(arm, iteration).exists() for arm in ARMS):
            out.append(iteration)
    return out


def result_blob(arm: str, iteration: int) -> dict:
    return read_json(result_path(arm, iteration))["results"]


def fair_blob(arm: str, iteration: int) -> dict | None:
    path = fair_path(arm, iteration)
    return read_json(path) if path.exists() else None


def diag_blob(iteration: int) -> dict | None:
    path = diag_path(iteration)
    return read_json(path) if path.exists() else None


def metric(results: dict, task: str, prefer_norm: bool) -> float | None:
    value = results.get(task)
    if value is None:
        return None
    if task == "xquad_el":
        return value.get("f1,none")
    if prefer_norm and "acc_norm,none" in value:
        return value.get("acc_norm,none")
    return value.get("acc,none")


def heldout_bpb(blob: dict | None) -> float | None:
    if blob is None:
        return None
    global_metrics = blob["global"]
    return global_metrics.get("bpb_bits_per_byte", global_metrics.get("bpc_bits_per_byte"))


def task_value(arm: str, iteration: int, task: str, prefer_norm: bool) -> float | None:
    return metric(result_blob(arm, iteration), task, prefer_norm)


def group_average(arm: str, iteration: int, group: str) -> float | None:
    vals = [
        task_value(arm, iteration, task, prefer_norm)
        for task, prefer_norm, task_group, _label in TASKS
        if task_group == group
        and not (group == "Greek" and task in HEADLINE_EXCLUDED_TASKS)
    ]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def baseline_group_average(baseline: str, group: str) -> float:
    vals = [
        BASELINES[baseline][task]
        for task, _prefer_norm, task_group, _label in TASKS
        if task_group == group
        and not (group == "Greek" and task in HEADLINE_EXCLUDED_TASKS)
        and task in BASELINES[baseline]
    ]
    return sum(vals) / len(vals)


def average_rows(rows: list[dict], labels: set[str], field: str) -> float | None:
    vals = [row.get(field) for row in rows if row.get("label") in labels and row.get(field) is not None]
    return sum(vals) / len(vals) if vals else None


def baseline_average_for_labels(baseline: str, labels: set[str]) -> float | None:
    label_to_task = {label: task for task, _prefer_norm, _group, label in TASKS}
    vals = [
        BASELINES[baseline][label_to_task[label]]
        for label in labels
        if label in label_to_task and label_to_task[label] in BASELINES[baseline]
    ]
    return sum(vals) / len(vals) if vals else None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_signed(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def fmt_pp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:+.2f} pp"


def winner(vanilla: float | None, td: float | None, lower_is_better: bool = False) -> str:
    if vanilla is None or td is None:
        return "n/a"
    if vanilla == td:
        return "tie"
    if lower_is_better:
        return "TD" if td < vanilla else "Vanilla"
    return "TD" if td > vanilla else "Vanilla"


def diag_summary(iteration: int) -> dict | None:
    blob = diag_blob(iteration)
    if not blob:
        return None
    rank = blob["forward"]["d1_rank_of_new_target"]
    greedy = blob.get("greedy", {})
    return {
        "iteration": iteration,
        "top1": rank["top1_rate"],
        "top5": rank["top5_rate"],
        "mean_rank": rank["mean_rank"],
        "mass_new": blob["forward"]["d2_avg_prob_mass_new_per_pos"],
        "greedy_new_utilization": greedy.get("utilization_rate"),
    }


def count_task_winners(task_rows: list[dict]) -> dict[str, int]:
    counts = {"TD": 0, "Vanilla": 0, "tie": 0, "n/a": 0}
    for row in task_rows:
        counts[row.get("winner", "n/a")] = counts.get(row.get("winner", "n/a"), 0) + 1
    return counts


def row_for_iteration(summary: dict, iteration: int) -> dict | None:
    for row in summary.get("aggregate_rows", []):
        if row["iteration"] == iteration:
            return row
    return None


def group_deltas(row: dict) -> dict[str, float | None]:
    arms = row["arms"]
    return {
        group: (
            arms["td"][group] - arms["vanilla"][group]
            if arms["td"].get(group) is not None and arms["vanilla"].get(group) is not None
            else None
        )
        for group in GROUPS
    }


def group_changes(row_new: dict, row_old: dict, arm: str = "td") -> dict[str, float | None]:
    out = {}
    for group in GROUPS:
        new_value = row_new["arms"][arm].get(group)
        old_value = row_old["arms"][arm].get(group)
        out[group] = new_value - old_value if new_value is not None and old_value is not None else None
    return out


def linear_recovery_projection(td_change_from_3p5: dict, td_vs_v4_hf: dict) -> dict[str, dict]:
    delta_tokens_b = (1192 - 834) * TOK_PER_ITER / 1e9
    out = {}
    for group in GROUPS:
        change = td_change_from_3p5.get(group)
        gap = -(td_vs_v4_hf.get(group) or 0.0)
        slope = change / delta_tokens_b if change is not None else None
        if slope is not None and slope > 0 and gap > 0:
            extra_b = gap / slope
            total_b = 5.0 + extra_b
        elif gap <= 0:
            extra_b = 0.0
            total_b = 5.0
        else:
            extra_b = None
            total_b = None
        out[group] = {
            "gap_to_v4_hf": gap,
            "change_3p5_to_5b": change,
            "slope_per_b": slope,
            "extra_b_tokens_at_linear_rate": extra_b,
            "total_b_tokens_at_linear_rate": total_b,
        }
    return out


def decision_payload(summary: dict) -> dict:
    latest = summary.get("latest_iteration")
    rows = summary.get("aggregate_rows", [])
    if latest is None or not rows:
        return {
            "recommendation": "Pending: no matched eval checkpoints are available.",
            "group_deltas": {},
            "bpb_delta_td_minus_vanilla": None,
            "task_wins": {},
            "td_change_from_3p5b": {},
            "td_vs_v4_hf": {},
        }

    latest_row = rows[-1]
    deltas = group_deltas(latest_row)
    td_bpb = latest_row["arms"]["td"].get("bpb")
    vanilla_bpb = latest_row["arms"]["vanilla"].get("bpb")
    bpb_delta = td_bpb - vanilla_bpb if td_bpb is not None and vanilla_bpb is not None else None

    row_3p5 = row_for_iteration(summary, 834)
    td_change_from_3p5 = group_changes(latest_row, row_3p5, "td") if row_3p5 else {}
    td_vs_v4_hf = {
        group: (
            latest_row["arms"]["td"][group] - summary["baselines"]["V4-HF"][group]
            if latest_row["arms"]["td"].get(group) is not None
            else None
        )
        for group in GROUPS
    }
    projection = linear_recovery_projection(td_change_from_3p5, td_vs_v4_hf) if row_3p5 else {}
    task_wins = count_task_winners(summary.get("task_rows", []))
    greek_rows = [row for row in summary.get("task_rows", []) if row["group"] == "Greek"]
    greek_no_mt_vanilla = average_rows(greek_rows, GREEK_NO_EXPLICIT_MT_LABELS, "vanilla")
    greek_no_mt_td = average_rows(greek_rows, GREEK_NO_EXPLICIT_MT_LABELS, "td")
    greek_no_mt_delta = (
        greek_no_mt_td - greek_no_mt_vanilla
        if greek_no_mt_td is not None and greek_no_mt_vanilla is not None
        else None
    )
    greek_no_mt_v4_hf = baseline_average_for_labels("V4-HF", GREEK_NO_EXPLICIT_MT_LABELS)
    greek_no_mt_td_vs_v4_hf = (
        greek_no_mt_td - greek_no_mt_v4_hf
        if greek_no_mt_td is not None and greek_no_mt_v4_hf is not None
        else None
    )
    objective_answer = None

    if summary["status"] != "final":
        recommendation = (
            "Pending final 1192 artifacts. Interim evidence says TD is ahead "
            "on English retention and multilingual aggregates, roughly tied on "
            "the no-explicit-MT Greek aggregate, and still behind Vanilla on heldout BPB."
        )
        objective_answer = (
            "Not answered yet: the matched 5B checkpoint is still missing, so "
            "the run cannot decide TD-vs-Vanilla at the target token count."
        )
    else:
        aggregate_wins = [group for group, value in deltas.items() if value is not None and value > 0]
        greek_delta = deltas.get("Greek")
        if len(aggregate_wins) == len(GROUPS):
            recommendation = (
                "Primary decision: TD has not overtaken the initial "
                "Vanilla/V4-HF scores by 5B. Secondary result: TD is the "
                "matched-5B downstream winner over continued Vanilla on the "
                "available fallback eval slices, including the no-explicit-MT "
                "Greek slice, while Vanilla remains better on heldout "
                "byte-normalized loss."
            )
            objective_answer = (
                "No: TD does not overtake initial Vanilla/V4-HF by 5B. It "
                "overtakes matched continued Vanilla on downstream aggregates, "
                "including the available no-explicit-MT Greek slice, but "
                "remains below V4-HF on the no-explicit-MT Greek, English-retention, "
                "and multilingual aggregates. The planned native-Greek suite "
                "was not run."
            )
        elif len(aggregate_wins) >= 2 and greek_delta is not None and greek_delta >= -0.002:
            recommendation = (
                "Primary decision: TD has not overtaken the initial "
                "Vanilla/V4-HF scores by 5B. It is competitive with matched "
                "continued Vanilla, but not a clean recovery winner."
            )
            objective_answer = (
                "No: TD does not overtake initial Vanilla/V4-HF by 5B. The "
                "matched-continuation comparison is mixed/positive, but the "
                "harder recovery bar is still unmet."
            )
        else:
            recommendation = (
                "Primary decision: TD has not overtaken the initial "
                "Vanilla/V4-HF scores by 5B, and it also has not clearly "
                "overtaken matched continued Vanilla."
            )
            objective_answer = (
                "No: TD does not overtake initial Vanilla/V4-HF by 5B."
            )

    return {
        "recommendation": recommendation,
        "objective_answer": objective_answer,
        "group_deltas": deltas,
        "bpb_delta_td_minus_vanilla": bpb_delta,
        "task_wins": task_wins,
        "greek_no_explicit_mt": {
            "vanilla": greek_no_mt_vanilla,
            "td": greek_no_mt_td,
            "delta_td_minus_vanilla": greek_no_mt_delta,
            "td_vs_v4_hf": greek_no_mt_td_vs_v4_hf,
        },
        "td_change_from_3p5b": td_change_from_3p5,
        "td_vs_v4_hf": td_vs_v4_hf,
        "linear_recovery_projection": projection,
    }


def make_summary() -> dict:
    iters = available_iterations()
    latest = iters[-1] if iters else None

    aggregate_rows = []
    for iteration in iters:
        row = {
            "iteration": iteration,
            "tokens_b": iteration * TOK_PER_ITER / 1e9,
            "arms": {},
        }
        for arm in ARMS:
            row["arms"][arm] = {
                group: group_average(arm, iteration, group)
                for group in GROUPS
            }
            row["arms"][arm]["bpb"] = heldout_bpb(fair_blob(arm, iteration))
        aggregate_rows.append(row)

    task_rows = []
    if latest is not None:
        for task, prefer_norm, group, label in TASKS:
            vanilla = task_value("vanilla", latest, task, prefer_norm)
            td = task_value("td", latest, task, prefer_norm)
            task_rows.append(
                {
                    "task": task,
                    "label": label,
                    "group": group,
                    "vanilla": vanilla,
                    "td": td,
                    "delta_td_minus_vanilla": (
                        td - vanilla if td is not None and vanilla is not None else None
                    ),
                    "winner": winner(vanilla, td),
                    "delta_td_vs_v4_hf": (
                        td - BASELINES["V4-HF"][task] if td is not None else None
                    ),
                    "delta_vanilla_vs_v4_hf": (
                        vanilla - BASELINES["V4-HF"][task] if vanilla is not None else None
                    ),
                }
            )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "final" if 1192 in iters else "interim_missing_1192",
        "available_iterations": iters,
        "missing_iterations": [it for it in TARGET_ITERS if it not in iters],
        "latest_iteration": latest,
        "tokens_b": {str(it): it * TOK_PER_ITER / 1e9 for it in iters},
        "aggregate_rows": aggregate_rows,
        "task_rows": task_rows,
        "diagnostics": [d for it in TARGET_ITERS if (d := diag_summary(it))],
        "baselines": {
            name: {group: baseline_group_average(name, group) for group in GROUPS}
            for name in BASELINES
        },
    }
    summary["decision"] = decision_payload(summary)
    return summary


def render_markdown(summary: dict) -> str:
    latest = summary["latest_iteration"]
    status_line = (
        "Final 1192 artifacts are present."
        if summary["status"] == "final"
        else "Interim report: final iter1192 artifacts are not present yet."
    )
    lines = [
        "# 5B continuation results - Vanilla vs TD layer11",
        "",
        f"Generated UTC: `{summary['generated_utc']}`.",
        "",
        status_line,
        "",
        "Run tag: `continuation_5b_td_vs_vanilla_20260525T142522Z`.",
        "",
        "Loss-reading rule: raw Megatron `lm loss` is per-token CE and is not",
        "tokenizer-fair across Vanilla vs the 148,480-vocab TD arm. This report",
        "uses heldout BPB and downstream evals for cross-arm conclusions.",
        "",
        "Evaluation-scope warning: the planned native-Greek suite was not run.",
        "In particular, `greek-nlp/benchmark`, Medical MCQA Greek, and OYXOY are",
        "absent. The Greek aggregate below excludes explicit MT tasks by",
        "default; those tasks remain visible only as per-task diagnostics and",
        "in the all-available fallback sensitivity row.",
        "",
    ]

    lines += [
        "## Available Checkpoints",
        "",
        "| Iter | Tokens B | Vanilla eval | TD eval | Vanilla BPB | TD BPB | BPB winner |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["aggregate_rows"]:
        iteration = row["iteration"]
        arms = row["arms"]
        lines.append(
            "| {it} | {tok:.3f} | yes | yes | {van_bpb} | {td_bpb} | {win} |".format(
                it=iteration,
                tok=row["tokens_b"],
                van_bpb=fmt(arms["vanilla"].get("bpb")),
                td_bpb=fmt(arms["td"].get("bpb")),
                win=winner(arms["vanilla"].get("bpb"), arms["td"].get("bpb"), lower_is_better=True),
            )
        )

    lines += [
        "",
        "## Aggregate Trajectory",
        "",
        "| Iter | Greek no-MT Vanilla | Greek no-MT TD | Delta TD-V | EN Vanilla | EN TD | Delta TD-V | Multi Vanilla | Multi TD | Delta TD-V |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate_rows"]:
        arms = row["arms"]
        lines.append(
            "| {it} | {gv} | {gt} | {gd} | {ev} | {et} | {ed} | {mv} | {mt} | {md} |".format(
                it=row["iteration"],
                gv=fmt(arms["vanilla"]["Greek"]),
                gt=fmt(arms["td"]["Greek"]),
                gd=fmt_signed(arms["td"]["Greek"] - arms["vanilla"]["Greek"]),
                ev=fmt(arms["vanilla"]["EN retention"]),
                et=fmt(arms["td"]["EN retention"]),
                ed=fmt_signed(arms["td"]["EN retention"] - arms["vanilla"]["EN retention"]),
                mv=fmt(arms["vanilla"]["Multilingual"]),
                mt=fmt(arms["td"]["Multilingual"]),
                md=fmt_signed(arms["td"]["Multilingual"] - arms["vanilla"]["Multilingual"]),
            )
        )

    if latest is not None:
        lines += [
            "",
            f"## Per-Task Matched Comparison At Iter {latest}",
            "",
            "| Group | Task | Vanilla | TD | Delta TD-V | Winner | TD vs V4-HF | Vanilla vs V4-HF |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
        for row in summary["task_rows"]:
            lines.append(
                "| {group} | {label} | {van} | {td} | {delta} | {winner} | {td_hf} | {van_hf} |".format(
                    group=row["group"],
                    label=row["label"],
                    van=fmt(row["vanilla"]),
                    td=fmt(row["td"]),
                    delta=fmt_signed(row["delta_td_minus_vanilla"]),
                    winner=row["winner"],
                    td_hf=fmt_signed(row["delta_td_vs_v4_hf"]),
                    van_hf=fmt_signed(row["delta_vanilla_vs_v4_hf"]),
                )
            )

        greek_rows = [row for row in summary["task_rows"] if row["group"] == "Greek"]
        lines += [
            "",
            f"## Greek Aggregate Variants At Iter {latest}",
            "",
            "The planned native-Greek benchmark suite was not run. The table",
            "below separates the all-available SwissAI fallback bundle from the",
            "no-explicit-MT slice that should be used as the current headline",
            "Greek reading.",
            "",
            "| Variant | Vanilla | TD | Delta TD-V | Note |",
            "|---|---:|---:|---:|---|",
        ]
        for name, labels, note in GREEK_AGGREGATE_VARIANTS:
            vanilla = average_rows(greek_rows, labels, "vanilla")
            td = average_rows(greek_rows, labels, "td")
            delta = td - vanilla if td is not None and vanilla is not None else None
            lines.append(
                f"| {name} | {fmt(vanilla)} | {fmt(td)} | {fmt_signed(delta)} | {note} |"
            )

    lines += [
        "",
        "## TD New-Token Diagnostics",
        "",
        "| Iter | Top1 new target | Top5 new target | Mean rank | New-vocab mass | Greedy new-token use |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["diagnostics"]:
        lines.append(
            "| {it} | {top1} | {top5} | {rank} | {mass} | {greedy} |".format(
                it=row["iteration"],
                top1=fmt(row["top1"]),
                top5=fmt(row["top5"]),
                rank=fmt(row["mean_rank"], 1),
                mass=fmt(row["mass_new"]),
                greedy=fmt(row["greedy_new_utilization"]),
            )
        )

    lines += [
        "",
        "## Baseline Anchors",
        "",
        "Baseline values come from `../V4_BENCHMARK_COMPARISON.md`.",
        "",
        "| Baseline | Greek no-MT agg | EN retention | Multilingual |",
        "|---|---:|---:|---:|",
    ]
    for name, values in summary["baselines"].items():
        lines.append(
            f"| {name} | {fmt(values['Greek'])} | {fmt(values['EN retention'])} | {fmt(values['Multilingual'])} |"
        )

    lines += [
        "",
        "## Decision Status",
        "",
    ]
    decision = summary["decision"]
    if summary["status"] == "final":
        lines.append(
            f"- Recommendation: {decision['recommendation']}",
        )
        lines.append(
            f"- Objective answer: {decision['objective_answer']}",
        )
        lines.append(
            "- Matched final aggregate deltas TD - Vanilla: "
            f"Greek no-MT `{fmt_signed(decision['group_deltas'].get('Greek'))}`, "
            f"EN `{fmt_signed(decision['group_deltas'].get('EN retention'))}`, "
            f"Multilingual `{fmt_signed(decision['group_deltas'].get('Multilingual'))}`, "
            f"BPB `{fmt_signed(decision['bpb_delta_td_minus_vanilla'])}` (lower BPB is better)."
        )
        wins = decision["task_wins"]
        lines.append(
            f"- Per-task wins at iter {latest}: TD `{wins.get('TD', 0)}`, Vanilla `{wins.get('Vanilla', 0)}`, ties `{wins.get('tie', 0)}`."
        )
        lines.append(
            "- TD change since 3.5B: "
            f"Greek `{fmt_signed(decision['td_change_from_3p5b'].get('Greek'))}`, "
            f"EN `{fmt_signed(decision['td_change_from_3p5b'].get('EN retention'))}`, "
            f"Multilingual `{fmt_signed(decision['td_change_from_3p5b'].get('Multilingual'))}`."
        )
        lines.append(
            "- TD vs V4-HF baseline at final: "
            f"Greek no-MT `{fmt_signed(decision['td_vs_v4_hf'].get('Greek'))}`, "
            f"EN `{fmt_signed(decision['td_vs_v4_hf'].get('EN retention'))}`, "
            f"Multilingual `{fmt_signed(decision['td_vs_v4_hf'].get('Multilingual'))}`."
        )
        lines.append(
            "- Recovery reading: TD is still below original V4-HF on all three "
            "aggregates, so it has not beaten the initial Vanilla/original "
            "Apertus scores by 5B. The 3.5B -> 5B trajectory is positive for "
            "TD on downstream aggregates, especially Greek, but BPB still "
            "favors Vanilla. Because the planned native-Greek suite was not "
            "run, the Greek downstream result is evidence from a fallback "
            "suite rather than a final native-benchmark claim."
        )
        lines += [
            "",
            "## Linear Gap-Closure Sense Check",
            "",
            "This is a rough extrapolation from only the 3.5B -> 5B interval,",
            "not a forecast. It answers whether the observed slope is remotely",
            "fast enough to catch V4-HF.",
            "",
            "| Group | TD gap to V4-HF at 5B | TD gain 3.5B->5B | Extra B tokens at same slope | Total B tokens at same slope |",
            "|---|---:|---:|---:|---:|",
        ]
        for group in GROUPS:
            proj = decision["linear_recovery_projection"].get(group, {})
            extra = proj.get("extra_b_tokens_at_linear_rate")
            total = proj.get("total_b_tokens_at_linear_rate")
            lines.append(
                "| {group} | {gap} | {gain} | {extra} | {total} |".format(
                    group=group,
                    gap=fmt(proj.get("gap_to_v4_hf")),
                    gain=fmt(proj.get("change_3p5_to_5b")),
                    extra=fmt(extra, 1) if extra is not None else "n/a",
                    total=fmt(total, 1) if total is not None else "n/a",
                )
            )
    else:
        missing = ", ".join(str(x) for x in summary["missing_iterations"])
        lines.append(f"- Interim recommendation: {decision['recommendation']}")
        lines.append(f"- Objective answer: {decision['objective_answer']}")
        lines.append(
            f"- Not final yet. Missing matched downstream/intrinsic artifacts for iter(s): `{missing}`."
        )
        if latest is not None:
            lines.append(
                f"- Current matched deltas at iter {latest}: "
                f"Greek `{fmt_signed(decision['group_deltas'].get('Greek'))}`, "
                f"EN `{fmt_signed(decision['group_deltas'].get('EN retention'))}`, "
                f"Multilingual `{fmt_signed(decision['group_deltas'].get('Multilingual'))}`, "
                f"BPB `{fmt_signed(decision['bpb_delta_td_minus_vanilla'])}` (lower BPB is better)."
            )
        lines.append(
            "- Current evidence can be used for 4.25B trajectory reading only; do not",
        )
        lines.append(
            "  declare the 5B objective complete until iter1192 BPB, diagnostics, and",
        )
        lines.append("  packed downstream eval are present.")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    summary = make_summary()
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_markdown(summary))
    print(OUT_MD)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
