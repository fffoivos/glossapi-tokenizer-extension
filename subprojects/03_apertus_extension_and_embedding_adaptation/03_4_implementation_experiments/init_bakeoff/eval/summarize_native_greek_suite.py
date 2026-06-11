#!/usr/bin/env python3
"""Summarize native-Greek eval outputs across checkpoints."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MCQ_GENERAL = ["greekmmlu", "ilsp_medical_mcqa", "ilsp_mcqa_asep"]
MCQ_DOMAIN_OPTIONAL = ["plutus_qa"]
GREEK_NLP_TRANSLATION_TASKS = {"machine_translation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcq-root", type=Path, required=True)
    parser.add_argument("--greek-nlp-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _unique_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def collect_mcq(root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    for summary_path in _unique_paths(
        sorted(root.glob("chunk_*/*/*_native_mcq_summary.csv"))
        + sorted(root.glob("*/*_native_mcq_summary.csv"))
        + sorted(root.glob("*_native_mcq_summary.csv"))
    ):
        model = summary_path.name.replace("_native_mcq_summary.csv", "")
        for row in _read_csv(summary_path):
            if row.get("subject") != "__all__":
                continue
            rows.append(
                {
                    "model": model,
                    "benchmark": row["benchmark"],
                    "n": int(row["n"]),
                    "accuracy": _safe_float(row["accuracy"]),
                    "correct": int(row["correct"]),
                    "source_file": str(summary_path),
                }
            )

    by_model: Dict[str, Dict[str, float]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["benchmark"]] = float(row["accuracy"])

    for model, scores in sorted(by_model.items()):
        general_values = [scores[key] for key in MCQ_GENERAL if key in scores]
        domain_values = [scores[key] for key in MCQ_DOMAIN_OPTIONAL if key in scores]
        aggregate_rows.append(
            {
                "model": model,
                "native_mcq_general": sum(general_values) / len(general_values) if general_values else "",
                "native_mcq_general_n_tasks": len(general_values),
                "native_mcq_with_domain": (
                    sum(general_values + domain_values) / len(general_values + domain_values)
                    if general_values or domain_values
                    else ""
                ),
                "native_mcq_with_domain_n_tasks": len(general_values + domain_values),
                **{f"mcq_{key}": scores.get(key, "") for key in MCQ_GENERAL + MCQ_DOMAIN_OPTIONAL},
            }
        )
    return rows, aggregate_rows


def collect_greek_nlp(root: Optional[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if root is None:
        return [], []
    rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    for summary_path in _unique_paths(
        sorted(root.glob("chunk_*/*/all_tasks_summary.csv"))
        + sorted(root.glob("*/*/all_tasks_summary.csv"))
        + sorted(root.glob("all_tasks_summary.csv"))
    ):
        model = summary_path.parent.name.rstrip("_")
        for row in _read_csv(summary_path):
            task = row.get("task_name") or row.get("task")
            metric = ""
            value: Optional[float] = None
            higher_is_better = True
            for candidate_metric in [
                "accuracy",
                "macro_f1",
                "entity_f1",
                "rouge_l",
                "bertscore_f1",
                "gleu_vs_reference",
                "bleu",
                "chrf",
            ]:
                candidate_value = _safe_float(row.get(candidate_metric))
                if candidate_value is not None:
                    metric = candidate_metric
                    value = candidate_value
                    break
            if value is None:
                for candidate_metric in ["wer_vs_reference", "cer_vs_reference"]:
                    candidate_value = _safe_float(row.get(candidate_metric))
                    if candidate_value is not None:
                        metric = candidate_metric
                        value = candidate_value
                        higher_is_better = False
                        break
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "target_lang": row.get("target_lang", ""),
                    "metric": metric,
                    "value": value if value is not None else "",
                    "higher_is_better": higher_is_better,
                    "headline_included": task not in GREEK_NLP_TRANSLATION_TASKS,
                    "source_file": str(summary_path),
                }
            )

    by_model: Dict[str, List[float]] = {}
    for row in rows:
        if not row["headline_included"]:
            continue
        value = row["value"]
        if value == "" or not row["higher_is_better"]:
            continue
        by_model.setdefault(row["model"], []).append(float(value))
    for model, values in sorted(by_model.items()):
        aggregate_rows.append(
            {
                "model": model,
                "greek_nlp_supporting_mean": sum(values) / len(values) if values else "",
                "greek_nlp_supporting_n_metrics": len(values),
            }
        )
    return rows, aggregate_rows


def render_markdown(
    path: Path,
    mcq_aggregate: List[Dict[str, Any]],
    greek_nlp_aggregate: List[Dict[str, Any]],
) -> None:
    by_model = {row["model"]: row for row in mcq_aggregate}
    for row in greek_nlp_aggregate:
        by_model.setdefault(row["model"], {}).update(row)

    lines = [
        "# Native Greek suite summary",
        "",
        "Native headline uses vetted native Greek tasks only. MT diagnostics are excluded.",
        "",
        "| Model | Native MCQ general | MCQ tasks | MCQ + Plutus | greek-nlp supporting mean | greek-nlp metrics |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in sorted(by_model.items()):
        lines.append(
            "| {model} | {native_mcq_general} | {native_mcq_general_n_tasks} | "
            "{native_mcq_with_domain} | {greek_nlp_supporting_mean} | {greek_nlp_supporting_n_metrics} |".format(
                model=model,
                native_mcq_general=_fmt(row.get("native_mcq_general")),
                native_mcq_general_n_tasks=row.get("native_mcq_general_n_tasks", ""),
                native_mcq_with_domain=_fmt(row.get("native_mcq_with_domain")),
                greek_nlp_supporting_mean=_fmt(row.get("greek_nlp_supporting_mean")),
                greek_nlp_supporting_n_metrics=row.get("greek_nlp_supporting_n_metrics", ""),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `native_mcq_general` averages GreekMMLU, ILSP Medical MCQA, and ILSP ASEP MCQA.",
            "- `MCQ + Plutus` adds the domain-specific Plutus QA finance task.",
            "- `greek-nlp supporting mean` excludes the upstream `machine_translation` task.",
            "- Per-task CSVs remain authoritative for domain-specific interpretation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mcq_rows, mcq_aggregate = collect_mcq(args.mcq_root)
    greek_nlp_rows, greek_nlp_aggregate = collect_greek_nlp(args.greek_nlp_root)

    _write_csv(
        args.output_dir / "native_mcq_per_task.csv",
        mcq_rows,
        ["model", "benchmark", "n", "accuracy", "correct", "source_file"],
    )
    _write_csv(
        args.output_dir / "native_mcq_aggregate.csv",
        mcq_aggregate,
        [
            "model",
            "native_mcq_general",
            "native_mcq_general_n_tasks",
            "native_mcq_with_domain",
            "native_mcq_with_domain_n_tasks",
            "mcq_greekmmlu",
            "mcq_ilsp_medical_mcqa",
            "mcq_ilsp_mcqa_asep",
            "mcq_plutus_qa",
        ],
    )
    if greek_nlp_rows:
        _write_csv(
            args.output_dir / "greek_nlp_per_task.csv",
            greek_nlp_rows,
            [
                "model",
                "task",
                "target_lang",
                "metric",
                "value",
                "higher_is_better",
                "headline_included",
                "source_file",
            ],
        )
        _write_csv(
            args.output_dir / "greek_nlp_supporting_aggregate.csv",
            greek_nlp_aggregate,
            ["model", "greek_nlp_supporting_mean", "greek_nlp_supporting_n_metrics"],
        )
    render_markdown(args.output_dir / "NATIVE_GREEK_SUITE_SUMMARY.md", mcq_aggregate, greek_nlp_aggregate)
    (args.output_dir / "summary_manifest.json").write_text(
        json.dumps(
            {
                "mcq_root": str(args.mcq_root),
                "greek_nlp_root": str(args.greek_nlp_root) if args.greek_nlp_root else None,
                "outputs": sorted(path.name for path in args.output_dir.iterdir()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
