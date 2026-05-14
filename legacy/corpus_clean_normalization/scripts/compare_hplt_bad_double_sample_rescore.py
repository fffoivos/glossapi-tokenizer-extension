#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


FILE_DETAIL_COLUMNS = [
    "rescored_greek_badness_score",
    "greek_latin_percentage",
    "table_ratio",
    "polytonic_ratio",
    "len_greek",
    "total_words",
    "v_pen",
    "c_pen",
    "bad_dbl",
    "misplaced_sigma",
    "invalid_bigram",
    "long_word_count",
    "longest_word",
    "short_word_count",
    "max_run",
    "v_rate",
    "c_rate",
    "d_rate",
    "sigma_end_rate",
    "bigram_rate",
    "long_word_rate",
    "short_ratio",
    "short_pen",
    "flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_noise_module():
    import glossapi_rs_noise  # type: ignore

    return glossapi_rs_noise


def derive_component_fields(detail: dict) -> dict[str, float | str]:
    combo_penalty = (
        float(detail["v_rate"])
        + 1.5 * float(detail["c_rate"])
        + 2.0 * float(detail["d_rate"])
        + 2.5 * float(detail["sigma_end_rate"])
        + 2.0 * float(detail["bigram_rate"])
    )
    short_penalty = float(detail["short_pen"])
    long_word_penalty = float(detail["long_word_rate"])
    shape_penalty = short_penalty + long_word_penalty
    component_values = {
        "vowel_combo": float(detail["v_rate"]),
        "consonant_combo": 1.5 * float(detail["c_rate"]),
        "bad_double": 2.0 * float(detail["d_rate"]),
        "misplaced_sigma": 2.5 * float(detail["sigma_end_rate"]),
        "invalid_bigram": 2.0 * float(detail["bigram_rate"]),
        "short_word_excess": short_penalty,
        "long_word_penalty": long_word_penalty,
    }
    dominant_component = max(component_values.items(), key=lambda kv: kv[1])[0]
    if combo_penalty > shape_penalty:
        dominant_group = "combination_driven"
    elif shape_penalty > combo_penalty:
        dominant_group = "word_shape_driven"
    else:
        dominant_group = "tied"
    return {
        "combo_penalty": combo_penalty,
        "shape_penalty": shape_penalty,
        "dominant_component": dominant_component,
        "dominant_group": dominant_group,
    }


def iter_sample_rows(sample_root: Path):
    for meta_path in sorted(sample_root.rglob("*.json")):
        if meta_path.name == "summary.json":
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text_name = meta.get("text_file")
        if not text_name:
            continue
        text_path = meta_path.with_name(text_name)
        if not text_path.exists():
            raise SystemExit(f"Missing text file for {meta_path}: {text_path}")
        yield meta_path, text_path, meta


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)

    noise_mod = load_noise_module()

    comparisons: list[dict] = []
    by_component = defaultdict(Counter)
    counts = Counter()

    for meta_path, text_path, meta in iter_sample_rows(args.sample_root):
        values = noise_mod.score_markdown_file_detailed(str(text_path))
        detail = dict(zip(FILE_DETAIL_COLUMNS, values))
        detail.update(derive_component_fields(detail))

        component = str(meta.get("component_folder") or meta_path.parent.name)
        old_score = float(meta["rescored_greek_badness_score"])
        new_score = float(detail["rescored_greek_badness_score"])
        old_bad_dbl = int(meta["bad_dbl"])
        new_bad_dbl = int(detail["bad_dbl"])
        old_dominant = str(meta["dominant_component"])
        new_dominant = str(detail["dominant_component"])

        row = {
            "component_folder": component,
            "source_doc_id": meta["source_doc_id"],
            "sample_index": int(meta["sample_index"]),
            "meta_path": str(meta_path),
            "text_path": str(text_path),
            "old_score": old_score,
            "new_score": new_score,
            "score_delta": new_score - old_score,
            "old_bad_dbl": old_bad_dbl,
            "new_bad_dbl": new_bad_dbl,
            "bad_dbl_delta": new_bad_dbl - old_bad_dbl,
            "old_d_rate": float(meta["d_rate"]),
            "new_d_rate": float(detail["d_rate"]),
            "d_rate_delta": float(detail["d_rate"]) - float(meta["d_rate"]),
            "old_max_run": int(meta["max_run"]),
            "new_max_run": int(detail["max_run"]),
            "old_dominant_component": old_dominant,
            "new_dominant_component": new_dominant,
            "dominant_component_changed": new_dominant != old_dominant,
            "new_combo_penalty": float(detail["combo_penalty"]),
            "new_shape_penalty": float(detail["shape_penalty"]),
        }
        comparisons.append(row)

        counts["sample_docs"] += 1
        by_component[component]["sample_docs"] += 1
        if new_bad_dbl < old_bad_dbl:
            counts["bad_dbl_decreased"] += 1
            by_component[component]["bad_dbl_decreased"] += 1
        if new_bad_dbl == 0 and old_bad_dbl > 0:
            counts["bad_dbl_zeroed"] += 1
            by_component[component]["bad_dbl_zeroed"] += 1
        if new_score < old_score:
            counts["score_decreased"] += 1
            by_component[component]["score_decreased"] += 1
        if new_dominant != old_dominant:
            counts["dominant_component_changed"] += 1
            by_component[component]["dominant_component_changed"] += 1

    comparisons.sort(key=lambda row: (row["bad_dbl_delta"], row["score_delta"]))
    top_reductions = comparisons[: args.top_k]

    component_summary = {}
    for component, counter in sorted(by_component.items()):
        component_rows = [row for row in comparisons if row["component_folder"] == component]
        component_summary[component] = {
            "sample_docs": counter["sample_docs"],
            "bad_dbl_decreased": counter["bad_dbl_decreased"],
            "bad_dbl_zeroed": counter["bad_dbl_zeroed"],
            "score_decreased": counter["score_decreased"],
            "dominant_component_changed": counter["dominant_component_changed"],
            "mean_old_bad_dbl": (
                sum(row["old_bad_dbl"] for row in component_rows) / len(component_rows)
                if component_rows
                else 0.0
            ),
            "mean_new_bad_dbl": (
                sum(row["new_bad_dbl"] for row in component_rows) / len(component_rows)
                if component_rows
                else 0.0
            ),
            "mean_score_delta": (
                sum(row["score_delta"] for row in component_rows) / len(component_rows)
                if component_rows
                else 0.0
            ),
        }

    summary = {
        "sample_root": str(args.sample_root),
        "sample_docs": counts["sample_docs"],
        "bad_dbl_decreased": counts["bad_dbl_decreased"],
        "bad_dbl_zeroed": counts["bad_dbl_zeroed"],
        "score_decreased": counts["score_decreased"],
        "dominant_component_changed": counts["dominant_component_changed"],
        "mean_old_bad_dbl": (
            sum(row["old_bad_dbl"] for row in comparisons) / len(comparisons)
            if comparisons
            else 0.0
        ),
        "mean_new_bad_dbl": (
            sum(row["new_bad_dbl"] for row in comparisons) / len(comparisons)
            if comparisons
            else 0.0
        ),
        "mean_score_delta": (
            sum(row["score_delta"] for row in comparisons) / len(comparisons)
            if comparisons
            else 0.0
        ),
        "component_summary": component_summary,
        "top_bad_double_reductions": top_reductions,
    }
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.output_md is not None:
        lines = [
            "# HPLT Bad Double Sample Rescore",
            "",
            f"- sample docs: `{summary['sample_docs']}`",
            f"- docs with lower `bad_dbl`: `{summary['bad_dbl_decreased']}`",
            f"- docs zeroed on `bad_dbl`: `{summary['bad_dbl_zeroed']}`",
            f"- docs with lower total score: `{summary['score_decreased']}`",
            f"- docs whose dominant component changed: `{summary['dominant_component_changed']}`",
            f"- mean `bad_dbl`: `{summary['mean_old_bad_dbl']:.2f}` -> `{summary['mean_new_bad_dbl']:.2f}`",
            f"- mean score delta: `{summary['mean_score_delta']:.2f}`",
            "",
            "## By Component",
            "",
        ]
        for component, data in component_summary.items():
            lines.append(
                f"- `{component}`: docs={data['sample_docs']}, "
                f"bad_dbl_decreased={data['bad_dbl_decreased']}, "
                f"bad_dbl_zeroed={data['bad_dbl_zeroed']}, "
                f"dominant_component_changed={data['dominant_component_changed']}, "
                f"mean_bad_dbl={data['mean_old_bad_dbl']:.2f}->{data['mean_new_bad_dbl']:.2f}, "
                f"mean_score_delta={data['mean_score_delta']:.2f}"
            )
        lines.extend(["", "## Top Reductions", ""])
        for row in top_reductions:
            lines.append(
                f"- `{row['component_folder']}` sample `{row['sample_index']}` "
                f"`{row['source_doc_id']}`: `bad_dbl` {row['old_bad_dbl']} -> {row['new_bad_dbl']}, "
                f"score {row['old_score']:.2f} -> {row['new_score']:.2f}, "
                f"dominant `{row['old_dominant_component']}` -> `{row['new_dominant_component']}`"
            )
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
