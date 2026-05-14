#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from heapq import heappush, heappushpop
from pathlib import Path
from glob import glob

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


DETAIL_COLUMNS = [
    "path",
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


def _push_top_k(
    heap: list[tuple[float, str, dict]], row: dict, key_name: str, k: int
) -> None:
    item = (float(row[key_name]), str(row["source_doc_id"]), row.copy())
    if len(heap) < k:
        heappush(heap, item)
        return
    if item[0] > heap[0][0]:
        heappushpop(heap, item)


def _finalize_top_k(heap: list[tuple[float, str, dict]]) -> list[dict]:
    return [row for _, _, row in sorted(heap, key=lambda x: x[0], reverse=True)]


def _sanitize_text(text: str | None) -> str:
    if not text:
        return "\n"
    text = text.rstrip("\n")
    return text + "\n"


def _load_noise_module():
    import glossapi_rs_noise  # type: ignore

    return glossapi_rs_noise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-glob",
        default="/home/foivos/data/glossapi_work/hf_release_publish/data/HPLT__ell_Grek_ge8_no_mt.*.parquet",
    )
    p.add_argument("--threshold", type=float, default=60.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4)))
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_mod = _load_noise_module()
    files = sorted(glob(args.data_glob))
    if not files:
        raise SystemExit(f"No parquet files matched {args.data_glob}")

    dataset = ds.dataset(files, format="parquet")
    columns = ["source_doc_id", "text", "greek_badness_score"]

    per_doc_rows: list[dict] = []
    component_positive = Counter()
    dominant_component_counts = Counter()
    dominant_group_counts = Counter()
    flag_counts = Counter()

    top_combo_share: list[tuple[float, dict]] = []
    top_shape_share: list[tuple[float, dict]] = []
    top_total_score: list[tuple[float, dict]] = []

    total_rows = 0
    flagged_rows = 0
    total_score_sum = 0.0
    total_combo_sum = 0.0
    total_shape_sum = 0.0
    total_other_sum = 0.0
    combo_gt60 = 0
    shape_gt60 = 0
    combo_ge_half = 0
    combo_ge_75 = 0
    shape_ge_half = 0
    shape_ge_75 = 0
    combo_positive = 0
    shape_positive = 0
    pure_shape_positive = 0
    pure_combo_positive = 0
    rescored_mismatch_count = 0

    batch_rows: list[dict] = []

    def flush_batch() -> None:
        nonlocal flagged_rows, total_score_sum, total_combo_sum, total_shape_sum, total_other_sum
        nonlocal combo_gt60, shape_gt60, combo_ge_half, combo_ge_75, shape_ge_half, shape_ge_75
        nonlocal combo_positive, shape_positive, pure_combo_positive, pure_shape_positive
        nonlocal rescored_mismatch_count
        if not batch_rows:
            return
        with tempfile.TemporaryDirectory(prefix="hplt_badness_batch_", dir=str(output_dir)) as tmpdir:
            tmp_path = Path(tmpdir)
            path_to_meta: dict[str, dict] = {}
            for i, row in enumerate(batch_rows):
                stem = f"doc_{i:05d}"
                md_path = tmp_path / f"{stem}.md"
                md_path.write_text(_sanitize_text(row["text"]), encoding="utf-8")
                path_to_meta[str(md_path)] = row

            detailed = noise_mod.score_markdown_directory_detailed(str(tmp_path), args.threads)
            for record in detailed:
                values = list(record)
                detail = dict(zip(DETAIL_COLUMNS, values))
                meta = path_to_meta[detail["path"]]
                stored_score = float(meta["greek_badness_score"])
                rescored = float(detail["rescored_greek_badness_score"])
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
                other_penalty = max(0.0, rescored - combo_penalty - shape_penalty)
                combo_share = combo_penalty / rescored if rescored > 0 else 0.0
                shape_share = shape_penalty / rescored if rescored > 0 else 0.0
                sigma_share = (2.5 * float(detail["sigma_end_rate"]) / rescored) if rescored > 0 else 0.0
                flags = list(detail["flags"])
                if any(flags):
                    for flag in flags:
                        flag_counts[str(flag)] += 1

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
                dominant_component_counts[dominant_component] += 1

                if combo_penalty > 0:
                    combo_positive += 1
                if shape_penalty > 0:
                    shape_positive += 1
                if combo_penalty > 0 and shape_penalty == 0:
                    pure_combo_positive += 1
                if shape_penalty > 0 and combo_penalty == 0:
                    pure_shape_positive += 1
                if combo_penalty > 60.0:
                    combo_gt60 += 1
                if shape_penalty > 60.0:
                    shape_gt60 += 1
                if combo_share >= 0.5:
                    combo_ge_half += 1
                if combo_share >= 0.75:
                    combo_ge_75 += 1
                if shape_share >= 0.5:
                    shape_ge_half += 1
                if shape_share >= 0.75:
                    shape_ge_75 += 1

                if combo_penalty > shape_penalty:
                    dominant_group = "combination_driven"
                elif shape_penalty > combo_penalty:
                    dominant_group = "word_shape_driven"
                else:
                    dominant_group = "tied"
                dominant_group_counts[dominant_group] += 1

                if abs(stored_score - rescored) > 0.05:
                    rescored_mismatch_count += 1

                if float(detail["v_pen"]) > 0:
                    component_positive["v_pen_docs"] += 1
                if float(detail["c_pen"]) > 0:
                    component_positive["c_pen_docs"] += 1
                if float(detail["bad_dbl"]) > 0:
                    component_positive["bad_dbl_docs"] += 1
                if float(detail["misplaced_sigma"]) > 0:
                    component_positive["misplaced_sigma_docs"] += 1
                if float(detail["invalid_bigram"]) > 0:
                    component_positive["invalid_bigram_docs"] += 1
                if short_penalty > 0:
                    component_positive["short_pen_docs"] += 1
                if long_word_penalty > 0:
                    component_positive["long_word_docs"] += 1

                flagged_rows += 1
                total_score_sum += rescored
                total_combo_sum += combo_penalty
                total_shape_sum += shape_penalty
                total_other_sum += other_penalty

                row_out = {
                    "source_doc_id": meta["source_doc_id"],
                    "stored_greek_badness_score": stored_score,
                    "rescored_greek_badness_score": rescored,
                    "len_greek": int(detail["len_greek"]),
                    "total_words": int(detail["total_words"]),
                    "v_pen": int(detail["v_pen"]),
                    "c_pen": int(detail["c_pen"]),
                    "bad_dbl": int(detail["bad_dbl"]),
                    "misplaced_sigma": int(detail["misplaced_sigma"]),
                    "invalid_bigram": int(detail["invalid_bigram"]),
                    "long_word_count": int(detail["long_word_count"]),
                    "longest_word": int(detail["longest_word"]),
                    "short_word_count": int(detail["short_word_count"]),
                    "max_run": int(detail["max_run"]),
                    "v_rate": float(detail["v_rate"]),
                    "c_rate": float(detail["c_rate"]),
                    "d_rate": float(detail["d_rate"]),
                    "sigma_end_rate": float(detail["sigma_end_rate"]),
                    "bigram_rate": float(detail["bigram_rate"]),
                    "long_word_rate": long_word_penalty,
                    "short_ratio": float(detail["short_ratio"]),
                    "short_pen": short_penalty,
                    "greek_latin_percentage": float(detail["greek_latin_percentage"]),
                    "table_ratio": float(detail["table_ratio"]),
                    "polytonic_ratio": float(detail["polytonic_ratio"]),
                    "flags": json.dumps(flags, ensure_ascii=False),
                    "combo_penalty": combo_penalty,
                    "shape_penalty": shape_penalty,
                    "combo_share": combo_share,
                    "shape_share": shape_share,
                    "sigma_share": sigma_share,
                    "dominant_component": dominant_component,
                    "dominant_group": dominant_group,
                }
                per_doc_rows.append(row_out)
                _push_top_k(top_combo_share, row_out, "combo_share", args.top_k)
                _push_top_k(top_shape_share, row_out, "shape_share", args.top_k)
                _push_top_k(top_total_score, row_out, "rescored_greek_badness_score", args.top_k)
        batch_rows.clear()

    for batch in dataset.to_batches(columns=columns):
        total_rows += batch.num_rows
        g = pc.cast(batch["greek_badness_score"], "float64", safe=False)
        mask = pc.fill_null(pc.greater(g, args.threshold), False)
        indices = pc.indices_nonzero(mask).to_pylist()
        for idx in indices:
            batch_rows.append(
                {
                    "source_doc_id": batch["source_doc_id"][idx].as_py(),
                    "text": batch["text"][idx].as_py(),
                    "greek_badness_score": batch["greek_badness_score"][idx].as_py(),
                }
            )
            if len(batch_rows) >= args.batch_size:
                flush_batch()
    flush_batch()

    per_doc_table = pa.Table.from_pylist(per_doc_rows)
    pq.write_table(per_doc_table, output_dir / "flagged_doc_components.parquet", compression="zstd")

    def pct(n: int, d: int) -> float:
        return round((100.0 * n / d), 4) if d else 0.0

    summary = {
        "dataset_total_rows": total_rows,
        "flagged_rows_greek_badness_gt_threshold": flagged_rows,
        "flagged_percentage_of_dataset": pct(flagged_rows, total_rows),
        "threshold": args.threshold,
        "rescored_mismatch_count_abs_gt_0_05": rescored_mismatch_count,
        "binary_classes": {
            "combo_penalty_gt_60_docs": combo_gt60,
            "combo_penalty_gt_60_pct_of_flagged": pct(combo_gt60, flagged_rows),
            "shape_penalty_gt_60_docs": shape_gt60,
            "shape_penalty_gt_60_pct_of_flagged": pct(shape_gt60, flagged_rows),
            "combo_penalty_ge_50pct_of_score_docs": combo_ge_half,
            "combo_penalty_ge_50pct_of_score_pct": pct(combo_ge_half, flagged_rows),
            "combo_penalty_ge_75pct_of_score_docs": combo_ge_75,
            "combo_penalty_ge_75pct_of_score_pct": pct(combo_ge_75, flagged_rows),
            "shape_penalty_ge_50pct_of_score_docs": shape_ge_half,
            "shape_penalty_ge_50pct_of_score_pct": pct(shape_ge_half, flagged_rows),
            "shape_penalty_ge_75pct_of_score_docs": shape_ge_75,
            "shape_penalty_ge_75pct_of_score_pct": pct(shape_ge_75, flagged_rows),
            "combo_positive_docs": combo_positive,
            "combo_positive_pct_of_flagged": pct(combo_positive, flagged_rows),
            "shape_positive_docs": shape_positive,
            "shape_positive_pct_of_flagged": pct(shape_positive, flagged_rows),
            "pure_combo_positive_docs": pure_combo_positive,
            "pure_combo_positive_pct_of_flagged": pct(pure_combo_positive, flagged_rows),
            "pure_shape_positive_docs": pure_shape_positive,
            "pure_shape_positive_pct_of_flagged": pct(pure_shape_positive, flagged_rows),
        },
        "score_mass": {
            "total_score_sum": round(total_score_sum, 4),
            "combo_penalty_sum": round(total_combo_sum, 4),
            "combo_penalty_pct_of_total_score_mass": round(100.0 * total_combo_sum / total_score_sum, 4) if total_score_sum else 0.0,
            "shape_penalty_sum": round(total_shape_sum, 4),
            "shape_penalty_pct_of_total_score_mass": round(100.0 * total_shape_sum / total_score_sum, 4) if total_score_sum else 0.0,
            "other_penalty_sum": round(total_other_sum, 4),
            "other_penalty_pct_of_total_score_mass": round(100.0 * total_other_sum / total_score_sum, 4) if total_score_sum else 0.0,
        },
        "component_presence_doc_counts": dict(component_positive),
        "component_presence_doc_percentages": {
            key: pct(val, flagged_rows) for key, val in component_positive.items()
        },
        "dominant_component_counts": dict(dominant_component_counts),
        "dominant_component_percentages": {
            key: pct(val, flagged_rows) for key, val in dominant_component_counts.items()
        },
        "dominant_group_counts": dict(dominant_group_counts),
        "dominant_group_percentages": {
            key: pct(val, flagged_rows) for key, val in dominant_group_counts.items()
        },
        "flags_counts": dict(flag_counts),
        "top_examples": {
            "highest_combo_share": _finalize_top_k(top_combo_share),
            "highest_shape_share": _finalize_top_k(top_shape_share),
            "highest_total_score": _finalize_top_k(top_total_score),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
