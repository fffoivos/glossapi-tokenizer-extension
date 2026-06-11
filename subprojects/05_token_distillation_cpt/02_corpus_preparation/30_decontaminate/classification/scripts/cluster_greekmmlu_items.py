#!/usr/bin/env python3
"""Discovery-pass clustering of GreekMMLU items.

Bottom-up EDA over `dascim/GreekMMLU` (or any compatible benchmark JSONL):
groups items by identical answer sets and identical question stems / prefixes,
ranks the resulting clusters by frequency, and writes a single discovery JSON
that the user reviews to assign semantic labels to the high-frequency
clusters.

This script does NOT label categories. It surfaces the empirical clusters
that the labeling step works from.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!\?\"'·]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dascim/GreekMMLU")
    parser.add_argument("--config", default="All")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--sample-per-cluster", type=int, default=3)
    parser.add_argument("--question-prefix-token-grid", default="5,8,12")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return " ".join(text.casefold().split())


def normalize_choice(choice: Any) -> str:
    text = normalize_text(str(choice))
    return TRAILING_PUNCT_RE.sub("", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def load_dataset_rows(dataset: str, config: str, split: str) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore[import-not-found]

    return load_dataset(dataset, config, split=split)


def cluster_summary(buckets: dict[str, list[dict[str, Any]]], top_k: int, sample_n: int, total: int) -> dict[str, Any]:
    ranked = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    top_clusters = []
    for key, items in ranked[:top_k]:
        top_clusters.append(
            {
                "key": key,
                "count": len(items),
                "fraction_of_items": round(len(items) / total, 6) if total else 0.0,
                "subjects": sorted({str(rec.get("subject")) for rec in items[:200]}),
                "samples": items[:sample_n],
            }
        )
    sizes = sorted(len(items) for items in buckets.values())
    return {
        "total_items": total,
        "unique_clusters": len(buckets),
        "top_k_shown": min(top_k, len(buckets)),
        "long_tail_summary": {
            "n_clusters_size_1": sum(1 for s in sizes if s == 1),
            "max_size": sizes[-1] if sizes else 0,
            "median_size": sizes[len(sizes) // 2] if sizes else 0,
            "items_in_top_5": sum(len(items) for _, items in ranked[:5]),
            "fraction_in_top_5": round(sum(len(items) for _, items in ranked[:5]) / total, 6) if total else 0.0,
            "items_in_top_20": sum(len(items) for _, items in ranked[:20]),
            "fraction_in_top_20": round(sum(len(items) for _, items in ranked[:20]) / total, 6) if total else 0.0,
        },
        "top_clusters": top_clusters,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix_widths = [int(x) for x in args.question_prefix_token_grid.split(",") if x.strip()]

    ds = load_dataset_rows(args.dataset, args.config, args.split)
    column_probe = list(ds.column_names) if hasattr(ds, "column_names") else []

    by_answer_set_raw: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_answer_set_norm: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_question_exact: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_prefix: dict[int, dict[str, list[dict[str, Any]]]] = {
        n: collections.defaultdict(list) for n in prefix_widths
    }

    total = 0
    answer_count_histogram: dict[int, int] = collections.Counter()
    answer_token_histogram: dict[int, int] = collections.Counter()

    for idx, row in enumerate(ds):
        question = str(row.get("question") or "")
        choices_raw = row.get("choices") or []
        if isinstance(choices_raw, dict):
            # some HF dataset shapes carry choices as {"text": [...], ...}
            choices_raw = choices_raw.get("text") or choices_raw.get("choices") or []
        choices = [str(c) for c in choices_raw]
        subject = row.get("subject") or row.get("category") or ""
        answer_idx = row.get("answer", row.get("answer_index"))

        try:
            answer_text = choices[int(answer_idx)] if answer_idx is not None and choices else ""
        except (ValueError, TypeError, IndexError):
            answer_text = ""

        answer_count_histogram[len(choices)] += 1
        if answer_text:
            answer_token_histogram[len(tokenize(answer_text))] += 1

        raw_key = json.dumps(sorted(c.strip() for c in choices), ensure_ascii=False)
        norm_key = json.dumps(sorted(normalize_choice(c) for c in choices), ensure_ascii=False)
        question_norm = normalize_text(question)
        question_tokens = tokenize(question)

        rec = {
            "example_id": f"greekmmlu:{idx}",
            "subject": str(subject),
            "question": question,
            "choices": choices,
            "answer_index": int(answer_idx) if answer_idx is not None else None,
            "answer_text": answer_text,
            "answer_token_count": len(tokenize(answer_text)) if answer_text else 0,
            "question_token_count": len(question_tokens),
        }

        by_answer_set_raw[raw_key].append(rec)
        by_answer_set_norm[norm_key].append(rec)
        by_question_exact[question_norm].append(rec)
        for n in prefix_widths:
            prefix_key = " ".join(question_tokens[:n])
            by_prefix[n][prefix_key].append(rec)
        total += 1

    generated = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    summary = {
        "schema": "greekmmlu-cluster-discovery-v1",
        "generated_utc": generated,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "total_items": total,
        "dataset_columns": column_probe,
        "answer_count_histogram": dict(sorted(answer_count_histogram.items())),
        "answer_token_histogram_top": dict(sorted(answer_token_histogram.items())[:30]),
        "answer_token_collapse_at_k13_count": sum(
            count for tokens, count in answer_token_histogram.items() if tokens < 13
        ),
        "answer_token_collapse_at_k13_fraction": round(
            sum(count for tokens, count in answer_token_histogram.items() if tokens < 13) / total,
            6,
        )
        if total
        else 0.0,
        "clusters": {
            "answer_set_raw": cluster_summary(by_answer_set_raw, args.top_k, args.sample_per_cluster, total),
            "answer_set_normalized": cluster_summary(by_answer_set_norm, args.top_k, args.sample_per_cluster, total),
            "question_exact_normalized": cluster_summary(by_question_exact, args.top_k, args.sample_per_cluster, total),
            **{
                f"question_prefix_{n}_tokens": cluster_summary(by_prefix[n], args.top_k, args.sample_per_cluster, total)
                for n in prefix_widths
            },
        },
    }

    out_path = args.output_dir / f"greekmmlu_clusters_{generated}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(out_path),
                "total_items": total,
                "answer_token_collapse_at_k13_fraction": summary["answer_token_collapse_at_k13_fraction"],
                "report_keys": list(summary["clusters"].keys()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
