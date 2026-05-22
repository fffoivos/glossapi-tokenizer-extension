#!/usr/bin/env python3
"""Build a small CPT held-out JSONL from the post-Apertus-dedup Greek pool.

The slice is for tokenizer-fair metrics and new-token diagnostics. It is not a
replacement for benchmark decontamination; it is a training-disjoint Greek text
slice built from the same cleaned pool that fed the bakeoff.
"""
from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_QUOTAS = {
    "greek_hplt_clean60": 250,
    "greek_literary": 130,
    "greek_dialogue_textbooks": 45,
    "greek_academic": 40,
    "greek_legal_civic": 25,
    "greek_dictionary_misc": 10,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--selected-parquet", required=True)
    p.add_argument("--bulk-mix-jsonl", required=True)
    p.add_argument("--recipe-json", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--manifest-json", required=True)
    p.add_argument("--seed", default="cpt-heldout-v1-20260522")
    p.add_argument("--batch-size", type=int, default=10_000)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--min-greek-percentage", type=float, default=50.0)
    p.add_argument("--quotas-json", default="")
    p.add_argument("--allow-partial", action="store_true")
    return p.parse_args()


def load_training_doc_ids(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            source = str(row.get("source") or "")
            if not source.startswith("greek_"):
                continue
            doc_id = row.get("doc_id")
            if doc_id:
                out.add(str(doc_id))
    return out


def load_greek_specs(recipe_path: Path) -> list[dict[str, Any]]:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    specs = []
    for spec in recipe["sources"]:
        if spec.get("bucket") != "greek":
            continue
        compiled = dict(spec)
        if spec.get("filter_values"):
            compiled["_filter_values_set"] = set(spec["filter_values"])
        if spec.get("filter_values_regex"):
            compiled["_filter_regex"] = re.compile(spec["filter_values_regex"])
        specs.append(compiled)
    if not specs:
        raise SystemExit(f"No Greek source specs found in {recipe_path}")
    return specs


def classify_source(source_dataset: str, specs: list[dict[str, Any]]) -> str | None:
    for spec in specs:
        field = spec.get("filter_field")
        if field != "source_dataset":
            continue
        values = spec.get("_filter_values_set")
        if values is not None and source_dataset in values:
            return str(spec["name"])
        regex = spec.get("_filter_regex")
        if regex is not None and regex.search(source_dataset):
            return str(spec["name"])
    return None


def score_doc(seed: str, doc_key: str) -> int:
    digest = hashlib.sha1(f"{seed}\0{doc_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def push_candidate(
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]],
    quotas: dict[str, int],
    source: str,
    score: int,
    row: dict[str, Any],
) -> None:
    quota = quotas.get(source, 0)
    if quota <= 0:
        return
    heap = heaps.setdefault(source, [])
    item = (-score, str(row["doc_id"]), row)
    if len(heap) < quota:
        heapq.heappush(heap, item)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def main() -> None:
    args = parse_args()
    selected = Path(args.selected_parquet)
    bulk_mix = Path(args.bulk_mix_jsonl)
    recipe = Path(args.recipe_json)
    output_jsonl = Path(args.output_jsonl)
    manifest_json = Path(args.manifest_json)

    quotas = dict(DEFAULT_QUOTAS)
    if args.quotas_json:
        quotas = {k: int(v) for k, v in json.loads(args.quotas_json).items()}

    specs = load_greek_specs(recipe)
    print(f"Loading Greek training doc ids from {bulk_mix} ...", flush=True)
    train_doc_ids = load_training_doc_ids(bulk_mix)
    print(f"Loaded {len(train_doc_ids):,} Greek training doc ids", flush=True)

    columns = [
        "text",
        "source_dataset",
        "doc_key",
        "is_historical_or_polytonic",
        "greek_percentage",
        "polytonic_ratio",
        "len_greek",
    ]

    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {k: [] for k in quotas}
    counters = {
        "rows_seen": 0,
        "matched_greek_recipe": 0,
        "excluded_training_doc_id": 0,
        "excluded_short_or_low_greek": 0,
        "candidate_rows": 0,
    }
    seen_by_source = {k: 0 for k in quotas}
    candidate_by_source = {k: 0 for k in quotas}

    pf = pq.ParquetFile(selected)
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=args.batch_size, columns=columns), 1):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            counters["rows_seen"] += 1
            doc_key = str(data["doc_key"][i])
            source_dataset = str(data["source_dataset"][i])
            source = classify_source(source_dataset, specs)
            if source is None or source not in quotas:
                continue
            counters["matched_greek_recipe"] += 1
            seen_by_source[source] += 1
            if doc_key in train_doc_ids:
                counters["excluded_training_doc_id"] += 1
                continue
            text = data["text"][i]
            if not isinstance(text, str):
                continue
            greek_percentage = data["greek_percentage"][i]
            if len(text) < args.min_chars or float(greek_percentage or 0.0) < args.min_greek_percentage:
                counters["excluded_short_or_low_greek"] += 1
                continue
            score = score_doc(args.seed, doc_key)
            row = {
                "text": text,
                "source": source,
                "register": source.replace("greek_", ""),
                "doc_id": doc_key,
                "source_dataset": source_dataset,
                "is_historical_or_polytonic": bool(data["is_historical_or_polytonic"][i]),
                "greek_percentage": greek_percentage,
                "polytonic_ratio": data["polytonic_ratio"][i],
                "len_greek": data["len_greek"][i],
                "heldout_seed": args.seed,
                "selection_score": score,
            }
            counters["candidate_rows"] += 1
            candidate_by_source[source] += 1
            push_candidate(heaps, quotas, source, score, row)
        if batch_idx % 250 == 0:
            filled = {k: len(v) for k, v in heaps.items()}
            print(f"batch={batch_idx:,} rows={counters['rows_seen']:,} filled={filled}", flush=True)

    selected_rows: list[dict[str, Any]] = []
    missing: dict[str, dict[str, int]] = {}
    for source, quota in quotas.items():
        rows = [item[2] for item in heaps.get(source, [])]
        rows.sort(key=lambda r: (int(r["selection_score"]), str(r["doc_id"])))
        selected_rows.extend(rows)
        if len(rows) < quota:
            missing[source] = {"wanted": quota, "got": len(rows)}

    if missing and not args.allow_partial:
        raise SystemExit(f"Could not fill all quotas: {missing}")

    selected_rows.sort(key=lambda r: (r["source"], int(r["selection_score"]), str(r["doc_id"])))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "output_jsonl": str(output_jsonl),
        "selected_parquet": str(selected),
        "bulk_mix_jsonl": str(bulk_mix),
        "recipe_json": str(recipe),
        "seed": args.seed,
        "quotas": quotas,
        "counts": counters,
        "seen_by_source": seen_by_source,
        "candidate_by_source": candidate_by_source,
        "selected_by_source": {k: len(v) for k, v in heaps.items()},
        "missing": missing,
        "selection_rule": "lowest sha1(seed + NUL + doc_key) per Greek recipe source after excluding bulk_mix Greek doc_ids",
        "filters": {
            "min_chars": args.min_chars,
            "min_greek_percentage": args.min_greek_percentage,
        },
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
