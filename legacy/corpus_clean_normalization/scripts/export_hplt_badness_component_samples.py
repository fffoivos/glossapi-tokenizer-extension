#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--components-parquet",
        default="/home/foivos/Projects/glossapi-tokenizer-extension/corpus_clean_normalization/runs/hplt_greek_badness_contributions_20260419_v1/flagged_doc_components.parquet",
    )
    p.add_argument(
        "--raw-data-glob",
        default="/home/foivos/data/glossapi_work/hf_release_publish/data/HPLT__ell_Grek_ge8_no_mt.*.parquet",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--components",
        nargs="*",
        default=None,
        help="Explicit component list. Defaults to top 5 dominant components by count.",
    )
    return p.parse_args()


def slugify(text: str) -> str:
    text = text.replace("::", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:120].strip("_") or "doc"


def extract_shard_key(source_doc_id: str) -> str:
    parts = source_doc_id.split("::")
    if len(parts) < 3:
        raise ValueError(f"Unexpected source_doc_id format: {source_doc_id}")
    return parts[1].replace(".jsonl.zst", "")


def choose_components(rows: list[dict], explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    counts = Counter(row["dominant_component"] for row in rows)
    return [name for name, _ in counts.most_common(5)]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.components_parquet)
    rows = table.to_pylist()
    components = choose_components(rows, args.components)

    rng = random.Random(args.seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["dominant_component"] in components:
            grouped[row["dominant_component"]].append(row)

    sampled: dict[str, list[dict]] = {}
    needed_ids: set[str] = set()
    for component in components:
        component_rows = grouped[component]
        if len(component_rows) < args.sample_size:
            chosen = list(component_rows)
        else:
            chosen = rng.sample(component_rows, args.sample_size)
        chosen = sorted(chosen, key=lambda r: r["rescored_greek_badness_score"], reverse=True)
        sampled[component] = chosen
        needed_ids.update(row["source_doc_id"] for row in chosen)

    shard_keys = sorted({extract_shard_key(source_doc_id) for source_doc_id in needed_ids})
    raw_files: list[str] = []
    for shard_key in shard_keys:
        raw_files.extend(
            sorted(glob(args.raw_data_glob.replace("*.parquet", f"{shard_key}.part-*.parquet")))
        )
    if not raw_files:
        raise SystemExit(
            f"No raw HPLT parquet files matched selected shard keys {', '.join(shard_keys)}"
        )

    raw_dataset = ds.dataset(raw_files, format="parquet")
    filter_expr = ds.field("source_doc_id").isin(sorted(needed_ids))
    raw_table = raw_dataset.to_table(columns=["source_doc_id", "text"], filter=filter_expr)
    text_by_id = {
        row["source_doc_id"]: (row["text"] or "") for row in raw_table.to_pylist()
    }

    missing = sorted(needed_ids - set(text_by_id))
    if missing:
        raise SystemExit(f"Missing {len(missing)} sampled source_doc_id values from raw parquet")

    top_summary = {
        "sample_size_per_component": args.sample_size,
        "seed": args.seed,
        "components": components,
        "component_doc_counts_in_flagged_slice": {
            component: len(grouped[component]) for component in components
        },
        "sampled_docs_total": sum(len(v) for v in sampled.values()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(top_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme_lines = [
        "# HPLT Badness Driver Samples",
        "",
        f"Seed: `{args.seed}`",
        f"Sample size per component: `{args.sample_size}`",
        "",
        "Included dominant components:",
    ]
    for component in components:
        readme_lines.append(f"- `{component}`: {len(sampled[component])} docs")
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    for component in components:
        component_dir = out_dir / component
        component_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for idx, row in enumerate(sampled[component], start=1):
            source_doc_id = row["source_doc_id"]
            safe_id = slugify(source_doc_id)
            stem = f"{idx:02d}__{safe_id}"
            text_path = component_dir / f"{stem}.md"
            meta_path = component_dir / f"{stem}.json"
            text_path.write_text(text_by_id[source_doc_id].rstrip("\n") + "\n", encoding="utf-8")
            meta = {
                **row,
                "component_folder": component,
                "sample_index": idx,
                "text_file": text_path.name,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_rows.append(meta)
        with (component_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
