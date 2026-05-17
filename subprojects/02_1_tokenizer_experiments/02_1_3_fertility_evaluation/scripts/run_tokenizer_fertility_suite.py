#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path


def import_fertility(repo_root: Path):
    path = repo_root / "tokenizer_analysis" / "run_wave4_fertility_eval.py"
    spec = importlib.util.spec_from_file_location("fertility_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_name_path(item: str) -> tuple[str, Path]:
    name, sep, path = item.partition("=")
    if not sep:
        raise SystemExit(f"expected name=path, got {item!r}")
    return name, Path(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/foivos/Projects/glossapi-tokenizer-extension", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", action="append", required=True, help="name=/path/to/tokenizer_dir")
    parser.add_argument("--slice", action="append", required=True, help="name=/path/to/parquet")
    parser.add_argument("--latest-glossapi-limit", type=int, default=6)
    parser.add_argument("--max-docs-per-slice", type=int, default=300)
    parser.add_argument("--max-chars-per-slice", type=int, default=3_000_000)
    parser.add_argument("--max-doc-chars", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--word-batch-size", type=int, default=1024)
    parser.add_argument("--parquet-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()

    fert = import_fertility(args.repo_root)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "hf_cache"

    tokenizer_specs = []
    for item in args.tokenizer:
        name, path = parse_name_path(item)
        tok_json = path / "tokenizer.json"
        vocab_size = len(fert.load_json(tok_json)["model"]["vocab"])
        base_vocab_size = 131072 if vocab_size >= 131072 else None
        tokenizer_specs.append({
            "name": name,
            "family": "apertus_continuous" if base_vocab_size else "other",
            "path": str(path),
            "vocab_size": vocab_size,
            "base_vocab_size": base_vocab_size,
            "added_units": (vocab_size - 131072) if base_vocab_size else None,
            "apertus_compatible": bool(base_vocab_size),
            "tokenizer_json_sha256": fert.sha256_path(tok_json),
            "tokenizer_json_size_bytes": tok_json.stat().st_size,
        })

    specs = []
    for item in args.slice:
        name, path = parse_name_path(item)
        specs.append(fert.SliceSpec(name=name, kind="heldout_or_experiment_split", paths=(path,)))
    latest_specs, latest_repos = fert.download_latest_glossapi_slices(cache_dir=cache_dir, limit=args.latest_glossapi_limit)
    specs.extend(latest_specs)

    write_json(out_dir / "tokenizers.json", tokenizer_specs)
    write_json(out_dir / "latest_glossapi_datasets.json", latest_repos)
    write_json(out_dir / "run_config.json", {
        "seed": args.seed,
        "max_docs_per_slice": args.max_docs_per_slice,
        "max_chars_per_slice": args.max_chars_per_slice,
        "max_doc_chars": args.max_doc_chars,
        "latest_glossapi_limit": args.latest_glossapi_limit,
    })

    sample_results = []
    sample_manifest = []
    for spec in specs:
        sample = fert.sample_slice(
            spec,
            out_dir=out_dir,
            seed=args.seed,
            max_docs=args.max_docs_per_slice,
            max_chars=args.max_chars_per_slice,
            max_doc_chars=args.max_doc_chars,
            batch_size=args.parquet_batch_size,
        )
        sample_results.append(sample)
        sample_manifest.append({k: v for k, v in sample.items() if k != "items"})
        write_json(out_dir / "sample_manifests" / f"{spec.name}.json", {k: v for k, v in sample.items() if k != "items"})
    write_json(out_dir / "sample_manifest.json", sample_manifest)

    rows = []
    for tokenizer_spec in tokenizer_specs:
        for sample in sample_results:
            row = fert.evaluate_tokenizer_on_slice(
                tokenizer_spec,
                sample,
                batch_size=args.batch_size,
                word_batch_size=args.word_batch_size,
            )
            rows.append(row)
            write_json(out_dir / "progress_latest.json", {"completed_rows": len(rows), "last": row})

    write_json(out_dir / "metrics_by_slice.json", rows)
    with (out_dir / "metrics_by_slice.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Tokenizer Fertility Suite",
        "",
        "| slice | tokenizer | chars/token | Greek word fertility | single-word share | added token rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['slice']}` | `{row['tokenizer']}` | {float(row.get('chars_per_token') or 0):.4f} | "
            f"{float(row.get('greek_word_space_fertility') or 0):.4f} | "
            f"{float(row.get('single_token_greek_word_share') or 0):.4f} | "
            f"{float(row.get('added_token_rate') or 0):.4f} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "summary.json", {
        "output_dir": str(out_dir),
        "metric_rows": len(rows),
        "samples": sample_manifest,
        "tokenizers": tokenizer_specs,
        "latest_glossapi_datasets": latest_repos,
    })
    print(json.dumps({"output_dir": str(out_dir), "metric_rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    raise SystemExit(main())
