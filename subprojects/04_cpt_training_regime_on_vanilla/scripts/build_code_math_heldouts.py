#!/usr/bin/env python3
"""Build code/math heldout JSONLs disjoint from the final 04 training mix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_SOURCES = {
    "code": "code_codeparrot_clean",
    "math": "math_finemath",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe-json", required=True)
    p.add_argument("--training-jsonl", required=True)
    p.add_argument("--mix-builder-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--code-output-jsonl", default="")
    p.add_argument("--math-output-jsonl", default="")
    p.add_argument("--manifest-json", required=True)
    p.add_argument("--code-quota", type=int, default=200)
    p.add_argument("--math-quota", type=int, default=200)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--allow-partial", action="store_true")
    return p.parse_args()


def load_training_ids(path: Path, source_names: set[str]) -> dict[str, set[str]]:
    ids = {source: set() for source in source_names}
    counts = {source: 0 for source in source_names}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            source = str(row.get("source") or "")
            if source not in ids:
                continue
            counts[source] += 1
            doc_id = row.get("doc_id")
            if doc_id:
                ids[source].add(str(doc_id))
    print("training source row counts:", counts, flush=True)
    print("training source doc-id counts:", {k: len(v) for k, v in ids.items()}, flush=True)
    return ids


def load_source_specs(recipe_path: Path, wanted: dict[str, str]) -> dict[str, dict[str, Any]]:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    by_name = {str(spec["name"]): spec for spec in recipe["sources"]}
    missing = [name for name in wanted.values() if name not in by_name]
    if missing:
        raise SystemExit(f"recipe is missing expected sources: {missing}")
    return {kind: by_name[name] for kind, name in wanted.items()}


def select_rows(
    *,
    kind: str,
    source_spec: dict[str, Any],
    train_ids: set[str],
    quota: int,
    min_chars: int,
    build_source_stream,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    counters = {
        "seen": 0,
        "excluded_training_doc_id": 0,
        "excluded_short": 0,
        "selected": 0,
    }
    for row in build_source_stream(source_spec, None):
        counters["seen"] += 1
        doc_id = str(row.get("doc_id") or "")
        if doc_id in train_ids:
            counters["excluded_training_doc_id"] += 1
            continue
        text = row.get("text")
        if not isinstance(text, str) or len(text) < min_chars:
            counters["excluded_short"] += 1
            continue
        selected.append(
            {
                "text": text,
                "source": str(row.get("source") or source_spec["name"]),
                "doc_id": doc_id,
                "heldout_kind": kind,
                "selection_rule": "first stream rows not present in final training JSONL",
            }
        )
        counters["selected"] += 1
        if len(selected) >= quota:
            break
    return selected, counters


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    recipe_json = Path(args.recipe_json)
    training_jsonl = Path(args.training_jsonl)
    mix_builder_dir = Path(args.mix_builder_dir)
    output_dir = Path(args.output_dir)
    manifest_json = Path(args.manifest_json)

    sys.path.insert(0, str(mix_builder_dir))
    try:
        from mix_builder import _build_source_stream  # type: ignore
    except Exception as exc:  # pragma: no cover - operational guard
        raise SystemExit(f"could not import mix_builder from {mix_builder_dir}: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    code_output = Path(args.code_output_jsonl) if args.code_output_jsonl else output_dir / "cpt_code_heldout_200_20260528.jsonl"
    math_output = Path(args.math_output_jsonl) if args.math_output_jsonl else output_dir / "cpt_math_heldout_200_20260528.jsonl"

    specs = load_source_specs(recipe_json, DEFAULT_SOURCES)
    train_ids = load_training_ids(training_jsonl, set(DEFAULT_SOURCES.values()))

    outputs: dict[str, str] = {}
    counters: dict[str, dict[str, int]] = {}
    selected_counts: dict[str, int] = {}

    for kind, quota, path in (
        ("code", args.code_quota, code_output),
        ("math", args.math_quota, math_output),
    ):
        rows, stats = select_rows(
            kind=kind,
            source_spec=specs[kind],
            train_ids=train_ids[DEFAULT_SOURCES[kind]],
            quota=quota,
            min_chars=args.min_chars,
            build_source_stream=_build_source_stream,
        )
        if len(rows) < quota and not args.allow_partial:
            raise SystemExit(f"{kind}: selected {len(rows)} rows, wanted {quota}; stats={stats}")
        write_jsonl(path, rows)
        outputs[kind] = str(path)
        counters[kind] = stats
        selected_counts[kind] = len(rows)
        print(f"{kind}: wrote {len(rows)} rows -> {path}", flush=True)

    manifest = {
        "recipe_json": str(recipe_json),
        "training_jsonl": str(training_jsonl),
        "source_names": DEFAULT_SOURCES,
        "outputs": outputs,
        "quotas": {"code": args.code_quota, "math": args.math_quota},
        "min_chars": args.min_chars,
        "selected_counts": selected_counts,
        "counters": counters,
        "disjointness_rule": "exact doc_id exclusion against the final training JSONL for each source",
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
