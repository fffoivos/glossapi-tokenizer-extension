#!/usr/bin/env python3
"""Stage a small StarCoderData subset with stable local doc ids.

The upstream StarCoderData shards expose an `id` column, but those ids are only
local to a shard. This script downloads a deterministic subset and rewrites it
with `doc_id = starcoderdata:<repo_path>:<row_id>`, so the forgetting holdout can
be excluded from replay with the same drop-doc-key mechanism used for web pools.
"""
import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


DEFAULT_PLAN = "python:8,javascript:6,java:4,go:3,rust:2,cpp:2,typescript:3"


def parse_plan(raw):
    plan = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        lang, count = item.split(":", 1)
        plan.append((lang.strip(), int(count)))
    if not plan:
        raise SystemExit("empty StarCoderData staging plan")
    return plan


def has_expected_schema(path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        cols = set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return False
    return {"content", "doc_id"}.issubset(cols)


def rewrite_shard(src, dst, relpath, batch_size):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    rows = chars = 0
    writer = None
    pf = pq.ParquetFile(src)
    cols = [c for c in ("content", "id") if c in pf.schema_arrow.names]
    if "content" not in cols:
        raise SystemExit(f"{relpath}: source shard has no content column")
    try:
        for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
            data = batch.to_pydict()
            content = data["content"]
            ids = data.get("id") or list(range(rows, rows + len(content)))
            out = pa.table(
                {
                    "content": content,
                    "id": [str(x) for x in ids],
                    "doc_id": [f"starcoderdata:{relpath}:{x}" for x in ids],
                    "source_file": [relpath] * len(content),
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(tmp, out.schema, compression="zstd")
            writer.write_table(out)
            rows += len(content)
            chars += sum(len(t) for t in content if isinstance(t, str))
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, dst)
    return rows, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--repo-id", default="bigcode/starcoderdata")
    ap.add_argument("--plan", default=os.environ.get("STARCODER_PLAN", DEFAULT_PLAN))
    ap.add_argument("--cache-dir")
    ap.add_argument("--batch-size", type=int, default=20_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    cache_dir = args.cache_dir or str(out_root.parent / ".hf_cache_starcoderdata_v2")
    files = [f for f in HfApi().list_repo_files(args.repo_id, repo_type="dataset") if f.endswith(".parquet")]
    staged = []
    for lang, count in parse_plan(args.plan):
        lang_files = sorted(f for f in files if f.startswith(f"{lang}/"))
        if len(lang_files) < count:
            raise SystemExit(f"{lang}: requested {count} shards, only found {len(lang_files)}")
        for relpath in lang_files[:count]:
            dst = out_root / relpath
            if not args.force and has_expected_schema(dst):
                pf = pq.ParquetFile(dst)
                rows = pf.metadata.num_rows
                chars = None
                print(f"[SKIP] {relpath}: already staged at {dst} ({rows:,} rows)", flush=True)
            else:
                src = hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename=relpath,
                    cache_dir=cache_dir,
                )
                rows, chars = rewrite_shard(src, dst, relpath, args.batch_size)
                print(f"[OK] {relpath}: {rows:,} rows -> {dst}", flush=True)
            staged.append({"repo_path": relpath, "local_path": str(dst), "rows": rows, "chars": chars})

    manifest = {
        "repo_id": args.repo_id,
        "plan": args.plan,
        "doc_id": "starcoderdata:<repo_path>:<upstream_id>",
        "text_column": "content",
        "id_column": "doc_id",
        "shards": staged,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"DONE: staged {len(staged)} StarCoderData shards -> {out_root}", flush=True)
    print(f"DONE: manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
