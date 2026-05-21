"""Streaming mix builder for the CPT bakeoff (and the future anneal phase).

Reads a JSON recipe describing per-source HuggingFace datasets + weights +
optional filters + optional drop-doc-keys parquets, interleaves them against
the requested token-budget shares, and writes a JSON-lines file. The output is
suitable as input to Megatron-LM's `tools/preprocess_data.py` for binary
tokenization (which is a separate, downstream step).

Determinism: with the same `seed` and the same recipe + dataset revisions,
the output text stream is reproducible. All bakeoff arms read the same JSONL
document order, then preprocessing intentionally produces two tokenized
Megatron datasets: base-tokenizer IDs for Vanilla, extended-tokenizer IDs for
ReTok/Centroid. Token IDs are therefore not identical across the base-vs-
extended families; the controlled invariant is the underlying text stream.

CLI:
    python3 mix_builder.py \\
        --recipe recipes/bulk.json \\
        --target-tokens 7000000000 \\
        --tokenizer /path/to/extended-tokenizer \\
        --output bulk_mix.jsonl \\
        --seed 20260520

For Slurm array builds, pass `--source-shard-index i --source-shard-count n`.
This partitions each source after filtering by eligible row index, so shard
files are disjoint per source instead of repeating the same prefixes.

Token counting is approximate (fast HF tokenizer encode without padding);
the budget is a stop condition, not a hard cap. Expect ±2% slack.

Hardware: streams from HF; needs internet to the HF mirror plus enough
disk for the JSONL (estimate ~chars-per-token-target × 4 bytes per char
for UTF-8 → ~28 GB JSONL for a 7 B-token target at chars/token ≈ 4).
Use Clariden `normal` partition (xfer is in maintenance till 2026-06-11);
allocate ~64 GB RAM and one CPU socket.

Compute justification (per [[feedback_compute_sweet_spot_justify]]):
  * Parallelism: sources stream sequentially within their iterators; multiple
    sources are interleaved by a deterministic token-fair scheduler, not by
    parallel readers. Per-source IO is the dominant cost for parquet sources
    (local_parquet) and is read in pyarrow's internal-threadpool batch_size,
    which uses all cores available. For HF-hosted sources (FineWeb-Edu etc.)
    IO is throttled by the HF mirror.
  * Tokenizer hot path: `tokenizer.encode(text)` is called once per row
    for budget tracking. The Apertus fast (rust) tokenizer is GIL-free
    but a single encode() call is single-threaded by design (parallelism
    is at the batch level via `encode_batch`, not within one encode).
    The row-at-a-time loop is intentionally cache-friendly for the
    streaming-mix pattern; for a 7 B-token budget at ~10 MB/s
    single-thread encode rate, walltime ≈ 45-50 min for the encode step,
    well under our 6 h sbatch ceiling.
  * Saturation: writes JSON-lines sequentially (single Python writer
    holds the output file). The bottleneck shifts between tokenizer
    encode (CPU-bound) and HF parquet load (IO-bound) depending on
    source. We do not parallelize the writer because the bakeoff requires
    determinism: identical interleave order across the three arms.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def _approx_token_count(text: str, tokenizer) -> int:
    """Fast token count via the tokenizer's encode. No padding."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _load_drop_keys(spec: str) -> set[str]:
    """Load a doc-key drop list from a HF spec like `repo::path/inside.parquet`,
    or from a local parquet path (env vars like `${APERTUS_DROP}` expanded).
    Returns the set of doc-key strings.

    NOTE: with the runbook-built `${SELECTED}` pool (Blocker 3 fix), the
    Apertus-drop is already applied upstream and this drop-keys hook is
    redundant for nanochat sources. It's still useful for ad-hoc HF sources.
    """
    import os
    import pyarrow.parquet as pq
    if "::" in spec:
        repo, internal = spec.split("::", 1)
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=repo, repo_type="dataset", filename=internal)
    else:
        local = os.path.expandvars(spec)
        if "$" in local:
            raise SystemExit(f"drop_doc_keys_parquet still contains unexpanded '$': {local!r}")
    tbl = pq.read_table(local)
    # Try common column names for doc-key
    for col in ("doc_key", "hf_pool_doc_key", "doc_id"):
        if col in tbl.column_names:
            return set(tbl.column(col).to_pylist())
    raise SystemExit(f"drop-keys parquet at {spec} has no doc_key/hf_pool_doc_key/doc_id column; cols: {tbl.column_names}")


def _build_source_stream(
    spec: dict[str, Any],
    drop_keys: set[str] | None,
    source_shard_index: int = 0,
    source_shard_count: int = 1,
) -> Iterable[dict]:
    """Yield {text, doc_id, source, ...} from one source per its spec.

    Two source modes:
    - HF dataset (`id`): streaming `load_dataset(id, config, split, streaming=True)`
    - Local parquet (`local_parquet`): a path on disk (or a glob). Supports
      `${VAR}` env-var expansion so recipes can reference `${SELECTED}` for the
      runbook-built post-Apertus-drop + post-internal-dedup pool. Reviewer
      round-2 Blocker 3: the runbook (`03_2/CPT_DATASET_BUILD_RUNBOOK.md`) is
      the canonical CPT-source path; this is the in-process consumer.

    Applies filter_field/filter_values, filter_values_regex, filter_min,
    drop_doc_keys filtering. Returns an iterator over normalized rows.
    """
    import os

    name = spec["name"]
    split = spec.get("split", "train")
    streaming = bool(spec.get("streaming", True))
    text_col = spec.get("text_column", "text")
    filter_field = spec.get("filter_field")
    filter_values = set(spec.get("filter_values") or [])
    filter_regex_pat = spec.get("filter_values_regex")
    filter_min = spec.get("filter_min")
    doc_key_field = spec.get("doc_key_field", "doc_key")

    if filter_regex_pat:
        import re
        regex = re.compile(filter_regex_pat)

    n_seen = 0
    n_yielded = 0

    def emit_or_none(row: dict[str, Any]) -> dict[str, str] | None:
        nonlocal n_seen, n_yielded
        n_seen += 1
        if filter_field:
            v = row.get(filter_field)
            if v is None:
                return None
            if filter_values and v not in filter_values:
                return None
            if filter_regex_pat and not regex.search(str(v)):
                return None
            if filter_min is not None:
                try:
                    if float(v) < float(filter_min):
                        return None
                except (TypeError, ValueError):
                    return None
        if drop_keys is not None:
            dk = row.get(doc_key_field)
            if dk is not None and dk in drop_keys:
                return None
        text = row.get(text_col)
        if not text or not isinstance(text, str):
            return None
        candidate_index = n_yielded
        n_yielded += 1
        if source_shard_count > 1 and candidate_index % source_shard_count != source_shard_index:
            return None
        doc_id = (
            row.get("doc_key")
            or row.get("doc_id")
            or row.get("id")
            or f"{name}_{candidate_index}"
        )
        return {"text": text, "source": name, "doc_id": str(doc_id)}

    local_parquet = spec.get("local_parquet")
    if local_parquet:
        # Expand env vars (e.g., ${SELECTED}) before opening
        local_path = os.path.expandvars(local_parquet)
        if "$" in local_path:
            raise SystemExit(
                f"{name}: local_parquet path still contains '$' after expansion: {local_path!r}. "
                f"Set the env var before running mix_builder."
            )
        import glob
        import pyarrow.parquet as pq

        paths = sorted(glob.glob(local_path))
        if not paths:
            raise SystemExit(f"{name}: local_parquet matched no files: {local_path}")
        for path in paths:
            pf = pq.ParquetFile(path)
            available = set(pf.schema_arrow.names)
            requested = [text_col, doc_key_field, "doc_key", "doc_id", "id"]
            if filter_field:
                requested.append(filter_field)
            columns = []
            for col in requested:
                if col in available and col not in columns:
                    columns.append(col)
            if text_col not in columns:
                raise SystemExit(f"{name}: local parquet {path} has no text column {text_col!r}")
            for batch in pf.iter_batches(batch_size=5_000, columns=columns):
                data = batch.to_pydict()
                for i in range(batch.num_rows):
                    out = emit_or_none({col: data[col][i] for col in columns})
                    if out is not None:
                        yield out
        return
    else:
        from datasets import load_dataset

        ds_id = spec["id"]
        config = spec.get("config")
        try:
            ds = load_dataset(ds_id, config, split=split, streaming=streaming)
        except Exception as exc:
            # Fallback chain (e.g., FW2-HQ doesn't have Persian → use FW2 standard)
            fb_id = spec.get("fallback_id")
            fb_cfg = spec.get("fallback_config")
            if fb_id:
                print(f"  [WARN] {name}: primary {ds_id}/{config} failed ({exc}); trying fallback {fb_id}/{fb_cfg}", file=sys.stderr)
                ds = load_dataset(fb_id, fb_cfg, split=split, streaming=streaming)
            else:
                raise

    for row in ds:
        out = emit_or_none(row)
        if out is not None:
            yield out


def _build_all_sources(
    recipe: dict,
    source_shard_index: int = 0,
    source_shard_count: int = 1,
) -> tuple[list[Iterable[dict]], list[float], list[str], list[str]]:
    """Build streams + normalized probabilities + names + bucket labels."""
    sources: list[Iterable[dict]] = []
    weights: list[float] = []
    names: list[str] = []
    buckets: list[str] = []

    drop_keys_cache: dict[str, set[str]] = {}
    for spec in recipe["sources"]:
        dks: set[str] | None = None
        if "drop_doc_keys_parquet" in spec:
            key = spec["drop_doc_keys_parquet"]
            if key not in drop_keys_cache:
                print(f"  loading drop-keys from {key} ...", flush=True)
                drop_keys_cache[key] = _load_drop_keys(key)
                print(f"    loaded {len(drop_keys_cache[key]):,} drop keys", flush=True)
            dks = drop_keys_cache[key]
        sources.append(_build_source_stream(spec, dks, source_shard_index, source_shard_count))
        weights.append(float(spec["weight"]))
        names.append(spec["name"])
        buckets.append(str(spec.get("bucket") or "default"))

    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    return sources, weights, names, buckets


def _choose_token_fair_source(
    active: list[int],
    weights: list[float],
    per_source_tokens: dict[str, int],
    names: list[str],
    source_token_targets: list[float],
    rng: random.Random,
) -> int:
    """Choose the active source furthest behind its target token budget.

    A probability-only row sampler is not enough here: selected nanochat rows
    vary from tiny snippets to multi-million-token academic documents. This
    scheduler uses observed token totals to keep source shares close to the
    recipe's token-budget weights. Overshoot is bounded by the largest emitted
    document for that source.
    """
    under_quota = [i for i in active if per_source_tokens[names[i]] < source_token_targets[i]]
    candidates = under_quota or active
    progress = {
        i: per_source_tokens[names[i]] / max(weights[i], 1e-12)
        for i in candidates
    }
    min_progress = min(progress.values())
    tied = [i for i in candidates if progress[i] == min_progress]
    return rng.choice(tied)


def _choose_bucket_token_fair_source(
    active: list[int],
    weights: list[float],
    per_source_tokens: dict[str, int],
    names: list[str],
    buckets: list[str],
    bucket_weights: dict[str, float],
    bucket_source_weight_sums: dict[str, float],
    per_bucket_tokens: dict[str, int],
    bucket_token_targets: dict[str, float],
    source_token_targets: list[float],
    rng: random.Random,
) -> int:
    """Choose a source while preserving top-level bucket token shares.

    Source exhaustion should normally redistribute within the same bucket first:
    if `greek_literary` exhausts, the remaining Greek sources should absorb the
    missing Greek budget before replay/code/math are allowed to grow.
    """
    active_buckets = sorted({buckets[i] for i in active if bucket_weights.get(buckets[i], 0.0) > 0.0})
    if not active_buckets:
        return _choose_token_fair_source(active, weights, per_source_tokens, names, source_token_targets, rng)

    under_quota_buckets = [
        b for b in active_buckets
        if per_bucket_tokens.get(b, 0) < bucket_token_targets.get(b, 0.0)
    ]
    bucket_candidates = under_quota_buckets or active_buckets
    bucket_progress = {
        b: per_bucket_tokens.get(b, 0) / max(bucket_weights[b], 1e-12)
        for b in bucket_candidates
    }
    min_bucket_progress = min(bucket_progress.values())
    tied_buckets = [b for b in bucket_candidates if bucket_progress[b] == min_bucket_progress]
    chosen_bucket = rng.choice(tied_buckets)

    source_candidates = [i for i in active if buckets[i] == chosen_bucket]
    under_quota_sources = [
        i for i in source_candidates
        if per_source_tokens[names[i]] < source_token_targets[i]
    ]
    candidates = under_quota_sources or source_candidates
    source_progress = {
        i: per_source_tokens[names[i]] / max(weights[i] / bucket_source_weight_sums[chosen_bucket], 1e-12)
        for i in candidates
    }
    min_source_progress = min(source_progress.values())
    tied_sources = [i for i in candidates if source_progress[i] == min_source_progress]
    return rng.choice(tied_sources)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--target-tokens", type=int, required=True,
                    help="stop after approximately this many tokens written")
    ap.add_argument("--tokenizer", type=Path, required=True,
                    help="HF tokenizer path (used for token counting only)")
    ap.add_argument("--output", type=Path, required=True, help="output JSONL path")
    ap.add_argument("--seed", type=int, default=None,
                    help="overrides recipe seed if given")
    ap.add_argument("--progress-every-tokens", type=int, default=50_000_000,
                    help="emit progress line every N tokens")
    ap.add_argument("--source-shard-index", type=int, default=0,
                    help="emit only eligible source rows where row_index %% source_shard_count equals this index")
    ap.add_argument("--source-shard-count", type=int, default=1,
                    help="split each source stream into this many disjoint row shards")
    args = ap.parse_args()
    if args.source_shard_count < 1:
        raise SystemExit("--source-shard-count must be >= 1")
    if args.source_shard_index < 0 or args.source_shard_index >= args.source_shard_count:
        raise SystemExit("--source-shard-index must be in [0, --source-shard-count)")

    recipe = json.loads(args.recipe.read_text())
    seed = args.seed if args.seed is not None else recipe.get("seed", 20_260_520)

    print(f"=== mix_builder ===")
    print(f"recipe: {args.recipe} (name={recipe.get('name')}, version={recipe.get('version')})")
    print(f"target_tokens: {args.target_tokens:,}")
    print(f"output: {args.output}")
    print(f"seed: {seed}")
    print(f"source_shard: {args.source_shard_index}/{args.source_shard_count}")

    from transformers import AutoTokenizer  # type: ignore

    print(f"loading tokenizer for token counting: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    print(f"  vocab_size: {tokenizer.vocab_size:,}")

    print(f"building source streams ...")
    sources, weights, names, buckets = _build_all_sources(
        recipe,
        source_shard_index=args.source_shard_index,
        source_shard_count=args.source_shard_count,
    )
    print(f"  {len(sources)} sources; per-source weights:")
    for name, bucket, w in zip(names, buckets, weights):
        print(f"    {w:.4f}  {bucket:<8}  {name}")

    source_weight_by_bucket: dict[str, float] = {}
    for bucket, weight in zip(buckets, weights):
        source_weight_by_bucket[bucket] = source_weight_by_bucket.get(bucket, 0.0) + weight
    raw_bucket_weights = {
        str(bucket): float(weight)
        for bucket, weight in (recipe.get("buckets") or source_weight_by_bucket).items()
    }
    for bucket in source_weight_by_bucket:
        raw_bucket_weights.setdefault(bucket, source_weight_by_bucket[bucket])
    bucket_weight_total = sum(w for w in raw_bucket_weights.values() if w > 0.0)
    if bucket_weight_total <= 0:
        raise SystemExit("bucket weights must sum to a positive value")
    bucket_weights = {
        bucket: weight / bucket_weight_total
        for bucket, weight in raw_bucket_weights.items()
        if weight > 0.0
    }
    bucket_source_weight_sums = {
        bucket: sum(weights[i] for i, source_bucket in enumerate(buckets) if source_bucket == bucket)
        for bucket in bucket_weights
    }
    for bucket, source_sum in bucket_source_weight_sums.items():
        if source_sum <= 0.0:
            raise SystemExit(f"bucket {bucket!r} has no positive-weight sources")
    print("  bucket target weights:")
    for bucket in sorted(bucket_weights):
        print(f"    {bucket_weights[bucket]:.4f}  {bucket}")

    print(f"interleaving with deterministic bucket-preserving token-fair scheduler ...")
    iterators = [iter(src) for src in sources]
    active = [i for i, weight in enumerate(weights) if weight > 0.0]
    rng = random.Random(seed)
    source_token_targets = [args.target_tokens * w for w in weights]
    bucket_token_targets = {
        bucket: args.target_tokens * weight
        for bucket, weight in bucket_weights.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = 0
    n_rows = 0
    last_progress = 0
    t0 = time.time()
    per_source_rows = {n: 0 for n in names}
    per_source_tokens = {n: 0 for n in names}
    per_bucket_rows = {b: 0 for b in bucket_weights}
    per_bucket_tokens = {b: 0 for b in bucket_weights}

    with args.output.open("w", encoding="utf-8") as fp:
        while active and n_tokens < args.target_tokens:
            idx = _choose_bucket_token_fair_source(
                active=active,
                weights=weights,
                per_source_tokens=per_source_tokens,
                names=names,
                buckets=buckets,
                bucket_weights=bucket_weights,
                bucket_source_weight_sums=bucket_source_weight_sums,
                per_bucket_tokens=per_bucket_tokens,
                bucket_token_targets=bucket_token_targets,
                source_token_targets=source_token_targets,
                rng=rng,
            )
            try:
                row = next(iterators[idx])
            except StopIteration:
                active.remove(idx)
                print(
                    f"  [WARN] source exhausted before target budget: {names[idx]} "
                    f"({per_source_tokens[names[idx]]:,}/{source_token_targets[idx]:,.0f} tokens)",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            text = row.get("text") or ""
            if not text:
                continue
            n_tokens_row = _approx_token_count(text, tokenizer)
            json.dump({
                "text": text,
                "source": row.get("source", "?"),
                "doc_id": row.get("doc_id", f"row_{n_rows}"),
            }, fp, ensure_ascii=False)
            fp.write("\n")
            n_tokens += n_tokens_row
            n_rows += 1
            src = row.get("source", "?")
            if src in per_source_rows:
                per_source_rows[src] += 1
                per_source_tokens[src] += n_tokens_row
            bucket = buckets[idx]
            if bucket in per_bucket_rows:
                per_bucket_rows[bucket] += 1
                per_bucket_tokens[bucket] += n_tokens_row

            if n_tokens - last_progress >= args.progress_every_tokens:
                elapsed = time.time() - t0
                rate = n_tokens / elapsed if elapsed > 0 else 0.0
                pct = 100 * n_tokens / args.target_tokens
                eta = (args.target_tokens - n_tokens) / rate if rate > 0 else 0.0
                print(f"  {n_tokens:>12,} tok  ({pct:5.1f}%)  rows={n_rows:>10,}  rate={rate/1000:6.1f}k tok/s  ETA={eta/60:5.1f} min", flush=True)
                last_progress = n_tokens

    elapsed = time.time() - t0
    print(f"\n=== summary ===")
    print(f"total tokens written: {n_tokens:,} (target {args.target_tokens:,})")
    print(f"total rows: {n_rows:,}")
    print(f"wall: {elapsed/60:.1f} min")
    print(f"output: {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)")
    print(f"\nper-source breakdown:")
    print(f"  {'source':<35} {'rows':>10} {'tokens':>14} {'effective_weight':>18}")
    for name in names:
        w = per_source_tokens[name] / max(n_tokens, 1)
        print(f"  {name:<35} {per_source_rows[name]:>10,} {per_source_tokens[name]:>14,} {w:>18.4f}")
    print(f"\nper-bucket breakdown:")
    print(f"  {'bucket':<12} {'rows':>10} {'tokens':>14} {'target_weight':>14} {'effective_weight':>18}")
    for bucket in sorted(bucket_weights):
        w = per_bucket_tokens[bucket] / max(n_tokens, 1)
        print(f"  {bucket:<12} {per_bucket_rows[bucket]:>10,} {per_bucket_tokens[bucket]:>14,} {bucket_weights[bucket]:>14.4f} {w:>18.4f}")

    # Write a sidecar manifest
    manifest = {
        "recipe": str(args.recipe),
        "recipe_name": recipe.get("name"),
        "recipe_version": recipe.get("version"),
        "seed": seed,
        "target_tokens": args.target_tokens,
        "source_shard_index": args.source_shard_index,
        "source_shard_count": args.source_shard_count,
        "actual_tokens": n_tokens,
        "actual_rows": n_rows,
        "wall_seconds": elapsed,
        "output": str(args.output),
        "scheduler": "bucket_preserving_token_fair_min_tokens_over_weight",
        "per_bucket": {
            bucket: {
                "rows": per_bucket_rows[bucket],
                "tokens": per_bucket_tokens[bucket],
                "effective_weight": per_bucket_tokens[bucket] / max(n_tokens, 1),
                "target_weight": bucket_weights[bucket],
                "target_tokens": bucket_token_targets[bucket],
                "token_delta_vs_target": per_bucket_tokens[bucket] - bucket_token_targets[bucket],
            } for bucket in sorted(bucket_weights)
        },
        "per_source": {
            name: {
                "rows": per_source_rows[name],
                "tokens": per_source_tokens[name],
                "effective_weight": per_source_tokens[name] / max(n_tokens, 1),
                "target_weight": weights[i],
                "target_tokens": source_token_targets[i],
                "token_delta_vs_target": per_source_tokens[name] - source_token_targets[i],
            } for i, name in enumerate(names)
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # On Clariden, a successful run can abort during teardown of native dataset
    # readers after all files are written. Preserve real Python exceptions, but
    # skip native finalizers on the clean path so Slurm sees the successful exit.
    os._exit(code)
