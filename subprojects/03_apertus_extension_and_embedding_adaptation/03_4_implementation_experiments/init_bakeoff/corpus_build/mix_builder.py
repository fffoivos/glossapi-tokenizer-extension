"""Streaming mix builder for the CPT bakeoff (and the future anneal phase).

Reads a JSON recipe describing per-source HuggingFace datasets + weights +
optional filters + optional drop-doc-keys parquets, interleaves them with
the requested probabilities, and writes a JSON-lines file. The output is
suitable as input to Megatron-LM's `tools/preprocess_data.py` for binary
tokenization (which is a separate, downstream step).

Determinism: with the same `seed` and the same recipe + dataset revisions,
the output stream is reproducible. The three bakeoff arms read the same
JSONL so token streams are identical and the only differential is the
init applied to the model.

CLI:
    python3 mix_builder.py \\
        --recipe recipes/bulk.json \\
        --target-tokens 7000000000 \\
        --tokenizer /path/to/extended-tokenizer \\
        --output bulk_mix.jsonl \\
        --seed 20260520

Token counting is approximate (fast HF tokenizer encode without padding);
the budget is a stop condition, not a hard cap. Expect ±2% slack.

Hardware: streams from HF; needs internet to the HF mirror plus enough
disk for the JSONL (estimate ~chars-per-token-target × 4 bytes per char
for UTF-8 → ~28 GB JSONL for a 7 B-token target at chars/token ≈ 4).
Use Clariden `xfer` for the actual run.
"""
from __future__ import annotations
import argparse
import json
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
    or from a local parquet path. Returns the set of doc-key strings.
    """
    import pyarrow.parquet as pq
    if "::" in spec:
        repo, internal = spec.split("::", 1)
        # Use huggingface_hub to download the specific file
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=repo, repo_type="dataset", filename=internal)
    else:
        local = spec
    tbl = pq.read_table(local)
    # Try common column names for doc-key
    for col in ("doc_key", "hf_pool_doc_key", "doc_id"):
        if col in tbl.column_names:
            return set(tbl.column(col).to_pylist())
    raise SystemExit(f"drop-keys parquet at {spec} has no doc_key/hf_pool_doc_key/doc_id column; cols: {tbl.column_names}")


def _build_source_stream(spec: dict[str, Any], drop_keys: set[str] | None) -> Iterable[dict]:
    """Yield {text, doc_id, source, ...} from one source per its spec.

    Applies filter_field/filter_values, filter_values_regex, filter_min,
    drop_doc_keys filtering. Returns an iterator over normalized rows.
    """
    from datasets import load_dataset

    ds_id = spec["id"]
    config = spec.get("config")
    split = spec.get("split", "train")
    streaming = bool(spec.get("streaming", True))
    text_col = spec.get("text_column", "text")
    name = spec["name"]

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
    for row in ds:
        n_seen += 1
        # Apply filter
        if filter_field:
            v = row.get(filter_field)
            if v is None:
                continue
            if filter_values and v not in filter_values:
                continue
            if filter_regex_pat and not regex.search(str(v)):
                continue
            if filter_min is not None:
                try:
                    if float(v) < float(filter_min):
                        continue
                except (TypeError, ValueError):
                    continue
        # Apply drop list
        if drop_keys is not None:
            dk = row.get(doc_key_field)
            if dk is not None and dk in drop_keys:
                continue
        # Normalize output
        text = row.get(text_col)
        if not text or not isinstance(text, str):
            continue
        doc_id = (
            row.get("doc_key")
            or row.get("doc_id")
            or row.get("id")
            or f"{name}_{n_yielded}"
        )
        yield {
            "text": text,
            "source": name,
            "doc_id": str(doc_id),
        }
        n_yielded += 1


def _build_all_sources(recipe: dict) -> tuple[list[Iterable[dict]], list[float], list[str]]:
    """Build streams + normalized probabilities + names for all sources in the recipe."""
    sources: list[Iterable[dict]] = []
    weights: list[float] = []
    names: list[str] = []

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
        sources.append(_build_source_stream(spec, dks))
        weights.append(float(spec["weight"]))
        names.append(spec["name"])

    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    return sources, weights, names


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
    args = ap.parse_args()

    recipe = json.loads(args.recipe.read_text())
    seed = args.seed if args.seed is not None else recipe.get("seed", 20_260_520)

    print(f"=== mix_builder ===")
    print(f"recipe: {args.recipe} (name={recipe.get('name')}, version={recipe.get('version')})")
    print(f"target_tokens: {args.target_tokens:,}")
    print(f"output: {args.output}")
    print(f"seed: {seed}")

    from datasets import interleave_datasets
    from transformers import AutoTokenizer  # type: ignore

    print(f"loading tokenizer for token counting: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    print(f"  vocab_size: {tokenizer.vocab_size:,}")

    print(f"building source streams ...")
    sources, weights, names = _build_all_sources(recipe)
    print(f"  {len(sources)} sources; per-source weights:")
    for name, w in zip(names, weights):
        print(f"    {w:.4f}  {name}")

    print(f"interleaving with stopping_strategy='all_exhausted' ...")
    mixed = interleave_datasets(
        list(sources),
        probabilities=weights,
        stopping_strategy="all_exhausted",
        seed=seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = 0
    n_rows = 0
    last_progress = 0
    t0 = time.time()
    per_source_rows = {n: 0 for n in names}
    per_source_tokens = {n: 0 for n in names}

    with args.output.open("w", encoding="utf-8") as fp:
        for row in mixed:
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

            if n_tokens - last_progress >= args.progress_every_tokens:
                elapsed = time.time() - t0
                rate = n_tokens / elapsed if elapsed > 0 else 0.0
                pct = 100 * n_tokens / args.target_tokens
                eta = (args.target_tokens - n_tokens) / rate if rate > 0 else 0.0
                print(f"  {n_tokens:>12,} tok  ({pct:5.1f}%)  rows={n_rows:>10,}  rate={rate/1000:6.1f}k tok/s  ETA={eta/60:5.1f} min", flush=True)
                last_progress = n_tokens

            if n_tokens >= args.target_tokens:
                break

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

    # Write a sidecar manifest
    manifest = {
        "recipe": str(args.recipe),
        "recipe_name": recipe.get("name"),
        "recipe_version": recipe.get("version"),
        "seed": seed,
        "target_tokens": args.target_tokens,
        "actual_tokens": n_tokens,
        "actual_rows": n_rows,
        "wall_seconds": elapsed,
        "output": str(args.output),
        "per_source": {
            name: {
                "rows": per_source_rows[name],
                "tokens": per_source_tokens[name],
                "effective_weight": per_source_tokens[name] / max(n_tokens, 1),
                "target_weight": weights[i],
            } for i, name in enumerate(names)
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
