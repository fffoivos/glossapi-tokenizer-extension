"""Aggregate per-shard worker partials → final per-source + component parquets.

Inputs (from GCS):
  {gcs_prefix}/per_source_counts/shard_NN_of_KK.parquet   (K files)
      schema: source_dataset, id, fire_count    (long format, sparse)
  {gcs_prefix}/per_source_denominators/shard_NN_of_KK.json   (K files)
      schema: {source_dataset → {rows, chars, tokenized_tokens}}
  {gcs_prefix}/_DONE_shard_NN_of_KK                       (K markers — gate)

Outputs (local):
  variants/c3_added_17408_curated_padded.firing_counts/
    per_source_counts.parquet           (long format, sparse, all shards summed)
    glossapi_nanochat_only.parquet      (id, decoded, fire_count, fire_rate)
    hplt_only.parquet                   (id, decoded, fire_count, fire_rate)
    glossapi_nanochat_plus_hplt.parquet (derived sum of the two components)
    run_summary.json                    (denominators + tail stats per component
                                          + per-source contribution summary)

Invariants enforced (per FIRING_COUNT_PLAN.md v2.2 §share-sum check):
  For each token id i:
    sum_over_sources( per_source_counts[s, i] )
        == glossapi_nanochat_only[i] + hplt_only[i]
  Per-source rows + chars + tokens (summed across shards) equal the
  per-component denominators after re-grouping by component.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HPLT_SOURCE_STRING = "HPLT/ell_Grek_ge8_no_mt_clean60"


def classify_source(s: str) -> str:
    return "hplt_only" if s == HPLT_SOURCE_STRING else "glossapi_nanochat_only"


def gsutil_ls(gcs_uri: str) -> list[str]:
    """List GCS objects under a prefix. Returns full gs:// URIs."""
    p = subprocess.run(
        ["gcloud", "storage", "ls", gcs_uri],
        capture_output=True, text=True, check=False,
    )
    if p.returncode != 0:
        return []
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def gsutil_cp(src: str, dst: str) -> None:
    subprocess.run(["gcloud", "storage", "cp", src, dst],
                   check=True, capture_output=True, text=True)


def expect_done_markers(gcs_prefix: str, k: int) -> None:
    """Fail-hard if any shard's _DONE marker is missing."""
    found = gsutil_ls(f"{gcs_prefix}/")
    done_uris = [u for u in found if "/_DONE_shard_" in u]
    expected = set(
        f"{gcs_prefix.rstrip('/')}/_DONE_shard_{i:02d}_of_{k:02d}"
        for i in range(k)
    )
    actually = set(done_uris)
    missing = expected - actually
    if missing:
        raise RuntimeError(
            f"Missing _DONE markers for {len(missing)} shards "
            f"(of {k}). Respawn those shards and rerun.\n"
            f"  missing: {sorted(missing)}"
        )


def load_partials(gcs_prefix: str, k: int, tmp_dir: Path) -> tuple[list, list]:
    """Download all K shard partials (counts + denoms) to local."""
    counts_paths: list[Path] = []
    denoms_paths: list[Path] = []
    for i in range(k):
        tag = f"shard_{i:02d}_of_{k:02d}"
        c_src = f"{gcs_prefix}/per_source_counts/{tag}.parquet"
        d_src = f"{gcs_prefix}/per_source_denominators/{tag}.json"
        c_dst = tmp_dir / f"{tag}_counts.parquet"
        d_dst = tmp_dir / f"{tag}_denoms.json"
        gsutil_cp(c_src, str(c_dst))
        gsutil_cp(d_src, str(d_dst))
        counts_paths.append(c_dst)
        denoms_paths.append(d_dst)
    return counts_paths, denoms_paths


def sum_partials_to_per_source(counts_paths: list[Path],
                               vocab_size: int) -> dict[str, np.ndarray]:
    """Sum per-shard long-format partials into a per-source dense counts dict.

    CRITICAL (reviewer round 3 #1): vocab_size must come from the tokenizer,
    not from max(id) across partials. Otherwise high-id tokens that never
    fire silently disappear from the output, and tail stats become wrong.
    """
    out: dict[str, np.ndarray] = {}
    observed_max_id = -1
    for p in counts_paths:
        tbl = pq.read_table(str(p))
        srcs = tbl.column("source_dataset").to_pylist()
        ids_arr = np.asarray(tbl.column("id").to_numpy(), dtype=np.int64)
        counts_arr = np.asarray(tbl.column("fire_count").to_numpy(), dtype=np.int64)
        if ids_arr.size:
            observed_max_id = max(observed_max_id, int(ids_arr.max()))
        # Add to per-source vectors; accumulate via direct index assignment
        for s, i, c in zip(srcs, ids_arr.tolist(), counts_arr.tolist()):
            if s not in out:
                out[s] = np.zeros(vocab_size, dtype=np.int64)
            out[s][i] += c
    if observed_max_id >= vocab_size:
        raise RuntimeError(
            f"observed token id {observed_max_id} >= tokenizer vocab_size "
            f"{vocab_size}; tokenizer/data mismatch"
        )
    return out


def sum_denominators(denoms_paths: list[Path]) -> dict[str, dict]:
    """Sum per-shard denoms into a per-source aggregate."""
    agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "chars": 0, "tokenized_tokens": 0}
    )
    for p in denoms_paths:
        d = json.loads(p.read_text())
        for s, fields in d.items():
            for k in ("rows", "chars", "tokenized_tokens"):
                agg[s][k] += fields.get(k, 0)
    return dict(agg)


def derive_component_counts(per_source: dict[str, np.ndarray],
                            vocab_size: int) -> dict[str, np.ndarray]:
    """Sum per-source counts into component (glossapi_nanochat_only / hplt_only)."""
    comp = {
        "glossapi_nanochat_only": np.zeros(vocab_size, dtype=np.int64),
        "hplt_only": np.zeros(vocab_size, dtype=np.int64),
    }
    for s, vec in per_source.items():
        comp[classify_source(s)] += vec
    return comp


def share_sum_invariant_check(per_source: dict[str, np.ndarray],
                              components: dict[str, np.ndarray]) -> None:
    """For every id: sum over sources == sum over components."""
    sums_src = sum(per_source.values())
    sums_comp = sum(components.values())
    if not np.array_equal(sums_src, sums_comp):
        diffs = np.where(sums_src != sums_comp)[0]
        raise RuntimeError(
            f"share-sum invariant violated on {len(diffs)} ids "
            f"(first 5: {diffs[:5].tolist()})"
        )


def tail_stats(vec: np.ndarray, base_vocab: int = 131_072) -> dict:
    """Per-component tail stats focused on the added 17,408 vocab range."""
    added = vec[base_vocab:]
    return {
        "n_added_tokens": int(added.size),
        "total_firings_added": int(added.sum()),
        "tokens_with_zero_firings": int((added == 0).sum()),
        "tokens_with_lt_100_firings": int((added < 100).sum()),
        "tokens_with_lt_1k_firings": int((added < 1000).sum()),
        "tokens_with_ge_10k_firings": int((added >= 10_000).sum()),
        "percentiles": {
            "p10": int(np.percentile(added, 10)),
            "p25": int(np.percentile(added, 25)),
            "p50": int(np.percentile(added, 50)),
            "p75": int(np.percentile(added, 75)),
            "p90": int(np.percentile(added, 90)),
            "p99": int(np.percentile(added, 99)),
        },
    }


def write_component_parquet(out_path: Path, vec: np.ndarray, denom_tokens: int,
                            tokenizer_dir: str) -> None:
    """Write (id, decoded, fire_count, fire_rate) for a single component."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    ids = np.arange(len(vec), dtype=np.int32)
    decoded = [tok.decode([int(i)], skip_special_tokens=False) for i in ids]
    rates = (vec.astype(np.float64) / denom_tokens) if denom_tokens > 0 else np.zeros_like(vec, dtype=np.float64)
    table = pa.table({
        "id": ids,
        "decoded": pa.array(decoded, type=pa.string()),
        "fire_count": vec.astype(np.int64),
        "fire_rate": rates,
    })
    pq.write_table(table, str(out_path), compression="zstd")


def write_per_source_long_parquet(out_path: Path, per_source: dict[str, np.ndarray],
                                  per_source_denoms: dict, components_tokens: dict,
                                  tokenizer_dir: str, vocab_size: int) -> int:
    """Sparse long format: only non-zero (source, id) rows."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    rows_sources, rows_groups, rows_ids, rows_decoded = [], [], [], []
    rows_counts, rows_rate_in_src, rows_share_of_component = [], [], []
    # Build component counts once for share_of_component
    comp_counts = derive_component_counts(per_source, vocab_size)
    for source, vec in per_source.items():
        nz = np.nonzero(vec)[0]
        if nz.size == 0:
            continue
        group = classify_source(source)
        src_tokens = per_source_denoms.get(source, {}).get("tokenized_tokens", 0)
        for i in nz:
            c = int(vec[i])
            rows_sources.append(source)
            rows_groups.append(group)
            rows_ids.append(int(i))
            rows_decoded.append(tok.decode([int(i)], skip_special_tokens=False))
            rows_counts.append(c)
            rows_rate_in_src.append(c / src_tokens if src_tokens > 0 else 0.0)
            comp_c = int(comp_counts[group][i])
            rows_share_of_component.append(c / comp_c if comp_c > 0 else 0.0)
    table = pa.table({
        "source_dataset": pa.array(rows_sources, type=pa.string()),
        "source_group": pa.array(rows_groups, type=pa.string()),
        "id": pa.array(rows_ids, type=pa.int32()),
        "decoded": pa.array(rows_decoded, type=pa.string()),
        "fire_count": pa.array(rows_counts, type=pa.int64()),
        "fire_rate_within_source_dataset": pa.array(rows_rate_in_src, type=pa.float64()),
        "share_of_component_token_firings": pa.array(rows_share_of_component, type=pa.float64()),
    })
    pq.write_table(table, str(out_path), compression="zstd")
    return table.num_rows


def write_source_dataset_summary_parquet(out_path: Path,
                                          per_source: dict[str, np.ndarray],
                                          per_source_denoms: dict,
                                          base_vocab: int = 131_072) -> None:
    """Per-source aggregate stats (one row per source_dataset).

    Reviewer round 3 #5: canonical second output named to match the plan.
    """
    rows = []
    for s, vec in per_source.items():
        d = per_source_denoms.get(s, {})
        added = vec[base_vocab:]
        rows.append({
            "source_dataset": s,
            "source_group": classify_source(s),
            "rows": int(d.get("rows", 0)),
            "chars": int(d.get("chars", 0)),
            "tokenized_tokens": int(d.get("tokenized_tokens", 0)),
            "n_nonzero_vocab_ids": int((vec > 0).sum()),
            "n_nonzero_added_ids": int((added > 0).sum()),
            "total_added_firings": int(added.sum()),
            "added_firing_share_of_total":
                float(added.sum() / vec.sum()) if vec.sum() > 0 else 0.0,
        })
    rows.sort(key=lambda r: -r["tokenized_tokens"])
    table = pa.table({
        "source_dataset": pa.array([r["source_dataset"] for r in rows], type=pa.string()),
        "source_group":   pa.array([r["source_group"] for r in rows], type=pa.string()),
        "rows":           pa.array([r["rows"] for r in rows], type=pa.int64()),
        "chars":          pa.array([r["chars"] for r in rows], type=pa.int64()),
        "tokenized_tokens": pa.array([r["tokenized_tokens"] for r in rows], type=pa.int64()),
        "n_nonzero_vocab_ids": pa.array([r["n_nonzero_vocab_ids"] for r in rows], type=pa.int64()),
        "n_nonzero_added_ids": pa.array([r["n_nonzero_added_ids"] for r in rows], type=pa.int64()),
        "total_added_firings": pa.array([r["total_added_firings"] for r in rows], type=pa.int64()),
        "added_firing_share_of_total": pa.array([r["added_firing_share_of_total"] for r in rows], type=pa.float64()),
    })
    pq.write_table(table, str(out_path), compression="zstd")


def aggregate(gcs_prefix: str, k: int, tokenizer_dir: str,
              out_dir: Path) -> dict:
    """Full aggregation pipeline. Returns run_summary dict."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # CRITICAL (reviewer round 3 #1): vocab_size from tokenizer, not from
    # max(observed id). Zero-firing tail tokens MUST appear in the output.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    vocab_size = tok.get_vocab_size()
    print(f"[agg] tokenizer vocab_size = {vocab_size:,}", flush=True)

    print(f"[agg] checking _DONE markers ({k} expected) ...", flush=True)
    expect_done_markers(gcs_prefix, k)
    print(f"  ✓ all {k} _DONE markers present", flush=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print(f"[agg] downloading {k} shard partials to {td} ...", flush=True)
        counts_paths, denoms_paths = load_partials(gcs_prefix, k, td)
        print(f"  ✓ downloaded {len(counts_paths)} counts + denoms files", flush=True)

        print(f"[agg] summing per-source counts (vocab={vocab_size:,}) ...", flush=True)
        per_source = sum_partials_to_per_source(counts_paths, vocab_size)
        print(f"  ✓ {len(per_source)} distinct sources, all vectors len {vocab_size:,}",
              flush=True)

        print(f"[agg] summing per-source denominators ...", flush=True)
        per_source_denoms = sum_denominators(denoms_paths)
        print(f"  ✓ {len(per_source_denoms)} sources in denoms", flush=True)

        print(f"[agg] deriving components ...", flush=True)
        components = derive_component_counts(per_source, vocab_size)
        component_denoms = {
            "glossapi_nanochat_only": {"rows": 0, "chars": 0, "tokenized_tokens": 0},
            "hplt_only": {"rows": 0, "chars": 0, "tokenized_tokens": 0},
        }
        for src, d in per_source_denoms.items():
            g = classify_source(src)
            for fld in ("rows", "chars", "tokenized_tokens"):
                component_denoms[g][fld] += d[fld]

        print(f"[agg] running share-sum invariant check ...", flush=True)
        share_sum_invariant_check(per_source, components)
        print(f"  ✓ invariant holds for all {vocab_size:,} ids", flush=True)

        # Write component parquets
        print(f"[agg] writing component parquets ...", flush=True)
        ga = components["glossapi_nanochat_only"]
        hp = components["hplt_only"]
        comb = ga + hp
        write_component_parquet(
            out_dir / "glossapi_nanochat_only.parquet", ga,
            component_denoms["glossapi_nanochat_only"]["tokenized_tokens"],
            tokenizer_dir,
        )
        if hp.sum() > 0:
            write_component_parquet(
                out_dir / "hplt_only.parquet", hp,
                component_denoms["hplt_only"]["tokenized_tokens"],
                tokenizer_dir,
            )
            write_component_parquet(
                out_dir / "glossapi_nanochat_plus_hplt.parquet", comb,
                component_denoms["glossapi_nanochat_only"]["tokenized_tokens"]
                + component_denoms["hplt_only"]["tokenized_tokens"],
                tokenizer_dir,
            )

        # Per-source long  (renamed per reviewer round 3 #5)
        print(f"[agg] writing source_dataset_token_counts (long format) ...", flush=True)
        per_source_components_tokens = {
            "glossapi_nanochat_only":
                component_denoms["glossapi_nanochat_only"]["tokenized_tokens"],
            "hplt_only": component_denoms["hplt_only"]["tokenized_tokens"],
        }
        n_per_source_rows = write_per_source_long_parquet(
            out_dir / "source_dataset_token_counts.parquet", per_source,
            per_source_denoms, per_source_components_tokens, tokenizer_dir,
            vocab_size,
        )
        print(f"  ✓ {n_per_source_rows:,} non-zero (source, id) rows", flush=True)

        # Per-source summary (one row per source, reviewer round 3 #5)
        print(f"[agg] writing source_dataset_summary.parquet ...", flush=True)
        write_source_dataset_summary_parquet(
            out_dir / "source_dataset_summary.parquet",
            per_source, per_source_denoms,
        )
        print(f"  ✓ {len(per_source)} source rows", flush=True)

        # Summary JSON
        summary = {
            "k_shards": k,
            "gcs_prefix": gcs_prefix,
            "wall_seconds": round(time.time() - t0, 1),
            "vocab_size": vocab_size,
            "n_distinct_sources": len(per_source),
            "component_denominators": component_denoms,
            "per_component_tail_stats_on_added_17408": {
                "glossapi_nanochat_only": tail_stats(ga),
                "hplt_only": tail_stats(hp) if hp.sum() > 0 else None,
                "glossapi_nanochat_plus_hplt": tail_stats(comb) if hp.sum() > 0 else None,
            },
            "per_source_summary": {
                s: {
                    **per_source_denoms[s],
                    "n_nonzero_ids": int((vec > 0).sum()),
                    "n_nonzero_added_ids": int((vec[131_072:] > 0).sum()),
                    "total_added_firings": int(vec[131_072:].sum()),
                    "component": classify_source(s),
                }
                for s, vec in per_source.items()
            },
        }
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=False)
        )
        print(f"[agg] DONE. wall {time.time()-t0:.1f}s", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-prefix", required=True,
                    help="e.g. gs://testbucketglossapi/firing_counts_TS")
    ap.add_argument("--k", type=int, required=True,
                    help="number of shards = number of workers")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="local output dir, e.g. "
                         "variants/c3_added_17408_curated_padded.firing_counts/")
    args = ap.parse_args()
    aggregate(args.gcs_prefix, args.k, args.tokenizer_dir, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[agg FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
