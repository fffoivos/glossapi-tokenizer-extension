#!/usr/bin/env python3
"""Lightweight production entry point for the established text deduplicator.

Unlike ``glossapi_corpus_cli.cli``, this launcher does not import the rest of
the corpus CLI (or Typer and the preparation pipeline).  It also establishes
the process-wide DuckDB resource contract before calling ``text_dedup``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glossapi_corpus_cli import text_dedup  # noqa: E402
from full_corpus_dedup_recipe import (  # noqa: E402
    APPROVED_PRODUCTION_RECIPE,
    validate_recipe_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", required=True)
    parser.add_argument("--duckdb-threads", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--experimental-parameters", action="store_true")
    parser.add_argument(
        "--greek-diacritic-policy",
        choices=sorted(text_dedup.GREEK_DIACRITIC_POLICIES),
        default=text_dedup.DEFAULT_GREEK_DIACRITIC_POLICY,
    )
    parser.add_argument("--minhash-threshold", type=float, default=text_dedup.DEFAULT_NEAR_THRESHOLD)
    parser.add_argument("--num-perm", type=int, default=text_dedup.DEFAULT_NUM_PERM)
    parser.add_argument("--bands", type=int, default=text_dedup.DEFAULT_BANDS)
    parser.add_argument("--rows-per-band", type=int, default=text_dedup.DEFAULT_ROWS_PER_BAND)
    parser.add_argument(
        "--shingle-mode",
        choices=["token", "char"],
        default=text_dedup.DEFAULT_SHINGLE_MODE,
    )
    parser.add_argument("--shingle-size", type=int, default=text_dedup.DEFAULT_SHINGLE_SIZE)
    parser.add_argument("--max-bucket-size", type=int, default=text_dedup.DEFAULT_MAX_BUCKET_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.duckdb_threads < 1:
        raise ValueError("--duckdb-threads must be >= 1")
    if not args.memory_limit.strip():
        raise ValueError("--memory-limit must not be empty")
    if args.exact_only and not args.experimental_parameters:
        raise ValueError("--exact-only is non-production and requires --experimental-parameters")
    requested_recipe = {
        key: getattr(args, key)
        for key in APPROVED_PRODUCTION_RECIPE
    }
    validate_recipe_parameters(
        requested_recipe,
        experimental=args.experimental_parameters,
    )

    temporary_directory = args.temporary_directory.expanduser().resolve()
    temporary_directory.mkdir(parents=True, exist_ok=True)
    os.environ[text_dedup.DUCKDB_TEMP_DIRECTORY_ENV] = str(temporary_directory)
    os.environ[text_dedup.DUCKDB_MEMORY_LIMIT_ENV] = args.memory_limit.strip()
    os.environ[text_dedup.DUCKDB_THREADS_ENV] = str(args.duckdb_threads)

    payload = text_dedup.run_dedup_pipeline(
        input_root=args.input_root,
        state_root=args.state_root,
        run_root=args.run_root,
        resume=args.resume,
        max_workers=args.max_workers,
        greek_diacritic_policy=args.greek_diacritic_policy,
        exact_only=args.exact_only,
        minhash_threshold=args.minhash_threshold,
        num_perm=args.num_perm,
        bands=args.bands,
        rows_per_band=args.rows_per_band,
        shingle_mode=args.shingle_mode,
        shingle_size=args.shingle_size,
        max_bucket_size=args.max_bucket_size,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
