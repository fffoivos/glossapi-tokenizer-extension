#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from time import time

from glossapi_corpus_cli.continuous_bpe import (
    aggregate_sequence_shards,
    build_extended_tokenizer_dir,
    build_sequence_shards,
    collect_segment_shards,
    load_base_tokenizer_artifacts,
    run_continuation_training,
    verify_front_end_contract,
    verify_tokenizer_identity,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue Apertus BPE training from the frozen base tokenizer without resetting model.vocab/model.merges.")
    parser.add_argument("--base-tokenizer-dir", required=True, help="Frozen local Apertus snapshot directory with tokenizer.json/tokenizer_config.json/special_tokens_map.json")
    parser.add_argument("--reference-tokenizer", default="swiss-ai/Apertus-8B-2509", help="Reference tokenizer repo or path for the identity check")
    parser.add_argument("--input-glob", action="append", required=True, help="Parquet glob(s) with a text column.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-vocab-size", type=int, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=min(64, os.cpu_count() or 16))
    parser.add_argument("--row-group-chunk-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--max-row-groups", type=int, default=None)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    parser.add_argument("--name", default="continuous_bpe")
    parser.add_argument("--skip-identity-check", action="store_true")
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(pattern))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError(f"No input parquet files matched: {patterns}")
    return unique


def count_rows(paths: list[Path]) -> int:
    import pyarrow.parquet as pq

    return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in paths)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    segment_shard_dir = work_dir / "segment_shards"
    sequence_shard_dir = work_dir / "sequence_shards"
    aggregated_sequence_path = work_dir / "sequence_counter.pkl"
    checkpoint_path = work_dir / "merge_state_latest.pkl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "training_summary.json"
    identity_path = output_dir / "replication_check.json"
    contract_path = output_dir / "front_end_contract_check.json"
    tokenizer_output_dir = output_dir / "tokenizer"

    input_paths = expand_inputs(args.input_glob)
    total_input_rows = count_rows(input_paths)

    base_artifacts = load_base_tokenizer_artifacts(args.base_tokenizer_dir)
    if args.target_vocab_size % 128 != 0:
        raise ValueError("target_vocab_size must be divisible by 128")
    if args.target_vocab_size <= base_artifacts.base_vocab_size:
        raise ValueError("target_vocab_size must exceed the base tokenizer size")

    started = time()
    phase_times: dict[str, float] = {}

    progress_state = {
        "name": args.name,
        "phase": "init",
        "updated_at_epoch": time(),
        "input_files": [str(path) for path in input_paths],
        "target_vocab_size": args.target_vocab_size,
    }

    def update_progress(patch: dict[str, object]) -> None:
        progress_state.update(patch)
        progress_state["updated_at_epoch"] = time()
        write_json_atomic(progress_path, progress_state)

    update_progress({"phase": "identity_check"})
    if not args.skip_identity_check and not identity_path.exists():
        identity = verify_tokenizer_identity(
            args.base_tokenizer_dir,
            args.reference_tokenizer,
            sample_texts=[
                "Καλημέρα κόσμε!",
                "Το ελληνικό κείμενο πρέπει να περάσει ακριβώς από το ίδιο front-end.",
                "Ἐν ἀρχῇ ἦν ὁ λόγος.",
            ],
        )
        write_json_atomic(identity_path, identity)
    phase_times["identity_check_seconds"] = time() - started

    update_progress({"phase": "count_segments"})
    phase_started = time()
    segment_shard_paths = collect_segment_shards(
        base_ref=args.base_tokenizer_dir,
        input_paths=input_paths,
        shard_dir=segment_shard_dir,
        text_column=args.text_column,
        batch_size=args.batch_size,
        row_group_chunk_size=args.row_group_chunk_size,
        max_row_groups=args.max_row_groups,
        num_workers=args.num_workers,
        progress_callback=update_progress,
    )
    phase_times["count_segments_seconds"] = time() - phase_started

    update_progress({"phase": "build_sequence_shards"})
    phase_started = time()
    sequence_shard_paths = build_sequence_shards(
        base_ref=args.base_tokenizer_dir,
        segment_shard_paths=segment_shard_paths,
        sequence_shard_dir=sequence_shard_dir,
        num_workers=args.num_workers,
        progress_callback=update_progress,
    )
    phase_times["build_sequence_shards_seconds"] = time() - phase_started

    update_progress({"phase": "aggregate_sequences"})
    phase_started = time()
    sequence_counter = aggregate_sequence_shards(sequence_shard_paths, aggregated_sequence_path)
    phase_times["aggregate_sequences_seconds"] = time() - phase_started

    update_progress(
        {
            "phase": "merge_loop",
            "unique_sequences": len(sequence_counter),
            "base_vocab_size": base_artifacts.base_vocab_size,
        }
    )
    phase_started = time()
    continuation = run_continuation_training(
        base_artifacts=base_artifacts,
        sequence_counter=sequence_counter,
        target_vocab_size=args.target_vocab_size,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        progress_callback=update_progress,
        min_pair_frequency=args.min_pair_frequency,
    )
    phase_times["merge_loop_seconds"] = time() - phase_started

    update_progress({"phase": "write_tokenizer"})
    phase_started = time()
    tokenizer_info = build_extended_tokenizer_dir(
        base_artifacts=base_artifacts,
        output_dir=tokenizer_output_dir,
        token_strings=continuation["token_strings"],
        added_merges=continuation["added_merges"],
    )
    phase_times["write_tokenizer_seconds"] = time() - phase_started

    update_progress({"phase": "front_end_contract"})
    contract = verify_front_end_contract(args.base_tokenizer_dir, str(tokenizer_output_dir))
    write_json_atomic(contract_path, contract)

    elapsed = time() - started
    summary = {
        "name": args.name,
        "base_tokenizer_dir": str(Path(args.base_tokenizer_dir).resolve()),
        "reference_tokenizer": args.reference_tokenizer,
        "output_dir": str(output_dir),
        "tokenizer_output_dir": str(tokenizer_output_dir),
        "target_vocab_size": args.target_vocab_size,
        "base_vocab_size": base_artifacts.base_vocab_size,
        "added_token_count": tokenizer_info["added_count"],
        "added_merge_count": len(continuation["added_merges"]),
        "input": {
            "patterns": args.input_glob,
            "files": [str(path) for path in input_paths],
            "total_rows": total_input_rows,
            "text_column": args.text_column,
        },
        "runtime_seconds": elapsed,
        "phase_times_seconds": phase_times,
        "work_artifacts": {
            "segment_shard_dir": str(segment_shard_dir),
            "sequence_shard_dir": str(sequence_shard_dir),
            "sequence_counter_path": str(aggregated_sequence_path),
            "checkpoint_path": str(checkpoint_path),
            "progress_path": str(progress_path),
            "replication_check_path": str(identity_path),
            "front_end_contract_path": str(contract_path),
        },
        "tokenizer": tokenizer_info,
        "special_tokens_map": base_artifacts.special_tokens_map,
    }
    write_json_atomic(summary_path, summary)
    update_progress({"phase": "completed", "summary_path": str(summary_path), "tokenizer_output_dir": str(tokenizer_output_dir)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
