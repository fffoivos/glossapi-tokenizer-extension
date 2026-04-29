from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pickle
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Iterable

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_pickle_atomic(path: Path, payload: Any) -> None:
    _atomic_write_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def read_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BaseTokenizerArtifacts:
    base_ref: str
    base_dir: Path
    tokenizer_json: dict[str, Any]
    tokenizer_config: dict[str, Any]
    special_tokens_map: dict[str, Any]

    @property
    def vocab(self) -> dict[str, int]:
        return self.tokenizer_json["model"]["vocab"]

    @property
    def merges(self) -> list[list[str]]:
        return self.tokenizer_json["model"]["merges"]

    @property
    def base_vocab_size(self) -> int:
        return len(self.vocab)


@dataclass(frozen=True)
class RowGroupTask:
    file_path: Path
    task_id: str
    row_groups: tuple[int, ...]


def load_base_tokenizer_artifacts(base_ref: str) -> BaseTokenizerArtifacts:
    base_dir = Path(base_ref)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base tokenizer path not found: {base_ref}")
    tokenizer_json = json.loads((base_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((base_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    special_tokens_map_path = base_dir / "special_tokens_map.json"
    if special_tokens_map_path.exists():
        special_tokens_map = json.loads(special_tokens_map_path.read_text(encoding="utf-8"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(base_dir), use_fast=True)
        special_tokens_map = {}
        for attr in ["bos_token", "eos_token", "pad_token", "unk_token"]:
            token_value = getattr(tokenizer, attr, None)
            if token_value is None:
                continue
            token_obj = tokenizer._special_tokens_map.get(attr, token_value)
            if hasattr(token_obj, "content"):
                special_tokens_map[attr] = {
                    "content": token_obj.content,
                    "lstrip": bool(getattr(token_obj, "lstrip", False)),
                    "normalized": bool(getattr(token_obj, "normalized", False)),
                    "rstrip": bool(getattr(token_obj, "rstrip", False)),
                    "single_word": bool(getattr(token_obj, "single_word", False)),
                }
            else:
                special_tokens_map[attr] = {
                    "content": str(token_value),
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                }
    return BaseTokenizerArtifacts(
        base_ref=base_ref,
        base_dir=base_dir.resolve(),
        tokenizer_json=tokenizer_json,
        tokenizer_config=tokenizer_config,
        special_tokens_map=special_tokens_map,
    )


def verify_tokenizer_identity(base_ref: str, reference_repo: str, sample_texts: Iterable[str]) -> dict[str, Any]:
    reference = AutoTokenizer.from_pretrained(reference_repo, use_fast=True)
    frozen = AutoTokenizer.from_pretrained(base_ref, use_fast=True)
    if len(reference) != len(frozen):
        raise ValueError(f"Tokenizer length mismatch: {len(reference)} != {len(frozen)}")
    if reference.special_tokens_map != frozen.special_tokens_map:
        raise ValueError("Special tokens map mismatch between reference repo and frozen base")
    checks = {
        "normalizer": reference.backend_tokenizer.normalizer.__getstate__() if reference.backend_tokenizer.normalizer else None,
        "pre_tokenizer": reference.backend_tokenizer.pre_tokenizer.__getstate__() if reference.backend_tokenizer.pre_tokenizer else None,
        "decoder": reference.backend_tokenizer.decoder.__getstate__() if reference.backend_tokenizer.decoder else None,
        "post_processor": reference.backend_tokenizer.post_processor.__getstate__() if reference.backend_tokenizer.post_processor else None,
    }
    frozen_checks = {
        "normalizer": frozen.backend_tokenizer.normalizer.__getstate__() if frozen.backend_tokenizer.normalizer else None,
        "pre_tokenizer": frozen.backend_tokenizer.pre_tokenizer.__getstate__() if frozen.backend_tokenizer.pre_tokenizer else None,
        "decoder": frozen.backend_tokenizer.decoder.__getstate__() if frozen.backend_tokenizer.decoder else None,
        "post_processor": frozen.backend_tokenizer.post_processor.__getstate__() if frozen.backend_tokenizer.post_processor else None,
    }
    if checks != frozen_checks:
        raise ValueError("Frozen base tokenizer front-end does not match the reference repo")
    sample_results: list[dict[str, Any]] = []
    for text in sample_texts:
        ref_ids = reference.encode(text, add_special_tokens=False)
        frozen_ids = frozen.encode(text, add_special_tokens=False)
        if ref_ids != frozen_ids:
            raise ValueError(f"Reference and frozen base tokenize differently on sample text: {text!r}")
        sample_results.append({"text": text, "ids": ref_ids})
    return {
        "reference_repo": reference_repo,
        "base_ref": base_ref,
        "vocab_size": len(reference),
        "special_tokens_map": reference.special_tokens_map,
        "sample_count": len(sample_results),
    }


def verify_front_end_contract(base_ref: str, candidate_dir: str) -> dict[str, Any]:
    base = AutoTokenizer.from_pretrained(base_ref, use_fast=True)
    candidate = AutoTokenizer.from_pretrained(candidate_dir, use_fast=True)
    base_backend = base.backend_tokenizer
    candidate_backend = candidate.backend_tokenizer
    checks = {
        "normalizer": base_backend.normalizer.__getstate__() if base_backend.normalizer else None,
        "pre_tokenizer": base_backend.pre_tokenizer.__getstate__() if base_backend.pre_tokenizer else None,
        "decoder": base_backend.decoder.__getstate__() if base_backend.decoder else None,
        "post_processor": base_backend.post_processor.__getstate__() if base_backend.post_processor else None,
    }
    candidate_checks = {
        "normalizer": candidate_backend.normalizer.__getstate__() if candidate_backend.normalizer else None,
        "pre_tokenizer": candidate_backend.pre_tokenizer.__getstate__() if candidate_backend.pre_tokenizer else None,
        "decoder": candidate_backend.decoder.__getstate__() if candidate_backend.decoder else None,
        "post_processor": candidate_backend.post_processor.__getstate__() if candidate_backend.post_processor else None,
    }
    if checks != candidate_checks:
        raise ValueError("Candidate tokenizer front-end does not match the base tokenizer")
    if base.special_tokens_map != candidate.special_tokens_map:
        raise ValueError("Candidate special tokens map does not match the base tokenizer")
    if candidate.bos_token_id != base.bos_token_id or candidate.eos_token_id != base.eos_token_id or candidate.pad_token_id != base.pad_token_id:
        raise ValueError("Candidate special token ids do not match the base tokenizer")
    return {
        "base_ref": base_ref,
        "candidate_dir": candidate_dir,
        "base_vocab_size": len(base),
        "candidate_vocab_size": len(candidate),
        "bos_token_id": candidate.bos_token_id,
        "eos_token_id": candidate.eos_token_id,
        "pad_token_id": candidate.pad_token_id,
    }


def build_row_group_tasks(paths: list[Path], row_group_chunk_size: int, max_row_groups: int | None = None) -> list[RowGroupTask]:
    tasks: list[RowGroupTask] = []
    remaining = max_row_groups
    for path in paths:
        pf = pq.ParquetFile(path)
        total = pf.metadata.num_row_groups
        if remaining is not None:
            total = min(total, remaining)
        for start in range(0, total, row_group_chunk_size):
            stop = min(start + row_group_chunk_size, total)
            task_id = f"{path.stem}_rg{start:06d}_{stop - 1:06d}"
            tasks.append(RowGroupTask(file_path=path.resolve(), task_id=task_id, row_groups=tuple(range(start, stop))))
        if remaining is not None:
            remaining -= total
            if remaining <= 0:
                break
    return tasks


def _count_segment_shard(
    base_ref: str,
    task: RowGroupTask,
    text_column: str,
    batch_size: int,
    output_path: str,
    heartbeat_path: str,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_ref, use_fast=True)
    backend = tokenizer.backend_tokenizer if hasattr(tokenizer, "backend_tokenizer") else tokenizer._tokenizer
    counter: Counter[str] = Counter()
    rows = 0
    segments = 0
    heartbeat = Path(heartbeat_path)
    write_json_atomic(
        heartbeat,
        {
            "task_id": task.task_id,
            "state": "running",
            "rows": 0,
            "segments": 0,
            "row_groups": list(task.row_groups),
            "file_path": str(task.file_path),
        },
    )
    pf = pq.ParquetFile(task.file_path)
    batch_idx = 0
    for batch in pf.iter_batches(columns=[text_column], row_groups=list(task.row_groups), batch_size=batch_size):
        column = batch.column(0)
        for value in column:
            text = value.as_py()
            if not text:
                continue
            rows += 1
            pretokenized = backend.pre_tokenizer.pre_tokenize_str(text)
            if not pretokenized:
                continue
            counter.update(piece for piece, _ in pretokenized)
            segments += len(pretokenized)
        batch_idx += 1
        if batch_idx % 1 == 0:
            write_json_atomic(
                heartbeat,
                {
                    "task_id": task.task_id,
                    "state": "running",
                    "rows": rows,
                    "segments": segments,
                    "row_groups": list(task.row_groups),
                    "file_path": str(task.file_path),
                },
            )
    payload = {
        "task_id": task.task_id,
        "file_path": str(task.file_path),
        "row_groups": list(task.row_groups),
        "rows": rows,
        "segments": segments,
        "unique_segments": len(counter),
        "counter": counter,
    }
    write_pickle_atomic(Path(output_path), payload)
    write_json_atomic(
        heartbeat,
        {
            "task_id": task.task_id,
            "state": "completed",
            "rows": rows,
            "segments": segments,
            "unique_segments": len(counter),
            "row_groups": list(task.row_groups),
            "file_path": str(task.file_path),
            "output_path": output_path,
        },
    )
    return {
        "task_id": task.task_id,
        "rows": rows,
        "segments": segments,
        "unique_segments": len(counter),
        "output_path": output_path,
    }


def _sequence_counter_from_segment_shard(base_ref: str, segment_shard_path: str, output_path: str, heartbeat_path: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_ref, use_fast=True)
    backend = tokenizer.backend_tokenizer if hasattr(tokenizer, "backend_tokenizer") else tokenizer._tokenizer
    shard = read_pickle(Path(segment_shard_path))
    write_json_atomic(
        Path(heartbeat_path),
        {
            "task_id": shard["task_id"],
            "state": "running",
            "segments": shard["segments"],
            "rows": shard["rows"],
        },
    )
    sequence_counter: Counter[tuple[int, ...]] = Counter()
    kept_segments = 0
    for segment, freq in shard["counter"].items():
        tokens = backend.model.tokenize(segment)
        ids = tuple(token.id for token in tokens)
        if len(ids) < 2:
            continue
        sequence_counter[ids] += int(freq)
        kept_segments += int(freq)
    payload = {
        "task_id": shard["task_id"],
        "rows": shard["rows"],
        "segments": shard["segments"],
        "kept_segments": kept_segments,
        "unique_sequences": len(sequence_counter),
        "sequence_counter": sequence_counter,
    }
    write_pickle_atomic(Path(output_path), payload)
    write_json_atomic(
        Path(heartbeat_path),
        {
            "task_id": shard["task_id"],
            "state": "completed",
            "segments": shard["segments"],
            "rows": shard["rows"],
            "kept_segments": kept_segments,
            "unique_sequences": len(sequence_counter),
            "output_path": output_path,
        },
    )
    return {
        "task_id": shard["task_id"],
        "rows": shard["rows"],
        "segments": shard["segments"],
        "kept_segments": kept_segments,
        "unique_sequences": len(sequence_counter),
        "output_path": output_path,
    }


def collect_segment_shards(
    *,
    base_ref: str,
    input_paths: list[Path],
    shard_dir: Path,
    text_column: str,
    batch_size: int,
    row_group_chunk_size: int,
    max_row_groups: int | None,
    num_workers: int,
    progress_callback,
) -> list[Path]:
    tasks = build_row_group_tasks(input_paths, row_group_chunk_size=row_group_chunk_size, max_row_groups=max_row_groups)
    shard_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_dir = shard_dir / "_progress"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [shard_dir / f"{task.task_id}.pkl" for task in tasks]
    missing = [(task, output_paths[idx]) for idx, task in enumerate(tasks) if not output_paths[idx].exists()]
    completed = len(tasks) - len(missing)
    progress_callback({"count_total_tasks": len(tasks), "count_completed_tasks": completed, "count_inflight_tasks": len(missing)})
    if missing:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(_count_segment_shard, base_ref, task, text_column, batch_size, str(path), str(heartbeat_dir / f"{task.task_id}.json"))
                for task, path in missing
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                progress_callback({"count_total_tasks": len(tasks), "count_completed_tasks": completed, "count_inflight_tasks": len(tasks) - completed})
    return output_paths


def build_sequence_shards(
    *,
    base_ref: str,
    segment_shard_paths: list[Path],
    sequence_shard_dir: Path,
    num_workers: int,
    progress_callback,
) -> list[Path]:
    sequence_shard_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_dir = sequence_shard_dir / "_progress"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for segment_path in segment_shard_paths:
        output_path = sequence_shard_dir / segment_path.name
        pairs.append((segment_path, output_path))
    missing = [(segment_path, output_path) for segment_path, output_path in pairs if not output_path.exists()]
    completed = len(pairs) - len(missing)
    progress_callback({"sequence_total_tasks": len(pairs), "sequence_completed_tasks": completed, "sequence_inflight_tasks": len(missing)})
    if missing:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(
                    _sequence_counter_from_segment_shard,
                    base_ref,
                    str(segment_path),
                    str(output_path),
                    str(heartbeat_dir / f"{segment_path.stem}.json"),
                )
                for segment_path, output_path in missing
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                progress_callback({"sequence_total_tasks": len(pairs), "sequence_completed_tasks": completed, "sequence_inflight_tasks": len(pairs) - completed})
    return [output_path for _, output_path in pairs]


def aggregate_sequence_shards(sequence_shard_paths: list[Path], output_path: Path) -> Counter[tuple[int, ...]]:
    if output_path.exists():
        payload = read_pickle(output_path)
        return payload["sequence_counter"]
    aggregate: Counter[tuple[int, ...]] = Counter()
    total_segments = 0
    kept_segments = 0
    for path in sequence_shard_paths:
        shard = read_pickle(path)
        aggregate.update(shard["sequence_counter"])
        total_segments += int(shard["segments"])
        kept_segments += int(shard["kept_segments"])
    payload = {
        "sequence_counter": aggregate,
        "unique_sequences": len(aggregate),
        "total_segments": total_segments,
        "kept_segments": kept_segments,
    }
    write_pickle_atomic(output_path, payload)
    return aggregate


def consolidate_sequences(sequences: list[tuple[int, ...]], frequencies: list[int]) -> tuple[list[tuple[int, ...]], list[int]]:
    counter: Counter[tuple[int, ...]] = Counter()
    for sequence, freq in zip(sequences, frequencies):
        if freq <= 0:
            continue
        counter[tuple(sequence)] += int(freq)
    consolidated_sequences = list(counter.keys())
    consolidated_frequencies = [counter[sequence] for sequence in consolidated_sequences]
    return consolidated_sequences, consolidated_frequencies


def _pair_counter(sequence: tuple[int, ...]) -> Counter[tuple[int, int]]:
    return Counter(zip(sequence, sequence[1:]))


def _merge_sequence(sequence: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    merged: list[int] = []
    idx = 0
    limit = len(sequence)
    while idx < limit:
        if idx + 1 < limit and sequence[idx] == pair[0] and sequence[idx + 1] == pair[1]:
            merged.append(new_id)
            idx += 2
        else:
            merged.append(sequence[idx])
            idx += 1
    return tuple(merged)


def rebuild_pair_statistics(
    sequences: list[tuple[int, ...]],
    frequencies: list[int],
) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], dict[int, int]], list[tuple[int, tuple[int, int]]]]:
    pair_totals: Counter[tuple[int, int]] = Counter()
    pair_to_seq_counts: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for seq_id, sequence in enumerate(sequences):
        if len(sequence) < 2 or frequencies[seq_id] <= 0:
            continue
        local_pairs = _pair_counter(sequence)
        freq = int(frequencies[seq_id])
        for pair, occurrences in local_pairs.items():
            pair_totals[pair] += occurrences * freq
            pair_to_seq_counts[pair][seq_id] = occurrences
    heap = [(-count, pair) for pair, count in pair_totals.items() if count > 0]
    import heapq

    heapq.heapify(heap)
    return pair_totals, pair_to_seq_counts, heap


def select_next_pair(
    *,
    pair_totals: Counter[tuple[int, int]],
    heap: list[tuple[int, tuple[int, int]]],
    token_strings: list[str],
    token_to_id: dict[str, int],
    blocked_pairs: set[tuple[int, int]],
    min_frequency: int,
) -> tuple[tuple[int, int], int, str] | None:
    import heapq

    while heap:
        neg_count, pair = heapq.heappop(heap)
        current_count = int(pair_totals.get(pair, 0))
        if current_count <= 0:
            continue
        if current_count != -neg_count:
            continue
        if pair in blocked_pairs:
            continue
        if current_count < min_frequency:
            return None
        new_token = token_strings[pair[0]] + token_strings[pair[1]]
        if new_token in token_to_id:
            blocked_pairs.add(pair)
            continue
        return pair, current_count, new_token
    return None


def save_continuation_state(
    *,
    checkpoint_path: Path,
    sequences: list[tuple[int, ...]],
    frequencies: list[int],
    token_strings: list[str],
    added_merges: list[list[str]],
    added_tokens: list[str],
    step: int,
) -> None:
    consolidated_sequences, consolidated_frequencies = consolidate_sequences(sequences, frequencies)
    payload = {
        "step": step,
        "sequences": consolidated_sequences,
        "frequencies": consolidated_frequencies,
        "token_strings": token_strings,
        "added_merges": added_merges,
        "added_tokens": added_tokens,
    }
    write_pickle_atomic(checkpoint_path, payload)


def load_continuation_state(checkpoint_path: Path) -> dict[str, Any]:
    return read_pickle(checkpoint_path)


def run_continuation_training(
    *,
    base_artifacts: BaseTokenizerArtifacts,
    sequence_counter: Counter[tuple[int, ...]],
    target_vocab_size: int,
    checkpoint_path: Path,
    checkpoint_every: int,
    progress_callback,
    min_pair_frequency: int = 2,
) -> dict[str, Any]:
    if target_vocab_size <= base_artifacts.base_vocab_size:
        raise ValueError("target_vocab_size must exceed the base tokenizer size")
    if checkpoint_path.exists():
        state = load_continuation_state(checkpoint_path)
        sequences = list(state["sequences"])
        frequencies = list(state["frequencies"])
        token_strings = list(state["token_strings"])
        added_merges = list(state["added_merges"])
        added_tokens = list(state["added_tokens"])
        start_step = int(state["step"])
    else:
        sequences = [tuple(sequence) for sequence in sequence_counter.keys()]
        frequencies = [int(sequence_counter[sequence]) for sequence in sequences]
        token_strings = [None] * base_artifacts.base_vocab_size
        for token, token_id in base_artifacts.vocab.items():
            token_strings[token_id] = token
        added_merges: list[list[str]] = []
        added_tokens: list[str] = []
        start_step = 0
    token_to_id = {token: token_id for token_id, token in enumerate(token_strings)}
    pair_totals, pair_to_seq_counts, heap = rebuild_pair_statistics(sequences, frequencies)
    blocked_pairs: set[tuple[int, int]] = set()
    target_added = target_vocab_size - base_artifacts.base_vocab_size
    progress_callback(
        {
            "merge_target_added": target_added,
            "merge_completed_added": len(added_tokens),
            "current_vocab_size": len(token_strings),
            "phase": "merge_loop",
        }
    )
    started = time()
    progress_every = max(1, checkpoint_every // 16)
    for step in range(start_step, target_added):
        choice = select_next_pair(
            pair_totals=pair_totals,
            heap=heap,
            token_strings=token_strings,
            token_to_id=token_to_id,
            blocked_pairs=blocked_pairs,
            min_frequency=min_pair_frequency,
        )
        if choice is None:
            raise RuntimeError(f"Unable to continue BPE to requested size; exhausted candidate pairs at step {step}")
        pair, frequency, new_token = choice
        new_id = len(token_strings)
        token_strings.append(new_token)
        token_to_id[new_token] = new_id
        added_tokens.append(new_token)
        added_merges.append([token_strings[pair[0]], token_strings[pair[1]]])
        affected = list(pair_to_seq_counts.pop(pair, {}).items())
        pair_totals[pair] = 0
        for seq_id, _occurrences in affected:
            old_sequence = sequences[seq_id]
            if len(old_sequence) < 2 or frequencies[seq_id] <= 0:
                continue
            old_pairs = _pair_counter(old_sequence)
            new_sequence = _merge_sequence(old_sequence, pair, new_id)
            if new_sequence == old_sequence:
                continue
            new_pairs = _pair_counter(new_sequence)
            sequences[seq_id] = new_sequence
            freq = int(frequencies[seq_id])
            for changed_pair in set(old_pairs) | set(new_pairs):
                delta = int(new_pairs.get(changed_pair, 0)) - int(old_pairs.get(changed_pair, 0))
                if delta == 0:
                    continue
                pair_totals[changed_pair] += delta * freq
                local = pair_to_seq_counts.setdefault(changed_pair, {})
                new_count = int(local.get(seq_id, 0)) + delta
                if new_count > 0:
                    local[seq_id] = new_count
                else:
                    local.pop(seq_id, None)
                if pair_totals[changed_pair] > 0:
                    import heapq

                    heapq.heappush(heap, (-int(pair_totals[changed_pair]), changed_pair))
        completed = step + 1
        if completed % checkpoint_every == 0 or completed == target_added:
            save_continuation_state(
                checkpoint_path=checkpoint_path,
                sequences=sequences,
                frequencies=frequencies,
                token_strings=token_strings,
                added_merges=added_merges,
                added_tokens=added_tokens,
                step=completed,
            )
        if completed <= 10 or completed % progress_every == 0 or completed == target_added:
            progress_callback(
                {
                    "phase": "merge_loop",
                    "merge_target_added": target_added,
                    "merge_completed_added": completed,
                    "current_vocab_size": len(token_strings),
                    "latest_pair_frequency": frequency,
                    "elapsed_merge_seconds": time() - started,
                }
            )
    return {
        "token_strings": token_strings,
        "added_tokens": added_tokens,
        "added_merges": added_merges,
    }


def build_extended_tokenizer_dir(
    *,
    base_artifacts: BaseTokenizerArtifacts,
    output_dir: Path,
    token_strings: list[str],
    added_merges: list[list[str]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = json.loads(json.dumps(base_artifacts.tokenizer_json))
    tokenizer_json["model"]["vocab"] = {token: token_id for token_id, token in enumerate(token_strings)}
    tokenizer_json["model"]["merges"] = list(base_artifacts.merges) + added_merges
    _atomic_write_bytes(output_dir / "tokenizer.json", (json.dumps(tokenizer_json, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    _atomic_write_bytes(
        output_dir / "tokenizer_config.json",
        (json.dumps(base_artifacts.tokenizer_config, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    _atomic_write_bytes(
        output_dir / "special_tokens_map.json",
        (json.dumps(base_artifacts.special_tokens_map, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return {
        "output_dir": str(output_dir),
        "tokenizer_json_sha256": sha256_path(output_dir / "tokenizer.json"),
        "vocab_size": len(token_strings),
        "added_count": len(token_strings) - base_artifacts.base_vocab_size,
        "merge_count": len(tokenizer_json["model"]["merges"]),
    }
