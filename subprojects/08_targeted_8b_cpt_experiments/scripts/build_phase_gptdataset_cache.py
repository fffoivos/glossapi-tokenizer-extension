#!/usr/bin/env python3
"""Build one exact c92402e GPTDataset cache without loading a language model."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

import numpy as np

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    sha256_file,
    write_json_atomic,
)
from finalize_training_megatron import validate_runtime as validate_megatron_runtime
from freeze_phase_blend_cache import (
    PHASE3_COMPONENT_REQUESTED_SAMPLES,
    PHASE_DATASET_SAMPLES,
    validate_data_path_spec,
)

RUNTIME_VALIDATION_SAMPLES = 132_096
RUNTIME_TEST_SAMPLES = 1_024
RUNTIME_DATASET_BUILDER_THREADS = 4


@contextmanager
def single_rank_process_group() -> Iterator[dict[str, object]]:
    """Provide the rank-0 process group required by c92402e blend caching.

    The low-level GPTDataset builder tolerates an uninitialized distributed
    runtime, but BlendedDataset calls ``torch.distributed.get_rank()``
    unconditionally on a cache miss.  Cache construction is intentionally a
    one-process CPU operation, so create an ephemeral world-size-one Gloo
    group and tear it down before the immutable receipt is published.
    """

    import torch.distributed as dist

    require(dist.is_available(), "torch.distributed is unavailable in the cache-build runtime")
    require(not dist.is_initialized(), "cache builder requires an uninitialized process group")
    lifecycle: dict[str, object] = {
        "backend": "gloo",
        "rank": 0,
        "world_size": 1,
        "started_by_builder": True,
        "destroyed_after_build": False,
    }
    with tempfile.TemporaryDirectory(prefix="h2g-cache-pg-") as temporary:
        init_file = (Path(temporary).resolve() / "store")
        dist.init_process_group(
            backend="gloo",
            init_method=init_file.as_uri(),
            rank=0,
            world_size=1,
        )
        require(dist.is_initialized(), "single-rank process group did not initialize")
        require(dist.get_rank() == 0, "cache-builder process-group rank drift")
        require(dist.get_world_size() == 1, "cache-builder process-group world-size drift")
        require(str(dist.get_backend()) == "gloo", "cache-builder process-group backend drift")
        try:
            yield lifecycle
        finally:
            dist.destroy_process_group()
            lifecycle["destroyed_after_build"] = not dist.is_initialized()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--data-path-spec", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--megatron-receipt", type=Path, required=True)
    parser.add_argument("--validation-cache-seed-root", type=Path)
    parser.add_argument("--validation-data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seed_validation_cache(
    *, source_root: Path, validation_root: Path, cache_root: Path
) -> dict[str, object]:
    """Copy only exact online-validation cache entries from a proven cache.

    Megatron stores the main blend and every extra-validation dataset in the
    same ``path_to_cache``.  The training-only builder above creates the main
    blend; this step imports the already-built validation indices without
    importing unrelated or stale main-training keys.
    """

    source_root = source_root.resolve()
    validation_root = validation_root.resolve()
    require(source_root.is_dir(), "validation cache seed root missing")
    require(validation_root.is_dir(), "validation data root missing")
    require(source_root != cache_root, "validation cache seed aliases target cache")
    copied: list[str] = []
    descriptions = 0
    for description in sorted(source_root.glob("*-description.txt")):
        value = json.loads(description.read_text(encoding="utf-8"))
        datasets = value.get("datasets", [value])
        paths = [Path(str(row.get("dataset_path", ""))).resolve() for row in datasets]
        if not paths or not all(path.is_relative_to(validation_root) for path in paths):
            continue
        descriptions += 1
        prefix = description.name[: -len("-description.txt")]
        members = sorted(source_root.glob(f"{prefix}-*"))
        require(members and description in members, f"validation cache entry is incomplete: {prefix}")
        for source in members:
            target = cache_root / source.name
            if target.exists():
                require(
                    target.stat().st_size == source.stat().st_size
                    and sha256_file(target) == sha256_file(source),
                    f"validation cache seed collides with different bytes: {source.name}",
                )
            else:
                shutil.copy2(source, target)
            copied.append(source.name)
    require(descriptions > 0 and copied, "validation cache seed selected no entries")
    return {
        "source_root": str(source_root),
        "validation_data_root": str(validation_root),
        "description_files": descriptions,
        "files": sorted(set(copied)),
    }


def no_epoch_wrap(cache_root: Path) -> dict[str, object]:
    files = sorted(cache_root.glob("*GPTDataset-train-document_index.npy"))
    require(bool(files), "Phase-3 cache has no train document-index arrays")
    rows = []
    for path in files:
        values = np.load(path, mmap_mode="r")
        unique = int(np.unique(values).size)
        total = int(values.size)
        require(unique == total, f"Phase-3 document-index wraps/repeats documents: {path}")
        rows.append({"path": str(path.resolve()), "entries": total, "unique_entries": unique})
    return {"passed": True, "files": rows}


def main() -> int:
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

    args = parse_args()
    require(not args.output.exists(), f"immutable cache-build receipt exists: {args.output}")
    require(args.tokenizer.is_dir(), "tokenizer directory missing")
    megatron = read_json(args.megatron_receipt)
    patch = Path(str(megatron.get("patch", {}).get("path", "")))
    validate_megatron_runtime(
        megatron,
        args.megatron_root,
        patch,
        require_helpers=True,
    )
    cache_root = args.cache_root.resolve()
    require(not cache_root.exists(), f"cache root already exists: {cache_root}")
    cache_root.mkdir(parents=True)
    spec = read_json(args.data_path_spec)
    _, prefixes = validate_data_path_spec(spec, args.phase)
    components = spec["components"]
    weights = [float(row["weight"]) for row in components]
    # Match the production ``--tokenizer-type HuggingFaceTokenizer`` path
    # exactly. Passing ``local_files_only=True`` changes the tokenizer kwargs
    # serialized into Megatron's dataset description and therefore changes the
    # cache key, even though the loaded tokenizer bytes are identical.
    tokenizer = _HuggingFaceTokenizer(str(args.tokenizer.resolve()))
    config = GPTDatasetConfig(
        random_seed=20260609,
        sequence_length=4096,
        blend=([str(path) for path in prefixes], weights),
        blend_per_split=None,
        split="100,0,0",
        # Cache descriptions include this construction contract. Match the
        # production Megatron invocation rather than a preparation-only value.
        num_dataset_builder_threads=RUNTIME_DATASET_BUILDER_THREADS,
        path_to_cache=str(cache_root),
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        create_attention_mask=True,
        goldfish_loss=True,
        goldfish_k=50,
        goldfish_h=50,
    )
    requested = PHASE_DATASET_SAMPLES[args.phase]
    target_sizes = [requested, RUNTIME_VALIDATION_SAMPLES, RUNTIME_TEST_SAMPLES]
    with single_rank_process_group() as process_group:
        train, valid, test = BlendedMegatronDatasetBuilder(
            GPTDataset,
            target_sizes,
            lambda: True,
            config,
        ).build()
    require(process_group["destroyed_after_build"] is True, "cache-builder process group was not destroyed")
    require(train is not None and len(train) >= requested, "built training dataset is too short")
    require(valid is None and test is None, "unexpected validation/test datasets were built")
    component_requested_samples = None
    component_built_samples = None
    if args.phase == 3:
        datasets = getattr(train, "datasets", None)
        require(isinstance(datasets, (list, tuple)) and len(datasets) == len(components), "Phase-3 blended component datasets are not inspectable")
        component_requested_samples = dict(PHASE3_COMPONENT_REQUESTED_SAMPLES)
        component_built_samples = {
            str(component["role"]): len(dataset)
            for component, dataset in zip(components, datasets, strict=True)
        }
        # ``BlendedMegatronDatasetBuilder`` materializes one non-wrapping epoch
        # for every component, then draws the phase-local blended horizon from
        # those component indices.  The selector's 1.005x figures are therefore
        # lower bounds on available samples, not the lengths of the individual
        # one-epoch GPTDataset arrays.  Requiring equality here would reject a
        # valid cache whenever a selected component contains more than its
        # minimum capacity.
        require(
            set(component_built_samples) == set(component_requested_samples)
            and all(
                component_built_samples[role] >= component_requested_samples[role]
                for role in component_requested_samples
            ),
            "Phase-3 component capacity is below its requested no-wrap margin: "
            f"built={component_built_samples}, requested={component_requested_samples}",
        )
    require(
        (args.validation_cache_seed_root is None)
        == (args.validation_data_root is None),
        "validation cache seed root and data root must be supplied together",
    )
    validation_cache_seed = None
    if args.validation_cache_seed_root is not None:
        validation_cache_seed = seed_validation_cache(
            source_root=args.validation_cache_seed_root,
            validation_root=args.validation_data_root,
            cache_root=cache_root,
        )
    cache_files = sorted(path for path in cache_root.rglob("*") if path.is_file())
    require(bool(cache_files), "GPTDataset builder created no cache files")
    phase3_no_wrap = no_epoch_wrap(cache_root) if args.phase == 3 else None
    payload: dict[str, object] = {
        "schema_version": "apertus_hard_h_to_g_phase_cache_build_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": args.phase,
        "requested_samples": requested,
        "runtime_target_sizes": target_sizes,
        "runtime_dataset_builder_threads": RUNTIME_DATASET_BUILDER_THREADS,
        "built_samples": len(train),
        "data_path_spec": file_binding(args.data_path_spec),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "tokenizer_eod": tokenizer.eod,
        "megatron_root": str(args.megatron_root.resolve()),
        "megatron_receipt": file_binding(args.megatron_receipt),
        "cache_root": str(cache_root),
        "cache_file_count": len(cache_files),
        "cache_build_process_group": process_group,
        "validation_cache_seed": validation_cache_seed,
        "component_requested_samples": component_requested_samples,
        "component_built_samples": component_built_samples,
        "phase3_no_epoch_wrap": phase3_no_wrap,
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "phase": args.phase, "built_samples": len(train)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
