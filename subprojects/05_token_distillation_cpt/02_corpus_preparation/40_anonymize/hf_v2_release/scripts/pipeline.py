#!/usr/bin/env python3
"""Prepare, transform, and finalize the row-preserving anonymized HF v2 release."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from release_common import (
    CONTRACT_SCHEMA,
    FINAL_SCHEMA,
    TASK_SCHEMA,
    canonical_sha256,
    code_inventory,
    file_receipt,
    load_contract,
    read_json,
    sha256_file,
    update_text_digest,
    utc_now,
    validate_config,
    write_json_atomic,
)


HERE = Path(__file__).resolve().parent
MASKER_DIR = HERE.parents[1] / "scripts"
if str(MASKER_DIR) not in sys.path:
    sys.path.insert(0, str(MASKER_DIR))
from pii_masker import mask  # noqa: E402


WORKER_TOKENIZER: Any = None


def _worker_init(tokenizer_json: str) -> None:
    global WORKER_TOKENIZER
    from tokenizers import Tokenizer

    WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_json)


def _mask_and_count(text: str) -> tuple[str, dict[str, int], int]:
    if WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    masked, counts = mask(text)
    second, residual = mask(masked)
    if second != masked or any(int(value) for value in residual.values()):
        raise RuntimeError("PII masking was not idempotent")
    tokens = len(WORKER_TOKENIZER.encode(masked, add_special_tokens=False).ids)
    return masked, {name: int(counts[name]) for name in ("email", "ip", "iban")}, tokens


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside frozen code root: {path}") from exc


def _code_paths(code_root: Path, config_path: Path) -> list[str]:
    paths = [
        _relative(Path(__file__), code_root),
        _relative(HERE / "release_common.py", code_root),
        _relative(HERE / "publish_release.py", code_root),
        _relative(MASKER_DIR / "pii_masker.py", code_root),
        _relative(config_path, code_root),
    ]
    clariden = HERE.parent / "clariden"
    paths.extend(
        _relative(clariden / name, code_root)
        for name in (
            "prepare.sbatch",
            "transform.sbatch",
            "transform_batch.sbatch",
            "finalize.sbatch",
            "finalize_overlay.sbatch",
            "publish.sbatch",
            "publish_overlay.sbatch",
            "submit.sh",
        )
    )
    return sorted(set(paths))


def _validate_external_receipt(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} checksum drift: {actual} != {expected_sha256}")
    value = read_json(path)
    return value


def prepare(args: argparse.Namespace) -> int:
    code_root = args.code_root.resolve()
    config_path = args.config.resolve()
    input_root = args.input_root.resolve()
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run root must be new and empty: {run_root}")
    config = read_json(config_path)
    validate_config(config)

    manifest_path = input_root / "manifests/deduplicated_manifest.json"
    if sha256_file(manifest_path) != config["input"]["deduplicated_manifest_sha256"]:
        raise ValueError("pinned HF v2 manifest checksum drift")
    manifest = read_json(manifest_path)
    if (
        manifest.get("status") != "passed"
        or manifest.get("repository_id") != config["input"]["repository_id"]
        or int(manifest.get("rows", -1)) != int(config["input"]["rows"])
        or len(manifest.get("files", [])) != int(config["input"]["shards"])
    ):
        raise ValueError("pinned HF v2 manifest does not match the release config")

    dedup = _validate_external_receipt(
        args.dedup_receipt.resolve(), config["deduplication"]["receipt_sha256"], "dedup receipt"
    )
    for key in (
        "input_rows",
        "retained_rows",
        "removed_rows",
        "clusters",
        "strict_exact_groups",
        "strict_exact_union_edges",
        "normalized_exact_groups",
        "normalized_exact_union_edges",
        "verified_minhash_edges",
        "verified_minhash_union_edges",
    ):
        if int(dedup.get(key, -1)) != int(config["deduplication"][key]):
            raise ValueError(f"dedup receipt/config mismatch: {key}")

    breakdown = _validate_external_receipt(
        args.source_breakdown.resolve(),
        config["deduplication"]["source_breakdown_sha256"],
        "source breakdown",
    )
    observed_sources = {
        str(row["source_dataset"]): int(row["retained_rows"])
        for row in breakdown.get("sources", [])
    }
    if observed_sources != {key: int(value) for key, value in config["expected_source_rows"].items()}:
        raise ValueError("source-breakdown/config rows differ")

    tokenizer_json = args.tokenizer_json.resolve()
    if tokenizer_json.is_symlink() or not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    if sha256_file(tokenizer_json) != config["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("production tokenizer checksum drift")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    if tokenizer.get_vocab_size(with_added_tokens=True) != int(config["tokenizer"]["vocab_size"]):
        raise ValueError("production tokenizer vocabulary-size drift")

    tasks = []
    seen_paths: set[str] = set()
    row_total = 0
    for index, row in enumerate(manifest["files"]):
        relative = str(row.get("path", ""))
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.suffix != ".parquet":
            raise ValueError(f"unsafe input path: {relative!r}")
        if relative in seen_paths:
            raise ValueError(f"duplicate input path: {relative}")
        seen_paths.add(relative)
        path = input_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"input file size drift: {relative}")
        task = {
            "task_index": index,
            "relative_path": relative,
            "input_path": str(path.resolve()),
            "input_bytes": int(row["bytes"]),
            "input_sha256": str(row["sha256"]),
            "rows": int(row["rows"]),
            "origin": str(row.get("origin", "")),
        }
        tasks.append(task)
        row_total += task["rows"]
    if row_total != int(config["input"]["rows"]):
        raise ValueError("task rows do not close to the release")

    code_rows = code_inventory(code_root, _code_paths(code_root, config_path))
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "receipts/tasks").mkdir(parents=True)
    (run_root / "release/data").mkdir(parents=True)
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "status": "frozen",
        "created_at": utc_now(),
        "run_root": str(run_root),
        "code_root": str(code_root),
        "code_inventory": code_rows,
        "config": {
            "path": _relative(config_path, code_root),
            "sha256": sha256_file(config_path),
            "canonical_sha256": canonical_sha256(config),
        },
        "input": {
            "root": str(input_root),
            "manifest": file_receipt(manifest_path),
            "repository_id": config["input"]["repository_id"],
            "revision": config["input"]["revision"],
            "rows": row_total,
            "shards": len(tasks),
        },
        "tokenizer": {
            **config["tokenizer"],
            "tokenizer_json": file_receipt(tokenizer_json),
        },
        "deduplication_receipt": file_receipt(args.dedup_receipt.resolve()),
        "source_breakdown": file_receipt(args.source_breakdown.resolve()),
        "tasks": tasks,
        "tasks_sha256": canonical_sha256(tasks),
        "policy": {
            "new_filtering": False,
            "new_deduplication": False,
            "new_decontamination": False,
            "row_removal_allowed": False,
            "only_text_may_change": True,
        },
    }
    write_json_atomic(run_root / "run_contract.json", contract)
    print(json.dumps({"ok": True, "run_root": str(run_root), "tasks": len(tasks), "rows": row_total}, sort_keys=True))
    return 0


def _validate_task_input(task: Mapping[str, Any]) -> Path:
    path = Path(str(task["input_path"]))
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(task["input_bytes"]):
        raise ValueError(f"input bytes drift: {path}")
    if sha256_file(path) != task["input_sha256"]:
        raise ValueError(f"input checksum drift: {path}")
    return path.resolve()


def _receipt_path(run_root: Path, task_index: int) -> Path:
    return run_root / "receipts/tasks" / f"task_{task_index:05d}.json"


def _output_path(run_root: Path, task: Mapping[str, Any]) -> Path:
    return run_root / "release" / str(task["relative_path"])


def _validate_resumed_task(receipt: Mapping[str, Any], task: Mapping[str, Any], output: Path) -> None:
    if (
        receipt.get("schema_version") != TASK_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("task_sha256") != canonical_sha256(task)
        or int(receipt.get("counts", {}).get("output_rows", -1)) != int(task["rows"])
    ):
        raise ValueError("existing task receipt binding drift")
    claimed = receipt.get("output", {})
    if output.is_symlink() or not output.is_file():
        raise FileNotFoundError(output)
    if output.stat().st_size != int(claimed.get("bytes", -1)) or sha256_file(output) != claimed.get("sha256"):
        raise ValueError("existing task output drift")


def _task_invariants_pass(receipt: Mapping[str, Any]) -> bool:
    expected = {
        "row_count_preserved": True,
        "row_order_preserved": True,
        "schema_and_metadata_preserved": True,
        "all_non_text_values_equal": True,
        "only_text_replaced": True,
        "new_filtering": False,
        "new_deduplication": False,
    }
    return receipt.get("invariants") == expected


def transform(args: argparse.Namespace) -> int:
    started = time.monotonic()
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path, executing_code_root=args.code_root.resolve())
    run_root = Path(contract["run_root"])
    config = read_json(args.code_root.resolve() / contract["config"]["path"])
    source_categories = validate_config(config)
    tasks = contract["tasks"]
    index = int(args.task_index)
    if index < 0 or index >= len(tasks):
        raise ValueError("task index outside frozen inventory")
    task = tasks[index]
    output = _output_path(run_root, task)
    receipt_path = _receipt_path(run_root, index)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        _validate_resumed_task(receipt, task, output)
        print(json.dumps({"ok": True, "resumed": True, "task": index}, sort_keys=True))
        return 0
    input_path = _validate_task_input(task)

    tokenizer_json = Path(contract["tokenizer"]["tokenizer_json"]["path"])
    if sha256_file(tokenizer_json) != contract["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("tokenizer drift after contract freeze")

    import pyarrow as pa
    import pyarrow.parquet as pq

    source_file = pq.ParquetFile(input_path)
    schema = source_file.schema_arrow
    text_index = schema.get_field_index("text")
    source_index = schema.get_field_index("source_dataset")
    if text_index < 0 or source_index < 0:
        raise ValueError("release shard lacks text/source_dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    for stale in output.parent.glob(output.name + ".partial-*"):
        if stale.is_symlink() or not stale.is_file():
            raise ValueError(f"unsafe stale task output: {stale}")
        stale.unlink()
    temporary = Path(str(output) + f".partial-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    raw_text_digest = hashlib.sha256()
    masked_text_digest = hashlib.sha256()
    counters: collections.Counter[str] = collections.Counter()
    source_rows: collections.Counter[str] = collections.Counter()
    source_text_tokens: collections.Counter[str] = collections.Counter()
    category_rows: collections.Counter[str] = collections.Counter()
    category_text_tokens: collections.Counter[str] = collections.Counter()

    context = mp.get_context("fork")
    with context.Pool(
        processes=int(args.workers),
        initializer=_worker_init,
        initargs=(str(tokenizer_json),),
    ) as pool, pq.ParquetWriter(
        temporary,
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    ) as writer:
        for row_group in range(source_file.metadata.num_row_groups):
            table = source_file.read_row_group(row_group)
            texts = table.column(text_index).to_pylist()
            sources = table.column(source_index).to_pylist()
            if any(not isinstance(text, str) for text in texts):
                raise ValueError(f"task {index} contains null or non-string text")
            if any(not isinstance(source, str) or source not in source_categories for source in sources):
                unknown = sorted({str(source) for source in sources if source not in source_categories})
                raise ValueError(f"task {index} has unknown source_dataset values: {unknown}")
            results = pool.imap(_mask_and_count, texts, chunksize=int(args.chunksize))
            masked_texts: list[str] = []
            for text, source, result in zip(texts, sources, results, strict=True):
                masked, pii, token_count = result
                update_text_digest(raw_text_digest, text)
                update_text_digest(masked_text_digest, masked)
                masked_texts.append(masked)
                category = source_categories[source]
                counters["input_rows"] += 1
                counters["output_rows"] += 1
                counters["changed_rows"] += int(masked != text)
                counters["text_tokens"] += int(token_count)
                for name in ("email", "ip", "iban"):
                    counters[f"{name}_matches"] += int(pii[name])
                source_rows[source] += 1
                source_text_tokens[source] += int(token_count)
                category_rows[category] += 1
                category_text_tokens[category] += int(token_count)
            replacement = pa.array(masked_texts, type=schema.field(text_index).type)
            transformed = table.set_column(text_index, schema.field(text_index), replacement)
            writer.write_table(transformed, row_group_size=transformed.num_rows)

    os.replace(temporary, output)
    if counters["input_rows"] != int(task["rows"]) or counters["output_rows"] != int(task["rows"]):
        raise RuntimeError("row preservation failed")

    output_file = pq.ParquetFile(output)
    if not schema.equals(output_file.schema_arrow, check_metadata=True):
        raise RuntimeError("Parquet schema or schema metadata changed")
    if output_file.metadata.num_rows != source_file.metadata.num_rows:
        raise RuntimeError("output Parquet row count changed")
    if output_file.metadata.num_row_groups != source_file.metadata.num_row_groups:
        raise RuntimeError("output Parquet row-group geometry changed")
    reread_digest = hashlib.sha256()
    for row_group in range(source_file.metadata.num_row_groups):
        before = source_file.read_row_group(row_group)
        after = output_file.read_row_group(row_group)
        if not before.drop(["text"]).equals(after.drop(["text"]), check_metadata=True):
            raise RuntimeError(f"non-text values changed in row group {row_group}")
        for text in after.column(text_index).to_pylist():
            update_text_digest(reread_digest, text)
    if reread_digest.hexdigest() != masked_text_digest.hexdigest():
        raise RuntimeError("written output text differs from the masked stream")

    receipt = {
        "schema_version": TASK_SCHEMA,
        "status": "passed",
        "completed_at": utc_now(),
        "task_index": index,
        "task_sha256": canonical_sha256(task),
        "contract_sha256": sha256_file(contract_path),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "input": task,
        "output": file_receipt(output, rows=int(counters["output_rows"]), relative_to=run_root / "release"),
        "invariants": {
            "row_count_preserved": True,
            "row_order_preserved": True,
            "schema_and_metadata_preserved": True,
            "all_non_text_values_equal": True,
            "only_text_replaced": True,
            "new_filtering": False,
            "new_deduplication": False,
        },
        "text_sha256": {
            "input_semantic": raw_text_digest.hexdigest(),
            "output_semantic": masked_text_digest.hexdigest(),
            "output_reread_semantic": reread_digest.hexdigest(),
        },
        "counts": dict(sorted(counters.items())),
        "source_rows": dict(sorted(source_rows.items())),
        "source_text_tokens": dict(sorted(source_text_tokens.items())),
        "category_rows": dict(sorted(category_rows.items())),
        "category_text_tokens": dict(sorted(category_text_tokens.items())),
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({"ok": True, "task": index, **receipt["counts"], "seconds": receipt["elapsed_seconds"]}, sort_keys=True))
    return 0


def transform_batch(args: argparse.Namespace) -> int:
    """Run one deterministic modulo lane inside a single debug allocation."""
    contract = load_contract(args.contract.resolve(), executing_code_root=args.code_root.resolve())
    run_root = Path(contract["run_root"]).resolve()
    stop_file = args.stop_file.resolve()
    expected_stop = run_root / "control" / "transform_stop_requested"
    if stop_file != expected_stop:
        raise ValueError(f"unexpected batch stop-file path: {stop_file}")
    lane = int(args.lane)
    lanes = int(args.lanes)
    if lanes < 1 or lane < 0 or lane >= lanes:
        raise ValueError("invalid transform lane geometry")
    completed = 0
    for index in range(lane, len(contract["tasks"]), lanes):
        if stop_file.exists():
            break
        transform(
            argparse.Namespace(
                code_root=args.code_root,
                contract=args.contract,
                task_index=index,
                workers=args.workers,
                chunksize=args.chunksize,
            )
        )
        completed += 1
    print(
        json.dumps(
            {
                "ok": True,
                "lane": lane,
                "lanes": lanes,
                "tasks_completed_or_resumed": completed,
                "stop_requested": stop_file.exists(),
            },
            sort_keys=True,
        )
    )
    return 0


def _sum_maps(receipts: Iterable[Mapping[str, Any]], key: str) -> collections.Counter[str]:
    total: collections.Counter[str] = collections.Counter()
    for receipt in receipts:
        for name, value in receipt.get(key, {}).items():
            total[str(name)] += int(value)
    return total


def _format_int(value: int) -> str:
    return f"{int(value):,}"


def _render_readme(config: Mapping[str, Any], category_rows: Mapping[str, int], category_tokens: Mapping[str, int], totals: Mapping[str, int]) -> str:
    tokenizer = config["tokenizer"]
    hplt = config["hplt_filter"]
    cutoff = config["glossapi_cutoff"]
    dedup = config["deduplication"]
    anonymization = config["anonymization"]
    lines = [
        "---",
        "license: other",
        "configs:",
        "- config_name: default",
        "  data_files: data/*.parquet",
        "---",
        "",
        "# GlossAPI Greek pretraining corpus v2",
        "",
        "## HPLT filtering method",
        "",
        f"The HPLT component is `{hplt['source_dataset']}`. It retains HPLT quality bins "
        f"{', '.join(str(value) for value in hplt['quality_bins'])} (GE8), uses the pre-applied no-MT/register filter, "
        f"requires `greek_badness_score <= {hplt['greek_badness_score_max']}`, and applies {hplt['cleaning']}. "
        f"The standalone filtered slice contained {_format_int(hplt['pre_dedup_rows'])} documents; "
        f"{_format_int(hplt['post_dedup_rows'])} remain after corpus-wide deduplication.",
        "",
        "## GlossAPI datasets and token counts",
        "",
        f"GlossAPI source-audit cutoff: `{cutoff['timestamp']}`. {cutoff['semantics']} "
        "The table reports the admitted, surviving datasets grouped by their actual subject matter.",
        "",
        f"Counts use [`{tokenizer['repository_id']}`](https://huggingface.co/{tokenizer['repository_id']}/tree/{tokenizer['revision']}/{tokenizer['subfolder']}) "
        f"at `{tokenizer['revision']}` ({_format_int(tokenizer['vocab_size'])} tokens). Training-token counts include one EOD token per document.",
        "",
        "| Category | Included source datasets | Documents | Training tokens |",
        "|---|---|---:|---:|",
    ]
    for category in config["source_categories"]:
        category_id = category["id"]
        sources = "<br>".join(f"`{source}`" for source in category["sources"])
        rows = int(category_rows.get(category_id, 0))
        tokens = int(category_tokens.get(category_id, 0)) + rows
        lines.append(f"| {category['label']} | {sources} | {_format_int(rows)} | {_format_int(tokens)} |")
    lines.extend(
        [
            f"| **Total** | **37 source datasets** | **{_format_int(totals['rows'])}** | **{_format_int(totals['training_tokens'])}** |",
            "",
            "## Deduplication",
            "",
            f"The combined {_format_int(dedup['input_rows'])}-document corpus was deduplicated once before this release; "
            f"{_format_int(dedup['removed_rows'])} documents were removed and {_format_int(dedup['retained_rows'])} retained. "
            "Matching combined strict UTF-8 text hashes, conservative normalized-exact hashes "
            f"({dedup['normalization']}), and MinHash candidates built from {dedup['minhash']['token_shingles']}-token Greek shingles "
            f"({dedup['minhash']['permutations']} permutations, {dedup['minhash']['bands']} x {dedup['minhash']['hashes_per_band']} bands). "
            f"Near-duplicate candidates were removed only after exact Jaccard verification at >= {dedup['minhash']['verified_jaccard_threshold']}. "
            "Representatives preferred the original Nanochat corpus, then lower Greek/mojibake badness, lower cleaner loss, longer retained text, and stable source identity/order. "
            "No additional deduplication was performed during anonymization.",
            "",
            "## Anonymization to Apertus standards",
            "",
            f"The `text` field was anonymized with the {anonymization['standard']} policy from upstream commit "
            f"`{anonymization['upstream_commit']}`: email addresses, IP addresses, and validated IBANs are replaced by "
            f"`{anonymization['replacements']['email']}`, `{anonymization['replacements']['ip']}`, and `{anonymization['replacements']['iban']}`. "
            f"IBAN matching adds {anonymization['iban_extension']}. Every row, its order and multiplicity, the schema, source identity, and every non-text value are preserved; provenance metadata is retained.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(args: argparse.Namespace) -> int:
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path, executing_code_root=args.code_root.resolve())
    run_root = Path(contract["run_root"])
    config = read_json(args.code_root.resolve() / contract["config"]["path"])
    validate_config(config)
    receipts: list[dict[str, Any]] = []
    output_files: list[dict[str, Any]] = []
    for task in contract["tasks"]:
        index = int(task["task_index"])
        receipt_path = _receipt_path(run_root, index)
        if not receipt_path.is_file():
            raise FileNotFoundError(receipt_path)
        receipt = read_json(receipt_path)
        output = _output_path(run_root, task)
        _validate_resumed_task(receipt, task, output)
        if receipt.get("contract_sha256") != sha256_file(contract_path):
            raise ValueError(f"task {index} contract binding drift")
        if not _task_invariants_pass(receipt):
            raise ValueError(f"task {index} invariant failed")
        receipts.append(receipt)
        output_files.append({**receipt["output"], "task_index": index})

    counts = _sum_maps(receipts, "counts")
    source_rows = _sum_maps(receipts, "source_rows")
    source_tokens = _sum_maps(receipts, "source_text_tokens")
    category_rows = _sum_maps(receipts, "category_rows")
    category_tokens = _sum_maps(receipts, "category_text_tokens")
    expected_sources = {key: int(value) for key, value in config["expected_source_rows"].items()}
    if dict(source_rows) != expected_sources:
        raise ValueError(
            "final source rows differ from the pinned release: "
            f"missing_or_changed={sorted(key for key in set(expected_sources) | set(source_rows) if expected_sources.get(key) != source_rows.get(key))}"
        )
    rows = int(config["input"]["rows"])
    if counts["input_rows"] != rows or counts["output_rows"] != rows:
        raise ValueError("final row accounting does not close")
    if counts["changed_rows"] < 1 or sum(counts[f"{name}_matches"] for name in ("email", "ip", "iban")) < 1:
        raise ValueError("anonymization produced no matches; refusing a likely no-op release")
    if sum(source_tokens.values()) != counts["text_tokens"] or sum(category_tokens.values()) != counts["text_tokens"]:
        raise ValueError("token accounting does not close")
    if sum(category_rows.values()) != rows:
        raise ValueError("category row accounting does not close")
    totals = {
        "rows": rows,
        "text_tokens": counts["text_tokens"],
        "eod_tokens": rows * int(config["tokenizer"]["eod_tokens_per_document"]),
        "training_tokens": counts["text_tokens"] + rows,
    }
    release_root = run_root / "release"
    manifests = release_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    token_counts = {
        "schema_version": "glossapi_hf_v2_anonymized_token_counts_v1",
        "status": "passed",
        "created_at": utc_now(),
        "tokenizer": config["tokenizer"],
        "totals": totals,
        "source_rows": dict(sorted(source_rows.items())),
        "source_text_tokens": dict(sorted(source_tokens.items())),
        "category_rows": dict(sorted(category_rows.items())),
        "category_text_tokens": dict(sorted(category_tokens.items())),
    }
    token_path = manifests / "token_counts.json"
    write_json_atomic(token_path, token_counts)
    readme_path = release_root / "README.md"
    readme_path.write_text(
        _render_readme(config, category_rows, category_tokens, totals), encoding="utf-8"
    )
    manifest = {
        "schema_version": FINAL_SCHEMA,
        "status": "passed",
        "completed_at": utc_now(),
        "contract": file_receipt(contract_path),
        "finalizer_overlay": {
            "executing_pipeline": file_receipt(Path(__file__)),
            "executing_release_common": file_receipt(HERE / "release_common.py"),
            "contract_code_root": str(args.code_root.resolve()),
        },
        "input": contract["input"],
        "tokenizer": contract["tokenizer"],
        "policy": contract["policy"],
        "counts": {
            **totals,
            "shards": len(output_files),
            "changed_rows": counts["changed_rows"],
            "email_matches": counts["email_matches"],
            "ip_matches": counts["ip_matches"],
            "iban_matches": counts["iban_matches"],
        },
        "source_rows": dict(sorted(source_rows.items())),
        "category_rows": dict(sorted(category_rows.items())),
        "files": output_files,
        "metadata": {
            "README.md": file_receipt(readme_path, relative_to=release_root),
            "manifests/token_counts.json": file_receipt(token_path, relative_to=release_root),
        },
        "invariants": {
            "all_431_shards_present_and_hashed": True,
            "all_51839746_rows_preserved": True,
            "all_37_source_counts_match": True,
            "only_text_changed": True,
            "schema_and_all_non_text_values_preserved": True,
            "no_new_filtering": True,
            "no_new_deduplication": True,
            "no_new_decontamination": True,
        },
    }
    manifest_path = manifests / "anonymization_manifest.json"
    # Its checksum is recorded by the publication state/receipt rather than
    # recursively embedded in the manifest itself.
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(manifest_path),
                "rows": rows,
                "training_tokens": totals["training_tokens"],
                "changed_rows": counts["changed_rows"],
                "pii_matches": sum(counts[f"{name}_matches"] for name in ("email", "ip", "iban")),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("prepare")
    command.add_argument("--code-root", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--input-root", type=Path, required=True)
    command.add_argument("--tokenizer-json", type=Path, required=True)
    command.add_argument("--dedup-receipt", type=Path, required=True)
    command.add_argument("--source-breakdown", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.set_defaults(func=prepare)

    command = subparsers.add_parser("transform")
    command.add_argument("--code-root", type=Path, required=True)
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.add_argument("--workers", type=int, default=64)
    command.add_argument("--chunksize", type=int, default=8)
    command.set_defaults(func=transform)

    command = subparsers.add_parser("finalize")
    command.add_argument("--code-root", type=Path, required=True)
    command.add_argument("--contract", type=Path, required=True)
    command.set_defaults(func=finalize)

    command = subparsers.add_parser("transform-batch")
    command.add_argument("--code-root", type=Path, required=True)
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--lane", type=int, required=True)
    command.add_argument("--lanes", type=int, required=True)
    command.add_argument("--workers", type=int, default=64)
    command.add_argument("--chunksize", type=int, default=8)
    command.add_argument("--stop-file", type=Path, required=True)
    command.set_defaults(func=transform_batch)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
