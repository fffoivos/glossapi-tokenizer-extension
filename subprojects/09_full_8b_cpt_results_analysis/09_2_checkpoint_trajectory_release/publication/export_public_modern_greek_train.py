#!/usr/bin/env python3
"""Export the exact selected Modern-Greek training documents as public Parquet.

The source is revision-pinned ``glossapi-greek-nanochat-pretraining-dataset-v2``.
Selection is rebuilt solely from the full-8B stage's immutable catalog45 and
content57 evidence.  Rows retain the source schema and values verbatim; an
identity or text-hash mismatch fails rather than silently applying a new
deduplication or anonymization policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any


CATALOG_BYTES = 45
CONTENT_BYTES = 57
EXPECTED_DOCUMENTS = {"hplt_new_greek": 46_535_439, "non_hplt_new_greek": 3_086_052}
EXPECTED_ACTIVE_TOKENS = {"hplt_new_greek": 41_512_804_679, "non_hplt_new_greek": 19_068_732_797}
EXPECTED_REPO = "fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2"

# Under Linux ``fork`` lets many Parquet workers share the read-only table.
_EXPECTED: dict[bytes, tuple[bytes, str]] | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_catalog_identities(path: Path) -> set[bytes]:
    raw = path.read_bytes()
    require(len(raw) % CATALOG_BYTES == 0, f"catalog byte length drift: {path}")
    return {raw[offset + 13 : offset + 29] for offset in range(0, len(raw), CATALOG_BYTES)}


def selected_expected(stage_root: Path) -> dict[bytes, tuple[bytes, str]]:
    selected: dict[bytes, str] = {}
    for pool in EXPECTED_DOCUMENTS:
        ids = read_catalog_identities(stage_root / "inventory" / "catalog" / f"{pool}.source_local_selected.catalog45")
        require(len(ids) == EXPECTED_DOCUMENTS[pool], f"selected catalog document count drift: {pool}")
        for identity in ids:
            require(identity not in selected, "selected identity occurs in more than one Modern-Greek pool")
            selected[identity] = pool
    content_path = stage_root / "inventory" / "raw" / "modern.content57"
    with content_path.open("rb") as handle:
        while True:
            row = handle.read(CONTENT_BYTES)
            if not row:
                break
            require(len(row) == CONTENT_BYTES, "modern.content57 is truncated")
            identity = row[41:57]
            pool = selected.get(identity)
            if pool is not None:
                content = row[:32]
                prior = selected[identity]
                require(isinstance(prior, str), "duplicate modern identity in content receipt")
                selected[identity] = (content, pool)  # type: ignore[assignment]
    missing = [identity.hex() for identity, value in selected.items() if isinstance(value, str)]
    require(not missing, f"selected identities have no content57 binding (first: {missing[:3]})")
    result = {identity: value for identity, value in selected.items() if isinstance(value, tuple)}
    require(len(result) == sum(EXPECTED_DOCUMENTS.values()), "selected modern document total drift")
    return result


def source_identity(source_doc_id: Any, text: Any) -> tuple[bytes, bytes]:
    require(source_doc_id is not None and text is not None, "source row has null source_doc_id or text")
    text_bytes = str(text).encode("utf-8")
    text_sha = hashlib.sha256(text_bytes).digest()
    identity = hashlib.sha256(str(source_doc_id).encode("utf-8") + b"\0" + text_sha).digest()[:16]
    return identity, text_sha


def _init_worker(expected: dict[bytes, tuple[bytes, str]]) -> None:
    global _EXPECTED
    _EXPECTED = expected


def export_one(task: tuple[str, str, int]) -> dict[str, Any]:
    source_value, output_value, batch_size = task
    source, output = Path(source_value), Path(output_value)
    expected = _EXPECTED
    require(expected is not None, "export worker was not initialized")
    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(source)
    names = reader.schema_arrow.names
    require("source_doc_id" in names and "text" in names, f"source schema lacks exact identity columns: {source}")
    source_index, text_index = names.index("source_doc_id"), names.index("text")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    writer: pq.ParquetWriter | None = None
    counts = {pool: 0 for pool in EXPECTED_DOCUMENTS}
    matched = 0
    try:
        for batch in reader.iter_batches(batch_size=batch_size):
            doc_ids, texts = batch.column(source_index), batch.column(text_index)
            rows: list[int] = []
            for index in range(batch.num_rows):
                identity, text_sha = source_identity(doc_ids[index].as_py(), texts[index].as_py())
                binding = expected.get(identity)
                if binding is None:
                    continue
                require(text_sha == binding[0], f"selected source text hash drift: {source}:{index}")
                counts[binding[1]] += 1
                matched += 1
                rows.append(index)
            if rows:
                table = pa.Table.from_batches([batch.take(pa.array(rows, type=pa.int64()))])
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary, output)
        else:
            temporary.unlink(missing_ok=True)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return {"source_relative_path": source.name, "output_relative_path": output.name if matched else None, "rows": matched, "pool_rows": counts, "bytes": output.stat().st_size if matched else 0, "sha256": sha256_file(output) if matched else None}


def snapshot(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    require(bool(token), "HF_TOKEN must be injected per command")
    require(args.source_repo == EXPECTED_REPO, "unexpected public source repository")
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    info = api.repo_info(repo_id=args.source_repo, repo_type="dataset", revision=args.source_revision)
    target = Path(snapshot_download(repo_id=args.source_repo, repo_type="dataset", revision=args.source_revision, allow_patterns=["data/*.parquet"], local_dir=args.cache_root, token=token, max_workers=args.workers)).resolve()
    parquet = sorted((target / "data").glob("*.parquet"))
    require(parquet, "source snapshot contains no data Parquet files")
    result = {"schema_version": "apertus_full8_modern_greek_source_snapshot_v1", "status": "completed", "repo_id": args.source_repo, "requested_revision": args.source_revision, "resolved_revision": info.sha, "snapshot_root": str(target), "files": [{"relative_path": path.relative_to(target).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in parquet]}
    write_json(args.output, result)
    return result


def export(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_snapshot.resolve()
    stage_root = args.training_stage.resolve()
    output = args.output_root.resolve()
    require(not output.exists(), f"refusing to overwrite immutable public snapshot: {output}")
    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    require(receipt.get("schema_version") == "apertus_full8_modern_greek_source_snapshot_v1" and receipt.get("status") == "completed", "source snapshot receipt drift")
    require(Path(str(receipt.get("snapshot_root", ""))).resolve() == source_root, "source snapshot root drift")
    expected = selected_expected(stage_root)
    source_files = sorted((source_root / "data").glob("*.parquet"))
    require(source_files, "source snapshot data directory is empty")
    output.mkdir(parents=True)
    tasks = [(str(source), str(output / "data" / source.name), args.batch_size) for source in source_files]
    context = mp.get_context("fork")
    with context.Pool(processes=args.workers, initializer=_init_worker, initargs=(expected,)) as pool:
        outputs = list(pool.imap_unordered(export_one, tasks))
    counts = {pool: 0 for pool in EXPECTED_DOCUMENTS}
    for row in outputs:
        for pool, value in row["pool_rows"].items():
            counts[pool] += int(value)
    require(counts == EXPECTED_DOCUMENTS, f"public export row accounting drift: {counts}")
    generated = sorted((output / "data").glob("*.parquet"))
    require(generated, "public export produced no Parquet files")
    readme = "\n".join((
        "---", "language: el", "license: apache-2.0", "---", "",
        "# Exact Modern-Greek training content for Apertus 8B Greek CPT", "",
        "This is the public Modern-Greek, train-only document snapshot selected for the full 8B D0 continued-pretraining run. It preserves the upstream v2 schema and row values; selection is reconstructed from the immutable post-mask training catalogs and content hashes. It contains no replay payload.", "",
        "## Exact selected content", "",
        f"- HPLT Modern Greek: {EXPECTED_DOCUMENTS['hplt_new_greek']:,} documents; {EXPECTED_ACTIVE_TOKENS['hplt_new_greek']:,} active training tokens.",
        f"- GlossAPI/non-HPLT Modern Greek: {EXPECTED_DOCUMENTS['non_hplt_new_greek']:,} documents; {EXPECTED_ACTIVE_TOKENS['non_hplt_new_greek']:,} active training tokens.",
        f"- Total: {sum(EXPECTED_DOCUMENTS.values()):,} documents; {sum(EXPECTED_ACTIVE_TOKENS.values()):,} active training tokens.", "",
        "## Processing and provenance", "",
        "The upstream corpus was revision-pinned. The training workflow applied heldout exclusion, GreekMMLU decontamination, Apertus-standard PII masking, and global exact post-mask deduplication before tokenization. This release does not perform an additional anonymization, deduplication, text transformation, or retokenization; it only selects the exact Modern-Greek documents that the frozen training catalogs identify.", "",
        "The companion private dataset `fffoivos/apertus-8b-greek-cpt-d0-full-mix` contains the complete packed 79/20/1 mixture and its restricted replay provenance. This public dataset does not grant redistribution rights for those replay sources.", "",
    ))
    (output / "README.md").write_text(readme, encoding="utf-8")
    inventory = [{"relative_path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in generated]
    source_binding = {"path": str(args.source_receipt.resolve()), "sha256": sha256_file(args.source_receipt), "resolved_revision": receipt["resolved_revision"]}
    manifest = {"schema_version": "apertus_full8_modern_greek_train_snapshot_v1", "status": "verified", "visibility": "public", "source": source_binding, "training_stage": {"path": str(stage_root), "catalogs": ["inventory/catalog/hplt_new_greek.source_local_selected.catalog45", "inventory/catalog/non_hplt_new_greek.source_local_selected.catalog45"], "content_receipt": "inventory/raw/modern.content57"}, "selection": {"row_counts": counts, "active_tokens": EXPECTED_ACTIVE_TOKENS, "total_documents": sum(counts.values()), "total_active_tokens": sum(EXPECTED_ACTIVE_TOKENS.values()), "identity": "sha256(str(source_doc_id) UTF-8 || NUL || sha256(text UTF-8))[:16]", "content_hash_verified": True}, "upload_payload_inventory": [{"relative_path": "README.md", "bytes": (output / "README.md").stat().st_size, "sha256": sha256_file(output / "README.md")}, *inventory], "shards": sorted(outputs, key=lambda row: row["source_relative_path"])}
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--source-repo", default=EXPECTED_REPO)
    snap.add_argument("--source-revision", required=True)
    snap.add_argument("--cache-root", type=Path, required=True)
    snap.add_argument("--workers", type=int, default=16)
    snap.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("export")
    build.add_argument("--source-snapshot", type=Path, required=True)
    build.add_argument("--source-receipt", type=Path, required=True)
    build.add_argument("--training-stage", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--workers", type=int, default=32)
    build.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    require(1 <= args.workers <= 96, "workers must be in [1,96]")
    result = snapshot(args) if args.command == "snapshot" else export(args)
    print(json.dumps({"ok": True, "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
