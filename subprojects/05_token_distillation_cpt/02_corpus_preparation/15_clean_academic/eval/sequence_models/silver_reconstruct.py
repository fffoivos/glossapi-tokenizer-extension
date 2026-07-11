#!/usr/bin/env python3
"""Audit and reconstruct the maximum defensible LLM-silver sequence evidence."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .contract import (
    GoldDocument,
    build_split_manifest,
    canonical_json_sha256,
    parse_gold_rows,
    read_gold,
    sha256_file,
    validate_silver,
)
from .evaluate import read_predictions

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
LINE_RE = re.compile(r"^L(\d+):\s?(.*)$")
TOKENIZER_SHA256 = "358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394"
TOKENIZER_REVISION = "a4826df7f76b54cdd6dc21d09fe97283c466999b"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())


def _require_new_outputs(*paths: str | Path) -> None:
    existing = [str(path) for path in map(Path, paths) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite immutable outputs: {existing}")


def _atomic_validated_silver(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Stream JSONL through the strict contract, then atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            def documents() -> Iterator[GoldDocument]:
                for row in rows:
                    payload = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                    handle.write(payload)
                    digest.update(payload)
                    yield parse_gold_rows([row])[0]

            receipt = validate_silver(documents(), policy, split_manifest=split_manifest)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        return receipt, digest.hexdigest()
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _hash(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()


class ExactTokenizer:
    def __init__(self, path: Path) -> None:
        actual = sha256_file(path)
        if actual != TOKENIZER_SHA256:
            raise ValueError(f"ModernGreek-148k hash mismatch: expected {TOKENIZER_SHA256}, got {actual}")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - Clariden/runtime hook
            raise RuntimeError("install the pinned tokenizers runtime to hydrate silver") from exc
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def counts(self, texts: Sequence[str]) -> list[int]:
        return [len(item.ids) for item in self.tokenizer.encode_batch(list(texts), add_special_tokens=False)]


def _annotation_family(path: Path) -> list[dict[str, Any]]:
    value = _load_json(path)
    return value["annotations"] if isinstance(value, dict) else value


def _json_record_count(path: Path) -> int | None:
    try:
        if path.suffix == ".jsonl":
            return sum(1 for _ in _iter_jsonl(path))
        value = _load_json(path)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("annotations", "batches", "assignments"):
            if isinstance(value.get(key), (list, dict)):
                return len(value[key])
    return None


def audit_tracked() -> dict[str, Any]:
    units = EVAL_DIR / "units"
    families = [
        ("span_windows", units / "SPAN_manifest.jsonl", EVAL_DIR / "annotations_span" / "all.json"),
        ("section_scale", units / "B_scale_manifest.jsonl", EVAL_DIR / "annotations_scale" / "all.json"),
        ("boundary", units / "A_boundary_manifest.jsonl", EVAL_DIR / "annotations_scale_A" / "all.json"),
        ("goal_sections", units / "G_sections_manifest.jsonl", EVAL_DIR / "annotations_goal_sections" / "all.json"),
        ("goal_windows_opus", units / "W_windows_manifest.jsonl", EVAL_DIR / "annotations_goal_windows" / "all.json"),
        ("goal_windows_haiku", units / "W_windows_manifest.jsonl", EVAL_DIR / "annotations_goal_windows_haiku" / "all.json"),
    ]
    reports: dict[str, Any] = {}
    for name, manifest_path, annotation_path in families:
        manifest = {row["unit_id"]: row for row in _jsonl(manifest_path)}
        annotations = {row["unit_id"]: row for row in _annotation_family(annotation_path)}
        joined = sorted(set(manifest) & set(annotations))
        missing = sorted(set(manifest) - set(annotations))
        extra = sorted(set(annotations) - set(manifest))
        documents_by_source: dict[str, set[str]] = collections.defaultdict(set)
        for unit in joined:
            if manifest[unit].get("doc_id"):
                documents_by_source[str(manifest[unit].get("source"))].add(
                    str(manifest[unit]["doc_id"])
                )
        reports[name] = {
            "manifest_rows": len(manifest),
            "annotation_rows": len(annotations),
            "joined_rows": len(joined),
            "missing_annotation_count": len(missing),
            "missing_annotation_ids_sha256": canonical_json_sha256(missing),
            "missing_annotation_first_last": [missing[0], missing[-1]] if missing else [],
            "extra_annotation_count": len(extra),
            "extra_annotation_ids_sha256": canonical_json_sha256(extra),
            "source_rows": dict(sorted(collections.Counter(
                str(manifest[unit].get("source")) for unit in joined
            ).items())),
            "source_documents": {
                source: len(document_ids)
                for source, document_ids in sorted(documents_by_source.items())
            },
            "document_count": len({manifest[unit].get("doc_id") for unit in joined if manifest[unit].get("doc_id")}),
            "manifest_sha256": sha256_file(manifest_path),
            "annotations_sha256": sha256_file(annotation_path),
        }
    span_annotations = _annotation_family(EVAL_DIR / "annotations_span" / "all.json")
    span_manifest = {row["unit_id"]: row for row in _jsonl(units / "SPAN_manifest.jsonl")}
    valid_span = [row for row in span_annotations if row.get("unit_id") in span_manifest]
    batchpaths = _load_json(units / "SPAN_batchpaths.json")
    expected_batches = sorted(Path(path).name for path in batchpaths)
    present_batches = sorted(path.name for path in (units / "SPAN").glob("batch_*.json"))
    missing_batches = sorted(set(expected_batches) - set(present_batches))
    scale = reports["section_scale"]
    annotation_files = sorted(
        path for directory in EVAL_DIR.glob("annotations*") if directory.is_dir()
        for path in directory.glob("*.json")
    )
    unit_files = sorted(
        path for path in units.iterdir() if path.is_file() and path.name != ".gitignore"
    )
    model_files = [
        "span_line_lr_model.json", "span_line_lr_struct_model.json",
        "span_line_lr_syn_model.json", "toc_line_lr_model.json",
        "span_smooth_params.json", "struct_smooth_params.json",
        "window_lr_model.json", "beta_gate_model.json", "feature_separability.json",
    ]
    reconstruction_scripts = [
        "build_span_dataset.py", "build_span_units.py", "span_loop_step.py",
        "span_seq_data.py", "span_signals.py", "line_lr.py", "train_struct.py",
        "struct_lines.py", "decode_spans.py", "decode_struct.py",
    ]
    report = {
        "schema_version": "academic-structure-silver-inventory-v1",
        "evidence_status": "LLM_silver",
        "production_eligible": False,
        "families": reports,
        "all_annotation_file_receipts": {
            str(path.relative_to(EVAL_DIR)): {
                "sha256": sha256_file(path), "records": _json_record_count(path)
            }
            for path in annotation_files
        },
        "all_unit_file_receipts": {
            str(path.relative_to(EVAL_DIR)): {
                "sha256": sha256_file(path), "records": _json_record_count(path)
            }
            for path in unit_files
        },
        "sequence_evidence": {
            "bibliography_span_annotations": sum(len(row.get("spans", [])) for row in valid_span),
            "annotated_windows": len(valid_span),
            "documents": reports["span_windows"]["document_count"],
            "source_windows": reports["span_windows"]["source_rows"],
            "expected_text_batch_files": len(expected_batches),
            "present_text_batch_files": len(present_batches),
            "missing_text_batch_file_count": len(missing_batches),
            "missing_text_batch_names_sha256": canonical_json_sha256(missing_batches),
            "missing_text_batch_first_last": (
                [missing_batches[0], missing_batches[-1]] if missing_batches else []
            ),
            "fit_ready_line_rows": 0 if missing_batches else "requires_hydration",
            "task_scope_if_hydrated": "bibliography_binary_windows",
            "toc_supervision_available": False,
        },
        "section_snippet_evidence": {
            "rows": scale["joined_rows"],
            "source_rows": scale["source_rows"],
            "sequence_fit_eligible": False,
            "reason": "evidence_quote snippets have no work/document identity or full section text",
        },
        "tracked_feature_model_artifacts": {
            name: {
                "sha256": sha256_file(EVAL_DIR / name),
                "reconstruction_role": (
                    "frozen_comparison_baseline" if name in {
                        "span_line_lr_model.json", "toc_line_lr_model.json",
                        "struct_smooth_params.json",
                    } else "research_or_diagnostic_only"
                ),
                "raw_supervision_embedded": False,
            }
            for name in model_files
        },
        "tracked_reconstruction_script_receipts": {
            name: sha256_file(EVAL_DIR / name) for name in reconstruction_scripts
        },
        "legacy_struct_2k": {
            "gold_path": "units/STRUCT_2K_gold.jsonl",
            "gold_present": (units / "STRUCT_2K_gold.jsonl").is_file(),
            "run_directory": "units/STRUCT_2K",
            "run_directory_present": (units / "STRUCT_2K").is_dir(),
            "required_for_bib_toc_reconstruction": [
                "STRUCT_2K_gold.jsonl",
                "or all STRUCT_2K/batch_*.json plus ann_*.json files",
            ],
        },
        "tracked_positive_span_ledger": {
            "path": "span_dataset.jsonl",
            "rows": sum(1 for _ in _iter_jsonl(EVAL_DIR / "span_dataset.jsonl")),
            "sha256": sha256_file(EVAL_DIR / "span_dataset.jsonl"),
        },
    }
    report["inventory_sha256"] = canonical_json_sha256(report)
    return report


def _load_units(unit_dir: Path, expected_paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    expected = {Path(path).name for path in expected_paths}
    actual_paths = sorted(unit_dir.glob("batch_*.json"))
    actual = {path.name for path in actual_paths}
    if actual != expected:
        raise ValueError(
            f"SPAN batch inventory mismatch: missing={sorted(expected-actual)[:20]}, "
            f"extra={sorted(actual-expected)[:20]}"
        )
    units: dict[str, dict[str, Any]] = {}
    for path in actual_paths:
        rows = _load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: batch root must be a list")
        for row in rows:
            unit_id = str(row.get("unit_id", ""))
            if not unit_id or unit_id in units:
                raise ValueError(f"{path}: empty/duplicate unit_id {unit_id!r}")
            if not isinstance(row.get("text_numbered"), str):
                raise ValueError(f"{path}:{unit_id}: missing text_numbered")
            units[unit_id] = row
    return units


def _line_rows(text_numbered: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for raw in text_numbered.splitlines():
        match = LINE_RE.match(raw)
        if match and match.group(2).strip():
            rows.append((int(match.group(1)), match.group(2)))
    return rows


@dataclass
class SilverDraft:
    upstream_doc_id: str
    source: str
    n_physical_lines: int
    lines: dict[int, str]
    bib_lines: set[int]
    annotation_units: list[str]


@dataclass(frozen=True)
class Identity:
    document_id: str
    work_id: str
    source: str


def _cluster_exact_work(rows: list[dict[str, Any]]) -> None:
    hashes = collections.Counter(row["observed_text_sha256"] for row in rows)
    for row in rows:
        if hashes[row["observed_text_sha256"]] > 1:
            row["work_id"] = _hash("academic-silver-exact-work-v1", row["observed_text_sha256"])


def _token_counts(tokenizer: ExactTokenizer, texts: Sequence[str]) -> list[int]:
    result: list[int] = []
    for start in range(0, len(texts), 512):
        result.extend(tokenizer.counts(texts[start:start + 512]))
    return result


def hydrate_span(args: argparse.Namespace) -> int:
    _require_new_outputs(args.output, args.split_manifest, args.receipt)
    config = _load_json(Path(args.config))
    manifest_rows = _jsonl(EVAL_DIR / "units" / "SPAN_manifest.jsonl")
    manifest = {row["unit_id"]: row for row in manifest_rows}
    annotations = {
        row["unit_id"]: row for row in _annotation_family(EVAL_DIR / "annotations_span" / "all.json")
    }
    expected = _load_json(EVAL_DIR / "units" / "SPAN_batchpaths.json")
    units = _load_units(Path(args.unit_dir), expected)
    missing_annotation = sorted(set(manifest) - set(annotations))
    joined = sorted(set(manifest) & set(annotations) & set(units))
    if set(units) != set(manifest):
        raise ValueError(
            f"unit payload/manifest mismatch: missing={len(set(manifest)-set(units))}, "
            f"extra={len(set(units)-set(manifest))}"
        )
    drafts: dict[tuple[str, str], SilverDraft] = {}
    for unit_id in joined:
        meta = manifest[unit_id]
        annotation = annotations[unit_id]
        key = (str(meta["source"]), str(meta["doc_id"]))
        draft = drafts.setdefault(key, SilverDraft(
            upstream_doc_id=key[1], source=key[0], n_physical_lines=int(meta["win_hi"]),
            lines={}, bib_lines=set(), annotation_units=[],
        ))
        draft.n_physical_lines = max(draft.n_physical_lines, int(meta["win_hi"]))
        draft.annotation_units.append(unit_id)
        for abs_idx, text in _line_rows(units[unit_id]["text_numbered"]):
            previous = draft.lines.setdefault(abs_idx, text)
            if previous != text:
                raise ValueError(f"{unit_id}: conflicting text at absolute line {abs_idx}")
        for span in annotation.get("spans", []):
            start, end = int(span["start_line"]), int(span["end_line"])
            if end < start:
                raise ValueError(f"{unit_id}: invalid silver span {start}..{end}")
            draft.bib_lines.update(range(start, end + 1))
    tokenizer = ExactTokenizer(Path(args.tokenizer_json))
    rows: list[dict[str, Any]] = []
    for (source, upstream_id), draft in sorted(drafts.items()):
        line_items = sorted(draft.lines.items())
        texts = [text for _, text in line_items]
        counts = _token_counts(tokenizer, texts)
        document_id = _hash("academic-silver-document-v1", source, upstream_id)
        observed_sha = canonical_json_sha256(line_items)
        lines = [
            {
                "line_id": _hash("academic-silver-line-v1", document_id, abs_idx, text),
                "abs_idx": abs_idx,
                "text": text,
                "label": "BIB" if abs_idx in draft.bib_lines else "O",
                "token_count": count,
                "is_running_prose": None,
            }
            for (abs_idx, text), count in zip(line_items, counts)
        ]
        rows.append({
            "schema_version": "academic-structure-gold-v1",
            "document_id": document_id,
            "work_id": _hash("academic-silver-work-v1", source, upstream_id),
            "representation_id": _hash(
                "academic-silver-representation-v1", source, upstream_id, observed_sha
            ),
            "source": source,
            "coverage": "annotated_windows",
            "n_physical_lines": draft.n_physical_lines,
            "n_present_lines": len(lines),
            "annotation": {
                "status": "LLM_silver",
                "engine": "Claude Opus span-annotation workflow",
                "task_scope": "bibliography_binary_windows",
                "annotator_ids": ["LLM:Claude-Opus"],
                "adjudicator_id": None,
                "unit_ids": sorted(draft.annotation_units),
                "source_annotations_sha256": sha256_file(EVAL_DIR / "annotations_span" / "all.json"),
                "toc_supervised": False,
            },
            "tokenizer": {"id": "ModernGreek-148k", "revision": TOKENIZER_REVISION},
            "upstream_document_id": upstream_id,
            "observed_text_sha256": observed_sha,
            "lines": lines,
        })
    _cluster_exact_work(rows)
    identities = [Identity(row["document_id"], row["work_id"], row["source"]) for row in rows]
    split_manifest = build_split_manifest(identities, config["split"])
    for row in rows:
        row["split"] = split_manifest["assignments"][row["document_id"]]
    split_path = Path(args.split_manifest)
    receipt, gold_sha256 = _atomic_validated_silver(
        Path(args.output), rows, config["silver_contract"], split_manifest
    )
    receipt.update({
        "gold_sha256": gold_sha256,
        "missing_annotation_unit_ids": missing_annotation,
        "source_manifest_sha256": sha256_file(EVAL_DIR / "units" / "SPAN_manifest.jsonl"),
        "source_batches_inventory_sha256": canonical_json_sha256(
            sorted((path.name, sha256_file(path)) for path in Path(args.unit_dir).glob("batch_*.json"))
        ),
    })
    _write_json(split_path, split_manifest)
    receipt["split_manifest_sha256"] = sha256_file(split_path)
    _write_json(Path(args.receipt), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _legacy_metadata(path: Path) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for row_number, row in enumerate(_iter_jsonl(path), 1):
        lines = row.get("lines")
        if not isinstance(lines, list) or not lines:
            raise ValueError(f"legacy row {row_number}: missing lines")
        source, upstream_id = str(row.get("source", "unknown")), str(row.get("doc_id", ""))
        if not upstream_id:
            raise ValueError(f"legacy row {row_number}: missing doc_id")
        observed = canonical_json_sha256([[int(item[0]), str(item[1])] for item in lines])
        metadata.append({
            "source": source, "upstream_id": upstream_id, "observed_text_sha256": observed,
            "document_id": _hash("academic-silver-document-v1", source, upstream_id),
            "work_id": _hash("academic-silver-work-v1", source, upstream_id),
        })
    _cluster_exact_work(metadata)
    return metadata


def import_legacy(args: argparse.Namespace) -> int:
    _require_new_outputs(args.output, args.split_manifest, args.receipt)
    path = Path(args.input)
    config = _load_json(Path(args.config))
    metadata = _legacy_metadata(path)
    lookup = {(row["source"], row["upstream_id"]): row for row in metadata}
    identities = [Identity(row["document_id"], row["work_id"], row["source"]) for row in metadata]
    split_manifest = build_split_manifest(identities, config["split"])
    tokenizer = ExactTokenizer(Path(args.tokenizer_json))
    legacy_sha256 = sha256_file(path)
    label_map = {0: "O", 1: "BIB", 2: "TOC", "0": "O", "1": "BIB", "2": "TOC"}
    def converted_rows() -> Iterator[dict[str, Any]]:
        for row in _iter_jsonl(path):
            source, upstream_id = str(row.get("source", "unknown")), str(row["doc_id"])
            identity = lookup[(source, upstream_id)]
            legacy_lines = row["lines"]
            texts = [str(item[1]) for item in legacy_lines]
            counts = _token_counts(tokenizer, texts)
            lines = []
            for item, count in zip(legacy_lines, counts):
                label = label_map.get(item[2])
                if label is None:
                    raise ValueError(f"{upstream_id}: unsupported legacy label {item[2]!r}")
                abs_idx, text = int(item[0]), str(item[1])
                lines.append({
                    "line_id": _hash("academic-silver-line-v1", identity["document_id"], abs_idx, text),
                    "abs_idx": abs_idx, "text": text, "label": label,
                    "token_count": count, "is_running_prose": None,
                })
            yield {
                "schema_version": "academic-structure-gold-v1",
                "document_id": identity["document_id"], "work_id": identity["work_id"],
                "representation_id": _hash(
                    "academic-silver-representation-v1", source, upstream_id,
                    identity["observed_text_sha256"],
                ),
                "source": source,
                "split": split_manifest["assignments"][identity["document_id"]],
                "coverage": str(row.get("mode") or "legacy_front_tail_windows"),
                "n_physical_lines": int(row.get("n_lines") or max(item[0] for item in legacy_lines) + 1),
                "n_present_lines": len(lines),
                "annotation": {
                    "status": "LLM_silver",
                    "engine": "legacy STRUCT_2K mixed LLM workflow; per-document engine ledger unavailable",
                    "task_scope": "bibliography_toc_windows",
                    "annotator_ids": ["LLM:legacy-STRUCT_2K"], "adjudicator_id": None,
                    "legacy_artifact_sha256": legacy_sha256, "toc_supervised": True,
                },
                "tokenizer": {"id": "ModernGreek-148k", "revision": TOKENIZER_REVISION},
                "upstream_document_id": upstream_id,
                "observed_text_sha256": identity["observed_text_sha256"],
                "lines": lines,
            }

    receipt, gold_sha256 = _atomic_validated_silver(
        Path(args.output), converted_rows(), config["silver_contract"], split_manifest
    )
    receipt.update({
        "gold_sha256": gold_sha256,
        "legacy_input_sha256": legacy_sha256,
    })
    _write_json(Path(args.split_manifest), split_manifest)
    receipt["split_manifest_sha256"] = sha256_file(args.split_manifest)
    _write_json(Path(args.receipt), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def false_deletion_packet(args: argparse.Namespace) -> int:
    _require_new_outputs(args.output)
    # Reuse the established review taxonomy instead of creating another one.
    from failure_analysis import categorize_fp
    from span_signals import line_signals

    documents = read_gold(args.silver)
    predictions = read_predictions(args.predictions, documents)
    per_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for document in documents:
        guesses = predictions[document.document_id]
        for index, (line, guess) in enumerate(zip(document.lines, guesses)):
            if line.label == "O" and guess != "O":
                lo, hi = max(0, index - 2), min(len(document.lines), index + 3)
                per_source[document.source].append({
                    "schema_version": "academic-structure-targeted-false-deletion-audit-v1",
                    "evidence_status": "manual_review_required",
                    "document_id": document.document_id,
                    "work_id": document.work_id,
                    "source": document.source,
                    "abs_idx": line.abs_idx,
                    "silver_label": line.label,
                    "candidate_prediction": guess,
                    "existing_review_category": categorize_fp(
                        line.text, line_signals(line.text),
                        line.abs_idx / max(document.n_physical_lines, 1),
                    ),
                    "token_count": line.token_count,
                    "context": [
                        {"abs_idx": context.abs_idx, "text": context.text}
                        for context in document.lines[lo:hi]
                    ],
                    "manual_is_false_deletion": None,
                    "manual_is_running_prose": None,
                    "reviewer_note": "",
                })
    chosen = []
    for source, rows in sorted(per_source.items()):
        rows.sort(key=lambda row: (-row["token_count"], _hash(args.seed, row["document_id"], row["abs_idx"])))
        chosen.extend(rows[:args.per_source])
    _atomic_write(
        Path(args.output),
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen).encode(),
    )
    print(json.dumps({"rows": len(chosen), "by_source": dict(collections.Counter(r["source"] for r in chosen))}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--output")
    hydrate = sub.add_parser("hydrate-span")
    hydrate.add_argument("--unit-dir", required=True)
    hydrate.add_argument("--tokenizer-json", required=True)
    hydrate.add_argument("--config", default=str(HERE / "config.json"))
    hydrate.add_argument("--output", required=True)
    hydrate.add_argument("--split-manifest", required=True)
    hydrate.add_argument("--receipt", required=True)
    legacy = sub.add_parser("import-legacy")
    legacy.add_argument("--input", required=True, help="future copied STRUCT_2K_gold.jsonl")
    legacy.add_argument("--tokenizer-json", required=True)
    legacy.add_argument("--config", default=str(HERE / "config.json"))
    legacy.add_argument("--output", required=True)
    legacy.add_argument("--split-manifest", required=True)
    legacy.add_argument("--receipt", required=True)
    packet = sub.add_parser("false-deletion-packet")
    packet.add_argument("--silver", required=True)
    packet.add_argument("--predictions", required=True)
    packet.add_argument("--output", required=True)
    packet.add_argument("--per-source", type=int, default=25)
    packet.add_argument("--seed", default="silver-high-risk-v1")
    args = parser.parse_args(argv)
    if args.command == "audit":
        report = audit_tracked()
        if args.output:
            _write_json(Path(args.output), report)
        else:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.command == "hydrate-span":
        return hydrate_span(args)
    if args.command == "import-legacy":
        return import_legacy(args)
    return false_deletion_packet(args)


if __name__ == "__main__":
    raise SystemExit(main())
