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
from .span_rehydration import (
    inspect_span_snapshot,
    rehydrate_span_units,
    verify_rehydration_receipt,
)

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
LINE_RE = re.compile(r"^L(\d+):\s?(.*)$")
TOKENIZER_SHA256 = "358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394"
TOKENIZER_REVISION = "a4826df7f76b54cdd6dc21d09fe97283c466999b"
SPAN_MANIFEST = EVAL_DIR / "units" / "SPAN_manifest.jsonl"
SPAN_BATCHPATHS = EVAL_DIR / "units" / "SPAN_batchpaths.json"
SPAN_REHYDRATION_LAYOUT = HERE / "span_rehydration_layout.json"


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
            "historical_raw_path": "units/STRUCT_2K_gold.jsonl",
            "evidence_status": "LLM_silver_despite_historical_filename",
            "task_scope": "bibliography_toc_windows",
            "expected_document_count": 2000,
            "raw_artifact_present": (units / "STRUCT_2K_gold.jsonl").is_file(),
            "new_annotation_proposed": False,
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


@dataclass(frozen=True)
class DeclaredSpan:
    unit_id: str
    span_index: int
    start_line: int
    end_line: int
    unit_win_lo: int
    unit_win_hi: int


def _validate_unit_coordinates(
    unit_id: str,
    meta: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> list[tuple[int, str]]:
    line_rows = _line_rows(str(unit["text_numbered"]))
    coordinates = [index for index, _ in line_rows]
    if not coordinates or coordinates != sorted(set(coordinates)):
        raise ValueError(f"{unit_id}: numbered text coordinates are empty, duplicate, or unordered")
    lo, hi = int(meta["win_lo"]), int(meta["win_hi"])
    if coordinates[0] < lo or coordinates[-1] >= hi:
        raise ValueError(f"{unit_id}: numbered text coordinates escape manifest window [{lo}, {hi})")
    return line_rows


def _validate_annotation(
    unit_id: str,
    meta: Mapping[str, Any],
    annotation: Mapping[str, Any],
) -> list[DeclaredSpan]:
    lo, hi = int(meta["win_lo"]), int(meta["win_hi"])
    spans = annotation.get("spans", [])
    if not isinstance(spans, list):
        raise ValueError(f"{unit_id}: annotation spans must be a list")
    has_bib = annotation.get("has_bib")
    if not isinstance(has_bib, bool):
        raise ValueError(f"{unit_id}: has_bib must be boolean")
    if has_bib is True and not spans:
        raise ValueError(f"{unit_id}: has_bib=true but no spans are present")
    if has_bib is False and spans:
        raise ValueError(f"{unit_id}: has_bib=false but spans are present")
    declared: list[DeclaredSpan] = []
    for span_index, span in enumerate(spans):
        if not isinstance(span, Mapping):
            raise ValueError(f"{unit_id}: span {span_index} must be an object")
        start, end = span.get("start_line"), span.get("end_line")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError(
                f"{unit_id}: span {span_index} must have nonnegative ordered integer coordinates"
            )
        declared.append(DeclaredSpan(unit_id, span_index, start, end, lo, hi))
    return declared


def _validate_unit_and_annotation(
    unit_id: str,
    meta: Mapping[str, Any],
    annotation: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> tuple[list[tuple[int, str]], list[DeclaredSpan]]:
    return (
        _validate_unit_coordinates(unit_id, meta, unit),
        _validate_annotation(unit_id, meta, annotation),
    )


@dataclass
class SilverDraft:
    upstream_doc_id: str
    source: str
    n_physical_lines: int
    lines: dict[int, str]
    bib_lines: set[int]
    declared_spans: list[DeclaredSpan]
    sampled_units: list[str]
    annotation_units: list[str]
    missing_annotation_units: list[str]


def _build_span_drafts(
    manifest: Mapping[str, Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    units: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], SilverDraft]:
    """Rebuild historical document unions while retaining unannotated unit text."""

    drafts: dict[tuple[str, str], SilverDraft] = {}
    for unit_id in sorted(manifest):
        meta = manifest[unit_id]
        key = (str(meta["source"]), str(meta["doc_id"]))
        draft = drafts.setdefault(key, SilverDraft(
            upstream_doc_id=key[1], source=key[0], n_physical_lines=int(meta["win_hi"]),
            lines={}, bib_lines=set(), declared_spans=[], sampled_units=[],
            annotation_units=[], missing_annotation_units=[],
        ))
        draft.n_physical_lines = max(draft.n_physical_lines, int(meta["win_hi"]))
        draft.sampled_units.append(unit_id)
        line_rows = _validate_unit_coordinates(unit_id, meta, units[unit_id])
        for abs_idx, text in line_rows:
            previous = draft.lines.setdefault(abs_idx, text)
            if previous != text:
                raise ValueError(f"{unit_id}: conflicting text at absolute line {abs_idx}")
        annotation = annotations.get(unit_id)
        if annotation is None:
            # Historical span_seq_data.load() retained every sampled unit's text
            # and added spans only when its annotation existed.  Preserve that
            # comparison behavior explicitly instead of dropping this document.
            draft.missing_annotation_units.append(unit_id)
        else:
            draft.annotation_units.append(unit_id)
            draft.declared_spans.extend(
                _validate_annotation(unit_id, meta, annotation)
            )
    return drafts


def _project_document_spans(
    draft: SilverDraft,
) -> tuple[collections.Counter[str], list[dict[str, Any]]]:
    """Apply the historical present-line document-union semantics without repairing labels."""
    present = set(draft.lines)
    counts: collections.Counter[str] = collections.Counter()
    anomalies: list[dict[str, Any]] = []
    for span in draft.declared_spans:
        counts["declared_span_count"] += 1
        effective = sorted(
            index
            for index in present
            if span.start_line <= index <= span.end_line
        )
        unit_window_contains = (
            span.unit_win_lo <= span.start_line <= span.end_line < span.unit_win_hi
        )
        start_present = span.start_line in present
        end_present = span.end_line in present
        within_document = (
            0 <= span.start_line <= span.end_line < draft.n_physical_lines
        )
        if not unit_window_contains:
            counts["unit_window_escape_span_count"] += 1
        if not (start_present and end_present):
            counts["declared_boundary_absence_span_count"] += 1
        if not within_document:
            counts["declared_outside_document_span_count"] += 1
        if not effective:
            outcome = "zero_effective"
            counts["zero_effective_span_count"] += 1
        elif unit_window_contains and start_present and end_present:
            outcome = "exact_nonempty"
            counts["exact_nonempty_span_count"] += 1
        else:
            outcome = "projected_nonempty"
            counts["adjusted_nonempty_span_count"] += 1
        draft.bib_lines.update(effective)
        counts["effective_positive_coordinate_assignments"] += len(effective)
        if outcome != "exact_nonempty":
            anomalies.append(
                {
                    "source": draft.source,
                    "upstream_document_id": draft.upstream_doc_id,
                    "unit_id": span.unit_id,
                    "span_index": span.span_index,
                    "declared_start_line": span.start_line,
                    "declared_end_line": span.end_line,
                    "unit_win_lo": span.unit_win_lo,
                    "unit_win_hi": span.unit_win_hi,
                    "unit_window_contains_declared_span": unit_window_contains,
                    "declared_span_within_document_physical_range": within_document,
                    "declared_start_present_in_document_union": start_present,
                    "declared_end_present_in_document_union": end_present,
                    "effective_present_line_count": len(effective),
                    "effective_first_present_line": effective[0] if effective else None,
                    "effective_last_present_line": effective[-1] if effective else None,
                    "outcome": outcome,
                }
            )
    return counts, anomalies


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
    manifest_rows = _jsonl(SPAN_MANIFEST)
    manifest = {row["unit_id"]: row for row in manifest_rows}
    annotations = {
        row["unit_id"]: row for row in _annotation_family(EVAL_DIR / "annotations_span" / "all.json")
    }
    expected = _load_json(SPAN_BATCHPATHS)
    if args.unit_rehydration_receipt:
        unit_snapshot = verify_rehydration_receipt(
            args.unit_dir,
            args.unit_rehydration_receipt,
            SPAN_MANIFEST,
            SPAN_BATCHPATHS,
            args.unit_layout,
        )
    else:
        unit_snapshot = {
            "receipt_path": None,
            "receipt_sha256": None,
            "snapshot_artifact_sha256": None,
            "snapshot_equivalence_status": "unreceipted_external_unit_directory",
            "snapshot_equivalence_verified": False,
            "research_fit_eligible": False,
            "research_evidence_scope": "none",
            "production_eligible": False,
        }
    units = _load_units(Path(args.unit_dir), expected)
    missing_annotation = sorted(set(manifest) - set(annotations))
    annotated_unit_ids = set(manifest) & set(annotations)
    if set(units) != set(manifest):
        raise ValueError(
            f"unit payload/manifest mismatch: missing={len(set(manifest)-set(units))}, "
            f"extra={len(set(units)-set(manifest))}"
        )
    drafts = _build_span_drafts(manifest, annotations, units)
    projection_counts: collections.Counter[str] = collections.Counter()
    projection_anomalies: list[dict[str, Any]] = []
    for draft in drafts.values():
        document_counts, document_anomalies = _project_document_spans(draft)
        projection_counts.update(document_counts)
        projection_anomalies.extend(document_anomalies)
    projection_counts["effective_unique_bib_line_count"] = sum(
        len(draft.bib_lines) for draft in drafts.values()
    )
    for key in (
        "declared_span_count",
        "exact_nonempty_span_count",
        "adjusted_nonempty_span_count",
        "zero_effective_span_count",
        "unit_window_escape_span_count",
        "declared_boundary_absence_span_count",
        "declared_outside_document_span_count",
        "effective_positive_coordinate_assignments",
    ):
        projection_counts.setdefault(key, 0)
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
                "sampled_unit_ids": sorted(draft.sampled_units),
                "unit_ids": sorted(draft.annotation_units),
                "missing_annotation_unit_ids": sorted(draft.missing_annotation_units),
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
    receipt, silver_sha256 = _atomic_validated_silver(
        Path(args.output), rows, config["silver_contract"], split_manifest
    )
    receipt.update({
        "silver_sha256": silver_sha256,
        "missing_annotation_unit_ids": missing_annotation,
        "missing_annotation_policy": (
            "retain sampled present lines as O, matching historical span_seq_data.load semantics"
        ),
        "source_manifest_sha256": sha256_file(SPAN_MANIFEST),
        "source_batches_inventory_sha256": canonical_json_sha256(
            sorted((path.name, sha256_file(path)) for path in Path(args.unit_dir).glob("batch_*.json"))
        ),
        "source_unit_snapshot": unit_snapshot,
        "sequence_fit_eligible": unit_snapshot["research_fit_eligible"],
        "sequence_evidence_scope": unit_snapshot["research_evidence_scope"],
        "annotation_coordinate_alignment": {
            "status": (
                "document_union_projection_with_silver_anomalies"
                if projection_anomalies
                else "exact_on_present_document_union"
            ),
            "unit_count": len(manifest),
            "annotated_unit_count": len(annotated_unit_ids),
            "missing_annotation_unit_count": len(missing_annotation),
            "semantics": (
                "merge present nonblank lines from all sampled windows by document, then label "
                "the intersection with each inclusive declared span"
            ),
            "zero_effective_policy": (
                "retain sampled lines as O, matching historical span_seq_data comparison semantics"
            ),
            "counts": dict(sorted(projection_counts.items())),
            "anomalies": projection_anomalies,
            "checks": [
                "numbered coordinates unique and ordered",
                "coordinates contained by manifest win_lo/win_hi",
                "annotation shape and coordinate types valid",
                "declared spans projected only onto present document-union coordinates",
            ],
        },
        "production_eligible": False,
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

    receipt, silver_sha256 = _atomic_validated_silver(
        Path(args.output), converted_rows(), config["silver_contract"], split_manifest
    )
    receipt.update({
        "silver_sha256": silver_sha256,
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
    ranked: dict[str, list[dict[str, Any]]] = {}
    for source, rows in sorted(per_source.items()):
        rows.sort(key=lambda row: (-row["token_count"], _hash(args.seed, row["document_id"], row["abs_idx"])))
        ranked[source] = rows
    available = sum(len(rows) for rows in ranked.values())
    if available < args.total:
        raise ValueError(
            f"only {available} high-risk predicted removals are available; "
            f"the deployment audit requires exactly {args.total}"
        )
    chosen: list[dict[str, Any]] = []
    offsets = {source: 0 for source in ranked}
    while len(chosen) < args.total:
        progressed = False
        for source in sorted(ranked):
            offset = offsets[source]
            if offset < len(ranked[source]) and len(chosen) < args.total:
                chosen.append(ranked[source][offset])
                offsets[source] += 1
                progressed = True
        if not progressed:  # guarded by the aggregate availability check
            raise RuntimeError("source-balanced audit selection stalled")
    _atomic_write(
        Path(args.output),
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen).encode(),
    )
    print(json.dumps({
        "schema_version": "academic-structure-targeted-false-deletion-packet-receipt-v1",
        "rows": len(chosen),
        "required_rows": args.total,
        "by_source": dict(collections.Counter(r["source"] for r in chosen)),
        "output_sha256": sha256_file(args.output),
        "manual_review_complete": False,
        "production_eligible": False,
    }, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--output")
    hydrate = sub.add_parser("hydrate-span")
    hydrate.add_argument("--unit-dir", required=True)
    hydrate.add_argument(
        "--unit-rehydration-receipt",
        help="receipt from rehydrate-span-units; receipted snapshots are silver-fit eligible even when historical equivalence is unverified",
    )
    hydrate.add_argument("--unit-layout", default=str(SPAN_REHYDRATION_LAYOUT))
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
    packet.add_argument("--total", type=int, choices=(100,), default=100)
    packet.add_argument("--seed", default="silver-high-risk-v1")
    rehydrate = sub.add_parser(
        "rehydrate-span-units",
        help="rebuild missing text-only SPAN batches from receipt-bound source artifacts",
    )
    rehydrate.add_argument("--manifest", default=str(SPAN_MANIFEST))
    rehydrate.add_argument("--batchpaths", default=str(SPAN_BATCHPATHS))
    rehydrate.add_argument("--layout", default=str(SPAN_REHYDRATION_LAYOUT))
    rehydrate.add_argument("--source-artifacts", required=True)
    rehydrate.add_argument("--output-dir", required=True)
    rehydrate.add_argument("--receipt", required=True)
    rehydrate.add_argument(
        "--expected-artifact-sha256",
        help="explicit expected span-unit-snapshot-artifact-v1 SHA-256; mismatch fails atomically",
    )
    rehydrate.add_argument(
        "--reference-unit-dir",
        help="optional original snapshot for byte/unit comparison only; never supplies labels",
    )
    snapshot = sub.add_parser(
        "span-snapshot-digest",
        help="validate an existing batch directory and calculate its artifact SHA-256",
    )
    snapshot.add_argument("--unit-dir", required=True)
    snapshot.add_argument("--manifest", default=str(SPAN_MANIFEST))
    snapshot.add_argument("--batchpaths", default=str(SPAN_BATCHPATHS))
    snapshot.add_argument("--layout", default=str(SPAN_REHYDRATION_LAYOUT))
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
    if args.command == "rehydrate-span-units":
        receipt = rehydrate_span_units(
            manifest_path=args.manifest,
            batchpaths_path=args.batchpaths,
            layout_path=args.layout,
            source_receipt_path=args.source_artifacts,
            output_dir=args.output_dir,
            receipt_path=args.receipt,
            expected_artifact_sha256=args.expected_artifact_sha256,
            reference_unit_dir=args.reference_unit_dir,
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "span-snapshot-digest":
        inspection = inspect_span_snapshot(
            unit_dir=args.unit_dir,
            manifest_path=args.manifest,
            batchpaths_path=args.batchpaths,
            layout_path=args.layout,
        )
        print(json.dumps(inspection, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    return false_deletion_packet(args)


if __name__ == "__main__":
    raise SystemExit(main())
