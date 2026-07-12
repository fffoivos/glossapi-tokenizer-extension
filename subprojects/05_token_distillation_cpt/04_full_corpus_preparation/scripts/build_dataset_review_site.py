#!/usr/bin/env python3
"""Build or serve the private, offline full-corpus dataset review site.

The generated site has no remote dependencies.  Dataset text is written only
to per-sample JSON files, fetched on demand, and rendered with ``textContent``.
The input sample packet must carry a validated masking/provenance receipt.
High-precision identifier patterns are masked, but generic names and addresses
can remain; generated material is therefore always treated as sensitive.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import hmac
import html
import json
import math
import os
import re
import secrets
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterable, Mapping

import export_dataset_review_samples as sample_exporter
from export_dataset_review_samples import (
    SITE_ATTESTATION_SCHEMA,
    redact_complete_text,
    validate_redaction_counts,
)
from profile_dataset_quality_rust import (
    require_exact_keys,
    require_nonnegative_int,
    require_sha256,
    sha256_json,
    validate_receipt_object,
    validate_quality_site_handoff,
)


SITE_SCHEMA = "dataset_review_site_manifest_v1"
SITE_DATA_SCHEMA = "dataset_review_site_data_v1"
SAMPLE_SCHEMA = "dataset_review_complete_sample_v1"
SAMPLE_RECEIPT_SCHEMA = "dataset_review_complete_sample_packet_receipt_v1"
SITE_SAMPLE_SCHEMA = "dataset_review_site_sample_v1"
INVENTORY_SCHEMA = "post_december_glossapi_inventory_v1"
QUALITY_SCHEMA = "dataset_quality_summary_v1"
EVALUATIONS_SCHEMA = "dataset_review_evaluations_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_OUTPUT = (
    Path.home()
    / "presentations"
    / "train-apertus-with-glossapi"
    / "full-corpus-v2-dataset-review"
)


def lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following any filesystem links."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def open_regular_nofollow(path: Path) -> BinaryIO:
    """Open one regular file while refusing symlinks in every path component."""

    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
    ):
        raise RuntimeError(
            "secure all-component no-follow input opening is unavailable"
        )
    absolute = lexical_absolute(path)
    parts = absolute.parts
    if len(parts) < 2 or parts[0] != absolute.anchor:
        raise ValueError(f"input path is not an absolute file path: {path}")
    directory_fd = os.open(
        absolute.anchor,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        for component in parts[1:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(
            f"input path contains a symlink/non-directory component: {absolute}"
        ) from exc
    finally:
        os.close(directory_fd)
    file_stat = os.fstat(file_fd)
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(file_fd)
        raise ValueError(f"input is not a regular file: {absolute}")
    return os.fdopen(file_fd, "rb", closefd=True)


@dataclass(frozen=True)
class SiteInputSnapshot:
    path: Path
    bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


def snapshot_site_input(
    path: Path, *, copy_to: Path | None = None
) -> SiteInputSnapshot:
    absolute = lexical_absolute(path)
    digest = hashlib.sha256()
    copied_bytes = 0
    output_handle = None
    with open_regular_nofollow(absolute) as handle:
        before = os.fstat(handle.fileno())
        if copy_to is not None:
            copy_to.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            output_fd = os.open(
                copy_to,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            output_handle = os.fdopen(output_fd, "wb", closefd=True)
        try:
            while chunk := handle.read(16 * 1024 * 1024):
                digest.update(chunk)
                copied_bytes += len(chunk)
                if output_handle is not None:
                    output_handle.write(chunk)
            after = os.fstat(handle.fileno())
            if output_handle is not None:
                output_handle.flush()
                os.fsync(output_handle.fileno())
        except BaseException:
            if output_handle is not None:
                output_handle.close()
            if copy_to is not None:
                copy_to.unlink(missing_ok=True)
            raise
        finally:
            if output_handle is not None and not output_handle.closed:
                output_handle.close()
    identity = (
        before.st_size,
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        identity
        != (
            after.st_size,
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or copied_bytes != after.st_size
    ):
        raise ValueError(f"input changed while snapshotting: {absolute}")
    return SiteInputSnapshot(
        path=absolute,
        bytes=after.st_size,
        sha256=digest.hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


def snapshot_site_inputs(
    paths: Iterable[Path | None], root: Path
) -> tuple[dict[Path, SiteInputSnapshot], dict[Path, Path]]:
    snapshots: dict[Path, SiteInputSnapshot] = {}
    copies: dict[Path, Path] = {}
    for path in paths:
        if path is None:
            continue
        absolute = lexical_absolute(path)
        if absolute in snapshots:
            continue
        target = root / Path(*absolute.parts[1:])
        snapshot = snapshot_site_input(absolute, copy_to=target)
        snapshots[snapshot.path] = snapshot
        copies[snapshot.path] = target
    return snapshots, copies


def verify_site_input_snapshots(
    snapshots: Mapping[Path, SiteInputSnapshot],
) -> None:
    for path, expected in snapshots.items():
        if snapshot_site_input(path) != expected:
            raise ValueError(f"input drift before atomic publication: {path}")


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def strict_json_loads(text: str, *, context: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{context}: non-finite JSON constant {value}")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{context}: non-finite JSON number {value}")
        return result

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def read_json(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"), context=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = strict_json_loads(line, context=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield line_number, value


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def input_receipt(
    path: Path | None,
    snapshots: Mapping[Path, Any] | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = lexical_absolute(path)
    if snapshots is not None:
        snapshot = snapshots.get(resolved)
        if snapshot is None:
            raise ValueError(f"input lacks a pre-parse snapshot: {resolved}")
        return {
            "path": str(resolved),
            "bytes": int(snapshot.bytes),
            "sha256": str(snapshot.sha256),
        }
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def safe_json(value: Any, *, indent: int | None = None) -> str:
    """Serialize JSON without literal HTML-significant characters."""

    result = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent else (",", ":"),
        allow_nan=False,
    )
    return (
        result.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )


def site_nonnegative_int(value: Any, *, context: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > (1 << 63) - 1
    ):
        raise ValueError(f"{context}: expected a nonnegative signed-64-bit integer")
    return value


def site_fraction(value: Any, *, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context}: expected a finite fraction")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{context}: expected a finite fraction in [0, 1]")
    return result


def canonical_display_document_id(value: str) -> str:
    return hashlib.sha256(
        f"dataset-review-display-id-v1\0{value}".encode("utf-8")
    ).hexdigest()[:16]


def opaque_site_id(key: bytes, label: str, *values: str) -> str:
    message = "\0".join(("dataset-review-site-opaque-id-v1", label, *values))
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def source_identity_map(sources: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        sources.get("schema_version") != "full_cpt_sources_v1"
        or not isinstance(sources.get("base"), Mapping)
        or not isinstance(sources.get("sources"), list)
    ):
        raise ValueError("unsupported local sources config")
    configured: list[tuple[str, Mapping[str, Any]]] = [
        ("nanochat_base", sources["base"]),
    ]
    for row in sources["sources"]:
        if not isinstance(row, Mapping):
            raise ValueError("local sources config contains a non-object source")
        raw_source_id = row.get("source_id")
        if not isinstance(raw_source_id, str):
            raise ValueError("local sources config contains a non-string source_id")
        configured.append((raw_source_id, row))
    result: dict[str, dict[str, Any]] = {}
    for source_id, row in configured:
        repo_id = row.get("repo_id")
        revision = row.get("revision")
        role = row.get("role")
        if (
            not source_id
            or not isinstance(repo_id, str)
            or not repo_id
            or not isinstance(revision, str)
            or not revision
            or not isinstance(role, str)
            or not role
            or source_id in result
        ):
            raise ValueError(
                "local sources config has an incomplete/duplicate identity"
            )
        acquisition_kind = row.get("acquisition_kind")
        if acquisition_kind in (None, ""):
            acquisition_kind = "hugging_face"
        if not isinstance(acquisition_kind, str) or not acquisition_kind:
            raise ValueError(f"{source_id}: invalid acquisition_kind")
        mdc_dataset_id = row.get("mdc_dataset_id")
        if mdc_dataset_id == "":
            mdc_dataset_id = None
        if acquisition_kind == "mozilla_data_collective":
            if not isinstance(mdc_dataset_id, str) or not mdc_dataset_id:
                raise ValueError(f"{source_id}: MDC source lacks mdc_dataset_id")
        elif mdc_dataset_id not in (None, ""):
            raise ValueError(f"{source_id}: non-MDC source declares mdc_dataset_id")
        canonical_config = {"source_id": source_id, **dict(row)}
        result[source_id] = {
            "source_id": source_id,
            "repo_id": repo_id,
            "revision": revision,
            "role": role,
            "acquisition_kind": acquisition_kind,
            "mdc_dataset_id": mdc_dataset_id or None,
            "source_config_sha256": sha256_json(canonical_config),
        }
    return result


def load_inventory(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if value.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError(f"{path}: unsupported inventory schema")
    entries: list[dict[str, Any]] = []
    for group, field in (
        ("post_cutoff", "post_cutoff_repositories"),
        (
            "older_material_change",
            "older_repositories_with_material_post_cutoff_changes",
        ),
    ):
        for position, row in enumerate(value.get(field, [])):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: inventory entry must be an object")
            entries.append(
                {**row, "inventory_group": group, "inventory_position": position}
            )
    repo_ids = [str(row.get("repo_id", "")) for row in entries]
    if (
        len(entries) != 29
        or len(set(repo_ids)) != 29
        or any(not name for name in repo_ids)
    ):
        raise ValueError(
            "dataset review site requires exactly 29 unique inventory repositories"
        )
    return entries


def load_evaluations(path: Path, expected_repos: set[str]) -> dict[str, dict[str, Any]]:
    value = read_json(path)
    if value.get("schema_version") != EVALUATIONS_SCHEMA:
        raise ValueError(f"{path}: unsupported evaluation schema")
    result: dict[str, dict[str, Any]] = {}
    for row in value.get("entries", []):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: evaluation row must be an object")
        repo_id = str(row.get("repo_id", ""))
        if not repo_id or repo_id in result:
            raise ValueError(f"{path}: duplicate or empty evaluation repo_id")
        if (
            not str(row.get("assessment", "")).strip()
            or not str(row.get("recommended_action", "")).strip()
        ):
            raise ValueError(f"{path}: incomplete evaluation for {repo_id}")
        result[repo_id] = row
    if set(result) != expected_repos:
        raise ValueError(
            f"evaluation coverage mismatch; missing={sorted(expected_repos - set(result))}, "
            f"unexpected={sorted(set(result) - expected_repos)}"
        )
    return result


def load_quality(
    path: Path | None,
    handoff_path: Path | None,
    *,
    sources_config_path: Path | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    if path is None and handoff_path is None:
        return {}, None, []
    if path is None or handoff_path is None:
        raise ValueError(
            "--quality-summary and --quality-handoff-receipt are an inseparable pair"
        )
    if sources_config_path is None:
        raise ValueError("quality handoff requires the exact local sources config")
    projection, acquired_identities = validate_quality_site_handoff(
        summary_path=path,
        handoff_path=handoff_path,
        expected_sources_config_sha256=sha256_file(sources_config_path),
    )
    tracked_sources = source_identity_map(read_json(sources_config_path))
    for identity in acquired_identities:
        tracked = tracked_sources.get(str(identity["source_id"]))
        if tracked is None or any(
            identity[name] != tracked[name]
            for name in (
                "repo_id",
                "revision",
                "role",
                "acquisition_kind",
                "mdc_dataset_id",
                "source_config_sha256",
            )
        ):
            raise ValueError(
                "quality handoff source identity differs from sources config"
            )
    if {str(row["repo_id"]) for row in acquired_identities} != {
        str(row["repo_id"]) for row in projection["repositories"]
    }:
        raise ValueError("quality handoff source/repository coverage drift")
    result = {str(row["repo_id"]): row for row in projection["repositories"]}
    scan_mode = str(projection["scan_mode"])
    return (
        result,
        {
            "scan_mode": scan_mode,
            "documents": int(projection["documents"]),
            # This profiler intentionally selects a population (and normally
            # excludes nanochat_base).  Never promote that to a corpus-wide claim.
            "is_corpus_wide": False,
            "label": (
                "Representative source-review sample"
                if scan_mode == "review_sample"
                else "Full scan of selected canonical sources"
            ),
            "selected_source_ids": list(projection["selected_source_ids"]),
            "excluded_source_ids": list(projection["excluded_source_ids"]),
        },
        acquired_identities,
    )


def load_requests(
    path: Path | None,
    tracked_sources: Mapping[str, Mapping[str, Any]],
    site_key: bytes,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, list[dict[str, Any]]]]:
    if path is None:
        return {}, {}, {}
    samples: dict[str, dict[str, Any]] = {}
    dataset_to_repo: dict[str, str] = {}
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, row in iter_jsonl(path):
        if row.get("schema_version") != "source_quality_review_request_v1":
            raise ValueError(f"{path}:{line_number}: unsupported request schema")
        sample_id = str(row.get("sample_id", ""))
        if not SHA256_RE.fullmatch(sample_id):
            raise ValueError(f"{path}:{line_number}: invalid sample_id")
        source = row.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"{path}:{line_number}: missing source envelope")
        source_id = str(source.get("source_id", ""))
        tracked = tracked_sources.get(source_id)
        repo_id = str(source.get("source_repo_id", ""))
        revision = str(source.get("source_revision", ""))
        dataset = str(row.get("source_dataset", ""))
        if (
            tracked is None
            or not repo_id
            or not revision
            or not dataset
            or repo_id != tracked["repo_id"]
            or revision != tracked["revision"]
        ):
            raise ValueError(
                f"{path}:{line_number}: request source identity differs from sources config"
            )
        previous = dataset_to_repo.setdefault(dataset, repo_id)
        if previous != repo_id:
            raise ValueError(
                f"{path}:{line_number}: source_dataset maps to multiple repositories"
            )
        if row.get("reviewer_slot") != "primary":
            continue
        if sample_id in samples:
            raise ValueError(f"{path}:{line_number}: duplicate primary sample")
        raw_doc_id = str(source.get("source_doc_id", ""))
        site_record = {
            "site_sample_id": opaque_site_id(
                site_key, "sample", repo_id, revision, sample_id
            ),
            "site_document_id": opaque_site_id(
                site_key, "document", repo_id, revision, raw_doc_id
            ),
            "source_dataset": dataset,
            "sampling_stratum": str(row.get("sampling_stratum", "")),
            "complete_document_available": False,
            "complete_document_path": None,
        }
        record = {
            "canonical_sample_id": sample_id,
            "source_id": source_id,
            "source_repo_id": repo_id,
            "source_dataset": dataset,
            "source_revision": revision,
            "canonical_display_document_id": canonical_display_document_id(raw_doc_id),
            "site_record": site_record,
        }
        samples[sample_id] = {**record, "repo_id": repo_id}
        by_repo[repo_id].append(site_record)
    for rows in by_repo.values():
        rows.sort(key=lambda row: row["site_sample_id"])
    return samples, dataset_to_repo, dict(by_repo)


def load_review_responses(
    path: Path | None,
    requests: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    scores: dict[str, list[int]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for line_number, row in iter_jsonl(path):
        if row.get("schema_version") != "source_quality_review_response_v1":
            raise ValueError(f"{path}:{line_number}: unsupported review response")
        sample_id = str(row.get("sample_id", ""))
        slot = str(row.get("reviewer_slot", ""))
        key = (sample_id, slot)
        if sample_id not in requests or key in seen:
            raise ValueError(f"{path}:{line_number}: unknown or duplicate response")
        seen.add(key)
        repo_id = str(requests[sample_id]["repo_id"])
        counters[repo_id][f"action:{row.get('action')}"] += 1
        counters[repo_id][f"value:{row.get('substantive_training_value')}"] += 1
        variability = row.get("variability")
        if isinstance(variability, dict):
            counters[repo_id][
                f"template_similarity:{variability.get('template_similarity')}"
            ] += 1
            counters[repo_id][
                f"substantive_variation:{variability.get('substantive_variation')}"
            ] += 1
        if bool(row.get("safety_or_license_blocker")):
            counters[repo_id]["safety_or_license_blockers"] += 1
        quality_score = site_nonnegative_int(
            row.get("quality_score"),
            context=f"{path}:{line_number}.quality_score",
        )
        if quality_score > 4:
            raise ValueError(f"{path}:{line_number}: quality_score exceeds 4")
        scores[repo_id].append(quality_score)
    result = {}
    for repo_id in sorted(set(counters) | set(scores)):
        result[repo_id] = {
            "response_rows": sum(
                value
                for key, value in counters[repo_id].items()
                if key.startswith("action:")
            ),
            "action_counts": {
                key.removeprefix("action:"): value
                for key, value in sorted(counters[repo_id].items())
                if key.startswith("action:")
            },
            "substantive_value_counts": {
                key.removeprefix("value:"): value
                for key, value in sorted(counters[repo_id].items())
                if key.startswith("value:")
            },
            "template_similarity_counts": {
                key.removeprefix("template_similarity:"): value
                for key, value in sorted(counters[repo_id].items())
                if key.startswith("template_similarity:")
            },
            "substantive_variation_counts": {
                key.removeprefix("substantive_variation:"): value
                for key, value in sorted(counters[repo_id].items())
                if key.startswith("substantive_variation:")
            },
            "safety_or_license_blockers": counters[repo_id][
                "safety_or_license_blockers"
            ],
            "mean_quality_score": (
                sum(scores[repo_id]) / len(scores[repo_id]) if scores[repo_id] else None
            ),
        }
    return result


def load_admission(
    path: Path | None,
    dataset_to_repo: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    value = read_json(path)
    if value.get("schema_version") != "source_quality_review_admission_v1":
        raise ValueError(f"{path}: unsupported source admission")
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in value.get("sources", []):
        if not isinstance(row, dict):
            continue
        dataset = str(row.get("source_dataset", ""))
        repo_id = dataset_to_repo.get(dataset)
        if repo_id:
            reasons = row.get("reasons", [])
            result[repo_id].append(
                {
                    "source_dataset": dataset,
                    "decision": str(row.get("decision", "")),
                    "reasons": [str(reason) for reason in reasons]
                    if isinstance(reasons, list)
                    else [],
                    "pending_count": len(row.get("pending", []))
                    if isinstance(row.get("pending"), list)
                    else 0,
                    "post_clean_review_required": bool(
                        row.get("post_clean_review_required", False)
                    ),
                }
            )
    return {
        key: sorted(rows, key=lambda row: str(row.get("source_dataset")))
        for key, rows in result.items()
    }


def load_novelty(
    path: Path | None,
    dataset_to_repo: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    value = read_json(path)
    if value.get("schema_version") != "full_cpt_source_novelty_v1":
        raise ValueError(f"{path}: unsupported novelty summary")
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"{path}: novelty sources must be an array")
    for index, row in enumerate(sources):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: novelty source {index} must be an object")
        source_dataset = str(row.get("source_dataset", ""))
        repo_id = dataset_to_repo.get(source_dataset)
        if repo_id:
            rows = site_nonnegative_int(
                row.get("rows"), context=f"{path}.sources[{index}].rows"
            )
            identity_tokens = site_nonnegative_int(
                row.get("identity_word_tokens"),
                context=f"{path}.sources[{index}].identity_word_tokens",
            )
            exact_tokens = site_nonnegative_int(
                row.get("exact_unique_word_tokens"),
                context=f"{path}.sources[{index}].exact_unique_word_tokens",
            )
            novel_tokens = site_nonnegative_int(
                row.get("novel_word_tokens_after_lineage_resolution"),
                context=(
                    f"{path}.sources[{index}]."
                    "novel_word_tokens_after_lineage_resolution"
                ),
            )
            fraction = site_fraction(
                row.get("novel_token_fraction"),
                context=f"{path}.sources[{index}].novel_token_fraction",
            )
            expected_fraction = (
                round(novel_tokens / identity_tokens, 8) if identity_tokens else 0.0
            )
            if (
                rows < 1
                or exact_tokens > identity_tokens
                or novel_tokens > exact_tokens
                or not math.isclose(fraction, expected_fraction, abs_tol=1e-8)
            ):
                raise ValueError(f"{path}.sources[{index}]: novelty denominator drift")
            result[repo_id].append(
                {
                    "source_dataset": source_dataset,
                    "rows": rows,
                    "identity_word_tokens": identity_tokens,
                    "exact_unique_word_tokens": exact_tokens,
                    "novel_word_tokens_after_lineage_resolution": novel_tokens,
                    "novel_token_fraction": fraction,
                }
            )
    return {
        key: sorted(rows, key=lambda row: str(row.get("source_dataset")))
        for key, rows in result.items()
    }


def payload_state(row: Mapping[str, Any]) -> str:
    status = str(row.get("payload_status", ""))
    if status == "empty_scaffold":
        return "empty"
    if status == "metadata_only_parquet":
        return "metadata_only"
    availability = str(row.get("availability", ""))
    if status == "no_hf_data" or (
        status == "external_full_text_parquet_archive"
        and availability.startswith("external_mozilla_registered")
    ):
        return "external_unavailable"
    if row.get("inventory_group") == "older_material_change":
        return "material_change"
    return "text_available"


def has_external_acquisition_evidence(
    inventory_row: Mapping[str, Any],
    *,
    quality_present: bool,
    acquired_identities: Iterable[Mapping[str, Any]],
    tracked_sources: Mapping[str, Mapping[str, Any]],
) -> bool:
    if not quality_present:
        return False
    repo_id = str(inventory_row.get("repo_id", ""))
    revision = str(inventory_row.get("revision", ""))
    for identity in acquired_identities:
        source_id = str(identity.get("source_id", ""))
        tracked = tracked_sources.get(source_id)
        if (
            tracked is not None
            and str(identity.get("repo_id", "")) == repo_id == tracked["repo_id"]
            and str(identity.get("revision", "")) == revision == tracked["revision"]
            and identity.get("acquisition_kind")
            == tracked["acquisition_kind"]
            == "mozilla_data_collective"
            and identity.get("mdc_dataset_id") == tracked["mdc_dataset_id"]
            and isinstance(identity.get("mdc_dataset_id"), str)
            and bool(identity["mdc_dataset_id"])
            and identity.get("source_config_sha256") == tracked["source_config_sha256"]
            and int(identity.get("documents", 0)) > 0
            and int(identity.get("shards", 0)) > 0
            and int(identity.get("acquisition_selected_file_count", 0)) > 0
        ):
            return True
    return False


def row_total(row: Mapping[str, Any]) -> int | None:
    rows = row.get("rows")
    if isinstance(rows, dict):
        for field in ("footer", "card"):
            if rows.get(field) is not None:
                return site_nonnegative_int(
                    rows[field], context=f"inventory.rows.{field}"
                )
    for field in (
        "new_asset_footer_rows",
        "current_metadata_footer_rows",
        "current_card_documents",
    ):
        if row.get(field) is not None:
            return site_nonnegative_int(row[field], context=f"inventory.{field}")
    return None


def byte_total(row: Mapping[str, Any]) -> int | None:
    for field in ("data_artifact_bytes", "new_asset_bytes", "payload_bytes_current"):
        if row.get(field) is not None:
            return site_nonnegative_int(row[field], context=f"inventory.{field}")
    return None


def token_total(row: Mapping[str, Any]) -> int | None:
    card = row.get("card_tokens")
    if isinstance(card, dict) and card.get("value") is not None:
        return site_nonnegative_int(
            card["value"], context="inventory.card_tokens.value"
        )
    for field in ("new_asset_card_tokens_sum", "current_card_tokens"):
        if row.get(field) is not None:
            return site_nonnegative_int(row[field], context=f"inventory.{field}")
    return None


def slug(index: int, repo_id: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", repo_id.casefold()).strip("-")
    return f"{index + 1:02d}-{name[:80]}"


def validate_sample_site_attestation(
    *,
    packet_path: Path,
    packet_receipt_path: Path,
    attestation_path: Path,
    review_requests_path: Path,
    primary_sample_ids: set[str],
    tracked_sources: Mapping[str, Mapping[str, Any]],
    local_sources_config_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], int]]:
    receipt = read_json(packet_receipt_path)
    require_exact_keys(
        receipt,
        required=(
            "schema_version",
            "status",
            "normalization_manifest",
            "canonical_root",
            "review_requests",
            "export_contract",
            "site_attestation",
            "input_shards",
            "checkpoint_inventory",
            "checkpoint_inventory_sha256",
            "output",
            "redaction_totals",
            "high_precision_identifier_patterns_masked",
        ),
        context="sample_packet_receipt",
    )
    if (
        receipt["schema_version"] != SAMPLE_RECEIPT_SCHEMA
        or receipt["status"] != "passed"
        or receipt["high_precision_identifier_patterns_masked"] is not True
    ):
        raise ValueError("sample packet receipt schema/status/masking drift")
    output = validate_receipt_object(
        receipt["output"], context="sample_packet_receipt.output", require_rows=True
    )
    declared_packet = Path(str(output["path"]))
    if not declared_packet.is_absolute():
        declared_packet = packet_receipt_path.resolve().parent / declared_packet
    if (
        declared_packet.resolve() != packet_path.resolve()
        or output["bytes"] != packet_path.stat().st_size
        or output["sha256"] != sha256_file(packet_path)
    ):
        raise ValueError("sample packet output receipt drift")
    requests_receipt = validate_receipt_object(
        receipt["review_requests"],
        context="sample_packet_receipt.review_requests",
        allow_rows=False,
    )
    declared_requests = Path(requests_receipt["path"])
    if not declared_requests.is_absolute():
        declared_requests = packet_receipt_path.resolve().parent / declared_requests
    if (
        declared_requests.resolve() != review_requests_path.resolve()
        or requests_receipt["bytes"] != review_requests_path.stat().st_size
        or requests_receipt["sha256"] != sha256_file(review_requests_path)
    ):
        raise ValueError("sample packet request receipt drift")
    attestation_receipt = validate_receipt_object(
        receipt["site_attestation"],
        context="sample_packet_receipt.site_attestation",
        allow_rows=False,
    )
    declared_attestation = Path(attestation_receipt["path"])
    if not declared_attestation.is_absolute():
        declared_attestation = (
            packet_receipt_path.resolve().parent / declared_attestation
        )
    if (
        declared_attestation.resolve() != attestation_path.resolve()
        or attestation_receipt["bytes"] != attestation_path.stat().st_size
        or attestation_receipt["sha256"] != sha256_file(attestation_path)
    ):
        raise ValueError("sample site attestation receipt drift")
    for name in ("normalization_manifest", "export_contract"):
        value = receipt[name]
        if not isinstance(value, Mapping):
            raise ValueError(f"sample_packet_receipt.{name}: expected object")
        required = {"path", "bytes", "sha256"}
        if name == "export_contract":
            required.add("contract_sha256")
        require_exact_keys(
            value, required=required, context=f"sample_packet_receipt.{name}"
        )
        require_nonnegative_int(
            value["bytes"], context=f"sample_packet_receipt.{name}.bytes"
        )
        require_sha256(value["sha256"], context=f"sample_packet_receipt.{name}.sha256")
        if name == "export_contract":
            require_sha256(
                value["contract_sha256"],
                context="sample_packet_receipt.export_contract.contract_sha256",
            )

    input_shards = receipt["input_shards"]
    if not isinstance(input_shards, list) or not input_shards:
        raise ValueError("sample packet receipt input shards missing")
    shard_rows: dict[tuple[str, str, str], int] = {}
    for index, row in enumerate(input_shards):
        if not isinstance(row, Mapping):
            raise ValueError("sample packet input shard must be object")
        require_exact_keys(
            row,
            required=("source_id", "path", "bytes", "sha256", "rows"),
            context=f"sample_packet_receipt.input_shards[{index}]",
        )
        source_id = str(row["source_id"])
        shard_path = str(row["path"])
        sha = require_sha256(
            row["sha256"], context="sample_packet_receipt.input_shard.sha256"
        )
        rows = require_nonnegative_int(
            row["rows"], context="sample_packet_receipt.input_shard.rows"
        )
        require_nonnegative_int(
            row["bytes"], context="sample_packet_receipt.input_shard.bytes"
        )
        if (
            not source_id
            or not shard_path
            or rows < 1
            or (
                source_id,
                shard_path,
                sha,
            )
            in shard_rows
        ):
            raise ValueError("sample packet input shard identity drift")
        shard_rows[(source_id, shard_path, sha)] = rows

    checkpoint_inventory = receipt["checkpoint_inventory"]
    if not isinstance(checkpoint_inventory, list) or len(checkpoint_inventory) != len(
        input_shards
    ):
        raise ValueError("sample packet checkpoint coverage drift")
    selected_rows = 0
    for index, row in enumerate(checkpoint_inventory):
        if not isinstance(row, Mapping):
            raise ValueError("sample packet checkpoint entry must be object")
        require_exact_keys(
            row,
            required=(
                "input_shard_sha256",
                "checkpoint_receipt_sha256",
                "output_sha256",
                "selected_rows",
            ),
            context=f"sample_packet_receipt.checkpoint_inventory[{index}]",
        )
        for name in (
            "input_shard_sha256",
            "checkpoint_receipt_sha256",
            "output_sha256",
        ):
            require_sha256(
                row[name], context=f"sample_packet_receipt.checkpoint.{name}"
            )
        if row["input_shard_sha256"] != input_shards[index]["sha256"]:
            raise ValueError("sample checkpoint/input-shard ordering drift")
        selected_rows += require_nonnegative_int(
            row["selected_rows"],
            context="sample_packet_receipt.checkpoint.selected_rows",
        )
    if (
        sha256_json(checkpoint_inventory) != receipt["checkpoint_inventory_sha256"]
        or selected_rows != output["rows"]
        or output["rows"] != len(primary_sample_ids)
    ):
        raise ValueError("sample packet checkpoint/request denominator drift")

    attestation = read_json(attestation_path)
    require_exact_keys(
        attestation,
        required=(
            "schema_version",
            "status",
            "created_at",
            "packet",
            "review_requests",
            "primary_sample_count",
            "primary_sample_id_inventory_sha256",
            "normalization",
            "export_contract",
            "checkpoint_closure",
            "masking",
        ),
        context="sample_site_attestation",
    )
    if (
        attestation["schema_version"] != SITE_ATTESTATION_SCHEMA
        or attestation["status"] != "passed"
    ):
        raise ValueError("sample site attestation schema/status drift")
    if (
        validate_receipt_object(
            attestation["packet"],
            context="sample_site_attestation.packet",
            require_rows=True,
        )
        != output
        or validate_receipt_object(
            attestation["review_requests"],
            context="sample_site_attestation.review_requests",
            allow_rows=False,
        )
        != requests_receipt
        or require_nonnegative_int(
            attestation["primary_sample_count"],
            context="sample_site_attestation.primary_sample_count",
        )
        != len(primary_sample_ids)
        or attestation["primary_sample_id_inventory_sha256"]
        != sha256_json(sorted(primary_sample_ids))
    ):
        raise ValueError("sample site attestation packet/request closure drift")

    normalization = attestation["normalization"]
    if not isinstance(normalization, Mapping):
        raise ValueError("sample attestation normalization closure missing")
    require_exact_keys(
        normalization,
        required=(
            "schema_version",
            "manifest",
            "sources_config_sha256",
            "acquisition_receipt_sha256",
            "source_identities",
            "source_identity_inventory_sha256",
            "normalized_shard_inventory_sha256",
            "input_shards",
            "input_shard_inventory_sha256",
        ),
        context="sample_site_attestation.normalization",
    )
    manifest_receipt = validate_receipt_object(
        normalization["manifest"],
        context="sample_site_attestation.normalization.manifest",
        allow_rows=False,
    )
    if (
        normalization["schema_version"] != "full_cpt_normalization_manifest_v1"
        or normalization["sources_config_sha256"] != local_sources_config_sha256
        or manifest_receipt != receipt["normalization_manifest"]
        or normalization["input_shards"] != input_shards
        or normalization["input_shard_inventory_sha256"] != sha256_json(input_shards)
    ):
        raise ValueError("sample attestation normalization/input-shard drift")
    for name in (
        "sources_config_sha256",
        "acquisition_receipt_sha256",
        "source_identity_inventory_sha256",
        "normalized_shard_inventory_sha256",
        "input_shard_inventory_sha256",
    ):
        require_sha256(
            normalization[name], context=f"sample_attestation.normalization.{name}"
        )
    identities = normalization["source_identities"]
    if not isinstance(identities, list) or not identities:
        raise ValueError("sample attestation source identity closure drift")
    seen_identity_sources: set[str] = set()
    for index, identity in enumerate(identities):
        if not isinstance(identity, Mapping):
            raise ValueError("sample attestation source identity must be an object")
        require_exact_keys(
            identity,
            required=(
                "source_id",
                "repo_id",
                "revision",
                "role",
                "documents",
                "shards",
                "shard_inventory_sha256",
                "acquisition_selected_file_count",
                "acquisition_selected_bytes",
                "acquisition_file_inventory_sha256",
                "acquisition_kind",
                "mdc_dataset_id",
                "source_config_sha256",
            ),
            context=f"sample_site_attestation.source_identities[{index}]",
        )
        source_id = str(identity["source_id"])
        tracked = tracked_sources.get(source_id)
        if (
            not source_id
            or tracked is None
            or not str(identity["repo_id"])
            or not str(identity["revision"])
            or identity["repo_id"] != tracked["repo_id"]
            or identity["revision"] != tracked["revision"]
            or identity["role"] != tracked["role"]
            or identity["acquisition_kind"] != tracked["acquisition_kind"]
            or identity["mdc_dataset_id"] != tracked["mdc_dataset_id"]
            or identity["source_config_sha256"] != tracked["source_config_sha256"]
            or source_id in seen_identity_sources
            or require_nonnegative_int(
                identity["documents"], context="sample_attestation.identity.documents"
            )
            < 1
            or require_nonnegative_int(
                identity["shards"], context="sample_attestation.identity.shards"
            )
            < 1
            or require_nonnegative_int(
                identity["acquisition_selected_file_count"],
                context="sample_attestation.identity.acquisition_file_count",
            )
            < 1
        ):
            raise ValueError("sample attestation source identity is incomplete")
        require_nonnegative_int(
            identity["acquisition_selected_bytes"],
            context="sample_attestation.identity.acquisition_bytes",
        )
        for name in (
            "shard_inventory_sha256",
            "acquisition_file_inventory_sha256",
            "source_config_sha256",
        ):
            require_sha256(
                identity[name], context=f"sample_attestation.identity.{name}"
            )
        seen_identity_sources.add(source_id)
    if (
        identities != sorted(identities, key=lambda row: str(row["source_id"]))
        or sha256_json(identities) != normalization["source_identity_inventory_sha256"]
        or not {str(row["source_id"]) for row in input_shards}.issubset(
            seen_identity_sources
        )
    ):
        raise ValueError("sample attestation source identity closure drift")

    export_contract = attestation["export_contract"]
    if not isinstance(export_contract, Mapping):
        raise ValueError("sample attestation export contract missing")
    require_exact_keys(
        export_contract,
        required=("receipt", "canonical_sha256", "value"),
        context="sample_site_attestation.export_contract",
    )
    contract_receipt = validate_receipt_object(
        export_contract["receipt"],
        context="sample_site_attestation.export_contract.receipt",
        allow_rows=False,
    )
    contract = export_contract["value"]
    if not isinstance(contract, Mapping):
        raise ValueError("sample attestation export contract value missing")
    require_exact_keys(
        contract,
        required=(
            "schema_version",
            "normalization_manifest_sha256",
            "review_requests_sha256",
            "exporter_script_sha256",
            "redaction_dependency_sha256",
            "redaction_pipeline",
            "batch_size",
            "selected_sample_count",
        ),
        context="sample_site_attestation.export_contract.value",
    )
    dependency_hashes = {
        "build_source_review_packet": sha256_file(
            Path(
                sample_exporter.redact_direct_identifiers.__code__.co_filename
            ).resolve()
        ),
        "greek_pii": sha256_file(
            Path(sample_exporter.mask_greek_identifiers.__code__.co_filename).resolve()
        ),
        "profile_dataset_quality_rust": sha256_file(
            Path(sample_exporter.metadata_flags.__code__.co_filename).resolve()
        ),
    }
    contract_batch_size = require_nonnegative_int(
        contract["batch_size"], context="sample_attestation.export_contract.batch_size"
    )
    contract_sample_count = require_nonnegative_int(
        contract["selected_sample_count"],
        context="sample_attestation.export_contract.selected_sample_count",
    )
    if (
        contract["schema_version"] != "dataset_review_sample_export_contract_v1"
        or contract["normalization_manifest_sha256"]
        != receipt["normalization_manifest"]["sha256"]
        or contract["review_requests_sha256"] != requests_receipt["sha256"]
        or contract["exporter_script_sha256"]
        != sha256_file(Path(sample_exporter.__file__).resolve())
        or contract["redaction_dependency_sha256"] != dependency_hashes
        or contract["redaction_pipeline"] != "high_precision_identifier_patterns_v1"
        or contract_batch_size < 1
        or contract_sample_count != len(primary_sample_ids)
        or sha256_json(contract) != export_contract["canonical_sha256"]
        or export_contract["canonical_sha256"]
        != receipt["export_contract"]["contract_sha256"]
        or contract_receipt
        != {
            name: receipt["export_contract"][name]
            for name in ("path", "bytes", "sha256")
        }
    ):
        raise ValueError("sample attestation export contract/implementation drift")

    checkpoint_closure = attestation["checkpoint_closure"]
    if not isinstance(checkpoint_closure, Mapping):
        raise ValueError("sample attestation checkpoint closure missing")
    require_exact_keys(
        checkpoint_closure,
        required=(
            "count",
            "selected_rows",
            "inventory_sha256",
            "receipt_closure_sha256",
            "checkpoint_text_outputs_rehashed_for_attestation",
        ),
        context="sample_site_attestation.checkpoint_closure",
    )
    if (
        require_nonnegative_int(
            checkpoint_closure["count"],
            context="sample_attestation.checkpoint_closure.count",
        )
        != len(checkpoint_inventory)
        or require_nonnegative_int(
            checkpoint_closure["selected_rows"],
            context="sample_attestation.checkpoint_closure.selected_rows",
        )
        != output["rows"]
        or checkpoint_closure["inventory_sha256"]
        != receipt["checkpoint_inventory_sha256"]
        or checkpoint_closure["checkpoint_text_outputs_rehashed_for_attestation"]
        is not True
    ):
        raise ValueError("sample attestation checkpoint closure drift")
    require_sha256(
        checkpoint_closure["receipt_closure_sha256"],
        context="sample_attestation.checkpoint_closure.receipt_closure_sha256",
    )

    masking = attestation["masking"]
    if not isinstance(masking, Mapping):
        raise ValueError("sample attestation masking closure missing")
    require_exact_keys(
        masking,
        required=(
            "pipeline",
            "implementation_sha256",
            "high_precision_identifier_patterns_masked",
            "private_data_true_rows",
            "redaction_totals",
            "redaction_totals_sha256",
        ),
        context="sample_site_attestation.masking",
    )
    expected_implementation = {
        "exporter": str(contract["exporter_script_sha256"]),
        **dependency_hashes,
    }
    redaction_totals = validate_redaction_counts(
        masking["redaction_totals"], context="sample attestation redaction totals"
    )
    receipt_redaction_totals = validate_redaction_counts(
        receipt["redaction_totals"], context="sample packet redaction totals"
    )
    implementation = masking["implementation_sha256"]
    if not isinstance(implementation, Mapping):
        raise ValueError("sample attestation masking implementation is invalid")
    require_exact_keys(
        implementation,
        required=(
            "exporter",
            "build_source_review_packet",
            "greek_pii",
            "profile_dataset_quality_rust",
        ),
        context="sample_site_attestation.masking.implementation_sha256",
    )
    if (
        masking["pipeline"] != "high_precision_identifier_patterns_v1"
        or masking["implementation_sha256"] != expected_implementation
        or masking["high_precision_identifier_patterns_masked"] is not True
        or require_nonnegative_int(
            masking["private_data_true_rows"],
            context="sample_attestation.masking.private_data_true_rows",
        )
        != 0
        or redaction_totals != receipt_redaction_totals
        or masking["redaction_totals_sha256"] != sha256_json(redaction_totals)
    ):
        raise ValueError("sample attestation masking closure drift")
    return receipt, shard_rows


def write_complete_samples(
    path: Path | None,
    *,
    packet_receipt_path: Path | None,
    site_attestation_path: Path | None,
    review_requests_path: Path | None,
    output: Path,
    requests: dict[str, dict[str, Any]],
    tracked_sources: Mapping[str, Mapping[str, Any]],
    local_sources_config_sha256: str,
    visible_repositories: set[str],
) -> tuple[list[dict[str, Any]], int]:
    if path is None and packet_receipt_path is None and site_attestation_path is None:
        return [], 0
    if (
        path is None
        or packet_receipt_path is None
        or site_attestation_path is None
        or review_requests_path is None
    ):
        raise ValueError(
            "--complete-samples requires its receipt, site attestation, and review requests"
        )
    packet_receipt, shard_rows = validate_sample_site_attestation(
        packet_path=path,
        packet_receipt_path=packet_receipt_path,
        attestation_path=site_attestation_path,
        review_requests_path=review_requests_path,
        primary_sample_ids=set(requests),
        tracked_sources=tracked_sources,
        local_sources_config_sha256=local_sources_config_sha256,
    )
    declared_output = packet_receipt["output"]

    written: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_addresses: set[tuple[str, str, str, int]] = set()
    observed_redactions: Counter[str] = Counter()
    excluded_outside_inventory = 0
    site_index = {
        request["canonical_sample_id"]: request["site_record"]
        for request in requests.values()
        if request["repo_id"] in visible_repositories
    }
    for line_number, row in iter_jsonl(path):
        require_exact_keys(
            row,
            required=(
                "schema_version",
                "sample_id",
                "source_id",
                "source_repo_id",
                "source_revision",
                "source_dataset",
                "display_document_id",
                "normalized_text_sha256",
                "profile_text_sha256",
                "profile_text_variant",
                "input_shard_path",
                "input_shard_sha256",
                "input_row_index",
                "private_data_true",
                "corrected_version_present",
                "high_precision_identifier_patterns_masked",
                "redaction_counts",
                "text",
            ),
            context=f"{path}:{line_number}",
        )
        if row.get("schema_version") != SAMPLE_SCHEMA:
            raise ValueError(
                f"{path}:{line_number}: unsupported complete sample schema"
            )
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in requests or sample_id in seen:
            raise ValueError(f"{path}:{line_number}: unknown or duplicate sample_id")
        text = row.get("text")
        if (
            row.get("high_precision_identifier_patterns_masked") is not True
            or row.get("private_data_true") is not False
            or not isinstance(row.get("corrected_version_present"), bool)
            or row.get("profile_text_variant")
            != "high_precision_identifier_masked_review_sample"
            or not isinstance(text, str)
            or not SHA256_RE.fullmatch(str(row.get("normalized_text_sha256", "")))
            or not SHA256_RE.fullmatch(str(row.get("profile_text_sha256", "")))
            or not SHA256_RE.fullmatch(str(row.get("input_shard_sha256", "")))
            or int(row.get("input_row_index", -1)) < 0
            or hashlib.sha256(text.encode("utf-8")).hexdigest()
            != row.get("profile_text_sha256")
        ):
            raise ValueError(
                f"{path}:{line_number}: complete sample lacks valid masking/text attestation"
            )
        remasked, residual_identifiers = redact_complete_text(text)
        if remasked != text or residual_identifiers:
            raise ValueError(
                f"{path}:{line_number}: residual known identifier or URL after masking"
            )
        redaction_counts = validate_redaction_counts(
            row.get("redaction_counts"),
            context=f"{path}:{line_number}.redaction_counts",
        )
        observed_redactions.update(redaction_counts)
        expected = requests[sample_id]
        repo_id = str(row.get("source_repo_id", ""))
        dataset = str(row.get("source_dataset", ""))
        identity = {
            "source_id": str(row.get("source_id", "")),
            "source_repo_id": repo_id,
            "source_revision": str(row.get("source_revision", "")),
            "source_dataset": dataset,
            "canonical_display_document_id": str(row.get("display_document_id", "")),
        }
        tracked = tracked_sources.get(identity["source_id"])
        if (
            tracked is None
            or identity["source_id"] != expected["source_id"]
            or repo_id != expected["repo_id"]
            or identity["source_revision"] != expected["source_revision"]
            or repo_id != tracked["repo_id"]
            or identity["source_revision"] != tracked["revision"]
            or dataset != expected["source_dataset"]
            or identity["canonical_display_document_id"]
            != expected["canonical_display_document_id"]
        ):
            raise ValueError(f"{path}:{line_number}: complete sample identity drift")
        input_address = (
            identity["source_id"],
            str(row["input_shard_path"]),
            str(row["input_shard_sha256"]),
            int(row["input_row_index"]),
        )
        declared_shard_rows = shard_rows.get(input_address[:3])
        if (
            declared_shard_rows is None
            or input_address[3] >= declared_shard_rows
            or input_address in seen_addresses
        ):
            raise ValueError(
                f"{path}:{line_number}: invalid/duplicate canonical row binding"
            )
        seen_addresses.add(input_address)
        seen.add(sample_id)
        if repo_id not in visible_repositories:
            excluded_outside_inventory += 1
            continue
        site_record = site_index[sample_id]
        site_sample_id = str(site_record["site_sample_id"])
        sample_path = output / "samples" / f"{site_sample_id}.json"
        payload = {
            "schema_version": SITE_SAMPLE_SCHEMA,
            "site_sample_id": site_sample_id,
            "site_document_id": str(site_record["site_document_id"]),
            "source_repo_id": repo_id,
            "source_dataset": dataset,
            "high_precision_identifier_patterns_masked": True,
            "redaction_counts": dict(redaction_counts),
            "characters": len(text),
            "text": text,
        }
        write_private(sample_path, safe_json(payload))
        relative = sample_path.relative_to(output).as_posix()
        site_record["complete_document_available"] = True
        site_record["complete_document_path"] = relative
        written.append(receipt(sample_path, output))
    if seen != set(requests) or len(seen) != int(declared_output["rows"]):
        raise ValueError(
            f"{path}: complete sample coverage differs from primary requests"
        )
    if dict(sorted(observed_redactions.items())) != packet_receipt["redaction_totals"]:
        raise ValueError(f"{path}: per-row redaction totals differ from receipt")
    return written, excluded_outside_inventory


def html_shell(
    *, title: str, body: str, base: str, page: str, repo_index: int | None = None
) -> str:
    data_attributes = (
        f' data-page="{html.escape(page)}" data-base="{html.escape(base)}"'
    )
    if repo_index is not None:
        data_attributes += f' data-repo-index="{repo_index}"'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{base}assets/site.css">
</head>
<body{data_attributes}>
{body}
<script src="{base}assets/site.js" defer></script>
</body>
</html>
"""


CSS = r"""
:root{color-scheme:light dark;--bg:#f5f4ef;--panel:#fff;--ink:#17201c;--muted:#68736d;--line:#d8ddd8;--accent:#1e6551;--warn:#a75c00;--bad:#a33a38;--good:#2d7355}
@media(prefers-color-scheme:dark){:root{--bg:#111614;--panel:#18201d;--ink:#edf4f0;--muted:#9eaaa4;--line:#334039;--accent:#77c9aa;--warn:#f0ac55;--bad:#f58d89;--good:#78c99d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--accent)}header,main,footer{max-width:1240px;margin:auto;padding:24px}header{padding-bottom:8px}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:12px;color:var(--muted)}h1{font-size:clamp(28px,4vw,48px);line-height:1.08;margin:.2em 0}h2{margin-top:2em}.lede{font-size:18px;color:var(--muted);max-width:850px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}.privacy-warning{border-color:var(--warn);color:var(--warn);font-weight:650}.kpi{font-size:28px;font-weight:750}.label,.muted{color:var(--muted);font-size:13px}.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}.toolbar input,.toolbar select{padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink)}.table-wrap{overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:14px}table{width:100%;border-collapse:collapse;min-width:920px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{cursor:pointer;position:sticky;top:0;background:var(--panel)}.badge{display:inline-block;padding:3px 8px;border-radius:999px;border:1px solid currentColor;font-size:12px;margin:1px 4px 1px 0}.badge.good{color:var(--good)}.badge.warn{color:var(--warn)}.badge.bad{color:var(--bad)}.metric{display:grid;grid-template-columns:240px 1fr 110px;gap:9px;align-items:center;margin:7px 0}progress.track{width:100%;height:9px;border:0;border-radius:9px;overflow:hidden;background:var(--line);accent-color:var(--accent)}progress.track::-webkit-progress-bar{background:var(--line)}progress.track::-webkit-progress-value{background:var(--accent)}progress.track::-moz-progress-bar{background:var(--accent)}.repo-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.repo-card h3{margin-top:0}.sample{border-top:1px solid var(--line);padding:12px 0}.sample button{padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink);cursor:pointer}.sample button:disabled{opacity:.5;cursor:not-allowed}pre.document{white-space:pre-wrap;word-break:break-word;max-height:70vh;overflow:auto;background:var(--bg);padding:14px;border-radius:9px;border:1px solid var(--line)}dl{display:grid;grid-template-columns:minmax(160px,240px) 1fr;gap:7px 14px}dt{color:var(--muted)}dd{margin:0}.back{display:inline-block;margin-bottom:12px}footer{color:var(--muted);font-size:12px}@media(max-width:650px){.metric{grid-template-columns:1fr}.metric .value{text-align:left}dl{display:block}dt{margin-top:9px}}
""".strip()


JS = r"""
(async()=>{'use strict';
const body=document.body,base=body.dataset.base||'',page=body.dataset.page;
const data=await fetch(base+'site_data.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error('site_data.json '+r.status);return r.json()});
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=String(text);return n};
const fmt=n=>n===null||n===undefined?'—':new Intl.NumberFormat('en',{notation:Math.abs(n)>=1e6?'compact':'standard',maximumFractionDigits:2}).format(n);
const pct=n=>n===null||n===undefined?'—':new Intl.NumberFormat('en',{style:'percent',maximumFractionDigits:2}).format(n);
const badge=(text,kind='warn')=>el('span','badge '+kind,text);
const badgeKind=s=>/exclude|empty|metadata|unavailable/.test(s)?'bad':/candidate|include|novel|replace|clean/.test(s)?'good':'warn';
const countSummary=o=>o&&Object.keys(o).length?Object.entries(o).map(([k,v])=>k+': '+fmt(v)).join(' · '):'—';
function metric(parent,label,value,max,display){const row=el('div','metric'),lab=el('div','label',label);if(value===null||value===undefined){row.append(lab,el('div','muted','Metric pending'),el('div','value','—'))}else{const track=el('progress','track');track.max=max>0?max:1;track.value=Math.max(0,Math.min(track.max,value));row.append(lab,track,el('div','value',display??fmt(value)))}parent.append(row)}
function renderOverview(){document.getElementById('repo-count').textContent=fmt(data.repositories.length);document.getElementById('text-count').textContent=fmt(data.overview.text_bearing_or_changed_repositories);document.getElementById('profiled-count').textContent=fmt(data.overview.profiled_repositories);document.getElementById('sample-count').textContent=fmt(data.overview.complete_samples);const scope=document.getElementById('scope-banner'),qs=data.overview.quality_scope;scope.textContent=qs?(qs.scan_mode==='review_sample'?qs.label+' · '+fmt(qs.documents)+' sampled documents. Representative sample statistics; not corpus-wide estimates.':qs.label+' · '+fmt(qs.documents)+' documents across '+fmt(qs.selected_source_ids.length)+' explicitly selected sources. Excluded sources: '+(qs.excluded_source_ids.join(', ')||'none declared')+'. This is a selected-population scan, not a corpus-wide claim.'):'Quality diagnostics have not run yet.';const tbody=document.querySelector('#repo-table tbody'),search=document.getElementById('search'),state=document.getElementById('state');let sort='repo_id',ascending=true;
const render=()=>{tbody.replaceChildren();const q=search.value.toLocaleLowerCase();const rows=data.repositories.filter(r=>(!q||JSON.stringify([r.repo_id,r.evaluation.assessment,r.recommended_action]).toLocaleLowerCase().includes(q))&&(!state.value||r.payload_state===state.value)).sort((a,b)=>{const av=a[sort]??a.evaluation?.[sort]??'',bv=b[sort]??b.evaluation?.[sort]??'';return (String(av).localeCompare(String(bv),undefined,{numeric:true}))*(ascending?1:-1)});for(const r of rows){const tr=el('tr'),name=el('td'),a=el('a',null,r.repo_id);a.href='datasets/'+r.slug+'.html';name.append(a);const action=el('td');action.append(badge(r.recommended_action,badgeKind(r.recommended_action)));tr.append(name,el('td',null,r.payload_state),action,el('td',null,fmt(r.declared_rows)),el('td',null,fmt(r.declared_bytes)),el('td',null,fmt(r.quality?.documents)),el('td',null,pct(r.quality?.document_rates?.html_rate)),el('td',null,fmt(r.quality?.distributions?.rust_noise_badness_score?.p50_approx)),el('td',null,fmt(r.samples.length)));tbody.append(tr)}};
for(const s of Object.keys(data.overview.payload_states).sort()){const o=el('option',null,s);o.value=s;state.append(o)}search.addEventListener('input',render);state.addEventListener('change',render);document.querySelectorAll('th[data-sort]').forEach(th=>th.addEventListener('click',()=>{ascending=sort===th.dataset.sort?!ascending:true;sort=th.dataset.sort;render()}));render();
const grid=document.getElementById('repo-grid');for(const r of data.repositories){const card=el('article','card repo-card'),h=el('h3'),a=el('a',null,r.repo_id);a.href='datasets/'+r.slug+'.html';h.append(a);card.append(h,badge(r.payload_state,badgeKind(r.payload_state)),badge(r.recommended_action,badgeKind(r.recommended_action)),el('p',null,r.evaluation.assessment));grid.append(card)}}
function addDefinition(dl,key,value){dl.append(el('dt',null,key),el('dd',null,value??'—'))}
function renderWaterfall(parent,row){const box=el('section','panel'),title=el('strong',null,row.source_dataset+' token lineage');box.append(title);const raw=row.identity_word_tokens,exact=row.exact_unique_word_tokens,novel=row.novel_word_tokens_after_lineage_resolution;metric(box,'Identity tokens',raw,Math.max(1,raw),fmt(raw));metric(box,'Exact-unique tokens',exact,Math.max(1,raw),fmt(exact));metric(box,'Novel after lineage review',novel,Math.max(1,raw),fmt(novel));box.append(el('p','muted','Near-duplicate removal remains deferred to global dedup.'));parent.append(box)}
function renderDetail(){const r=data.repositories[Number(body.dataset.repoIndex)],root=document.getElementById('detail');document.title=r.repo_id+' · Dataset review';document.getElementById('repo-title').textContent=r.repo_id;document.getElementById('assessment').textContent=r.evaluation.assessment;const badges=document.getElementById('badges');badges.append(badge(r.payload_state,badgeKind(r.payload_state)),badge(r.recommended_action,badgeKind(r.recommended_action)));const dl=document.getElementById('facts');addDefinition(dl,'Inventory group',r.inventory_group);addDefinition(dl,'Declared rows',fmt(r.declared_rows));addDefinition(dl,'Declared bytes',fmt(r.declared_bytes));addDefinition(dl,'Card-reported tokens',fmt(r.declared_tokens));addDefinition(dl,'Relation to Nanochat',r.relation_to_first_nanochat);addDefinition(dl,'Inventory disposition',r.inventory_disposition);addDefinition(dl,'Main risks',r.evaluation.main_risks.join(' · '));
const quality=document.getElementById('quality');if(!r.quality){quality.append(el('p','muted','No Rust/profile metrics are available for this repository yet. Structural, ToC, bibliography, and template metrics are pending.'))}else{const scope=r.quality_scope;quality.append(el('p','panel',scope.scan_mode==='review_sample'?scope.label+' · '+fmt(scope.repository_documents)+' sampled documents for this repository. Not corpus-wide. High-precision identifier patterns were masked before profiling; generic names and addresses may remain.':scope.label+' · '+fmt(scope.repository_documents)+' documents from the selected population for this repository. Not a corpus-wide claim.'));const rates=r.quality.document_rates||{},d=r.quality.distributions||{},tc=r.quality.template_concentration||{};metric(quality,'Median document length',d.original_characters?.p50_approx,d.original_characters?.p99_approx||1,fmt(d.original_characters?.p50_approx)+' chars');metric(quality,'90th-percentile length',d.original_characters?.p90_approx,d.original_characters?.p99_approx||1,fmt(d.original_characters?.p90_approx)+' chars');metric(quality,'Median Greek-letter share',d.raw_greek_letter_fraction?.p50_approx,1,pct(d.raw_greek_letter_fraction?.p50_approx));metric(quality,'Median repeated-line fraction',d.raw_repeated_line_fraction?.p50_approx,1,pct(d.raw_repeated_line_fraction?.p50_approx));metric(quality,'Median one-token-line fraction',d.raw_one_token_line_fraction?.p50_approx,1,pct(d.raw_one_token_line_fraction?.p50_approx));metric(quality,'Median mojibake markers / 1k chars',d.raw_mojibake_per_1000_chars?.p50_approx,d.raw_mojibake_per_1000_chars?.p99_approx||1,fmt(d.raw_mojibake_per_1000_chars?.p50_approx));metric(quality,'Median replacement chars / 1k',d.raw_replacement_per_1000_chars?.p50_approx,d.raw_replacement_per_1000_chars?.p99_approx||1,fmt(d.raw_replacement_per_1000_chars?.p50_approx));metric(quality,'HTML-bearing documents',rates.html_rate,1,pct(rates.html_rate));metric(quality,'Bibliography-header heuristic',rates.bibliography_header_rate,1,pct(rates.bibliography_header_rate));metric(quality,'ToC-header heuristic',rates.toc_header_rate,1,pct(rates.toc_header_rate));metric(quality,'Markdown-table documents',rates.markdown_table_rate,1,pct(rates.markdown_table_rate));metric(quality,'Large Markdown-table documents',rates.large_markdown_table_rate,1,pct(rates.large_markdown_table_rate));metric(quality,'Digital-governance footer documents',rates.digital_governance_footer_rate,1,pct(rates.digital_governance_footer_rate));metric(quality,'Personnel-cue documents',rates.personnel_cue_rate,1,pct(rates.personnel_cue_rate));metric(quality,'Isolated ADA-stamp documents',rates.isolated_ada_stamp_rate,1,pct(rates.isolated_ada_stamp_rate));metric(quality,'privateData=true documents',rates.private_data_true_rate,1,pct(rates.private_data_true_rate));metric(quality,'Corrected-version documents',rates.corrected_version_rate,1,pct(rates.corrected_version_rate));metric(quality,'Top structural-template concentration',tc.top_1_fraction,1,pct(tc.top_1_fraction));metric(quality,'Top-10 structural-template concentration',tc.top_10_fraction,1,pct(tc.top_10_fraction));metric(quality,scope.scan_mode==='review_sample'?'Residual high-precision identifier signals':'High-precision identifier signals',rates.direct_identifier_rate,1,pct(rates.direct_identifier_rate));metric(quality,'Guarded zero/no-Greek score',rates.zero_badness_zero_greek_guard_rate,1,pct(rates.zero_badness_zero_greek_guard_rate));metric(quality,'Median Rust badness',d.rust_noise_badness_score?.p50_approx,Math.max(1,d.rust_noise_badness_score?.p99_approx||1),fmt(d.rust_noise_badness_score?.p50_approx));metric(quality,'Median cleaner removal fraction',d.cleaner_removed_character_fraction?.p50_approx,1,pct(d.cleaner_removed_character_fraction?.p50_approx))}
const evidence=document.getElementById('review-evidence'),edl=el('dl');if(r.review){addDefinition(edl,'Review response rows',fmt(r.review.response_rows));addDefinition(edl,'Mean quality score (0–4)',fmt(r.review.mean_quality_score));addDefinition(edl,'Reviewer actions',countSummary(r.review.action_counts));addDefinition(edl,'Substantive value',countSummary(r.review.substantive_value_counts));addDefinition(edl,'Template similarity',countSummary(r.review.template_similarity_counts));addDefinition(edl,'Substantive variation',countSummary(r.review.substantive_variation_counts))}if(r.admissions.length){addDefinition(edl,'Admission decisions',r.admissions.map(x=>x.source_dataset+': '+x.decision).join(' · '));addDefinition(edl,'Admission reasons',r.admissions.map(x=>x.reasons.join(', ')||'none').join(' · '))}if(r.novelty.length)addDefinition(edl,'Exact-lineage novel token fraction',r.novelty.map(x=>x.source_dataset+': '+pct(x.novel_token_fraction)).join(' · '));if(edl.children.length)evidence.append(edl);else evidence.append(el('p','muted','Review responses, admission, and lineage novelty are not available yet.'));for(const row of r.novelty)renderWaterfall(evidence,row);
const notes=document.getElementById('notes');for(const note of r.notes){notes.append(el('li',null,note))}const samples=document.getElementById('samples');if(!r.samples.length)samples.append(el('p','muted','No review sample has been prepared.'));for(const s of r.samples){const box=el('section','sample'),head=el('div',null,s.source_dataset+' · local document '+s.site_document_id.slice(0,12)),button=el('button',null,s.complete_document_available?'Load complete masked document':'Complete document unavailable');button.disabled=!s.complete_document_available;const pre=el('pre','document');pre.hidden=true;button.addEventListener('click',async()=>{button.disabled=true;button.textContent='Loading…';try{const doc=await fetch(base+s.complete_document_path,{cache:'no-store'}).then(x=>{if(!x.ok)throw new Error(String(x.status));return x.json()});pre.textContent=doc.text;pre.hidden=false;button.textContent='Loaded'}catch(e){button.textContent='Load failed: '+e.message;button.disabled=false}});box.append(head,el('div','muted',s.sampling_stratum+' · local sample '+s.site_sample_id.slice(0,12)),button,pre);samples.append(box)}}
if(page==='overview')renderOverview();else if(page==='detail')renderDetail();
})().catch(error=>{const p=document.createElement('pre');p.textContent='Dataset review site failed safely: '+error.stack;document.body.append(p)});
""".strip()


def build_site(args: argparse.Namespace) -> int:
    output = lexical_absolute(args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    temporary.chmod(0o700)
    stable_inputs_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.inputs-", dir=output.parent)
    )
    stable_inputs_root.chmod(0o700)
    try:
        input_names = (
            "inventory",
            "evaluations",
            "sources_config",
            "quality_summary",
            "quality_handoff_receipt",
            "review_requests",
            "review_responses",
            "admission",
            "novelty",
            "complete_samples",
            "complete_samples_receipt",
            "complete_samples_attestation",
        )
        original_inputs = {name: getattr(args, name, None) for name in input_names}
        input_snapshots, stable_paths = snapshot_site_inputs(
            original_inputs.values(), stable_inputs_root
        )
        args = copy.copy(args)
        for name, path in original_inputs.items():
            if path is not None:
                setattr(args, name, stable_paths[lexical_absolute(path)])

        inventory = load_inventory(args.inventory)
        repos = {str(row["repo_id"]) for row in inventory}
        evaluations = load_evaluations(args.evaluations, repos)
        sources = read_json(args.sources_config)
        tracked_sources = source_identity_map(sources)
        quality, quality_scope, acquired_identities = load_quality(
            args.quality_summary,
            getattr(args, "quality_handoff_receipt", None),
            sources_config_path=args.sources_config,
        )
        supplemental_quality_repositories = sorted(set(quality) - repos)
        site_key = secrets.token_bytes(32)
        requests, dataset_to_repo, by_repo = load_requests(
            args.review_requests, tracked_sources, site_key
        )
        opaque_ids = [
            str(value)
            for rows in by_repo.values()
            for row in rows
            for value in (row["site_sample_id"], row["site_document_id"])
        ]
        if len(opaque_ids) != len(set(opaque_ids)):
            raise ValueError("site-local opaque identifier collision")
        responses = load_review_responses(args.review_responses, requests)
        admissions = load_admission(args.admission, dataset_to_repo)
        novelty = load_novelty(args.novelty, dataset_to_repo)

        sample_receipts, excluded_complete_samples = write_complete_samples(
            args.complete_samples,
            packet_receipt_path=getattr(args, "complete_samples_receipt", None),
            site_attestation_path=getattr(args, "complete_samples_attestation", None),
            review_requests_path=args.review_requests,
            output=temporary,
            requests=requests,
            tracked_sources=tracked_sources,
            local_sources_config_sha256=sha256_file(args.sources_config),
            visible_repositories=repos,
        )
        repositories: list[dict[str, Any]] = []
        for index, row in enumerate(inventory):
            repo_id = str(row["repo_id"])
            evaluation = evaluations[repo_id]
            state = payload_state(row)
            if state == "external_unavailable" and has_external_acquisition_evidence(
                row,
                quality_present=repo_id in quality,
                acquired_identities=acquired_identities,
                tracked_sources=tracked_sources,
            ):
                state = "external_acquired"
            notes = row.get("notes", row.get("warnings", []))
            if not isinstance(notes, list):
                notes = []
            repositories.append(
                {
                    "repo_id": repo_id,
                    "slug": slug(index, repo_id),
                    "inventory_group": row["inventory_group"],
                    "payload_state": state,
                    "declared_rows": row_total(row),
                    "declared_bytes": byte_total(row),
                    "declared_tokens": token_total(row),
                    "relation_to_first_nanochat": row.get("relation_to_first_nanochat"),
                    "inventory_disposition": row.get("disposition"),
                    "recommended_action": evaluation["recommended_action"],
                    "evaluation": evaluation,
                    "notes": [str(note) for note in notes],
                    "quality": quality.get(repo_id),
                    "quality_scope": (
                        {
                            **quality_scope,
                            "repository_documents": int(
                                quality[repo_id].get("documents", 0)
                            ),
                        }
                        if quality_scope is not None and repo_id in quality
                        else None
                    ),
                    "review": responses.get(repo_id),
                    "admissions": admissions.get(repo_id, []),
                    "novelty": novelty.get(repo_id, []),
                    "samples": by_repo.get(repo_id, []),
                }
            )

        state_counts = Counter(row["payload_state"] for row in repositories)
        site_data = {
            "schema_version": SITE_DATA_SCHEMA,
            "generated_at": utc_now(),
            "privacy": {
                "local_only": True,
                "sensitive_do_not_share": True,
                "high_precision_identifier_patterns_masked": True,
                "generic_names_and_addresses_may_remain": True,
                "sample_rendering": "JSON fetched on demand and assigned with textContent",
                "external_resources": False,
            },
            "overview": {
                "repositories": len(repositories),
                "payload_states": dict(sorted(state_counts.items())),
                "text_bearing_or_changed_repositories": sum(
                    state_counts[name]
                    for name in (
                        "text_available",
                        "material_change",
                        "external_acquired",
                    )
                ),
                "profiled_repositories": sum(
                    row["quality"] is not None for row in repositories
                ),
                "reviewed_repositories": sum(
                    row["review"] is not None for row in repositories
                ),
                "complete_samples": len(sample_receipts),
                "complete_samples_excluded_outside_inventory": excluded_complete_samples,
                "supplemental_profiled_repositories_outside_inventory": (
                    supplemental_quality_repositories
                ),
                "quality_scope": quality_scope,
            },
            "repositories": repositories,
        }
        write_private(temporary / "site_data.json", safe_json(site_data))
        write_private(temporary / "assets" / "site.css", CSS + "\n")
        write_private(temporary / "assets" / "site.js", JS + "\n")

        overview_body = """
<header><div class="eyebrow">Greek Apertus · full-corpus review</div><h1>What is in the new data?</h1><p class="lede">A private, receipt-backed view of all 29 post-cutoff or materially changed dataset repositories. Rust metrics are diagnostic; admission remains a lineage, quality, privacy, and cleaning decision.</p></header>
<main><p class="panel privacy-warning">Sensitive local review material — do not share. High-precision identifier patterns are masked, but generic names, addresses, and identifying context may remain.</p><section class="cards"><div class="card"><div id="repo-count" class="kpi">—</div><div class="label">inventory repositories</div></div><div class="card"><div id="text-count" class="kpi">—</div><div class="label">available or materially changed text repositories</div></div><div class="card"><div id="profiled-count" class="kpi">—</div><div class="label">Rust-profiled repositories</div></div><div class="card"><div id="sample-count" class="kpi">—</div><div class="label">complete local samples</div></div></section><p id="scope-banner" class="panel"></p>
<h2>Comparison</h2><div class="toolbar"><input id="search" type="search" placeholder="Filter repositories or evaluations"><select id="state"><option value="">All payload states</option></select></div><div class="table-wrap"><table id="repo-table"><thead><tr><th data-sort="repo_id">Repository</th><th data-sort="payload_state">Payload</th><th data-sort="recommended_action">Evaluation</th><th>Rows</th><th>Bytes</th><th>Profiled docs</th><th>HTML rate</th><th>Median Rust badness</th><th>Samples</th></tr></thead><tbody></tbody></table></div>
<h2>Dataset cards</h2><section id="repo-grid" class="repo-grid"></section></main><footer>Local-only static report. No CDN, analytics, remote fonts, or external requests.</footer>
""".strip()
        write_private(
            temporary / "index.html",
            html_shell(
                title="Greek Apertus dataset review",
                body=overview_body,
                base="",
                page="overview",
            ),
        )
        for index, row in enumerate(repositories):
            detail_body = """
<header><a class="back" href="../index.html">← All datasets</a><div class="eyebrow">Dataset review</div><h1 id="repo-title">Loading…</h1><div id="badges"></div><p id="assessment" class="lede"></p></header>
<main id="detail"><p class="panel privacy-warning">Sensitive local review material — do not share. High-precision patterns are masked; generic names, addresses, and identifying context may remain.</p><section class="panel"><h2>Inventory and evaluation</h2><dl id="facts"></dl></section><section class="panel"><h2>Cleanliness and Rust diagnostics</h2><div id="quality"></div><p class="muted">Approximate quantiles use a deterministic bounded sample. ToC/BIB values are simple header heuristics, not classifier accuracy or removal authorization. Template concentration is an edge-template diagnostic. A zero badness score with zero Greek characters is explicitly guarded, not labelled clean.</p></section><section class="panel"><h2>Reviewer, variability, and lineage evidence</h2><div id="review-evidence"></div></section><section class="panel"><h2>Inventory notes</h2><ul id="notes"></ul></section><section class="panel"><h2>Review documents</h2><p class="muted">Complete documents have high-precision identifier patterns masked, are stored outside HTML, fetched only when requested, and rendered as plain text. Generic names and addresses may remain.</p><div id="samples"></div></section></main><footer>Diagnostic review page; no cleaning or training admission is implied.</footer>
""".strip()
            write_private(
                temporary / "datasets" / f"{row['slug']}.html",
                html_shell(
                    title=f"{row['repo_id']} · Dataset review",
                    body=detail_body,
                    base="../",
                    page="detail",
                    repo_index=index,
                ),
            )

        files = [
            receipt(path, temporary)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        ]
        manifest = {
            "schema_version": SITE_SCHEMA,
            "status": "passed",
            "generated_at": utc_now(),
            "output_root": str(output),
            "repository_count": len(repositories),
            "dataset_page_count": len(repositories),
            "complete_sample_count": len(sample_receipts),
            "inputs": {
                "inventory": input_receipt(
                    original_inputs["inventory"], input_snapshots
                ),
                "evaluations": input_receipt(
                    original_inputs["evaluations"], input_snapshots
                ),
                "sources_config": input_receipt(
                    original_inputs["sources_config"], input_snapshots
                ),
                "quality_summary": input_receipt(
                    original_inputs["quality_summary"], input_snapshots
                ),
                "quality_handoff_receipt": input_receipt(
                    original_inputs["quality_handoff_receipt"], input_snapshots
                ),
                "review_requests": input_receipt(
                    original_inputs["review_requests"], input_snapshots
                ),
                "review_responses": input_receipt(
                    original_inputs["review_responses"], input_snapshots
                ),
                "admission": input_receipt(
                    original_inputs["admission"], input_snapshots
                ),
                "novelty": input_receipt(original_inputs["novelty"], input_snapshots),
                "complete_samples": input_receipt(
                    original_inputs["complete_samples"], input_snapshots
                ),
                "complete_samples_receipt": input_receipt(
                    original_inputs["complete_samples_receipt"], input_snapshots
                ),
                "complete_samples_attestation": input_receipt(
                    original_inputs["complete_samples_attestation"], input_snapshots
                ),
            },
            "security": {
                "bind_address": "127.0.0.1",
                "external_resources": False,
                "content_security_policy": True,
                "sample_text_inserted_with_text_content": True,
                "file_mode": "0600",
                "directory_mode": "0700",
                "sensitive_do_not_share": True,
                "opaque_id_key_sha256": hashlib.sha256(site_key).hexdigest(),
            },
            "files": files,
        }
        write_private(
            temporary / "site_manifest.json", safe_json(manifest, indent=2) + "\n"
        )
        validate_site_directory(temporary)
        verify_site_input_snapshots(input_snapshots)

        if output.exists():
            if not args.replace:
                raise FileExistsError(f"site already exists; use --replace: {output}")
            marker = output / "site_manifest.json"
            if (
                not marker.is_file()
                or read_json(marker).get("schema_version") != SITE_SCHEMA
            ):
                raise ValueError(
                    f"refusing to replace a directory not generated by this tool: {output}"
                )
            backup = output.parent / f".{output.name}.previous-{os.getpid()}"
            if backup.exists():
                raise FileExistsError(backup)
            os.replace(output, backup)
            try:
                verify_site_input_snapshots(input_snapshots)
                os.replace(temporary, output)
            except BaseException:
                os.replace(backup, output)
                raise
            shutil.rmtree(backup)
        else:
            verify_site_input_snapshots(input_snapshots)
            os.replace(temporary, output)
        print(
            safe_json(
                {
                    "ok": True,
                    "site": str(output / "index.html"),
                    "repositories": len(repositories),
                    "complete_samples": len(sample_receipts),
                    "serve": f"python {Path(__file__).resolve()} serve --site-dir {output}",
                }
            )
        )
        return 0
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if stable_inputs_root.exists():
            shutil.rmtree(stable_inputs_root)


def validate_site_directory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "site_manifest.json"
    if not (root / "index.html").is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"not a generated dataset review site: {root}")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != SITE_SCHEMA
        or manifest.get("status") != "passed"
    ):
        raise ValueError(f"unsupported or incomplete site manifest: {manifest_path}")
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"site must not contain symlinks: {candidate}")
        if not candidate.is_file() and not candidate.is_dir():
            raise ValueError(f"site contains a special filesystem entry: {candidate}")
    declared: dict[str, Mapping[str, Any]] = {}
    for row in manifest.get("files", []):
        if not isinstance(row, Mapping):
            raise ValueError("site manifest file row must be an object")
        raw = str(row.get("path", ""))
        relative = Path(raw)
        if (
            not raw
            or relative.is_absolute()
            or ".." in relative.parts
            or raw != relative.as_posix()
            or raw == "site_manifest.json"
            or raw in declared
        ):
            raise ValueError(f"unsafe or duplicate site manifest path: {raw!r}")
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"missing/non-regular site file: {target}")
        try:
            target.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"site file escapes root: {target}") from exc
        if int(row.get("bytes", -1)) != target.stat().st_size or str(
            row.get("sha256", "")
        ) != sha256_file(target):
            raise ValueError(f"site file receipt drift: {target}")
        declared[raw] = row
    actual = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != manifest_path
    }
    if actual != set(declared):
        raise ValueError(
            "site file inventory drift; "
            f"missing={sorted(set(declared) - actual)}, "
            f"unexpected={sorted(actual - set(declared))}"
        )
    allowed_directories = {
        parent.as_posix()
        for name in declared
        for parent in Path(name).parents
        if parent.as_posix() != "."
    }
    actual_directories = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_dir()
    }
    if actual_directories != allowed_directories:
        raise ValueError(
            "site directory inventory drift; "
            f"missing={sorted(allowed_directories - actual_directories)}, "
            f"unexpected={sorted(actual_directories - allowed_directories)}"
        )
    return manifest


class PrivateSiteHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def list_directory(self, path: str):  # type: ignore[no-untyped-def]
        self.send_error(404, "Directory listing disabled")
        return None


def serve_site(args: argparse.Namespace) -> int:
    root = args.site_dir.expanduser().resolve()
    validate_site_directory(root)
    handler = functools.partial(PrivateSiteHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Serving private dataset review at http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument(
        "--inventory",
        type=Path,
        default=here / "configs" / "post_december_inventory.json",
    )
    build.add_argument(
        "--evaluations",
        type=Path,
        default=here / "configs" / "dataset_review_evaluations.json",
    )
    build.add_argument(
        "--sources-config", type=Path, default=here / "configs" / "sources.json"
    )
    build.add_argument("--quality-summary", type=Path)
    build.add_argument("--quality-handoff-receipt", type=Path)
    build.add_argument("--review-requests", type=Path)
    build.add_argument("--review-responses", type=Path)
    build.add_argument("--admission", type=Path)
    build.add_argument("--novelty", type=Path)
    build.add_argument(
        "--complete-samples",
        type=Path,
        help="high-precision-pattern-masked dataset_review_complete_sample_v1 JSONL",
    )
    build.add_argument(
        "--complete-samples-receipt",
        type=Path,
        help="receipt binding the complete sample packet and review requests",
    )
    build.add_argument("--complete-samples-attestation", type=Path)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--replace", action="store_true")
    build.set_defaults(function=build_site)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--site-dir", type=Path, default=DEFAULT_OUTPUT)
    serve.add_argument("--port", type=int, default=8766)
    serve.set_defaults(function=serve_site)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
