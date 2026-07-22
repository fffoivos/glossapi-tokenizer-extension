#!/usr/bin/env python3
"""Fetch a small, revision-bound public preview for the dataset review site.

This is deliberately *not* a corpus acquisition tool.  It reads only a few
rows per repository through Hugging Face's Dataset Server and writes bounded
public-source excerpts for visual review.  The full CPU pipeline remains the
source of truth for normalization, quality profiling, anonymization, and
training admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PACKET_SCHEMA = "dataset_review_public_sample_packet_v1"
INVENTORY_SCHEMA = "post_december_glossapi_inventory_v1"
MAX_RESPONSE_BYTES = 12 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 15
DEFAULT_MAX_TEXT_CHARS = 16_000
DEFAULT_SAMPLES_PER_REPOSITORY = 3
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,200}>")
GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "�", "Î", "Ï")
TEXT_FALLBACK_COLUMNS = (
    "text",
    "content",
    "markdown_text",
    "text_markdown",
    "text_content",
    "plain_text",
    "extracted_md",
    "articles",
    "documents",
    "full_transcript_text",
    "transcription",
    "entry_text",
    "section",
)
ID_FALLBACK_COLUMNS = (
    "id",
    "doc_id",
    "document_id",
    "source_id",
    "url",
    "pdf_url",
    "handle_url",
    "book_id",
    "consultation_id",
    "subject_id",
    "title",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def load_inventory(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if value.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError(f"{path}: unsupported inventory schema")
    rows: list[dict[str, Any]] = []
    for field in (
        "post_cutoff_repositories",
        "older_repositories_with_material_post_cutoff_changes",
    ):
        entries = value.get(field, [])
        if not isinstance(entries, list):
            raise ValueError(f"{path}: {field} must be a list")
        for row in entries:
            if not isinstance(row, dict) or not isinstance(row.get("repo_id"), str):
                raise ValueError(f"{path}: invalid inventory entry")
            rows.append(dict(row))
    if len(rows) != 29 or len({row["repo_id"] for row in rows}) != 29:
        raise ValueError("public sample preview requires the exact 29-repository inventory")
    return rows


def inventory_revision(row: Mapping[str, Any]) -> str:
    revision = row.get("revision") or row.get("current_revision")
    if not isinstance(revision, str) or not GIT_REVISION_RE.fullmatch(revision):
        raise ValueError(f"{row.get('repo_id')}: missing immutable current revision")
    return revision


def load_source_columns(path: Path) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    value = read_json(path)
    if value.get("schema_version") != "full_cpt_sources_v1":
        raise ValueError(f"{path}: unsupported sources config")
    result: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for row in value.get("sources", []):
        if not isinstance(row, dict):
            continue
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            continue
        text_columns = tuple(
            str(name)
            for name in row.get("text_columns", [])
            if isinstance(name, str) and name
        )
        id_columns = tuple(
            str(name)
            for name in row.get("id_columns", [])
            if isinstance(name, str) and name
        )
        result[repo_id] = (text_columns, id_columns)
    return result


def request_json(url: str, token: str, *, limit: int = MAX_RESPONSE_BYTES) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "GlossAPI-public-review-preview/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        body = exc.read(2_048).decode("utf-8", "replace").replace("\n", " ")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from None
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"network error: {getattr(exc, 'reason', exc)}") from None
    if len(payload) > limit:
        raise RuntimeError(f"response exceeds bounded preview limit ({limit} bytes)")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response: {exc}") from None
    if not isinstance(value, dict):
        raise RuntimeError("API response root is not an object")
    return value


def hf_quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def verify_pinned_revision(repo_id: str, revision: str, token: str) -> None:
    url = (
        "https://huggingface.co/api/datasets/"
        f"{repo_id}/revision/{revision}?blobs=true"
    )
    value = request_json(url, token)
    if value.get("id") != repo_id or value.get("sha") != revision:
        raise RuntimeError("Hugging Face repository revision differs from pinned inventory")


def verify_current_head(repo_id: str, revision: str, token: str) -> str:
    value = request_json(
        f"https://huggingface.co/api/datasets/{repo_id}?blobs=true", token
    )
    head = value.get("sha")
    if not isinstance(head, str) or head != revision:
        raise RuntimeError("Hugging Face HEAD differs from the pinned inventory revision")
    return head


def select_split(repo_id: str, token: str) -> tuple[str, str]:
    value = request_json(
        "https://datasets-server.huggingface.co/splits?dataset=" + hf_quote(repo_id),
        token,
    )
    rows = value.get("splits")
    if not isinstance(rows, list) or not rows:
        error = value.get("error") or value.get("failed") or "no Dataset Server split"
        raise RuntimeError(str(error)[:500])
    candidates: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        config = row.get("config")
        split = row.get("split")
        if isinstance(config, str) and config and isinstance(split, str) and split:
            candidates.append((config, split))
    if not candidates:
        raise RuntimeError("Dataset Server returned no usable config/split")
    return sorted(candidates, key=lambda value: (value != ("default", "train"), value))[0]


def fetch_row(
    repo_id: str, config: str, split: str, offset: int, token: str
) -> tuple[dict[str, Any], int, int, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": repo_id,
            "config": config,
            "split": split,
            "offset": str(offset),
            "length": "1",
        }
    )
    value = request_json("https://datasets-server.huggingface.co/rows?" + query, token)
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        error = value.get("error") or "Dataset Server returned no row"
        raise RuntimeError(str(error)[:500])
    response_row = rows[0]
    row = response_row.get("row")
    total = value.get("num_rows_total")
    returned_index = response_row.get("row_idx")
    if (
        not isinstance(row, dict)
        or not isinstance(total, int)
        or total <= 0
        or not isinstance(returned_index, int)
        or returned_index != offset
    ):
        raise RuntimeError("Dataset Server row response lacks a usable row count")
    return row, total, returned_index, response_row.get("truncated_cells", [])


def text_from_value(value: Any, *, budget: int = 2_000_000) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = text_from_value(item, budget=budget)
            if text:
                parts.append(text)
                budget -= len(text)
                if budget <= 0:
                    break
        return "\n\n".join(parts)
    if isinstance(value, dict):
        for key in TEXT_FALLBACK_COLUMNS:
            if key in value:
                text = text_from_value(value[key], budget=budget)
                if text:
                    return text
        parts = [
            text_from_value(item, budget=budget)
            for item in value.values()
            if isinstance(item, (str, list, dict))
        ]
        return "\n\n".join(part for part in parts if part)
    return ""


def select_text(row: Mapping[str, Any], preferred_columns: Iterable[str]) -> tuple[str, str]:
    preferred = tuple(preferred_columns)
    candidates = preferred or TEXT_FALLBACK_COLUMNS
    for name in candidates:
        if name not in row:
            continue
        text = text_from_value(row[name])
        if text.strip():
            return name, text
    if preferred:
        raise RuntimeError(
            "configured text column is absent or empty in this Dataset Server row"
        )
    string_fields = [
        (name, value)
        for name, value in row.items()
        if isinstance(value, str) and len(value.strip()) >= 80
    ]
    if not string_fields:
        raise RuntimeError("no text-bearing column found in Dataset Server row")
    name, text = max(string_fields, key=lambda value: len(value[1]))
    return str(name), str(text)


def select_document_id(row: Mapping[str, Any], preferred_columns: Iterable[str], offset: int) -> str:
    for name in tuple(preferred_columns) + ID_FALLBACK_COLUMNS:
        value = row.get(name)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return f"row-{offset}"


def public_metadata(row: Mapping[str, Any], excluded: set[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in row.items():
        if key in excluded or len(metadata) >= 6:
            continue
        if isinstance(value, (str, int, float, bool)) and not isinstance(value, bytes):
            rendered = str(value).strip()
            if rendered:
                metadata[str(key)] = rendered[:400]
    return metadata


def clip_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    cut = text.rfind("\n", 0, limit)
    if cut < limit // 2:
        cut = text.rfind(" ", 0, limit)
    if cut < limit // 2:
        cut = limit
    return text[:cut].rstrip() + "\n\n[excerpt truncated]", True


def preview_metrics(text: str) -> dict[str, int | float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = sum(count - 1 for count in Counter(lines).values() if count > 1)
    greek = len(GREEK_RE.findall(text))
    return {
        "characters": len(text),
        "lines": len(lines),
        "greek_letter_fraction": round(greek / max(1, len(text)), 6),
        "html_tag_like_count": len(HTML_TAG_RE.findall(text)),
        "mojibake_marker_count": sum(text.count(marker) for marker in MOJIBAKE_MARKERS),
        "replacement_character_count": text.count("�"),
        "repeated_nonblank_line_fraction": round(repeated / max(1, len(lines)), 6),
    }


def make_sample(
    *,
    repo_id: str,
    revision: str,
    config: str,
    split: str,
    offset: int,
    row: Mapping[str, Any],
    returned_row_index: int,
    truncated_cells: Any,
    head_before: str,
    text_columns: tuple[str, ...],
    id_columns: tuple[str, ...],
    max_text_chars: int,
) -> dict[str, Any]:
    text_column, full_text = select_text(row, text_columns)
    displayed_text, truncated = clip_text(full_text, max_text_chars)
    document_id = select_document_id(row, id_columns, offset)
    source_text_sha256 = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    displayed_text_sha256 = hashlib.sha256(displayed_text.encode("utf-8")).hexdigest()
    sample_id = sha256_json(
        {
            "repo_id": repo_id,
            "revision": revision,
            "config": config,
            "split": split,
            "row_index": offset,
            "source_text_sha256": source_text_sha256,
        }
    )
    return {
        "schema_version": "dataset_review_public_sample_v1",
        "sample_id": sample_id,
        "repo_id": repo_id,
        "source_revision": revision,
        "head_before": head_before,
        "source_url": f"https://huggingface.co/datasets/{repo_id}/tree/{revision}",
        "dataset_server_config": config,
        "dataset_server_split": split,
        "row_index": offset,
        "dataset_server_row_index": returned_row_index,
        "dataset_server_truncated_cells": truncated_cells,
        "retrieved_at": utc_now(),
        "dataset_server_row_sha256": sha256_json(row),
        "source_document_id": document_id,
        "text_column": text_column,
        "source_text_characters": len(full_text),
        "displayed_text_characters": len(displayed_text),
        "displayed_text_is_excerpt": truncated,
        "source_text_sha256": source_text_sha256,
        "displayed_text_sha256": displayed_text_sha256,
        "metadata": public_metadata(row, set(text_columns) | {text_column}),
        "preview_metrics": preview_metrics(displayed_text),
        "text": displayed_text,
    }


def availability(repo_id: str, revision: str, reason: str, detail: str) -> dict[str, str]:
    return {
        "repo_id": repo_id,
        "source_revision": revision,
        "reason": reason,
        "detail": detail[:500],
    }


def public_preview_unavailable_detail(error: Exception) -> str:
    """Keep source-browser status useful without copying transport error pages."""

    message = str(error).lower()
    if "scan size limit exceeded" in message:
        return (
            "Dataset Server cannot read a bounded row from this large Parquet "
            "layout; the CPU acquisition stage will provide the preview."
        )
    if "http 429" in message:
        return (
            "The public preview service rate-limited this request; retry from "
            "the worker sample stage is pending."
        )
    if "emptydataseterror" in message or "no (supported) data files" in message:
        return (
            "No Hugging Face text payload is available for a bounded preview; "
            "this source follows a separate acquisition path."
        )
    if "not supported: dataset repository" in message:
        return (
            "The Dataset Server cannot expose this repository for a bounded "
            "preview; worker acquisition is required."
        )
    if "configured text column" in message:
        return "The selected Dataset Server row lacks the configured text field; retry is pending."
    return "A bounded public preview is not currently available; worker acquisition is required."


def build_packet(args: argparse.Namespace) -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("HF_TOKEN is required to access auto-gated public source datasets")
    if args.samples_per_repository < 1 or args.samples_per_repository > 5:
        raise ValueError("--samples-per-repository must be between 1 and 5")
    if args.max_text_chars < 1_000 or args.max_text_chars > 100_000:
        raise ValueError("--max-text-chars must be between 1000 and 100000")
    inventory = load_inventory(args.inventory)
    source_columns = load_source_columns(args.sources_config)
    inventory_by_repo = {str(row["repo_id"]): row for row in inventory}
    selected_repositories = set(args.repository or inventory_by_repo)
    unknown = selected_repositories - set(inventory_by_repo)
    if unknown:
        raise ValueError(f"--repository is not in the review inventory: {sorted(unknown)}")
    retained_samples: list[dict[str, Any]] = []
    retained_unavailable: list[dict[str, Any]] = []
    if args.merge_existing is not None:
        previous = read_json(args.merge_existing)
        if (
            previous.get("schema_version") != PACKET_SCHEMA
            or previous.get("inventory_sha256") != sha256_file(args.inventory)
            or previous.get("sources_config_sha256") != sha256_file(args.sources_config)
        ):
            raise ValueError("--merge-existing packet has incompatible source bindings")
        for row in previous.get("samples", []):
            if isinstance(row, dict) and row.get("repo_id") not in selected_repositories:
                retained_samples.append(row)
        for row in previous.get("unavailable_repositories", []):
            if isinstance(row, dict) and row.get("repo_id") not in selected_repositories:
                retained_unavailable.append(row)
    samples: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for entry in inventory:
        repo_id = str(entry["repo_id"])
        if repo_id not in selected_repositories:
            continue
        revision = inventory_revision(entry)
        try:
            verify_pinned_revision(repo_id, revision, token)
            head_before = verify_current_head(repo_id, revision, token)
            config, split = select_split(repo_id, token)
            text_columns, id_columns = source_columns.get(repo_id, ((), ()))
            first_row, total_rows, first_returned_index, first_truncated_cells = fetch_row(
                repo_id, config, split, 0, token
            )
            offsets = {0}
            if args.samples_per_repository >= 2:
                offsets.add(total_rows // 2)
            if args.samples_per_repository >= 3:
                offsets.add(total_rows - 1)
            if args.samples_per_repository >= 4:
                offsets.add(total_rows // 3)
            if args.samples_per_repository >= 5:
                offsets.add((2 * total_rows) // 3)
            repo_samples: list[dict[str, Any]] = []
            for offset in sorted(offsets):
                if offset == 0:
                    row = first_row
                    returned_row_index = first_returned_index
                    truncated_cells = first_truncated_cells
                else:
                    row, _, returned_row_index, truncated_cells = fetch_row(
                        repo_id, config, split, offset, token
                    )
                repo_samples.append(
                    make_sample(
                        repo_id=repo_id,
                        revision=revision,
                        config=config,
                        split=split,
                        offset=offset,
                        row=row,
                        returned_row_index=returned_row_index,
                        truncated_cells=truncated_cells,
                        head_before=head_before,
                        text_columns=text_columns,
                        id_columns=id_columns,
                        max_text_chars=args.max_text_chars,
                    )
                )
            head_after = verify_current_head(repo_id, revision, token)
            for sample in repo_samples:
                sample["head_after"] = head_after
            samples.extend(repo_samples)
        except (RuntimeError, ValueError) as exc:
            unavailable.append(
                availability(
                    repo_id,
                    revision,
                    "preview_unavailable",
                    public_preview_unavailable_detail(exc),
                )
            )
    samples = retained_samples + samples
    unavailable = retained_unavailable + unavailable
    samples.sort(key=lambda row: (str(row["repo_id"]), int(row["row_index"])))
    unavailable.sort(key=lambda row: row["repo_id"])
    sampled_repositories = {str(row["repo_id"]) for row in samples}
    unavailable_repositories = {str(row["repo_id"]) for row in unavailable}
    if (
        sampled_repositories & unavailable_repositories
        or sampled_repositories | unavailable_repositories != set(inventory_by_repo)
    ):
        raise ValueError("preview packet does not close over the 29 review repositories")
    packet = {
        "schema_version": PACKET_SCHEMA,
        "generated_at": utc_now(),
        "mode": "bounded_public_source_preview",
        "inventory_sha256": sha256_file(args.inventory),
        "sources_config_sha256": sha256_file(args.sources_config),
        "samples_per_repository_requested": args.samples_per_repository,
        "max_text_chars": args.max_text_chars,
        "sampled_repositories": sorted(sampled_repositories),
        "unavailable_repositories": unavailable,
        "samples": samples,
    }
    if args.output.exists() and not args.replace:
        raise FileExistsError(f"refusing to overwrite {args.output}; use --replace")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "samples": len(samples),
                "sampled_repositories": len(packet["sampled_repositories"]),
                "unavailable_repositories": len(unavailable),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", type=Path, default=here / "configs" / "post_december_inventory.json"
    )
    parser.add_argument("--sources-config", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-repository", type=int, default=DEFAULT_SAMPLES_PER_REPOSITORY)
    parser.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS)
    parser.add_argument(
        "--repository",
        action="append",
        help="one review-inventory repository to refresh; may be given repeatedly",
    )
    parser.add_argument(
        "--merge-existing",
        type=Path,
        help="retain preview rows for repositories not selected with --repository",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(build_packet(parse_args()))
    except (ValueError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
