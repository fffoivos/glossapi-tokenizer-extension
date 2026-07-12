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
import functools
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping


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


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
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


def input_receipt(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
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
    )
    return (
        result.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )


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


def source_repo_map(sources: Mapping[str, Any]) -> dict[str, str]:
    result = {"nanochat_base": str(sources.get("base", {}).get("repo_id", ""))}
    for row in sources.get("sources", []):
        if isinstance(row, dict):
            result[str(row.get("source_id", ""))] = str(row.get("repo_id", ""))
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
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    value = read_json(path)
    if value.get("schema_version") != QUALITY_SCHEMA or value.get("status") != "passed":
        raise ValueError(f"{path}: unsupported or incomplete quality summary")
    result = {}
    for row in value.get("repositories", []):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: quality repository row must be an object")
        repo_id = str(row.get("repo_id", ""))
        if not repo_id or repo_id in result:
            raise ValueError(f"{path}: duplicate or empty quality repo_id")
        result[repo_id] = row
    scan_mode = str(value.get("scan_mode", ""))
    if scan_mode not in {"review_sample", "full_scan"}:
        raise ValueError(f"{path}: invalid quality scan_mode")
    global_summary = (
        value.get("global") if isinstance(value.get("global"), dict) else {}
    )
    selected_source_ids = sorted(
        str(item) for item in value.get("selected_source_ids", []) if str(item)
    )
    excluded_source_ids = sorted(
        str(item) for item in value.get("excluded_source_ids", []) if str(item)
    )
    if set(selected_source_ids) & set(excluded_source_ids):
        raise ValueError(f"{path}: selected/excluded quality sources overlap")
    return result, {
        "scan_mode": scan_mode,
        "documents": int(global_summary.get("documents", 0)),
        # This profiler intentionally selects a population (and normally
        # excludes nanochat_base).  Never promote that to a corpus-wide claim.
        "is_corpus_wide": False,
        "label": (
            "Representative source-review sample"
            if scan_mode == "review_sample"
            else "Full scan of selected canonical sources"
        ),
        "selected_source_ids": selected_source_ids,
        "excluded_source_ids": excluded_source_ids,
    }


def load_requests(
    path: Path | None,
    source_id_to_repo: Mapping[str, str],
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
        repo_id = str(
            source.get("source_repo_id") or source_id_to_repo.get(source_id, "")
        )
        dataset = str(row.get("source_dataset", ""))
        if not repo_id or not dataset:
            raise ValueError(
                f"{path}:{line_number}: request cannot be mapped to a repository"
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
        revision = str(source.get("source_revision", ""))
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
        if isinstance(row.get("quality_score"), int):
            scores[repo_id].append(int(row["quality_score"]))
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
    for row in value.get("sources", []):
        if not isinstance(row, dict):
            continue
        repo_id = dataset_to_repo.get(str(row.get("source_dataset", "")))
        if repo_id:
            result[repo_id].append(
                {
                    "source_dataset": str(row.get("source_dataset", "")),
                    "rows": int(row.get("rows", 0)),
                    "identity_word_tokens": int(row.get("identity_word_tokens", 0)),
                    "exact_unique_word_tokens": (
                        int(row["exact_unique_word_tokens"])
                        if isinstance(row.get("exact_unique_word_tokens"), int)
                        else None
                    ),
                    "novel_word_tokens_after_lineage_resolution": int(
                        row.get("novel_word_tokens_after_lineage_resolution", 0)
                    ),
                    "novel_token_fraction": float(row.get("novel_token_fraction", 0.0)),
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


def row_total(row: Mapping[str, Any]) -> int | None:
    rows = row.get("rows")
    if isinstance(rows, dict):
        for field in ("footer", "card"):
            if isinstance(rows.get(field), int):
                return int(rows[field])
    for field in (
        "new_asset_footer_rows",
        "current_metadata_footer_rows",
        "current_card_documents",
    ):
        if isinstance(row.get(field), int):
            return int(row[field])
    return None


def byte_total(row: Mapping[str, Any]) -> int | None:
    for field in ("data_artifact_bytes", "new_asset_bytes", "payload_bytes_current"):
        if isinstance(row.get(field), int):
            return int(row[field])
    return None


def token_total(row: Mapping[str, Any]) -> int | None:
    card = row.get("card_tokens")
    if isinstance(card, dict) and isinstance(card.get("value"), int):
        return int(card["value"])
    for field in ("new_asset_card_tokens_sum", "current_card_tokens"):
        if isinstance(row.get(field), int):
            return int(row[field])
    return None


def slug(index: int, repo_id: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", repo_id.casefold()).strip("-")
    return f"{index + 1:02d}-{name[:80]}"


def write_complete_samples(
    path: Path | None,
    *,
    packet_receipt_path: Path | None,
    review_requests_path: Path | None,
    output: Path,
    requests: dict[str, dict[str, Any]],
    visible_repositories: set[str],
) -> tuple[list[dict[str, Any]], int]:
    if path is None and packet_receipt_path is None:
        return [], 0
    if path is None or packet_receipt_path is None or review_requests_path is None:
        raise ValueError(
            "--complete-samples requires --complete-samples-receipt and --review-requests"
        )
    packet_receipt = read_json(packet_receipt_path)
    if (
        packet_receipt.get("schema_version") != SAMPLE_RECEIPT_SCHEMA
        or packet_receipt.get("status") != "passed"
        or packet_receipt.get("high_precision_identifier_patterns_masked") is not True
    ):
        raise ValueError(f"{packet_receipt_path}: incomplete sample-packet receipt")
    declared_output = packet_receipt.get("output")
    if not isinstance(declared_output, dict):
        raise ValueError(f"{packet_receipt_path}: missing output receipt")
    declared_path = Path(str(declared_output.get("path", "")))
    if not declared_path.is_absolute():
        declared_path = packet_receipt_path.resolve().parent / declared_path
    if declared_path.resolve() != path.resolve():
        raise ValueError(f"{packet_receipt_path}: sample packet path drift")
    if (
        int(declared_output.get("bytes", -1)) != path.stat().st_size
        or str(declared_output.get("sha256", "")) != sha256_file(path)
        or packet_receipt.get("review_requests", {}).get("sha256")
        != sha256_file(review_requests_path)
    ):
        raise ValueError(f"{packet_receipt_path}: sample packet/upstream receipt drift")

    written: list[dict[str, Any]] = []
    seen: set[str] = set()
    excluded_outside_inventory = 0
    site_index = {
        request["canonical_sample_id"]: request["site_record"]
        for request in requests.values()
        if request["repo_id"] in visible_repositories
    }
    for line_number, row in iter_jsonl(path):
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
        if (
            identity["source_id"] != expected["source_id"]
            or repo_id != expected["repo_id"]
            or identity["source_revision"] != expected["source_revision"]
            or dataset != expected["source_dataset"]
            or identity["canonical_display_document_id"]
            != expected["canonical_display_document_id"]
        ):
            raise ValueError(f"{path}:{line_number}: complete sample identity drift")
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
            "redaction_counts": row.get("redaction_counts", {}),
            "characters": len(text),
            "text": text,
        }
        write_private(sample_path, safe_json(payload))
        relative = sample_path.relative_to(output).as_posix()
        site_record["complete_document_available"] = True
        site_record["complete_document_path"] = relative
        written.append(receipt(sample_path, output))
    if len(seen) != int(declared_output.get("rows", -1)):
        raise ValueError(f"{path}: sample row count differs from receipt")
    missing = set(site_index) - seen
    if missing:
        # Missing complete documents remain visible as explicit unavailable states.
        for sample_id in missing:
            site_index[sample_id]["complete_document_unavailable_reason"] = (
                "not_present_in_complete_sample_packet"
            )
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
    inventory = load_inventory(args.inventory)
    repos = {str(row["repo_id"]) for row in inventory}
    evaluations = load_evaluations(args.evaluations, repos)
    sources = read_json(args.sources_config)
    quality, quality_scope = load_quality(args.quality_summary)
    supplemental_quality_repositories = sorted(set(quality) - repos)
    site_key = secrets.token_bytes(32)
    requests, dataset_to_repo, by_repo = load_requests(
        args.review_requests, source_repo_map(sources), site_key
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

    output = args.output_dir.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    temporary.chmod(0o700)
    try:
        sample_receipts, excluded_complete_samples = write_complete_samples(
            args.complete_samples,
            packet_receipt_path=getattr(args, "complete_samples_receipt", None),
            review_requests_path=args.review_requests,
            output=temporary,
            requests=requests,
            visible_repositories=repos,
        )
        repositories: list[dict[str, Any]] = []
        for index, row in enumerate(inventory):
            repo_id = str(row["repo_id"])
            evaluation = evaluations[repo_id]
            state = payload_state(row)
            if state == "external_unavailable" and repo_id in quality:
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
                "inventory": input_receipt(args.inventory),
                "evaluations": input_receipt(args.evaluations),
                "sources_config": input_receipt(args.sources_config),
                "quality_summary": input_receipt(args.quality_summary),
                "review_requests": input_receipt(args.review_requests),
                "review_responses": input_receipt(args.review_responses),
                "admission": input_receipt(args.admission),
                "novelty": input_receipt(args.novelty),
                "complete_samples": input_receipt(args.complete_samples),
                "complete_samples_receipt": input_receipt(
                    getattr(args, "complete_samples_receipt", None)
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
                os.replace(temporary, output)
            except BaseException:
                os.replace(backup, output)
                raise
            shutil.rmtree(backup)
        else:
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
