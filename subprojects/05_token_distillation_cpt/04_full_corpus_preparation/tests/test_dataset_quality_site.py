from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


QUALITY = load_module(
    "phase04_dataset_quality_rust",
    HERE / "scripts" / "profile_dataset_quality_rust.py",
)
sys.modules["profile_dataset_quality_rust"] = QUALITY
SITE = load_module(
    "phase04_dataset_review_site",
    HERE / "scripts" / "build_dataset_review_site.py",
)
EXPORTER = load_module(
    "phase04_export_dataset_review_samples",
    HERE / "scripts" / "export_dataset_review_samples.py",
)
PRESENTATION = load_module(
    "phase04_dataset_review_presentation",
    HERE / "scripts" / "build_dataset_review_presentation.py",
)
SOURCES_CONFIG = HERE / "configs" / "sources.json"


def source_identities(path: Path = SOURCES_CONFIG) -> dict[str, dict[str, object]]:
    return SITE.source_identity_map(json.loads(path.read_text(encoding="utf-8")))


def write_sources_config(
    path: Path,
    sources: list[dict[str, object]],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_sources_v1",
                "base": {
                    "repo_id": "owner/base",
                    "revision": "b" * 40,
                    "role": "base",
                },
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return path


def tracked_inventory() -> dict:
    return json.loads((HERE / "configs" / "post_december_inventory.json").read_text())


def write_public_preview_packet(path: Path) -> Path:
    inventory = tracked_inventory()
    entries = [
        *inventory["post_cutoff_repositories"],
        *inventory["older_repositories_with_material_post_cutoff_changes"],
    ]
    repo_id = "glossAPI/e-nautilia"
    row = next(value for value in entries if value["repo_id"] == repo_id)
    revision = str(row["revision"])
    text = "Δημόσιο δείγμα <b>κειμένου</b> για οπτική επιθεώρηση."
    source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sample_id = QUALITY.sha256_json(
        {
            "repo_id": repo_id,
            "revision": revision,
            "config": "default",
            "split": "train",
            "row_index": 0,
            "source_text_sha256": source_sha,
        }
    )
    unavailable = [
        {
            "repo_id": str(value["repo_id"]),
            "source_revision": str(value.get("revision") or value.get("current_revision")),
            "reason": "preview_unavailable",
            "detail": "Worker acquisition required for this bounded preview.",
        }
        for value in entries
        if value["repo_id"] != repo_id
    ]
    packet = {
        "schema_version": "dataset_review_public_sample_packet_v1",
        "generated_at": "2026-07-12T00:00:00Z",
        "mode": "bounded_public_source_preview",
        "inventory_sha256": QUALITY.sha256_file(
            HERE / "configs" / "post_december_inventory.json"
        ),
        "sources_config_sha256": QUALITY.sha256_file(SOURCES_CONFIG),
        "samples_per_repository_requested": 1,
        "max_text_chars": 16000,
        "sampled_repositories": [repo_id],
        "unavailable_repositories": unavailable,
        "samples": [
            {
                "schema_version": "dataset_review_public_sample_v1",
                "sample_id": sample_id,
                "repo_id": repo_id,
                "source_revision": revision,
                "head_before": revision,
                "source_url": f"https://huggingface.co/datasets/{repo_id}/tree/{revision}",
                "dataset_server_config": "default",
                "dataset_server_split": "train",
                "row_index": 0,
                "dataset_server_row_index": 0,
                "dataset_server_truncated_cells": [],
                "retrieved_at": "2026-07-12T00:00:00Z",
                "dataset_server_row_sha256": "a" * 64,
                "source_document_id": "public-doc-1",
                "text_column": "content",
                "source_text_characters": len(text),
                "displayed_text_characters": len(text),
                "displayed_text_is_excerpt": False,
                "source_text_sha256": source_sha,
                "displayed_text_sha256": source_sha,
                "metadata": {"title": "Public sample"},
                "preview_metrics": {
                    "characters": len(text),
                    "lines": 1,
                    "greek_letter_fraction": 0.6,
                    "html_tag_like_count": 2,
                    "mojibake_marker_count": 0,
                    "replacement_character_count": 0,
                    "repeated_nonblank_line_fraction": 0.0,
                },
                "text": text,
                "head_after": revision,
            }
        ],
    }
    path.write_text(json.dumps(packet), encoding="utf-8")
    return path


def review_request(
    uid: str,
    *,
    source_id: str = "diavgeia",
    repo_id: str = "glossAPI/diavgeia",
    dataset: str = "diavgeia",
    doc_id: str = "ADA-1",
    preview_sentinel: str | None = None,
    revision: str | None = None,
) -> dict[str, object]:
    if revision is None:
        revision = str(source_identities()[source_id]["revision"])
    row: dict[str, object] = {
        "schema_version": "source_quality_review_request_v1",
        "reviewer_slot": "primary",
        "sample_id": uid,
        "source_dataset": dataset,
        "sampling_stratum": "risk",
        "source": {
            "source_id": source_id,
            "source_repo_id": repo_id,
            "source_revision": revision,
            "source_doc_id": doc_id,
        },
    }
    if preview_sentinel is not None:
        row["document"] = {"mode": "full", "text": preview_sentinel}
    return row


def complete_sample(
    uid: str,
    text: str,
    *,
    source_id: str = "diavgeia",
    repo_id: str = "glossAPI/diavgeia",
    dataset: str = "diavgeia",
    doc_id: str = "ADA-1",
    revision: str | None = None,
) -> dict[str, object]:
    if revision is None:
        revision = str(source_identities()[source_id]["revision"])
    return {
        "schema_version": "dataset_review_complete_sample_v1",
        "sample_id": uid,
        "source_id": source_id,
        "source_repo_id": repo_id,
        "source_revision": revision,
        "source_dataset": dataset,
        "display_document_id": QUALITY.display_document_id(doc_id),
        "normalized_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "profile_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "profile_text_variant": "high_precision_identifier_masked_review_sample",
        "input_shard_path": f"{source_id}/part.parquet",
        "input_shard_sha256": hashlib.sha256(
            f"fixture-shard:{source_id}".encode()
        ).hexdigest(),
        "input_row_index": 7,
        "private_data_true": False,
        "corrected_version_present": False,
        "high_precision_identifier_patterns_masked": True,
        "redaction_counts": {},
        "text": text,
    }


def write_packet_receipt(
    path: Path,
    *,
    packet: Path,
    requests: Path,
    rows: int,
    normalization: Path | None = None,
    sources_config: Path = SOURCES_CONFIG,
) -> Path:
    packet_rows = [
        json.loads(line) for line in packet.read_text().splitlines() if line.strip()
    ]
    assert len(packet_rows) == rows
    normalization_bytes = (
        normalization.read_bytes() if normalization else b"normalization"
    )
    normalization_receipt = {
        "path": "normalization_manifest.json",
        "bytes": len(normalization_bytes),
        "sha256": hashlib.sha256(normalization_bytes).hexdigest(),
    }
    requests_receipt = {
        "path": requests.name,
        "bytes": requests.stat().st_size,
        "sha256": hashlib.sha256(requests.read_bytes()).hexdigest(),
    }
    shard_groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in packet_rows:
        key = (
            str(row["source_id"]),
            str(row["input_shard_path"]),
            str(row["input_shard_sha256"]),
        )
        shard_groups.setdefault(key, []).append(row)
    input_shards = [
        {
            "source_id": source_id,
            "path": shard_path,
            "bytes": 1,
            "rows": max(int(row["input_row_index"]) for row in group) + 1,
            "sha256": shard_sha,
        }
        for (source_id, shard_path, shard_sha), group in sorted(shard_groups.items())
    ]
    checkpoint_inventory = [
        {
            "input_shard_sha256": shard["sha256"],
            "checkpoint_receipt_sha256": hashlib.sha256(
                f"checkpoint:{shard['sha256']}".encode()
            ).hexdigest(),
            "output_sha256": hashlib.sha256(
                f"output:{shard['sha256']}".encode()
            ).hexdigest(),
            "selected_rows": sum(
                row["input_shard_sha256"] == shard["sha256"] for row in packet_rows
            ),
        }
        for shard in input_shards
    ]
    redaction_totals: dict[str, int] = {}
    for row in packet_rows:
        for name, count in row["redaction_counts"].items():
            redaction_totals[name] = redaction_totals.get(name, 0) + int(count)
    dependency_hashes = {
        "build_source_review_packet": QUALITY.sha256_file(
            Path(EXPORTER.redact_direct_identifiers.__code__.co_filename).resolve()
        ),
        "greek_pii": QUALITY.sha256_file(
            Path(EXPORTER.mask_greek_identifiers.__code__.co_filename).resolve()
        ),
        "profile_dataset_quality_rust": QUALITY.sha256_file(
            Path(EXPORTER.metadata_flags.__code__.co_filename).resolve()
        ),
    }
    contract = {
        "schema_version": "dataset_review_sample_export_contract_v1",
        "normalization_manifest_sha256": normalization_receipt["sha256"],
        "review_requests_sha256": requests_receipt["sha256"],
        "exporter_script_sha256": QUALITY.sha256_file(
            Path(EXPORTER.__file__).resolve()
        ),
        "redaction_dependency_sha256": dependency_hashes,
        "redaction_pipeline": "high_precision_identifier_patterns_v1",
        "batch_size": 8,
        "selected_sample_count": rows,
    }
    contract_sha = QUALITY.sha256_json(contract)
    contract_receipt = {
        "path": "sample-export-checkpoints/contract.json",
        "bytes": 1,
        "sha256": "c" * 64,
    }
    identities = []
    tracked_sources = source_identities(sources_config)
    for source_id in sorted({str(row["source_id"]) for row in packet_rows}):
        source_rows = [row for row in packet_rows if row["source_id"] == source_id]
        source_shards = [row for row in input_shards if row["source_id"] == source_id]
        tracked = tracked_sources[source_id]
        assert source_rows[0]["source_repo_id"] == tracked["repo_id"]
        assert source_rows[0]["source_revision"] == tracked["revision"]
        identities.append(
            {
                "source_id": source_id,
                "repo_id": tracked["repo_id"],
                "revision": tracked["revision"],
                "role": tracked["role"],
                "acquisition_kind": tracked["acquisition_kind"],
                "mdc_dataset_id": tracked["mdc_dataset_id"],
                "source_config_sha256": tracked["source_config_sha256"],
                "documents": len(source_rows),
                "shards": len(source_shards),
                "shard_inventory_sha256": QUALITY.sha256_json(source_shards),
                "acquisition_selected_file_count": 1,
                "acquisition_selected_bytes": 1,
                "acquisition_file_inventory_sha256": hashlib.sha256(
                    f"acquisition:{source_id}".encode()
                ).hexdigest(),
            }
        )
    packet_output = {
        "path": packet.name,
        "bytes": packet.stat().st_size,
        "rows": rows,
        "sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
    }
    attestation = {
        "schema_version": "dataset_review_complete_sample_site_attestation_v1",
        "status": "passed",
        "created_at": "2026-07-12T00:00:00Z",
        "packet": packet_output,
        "review_requests": requests_receipt,
        "primary_sample_count": rows,
        "primary_sample_id_inventory_sha256": QUALITY.sha256_json(
            sorted(str(row["sample_id"]) for row in packet_rows)
        ),
        "normalization": {
            "schema_version": "full_cpt_normalization_manifest_v1",
            "manifest": normalization_receipt,
            "sources_config_sha256": QUALITY.sha256_file(sources_config),
            "acquisition_receipt_sha256": "2" * 64,
            "source_identities": identities,
            "source_identity_inventory_sha256": QUALITY.sha256_json(identities),
            "normalized_shard_inventory_sha256": QUALITY.sha256_json(input_shards),
            "input_shards": input_shards,
            "input_shard_inventory_sha256": QUALITY.sha256_json(input_shards),
        },
        "export_contract": {
            "receipt": contract_receipt,
            "canonical_sha256": contract_sha,
            "value": contract,
        },
        "checkpoint_closure": {
            "count": len(checkpoint_inventory),
            "selected_rows": rows,
            "inventory_sha256": QUALITY.sha256_json(checkpoint_inventory),
            "receipt_closure_sha256": "3" * 64,
            "checkpoint_text_outputs_rehashed_for_attestation": True,
        },
        "masking": {
            "pipeline": "high_precision_identifier_patterns_v1",
            "implementation_sha256": {
                "exporter": contract["exporter_script_sha256"],
                **dependency_hashes,
            },
            "high_precision_identifier_patterns_masked": True,
            "private_data_true_rows": 0,
            "redaction_totals": redaction_totals,
            "redaction_totals_sha256": QUALITY.sha256_json(redaction_totals),
        },
    }
    attestation_path = path.with_name("samples-attestation.json")
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    value = {
        "schema_version": "dataset_review_complete_sample_packet_receipt_v1",
        "status": "passed",
        "normalization_manifest": normalization_receipt,
        "canonical_root": "/receipt-bound/canonical",
        "review_requests": requests_receipt,
        "export_contract": {
            **contract_receipt,
            "contract_sha256": contract_sha,
        },
        "site_attestation": {
            "path": attestation_path.name,
            "bytes": attestation_path.stat().st_size,
            "sha256": hashlib.sha256(attestation_path.read_bytes()).hexdigest(),
        },
        "input_shards": input_shards,
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_inventory_sha256": QUALITY.sha256_json(checkpoint_inventory),
        "output": packet_output,
        "redaction_totals": redaction_totals,
        "high_precision_identifier_patterns_masked": True,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return attestation_path


def refresh_packet_attestation_receipts(
    *, packet: Path, receipt_path: Path, attestation_path: Path
) -> None:
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["packet"].update(
        {
            "bytes": packet.stat().st_size,
            "sha256": QUALITY.sha256_file(packet),
        }
    )
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output"] = attestation["packet"]
    receipt["site_attestation"].update(
        {
            "bytes": attestation_path.stat().st_size,
            "sha256": QUALITY.sha256_file(attestation_path),
        }
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def quality_distribution(documents: int, value: float = 0.0) -> dict[str, object]:
    metric = value if documents else None
    return {
        "count": documents,
        "min": metric,
        "mean": metric,
        "p10_approx": metric,
        "p50_approx": metric,
        "p90_approx": metric,
        "p99_approx": metric,
        "max": metric,
        "quantile_sample_documents": documents,
    }


def quality_statistics(repo_id: str | None, documents: int) -> dict[str, object]:
    counts = {name: 0 for name in QUALITY.DOCUMENT_COUNTERS}
    rates = {
        name.removesuffix("_documents") + "_rate": 0.0
        for name in QUALITY.DOCUMENT_COUNTERS
    }
    value: dict[str, object] = {
        "documents": documents,
        "characters": documents * 100,
        "bytes_utf8": documents * 150,
        "source_datasets": [repo_id.rsplit("/", 1)[-1]] if repo_id else ["global"],
        "document_counts": counts,
        "document_rates": rates,
        "distributions": {
            name: quality_distribution(documents)
            for name in QUALITY.DISTRIBUTION_METRICS
        },
        "template_concentration": {
            "documents_with_template": 0,
            "unique_templates": 0,
            "top_1_fraction": 0.0,
            "top_10_fraction": 0.0,
        },
    }
    if repo_id is not None:
        value["repo_id"] = repo_id
    return value


def write_quality_summary_and_handoff(
    tmp_path: Path,
    *,
    repositories: list[tuple[str, str, str]] | None = None,
    scan_mode: str = "review_sample",
    sources_config: Path = SOURCES_CONFIG,
) -> tuple[Path, Path]:
    tracked_sources = source_identities(sources_config)
    repositories = repositories or [
        (
            "kallipos_sections",
            str(tracked_sources["kallipos_sections"]["repo_id"]),
            str(tracked_sources["kallipos_sections"]["revision"]),
        ),
        (
            "diavgeia",
            str(tracked_sources["diavgeia"]["repo_id"]),
            str(tracked_sources["diavgeia"]["revision"]),
        ),
    ]
    repositories = sorted(repositories, key=lambda row: row[1])
    shards = [
        {
            "source_id": source_id,
            "path": f"{source_id}/part.parquet",
            "bytes": 10,
            "sha256": hashlib.sha256(f"quality-shard:{source_id}".encode()).hexdigest(),
            "rows": 1,
            **({"batches": 1} if scan_mode == "full_scan" else {}),
        }
        for source_id, _, _ in sorted(repositories)
    ]
    contract_shards = [
        {key: shard[key] for key in ("source_id", "path", "bytes", "sha256", "rows")}
        for shard in shards
    ]
    if scan_mode == "review_sample":
        checkpoint_inventory = [
            {
                "receipt_path": "batches/review/receipt.json",
                "receipt_sha256": hashlib.sha256(b"receipt:review").hexdigest(),
                "output_sha256": hashlib.sha256(b"output:review").hexdigest(),
                "rows": len(repositories),
                "input_shard_source_id": "exact_source_review_sample",
                "input_shard_path": "review-sample/samples",
                "input_shard_sha256": "1" * 64,
                "batch_index": 0,
                "row_start": 0,
                "row_end_exclusive": len(repositories),
            }
        ]
    else:
        checkpoint_inventory = [
            {
                "receipt_path": f"batches/{index}/receipt.json",
                "receipt_sha256": hashlib.sha256(
                    f"receipt:{index}".encode()
                ).hexdigest(),
                "output_sha256": hashlib.sha256(f"output:{index}".encode()).hexdigest(),
                "rows": 1,
                "input_shard_source_id": str(shard["source_id"]),
                "input_shard_path": str(shard["path"]),
                "input_shard_sha256": shard["sha256"],
                "batch_index": 0,
                "row_start": 0,
                "row_end_exclusive": 1,
            }
            for index, shard in enumerate(shards)
        ]
    contract_receipt = {"path": "contract.json", "bytes": 1, "sha256": "c" * 64}
    normalization_receipt = {
        "path": "/clariden/normalization_manifest.json",
        "bytes": 1,
        "sha256": "d" * 64,
    }
    build_receipt = {"path": "/clariden/build.json", "bytes": 1, "sha256": "e" * 64}
    documents = len(repositories)
    selected_sources = sorted(row[0] for row in repositories)
    global_statistics = quality_statistics(None, documents)
    global_statistics["source_datasets"] = sorted(
        {repo_id.rsplit("/", 1)[-1] for _, repo_id, _ in repositories}
    )
    route_accumulator = QUALITY.RouteCoverageAccumulator()
    for source_id, _, _ in repositories:
        route_accumulator.add(
            {
                "source_id": source_id,
                "source_route": "mixed",
                "review_route": "mixed",
                "extraction_route": "mixed",
                "observed_extraction_route": "mixed",
                "observed_extraction_route_basis": "declared_extraction_route_fallback",
                "observed_extraction_route_evidence": "roster:extraction_route",
                "observed_extraction_route_priority": "logical_primary",
            },
            context=f"fixture.{source_id}",
        )
    route_coverage = route_accumulator.finish()
    summary = {
        "schema_version": "dataset_quality_summary_v2",
        "status": "passed",
        "created_at": "2026-07-12T00:00:00Z",
        "mode": "diagnostic_only_no_cleaned_text_persisted",
        "scan_mode": scan_mode,
        "contract_sha256": "f" * 64,
        "contract": contract_receipt,
        "normalization_manifest": normalization_receipt,
        "normalization_schema_version": "full_cpt_normalization_manifest_v1",
        "glossapi_build_receipt": build_receipt,
        "glossapi_commit": "a" * 40,
        "batch_size": 4096,
        "threads": 256,
        "quantile_sample_size": 8192,
        "selected_source_ids": selected_sources,
        "excluded_source_ids": ["nanochat_base"],
        "input_shards": shards,
        "batch_checkpoints": {
            "count": len(checkpoint_inventory),
            "rows": documents,
            "inventory_sha256": QUALITY.sha256_json(checkpoint_inventory),
            "inventory": checkpoint_inventory,
        },
        "document_output": {
            "path": "dataset_quality_document_v2.parquet",
            "bytes": 10,
            "sha256": "9" * 64,
            "rows": documents,
        },
        "global": global_statistics,
        "repositories": [
            quality_statistics(repo_id, 1) for _, repo_id, _ in repositories
        ],
        "route_coverage": route_coverage,
        "metric_notes": {
            "rust_noise_badness_score": "diagnostic",
            "cleaner_removed_character_fraction": "diagnostic",
            "approximate_quantiles": "bounded",
            "zero_badness_zero_greek_guard": "guarded",
            "profile_scope": "selected population",
        },
    }
    summary_path = tmp_path / "dataset_quality_summary_v2.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    projection = QUALITY.validate_and_project_quality_summary(summary)
    identities = [
        {
            "source_id": source_id,
            "repo_id": repo_id,
            "revision": revision,
            "role": tracked_sources[source_id]["role"],
            "acquisition_kind": tracked_sources[source_id]["acquisition_kind"],
            "mdc_dataset_id": tracked_sources[source_id]["mdc_dataset_id"],
            "source_config_sha256": tracked_sources[source_id]["source_config_sha256"],
            "documents": 1,
            "shards": 1,
            "shard_inventory_sha256": hashlib.sha256(
                f"identity-shards:{source_id}".encode()
            ).hexdigest(),
            "acquisition_selected_file_count": 1,
            "acquisition_selected_bytes": 10,
            "acquisition_file_inventory_sha256": hashlib.sha256(
                f"identity-files:{source_id}".encode()
            ).hexdigest(),
        }
        for source_id, repo_id, revision in sorted(repositories)
    ]
    review_sample = None
    if scan_mode == "review_sample":
        review_sample = {
            "review_sample_packet": {
                "path": "/clariden/samples",
                "bytes": 1,
                "sha256": "1" * 64,
            },
            "review_sample_receipt": {
                "path": "/clariden/receipt",
                "bytes": 1,
                "sha256": "2" * 64,
            },
            "review_sample_attestation": {
                "path": "/clariden/attestation",
                "bytes": 1,
                "sha256": "3" * 64,
            },
            "review_requests": {
                "path": "/clariden/requests",
                "bytes": 1,
                "sha256": "4" * 64,
            },
            "documents": documents,
            "text_variant": "high_precision_identifier_masked_review_sample",
        }
    handoff = {
        "schema_version": "dataset_quality_site_handoff_v2",
        "status": "passed",
        "created_at": "2026-07-12T00:00:00Z",
        "summary": {
            "path": summary_path.name,
            "bytes": summary_path.stat().st_size,
            "sha256": QUALITY.sha256_file(summary_path),
        },
        "scan_mode": scan_mode,
        "aggregate_projection_sha256": QUALITY.sha256_json(projection),
        "normalization": {
            "schema_version": "full_cpt_normalization_manifest_v1",
            "manifest": {
                "path": "normalization_manifest.json",
                "bytes": normalization_receipt["bytes"],
                "sha256": normalization_receipt["sha256"],
            },
            "sources_config_sha256": QUALITY.sha256_file(sources_config),
            "acquisition_receipt_sha256": "6" * 64,
            "source_identities": identities,
            "source_identity_inventory_sha256": QUALITY.sha256_json(identities),
            "normalized_shard_inventory_sha256": "7" * 64,
            "selected_normalized_source_ids": selected_sources,
            "selected_normalized_shard_inventory_sha256": QUALITY.sha256_json(
                contract_shards
            ),
        },
        "build": {
            "receipt_sha256": build_receipt["sha256"],
            "commit": summary["glossapi_commit"],
            "cargo_lock_inventory_sha256": "8" * 64,
            "module_inventory_sha256": "a" * 64,
            "runtime": {
                "python": "3.12",
                "platform": "linux",
                "machine": "x86_64",
                "rustc": "rustc",
                "cargo": "cargo",
                "maturin": "maturin",
            },
        },
        "contract": {
            "receipt": contract_receipt,
            "canonical_sha256": summary["contract_sha256"],
            "schema_version": "dataset_quality_rust_contract_v1",
            "document_schema": "dataset_quality_document_v2",
            "selected_shard_inventory_sha256": QUALITY.sha256_json(contract_shards),
            "excluded_source_ids": summary["excluded_source_ids"],
            "profiler_script_sha256": "b" * 64,
            "review_sample": review_sample,
        },
        "document_output": summary["document_output"],
        "route_coverage": route_coverage,
        "checkpoint_closure": {
            "count": len(checkpoint_inventory),
            "rows": documents,
            "inventory_sha256": summary["batch_checkpoints"]["inventory_sha256"],
            "receipt_closure_sha256": "0" * 64,
            "checkpoint_outputs_rehashed_for_handoff": True,
            "consolidated_document_output_rehashed_for_handoff": True,
        },
    }
    handoff_path = tmp_path / "quality-handoff.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    return summary_path, handoff_path


def write_normalization_fixture(
    tmp_path: Path,
    *,
    canonical_root: Path,
    shard: Path,
    source_id: str,
    repo_id: str,
    revision: str,
    rows: int,
    mdc_dataset_id: str | None = None,
    acquisition_receipt_kind: str | None = None,
    structured_profile_contract: dict[str, object] | None = None,
) -> Path:
    shard_sha = QUALITY.sha256_file(shard)
    configured_source: dict[str, object] = {
        "source_id": source_id,
        "repo_id": repo_id,
        "revision": revision,
        "role": "additive_candidate",
    }
    if mdc_dataset_id is not None:
        configured_source.update(
            {
                "acquisition_kind": "mozilla_data_collective",
                "mdc_dataset_id": mdc_dataset_id,
            }
        )
    if structured_profile_contract is not None:
        configured_source["structured_profile_contract"] = structured_profile_contract
    sources_config = tmp_path / "sources.json"
    sources_config.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_sources_v1",
                "base": {
                    "repo_id": "owner/base",
                    "revision": "b" * 40,
                    "role": "base",
                },
                "sources": [configured_source],
            }
        )
    )
    acquisition_source: dict[str, object] = {
        "source_id": source_id,
        "repo_id": repo_id,
        "revision": revision,
        "role": "additive_candidate",
        "selected_file_count": 1,
        "selected_bytes": shard.stat().st_size,
        "files": [
            {
                "path": shard.name,
                "size": shard.stat().st_size,
                "hash_kind": "sha256",
                "expected_hash": shard_sha,
            }
        ],
    }
    if mdc_dataset_id is not None:
        # This is the current realistic MDC receipt shape: the pinned dataset
        # ID predates the optional explicit acquisition_kind field.
        acquisition_source["mdc_dataset_id"] = mdc_dataset_id
    if acquisition_receipt_kind is not None:
        acquisition_source["acquisition_kind"] = acquisition_receipt_kind
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_acquisition_receipt_v1",
                "status": "passed",
                "sources_config_sha256": QUALITY.sha256_file(sources_config),
                "sources": [acquisition_source],
            }
        )
    )
    source_receipt = tmp_path / "normalization-source-receipt.json"
    source_receipt.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_normalization_source_receipt_v1",
                "source_id": source_id,
                "repo_id": repo_id,
                "revision": revision,
                "role": "additive_candidate",
                "counts": {"documents_emitted": rows},
                "shards": [
                    {
                        "path": str(shard.resolve()),
                        "bytes": shard.stat().st_size,
                        "sha256": shard_sha,
                        "rows": rows,
                    }
                ],
            }
        )
    )
    shard_receipt = tmp_path / "normalization-shard-receipt.json"
    shard_receipt.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_normalization_shard_receipt_v1",
                "source_id": source_id,
                "output": {
                    "path": str(shard.resolve()),
                    "bytes": shard.stat().st_size,
                    "sha256": shard_sha,
                    "rows": rows,
                },
            }
        )
    )
    normalization_contract = tmp_path / "normalization-contract.json"
    normalization_contract.write_text(json.dumps({"schema_version": "fixture"}))
    manifest = {
        "schema_version": "full_cpt_normalization_manifest_v1",
        "sources_config": str(sources_config.resolve()),
        "sources_config_sha256": QUALITY.sha256_file(sources_config),
        "acquisition_receipt": str(acquisition.resolve()),
        "acquisition_receipt_sha256": QUALITY.sha256_file(acquisition),
        "contract": {
            "path": str(normalization_contract.resolve()),
            "bytes": normalization_contract.stat().st_size,
            "sha256": QUALITY.sha256_file(normalization_contract),
        },
        "output": str(canonical_root.resolve()),
        "sources": [
            {
                "source_id": source_id,
                "repo_id": repo_id,
                "revision": revision,
                "role": "additive_candidate",
                "counts": {"documents_emitted": rows},
                "receipt": {
                    "path": str(source_receipt.resolve()),
                    "bytes": source_receipt.stat().st_size,
                    "sha256": QUALITY.sha256_file(source_receipt),
                },
                "shards": [
                    {
                        "path": str(shard.resolve()),
                        "bytes": shard.stat().st_size,
                        "sha256": shard_sha,
                        "rows": rows,
                        "receipt": {
                            "path": str(shard_receipt.resolve()),
                            "bytes": shard_receipt.stat().st_size,
                            "sha256": QUALITY.sha256_file(shard_receipt),
                        },
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "normalization.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_profile_route_fields_binds_declared_fallback_to_frozen_extraction_route() -> None:
    # The logical source is HTML even when this document's representation came
    # from the source's declared PDF extraction route.  The latter is a
    # secondary diagnostic, never a replacement for the primary error model.
    valid = {
        "source_route": "html_web",
        "review_route": "html_web",
        "extraction_route": "pdf_ocr",
        "observed_extraction_route": "pdf_ocr",
        "observed_extraction_route_basis": "declared_extraction_route_fallback",
        "observed_extraction_route_evidence": "roster:extraction_route",
        "observed_extraction_route_priority": "secondary_exception_only",
    }
    routes = QUALITY.profile_route_fields(valid, context="test.valid")
    assert routes["source_route"] == "html_web"
    assert routes["observed_extraction_route"] == "pdf_ocr"
    assert routes["observed_extraction_route_priority"] == "secondary_exception_only"

    false_fallback = {
        **valid,
        "observed_extraction_route": "html_web",
        "observed_extraction_route_priority": "logical_primary",
    }
    with pytest.raises(ValueError, match="declared extraction route fallback must equal"):
        QUALITY.profile_route_fields(false_fallback, context="test.false_fallback")

    unavailable_with_route = {
        **valid,
        "observed_extraction_route_basis": "unavailable",
        "observed_extraction_route_evidence": "none",
    }
    with pytest.raises(ValueError, match="unavailable observed extraction route cannot carry"):
        QUALITY.profile_route_fields(unavailable_with_route, context="test.unavailable")


def test_evaluations_cover_exact_29_repository_inventory() -> None:
    inventory = SITE.load_inventory(HERE / "configs" / "post_december_inventory.json")
    repos = {row["repo_id"] for row in inventory}
    evaluations = SITE.load_evaluations(
        HERE / "configs" / "dataset_review_evaluations.json", repos
    )
    assert len(inventory) == len(evaluations) == 29
    assert {row["inventory_group"] for row in inventory} == {
        "post_cutoff",
        "older_material_change",
    }
    assert evaluations["glossAPI/diavgeia"]["recommended_action"] == (
        "source_specific_cleaning"
    )
    assert evaluations["glossAPI/pandemos"]["recommended_action"] == "exclude_no_text"
    assert (
        SITE.payload_state(
            {
                "payload_status": "external_full_text_parquet_archive",
                "availability": "external_mozilla_registered_download_required",
            }
        )
        == "external_unavailable"
    )


def test_inventory_loader_accepts_receipt_bound_roster_changes(tmp_path: Path) -> None:
    inventory = tracked_inventory()
    inventory["post_cutoff_repositories"] = inventory["post_cutoff_repositories"][:-1]
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    loaded = SITE.load_inventory(path)
    assert len(loaded) == 28


def test_site_build_is_offline_complete_and_safe_for_hostile_sample(
    tmp_path: Path,
) -> None:
    uid = hashlib.sha256(b"hostile-sample").hexdigest()
    requests = tmp_path / "requests.jsonl"
    preview_sentinel = "PREVIEW_SENTINEL_MUST_NOT_LEAVE_REQUESTS"
    requests.write_text(
        json.dumps(review_request(uid, preview_sentinel=preview_sentinel)) + "\n",
        encoding="utf-8",
    )
    hostile = "</script><img src=x onerror=alert(1)> & harmless Greek κείμενο"
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(complete_sample(uid, hostile)) + "\n", encoding="utf-8"
    )
    sample_receipt = tmp_path / "samples-receipt.json"
    sample_attestation = write_packet_receipt(
        sample_receipt, packet=samples, requests=requests, rows=1
    )
    quality, quality_handoff = write_quality_summary_and_handoff(tmp_path)
    output = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "build_dataset_review_site.py"),
            "build",
            "--review-requests",
            str(requests),
            "--complete-samples",
            str(samples),
            "--complete-samples-receipt",
            str(sample_receipt),
            "--complete-samples-attestation",
            str(sample_attestation),
            "--quality-summary",
            str(quality),
            "--quality-handoff-receipt",
            str(quality_handoff),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    data = json.loads((output / "site_data.json").read_text())
    assert len(data["repositories"]) == 29
    assert data["overview"]["complete_samples"] == 1
    assert data["overview"]["quality_scope"] == {
        "documents": 2,
        "excluded_source_ids": ["nanochat_base"],
        "is_corpus_wide": False,
        "label": "Representative source-review sample",
        "scan_mode": "review_sample",
        "selected_source_ids": ["diavgeia", "kallipos_sections"],
    }
    assert data["overview"]["supplemental_profiled_repositories_outside_inventory"] == [
        "glossAPI/Apothetirio_Kallipos"
    ]
    diavgeia = next(
        row for row in data["repositories"] if row["repo_id"] == "glossAPI/diavgeia"
    )
    assert diavgeia["quality_scope"]["repository_documents"] == 1
    assert len(list((output / "datasets").glob("*.html"))) == 29
    sample_files = list((output / "samples").glob("*.json"))
    assert len(sample_files) == 1
    sample_path = sample_files[0]
    assert sample_path.stem != uid and len(sample_path.stem) == 32
    assert "<" not in sample_path.read_text(encoding="utf-8")
    parsed_sample = json.loads(sample_path.read_text())
    assert parsed_sample["schema_version"] == "dataset_review_site_sample_v1"
    assert parsed_sample["site_sample_id"] == sample_path.stem
    assert parsed_sample["text"] == hostile
    assert "ADA-1" not in sample_path.read_text()
    assert "ADA-1" not in (output / "site_data.json").read_text()
    assert 'id="scope-banner"' in (output / "index.html").read_text()
    assert hostile not in (output / "index.html").read_text()
    all_output = "\n".join(
        path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file()
    )
    assert uid not in all_output
    assert preview_sentinel not in all_output
    javascript = (output / "assets" / "site.js").read_text()
    assert "textContent=doc.text" in javascript
    assert "style:'percent'" in javascript
    assert "not corpus-wide" in javascript.lower()
    assert "innerHTML" not in javascript
    for page in [output / "index.html", *sorted((output / "datasets").glob("*.html"))]:
        source = page.read_text(encoding="utf-8")
        assert "Content-Security-Policy" in source
        assert "https://" not in source
        assert 'src="http' not in source
    manifest = json.loads((output / "site_manifest.json").read_text())
    assert manifest["repository_count"] == manifest["dataset_page_count"] == 29
    assert manifest["security"]["bind_address"] == "127.0.0.1"
    assert manifest["security"]["external_resources"] is False
    assert manifest["inputs"]["complete_samples_receipt"][
        "sha256"
    ] == QUALITY.sha256_file(sample_receipt)
    assert manifest["inputs"]["complete_samples_attestation"][
        "sha256"
    ] == QUALITY.sha256_file(sample_attestation)
    assert os.stat(sample_path).st_mode & 0o777 == 0o600
    assert SITE.validate_site_directory(output)["status"] == "passed"
    jsonschema = pytest.importorskip("jsonschema")
    manifest_schema = json.loads(
        (HERE / "schemas" / "dataset_review_site_manifest.schema.json").read_text()
    )
    sample_schema = json.loads(
        (HERE / "schemas" / "dataset_review_site_sample.schema.json").read_text()
    )
    packet_receipt_schema = json.loads(
        (
            HERE
            / "schemas"
            / "dataset_review_complete_sample_packet_receipt.schema.json"
        ).read_text()
    )
    sample_attestation_schema = json.loads(
        (
            HERE
            / "schemas"
            / "dataset_review_complete_sample_site_attestation.schema.json"
        ).read_text()
    )
    quality_schema = json.loads(
        (HERE / "schemas" / "dataset_quality_summary.schema.json").read_text()
    )
    quality_handoff_schema = json.loads(
        (HERE / "schemas" / "dataset_quality_site_handoff.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(manifest_schema).validate(manifest)
    jsonschema.Draft202012Validator(sample_schema).validate(parsed_sample)
    jsonschema.Draft202012Validator(packet_receipt_schema).validate(
        json.loads(sample_receipt.read_text())
    )
    jsonschema.Draft202012Validator(sample_attestation_schema).validate(
        json.loads(sample_attestation.read_text())
    )
    jsonschema.Draft202012Validator(quality_schema).validate(
        json.loads(quality.read_text())
    )
    jsonschema.Draft202012Validator(quality_handoff_schema).validate(
        json.loads(quality_handoff.read_text())
    )


def test_site_rejects_unredacted_complete_sample(tmp_path: Path) -> None:
    uid = "a" * 64
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(review_request(uid, doc_id="1")) + "\n")
    samples = tmp_path / "samples.jsonl"
    row = complete_sample(uid, "secret", doc_id="1")
    row["high_precision_identifier_patterns_masked"] = False
    samples.write_text(json.dumps(row) + "\n")
    sample_receipt = tmp_path / "samples-receipt.json"
    sample_attestation = write_packet_receipt(
        sample_receipt, packet=samples, requests=requests, rows=1
    )
    with pytest.raises(ValueError, match="masking/text attestation"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=HERE / "configs" / "sources.json",
                quality_summary=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=samples,
                complete_samples_receipt=sample_receipt,
                complete_samples_attestation=sample_attestation,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )


def test_site_filters_supplemental_complete_samples_and_emits_no_hidden_text(
    tmp_path: Path,
) -> None:
    visible_uid = hashlib.sha256(b"visible").hexdigest()
    hidden_uid = hashlib.sha256(b"supplemental").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        "\n".join(
            [
                json.dumps(review_request(visible_uid)),
                json.dumps(
                    review_request(
                        hidden_uid,
                        source_id="kallipos_sections",
                        repo_id="glossAPI/Apothetirio_Kallipos",
                        dataset="kallipos",
                        doc_id="hidden-doc",
                    )
                ),
            ]
        )
        + "\n"
    )
    hidden_text = "HIDDEN_SUPPLEMENTAL_TEXT_MUST_NOT_ENTER_SITE"
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        "\n".join(
            [
                json.dumps(complete_sample(visible_uid, "ορατό κείμενο")),
                json.dumps(
                    complete_sample(
                        hidden_uid,
                        hidden_text,
                        source_id="kallipos_sections",
                        repo_id="glossAPI/Apothetirio_Kallipos",
                        dataset="kallipos",
                        doc_id="hidden-doc",
                    )
                ),
            ]
        )
        + "\n"
    )
    sample_receipt = tmp_path / "samples-receipt.json"
    sample_attestation = write_packet_receipt(
        sample_receipt, packet=samples, requests=requests, rows=2
    )
    output = tmp_path / "site"
    SITE.build_site(
        SimpleNamespace(
            inventory=HERE / "configs" / "post_december_inventory.json",
            evaluations=HERE / "configs" / "dataset_review_evaluations.json",
            sources_config=HERE / "configs" / "sources.json",
            quality_summary=None,
            review_requests=requests,
            review_responses=None,
            admission=None,
            novelty=None,
            complete_samples=samples,
            complete_samples_receipt=sample_receipt,
            complete_samples_attestation=sample_attestation,
            output_dir=output,
            replace=False,
        )
    )
    data = json.loads((output / "site_data.json").read_text())
    assert data["overview"]["complete_samples"] == 1
    assert data["overview"]["complete_samples_excluded_outside_inventory"] == 1
    assert len(list((output / "samples").glob("*.json"))) == 1
    emitted = "\n".join(
        path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file()
    )
    assert hidden_uid not in emitted
    assert hidden_text not in emitted


def test_site_rejects_sample_receipt_and_profile_hash_drift(tmp_path: Path) -> None:
    uid = hashlib.sha256(b"receipt-drift").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(review_request(uid)) + "\n")
    samples = tmp_path / "samples.jsonl"
    row = complete_sample(uid, "κείμενο")
    row["profile_text_sha256"] = "f" * 64
    samples.write_text(json.dumps(row) + "\n")
    receipt_path = tmp_path / "samples-receipt.json"
    sample_attestation = write_packet_receipt(
        receipt_path, packet=samples, requests=requests, rows=1
    )
    with pytest.raises(ValueError, match="masking/text attestation"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=HERE / "configs" / "sources.json",
                quality_summary=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=samples,
                complete_samples_receipt=receipt_path,
                complete_samples_attestation=sample_attestation,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )
    receipt = json.loads(receipt_path.read_text())
    receipt["review_requests"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="request receipt drift"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=HERE / "configs" / "sources.json",
                quality_summary=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=samples,
                complete_samples_receipt=receipt_path,
                complete_samples_attestation=sample_attestation,
                output_dir=tmp_path / "site-two",
                replace=False,
            )
        )


def test_site_manifest_validation_rejects_tamper_extra_and_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "site"
    args = SimpleNamespace(
        inventory=HERE / "configs" / "post_december_inventory.json",
        evaluations=HERE / "configs" / "dataset_review_evaluations.json",
        sources_config=HERE / "configs" / "sources.json",
        quality_summary=None,
        review_requests=None,
        review_responses=None,
        admission=None,
        novelty=None,
        complete_samples=None,
        complete_samples_receipt=None,
        complete_samples_attestation=None,
        output_dir=output,
        replace=False,
    )
    SITE.build_site(args)
    site_data = output / "site_data.json"
    original = site_data.read_bytes()
    site_data.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="receipt drift"):
        SITE.validate_site_directory(output)
    site_data.write_bytes(original)
    extra = output / "unexpected.txt"
    extra.write_text("unexpected")
    with pytest.raises(ValueError, match="inventory drift"):
        SITE.validate_site_directory(output)
    extra.unlink()
    symlink = output / "linked"
    symlink.symlink_to(output / "index.html")
    with pytest.raises(ValueError, match="symlinks"):
        SITE.validate_site_directory(output)


def test_site_inputs_refuse_symlinked_parent_components(tmp_path: Path) -> None:
    real_inputs = tmp_path / "real-inputs"
    real_inputs.mkdir()
    for name, source in (
        ("inventory.json", HERE / "configs" / "post_december_inventory.json"),
        ("evaluations.json", HERE / "configs" / "dataset_review_evaluations.json"),
        ("sources.json", SOURCES_CONFIG),
    ):
        shutil.copy2(source, real_inputs / name)
    linked_inputs = tmp_path / "linked-inputs"
    linked_inputs.symlink_to(real_inputs, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink/non-directory"):
        SITE.build_site(
            SimpleNamespace(
                inventory=linked_inputs / "inventory.json",
                evaluations=linked_inputs / "evaluations.json",
                sources_config=linked_inputs / "sources.json",
                quality_summary=None,
                quality_handoff_receipt=None,
                review_requests=None,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=None,
                complete_samples_receipt=None,
                complete_samples_attestation=None,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )


def test_site_parses_private_stable_copies_during_parent_swap_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    attacker = tmp_path / "attacker"
    held = tmp_path / "inputs-held"
    inputs.mkdir()
    attacker.mkdir()
    source_files = {
        "inventory.json": HERE / "configs" / "post_december_inventory.json",
        "evaluations.json": HERE / "configs" / "dataset_review_evaluations.json",
        "sources.json": SOURCES_CONFIG,
    }
    for name, source in source_files.items():
        shutil.copy2(source, inputs / name)
        shutil.copy2(source, attacker / name)
    sentinel = "ATTACKER_PARENT_SWAP_SENTINEL"
    attacker_evaluations = json.loads(
        (attacker / "evaluations.json").read_text(encoding="utf-8")
    )
    attacker_evaluations["entries"][0]["assessment"] = sentinel
    (attacker / "evaluations.json").write_text(json.dumps(attacker_evaluations))

    original_load_evaluations = SITE.load_evaluations

    def load_while_parent_is_swapped(path: Path, repos: set[str]):
        inputs.rename(held)
        inputs.symlink_to(attacker, target_is_directory=True)
        try:
            return original_load_evaluations(path, repos)
        finally:
            inputs.unlink()
            held.rename(inputs)

    monkeypatch.setattr(SITE, "load_evaluations", load_while_parent_is_swapped)
    output = tmp_path / "site"
    SITE.build_site(
        SimpleNamespace(
            inventory=inputs / "inventory.json",
            evaluations=inputs / "evaluations.json",
            sources_config=inputs / "sources.json",
            quality_summary=None,
            quality_handoff_receipt=None,
            review_requests=None,
            review_responses=None,
            admission=None,
            novelty=None,
            complete_samples=None,
            complete_samples_receipt=None,
            complete_samples_attestation=None,
            output_dir=output,
            replace=False,
        )
    )
    assert sentinel not in (output / "site_data.json").read_text(encoding="utf-8")
    snapshot = SITE.snapshot_site_input(inputs / "inventory.json")
    assert not hasattr(snapshot, "content")


def test_full_quality_scope_is_selected_population_not_corpus_wide(
    tmp_path: Path,
) -> None:
    sources_config = write_sources_config(
        tmp_path / "sources.json",
        [
            {
                "source_id": "alpha",
                "repo_id": "owner/alpha",
                "revision": "a" * 40,
                "role": "additive_candidate",
            },
            {
                "source_id": "beta",
                "repo_id": "owner/beta",
                "revision": "b" * 40,
                "role": "additive_candidate",
            },
        ],
    )
    path, handoff = write_quality_summary_and_handoff(
        tmp_path,
        repositories=[
            ("alpha", "owner/alpha", "a" * 40),
            ("beta", "owner/beta", "b" * 40),
        ],
        scan_mode="full_scan",
        sources_config=sources_config,
    )
    _, scope, _ = SITE.load_quality(path, handoff, sources_config_path=sources_config)
    assert scope == {
        "scan_mode": "full_scan",
        "documents": 2,
        "is_corpus_wide": False,
        "label": "Full scan of selected canonical sources",
        "selected_source_ids": ["alpha", "beta"],
        "excluded_source_ids": ["nanochat_base"],
    }
    value = json.loads(path.read_text())
    value["excluded_source_ids"] = ["alpha"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="overlap"):
        SITE.load_quality(path, handoff, sources_config_path=sources_config)


def test_quality_summary_rejects_checkpoint_path_traversal_with_rehashed_inventory(
    tmp_path: Path,
) -> None:
    summary_path, _ = write_quality_summary_and_handoff(tmp_path, scan_mode="full_scan")
    summary = json.loads(summary_path.read_text())
    inventory = summary["batch_checkpoints"]["inventory"]
    inventory[0]["receipt_path"] = "../outside/receipt.json"
    summary["batch_checkpoints"]["inventory_sha256"] = QUALITY.sha256_json(inventory)
    with pytest.raises(ValueError, match="canonical relative path"):
        QUALITY.validate_and_project_quality_summary(summary)

    for receipt_name, bad_path in (
        ("contract", "../outside/contract.json"),
        ("document_output", "/outside/documents.parquet"),
    ):
        summary = json.loads(summary_path.read_text())
        summary[receipt_name]["path"] = bad_path
        with pytest.raises(ValueError, match="canonical relative path"):
            QUALITY.validate_and_project_quality_summary(summary)


def test_quality_v2_boundary_rejects_legacy_v1_artifacts(tmp_path: Path) -> None:
    summary_path, handoff_path = write_quality_summary_and_handoff(
        tmp_path, scan_mode="full_scan"
    )
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "dataset_quality_summary_v1"
    with pytest.raises(ValueError, match="unsupported schema/status/mode"):
        QUALITY.validate_and_project_quality_summary(summary)

    summary = json.loads(summary_path.read_text())
    summary["document_output"]["path"] = "dataset_quality_document_v1.parquet"
    with pytest.raises(ValueError, match="unexpected output"):
        QUALITY.validate_and_project_quality_summary(summary)

    handoff = json.loads(handoff_path.read_text())
    handoff["schema_version"] = "dataset_quality_site_handoff_v1"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    with pytest.raises(ValueError, match="schema/status/projection drift"):
        QUALITY.validate_quality_site_handoff(
            summary_path=summary_path, handoff_path=handoff_path
        )


def test_quality_summary_rejects_duplicate_checkpoint_paths_and_batch_identities(
    tmp_path: Path,
) -> None:
    summary_path, _ = write_quality_summary_and_handoff(tmp_path, scan_mode="full_scan")
    summary = json.loads(summary_path.read_text())
    inventory = summary["batch_checkpoints"]["inventory"]
    inventory[1]["receipt_path"] = inventory[0]["receipt_path"]
    summary["batch_checkpoints"]["inventory_sha256"] = QUALITY.sha256_json(inventory)
    with pytest.raises(ValueError, match="duplicate checkpoint receipt path"):
        QUALITY.validate_and_project_quality_summary(summary)

    identity_root = tmp_path / "identity"
    identity_root.mkdir()
    summary_path, _ = write_quality_summary_and_handoff(
        identity_root, scan_mode="full_scan"
    )
    summary = json.loads(summary_path.read_text())
    inventory = summary["batch_checkpoints"]["inventory"]
    for key in (
        "input_shard_source_id",
        "input_shard_path",
        "input_shard_sha256",
        "batch_index",
    ):
        inventory[1][key] = inventory[0][key]
    summary["batch_checkpoints"]["inventory_sha256"] = QUALITY.sha256_json(inventory)
    with pytest.raises(ValueError, match="duplicate physical-shard batch identity"):
        QUALITY.validate_and_project_quality_summary(summary)


def test_quality_summary_rejects_duplicate_physical_shard_paths(
    tmp_path: Path,
) -> None:
    summary_path, _ = write_quality_summary_and_handoff(tmp_path, scan_mode="full_scan")
    summary = json.loads(summary_path.read_text())
    first_path = summary["input_shards"][0]["path"]
    second_source = summary["input_shards"][1]["source_id"]
    summary["input_shards"][1]["path"] = first_path
    inventory = summary["batch_checkpoints"]["inventory"]
    second_checkpoint = next(
        row for row in inventory if row["input_shard_source_id"] == second_source
    )
    second_checkpoint["input_shard_path"] = first_path
    summary["batch_checkpoints"]["inventory_sha256"] = QUALITY.sha256_json(inventory)
    with pytest.raises(ValueError, match="duplicate physical shard path"):
        QUALITY.validate_and_project_quality_summary(summary)


def test_quality_summary_rejects_checkpoint_receipt_filename_aliases(
    tmp_path: Path,
) -> None:
    summary_path, _ = write_quality_summary_and_handoff(tmp_path, scan_mode="full_scan")
    summary = json.loads(summary_path.read_text())
    inventory = summary["batch_checkpoints"]["inventory"]
    inventory[0]["receipt_path"] = "batches/shared/a.json"
    inventory[1]["receipt_path"] = "batches/shared/b.json"
    summary["batch_checkpoints"]["inventory_sha256"] = QUALITY.sha256_json(inventory)
    with pytest.raises(ValueError, match="expected receipt.json basename"):
        QUALITY.validate_and_project_quality_summary(summary)


def test_review_sample_checkpoint_is_bound_to_exact_packet_identity(
    tmp_path: Path,
) -> None:
    summary_path, handoff_path = write_quality_summary_and_handoff(tmp_path)
    summary = json.loads(summary_path.read_text())
    inventory = summary["batch_checkpoints"]["inventory"]
    inventory[0].update(
        {
            "input_shard_source_id": "forged-review-source",
            "input_shard_path": "review-sample/forged.jsonl",
            "input_shard_sha256": "f" * 64,
        }
    )
    summary["batch_checkpoints"]["inventory_sha256"] = QUALITY.sha256_json(inventory)
    with pytest.raises(ValueError, match="review-sample checkpoint identity drift"):
        QUALITY.validate_and_project_quality_summary(summary)

    summary = json.loads(summary_path.read_text())
    inventory = summary["batch_checkpoints"]["inventory"]
    inventory[0]["input_shard_sha256"] = "f" * 64
    summary["batch_checkpoints"]["inventory_sha256"] = QUALITY.sha256_json(inventory)
    summary_path.write_text(json.dumps(summary))
    projection = QUALITY.validate_and_project_quality_summary(summary)
    handoff = json.loads(handoff_path.read_text())
    handoff["aggregate_projection_sha256"] = QUALITY.sha256_json(projection)
    handoff["summary"].update(
        {
            "bytes": summary_path.stat().st_size,
            "sha256": QUALITY.sha256_file(summary_path),
        }
    )
    handoff["checkpoint_closure"]["inventory_sha256"] = summary["batch_checkpoints"][
        "inventory_sha256"
    ]
    handoff_path.write_text(json.dumps(handoff))
    with pytest.raises(ValueError, match="review-sample checkpoint binding drift"):
        QUALITY.validate_quality_site_handoff(
            summary_path=summary_path, handoff_path=handoff_path
        )


def test_full_scan_rejects_partial_selected_shard_coverage(tmp_path: Path) -> None:
    sources_config = write_sources_config(
        tmp_path / "sources.json",
        [
            {
                "source_id": "alpha",
                "repo_id": "owner/alpha",
                "revision": "a" * 40,
                "role": "additive_candidate",
            }
        ],
    )
    summary_path, _ = write_quality_summary_and_handoff(
        tmp_path,
        repositories=[("alpha", "owner/alpha", "a" * 40)],
        scan_mode="full_scan",
        sources_config=sources_config,
    )
    summary = json.loads(summary_path.read_text())
    summary["input_shards"][0]["rows"] = 100
    with pytest.raises(ValueError, match="incomplete full-scan row coverage"):
        QUALITY.validate_and_project_quality_summary(summary)


@pytest.mark.parametrize("second_start", [0, 2])
def test_full_scan_rejects_overlapping_or_gapped_checkpoint_intervals(
    tmp_path: Path, second_start: int
) -> None:
    sources_config = write_sources_config(
        tmp_path / "sources.json",
        [
            {
                "source_id": "alpha",
                "repo_id": "owner/alpha",
                "revision": "a" * 40,
                "role": "additive_candidate",
            }
        ],
    )
    summary_path, _ = write_quality_summary_and_handoff(
        tmp_path,
        repositories=[("alpha", "owner/alpha", "a" * 40)],
        scan_mode="full_scan",
        sources_config=sources_config,
    )
    summary = json.loads(summary_path.read_text())
    summary["input_shards"][0].update({"rows": 3, "batches": 2})
    inventory = summary["batch_checkpoints"]["inventory"]
    second = {
        **inventory[0],
        "receipt_path": "batches/alpha/second/receipt.json",
        "receipt_sha256": "1" * 64,
        "output_sha256": "2" * 64,
        "batch_index": 1,
        "row_start": second_start,
        "row_end_exclusive": second_start + 1,
    }
    inventory.append(second)
    summary["batch_checkpoints"].update(
        {
            "count": 2,
            "rows": 2,
            "inventory_sha256": QUALITY.sha256_json(inventory),
        }
    )
    with pytest.raises(ValueError, match="checkpoint coverage is noncontiguous"):
        QUALITY.validate_and_project_quality_summary(summary)


def test_quality_summary_rejects_unconstrained_preview_or_raw_identifier_keys(
    tmp_path: Path,
) -> None:
    summary_path, handoff_path = write_quality_summary_and_handoff(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["repositories"][0]["document_preview"] = "RAW-DOCUMENT-ID-123"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="unexpected=.*document_preview"):
        SITE.load_quality(
            summary_path, handoff_path, sources_config_path=SOURCES_CONFIG
        )
    second = tmp_path / "handoff-case"
    second.mkdir()
    summary_path, handoff_path = write_quality_summary_and_handoff(second)
    handoff = json.loads(handoff_path.read_text())
    handoff["normalization"]["source_identities"][0]["raw_document_id"] = "secret"
    handoff_path.write_text(json.dumps(handoff))
    with pytest.raises(ValueError, match="unexpected=.*raw_document_id"):
        SITE.load_quality(
            summary_path, handoff_path, sources_config_path=SOURCES_CONFIG
        )


def test_json_numbers_are_finite_and_serialization_cannot_emit_nan() -> None:
    for loader in (SITE.strict_json_loads, QUALITY.strict_json_loads):
        with pytest.raises(ValueError, match="non-finite JSON number"):
            loader('{"overflow":1e999}', context="fixture")
    with pytest.raises(ValueError, match="Out of range float values"):
        SITE.safe_json({"invalid": float("inf")})
    with pytest.raises(ValueError, match="Out of range float values"):
        EXPORTER.canonical_json({"invalid": float("nan")})


def test_quality_fraction_distributions_reject_out_of_range_values(
    tmp_path: Path,
) -> None:
    summary_path, _ = write_quality_summary_and_handoff(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for statistics in [summary["global"], *summary["repositories"]]:
        statistics["distributions"]["raw_greek_letter_fraction"] = quality_distribution(
            int(statistics["documents"]), 1.01
        )
    with pytest.raises(ValueError, match=r"raw_greek_letter_fraction.*\[0, 1\]"):
        QUALITY.validate_and_project_quality_summary(summary)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("novel_token_fraction", 1.01, "finite fraction in"),
        ("exact_unique_word_tokens", 11, "denominator drift"),
        ("rows", True, "signed-64-bit integer"),
    ],
)
def test_novelty_values_obey_ranges_and_denominators(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    novelty = {
        "schema_version": "full_cpt_source_novelty_v1",
        "sources": [
            {
                "source_dataset": "candidate",
                "rows": 1,
                "identity_word_tokens": 10,
                "exact_unique_word_tokens": 9,
                "novel_word_tokens_after_lineage_resolution": 8,
                "novel_token_fraction": 0.8,
            }
        ],
    }
    novelty["sources"][0][field] = value
    path = tmp_path / "novelty.json"
    path.write_text(json.dumps(novelty), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        SITE.load_novelty(path, {"candidate": "owner/candidate"})


def test_novelty_exponent_overflow_is_rejected_during_parse(tmp_path: Path) -> None:
    path = tmp_path / "novelty.json"
    path.write_text(
        '{"schema_version":"full_cpt_source_novelty_v1","sources":['
        '{"source_dataset":"candidate","rows":1,"identity_word_tokens":1,'
        '"exact_unique_word_tokens":1,"novel_word_tokens_after_lineage_resolution":1,'
        '"novel_token_fraction":1e999}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        SITE.load_novelty(path, {"candidate": "owner/candidate"})


def test_external_repository_needs_exact_acquired_revision_evidence(
    tmp_path: Path,
) -> None:
    sources_config = write_sources_config(
        tmp_path / "sources.json",
        [
            {
                "source_id": "istorima",
                "repo_id": "glossAPI/istorima",
                "revision": "0" * 40,
                "role": "additive_candidate",
            }
        ],
    )
    summary_path, handoff_path = write_quality_summary_and_handoff(
        tmp_path,
        repositories=[("istorima", "glossAPI/istorima", "0" * 40)],
        scan_mode="full_scan",
        sources_config=sources_config,
    )
    output = tmp_path / "site"
    SITE.build_site(
        SimpleNamespace(
            inventory=HERE / "configs" / "post_december_inventory.json",
            evaluations=HERE / "configs" / "dataset_review_evaluations.json",
            sources_config=sources_config,
            quality_summary=summary_path,
            quality_handoff_receipt=handoff_path,
            review_requests=None,
            review_responses=None,
            admission=None,
            novelty=None,
            complete_samples=None,
            complete_samples_receipt=None,
            complete_samples_attestation=None,
            output_dir=output,
            replace=False,
        )
    )
    data = json.loads((output / "site_data.json").read_text())
    istorima = next(
        row for row in data["repositories"] if row["repo_id"] == "glossAPI/istorima"
    )
    assert istorima["quality"] is not None
    assert istorima["payload_state"] == "external_unavailable"


def test_external_repository_promotes_only_with_exact_mdc_identity(
    tmp_path: Path,
) -> None:
    revision = "0f3f32db50235b0e42e130983e6a06c709835e16"
    sources_config = write_sources_config(
        tmp_path / "sources.json",
        [
            {
                "source_id": "istorima",
                "repo_id": "glossAPI/istorima",
                "revision": revision,
                "role": "additive_candidate",
                "acquisition_kind": "mozilla_data_collective",
                "mdc_dataset_id": "mdc-istorima-pinned-id",
            }
        ],
    )
    summary_path, handoff_path = write_quality_summary_and_handoff(
        tmp_path,
        repositories=[("istorima", "glossAPI/istorima", revision)],
        scan_mode="full_scan",
        sources_config=sources_config,
    )
    output = tmp_path / "site"
    SITE.build_site(
        SimpleNamespace(
            inventory=HERE / "configs" / "post_december_inventory.json",
            evaluations=HERE / "configs" / "dataset_review_evaluations.json",
            sources_config=sources_config,
            quality_summary=summary_path,
            quality_handoff_receipt=handoff_path,
            review_requests=None,
            review_responses=None,
            admission=None,
            novelty=None,
            complete_samples=None,
            complete_samples_receipt=None,
            complete_samples_attestation=None,
            output_dir=output,
            replace=False,
        )
    )
    data = json.loads((output / "site_data.json").read_text(encoding="utf-8"))
    istorima = next(
        row for row in data["repositories"] if row["repo_id"] == "glossAPI/istorima"
    )
    assert istorima["payload_state"] == "external_acquired"


def test_quality_handoff_is_bound_to_exact_local_sources_config(
    tmp_path: Path,
) -> None:
    summary_path, handoff_path = write_quality_summary_and_handoff(tmp_path)
    alternate = tmp_path / "sources-copy.json"
    alternate.write_text(SOURCES_CONFIG.read_text(encoding="utf-8") + "\n")
    with pytest.raises(ValueError, match="local sources config identity drift"):
        SITE.load_quality(summary_path, handoff_path, sources_config_path=alternate)


def test_review_request_cannot_relabel_supplemental_source_as_visible(
    tmp_path: Path,
) -> None:
    source = source_identities()["kallipos_sections"]
    request = review_request(
        hashlib.sha256(b"relabelled-request").hexdigest(),
        source_id="kallipos_sections",
        repo_id="glossAPI/diavgeia",
        dataset="kallipos",
        revision=str(source["revision"]),
    )
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from sources config"):
        SITE.load_requests(path, source_identities(), b"site-key")


def test_sample_attestation_identity_cannot_relabel_a_tracked_source(
    tmp_path: Path,
) -> None:
    uid = hashlib.sha256(b"attested-source-relabel").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(review_request(uid)) + "\n")
    packet = tmp_path / "samples.jsonl"
    packet.write_text(json.dumps(complete_sample(uid, "κείμενο")) + "\n")
    receipt_path = tmp_path / "samples-receipt.json"
    attestation_path = write_packet_receipt(
        receipt_path, packet=packet, requests=requests, rows=1
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["normalization"]["source_identities"][0]["repo_id"] = (
        "glossAPI/Apothetirio_Kallipos"
    )
    attestation["normalization"]["source_identity_inventory_sha256"] = (
        QUALITY.sha256_json(attestation["normalization"]["source_identities"])
    )
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    refresh_packet_attestation_receipts(
        packet=packet,
        receipt_path=receipt_path,
        attestation_path=attestation_path,
    )
    with pytest.raises(ValueError, match="source identity is incomplete"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=SOURCES_CONFIG,
                quality_summary=None,
                quality_handoff_receipt=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=packet,
                complete_samples_receipt=receipt_path,
                complete_samples_attestation=attestation_path,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )


def test_complete_sample_row_cannot_relabel_a_tracked_source(
    tmp_path: Path,
) -> None:
    uid = hashlib.sha256(b"sample-row-relabel").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(review_request(uid)) + "\n")
    packet = tmp_path / "samples.jsonl"
    row = complete_sample(uid, "κείμενο")
    packet.write_text(json.dumps(row) + "\n")
    receipt_path = tmp_path / "samples-receipt.json"
    attestation_path = write_packet_receipt(
        receipt_path, packet=packet, requests=requests, rows=1
    )
    row["source_repo_id"] = "glossAPI/Apothetirio_Kallipos"
    packet.write_text(json.dumps(row) + "\n")
    refresh_packet_attestation_receipts(
        packet=packet,
        receipt_path=receipt_path,
        attestation_path=attestation_path,
    )
    with pytest.raises(ValueError, match="complete sample identity drift"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=SOURCES_CONFIG,
                quality_summary=None,
                quality_handoff_receipt=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=packet,
                complete_samples_receipt=receipt_path,
                complete_samples_attestation=attestation_path,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )


def test_site_rejects_forged_checkpoint_attestation_even_with_updated_self_hash(
    tmp_path: Path,
) -> None:
    uid = hashlib.sha256(b"forged-attestation").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(review_request(uid)) + "\n")
    packet = tmp_path / "samples.jsonl"
    packet.write_text(json.dumps(complete_sample(uid, "ασφαλές κείμενο")) + "\n")
    receipt_path = tmp_path / "samples-receipt.json"
    attestation_path = write_packet_receipt(
        receipt_path, packet=packet, requests=requests, rows=1
    )
    attestation = json.loads(attestation_path.read_text())
    attestation["checkpoint_closure"][
        "checkpoint_text_outputs_rehashed_for_attestation"
    ] = False
    attestation_path.write_text(json.dumps(attestation))
    receipt = json.loads(receipt_path.read_text())
    receipt["site_attestation"].update(
        {
            "bytes": attestation_path.stat().st_size,
            "sha256": QUALITY.sha256_file(attestation_path),
        }
    )
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="checkpoint closure drift"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=HERE / "configs" / "sources.json",
                quality_summary=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=packet,
                complete_samples_receipt=receipt_path,
                complete_samples_attestation=attestation_path,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )


def test_site_rejects_complete_samples_without_cluster_attestation(
    tmp_path: Path,
) -> None:
    uid = hashlib.sha256(b"missing-attestation").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(review_request(uid)) + "\n")
    packet = tmp_path / "samples.jsonl"
    packet.write_text(json.dumps(complete_sample(uid, "κείμενο")) + "\n")
    receipt_path = tmp_path / "samples-receipt.json"
    write_packet_receipt(receipt_path, packet=packet, requests=requests, rows=1)
    with pytest.raises(ValueError, match="requires its receipt, site attestation"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=HERE / "configs" / "sources.json",
                quality_summary=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=packet,
                complete_samples_receipt=receipt_path,
                complete_samples_attestation=None,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )


def test_site_remasks_immediately_before_emission_and_rejects_residual_url(
    tmp_path: Path,
) -> None:
    uid = hashlib.sha256(b"residual-url").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(review_request(uid)) + "\n")
    packet = tmp_path / "samples.jsonl"
    residual = "κείμενο https://example.org/view?token=private#person-42"
    packet.write_text(json.dumps(complete_sample(uid, residual)) + "\n")
    receipt_path = tmp_path / "samples-receipt.json"
    attestation_path = write_packet_receipt(
        receipt_path, packet=packet, requests=requests, rows=1
    )
    with pytest.raises(ValueError, match="residual known identifier or URL"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=HERE / "configs" / "sources.json",
                quality_summary=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=packet,
                complete_samples_receipt=receipt_path,
                complete_samples_attestation=attestation_path,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )


def test_site_detects_input_drift_after_parse_before_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = tmp_path / "inventory.json"
    evaluations = tmp_path / "evaluations.json"
    sources = tmp_path / "sources.json"
    inventory.write_bytes(
        (HERE / "configs" / "post_december_inventory.json").read_bytes()
    )
    evaluations.write_bytes(
        (HERE / "configs" / "dataset_review_evaluations.json").read_bytes()
    )
    sources.write_bytes((HERE / "configs" / "sources.json").read_bytes())
    original_validate = SITE.validate_site_directory

    def validate_then_drift(root: Path) -> dict[str, object]:
        result = original_validate(root)
        inventory.write_bytes(inventory.read_bytes() + b" ")
        return result

    monkeypatch.setattr(SITE, "validate_site_directory", validate_then_drift)
    output = tmp_path / "site"
    with pytest.raises(ValueError, match="input drift before atomic publication"):
        SITE.build_site(
            SimpleNamespace(
                inventory=inventory,
                evaluations=evaluations,
                sources_config=sources,
                quality_summary=None,
                review_requests=None,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=None,
                complete_samples_receipt=None,
                complete_samples_attestation=None,
                output_dir=output,
                replace=False,
            )
        )
    assert not output.exists()


def test_normalized_shard_loader_is_manifest_exact(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    shard = root / "candidate" / "part.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"PAR1-fixture")
    manifest = {
        "schema_version": "full_cpt_normalization_manifest_v1",
        "output": str(root.resolve()),
        "sources": [
            {
                "source_id": "candidate",
                "shards": [
                    {
                        "path": str(shard.resolve()),
                        "bytes": shard.stat().st_size,
                        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                        "rows": 1,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    _, bindings, excluded = QUALITY.load_normalized_shards(
        path, root, include_source_ids=set(), include_base=False
    )
    assert len(bindings) == 1
    assert excluded == []
    rogue = root / "rogue.parquet"
    rogue.write_bytes(b"PAR1-rogue")
    with pytest.raises(ValueError, match="inventory differs"):
        QUALITY.load_normalized_shards(
            path, root, include_source_ids=set(), include_base=False
        )


def test_normalization_identity_closure_rejects_acquisition_file_total_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "canonical"
    shard = root / "candidate" / "part.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"fixture")
    manifest_path = write_normalization_fixture(
        tmp_path,
        canonical_root=root,
        shard=shard,
        source_id="candidate",
        repo_id="owner/candidate",
        revision="a" * 40,
        rows=1,
    )
    manifest = json.loads(manifest_path.read_text())
    acquisition_path = Path(manifest["acquisition_receipt"])
    acquisition = json.loads(acquisition_path.read_text())
    acquisition["sources"][0]["selected_file_count"] = 2
    acquisition_path.write_text(json.dumps(acquisition))
    manifest["acquisition_receipt_sha256"] = QUALITY.sha256_file(acquisition_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="acquisition file totals drift"):
        QUALITY.normalization_identity_closure(manifest_path)


def test_normalization_identity_closure_accepts_realistic_mdc_receipt_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "canonical"
    shard = root / "istorima" / "part.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"fixture")
    manifest_path = write_normalization_fixture(
        tmp_path,
        canonical_root=root,
        shard=shard,
        source_id="istorima",
        repo_id="glossAPI/istorima",
        revision="0" * 40,
        rows=1,
        mdc_dataset_id="mdc-istorima-pinned-id",
    )
    closure = QUALITY.normalization_identity_closure(manifest_path)
    identity = closure["source_identities"][0]
    sources_path = Path(json.loads(manifest_path.read_text())["sources_config"])
    configured = source_identities(sources_path)["istorima"]
    assert identity["acquisition_kind"] == "mozilla_data_collective"
    assert identity["mdc_dataset_id"] == "mdc-istorima-pinned-id"
    assert identity["source_config_sha256"] == configured["source_config_sha256"]


def test_normalization_identity_closure_rejects_explicit_mdc_kind_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "canonical"
    shard = root / "istorima" / "part.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"fixture")
    manifest_path = write_normalization_fixture(
        tmp_path,
        canonical_root=root,
        shard=shard,
        source_id="istorima",
        repo_id="glossAPI/istorima",
        revision="0" * 40,
        rows=1,
        mdc_dataset_id="mdc-istorima-pinned-id",
        acquisition_receipt_kind="hugging_face",
    )
    with pytest.raises(ValueError, match="provenance identity drift"):
        QUALITY.normalization_identity_closure(manifest_path)


def test_quantile_sample_size_is_bound_into_resume_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "normalization.json"
    build = tmp_path / "build.json"
    shard_path = tmp_path / "part.parquet"
    manifest.write_text("{}")
    build.write_text("{}")
    shard_path.write_bytes(b"fixture")
    shard = QUALITY.ShardBinding(
        source_id="candidate",
        path=shard_path,
        relative_path="candidate/part.parquet",
        bytes=shard_path.stat().st_size,
        sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
        rows=1,
    )
    base = dict(
        scan_mode="full_scan",
        normalization_manifest=manifest,
        canonical_root=tmp_path,
        build_receipt=build,
        expected_commit=QUALITY.PINNED_GLOSSAPI_COMMIT,
        batch_size=4096,
        threads=256,
    )
    first = QUALITY.diagnostics_contract(
        SimpleNamespace(**base, quantile_sample_size=1024),
        shards=[shard],
        excluded=["nanochat_base"],
        sample_input_shards=None,
        sample_contract=None,
    )
    second = QUALITY.diagnostics_contract(
        SimpleNamespace(**base, quantile_sample_size=2048),
        shards=[shard],
        excluded=["nanochat_base"],
        sample_input_shards=None,
        sample_contract=None,
    )
    assert first["quantile_sample_size"] == 1024
    assert QUALITY.sha256_json(first) != QUALITY.sha256_json(second)


def test_raw_structural_metrics_and_replacement_character_are_not_double_counted() -> (
    None
):
    text = """ΠΕΡΙΕΧΟΜΕΝΑ
1. Εισαγωγή ........ 3
Βιβλιογραφία
| α | β |
|---|---|
ΑΔΑ: ΑΒΓΔ-123
χαλασμένο � Ã©
"""
    metrics = QUALITY.raw_metrics(
        text, private_data_true=True, corrected_version_present=True
    )
    assert metrics["raw_replacement_characters"] == 1
    assert metrics["raw_mojibake_markers"] == 1
    assert metrics["raw_replacement_per_1000_chars"] > 0
    assert metrics["raw_mojibake_per_1000_chars"] > 0
    assert metrics["bibliography_header_detected"] is True
    assert metrics["toc_header_detected"] is True
    assert metrics["raw_markdown_table_lines"] == 2
    assert metrics["isolated_ada_stamp_lines"] == 1
    assert metrics["private_data_true"] is True
    assert metrics["corrected_version_present"] is True

    assert QUALITY.metadata_flags(
        json.dumps(
            {
                "metadata_json": json.dumps(
                    {"privateData": "true", "correctedVersionId": "v2"}
                )
            }
        )
    ) == (True, True)
    with pytest.raises(ValueError, match="source_metadata_json"):
        QUALITY.metadata_flags("{broken")


def test_phase2_route_metrics_are_observable_and_deterministic() -> None:
    text = "\n".join(
        (
            "&amp; &#169; &copy; not-an-entity&",
            "<script>ignored()</script><style>.x{}</style>",
            "<nav>menu</nav><footer>legal</footer>",
            "ανα-",
            "φορά",
            "REPEATED PAGE HEADER",
            "body",
            "REPEATED PAGE HEADER",
        )
    )
    first = QUALITY.raw_metrics(text)
    second = QUALITY.raw_metrics(text)
    assert first == second
    assert first["raw_html_entity_count"] == 3
    assert first["raw_html_entity_per_1000_chars"] == pytest.approx(
        3_000 / len(text)
    )
    # Opening and closing tags are observable markup events, so each explicit
    # script/style or nav/footer pair contributes two.
    assert first["raw_script_style_tag_count"] == 4
    assert first["raw_navigation_markup_tag_count"] == 4
    assert first["raw_line_break_hyphenation_fraction"] == pytest.approx(
        1 / text.count("\n")
    )
    assert QUALITY.raw_metrics("α-\nβ-\nγ")[
        "raw_line_break_hyphenation_fraction"
    ] == pytest.approx(1.0)
    # This is only a repeated-short-line proxy: both repeated header
    # occurrences count, divided by all nonempty physical lines.
    assert first["raw_repeated_short_line_fraction"] == pytest.approx(2 / 8)


def test_frozen_structured_profile_contracts_and_metadata_facts(tmp_path: Path) -> None:
    config = json.loads(SOURCES_CONFIG.read_text(encoding="utf-8"))
    contracts = QUALITY._source_config_structured_profile_contracts(
        config, context="fixture.sources"
    )
    assert contracts["open_council"] is not None
    assert contracts["open_council"].required_all_fields == (
        "subject_id",
        "meeting_id",
    )
    assert contracts["istorima"].required_all_fields == ("id", "title")
    assert contracts["modern_greek_dictionary"].required_all_fields == (
        "lemma",
        "source_url",
    )
    assert contracts["opengov_deliberations_v2"].required_all_fields == (
        "consultation_id",
        "post_id",
        "url",
    )
    school = contracts["school_books_new_editions"]
    assert school is not None
    assert school.required_all_fields == ()
    assert school.required_any_field_groups == (
        ("book_id", "handle", "identifier", "mdb_code"),
        ("PDF_Link", "pdf_urls", "pdf_files"),
    )

    open_council = QUALITY.structured_profile_metrics(
        json.dumps(
            {
                "subject_id": "subject-1",
                "nested": {"meeting": {"location": "Athens"}},
            }
        ),
        contract=contracts["open_council"],
        context="fixture.open_council",
    )
    assert open_council == {
        "structured_contract_declared": True,
        "structured_required_field_count": 2,
        "structured_present_required_field_count": 1,
        "structured_missing_required_field_count": 1,
        "structured_metadata_max_depth": 3,
    }
    school_complete = QUALITY.structured_profile_metrics(
        json.dumps({"identifier": "book-1", "pdf_files": ["book.pdf"]}),
        contract=school,
        context="fixture.school.complete",
    )
    assert school_complete["structured_required_field_count"] == 2
    assert school_complete["structured_present_required_field_count"] == 2
    assert school_complete["structured_missing_required_field_count"] == 0
    school_missing_link = QUALITY.structured_profile_metrics(
        json.dumps({"identifier": "book-1"}),
        contract=school,
        context="fixture.school.missing",
    )
    assert school_missing_link["structured_present_required_field_count"] == 1
    assert school_missing_link["structured_missing_required_field_count"] == 1
    assert QUALITY.structured_profile_metrics(
        None, contract=None, context="fixture.legacy"
    ) == {
        "structured_contract_declared": False,
        "structured_required_field_count": 0,
        "structured_present_required_field_count": 0,
        "structured_missing_required_field_count": 0,
        "structured_metadata_max_depth": 0,
    }

    with pytest.raises(ValueError, match="nonempty field list"):
        QUALITY.validate_structured_profile_contract(
            {"required_all_fields": [], "required_any_field_groups": [[]]},
            context="fixture.malformed",
        )
    with pytest.raises(ValueError, match="key contract drift"):
        QUALITY.validate_structured_profile_contract(
            {"required_all_fields": ["id"], "unexpected": []},
            context="fixture.malformed",
        )

    shard = tmp_path / "input.parquet"
    shard.write_bytes(b"fixture")
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    manifest = write_normalization_fixture(
        tmp_path,
        canonical_root=canonical_root,
        shard=shard,
        source_id="structured-fixture",
        repo_id="owner/structured-fixture",
        revision="a" * 40,
        rows=1,
        structured_profile_contract={
            "required_all_fields": ["id"],
            "required_any_field_groups": [],
        },
    )
    receipt_bound = QUALITY.load_receipt_bound_structured_profile_contracts(manifest)
    assert receipt_bound["structured-fixture"] is not None
    assert receipt_bound["structured-fixture"].required_field_count == 1
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sources config receipt drift"):
        QUALITY.load_receipt_bound_structured_profile_contracts(manifest)


def test_group_stats_reports_zero_rates_and_template_concentration() -> None:
    base = {
        "source_dataset": "diavgeia",
        "original_characters": 100,
        "original_bytes_utf8": 120,
        **{name: 0.0 for name in QUALITY.DISTRIBUTION_METRICS},
        "raw_html_tags": 0,
        "raw_mojibake_markers": 0,
        "raw_replacement_characters": 0,
        "raw_control_characters": 0,
        "raw_unique_line_fraction": 1.0,
        "raw_one_token_line_fraction": 0.0,
        "raw_markdown_table_lines": 0,
        "bibliography_header_detected": False,
        "toc_header_detected": False,
        "digital_governance_footer_detected": False,
        "personnel_cue_detected": False,
        "isolated_ada_stamp_lines": 0,
        "private_data_true": False,
        "corrected_version_present": False,
        "direct_identifier_match_count": 0,
        "cleaner_is_empty": False,
        "zero_badness_zero_greek_guard": False,
    }
    stats = QUALITY.GroupStats(reservoir_size=100)
    for index, template in enumerate(["a" * 64, "a" * 64, "b" * 64]):
        stats.add(
            {
                **base,
                "document_id": hashlib.sha256(str(index).encode()).hexdigest(),
                "structural_template_id": template,
            }
        )
    result = stats.finish()
    assert result["document_rates"]["html_rate"] == 0.0
    assert result["document_rates"]["bibliography_header_rate"] == 0.0
    assert result["template_concentration"] == {
        "documents_with_template": 3,
        "unique_templates": 2,
        "top_1_fraction": pytest.approx(2 / 3),
        "top_10_fraction": 1.0,
    }


def detailed_noise_row(path: Path, *, score: float, greek: int) -> tuple[object, ...]:
    values: dict[str, object] = {}
    for name in QUALITY.NOISE_FIELDS:
        if name == "rust_noise_badness_score":
            values[name] = score
        elif name == "rust_noise_greek_characters":
            values[name] = greek
        elif name in QUALITY.FLOAT_NOISE_FIELDS:
            values[name] = 0.0
        elif name in QUALITY.INTEGER_NOISE_FIELDS:
            values[name] = 0
        else:
            values[name] = ""
    return (str(path), *(values[name] for name in QUALITY.NOISE_FIELDS))


def test_exact_review_sample_packet_is_bound_and_uses_hashed_display_id(
    tmp_path: Path,
) -> None:
    normalization = tmp_path / "normalization.json"
    normalization.write_text('{"schema_version":"full_cpt_normalization_manifest_v1"}')
    raw_doc_id = "https://private.example/person/123"
    display_id = QUALITY.display_document_id(raw_doc_id)
    uid = hashlib.sha256(b"selected").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps(
            {
                "schema_version": "source_quality_review_request_v1",
                "reviewer_slot": "primary",
                "sample_id": uid,
                "source_dataset": "diavgeia",
                "source": {
                    "source_id": "diavgeia",
                    "source_repo_id": "glossAPI/diavgeia",
                    "source_revision": source_identities()["diavgeia"]["revision"],
                    "source_doc_id": raw_doc_id,
                },
            }
        )
        + "\n"
    )
    text = "πλήρως ανωνυμοποιημένο κείμενο"
    packet = tmp_path / "samples.jsonl"
    row = complete_sample(uid, text, doc_id=raw_doc_id)
    row["normalized_text_sha256"] = "c" * 64
    assert row["display_document_id"] == display_id
    packet.write_text(json.dumps(row) + "\n")
    receipt_path = tmp_path / "sample-receipt.json"
    attestation_path = write_packet_receipt(
        receipt_path,
        packet=packet,
        requests=requests,
        rows=1,
        normalization=normalization,
    )
    rows, inputs = QUALITY.load_review_sample_packet(
        packet_path=packet,
        receipt_path=receipt_path,
        attestation_path=attestation_path,
        requests_path=requests,
        normalization_manifest=normalization,
    )
    assert len(rows) == len(inputs) == 1
    assert "source_doc_id" not in rows[0]
    assert "display_document_id" not in rows[0]
    assert raw_doc_id not in json.dumps(rows)
    assert rows[0]["profile_text_variant"] == (
        "high_precision_identifier_masked_review_sample"
    )


def test_rust_batch_checkpoint_and_zero_greek_guard(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    class FakeNoise:
        calls = 0

        def score_markdown_directory_detailed(self, root: str, threads: int):
            self.calls += 1
            paths = sorted(Path(root).glob("*.md"))
            return [
                detailed_noise_row(paths[0], score=0.0, greek=0),
                detailed_noise_row(paths[1], score=0.0, greek=6),
            ]

    class FakeCleaner:
        calls = 0

        def run_complete_pipeline(
            self,
            input_dir: str,
            output_dir: str,
            report: str,
            scripts: list[str],
            threads: int,
            write_cleaned_files: bool,
        ) -> None:
            self.calls += 1
            assert scripts == ["greek", "latin"]
            assert write_cleaned_files is False
            names = [
                f"{path.stem}.pdf" for path in sorted(Path(input_dir).glob("*.md"))
            ]
            pq.write_table(
                pa.table(
                    {
                        "file_name": names,
                        "badness_score_all_chars": [0.0, 0.0],
                        "percentage_greek_cleaned": [0.0, 100.0],
                        "percentage_latin_cleaned": [100.0, 0.0],
                        "char_count_no_comments": [7, 6],
                        "is_empty": [False, False],
                    }
                ),
                report,
            )

    texts = ["English", "κείμενο"]
    rows = []
    for index, text in enumerate(texts):
        rows.append(
            {
                "source_id": "candidate",
                "source_dataset": "candidate",
                "source_repo_id": "glossAPI/candidate",
                "source_revision": "b" * 40,
                "source_route": "pdf_ocr",
                "review_route": "pdf_ocr",
                "extraction_route": "html_web",
                "observed_extraction_route": "html_web",
                "observed_extraction_route_basis": "explicit_row_route",
                "observed_extraction_route_evidence": "raw_field:format",
                "observed_extraction_route_priority": "secondary_exception_only",
                "source_doc_id": str(index),
                "stable_uid": hashlib.sha256(f"uid-{index}".encode()).hexdigest(),
                "normalized_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text": text,
            }
        )
    input_path = tmp_path / "input.parquet"
    input_path.write_bytes(b"fixture")
    shard = QUALITY.ShardBinding(
        source_id="candidate",
        path=input_path,
        relative_path="candidate/input.parquet",
        bytes=input_path.stat().st_size,
        sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        rows=2,
    )
    build_receipt = tmp_path / "rust-build.json"
    build_receipt.write_text("{}")
    noise = FakeNoise()
    cleaner = FakeCleaner()
    runtime = QUALITY.RustRuntime(
        noise=noise,
        cleaner=cleaner,
        receipt={},
        receipt_path=build_receipt,
    )
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    receipt = QUALITY.process_batch(
        rows=rows,
        shard=shard,
        batch_index=0,
        row_start=0,
        output_root=output,
        scratch_root=scratch,
        contract_sha256="c" * 64,
        runtime=runtime,
        threads=2,
    )
    data = pq.read_table(
        Path(receipt["receipt"]["path"]).parent / "documents.parquet"
    ).to_pylist()
    assert [row["zero_badness_zero_greek_guard"] for row in data] == [True, False]
    assert [row["noise_score_interpretation"] for row in data] == [
        "guarded_zero_score_without_greek",
        "zero_score_with_greek",
    ]
    # Direct/legacy callers that have no canonical metadata contract remain
    # valid and explicitly report no invented structured completeness facts.
    assert [row["structured_contract_declared"] for row in data] == [False, False]
    assert [row["structured_required_field_count"] for row in data] == [0, 0]
    assert [row["structured_missing_required_field_count"] for row in data] == [0, 0]
    assert not list(scratch.iterdir())
    assert noise.calls == cleaner.calls == 1

    class MustNotRun:
        def __getattr__(self, name: str):
            raise AssertionError(f"checkpoint resume called Rust: {name}")

    resumed = QUALITY.process_batch(
        rows=rows,
        shard=shard,
        batch_index=0,
        row_start=0,
        output_root=output,
        scratch_root=scratch,
        contract_sha256="c" * 64,
        runtime=QUALITY.RustRuntime(
            noise=MustNotRun(),
            cleaner=MustNotRun(),
            receipt={},
            receipt_path=build_receipt,
        ),
        threads=2,
    )
    assert resumed["output"]["sha256"] == receipt["output"]["sha256"]

    checkpoint_directory = Path(receipt["receipt"]["path"]).parent
    external = tmp_path / "external-checkpoint"
    external.mkdir()
    for name in ("receipt.json", "documents.parquet"):
        (external / name).write_bytes((checkpoint_directory / name).read_bytes())

    directory_link = output / "linked-checkpoint"
    directory_link.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe directory path"):
        QUALITY.validate_batch_checkpoint(
            directory_link,
            output_root=output,
            contract_sha256="c" * 64,
            shard=shard,
            batch_index=0,
            row_start=0,
            row_end=2,
        )

    receipt_link_checkpoint = output / "receipt-link-checkpoint"
    receipt_link_checkpoint.mkdir()
    (receipt_link_checkpoint / "receipt.json").symlink_to(external / "receipt.json")
    (receipt_link_checkpoint / "documents.parquet").write_bytes(
        (external / "documents.parquet").read_bytes()
    )
    with pytest.raises(ValueError, match="unsafe relative file path"):
        QUALITY.validate_batch_checkpoint(
            receipt_link_checkpoint,
            output_root=output,
            contract_sha256="c" * 64,
            shard=shard,
            batch_index=0,
            row_start=0,
            row_end=2,
        )

    output_link_checkpoint = output / "output-link-checkpoint"
    output_link_checkpoint.mkdir()
    (output_link_checkpoint / "receipt.json").write_bytes(
        (external / "receipt.json").read_bytes()
    )
    (output_link_checkpoint / "documents.parquet").symlink_to(
        external / "documents.parquet"
    )
    with pytest.raises(ValueError, match="unsafe relative file path"):
        QUALITY.validate_batch_checkpoint(
            output_link_checkpoint,
            output_root=output,
            contract_sha256="c" * 64,
            shard=shard,
            batch_index=0,
            row_start=0,
            row_end=2,
        )

    document_output, global_summary, repositories, route_coverage = QUALITY.consolidate_batches(
        [receipt], output_root=output, reservoir_size=100
    )
    assert document_output["rows"] == global_summary["documents"] == 2
    assert repositories[0]["repo_id"] == "glossAPI/candidate"
    assert route_coverage["sources"] == [
        {
            "source_id": "candidate",
            "documents": 2,
            "source_route": "pdf_ocr",
            "review_route": "pdf_ocr",
            "extraction_route": "html_web",
            "observed_extraction_route_counts": [
                {"route": "html_web", "documents": 2}
            ],
            "observed_extraction_route_basis_counts": [
                {"basis": "explicit_row_route", "documents": 2}
            ],
            "observed_extraction_route_priority_counts": [
                {"priority": "secondary_exception_only", "documents": 2}
            ],
        }
    ]
    jsonschema = pytest.importorskip("jsonschema")
    document_contract = json.loads(
        (HERE / "schemas" / "dataset_quality_document.schema.json").read_text()
    )
    assert set(QUALITY.document_schema().names) == set(document_contract["required"])
    for row in pq.read_table(
        output / "dataset_quality_document_v2.parquet"
    ).to_pylist():
        jsonschema.Draft202012Validator(document_contract).validate(row)

    original_checkpoint = Path(receipt["receipt"]["path"]).parent
    alias_checkpoint = output / "alias-checkpoint"
    alias_checkpoint.mkdir()
    (alias_checkpoint / "receipt.json").write_text("{}")
    os.link(
        original_checkpoint / "documents.parquet",
        alias_checkpoint / "documents.parquet",
    )
    aliased_receipt = {
        **receipt,
        "receipt": {
            **receipt["receipt"],
            "path": str(alias_checkpoint / "receipt.json"),
        },
        "batch_index": 1,
    }
    with pytest.raises(ValueError, match="aliased quality checkpoint output identity"):
        QUALITY.consolidate_batches(
            [receipt, aliased_receipt], output_root=output, reservoir_size=100
        )


def test_nofollow_reader_rejects_path_swap_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    receipt = root / "receipt.json"
    receipt.write_text('{"status":"passed"}')
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"status":"forged"}')
    original_read = QUALITY.os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        data = original_read(descriptor, size)
        if data and not swapped:
            swapped = True
            receipt.rename(root / "receipt.original.json")
            receipt.symlink_to(replacement)
        return data

    monkeypatch.setattr(QUALITY.os, "read", swapping_read)
    with pytest.raises(
        ValueError,
        match="file changed while reading|identity recheck|unsafe relative file path",
    ):
        QUALITY.load_relative_json_nofollow(
            root, "receipt.json", context="swap fixture"
        )


def test_build_receipt_can_bind_staged_modules_to_atomic_publish_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "glossapi"
    for crate in ("glossapi_rs_noise", "glossapi_rs_cleaner"):
        lock = source / "rust" / crate / "Cargo.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"lock for {crate}")
    staged = tmp_path / "runtime.partial" / "modules"
    staged.mkdir(parents=True)
    module_paths = {}
    for name in ("glossapi_rs_noise", "glossapi_rs_cleaner"):
        path = staged / f"{name}.so"
        path.write_bytes(name.encode())
        module_paths[name] = path.resolve()
    published = tmp_path / "runtime" / "modules"

    def fake_git_output(root: Path, *args: str) -> str:
        if args == ("rev-parse", "--is-inside-work-tree"):
            return "true"
        if args == ("rev-parse", "HEAD"):
            return QUALITY.PINNED_GLOSSAPI_COMMIT
        if args == ("status", "--porcelain", "--untracked-files=normal"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(QUALITY, "git_output", fake_git_output)
    monkeypatch.setattr(QUALITY, "module_path", lambda name: module_paths[name])
    monkeypatch.setattr(
        QUALITY, "tool_version", lambda command, *arguments: f"{command} test-version"
    )
    output = tmp_path / "runtime.partial" / "build_receipt.json"
    assert (
        QUALITY.build_runtime_receipt(
            SimpleNamespace(
                glossapi_root=source,
                expected_commit=QUALITY.PINNED_GLOSSAPI_COMMIT,
                module_root=staged,
                published_module_root=published,
                maturin_version="1.9.4",
                output=output,
            )
        )
        == 0
    )
    value = json.loads(output.read_text())
    assert {Path(row["path"]).parent for row in value["modules"]} == {
        published.resolve()
    }
    assert all(Path(row["path"]).name.endswith(".so") for row in value["modules"])
    assert all(not Path(row["path"]).exists() for row in value["modules"])
    assert value["runtime"]["rustc"] == "rustc test-version"
    assert value["runtime"]["cargo"] == "cargo test-version"
    assert value["runtime"]["maturin"] == "1.9.4"


@pytest.mark.parametrize(
    "counts",
    [
        {"raw_identifier": 1},
        {"email": 0},
        {"email": -1},
        {"email": True},
    ],
)
def test_redaction_count_registry_is_closed_and_positive(
    counts: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="invalid redaction count"):
        EXPORTER.validate_redaction_counts(counts, context="fixture")


def test_redaction_count_schema_rejects_unknown_and_zero_values() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (HERE / "schemas" / "dataset_review_complete_sample.schema.json").read_text()
    )
    row = complete_sample("a" * 64, "κείμενο")
    row["redaction_counts"] = {"raw_identifier": 0}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(row)


def test_complete_sample_redaction_covers_identity_ipv6_and_email() -> None:
    text = (
        "email user@example.gr IPv6 2001:0db8:85a3:0000:0000:8a2e:0370:7334 "
        "ΑΔΤ: ΑΒ123456"
    )
    redacted, counts = EXPORTER.redact_complete_text(text)
    assert "user@example.gr" not in redacted
    assert "2001:0db8" not in redacted
    assert "ΑΒ123456" not in redacted
    assert counts["email"] == counts["ipv6"] == counts["identity"] == 1


def test_sample_export_checkpoint_rejects_symlinked_directory_and_files(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    fragment = external / "samples.jsonl"
    fragment.write_text('{"sample_id":"one"}\n')
    shard_receipt = {
        "source_id": "candidate",
        "path": "candidate/part.parquet",
        "bytes": 10,
        "sha256": "a" * 64,
        "rows": 1,
    }
    checkpoint = {
        "schema_version": EXPORTER.CHECKPOINT_SCHEMA,
        "status": "passed",
        "contract_sha256": "c" * 64,
        "input_shard": shard_receipt,
        "rows_scanned": 1,
        "redaction_totals": {},
        "output": {
            "path": "samples.jsonl",
            "bytes": fragment.stat().st_size,
            "sha256": QUALITY.sha256_file(fragment),
            "rows": 1,
        },
    }
    (external / "receipt.json").write_text(json.dumps(checkpoint))

    directory_link = checkpoint_root / "directory-link"
    directory_link.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe directory path"):
        EXPORTER.validate_checkpoint(
            directory_link,
            checkpoint_root=checkpoint_root,
            contract_sha256="c" * 64,
            shard_receipt=shard_receipt,
        )

    receipt_link = checkpoint_root / "receipt-link"
    receipt_link.mkdir()
    (receipt_link / "receipt.json").symlink_to(external / "receipt.json")
    (receipt_link / "samples.jsonl").write_bytes(fragment.read_bytes())
    with pytest.raises(ValueError, match="unsafe relative file path"):
        EXPORTER.validate_checkpoint(
            receipt_link,
            checkpoint_root=checkpoint_root,
            contract_sha256="c" * 64,
            shard_receipt=shard_receipt,
        )

    output_link = checkpoint_root / "output-link"
    output_link.mkdir()
    (output_link / "receipt.json").write_text(json.dumps(checkpoint))
    (output_link / "samples.jsonl").symlink_to(fragment)
    with pytest.raises(ValueError, match="unsafe relative file path"):
        EXPORTER.validate_checkpoint(
            output_link,
            checkpoint_root=checkpoint_root,
            contract_sha256="c" * 64,
            shard_receipt=shard_receipt,
        )


def test_sample_export_checkpoint_runtime_matches_strict_schema(
    tmp_path: Path,
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            HERE
            / "schemas"
            / "dataset_review_sample_export_shard_checkpoint.schema.json"
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    shard_receipt = {
        "source_id": "candidate",
        "path": "candidate/part.parquet",
        "bytes": 10,
        "sha256": "a" * 64,
        "rows": 1,
    }
    fragment_bytes = b'{"sample_id":"one"}\n'
    base = {
        "schema_version": EXPORTER.CHECKPOINT_SCHEMA,
        "status": "passed",
        "contract_sha256": "c" * 64,
        "input_shard": shard_receipt,
        "rows_scanned": 1,
        "redaction_totals": {},
        "output": {
            "path": "samples.jsonl",
            "bytes": len(fragment_bytes),
            "sha256": hashlib.sha256(fragment_bytes).hexdigest(),
            "rows": 1,
        },
    }
    cases: list[dict[str, object]] = []
    for mutation in (
        ("unexpected_root", "root"),
        ("unexpected_output", "output"),
        ("unexpected_input", "input"),
        ("boolean_rows_scanned", "rows_scanned"),
        ("boolean_output_rows", "output_rows"),
        ("string_output_bytes", "output_bytes"),
    ):
        value = json.loads(json.dumps(base))
        if mutation[1] == "root":
            value["unexpected_root"] = True
        elif mutation[1] == "output":
            value["output"]["unexpected"] = True
        elif mutation[1] == "input":
            value["input_shard"]["unexpected"] = True
        elif mutation[1] == "rows_scanned":
            value["rows_scanned"] = True
        elif mutation[1] == "output_rows":
            value["output"]["rows"] = True
        else:
            value["output"]["bytes"] = str(len(fragment_bytes))
        value["_case_name"] = mutation[0]
        cases.append(value)

    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    for index, value in enumerate(cases):
        case_name = str(value.pop("_case_name"))
        directory = checkpoint_root / f"case-{index}"
        directory.mkdir()
        (directory / "samples.jsonl").write_bytes(fragment_bytes)
        (directory / "receipt.json").write_text(json.dumps(value))
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(value)
        with pytest.raises(ValueError):
            EXPORTER.validate_checkpoint(
                directory,
                checkpoint_root=checkpoint_root,
                contract_sha256="c" * 64,
                shard_receipt=shard_receipt,
            )
        assert case_name


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.org:8443/private/report.pdf?email="
        "alice%40example.org&token=s3cr3t%2Bvalue#account-123",
        "HTTP://example.org/a_(balanced)/b;params?q=one%20two&next="
        "https%3A%2F%2Fevil.example%2Fprivate#Greek-\u03b1\u03c0\u03cc\u03c1\u03c1\u03b7\u03c4\u03bf",
        "www.example.gr/path/to/page?session=abc123&empty=#private-fragment",
    ],
)
def test_complete_sample_redaction_masks_full_url_query_and_fragment(url: str) -> None:
    redacted, counts = EXPORTER.redact_complete_text(f"before <{url}> after")
    assert redacted == "before <[REDACTED_URL]> after"
    assert counts == {"url": 1}
    for secret in ("password", "alice", "s3cr3t", "session", "private"):
        assert secret not in redacted


def test_complete_sample_url_redaction_handles_multiple_contexts_without_email_overlap() -> (
    None
):
    text = (
        'href="https://example.org/download?owner=user@example.gr#record-7" '
        "markdown=(www.example.gr/a?x=1&y=2#section) "
        "standalone=user@www.example.gr"
    )
    redacted, counts = EXPORTER.redact_complete_text(text)
    assert "https://" not in redacted
    assert "www.example.gr/a" not in redacted
    assert "owner=" not in redacted
    assert "#record-7" not in redacted
    assert "#section" not in redacted
    assert "user@www.example.gr" not in redacted
    assert counts == {"email": 1, "url": 2}


def test_sample_export_omits_raw_source_document_identifier(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "canonical"
    shard = root / "diavgeia" / "part.parquet"
    shard.parent.mkdir(parents=True)
    raw_doc_id = "https://private.example/records/person-123"
    text = "κείμενο user@example.gr"
    uid = hashlib.sha256(b"exported").hexdigest()
    pq.write_table(
        pa.table(
            {
                "source_id": ["diavgeia"],
                "stable_uid": [uid],
                "source_repo_id": ["glossAPI/diavgeia"],
                "source_revision": ["a" * 40],
                "source_dataset": ["diavgeia"],
                "source_doc_id": [raw_doc_id],
                "normalized_text_sha256": [hashlib.sha256(text.encode()).hexdigest()],
                "source_metadata_json": [json.dumps({"correctedVersionId": "v2"})],
                "text": [text],
            }
        ),
        shard,
    )
    manifest_path = write_normalization_fixture(
        tmp_path,
        canonical_root=root,
        shard=shard,
        source_id="diavgeia",
        repo_id="glossAPI/diavgeia",
        revision="a" * 40,
        rows=1,
    )
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps(
            {
                "schema_version": "source_quality_review_request_v1",
                "reviewer_slot": "primary",
                "sample_id": uid,
                "source_dataset": "diavgeia",
                "source": {
                    "source_id": "diavgeia",
                    "source_repo_id": "glossAPI/diavgeia",
                    "source_revision": "a" * 40,
                    "source_doc_id": raw_doc_id,
                },
            }
        )
        + "\n"
    )
    packet = tmp_path / "complete.jsonl"
    packet_receipt = tmp_path / "complete-receipt.json"
    site_attestation = tmp_path / "complete-site-attestation.json"
    assert (
        EXPORTER.export_samples(
            SimpleNamespace(
                output=packet,
                receipt=packet_receipt,
                site_attestation=site_attestation,
                resume=False,
                review_requests=requests,
                normalization_manifest=manifest_path,
                canonical_root=root,
                scratch_dir=tmp_path / "scratch",
                batch_size=8,
            )
        )
        == 0
    )
    source = packet.read_text(encoding="utf-8")
    row = json.loads(source)
    assert raw_doc_id not in source
    assert "source_doc_id" not in row
    assert row["display_document_id"] == QUALITY.display_document_id(raw_doc_id)
    assert "user@example.gr" not in row["text"]
    assert row["profile_text_variant"] == (
        "high_precision_identifier_masked_review_sample"
    )
    assert row["private_data_true"] is False
    assert row["corrected_version_present"] is True
    receipt_value = json.loads(packet_receipt.read_text())
    assert receipt_value["output"]["path"] == packet.name
    assert (
        receipt_value["checkpoint_inventory_sha256"]
        == hashlib.sha256(
            EXPORTER.canonical_json(receipt_value["checkpoint_inventory"]).encode()
        ).hexdigest()
    )
    jsonschema = pytest.importorskip("jsonschema")
    packet_schema = json.loads(
        (HERE / "schemas" / "dataset_review_complete_sample.schema.json").read_text()
    )
    receipt_schema = json.loads(
        (
            HERE
            / "schemas"
            / "dataset_review_complete_sample_packet_receipt.schema.json"
        ).read_text()
    )
    attestation_schema = json.loads(
        (
            HERE
            / "schemas"
            / "dataset_review_complete_sample_site_attestation.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(packet_schema).validate(row)
    jsonschema.Draft202012Validator(receipt_schema).validate(receipt_value)
    jsonschema.Draft202012Validator(attestation_schema).validate(
        json.loads(site_attestation.read_text())
    )
    assert (
        EXPORTER.export_samples(
            SimpleNamespace(
                output=packet,
                receipt=packet_receipt,
                site_attestation=site_attestation,
                resume=True,
                review_requests=requests,
                normalization_manifest=manifest_path,
                canonical_root=root,
                scratch_dir=tmp_path / "scratch",
                batch_size=8,
            )
        )
        == 0
    )


def test_clariden_wrapper_is_cpu_only_resumable_and_4096_bounded() -> None:
    wrapper = (HERE / "clariden" / "41_profile_dataset_quality_rust.sbatch").read_text()
    builder = (
        HERE / "clariden" / "06_build_glossapi_quality_runtime.sbatch"
    ).read_text()
    submit = (HERE / "clariden" / "submit.sh").read_text()
    assert "#SBATCH --cpus-per-task=256" in wrapper
    assert "#SBATCH --gres" not in wrapper
    assert "phase04_require_cpu_request" in wrapper
    assert "BATCH_SIZE=4096" in wrapper
    assert "QUALITY_STAGE=35-dataset-quality-sample" in wrapper
    assert "QUALITY_STAGE=15-dataset-quality-full" in wrapper
    assert 'phase04_stage_require_upstream "10-normalize"' in wrapper
    assert 'phase04_stage_require_upstream "30-review-packet"' in wrapper
    assert "--review-sample-packet" in wrapper
    assert '--checkpoint-dir "$PHASE04_STAGE_DIR/sample-export-checkpoints"' in wrapper
    assert "--site-attestation" in wrapper
    assert "--review-sample-attestation" in wrapper
    assert "--site-handoff" in wrapper
    assert 'phase04_stage_bind_parameter scan_mode "$QUALITY_MODE"' in wrapper
    assert "--resume" in wrapper
    assert 'CUDA_VISIBLE_DEVICES=""' in wrapper
    assert "dataset-quality|dataset-quality-sample|35-dataset-quality-sample" in submit
    assert "dataset-quality-full|15-dataset-quality-full" in submit
    assert "QUALITY_MODE=review_sample" in submit
    assert "QUALITY_MODE=full_scan" in submit

    assert "#SBATCH --cpus-per-task=128" in builder
    assert "#SBATCH --gres" not in builder
    assert "phase04_require_cpu_request" in builder
    assert "maturin" in builder and "--locked" in builder
    assert "CARGO_TARGET_DIR" in builder
    assert '--module-root "$PARTIAL/modules"' in builder
    assert '--published-module-root "$GLOSSAPI_QUALITY_MODULE_DIR"' in builder
    assert 'mv "$PARTIAL" "$GLOSSAPI_QUALITY_RUNTIME_ROOT"' in builder
    assert "build-quality-runtime" in submit


def test_new_json_schemas_are_parseable_and_versioned() -> None:
    expected = {
        "glossapi_rust_quality_build_receipt.schema.json": "glossapi_rust_quality_build_receipt_v1",
        "dataset_quality_document.schema.json": "dataset_quality_document_v2",
        "dataset_quality_summary.schema.json": "dataset_quality_summary_v2",
        "dataset_quality_site_handoff.schema.json": "dataset_quality_site_handoff_v2",
        "dataset_review_complete_sample.schema.json": "dataset_review_complete_sample_v1",
        "dataset_review_complete_sample_packet_receipt.schema.json": (
            "dataset_review_complete_sample_packet_receipt_v1"
        ),
        "dataset_review_complete_sample_site_attestation.schema.json": (
            "dataset_review_complete_sample_site_attestation_v1"
        ),
        "dataset_review_sample_export_contract.schema.json": (
            "dataset_review_sample_export_contract_v1"
        ),
        "dataset_review_sample_export_shard_checkpoint.schema.json": (
            "dataset_review_sample_export_shard_checkpoint_v1"
        ),
        "dataset_review_site_sample.schema.json": "dataset_review_site_sample_v1",
        "dataset_review_public_sample_packet.schema.json": (
            "dataset_review_public_sample_packet_v1"
        ),
        "dataset_review_public_site_sample.schema.json": (
            "dataset_review_public_site_sample_v1"
        ),
        "dataset_review_site_manifest.schema.json": "dataset_review_site_manifest_v1",
        "dataset_review_presentation_handoff.schema.json": (
            "dataset_review_presentation_handoff_v1"
        ),
    }
    for name, version in expected.items():
        value = json.loads((HERE / "schemas" / name).read_text())
        assert value["$schema"].endswith("2020-12/schema")
        schema_version = value["properties"]["schema_version"]
        assert schema_version["const"] == version


def write_presentation_handoff(tmp_path: Path, *, kind: str = "fixture") -> Path:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    entries = []
    for role, source, schema_version in (
        ("inventory", HERE / "configs" / "post_december_inventory.json", "post_december_glossapi_inventory_v1"),
        ("evaluations", HERE / "configs" / "dataset_review_evaluations.json", "dataset_review_evaluations_v1"),
        ("sources_config", SOURCES_CONFIG, "full_cpt_sources_v1"),
    ):
        target = handoff / source.name
        shutil.copy2(source, target)
        entries.append(
            {
                "role": role,
                "path": target.name,
                "bytes": target.stat().st_size,
                "sha256": QUALITY.sha256_file(target),
                "schema_version": schema_version,
            }
        )
    (handoff / "dataset_review_site_handoff.json").write_text(
        json.dumps(
            {
                "schema_version": "dataset_review_presentation_handoff_v1",
                "status": "passed",
                "run_id": "fixture-20260712",
                "producer_commit": "a" * 40,
                "created_at": "2026-07-12T00:00:00Z",
                "handoff_kind": kind,
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    return handoff


def test_presentation_handoff_build_is_deterministic_and_closed(tmp_path: Path) -> None:
    handoff = write_presentation_handoff(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert PRESENTATION.build(
        SimpleNamespace(handoff_dir=handoff, output_dir=first)
    ) == 0
    assert PRESENTATION.build(
        SimpleNamespace(handoff_dir=handoff, output_dir=second)
    ) == 0
    assert PRESENTATION.validate_site_directory(first)["status"] == "passed"
    first_files = {
        item.relative_to(first): QUALITY.sha256_file(item)
        for item in first.rglob("*")
        if item.is_file()
    }
    second_files = {
        item.relative_to(second): QUALITY.sha256_file(item)
        for item in second.rglob("*")
        if item.is_file()
    }
    assert first_files == second_files
    report = (first / "site_acceptance_report.md").read_text(encoding="utf-8")
    assert "fixture-20260712" in report
    assert "public/masked variant toggle" in report
    assert 'href="documents.html"' in (first / "index.html").read_text()
    assert "textContent=doc.text" in (first / "assets" / "site.js").read_text()
    stylesheet = (first / "assets" / "site.css").read_text()
    assert "dd{overflow-wrap:anywhere}" in stylesheet
    assert ".coverage-table{min-width:0;table-layout:fixed}" in stylesheet


def test_presentation_handoff_rejects_extra_file_and_symlink(tmp_path: Path) -> None:
    handoff = write_presentation_handoff(tmp_path)
    (handoff / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="directory closure"):
        PRESENTATION.accept_handoff(handoff, require_complete=False)
    (handoff / "extra.json").unlink()
    (handoff / "linked.json").symlink_to(handoff / "post_december_inventory.json")
    with pytest.raises(ValueError, match="symlink"):
        PRESENTATION.accept_handoff(handoff, require_complete=False)


def test_fixture_handoff_cannot_be_published(tmp_path: Path) -> None:
    handoff = write_presentation_handoff(tmp_path)
    with pytest.raises(ValueError, match="only an Agent-1 handoff"):
        PRESENTATION.accept_handoff(handoff, require_complete=True)


def test_public_preview_is_plainly_labelled_and_browseable(tmp_path: Path) -> None:
    source = source_identities()["diavgeia"]
    text = "</script><img src=x onerror=alert(1)> δημόσιο δείγμα"
    sample_id = hashlib.sha256(b"public-preview").hexdigest()
    previews = tmp_path / "public-previews.json"
    previews.write_text(
        json.dumps(
            {
                "schema_version": "dataset_review_public_sample_packet_v1",
                "samples": [
                    {
                        "schema_version": "dataset_review_public_sample_v1",
                        "sample_id": sample_id,
                        "repo_id": "glossAPI/diavgeia",
                        "source_revision": source["revision"],
                        "source_document_id": "public-document-1",
                        "displayed_text_is_excerpt": True,
                        "displayed_text_characters": len(text),
                        "displayed_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "text": text,
                        "metadata": {"kind": "fixture"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "site"
    SITE.build_site(
        SimpleNamespace(
            inventory=HERE / "configs" / "post_december_inventory.json",
            evaluations=HERE / "configs" / "dataset_review_evaluations.json",
            sources_config=SOURCES_CONFIG,
            quality_summary=None,
            quality_handoff_receipt=None,
            review_requests=None,
            review_responses=None,
            admission=None,
            novelty=None,
            complete_samples=None,
            complete_samples_receipt=None,
            complete_samples_attestation=None,
            public_previews=previews,
            pipeline_waterfall=None,
            output_dir=output,
            replace=False,
        )
    )
    sample = json.loads(next((output / "samples").glob("*-public.json")).read_text())
    assert sample["label"] == "public source excerpt"
    assert sample["high_precision_identifier_patterns_masked"] is False
    assert text not in (output / "documents.html").read_text()
    assert "Browse documents" in (output / "documents.html").read_text()
    assert "textContent=doc.text" in (output / "assets" / "site.js").read_text()
