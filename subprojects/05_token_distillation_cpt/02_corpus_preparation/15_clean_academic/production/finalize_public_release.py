#!/usr/bin/env python3
"""Finalize a verified candidate as an immutable public Hugging Face release."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import atomic_write_json, load_json, sha256_file, validate_contract
from .count_tokens import TOKEN_SUMMARY_SCHEMA
from .materialize_release import RECONSTRUCTION_SCHEMA

PUBLIC_POLICY_SCHEMA = "bibliography-cleaning-public-release-policy-v1"


def _receipt(path: Path, *, root: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        raise RuntimeError(
            "final release requires same-filesystem hardlinks"
        ) from error


def _format_integer(value: int) -> str:
    return f"{value:,}"


def _readme(
    *,
    manifest: dict[str, Any],
    token_summary: dict[str, Any],
    cleaning_summary: dict[str, Any],
    created_at: str,
) -> str:
    stats = cleaning_summary["overall"]
    base_tokens = int(token_summary["base"]["training_tokens"])
    cleaned_tokens = int(token_summary["cleaned"]["training_tokens"])
    untouched = len(manifest["bibliography_cleaning"]["untouched_ranks"])
    return f"""---
license: other
configs:
- config_name: default
  data_files: data/*.parquet
---

# Greek Nanochat plus new sources — bibliography-cleaned v2

This is the public, manually gated v2 dataset revision materialized on
{created_at}. It retains the original deduplicated 51,839,746 documents and
removes detected bibliography/reference-list lines from 175,242 documents in
`greek_phd`, `openarchives.gr`, `glossAPI/elocus`, and `glossAPI/libduth`.
Kallipos was analyzed but was not changed.

## Cleaning result

- Documents in release: {_format_integer(int(manifest["rows"]))}
- Documents examined in the apply scope: {_format_integer(int(stats["docs"]))}
- Documents with at least one removal: {_format_integer(int(stats["docs_cleaned"]))}
- Characters removed: {_format_integer(int(stats["total_chars_removed"]))}
- Apply-scope character removal: {float(stats["char_removal_pct"]):.6f}%
- Documents emptied: {int(stats["would_empty"])}
- Shards transformed: {len(manifest["bibliography_cleaning"]["transformed_ranks"])}
- Shards preserved checksum-identically: {untouched}

## Exact tokenizer accounting

Tokenizer: `fffoivos/apertus-tokenizer-extension` at
`a4826df7f76b54cdd6dc21d09fe97283c466999b` (vocabulary 148,480).
Training-token totals include one EOS token per document.

- Previous v2: {_format_integer(base_tokens)} training tokens
- Bibliography-cleaned v2: {_format_integer(cleaned_tokens)} training tokens
- Difference: {_format_integer(cleaned_tokens - base_tokens)} tokens

## Provenance and rights

The release is receipt-bound: `manifests/deduplicated_manifest.json` records
all 431 shard hashes, the cleaning contract and summary, reconstruction
evidence, and exact token-count evidence.

This is a mixed-source corpus and no single repository-level license replaces
the terms or restrictions of its component sources. In particular, inclusion
of `glossAPI/libduth` follows an explicit dataset-owner operational directive;
that directive is not represented as rightsholder permission and does not
supersede the recorded source-license warning. Users must review source
metadata and comply with applicable rights and restrictions.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract, contract_sha = validate_contract(args.contract)
    policy = contract["policy"]
    target = policy["publication_target"]
    if (
        target["repo_id"] != args.repo_id
        or target["visibility"] != "public"
        or policy["publication_authorized"] is not True
    ):
        raise ValueError("contract policy does not authorize this public target")

    candidate = Path(args.candidate).resolve()
    candidate_manifest_path = candidate / "manifests" / "deduplicated_manifest.json"
    candidate_manifest = load_json(candidate_manifest_path)
    if (
        Path(candidate_manifest["root"]).resolve() != candidate
        or candidate_manifest.get("publication_ready") is not False
        or candidate_manifest.get("private_only") is not False
        or len(candidate_manifest["files"]) != 431
    ):
        raise ValueError("input is not a non-publishable validated candidate")
    reconstruction_path = (
        candidate
        / candidate_manifest["bibliography_cleaning"]["reconstruction"]["path"]
    )
    reconstruction = load_json(reconstruction_path)
    if (
        reconstruction.get("schema_version") != RECONSTRUCTION_SCHEMA
        or reconstruction.get("status") != "passed"
        or sha256_file(reconstruction_path)
        != candidate_manifest["bibliography_cleaning"]["reconstruction"]["sha256"]
        or reconstruction["contract"]["sha256"] != contract_sha
    ):
        raise ValueError("candidate reconstruction evidence is invalid")
    source_manifest_path = Path(reconstruction["source_manifest"]["path"])
    source_manifest = load_json(source_manifest_path)
    if (
        sha256_file(source_manifest_path)
        != reconstruction["source_manifest"]["sha256"]
        or candidate_manifest["decision_ledger"]["sha256"]
        != source_manifest["decision_ledger"]["sha256"]
        or int(candidate_manifest["decision_ledger"]["bytes"])
        != int(source_manifest["decision_ledger"]["bytes"])
        or int(candidate_manifest["decision_ledger"]["rows"])
        != int(source_manifest["decision_ledger"]["rows"])
    ):
        raise ValueError("deduplication decision ledger provenance failed")

    token_summary_path = Path(args.token_summary).resolve()
    token_summary = load_json(token_summary_path)
    if (
        token_summary.get("schema_version") != TOKEN_SUMMARY_SCHEMA
        or token_summary.get("status") != "passed"
        or token_summary["plan"]["sha256"] != sha256_file(token_summary["plan"]["path"])
    ):
        raise ValueError("token summary is not passed")
    token_plan = load_json(token_summary["plan"]["path"])
    if (
        token_plan["cleaned_manifest"]["sha256"] != sha256_file(candidate_manifest_path)
        or Path(token_plan["cleaned_manifest"]["path"]).resolve()
        != candidate_manifest_path
        or int(token_summary["cleaned"]["documents"]) != int(candidate_manifest["rows"])
    ):
        raise ValueError("token summary is not bound to this candidate")

    apply_summary_path = Path(contract["run_root"]) / "apply" / "summary.json"
    apply_summary = load_json(apply_summary_path)
    if (
        candidate_manifest["bibliography_cleaning"]["apply_summary_sha256"]
        != sha256_file(apply_summary_path)
        or apply_summary.get("status") != "passed"
    ):
        raise ValueError("apply summary changed after candidate reconstruction")

    output = Path(args.output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable final release already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        files = []
        for row in sorted(
            candidate_manifest["files"], key=lambda value: int(value["rank"])
        ):
            source = candidate / row["path"]
            if (
                source.is_symlink()
                or source.stat().st_size != int(row["bytes"])
                or sha256_file(source) != row["sha256"]
                or pq.ParquetFile(source).metadata.num_rows != int(row["rows"])
            ):
                raise ValueError(f"candidate shard drift: {row['path']}")
            destination = temporary / row["path"]
            _hardlink(source, destination)
            files.append(
                {**row, **_receipt(destination, root=temporary, rows=int(row["rows"]))}
            )

        decision_source = candidate / candidate_manifest["decision_ledger"]["path"]
        decision_destination = temporary / "manifests" / "dedup_decision_ledger.parquet"
        _hardlink(decision_source, decision_destination)
        reconstruction_destination = (
            temporary / "manifests" / "bibliography_reconstruction.json"
        )
        _hardlink(reconstruction_path, reconstruction_destination)
        token_destination = temporary / "manifests" / "token_counts.json"
        _hardlink(token_summary_path, token_destination)

        prior_policy_path = (
            Path(contract["release_root"])
            / "manifests"
            / "license_override_receipt.json"
        )
        libduth = policy["license_overrides"]["glossAPI/libduth"]
        public_policy = {
            "schema_version": PUBLIC_POLICY_SCHEMA,
            "status": "passed",
            "created_at": args.created_at,
            "approved_by": libduth["approved_by"],
            "approved_on": libduth["approved_on"],
            "repository_id": args.repo_id,
            "visibility": "public",
            "gating": target["gating"],
            "scope": libduth["scope"],
            "includes_cleaned_libduth": True,
            "authorization_basis": libduth["authorization_basis"],
            "does_not_supersede_source_terms": True,
            "mixed_source_rights_warning": (
                "No repository-level license replaces component-source terms. "
                "Users remain responsible for complying with applicable restrictions."
            ),
            "prior_private_override": {
                "path": str(prior_policy_path),
                "sha256": sha256_file(prior_policy_path),
            },
            "contract": {
                "path": str(Path(args.contract).resolve()),
                "sha256": contract_sha,
            },
        }
        policy_destination = temporary / "manifests" / "license_override_receipt.json"
        atomic_write_json(policy_destination, public_policy)

        inventory_path = temporary / "manifests" / "deduplicated_inventory.parquet"
        pq.write_table(pa.Table.from_pylist(files), inventory_path, compression="zstd")
        final_manifest = {
            **candidate_manifest,
            "created_at": args.created_at,
            "root": str(output.resolve()),
            "repository_id": args.repo_id,
            "private_only": False,
            "publication_ready": True,
            "files": files,
            "inventory": _receipt(inventory_path, root=temporary, rows=431),
            "decision_ledger": _receipt(
                decision_destination,
                root=temporary,
                rows=int(candidate_manifest["decision_ledger"]["rows"]),
            ),
            "bibliography_cleaning": {
                **candidate_manifest["bibliography_cleaning"],
                "reconstruction": _receipt(reconstruction_destination, root=temporary),
                "apply_summary": {
                    "path": str(apply_summary_path),
                    "sha256": sha256_file(apply_summary_path),
                },
                "token_counts": _receipt(token_destination, root=temporary),
                "public_policy": _receipt(policy_destination, root=temporary),
            },
            "token_counts": {
                "tokenizer": token_summary["tokenizer"],
                "base": token_summary["base"],
                "cleaned": token_summary["cleaned"],
                "delta": token_summary["delta"],
            },
        }
        atomic_write_json(
            temporary / "manifests" / "deduplicated_manifest.json", final_manifest
        )
        (temporary / "README.md").write_text(
            _readme(
                manifest=final_manifest,
                token_summary=token_summary,
                cleaning_summary=apply_summary,
                created_at=args.created_at,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"{output}: public release finalized with {len(files)} data shards")
    return final_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--token-summary", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
