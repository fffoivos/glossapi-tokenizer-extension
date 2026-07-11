#!/usr/bin/env python3
"""Materialize a manifest-bound private training and safe public release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from build_token_waterfall import build_waterfall
from finalization_io import (
    atomic_output_path,
    configure_duckdb,
    discover_parquet,
    parquet_file_receipt,
    read_json_object,
    sha256_file,
    sql_path_list,
    sql_string,
    utc_now,
    write_json_atomic,
)
from source_license import load_adjudication as load_license_adjudication


INTEGRITY_CONTRACT_VERSION = "full_cpt_release_integrity_v1"
PUBLIC_METADATA_POLICY_VERSION = "full_cpt_public_metadata_v1"
DECONTAM_POLICY_VERSION = "greekmmlu_decontamination_v1"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# The public corpus is an allowlist.  Newly introduced canonical columns do not
# silently become redistributable.  Raw titles, people, metadata, work IDs and
# storage locations are either omitted or represented by domain-separated
# hashes below.
PUBLIC_SHARED_ALLOWLIST = (
    "source_id",
    "source_dataset",
    "text",
    "greek_badness_score",
    "mojibake_badness_score",
    "needs_ocr",
    "is_empty",
    "ocr_success",
    "is_historical_or_polytonic",
    "source_family_id",
    "acquisition_source_id",
    "source_repo_id",
    "source_revision",
    "source_text_field",
    "stable_uid",
    "work_key",
    "representation_generation",
    "lineage_alias_id",
    "cleaning_profile",
    "structural_policy",
    "training_eligibility",
    "source_role",
    "cleaned_text_sha256",
    "eligible_for_training",
    "eligible_for_redistribution",
)
PUBLIC_HASHED_COLUMNS = {
    "source_doc_id": "source_doc_id_sha256",
    "source_artifact_path": "source_artifact_path_sha256",
    "source_row_id": "source_row_id_sha256",
    "work_id": "work_id_sha256",
}
PUBLIC_ALWAYS_FORBIDDEN = {
    "author",
    "title",
    "source_metadata_json",
    "source_doc_id",
    "source_artifact_path",
    "source_row_id",
    "work_id",
    "original_text_sha256",
    "normalized_text_sha256",
}
DEDUP_COLUMNS = [
    "decision_stage AS dedup_decision_stage",
    "cluster_id AS dedup_cluster_id",
    "kept_doc_key AS dedup_kept_doc_key",
    "exact_strict_version AS dedup_exact_strict_version",
    "exact_relaxed_version AS dedup_exact_relaxed_version",
    "near_norm_version AS dedup_near_norm_version",
    "shingle_version AS dedup_shingle_version",
    "selection_version AS dedup_selection_version",
]


def write_text_atomic(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"immutable dataset card drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_dataset_card(
    *,
    public_sources: list[dict[str, Any]],
    license_payload: Mapping[str, Any],
    redistribution_rows: int,
    structural_applied: bool,
) -> str:
    adjudication = {
        str(row["source_id"]): row
        for row in license_payload.get("sources", [])
        if isinstance(row, Mapping) and row.get("source_id")
    }
    source_ids = [str(row.get("source_id")) for row in public_sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("public source has multiple repository/revision identities")
    lines = [
        "---",
        "license: other",
        "language:",
        "- el",
        "task_categories:",
        "- text-generation",
        "pretty_name: GlossAPI Greek CPT Redistributable Delta v2",
        "---",
        "",
        "# GlossAPI Greek CPT Redistributable Delta v2",
        "",
        "This is the cleaned, GreekMMLU-decontaminated, globally deduplicated "
        "redistributable delta used by the Greek Apertus continued-pretraining pipeline. "
        "It is **not** the full private training corpus: the Nanochat base and every source "
        "without affirmative redistribution evidence are excluded.",
        "",
        f"The validated public subset contains **{redistribution_rows:,} rows**.",
        "",
        "## Sources, licenses, and attribution",
        "",
        "The repository uses `license: other` because its sources have different terms. "
        "Reuse must follow each source's terms; Hugging Face access gating is not a license.",
        "",
        "| Source | Upstream dataset (pinned revision) | Declared terms | Rows | Required conditions |",
        "|---|---|---|---:|---|",
    ]
    for source in sorted(public_sources, key=lambda row: str(row["source_id"])):
        source_id = str(source["source_id"])
        decision = adjudication.get(source_id)
        if decision is None:
            raise ValueError(f"public source lacks license adjudication: {source_id}")
        redistribution = decision.get("redistribution")
        if not isinstance(redistribution, Mapping) or redistribution.get("eligible") is not True:
            raise ValueError(f"dataset card source is not redistribution-eligible: {source_id}")
        repo_id = str(decision["repo_id"])
        revision = str(decision["revision"])
        if str(source.get("repo_id")) != repo_id or str(source.get("revision")) != revision:
            raise ValueError(f"public source provenance differs from adjudication: {source_id}")
        repo_link = f"https://huggingface.co/datasets/{repo_id}/tree/{revision}"
        conditions = ", ".join(
            f"`{item}`" for item in redistribution.get("conditions", [])
        )
        lines.append(
            f"| `{source_id}` | [{repo_id}]({repo_link}) (`{revision}`) | "
            f"`{decision.get('declared_license', 'other')}` | {int(source['rows']):,} | "
            f"{conditions or 'See upstream terms'} |"
        )
    lines.extend(
        [
            "",
            "The pinned technical license evidence and upstream terms links are preserved in "
            "`provenance/source_license_adjudication.json`. EELLAK material is CC BY-SA 4.0; "
            "downstream redistribution or adaptations must preserve its attribution and "
            "ShareAlike obligations. This technical evidence review is not legal advice.",
            "",
            "## Processing",
            "",
            "- Source-specific normalization and quality filtering.",
            "- PII/anonymization policy application with row-level ledgers.",
            "- Conservative GreekMMLU decontamination against a pinned query manifest.",
            "- Exact and MinHash near-deduplication with content-bound decisions.",
            (
                "- Reviewed ToC/bibliography structural deletion was applied."
                if structural_applied
                else "- ToC/bibliography deletion was not applied; the structural gate completed as a no-op."
            ),
            "",
            "Exact manifests, validation receipts, token-loss accounting, source-license "
            "adjudication, and upstream stage bindings are under `provenance/`.",
            "",
            "## Intended use",
            "",
            "This dataset is prepared for Greek language-model research. Inspect the per-source "
            "terms and provenance before redistribution or use in another context.",
            "",
        ]
    )
    return "\n".join(lines)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _resolved_manifest_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {field!r} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"manifest field {field!r} must be an absolute path: {value!r}")
    return path.resolve()


def _require_completed_manifest(path: Path, schema_version: str, *, status: str | None = None) -> dict[str, Any]:
    payload = read_json_object(path)
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"{path}: expected schema_version={schema_version!r}")
    if not isinstance(payload.get("completed_at"), str) or not payload["completed_at"]:
        raise ValueError(f"{path}: completed_at is required")
    if status is not None and payload.get("status") != status:
        raise ValueError(f"{path}: expected status={status!r}, got {payload.get('status')!r}")
    return payload


def _receipt_path(receipt: Mapping[str, Any], *, root: Path | None = None) -> Path:
    value = receipt.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("file receipt path must be non-empty")
    path = Path(value)
    if not path.is_absolute():
        if root is None:
            raise ValueError(f"relative file receipt has no declared root: {value}")
        path = root / path
    return path.resolve()


def _validate_receipt(receipt: Mapping[str, Any], expected_path: Path, *, verify_hash: bool) -> None:
    import pyarrow.parquet as pq

    path = _receipt_path(receipt)
    if path != expected_path.resolve():
        raise ValueError(f"manifest file receipt points to {path}, expected {expected_path.resolve()}")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = int(receipt.get("bytes", -1))
    if expected_bytes < 0 or path.stat().st_size != expected_bytes:
        raise ValueError(f"file size differs from manifest receipt: {path}")
    expected_sha = str(receipt.get("sha256", ""))
    if not HEX_SHA256.fullmatch(expected_sha):
        raise ValueError(f"manifest receipt has invalid sha256: {path}")
    if verify_hash and sha256_file(path) != expected_sha:
        raise ValueError(f"file checksum differs from manifest receipt: {path}")
    metadata = pq.ParquetFile(path).metadata
    if metadata.num_rows != int(receipt.get("rows", -1)):
        raise ValueError(f"file row count differs from manifest receipt: {path}")
    if metadata.num_row_groups != int(receipt.get("row_groups", -1)):
        raise ValueError(f"file row-group count differs from manifest receipt: {path}")


def _validate_decontamination_policy(payload: Mapping[str, Any]) -> None:
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("decontamination manifest lacks a policy object")
    if policy.get("policy_version") != DECONTAM_POLICY_VERSION:
        raise ValueError(
            "decontamination policy version mismatch: "
            f"expected {DECONTAM_POLICY_VERSION!r}, got {policy.get('policy_version')!r}"
        )
    if policy.get("normalization") != "NFKC+strip_combining_marks+casefold+unicode_word_tokens_v1":
        raise ValueError("decontamination normalization policy drift")
    numeric_bounds = {
        "k": (8, None),
        "min_coverage": (0.85, 1.0),
        "minhash_threshold": (0.85, 1.0),
        "min_matched_grams": (4, None),
        "max_gap_tokens": (0, 40),
    }
    for name, (lower, upper) in numeric_bounds.items():
        value = policy.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"decontamination policy {name!r} is not numeric")
        if float(value) < lower or (upper is not None and float(value) > upper):
            raise ValueError(f"decontamination policy {name!r} is outside the production bound")
    for name in ("k", "min_matched_grams", "max_gap_tokens"):
        if not isinstance(policy.get(name), int) or isinstance(policy.get(name), bool):
            raise ValueError(f"decontamination policy {name!r} must be an integer")
    if policy.get("minhash_permutations") != 64:
        raise ValueError("decontamination MinHash permutation identity mismatch")
    expected_rules = {
        "greekmmlu_exact_prompt",
        "greekmmlu_exact_question_answer",
        "greekmmlu_ngram_minhash_answer",
    }
    if set(policy.get("drop_rules", [])) != expected_rules:
        raise ValueError("decontamination drop-rule identity mismatch")
    if policy.get("answer_only_action") != "audit_only":
        raise ValueError("answer-only GreekMMLU matches must remain audit-only")


def verify_upstream_manifests(
    *,
    cleaning_manifest_path: Path,
    decontamination_manifest_path: Path,
    dedup_manifest_path: Path,
    input_root: Path,
    dedup_decisions: Path,
    verify_decisions_hash: bool = True,
) -> dict[str, dict[str, Any]]:
    """Validate the immutable stage chain and return hash-bound receipts."""

    cleaning = _require_completed_manifest(
        cleaning_manifest_path, "full_cpt_cleaning_manifest_v1", status="completed"
    )
    decontamination = _require_completed_manifest(
        decontamination_manifest_path,
        "full_cpt_greekmmlu_decontamination_v1",
        status="completed",
    )
    dedup = _require_completed_manifest(
        dedup_manifest_path, "full_cpt_dedup_wrapper_manifest_v1", status="completed"
    )
    cleaning_output = _resolved_manifest_path(cleaning.get("output"), field="cleaning.output")
    decontam_input = _resolved_manifest_path(decontamination.get("input"), field="decontamination.input")
    decontam_output = _resolved_manifest_path(decontamination.get("output"), field="decontamination.output")
    dedup_input = _resolved_manifest_path(dedup.get("input"), field="dedup.input")
    if cleaning_output != decontam_input:
        raise ValueError("cleaning/decontamination path identity mismatch")
    if decontam_output != input_root.resolve() or dedup_input != input_root.resolve():
        raise ValueError("decontamination/dedup input identity differs from release input")

    declared_outputs: dict[Path, Mapping[str, Any]] = {}
    file_rows = decontamination.get("files")
    if not isinstance(file_rows, list) or not file_rows:
        raise ValueError("decontamination manifest has no output inventory")
    for row in file_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("output"), Mapping):
            raise ValueError("decontamination manifest output receipt is malformed")
        receipt = row["output"]
        path = _receipt_path(receipt, root=decontam_output)
        if path in declared_outputs:
            raise ValueError(f"duplicate decontamination output receipt: {path}")
        declared_outputs[path] = receipt
        absolute_receipt = {**receipt, "path": str(path)}
        # The upstream stage receipt already performed the full SHA pass.  Here
        # we recheck exact paths, bytes and Parquet accounting before scanning
        # every row and recomputing its cleaned-text hash below.
        _validate_receipt(absolute_receipt, path, verify_hash=False)
    actual_outputs = set(discover_parquet(decontam_output))
    if actual_outputs != set(declared_outputs):
        raise ValueError("decontamination output inventory differs from its completed manifest")
    declared_kept = sum(int(receipt.get("rows", -1)) for receipt in declared_outputs.values())
    if declared_kept != int(decontamination.get("counts", {}).get("kept", -1)):
        raise ValueError("decontamination kept-row accounting differs from its output inventory")

    cleaning_artifacts = {
        "tokenizer_json": "tokenizer_sha256",
        "source_admission": "source_admission_sha256",
        "source_config": "source_config_sha256",
        "license_adjudication": "license_adjudication_sha256",
        "eligibility_policy": "eligibility_policy_sha256",
        "cleaning_policy": "cleaning_policy_sha256",
    }
    for path_field, sha_field in cleaning_artifacts.items():
        path = _resolved_manifest_path(cleaning.get(path_field), field=f"cleaning.{path_field}")
        expected_sha = str(cleaning.get(sha_field, ""))
        if not HEX_SHA256.fullmatch(expected_sha):
            raise ValueError(f"cleaning manifest does not bind a valid {sha_field}")
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"cleaning policy artifact checksum drift: {path}")
    license_path = _resolved_manifest_path(
        cleaning.get("license_adjudication"), field="cleaning.license_adjudication"
    )
    source_config_path = _resolved_manifest_path(
        cleaning.get("source_config"), field="cleaning.source_config"
    )
    load_license_adjudication(license_path, source_registry_path=source_config_path)
    if not HEX_SHA256.fullmatch(str(cleaning.get("config_sha256", ""))):
        raise ValueError("cleaning manifest lacks a valid combined config_sha256")
    if cleaning.get("cleaning_pass") not in {"post_source_post_pii", "structural_last_final"}:
        raise ValueError("cleaning manifest has an unsupported cleaning_pass")
    if not isinstance(cleaning.get("structural_applied"), bool):
        raise ValueError("cleaning manifest must record structural_applied as a boolean")
    if cleaning.get("cleaning_pass") == "post_source_post_pii" and cleaning.get("structural_applied"):
        raise ValueError("cleaning_pass and structural_applied policy identity mismatch")
    if cleaning.get("cleaning_pass") == "structural_last_final":
        for path_field, sha_field in (
            ("stage50_cleaning_manifest", "stage50_cleaning_manifest_sha256"),
            ("structural_decision", "structural_decision_sha256"),
        ):
            path = _resolved_manifest_path(cleaning.get(path_field), field=f"cleaning.{path_field}")
            expected_sha = str(cleaning.get(sha_field, ""))
            if not HEX_SHA256.fullmatch(expected_sha) or not path.is_file() or sha256_file(path) != expected_sha:
                raise ValueError(f"final cleaning structural-chain checksum drift: {path}")
        decision = read_json_object(Path(str(cleaning["structural_decision"])))
        if decision.get("schema_version") != "full_cpt_structural_application_decision_v1":
            raise ValueError("final cleaning structural decision schema mismatch")
        if decision.get("stage50_cleaning_manifest_sha256") != cleaning.get(
            "stage50_cleaning_manifest_sha256"
        ):
            raise ValueError("final cleaning structural decision has a different Stage50 parent")
        if decision.get("cleaning_policy_sha256") != cleaning.get("cleaning_policy_sha256"):
            raise ValueError("final cleaning structural decision has a different cleaning policy")
        if decision.get("apply_structural") is not cleaning.get("structural_applied"):
            raise ValueError("final cleaning structural decision/action mismatch")
        expected_status = "passed" if cleaning["structural_applied"] else "no_op"
        if decision.get("status") != expected_status:
            raise ValueError("final cleaning structural decision status mismatch")
    _validate_decontamination_policy(decontamination)

    identity = dedup.get("identity_contract")
    if not isinstance(identity, Mapping):
        raise ValueError("dedup manifest lacks identity_contract")
    expected_identity = {
        "dedup_source_dataset": "source_dataset (unchanged)",
        "dedup_source_doc_id": "stable_uid",
        "upstream_source_doc_id": "source_doc_id before staging",
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"dedup identity contract mismatch for {key!r}")
    recipe = dedup.get("recipe")
    if not isinstance(recipe, Mapping) or recipe.get("id") != "greek_cpt_text_dedup_v1":
        raise ValueError("dedup manifest does not use the approved production recipe")
    if recipe.get("mode") != "production":
        raise ValueError("experimental dedup manifests cannot be materialized as a release")
    output = dedup.get("dedup_output")
    if not isinstance(output, Mapping) or not isinstance(output.get("content_bound_decisions"), Mapping):
        raise ValueError("completed dedup manifest lacks dedup_output.content_bound_decisions receipt")
    bound_receipt = output["content_bound_decisions"]
    if bound_receipt.get("schema_version") != "full_cpt_dedup_decisions_content_bound_v1":
        raise ValueError("dedup content-bound decision receipt schema mismatch")
    _validate_receipt(bound_receipt, dedup_decisions, verify_hash=verify_decisions_hash)
    alias = output.get("decisions")
    if not isinstance(alias, Mapping):
        raise ValueError("completed dedup manifest lacks the content-bound decisions alias")
    for field in ("path", "sha256", "bytes", "rows", "row_groups"):
        if alias.get(field) != bound_receipt.get(field):
            raise ValueError(f"dedup decisions alias differs from content-bound receipt field {field!r}")
    content_binding = output.get("content_binding")
    expected_binding = {
        "schema_version": "full_cpt_dedup_decisions_content_bound_v1",
        "stable_uid_column": "stable_uid",
        "input_text_sha256_column": "input_text_sha256",
    }
    if not isinstance(content_binding, Mapping) or any(
        content_binding.get(key) != value for key, value in expected_binding.items()
    ):
        raise ValueError("dedup manifest content-binding identity mismatch")

    return {
        "cleaning": {
            "path": str(cleaning_manifest_path.resolve()),
            "sha256": sha256_file(cleaning_manifest_path),
            "schema_version": cleaning["schema_version"],
            "completed_at": cleaning["completed_at"],
        },
        "decontamination": {
            "path": str(decontamination_manifest_path.resolve()),
            "sha256": sha256_file(decontamination_manifest_path),
            "schema_version": decontamination["schema_version"],
            "completed_at": decontamination["completed_at"],
        },
        "dedup": {
            "path": str(dedup_manifest_path.resolve()),
            "sha256": sha256_file(dedup_manifest_path),
            "schema_version": dedup["schema_version"],
            "completed_at": dedup["completed_at"],
        },
    }


def validate_inputs(connection: Any) -> dict[str, int]:
    result = connection.execute(
        """
        SELECT
          (SELECT count(*) FROM corpus WHERE eligible_for_training) AS eligible_rows,
          (SELECT count(*) - count(DISTINCT stable_uid) FROM corpus WHERE eligible_for_training)
            AS duplicate_stable_uid,
          (SELECT count(*) FROM decisions) AS decision_rows,
          (SELECT count(*) - count(DISTINCT (source_dataset, source_doc_id)) FROM decisions)
            AS duplicate_decision_identity,
          (SELECT count(*) FROM decisions WHERE decision NOT IN ('keep', 'drop')) AS bad_decisions,
          (SELECT count(*) FROM decisions WHERE source_dataset IS NULL OR trim(source_dataset) = ''
              OR source_doc_id IS NULL OR trim(source_doc_id) = ''
              OR stable_uid IS NULL OR trim(stable_uid) = '' OR stable_uid <> source_doc_id
              OR input_text_sha256 IS NULL OR trim(input_text_sha256) = '') AS missing_decision_identity,
          (SELECT count(*) FROM corpus c LEFT JOIN decisions d
             ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset
             WHERE c.eligible_for_training AND d.source_doc_id IS NULL) AS eligible_without_decision,
          (SELECT count(*) FROM decisions d LEFT JOIN corpus c
             ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset
             AND c.eligible_for_training
             WHERE c.stable_uid IS NULL) AS decision_without_eligible_input,
          (SELECT count(*) FROM decisions d JOIN corpus c
             ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset
             WHERE d.input_text_sha256 <> c.cleaned_text_sha256) AS decision_text_hash_mismatch,
          (SELECT count(*) FROM corpus
             WHERE eligible_for_training AND cleaned_text_sha256 <> sha256(text)) AS corpus_text_hash_mismatch,
          (SELECT count(*) FROM decisions WHERE decision = 'keep') AS expected_training_rows,
          (SELECT count(*) FROM decisions d JOIN corpus c
             ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset
             WHERE d.decision = 'keep' AND c.eligible_for_redistribution) AS expected_redistribution_rows
        """
    ).fetchone()
    names = [
        "eligible_rows",
        "duplicate_stable_uid",
        "decision_rows",
        "duplicate_decision_identity",
        "bad_decisions",
        "missing_decision_identity",
        "eligible_without_decision",
        "decision_without_eligible_input",
        "decision_text_hash_mismatch",
        "corpus_text_hash_mismatch",
        "expected_training_rows",
        "expected_redistribution_rows",
    ]
    payload = {name: int(value) for name, value in zip(names, result, strict=True)}
    if payload["eligible_rows"] != payload["decision_rows"]:
        raise ValueError(f"eligible input and dedup decision counts differ: {payload}")
    failures = (
        "duplicate_stable_uid",
        "duplicate_decision_identity",
        "bad_decisions",
        "missing_decision_identity",
        "eligible_without_decision",
        "decision_without_eligible_input",
        "decision_text_hash_mismatch",
        "corpus_text_hash_mismatch",
    )
    if any(payload[key] for key in failures):
        raise ValueError(f"release materialization input gate failed: {payload}")
    return payload


def _domain_hash_expression(column: str, output: str) -> str:
    source = f"r.{quote_identifier(column)}"
    domain = f"{PUBLIC_METADATA_POLICY_VERSION}:{column}:"
    return (
        f"CASE WHEN {source} IS NULL OR trim(CAST({source} AS VARCHAR)) = '' THEN NULL "
        f"ELSE sha256({sql_string(domain)} || CAST({source} AS VARCHAR)) END "
        f"AS {quote_identifier(output)}"
    )


def copy_partition(
    connection: Any,
    *,
    input_path: Path,
    output_path: Path,
    columns: list[str],
    hashed_columns: Mapping[str, str] | None = None,
    redistribution: bool,
) -> dict[str, Any]:
    selected = [f"r.{quote_identifier(column)}" for column in columns]
    if hashed_columns:
        selected.extend(_domain_hash_expression(source, output) for source, output in hashed_columns.items())
    eligibility = "AND r.eligible_for_redistribution" if redistribution else ""
    query = (
        f"SELECT {', '.join(selected)} FROM release_rows r "
        f"WHERE r._input_path = {sql_string(input_path.resolve())} {eligibility} "
        "ORDER BY r.stable_uid"
    )
    temporary = atomic_output_path(output_path)
    connection.execute(
        f"COPY ({query}) TO {sql_string(temporary.resolve())} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    os.replace(temporary, output_path)
    return parquet_file_receipt(output_path)


def _checkpoint_path(output_root: Path, relative: Path) -> Path:
    return output_root / ".materialization-checkpoints" / relative.parent / f"{relative.name}.json"


def _checkpoint_contract(
    *,
    relative: Path,
    input_receipt: Mapping[str, Any],
    upstream: Mapping[str, Mapping[str, Any]],
    decisions_sha256: str,
    training_columns: list[str],
    public_columns: list[str],
) -> str:
    value = {
        "version": "full_cpt_materialization_checkpoint_v1",
        "relative_path": relative.as_posix(),
        "input_sha256": input_receipt.get("sha256"),
        "input_bytes": input_receipt.get("bytes"),
        "input_rows": input_receipt.get("rows"),
        "upstream_manifest_sha256": {key: row["sha256"] for key, row in sorted(upstream.items())},
        "dedup_decisions_sha256": decisions_sha256,
        "training_columns": training_columns,
        "public_columns": public_columns,
        "public_metadata_policy_version": PUBLIC_METADATA_POLICY_VERSION,
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_materialized_receipt(receipt: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    relative = Path(str(receipt.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".parquet":
        raise ValueError(f"unsafe materialization checkpoint path: {relative}")
    path = output_root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"checkpointed release file is missing or a symlink: {path}")
    actual = {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "row_groups": pq.ParquetFile(path).metadata.num_row_groups,
    }
    for field in ("sha256", "bytes", "rows", "row_groups"):
        if actual[field] != receipt.get(field):
            raise ValueError(f"checkpointed release file drift for {path}: {field}")
    return actual


def _resume_partition(
    *,
    checkpoint: Path,
    expected_contract: str,
    actual_input_sha256: str,
    input_receipt: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not checkpoint.exists():
        return None
    value = read_json_object(checkpoint)
    if value.get("schema_version") != "full_cpt_materialization_checkpoint_v1":
        raise ValueError(f"unsupported materialization checkpoint: {checkpoint}")
    if value.get("contract_sha256") != expected_contract:
        raise ValueError(f"materialization checkpoint contract drift: {checkpoint}")
    expected_input_sha = str(input_receipt.get("sha256", ""))
    if value.get("input_sha256") != expected_input_sha or actual_input_sha256 != expected_input_sha:
        raise ValueError("checkpointed materialization input checksum drift")
    training = _validate_materialized_receipt(value.get("training", {}), output_root=output_root)
    redistribution = _validate_materialized_receipt(
        value.get("redistribution", {}), output_root=output_root
    )
    return training, redistribution


def _remove_uncheckpointed_outputs(*paths: Path) -> None:
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"refusing to replace a symlinked generated output: {path}")
        path.unlink(missing_ok=True)
        path.with_name(f".{path.name}.partial").unlink(missing_ok=True)


def _validate_resume_inventory(output_root: Path, expected_files: set[Path], checkpoints: set[Path]) -> None:
    allowed = {path.resolve() for path in expected_files | checkpoints}
    allowed.update(path.with_name(f".{path.name}.partial").resolve() for path in expected_files)
    checkpoint_root = output_root / ".materialization-checkpoints"
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"release resume tree contains a symlink: {path}")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in allowed:
            continue
        # Atomic JSON writers may leave only a random-named .partial beneath
        # the private checkpoint root.  It was never a reusable receipt.
        if checkpoint_root in path.parents and path.name.startswith(".") and path.suffix == ".partial":
            path.unlink()
            continue
        raise ValueError(f"release resume tree contains an unknown file: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Decontaminated canonical Parquet root")
    parser.add_argument("--cleaning-manifest", type=Path, required=True)
    parser.add_argument("--decontamination-manifest", type=Path, required=True)
    parser.add_argument("--dedup-manifest", type=Path, required=True)
    parser.add_argument("--dedup-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--token-waterfall", type=Path, required=True)
    parser.add_argument("--cleaning-ledger", type=Path)
    parser.add_argument("--decontam-ledger", type=Path)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", default="200GB")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Reuse only checksum-valid shard checkpoints")
    return parser.parse_args()


def main() -> int:
    import duckdb
    import pyarrow.parquet as pq

    args = parse_args()
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.manifest}")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to materialize into non-empty output root: {args.output}")
    if not args.dedup_decisions.is_file():
        raise FileNotFoundError(args.dedup_decisions)
    upstream = verify_upstream_manifests(
        cleaning_manifest_path=args.cleaning_manifest,
        decontamination_manifest_path=args.decontamination_manifest,
        dedup_manifest_path=args.dedup_manifest,
        input_root=args.input,
        dedup_decisions=args.dedup_decisions,
    )
    cleaning_payload = read_json_object(args.cleaning_manifest)
    license_path = _resolved_manifest_path(
        cleaning_payload.get("license_adjudication"),
        field="cleaning.license_adjudication",
    )
    license_payload = read_json_object(license_path)
    input_files = discover_parquet(args.input)
    first_schema = pq.ParquetFile(input_files[0]).schema_arrow
    required = {
        "source_dataset",
        "source_doc_id",
        "text",
        "stable_uid",
        "cleaned_text_sha256",
        "acquisition_source_id",
        "source_repo_id",
        "source_revision",
        "source_artifact_path",
        "source_row_id",
        "eligible_for_training",
        "eligible_for_redistribution",
    }
    missing = required - set(first_schema.names)
    if missing:
        raise ValueError(f"release input schema misses required columns: {sorted(missing)}")
    dedup_output_columns = {expression.split(" AS ", 1)[1] for expression in DEDUP_COLUMNS}
    collisions = (dedup_output_columns | set(PUBLIC_HASHED_COLUMNS.values())) & set(first_schema.names)
    if collisions:
        raise ValueError(f"release input already contains reserved release columns: {sorted(collisions)}")
    for path in input_files[1:]:
        if pq.ParquetFile(path).schema_arrow != first_schema:
            raise ValueError(f"canonical input schema drift: {path}")

    if not args.token_waterfall.exists():
        if args.cleaning_ledger is None or args.decontam_ledger is None:
            raise FileNotFoundError(
                f"{args.token_waterfall} is missing; provide --cleaning-ledger and --decontam-ledger to build it"
            )
        build_waterfall(
            cleaning_ledger=args.cleaning_ledger,
            decontam_ledger=args.decontam_ledger,
            dedup_decisions=args.dedup_decisions,
            output=args.token_waterfall,
            temporary_directory=args.temporary_directory / "waterfall",
            memory_limit=args.memory_limit,
            threads=args.threads,
            cleaning_manifest=args.cleaning_manifest,
            decontamination_manifest=args.decontamination_manifest,
            dedup_manifest=args.dedup_manifest,
        )
    waterfall = read_json_object(args.token_waterfall)
    if waterfall.get("schema_version") != "full_cpt_token_waterfall_v1":
        raise ValueError("--token-waterfall has an unsupported schema")
    if not bool(waterfall.get("invariants", {}).get("reconciled")):
        raise ValueError("--token-waterfall is not reconciled")
    expected_waterfall_inputs = {
        "dedup_decisions_sha256": sha256_file(args.dedup_decisions),
        "cleaning_manifest_sha256": upstream["cleaning"]["sha256"],
        "decontamination_manifest_sha256": upstream["decontamination"]["sha256"],
        "dedup_manifest_sha256": upstream["dedup"]["sha256"],
    }
    waterfall_inputs = waterfall.get("inputs", {})
    for key, expected in expected_waterfall_inputs.items():
        if waterfall_inputs.get(key) != expected:
            raise ValueError(f"--token-waterfall {key} binding mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    training_root = args.output / "training" / "data"
    redistribution_root = args.output / "redistribution" / "data"
    training_columns = [*first_schema.names, *sorted(dedup_output_columns)]
    public_shared = [name for name in PUBLIC_SHARED_ALLOWLIST if name in first_schema.names]
    public_shared.extend(sorted(dedup_output_columns))
    public_hashed = {
        source: output for source, output in PUBLIC_HASHED_COLUMNS.items() if source in first_schema.names
    }
    public_columns = [*public_shared, *public_hashed.values()]
    dropped_columns = sorted(set(training_columns) - set(public_shared) - set(public_hashed))

    decontamination_run = read_json_object(args.decontamination_manifest)
    input_receipts: dict[str, Mapping[str, Any]] = {}
    for row in decontamination_run["files"]:
        receipt = row["output"]
        relative = Path(str(receipt["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe decontamination output receipt path: {relative}")
        input_receipts[relative.as_posix()] = receipt
    if set(input_receipts) != {path.relative_to(args.input).as_posix() for path in input_files}:
        raise ValueError("decontamination receipt paths do not match materialization inputs")
    decisions_sha256 = sha256_file(args.dedup_decisions)
    expected_output_files = {
        *(training_root / path.relative_to(args.input) for path in input_files),
        *(redistribution_root / path.relative_to(args.input) for path in input_files),
        args.output / "publication" / "README.md",
    }
    expected_checkpoints = {
        _checkpoint_path(args.output, path.relative_to(args.input)) for path in input_files
    }
    if args.resume:
        _validate_resume_inventory(args.output, expected_output_files, expected_checkpoints)

    connection = duckdb.connect()
    configure_duckdb(
        connection,
        temporary_directory=args.temporary_directory / "materialize",
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    try:
        connection.execute(
            f"CREATE TEMP VIEW corpus AS SELECT * FROM read_parquet({sql_path_list(input_files)}, "
            "union_by_name=true)"
        )
        # Decisions are copied once and indexed.  The corpus/decision join is
        # then materialized once with its input filename, avoiding the former
        # full decisions scan for every output shard.
        connection.execute(
            f"CREATE TEMP TABLE decisions AS SELECT * FROM read_parquet({sql_string(args.dedup_decisions.resolve())})"
        )
        connection.execute("CREATE INDEX decisions_identity ON decisions(source_dataset, source_doc_id)")
        gates = validate_inputs(connection)
        corpus_select = ", ".join(f"c.{quote_identifier(name)}" for name in first_schema.names)
        dedup_select = ", ".join(f"d.{expression}" for expression in DEDUP_COLUMNS)
        connection.execute(
            f"CREATE TEMP TABLE release_rows AS SELECT {corpus_select}, {dedup_select}, "
            "c.filename AS _input_path "
            f"FROM read_parquet({sql_path_list(input_files)}, union_by_name=true, filename=true) c "
            "JOIN decisions d ON d.source_doc_id = c.stable_uid AND d.source_dataset = c.source_dataset "
            "WHERE d.decision = 'keep' AND c.eligible_for_training"
        )
        connection.execute("CREATE INDEX release_rows_input ON release_rows(_input_path)")
        files: list[dict[str, Any]] = []
        totals: dict[str, int] = defaultdict(int)
        for input_path in input_files:
            relative = input_path.relative_to(args.input)
            actual_input_sha256 = sha256_file(input_path)
            if actual_input_sha256 != input_receipts[relative.as_posix()].get("sha256"):
                raise ValueError(f"decontamination input shard checksum drift: {input_path}")
            training_path = training_root / relative
            redistribution_path = redistribution_root / relative
            checkpoint = _checkpoint_path(args.output, relative)
            checkpoint_contract = _checkpoint_contract(
                relative=relative,
                input_receipt=input_receipts[relative.as_posix()],
                upstream=upstream,
                decisions_sha256=decisions_sha256,
                training_columns=training_columns,
                public_columns=public_columns,
            )
            resumed = _resume_partition(
                checkpoint=checkpoint,
                expected_contract=checkpoint_contract,
                actual_input_sha256=actual_input_sha256,
                input_receipt=input_receipts[relative.as_posix()],
                output_root=args.output,
            )
            if resumed is None:
                _remove_uncheckpointed_outputs(training_path, redistribution_path)
                training_receipt = copy_partition(
                    connection,
                    input_path=input_path,
                    output_path=training_path,
                    columns=training_columns,
                    redistribution=False,
                )
                redistribution_receipt = copy_partition(
                    connection,
                    input_path=input_path,
                    output_path=redistribution_path,
                    columns=public_shared,
                    hashed_columns=public_hashed,
                    redistribution=True,
                )
                training_receipt["path"] = str(training_path.relative_to(args.output))
                redistribution_receipt["path"] = str(redistribution_path.relative_to(args.output))
                write_json_atomic(
                    checkpoint,
                    {
                        "schema_version": "full_cpt_materialization_checkpoint_v1",
                        "completed_at": utc_now(),
                        "contract_sha256": checkpoint_contract,
                        "input_path": str(input_path.resolve()),
                        "input_sha256": input_receipts[relative.as_posix()]["sha256"],
                        "training": training_receipt,
                        "redistribution": redistribution_receipt,
                    },
                )
            else:
                training_receipt, redistribution_receipt = resumed
            files.append(
                {
                    "input": str(input_path.resolve()),
                    "training": training_receipt,
                    "redistribution": redistribution_receipt,
                }
            )
            totals["training_rows"] += int(training_receipt["rows"])
            totals["redistribution_rows"] += int(redistribution_receipt["rows"])
        public_sources = [
            {
                "source_id": str(row[0]),
                "repo_id": str(row[1]),
                "revision": str(row[2]),
                "rows": int(row[3]),
            }
            for row in connection.execute(
                """
                SELECT acquisition_source_id, source_repo_id, source_revision, count(*)
                FROM release_rows
                WHERE eligible_for_redistribution
                GROUP BY acquisition_source_id, source_repo_id, source_revision
                ORDER BY acquisition_source_id, source_repo_id, source_revision
                """
            ).fetchall()
        ]
    finally:
        connection.close()
    if totals["training_rows"] != gates["expected_training_rows"]:
        raise RuntimeError(f"materialized training rows do not match kept decisions: {dict(totals)} vs {gates}")
    if totals["redistribution_rows"] != gates["expected_redistribution_rows"]:
        raise RuntimeError(
            f"materialized redistribution rows do not match eligible kept inputs: {dict(totals)} vs {gates}"
        )
    dataset_card_path = args.output / "publication" / "README.md"
    dataset_card = build_dataset_card(
        public_sources=public_sources,
        license_payload=license_payload,
        redistribution_rows=int(totals["redistribution_rows"]),
        structural_applied=bool(cleaning_payload.get("structural_applied", False)),
    )
    write_text_atomic(dataset_card_path, dataset_card)
    dataset_card_receipt = {
        "path": str(dataset_card_path.relative_to(args.output)),
        "remote_path": "README.md",
        "sha256": sha256_file(dataset_card_path),
        "bytes": dataset_card_path.stat().st_size,
    }
    payload = {
        "schema_version": "full_cpt_release_manifest_v1",
        "integrity_contract_version": INTEGRITY_CONTRACT_VERSION,
        "completed_at": utc_now(),
        "input": str(args.input.resolve()),
        "upstream_manifests": upstream,
        "dedup_decisions": str(args.dedup_decisions.resolve()),
        "dedup_decisions_sha256": sha256_file(args.dedup_decisions),
        "token_waterfall": str(args.token_waterfall.resolve()),
        "token_waterfall_sha256": sha256_file(args.token_waterfall),
        "source_license_adjudication": {
            "path": str(license_path),
            "sha256": sha256_file(license_path),
            "schema_version": str(license_payload["schema_version"]),
            "status": str(license_payload["status"]),
            "audited_at": str(license_payload["audited_at"]),
        },
        "output": str(args.output.resolve()),
        "training_root": "training/data",
        "redistribution_root": "redistribution/data",
        "dataset_card": dataset_card_receipt,
        "redistribution_sources": public_sources,
        "redistribution_policy": {
            "policy_version": PUBLIC_METADATA_POLICY_VERSION,
            "mode": "explicit_allowlist_with_domain_separated_hashes",
            "requires_eligible_for_training": True,
            "requires_eligible_for_redistribution": True,
            "shared_columns": public_shared,
            "hashed_columns": public_hashed,
            "public_columns": public_columns,
            "dropped_columns": dropped_columns,
            "always_forbidden_columns": sorted(PUBLIC_ALWAYS_FORBIDDEN),
        },
        "materialization": {
            "dedup_decisions_materialized_once": True,
            "resume_contract": "full_cpt_materialization_checkpoint_v1",
            "resumed": args.resume,
            "join_identity": ["source_dataset", "stable_uid=source_doc_id"],
            "decision_content_binding": "input_text_sha256=cleaned_text_sha256=sha256(text)",
        },
        "input_gates": gates,
        "counts": dict(totals),
        "files": files,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": dict(totals)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
