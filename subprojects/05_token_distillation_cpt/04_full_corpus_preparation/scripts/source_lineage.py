#!/usr/bin/env python3
"""Deterministic source-lineage primitives for the full CPT corpus.

The functions in this module deliberately operate on canonical row envelopes,
not raw upstream schemas.  Source-specific normalizers must supply stable row and
document identifiers before a row reaches this boundary.  This keeps lineage
identity independent from file order and prevents a reprocessed repository from
being appended merely because it uses a new Hugging Face repository name.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit, urlunsplit


LINEAGE_ROW_SCHEMA = "full_cpt_lineage_row_v1"
REGISTRY_MANIFEST_SCHEMA = "full_cpt_lineage_registry_v1"
IDENTITY_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

BOUND_CANONICAL_STRING_FIELDS = (
    "source_id",
    "source_dataset",
    "source_doc_id",
    "source_family_id",
    "source_repo_id",
    "source_revision",
    "source_artifact_path",
    "source_row_id",
    "source_text_field",
    "original_text_sha256",
    "normalized_text_sha256",
    "stable_uid",
    "work_key",
    "work_id",
    "representation_generation",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_parts(namespace: str, *parts: object) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    for part in parts:
        encoded = canonical_json(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def normalize_identity_text(text: str) -> str:
    """Normalize only representation-neutral whitespace for exact identity.

    NFC preserves distinctions such as Greek compatibility characters that a
    cleaner may need to audit.  No case folding or internal whitespace collapse
    is performed.
    """

    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip(" \t") for line in text.split("\n")).strip()


def normalize_work_identifier(value: object) -> str:
    raw = unicodedata.normalize("NFC", str(value)).strip()
    if not raw:
        raise ValueError("work identifier must not be empty")
    compact = re.sub(r"\s+", " ", raw)
    lowered = compact.lower()
    if lowered.startswith("doi:"):
        return "doi:" + compact[4:].strip().lower()
    if lowered.startswith("https://doi.org/") or lowered.startswith("http://doi.org/"):
        return "doi:" + compact.split("doi.org/", 1)[1].strip().lower()
    if lowered.startswith(("http://", "https://")):
        parsed = urlsplit(compact)
        path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/") or "/"
        return urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, "")
        )
    return compact


def first_appearance(source_dataset: str, roster: Mapping[str, Any]) -> dict[str, Any]:
    repository = roster.get("repository", {})
    for entry in roster.get("sources", []):
        if isinstance(entry, dict) and entry.get("source_dataset") == source_dataset:
            return {
                "cohort": "nanochat_first_data_revision",
                "revision": repository.get("first_data_revision"),
                "committed_at": repository.get("first_data_committed_at"),
                "anchor_revision": repository.get("first_data_revision"),
            }
    for entry in roster.get("later_source_name_additions", []):
        if isinstance(entry, dict) and entry.get("source_dataset") == source_dataset:
            return {
                "cohort": "nanochat_later_source_name_addition",
                "revision": entry.get("first_roster_revision"),
                "committed_at": entry.get("committed_at"),
                "anchor_revision": repository.get("first_data_revision"),
            }
    return {
        "cohort": "not_present_at_nanochat_anchor",
        "revision": None,
        "committed_at": None,
        "anchor_revision": repository.get("first_data_revision"),
    }


def source_indexes(
    sources: Mapping[str, Any],
) -> tuple[dict[str, dict], dict[str, dict]]:
    by_id: dict[str, dict] = {}
    by_repo: dict[str, dict] = {}
    for source in sources.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_id = source.get("source_id")
        repo_id = source.get("repo_id")
        if isinstance(source_id, str):
            by_id[source_id] = source
        if isinstance(repo_id, str):
            by_repo[repo_id] = source
    return by_id, by_repo


def aliases_for_repo(repo_id: str, aliases: Mapping[str, Any]) -> list[dict]:
    return sorted(
        [
            entry
            for entry in aliases.get("aliases", [])
            if isinstance(entry, dict) and entry.get("current_repo_id") == repo_id
        ],
        key=lambda entry: str(entry.get("alias_id", "")),
    )


def choose_alias_id(
    row: Mapping[str, Any],
    repo_id: str,
    source_dataset: str,
    aliases: Mapping[str, Any],
) -> str | None:
    candidates = aliases_for_repo(repo_id, aliases)
    requested = row.get("lineage_alias_id")
    if requested is not None:
        if not isinstance(requested, str) or requested not in {
            str(entry.get("alias_id")) for entry in candidates
        }:
            raise ValueError(f"invalid lineage_alias_id {requested!r} for {repo_id}")
        return requested
    matching = [
        entry
        for entry in candidates
        if source_dataset in entry.get("initial_source_datasets", [])
    ]
    if len(matching) == 1:
        return str(matching[0]["alias_id"])
    if len(candidates) == 1:
        return str(candidates[0]["alias_id"])
    return None


def representation_generation(origin: str, source: Mapping[str, Any]) -> str:
    if origin == "base":
        return "nanochat_pinned_release"
    relation = str(source.get("content_relation", ""))
    if relation == "new_family":
        return "candidate_first_representation"
    if relation == "same_source_replacement":
        return "candidate_reprocessed"
    if relation == "same_source_resegmentation":
        return "candidate_sectioned"
    if relation == "same_source_replacement_resegmentation":
        return "candidate_reprocessed_sectioned"
    if relation.startswith("hybrid"):
        return "candidate_hybrid"
    if "internal_repo_overlap" in relation:
        return "candidate_parallel_repository"
    if "base_overlap" in relation:
        return "candidate_independent_overlap_risk"
    return "candidate_unspecified_representation"


def lineage_class(
    origin: str,
    source: Mapping[str, Any],
    alias_id: str | None,
    aliases: Mapping[str, Any],
) -> str:
    if origin == "base":
        return "nanochat_base"
    if alias_id:
        for entry in aliases.get("aliases", []):
            if isinstance(entry, dict) and entry.get("alias_id") == alias_id:
                return f"reviewed_{entry.get('alias_kind')}_lineage"
    relation = str(source.get("content_relation", ""))
    if relation == "new_family":
        return "new_source_family_candidate"
    if "internal_repo_overlap" in relation:
        return "candidate_family_internal_overlap"
    if "base_overlap" in relation:
        return "candidate_possible_base_overlap"
    if relation.startswith("same_source"):
        return "same_source_alternate_representation"
    if relation.startswith("hybrid"):
        return "hybrid_alternate_representation"
    return "candidate_requires_lineage_review"


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None or not str(value):
        raise ValueError(f"canonical row requires non-empty {field}")
    return str(value)


def canonicalize_row(
    row: Mapping[str, Any],
    *,
    origin: str,
    sources: Mapping[str, Any],
    roster: Mapping[str, Any],
    aliases: Mapping[str, Any],
    canonical_bound: bool = False,
    verify_bound: bool = False,
    source_by_id: Mapping[str, dict[str, Any]] | None = None,
    first_appearance_cache: MutableMapping[str, dict[str, Any]] | None = None,
    alias_cache: MutableMapping[tuple[str, str, str], str | None] | None = None,
) -> dict[str, Any]:
    """Create a stable, source-preserving row lineage record.

    Required canonical-envelope fields are ``text``, ``source_artifact_path``,
    ``source_row_id`` and ``source_doc_id``.  Candidate rows additionally need
    ``source_id``.  Sectioned/resegmented candidates must provide ``work_id`` so
    a section ID is never mistaken for a work-level identity.
    """

    if origin not in {"base", "candidate"}:
        raise ValueError("origin must be base or candidate")
    text = row.get("text")
    if not isinstance(text, str) and not (
        canonical_bound and origin == "base" and text is None
    ):
        raise ValueError("canonical row requires string text")
    by_id = source_by_id if source_by_id is not None else source_indexes(sources)[0]
    if origin == "base":
        source: dict[str, Any] = dict(sources.get("base", {}))
        source["source_id"] = "nanochat_base"
        source["source_family_id"] = str(
            row.get("source_family_id") or row.get("source_dataset") or ""
        )
        source["content_relation"] = "base"
        source["role"] = "base"
    else:
        source_id = _required_string(row, "source_id")
        if source_id not in by_id:
            raise ValueError(f"unknown candidate source_id {source_id!r}")
        source = by_id[source_id]

    repo_id = str(source.get("repo_id", ""))
    revision = str(source.get("revision", ""))
    supplied_repo = row.get("source_repo_id")
    supplied_revision = row.get("source_revision")
    if supplied_repo is not None and supplied_repo != repo_id:
        raise ValueError(f"source_repo_id drift: {supplied_repo!r} != {repo_id!r}")
    if supplied_revision is not None and supplied_revision != revision:
        raise ValueError("source_revision does not match the pinned registry")

    upstream_name = row.get("source_dataset")
    if upstream_name is None or upstream_name == "":
        source_dataset = repo_id
        source_dataset_origin = "pinned_repo_fallback"
    elif not isinstance(upstream_name, str):
        raise ValueError("source_dataset must be a string when present")
    else:
        source_dataset = upstream_name
        source_dataset_origin = "preserved_upstream_value"

    source_artifact_path = _required_string(row, "source_artifact_path")
    source_row_id = _required_string(row, "source_row_id")
    source_doc_id = _required_string(row, "source_doc_id")
    source_text_field = str(
        row.get("source_text_field") or (source.get("text_columns") or ["text"])[0]
    )
    relation = str(source.get("content_relation", ""))
    if (
        origin == "candidate"
        and "resegmentation" in relation
        and not row.get("work_id")
    ):
        raise ValueError(
            f"{source.get('source_id')}: resegmented rows require an explicit work_id"
        )
    work_id = normalize_work_identifier(row.get("work_id") or source_doc_id)
    source_family_id = str(
        row.get("source_family_id") or source.get("source_family_id") or source_dataset
    )
    if not source_family_id:
        raise ValueError("source_family_id must not be empty")
    requested_alias = str(row.get("lineage_alias_id") or "")
    alias_key = (repo_id, source_dataset, requested_alias)
    if alias_cache is not None and alias_key in alias_cache:
        alias_id = alias_cache[alias_key]
    else:
        alias_id = choose_alias_id(row, repo_id, source_dataset, aliases)
        if alias_cache is not None:
            alias_cache[alias_key] = alias_id
    generation = str(
        row.get("representation_generation")
        or representation_generation(origin, source)
    )

    if canonical_bound:
        for field in BOUND_CANONICAL_STRING_FIELDS:
            _required_string(row, field)
        for field in (
            "original_text_sha256",
            "normalized_text_sha256",
            "stable_uid",
            "work_key",
        ):
            if not HEX_SHA256_RE.fullmatch(str(row[field])):
                raise ValueError(f"bound canonical row has invalid {field}")
        original_text_sha256 = str(row["original_text_sha256"])
        normalized_text_sha256 = str(row["normalized_text_sha256"])
        stable_uid = str(row["stable_uid"])
        work_key = str(row["work_key"])
        supplied_work_id = str(row["work_id"])
        if supplied_work_id != work_id:
            raise ValueError(
                "bound canonical work_id differs from its canonical envelope"
            )
        if verify_bound:
            expected_uid = sha256_parts(
                "full_cpt_stable_uid_v1",
                repo_id,
                revision,
                source_artifact_path,
                source_row_id,
                source_dataset,
                source_text_field,
            )
            expected_work_key = sha256_parts(
                "full_cpt_work_key_v1", source_family_id, work_id
            )
            if stable_uid != expected_uid:
                raise ValueError("sampled canonical stable_uid invariant failed")
            if work_key != expected_work_key:
                raise ValueError("sampled canonical work_key invariant failed")
            if isinstance(text, str):
                sampled_hash = hashlib.sha256(
                    normalize_identity_text(text).encode("utf-8")
                ).hexdigest()
                if normalized_text_sha256 != sampled_hash:
                    raise ValueError(
                        "sampled canonical normalized text hash invariant failed"
                    )
    else:
        assert isinstance(text, str)
        normalized_text = normalize_identity_text(text)
        original_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        normalized_text_sha256 = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()
        stable_uid = sha256_parts(
            "full_cpt_stable_uid_v1",
            repo_id,
            revision,
            source_artifact_path,
            source_row_id,
            source_dataset,
            source_text_field,
        )
        work_key = sha256_parts("full_cpt_work_key_v1", source_family_id, work_id)
    if first_appearance_cache is not None:
        appearance = first_appearance_cache.get(source_dataset)
        if appearance is None:
            appearance = first_appearance(source_dataset, roster)
            first_appearance_cache[source_dataset] = appearance
    else:
        appearance = first_appearance(source_dataset, roster)
    result = {
        "schema_version": LINEAGE_ROW_SCHEMA,
        "origin": origin,
        "source_id": source.get("source_id"),
        "source_dataset": source_dataset,
        "source_dataset_origin": source_dataset_origin,
        "source_family_id": source_family_id,
        "source_repo_id": repo_id,
        "source_revision": revision,
        "source_artifact_path": source_artifact_path,
        "source_row_id": source_row_id,
        "source_doc_id": source_doc_id,
        "source_text_field": source_text_field,
        "original_text_sha256": original_text_sha256,
        "normalized_text_sha256": normalized_text_sha256,
        # Novelty is only defined for additive candidates. Avoid a second full
        # token scan over the ~60B-token Nanochat base; base rows participate
        # through exact/work hashes and contribute no novelty denominator.
        "identity_word_tokens": (
            sum(1 for _ in IDENTITY_TOKEN_RE.finditer(text))
            if origin == "candidate" and isinstance(text, str)
            else 0
        ),
        "stable_uid": stable_uid,
        "work_key": work_key,
        "work_id_normalized": work_id,
        "representation_generation": generation,
        "representation_key": sha256_parts(
            "full_cpt_representation_key_v1", work_key, generation
        ),
        "lineage_alias_id": alias_id,
        "lineage_class": lineage_class(origin, source, alias_id, aliases),
        "content_relation": source.get("content_relation"),
        "merge_policy": source.get("merge_policy"),
        "first_appearance": appearance,
    }
    return result


def build_registry_manifest(
    sources: Mapping[str, Any], roster: Mapping[str, Any], aliases: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a deterministic route-level manifest before any payload is read."""

    initial_names = sorted(
        str(entry["source_dataset"])
        for entry in roster.get("sources", [])
        if isinstance(entry, dict) and entry.get("source_dataset")
    )
    later_names = sorted(
        str(entry["source_dataset"])
        for entry in roster.get("later_source_name_additions", [])
        if isinstance(entry, dict) and entry.get("source_dataset")
    )
    candidates: list[dict[str, Any]] = []
    for source in sorted(
        [entry for entry in sources.get("sources", []) if isinstance(entry, dict)],
        key=lambda entry: str(entry.get("source_id", "")),
    ):
        repo_id = str(source.get("repo_id", ""))
        relation = str(source.get("content_relation", ""))
        route_aliases = aliases_for_repo(repo_id, aliases)
        needs_base_audit = (
            source.get("role") != "additive_candidate"
            or relation != "new_family"
            or bool(route_aliases)
        )
        candidates.append(
            {
                "source_id": source.get("source_id"),
                "repo_id": repo_id,
                "revision": source.get("revision"),
                "source_family_id": source.get("source_family_id"),
                "fallback_source_dataset": repo_id,
                "fallback_first_appearance": first_appearance(repo_id, roster),
                "role": source.get("role"),
                "content_relation": relation,
                "merge_policy": source.get("merge_policy"),
                "representation_generation": representation_generation(
                    "candidate", source
                ),
                "reviewed_aliases": [
                    {
                        "lineage_alias_id": entry.get("alias_id"),
                        "alias_kind": entry.get("alias_kind"),
                        "initial_source_datasets": entry.get("initial_source_datasets"),
                        "snapshot_equivalence": entry.get("snapshot_equivalence"),
                    }
                    for entry in route_aliases
                ],
                "requires_base_identity_audit": needs_base_audit,
                "requires_family_internal_dedup": "internal_repo_overlap" in relation,
                "blind_append_allowed": False,
            }
        )
    return {
        "schema_version": REGISTRY_MANIFEST_SCHEMA,
        "authority": "first_nanochat_data_revision_plus_row_level_identity",
        "nanochat_anchor": {
            "repo_id": roster.get("repository", {}).get("repo_id"),
            "revision": roster.get("repository", {}).get("first_data_revision"),
            "committed_at": roster.get("repository", {}).get("first_data_committed_at"),
            "initial_source_datasets": initial_names,
            "later_source_name_additions": later_names,
        },
        "base": {
            "repo_id": sources.get("base", {}).get("repo_id"),
            "revision": sources.get("base", {}).get("revision"),
            "source_dataset_policy": "preserve_exact_upstream_value",
        },
        "candidates": candidates,
        "global_invariant": (
            "No candidate is blindly appended. New-family candidates still pass exact and "
            "near deduplication against the pinned Nanochat base and all other candidates."
        ),
    }


def resolve_canonical_inputs(paths: Sequence[Path]) -> list[Path]:
    """Resolve canonical JSONL or Parquet inputs without making a full copy."""

    result: list[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and (
                    candidate.suffix == ".parquet" or candidate.name.endswith(".jsonl")
                )
            )
        elif path.is_file() and (
            path.suffix == ".parquet" or path.name.endswith(".jsonl")
        ):
            result.append(path)
        else:
            raise ValueError(f"unsupported or missing canonical input: {path}")
    unique = sorted({path.resolve() for path in result})
    if not unique:
        raise ValueError("no canonical JSONL or Parquet inputs resolved")
    return unique


def _validate_bound_parquet_schema(path: Path, parquet: Any, *, origin: str) -> None:
    fields = {field.name: str(field.type) for field in parquet.schema_arrow}
    required = set(BOUND_CANONICAL_STRING_FIELDS)
    if origin == "candidate":
        required.add("text")
    missing = sorted(required - set(fields))
    wrong_type = sorted(field for field in required if fields.get(field) != "string")
    if missing or wrong_type:
        raise ValueError(
            f"{path}: canonical Parquet schema mismatch; missing={missing}, "
            f"non_string={wrong_type}"
        )


def iter_lineage_rows(
    paths: Sequence[Path],
    *,
    origin: str,
    bound_inputs: Mapping[Path, Mapping[str, Any]] | None = None,
) -> Iterator[tuple[Path, int, dict[str, Any], bool]]:
    """Read only the columns needed by lineage from receipt-bound Parquet.

    A bound Nanochat Parquet deliberately does not scan ``text`` at all: its
    canonical hashes and identities are already covered by the normalization
    shard checksum.  Candidate text remains necessary for the exact novelty
    word-token proxy.  Legacy/unbound JSONL and Parquet retain the full slow
    canonicalization path.
    """

    if origin not in {"base", "candidate"}:
        raise ValueError("origin must be base or candidate")
    bindings = {path.resolve(): value for path, value in (bound_inputs or {}).items()}
    for path in resolve_canonical_inputs(paths):
        binding = bindings.get(path.resolve())
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:  # pragma: no cover - Clariden runtime contract
                raise RuntimeError(
                    "pyarrow is required for canonical Parquet inputs"
                ) from exc
            parquet = pq.ParquetFile(path)
            if binding is not None:
                _validate_bound_parquet_schema(path, parquet, origin=origin)
                if path.stat().st_size != int(binding.get("bytes", -1)):
                    raise ValueError(f"{path}: bound canonical shard size drift")
                expected_rows = int(binding.get("rows", -1))
                if expected_rows >= 0 and parquet.metadata.num_rows != expected_rows:
                    raise ValueError(f"{path}: bound canonical shard row-count drift")
                columns = list(BOUND_CANONICAL_STRING_FIELDS)
                if origin == "candidate":
                    columns.append("text")
                row_number = 0
                for batch in parquet.iter_batches(
                    batch_size=8192, columns=columns, use_threads=False
                ):
                    payload = batch.to_pydict()
                    for index in range(batch.num_rows):
                        row_number += 1
                        yield (
                            path,
                            row_number,
                            {
                                column: values[index]
                                for column, values in payload.items()
                            },
                            True,
                        )
                continue
            row_number = 0
            for batch in parquet.iter_batches(batch_size=4096, use_threads=False):
                for value in batch.to_pylist():
                    row_number += 1
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{row_number}: row must be an object")
                    yield path, row_number, value, False
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row must be an object")
                yield path, line_number, value, False


def iter_jsonl(paths: Sequence[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    """Iterate canonical rows from legacy JSONL or sharded Parquet roots.

    The historical name remains for API compatibility. Production Phase-04
    normalization is Parquet-native, so reviewer sampling scans it in place
    rather than first writing a second, much larger JSONL representation.
    """

    for path in resolve_canonical_inputs(paths):
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:  # pragma: no cover - Clariden runtime contract
                raise RuntimeError(
                    "pyarrow is required for canonical Parquet inputs"
                ) from exc
            parquet = pq.ParquetFile(path)
            row_number = 0
            for batch in parquet.iter_batches(batch_size=4096, use_threads=False):
                for value in batch.to_pylist():
                    row_number += 1
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{row_number}: row must be an object")
                    yield path, row_number, value
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row must be an object")
                yield path, line_number, value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
