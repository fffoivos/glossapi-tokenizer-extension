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
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


LINEAGE_ROW_SCHEMA = "full_cpt_lineage_row_v1"
REGISTRY_MANIFEST_SCHEMA = "full_cpt_lineage_registry_v1"


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
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))
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


def source_indexes(sources: Mapping[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
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
    row: Mapping[str, Any], repo_id: str, source_dataset: str, aliases: Mapping[str, Any]
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


def lineage_class(origin: str, source: Mapping[str, Any], alias_id: str | None, aliases: Mapping[str, Any]) -> str:
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
    if not isinstance(text, str):
        raise ValueError("canonical row requires string text")
    by_id, _ = source_indexes(sources)
    if origin == "base":
        source: dict[str, Any] = dict(sources.get("base", {}))
        source["source_id"] = "nanochat_base"
        source["source_family_id"] = str(row.get("source_family_id") or row.get("source_dataset") or "")
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
        row.get("source_text_field")
        or (source.get("text_columns") or ["text"])[0]
    )
    relation = str(source.get("content_relation", ""))
    if origin == "candidate" and "resegmentation" in relation and not row.get("work_id"):
        raise ValueError(
            f"{source.get('source_id')}: resegmented rows require an explicit work_id"
        )
    work_id = normalize_work_identifier(row.get("work_id") or source_doc_id)
    source_family_id = str(row.get("source_family_id") or source.get("source_family_id") or source_dataset)
    if not source_family_id:
        raise ValueError("source_family_id must not be empty")
    alias_id = choose_alias_id(row, repo_id, source_dataset, aliases)
    generation = str(row.get("representation_generation") or representation_generation(origin, source))
    normalized_text = normalize_identity_text(text)

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
        "original_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_text_sha256": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
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
        "first_appearance": first_appearance(source_dataset, roster),
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
                "representation_generation": representation_generation("candidate", source),
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


def iter_jsonl(paths: Sequence[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row must be an object")
                yield path, line_number, value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
