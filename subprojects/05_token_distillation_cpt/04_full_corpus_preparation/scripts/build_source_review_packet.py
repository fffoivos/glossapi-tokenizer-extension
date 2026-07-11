#!/usr/bin/env python3
"""Create deterministic, source-stratified Codex review requests.

The command consumes canonical candidate JSONL envelopes, samples each exact
``source_dataset`` value independently, redacts direct identifiers, and emits a
structured review packet.  It does not call a model or make admission decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from source_lineage import (
    canonical_json,
    canonicalize_row,
    iter_jsonl,
    load_json,
    resolve_canonical_inputs,
    sha256_parts,
    write_json,
)


REQUEST_SCHEMA = "source_quality_review_request_v1"
SUMMARY_SCHEMA = "source_quality_review_packet_summary_v1"

EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-zΑ-Ωα-ω]{2,}(?![\w.-])")
IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
IBAN_RE = re.compile(r"(?i)\bGR\s*\d{2}(?:\s*[0-9A-Z]){23}\b")
AFM_RE = re.compile(r"(?i)\b(?:Α\.?\s*Φ\.?\s*Μ\.?|AFM)\s*[:#-]?\s*\d{9}\b")
AMKA_RE = re.compile(r"(?i)\b(?:Α\.?\s*Μ\.?\s*Κ\.?\s*Α\.?|AMKA)\s*[:#-]?\s*\d{11}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+30\s*)?(?:2\d{9}|69\d{8})(?!\d)")
HTML_RE = re.compile(r"<\s*/?\s*[A-Za-z][^>]{0,200}>")
MOJIBAKE_RE = re.compile(r"(?:Ã.|Â.|â€|Î[\x80-\xbf]|Ï[\x80-\xbf]|\ufffd)")
GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
TEMPLATE_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
TEMPLATE_NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
TEMPLATE_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")


def metadata_private(row: Mapping[str, Any]) -> bool:
    values: list[Any] = [row]
    for field in ("metadata", "source_metadata", "metadata_json", "source_metadata_json"):
        value = row.get(field)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            values.append(value)
    for value in values:
        if not isinstance(value, Mapping):
            continue
        flag = value.get("privateData", value.get("private_data"))
        if flag is True or (isinstance(flag, str) and flag.strip().lower() == "true"):
            return True
    return False


def redact_direct_identifiers(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, pattern in (
        ("email", EMAIL_RE),
        ("ipv4", IPV4_RE),
        ("iban", IBAN_RE),
        ("afm", AFM_RE),
        ("amka", AMKA_RE),
        ("phone", PHONE_RE),
    ):
        text, count = pattern.subn(f"[REDACTED_{name.upper()}]", text)
        if count:
            counts[name] = count
    return text, counts


def quality_metrics(text: str) -> dict[str, Any]:
    characters = len(text)
    html_tags = len(HTML_RE.findall(text))
    mojibake = len(MOJIBAKE_RE.findall(text))
    replacement = text.count("\ufffd")
    control = sum(1 for char in text if unicontrol(char))
    letters = LETTER_RE.findall(text)
    greek = GREEK_RE.findall(text)
    greek_letter_fraction = len(greek) / len(letters) if letters else 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_lines = sum(1 for line in lines if len(line) <= 3)
    repeated_lines = len(lines) - len(set(lines))
    short_line_fraction = short_lines / len(lines) if lines else 0.0
    repeated_line_fraction = repeated_lines / len(lines) if lines else 0.0
    tag_density = html_tags * 1000.0 / max(characters, 1)
    mojibake_density = (mojibake + replacement) * 1000.0 / max(characters, 1)
    control_density = control * 1000.0 / max(characters, 1)
    risk_score = (
        min(tag_density / 2.0, 4.0)
        + min(mojibake_density * 5.0, 4.0)
        + min(control_density * 5.0, 3.0)
        + min(short_line_fraction * 3.0, 2.0)
        + min(repeated_line_fraction * 4.0, 3.0)
        + (2.0 if letters and greek_letter_fraction < 0.2 else 0.0)
        + (2.0 if characters < 200 else 0.0)
    )
    return {
        "characters": characters,
        "lines": len(lines),
        "html_tags": html_tags,
        "html_tags_per_1000_chars": round(tag_density, 6),
        "mojibake_markers": mojibake,
        "replacement_characters": replacement,
        "control_characters": control,
        "greek_letter_fraction": round(greek_letter_fraction, 6),
        "short_line_fraction": round(short_line_fraction, 6),
        "repeated_line_fraction": round(repeated_line_fraction, 6),
        "risk_score": round(risk_score, 6),
    }


def unicontrol(char: str) -> bool:
    code = ord(char)
    return (code < 32 and char not in "\n\r\t") or 127 <= code <= 159


def review_text(text: str, excerpt_cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    text, redactions = redact_direct_identifiers(text)
    full_limit = int(excerpt_cfg["full_text_max_characters"])
    segment = int(excerpt_cfg["segment_characters"])
    if len(text) <= full_limit:
        return {"mode": "full", "text": text}, redactions
    middle_start = max(0, (len(text) - segment) // 2)
    return {
        "mode": "front_middle_end",
        "original_characters": len(text),
        "front": text[:segment],
        "middle": text[middle_start : middle_start + segment],
        "end": text[-segment:],
    }, redactions


def cluster_id(row: Mapping[str, Any], lineage: Mapping[str, Any], policy: Mapping[str, Any]) -> str:
    for field in policy.get("cluster_fields_in_priority_order", []):
        value = row.get(field)
        if value is not None and str(value):
            return f"{field}:{value}"
    text = str(row.get("text") or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3:
        selected = lines[:16] + (lines[-4:] if len(lines) > 20 else [])
        skeleton: list[str] = []
        for line in selected:
            value = TEMPLATE_URL_RE.sub(" URL ", line)
            value = EMAIL_RE.sub(" EMAIL ", value)
            value = TEMPLATE_NUMBER_RE.sub(" N ", value)
            value = TEMPLATE_WORD_RE.sub(" W ", value)
            value = re.sub(r"\s+", " ", value).strip()[:240]
            skeleton.append(value)
        digest = hashlib.sha256("\n".join(skeleton).encode("utf-8")).hexdigest()
        return "structural_template_v1:" + digest
    return "exact:" + str(lineage["normalized_text_sha256"])


def deterministic_rank(seed: str, namespace: str, stable_uid: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{stable_uid}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def make_item(
    row: Mapping[str, Any],
    lineage: Mapping[str, Any],
    *,
    cluster: str,
    cluster_size: int,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    text = str(row["text"])
    document, redactions = review_text(text, policy["excerpt"])
    paired = None
    if isinstance(row.get("base_comparison_text"), str):
        paired_document, paired_redactions = review_text(
            str(row["base_comparison_text"]), policy["excerpt"]
        )
        paired = {
            "stable_uid": row.get("base_comparison_uid"),
            "document": paired_document,
            "redactions": paired_redactions,
        }
    return {
        "sample_id": lineage["stable_uid"],
        "source_id": lineage["source_id"],
        "source_dataset": lineage["source_dataset"],
        "source_dataset_origin": lineage["source_dataset_origin"],
        "source_family_id": lineage["source_family_id"],
        "source_repo_id": lineage["source_repo_id"],
        "source_revision": lineage["source_revision"],
        "source_doc_id": lineage["source_doc_id"],
        "work_key": lineage["work_key"],
        "representation_generation": lineage["representation_generation"],
        "lineage_class": lineage["lineage_class"],
        "first_appearance": lineage["first_appearance"],
        "review_cluster_id": cluster,
        "review_cluster_size": cluster_size,
        "quality_metrics": quality_metrics(text),
        "document": document,
        "redactions": redactions,
        "paired_base_representation": paired,
    }


def push_random(heap: list, capacity: int, rank: int, uid: str, item: dict) -> None:
    entry = (-rank, uid, item)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)


def push_risk(heap: list, capacity: int, rank: int, uid: str, item: dict) -> None:
    score = float(item["quality_metrics"]["risk_score"])
    entry = (score, -rank, uid, item)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif entry[:3] > heap[0][:3]:
        heapq.heapreplace(heap, entry)


def source_plan(source_id: str, count: int, policy: Mapping[str, Any]) -> dict[str, int]:
    is_large = count >= int(policy["large_source_min_documents"])
    is_heterogeneous = source_id in set(policy.get("heterogeneous_source_ids", []))
    key = "large_or_heterogeneous_sample" if is_large or is_heterogeneous else "default_sample"
    return {name: int(value) for name, value in policy[key].items()}


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema_version") != "full_cpt_source_review_policy_v1":
        raise ValueError("unsupported source review policy schema")
    if policy.get("grouping_field") != "source_dataset":
        raise ValueError("review sampling must group by exact source_dataset")
    for key in ("default_sample", "large_or_heterogeneous_sample"):
        sample = policy.get(key, {})
        total = sample.get("total_unique_documents")
        if total != sum(sample.get(name, 0) for name in ("random", "risk", "cluster")):
            raise ValueError(f"{key} quotas must sum to total_unique_documents")
    if policy["default_sample"] != {
        "total_unique_documents": 100,
        "random": 60,
        "risk": 20,
        "cluster": 20,
    }:
        raise ValueError("default review sample must remain 100 = 60 random + 20 risk + 20 cluster")
    if policy["large_or_heterogeneous_sample"] != {
        "total_unique_documents": 200,
        "random": 100,
        "risk": 50,
        "cluster": 50,
    }:
        raise ValueError("large/heterogeneous sample must remain 200 = 100 + 50 + 50")
    fraction = policy.get("double_review_fraction")
    if not isinstance(fraction, (float, int)) or not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("double_review_fraction must be in [0,1]")


def admission_filter(path: Path | None, decisions: set[str]) -> dict[str, str] | None:
    if path is None:
        if decisions:
            raise ValueError("--decision requires --source-admission")
        return None
    value = load_json(path)
    if value.get("schema_version") != "source_quality_review_admission_v1":
        raise ValueError("source admission must use source_quality_review_admission_v1")
    if int(value.get("pending_adjudications", 0)):
        raise ValueError("source admission still has pending adjudications")
    rows = value.get("sources")
    if not isinstance(rows, list):
        raise ValueError("source admission sources must be a list")
    result = {str(row["source_dataset"]): str(row["decision"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("source admission has duplicate source_dataset values")
    return result


def review_row_selected(
    row: Mapping[str, Any],
    *,
    source_ids: set[str],
    admissions: Mapping[str, str] | None,
    decisions: set[str],
) -> bool:
    source_id = str(row.get("source_id") or row.get("acquisition_source_id") or "")
    if source_id == "nanochat_base":
        return False
    if source_ids and source_id not in source_ids:
        return False
    if admissions is not None:
        source_dataset = str(row.get("source_dataset") or "")
        if source_dataset not in admissions:
            return False
        if decisions and admissions[source_dataset] not in decisions:
            return False
    return True


def first_pass(
    paths: list[Path],
    *,
    sources: dict,
    roster: dict,
    aliases: dict,
    policy: dict,
    connection: sqlite3.Connection,
    selected_source_ids: set[str],
    admissions: Mapping[str, str] | None,
    decisions: set[str],
) -> tuple[Counter, dict[str, set[str]], Counter]:
    counts: Counter = Counter()
    source_ids: dict[str, set[str]] = defaultdict(set)
    skipped: Counter = Counter()
    connection.executescript(
        """
        CREATE TABLE clusters(
            source_dataset TEXT,
            cluster_id TEXT,
            n INTEGER,
            PRIMARY KEY(source_dataset, cluster_id)
        );
        CREATE TABLE review_rows(stable_uid TEXT PRIMARY KEY);
        """
    )
    for path, line_number, row in iter_jsonl(paths):
        if not review_row_selected(
            row,
            source_ids=selected_source_ids,
            admissions=admissions,
            decisions=decisions,
        ):
            continue
        try:
            lineage = canonicalize_row(
                row,
                origin="candidate",
                sources=sources,
                roster=roster,
                aliases=aliases,
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        source_dataset = lineage["source_dataset"]
        if metadata_private(row):
            skipped[source_dataset] += 1
            continue
        cluster = cluster_id(row, lineage, policy)
        try:
            connection.execute(
                "INSERT INTO review_rows(stable_uid) VALUES (?)", (lineage["stable_uid"],)
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"{path}:{line_number}: duplicate stable_uid {lineage['stable_uid']} in review input"
            ) from exc
        counts[source_dataset] += 1
        source_ids[source_dataset].add(str(lineage["source_id"]))
        connection.execute(
            """
            INSERT INTO clusters(source_dataset, cluster_id, n) VALUES (?, ?, 1)
            ON CONFLICT(source_dataset, cluster_id) DO UPDATE SET n = n + 1
            """,
            (source_dataset, cluster),
        )
    connection.commit()
    return counts, source_ids, skipped


def top_cluster_map(
    connection: sqlite3.Connection,
    counts: Counter,
    source_ids: Mapping[str, set[str]],
    policy: Mapping[str, Any],
) -> tuple[dict[str, list[tuple[str, int]]], dict[tuple[str, str], int]]:
    result: dict[str, list[tuple[str, int]]] = {}
    sizes: dict[tuple[str, str], int] = {}
    for source_dataset in sorted(counts):
        ids = sorted(source_ids[source_dataset])
        plan = source_plan(ids[0], counts[source_dataset], policy)
        limit = max(plan["cluster"] * 4, plan["cluster"])
        rows = list(
            connection.execute(
                """
                SELECT cluster_id, n FROM clusters WHERE source_dataset = ?
                ORDER BY n DESC, cluster_id ASC LIMIT ?
                """,
                (source_dataset, limit),
            )
        )
        result[source_dataset] = [(str(cluster), int(n)) for cluster, n in rows]
        for cluster, n in rows:
            sizes[(source_dataset, str(cluster))] = int(n)
    return result, sizes


def select_items(
    paths: list[Path],
    *,
    sources: dict,
    roster: dict,
    aliases: dict,
    policy: dict,
    counts: Counter,
    source_ids: Mapping[str, set[str]],
    top_clusters: Mapping[str, list[tuple[str, int]]],
    cluster_sizes: Mapping[tuple[str, str], int],
    selected_source_ids: set[str],
    admissions: Mapping[str, str] | None,
    decisions: set[str],
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    random_heaps: dict[str, list] = defaultdict(list)
    risk_heaps: dict[str, list] = defaultdict(list)
    cluster_reps: dict[tuple[str, str], tuple[int, dict]] = {}
    cluster_sets = {source: {cluster for cluster, _ in values} for source, values in top_clusters.items()}
    seed = str(policy["seed"])

    for path, line_number, row in iter_jsonl(paths):
        if not review_row_selected(
            row,
            source_ids=selected_source_ids,
            admissions=admissions,
            decisions=decisions,
        ):
            continue
        try:
            lineage = canonicalize_row(
                row,
                origin="candidate",
                sources=sources,
                roster=roster,
                aliases=aliases,
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        source_dataset = lineage["source_dataset"]
        if metadata_private(row):
            continue
        cluster = cluster_id(row, lineage, policy)
        item = make_item(
            row,
            lineage,
            cluster=cluster,
            cluster_size=cluster_sizes.get((source_dataset, cluster), 1),
            policy=policy,
        )
        uid = str(item["sample_id"])
        rank = deterministic_rank(seed, "sample", uid)
        plan = source_plan(sorted(source_ids[source_dataset])[0], counts[source_dataset], policy)
        capacity = max(plan["total_unique_documents"] * 3, plan["total_unique_documents"])
        push_random(random_heaps[source_dataset], capacity, rank, uid, item)
        push_risk(risk_heaps[source_dataset], capacity, rank, uid, item)
        if cluster in cluster_sets.get(source_dataset, set()):
            key = (source_dataset, cluster)
            previous = cluster_reps.get(key)
            if previous is None or rank < previous[0]:
                cluster_reps[key] = (rank, item)

    selected: dict[str, list[dict]] = {}
    actual_strata: dict[str, dict[str, int]] = {}
    for source_dataset in sorted(counts):
        plan = source_plan(sorted(source_ids[source_dataset])[0], counts[source_dataset], policy)
        chosen: list[dict] = []
        selected_ids: set[str] = set()
        strata = Counter()

        for cluster, _ in top_clusters[source_dataset]:
            representative = cluster_reps.get((source_dataset, cluster))
            if representative is None:
                continue
            item = representative[1]
            if item["sample_id"] in selected_ids:
                continue
            item = dict(item, sampling_stratum="cluster")
            chosen.append(item)
            selected_ids.add(item["sample_id"])
            strata["cluster"] += 1
            if strata["cluster"] >= plan["cluster"]:
                break

        risk_candidates = sorted(
            (entry[3] for entry in risk_heaps[source_dataset]),
            key=lambda item: (
                -float(item["quality_metrics"]["risk_score"]),
                deterministic_rank(seed, "sample", item["sample_id"]),
                item["sample_id"],
            ),
        )
        for item in risk_candidates:
            if item["sample_id"] in selected_ids:
                continue
            item = dict(item, sampling_stratum="risk")
            chosen.append(item)
            selected_ids.add(item["sample_id"])
            strata["risk"] += 1
            if strata["risk"] >= plan["risk"]:
                break

        random_candidates = sorted(
            (entry[2] for entry in random_heaps[source_dataset]),
            key=lambda item: (
                deterministic_rank(seed, "sample", item["sample_id"]), item["sample_id"]
            ),
        )
        for item in random_candidates:
            if item["sample_id"] in selected_ids:
                continue
            item = dict(item, sampling_stratum="random")
            chosen.append(item)
            selected_ids.add(item["sample_id"])
            strata["random"] += 1
            if strata["random"] >= plan["random"]:
                break

        if len(chosen) < min(plan["total_unique_documents"], counts[source_dataset]):
            fallback = sorted(
                [*risk_candidates, *random_candidates],
                key=lambda item: (
                    deterministic_rank(seed, "fallback", item["sample_id"]), item["sample_id"]
                ),
            )
            target = min(plan["total_unique_documents"], counts[source_dataset])
            for item in fallback:
                if item["sample_id"] in selected_ids:
                    continue
                item = dict(item, sampling_stratum="quota_fallback")
                chosen.append(item)
                selected_ids.add(item["sample_id"])
                strata["quota_fallback"] += 1
                if len(chosen) >= target:
                    break
        selected[source_dataset] = sorted(chosen, key=lambda item: item["sample_id"])
        actual_strata[source_dataset] = dict(sorted(strata.items()))
    return selected, actual_strata


def request_record(item: dict, *, slot: str, review_phase: str) -> dict[str, Any]:
    review_id = sha256_parts("source_quality_review_request_v1", item["sample_id"], slot, review_phase)
    return {
        "schema_version": REQUEST_SCHEMA,
        "review_id": review_id,
        "sample_id": item["sample_id"],
        "reviewer_slot": slot,
        "review_phase": review_phase,
        "source_dataset": item["source_dataset"],
        "sampling_stratum": item["sampling_stratum"],
        "response_schema_version": "source_quality_review_response_v1",
        "task": (
            "Judge training value, source-appropriate cleanliness (HTML and/or PDF/OCR), "
            "and substantive variability. Return one JSON object matching the response schema."
        ),
        "source": {
            key: item[key]
            for key in (
                "source_id",
                "source_dataset",
                "source_dataset_origin",
                "source_family_id",
                "source_repo_id",
                "source_revision",
                "source_doc_id",
                "work_key",
                "representation_generation",
                "lineage_class",
                "first_appearance",
            )
        },
        "cluster": {
            "id": item["review_cluster_id"],
            "size": item["review_cluster_size"],
        },
        "quality_metrics": item["quality_metrics"],
        "document": item["document"],
        "redactions": item["redactions"],
        "paired_base_representation": item["paired_base_representation"],
    }


def build_packet(args: argparse.Namespace) -> int:
    sources = load_json(args.sources_config)
    roster = load_json(args.roster_config)
    aliases = load_json(args.aliases_config)
    policy = load_json(args.review_policy)
    validate_policy(policy)
    candidate_inputs = resolve_canonical_inputs(args.candidate_jsonl)
    decisions = set(args.decision or [])
    admissions = admission_filter(args.source_admission, decisions)
    selected_source_ids = set(args.source_id or [])

    with tempfile.TemporaryDirectory(prefix="source-review-clusters-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "clusters.sqlite")
        counts, source_ids, skipped = first_pass(
            candidate_inputs,
            sources=sources,
            roster=roster,
            aliases=aliases,
            policy=policy,
            connection=connection,
            selected_source_ids=selected_source_ids,
            admissions=admissions,
            decisions=decisions,
        )
        if not counts:
            raise ValueError("no review-eligible candidate rows")
        for name, ids in source_ids.items():
            if len(ids) != 1:
                raise ValueError(
                    f"exact source_dataset {name!r} maps to multiple source_id values: {sorted(ids)}"
                )
        top_clusters, cluster_sizes = top_cluster_map(
            connection, counts, source_ids, policy
        )
        selected, actual_strata = select_items(
            candidate_inputs,
            sources=sources,
            roster=roster,
            aliases=aliases,
            policy=policy,
            counts=counts,
            source_ids=source_ids,
            top_clusters=top_clusters,
            cluster_sizes=cluster_sizes,
            selected_source_ids=selected_source_ids,
            admissions=admissions,
            decisions=decisions,
        )
        connection.close()

    args.requests_out.parent.mkdir(parents=True, exist_ok=True)
    request_count = 0
    source_reports: list[dict[str, Any]] = []
    with args.requests_out.open("w", encoding="utf-8") as handle:
        for source_dataset in sorted(selected):
            items = selected[source_dataset]
            source_id = sorted(source_ids[source_dataset])[0]
            plan = source_plan(source_id, counts[source_dataset], policy)
            double_count = min(
                len(items), int(math.ceil(len(items) * float(policy["double_review_fraction"])))
            )
            double_ids = {
                item["sample_id"]
                for item in sorted(
                    items,
                    key=lambda item: (
                        deterministic_rank(str(policy["seed"]), "double", item["sample_id"]),
                        item["sample_id"],
                    ),
                )[:double_count]
            }
            records: list[dict] = []
            for item in items:
                records.append(request_record(item, slot="primary", review_phase=args.review_phase))
                if item["sample_id"] in double_ids:
                    records.append(
                        request_record(item, slot="secondary", review_phase=args.review_phase)
                    )
            for record in sorted(records, key=lambda value: value["review_id"]):
                handle.write(canonical_json(record) + "\n")
                request_count += 1
            source_reports.append(
                {
                    "source_id": source_id,
                    "source_dataset": source_dataset,
                    "eligible_documents": counts[source_dataset],
                    "private_data_documents_excluded": skipped[source_dataset],
                    "planned_sample": plan,
                    "unique_sampled_documents": len(items),
                    "sampling_strata": actual_strata[source_dataset],
                    "double_review_documents": double_count,
                    "request_rows": len(records),
                    "shortfall": max(0, plan["total_unique_documents"] - len(items)),
                }
            )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "review_phase": args.review_phase,
        "seed": policy["seed"],
        "grouping_field": "source_dataset",
        "input_files": [
            {"path": str(path), "bytes": path.stat().st_size} for path in candidate_inputs
        ],
        "response_schema": str(args.response_schema),
        "requests": {"path": str(args.requests_out), "rows": request_count},
        "sources": source_reports,
        "unique_sampled_documents": sum(
            report["unique_sampled_documents"] for report in source_reports
        ),
        "private_data_documents_excluded": sum(skipped.values()),
    }
    write_json(args.summary_out, summary)
    return 0


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-jsonl",
        "--candidate-input",
        dest="candidate_jsonl",
        action="append",
        type=Path,
        required=True,
        help="canonical JSONL, Parquet file, or sharded Parquet root (repeatable)",
    )
    parser.add_argument("--source-id", action="append", help="limit to acquisition source_id")
    parser.add_argument(
        "--source-admission",
        type=Path,
        help="optional completed admission report used to select exact source_dataset values",
    )
    parser.add_argument(
        "--decision",
        action="append",
        choices=("include", "include_after_cleaning", "quarantine", "exclude"),
        help="with --source-admission, retain only these decisions (repeatable)",
    )
    parser.add_argument("--sources-config", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument(
        "--roster-config", type=Path, default=here / "configs" / "nanochat_initial_roster.json"
    )
    parser.add_argument(
        "--aliases-config", type=Path, default=here / "configs" / "source_lineage_aliases.json"
    )
    parser.add_argument(
        "--review-policy", type=Path, default=here / "configs" / "source_review_policy.json"
    )
    parser.add_argument(
        "--response-schema",
        type=Path,
        default=here / "schemas" / "source_review_response.schema.json",
    )
    parser.add_argument("--review-phase", choices=("pre_clean", "post_clean"), default="pre_clean")
    parser.add_argument("--requests-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    if not args.response_schema.is_file():
        parser.error(f"response schema not found: {args.response_schema}")
    return build_packet(args)


if __name__ == "__main__":
    raise SystemExit(main())
