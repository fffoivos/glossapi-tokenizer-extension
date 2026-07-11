#!/usr/bin/env python3
"""Build an exact, source-aware token waterfall from stage ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finalization_io import (
    configure_duckdb,
    discover_parquet,
    read_json_object,
    sha256_file,
    sql_path_list,
    sql_string,
    utc_now,
    write_json_atomic,
)


STAGES = [
    "normalized_input",
    "source_cleaning",
    "pii_anonymization",
    "toc_bib",
    "policy_filter",
    "greekmmlu_decontamination",
    "strict_exact",
    "relaxed_exact",
    "near_duplicate",
    "final_retained",
]


def _rows(cursor: Any) -> list[dict[str, Any]]:
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def build_waterfall(
    *,
    cleaning_ledger: Path,
    decontam_ledger: Path,
    dedup_decisions: Path,
    cleaning_manifest: Path,
    decontamination_manifest: Path,
    dedup_manifest: Path,
    output: Path,
    temporary_directory: Path,
    memory_limit: str,
    threads: int,
) -> dict[str, Any]:
    import duckdb

    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable token waterfall: {output}")
    cleaning_files = discover_parquet(cleaning_ledger)
    decontam_files = discover_parquet(decontam_ledger)
    if not dedup_decisions.is_file():
        raise FileNotFoundError(dedup_decisions)
    cleaning_run = read_json_object(cleaning_manifest)
    decontamination_run = read_json_object(decontamination_manifest)
    dedup_run = read_json_object(dedup_manifest)
    if cleaning_run.get("schema_version") not in {
        "full_cpt_cleaning_manifest_v1",
        "full_cpt_structural_finalization_manifest_v1",
    }:
        raise ValueError("unsupported cleaning manifest")
    if decontamination_run.get("schema_version") != "full_cpt_greekmmlu_decontamination_v1":
        raise ValueError("unsupported decontamination manifest")
    if decontamination_run.get("policy", {}).get("policy_version") != (
        "greekmmlu_decontamination_v1"
    ):
        raise ValueError("decontamination manifest lacks the approved policy version")
    if dedup_run.get("schema_version") != "full_cpt_dedup_wrapper_manifest_v1":
        raise ValueError("unsupported dedup manifest")
    if dedup_run.get("status") != "completed":
        raise ValueError("dedup manifest is not completed")
    decision_receipt = dedup_run.get("dedup_output", {}).get("content_bound_decisions", {})
    if decision_receipt.get("sha256") != sha256_file(dedup_decisions):
        raise ValueError("dedup decisions differ from the completed dedup manifest")
    connection = duckdb.connect()
    configure_duckdb(connection, temporary_directory=temporary_directory, memory_limit=memory_limit, threads=threads)
    try:
        connection.execute(
            f"CREATE TEMP VIEW cleaning AS SELECT * FROM read_parquet({sql_path_list(cleaning_files)}, union_by_name=true)"
        )
        connection.execute(
            f"CREATE TEMP VIEW decontam AS SELECT * FROM read_parquet({sql_path_list(decontam_files)}, union_by_name=true)"
        )
        connection.execute(
            f"CREATE TEMP VIEW decisions AS SELECT * FROM read_parquet({sql_string(dedup_decisions.resolve())})"
        )
        duplicate_counts = connection.execute(
            """
            SELECT
              (SELECT count(*) - count(DISTINCT stable_uid) FROM cleaning),
              (SELECT count(*) - count(DISTINCT stable_uid) FROM decontam),
              (SELECT count(*) - count(DISTINCT stable_uid) FROM decisions)
            """
        ).fetchone()
        if any(int(value) for value in duplicate_counts):
            raise ValueError(f"waterfall input identity is non-unique: {duplicate_counts}")
        coverage = connection.execute(
            """
            SELECT
              count(*) FILTER (WHERE c.action = 'keep') AS cleaning_kept,
              count(d.stable_uid) FILTER (WHERE c.action = 'keep') AS decontam_covered,
              count(*) FILTER (WHERE c.action = 'keep' AND d.stable_uid IS NULL) AS decontam_missing,
              count(*) FILTER (
                WHERE c.action = 'keep' AND d.stable_uid IS NOT NULL
                  AND c.final_text_sha256 IS DISTINCT FROM d.input_text_sha256
              ) AS cleaning_decontam_text_hash_drift
            FROM cleaning c
            LEFT JOIN decontam d USING (stable_uid)
            """
        ).fetchone()
        if int(coverage[2]) or int(coverage[3]):
            raise ValueError(
                "cleaning/decontamination coverage or text binding failed: "
                f"missing={coverage[2]}, hash_drift={coverage[3]}"
            )
        decision_binding = connection.execute(
            """
            SELECT
              count(*) AS decisions,
              count(dc.stable_uid) AS decontam_covered,
              count(*) FILTER (WHERE dc.stable_uid IS NULL) AS missing_decontam_identity,
              count(*) FILTER (
                WHERE dc.stable_uid IS NOT NULL
                  AND d.input_text_sha256 IS DISTINCT FROM dc.input_text_sha256
              ) AS input_text_hash_drift
            FROM decisions d
            LEFT JOIN decontam dc ON dc.stable_uid = d.stable_uid AND dc.action = 'keep'
            """
        ).fetchone()
        if int(decision_binding[2]) or int(decision_binding[3]):
            raise ValueError(f"dedup decision content binding failed: {decision_binding}")

        source_totals = _rows(
            connection.execute(
                """
                WITH base AS (
                  SELECT
                    c.acquisition_source_id,
                    c.source_dataset,
                    sum(c.tokens_normalized)::BIGINT AS normalized_input,
                    sum(c.tokens_source_cleaned)::BIGINT AS source_cleaning,
                    sum(c.tokens_pii_masked)::BIGINT AS pii_anonymization,
                    sum(c.tokens_structural_cleaned)::BIGINT AS toc_bib,
                    sum(c.tokens_final)::BIGINT AS policy_filter,
                    sum(CASE WHEN dc.action = 'keep' THEN c.tokens_final ELSE 0 END)::BIGINT
                      AS greekmmlu_decontamination
                  FROM cleaning c
                  LEFT JOIN decontam dc USING (stable_uid)
                  GROUP BY 1, 2
                ), dropped AS (
                  SELECT
                    c.acquisition_source_id,
                    c.source_dataset,
                    sum(CASE WHEN d.decision_stage = 'strict_exact' AND d.decision = 'drop'
                             THEN c.tokens_final ELSE 0 END)::BIGINT AS strict_loss,
                    sum(CASE WHEN d.decision_stage = 'relaxed_exact' AND d.decision = 'drop'
                             THEN c.tokens_final ELSE 0 END)::BIGINT AS relaxed_loss,
                    sum(CASE WHEN d.decision_stage = 'near_duplicate' AND d.decision = 'drop'
                             THEN c.tokens_final ELSE 0 END)::BIGINT AS near_loss,
                    sum(CASE WHEN d.decision = 'keep' THEN c.tokens_final ELSE 0 END)::BIGINT AS final_retained
                  FROM decisions d
                  JOIN cleaning c ON c.stable_uid = d.stable_uid
                  GROUP BY 1, 2
                )
                SELECT
                  b.acquisition_source_id,
                  b.source_dataset,
                  b.normalized_input,
                  b.source_cleaning,
                  b.pii_anonymization,
                  b.toc_bib,
                  b.policy_filter,
                  b.greekmmlu_decontamination,
                  (b.greekmmlu_decontamination - coalesce(x.strict_loss, 0))::BIGINT AS strict_exact,
                  (b.greekmmlu_decontamination - coalesce(x.strict_loss, 0)
                    - coalesce(x.relaxed_loss, 0))::BIGINT AS relaxed_exact,
                  (b.greekmmlu_decontamination - coalesce(x.strict_loss, 0)
                    - coalesce(x.relaxed_loss, 0) - coalesce(x.near_loss, 0))::BIGINT AS near_duplicate,
                  coalesce(x.final_retained, 0)::BIGINT AS final_retained
                FROM base b
                LEFT JOIN dropped x USING (acquisition_source_id, source_dataset)
                ORDER BY 1, 2
                """
            )
        )
        for row in source_totals:
            if int(row["near_duplicate"]) != int(row["final_retained"]):
                raise RuntimeError(f"token waterfall does not reconcile for {row['acquisition_source_id']}")

        event_queries = [
            (
                "source_cleaning",
                "source_cleaning_rules_combined",
                "tokens_normalized - tokens_source_cleaned",
                "tokens_normalized <> tokens_source_cleaned",
            ),
            (
                "pii_anonymization",
                "high_confidence_pii_masking",
                "tokens_source_cleaned - tokens_pii_masked",
                "tokens_source_cleaned <> tokens_pii_masked",
            ),
            (
                "toc_bib",
                "toc_bibliography_structural_combined",
                "tokens_pii_masked - tokens_structural_cleaned",
                "tokens_pii_masked <> tokens_structural_cleaned",
            ),
        ]
        events: list[dict[str, Any]] = []
        for stage, reason, expression, affected in event_queries:
            events.extend(
                _rows(
                    connection.execute(
                        f"""
                        SELECT acquisition_source_id, source_dataset,
                               {sql_string(stage)} AS stage, {sql_string(reason)} AS reason,
                               count(*) FILTER (WHERE {affected})::BIGINT AS documents,
                               sum({expression})::BIGINT AS tokens_removed
                        FROM cleaning GROUP BY 1, 2 ORDER BY 1, 2
                        """
                    )
                )
            )
        for stage, reason, column in (
            ("bibliography", "approved_bibliography_spans", "tokens_bibliography_removed"),
            ("toc", "approved_toc_spans", "tokens_toc_removed"),
            ("toc_bib_union", "approved_structural_span_union", "tokens_structural_union_removed"),
        ):
            events.extend(
                _rows(
                    connection.execute(
                        f"""
                        SELECT acquisition_source_id, source_dataset,
                               {sql_string(stage)} AS stage, {sql_string(reason)} AS reason,
                               count(*) FILTER (WHERE {column} > 0)::BIGINT AS documents,
                               sum({column})::BIGINT AS tokens_removed
                        FROM cleaning GROUP BY 1, 2 ORDER BY 1, 2
                        """
                    )
                )
            )
        structural_loss_by_source = _rows(
            connection.execute(
                """
                SELECT acquisition_source_id, source_dataset,
                       sum(tokens_bibliography_removed)::BIGINT AS bibliography_tokens_removed,
                       sum(tokens_toc_removed)::BIGINT AS toc_tokens_removed,
                       sum(tokens_structural_union_removed)::BIGINT AS union_tokens_removed
                FROM cleaning GROUP BY 1, 2 ORDER BY 1, 2
                """
            )
        )
        events.extend(
            _rows(
                connection.execute(
                    """
                    SELECT acquisition_source_id, source_dataset, 'policy_filter' AS stage,
                           CASE WHEN reasons_json IN ('', '[]') THEN action ELSE reasons_json END AS reason,
                           count(*)::BIGINT AS documents,
                           sum(tokens_structural_cleaned - tokens_final)::BIGINT AS tokens_removed
                    FROM cleaning
                    WHERE action <> 'keep' OR NOT eligible_for_training
                    GROUP BY 1, 2, 4 ORDER BY 1, 2, 4
                    """
                )
            )
        )
        events.extend(
            _rows(
                connection.execute(
                    """
                    SELECT c.acquisition_source_id, c.source_dataset,
                           'greekmmlu_decontamination' AS stage, dc.reason,
                           count(*)::BIGINT AS documents, sum(c.tokens_final)::BIGINT AS tokens_removed
                    FROM decontam dc JOIN cleaning c USING (stable_uid)
                    WHERE dc.action = 'drop'
                    GROUP BY 1, 2, 4 ORDER BY 1, 2, 4
                    """
                )
            )
        )
        events.extend(
            _rows(
                connection.execute(
                    """
                    SELECT c.acquisition_source_id, c.source_dataset, d.decision_stage AS stage,
                           d.reason, count(*)::BIGINT AS documents,
                           sum(c.tokens_final)::BIGINT AS tokens_removed
                    FROM decisions d JOIN cleaning c ON c.stable_uid = d.stable_uid
                    WHERE d.decision = 'drop'
                    GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4
                    """
                )
            )
        )
    finally:
        connection.close()

    global_totals = {stage: sum(int(row[stage]) for row in source_totals) for stage in STAGES}
    stage_rows: list[dict[str, Any]] = []
    previous = global_totals["normalized_input"]
    for order, stage in enumerate(STAGES):
        after = global_totals[stage]
        before = previous if stage != "normalized_input" else after
        stage_rows.append(
            {
                "stage_order": order,
                "stage": stage,
                "tokens_before": before,
                "tokens_after": after,
                "tokens_removed": before - after,
            }
        )
        previous = after
    global_events: dict[tuple[str, str], dict[str, int]] = {}
    for event in events:
        key = (str(event["stage"]), str(event["reason"]))
        target = global_events.setdefault(key, {"documents": 0, "tokens_removed": 0})
        target["documents"] += int(event["documents"] or 0)
        target["tokens_removed"] += int(event["tokens_removed"] or 0)
    payload = {
        "schema_version": "full_cpt_token_waterfall_v1",
        "completed_at": utc_now(),
        "inputs": {
            "cleaning_ledger": str(cleaning_ledger.resolve()),
            "cleaning_manifest": str(cleaning_manifest.resolve()),
            "cleaning_manifest_sha256": sha256_file(cleaning_manifest),
            "decontamination_ledger": str(decontam_ledger.resolve()),
            "decontamination_manifest": str(decontamination_manifest.resolve()),
            "decontamination_manifest_sha256": sha256_file(decontamination_manifest),
            "dedup_decisions": str(dedup_decisions.resolve()),
            "dedup_decisions_sha256": sha256_file(dedup_decisions),
            "dedup_manifest": str(dedup_manifest.resolve()),
            "dedup_manifest_sha256": sha256_file(dedup_manifest),
        },
        "stage_totals": stage_rows,
        "source_stage_totals": source_totals,
        "events_by_source_and_reason": events,
        "events_global": [
            {"stage": stage, "reason": reason, **counts}
            for (stage, reason), counts in sorted(global_events.items())
        ],
        "structural_token_loss": {
            "by_source": structural_loss_by_source,
            "global": {
                "bibliography_tokens_removed": sum(
                    int(row["bibliography_tokens_removed"] or 0)
                    for row in structural_loss_by_source
                ),
                "toc_tokens_removed": sum(
                    int(row["toc_tokens_removed"] or 0) for row in structural_loss_by_source
                ),
                "union_tokens_removed": sum(
                    int(row["union_tokens_removed"] or 0) for row in structural_loss_by_source
                ),
            },
        },
        "invariants": {
            "decontam_coverage": {
                "cleaning_kept": int(coverage[0]),
                "decontam_covered": int(coverage[1]),
                "missing": int(coverage[2]),
                "input_text_hash_drift": int(coverage[3]),
            },
            "dedup_content_binding": {
                "decisions": int(decision_binding[0]),
                "decontam_covered": int(decision_binding[1]),
                "missing_decontam_identity": int(decision_binding[2]),
                "input_text_hash_drift": int(decision_binding[3]),
            },
            "final_tokens": global_totals["final_retained"],
            "reconciled": global_totals["near_duplicate"] == global_totals["final_retained"],
        },
    }
    write_json_atomic(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaning-ledger", type=Path, required=True)
    parser.add_argument("--decontam-ledger", type=Path, required=True)
    parser.add_argument("--dedup-decisions", type=Path, required=True)
    parser.add_argument("--cleaning-manifest", type=Path, required=True)
    parser.add_argument("--decontamination-manifest", type=Path, required=True)
    parser.add_argument("--dedup-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", default="200GB")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    payload = build_waterfall(
        cleaning_ledger=args.cleaning_ledger,
        decontam_ledger=args.decontam_ledger,
        dedup_decisions=args.dedup_decisions,
        cleaning_manifest=args.cleaning_manifest,
        decontamination_manifest=args.decontamination_manifest,
        dedup_manifest=args.dedup_manifest,
        output=args.output,
        temporary_directory=args.temporary_directory,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    print(json.dumps({"ok": True, "output": str(args.output), "final_tokens": payload["invariants"]["final_tokens"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
