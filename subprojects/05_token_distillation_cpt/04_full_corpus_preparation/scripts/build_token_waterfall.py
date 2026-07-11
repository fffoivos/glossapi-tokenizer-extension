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
              (SELECT count(*) - count(DISTINCT source_doc_id) FROM decisions)
            """
        ).fetchone()
        if any(int(value) for value in duplicate_counts):
            raise ValueError(f"waterfall input identity is non-unique: {duplicate_counts}")
        coverage = connection.execute(
            """
            SELECT
              count(*) FILTER (WHERE c.action = 'keep') AS cleaning_kept,
              count(d.stable_uid) FILTER (WHERE c.action = 'keep') AS decontam_covered,
              count(*) FILTER (WHERE c.action = 'keep' AND d.stable_uid IS NULL) AS decontam_missing
            FROM cleaning c
            LEFT JOIN decontam d USING (stable_uid)
            """
        ).fetchone()
        if int(coverage[2]):
            raise ValueError(f"decontamination ledger misses {coverage[2]} cleaning-kept documents")

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
                  JOIN cleaning c ON c.stable_uid = d.source_doc_id
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
        events.extend(
            _rows(
                connection.execute(
                    """
                    SELECT acquisition_source_id, source_dataset, 'policy_filter' AS stage,
                           CASE WHEN reasons_json IN ('', '[]') THEN action ELSE reasons_json END AS reason,
                           count(*)::BIGINT AS documents,
                           sum(tokens_structural_cleaned - tokens_final)::BIGINT AS tokens_removed
                    FROM cleaning
                    WHERE action <> 'keep'
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
                    FROM decisions d JOIN cleaning c ON c.stable_uid = d.source_doc_id
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
            "decontamination_ledger": str(decontam_ledger.resolve()),
            "dedup_decisions": str(dedup_decisions.resolve()),
            "dedup_decisions_sha256": sha256_file(dedup_decisions),
        },
        "stage_totals": stage_rows,
        "source_stage_totals": source_totals,
        "events_by_source_and_reason": events,
        "events_global": [
            {"stage": stage, "reason": reason, **counts}
            for (stage, reason), counts in sorted(global_events.items())
        ],
        "invariants": {
            "decontam_coverage": {
                "cleaning_kept": int(coverage[0]),
                "decontam_covered": int(coverage[1]),
                "missing": int(coverage[2]),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--memory-limit", default="200GB")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    payload = build_waterfall(
        cleaning_ledger=args.cleaning_ledger,
        decontam_ledger=args.decontam_ledger,
        dedup_decisions=args.dedup_decisions,
        output=args.output,
        temporary_directory=args.temporary_directory,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    print(json.dumps({"ok": True, "output": str(args.output), "final_tokens": payload["invariants"]["final_tokens"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
