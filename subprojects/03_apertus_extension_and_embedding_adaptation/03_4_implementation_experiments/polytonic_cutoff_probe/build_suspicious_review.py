#!/usr/bin/env python3
"""Join structural-token audit results to actual corpus firing evidence."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    asset_manifest = json.loads(
        (args.run_root / "asset_manifest.json").read_text(encoding="utf-8")
    )
    cutoffs: dict[str, object] = {}
    for cutoff in (512, 1024):
        label = f"{cutoff:04d}"
        candidate = (
            args.run_root / "candidates" / f"c3p_poly_added_{label}"
        )
        audit_path = candidate / "byte_fragment_audit.json"
        coverage_path = (
            args.run_root / f"coverage_{label}" / "td_coverage_prepass.jsonl"
        )
        snippets_path = (
            args.run_root
            / f"coverage_{label}"
            / "td_snippet_index"
            / "snippets.jsonl"
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "passed":
            raise SystemExit(f"cutoff {cutoff}: byte-fragment audit did not pass")
        reviewed_ids = set(audit["reviewed_ids_in_candidate"])
        coverage = {
            row["new_token_id"]: row for row in read_jsonl(coverage_path)
        }
        surfaces: dict[int, list[str]] = {token_id: [] for token_id in reviewed_ids}
        for snippet in read_jsonl(snippets_path):
            token_id = snippet.get("new_token_id")
            surface = snippet.get("surface")
            if (
                token_id in reviewed_ids
                and isinstance(surface, str)
                and surface not in surfaces[token_id]
                and len(surfaces[token_id]) < 8
            ):
                surfaces[token_id].append(surface)

        rows = []
        for review in audit["reviews"]:
            token_id = review["id"]
            if token_id not in reviewed_ids:
                continue
            cov = coverage[token_id]
            if review.get("decision") != "keep_structural_bytelevel":
                raise SystemExit(
                    f"cutoff {cutoff}, id {token_id}: unexpected audit decision"
                )
            if cov.get("extended_firings", 0) <= 0 or not surfaces[token_id]:
                raise SystemExit(
                    f"cutoff {cutoff}, id {token_id}: no actual firing witness"
                )
            rows.append(
                {
                    "id": token_id,
                    "raw_bytes_hex": review["raw_bytes_hex"],
                    "decision": "keep",
                    "classification": "structural ByteLevel merge component",
                    "valid_utf8_descendants": review["valid_utf8_descendants"],
                    "actual_firings_in_30m_token_scan": cov["extended_firings"],
                    "docs_with_firing": cov["docs_with_firing"],
                    "coverage_status": cov["status"],
                    "actual_surfaces": surfaces[token_id],
                    "training_policy": (
                        "retain and merge-chain initialize; exclude from "
                        "standalone token-distillation targets"
                    ),
                }
            )
        cutoffs[str(cutoff)] = {
            "candidate_tokenizer_sha256": audit["candidate_tokenizer"]["sha256"],
            "audit_path": str(audit_path),
            "coverage_path": str(coverage_path),
            "reviewed_count": len(rows),
            "unresolved_count": len(audit["unresolved_ids_in_candidate"]),
            "tokens": rows,
        }

    report = {
        "schema_version": "polytonic-suspicious-token-review-v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "passed",
        "scope": (
            "all historically flagged byte-fragment IDs that occur in the "
            "+512 or +1024 production cutoffs"
        ),
        "corpus_evidence": {
            "source": "FineWeb-2 grc_Grek train",
            "tokens_scanned_per_candidate": 30_000_000,
            "ancient_train_jsonl": asset_manifest["files"][
                "probe_data/ancient_train.jsonl"
            ],
        },
        "decision": (
            "keep all flagged cutoff tokens; none is mojibake or an unresolved "
            "fragment, and all fire on valid polytonic surfaces"
        ),
        "cutoffs": cutoffs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
