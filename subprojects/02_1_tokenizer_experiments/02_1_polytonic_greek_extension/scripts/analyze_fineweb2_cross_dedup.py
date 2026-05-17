#!/usr/bin/env python3
"""Analyze cross-dedup and metadata overlap with FineWeb-2."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import duckdb
import pyarrow.parquet as pq


def q(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def safe_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def collect_urls(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                if value.startswith(("http://", "https://")):
                    out.append(value)
                elif "url" in key.lower() and value.startswith(("http://", "https://")):
                    out.append(value)
            else:
                collect_urls(value, out)
    elif isinstance(obj, list):
        for item in obj:
            collect_urls(item, out)


def host(url: str) -> str:
    if not url:
        return ""
    h = urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def canonical_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    h = parsed.netloc.lower()
    if h.startswith("www."):
        h = h[4:]
    path = unquote(parsed.path or "/").rstrip("/")
    return f"{h}{path}".casefold()


def bucket_domain(domain: str) -> str:
    d = domain.lower()
    if "wikisource.org" in d:
        return "wikisource"
    if "perseus" in d or "perseids.org" in d:
        return "perseus"
    if "goarch.org" in d:
        return "goarch"
    if any(x in d for x in ["bible", "bibbia", "sacred-texts", "catholic", "orthodox", "agia-grafi", "textusreceptus", "bibel"]):
        return "biblical_church"
    if any(x in d for x in ["digitalathenaeus", "hellas.bab2min", "skuolasprint"]):
        return "classical_other"
    return "other"


def metadata_overlap(ours: Path, fineweb: Path) -> dict[str, Any]:
    fine_urls: dict[str, list[dict[str, str]]] = defaultdict(list)
    fine_domains: Counter[str] = Counter()
    fine_buckets: Counter[str] = Counter()
    pf = pq.ParquetFile(fineweb)
    for batch in pf.iter_batches(columns=["id", "url", "text"], batch_size=1024):
        data = batch.to_pydict()
        for doc_id, url in zip(data["id"], data["url"], strict=True):
            url = url or ""
            c = canonical_url(url)
            d = host(url)
            if c:
                fine_urls[c].append({"id": str(doc_id), "url": url, "domain": d})
            if d:
                fine_domains[d] += 1
                fine_buckets[bucket_domain(d)] += 1

    our_domains: Counter[str] = Counter()
    our_source_domains: dict[str, Counter[str]] = defaultdict(Counter)
    exact_url_matches: list[dict[str, str]] = []
    our_docs_with_url = 0
    our_pf = pq.ParquetFile(ours)
    cols = ["source_dataset", "source_doc_id", "source_metadata_json", "title", "author"]
    for batch in our_pf.iter_batches(columns=cols, batch_size=1024):
        data = batch.to_pydict()
        for i, meta_s in enumerate(data["source_metadata_json"]):
            source = data["source_dataset"][i] or ""
            doc_id = data["source_doc_id"][i] or ""
            meta = safe_json(meta_s)
            urls: list[str] = []
            collect_urls(meta, urls)
            if urls:
                our_docs_with_url += 1
            seen_domains = set()
            for url in urls:
                d = host(url)
                if d and d not in seen_domains:
                    our_domains[d] += 1
                    our_source_domains[source][d] += 1
                    seen_domains.add(d)
                c = canonical_url(url)
                for hit in fine_urls.get(c, []):
                    exact_url_matches.append(
                        {
                            "our_source": source,
                            "our_doc_id": str(doc_id),
                            "our_title": str(data["title"][i] or ""),
                            "our_author": str(data["author"][i] or ""),
                            "our_url": url,
                            "fineweb_id": hit["id"],
                            "fineweb_url": hit["url"],
                            "fineweb_domain": hit["domain"],
                        }
                    )

    return {
        "fineweb_top_domains": dict(fine_domains.most_common(80)),
        "fineweb_domain_buckets": dict(fine_buckets.most_common()),
        "our_docs_with_url_metadata": our_docs_with_url,
        "our_top_metadata_domains": dict(our_domains.most_common(80)),
        "our_source_domains": {
            source: dict(counter.most_common(25))
            for source, counter in sorted(our_source_domains.items())
        },
        "exact_canonical_url_match_count": len(exact_url_matches),
        "exact_canonical_url_match_samples": exact_url_matches[:100],
    }


def dedup_analysis(run_root: Path) -> dict[str, Any]:
    con = duckdb.connect()
    decisions = run_root / "final" / "dedup_decisions.parquet"
    con.execute(f"create or replace view decisions as select * from read_parquet({q(decisions)})")
    summary = {
        "by_source_decision": con.execute(
            """
            select source_dataset, decision, count(*) as rows
            from decisions
            group by 1,2
            order by source_dataset, decision
            """
        ).fetchdf().to_dict("records"),
        "drops_by_source_stage": con.execute(
            """
            select source_dataset, decision_stage, count(*) as rows
            from decisions
            where decision = 'drop'
            group by 1,2
            order by source_dataset, decision_stage
            """
        ).fetchdf().to_dict("records"),
        "cross_source_family_summary": con.execute(
            """
            with fam as (
              select
                cluster_id,
                count(*) as family_size,
                count(distinct source_dataset) as source_count,
                bool_or(source_dataset = 'fineweb2_main_grc_Grek') as has_fineweb,
                bool_or(source_dataset <> 'fineweb2_main_grc_Grek') as has_ours
              from decisions
              group by 1
            )
            select
              count(*) filter (where source_count > 1) as cross_source_families,
              sum(family_size) filter (where source_count > 1) as cross_source_rows,
              count(*) filter (where has_fineweb and has_ours) as fineweb_ours_families,
              sum(family_size) filter (where has_fineweb and has_ours) as fineweb_ours_rows
            from fam
            """
        ).fetchdf().to_dict("records")[0],
        "fineweb_mixed_families_by_our_source": con.execute(
            """
            with fw_fams as (
              select distinct cluster_id
              from decisions
              where source_dataset = 'fineweb2_main_grc_Grek'
            )
            select d.source_dataset, count(*) as rows, count(distinct d.cluster_id) as families
            from decisions d
            join fw_fams using(cluster_id)
            where d.source_dataset <> 'fineweb2_main_grc_Grek'
            group by 1
            order by rows desc
            """
        ).fetchdf().to_dict("records"),
        "fineweb_mixed_families_top50": con.execute(
            """
            with fam as (
              select
                cluster_id,
                count(*) as family_size,
                string_agg(distinct source_dataset, ', ' order by source_dataset) as sources,
                sum(case when source_dataset = 'fineweb2_main_grc_Grek' then 1 else 0 end) as fineweb_rows,
                sum(case when source_dataset <> 'fineweb2_main_grc_Grek' then 1 else 0 end) as our_rows
              from decisions
              group by 1
            )
            select *
            from fam
            where fineweb_rows > 0 and our_rows > 0
            order by family_size desc, fineweb_rows desc
            limit 50
            """
        ).fetchdf().to_dict("records"),
        "drop_direction": con.execute(
            """
            with kept as (
              select doc_key as kept_doc_key, source_dataset as kept_source
              from decisions
              where decision = 'keep'
            )
            select
              d.source_dataset as dropped_source,
              coalesce(k.kept_source, 'unknown') as kept_source,
              d.decision_stage,
              count(*) as rows
            from decisions d
            left join kept k on d.kept_doc_key = k.kept_doc_key
            where d.decision = 'drop'
            group by 1,2,3
            order by rows desc
            """
        ).fetchdf().to_dict("records"),
    }
    return summary


def write_report(path: Path, result: dict[str, Any]) -> None:
    d = result["dedup"]
    m = result["metadata"]
    lines = [
        "# FineWeb-2 Cross-Dedup And Metadata Analysis",
        "",
        "## Cross-Dedup",
        "",
    ]
    c = d["cross_source_family_summary"]
    lines.extend(
        [
            f"- Cross-source families: `{c['cross_source_families']}`",
            f"- Cross-source rows: `{c['cross_source_rows']}`",
            f"- FineWeb + our-source families: `{c['fineweb_ours_families']}`",
            f"- FineWeb + our-source rows: `{c['fineweb_ours_rows']}`",
            "",
            "Drops by source/stage:",
            "",
        ]
    )
    for row in d["drops_by_source_stage"]:
        lines.append(f"- `{row['source_dataset']}` / `{row['decision_stage']}`: {row['rows']:,}")
    lines.extend(["", "FineWeb-overlap rows by our source:", ""])
    for row in d["fineweb_mixed_families_by_our_source"]:
        lines.append(f"- `{row['source_dataset']}`: {row['rows']:,} rows in {row['families']:,} families")
    lines.extend(["", "Drop direction:", ""])
    for row in d["drop_direction"]:
        lines.append(
            f"- dropped `{row['dropped_source']}` kept-by `{row['kept_source']}` via `{row['decision_stage']}`: {row['rows']:,}"
        )
    lines.extend(
        [
            "",
            "## Metadata",
            "",
            f"- Our docs with URL-like metadata: `{m['our_docs_with_url_metadata']}`",
            f"- Exact canonical URL matches: `{m['exact_canonical_url_match_count']}`",
            "",
            "FineWeb domain buckets:",
            "",
        ]
    )
    for bucket, count in m["fineweb_domain_buckets"].items():
        lines.append(f"- `{bucket}`: {count:,}")
    lines.extend(["", "FineWeb top domains:", ""])
    for domain, count in list(m["fineweb_top_domains"].items())[:30]:
        lines.append(f"- `{domain}`: {count:,}")
    lines.extend(["", "Our metadata domains:", ""])
    for domain, count in list(m["our_top_metadata_domains"].items())[:30]:
        lines.append(f"- `{domain}`: {count:,}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--fineweb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "dedup": dedup_analysis(args.run_root),
        "metadata": metadata_overlap(args.ours, args.fineweb),
    }
    (args.out_dir / "cross_dedup_metadata_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.out_dir / "cross_dedup_metadata_report.md", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
