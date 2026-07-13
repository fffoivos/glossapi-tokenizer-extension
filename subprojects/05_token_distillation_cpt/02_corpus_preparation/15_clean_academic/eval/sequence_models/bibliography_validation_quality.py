#!/usr/bin/env python3
"""Audit extraction quality independently of bibliography predictions.

The quality decision is deliberately separated from evaluation.  Candidate
signals are computed only from source text and the canonical GlossAPI Rust
noise scorer.  A separately reviewed decisions file can then exclude unusable
documents before the frozen predictions are evaluated on a readability-
qualified subset.  The original validation result is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import statistics
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import evaluate_prediction
from .bibliography_entry_models import load_table


SCHEMA_VERSION = "bibliography-validation-quality-audit-v1"
DECISIONS_SCHEMA_VERSION = "bibliography-validation-quality-decisions-v1"
CRITERIA_VERSION = "text-only-conservative-v1"
CANONICAL_BADNESS_MAX = 60.0

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
GLYPH_RE = re.compile(
    r"(?:GLYPH(?:<|&lt;)\d+(?:>|&gt;)|/(?:a|pi|uni)\d+|\ufffd)", re.IGNORECASE
)


@dataclass(frozen=True)
class TextQuality:
    nonempty_lines: int
    character_count: int
    lexical_word_count: int
    median_words_per_line: float
    lines_at_most_one_word_fraction: float
    single_character_word_fraction: float
    glyph_placeholder_count: int
    glyph_placeholders_per_1000_chars: float
    suspicious_symbol_fraction: float
    longest_at_most_one_word_run: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"input row {row_number} is not an object")
            yield row


def analyze_text(lines: Sequence[str]) -> TextQuality:
    nonempty = [line.strip() for line in lines if line.strip()]
    word_rows = [WORD_RE.findall(line) for line in nonempty]
    word_counts = [len(words) for words in word_rows]
    all_words = [word for words in word_rows for word in words]
    text = "\n".join(nonempty)
    characters = len(text)
    glyph_count = len(GLYPH_RE.findall(text))
    suspicious_symbols = sum(
        unicodedata.category(character) in {"So", "Co", "Cs"} for character in text
    )
    longest_run = run = 0
    for count in word_counts:
        run = run + 1 if count <= 1 else 0
        longest_run = max(longest_run, run)
    return TextQuality(
        nonempty_lines=len(nonempty),
        character_count=characters,
        lexical_word_count=len(all_words),
        median_words_per_line=float(statistics.median(word_counts)) if word_counts else 0.0,
        lines_at_most_one_word_fraction=(
            sum(count <= 1 for count in word_counts) / len(word_counts)
            if word_counts
            else 1.0
        ),
        single_character_word_fraction=(
            sum(len(word) == 1 for word in all_words) / len(all_words)
            if all_words
            else 0.0
        ),
        glyph_placeholder_count=glyph_count,
        glyph_placeholders_per_1000_chars=1000.0 * glyph_count / max(characters, 1),
        suspicious_symbol_fraction=suspicious_symbols / max(characters, 1),
        longest_at_most_one_word_run=longest_run,
    )


def candidate_reasons(quality: TextQuality, rust_badness: float | None) -> list[str]:
    reasons: list[str] = []
    if rust_badness is not None and rust_badness > CANONICAL_BADNESS_MAX:
        reasons.append("canonical_greek_badness_gt_60")
    if (
        quality.nonempty_lines >= 500
        and quality.lines_at_most_one_word_fraction >= 0.80
        and quality.longest_at_most_one_word_run >= 100
    ):
        reasons.append("extreme_line_fragmentation")
    if (
        quality.lexical_word_count >= 500
        and quality.single_character_word_fraction >= 0.35
    ):
        reasons.append("character_spaced_extraction")
    if (
        quality.glyph_placeholder_count >= 50
        or quality.glyph_placeholders_per_1000_chars >= 2.0
    ):
        reasons.append("glyph_placeholder_corruption")
    return reasons


def _score_rust(text: str, module: Any) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        result = module.score_markdown_file_detailed(handle.name)
    if not result or len(result) != 24:
        raise RuntimeError("unexpected glossapi_rs_noise detailed result")
    keys = (
        "greek_badness_score",
        "latin_percentage",
        "table_line_ratio",
        "polytonic_word_ratio",
        "greek_character_count",
        "greek_word_count",
        "vowel_penalty_count",
        "consonant_penalty_count",
        "bad_double_count",
        "misplaced_sigma_count",
        "invalid_bigram_count",
        "long_word_count",
        "longest_word",
        "short_word_count",
        "maximum_run",
        "vowel_rate",
        "consonant_rate",
        "bad_double_rate",
        "sigma_end_rate",
        "bigram_rate",
        "long_word_rate",
        "short_word_ratio",
        "short_excess_per_1000",
        "flags",
    )
    return dict(zip(keys, result, strict=True))


def _load_decisions(path: Path | None) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    if path is None:
        return {}, None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("invalid quality-decisions schema")
    if value.get("criteria_version") != CRITERIA_VERSION:
        raise ValueError("quality decisions use different criteria")
    decisions: dict[str, Mapping[str, Any]] = {}
    for row in value.get("documents", []):
        document_id = str(row.get("document_id", ""))
        decision = str(row.get("decision", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", document_id):
            raise ValueError("invalid document ID in quality decisions")
        if decision not in {"exclude", "keep"}:
            raise ValueError("quality decision must be exclude or keep")
        if document_id in decisions:
            raise ValueError("duplicate document in quality decisions")
        decisions[document_id] = row
    return decisions, _sha256(path)


def _render_html(packet: Mapping[str, Any]) -> str:
    candidates = packet["candidates"]
    payload = json.dumps(candidates, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Validation extraction-quality review</title><style>
body{{margin:0;background:#f6f2e9;color:#242a26;font:14px/1.45 system-ui}}header{{position:sticky;top:0;background:#fffdf8ee;border-bottom:1px solid #d8d0c2;padding:12px 18px;z-index:3}}h1{{margin:0;font:700 22px Georgia,serif}}main{{max-width:1450px;margin:16px auto;padding:0 16px}}select{{padding:8px;border:1px solid #ccc3b5;border-radius:8px;background:white;min-width:420px}}.metrics{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}}.metric,.flag{{border-radius:7px;padding:5px 8px;background:#ebe5da}}.flag{{background:#f7d9d5;color:#7a2522;font-weight:700}}.reader{{background:#fffdf8;border:1px solid #d8d0c2;border-radius:12px;overflow:hidden}}.line{{display:grid;grid-template-columns:70px 1fr;gap:12px;padding:5px 12px;border-bottom:1px solid #eee9df}}.ln{{color:#918a7f;text-align:right;font:11px ui-monospace,monospace}}.text{{font:15px/1.45 Georgia,serif;overflow-wrap:anywhere}}.decision{{font-weight:800;margin-left:12px}} </style></head><body><header><h1>Validation extraction-quality review</h1><select id="pick"></select><span id="decision" class="decision"></span></header><main><div id="metrics" class="metrics"></div><section id="reader" class="reader"></section></main><script>
const DOCS={payload}; const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); const pick=document.getElementById('pick');
DOCS.forEach((d,i)=>pick.add(new Option(`${{i+1}}. ${{d.source}} — ${{d.document_id.slice(0,12)}} — ${{d.candidate_reasons.join(', ')}}`,i)));
function render(){{const d=DOCS[+pick.value||0],q=d.text_quality,r=d.rust_metrics||{{}};document.getElementById('decision').textContent=d.decision?d.decision.toUpperCase():"UNDECIDED";document.getElementById('metrics').innerHTML=[...d.candidate_reasons.map(x=>`<span class="flag">${{esc(x)}}</span>`),`<span class="metric">Rust badness ${{r.greek_badness_score?.toFixed?.(2)??'n/a'}}</span>`,`<span class="metric">${{q.nonempty_lines}} lines</span>`,`<span class="metric">${{(100*q.lines_at_most_one_word_fraction).toFixed(1)}}% ≤1 word</span>`,`<span class="metric">${{(100*q.single_character_word_fraction).toFixed(1)}}% one-character words</span>`,`<span class="metric">${{q.glyph_placeholder_count}} glyph placeholders</span>`].join('');document.getElementById('reader').innerHTML=d.lines.map(x=>`<div class="line"><span class="ln">L${{x.abs_idx}}</span><span class="text">${{esc(x.text)}}</span></div>`).join('')}}pick.onchange=render;render();</script></body></html>'''


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    validation_root = Path(args.validation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    decisions_path = Path(args.decisions).resolve() if args.decisions else None
    decisions, decisions_sha = _load_decisions(decisions_path)
    rust_module = importlib.import_module("glossapi_rs_noise")

    table = load_table(validation_root / "validation_table", expected_split="validation")
    validation_ids = {str(row["document_id"]): index for index, row in enumerate(table.documents)}
    documents: list[dict[str, Any]] = []
    observed: set[str] = set()
    for row in _iter_rows(input_path):
        if row.get("split") != "validation":
            continue
        document_id = str(row.get("document_id", ""))
        if document_id not in validation_ids:
            raise ValueError(f"validation input/table mismatch: {document_id}")
        raw_lines = row.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError(f"{document_id}: missing line inventory")
        line_rows = [
            {"abs_idx": int(line["abs_idx"]), "text": str(line["text"])}
            for line in raw_lines
        ]
        text_lines = [line["text"] for line in line_rows]
        quality = analyze_text(text_lines)
        rust = _score_rust("\n".join(text_lines), rust_module)
        reasons = candidate_reasons(quality, float(rust["greek_badness_score"]))
        decision = decisions.get(document_id)
        documents.append(
            {
                "document_id": document_id,
                "source": str(row.get("source", "")),
                "text_quality": asdict(quality),
                "rust_metrics": rust,
                "candidate_reasons": reasons,
                "decision": decision.get("decision") if decision else None,
                "decision_reason": decision.get("reason") if decision else None,
                "lines": line_rows if reasons else [],
            }
        )
        observed.add(document_id)
    if observed != set(validation_ids):
        raise ValueError("validation document inventory mismatch")
    unknown_decisions = set(decisions) - observed
    if unknown_decisions:
        raise ValueError(f"quality decisions contain absent documents: {sorted(unknown_decisions)}")

    candidates = [row for row in documents if row["candidate_reasons"]]
    excluded = {
        document_id for document_id, row in decisions.items() if row["decision"] == "exclude"
    }
    comparison = None
    if decisions:
        prediction = np.load(
            validation_root / "validation_block_prediction.npy", mmap_mode="r", allow_pickle=False
        )
        included_indices = {
            index for document_id, index in validation_ids.items() if document_id not in excluded
        }
        comparison = {
            "original": evaluate_prediction(table, prediction),
            "readability_qualified": evaluate_prediction(
                table, prediction, document_subset=included_indices
            ),
            "excluded_document_count": len(excluded),
            "included_document_count": len(included_indices),
        }

    packet = {
        "schema_version": SCHEMA_VERSION,
        "criteria_version": CRITERIA_VERSION,
        "input_sha256": _sha256(input_path),
        "validation_report_sha256": _sha256(validation_root / "validation_report.json"),
        "glossapi_rs_noise_version": str(getattr(rust_module, "__version__", "0.1.0")),
        "document_count": len(documents),
        "candidate_count": len(candidates),
        "excluded_document_count": len(excluded),
        "decisions_sha256": decisions_sha,
        "candidates": candidates,
        "comparison": comparison,
    }
    _write_json(output_dir / "quality_audit.json", packet)
    (output_dir / "index.html").write_text(_render_html(packet), encoding="utf-8")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "quality_audit_sha256": _sha256(output_dir / "quality_audit.json"),
        "index_sha256": _sha256(output_dir / "index.html"),
        "prediction_blind_quality_screen": True,
        "labels_used_only_after_locked_decisions_for_metric_recalculation": True,
    }
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decisions")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
