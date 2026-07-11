from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LOSS = load_module("phase04_structural_token_loss", HERE / "scripts" / "structural_token_loss.py")
VALIDATE = load_module("phase04_validate_configs", HERE / "scripts" / "validate_configs.py")


def span(doc_id: str, kind: str, start: int, end: int, line_start: int, line_end: int, text: str):
    source = "fixture"
    return LOSS.Span(
        source,
        doc_id,
        hashlib.sha256(f"{source}\0{doc_id}".encode()).hexdigest(),
        hashlib.sha256(text.encode()).hexdigest(),
        len(text),
        "fixture-model",
        kind,
        start,
        end,
        line_start,
        line_end,
        "fixture",
        "fixture",
    )


def test_apply_spans_merges_overlap_and_preserves_retained_text() -> None:
    text = "αρχή\nΠΕΡΙΕΧΟΜΕΝΑ\nκεφάλαιο .... 2\nσώμα\nΒΙΒΛΙΟΓΡΑΦΙΑ\nSmith 2020\nτέλος"
    toc_start = text.index("ΠΕΡΙΕΧΟΜΕΝΑ")
    toc_end = text.index("σώμα") - 1
    bib_start = text.index("ΒΙΒΛΙΟΓΡΑΦΙΑ")
    bib_end = text.index("τέλος") - 1
    spans = [
        span("d", LOSS.TOC_KIND, toc_start, toc_end, 1, 2, text),
        # Overlap a second ToC record to prove union, not double removal.
        span("d", LOSS.TOC_KIND, toc_start + 3, toc_end, 1, 2, text),
        span("d", LOSS.BIB_KIND, bib_start, bib_end, 4, 5, text),
    ]

    LOSS.validate_spans(text, spans, doc_id="d")
    toc_only = LOSS.apply_spans(text, spans, {LOSS.TOC_KIND})
    both = LOSS.apply_spans(text, spans, {LOSS.TOC_KIND, LOSS.BIB_KIND})

    assert "ΠΕΡΙΕΧΟΜΕΝΑ" not in toc_only
    assert "ΒΙΒΛΙΟΓΡΑΦΙΑ" in toc_only
    assert "ΠΕΡΙΕΧΟΜΕΝΑ" not in both
    assert "ΒΙΒΛΙΟΓΡΑΦΙΑ" not in both
    assert "αρχή" in both and "σώμα" in both and "τέλος" in both


def test_unicode_offsets_and_line_validation_are_codepoint_based() -> None:
    text = "ASCII\nἙλληνικὸ κείμενο\nτέλος"
    start = text.index("Ἑλληνικὸ")
    end = start + len("Ἑλληνικὸ κείμενο")
    good = span("poly", LOSS.BIB_KIND, start, end, 1, 1, text)
    LOSS.validate_spans(text, [good], doc_id="poly")
    assert LOSS.apply_spans(text, [good], {LOSS.BIB_KIND}).endswith("τέλος")

    bad = span("poly", LOSS.BIB_KIND, start, end, 0, 1, text)
    with pytest.raises(ValueError, match="line_start"):
        LOSS.validate_spans(text, [bad], doc_id="poly")


def test_script_counts_separate_polytonic_and_latin() -> None:
    greek, latin, polytonic = LOSS.script_counts("ἱστορίαι Smith Αθήνα")
    assert greek > 0
    assert latin == 5
    assert polytonic > 0


def test_span_ledger_row_uid_must_bind_source_and_document(tmp_path: Path) -> None:
    path = tmp_path / "spans.jsonl"
    path.write_text(
        json.dumps(
            {
                "source": "fixture",
                "doc_id": "d",
                "row_uid": "0" * 64,
                "original_sha256": "0" * 64,
                "original_chars": 1,
                "model_id": "fixture",
                "kind": "bib_span",
                "char_start": 0,
                "char_end": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row_uid"):
        LOSS.load_spans(path)


def test_tracked_configs_validate() -> None:
    sources = VALIDATE.load_json(HERE / "configs" / "sources.json")
    backlog = VALIDATE.load_json(HERE / "configs" / "source_backlog.json")
    policy = VALIDATE.load_json(HERE / "configs" / "cleaning_policy.json")
    assert VALIDATE.validate_sources(sources) == []
    assert VALIDATE.validate_backlog(backlog, sources) == []
    assert VALIDATE.validate_policy(policy) == []


def test_cli_exact_counterfactual_smoke(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = tokenizers.Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "body": 1,
                "contents": 2,
                "chapter": 3,
                "bibliography": 4,
                "smith": 5,
                "end": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    text = "body\ncontents\nchapter\nbibliography\nsmith\nend"
    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.table({"source_doc_id": ["doc-1"], "text": [text]}),
        input_path,
    )
    spans_path = tmp_path / "spans.jsonl"
    toc_start = text.index("contents")
    toc_end = text.index("bibliography") - 1
    bib_start = text.index("bibliography")
    bib_end = text.index("end") - 1
    spans_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source": "fixture",
                        "doc_id": "doc-1",
                        "row_uid": hashlib.sha256(b"fixture\0doc-1").hexdigest(),
                        "original_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "original_chars": len(text),
                        "model_id": "fixture-toc",
                        "kind": "toc_span",
                        "char_start": toc_start,
                        "char_end": toc_end,
                        "line_start": 1,
                        "line_end": 2,
                    }
                ),
                json.dumps(
                    {
                        "source": "fixture",
                        "doc_id": "doc-1",
                        "row_uid": hashlib.sha256(b"fixture\0doc-1").hexdigest(),
                        "original_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "original_chars": len(text),
                        "model_id": "fixture-bib",
                        "kind": "bib_span",
                        "char_start": bib_start,
                        "char_end": bib_end,
                        "line_start": 3,
                        "line_end": 4,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    counters_path = tmp_path / "structure_counters.jsonl"
    counters_path.write_text(
        json.dumps(
            {
                "source": "fixture",
                "doc_id": "doc-1",
                "row_uid": hashlib.sha256(b"fixture\0doc-1").hexdigest(),
                "original_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "original_chars": len(text),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    detector_manifest = tmp_path / "detector_run_manifest.json"
    input_receipt = tmp_path / "input_receipt.json"
    input_receipt.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_acquisition_receipt_v1",
                "status": "passed",
            }
        ),
        encoding="utf-8",
    )
    detector_manifest.write_text(
        json.dumps(
            {
                "schema_version": "structural_detector_run_v1",
                "status": "passed",
                "source": "fixture",
                "binary_sha256": "f" * 64,
                "input_receipt_sha256": LOSS.sha256_file(input_receipt),
                "stream_manifest": {"source_regex": None},
                "spans": {"sha256": LOSS.sha256_file(spans_path)},
                "counters": {
                    "path": counters_path.name,
                    "sha256": LOSS.sha256_file(counters_path),
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "structural_token_loss.py"),
            "--input",
            str(input_path),
            "--spans",
            str(spans_path),
            "--detector-run-manifest",
            str(detector_manifest),
            "--input-receipt",
            str(input_receipt),
            "--tokenizer-json",
            str(tokenizer_path),
            "--text-column",
            "text",
            "--id-column",
            "source_doc_id",
            "--source-column",
            "",
            "--source-name",
            "fixture",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert '"ok": true' in result.stdout.lower()
    summary = json.loads((output_dir / "structural_token_loss_summary.json").read_text(encoding="utf-8"))
    source = summary["by_source"][0]
    assert source["tokens_before"] == 6
    assert source["tokens_removed_bib"] == 2
    assert source["tokens_removed_toc"] == 2
    assert source["tokens_removed_both"] == 4
    assert source["interaction_tokens"] == 0
