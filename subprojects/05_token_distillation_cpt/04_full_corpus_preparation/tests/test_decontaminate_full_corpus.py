from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from decontaminate_full_corpus import (  # noqa: E402
    BenchmarkIndex,
    BenchmarkItem,
    load_benchmark_index,
    match_document,
    minhash_signature,
    tokenize,
    kgrams,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_is_mandatory_and_checksum_bound(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    query = {
        "benchmark": "greekmmlu",
        "split": "test",
        "example_id": "q1",
        "question": "Ποια είναι η πρωτεύουσα της Ελλάδας;",
        "choices": ["Αθήνα", "Πάτρα", "Λάρισα", "Βόλος"],
        "answer_index": 0,
    }
    queries.write_text(json.dumps(query, ensure_ascii=False) + "\n", encoding="utf-8")
    missing = tmp_path / "missing.manifest.json"
    with pytest.raises(FileNotFoundError, match="required manifest"):
        load_benchmark_index(
            queries,
            missing,
            k=3,
            min_coverage=0.85,
            minhash_threshold=0.85,
            min_matched_grams=2,
            max_gap_tokens=20,
        )
    manifest = tmp_path / "queries.jsonl.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "greekmmlu_query_manifest_v1",
                "benchmark_id": "greekmmlu",
                "dataset_repo_id": "dascim/GreekMMLU",
                "dataset_revision": "0123456789abcdef",
                "required_splits": ["test"],
                "queries_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_benchmark_index(
            queries,
            manifest,
            k=3,
            min_coverage=0.85,
            minhash_threshold=0.85,
            min_matched_grams=2,
            max_gap_tokens=20,
        )
    payload = json.loads(manifest.read_text())
    payload["queries_sha256"] = _sha(queries)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    index, receipt = load_benchmark_index(
        queries,
        manifest,
        k=3,
        min_coverage=0.85,
        minhash_threshold=0.85,
        min_matched_grams=2,
        max_gap_tokens=20,
    )
    assert len(index.items) == 1
    assert receipt["observed_splits"] == ["test"]


def _index(question: str, answer: str, prompt: str = "", *, k: int = 8) -> BenchmarkIndex:
    question_tokens = tokenize(question)
    grams = tuple(dict.fromkeys(gram for _, gram in kgrams(question_tokens, k)))
    surfaces = (("eval_prompt", tokenize(prompt)),) if prompt else ()
    item = BenchmarkItem(
        index=0,
        item_id="item-1",
        split="test",
        subject="demo",
        question_tokens=question_tokens,
        answer_tokens=tokenize(answer),
        question_grams=grams,
        question_signature=minhash_signature(grams),
        prompt_surfaces=surfaces,
    )
    qindex: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for offset, gram in kgrams(question_tokens, k):
        qindex.setdefault(gram, []).append((0, offset))
    pindex = {surfaces[0][1][:k]: ((0, "eval_prompt", surfaces[0][1]),)} if surfaces else {}
    return BenchmarkIndex(
        items=(item,),
        qgram_index={key: tuple(value) for key, value in qindex.items()},
        prompt_anchor_index=pindex,
        k=k,
        min_coverage=0.85,
        minhash_threshold=0.85,
        min_matched_grams=4,
        max_gap_tokens=40,
    )


def test_only_high_confidence_question_bound_rules_drop() -> None:
    question = "Ποια είναι η πρωτεύουσα της Ελλάδας και ποια πόλη αποτελεί το μεγαλύτερο διοικητικό της κέντρο σήμερα;"
    prompt = question + "\nΑθήνα\nΠάτρα\nΛάρισα\nΗράκλειο\nΕπιλέξτε μία απάντηση από τις παραπάνω επιλογές."
    index = _index(question, "Αθήνα", prompt)
    action, reason, _ = match_document("Η Αθήνα είναι μεγάλη πόλη αλλά εδώ δεν υπάρχει ερώτηση.", index)
    assert (action, reason) == ("keep", "no_high_confidence_match")
    action, reason, evidence = match_document(question + " Η σωστή απάντηση είναι Αθήνα.", index)
    assert (action, reason) == ("drop", "greekmmlu_exact_question_answer")
    assert evidence[0]["method"] == "exact_question_answer"
    action, reason, _ = match_document(question + " Δεν παρατίθεται απάντηση.", index)
    assert (action, reason) == ("keep", "no_high_confidence_match")
    action, reason, evidence = match_document("Πρόλογος. " + prompt + " Επίλογος.", index)
    assert (action, reason) == ("drop", "greekmmlu_exact_prompt")
    assert evidence[0]["surface"] == "eval_prompt"


def test_conservative_aligned_ngram_minhash_requires_nearby_answer() -> None:
    words = [f"λέξη{i}" for i in range(220)]
    question = " ".join(words)
    index = _index(question, "Αθήνα")
    approximate = words.copy()
    approximate[100] = "διαφορετική"
    action, reason, evidence = match_document(" ".join(approximate) + " απάντηση Αθήνα", index)
    assert (action, reason) == ("drop", "greekmmlu_ngram_minhash_answer")
    assert evidence[0]["question_coverage"] >= 0.85
    action, reason, _ = match_document(" ".join(approximate) + " χωρίς αποτέλεσμα", index)
    assert (action, reason) == ("keep", "no_high_confidence_match")


def test_cli_streams_kept_dropped_and_hash_bound_ledger(tmp_path: Path) -> None:
    question = "Ποια είναι η πρωτεύουσα της Ελλάδας και ποια πόλη αποτελεί το μεγαλύτερο διοικητικό της κέντρο σήμερα;"
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "benchmark": "greekmmlu",
                "split": "test",
                "example_id": "q1",
                "question": question,
                "choices": ["Αθήνα", "Πάτρα", "Λάρισα", "Βόλος"],
                "answer_index": 0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = Path(str(queries) + ".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "greekmmlu_query_manifest_v1",
                "benchmark_id": "greekmmlu",
                "dataset_repo_id": "dascim/GreekMMLU",
                "dataset_revision": "0123456789abcdef",
                "required_splits": ["test"],
                "queries_sha256": _sha(queries),
            }
        ),
        encoding="utf-8",
    )
    kept_text = "Η Αθήνα αναφέρεται μόνη της, χωρίς κανένα κείμενο ερώτησης."
    dropped_text = question + " Η σωστή απάντηση είναι Αθήνα."
    input_root = tmp_path / "input"
    input_path = input_root / "part.parquet"
    input_path.parent.mkdir()
    rows = [
        {
            "stable_uid": "keep",
            "acquisition_source_id": "demo",
            "source_dataset": "demo_source",
            "source_doc_id": "upstream-keep",
            "text": kept_text,
            "cleaned_text_sha256": hashlib.sha256(kept_text.encode()).hexdigest(),
        },
        {
            "stable_uid": "drop",
            "acquisition_source_id": "demo",
            "source_dataset": "demo_source",
            "source_doc_id": "upstream-drop",
            "text": dropped_text,
            "cleaned_text_sha256": hashlib.sha256(dropped_text.encode()).hexdigest(),
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), input_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "decontaminate_full_corpus.py"),
            "--input",
            str(input_root),
            "--output",
            str(tmp_path / "output"),
            "--dropped",
            str(tmp_path / "dropped"),
            "--ledger",
            str(tmp_path / "ledger"),
            "--manifest",
            str(tmp_path / "run.json"),
            "--queries-jsonl",
            str(queries),
            "--workers",
            "1",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert pq.read_table(tmp_path / "output" / "part.parquet")["stable_uid"].to_pylist() == ["keep"]
    assert pq.read_table(tmp_path / "dropped" / "part.parquet")["stable_uid"].to_pylist() == ["drop"]
    ledger = {row["stable_uid"]: row for row in pq.read_table(tmp_path / "ledger" / "part.parquet").to_pylist()}
    assert ledger["keep"]["action"] == "keep"
    assert ledger["drop"]["reason"] == "greekmmlu_exact_question_answer"
    assert ledger["drop"]["input_text_sha256"] == hashlib.sha256(dropped_text.encode()).hexdigest()
