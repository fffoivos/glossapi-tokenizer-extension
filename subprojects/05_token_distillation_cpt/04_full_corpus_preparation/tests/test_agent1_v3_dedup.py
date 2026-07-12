from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "agent1_v3_dedup.py"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _row(uid: str, *, base: bool, text: str) -> dict[str, object]:
    return {
        "stable_uid": uid,
        "source_role": "base" if base else "additive_candidate",
        "acquisition_source_id": "nanochat_base" if base else "candidate",
        "source_dataset": "base" if base else "candidate",
        "source_repo_id": "repo/base" if base else "repo/candidate",
        "source_revision": "a" if base else "b",
        "training_eligibility": "eligible_open",
        "needs_ocr": False,
        "ocr_success": True,
        "greek_badness_score": 1.0,
        "mojibake_badness_score": 1.0,
        "text": text,
        "normalized_text_sha256": _sha(text),
    }


def _run(*args: str) -> None:
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True)


def test_v3_reconciles_legacy_choice_to_nanochat_base(tmp_path: Path) -> None:
    text = "ίδιο ελληνικό κείμενο"
    normalized = tmp_path / "normalized"
    _write(normalized / "data.parquet", [_row("base", base=True, text=text), _row("candidate", base=False, text=text)])
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({
        "schema_version": "agent1_full_corpus_v3_source_admission_confirmation_v1",
        "status": "approved",
        "sources": [{"source_id": "candidate", "decision": "include"}],
    }), encoding="utf-8")
    pool = tmp_path / "pool"
    _run("prepare-pool", "--input", str(normalized), "--admission-confirmation", str(confirmation), "--output", str(pool), "--manifest", str(tmp_path / "pool.json"))

    raw = tmp_path / "raw.parquet"
    # Simulate an old deduper selecting candidate based on its old heuristic.
    _write(raw, [
        {"doc_key": "base-key", "source_doc_id": "base", "decision": "drop", "kept_doc_key": "candidate-key", "cluster_id": "old", "decision_stage": "near", "reason": "old"},
        {"doc_key": "candidate-key", "source_doc_id": "candidate", "decision": "keep", "kept_doc_key": "candidate-key", "cluster_id": "old", "decision_stage": "near", "reason": "old"},
    ])
    ledger = tmp_path / "ledger.parquet"
    _run("reconcile", "--pool", str(pool), "--raw-decisions", str(raw), "--output-ledger", str(ledger), "--work-database", str(tmp_path / "reconcile.sqlite"), "--manifest", str(tmp_path / "ledger.json"))

    decisions = {row["stable_uid"]: row for row in pq.read_table(ledger).to_pylist()}
    assert decisions["base"]["action"] == "keep"
    assert decisions["candidate"]["action"] == "drop"
    assert decisions["candidate"]["representative_stable_uid"] == "base"

    output = tmp_path / "survivors"
    _run("materialize", "--pool", str(pool), "--ledger", str(ledger), "--output", str(output), "--work-database", str(tmp_path / "materialize.sqlite"), "--manifest", str(tmp_path / "survivors.json"))
    assert [row["stable_uid"] for row in pq.read_table(output / "data.parquet").to_pylist()] == ["base"]


def test_pool_excludes_unadmitted_candidate_but_keeps_base(tmp_path: Path) -> None:
    text = "κείμενο"
    normalized = tmp_path / "normalized"
    _write(normalized / "data.parquet", [_row("base", base=True, text=text), _row("candidate", base=False, text=text)])
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({
        "schema_version": "agent1_full_corpus_v3_source_admission_confirmation_v1",
        "status": "approved",
        "sources": [{"source_id": "candidate", "decision": "exclude"}],
    }), encoding="utf-8")
    output = tmp_path / "pool"
    _run("prepare-pool", "--input", str(normalized), "--admission-confirmation", str(confirmation), "--output", str(output), "--manifest", str(tmp_path / "pool.json"))
    assert [row["stable_uid"] for row in pq.read_table(output / "data.parquet").to_pylist()] == ["base"]
