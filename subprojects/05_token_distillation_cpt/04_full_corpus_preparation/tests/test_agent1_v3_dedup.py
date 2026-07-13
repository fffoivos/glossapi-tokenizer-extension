from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq


PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "agent1_v3_dedup.py"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _row(
    uid: str,
    *,
    base: bool,
    text: str,
    work_key: str | None = None,
    representation_generation: str = "canonical",
    source_id: str | None = None,
) -> dict[str, object]:
    resolved_source_id = source_id or ("nanochat_base" if base else "candidate")
    return {
        "stable_uid": uid,
        "source_role": "base" if base else "additive_candidate",
        "acquisition_source_id": resolved_source_id,
        "source_dataset": "base" if base else resolved_source_id,
        "source_repo_id": "repo/base" if base else f"repo/{resolved_source_id}",
        "source_revision": "a" if base else "b",
        "training_eligibility": "eligible_open",
        "needs_ocr": False,
        "ocr_success": True,
        "greek_badness_score": 1.0,
        "mojibake_badness_score": 1.0,
        "text": text,
        "normalized_text_sha256": _sha(text),
        "work_key": work_key or f"work-{uid}",
        "representation_generation": representation_generation,
    }


def _run(*args: str) -> None:
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True)


def _provisional_row(
    pool_row: dict[str, object],
    *,
    cluster_id: str,
    action: str,
    representative: dict[str, object],
) -> dict[str, object]:
    return {
        "stable_uid": pool_row["stable_uid"],
        "input_representation_id": pool_row["input_representation_id"],
        "input_text_sha256": pool_row["cleaned_text_sha256"],
        "action": action,
        "representative_stable_uid": representative["stable_uid"],
        "representative_input_representation_id": representative["input_representation_id"],
        "cluster_id": cluster_id,
        "method": "content_work_representation_near_precedence_v1",
        "raw_decision_stage": "near",
        "reason": "generic-near",
    }


def _raw_decision(uid: str, representative: str, cluster_id: str) -> dict[str, object]:
    return {
        "doc_key": f"doc:{uid}",
        "source_doc_id": uid,
        "decision": "keep" if uid == representative else "drop",
        "kept_doc_key": f"doc:{representative}",
        "cluster_id": cluster_id,
        "decision_stage": "near",
        "reason": "fixture",
    }


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


def test_identity_reconcile_closes_near_content_work_and_representation_before_precedence(tmp_path: Path) -> None:
    """A multi-hop graph must be closed before Nanochat wins selection.

    ``base`` connects to ``content`` by exact text; the generic detector joins
    ``content`` to ``work-left``; work identity joins both work variants; and
    the first two variants also carry the same representation generation.
    No single edge kind alone reaches the full component.
    """

    normalized = tmp_path / "normalized"
    _write(normalized / "data.parquet", [
        _row("base", base=True, text="same exact content", work_key="work-base"),
        _row("content", base=False, text="same exact content", work_key="work-content"),
        _row("work-left", base=False, text="left text", work_key="work-shared", representation_generation="edition-a"),
        _row("work-right", base=False, text="right text", work_key="work-shared", representation_generation="edition-a"),
        _row("work-variant", base=False, text="variant text", work_key="work-shared", representation_generation="edition-b"),
        _row("unrelated", base=False, text="unrelated text", work_key="work-unrelated"),
    ])
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({
        "schema_version": "agent1_full_corpus_v3_source_admission_confirmation_v1",
        "status": "approved",
        "sources": [{"source_id": "candidate", "decision": "include"}],
    }), encoding="utf-8")
    pool = tmp_path / "pool"
    _run(
        "prepare-pool", "--input", str(normalized),
        "--admission-confirmation", str(confirmation), "--output", str(pool),
        "--manifest", str(tmp_path / "pool.json"),
    )
    pool_rows = {row["stable_uid"]: row for row in pq.read_table(pool / "data.parquet").to_pylist()}
    provisional = tmp_path / "provisional.parquet"
    _write(provisional, [
        _provisional_row(pool_rows["base"], cluster_id="near-base", action="keep", representative=pool_rows["base"]),
        _provisional_row(pool_rows["content"], cluster_id="near-bridge", action="keep", representative=pool_rows["content"]),
        _provisional_row(pool_rows["work-left"], cluster_id="near-bridge", action="drop", representative=pool_rows["content"]),
        _provisional_row(pool_rows["work-right"], cluster_id="near-right", action="keep", representative=pool_rows["work-right"]),
        _provisional_row(pool_rows["work-variant"], cluster_id="near-variant", action="keep", representative=pool_rows["work-variant"]),
        _provisional_row(pool_rows["unrelated"], cluster_id="near-unrelated", action="keep", representative=pool_rows["unrelated"]),
    ])

    final_ledger = tmp_path / "identity-ledger.parquet"
    final_manifest = tmp_path / "identity-ledger.json"
    _run(
        "identity-reconcile", "--pool", str(pool), "--provisional-ledger", str(provisional),
        "--output-ledger", str(final_ledger), "--work-database", str(tmp_path / "identity.sqlite"),
        "--manifest", str(final_manifest),
    )

    decisions = {row["stable_uid"]: row for row in pq.read_table(final_ledger).to_pylist()}
    assert decisions["base"]["action"] == "keep"
    for uid in ("content", "work-left", "work-right", "work-variant"):
        assert decisions[uid]["action"] == "drop"
        assert decisions[uid]["representative_stable_uid"] == "base"
    assert decisions["unrelated"]["action"] == "keep"
    assert len({decisions[uid]["cluster_id"] for uid in ("base", "content", "work-left", "work-right", "work-variant")}) == 1

    manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
    identity = manifest["identity_reconciliation"]
    assert identity["closure"]["components"] == 2
    assert identity["closure"]["unresolved_memberships"] == 0
    assert identity["edge_groups"]["generic_near_component"]["linked_groups"] == 1
    assert identity["edge_groups"]["exact_content"]["linked_groups"] == 1
    assert identity["edge_groups"]["exact_work"]["linked_groups"] == 1
    assert identity["edge_groups"]["exact_representation"]["linked_groups"] == 1

    survivors = tmp_path / "survivors"
    _run(
        "materialize", "--pool", str(pool), "--ledger", str(final_ledger),
        "--output", str(survivors), "--work-database", str(tmp_path / "materialize.sqlite"),
        "--manifest", str(tmp_path / "survivors.json"),
    )
    assert [row["stable_uid"] for row in pq.read_table(survivors / "data.parquet").to_pylist()] == ["base", "unrelated"]


def test_identity_reconcile_and_materialize_refuse_pool_rows_outside_ledger(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    _write(normalized / "data.parquet", [
        _row("base", base=True, text="base"),
        _row("candidate", base=False, text="candidate"),
    ])
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({
        "schema_version": "agent1_full_corpus_v3_source_admission_confirmation_v1",
        "status": "approved",
        "sources": [{"source_id": "candidate", "decision": "include"}],
    }), encoding="utf-8")
    pool = tmp_path / "pool"
    _run(
        "prepare-pool", "--input", str(normalized),
        "--admission-confirmation", str(confirmation), "--output", str(pool),
        "--manifest", str(tmp_path / "pool.json"),
    )
    base = pq.read_table(pool / "data.parquet").to_pylist()[0]
    incomplete = tmp_path / "incomplete.parquet"
    _write(incomplete, [
        _provisional_row(base, cluster_id="near-base", action="keep", representative=base),
    ])

    with pytest.raises(subprocess.CalledProcessError):
        _run(
            "identity-reconcile", "--pool", str(pool), "--provisional-ledger", str(incomplete),
            "--output-ledger", str(tmp_path / "identity-ledger.parquet"),
            "--work-database", str(tmp_path / "identity.sqlite"), "--manifest", str(tmp_path / "identity.json"),
        )

    with pytest.raises(subprocess.CalledProcessError):
        _run(
            "materialize", "--pool", str(pool), "--ledger", str(incomplete),
            "--output", str(tmp_path / "survivors"), "--work-database", str(tmp_path / "materialize.sqlite"),
            "--manifest", str(tmp_path / "survivors.json"),
        )


def test_ordered_exact_then_within_cross_and_candidate_to_base_ledgers_compose(tmp_path: Path) -> None:
    """Stage 50 must retain four separately receipt-bound, deterministic passes."""

    normalized = tmp_path / "normalized"
    _write(normalized / "data.parquet", [
        _row("base", base=True, text="nanochat text", work_key="base-work"),
        _row("exact-clone", base=False, source_id="candidate-a", text="nanochat text", work_key="clone-work"),
        _row("a-within-a-keep", base=False, source_id="candidate-a", text="within keep", work_key="within-keep"),
        _row("a-within-z-drop", base=False, source_id="candidate-a", text="within drop", work_key="within-drop"),
        _row("a-cross", base=False, source_id="candidate-a", text="cross left", work_key="cross-left"),
        _row("b-cross", base=False, source_id="candidate-b", text="cross right", work_key="cross-right"),
        _row("a-base", base=False, source_id="candidate-a", text="base overlap", work_key="base-overlap"),
    ])
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({
        "schema_version": "agent1_full_corpus_v3_source_admission_confirmation_v1",
        "status": "approved",
        "sources": [
            {"source_id": "candidate-a", "decision": "include"},
            {"source_id": "candidate-b", "decision": "include"},
        ],
    }), encoding="utf-8")
    pool = tmp_path / "pool"
    _run(
        "prepare-pool", "--input", str(normalized), "--admission-confirmation", str(confirmation),
        "--output", str(pool), "--manifest", str(tmp_path / "pool.json"),
    )

    exact = tmp_path / "exact.parquet"
    exact_manifest = tmp_path / "exact.json"
    _run(
        "exact-reconcile", "--pool", str(pool), "--output-ledger", str(exact),
        "--work-database", str(tmp_path / "exact.sqlite"), "--manifest", str(exact_manifest),
    )
    exact_rows = {row["stable_uid"]: row for row in pq.read_table(exact).to_pylist()}
    assert exact_rows["exact-clone"]["action"] == "drop"
    assert exact_rows["exact-clone"]["representative_stable_uid"] == "base"
    exact_survivors = tmp_path / "exact-survivors"
    _run(
        "materialize", "--pool", str(pool), "--ledger", str(exact), "--output", str(exact_survivors),
        "--work-database", str(tmp_path / "exact-materialize.sqlite"), "--manifest", str(tmp_path / "exact-materialize.json"),
    )
    within_partition_manifest = tmp_path / "within-partition.json"
    _run(
        "partition-within-source", "--input", str(exact_survivors), "--output", str(tmp_path / "within-partition"),
        "--manifest", str(within_partition_manifest),
    )
    partition = json.loads(within_partition_manifest.read_text(encoding="utf-8"))
    assert [row["acquisition_source_id"] for row in partition["sources"]] == [
        "candidate-a", "candidate-b", "nanochat_base",
    ]

    within_raw_a = tmp_path / "within-a.parquet"
    within_raw_b = tmp_path / "within-b.parquet"
    within_raw_base = tmp_path / "within-base.parquet"
    _write(within_raw_a, [
        _raw_decision("a-within-a-keep", "a-within-a-keep", "within-a"),
        _raw_decision("a-within-z-drop", "a-within-a-keep", "within-a"),
        _raw_decision("a-cross", "a-cross", "within-a-cross"),
        _raw_decision("a-base", "a-base", "within-a-base"),
    ])
    _write(within_raw_b, [_raw_decision("b-cross", "b-cross", "within-b")])
    _write(within_raw_base, [_raw_decision("base", "base", "within-base")])
    within = tmp_path / "within.parquet"
    within_manifest = tmp_path / "within.json"
    _run(
        "near-reconcile", "--pool", str(exact_survivors), "--pass-kind", "within-source",
        "--raw-decisions", str(within_raw_a), "--raw-decisions", str(within_raw_b),
        "--raw-decisions", str(within_raw_base), "--output-ledger", str(within),
        "--work-database", str(tmp_path / "within.sqlite"), "--manifest", str(within_manifest),
    )
    within_survivors = tmp_path / "within-survivors"
    _run(
        "materialize", "--pool", str(exact_survivors), "--ledger", str(within), "--output", str(within_survivors),
        "--work-database", str(tmp_path / "within-materialize.sqlite"), "--manifest", str(tmp_path / "within-materialize.json"),
    )

    candidate_scope = tmp_path / "candidate-scope"
    _run(
        "filter-candidates", "--input", str(within_survivors), "--output", str(candidate_scope),
        "--manifest", str(tmp_path / "candidate-scope.json"),
    )
    cross_raw = tmp_path / "cross.parquet"
    _write(cross_raw, [
        _raw_decision("a-within-a-keep", "a-within-a-keep", "cross-singleton-a"),
        _raw_decision("a-cross", "a-cross", "cross-candidates"),
        _raw_decision("b-cross", "a-cross", "cross-candidates"),
        _raw_decision("a-base", "a-base", "cross-singleton-base"),
    ])
    cross_scope = tmp_path / "cross-scope.parquet"
    cross_scope_manifest = tmp_path / "cross-scope.json"
    _run(
        "near-reconcile", "--pool", str(candidate_scope), "--pass-kind", "cross-candidate",
        "--raw-decisions", str(cross_raw), "--output-ledger", str(cross_scope),
        "--work-database", str(tmp_path / "cross-scope.sqlite"), "--manifest", str(cross_scope_manifest),
    )
    cross = tmp_path / "cross.parquet.final"
    cross_manifest = tmp_path / "cross.json"
    _run(
        "extend-candidate-scope-ledger", "--pool", str(within_survivors), "--scope-ledger", str(cross_scope),
        "--output-ledger", str(cross), "--work-database", str(tmp_path / "cross.sqlite"),
        "--manifest", str(cross_manifest),
    )
    cross_survivors = tmp_path / "cross-survivors"
    _run(
        "materialize", "--pool", str(within_survivors), "--ledger", str(cross), "--output", str(cross_survivors),
        "--work-database", str(tmp_path / "cross-materialize.sqlite"), "--manifest", str(tmp_path / "cross-materialize.json"),
    )

    candidate_to_base_raw = tmp_path / "candidate-to-base.parquet"
    _write(candidate_to_base_raw, [
        _raw_decision("base", "base", "candidate-to-base"),
        _raw_decision("a-base", "base", "candidate-to-base"),
        _raw_decision("a-within-a-keep", "a-within-a-keep", "candidate-singleton-a"),
        _raw_decision("a-cross", "a-cross", "candidate-singleton-cross"),
    ])
    candidate_to_base = tmp_path / "candidate-to-base.final.parquet"
    candidate_to_base_manifest = tmp_path / "candidate-to-base.json"
    _run(
        "near-reconcile", "--pool", str(cross_survivors), "--pass-kind", "candidate-to-nanochat",
        "--raw-decisions", str(candidate_to_base_raw), "--output-ledger", str(candidate_to_base),
        "--work-database", str(tmp_path / "candidate-to-base.sqlite"), "--manifest", str(candidate_to_base_manifest),
    )

    final = tmp_path / "final.parquet"
    final_manifest = tmp_path / "final.json"
    _run(
        "compose-ordered-ledgers", "--pool", str(pool),
        "--stage", "exact_content_work_representation", "--stage-ledger", str(exact), "--stage-manifest", str(exact_manifest),
        "--stage", "within_source_near", "--stage-ledger", str(within), "--stage-manifest", str(within_manifest),
        "--stage", "cross_candidate_near", "--stage-ledger", str(cross), "--stage-manifest", str(cross_manifest),
        "--stage", "candidate_to_nanochat_near", "--stage-ledger", str(candidate_to_base), "--stage-manifest", str(candidate_to_base_manifest),
        "--output-ledger", str(final), "--work-database", str(tmp_path / "compose.sqlite"), "--manifest", str(final_manifest),
    )
    decisions = {row["stable_uid"]: row for row in pq.read_table(final).to_pylist()}
    assert decisions["exact-clone"]["representative_stable_uid"] == "base"
    assert decisions["a-within-z-drop"]["representative_stable_uid"] == "a-within-a-keep"
    assert decisions["b-cross"]["representative_stable_uid"] == "a-cross"
    assert decisions["a-base"]["representative_stable_uid"] == "base"
    assert decisions["base"]["action"] == "keep"
    assert decisions["a-within-a-keep"]["action"] == "keep"
    assert decisions["a-cross"]["action"] == "keep"
    manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert manifest["ordered_dedup"]["pass_order"] == [
        "exact_content_work_representation",
        "within_source_near",
        "cross_candidate_near",
        "candidate_to_nanochat_near",
    ]
    assert manifest["ordered_dedup"]["exact_identity_precedes_near_passes"] is True


def test_cross_candidate_pass_rejects_a_within_source_component(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    _write(normalized / "data.parquet", [
        _row("candidate-left", base=False, source_id="candidate-a", text="left"),
        _row("candidate-right", base=False, source_id="candidate-a", text="right"),
    ])
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({
        "schema_version": "agent1_full_corpus_v3_source_admission_confirmation_v1",
        "status": "approved",
        "sources": [{"source_id": "candidate-a", "decision": "include"}],
    }), encoding="utf-8")
    pool = tmp_path / "pool"
    _run(
        "prepare-pool", "--input", str(normalized), "--admission-confirmation", str(confirmation),
        "--output", str(pool), "--manifest", str(tmp_path / "pool.json"),
    )
    raw = tmp_path / "raw.parquet"
    _write(raw, [
        _raw_decision("candidate-left", "candidate-left", "wrong-cross-scope"),
        _raw_decision("candidate-right", "candidate-left", "wrong-cross-scope"),
    ])
    with pytest.raises(subprocess.CalledProcessError):
        _run(
            "near-reconcile", "--pool", str(pool), "--pass-kind", "cross-candidate",
            "--raw-decisions", str(raw), "--output-ledger", str(tmp_path / "ledger.parquet"),
            "--work-database", str(tmp_path / "scope.sqlite"), "--manifest", str(tmp_path / "ledger.json"),
        )
