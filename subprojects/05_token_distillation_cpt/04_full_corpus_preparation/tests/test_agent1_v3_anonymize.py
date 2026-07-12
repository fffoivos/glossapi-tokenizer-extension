from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
SCRIPT = SCRIPTS / "agent1_v3_anonymize.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agent1_v3_anonymize_test_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


ANONYMIZE = load_module()


def valid_afm() -> str:
    return next(
        f"{number:09d}"
        for number in range(1, 100_000_000)
        if ANONYMIZE.afm_valid(f"{number:09d}")
    )


def canonical_row(**updates: object) -> dict[str, object]:
    text = str(updates.pop("text", "Κανονικό κείμενο"))
    row: dict[str, object] = {
        "source_id": "demo",
        "source_dataset": "demo",
        "acquisition_source_id": "demo",
        "source_family_id": "demo",
        "source_doc_id": "doc-1",
        "stable_uid": "uid-1",
        "representation_generation": "candidate_first_representation",
        "source_metadata_json": "{}",
        "text": text,
    }
    row.update(updates)
    return row


def write_input(path: Path, rows: list[dict[str, object]]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Make optional v3 lineage columns part of the schema even when the first
    # fixture row is a private-data drop row and therefore lacks them.
    columns = sorted({key for row in rows for key in row})
    normalized = [{key: row.get(key) for key in columns} for row in rows]
    pq.write_table(pa.Table.from_pylist(normalized), path, compression="zstd")


def run_cli(tmp_path: Path, input_root: Path, *, policy: Path | None = None) -> dict[str, Path]:
    roots = {
        "output": tmp_path / "masked",
        "dropped": tmp_path / "dropped",
        "quarantine": tmp_path / "quarantine",
        "ledger": tmp_path / "protected-ledger",
        "manifest": tmp_path / "manifest.json",
    }
    command = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(input_root),
        "--output",
        str(roots["output"]),
        "--dropped",
        str(roots["dropped"]),
        "--quarantine",
        str(roots["quarantine"]),
        "--protected-ledger",
        str(roots["ledger"]),
        "--manifest",
        str(roots["manifest"]),
        "--batch-rows",
        "1",
    ]
    if policy is not None:
        command.extend(["--policy", str(policy)])
    subprocess.run(command, check=True, text=True, capture_output=True)
    return roots


def read_rows(path: Path) -> list[dict[str, object]]:
    pq = pytest.importorskip("pyarrow.parquet")
    return pq.read_table(path).to_pylist()


def test_masks_only_approved_direct_identifiers_and_keeps_html_names_addresses() -> None:
    afm = valid_afm()
    text = (
        "<p>Ο Γιάννης Παπαδόπουλος μένει στην Οδό Ερμού 1.</p> "
        f"Email demo@example.gr, ΑΦΜ: {afm}, τηλ. +30 2101234567, "
        "IP 192.0.2.4, IPv6 2001:db8::1, ΑΔΤ: ΑΒ123456 και διαβατήριο: ZZ123456."
    )
    masked, spans, counts = ANONYMIZE.mask_direct_identifiers(text)
    assert "<p>" in masked  # no HTML cleaning
    assert "Γιάννης Παπαδόπουλος" in masked  # no generic name masking
    assert "Οδό Ερμού 1" in masked  # no generic address masking
    assert "<email-pii>" in masked
    assert "<afm-pii>" in masked
    assert "<phone-pii>" in masked
    assert "<ip-pii>" in masked
    assert "<identity-pii>" in masked
    assert counts == {
        "afm": 1,
        "email": 1,
        "identity_or_passport": 2,
        "ip": 2,
        "phone": 1,
    }
    assert [span.raw_value for span in spans] == [
        "demo@example.gr",
        afm,
        "+30 2101234567",
        "192.0.2.4",
        "2001:db8::1",
        "ΑΒ123456",
        "ZZ123456",
    ]


def test_cli_streams_actions_preserves_uid_and_keeps_raw_spans_private(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    afm = valid_afm()
    input_root = tmp_path / "input"
    source = input_root / "source-a" / "part.parquet"
    public_email = "visible@example.gr"
    private_row = canonical_row(
        source_id="diavgeia",
        source_dataset="diavgeia",
        acquisition_source_id="diavgeia",
        source_family_id="diavgeia",
        source_doc_id="private",
        stable_uid="uid-private",
        text="Ιδιωτική απόφαση",
        source_metadata_json=json.dumps({"nested": {"privateData": True}}),
    )
    personnel_row = canonical_row(
        source_id="diavgeia",
        source_dataset="diavgeia",
        acquisition_source_id="diavgeia",
        source_family_id="diavgeia",
        source_doc_id="personnel",
        stable_uid="uid-personnel",
        text=(
            "ΠΙΝΑΚΑΣ ΥΠΟΨΗΦΙΩΝ\n"
            f"Επικοινωνία {public_email}; ΑΦΜ: {afm}; ΑΔΤ: ΑΒ123456"
        ),
    )
    keep_text = "<p>Μαρία Κωνσταντίνου, Οδός Αθηνάς 1.</p> email@example.gr +30 2101234567"
    keep_hash = hashlib.sha256(keep_text.encode("utf-8")).hexdigest()
    keep_row = canonical_row(
        source_doc_id="keep",
        stable_uid="uid-keep",
        text=keep_text,
        input_representation_id="normalized-v1:uid-keep",
        representation_id="dedup-v1:uid-keep",
        parent_representation_id="older-parent",
        parent_text_sha256=keep_hash,
        text_sha256=keep_hash,
        cleaned_text_sha256=keep_hash,
    )
    write_input(source, [private_row, personnel_row, keep_row])

    roots = run_cli(tmp_path, input_root)
    masked = read_rows(roots["output"] / "source-a" / "part.parquet")
    quarantined = read_rows(roots["quarantine"] / "source-a" / "part.parquet")
    dropped = read_rows(roots["dropped"] / "source-a" / "part.parquet")
    ledger = read_rows(roots["ledger"] / "source-a" / "part.parquet")

    assert [row["stable_uid"] for row in masked] == ["uid-keep"]
    assert len(quarantined) == 1
    assert quarantined[0]["stable_uid"] == "uid-personnel"
    assert quarantined[0]["anonymization_action"] == "quarantine"
    assert "<email-pii>" in str(quarantined[0]["text"])
    assert len(dropped) == 1
    assert dropped[0]["stable_uid"] == "uid-private"
    assert "text" not in dropped[0]  # no raw privateData document duplicate

    kept = masked[0]
    assert "<p>" in str(kept["text"])  # no HTML cleanup slipped in
    assert "Μαρία Κωνσταντίνου, Οδός Αθηνάς 1" in str(kept["text"])
    assert "<email-pii>" in str(kept["text"])
    assert "<phone-pii>" in str(kept["text"])
    assert kept["stable_uid"] == "uid-keep"
    assert kept["anonymization_parent_text_sha256"] == hashlib.sha256(
        str(keep_row["text"]).encode("utf-8")
    ).hexdigest()
    assert kept["anonymization_parent_representation_id"] != kept[
        "anonymization_child_representation_id"
    ]
    assert kept["representation_id"] == kept["anonymization_child_representation_id"]
    assert kept["parent_representation_id"] == "dedup-v1:uid-keep"
    assert kept["anonymization_parent_representation_id"] == "dedup-v1:uid-keep"
    assert kept["parent_text_sha256"] == kept["anonymization_parent_text_sha256"]
    assert kept["text_sha256"] == kept["anonymization_output_text_sha256"]
    assert kept["cleaned_text_sha256"] == kept["anonymization_output_text_sha256"]

    actions = {str(row["stable_uid"]): row for row in ledger}
    assert set(actions) == {"uid-private", "uid-personnel", "uid-keep"}
    assert actions["uid-private"]["action"] == "drop"
    assert json.loads(str(actions["uid-private"]["protected_spans_json"])) == []
    personnel_spans = json.loads(str(actions["uid-personnel"]["protected_spans_json"]))
    assert any(span["raw_value"] == public_email for span in personnel_spans)
    assert stat.S_IMODE((roots["ledger"] / "source-a" / "part.parquet").stat().st_mode) == 0o600
    assert stat.S_IMODE(roots["ledger"].stat().st_mode) == 0o700

    manifest = roots["manifest"].read_text(encoding="utf-8")
    assert public_email not in manifest
    payload = json.loads(manifest)
    assert payload["counts"]["input_rows"] == 3
    assert payload["counts"]["action:keep"] == 1
    assert payload["counts"]["action:drop"] == 1
    assert payload["counts"]["action:quarantine"] == 1
    assert payload["transform_boundaries"]["html_cleaning"] is False
    assert payload["protected_ledger"]["public_training_output"] is False


def test_cli_rejects_protected_ledger_inside_public_output(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    input_root = tmp_path / "input"
    write_input(input_root / "part.parquet", [canonical_row()])
    command = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(input_root),
        "--output",
        str(tmp_path / "output"),
        "--dropped",
        str(tmp_path / "dropped"),
        "--quarantine",
        str(tmp_path / "quarantine"),
        "--protected-ledger",
        str(tmp_path / "output" / "ledger"),
        "--manifest",
        str(tmp_path / "manifest.json"),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "must be disjoint" in completed.stderr


def test_existing_v3_representation_becomes_the_parent_of_the_masked_child() -> None:
    row = canonical_row(
        stable_uid="uid-lineage",
        representation_id="dedup-representation:uid-lineage",
        parent_representation_id="older-parent",
        text="email lineage@example.gr",
    )
    parent_hash = ANONYMIZE.sha256_text(str(row["text"]))
    parent_id = ANONYMIZE.parent_representation_id(row, parent_hash)
    masked, _, counts = ANONYMIZE.mask_direct_identifiers(str(row["text"]))
    output_hash = ANONYMIZE.sha256_text(masked)
    child_id = ANONYMIZE.child_representation_id(parent_id, output_hash)
    derivative = ANONYMIZE._derivative_row(
        row,
        text=masked,
        parent_hash=parent_hash,
        output_hash=output_hash,
        parent_id=parent_id,
        child_id=child_id,
        action="keep",
        reasons=["approved_high_precision_direct_identifier_masking"],
        pii_counts=counts,
    )
    assert parent_id == "dedup-representation:uid-lineage"
    assert derivative["stable_uid"] == "uid-lineage"
    assert derivative["parent_representation_id"] == parent_id
    assert derivative["representation_id"] == child_id
    assert derivative["parent_text_sha256"] == parent_hash
    assert derivative["text_sha256"] == derivative["cleaned_text_sha256"] == output_hash


def test_policy_is_bound_and_cannot_enable_generic_name_or_address_masking(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "anonymization": {
                    "drop_private_data_true": True,
                    "mask_types": list(ANONYMIZE.ALLOWED_MASK_TYPES),
                    "generic_person_name_ner": True,
                    "street_address_policy": "enabled",
                    "diavgeia_personnel_table": "quarantine_when_pii_heavy",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generic person-name NER"):
        ANONYMIZE._policy_receipt(policy, 3)
