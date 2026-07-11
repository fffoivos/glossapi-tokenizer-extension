from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PHASE = Path(__file__).resolve().parents[1]
SCRIPTS = PHASE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_token_waterfall import build_waterfall  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _corpus_schema() -> pa.Schema:
    return pa.schema(
        [
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("text", pa.string()),
            ("title", pa.string()),
            ("author", pa.string()),
            ("greek_badness_score", pa.float64()),
            ("mojibake_badness_score", pa.float64()),
            ("needs_ocr", pa.bool_()),
            ("is_empty", pa.bool_()),
            ("ocr_success", pa.bool_()),
            ("is_historical_or_polytonic", pa.bool_()),
            ("source_family_id", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_repo_id", pa.string()),
            ("source_revision", pa.string()),
            ("source_artifact_path", pa.string()),
            ("source_row_id", pa.string()),
            ("stable_uid", pa.string()),
            ("cleaned_text_sha256", pa.string()),
            ("source_metadata_json", pa.string()),
            ("eligible_for_training", pa.bool_()),
            ("eligible_for_redistribution", pa.bool_()),
        ]
    )


def _corpus_row(uid: str, text: str, *, redistributable: bool) -> dict[str, object]:
    return {
        "source_dataset": "demo_source",
        "source_doc_id": f"upstream-{uid}",
        "text": text,
        "title": "Δοκιμή",
        "author": "Author withheld from public output",
        "greek_badness_score": 0.0,
        "mojibake_badness_score": 0.0,
        "needs_ocr": False,
        "is_empty": False,
        "ocr_success": True,
        "is_historical_or_polytonic": False,
        "source_family_id": "demo",
        "acquisition_source_id": "eellak_articles",
        "source_repo_id": "glossAPI/eellak-articles",
        "source_revision": "59fd681c483e6bdcdabe7c1a1f8685c5eebf7883",
        "source_artifact_path": "data/train-00000.parquet",
        "source_row_id": uid,
        "stable_uid": uid,
        "cleaned_text_sha256": _sha(text),
        "source_metadata_json": json.dumps({"private": "not public"}),
        "eligible_for_training": True,
        "eligible_for_redistribution": redistributable,
    }


def _write_parquet(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _decision_schema() -> pa.Schema:
    return pa.schema(
        [
            ("doc_key", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("decision", pa.string()),
            ("decision_stage", pa.string()),
            ("cluster_id", pa.string()),
            ("kept_doc_key", pa.string()),
            ("reason", pa.string()),
            ("exact_strict_version", pa.string()),
            ("exact_relaxed_version", pa.string()),
            ("near_norm_version", pa.string()),
            ("tokenization_version", pa.string()),
            ("shingle_version", pa.string()),
            ("minhash_version", pa.string()),
            ("lsh_version", pa.string()),
            ("selection_version", pa.string()),
            ("stable_uid", pa.string()),
            ("input_text_sha256", pa.string()),
        ]
    )


def _decision(uid: str, decision: str, stage: str, kept: str, input_text_sha256: str) -> dict[str, object]:
    return {
        "doc_key": f"key-{uid}",
        "source_dataset": "demo_source",
        "source_doc_id": uid,
        "decision": decision,
        "decision_stage": stage,
        "cluster_id": None if decision == "keep" else "strict:duplicate",
        "kept_doc_key": f"key-{kept}",
        "reason": "kept" if decision == "keep" else stage,
        "exact_strict_version": "strict-v1",
        "exact_relaxed_version": "relaxed-v1",
        "near_norm_version": "near-v1",
        "tokenization_version": "tokens-v1",
        "shingle_version": "shingles-v1",
        "minhash_version": "minhash-v1",
        "lsh_version": "lsh-v1",
        "selection_version": "selection-v1",
        "stable_uid": uid,
        "input_text_sha256": input_text_sha256,
    }


def _ledger_schema() -> pa.Schema:
    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("action", pa.string()),
            ("reasons_json", pa.string()),
            ("tokens_normalized", pa.int64()),
            ("tokens_source_cleaned", pa.int64()),
            ("tokens_pii_masked", pa.int64()),
            ("tokens_structural_cleaned", pa.int64()),
            ("tokens_bibliography_removed", pa.int64()),
            ("tokens_toc_removed", pa.int64()),
            ("tokens_structural_union_removed", pa.int64()),
            ("tokens_final", pa.int64()),
            ("final_text_sha256", pa.string()),
            ("eligible_for_training", pa.bool_()),
        ]
    )


def _decontam_schema() -> pa.Schema:
    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("input_text_sha256", pa.string()),
            ("action", pa.string()),
            ("reason", pa.string()),
            ("benchmark_matches_json", pa.string()),
        ]
    )


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(SCRIPTS / script), *args]
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if result.returncode:
        raise AssertionError(
            f"{script} failed with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def test_dedup_staging_materialization_waterfall_and_validation(tmp_path: Path) -> None:
    corpus = tmp_path / "decontaminated"
    duplicate_text = "Αυτό είναι ένα πλήρες ελληνικό έγγραφο για δοκιμή."
    rows = [
        _corpus_row("uid-a", duplicate_text, redistributable=True),
        _corpus_row("uid-b", duplicate_text, redistributable=True),
        _corpus_row("uid-c", "Ένα δεύτερο και μοναδικό ελληνικό έγγραφο.", redistributable=False),
    ]
    _write_parquet(corpus / "demo" / "part-00000.parquet", rows, _corpus_schema())

    staged = tmp_path / "dedup-staged"
    stage_manifest = tmp_path / "dedup-stage.json"
    _run(
        "run_full_corpus_dedup.py",
        "--input",
        str(corpus),
        "--staged-input",
        str(staged),
        "--state-root",
        str(tmp_path / "state"),
        "--run-root",
        str(tmp_path / "dedup-run"),
        "--manifest",
        str(stage_manifest),
        "--temporary-directory",
        str(tmp_path / "duck-stage"),
        "--workers",
        "2",
        "--stage-only",
    )
    staged_rows = pq.read_table(staged / "demo" / "part-00000.parquet").to_pylist()
    assert [row["source_doc_id"] for row in staged_rows] == ["uid-a", "uid-b", "uid-c"]
    assert [row["upstream_source_doc_id"] for row in staged_rows] == [
        "upstream-uid-a",
        "upstream-uid-b",
        "upstream-uid-c",
    ]
    assert json.loads(stage_manifest.read_text())["dedup_implementation"]["reimplemented"] is False

    decisions = tmp_path / "dedup-run" / "final" / "dedup_decisions.parquet"
    _write_parquet(
        decisions,
        [
            _decision("uid-a", "keep", "kept_after_exact", "uid-a", _sha(duplicate_text)),
            _decision("uid-b", "drop", "strict_exact", "uid-a", _sha(duplicate_text)),
            _decision(
                "uid-c",
                "keep",
                "kept_after_exact",
                "uid-c",
                _sha("Ένα δεύτερο και μοναδικό ελληνικό έγγραφο."),
            ),
        ],
        _decision_schema(),
    )

    cleaned_root = tmp_path / "cleaned"
    cleaning_manifest = tmp_path / "cleaning-manifest.json"
    cleaning_artifacts = {}
    for name in ("tokenizer.json", "admission.json", "eligibility.json", "cleaning.json"):
        path = tmp_path / name
        path.write_text(f"fixture:{name}", encoding="utf-8")
        cleaning_artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for name, path in (
        ("sources.json", PHASE / "configs" / "sources.json"),
        (
            "source_license_adjudication.json",
            PHASE / "configs" / "source_license_adjudication.json",
        ),
    ):
        cleaning_artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    cleaning_manifest.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_cleaning_manifest_v1",
                "status": "completed",
                "completed_at": "2026-07-12T00:00:00+00:00",
                "input": str((tmp_path / "normalized").resolve()),
                "output": str(cleaned_root.resolve()),
                "tokenizer_json": cleaning_artifacts["tokenizer.json"]["path"],
                "tokenizer_sha256": cleaning_artifacts["tokenizer.json"]["sha256"],
                "source_admission": cleaning_artifacts["admission.json"]["path"],
                "source_admission_sha256": cleaning_artifacts["admission.json"]["sha256"],
                "source_config": cleaning_artifacts["sources.json"]["path"],
                "source_config_sha256": cleaning_artifacts["sources.json"]["sha256"],
                "license_adjudication": cleaning_artifacts["source_license_adjudication.json"]["path"],
                "license_adjudication_sha256": cleaning_artifacts["source_license_adjudication.json"]["sha256"],
                "eligibility_policy": cleaning_artifacts["eligibility.json"]["path"],
                "eligibility_policy_sha256": cleaning_artifacts["eligibility.json"]["sha256"],
                "cleaning_policy": cleaning_artifacts["cleaning.json"]["path"],
                "cleaning_policy_sha256": cleaning_artifacts["cleaning.json"]["sha256"],
                "config_sha256": "1" * 64,
                "cleaning_pass": "post_source_post_pii",
                "structural_applied": False,
            }
        ),
        encoding="utf-8",
    )
    decontamination_manifest = tmp_path / "decontamination-manifest.json"
    corpus_file = corpus / "demo" / "part-00000.parquet"
    corpus_receipt = {
        "path": "demo/part-00000.parquet",
        "sha256": hashlib.sha256(corpus_file.read_bytes()).hexdigest(),
        "bytes": corpus_file.stat().st_size,
        "rows": 3,
        "row_groups": pq.ParquetFile(corpus_file).metadata.num_row_groups,
    }
    decontamination_manifest.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_greekmmlu_decontamination_v1",
                "status": "completed",
                "completed_at": "2026-07-12T00:00:01+00:00",
                "input": str(cleaned_root.resolve()),
                "output": str(corpus.resolve()),
                "counts": {"input": 3, "kept": 3, "dropped": 0},
                "files": [{"output": corpus_receipt}],
                "policy": {
                    "policy_version": "greekmmlu_decontamination_v1",
                    "normalization": "NFKC+strip_combining_marks+casefold+unicode_word_tokens_v1",
                    "k": 8,
                    "min_coverage": 0.85,
                    "minhash_threshold": 0.85,
                    "minhash_permutations": 64,
                    "min_matched_grams": 4,
                    "max_gap_tokens": 40,
                    "drop_rules": [
                        "greekmmlu_exact_prompt",
                        "greekmmlu_exact_question_answer",
                        "greekmmlu_ngram_minhash_answer",
                    ],
                    "answer_only_action": "audit_only",
                },
            }
        ),
        encoding="utf-8",
    )
    decision_receipt = {
        "path": str(decisions.resolve()),
        "sha256": hashlib.sha256(decisions.read_bytes()).hexdigest(),
        "bytes": decisions.stat().st_size,
        "rows": 3,
        "row_groups": pq.ParquetFile(decisions).metadata.num_row_groups,
        "schema_version": "full_cpt_dedup_decisions_content_bound_v1",
    }
    dedup_manifest = tmp_path / "dedup-manifest.json"
    dedup_manifest.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_dedup_wrapper_manifest_v1",
                "completed_at": "2026-07-12T00:00:02+00:00",
                "status": "completed",
                "input": str(corpus.resolve()),
                "identity_contract": {
                    "dedup_source_dataset": "source_dataset (unchanged)",
                    "dedup_source_doc_id": "stable_uid",
                    "upstream_source_doc_id": "source_doc_id before staging",
                },
                "recipe": {"id": "greek_cpt_text_dedup_v1", "mode": "production"},
                "dedup_output": {
                    "content_bound_decisions": decision_receipt,
                    "decisions": decision_receipt,
                    "content_binding": {
                        "schema_version": "full_cpt_dedup_decisions_content_bound_v1",
                        "stable_uid_column": "stable_uid",
                        "input_text_sha256_column": "input_text_sha256",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    cleaning = tmp_path / "cleaning-ledger"
    cleaning_rows = []
    text_hashes = {
        "uid-a": _sha(duplicate_text),
        "uid-b": _sha(duplicate_text),
        "uid-c": _sha("Ένα δεύτερο και μοναδικό ελληνικό έγγραφο."),
        "uid-d": "hash",
        "uid-e": "hash",
    }
    for uid in ("uid-a", "uid-b", "uid-c", "uid-d", "uid-e"):
        eligible = uid != "uid-e"
        cleaning_rows.append(
            {
                "stable_uid": uid,
                "acquisition_source_id": "eellak_articles",
                "source_dataset": "demo_source",
                "source_doc_id": f"upstream-{uid}",
                "action": "keep",
                "reasons_json": (
                    "[]"
                    if eligible
                    else '["training_eligibility_not_approved:noncommercial_review"]'
                ),
                "tokens_normalized": 12,
                "tokens_source_cleaned": 11,
                "tokens_pii_masked": 11,
                "tokens_structural_cleaned": 10,
                "tokens_bibliography_removed": 1 if uid in {"uid-a", "uid-c", "uid-e"} else 0,
                "tokens_toc_removed": 1 if uid in {"uid-b", "uid-d"} else 0,
                "tokens_structural_union_removed": 1,
                "tokens_final": 10 if eligible else 0,
                "final_text_sha256": text_hashes[uid],
                "eligible_for_training": eligible,
            }
        )
    _write_parquet(cleaning / "part.parquet", cleaning_rows, _ledger_schema())
    decontam = tmp_path / "decontam-ledger"
    decontam_rows = []
    for uid in ("uid-a", "uid-b", "uid-c", "uid-d", "uid-e"):
        decontam_rows.append(
            {
                "stable_uid": uid,
                "acquisition_source_id": "eellak_articles",
                "source_dataset": "demo_source",
                "source_doc_id": f"upstream-{uid}",
                "input_text_sha256": text_hashes[uid],
                "action": "drop" if uid == "uid-d" else "keep",
                "reason": "greekmmlu_exact_prompt" if uid == "uid-d" else "no_high_confidence_match",
                "benchmark_matches_json": "[]",
            }
        )
    _write_parquet(decontam / "part.parquet", decontam_rows, _decontam_schema())

    waterfall = tmp_path / "token-waterfall.json"
    payload = build_waterfall(
        cleaning_ledger=cleaning,
        decontam_ledger=decontam,
        dedup_decisions=decisions,
        output=waterfall,
        temporary_directory=tmp_path / "duck-waterfall",
        memory_limit="1GB",
        threads=2,
        cleaning_manifest=cleaning_manifest,
        decontamination_manifest=decontamination_manifest,
        dedup_manifest=dedup_manifest,
    )
    assert payload["invariants"]["final_tokens"] == 20
    assert payload["invariants"]["reconciled"] is True
    assert any(row["stage"] == "toc_bib" and row["tokens_removed"] == 5 for row in payload["events_global"])
    assert payload["structural_token_loss"]["global"] == {
        "bibliography_tokens_removed": 3,
        "toc_tokens_removed": 2,
        "union_tokens_removed": 5,
    }
    assert any(row["stage"] == "greekmmlu_decontamination" and row["tokens_removed"] == 10 for row in payload["events_global"])
    assert any(row["stage"] == "strict_exact" and row["tokens_removed"] == 10 for row in payload["events_global"])
    assert any(
        row["stage"] == "policy_filter"
        and "training_eligibility_not_approved" in row["reason"]
        and row["tokens_removed"] == 10
        for row in payload["events_global"]
    )

    release = tmp_path / "release"
    release_manifest = tmp_path / "release-manifest.json"
    _run(
        "materialize_release.py",
        "--input",
        str(corpus),
        "--cleaning-manifest",
        str(cleaning_manifest),
        "--decontamination-manifest",
        str(decontamination_manifest),
        "--dedup-manifest",
        str(dedup_manifest),
        "--dedup-decisions",
        str(decisions),
        "--output",
        str(release),
        "--manifest",
        str(release_manifest),
        "--token-waterfall",
        str(waterfall),
        "--temporary-directory",
        str(tmp_path / "duck-materialize"),
        "--memory-limit",
        "1GB",
        "--threads",
        "2",
    )
    training = pq.read_table(release / "training" / "data" / "demo" / "part-00000.parquet")
    redistribution = pq.read_table(release / "redistribution" / "data" / "demo" / "part-00000.parquet")
    assert set(training["stable_uid"].to_pylist()) == {"uid-a", "uid-c"}
    assert redistribution["stable_uid"].to_pylist() == ["uid-a"]
    assert "source_metadata_json" not in redistribution.schema.names
    assert "author" not in redistribution.schema.names
    assert "title" not in redistribution.schema.names
    assert "source_doc_id" not in redistribution.schema.names
    assert "source_doc_id_sha256" in redistribution.schema.names

    training_sha_before_resume = hashlib.sha256(
        (release / "training" / "data" / "demo" / "part-00000.parquet").read_bytes()
    ).hexdigest()
    redistribution_sha_before_resume = hashlib.sha256(
        (release / "redistribution" / "data" / "demo" / "part-00000.parquet").read_bytes()
    ).hexdigest()
    card_sha_before_resume = hashlib.sha256(
        (release / "publication" / "README.md").read_bytes()
    ).hexdigest()
    assert list((release / ".materialization-checkpoints").rglob("*.json"))
    release_manifest.unlink()
    _run(
        "materialize_release.py",
        "--input",
        str(corpus),
        "--cleaning-manifest",
        str(cleaning_manifest),
        "--decontamination-manifest",
        str(decontamination_manifest),
        "--dedup-manifest",
        str(dedup_manifest),
        "--dedup-decisions",
        str(decisions),
        "--output",
        str(release),
        "--manifest",
        str(release_manifest),
        "--token-waterfall",
        str(waterfall),
        "--temporary-directory",
        str(tmp_path / "duck-materialize-resume"),
        "--memory-limit",
        "1GB",
        "--threads",
        "2",
        "--resume",
    )
    assert hashlib.sha256(
        (release / "training" / "data" / "demo" / "part-00000.parquet").read_bytes()
    ).hexdigest() == training_sha_before_resume
    assert hashlib.sha256(
        (release / "redistribution" / "data" / "demo" / "part-00000.parquet").read_bytes()
    ).hexdigest() == redistribution_sha_before_resume
    assert hashlib.sha256(
        (release / "publication" / "README.md").read_bytes()
    ).hexdigest() == card_sha_before_resume

    receipt = tmp_path / "validation.json"
    _run(
        "validate_release.py",
        "--release",
        str(release),
        "--manifest",
        str(release_manifest),
        "--cleaning-manifest",
        str(cleaning_manifest),
        "--decontamination-manifest",
        str(decontamination_manifest),
        "--dedup-manifest",
        str(dedup_manifest),
        "--dedup-decisions",
        str(decisions),
        "--output",
        str(receipt),
        "--temporary-directory",
        str(tmp_path / "duck-validate"),
        "--memory-limit",
        "1GB",
        "--threads",
        "2",
    )
    validation = json.loads(receipt.read_text())
    assert validation["status"] == "passed"
    assert not validation["failed_checks"]

    publication_dry_run = tmp_path / "publication-dry-run.json"
    _run(
        "publish_release.py",
        "--release",
        str(release),
        "--release-manifest",
        str(release_manifest),
        "--validation-receipt",
        str(receipt),
        "--repo-id",
        "fffoivos/test-release",
        "--output",
        str(publication_dry_run),
    )
    publication = json.loads(publication_dry_run.read_text())
    assert publication["status"] == "dry_run"
    assert publication["gate_mode"] == "manual"
    assert publication["remote_mode"] == "new-empty"
    assert publication["commit_sha"] is None
    assert publication["counts"]["rows"] == 1

    # A public row cannot drift independently of the private training row even
    # if its identity remains unchanged.  The validator emits a failed receipt
    # and identifies content parity, in addition to the file checksum failure.
    redistribution_path = release / "redistribution" / "data" / "demo" / "part-00000.parquet"
    public_rows = pq.read_table(redistribution_path).to_pylist()
    public_rows[0]["text"] = "Αλλοιωμένο δημόσιο κείμενο."
    pq.write_table(pa.Table.from_pylist(public_rows, schema=redistribution.schema), redistribution_path)
    failed_receipt = tmp_path / "validation-failed.json"
    command = [
        sys.executable,
        str(SCRIPTS / "validate_release.py"),
        "--release",
        str(release),
        "--manifest",
        str(release_manifest),
        "--cleaning-manifest",
        str(cleaning_manifest),
        "--decontamination-manifest",
        str(decontamination_manifest),
        "--dedup-manifest",
        str(dedup_manifest),
        "--dedup-decisions",
        str(decisions),
        "--output",
        str(failed_receipt),
        "--temporary-directory",
        str(tmp_path / "duck-validate-failed"),
        "--memory-limit",
        "1GB",
        "--threads",
        "2",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 1
    failed_validation = json.loads(failed_receipt.read_text())
    assert failed_validation["status"] == "failed"
    assert "redistribution_content_parity" in failed_validation["failed_checks"]
    assert "redistribution_bad_text_hash" in failed_validation["failed_checks"]

    release_manifest.unlink()
    resume_after_drift = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "materialize_release.py"),
            "--input",
            str(corpus),
            "--cleaning-manifest",
            str(cleaning_manifest),
            "--decontamination-manifest",
            str(decontamination_manifest),
            "--dedup-manifest",
            str(dedup_manifest),
            "--dedup-decisions",
            str(decisions),
            "--output",
            str(release),
            "--manifest",
            str(release_manifest),
            "--token-waterfall",
            str(waterfall),
            "--temporary-directory",
            str(tmp_path / "duck-materialize-drifted-resume"),
            "--memory-limit",
            "1GB",
            "--threads",
            "2",
            "--resume",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert resume_after_drift.returncode != 0
    assert "checkpointed release file drift" in resume_after_drift.stderr
