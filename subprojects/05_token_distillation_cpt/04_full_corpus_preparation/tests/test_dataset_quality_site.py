from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


QUALITY = load_module(
    "phase04_dataset_quality_rust",
    HERE / "scripts" / "profile_dataset_quality_rust.py",
)
sys.modules["profile_dataset_quality_rust"] = QUALITY
SITE = load_module(
    "phase04_dataset_review_site",
    HERE / "scripts" / "build_dataset_review_site.py",
)
EXPORTER = load_module(
    "phase04_export_dataset_review_samples",
    HERE / "scripts" / "export_dataset_review_samples.py",
)


def tracked_inventory() -> dict:
    return json.loads((HERE / "configs" / "post_december_inventory.json").read_text())


def review_request(
    uid: str,
    *,
    source_id: str = "diavgeia",
    repo_id: str = "glossAPI/diavgeia",
    dataset: str = "diavgeia",
    doc_id: str = "ADA-1",
    preview_sentinel: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "source_quality_review_request_v1",
        "reviewer_slot": "primary",
        "sample_id": uid,
        "source_dataset": dataset,
        "sampling_stratum": "risk",
        "source": {
            "source_id": source_id,
            "source_repo_id": repo_id,
            "source_revision": "a" * 40,
            "source_doc_id": doc_id,
        },
    }
    if preview_sentinel is not None:
        row["document"] = {"mode": "full", "text": preview_sentinel}
    return row


def complete_sample(
    uid: str,
    text: str,
    *,
    source_id: str = "diavgeia",
    repo_id: str = "glossAPI/diavgeia",
    dataset: str = "diavgeia",
    doc_id: str = "ADA-1",
) -> dict[str, object]:
    return {
        "schema_version": "dataset_review_complete_sample_v1",
        "sample_id": uid,
        "source_id": source_id,
        "source_repo_id": repo_id,
        "source_revision": "a" * 40,
        "source_dataset": dataset,
        "display_document_id": QUALITY.display_document_id(doc_id),
        "normalized_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "profile_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "profile_text_variant": "high_precision_identifier_masked_review_sample",
        "input_shard_path": f"{source_id}/part.parquet",
        "input_shard_sha256": "b" * 64,
        "input_row_index": 7,
        "private_data_true": False,
        "corrected_version_present": False,
        "high_precision_identifier_patterns_masked": True,
        "redaction_counts": {},
        "text": text,
    }


def write_packet_receipt(
    path: Path,
    *,
    packet: Path,
    requests: Path,
    rows: int,
    normalization: Path | None = None,
) -> None:
    value = {
        "schema_version": "dataset_review_complete_sample_packet_receipt_v1",
        "status": "passed",
        "normalization_manifest": {
            "path": str(normalization.resolve())
            if normalization
            else "normalization.json",
            "sha256": hashlib.sha256(
                normalization.read_bytes() if normalization else b"normalization"
            ).hexdigest(),
        },
        "canonical_root": "/receipt-bound/canonical",
        "review_requests": {
            "path": str(requests.resolve()),
            "sha256": hashlib.sha256(requests.read_bytes()).hexdigest(),
        },
        "export_contract": {
            "path": "contract.json",
            "sha256": "c" * 64,
            "contract_sha256": "d" * 64,
        },
        "input_shards": [
            {
                "source_id": "diavgeia",
                "path": "diavgeia/part.parquet",
                "bytes": 1,
                "rows": max(rows, 1),
                "sha256": "b" * 64,
            }
        ],
        "checkpoint_inventory": [{}],
        "checkpoint_inventory_sha256": "e" * 64,
        "output": {
            "path": packet.name,
            "bytes": packet.stat().st_size,
            "rows": rows,
            "sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
        },
        "redaction_totals": {},
        "high_precision_identifier_patterns_masked": True,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_evaluations_cover_exact_29_repository_inventory() -> None:
    inventory = SITE.load_inventory(HERE / "configs" / "post_december_inventory.json")
    repos = {row["repo_id"] for row in inventory}
    evaluations = SITE.load_evaluations(
        HERE / "configs" / "dataset_review_evaluations.json", repos
    )
    assert len(inventory) == len(evaluations) == 29
    assert {row["inventory_group"] for row in inventory} == {
        "post_cutoff",
        "older_material_change",
    }
    assert evaluations["glossAPI/diavgeia"]["recommended_action"] == (
        "source_specific_cleaning"
    )
    assert evaluations["glossAPI/pandemos"]["recommended_action"] == "exclude_no_text"
    assert (
        SITE.payload_state(
            {
                "payload_status": "external_full_text_parquet_archive",
                "availability": "external_mozilla_registered_download_required",
            }
        )
        == "external_unavailable"
    )


def test_site_build_is_offline_complete_and_safe_for_hostile_sample(
    tmp_path: Path,
) -> None:
    uid = hashlib.sha256(b"hostile-sample").hexdigest()
    requests = tmp_path / "requests.jsonl"
    preview_sentinel = "PREVIEW_SENTINEL_MUST_NOT_LEAVE_REQUESTS"
    requests.write_text(
        json.dumps(review_request(uid, preview_sentinel=preview_sentinel)) + "\n",
        encoding="utf-8",
    )
    hostile = "</script><img src=x onerror=alert(1)> & harmless Greek κείμενο"
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(complete_sample(uid, hostile)) + "\n", encoding="utf-8"
    )
    sample_receipt = tmp_path / "samples-receipt.json"
    write_packet_receipt(sample_receipt, packet=samples, requests=requests, rows=1)
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "schema_version": "dataset_quality_summary_v1",
                "status": "passed",
                "scan_mode": "review_sample",
                "selected_source_ids": ["diavgeia"],
                "excluded_source_ids": ["nanochat_base"],
                "global": {"documents": 1},
                "repositories": [
                    {
                        "repo_id": "glossAPI/diavgeia",
                        "documents": 1,
                        "document_rates": {"html_rate": 0.25},
                        "distributions": {},
                        "template_concentration": {},
                    },
                    {
                        "repo_id": "glossAPI/Apothetirio_Kallipos",
                        "documents": 1,
                        "document_rates": {},
                        "distributions": {},
                        "template_concentration": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "build_dataset_review_site.py"),
            "build",
            "--review-requests",
            str(requests),
            "--complete-samples",
            str(samples),
            "--complete-samples-receipt",
            str(sample_receipt),
            "--quality-summary",
            str(quality),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    data = json.loads((output / "site_data.json").read_text())
    assert len(data["repositories"]) == 29
    assert data["overview"]["complete_samples"] == 1
    assert data["overview"]["quality_scope"] == {
        "documents": 1,
        "excluded_source_ids": ["nanochat_base"],
        "is_corpus_wide": False,
        "label": "Representative source-review sample",
        "scan_mode": "review_sample",
        "selected_source_ids": ["diavgeia"],
    }
    assert data["overview"]["supplemental_profiled_repositories_outside_inventory"] == [
        "glossAPI/Apothetirio_Kallipos"
    ]
    diavgeia = next(
        row for row in data["repositories"] if row["repo_id"] == "glossAPI/diavgeia"
    )
    assert diavgeia["quality_scope"]["repository_documents"] == 1
    assert len(list((output / "datasets").glob("*.html"))) == 29
    sample_files = list((output / "samples").glob("*.json"))
    assert len(sample_files) == 1
    sample_path = sample_files[0]
    assert sample_path.stem != uid and len(sample_path.stem) == 32
    assert "<" not in sample_path.read_text(encoding="utf-8")
    parsed_sample = json.loads(sample_path.read_text())
    assert parsed_sample["schema_version"] == "dataset_review_site_sample_v1"
    assert parsed_sample["site_sample_id"] == sample_path.stem
    assert parsed_sample["text"] == hostile
    assert "ADA-1" not in sample_path.read_text()
    assert "ADA-1" not in (output / "site_data.json").read_text()
    assert 'id="scope-banner"' in (output / "index.html").read_text()
    assert hostile not in (output / "index.html").read_text()
    all_output = "\n".join(
        path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file()
    )
    assert uid not in all_output
    assert preview_sentinel not in all_output
    javascript = (output / "assets" / "site.js").read_text()
    assert "textContent=doc.text" in javascript
    assert "style:'percent'" in javascript
    assert "not corpus-wide" in javascript.lower()
    assert "innerHTML" not in javascript
    for page in [output / "index.html", *sorted((output / "datasets").glob("*.html"))]:
        source = page.read_text(encoding="utf-8")
        assert "Content-Security-Policy" in source
        assert "https://" not in source
        assert 'src="http' not in source
    manifest = json.loads((output / "site_manifest.json").read_text())
    assert manifest["repository_count"] == manifest["dataset_page_count"] == 29
    assert manifest["security"]["bind_address"] == "127.0.0.1"
    assert manifest["security"]["external_resources"] is False
    assert manifest["inputs"]["complete_samples_receipt"][
        "sha256"
    ] == QUALITY.sha256_file(sample_receipt)
    assert os.stat(sample_path).st_mode & 0o777 == 0o600
    assert SITE.validate_site_directory(output)["status"] == "passed"
    jsonschema = pytest.importorskip("jsonschema")
    manifest_schema = json.loads(
        (HERE / "schemas" / "dataset_review_site_manifest.schema.json").read_text()
    )
    sample_schema = json.loads(
        (HERE / "schemas" / "dataset_review_site_sample.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(manifest_schema).validate(manifest)
    jsonschema.Draft202012Validator(sample_schema).validate(parsed_sample)


def test_site_rejects_unredacted_complete_sample(tmp_path: Path) -> None:
    uid = "a" * 64
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(review_request(uid, doc_id="1")) + "\n")
    samples = tmp_path / "samples.jsonl"
    row = complete_sample(uid, "secret", doc_id="1")
    row["high_precision_identifier_patterns_masked"] = False
    samples.write_text(json.dumps(row) + "\n")
    sample_receipt = tmp_path / "samples-receipt.json"
    write_packet_receipt(sample_receipt, packet=samples, requests=requests, rows=1)
    with pytest.raises(ValueError, match="masking/text attestation"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=HERE / "configs" / "sources.json",
                quality_summary=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=samples,
                complete_samples_receipt=sample_receipt,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )


def test_site_filters_supplemental_complete_samples_and_emits_no_hidden_text(
    tmp_path: Path,
) -> None:
    visible_uid = hashlib.sha256(b"visible").hexdigest()
    hidden_uid = hashlib.sha256(b"supplemental").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        "\n".join(
            [
                json.dumps(review_request(visible_uid)),
                json.dumps(
                    review_request(
                        hidden_uid,
                        source_id="kallipos",
                        repo_id="glossAPI/Apothetirio_Kallipos",
                        dataset="kallipos",
                        doc_id="hidden-doc",
                    )
                ),
            ]
        )
        + "\n"
    )
    hidden_text = "HIDDEN_SUPPLEMENTAL_TEXT_MUST_NOT_ENTER_SITE"
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        "\n".join(
            [
                json.dumps(complete_sample(visible_uid, "ορατό κείμενο")),
                json.dumps(
                    complete_sample(
                        hidden_uid,
                        hidden_text,
                        source_id="kallipos",
                        repo_id="glossAPI/Apothetirio_Kallipos",
                        dataset="kallipos",
                        doc_id="hidden-doc",
                    )
                ),
            ]
        )
        + "\n"
    )
    sample_receipt = tmp_path / "samples-receipt.json"
    write_packet_receipt(sample_receipt, packet=samples, requests=requests, rows=2)
    output = tmp_path / "site"
    SITE.build_site(
        SimpleNamespace(
            inventory=HERE / "configs" / "post_december_inventory.json",
            evaluations=HERE / "configs" / "dataset_review_evaluations.json",
            sources_config=HERE / "configs" / "sources.json",
            quality_summary=None,
            review_requests=requests,
            review_responses=None,
            admission=None,
            novelty=None,
            complete_samples=samples,
            complete_samples_receipt=sample_receipt,
            output_dir=output,
            replace=False,
        )
    )
    data = json.loads((output / "site_data.json").read_text())
    assert data["overview"]["complete_samples"] == 1
    assert data["overview"]["complete_samples_excluded_outside_inventory"] == 1
    assert len(list((output / "samples").glob("*.json"))) == 1
    emitted = "\n".join(
        path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file()
    )
    assert hidden_uid not in emitted
    assert hidden_text not in emitted


def test_site_rejects_sample_receipt_and_profile_hash_drift(tmp_path: Path) -> None:
    uid = hashlib.sha256(b"receipt-drift").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(review_request(uid)) + "\n")
    samples = tmp_path / "samples.jsonl"
    row = complete_sample(uid, "κείμενο")
    row["profile_text_sha256"] = "f" * 64
    samples.write_text(json.dumps(row) + "\n")
    receipt_path = tmp_path / "samples-receipt.json"
    write_packet_receipt(receipt_path, packet=samples, requests=requests, rows=1)
    with pytest.raises(ValueError, match="masking/text attestation"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=HERE / "configs" / "sources.json",
                quality_summary=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=samples,
                complete_samples_receipt=receipt_path,
                output_dir=tmp_path / "site",
                replace=False,
            )
        )
    receipt = json.loads(receipt_path.read_text())
    receipt["review_requests"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="packet/upstream receipt drift"):
        SITE.build_site(
            SimpleNamespace(
                inventory=HERE / "configs" / "post_december_inventory.json",
                evaluations=HERE / "configs" / "dataset_review_evaluations.json",
                sources_config=HERE / "configs" / "sources.json",
                quality_summary=None,
                review_requests=requests,
                review_responses=None,
                admission=None,
                novelty=None,
                complete_samples=samples,
                complete_samples_receipt=receipt_path,
                output_dir=tmp_path / "site-two",
                replace=False,
            )
        )


def test_site_manifest_validation_rejects_tamper_extra_and_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "site"
    args = SimpleNamespace(
        inventory=HERE / "configs" / "post_december_inventory.json",
        evaluations=HERE / "configs" / "dataset_review_evaluations.json",
        sources_config=HERE / "configs" / "sources.json",
        quality_summary=None,
        review_requests=None,
        review_responses=None,
        admission=None,
        novelty=None,
        complete_samples=None,
        complete_samples_receipt=None,
        output_dir=output,
        replace=False,
    )
    SITE.build_site(args)
    site_data = output / "site_data.json"
    original = site_data.read_bytes()
    site_data.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="receipt drift"):
        SITE.validate_site_directory(output)
    site_data.write_bytes(original)
    extra = output / "unexpected.txt"
    extra.write_text("unexpected")
    with pytest.raises(ValueError, match="inventory drift"):
        SITE.validate_site_directory(output)
    extra.unlink()
    symlink = output / "linked"
    symlink.symlink_to(output / "index.html")
    with pytest.raises(ValueError, match="symlinks"):
        SITE.validate_site_directory(output)


def test_full_quality_scope_is_selected_population_not_corpus_wide(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "dataset_quality_summary_v1",
                "status": "passed",
                "scan_mode": "full_scan",
                "selected_source_ids": ["alpha", "beta"],
                "excluded_source_ids": ["nanochat_base"],
                "global": {"documents": 9},
                "repositories": [],
            }
        )
    )
    _, scope = SITE.load_quality(path)
    assert scope == {
        "scan_mode": "full_scan",
        "documents": 9,
        "is_corpus_wide": False,
        "label": "Full scan of selected canonical sources",
        "selected_source_ids": ["alpha", "beta"],
        "excluded_source_ids": ["nanochat_base"],
    }
    value = json.loads(path.read_text())
    value["excluded_source_ids"] = ["alpha"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="overlap"):
        SITE.load_quality(path)


def test_normalized_shard_loader_is_manifest_exact(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    shard = root / "candidate" / "part.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"PAR1-fixture")
    manifest = {
        "schema_version": "full_cpt_normalization_manifest_v1",
        "output": str(root.resolve()),
        "sources": [
            {
                "source_id": "candidate",
                "shards": [
                    {
                        "path": str(shard.resolve()),
                        "bytes": shard.stat().st_size,
                        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                        "rows": 1,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    _, bindings, excluded = QUALITY.load_normalized_shards(
        path, root, include_source_ids=set(), include_base=False
    )
    assert len(bindings) == 1
    assert excluded == []
    rogue = root / "rogue.parquet"
    rogue.write_bytes(b"PAR1-rogue")
    with pytest.raises(ValueError, match="inventory differs"):
        QUALITY.load_normalized_shards(
            path, root, include_source_ids=set(), include_base=False
        )


def test_quantile_sample_size_is_bound_into_resume_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "normalization.json"
    build = tmp_path / "build.json"
    shard_path = tmp_path / "part.parquet"
    manifest.write_text("{}")
    build.write_text("{}")
    shard_path.write_bytes(b"fixture")
    shard = QUALITY.ShardBinding(
        source_id="candidate",
        path=shard_path,
        relative_path="candidate/part.parquet",
        bytes=shard_path.stat().st_size,
        sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
        rows=1,
    )
    base = dict(
        scan_mode="full_scan",
        normalization_manifest=manifest,
        canonical_root=tmp_path,
        build_receipt=build,
        expected_commit=QUALITY.PINNED_GLOSSAPI_COMMIT,
        batch_size=4096,
        threads=256,
    )
    first = QUALITY.diagnostics_contract(
        SimpleNamespace(**base, quantile_sample_size=1024),
        shards=[shard],
        excluded=["nanochat_base"],
        sample_input_shards=None,
        sample_contract=None,
    )
    second = QUALITY.diagnostics_contract(
        SimpleNamespace(**base, quantile_sample_size=2048),
        shards=[shard],
        excluded=["nanochat_base"],
        sample_input_shards=None,
        sample_contract=None,
    )
    assert first["quantile_sample_size"] == 1024
    assert QUALITY.sha256_json(first) != QUALITY.sha256_json(second)


def test_raw_structural_metrics_and_replacement_character_are_not_double_counted() -> (
    None
):
    text = """ΠΕΡΙΕΧΟΜΕΝΑ
1. Εισαγωγή ........ 3
Βιβλιογραφία
| α | β |
|---|---|
ΑΔΑ: ΑΒΓΔ-123
χαλασμένο � Ã©
"""
    metrics = QUALITY.raw_metrics(
        text, private_data_true=True, corrected_version_present=True
    )
    assert metrics["raw_replacement_characters"] == 1
    assert metrics["raw_mojibake_markers"] == 1
    assert metrics["raw_replacement_per_1000_chars"] > 0
    assert metrics["raw_mojibake_per_1000_chars"] > 0
    assert metrics["bibliography_header_detected"] is True
    assert metrics["toc_header_detected"] is True
    assert metrics["raw_markdown_table_lines"] == 2
    assert metrics["isolated_ada_stamp_lines"] == 1
    assert metrics["private_data_true"] is True
    assert metrics["corrected_version_present"] is True

    assert QUALITY.metadata_flags(
        json.dumps(
            {
                "metadata_json": json.dumps(
                    {"privateData": "true", "correctedVersionId": "v2"}
                )
            }
        )
    ) == (True, True)
    with pytest.raises(ValueError, match="source_metadata_json"):
        QUALITY.metadata_flags("{broken")


def test_group_stats_reports_zero_rates_and_template_concentration() -> None:
    base = {
        "source_dataset": "diavgeia",
        "original_characters": 100,
        "original_bytes_utf8": 120,
        **{name: 0.0 for name in QUALITY.DISTRIBUTION_METRICS},
        "raw_html_tags": 0,
        "raw_mojibake_markers": 0,
        "raw_replacement_characters": 0,
        "raw_control_characters": 0,
        "raw_unique_line_fraction": 1.0,
        "raw_one_token_line_fraction": 0.0,
        "raw_markdown_table_lines": 0,
        "bibliography_header_detected": False,
        "toc_header_detected": False,
        "digital_governance_footer_detected": False,
        "personnel_cue_detected": False,
        "isolated_ada_stamp_lines": 0,
        "private_data_true": False,
        "corrected_version_present": False,
        "direct_identifier_match_count": 0,
        "cleaner_is_empty": False,
        "zero_badness_zero_greek_guard": False,
    }
    stats = QUALITY.GroupStats(reservoir_size=100)
    for index, template in enumerate(["a" * 64, "a" * 64, "b" * 64]):
        stats.add(
            {
                **base,
                "document_id": hashlib.sha256(str(index).encode()).hexdigest(),
                "structural_template_id": template,
            }
        )
    result = stats.finish()
    assert result["document_rates"]["html_rate"] == 0.0
    assert result["document_rates"]["bibliography_header_rate"] == 0.0
    assert result["template_concentration"] == {
        "documents_with_template": 3,
        "unique_templates": 2,
        "top_1_fraction": pytest.approx(2 / 3),
        "top_10_fraction": 1.0,
    }


def detailed_noise_row(path: Path, *, score: float, greek: int) -> tuple[object, ...]:
    values: dict[str, object] = {}
    for name in QUALITY.NOISE_FIELDS:
        if name == "rust_noise_badness_score":
            values[name] = score
        elif name == "rust_noise_greek_characters":
            values[name] = greek
        elif name in QUALITY.FLOAT_NOISE_FIELDS:
            values[name] = 0.0
        elif name in QUALITY.INTEGER_NOISE_FIELDS:
            values[name] = 0
        else:
            values[name] = ""
    return (str(path), *(values[name] for name in QUALITY.NOISE_FIELDS))


def test_exact_review_sample_packet_is_bound_and_uses_hashed_display_id(
    tmp_path: Path,
) -> None:
    normalization = tmp_path / "normalization.json"
    normalization.write_text('{"schema_version":"full_cpt_normalization_manifest_v1"}')
    raw_doc_id = "https://private.example/person/123"
    display_id = QUALITY.display_document_id(raw_doc_id)
    uid = hashlib.sha256(b"selected").hexdigest()
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps(
            {
                "schema_version": "source_quality_review_request_v1",
                "reviewer_slot": "primary",
                "sample_id": uid,
                "source_dataset": "diavgeia",
                "source": {
                    "source_id": "diavgeia",
                    "source_repo_id": "glossAPI/diavgeia",
                    "source_revision": "a" * 40,
                    "source_doc_id": raw_doc_id,
                },
            }
        )
        + "\n"
    )
    text = "πλήρως ανωνυμοποιημένο κείμενο"
    packet = tmp_path / "samples.jsonl"
    row = complete_sample(uid, text, doc_id=raw_doc_id)
    row["normalized_text_sha256"] = "c" * 64
    assert row["display_document_id"] == display_id
    packet.write_text(json.dumps(row) + "\n")
    receipt_path = tmp_path / "sample-receipt.json"
    write_packet_receipt(
        receipt_path,
        packet=packet,
        requests=requests,
        rows=1,
        normalization=normalization,
    )
    rows, inputs = QUALITY.load_review_sample_packet(
        packet_path=packet,
        receipt_path=receipt_path,
        requests_path=requests,
        normalization_manifest=normalization,
    )
    assert len(rows) == len(inputs) == 1
    assert "source_doc_id" not in rows[0]
    assert "display_document_id" not in rows[0]
    assert raw_doc_id not in json.dumps(rows)
    assert rows[0]["profile_text_variant"] == (
        "high_precision_identifier_masked_review_sample"
    )


def test_rust_batch_checkpoint_and_zero_greek_guard(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    class FakeNoise:
        calls = 0

        def score_markdown_directory_detailed(self, root: str, threads: int):
            self.calls += 1
            paths = sorted(Path(root).glob("*.md"))
            return [
                detailed_noise_row(paths[0], score=0.0, greek=0),
                detailed_noise_row(paths[1], score=0.0, greek=6),
            ]

    class FakeCleaner:
        calls = 0

        def run_complete_pipeline(
            self,
            input_dir: str,
            output_dir: str,
            report: str,
            scripts: list[str],
            threads: int,
            write_cleaned_files: bool,
        ) -> None:
            self.calls += 1
            assert scripts == ["greek", "latin"]
            assert write_cleaned_files is False
            names = [
                f"{path.stem}.pdf" for path in sorted(Path(input_dir).glob("*.md"))
            ]
            pq.write_table(
                pa.table(
                    {
                        "file_name": names,
                        "badness_score_all_chars": [0.0, 0.0],
                        "percentage_greek_cleaned": [0.0, 100.0],
                        "percentage_latin_cleaned": [100.0, 0.0],
                        "char_count_no_comments": [7, 6],
                        "is_empty": [False, False],
                    }
                ),
                report,
            )

    texts = ["English", "κείμενο"]
    rows = []
    for index, text in enumerate(texts):
        rows.append(
            {
                "source_id": "candidate",
                "source_dataset": "candidate",
                "source_repo_id": "glossAPI/candidate",
                "source_revision": "b" * 40,
                "source_doc_id": str(index),
                "stable_uid": hashlib.sha256(f"uid-{index}".encode()).hexdigest(),
                "normalized_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text": text,
            }
        )
    input_path = tmp_path / "input.parquet"
    input_path.write_bytes(b"fixture")
    shard = QUALITY.ShardBinding(
        source_id="candidate",
        path=input_path,
        relative_path="candidate/input.parquet",
        bytes=input_path.stat().st_size,
        sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        rows=2,
    )
    build_receipt = tmp_path / "rust-build.json"
    build_receipt.write_text("{}")
    noise = FakeNoise()
    cleaner = FakeCleaner()
    runtime = QUALITY.RustRuntime(
        noise=noise,
        cleaner=cleaner,
        receipt={},
        receipt_path=build_receipt,
    )
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    receipt = QUALITY.process_batch(
        rows=rows,
        shard=shard,
        batch_index=0,
        row_start=0,
        output_root=output,
        scratch_root=scratch,
        contract_sha256="c" * 64,
        runtime=runtime,
        threads=2,
    )
    data = pq.read_table(
        Path(receipt["receipt"]["path"]).parent / "documents.parquet"
    ).to_pylist()
    assert [row["zero_badness_zero_greek_guard"] for row in data] == [True, False]
    assert [row["noise_score_interpretation"] for row in data] == [
        "guarded_zero_score_without_greek",
        "zero_score_with_greek",
    ]
    assert not list(scratch.iterdir())
    assert noise.calls == cleaner.calls == 1

    class MustNotRun:
        def __getattr__(self, name: str):
            raise AssertionError(f"checkpoint resume called Rust: {name}")

    resumed = QUALITY.process_batch(
        rows=rows,
        shard=shard,
        batch_index=0,
        row_start=0,
        output_root=output,
        scratch_root=scratch,
        contract_sha256="c" * 64,
        runtime=QUALITY.RustRuntime(
            noise=MustNotRun(),
            cleaner=MustNotRun(),
            receipt={},
            receipt_path=build_receipt,
        ),
        threads=2,
    )
    assert resumed["output"]["sha256"] == receipt["output"]["sha256"]
    document_output, global_summary, repositories = QUALITY.consolidate_batches(
        [receipt], output_root=output, reservoir_size=100
    )
    assert document_output["rows"] == global_summary["documents"] == 2
    assert repositories[0]["repo_id"] == "glossAPI/candidate"
    jsonschema = pytest.importorskip("jsonschema")
    document_contract = json.loads(
        (HERE / "schemas" / "dataset_quality_document.schema.json").read_text()
    )
    for row in pq.read_table(
        output / "dataset_quality_document_v1.parquet"
    ).to_pylist():
        jsonschema.Draft202012Validator(document_contract).validate(row)


def test_build_receipt_can_bind_staged_modules_to_atomic_publish_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "glossapi"
    for crate in ("glossapi_rs_noise", "glossapi_rs_cleaner"):
        lock = source / "rust" / crate / "Cargo.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"lock for {crate}")
    staged = tmp_path / "runtime.partial" / "modules"
    staged.mkdir(parents=True)
    module_paths = {}
    for name in ("glossapi_rs_noise", "glossapi_rs_cleaner"):
        path = staged / f"{name}.so"
        path.write_bytes(name.encode())
        module_paths[name] = path.resolve()
    published = tmp_path / "runtime" / "modules"

    def fake_git_output(root: Path, *args: str) -> str:
        if args == ("rev-parse", "--is-inside-work-tree"):
            return "true"
        if args == ("rev-parse", "HEAD"):
            return QUALITY.PINNED_GLOSSAPI_COMMIT
        if args == ("status", "--porcelain", "--untracked-files=normal"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(QUALITY, "git_output", fake_git_output)
    monkeypatch.setattr(QUALITY, "module_path", lambda name: module_paths[name])
    monkeypatch.setattr(
        QUALITY, "tool_version", lambda command, *arguments: f"{command} test-version"
    )
    output = tmp_path / "runtime.partial" / "build_receipt.json"
    assert (
        QUALITY.build_runtime_receipt(
            SimpleNamespace(
                glossapi_root=source,
                expected_commit=QUALITY.PINNED_GLOSSAPI_COMMIT,
                module_root=staged,
                published_module_root=published,
                maturin_version="1.9.4",
                output=output,
            )
        )
        == 0
    )
    value = json.loads(output.read_text())
    assert {Path(row["path"]).parent for row in value["modules"]} == {
        published.resolve()
    }
    assert all(Path(row["path"]).name.endswith(".so") for row in value["modules"])
    assert all(not Path(row["path"]).exists() for row in value["modules"])
    assert value["runtime"]["rustc"] == "rustc test-version"
    assert value["runtime"]["cargo"] == "cargo test-version"
    assert value["runtime"]["maturin"] == "1.9.4"


def test_complete_sample_redaction_covers_identity_ipv6_and_email() -> None:
    text = (
        "email user@example.gr IPv6 2001:0db8:85a3:0000:0000:8a2e:0370:7334 "
        "ΑΔΤ: ΑΒ123456"
    )
    redacted, counts = EXPORTER.redact_complete_text(text)
    assert "user@example.gr" not in redacted
    assert "2001:0db8" not in redacted
    assert "ΑΒ123456" not in redacted
    assert counts["email"] == counts["ipv6"] == counts["identity"] == 1


def test_sample_export_omits_raw_source_document_identifier(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "canonical"
    shard = root / "diavgeia" / "part.parquet"
    shard.parent.mkdir(parents=True)
    raw_doc_id = "https://private.example/records/person-123"
    text = "κείμενο user@example.gr"
    uid = hashlib.sha256(b"exported").hexdigest()
    pq.write_table(
        pa.table(
            {
                "source_id": ["diavgeia"],
                "stable_uid": [uid],
                "source_repo_id": ["glossAPI/diavgeia"],
                "source_revision": ["a" * 40],
                "source_dataset": ["diavgeia"],
                "source_doc_id": [raw_doc_id],
                "normalized_text_sha256": [hashlib.sha256(text.encode()).hexdigest()],
                "source_metadata_json": [json.dumps({"correctedVersionId": "v2"})],
                "text": [text],
            }
        ),
        shard,
    )
    manifest_path = tmp_path / "normalization.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_normalization_manifest_v1",
                "output": str(root.resolve()),
                "sources": [
                    {
                        "source_id": "diavgeia",
                        "shards": [
                            {
                                "path": str(shard.resolve()),
                                "bytes": shard.stat().st_size,
                                "sha256": hashlib.sha256(
                                    shard.read_bytes()
                                ).hexdigest(),
                                "rows": 1,
                            }
                        ],
                    }
                ],
            }
        )
    )
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps(
            {
                "schema_version": "source_quality_review_request_v1",
                "reviewer_slot": "primary",
                "sample_id": uid,
                "source_dataset": "diavgeia",
                "source": {
                    "source_id": "diavgeia",
                    "source_repo_id": "glossAPI/diavgeia",
                    "source_revision": "a" * 40,
                    "source_doc_id": raw_doc_id,
                },
            }
        )
        + "\n"
    )
    packet = tmp_path / "complete.jsonl"
    packet_receipt = tmp_path / "complete-receipt.json"
    assert (
        EXPORTER.export_samples(
            SimpleNamespace(
                output=packet,
                receipt=packet_receipt,
                resume=False,
                review_requests=requests,
                normalization_manifest=manifest_path,
                canonical_root=root,
                scratch_dir=tmp_path / "scratch",
                batch_size=8,
            )
        )
        == 0
    )
    source = packet.read_text(encoding="utf-8")
    row = json.loads(source)
    assert raw_doc_id not in source
    assert "source_doc_id" not in row
    assert row["display_document_id"] == QUALITY.display_document_id(raw_doc_id)
    assert "user@example.gr" not in row["text"]
    assert row["profile_text_variant"] == (
        "high_precision_identifier_masked_review_sample"
    )
    assert row["private_data_true"] is False
    assert row["corrected_version_present"] is True
    receipt_value = json.loads(packet_receipt.read_text())
    assert receipt_value["output"]["path"] == packet.name
    assert (
        receipt_value["checkpoint_inventory_sha256"]
        == hashlib.sha256(
            EXPORTER.canonical_json(receipt_value["checkpoint_inventory"]).encode()
        ).hexdigest()
    )
    jsonschema = pytest.importorskip("jsonschema")
    packet_schema = json.loads(
        (HERE / "schemas" / "dataset_review_complete_sample.schema.json").read_text()
    )
    receipt_schema = json.loads(
        (
            HERE
            / "schemas"
            / "dataset_review_complete_sample_packet_receipt.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(packet_schema).validate(row)
    jsonschema.Draft202012Validator(receipt_schema).validate(receipt_value)
    assert (
        EXPORTER.export_samples(
            SimpleNamespace(
                output=packet,
                receipt=packet_receipt,
                resume=True,
                review_requests=requests,
                normalization_manifest=manifest_path,
                canonical_root=root,
                scratch_dir=tmp_path / "scratch",
                batch_size=8,
            )
        )
        == 0
    )


def test_clariden_wrapper_is_cpu_only_resumable_and_4096_bounded() -> None:
    wrapper = (HERE / "clariden" / "41_profile_dataset_quality_rust.sbatch").read_text()
    builder = (
        HERE / "clariden" / "06_build_glossapi_quality_runtime.sbatch"
    ).read_text()
    submit = (HERE / "clariden" / "submit.sh").read_text()
    assert "#SBATCH --cpus-per-task=256" in wrapper
    assert "#SBATCH --gres" not in wrapper
    assert "phase04_require_cpu_request" in wrapper
    assert "BATCH_SIZE=4096" in wrapper
    assert "QUALITY_STAGE=35-dataset-quality-sample" in wrapper
    assert "QUALITY_STAGE=15-dataset-quality-full" in wrapper
    assert 'phase04_stage_require_upstream "10-normalize"' in wrapper
    assert 'phase04_stage_require_upstream "30-review-packet"' in wrapper
    assert "--review-sample-packet" in wrapper
    assert '--checkpoint-dir "$PHASE04_STAGE_DIR/sample-export-checkpoints"' in wrapper
    assert 'phase04_stage_bind_parameter scan_mode "$QUALITY_MODE"' in wrapper
    assert "--resume" in wrapper
    assert 'CUDA_VISIBLE_DEVICES=""' in wrapper
    assert "dataset-quality|dataset-quality-sample|35-dataset-quality-sample" in submit
    assert "dataset-quality-full|15-dataset-quality-full" in submit
    assert "QUALITY_MODE=review_sample" in submit
    assert "QUALITY_MODE=full_scan" in submit

    assert "#SBATCH --cpus-per-task=128" in builder
    assert "#SBATCH --gres" not in builder
    assert "phase04_require_cpu_request" in builder
    assert "maturin" in builder and "--locked" in builder
    assert "CARGO_TARGET_DIR" in builder
    assert '--module-root "$PARTIAL/modules"' in builder
    assert '--published-module-root "$GLOSSAPI_QUALITY_MODULE_DIR"' in builder
    assert 'mv "$PARTIAL" "$GLOSSAPI_QUALITY_RUNTIME_ROOT"' in builder
    assert "build-quality-runtime" in submit


def test_new_json_schemas_are_parseable_and_versioned() -> None:
    expected = {
        "glossapi_rust_quality_build_receipt.schema.json": "glossapi_rust_quality_build_receipt_v1",
        "dataset_quality_document.schema.json": "dataset_quality_document_v1",
        "dataset_quality_summary.schema.json": "dataset_quality_summary_v1",
        "dataset_review_complete_sample.schema.json": "dataset_review_complete_sample_v1",
        "dataset_review_complete_sample_packet_receipt.schema.json": (
            "dataset_review_complete_sample_packet_receipt_v1"
        ),
        "dataset_review_sample_export_contract.schema.json": (
            "dataset_review_sample_export_contract_v1"
        ),
        "dataset_review_sample_export_shard_checkpoint.schema.json": (
            "dataset_review_sample_export_shard_checkpoint_v1"
        ),
        "dataset_review_site_sample.schema.json": "dataset_review_site_sample_v1",
        "dataset_review_site_manifest.schema.json": "dataset_review_site_manifest_v1",
    }
    for name, version in expected.items():
        value = json.loads((HERE / "schemas" / name).read_text())
        assert value["$schema"].endswith("2020-12/schema")
        schema_version = value["properties"]["schema_version"]
        assert schema_version["const"] == version
