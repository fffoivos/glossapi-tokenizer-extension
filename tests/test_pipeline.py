from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from glossapi_corpus_cli import pipeline


def test_kallipos_title_and_author_extraction() -> None:
    raw_json = json.dumps(
        {
            "Συγγραφείς": ["Καλογηράτου, Ζαχαρούλα", "Μονοβασίλης, Θεόδωρος"],
            "Βιβλιογραφική Αναφορά": [
                "Καλογηράτου, Ζ., & Μονοβασίλης, Θ. (2024).",
                "Αριθμητική επίλυση διαφορικών εξισώσεων",
                "[Προπτυχιακό εγχειρίδιο].",
            ],
            "Υπότιτλος": "Παράδειγμα",
            "ISBN": "123456789",
        },
        ensure_ascii=False,
    )
    title, author, payload = pipeline.kallipos_metadata_payload(raw_json, "book")
    assert title == "Αριθμητική επίλυση διαφορικών εξισώσεων"
    assert author == "Καλογηράτου, Ζαχαρούλα; Μονοβασίλης, Θεόδωρος"
    parsed = json.loads(payload)
    assert parsed["Υπότιτλος"] == "Παράδειγμα"
    assert parsed["document_type"] == "book"
    assert "ISBN" not in parsed


def test_extract_europarl_code() -> None:
    text = "ΚΕΙΜΕΝΑ ΠΟΥ ΕΓΚΡΙΘΗΚΑΝ\nP8_TA(2019)0030\nΤραπεζική Ένωση"
    assert pipeline.extract_europarl_code(text) == "TA-8-2019-0030"


def test_apply_quality_preset_modern_strict() -> None:
    df = pd.DataFrame(
        [
            {
                "source_dataset": "openarchives.gr",
                "greek_badness_score": 24.0,
                "needs_ocr": False,
                "is_empty": False,
                "filter": "ok",
                "is_historical_or_polytonic": False,
            },
            {
                "source_dataset": "openarchives.gr",
                "greek_badness_score": 40.0,
                "needs_ocr": False,
                "is_empty": False,
                "filter": "ok",
                "is_historical_or_polytonic": False,
            },
            {
                "source_dataset": "1000_prwta_xronia_ellhnikhs",
                "greek_badness_score": 80.0,
                "needs_ocr": False,
                "is_empty": False,
                "filter": "ok",
                "is_historical_or_polytonic": True,
            },
        ]
    )
    filtered = pipeline.apply_quality_preset(df, "modern_strict")
    assert len(filtered) == 2


def test_apply_quality_preset_keeps_only_successful_ocr_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "source_doc_id": "plain-ok",
                "source_dataset": "openarchives.gr",
                "greek_badness_score": 24.0,
                "needs_ocr": False,
                "ocr_success": False,
                "is_empty": False,
                "filter": "ok",
                "is_historical_or_polytonic": False,
            },
            {
                "source_doc_id": "ocr-ok",
                "source_dataset": "openarchives.gr",
                "greek_badness_score": 24.0,
                "needs_ocr": True,
                "ocr_success": True,
                "is_empty": False,
                "filter": "ok",
                "is_historical_or_polytonic": False,
            },
            {
                "source_doc_id": "ocr-missing",
                "source_dataset": "openarchives.gr",
                "greek_badness_score": 24.0,
                "needs_ocr": True,
                "ocr_success": False,
                "is_empty": False,
                "filter": "ok",
                "is_historical_or_polytonic": False,
            },
        ]
    )
    filtered = pipeline.apply_quality_preset(df, "modern_strict")
    assert set(filtered["source_doc_id"]) == {"plain-ok", "ocr-ok"}


def test_estimate_nanochat_tokens_uses_ratio() -> None:
    params = pipeline.estimate_nanochat_scaling_params(depth=12)
    target = pipeline.target_nanochat_tokens(depth=12, target_param_data_ratio=10.5)
    assert target == int(params * 10.5)


def test_write_parquet_rows_keeps_canonical_schema_when_later_batches_are_null(tmp_path: Path) -> None:
    rows = [
        {
            "source_dataset": "testset",
            "source_doc_id": "doc-1",
            "text": "κειμενο",
            "title": "Τίτλος",
            "author": None,
            "source_metadata_json": None,
            "is_historical_or_polytonic": False,
            "contains_math": False,
            "contains_latex": False,
            "greek_percentage": 99.0,
            "latin_percentage": 1.0,
            "polytonic_ratio": 0.0,
            "table_ratio": None,
            "greek_badness_score": 5.0,
            "len_greek": 7,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "is_empty": False,
            "filter": "ok",
            "ocr_success": True,
            "quality_method": "test",
            "reevaluated_at": "2026-03-15T00:00:00Z",
        },
        {
            "source_dataset": "testset",
            "source_doc_id": "doc-2",
            "text": "κειμενο",
            "title": None,
            "author": None,
            "source_metadata_json": None,
            "is_historical_or_polytonic": False,
            "contains_math": False,
            "contains_latex": False,
            "greek_percentage": None,
            "latin_percentage": None,
            "polytonic_ratio": None,
            "table_ratio": None,
            "greek_badness_score": None,
            "len_greek": None,
            "mojibake_badness_score": None,
            "needs_ocr": None,
            "is_empty": None,
            "filter": None,
            "ocr_success": None,
            "quality_method": None,
            "reevaluated_at": None,
        },
    ]
    path = tmp_path / "test.parquet"
    count = pipeline.write_parquet_rows(path, rows, batch_size=1, score_missing_quality=False)
    assert count == 2
    frame = pd.read_parquet(path)
    assert len(frame) == 2
    assert frame["greek_percentage"].dtype == "float64"
    assert frame.loc[0, "len_greek"] == 7
    assert pd.isna(frame.loc[1, "len_greek"])


def test_batch_score_missing_quality_scores_missing_badness_even_with_existing_method(monkeypatch) -> None:
    def fake_score_markdown_directory_detailed(input_dir: str, n_threads: int | None):
        paths = sorted(Path(input_dir).glob("*.md"))
        assert len(paths) == 1
        return [(str(paths[0]), 12.5, 3.0, 0.2, 0.01, 123)]

    monkeypatch.setattr(
        pipeline.glossapi_rs_noise,
        "score_markdown_directory_detailed",
        fake_score_markdown_directory_detailed,
    )
    rows = [
        {
            "source_dataset": "HPLT/ell_Grek_ge8_no_mt_clean60",
            "source_doc_id": "hplt-1",
            "text": "κείμενο",
            "greek_badness_score": None,
            "mojibake_badness_score": 0.07,
            "quality_method": "corpus.clean",
            "latin_percentage": 4.0,
        }
    ]

    scored = pipeline.batch_score_missing_quality(rows, "HPLT/ell_Grek_ge8_no_mt_clean60")

    assert scored[0]["greek_badness_score"] == 12.5
    assert scored[0]["mojibake_badness_score"] == 0.07
    assert scored[0]["quality_method"] == "corpus.clean"
    assert scored[0]["latin_percentage"] == 4.0
    assert scored[0]["table_ratio"] == 0.2
    assert scored[0]["polytonic_ratio"] == 0.01
    assert scored[0]["len_greek"] == 123


def test_batch_score_missing_quality_preserves_existing_badness_without_rescoring(monkeypatch) -> None:
    def fail_score_markdown_directory_detailed(input_dir: str, n_threads: int | None):
        raise AssertionError("existing scored rows should not be rescored")

    monkeypatch.setattr(
        pipeline.glossapi_rs_noise,
        "score_markdown_directory_detailed",
        fail_score_markdown_directory_detailed,
    )
    rows = [
        {
            "source_dataset": "already-scored",
            "source_doc_id": "doc-1",
            "text": "κείμενο",
            "greek_badness_score": 3.0,
            "mojibake_badness_score": 0.08,
            "quality_method": "previous-cleaner",
        }
    ]

    scored = pipeline.batch_score_missing_quality(rows, "already-scored")

    assert scored == rows


def test_metadata_json_normalizes_placeholder_url_to_null() -> None:
    payload = pipeline.metadata_json({"url": "  NA  ", "permanent_url": "https://example.com/doc"})
    parsed = json.loads(payload)
    assert "url" not in parsed
    assert parsed["permanent_url"] == "https://example.com/doc"


def test_exact_metadata_dedup_keeps_best_row_for_same_safe_key() -> None:
    rows = [
        {
            "source_dataset": "HuggingFaceFW/finewiki",
            "source_doc_id": "doc-1",
            "text": "Κείμενο",
            "title": "Τίτλος",
            "author": None,
            "source_metadata_json": json.dumps({"url": " https://example.org/wiki/A "}, ensure_ascii=False),
            "needs_ocr": False,
            "is_empty": False,
            "greek_badness_score": None,
        },
        {
            "source_dataset": "HuggingFaceFW/finewiki",
            "source_doc_id": "doc-2",
            "text": "Κείμενο με λίγο περισσότερο περιεχόμενο",
            "title": " τίτλος ",
            "author": None,
            "source_metadata_json": json.dumps({"url": "https://example.org/wiki/a"}, ensure_ascii=False),
            "needs_ocr": False,
            "is_empty": False,
            "greek_badness_score": None,
        },
        {
            "source_dataset": "HuggingFaceFW/finewiki",
            "source_doc_id": "doc-3",
            "text": "Άσχετο κείμενο",
            "title": "Διαφορετικός τίτλος",
            "author": None,
            "source_metadata_json": json.dumps({"url": "https://example.org/wiki/b"}, ensure_ascii=False),
            "needs_ocr": False,
            "is_empty": False,
            "greek_badness_score": None,
        },
    ]
    deduped = list(pipeline.iter_exact_metadata_dedup(rows))
    assert [row["source_doc_id"] for row in deduped] == ["doc-2", "doc-3"]


def test_maybe_iter_exact_metadata_dedup_passthroughs_unknown_datasets() -> None:
    rows = iter(
        [
            make_canonical_row(
                source_dataset=pipeline.OPENSUBTITLES_EL_DATASET,
                source_doc_id="doc-1",
                text="γεια",
            )
        ]
    )
    deduped = pipeline.maybe_iter_exact_metadata_dedup(pipeline.OPENSUBTITLES_EL_DATASET, rows)
    assert deduped is rows


def test_iter_opensubtitles_el_rows_reads_xml_zip_corpus(tmp_path: Path, monkeypatch) -> None:
    external_root = tmp_path / "external_hf"
    dataset_root = external_root / "OPUS__OpenSubtitles-el-v2018"
    dataset_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset_root / "el.zip", "w") as archive:
        archive.writestr(
            "OpenSubtitles/xml/el/2020/123/1.xml",
            (
                "<document id='1'>"
                "<meta>"
                "<subtitle><confidence>0.75</confidence><language>Greek</language><blocks>12</blocks></subtitle>"
                "<conversion><encoding>utf-8</encoding><tokens>3</tokens><sentences>2</sentences></conversion>"
                "<source><year>2020</year><genre>Drama</genre><country>GR</country></source>"
                "</meta>"
                "<s id='1'><w id='1.1'>γεια</w><w id='1.2'>σου</w></s>"
                "<s id='2'><w id='2.1'>καλημερα</w></s>"
                "</document>"
            ),
        )
        archive.writestr(
            "OpenSubtitles/xml/el/2020/124/2.xml.gz",
            gzip.compress("<document id='2'><s id='1'>τι κανεις</s></document>".encode("utf-8")),
        )
    monkeypatch.setattr(pipeline, "EXTERNAL_ROOT", external_root)
    rows = list(pipeline.iter_opensubtitles_el_rows())
    assert [row["source_doc_id"] for row in rows] == [
        "OpenSubtitles::xml::el::2020::123::1",
        "OpenSubtitles::xml::el::2020::124::2",
    ]
    assert [row["text"] for row in rows] == ["γεια σου\nκαλημερα", "τι κανεις"]
    assert all(row["source_dataset"] == pipeline.OPENSUBTITLES_EL_DATASET for row in rows)
    metadata = json.loads(rows[0]["source_metadata_json"])
    assert metadata["language"] == "el"
    assert metadata["zip_member"] == "OpenSubtitles/xml/el/2020/123/1.xml"
    assert metadata["document_id"] == "1"
    assert metadata["subtitle_language"] == "Greek"
    assert metadata["subtitle_confidence"] == 0.75
    assert metadata["subtitle_blocks"] == 12
    assert metadata["source_year"] == 2020
    assert metadata["source_genre"] == "Drama"
    assert metadata["source_country"] == "GR"
    assert "member_year" not in metadata
    assert "member_group_id" not in metadata
    assert "conversion_tokens" not in metadata


def make_canonical_row(
    *,
    source_dataset: str,
    source_doc_id: str,
    text: str,
    title: str | None = None,
    author: str | None = None,
    greek_badness_score: float | None = None,
    needs_ocr: bool = False,
) -> dict[str, object]:
    return {
        "source_dataset": source_dataset,
        "source_doc_id": source_doc_id,
        "text": text,
        "title": title,
        "author": author,
        "source_metadata_json": None,
        "is_historical_or_polytonic": False,
        "contains_math": False,
        "contains_latex": False,
        "greek_percentage": 99.0,
        "latin_percentage": 1.0,
        "polytonic_ratio": 0.0,
        "table_ratio": 0.0,
        "greek_badness_score": greek_badness_score,
        "mojibake_badness_score": 0.0,
        "needs_ocr": needs_ocr,
        "is_empty": False,
        "filter": "ok",
        "ocr_success": True,
        "quality_method": "test",
        "reevaluated_at": None,
    }


def write_canonical_snapshot(output_root: Path, rows: list[dict[str, object]]) -> None:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pipeline.CANONICAL_ARROW_SCHEMA), data_root / "sample.parquet")


def write_canonical_snapshot_parts(output_root: Path, parts: list[list[dict[str, object]]]) -> None:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    for idx, rows in enumerate(parts):
        pq.write_table(
            pa.Table.from_pylist(rows, schema=pipeline.CANONICAL_ARROW_SCHEMA),
            data_root / f"sample_{idx:02d}.parquet",
        )


def write_source_mix_config(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_builder_bundle(
    bundle_root: Path,
    *,
    rows: list[dict[str, object]],
    strict_groups: dict[str, tuple[str | None, int]] | None = None,
    relaxed_groups: dict[str, tuple[str | None, int]] | None = None,
    near_pairs: list[dict[str, object]] | None = None,
    family_membership: list[dict[str, object]] | None = None,
    candidate_score_floor: float = 0.85,
) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    strict_groups = strict_groups or {}
    relaxed_groups = relaxed_groups or {}
    near_pairs = near_pairs or []
    family_membership = family_membership or []
    doc_rows = []
    for row in rows:
        source_dataset = str(row["source_dataset"])
        source_doc_id = str(row["source_doc_id"])
        doc_key = pipeline.stable_doc_key(source_dataset, source_doc_id)
        strict_hash, strict_size = strict_groups.get(source_doc_id, (f"strict:{source_doc_id}", 1))
        relaxed_hash, relaxed_size = relaxed_groups.get(source_doc_id, (None, 0))
        len_greek = int(row.get("len_greek", len(str(row["text"]))))
        representative_score = float(
            row.get(
                "representative_score",
                max(0.0, len_greek * (1.0 - (float(row.get("greek_badness_score", 0.0) or 0.0) / 10.0))),
            )
        )
        doc_rows.append(
            {
                "doc_key": doc_key,
                "source_dataset": source_dataset,
                "source_doc_id": source_doc_id,
                "strict_exact_group_hash": strict_hash,
                "strict_exact_group_size": strict_size,
                "strict_exact_mixed_source": False,
                "strict_exact_kept_doc_key": doc_key,
                "relaxed_exact_group_hash": relaxed_hash,
                "relaxed_exact_group_size": relaxed_size,
                "relaxed_exact_mixed_source": False,
                "relaxed_exact_kept_doc_key": doc_key if relaxed_hash else None,
                "near_candidate_count": 0,
                "near_best_match_doc_key": None,
                "near_best_match_source_dataset": None,
                "near_best_estimated_jaccard": None,
                "near_best_length_ratio": None,
                "near_cross_dataset_candidate_count": 0,
                "near_same_dataset_candidate_count": 0,
                "needs_ocr": 0,
                "greek_badness_score": row.get("greek_badness_score"),
                "len_greek": len_greek,
                "mojibake_badness_score": 0.0,
                "ocr_success": 1,
                "text_length_for_selection": len(str(row["text"])),
                "representative_score": representative_score,
                "representative_score_version": "representative_score_v1_len_greek_badness10",
                "has_title": bool(row.get("title")),
                "has_author": bool(row.get("author")),
                "dedup_run_id": "test-run",
                "greek_diacritic_policy": "strip",
                "exact_strict_version": "exact_strict_norm_v1",
                "exact_relaxed_version": "exact_relaxed_norm_v5_strip",
                "near_norm_version": "near_norm_v5_strip",
                "tokenization_version": "tokenization_v2",
                "shingle_version": "token_5gram_v1",
                "minhash_version": "minhash_v2",
                "lsh_version": "lsh_v1",
                "selection_version": "selection_v2",
            }
        )
    pq.write_table(pa.Table.from_pylist(doc_rows), bundle_root / "doc_dedup_metadata.parquet")
    if family_membership:
        pq.write_table(pa.Table.from_pylist(family_membership), bundle_root / "dedup_family_membership.parquet")
    pq.write_table(
        pa.Table.from_pylist(near_pairs or []),
        bundle_root / "near_candidate_pairs.parquet",
    )
    files = {
        "doc_metadata": "doc_dedup_metadata.parquet",
        "near_candidate_pairs": "near_candidate_pairs.parquet",
    }
    if family_membership:
        files["family_membership"] = "dedup_family_membership.parquet"
    (bundle_root / "manifest.json").write_text(
        json.dumps(
            {
                "builder_metadata_version": "builder_metadata_v2",
                "candidate_score_floor": candidate_score_floor,
                "builder_default_threshold": candidate_score_floor,
                "builder_exact_stage": "strict_and_relaxed",
                "near_cluster_mode": "representative_validation",
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_builder_dedup_drop_intra_preserves_cross_dataset_duplicates(tmp_path: Path) -> None:
    output_root = tmp_path / "corpus"
    bundle_root = tmp_path / "bundle"
    rows = [
        make_canonical_row(source_dataset="openarchives.gr", source_doc_id="oa-1", text="ενδο 1", greek_badness_score=2.0),
        make_canonical_row(source_dataset="openarchives.gr", source_doc_id="oa-2", text="ενδο 1 περισσότερο", greek_badness_score=1.0),
        make_canonical_row(source_dataset="openarchives.gr", source_doc_id="oa-shared", text="κοινό κείμενο", greek_badness_score=1.0),
        make_canonical_row(source_dataset="greek_phd", source_doc_id="phd-shared", text="κοινό κείμενο", greek_badness_score=1.0),
        make_canonical_row(source_dataset="greek_phd", source_doc_id="phd-unique", text="μοναδικό", greek_badness_score=1.0),
    ]
    write_canonical_snapshot(output_root, rows)
    write_builder_bundle(
        bundle_root,
        rows=rows,
        strict_groups={
            "oa-1": ("strict-intra-oa", 2),
            "oa-2": ("strict-intra-oa", 2),
            "oa-shared": ("strict-shared", 1),
            "phd-shared": ("strict-phd-shared", 1),
            "phd-unique": ("strict-phd-unique", 1),
        },
        near_pairs=[
            {
                "doc_key_left": pipeline.stable_doc_key("openarchives.gr", "oa-shared"),
                "source_dataset_left": "openarchives.gr",
                "source_doc_id_left": "oa-shared",
                "doc_key_right": pipeline.stable_doc_key("greek_phd", "phd-shared"),
                "source_dataset_right": "greek_phd",
                "source_doc_id_right": "phd-shared",
                "estimated_jaccard": 0.91,
                "length_ratio": 1.0,
                "likely_containment_flag": False,
                "accepted_reason": "threshold",
                "bucket_match_bands": 3,
                "shingle_mode": "token",
                "shingle_size": 5,
                "num_perm": 128,
                "bands": 32,
                "rows_per_band": 4,
                "candidate_score_floor": 0.85,
            }
        ],
    )
    payload = pipeline.build_mix_export(
        output_root=output_root,
        mix_output_path=tmp_path / "mix.parquet",
        dedup_metadata_root=bundle_root,
        dedup_action="drop_intra",
        dedup_similarity_threshold=0.9,
    )
    frame = pd.read_parquet(tmp_path / "mix.parquet")
    assert payload["rows_kept"] == 4
    assert set(frame["source_doc_id"]) == {"oa-2", "oa-shared", "phd-shared", "phd-unique"}


def test_builder_dedup_drop_intra_and_inter_share_aware_balances_shared_pool(tmp_path: Path) -> None:
    output_root = tmp_path / "corpus"
    bundle_root = tmp_path / "bundle"
    rows = [
        make_canonical_row(source_dataset="alpha", source_doc_id="a-1", text="κοινό α1"),
        make_canonical_row(source_dataset="beta", source_doc_id="b-1", text="κοινό β1"),
        make_canonical_row(source_dataset="alpha", source_doc_id="a-2", text="κοινό α2"),
        make_canonical_row(source_dataset="beta", source_doc_id="b-2", text="κοινό β2"),
    ]
    write_canonical_snapshot(output_root, rows)
    write_builder_bundle(
        bundle_root,
        rows=rows,
        strict_groups={row["source_doc_id"]: (f"strict:{row['source_doc_id']}", 1) for row in rows},
        near_pairs=[
            {
                "doc_key_left": pipeline.stable_doc_key("alpha", "a-1"),
                "source_dataset_left": "alpha",
                "source_doc_id_left": "a-1",
                "doc_key_right": pipeline.stable_doc_key("beta", "b-1"),
                "source_dataset_right": "beta",
                "source_doc_id_right": "b-1",
                "estimated_jaccard": 0.92,
                "length_ratio": 1.0,
                "likely_containment_flag": False,
                "accepted_reason": "threshold",
                "bucket_match_bands": 3,
                "shingle_mode": "token",
                "shingle_size": 5,
                "num_perm": 128,
                "bands": 32,
                "rows_per_band": 4,
                "candidate_score_floor": 0.85,
            },
            {
                "doc_key_left": pipeline.stable_doc_key("alpha", "a-2"),
                "source_dataset_left": "alpha",
                "source_doc_id_left": "a-2",
                "doc_key_right": pipeline.stable_doc_key("beta", "b-2"),
                "source_dataset_right": "beta",
                "source_doc_id_right": "b-2",
                "estimated_jaccard": 0.92,
                "length_ratio": 1.0,
                "likely_containment_flag": False,
                "accepted_reason": "threshold",
                "bucket_match_bands": 3,
                "shingle_mode": "token",
                "shingle_size": 5,
                "num_perm": 128,
                "bands": 32,
                "rows_per_band": 4,
                "candidate_score_floor": 0.85,
            },
        ],
    )
    payload = pipeline.build_mix_export(
        output_root=output_root,
        mix_output_path=tmp_path / "share_aware.parquet",
        dedup_metadata_root=bundle_root,
        dedup_action="drop_intra_and_inter",
        dedup_similarity_threshold=0.9,
        dedup_inter_dataset_policy="share_aware",
    )
    frame = pd.read_parquet(tmp_path / "share_aware.parquet")
    assert payload["rows_kept"] == 2
    assert sorted(frame["source_dataset"].tolist()) == ["alpha", "beta"]
    assert set(frame["dedup_pool_key"]) == {"shared:alpha+beta"}


def test_external_doc_key_drop_applies_before_selected_input_materialization(tmp_path: Path) -> None:
    output_root = tmp_path / "corpus"
    rows = [
        make_canonical_row(source_dataset="alpha", source_doc_id="a-keep", text="άλφα"),
        make_canonical_row(source_dataset="alpha", source_doc_id="a-drop", text="άλφα seen"),
        make_canonical_row(source_dataset="beta", source_doc_id="b-keep", text="βήτα"),
    ]
    write_canonical_snapshot(output_root, rows)
    drop_path = tmp_path / "apertus_drop.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"doc_key": pipeline.stable_doc_key("alpha", "a-drop")}],
            schema=pa.schema([("doc_key", pa.string())]),
        ),
        drop_path,
    )

    payload = pipeline.materialize_streaming_mix_selected_input(
        output_root=output_root,
        destination=tmp_path / "selected_input.parquet",
        include_sources=None,
        exclude_sources=None,
        exclude_needs_ocr_sources=None,
        quality_preset="none",
        historical_mode="include",
        math_mode="include",
        latex_mode="include",
        dedup_metadata_root=None,
        dedup_action="ignore",
        dedup_exact_stage="strict_and_relaxed",
        dedup_similarity_threshold=None,
        dedup_inter_dataset_policy="share_aware",
        dedup_source_weights_path=None,
        exclude_doc_keys_path=drop_path,
    )

    frame = pd.read_parquet(tmp_path / "selected_input.parquet")
    assert sorted(frame["source_doc_id"].tolist()) == ["a-keep", "b-keep"]
    assert payload["external_drop_summary"]["excluded_rows"] == 1
    assert payload["selected_input"]["rows"] == 2


def test_builder_dedup_share_aware_uses_projected_chars_not_family_counts(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    frame = pd.DataFrame(
        [
            {"source_dataset": "alpha", "source_doc_id": "a-1", "text": "α" * 10, "chars": 10},
            {"source_dataset": "beta", "source_doc_id": "b-1", "text": "β" * 100, "chars": 100},
            {"source_dataset": "alpha", "source_doc_id": "a-2", "text": "γ" * 10, "chars": 10},
            {"source_dataset": "beta", "source_doc_id": "b-2", "text": "δ" * 100, "chars": 100},
        ]
    )
    rows = [
        make_canonical_row(source_dataset="alpha", source_doc_id="a-1", text="α" * 10, greek_badness_score=1.0),
        make_canonical_row(source_dataset="beta", source_doc_id="b-1", text="β" * 100, greek_badness_score=1.0),
        make_canonical_row(source_dataset="alpha", source_doc_id="a-2", text="γ" * 10, greek_badness_score=1.0),
        make_canonical_row(source_dataset="beta", source_doc_id="b-2", text="δ" * 100, greek_badness_score=1.0),
    ]
    write_builder_bundle(
        bundle_root,
        rows=rows,
        strict_groups={row["source_doc_id"]: (f"strict:{row['source_doc_id']}", 1) for row in rows},
        near_pairs=[
            {
                "doc_key_left": pipeline.stable_doc_key("alpha", "a-1"),
                "source_dataset_left": "alpha",
                "source_doc_id_left": "a-1",
                "doc_key_right": pipeline.stable_doc_key("beta", "b-1"),
                "source_dataset_right": "beta",
                "source_doc_id_right": "b-1",
                "estimated_jaccard": 0.92,
                "length_ratio": 1.0,
                "likely_containment_flag": False,
                "accepted_reason": "threshold",
                "bucket_match_bands": 3,
                "shingle_mode": "token",
                "shingle_size": 5,
                "num_perm": 128,
                "bands": 32,
                "rows_per_band": 4,
                "candidate_score_floor": 0.85,
            },
            {
                "doc_key_left": pipeline.stable_doc_key("alpha", "a-2"),
                "source_dataset_left": "alpha",
                "source_doc_id_left": "a-2",
                "doc_key_right": pipeline.stable_doc_key("beta", "b-2"),
                "source_dataset_right": "beta",
                "source_doc_id_right": "b-2",
                "estimated_jaccard": 0.92,
                "length_ratio": 1.0,
                "likely_containment_flag": False,
                "accepted_reason": "threshold",
                "bucket_match_bands": 3,
                "shingle_mode": "token",
                "shingle_size": 5,
                "num_perm": 128,
                "bands": 32,
                "rows_per_band": 4,
                "candidate_score_floor": 0.85,
            },
        ],
    )

    deduped, _ = pipeline.apply_builder_dedup(
        frame,
        dedup_metadata_root=bundle_root,
        dedup_action="drop_intra_and_inter",
        dedup_exact_stage="strict_and_relaxed",
        dedup_similarity_threshold=0.9,
        dedup_inter_dataset_policy="share_aware",
        dedup_source_weights_path=None,
    )

    assert deduped["source_doc_id"].tolist() == ["a-1", "a-2"]
    assert deduped["chars"].sum() == 20
    assert set(deduped["dedup_pool_key"]) == {"shared:alpha+beta"}


def test_builder_dedup_family_membership_prefers_best_effective_greek_length(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    frame = pd.DataFrame(
        [
            {"source_dataset": "alpha", "source_doc_id": "doc-a", "text": "άλφα"},
            {"source_dataset": "alpha", "source_doc_id": "doc-b", "text": "βήτα"},
        ]
    )
    rows = [
        make_canonical_row(source_dataset="alpha", source_doc_id="doc-a", text="άλφα", greek_badness_score=1.0),
        make_canonical_row(source_dataset="alpha", source_doc_id="doc-b", text="βήτα", greek_badness_score=6.0),
    ]
    rows[0]["len_greek"] = 600
    rows[1]["len_greek"] = 1000
    doc_a_key = pipeline.stable_doc_key("alpha", "doc-a")
    doc_b_key = pipeline.stable_doc_key("alpha", "doc-b")
    write_builder_bundle(
        bundle_root,
        rows=rows,
        family_membership=[
            {
                "doc_key": doc_a_key,
                "source_dataset": "alpha",
                "source_doc_id": "doc-a",
                "family_id": "family-1",
                "family_size": 2,
                "family_source_count": 1,
                "family_mixed_source": False,
                "canonical_kept_doc_key": doc_a_key,
                "canonical_decision": "keep",
                "canonical_decision_stage": "near_duplicate",
                "dedup_run_id": "test-run",
                "selection_version": "selection_v2",
                "representative_score_version": "representative_score_v1_len_greek_badness10",
            },
            {
                "doc_key": doc_b_key,
                "source_dataset": "alpha",
                "source_doc_id": "doc-b",
                "family_id": "family-1",
                "family_size": 2,
                "family_source_count": 1,
                "family_mixed_source": False,
                "canonical_kept_doc_key": doc_a_key,
                "canonical_decision": "drop",
                "canonical_decision_stage": "near_duplicate",
                "dedup_run_id": "test-run",
                "selection_version": "selection_v2",
                "representative_score_version": "representative_score_v1_len_greek_badness10",
            },
        ],
    )

    deduped, summary = pipeline.apply_builder_dedup(
        frame,
        dedup_metadata_root=bundle_root,
        dedup_action="drop_intra",
        dedup_exact_stage="strict_and_relaxed",
        dedup_similarity_threshold=0.85,
        dedup_inter_dataset_policy="quality_first",
        dedup_source_weights_path=None,
    )

    assert summary["bundle_replay_mode"] == "family_membership"
    assert deduped["source_doc_id"].tolist() == ["doc-a"]


def test_build_mix_export_supports_group_and_total_share_source_mix(tmp_path: Path) -> None:
    output_root = tmp_path / "corpus"
    mix_config_path = tmp_path / "source_mix.json"
    rows = [
        make_canonical_row(source_dataset="alpha", source_doc_id="a-1", text="α" * 35),
        make_canonical_row(source_dataset="beta", source_doc_id="b-1", text="β" * 35),
        make_canonical_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="h-1", text="η" * 15),
        make_canonical_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="h-2", text="θ" * 15),
        make_canonical_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="h-3", text="ι" * 15),
    ]
    write_canonical_snapshot(output_root, rows)
    write_source_mix_config(
        mix_config_path,
        [
            {
                "name": "all_non_hplt",
                "exclude_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 1.0,
                "fraction_mode": "of_group",
            },
            {
                "name": "hplt",
                "include_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 0.30,
                "fraction_mode": "of_total",
            },
        ],
    )

    payload = pipeline.build_mix_export(
        output_root=output_root,
        mix_output_path=tmp_path / "mix.parquet",
        source_mix_config_path=mix_config_path,
    )

    frame = pd.read_parquet(tmp_path / "mix.parquet")
    assert set(frame["source_dataset"]) == {"alpha", "beta", "HPLT/ell_Grek_ge8_no_mt_clean60"}
    assert frame["source_mix_component"].value_counts().to_dict() == {"hplt": 2, "all_non_hplt": 2}
    hplt_chars = int(frame.loc[frame["source_dataset"] == "HPLT/ell_Grek_ge8_no_mt_clean60", "text"].str.len().sum())
    non_hplt_chars = int(frame.loc[frame["source_dataset"] != "HPLT/ell_Grek_ge8_no_mt_clean60", "text"].str.len().sum())
    assert hplt_chars == 30
    assert non_hplt_chars == 70
    assert payload["source_mix"]["chars_after"] == 100
    assert [item["name"] for item in payload["source_mix"]["entries"]] == ["all_non_hplt", "hplt"]


def test_shared_selected_input_finalize_matches_direct_streaming_mix(tmp_path: Path) -> None:
    output_root = tmp_path / "corpus"
    mix_config_path = tmp_path / "source_mix.json"
    rows = [
        make_canonical_row(source_dataset="alpha", source_doc_id="a-1", text="α" * 35),
        make_canonical_row(source_dataset="beta", source_doc_id="b-1", text="β" * 35),
        make_canonical_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="h-1", text="η" * 15),
        make_canonical_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="h-2", text="θ" * 15),
        make_canonical_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="h-3", text="ι" * 15),
    ]
    write_canonical_snapshot(output_root, rows)
    write_source_mix_config(
        mix_config_path,
        [
            {
                "name": "all_non_hplt",
                "exclude_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 1.0,
                "fraction_mode": "of_group",
            },
            {
                "name": "hplt",
                "include_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 0.30,
                "fraction_mode": "of_total",
            },
        ],
    )

    direct_payload = pipeline.build_mix_export(
        output_root=output_root,
        mix_output_path=tmp_path / "direct_mix.parquet",
        source_mix_config_path=mix_config_path,
    )
    prelude_payload = pipeline.materialize_streaming_mix_selected_input(
        output_root=output_root,
        destination=tmp_path / "selected_input.parquet",
        include_sources=None,
        exclude_sources=None,
        exclude_needs_ocr_sources=None,
        quality_preset="none",
        historical_mode="include",
        math_mode="include",
        latex_mode="include",
        dedup_metadata_root=None,
        dedup_action="ignore",
        dedup_exact_stage="strict_and_relaxed",
        dedup_similarity_threshold=None,
        dedup_inter_dataset_policy="share_aware",
        dedup_source_weights_path=None,
    )
    shared_payload = pipeline.build_mix_output_from_selected_input(
        selected_input_path=tmp_path / "selected_input.parquet",
        mix_output_path=tmp_path / "shared_mix.parquet",
        source_mix_config_path=mix_config_path,
    )

    direct_frame = pd.read_parquet(tmp_path / "direct_mix.parquet")
    shared_frame = pd.read_parquet(tmp_path / "shared_mix.parquet")
    pd.testing.assert_frame_equal(direct_frame, shared_frame)
    assert prelude_payload["selected_input"]["rows"] == len(rows)
    assert shared_payload["rows_kept"] == direct_payload["rows_kept"]
    assert shared_payload["estimated_tokens"] == direct_payload["estimated_tokens"]
    assert shared_payload["source_mix"]["chars_after"] == direct_payload["source_mix"]["chars_after"]


def test_build_mix_from_selected_input_can_apply_standard_filters_before_source_share(tmp_path: Path) -> None:
    mix_config_path = tmp_path / "source_mix.json"
    selected_input_path = tmp_path / "selected_input.parquet"
    mix_output_path = tmp_path / "mix.parquet"
    write_source_mix_config(
        mix_config_path,
        [
            {
                "name": "all_non_hplt",
                "exclude_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 1.0,
                "fraction_mode": "of_group",
            },
            {
                "name": "hplt",
                "include_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 0.30,
                "fraction_mode": "of_total",
            },
        ],
    )
    rows = []
    for row in [
        make_canonical_row(source_dataset="alpha", source_doc_id="a-valid", text="α" * 70, greek_badness_score=1.0),
        make_canonical_row(source_dataset="beta", source_doc_id="b-missing-mojibake", text="β" * 100, greek_badness_score=1.0),
        make_canonical_row(
            source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
            source_doc_id="h-1",
            text="η" * 15,
            greek_badness_score=1.0,
        ),
        make_canonical_row(
            source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
            source_doc_id="h-2",
            text="θ" * 15,
            greek_badness_score=1.0,
        ),
        make_canonical_row(
            source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
            source_doc_id="h-3",
            text="ι" * 15,
            greek_badness_score=1.0,
        ),
    ]:
        row["doc_key"] = pipeline.stable_doc_key(str(row["source_dataset"]), str(row["source_doc_id"]))
        row["source_mix_chars"] = len(str(row["text"]))
        rows.append(row)
    rows[1]["mojibake_badness_score"] = None
    pq.write_table(pa.Table.from_pylist(rows, schema=pipeline.mix_intermediate_schema()), selected_input_path)

    payload = pipeline.build_mix_output_from_selected_input(
        selected_input_path,
        mix_output_path=mix_output_path,
        source_mix_config_path=mix_config_path,
        apply_standard_split_filters=True,
    )

    frame = pd.read_parquet(mix_output_path)
    assert "b-missing-mojibake" not in set(frame["source_doc_id"])
    assert int((frame["source_dataset"] == "HPLT/ell_Grek_ge8_no_mt_clean60").sum()) == 2
    assert payload["standard_split_filter"]["rows_before"] == 5
    assert payload["standard_split_filter"]["rows_after"] == 4
    assert payload["source_mix"]["entries"][0]["available_chars"] == 70
    assert payload["source_mix"]["entries"][1]["selected_chars"] == 30
    hplt_chars = int(frame.loc[frame["source_dataset"] == "HPLT/ell_Grek_ge8_no_mt_clean60", "text"].str.len().sum())
    assert hplt_chars / int(frame["text"].str.len().sum()) == 0.30


def test_parallel_selected_input_prelude_matches_direct_mix_for_multi_file_input(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "corpus"
    mix_config_path = tmp_path / "source_mix.json"
    part_one = [
        make_canonical_row(source_dataset="alpha", source_doc_id="a-1", text="α" * 35),
        make_canonical_row(source_dataset="beta", source_doc_id="b-1", text="β" * 35),
    ]
    part_two = [
        make_canonical_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="h-1", text="η" * 15),
        make_canonical_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="h-2", text="θ" * 15),
        make_canonical_row(source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60", source_doc_id="h-3", text="ι" * 15),
    ]
    write_canonical_snapshot_parts(output_root, [part_one, part_two])
    write_source_mix_config(
        mix_config_path,
        [
            {
                "name": "all_non_hplt",
                "exclude_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 1.0,
                "fraction_mode": "of_group",
            },
            {
                "name": "hplt",
                "include_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 0.30,
                "fraction_mode": "of_total",
            },
        ],
    )

    direct_payload = pipeline.build_mix_export(
        output_root=output_root,
        mix_output_path=tmp_path / "direct_mix.parquet",
        source_mix_config_path=mix_config_path,
    )
    monkeypatch.setenv("GLOSSAPI_MIX_PREPARE_WORKERS", "2")
    prelude_payload = pipeline.materialize_streaming_mix_selected_input(
        output_root=output_root,
        destination=tmp_path / "selected_input.parquet",
        include_sources=None,
        exclude_sources=None,
        exclude_needs_ocr_sources=None,
        quality_preset="none",
        historical_mode="include",
        math_mode="include",
        latex_mode="include",
        dedup_metadata_root=None,
        dedup_action="ignore",
        dedup_exact_stage="strict_and_relaxed",
        dedup_similarity_threshold=None,
        dedup_inter_dataset_policy="share_aware",
        dedup_source_weights_path=None,
    )
    shared_payload = pipeline.build_mix_output_from_selected_input(
        selected_input_path=tmp_path / "selected_input.parquet",
        mix_output_path=tmp_path / "shared_mix.parquet",
        source_mix_config_path=mix_config_path,
    )

    direct_frame = pd.read_parquet(tmp_path / "direct_mix.parquet")
    shared_frame = pd.read_parquet(tmp_path / "shared_mix.parquet")
    pd.testing.assert_frame_equal(direct_frame, shared_frame)
    assert prelude_payload["filtered_input"]["worker_count"] == 2
    assert prelude_payload["filtered_input"]["source_file_count"] == 2
    assert shared_payload["rows_kept"] == direct_payload["rows_kept"]


def test_build_mix_from_selected_input_allows_repeated_doc_keys_within_one_entry(tmp_path: Path) -> None:
    mix_config_path = tmp_path / "source_mix.json"
    selected_input_path = tmp_path / "selected_input.parquet"
    mix_output_path = tmp_path / "mix.parquet"
    write_source_mix_config(
        mix_config_path,
        [
            {
                "name": "all_non_hplt",
                "exclude_sources": ["HPLT/ell_Grek_ge8_no_mt_clean60"],
                "fraction": 1.0,
                "fraction_mode": "of_group",
            }
        ],
    )
    repeated_doc_key = pipeline.stable_doc_key("alpha", "a-1")
    rows = [
        {
            **make_canonical_row(source_dataset="alpha", source_doc_id="a-1", text="α" * 20),
            "doc_key": repeated_doc_key,
            "source_mix_chars": 20,
        },
        {
            **make_canonical_row(source_dataset="alpha", source_doc_id="a-1", text="β" * 15),
            "doc_key": repeated_doc_key,
            "source_mix_chars": 15,
        },
        {
            **make_canonical_row(
                source_dataset="HPLT/ell_Grek_ge8_no_mt_clean60",
                source_doc_id="h-1",
                text="η" * 10,
            ),
            "doc_key": pipeline.stable_doc_key("HPLT/ell_Grek_ge8_no_mt_clean60", "h-1"),
            "source_mix_chars": 10,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=pipeline.mix_intermediate_schema()), selected_input_path)

    payload = pipeline.build_mix_output_from_selected_input(
        selected_input_path,
        mix_output_path=mix_output_path,
        source_mix_config_path=mix_config_path,
    )

    frame = pd.read_parquet(mix_output_path)
    assert frame["source_dataset"].tolist() == ["alpha", "alpha"]
    assert frame["source_doc_id"].tolist() == ["a-1", "a-1"]
    assert payload["rows_kept"] == 2
    assert payload["source_mix"]["entries"][0]["selected_rows"] == 2


def test_apply_source_mix_config_supports_total_share_only_configs(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"source_dataset": "alpha", "source_doc_id": "a-1", "text": "α" * 40},
            {"source_dataset": "alpha", "source_doc_id": "a-2", "text": "β" * 30},
            {"source_dataset": "beta", "source_doc_id": "b-1", "text": "γ" * 10},
            {"source_dataset": "beta", "source_doc_id": "b-2", "text": "δ" * 20},
        ]
    )
    mix_config_path = tmp_path / "ratio_only.json"
    write_source_mix_config(
        mix_config_path,
        [
            {"name": "alpha", "source_dataset": "alpha", "fraction": 0.7, "fraction_mode": "of_total"},
            {"name": "beta", "source_dataset": "beta", "fraction": 0.3, "fraction_mode": "of_total"},
        ],
    )

    selected, summary = pipeline.apply_source_mix_config(frame, source_mix_config_path=mix_config_path, annotate_component=True)

    assert len(selected) == 4
    assert int(selected.loc[selected["source_dataset"] == "alpha", "text"].str.len().sum()) == 70
    assert int(selected.loc[selected["source_dataset"] == "beta", "text"].str.len().sum()) == 30
    assert summary["chars_after"] == 100


def test_select_rows_for_char_budget_prefers_closer_prefix_over_overshoot() -> None:
    frame = pd.DataFrame(
        [
            {
                "source_dataset": "HPLT/ell_Grek_ge8_no_mt_clean60",
                "source_doc_id": "h-1",
                "_source_mix_doc_key": "001",
                "_source_mix_chars": 42168,
            },
            {
                "source_dataset": "HPLT/ell_Grek_ge8_no_mt_clean60",
                "source_doc_id": "h-2",
                "_source_mix_doc_key": "002",
                "_source_mix_chars": 42169,
            },
        ]
    )

    selected = pipeline.select_rows_for_char_budget(frame, target_chars=39495, chars_column="_source_mix_chars")

    assert selected["source_doc_id"].tolist() == ["h-1"]
    assert int(selected["_source_mix_chars"].sum()) == 42168


def test_load_filtered_mix_can_exclude_needs_ocr_sources(tmp_path: Path) -> None:
    output_root = tmp_path / "corpus"
    rows = [
        make_canonical_row(source_dataset="openarchives.gr", source_doc_id="oa-drop", text="α" * 20, needs_ocr=True),
        make_canonical_row(source_dataset="openarchives.gr", source_doc_id="oa-keep", text="β" * 20, needs_ocr=False),
        make_canonical_row(source_dataset="other", source_doc_id="other-keep", text="γ" * 20, needs_ocr=True),
    ]
    write_canonical_snapshot(output_root, rows)

    frame = pipeline.load_filtered_mix(
        output_root=output_root,
        exclude_needs_ocr_sources=["openarchives.gr"],
    )

    assert sorted(frame["source_doc_id"].tolist()) == ["oa-keep", "other-keep"]


def test_nanochat_export_uses_shared_pool_allocation_when_dedup_present(tmp_path: Path) -> None:
    output_root = tmp_path / "corpus"
    bundle_root = tmp_path / "bundle"
    rows = [
        make_canonical_row(source_dataset="alpha", source_doc_id="a-big", text=" ".join(["άλφα"] * 30)),
        make_canonical_row(source_dataset="beta", source_doc_id="b-big", text=" ".join(["βήτα"] * 30)),
        make_canonical_row(source_dataset="alpha", source_doc_id="a-shared", text=" ".join(["κοινό"] * 5)),
        make_canonical_row(source_dataset="beta", source_doc_id="b-shared", text=" ".join(["κοινό"] * 5)),
    ]
    write_canonical_snapshot(output_root, rows)
    write_builder_bundle(
        bundle_root,
        rows=rows,
        strict_groups={row["source_doc_id"]: (f"strict:{row['source_doc_id']}", 1) for row in rows},
        near_pairs=[
            {
                "doc_key_left": pipeline.stable_doc_key("alpha", "a-shared"),
                "source_dataset_left": "alpha",
                "source_doc_id_left": "a-shared",
                "doc_key_right": pipeline.stable_doc_key("beta", "b-shared"),
                "source_dataset_right": "beta",
                "source_doc_id_right": "b-shared",
                "estimated_jaccard": 0.93,
                "length_ratio": 1.0,
                "likely_containment_flag": False,
                "accepted_reason": "threshold",
                "bucket_match_bands": 3,
                "shingle_mode": "token",
                "shingle_size": 5,
                "num_perm": 128,
                "bands": 32,
                "rows_per_band": 4,
                "candidate_score_floor": 0.85,
            }
        ],
    )
    payload = pipeline.export_nanochat_shards(
        output_root=output_root,
        export_root=tmp_path / "nanochat",
        nanochat_depth=4,
        target_tokens=20,
        dedup_metadata_root=bundle_root,
        dedup_action="annotate",
        dedup_similarity_threshold=0.9,
        dedup_pool_full_include_threshold=0.10,
    )
    shard_tables = [pd.read_parquet(path) for path in sorted((tmp_path / "nanochat").glob("shard_*.parquet"))]
    shard_texts = pd.concat(shard_tables, ignore_index=True)["text"].tolist()
    assert payload["allocation_group"] == "dedup_pool_key"
    assert any("κοινό" in text for text in shard_texts)
