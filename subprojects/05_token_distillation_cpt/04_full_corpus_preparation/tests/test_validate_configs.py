from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATE = load_module("phase04_validate_backlog", HERE / "scripts" / "validate_configs.py")


def tracked_configs():
    sources = VALIDATE.load_json(HERE / "configs" / "sources.json")
    backlog = VALIDATE.load_json(HERE / "configs" / "source_backlog.json")
    return sources, backlog


def tracked_lineage_configs():
    roster = VALIDATE.load_json(HERE / "configs" / "nanochat_initial_roster.json")
    aliases = VALIDATE.load_json(HERE / "configs" / "source_lineage_aliases.json")
    return roster, aliases


def tracked_inventory_config():
    return VALIDATE.load_json(HERE / "configs" / "post_december_inventory.json")


def tracked_cleaning_policy():
    return VALIDATE.load_json(HERE / "configs" / "cleaning_policy.json")


EXPECTED_INITIAL_ROWS = {
    "1000_prwta_xronia_ellhnikhs": 418,
    "AI-team-UoA/greek_legal_code": 43525,
    "Apothetirio_Kallipos": 4588,
    "Apothetirio_Pergamos": 14416,
    "Ekklisiastika_Keimena": 39,
    "Ellinika_Keimena_Project_Gutenberg": 180,
    "HuggingFaceFW/finepdfs-edu": 180586,
    "HuggingFaceFW/finewiki": 239099,
    "Sxolika_vivlia": 114,
    "Wikisource_Greek_texts": 3735,
    "dimodis_logotexnia": 11,
    "ellinika_dedomena_europaikou_koinovouliou": 28652,
    "eurlex-greek-legislation": 20993,
    "greek_phd": 29634,
    "klasikh_arx_ell_grammateia": 753,
    "openarchives.gr": 146038,
    "openbook_gr": 3374,
    "opengov.gr-diaboyleuseis": 1110,
}


def test_cleaning_policy_does_not_invent_human_gold_requirement() -> None:
    policy = tracked_cleaning_policy()
    assert VALIDATE.validate_policy(policy) == []
    approved_nonstructural = copy.deepcopy(policy)
    approved_nonstructural["status"] = "approved"
    assert VALIDATE.validate_policy(approved_nonstructural) == []

    enabled = copy.deepcopy(approved_nonstructural)
    enabled["structural"]["toc"]["enabled_for_materialization"] = True
    enabled["validation"]["structural_application_receipt_required"] = False
    errors = VALIDATE.validate_policy(enabled)
    assert any("requires an application receipt" in error for error in errors)


def test_tracked_nanochat_initial_roster_is_exact_and_valid() -> None:
    roster, _ = tracked_lineage_configs()
    assert VALIDATE.validate_initial_roster(roster) == []
    assert {
        source["source_dataset"]: source["rows"] for source in roster["sources"]
    } == EXPECTED_INITIAL_ROWS
    assert roster["row_counts"]["total_rows"] == sum(EXPECTED_INITIAL_ROWS.values())
    assert {
        addition["source_dataset"] for addition in roster["later_source_name_additions"]
    } == {
        "HPLT/ell_Grek_ge8_no_mt_clean60",
        "OPUS/OpenSubtitles-el-v2018",
    }


def test_initial_roster_rejects_latest_commit_as_the_origin_anchor() -> None:
    roster, _ = tracked_lineage_configs()
    mutated = copy.deepcopy(roster)
    mutated["repository"]["first_data_revision"] = (
        "e1d54136a880ed1df2ed95a5445dabd230453207"
    )
    errors = VALIDATE.validate_initial_roster(mutated)
    assert any("first_data_revision does not match HF history" in error for error in errors)


def test_initial_roster_rejects_row_or_artifact_drift() -> None:
    roster, _ = tracked_lineage_configs()
    mutated = copy.deepcopy(roster)
    mutated["sources"][0]["rows"] += 1
    mutated["sources"][1]["artifacts"] = mutated["sources"][0]["artifacts"]
    errors = VALIDATE.validate_initial_roster(mutated)
    assert any("source rows must sum" in error for error in errors)
    assert any("duplicate artifact path" in error for error in errors)


def test_tracked_lineage_aliases_cover_initial_names_without_equivalence_claims() -> None:
    roster, aliases = tracked_lineage_configs()
    assert VALIDATE.validate_lineage_aliases(aliases, roster) == []
    assert {alias["alias_kind"] for alias in aliases["aliases"]} >= {
        "direct",
        "replacement",
        "hybrid",
    }
    assert all(alias["snapshot_equivalence"] == "unproven" for alias in aliases["aliases"])
    assert all(alias["requires_document_key_audit"] is True for alias in aliases["aliases"])


def test_lineage_aliases_reject_unknown_names_and_append_equivalence() -> None:
    roster, aliases = tracked_lineage_configs()
    mutated = copy.deepcopy(aliases)
    mutated["aliases"][0]["initial_source_datasets"] = ["not_in_nanochat"]
    mutated["aliases"][1]["snapshot_equivalence"] = "equivalent"
    mutated["aliases"][2]["requires_document_key_audit"] = False
    errors = VALIDATE.validate_lineage_aliases(mutated, roster)
    assert any("unknown initial source_dataset" in error for error in errors)
    assert any("snapshot_equivalence must remain unproven" in error for error in errors)
    assert any("requires_document_key_audit must be true" in error for error in errors)


def test_lineage_alias_revisions_match_the_selected_or_backlog_registry() -> None:
    sources, backlog = tracked_configs()
    _, aliases = tracked_lineage_configs()
    assert VALIDATE.validate_alias_registry_revisions(aliases, sources, backlog) == []
    mutated = copy.deepcopy(aliases)
    mutated["aliases"][0]["reviewed_revision"] = "0" * 40
    assert any(
        "reviewed revision drift" in error
        for error in VALIDATE.validate_alias_registry_revisions(mutated, sources, backlog)
    )


def test_tracked_post_december_inventory_is_complete_and_arithmetic_is_valid() -> None:
    sources, backlog = tracked_configs()
    roster, _ = tracked_lineage_configs()
    inventory = tracked_inventory_config()
    assert VALIDATE.validate_post_december_inventory(
        inventory, roster, sources, backlog
    ) == []
    assert inventory["summary"]["post_cutoff_repository_count"] == 25
    assert inventory["summary"]["post_cutoff_new_family_full_text_repository_count"] == 19
    assert inventory["summary"]["new_family_full_text_data_artifact_bytes"] == 16_127_958_471
    assert inventory["summary"]["new_family_card_reported_tokens_arithmetic_sum"] == 4_478_171_892


def test_post_december_inventory_rejects_bad_cutoff_and_token_arithmetic() -> None:
    sources, backlog = tracked_configs()
    roster, _ = tracked_lineage_configs()
    inventory = copy.deepcopy(tracked_inventory_config())
    inventory["cutoff"]["timestamp"] = "2026-03-16T00:00:00Z"
    inventory["summary"]["new_family_card_reported_tokens_arithmetic_sum"] += 1
    errors = VALIDATE.validate_post_december_inventory(
        inventory, roster, sources, backlog
    )
    assert any("cutoff must remain" in error for error in errors)
    assert any("card_reported_tokens_arithmetic_sum" in error for error in errors)


def test_source_registry_requires_explicit_lineage_and_nonadditive_overlap_roles() -> None:
    sources, _ = tracked_configs()
    mutated = copy.deepcopy(sources)
    mutated["sources"][0].pop("source_family_id")
    mutated["sources"][1]["content_relation"] = "same_source_replacement"
    errors = VALIDATE.validate_sources(mutated)
    assert any("source_family_id required" in error for error in errors)
    assert any("cannot use additive_candidate" in error for error in errors)


def test_source_registry_freezes_exact_source_name_provenance() -> None:
    sources, _ = tracked_configs()
    assert VALIDATE.validate_sources(sources) == []
    assert sources["base"]["source_column"] == "source_dataset"
    fields = set(sources["normalized_provenance_contract"]["required_fields"])
    assert {"source_dataset", "source_family_id", "source_revision", "stable_uid"} <= fields

    mutated = copy.deepcopy(sources)
    mutated["base"]["source_column"] = "source_family_id"
    mutated["normalized_provenance_contract"]["required_fields"].remove("source_dataset")
    errors = VALIDATE.validate_sources(mutated)
    assert any("source_column must remain source_dataset" in error for error in errors)
    assert any("must match the frozen contract" in error for error in errors)

    mutated = copy.deepcopy(sources)
    mutated["sources"][0]["required_text_columns"] = ["not_a_candidate"]
    errors = VALIDATE.validate_sources(mutated)
    assert any("required_text_columns must be a subset" in error for error in errors)

    mutated = copy.deepcopy(sources)
    next(
        row
        for row in mutated["sources"]
        if row.get("acquisition_kind") == "mozilla_data_collective"
    )["mdc_expected_sha256"] = None
    errors = VALIDATE.validate_sources(mutated)
    assert any("mdc_expected_sha256 must be pinned" in error for error in errors)


def test_all_mdc_archives_have_exact_registry_sha256_pins() -> None:
    sources, _ = tracked_configs()
    mdc = [
        row
        for row in sources["sources"]
        if row.get("acquisition_kind") == "mozilla_data_collective"
    ]
    assert {row["source_id"]: row["mdc_expected_sha256"] for row in mdc} == {
        "istorima": "21ab85d8f64ec29d1e30c18b63ace260f854a40734270f5f116239fb503304c3",
        "modern_greek_dictionary": "7905b55117e68dccd6250ef71d02feced8c7bc0f1afe353150782af9696298d2",
        "ert_press": "1e47de8c336ce51e1b5ffb162a804d753a7606d9d4b1c644691e3ba20ec414cc",
    }


def test_embedded_structural_routes_have_static_positive_coverage_proof() -> None:
    sources, _ = tracked_configs()
    roster, _ = tracked_lineage_configs()
    assert VALIDATE.validate_embedded_route_roster_coverage(sources, roster) == []
    assert all(
        route["coverage_contract"]["minimum_normalized_rows"] >= 1
        for route in sources["embedded_structural_routes"]
    )

    bad_glob = copy.deepcopy(sources)
    bad_glob["embedded_structural_routes"][0]["acquisition_include_globs"] = [
        "data/not-greek-phd*.parquet"
    ]
    assert any(
        "acquisition globs miss frozen artifacts" in error
        for error in VALIDATE.validate_embedded_route_roster_coverage(
            bad_glob, roster
        )
    )

    bad_regex = copy.deepcopy(sources)
    bad_regex["embedded_structural_routes"][0]["source_regex"] = "^absent$"
    assert any(
        "source_regex must match exactly" in error
        for error in VALIDATE.validate_embedded_route_roster_coverage(
            bad_regex, roster
        )
    )


def test_tracked_source_backlog_is_non_acquiring_and_valid() -> None:
    sources, backlog = tracked_configs()
    assert VALIDATE.validate_backlog(backlog, sources) == []
    assert backlog["entries"]
    assert all(entry["acquisition_eligible"] is False for entry in backlog["entries"])


def test_backlog_rejects_acquisition_enablement() -> None:
    sources, backlog = tracked_configs()
    mutated = copy.deepcopy(backlog)
    mutated["entries"][0]["acquisition_eligible"] = True
    assert any(
        "acquisition_eligible must remain false" in error
        for error in VALIDATE.validate_backlog(mutated, sources)
    )


def test_backlog_rejects_source_registry_collision() -> None:
    sources, backlog = tracked_configs()
    mutated = copy.deepcopy(backlog)
    mutated["entries"][0]["repo_id"] = sources["sources"][0]["repo_id"]
    assert any(
        "repo is already present in sources.json" in error
        for error in VALIDATE.validate_backlog(mutated, sources)
    )


def test_backlog_rejects_unpinned_revision_and_untyped_metrics() -> None:
    sources, backlog = tracked_configs()
    mutated = copy.deepcopy(backlog)
    mutated["entries"][0]["revision"] = "main"
    mutated["entries"][0]["known_metrics"]["repository_rows"] = "1016"
    errors = VALIDATE.validate_backlog(mutated, sources)
    assert any("revision must be a lowercase 40-hex" in error for error in errors)
    assert any("known_metrics.repository_rows" in error for error in errors)


def test_backlog_repository_bytes_cover_every_candidate_artifact() -> None:
    sources, backlog = tracked_configs()
    mutated = copy.deepcopy(backlog)
    mutated["entries"][0]["known_metrics"]["repository_bytes"] -= 1
    errors = VALIDATE.validate_backlog(mutated, sources)
    assert any("must equal candidate file bytes" in error for error in errors)
