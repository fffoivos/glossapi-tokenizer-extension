from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
AREA = ROOT / "subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/hf_v2_release"
sys.path.insert(0, str(AREA / "scripts"))

from release_common import validate_config  # noqa: E402
sys.path.insert(0, str(AREA.parent / "scripts"))
from pii_masker import mask  # noqa: E402


def test_source_taxonomy_is_exact_and_row_closed() -> None:
    config = json.loads((AREA / "configs/release.json").read_text(encoding="utf-8"))
    mapping = validate_config(config)
    assert len(mapping) == 37
    assert sum(config["expected_source_rows"].values()) == 51_839_746
    assert config["expected_source_rows"]["HPLT/ell_Grek_ge8_no_mt_clean60"] == 48_629_460
    assert "glossAPI/libduth" in mapping


def test_policy_forbids_scope_changes() -> None:
    config = json.loads((AREA / "configs/release.json").read_text(encoding="utf-8"))
    assert config["anonymization"]["fields"] == ["text"]
    assert "preserve every row" in config["anonymization"]["row_policy"]
    assert config["deduplication"]["retained_rows"] == config["input"]["rows"]


def test_masker_reaches_fixed_point_for_adjacent_distinct_ibans() -> None:
    text = "GR1601101250000000012300695DE89370400440532013000"
    masked, counts = mask(text)
    masked_again, residual = mask(masked)
    assert masked == "<iban-pii><iban-pii>"
    assert counts == {"email": 0, "ip": 0, "iban": 2}
    assert masked_again == masked
    assert residual == {"email": 0, "ip": 0, "iban": 0}


def test_masker_skips_candidates_invalidated_by_an_earlier_global_replacement() -> None:
    # Replacing the first address also removes its bytes from the leading-zero
    # address, making the latter entry in the regex snapshot stale.
    masked, counts = mask("1.1.1.1 01.1.1.1")
    masked_again, residual = mask(masked)
    assert masked_again == masked
    assert counts["ip"] == 1
    assert residual == {"email": 0, "ip": 0, "iban": 0}


def test_task_invariant_gate_distinguishes_preservation_from_scope_changes() -> None:
    module_path = AREA / "scripts/pipeline.py"
    spec = importlib.util.spec_from_file_location("hf_v2_pipeline_invariants", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = {
        "invariants": {
            "row_count_preserved": True,
            "row_order_preserved": True,
            "schema_and_metadata_preserved": True,
            "all_non_text_values_equal": True,
            "only_text_replaced": True,
            "new_filtering": False,
            "new_deduplication": False,
        }
    }
    assert module._task_invariants_pass(receipt)
    receipt["invariants"]["new_deduplication"] = True
    assert not module._task_invariants_pass(receipt)


def test_hugging_face_draft_pr_is_a_resumable_publication_state() -> None:
    module_path = AREA / "scripts/publish_release.py"
    spec = importlib.util.spec_from_file_location("hf_v2_publisher_state", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._discussion_can_continue("draft")
    assert module._discussion_can_continue("open")
    assert module._discussion_can_continue("merged")
    assert not module._discussion_can_continue("closed")


def test_publisher_uses_resumable_arm64_safe_hub_transport() -> None:
    publisher = (AREA / "scripts/publish_release.py").read_text(encoding="utf-8")
    launcher = (AREA / "clariden/publish_overlay.sbatch").read_text(encoding="utf-8")
    assert "api.upload_large_folder(" in publisher
    assert "api.upload_folder(" not in publisher
    assert 'HF_HUB_DISABLE_XET=1 HF_TOKEN="$HF_TOKEN"' in launcher


def test_publisher_opens_verified_draft_before_merge() -> None:
    publisher = (AREA / "scripts/publish_release.py").read_text(encoding="utf-8")
    open_call = publisher.index("api.change_discussion_status(")
    merge_call = publisher.index("api.merge_pull_request(")
    assert open_call < merge_call
    assert 'new_status="open"' in publisher[open_call:merge_call]


def test_hf_readme_renderer_has_only_requested_sections() -> None:
    module_path = AREA / "scripts/pipeline.py"
    spec = importlib.util.spec_from_file_location("hf_v2_pipeline", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((AREA / "configs/release.json").read_text(encoding="utf-8"))
    category_rows = {row["id"]: 1 for row in config["source_categories"]}
    category_tokens = {row["id"]: 2 for row in config["source_categories"]}
    readme = module._render_readme(
        config,
        category_rows,
        category_tokens,
        {"rows": 9, "training_tokens": 27},
    )
    headings = [line for line in readme.splitlines() if line.startswith("## ")]
    assert headings == [
        "## HPLT filtering method",
        "## GlossAPI datasets and token counts",
        "## Deduplication",
        "## Anonymization to Apertus standards",
    ]
    assert "license" not in "\n".join(headings).lower()
    assert "provenance and rights" not in readme.lower()


def test_slurm_jobs_use_debug_and_transform_is_bounded() -> None:
    for name in ("prepare.sbatch", "transform.sbatch", "transform_batch.sbatch", "finalize.sbatch", "finalize_overlay.sbatch", "publish.sbatch", "publish_overlay.sbatch", "make_public.sbatch"):
        text = (AREA / "clariden" / name).read_text(encoding="utf-8")
        assert "#SBATCH --partition=debug" in text
        assert "#SBATCH --nodes=1" in text
    submit = (AREA / "clariden/submit.sh").read_text(encoding="utf-8")
    assert 'HFV2_STAGE:?set to prepare' in submit
    assert 'transform) script=transform_batch.sbatch' in submit
    assert "HFV2_DRY_RUN" in submit
    assert "HF_TOKEN=" not in submit
    publish = (AREA / "clariden/publish.sbatch").read_text(encoding="utf-8")
    assert '--contract "$HFV2_RUN_ROOT/run_contract.json"' in publish
    assert '--code-root "$HFV2_CODE_ROOT"' in publish


def test_publication_settings_job_is_commit_pinned_and_ungates() -> None:
    script = (AREA / "scripts/ensure_public_release.py").read_text(encoding="utf-8")
    launcher = (AREA / "clariden/make_public.sbatch").read_text(encoding="utf-8")
    assert "gated=False" in script
    assert 'visibility="public"' in script
    assert "anonymous manifest checksum mismatch" in script
    assert "987b8955fcd395c6219e39df9e64715457f69065" in launcher
