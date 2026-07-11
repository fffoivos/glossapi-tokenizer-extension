from __future__ import annotations

import re
from pathlib import Path


PHASE = Path(__file__).resolve().parents[1]
CLARIDEN = PHASE / "clariden"


def text(name: str) -> str:
    return (CLARIDEN / name).read_text(encoding="utf-8")


def test_live_normalization_consumers_validate_stage10_directly() -> None:
    for name in (
        "42_build_lineage.sbatch",
        "44_build_review_packet.sbatch",
        "60_apply_cleaning.sbatch",
    ):
        source = text(name)
        assert 'phase04_stage_require_upstream "10-normalize"' in source
        assert "--exact-parquet-root" in source


def test_nonadjacent_inputs_have_direct_upstream_receipt_gates() -> None:
    assert 'phase04_stage_require_upstream "20-lineage"' in text(
        "46_aggregate_reviews.sbatch"
    )
    assert 'phase04_stage_require_upstream "50-clean"' in text(
        "64_aggregate_post_clean_reviews.sbatch"
    )
    assert 'phase04_stage_require_upstream "60-greekmmlu-decontam"' in text(
        "90_materialize_validate.sbatch"
    )


def test_cleaning_chains_stop_before_final_clean_and_finalization_is_explicit() -> None:
    submit = text("submit.sh")
    after_admission = re.search(
        r"chain_after_admission\(\) \{(?P<body>.*?)\n\}", submit, re.DOTALL
    )
    after_post_clean = re.search(
        r"chain_after_post_clean\(\) \{(?P<body>.*?)\n\}", submit, re.DOTALL
    )
    assert after_admission is not None and after_post_clean is not None
    assert "submit_one final-clean" not in after_admission.group("body")
    assert "submit_one final-clean" not in after_post_clean.group("body")
    assert "No Stage58 or downstream job was submitted" in after_admission.group("body")
    assert "No job was submitted" in after_post_clean.group("body")
    assert "chain-finalize-noop" in submit
    assert "chain-finalize-promoted" in submit
    assert "CONFIRM_STRUCTURAL_NOOP" in submit
    assert "CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256" in submit


def test_final_clean_request_is_immutable_and_structural_apply_is_stage54_bound() -> (
    None
):
    submit = text("submit.sh")
    finalizer = text("66_finalize_cleaning.sbatch")
    assert "final_clean=$(submit_one final-clean" in submit
    assert "export FINAL_CLEAN_STAGE=58-final-clean" in submit
    assert "export FINAL_CLEAN_STAGE=50-clean" not in submit
    assert 'phase04_stage_require_upstream "54-structural-promote"' in finalizer
    assert "Stage58 only accepts the receipt emitted by Stage54" in finalizer
    assert "Stage58 only accepts the spans emitted by Stage54" in finalizer
    assert "freeze-structural-request" in finalizer
    assert '[[ "${APPLY_STRUCTURAL+x}" == "x" ]]' in finalizer
    assert "CONFIRM_STRUCTURAL_NOOP" in finalizer
    assert "CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256" in finalizer
    assert "phase04_stage_add_input structural_finalization_request" in finalizer
    assert 'phase04_contract_python "${decision_args[@]}"' in finalizer
    assert "validate-cleaning-replay" in finalizer
    assert "--finalizer" in finalizer
    assert "cleaning_replay_validation.json" in finalizer


def test_existing_only_acquisition_uses_authenticated_lock_and_lineage_debug_is_bound() -> (
    None
):
    acquisition = text("00_acquire_sources.sbatch")
    submit = text("submit.sh")
    lineage = text("42_build_lineage.sbatch")
    assert "exact Hugging Face lock resolution requires HF_TOKEN" in acquisition
    assert "resolve_args+=(--anonymous)" not in acquisition
    assert (
        "HF_TOKEN is required to resolve exact Hugging Face content identifiers"
        in submit
    )
    assert "phase04_stage_bind_parameter lineage_debug_exports" in lineage
