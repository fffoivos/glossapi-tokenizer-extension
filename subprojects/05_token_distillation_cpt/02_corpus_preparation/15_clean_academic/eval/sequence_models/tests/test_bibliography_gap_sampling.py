import numpy as np

from sequence_models.bibliography_gap_sampling import (
    fit_weights,
    select_training_rows,
    size_rungs,
)


def _row(index: int, target: int, regime: str, *, fold: int = 0, work: str | None = None) -> dict:
    return {
        "boundary_group_id": f"g{index}",
        "variant_id": f"v{index}",
        "document_id": f"d{index}",
        "work_id": work or f"w{index}",
        "source": "greek_phd",
        "fold": fold,
        "model_line_count": 2,
        "target_connect": target,
        "regime": regime,
        "base_weight": 1.0,
        "gold_block_group_id": f"gold{index // 4}" if target else None,
    }


def test_size_rungs_are_nested_and_end_in_all() -> None:
    assert size_rungs(700) == (250, 500, None)
    assert size_rungs(200) == (None,)


def test_regime_selection_is_cumulative_and_caps_positive_ratio() -> None:
    rows = [
        *(_row(index, 0, "deployment_real") for index in range(3)),
        *(_row(10 + index, 1, "deployment_real") for index in range(12)),
        _row(30, 0, "threshold_ladder"),
    ]
    targets = np.asarray([row["target_connect"] for row in rows], dtype=np.uint8)
    real = select_training_rows(
        rows, targets, regime="deployment_real", negative_group_limit=None, seed=7,
    )
    augmented = select_training_rows(
        rows, targets, regime="threshold_ladder", negative_group_limit=None, seed=7,
    )
    assert np.count_nonzero(targets[real] == 0) == 3
    assert np.count_nonzero(targets[real] == 1) <= 6
    assert np.count_nonzero(targets[augmented] == 0) == 4
    assert set(real).issubset(set(augmented))


def test_correlated_variants_are_selected_as_one_boundary_group() -> None:
    rows = [
        _row(0, 0, "deployment_real"),
        {**_row(1, 0, "header_ablation"), "boundary_group_id": "g0", "base_weight": 0.25},
        _row(2, 1, "deployment_real"),
        _row(3, 1, "deployment_real"),
    ]
    targets = np.asarray([row["target_connect"] for row in rows], dtype=np.uint8)
    selected = select_training_rows(
        rows, targets, regime="header_ablation", negative_group_limit=1, seed=9,
    )
    assert {index for index in selected if rows[index]["boundary_group_id"] == "g0"} == {0, 1}


def test_fit_weights_balance_works_and_cap_synthetic_fraction() -> None:
    rows = [
        _row(0, 0, "deployment_real", work="natural-a"),
        _row(1, 1, "deployment_real", work="natural-b"),
        *(
            _row(10 + index, index % 2, "hard_nonbib", work=f"synthetic-{index}")
            for index in range(8)
        ),
    ]
    targets = np.asarray([row["target_connect"] for row in rows], dtype=np.uint8)
    selected = np.arange(len(rows))
    weights = fit_weights(rows, targets, selected)
    synthetic = np.asarray([row["regime"] != "deployment_real" for row in rows])
    assert np.isclose(weights.sum(), 1.0)
    assert weights[synthetic].sum() <= 0.5 + 1.0e-12
    assert np.all(weights > 0)
