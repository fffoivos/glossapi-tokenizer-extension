from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_gap_candidate_screen import _run_configuration


def test_pooled_configuration_produces_complete_grouped_oof_predictions() -> None:
    count = 100
    targets = np.asarray([index % 2 for index in range(count)], dtype=np.uint8)
    folds = np.asarray([(index // 2) % 5 for index in range(count)], dtype=np.uint8)
    metadata = tuple(
        {
            "boundary_group_id": f"g{index}",
            "variant_id": f"v{index}",
            "document_id": f"d{index}",
            "work_id": f"w{index}",
            "source": ("greek_phd", "kallipos", "openarchives")[index % 3],
            "fold": int(folds[index]),
            "model_line_count": 2,
            "target_connect": int(targets[index]),
            "regime": "deployment_real",
            "base_weight": 1.0,
            "genuine_deployment_candidate": True,
        }
        for index in range(count)
    )
    rng = np.random.default_rng(7)
    pooled = rng.normal(size=(count, 8)).astype(np.float32)
    pooled[:, 0] += targets * 1.5
    table = SimpleNamespace(targets=targets, folds=folds, metadata=metadata)
    report, probability, threshold = _run_configuration(
        table=table,
        pooled=pooled,
        genuine_rows=np.arange(count),
        train_rows=np.arange(count),
        arm="pooled_logistic",
        regime="deployment_real",
        size_label="all",
        seed=11,
        maximum_false_connect_rate=0.0,
        bootstrap_replicates=10,
    )
    assert probability.shape == threshold.shape == (count,)
    assert np.isfinite(probability).all()
    assert len(report["folds"]) == 5
    assert report["train_negative_boundary_group_count"] == count // 2
