from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.feature_crf import LinearChainCRF  # noqa: E402
from sequence_models.features import TAGS, allowed_transition_mask  # noqa: E402


def _legal_paths(length: int, active: np.ndarray) -> list[tuple[int, ...]]:
    transition, start, end = allowed_transition_mask()
    return [
        path
        for path in itertools.product(range(len(TAGS)), repeat=length)
        if active[list(path)].all()
        and start[path[0]]
        and end[path[-1]]
        and all(transition[left, right] for left, right in zip(path, path[1:]))
    ]


def _path_score(model: LinearChainCRF, emissions: np.ndarray, path: tuple[int, ...]) -> float:
    value = model.start[path[0]] + emissions[0, path[0]] + model.end[path[-1]]
    for index in range(1, len(path)):
        value += model.transition[path[index - 1], path[index]]
        value += emissions[index, path[index]]
    return float(value)


def test_numpy_crf_partition_gold_viterbi_and_gradients_match_bruteforce() -> None:
    model = LinearChainCRF(2, seed=11, active_classes=("BIB",))
    model.emission[:] = np.linspace(-0.3, 0.4, model.emission.size).reshape(
        model.emission.shape
    )
    model.emission_bias[:] = np.linspace(-0.1, 0.1, len(TAGS))
    model.transition[:] = np.linspace(-0.2, 0.2, len(TAGS) ** 2).reshape(
        len(TAGS), len(TAGS)
    )
    model.start[:] = np.linspace(-0.05, 0.05, len(TAGS))
    model.end[:] = np.linspace(0.03, -0.03, len(TAGS))
    rows = [{0: 1.0}, {1: 0.5}, {0: -0.25, 1: 1.0}]
    emissions = model.emission_scores(rows)
    paths = _legal_paths(len(rows), model.active_tag_mask)
    scores = np.asarray([_path_score(model, emissions, path) for path in paths])
    maximum = float(scores.max())
    brute_log_z = maximum + math.log(float(np.exp(scores - maximum).sum()))
    log_z, *_ = model._forward_backward(emissions)
    assert log_z == pytest.approx(brute_log_z, abs=1e-10)

    best_path = paths[int(scores.argmax())]
    assert tuple(int(value) for value in model.viterbi(rows)) == best_path
    gold = np.asarray(paths[len(paths) // 2], dtype=np.int64)
    nll, emission_grad, bias_grad, transition_grad, start_grad, end_grad = (
        model.nll_and_grad(rows, gold)
    )
    assert nll == pytest.approx(log_z - _path_score(model, emissions, tuple(gold)), abs=1e-10)
    for value in [*emission_grad.values(), bias_grad, transition_grad, start_grad, end_grad]:
        assert np.isfinite(value).all()
    assert np.count_nonzero(transition_grad[~model.transition_mask]) == 0


def test_numpy_crf_deletion_bias_is_monotone_and_paths_stay_legal() -> None:
    model = LinearChainCRF(1, seed=3, active_classes=("BIB",))
    rows = [{0: 1.0}, {0: 1.0}, {0: 1.0}]
    low = model.viterbi(rows, deletion_bias=0.0)
    high = model.viterbi(rows, deletion_bias=100.0)
    transition, start, end = allowed_transition_mask()
    for path in (low, high):
        assert start[path[0]] and end[path[-1]]
        assert all(transition[left, right] for left, right in zip(path, path[1:]))
    assert sum(TAGS[index] != "O" for index in high) <= sum(
        TAGS[index] != "O" for index in low
    )
    illegal = np.asarray([TAGS.index("I-BIB"), TAGS.index("O"), 0])
    with pytest.raises(ValueError, match="illegal BIOES"):
        model.nll_and_grad(rows, illegal)


def test_numpy_crf_forbidden_states_cannot_win_with_extreme_scores() -> None:
    model = LinearChainCRF(1, seed=3, active_classes=("BIB",))
    i_bib = TAGS.index("I-BIB")
    s_toc = TAGS.index("S-TOC")
    model.emission_bias[i_bib] = 3.0e30
    model.emission_bias[s_toc] = 3.0e30
    model.start[i_bib] = 3.0e30
    decoded = model.viterbi([{0: 0.0}])
    assert TAGS[int(decoded[0])] in {"O", "S-BIB"}


def test_torch_masked_crf_padding_equivalence_bias_and_finite_gradients() -> None:
    torch = pytest.importorskip("torch")
    from sequence_models.char_tcn_crf import MaskedBIOESCRF

    torch.manual_seed(7)
    crf = MaskedBIOESCRF(len(TAGS), ("BIB",))
    short = torch.randn((1, 3, len(TAGS)), requires_grad=True)
    padded = torch.cat((short.detach(), torch.randn((1, 2, len(TAGS)))), dim=1)
    padded.requires_grad_(True)
    short_mask = torch.ones((1, 3), dtype=torch.bool)
    padded_mask = torch.tensor([[True, True, True, False, False]])
    gold_short = torch.tensor([[0, TAGS.index("S-BIB"), 0]])
    gold_padded = torch.tensor([[0, TAGS.index("S-BIB"), 0, 0, 0]])
    assert torch.allclose(
        crf.log_partition(short, short_mask),
        crf.log_partition(padded, padded_mask),
        atol=1e-6,
    )
    assert torch.allclose(
        crf.gold_score(short, gold_short, short_mask),
        crf.gold_score(padded, gold_padded, padded_mask),
        atol=1e-6,
    )
    assert crf.decode(short, short_mask) == crf.decode(padded, padded_mask)
    high_bias = crf.decode(short, short_mask, deletion_bias=100.0)[0]
    assert all(TAGS[index] == "O" for index in high_bias)
    loss = crf.neg_log_likelihood(short, gold_short, short_mask)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert short.grad is not None and bool(torch.isfinite(short.grad).all())
    for parameter in crf.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
    illegal = torch.tensor([[TAGS.index("I-BIB"), 0, 0]])
    with pytest.raises(ValueError, match="illegal BIOES"):
        crf.gold_score(short.detach(), illegal, short_mask)


def test_torch_crf_forbidden_states_cannot_win_with_extreme_scores() -> None:
    torch = pytest.importorskip("torch")
    from sequence_models.char_tcn_crf import MaskedBIOESCRF

    crf = MaskedBIOESCRF(len(TAGS), ("BIB",))
    emissions = torch.zeros((1, 1, len(TAGS)), requires_grad=True)
    with torch.no_grad():
        emissions[0, 0, TAGS.index("I-BIB")] = 30000.0
        emissions[0, 0, TAGS.index("S-TOC")] = 30000.0
        crf.start[TAGS.index("I-BIB")] = 30000.0
    mask = torch.ones((1, 1), dtype=torch.bool)
    decoded = crf.decode(emissions, mask)[0]
    assert [TAGS[index] for index in decoded] in (["O"], ["S-BIB"])
    loss = crf.neg_log_likelihood(emissions, torch.tensor([[0]]), mask)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert emissions.grad is not None and bool(torch.isfinite(emissions.grad).all())
    for parameter in crf.parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all())
