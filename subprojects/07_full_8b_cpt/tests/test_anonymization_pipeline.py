from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANON = ROOT / "subprojects/07_full_8b_cpt/dataset/anonymization"
MASKER = ROOT / "subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/scripts"
for path in (ANON, MASKER):
    sys.path.insert(0, str(path))

from anonymization_common import REPO_ROOT  # noqa: E402
from finalize_postmask_dedup import parse_catalog_line, survivor_key  # noqa: E402
from finalize_sanitized_bridge import nearest_percent, nearest_ratio_total  # noqa: E402
from pii_masker import mask  # noqa: E402


def test_repository_root_resolution() -> None:
    assert REPO_ROOT == ROOT


def test_masking_creates_deterministic_collision_key_without_logging_value() -> None:
    left, left_counts = mask("contact alpha@example.org")
    right, right_counts = mask("contact beta@example.org")
    assert left == right == "contact <email-pii>"
    assert left_counts == right_counts == {"email": 1, "ip": 0, "iban": 0}


def test_postmask_catalog_parser_uses_deterministic_order_fields() -> None:
    digest, task, doc_id = parse_catalog_line(f"{'a' * 64}\t00012\tdocv2:{'b' * 64}\n")
    assert digest == "a" * 64
    assert task == 12
    assert doc_id == "docv2:" + "b" * 64


def test_postmask_survivor_prefers_old_greek_without_changing_other_ordering() -> None:
    tasks = [
        {"pool": "new_greek"},
        {"pool": "foreign_replay"},
        {"pool": "old_greek_replay"},
    ]
    rows = [(0, "z"), (1, "a"), (2, "old")]
    assert min(rows, key=lambda row: survivor_key(tasks, row[0], row[1])) == (2, "old")
    assert min(rows[:2], key=lambda row: survivor_key(tasks, row[0], row[1])) == (0, "z")


def test_stationary_capacity_rounding_closes_to_integer_79_20_1() -> None:
    modern = 63_225_540_570
    active = nearest_ratio_total(modern)
    old_greek = nearest_percent(active)
    foreign = active - modern - old_greek
    assert active == modern + foreign + old_greek
    assert old_greek == 800_323_298
    assert foreign == 16_006_465_967
