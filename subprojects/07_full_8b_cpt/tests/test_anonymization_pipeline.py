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
from finalize_postmask_dedup import parse_catalog_line  # noqa: E402
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
