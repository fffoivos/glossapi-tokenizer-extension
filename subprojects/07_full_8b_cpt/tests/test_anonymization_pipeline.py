from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANON = ROOT / "subprojects/07_full_8b_cpt/dataset/anonymization"
DATASET = ROOT / "subprojects/07_full_8b_cpt/dataset"
MASKER = ROOT / "subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/scripts"
for path in (ANON, DATASET, MASKER):
    sys.path.insert(0, str(path))

from anonymization_common import REPO_ROOT, absolute_receipt  # noqa: E402
from build_sanitized_binary_shard import consume_postmask_drop, load_drops  # noqa: E402
from build_clean_replay_validation import sanitize_replay_text  # noqa: E402
from finalize_postmask_dedup import (  # noqa: E402
    DropWriters,
    parse_catalog_line,
    survivor_key,
)
from finalize_sanitized_bridge import (  # noqa: E402
    accumulate_index_accounting,
    bound_overlay_script,
    nearest_percent,
    nearest_ratio_total,
)
from pii_masker import mask  # noqa: E402


def test_repository_root_resolution() -> None:
    assert REPO_ROOT == ROOT


def test_masking_creates_deterministic_collision_key_without_logging_value() -> None:
    left, left_counts = mask("contact alpha@example.org")
    right, right_counts = mask("contact beta@example.org")
    assert left == right == "contact <email-pii>"
    assert left_counts == right_counts == {"email": 1, "ip": 0, "iban": 0}


def test_replay_validation_reconstructs_the_sanitized_training_text() -> None:
    assert sanitize_replay_text("mail alpha@example.org from 192.0.2.1") == (
        "mail <email-pii> from <ip-pii>"
    )


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


def test_postmask_drop_consumes_exact_content_multiplicity_not_every_doc_id() -> None:
    from collections import Counter

    digest = "a" * 64
    other_digest = "b" * 64
    remaining = Counter({("repeated-doc-id", digest): 2})
    assert consume_postmask_drop(remaining, "repeated-doc-id", digest)
    assert consume_postmask_drop(remaining, "repeated-doc-id", digest)
    assert not consume_postmask_drop(remaining, "repeated-doc-id", digest)
    assert not consume_postmask_drop(remaining, "repeated-doc-id", other_digest)
    assert not remaining


def test_drop_receipt_preserves_row_multiplicity_and_masked_identity(tmp_path: Path) -> None:
    digest = "a" * 64
    other_digest = "b" * 64
    writers = DropWriters(tmp_path / "drops")
    writers.write(7, "repeated-doc-id", digest, "postmask_exact_duplicate")
    writers.write(7, "repeated-doc-id", digest, "postmask_exact_duplicate")
    writers.write(7, "repeated-doc-id", other_digest, "validation_content_collision")
    writers.close()
    path = tmp_path / "drops" / "task_00007.drops.tsv"
    receipt = {
        "task_drop_files": [
            {"task_index": 7, **absolute_receipt(path, rows=3)}
        ]
    }
    counts, _ = load_drops(receipt, 7)
    assert counts[("repeated-doc-id", digest)] == 2
    assert counts[("repeated-doc-id", other_digest)] == 1
    assert sum(counts.values()) == 3


def test_stationary_capacity_rounding_closes_to_integer_79_20_1() -> None:
    modern = 63_225_540_570
    active = nearest_ratio_total(modern)
    old_greek = nearest_percent(active)
    foreign = active - modern - old_greek
    assert active == modern + foreign + old_greek
    assert old_greek == 800_323_298
    assert foreign == 16_006_465_967


def test_bridge_counts_logical_documents_not_terminal_index_sentinels() -> None:
    row = {"documents": 0, "document_index_entries": 0, "tokens": 0}
    accumulate_index_accounting(row, sequences=0, index_entries=1, tokens=0)
    accumulate_index_accounting(row, sequences=17, index_entries=18, tokens=91)
    assert row == {"documents": 17, "document_index_entries": 19, "tokens": 91}


def test_bridge_migration_preserves_original_overlay_code_binding() -> None:
    overlay = {
        "repository": {
            "code_files": [
                {"path": "/immutable/v27/finalize_sanitized_bridge.py"},
                {"path": "/immutable/v27/anonymization_common.py"},
            ]
        }
    }
    assert bound_overlay_script(overlay, "finalize_sanitized_bridge.py") == Path(
        "/immutable/v27/finalize_sanitized_bridge.py"
    )
