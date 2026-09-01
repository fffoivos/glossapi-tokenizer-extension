from __future__ import annotations

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


packing = load_module("build_packing_plan", ROOT / "dataset" / "build_packing_plan.py")
packed = load_module(
    "finalize_packed_corpus", ROOT / "dataset" / "finalize_packed_corpus.py"
)
bucket = load_module("pack_catalog_bucket", ROOT / "dataset" / "pack_catalog_bucket.py")


class SourceLocalPackingTests(unittest.TestCase):
    def test_bucket_index_writer_is_self_contained_and_megatron_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.idx"
            bucket.write_index(path, [4097, 4097])
            with path.open("rb") as handle:
                self.assertEqual(handle.read(9), b"MMIDIDX\x00\x00")
                self.assertEqual(struct.unpack("<Q", handle.read(8))[0], 1)
                self.assertEqual(struct.unpack("<B", handle.read(1))[0], 4)
                self.assertEqual(struct.unpack("<Q", handle.read(8))[0], 2)
                self.assertEqual(struct.unpack("<Q", handle.read(8))[0], 3)
                self.assertEqual(struct.unpack("<2i", handle.read(8)), (4097, 4097))
                self.assertEqual(struct.unpack("<2q", handle.read(16)), (0, 4097 * 4))
                self.assertEqual(struct.unpack("<3q", handle.read(24)), (0, 1, 2))
                self.assertEqual(handle.read(), b"")

    def test_token_balanced_boundaries_are_complete_and_nonempty(self) -> None:
        tokens = np.asarray([3, 7, 2, 11, 5, 13, 4], dtype=np.uint32)
        boundaries = packing.token_balanced_boundaries(tokens, 4)
        self.assertEqual(boundaries[0], 0)
        self.assertEqual(boundaries[-1], len(tokens))
        self.assertTrue(all(right > left for left, right in zip(boundaries, boundaries[1:])))

    def test_splitmix_sequence_order_is_a_permutation(self) -> None:
        ids = np.arange(10_000, dtype=np.uint64)
        keys = packed.splitmix64(ids ^ packed.SEQUENCE_ORDER_SEED)
        order = np.argsort(keys, kind="stable")
        self.assertEqual(np.unique(keys).size, ids.size)
        self.assertEqual(np.unique(ids[order]).size, ids.size)
        self.assertFalse(np.array_equal(order, ids))


if __name__ == "__main__":
    unittest.main()
