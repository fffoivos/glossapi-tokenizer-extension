from __future__ import annotations

import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.mdc_safe_extract import (  # noqa: E402
    SafeExtractionError,
    safe_extract,
    tree_manifest,
)


class MdcSafeExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _archive(self, name: str, members: list[tuple[tarfile.TarInfo, bytes]]) -> Path:
        path = self.root / f"{name}.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for member, payload in members:
                member.size = len(payload) if member.isreg() else 0
                archive.addfile(member, io.BytesIO(payload) if member.isreg() else None)
        return path

    def _run(self, archive: Path, suffix: str) -> dict[str, object]:
        return safe_extract(
            archive,
            self.root / f"extracted-{suffix}",
            self.root / f"manifest-{suffix}.json",
            self.root / f"receipt-{suffix}.json",
        )

    def test_fresh_extract_is_reproducible_and_existing_tree_drift_fails(self) -> None:
        directory = tarfile.TarInfo("dataset/")
        directory.type = tarfile.DIRTYPE
        first = tarfile.TarInfo("dataset/a.txt")
        second = tarfile.TarInfo("dataset/sub/b.bin")
        archive = self._archive(
            "good", [(directory, b""), (first, b"alpha"), (second, b"\x00beta")]
        )
        receipt = self._run(archive, "good")
        self.assertEqual(receipt["status"], "passed_fresh_archive_tree_matches")
        extracted = self.root / "extracted-good"
        self.assertEqual(
            [row["path"] for row in tree_manifest(extracted)["files"]],
            ["dataset/a.txt", "dataset/sub/b.bin"],
        )
        self.assertEqual(self._run(archive, "good"), receipt)
        (extracted / "unexpected.txt").write_text("drift", encoding="utf-8")
        with self.assertRaisesRegex(
            SafeExtractionError, "existing extraction differs"
        ):
            self._run(archive, "good")

    def test_rejects_traversal_duplicates_links_and_special_entries(self) -> None:
        cases: list[tuple[str, list[tuple[tarfile.TarInfo, bytes]], str]] = []
        absolute = tarfile.TarInfo("/absolute.txt")
        cases.append(("absolute", [(absolute, b"x")], "absolute"))
        parent = tarfile.TarInfo("../parent.txt")
        cases.append(("parent", [(parent, b"x")], "contains '..'"))
        duplicate_a = tarfile.TarInfo("same.txt")
        duplicate_b = tarfile.TarInfo("./same.txt")
        cases.append(
            ("duplicate", [(duplicate_a, b"x"), (duplicate_b, b"y")], "duplicate")
        )
        symlink = tarfile.TarInfo("link")
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "target"
        cases.append(("symlink", [(symlink, b"")], "special"))
        fifo = tarfile.TarInfo("fifo")
        fifo.type = tarfile.FIFOTYPE
        cases.append(("fifo", [(fifo, b"")], "special"))
        device = tarfile.TarInfo("device")
        device.type = tarfile.CHRTYPE
        cases.append(("device", [(device, b"")], "special"))

        for name, members, pattern in cases:
            with self.subTest(name=name):
                archive = self._archive(name, members)
                with self.assertRaisesRegex(SafeExtractionError, pattern):
                    self._run(archive, name)

    def test_rejects_manifest_or_receipt_nested_inside_extraction_root(self) -> None:
        member = tarfile.TarInfo("dataset/a.txt")
        archive = self._archive("nested-output", [(member, b"alpha")])
        extracted = self.root / "extracted-nested"
        with self.assertRaisesRegex(SafeExtractionError, "outputs collide"):
            safe_extract(
                archive,
                extracted,
                extracted / "manifest.json",
                self.root / "receipt.json",
            )
        with self.assertRaisesRegex(SafeExtractionError, "outputs collide"):
            safe_extract(
                archive,
                extracted,
                self.root / "manifest.json",
                extracted / "receipt.json",
            )


if __name__ == "__main__":
    unittest.main()
