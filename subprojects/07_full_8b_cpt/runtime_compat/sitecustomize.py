"""Narrow compatibility fixes for the pinned CSCS checkpoint runtime.

The SwissAI Megatron revision used by this campaign still calls
``numpy.product``, which NumPy 2 removed. PyTorch 2.9.1 also rebuilds its
distributed-checkpoint ``Metadata`` dataclass during ``FileSystemWriter.finish``
and otherwise drops dynamic Megatron fields such as ``mcore_data``. Both fixes
only preserve the historical checkpoint API/metadata behavior; they do not
alter tensor values, optimizer math, model precision, or the training data.
"""

import os
import pickle
from pathlib import Path

try:
    import dataclasses
except ImportError:  # pragma: no cover - compatibility with helper interpreters
    dataclasses = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - helper interpreters need not have NumPy
    np = None


if np is not None and not hasattr(np, "product"):
    np.product = np.prod


def _install_dcp_metadata_preservation() -> None:
    if dataclasses is None:
        return
    try:
        from torch.distributed.checkpoint import FileSystemWriter
    except ImportError:
        return

    original = FileSystemWriter.finish
    if getattr(original, "_apertus_preserves_dynamic_metadata", False):
        return

    def finish_preserving_dynamic_metadata(self, metadata, results):
        declared = {field.name for field in dataclasses.fields(metadata)}
        dynamic = {
            key: value for key, value in vars(metadata).items() if key not in declared
        }
        original(self, metadata, results)
        if not dynamic:
            return

        if not getattr(self, "use_collectives", True) and self.rank is not None:
            metadata_path = Path(self._get_metadata_path(self.rank))
        else:
            metadata_path = Path(self._get_metadata_path())
        with metadata_path.open("rb") as stream:
            completed = pickle.load(stream)
        for key, value in dynamic.items():
            setattr(completed, key, value)

        temporary = metadata_path.with_name(metadata_path.name + ".apertus.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(completed, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metadata_path)

    finish_preserving_dynamic_metadata._apertus_preserves_dynamic_metadata = True
    FileSystemWriter.finish = finish_preserving_dynamic_metadata


_install_dcp_metadata_preservation()
