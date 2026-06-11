"""datatrove PIIFormatter wrapper for the production (Clariden) run.

Thin wrapper over `pii_masker.mask` so the per-doc masking logic is
single-sourced and testable without datatrove. Behaviour matches Apertus's
`swiss-ai/pretrain-data` PIIFormatter
(commit 8af990b9401101cf95acd02b066ed0c449789126) for email/IP and extends
IBAN for Greek (see pii_masker.py). Replacement tokens unchanged:
<email-pii> / <ip-pii> / <iban-pii>.

Used by anonymize.py:
    LocalPipelineExecutor([JsonlReader(src), PIIFormatter(), JsonlWriter(out)], tasks=64)
"""
from __future__ import annotations

from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.data import Document
from typing import Iterable

from pii_masker import mask


class PIIFormatter(BaseFormatter):
    name = "PII (Apertus-parity: email/IP/IBAN)"

    def __init__(self, add_pii_count_to_metadata: bool = True):
        super().__init__()
        self.add_pii_count_to_metadata = add_pii_count_to_metadata

    def format(self, text: str) -> str:
        masked, _counts = mask(text)
        return masked

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1) -> Iterable[Document]:
        for doc in data:
            masked, counts = mask(doc.text)
            doc.text = masked
            if self.add_pii_count_to_metadata:
                doc.metadata["pii_count"] = sum(counts.values())
                doc.metadata["pii_by_type"] = counts
            yield doc

    def format_document(self, doc: Document) -> str:
        masked, counts = mask(doc.text)
        if self.add_pii_count_to_metadata:
            doc.metadata["pii_count"] = sum(counts.values())
            doc.metadata["pii_by_type"] = counts
        return masked
