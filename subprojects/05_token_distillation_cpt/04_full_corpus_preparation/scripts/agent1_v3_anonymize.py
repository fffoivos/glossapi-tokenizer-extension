#!/usr/bin/env python3
"""Receipt-bound direct-identifier anonymization for Agent 1's v3 lane.

This is intentionally *not* a variant of the legacy mixed cleaning stage.  It
does not strip HTML, repair OCR, normalize whitespace, remove structural text,
or use generic person/address detection.  Its only text transformation is the
frozen high-precision direct-identifier registry:

* email, IPv4, and checksum-valid IBANs from the Apertus-compatible masker;
* syntax-validated IPv6 addresses (also represented by the approved ``ip``
  mask type);
* Greek phone numbers, checksum-valid AFM/AMKA values, and labelled
  identity/passport numbers from :mod:`greek_pii`.

The input is streamed one Parquet batch at a time.  For every input row the
protected ledger records an action and, for emitted representations, reversible
source offsets/value metadata.  The public output never contains that ledger
or raw span values.  ``stable_uid`` remains the upstream document identity;
the stage creates explicit parent/child *representation* identifiers instead.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
LEGACY_PII_DIR = PHASE_ROOT.parent / "02_corpus_preparation" / "40_anonymize" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(LEGACY_PII_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_PII_DIR))

# These two modules are deliberately the source of truth for the actual mask
# semantics.  The tracker below mirrors their deterministic replacement order
# only to make a protected, reversible span ledger; it fails closed if the two
# results ever diverge.
import pii_masker  # type: ignore  # noqa: E402
from pii_masker import mask as mask_apertus_pii  # type: ignore  # noqa: E402

from greek_pii import (  # noqa: E402
    AFM_LABEL,
    AMKA_LABEL,
    ELEVEN_DIGITS,
    IDENTITY,
    NINE_DIGITS,
    PHONE,
    afm_valid,
    amka_valid,
    mask_greek_identifiers,
)


MANIFEST_SCHEMA = "agent1_full_corpus_v3_anonymization_manifest_v1"
LEDGER_SCHEMA = "agent1_full_corpus_v3_protected_anonymization_ledger_v1"
POLICY_VERSION = "agent1_v3_direct_identifier_anonymization_v1"
CHILD_REPRESENTATION_VERSION = "agent1_v3_masked_direct_identifier_representation_v1"
ALLOWED_MASK_TYPES = (
    "email",
    "phone",
    "afm",
    "amka",
    "iban",
    "identity_or_passport",
    "ip",
)
DEFAULT_DIAVGEIA_HEAVY_PII_THRESHOLD = 3

# This is the deliberately narrow structured-policy cue set inherited from the
# reviewed v2 policy.  It is a table/personnel signal, not generic NER.
DIAVGEIA_PERSONNEL = re.compile(
    r"(?i)(?:πατρώνυμο|μητρώνυμο|αριθμός\s+ταυτότητας|Α\.?Δ\.?Τ\.?|"
    r"πίνακας\s+(?:κατάταξης|υποψηφίων|προσληπτέων))"
)
IPV6_CANDIDATE = re.compile(
    r"(?i)(?<![0-9A-F:])(?:[0-9A-F]{1,4}:){2,7}[0-9A-F:]{1,4}(?![0-9A-F:])"
)
# ``greek_pii.IDENTITY`` covers the established Greek identity-card labels and
# the compact English ``passport`` form.  This extension covers the explicit
# Greek passport label without enabling any unlabelled alphanumeric masking.
PASSPORT_OR_IDENTITY_EXTENSION = re.compile(
    r"(?i)(?:διαβατ[ηή]ρ[ίι](?:ου|ο)|passport(?:\s+(?:number|no\.?))?|"
    r"identity\s*(?:card|number|no\.?)?)\s*[:#№-]?\s*"
    r"([A-ZΑ-Ω]{1,3}[\s.-]?\d{5,10}|[A-Z0-9Α-Ω]{5,12})\b"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_parts(namespace: str, *parts: object) -> str:
    """Hash unambiguously delimited representation identity components."""

    digest = hashlib.sha256(namespace.encode("utf-8"))
    for part in parts:
        encoded = canonical_json(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _true_value(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return isinstance(value, str) and value.strip().casefold() in {"true", "1"}


def recursive_private_data(value: object) -> bool:
    """Find an explicit ``privateData=true`` flag in source metadata only."""

    if isinstance(value, str):
        try:
            return recursive_private_data(json.loads(value))
        except (TypeError, json.JSONDecodeError):
            return False
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).replace("_", "").casefold()
            if normalized == "privatedata" and _true_value(item):
                return True
            if recursive_private_data(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(recursive_private_data(item) for item in value)
    return False


def is_diavgeia_row(row: Mapping[str, Any]) -> bool:
    return any(
        str(row.get(field) or "").strip().casefold() == "diavgeia"
        for field in ("acquisition_source_id", "source_id", "source_dataset", "source_family_id")
    )


def diavgeia_private_data_true(row: Mapping[str, Any]) -> bool:
    """Check direct and compacted upstream metadata forms of privateData."""

    if not is_diavgeia_row(row):
        return False
    for key, value in row.items():
        normalized = str(key).replace("_", "").casefold()
        if normalized == "privatedata" and _true_value(value):
            return True
        if normalized in {"sourcemetadatajson", "metadatajson", "metadata"} and recursive_private_data(value):
            return True
    return False


@dataclass(frozen=True)
class MaskSpan:
    """A source-text span kept only in the protected ledger."""

    pii_type: str
    char_start: int
    char_end: int
    raw_value: str
    replacement: str

    def ledger_value(self) -> dict[str, object]:
        return {
            "char_end": self.char_end,
            "char_start": self.char_start,
            "pii_type": self.pii_type,
            "raw_value": self.raw_value,
            "raw_value_sha256": sha256_text(self.raw_value),
            "replacement": self.replacement,
        }


class _TrackedText:
    """Small source-offset tracker for a fixed, non-overlapping mask sequence."""

    def __init__(self, original: str) -> None:
        self.original = original
        self.text = original
        self._source_offsets: list[int | None] = list(range(len(original)))
        self.spans: list[MaskSpan] = []

    def replace(self, start: int, end: int, replacement: str, pii_type: str) -> None:
        if start < 0 or end <= start or end > len(self.text):
            raise ValueError(f"invalid {pii_type} span {start}:{end}")
        mapped = [value for value in self._source_offsets[start:end] if value is not None]
        # Identifier expressions must never consume a prior mask token.  Treat
        # a deviation as implementation drift rather than write an ambiguous
        # reversible ledger.
        if not mapped:
            raise RuntimeError(f"{pii_type} match overlaps an already masked value")
        first, last = min(mapped), max(mapped)
        if mapped != list(range(first, last + 1)):
            raise RuntimeError(f"{pii_type} source span is not contiguous")
        raw_value = self.original[first : last + 1]
        self.spans.append(
            MaskSpan(
                pii_type=pii_type,
                char_start=first,
                char_end=last + 1,
                raw_value=raw_value,
                replacement=replacement,
            )
        )
        self.text = self.text[:start] + replacement + self.text[end:]
        self._source_offsets = (
            self._source_offsets[:start]
            + [None] * len(replacement)
            + self._source_offsets[end:]
        )

    def replace_literal_all(self, value: str, replacement: str, pii_type: str) -> None:
        """Mirror ``str.replace`` while retaining every original occurrence."""

        if not value:
            return
        positions: list[tuple[int, int]] = []
        cursor = 0
        while True:
            start = self.text.find(value, cursor)
            if start < 0:
                break
            positions.append((start, start + len(value)))
            cursor = start + len(value)
        for start, end in reversed(positions):
            self.replace(start, end, replacement, pii_type)


def _replace_generic_masks(tracker: _TrackedText) -> None:
    """Mirror the Apertus-compatible email/IP/IBAN implementation exactly."""

    seen: set[str] = set()
    for pii_type, pattern, replacement in (
        ("email", pii_masker.EMAIL_RE, "<email-pii>"),
        ("ip", pii_masker.IP_RE, "<ip-pii>"),
    ):
        values = [match.group(0) for match in pattern.finditer(tracker.text)]
        for value in values:
            if value and value not in seen:
                seen.add(value)
                tracker.replace_literal_all(value, replacement, pii_type)
    candidates = [match.group(0) for match in pii_masker.IBAN_CAND_RE.finditer(tracker.text)]
    for candidate in candidates:
        value = pii_masker._iban_from_candidate(candidate)
        if value and value not in seen:
            seen.add(value)
            tracker.replace_literal_all(value, "<iban-pii>", "iban")


def _validated_values(
    text: str,
    labelled: re.Pattern[str],
    bare: re.Pattern[str],
    validator: Any,
) -> set[str]:
    """The equivalent of greek_pii's deliberately private helper."""

    values = {match.group(1) for match in labelled.finditer(text) if validator(match.group(1))}
    values.update(match.group(1) for match in bare.finditer(text) if validator(match.group(1)))
    return values


def _replace_greek_masks(tracker: _TrackedText) -> None:
    for pii_type, labelled, bare, validator, replacement in (
        ("afm", AFM_LABEL, NINE_DIGITS, afm_valid, "<afm-pii>"),
        ("amka", AMKA_LABEL, ELEVEN_DIGITS, amka_valid, "<amka-pii>"),
    ):
        for value in sorted(
            _validated_values(tracker.text, labelled, bare, validator), key=len, reverse=True
        ):
            tracker.replace_literal_all(value, replacement, pii_type)
    for match in list(IDENTITY.finditer(tracker.text))[::-1]:
        start, end = match.span(1)
        tracker.replace(start, end, "<identity-pii>", "identity_or_passport")
    for match in list(PHONE.finditer(tracker.text))[::-1]:
        start, end = match.span(0)
        tracker.replace(start, end, "<phone-pii>", "phone")


def _replace_ipv6_masks(tracker: _TrackedText) -> None:
    """Mask only syntactically valid IPv6 candidates with the approved IP token."""

    matches = []
    for match in IPV6_CANDIDATE.finditer(tracker.text):
        try:
            valid = isinstance(ipaddress.ip_address(match.group(0)), ipaddress.IPv6Address)
        except ValueError:
            valid = False
        if valid:
            matches.append(match)
    for match in reversed(matches):
        tracker.replace(match.start(), match.end(), "<ip-pii>", "ip")


def _replace_identity_passport_extension(tracker: _TrackedText) -> None:
    for match in list(PASSPORT_OR_IDENTITY_EXTENSION.finditer(tracker.text))[::-1]:
        start, end = match.span(1)
        tracker.replace(start, end, "<identity-pii>", "identity_or_passport")


def mask_direct_identifiers(text: str) -> tuple[str, list[MaskSpan], dict[str, int]]:
    """Mask exactly the approved registry and produce a protected span record.

    The call to the existing maskers is an executable compatibility assertion:
    a future regex change cannot silently cause public text and audit spans to
    describe different transformations.
    """

    tracker = _TrackedText(text)
    _replace_generic_masks(tracker)
    _replace_greek_masks(tracker)
    expected, _ = mask_apertus_pii(text)
    expected, _ = mask_greek_identifiers(expected)
    if tracker.text != expected:
        raise RuntimeError(
            "protected span tracker diverged from approved pii_masker/greek_pii output"
        )
    # ``pii_masker`` is the existing source of truth for the legacy IPv4
    # policy. IPv6 is the same approved direct-ID category, but needs syntax
    # validation because a loose colon regex would otherwise over-mask prose.
    _replace_ipv6_masks(tracker)
    _replace_identity_passport_extension(tracker)
    spans = sorted(tracker.spans, key=lambda value: (value.char_start, value.char_end, value.pii_type))
    counts = Counter(span.pii_type for span in spans)
    return tracker.text, spans, dict(sorted(counts.items()))


def parent_representation_id(row: Mapping[str, Any], parent_text_sha256: str) -> str:
    for field in ("representation_id", "input_representation_id"):
        existing = row.get(field)
        if isinstance(existing, str) and existing:
            return existing
    return sha256_parts(
        "agent1_v3_parent_representation_id_v1",
        str(row["stable_uid"]),
        str(row.get("representation_generation") or "canonical"),
        parent_text_sha256,
    )


def child_representation_id(parent_id: str, output_text_sha256: str) -> str:
    return sha256_parts(
        CHILD_REPRESENTATION_VERSION,
        parent_id,
        output_text_sha256,
        POLICY_VERSION,
    )


def _source_label(row: Mapping[str, Any]) -> str:
    for field in ("acquisition_source_id", "source_id", "source_dataset", "source_family_id"):
        value = row.get(field)
        if value is not None and str(value):
            return str(value)
    return "unknown_source"


def _source_doc_id(row: Mapping[str, Any]) -> str:
    value = row.get("source_doc_id")
    return "" if value is None else str(value)


def _validate_input_row(row: Mapping[str, Any], *, input_path: Path) -> tuple[str, str]:
    stable_uid = row.get("stable_uid")
    if not isinstance(stable_uid, str) or not stable_uid:
        raise ValueError(f"{input_path}: every row needs a non-empty stable_uid")
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{input_path}: stable_uid={stable_uid}: text must be a string")
    actual_hash = sha256_text(text)
    for field in ("text_sha256", "cleaned_text_sha256"):
        claimed = row.get(field)
        if claimed is not None and str(claimed) != actual_hash:
            raise ValueError(
                f"{input_path}: stable_uid={stable_uid}: {field} differs from input text"
            )
    return stable_uid, text


def output_schema(input_schema: Any) -> Any:
    import pyarrow as pa

    required_v3_lineage = (
        ("representation_id", pa.string()),
        ("parent_representation_id", pa.string()),
        ("parent_text_sha256", pa.string()),
        ("text_sha256", pa.string()),
        ("cleaned_text_sha256", pa.string()),
    )
    additions = (
        ("anonymization_parent_text_sha256", pa.string()),
        ("anonymization_output_text_sha256", pa.string()),
        ("anonymization_parent_representation_id", pa.string()),
        ("anonymization_child_representation_id", pa.string()),
        ("anonymization_policy_version", pa.string()),
        ("anonymization_action", pa.string()),
        ("anonymization_reasons_json", pa.string()),
        ("anonymization_pii_by_type_json", pa.string()),
        ("anonymization_changed", pa.bool_()),
    )
    existing = set(input_schema.names)
    collisions = existing.intersection(name for name, _ in additions)
    if collisions:
        raise ValueError(
            "input already contains Agent 1 v3 anonymization output columns; "
            f"refusing to layer the immutable stage: {sorted(collisions)}"
        )
    fields = list(input_schema)
    for name, value in required_v3_lineage:
        if name not in existing:
            fields.append(pa.field(name, value))
        elif not (
            pa.types.is_string(input_schema.field(name).type)
            or pa.types.is_large_string(input_schema.field(name).type)
        ):
            raise ValueError(f"v3 lineage field {name!r} must be a string when present")
    fields.extend(pa.field(name, value) for name, value in additions)
    return pa.schema(fields, metadata=input_schema.metadata)


def dropped_schema() -> Any:
    import pyarrow as pa

    # Deliberately no source text or raw metadata: privateData=true rows are
    # removed, and their audit evidence lives only in the protected ledger.
    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("anonymization_parent_text_sha256", pa.string()),
            ("anonymization_parent_representation_id", pa.string()),
            ("anonymization_action", pa.string()),
            ("anonymization_reasons_json", pa.string()),
        ]
    )


def protected_ledger_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("input_text_sha256", pa.string()),
            ("output_text_sha256", pa.string()),
            ("parent_representation_id", pa.string()),
            ("child_representation_id", pa.string()),
            ("action", pa.string()),
            ("reasons_json", pa.string()),
            ("pii_by_type_json", pa.string()),
            ("span_count", pa.int64()),
            # This JSON deliberately contains raw span values.  The separate
            # protected-ledger root is chmod 0700/0600 and must never be part
            # of a training/public artifact inventory.
            ("protected_spans_json", pa.string()),
            ("ledger_schema_version", pa.string()),
        ]
    )


def discover_parquet(input_path: Path) -> list[tuple[Path, Path]]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix != ".parquet":
            raise ValueError(f"--input file must be Parquet: {input_path}")
        return [(input_path, Path(input_path.name))]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    files = sorted(path for path in input_path.rglob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no Parquet input shards beneath {input_path}")
    return [(path, path.relative_to(input_path)) for path in files]


def _safe_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _ensure_distinct_paths(
    *, input_path: Path, output: Path, dropped: Path, quarantine: Path, protected_ledger: Path
) -> None:
    destinations = {
        "output": output.resolve(),
        "dropped": dropped.resolve(),
        "quarantine": quarantine.resolve(),
        "protected ledger": protected_ledger.resolve(),
    }
    values = list(destinations.items())
    for index, (name, path) in enumerate(values):
        for other_name, other_path in values[index + 1 :]:
            if path == other_path or _safe_relative_to(path, other_path) or _safe_relative_to(other_path, path):
                raise ValueError(f"{name} and {other_name} paths must be disjoint")
    resolved_input = input_path.resolve()
    if resolved_input.is_dir():
        for name, path in destinations.items():
            if _safe_relative_to(path, resolved_input) or _safe_relative_to(resolved_input, path):
                raise ValueError(f"{name} must not overlap the immutable input tree")
    elif any(path == resolved_input for path in destinations.values()):
        raise ValueError("anonymization output must not replace its input file")


def _prepare_root(path: Path, *, protected: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output root is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty immutable output root: {path}")
    if protected:
        os.chmod(path, 0o700)


def _atomic_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    temporary_path.unlink(missing_ok=True)
    return temporary_path


def parquet_receipt(path: Path, *, relative_to: Path) -> dict[str, object]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    return {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(metadata.num_rows),
        "row_groups": int(metadata.num_row_groups),
    }


def input_receipt(path: Path, *, relative_to: Path | None) -> dict[str, object]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    display = path.resolve().as_posix() if relative_to is None else path.resolve().relative_to(relative_to.resolve()).as_posix()
    return {
        "path": display,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(metadata.num_rows),
        "row_groups": int(metadata.num_row_groups),
    }


def _atomic_json_no_replace(path: Path, value: Mapping[str, object], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Do not silently remove a partially published immutable manifest; its
        # presence is evidence that an operator must inspect the failed run.
        raise


def _policy_receipt(path: Path | None, heavy_threshold: int) -> dict[str, object]:
    default = {
        "policy_version": POLICY_VERSION,
        "mask_types": list(ALLOWED_MASK_TYPES),
        "drop_private_data_true": True,
        "generic_person_name_ner": False,
        "street_address_policy": "precision_gated_not_enabled",
        "html_or_ocr_cleaning": "not_enabled",
        "diavgeia_personnel_table": "quarantine_when_pii_heavy",
        "diavgeia_pii_heavy_threshold": heavy_threshold,
    }
    if path is None:
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: policy root must be an object")
    section = payload.get("anonymization")
    if not isinstance(section, Mapping):
        raise ValueError(f"{path}: policy lacks anonymization object")
    if section.get("drop_private_data_true") is not True:
        raise ValueError(f"{path}: v3 anonymization requires drop_private_data_true=true")
    if section.get("generic_person_name_ner") is not False:
        raise ValueError(f"{path}: generic person-name NER must remain disabled")
    if section.get("street_address_policy") != "precision_gated_not_enabled":
        raise ValueError(f"{path}: street address policy is not enabled in this stage")
    if set(section.get("mask_types", [])) != set(ALLOWED_MASK_TYPES):
        raise ValueError(f"{path}: mask_types do not match the frozen approved direct-ID registry")
    if section.get("diavgeia_personnel_table") != "quarantine_when_pii_heavy":
        raise ValueError(f"{path}: unsupported Diavgeia personnel-table policy")
    return {
        **default,
        "source_policy": {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
    }


def _derivative_row(
    row: Mapping[str, Any],
    *,
    text: str,
    parent_hash: str,
    output_hash: str,
    parent_id: str,
    child_id: str,
    action: str,
    reasons: Sequence[str],
    pii_counts: Mapping[str, int],
) -> dict[str, object]:
    derivative = {
        **dict(row),
        "text": text,
        # Advance the generic v3 lineage contract as well as retaining the
        # explicit anonymization audit fields.  ``stable_uid`` stays put: it
        # identifies the original source document, not this text variant.
        "parent_representation_id": parent_id,
        "representation_id": child_id,
        "parent_text_sha256": parent_hash,
        "text_sha256": output_hash,
        "cleaned_text_sha256": output_hash,
        "anonymization_parent_text_sha256": parent_hash,
        "anonymization_output_text_sha256": output_hash,
        "anonymization_parent_representation_id": parent_id,
        "anonymization_child_representation_id": child_id,
        "anonymization_policy_version": POLICY_VERSION,
        "anonymization_action": action,
        "anonymization_reasons_json": canonical_json(sorted(set(reasons))),
        "anonymization_pii_by_type_json": canonical_json(dict(sorted(pii_counts.items()))),
        "anonymization_changed": parent_hash != output_hash,
    }
    return derivative


def _ledger_row(
    row: Mapping[str, Any],
    *,
    parent_hash: str,
    output_hash: str | None,
    parent_id: str,
    child_id: str | None,
    action: str,
    reasons: Sequence[str],
    spans: Sequence[MaskSpan],
    pii_counts: Mapping[str, int],
) -> dict[str, object]:
    return {
        "stable_uid": str(row["stable_uid"]),
        "acquisition_source_id": str(row.get("acquisition_source_id") or ""),
        "source_dataset": str(row.get("source_dataset") or ""),
        "source_doc_id": _source_doc_id(row),
        "input_text_sha256": parent_hash,
        "output_text_sha256": output_hash,
        "parent_representation_id": parent_id,
        "child_representation_id": child_id,
        "action": action,
        "reasons_json": canonical_json(sorted(set(reasons))),
        "pii_by_type_json": canonical_json(dict(sorted(pii_counts.items()))),
        "span_count": len(spans),
        "protected_spans_json": canonical_json([span.ledger_value() for span in spans]),
        "ledger_schema_version": LEDGER_SCHEMA,
    }


def _dropped_row(
    row: Mapping[str, Any], *, parent_hash: str, parent_id: str, reasons: Sequence[str]
) -> dict[str, object]:
    return {
        "stable_uid": str(row["stable_uid"]),
        "acquisition_source_id": str(row.get("acquisition_source_id") or ""),
        "source_dataset": str(row.get("source_dataset") or ""),
        "source_doc_id": _source_doc_id(row),
        "anonymization_parent_text_sha256": parent_hash,
        "anonymization_parent_representation_id": parent_id,
        "anonymization_action": "drop",
        "anonymization_reasons_json": canonical_json(sorted(set(reasons))),
    }


def _process_file(
    *,
    input_path: Path,
    relative: Path,
    output_root: Path,
    dropped_root: Path,
    quarantine_root: Path,
    ledger_root: Path,
    batch_rows: int,
    diavgeia_heavy_pii_threshold: int,
) -> dict[str, object]:
    """Stream one input shard into four content-bound, atomically published shards."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(input_path)
    missing = {"stable_uid", "text"} - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{input_path}: missing required columns: {sorted(missing)}")
    derived_schema = output_schema(parquet.schema_arrow)
    paths = {
        "output": output_root / relative,
        "dropped": dropped_root / relative,
        "quarantine": quarantine_root / relative,
        "ledger": ledger_root / relative,
    }
    temporary = {name: _atomic_output_path(path) for name, path in paths.items()}
    writers = {
        "output": pq.ParquetWriter(temporary["output"], derived_schema, compression="zstd"),
        "dropped": pq.ParquetWriter(temporary["dropped"], dropped_schema(), compression="zstd"),
        "quarantine": pq.ParquetWriter(temporary["quarantine"], derived_schema, compression="zstd"),
        "ledger": pq.ParquetWriter(temporary["ledger"], protected_ledger_schema(), compression="zstd"),
    }
    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        for batch in parquet.iter_batches(batch_size=batch_rows, use_threads=False):
            output_rows: list[dict[str, object]] = []
            dropped_rows: list[dict[str, object]] = []
            quarantine_rows: list[dict[str, object]] = []
            ledger_rows: list[dict[str, object]] = []
            for row in batch.to_pylist():
                stable_uid, text = _validate_input_row(row, input_path=input_path)
                parent_hash = sha256_text(text)
                parent_id = parent_representation_id(row, parent_hash)
                source = _source_label(row)
                action = "keep"
                reasons: list[str] = []
                spans: list[MaskSpan] = []
                pii_counts: dict[str, int] = {}
                output_text: str | None = None
                child_id: str | None = None
                output_hash: str | None = None

                if diavgeia_private_data_true(row):
                    action = "drop"
                    reasons.append("diavgeia_privateData_true")
                else:
                    output_text, spans, pii_counts = mask_direct_identifiers(text)
                    output_hash = sha256_text(output_text)
                    child_id = child_representation_id(parent_id, output_hash)
                    if (
                        is_diavgeia_row(row)
                        and DIAVGEIA_PERSONNEL.search(text)
                        and sum(pii_counts.values()) >= diavgeia_heavy_pii_threshold
                    ):
                        action = "quarantine"
                        reasons.append("diavgeia_pii_heavy_personnel_table")
                    if pii_counts:
                        reasons.append("approved_high_precision_direct_identifier_masking")

                if action == "drop":
                    dropped_rows.append(
                        _dropped_row(
                            row,
                            parent_hash=parent_hash,
                            parent_id=parent_id,
                            reasons=reasons,
                        )
                    )
                else:
                    assert output_text is not None and output_hash is not None and child_id is not None
                    derivative = _derivative_row(
                        row,
                        text=output_text,
                        parent_hash=parent_hash,
                        output_hash=output_hash,
                        parent_id=parent_id,
                        child_id=child_id,
                        action=action,
                        reasons=reasons,
                        pii_counts=pii_counts,
                    )
                    if action == "keep":
                        output_rows.append(derivative)
                    else:
                        quarantine_rows.append(derivative)
                ledger_rows.append(
                    _ledger_row(
                        row,
                        parent_hash=parent_hash,
                        output_hash=output_hash,
                        parent_id=parent_id,
                        child_id=child_id,
                        action=action,
                        reasons=reasons,
                        spans=spans,
                        pii_counts=pii_counts,
                    )
                )
                counts["input_rows"] += 1
                counts[f"action:{action}"] += 1
                counts["protected_ledger_rows"] += 1
                counts["masked_spans"] += len(spans)
                counts["characters_input"] += len(text)
                counts["characters_output"] += len(output_text or "")
                for pii_type, amount in pii_counts.items():
                    counts[f"pii:{pii_type}"] += int(amount)
                    source_counts[source][f"pii:{pii_type}"] += int(amount)
                for key in (
                    "input_rows",
                    f"action:{action}",
                    "protected_ledger_rows",
                    "masked_spans",
                ):
                    source_counts[source][key] += 1 if key != "masked_spans" else len(spans)
                source_counts[source]["characters_input"] += len(text)
                source_counts[source]["characters_output"] += len(output_text or "")
            if output_rows:
                writers["output"].write_table(pa.Table.from_pylist(output_rows, schema=derived_schema))
            if dropped_rows:
                writers["dropped"].write_table(pa.Table.from_pylist(dropped_rows, schema=dropped_schema()))
            if quarantine_rows:
                writers["quarantine"].write_table(pa.Table.from_pylist(quarantine_rows, schema=derived_schema))
            if ledger_rows:
                writers["ledger"].write_table(pa.Table.from_pylist(ledger_rows, schema=protected_ledger_schema()))
        for writer in writers.values():
            writer.close()
        for name, path in paths.items():
            os.replace(temporary[name], path)
            if name == "ledger":
                os.chmod(path, 0o600)
    except BaseException:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    if counts["input_rows"] != (
        counts["action:keep"] + counts["action:drop"] + counts["action:quarantine"]
    ):
        raise RuntimeError(f"{input_path}: anonymization action accounting does not close")
    if counts["input_rows"] != counts["protected_ledger_rows"]:
        raise RuntimeError(f"{input_path}: protected ledger does not cover every input row")
    return {
        "relative_path": relative.as_posix(),
        "input": input_receipt(input_path, relative_to=None),
        "output": parquet_receipt(paths["output"], relative_to=output_root),
        "dropped": parquet_receipt(paths["dropped"], relative_to=dropped_root),
        "quarantine": parquet_receipt(paths["quarantine"], relative_to=quarantine_root),
        "protected_ledger": parquet_receipt(paths["ledger"], relative_to=ledger_root),
        "counts": dict(sorted(counts.items())),
        "per_source": {key: dict(sorted(value.items())) for key, value in sorted(source_counts.items())},
    }


def _aggregate(receipts: Iterable[Mapping[str, object]]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    total: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    for receipt in receipts:
        total.update({str(key): int(value) for key, value in dict(receipt["counts"]).items()})
        for source, values in dict(receipt["per_source"]).items():
            per_source[str(source)].update({str(key): int(value) for key, value in dict(values).items()})
    return total, per_source


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Decontaminated Parquet tree or one shard")
    parser.add_argument("--output", type=Path, required=True, help="Masked training-candidate Parquet tree")
    parser.add_argument("--dropped", type=Path, required=True, help="Text-free privateData=true tombstone tree")
    parser.add_argument("--quarantine", type=Path, required=True, help="Masked Diavgeia personnel-table quarantine tree")
    parser.add_argument(
        "--protected-ledger",
        "--ledger",
        dest="protected_ledger",
        type=Path,
        required=True,
        help="0700/0600 private audit ledger; never a public corpus input",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Immutable, non-public anonymization manifest")
    parser.add_argument("--policy", type=Path, help="Optional frozen Agent 1 v3 policy JSON")
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument(
        "--diavgeia-pii-heavy-threshold",
        type=int,
        default=DEFAULT_DIAVGEIA_HEAVY_PII_THRESHOLD,
        help="Minimum approved direct-ID spans for personnel-table quarantine",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_rows < 1:
        raise ValueError("--batch-rows must be >= 1")
    if args.diavgeia_pii_heavy_threshold < 1:
        raise ValueError("--diavgeia-pii-heavy-threshold must be >= 1")
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.manifest}")
    if args.policy is not None and not args.policy.is_file():
        raise FileNotFoundError(args.policy)

    input_path = args.input.resolve()
    output_root = args.output.resolve()
    dropped_root = args.dropped.resolve()
    quarantine_root = args.quarantine.resolve()
    ledger_root = args.protected_ledger.resolve()
    _ensure_distinct_paths(
        input_path=input_path,
        output=output_root,
        dropped=dropped_root,
        quarantine=quarantine_root,
        protected_ledger=ledger_root,
    )
    policy = _policy_receipt(args.policy.resolve() if args.policy else None, args.diavgeia_pii_heavy_threshold)
    files = discover_parquet(input_path)
    for root, protected in (
        (output_root, False),
        (dropped_root, True),
        (quarantine_root, True),
        (ledger_root, True),
    ):
        _prepare_root(root, protected=protected)

    receipts = [
        _process_file(
            input_path=path,
            relative=relative,
            output_root=output_root,
            dropped_root=dropped_root,
            quarantine_root=quarantine_root,
            ledger_root=ledger_root,
            batch_rows=args.batch_rows,
            diavgeia_heavy_pii_threshold=args.diavgeia_pii_heavy_threshold,
        )
        for path, relative in files
    ]
    totals, per_source = _aggregate(receipts)
    if totals["input_rows"] != (
        totals["action:keep"] + totals["action:drop"] + totals["action:quarantine"]
    ):
        raise RuntimeError("global anonymization action accounting does not close")
    if totals["input_rows"] != totals["protected_ledger_rows"]:
        raise RuntimeError("global protected ledger coverage does not close")
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "completed",
        "completed_at": utc_now(),
        "implementation": {
            "policy_version": POLICY_VERSION,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "pii_masker_sha256": sha256_file(Path(pii_masker.__file__).resolve()),
            "greek_pii_sha256": sha256_file(Path(mask_greek_identifiers.__code__.co_filename).resolve()),
        },
        "policy": policy,
        "input": str(input_path),
        "output": str(output_root),
        "dropped": str(dropped_root),
        "quarantine": str(quarantine_root),
        "protected_ledger": {
            "path": str(ledger_root),
            "contains_raw_span_values": True,
            "directory_mode": "0700",
            "file_mode": "0600",
            "public_training_output": False,
        },
        "transform_boundaries": {
            "generic_person_name_ner": False,
            "street_address_masking": False,
            "html_cleaning": False,
            "ocr_cleaning": False,
            "structural_cleaning": False,
            "stable_uid_preserved": True,
            "new_child_representation_ids": True,
        },
        "counts": dict(sorted(totals.items())),
        "per_source": {key: dict(sorted(value.items())) for key, value in sorted(per_source.items())},
        "files": sorted(receipts, key=lambda value: str(value["relative_path"])),
    }
    _atomic_json_no_replace(args.manifest.resolve(), manifest, mode=0o600)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(args.manifest.resolve()),
                "counts": dict(sorted(totals.items())),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
