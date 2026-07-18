#!/usr/bin/env python3
"""Directional heading semantics for bibliography block evolution.

The controller consumes heading-role predictions only after entry anchors have
formed core blocks.  No heading can seed a block.  It exposes pre-decoder walls
and a post-decoder attachment/connection pass so experiments cannot blur the
three deliberately different roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bibliography_entry_dataset import MAX_PHYSICAL_GAP


HEADER_ROLES = ("NONE", "BIB_HEADER", "BIB_SUBHEADER", "NON_BIB_HEADER")
ROLE_TO_ID = {name: index for index, name in enumerate(HEADER_ROLES)}


@dataclass(frozen=True)
class HeaderControllerConfig:
    attachment_window: int = 3
    connector_window: int = 4

    def __post_init__(self) -> None:
        if self.attachment_window < 1 or self.connector_window < 1:
            raise ValueError("header windows must be positive")


@dataclass(frozen=True)
class HeaderActions:
    hard_upward_stop: np.ndarray
    hard_downward_stop: np.ndarray
    connector: np.ndarray
    excluded: np.ndarray
    non_seed: np.ndarray


def header_actions(role_ids: np.ndarray) -> HeaderActions:
    role_ids = np.asarray(role_ids)
    if role_ids.ndim != 1 or np.any((role_ids < 0) | (role_ids >= len(HEADER_ROLES))):
        raise ValueError("invalid header-role vector")
    bib = role_ids == ROLE_TO_ID["BIB_HEADER"]
    sub = role_ids == ROLE_TO_ID["BIB_SUBHEADER"]
    non_bib = role_ids == ROLE_TO_ID["NON_BIB_HEADER"]
    return HeaderActions(
        # A bibliography heading belongs to the block below, but expansion
        # from that block must stop at it rather than consuming preceding text.
        hard_upward_stop=bib,
        # A non-bibliography heading is excluded and protects the section below
        # from a bibliography block above.
        hard_downward_stop=non_bib,
        connector=sub,
        excluded=non_bib,
        non_seed=bib | sub | non_bib,
    )


def _physical_step(abs_indices: np.ndarray, left: int, right: int) -> bool:
    return (
        0 <= left < len(abs_indices)
        and 0 <= right < len(abs_indices)
        and abs(int(abs_indices[right]) - int(abs_indices[left])) <= MAX_PHYSICAL_GAP
    )


def predecoder_walls(role_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return upward and downward directional walls before anchor decoding."""

    actions = header_actions(role_ids)
    return actions.hard_upward_stop.copy(), actions.hard_downward_stop.copy()


def apply_header_controller(
    core_prediction: np.ndarray,
    role_ids: np.ndarray,
    abs_indices: np.ndarray,
    *,
    hard_wall_mask: np.ndarray | None = None,
    config: HeaderControllerConfig = HeaderControllerConfig(),
) -> np.ndarray:
    """Attach main headings and connect subheadings without creating blocks.

    `core_prediction` must be the output of the anchored decoder with heading
    lines masked as non-seeds.  Exact scope walls are applied by the caller
    before this controller and remain higher-priority walls.
    """

    core = np.asarray(core_prediction, dtype=bool)
    roles = np.asarray(role_ids)
    absolute = np.asarray(abs_indices)
    if not (core.ndim == roles.ndim == absolute.ndim == 1):
        raise ValueError("header-controller inputs must be vectors")
    if not (len(core) == len(roles) == len(absolute)):
        raise ValueError("header-controller vectors differ in length")
    upward_stop, downward_stop = predecoder_walls(roles)
    actions = header_actions(roles)
    walls = (
        np.zeros(len(core), dtype=bool)
        if hard_wall_mask is None
        else np.asarray(hard_wall_mask, dtype=bool)
    )
    if walls.shape != core.shape:
        raise ValueError("hard wall mask does not align")
    # A malformed upstream candidate may have emitted a heading as an entry.
    # Strip all heading roles first so a heading can never be its own seed.
    result = core.copy()
    result[actions.non_seed] = False
    result[walls] = False

    # Trim a parent block that already crossed a directional heading before
    # this controller was introduced.  BIB_HEADER protects text above the
    # bibliography below; NON_BIB_HEADER protects the new section below a
    # bibliography above.  This makes G2 corrective, not merely additive.
    for header in np.flatnonzero(upward_stop):
        below = None
        cursor_below = int(header)
        for _ in range(config.attachment_window):
            candidate = cursor_below + 1
            if (
                candidate >= len(result)
                or walls[candidate]
                or not _physical_step(absolute, cursor_below, candidate)
            ):
                break
            if result[candidate]:
                below = candidate
                break
            cursor_below = candidate
        if below is None:
            continue
        cursor = int(header) - 1
        while cursor >= 0 and result[cursor] and not walls[cursor]:
            result[cursor] = False
            if cursor == 0 or not _physical_step(absolute, cursor - 1, cursor):
                break
            cursor -= 1
    for header in np.flatnonzero(downward_stop):
        if header == 0 or not result[header - 1]:
            continue
        cursor = int(header) + 1
        while cursor < len(result) and result[cursor] and not walls[cursor]:
            result[cursor] = False
            if cursor + 1 >= len(result) or not _physical_step(absolute, cursor, cursor + 1):
                break
            cursor += 1

    non_bib_positions = set(np.flatnonzero(actions.hard_downward_stop).tolist())

    # Attach BIB_HEADER only to an already-established block below.  The
    # heading becomes the first line and consequently the upward boundary.
    for header in np.flatnonzero(actions.hard_upward_stop):
        for distance in range(1, config.attachment_window + 1):
            candidate = int(header) + distance
            if candidate >= len(result) or not _physical_step(absolute, candidate - 1, candidate):
                break
            if candidate in non_bib_positions or walls[candidate]:
                break
            if result[candidate]:
                if not np.any(walls[int(header) : candidate]):
                    result[int(header) : candidate] = True
                break
            if actions.hard_upward_stop[candidate]:
                break

    # A subheading is a connector only when established material is visible on
    # both sides.  It cannot create a one-sided or all-heading block.
    for subheader in np.flatnonzero(actions.connector):
        left = right = None
        cursor = int(subheader)
        for _ in range(config.connector_window):
            candidate = cursor - 1
            if candidate < 0 or walls[candidate] or not _physical_step(absolute, candidate, cursor):
                break
            if actions.hard_downward_stop[candidate] or actions.hard_upward_stop[candidate]:
                break
            if result[candidate]:
                left = candidate
                break
            cursor = candidate
        cursor = int(subheader)
        for _ in range(config.connector_window):
            candidate = cursor + 1
            if candidate >= len(result) or walls[candidate] or not _physical_step(absolute, cursor, candidate):
                break
            if actions.hard_downward_stop[candidate] or actions.hard_upward_stop[candidate]:
                break
            if result[candidate]:
                right = candidate
                break
            cursor = candidate
        if left is not None and right is not None and not np.any(walls[left : right + 1]):
            result[left : right + 1] = True

    # NON_BIB_HEADER is unconditionally excluded even if it lies between two
    # blocks.  It therefore cannot be swallowed by a connector operation.
    result[actions.excluded] = False
    result[walls] = False
    return result


def assert_header_invariants(
    prediction: Sequence[bool], role_ids: Sequence[int]
) -> None:
    prediction = np.asarray(prediction, dtype=bool)
    roles = np.asarray(role_ids)
    actions = header_actions(roles)
    if np.any(prediction & actions.excluded):
        raise ValueError("NON_BIB_HEADER was included in a bibliography block")
