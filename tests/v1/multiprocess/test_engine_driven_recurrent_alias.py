# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic engine-driven recurrent-state restores."""

import torch

from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    _collapse_chunks_for_single_destination,
)


def _chunks() -> list[torch.Tensor]:
    return [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])]


def test_complete_full_chunk_alias_keeps_newest_snapshot() -> None:
    chunks = _chunks()

    selected_chunks, selected_ids = _collapse_chunks_for_single_destination(
        chunks,
        [4, 5, 4, 5, 4, 5],
        blocks_in_chunk=2,
        blocks_per_window=2,
    )

    assert selected_chunks == chunks[-1:]
    assert selected_ids == [4, 5]


def test_complete_window_alias_keeps_newest_full_mapping() -> None:
    chunks = _chunks()
    block_ids = [10, 11, 12, 7, 20, 21, 22, 7, 30, 31, 32, 7]

    selected_chunks, selected_ids = _collapse_chunks_for_single_destination(
        chunks,
        block_ids,
        blocks_in_chunk=4,
        blocks_per_window=1,
    )

    assert selected_chunks == chunks[-1:]
    assert selected_ids == [30, 31, 32, 7]


def test_distinct_window_destinations_remain_unchanged() -> None:
    chunks = _chunks()
    block_ids = [10, 11, 12, 7, 20, 21, 22, 8, 30, 31, 32, 7]

    selected_chunks, selected_ids = _collapse_chunks_for_single_destination(
        chunks,
        block_ids,
        blocks_in_chunk=4,
        blocks_per_window=1,
    )

    assert selected_chunks is chunks
    assert selected_ids is block_ids


def test_incomplete_or_invalid_geometry_remains_unchanged() -> None:
    chunks = _chunks()
    cases = (
        ([0, 0], 1, 1),
        ([0, 0, 0], 2, 1),
        ([0, 0, 0], 1, 2),
    )

    for block_ids, blocks_in_chunk, blocks_per_window in cases:
        selected_chunks, selected_ids = _collapse_chunks_for_single_destination(
            chunks,
            block_ids,
            blocks_in_chunk=blocks_in_chunk,
            blocks_per_window=blocks_per_window,
        )
        assert selected_chunks is chunks
        assert selected_ids is block_ids
