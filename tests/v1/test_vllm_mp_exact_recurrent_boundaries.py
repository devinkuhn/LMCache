# SPDX-License-Identifier: Apache-2.0
"""Exact recurrent-boundary tests for the vLLM MP connector."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module load")

# Third Party
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
    _is_mamba_group_spec,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)

GROUP_TOKENS_PER_BLOCK = [512, 512, 512, 512, 2048, 512]
CHUNK_TOKENS = 4096
RECURRENT_GROUP_ID = 5


class MambaSpec:
    """Compatibility stand-in identified by vLLM's recurrent spec name."""


class AttentionSpec:
    """Non-recurrent compatibility stand-in."""


def test_recurrent_group_identity_handles_uniform_group_specs() -> None:
    recurrent_group = SimpleNamespace(
        kv_cache_specs={
            "attention": AttentionSpec(),
            "recurrent": MambaSpec(),
        }
    )

    assert _is_mamba_group_spec(recurrent_group)
    assert not _is_mamba_group_spec(AttentionSpec())


def _store_tracker(num_chunks: int = 2) -> LMCacheMPRequestTracker:
    """Build a tracker with complete positional tables for DFlash geometry."""
    num_tokens = num_chunks * CHUNK_TOKENS
    token_ids = list(range(num_tokens))
    request = SimpleNamespace(
        request_id="dflash-store",
        cache_salt="",
        prompt_token_ids=token_ids,
        all_token_ids=token_ids,
        mm_features=[],
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {
        group_id: list(
            range(
                1000 * (group_id + 1),
                1000 * (group_id + 1) + num_tokens // group_tokens_per_block,
            )
        )
        for group_id, group_tokens_per_block in enumerate(GROUP_TOKENS_PER_BLOCK)
    }
    tracker.num_scheduled_tokens = num_tokens
    return tracker


def test_tracker_initializes_exact_recurrent_boundary_map() -> None:
    tracker = _store_tracker()

    assert tracker.exact_mamba_boundary_blocks == {}


def test_store_uses_exact_boundaries_for_six_group_dflash_geometry() -> None:
    tracker = _store_tracker()
    tracker.exact_mamba_boundary_blocks = {
        RECURRENT_GROUP_ID: {CHUNK_TOKENS: 91, 2 * CHUNK_TOKENS: 97}
    }
    positional_attention_ids = [
        list(tracker.allocated_block_ids[group_id])
        for group_id in range(RECURRENT_GROUP_ID)
    ]

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK_TOKENS,
        GROUP_TOKENS_PER_BLOCK,
        {RECURRENT_GROUP_ID},
    )

    assert metadata is not None
    assert metadata.op.block_ids[:RECURRENT_GROUP_ID] == positional_attention_ids
    assert metadata.op.block_ids[RECURRENT_GROUP_ID] == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        91,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        97,
    ]
    assert tracker.num_stored_tokens == 2 * CHUNK_TOKENS


@pytest.mark.parametrize(
    "exact_boundaries",
    [
        {},
        {CHUNK_TOKENS: 91},
        {2 * CHUNK_TOKENS: 97},
    ],
)
def test_store_fails_closed_when_any_exact_boundary_is_missing(
    exact_boundaries: dict[int, int],
) -> None:
    tracker = _store_tracker()
    tracker.exact_mamba_boundary_blocks = {RECURRENT_GROUP_ID: exact_boundaries}
    positional_tables = {
        group_id: list(block_ids)
        for group_id, block_ids in tracker.allocated_block_ids.items()
    }

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK_TOKENS,
        GROUP_TOKENS_PER_BLOCK,
        {RECURRENT_GROUP_ID},
    )

    assert metadata is None
    assert tracker.num_stored_tokens == 0
    assert tracker.allocated_block_ids == positional_tables


def test_exact_store_remains_enabled_after_mixed_recurrent_suppression() -> None:
    tracker = _store_tracker(num_chunks=1)
    tracker.suppress_mixed_recurrent_retrieve = True
    tracker.exact_mamba_boundary_blocks = {RECURRENT_GROUP_ID: {CHUNK_TOKENS: 91}}

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK_TOKENS,
        GROUP_TOKENS_PER_BLOCK,
        {RECURRENT_GROUP_ID},
    )

    assert metadata is not None
    assert metadata.op.block_ids[RECURRENT_GROUP_ID] == [0] * 7 + [91]


def test_connector_ingests_only_exact_handoffs_for_recurrent_group() -> None:
    tracker = _store_tracker(num_chunks=1)
    tracker.state = LMCacheMPRequestState.READY
    connector = cast(LMCacheMPConnector, object.__new__(LMCacheMPConnector))
    connector.request_trackers = {"known": tracker}
    connector._mamba_group_ids = {RECURRENT_GROUP_ID}
    connector._group_tokens_per_block = GROUP_TOKENS_PER_BLOCK
    connector.lazy_offload = False
    connector.scheduler_adapter = SimpleNamespace(  # type: ignore[assignment]
        lmcache_tokens_per_chunk=CHUNK_TOKENS,
        report_block_allocations=lambda records: None,
    )
    scheduler_output = SimpleNamespace(
        partial_tail_offloads={
            "unknown": [(RECURRENT_GROUP_ID, 71, CHUNK_TOKENS)],
            "known": [
                (RECURRENT_GROUP_ID, 91, CHUNK_TOKENS),
                (4, 92, CHUNK_TOKENS),
                (RECURRENT_GROUP_ID, 0, 2 * CHUNK_TOKENS),
                (RECURRENT_GROUP_ID, -1, 2 * CHUNK_TOKENS),
            ],
        },
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["known"],
            new_block_ids=[None],
            resumed_req_ids={"known"},
        ),
        num_scheduled_tokens={"known": 0},
        total_num_scheduled_tokens=0,
        preempted_req_ids=[],
    )

    connector.build_connector_meta(scheduler_output)

    assert tracker.exact_mamba_boundary_blocks == {
        RECURRENT_GROUP_ID: {CHUNK_TOKENS: 91}
    }


def test_connector_retains_new_request_handoff_before_first_store() -> None:
    tracker = _store_tracker(num_chunks=1)
    tracker.num_scheduled_tokens = 0

    class NewRequestConnector(LMCacheMPConnector):
        def _process_new_requests(
            self,
            scheduler_output: SchedulerOutput,
            metadata: LMCacheMPConnectorMetadata,
        ) -> None:
            self.request_trackers["new-request"] = tracker
            super()._process_new_requests(scheduler_output, metadata)

    connector = cast(NewRequestConnector, object.__new__(NewRequestConnector))
    connector.request_trackers = {}
    connector._mamba_group_ids = {RECURRENT_GROUP_ID}
    connector._group_tokens_per_block = GROUP_TOKENS_PER_BLOCK
    connector.lazy_offload = False
    connector.scheduler_adapter = SimpleNamespace(  # type: ignore[assignment]
        lmcache_tokens_per_chunk=CHUNK_TOKENS,
        report_block_allocations=lambda records: None,
    )
    scheduler_output = SimpleNamespace(
        partial_tail_offloads={"new-request": [(RECURRENT_GROUP_ID, 91, CHUNK_TOKENS)]},
        scheduled_new_reqs=[SimpleNamespace(req_id="new-request")],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={"new-request": CHUNK_TOKENS},
        total_num_scheduled_tokens=CHUNK_TOKENS,
        preempted_req_ids=[],
    )

    metadata = connector.build_connector_meta(cast(SchedulerOutput, scheduler_output))

    assert isinstance(metadata, LMCacheMPConnectorMetadata)
    assert tracker.exact_mamba_boundary_blocks == {
        RECURRENT_GROUP_ID: {CHUNK_TOKENS: 91}
    }
    assert len(metadata.requests) == 1
    assert metadata.requests[0].direction == "STORE"
    assert metadata.requests[0].op.block_ids[RECURRENT_GROUP_ID] == [0] * 7 + [91]
