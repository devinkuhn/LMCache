# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    create_engine_group_infos_from_vllm,
    effective_tokens_per_block,
)
from lmcache.v1.multiprocess.group_view import (
    expand_engine_block_ids,
    get_engine_group_indices,
    num_engine_groups,
    slice_block_ids_per_group,
)

# Test doubles for the vLLM KV cache spec classes. Unit tests must run
# without vLLM installed; sliding-window specs are detected by class name,
# so the doubles share the vLLM class names.


@dataclass
class MockKVCacheSpec:
    block_size: int


@dataclass
class SlidingWindowSpec:
    block_size: int
    sliding_window: int


@dataclass
class SlidingWindowMLASpec(SlidingWindowSpec):
    pass


@dataclass
class FullAttentionSpec:
    block_size: int
    sliding_window: "int | None" = None
    dcp_replicated: bool = False


@dataclass
class MLAAttentionSpec:
    """Key-only, one-vector-per-token spec (an MLA index cache)."""

    block_size: int
    dcp_replicated: bool = False
    dcp_kv_shard_count: "int | None" = None


@dataclass
class MambaSpec:
    block_size: int
    mamba_cache_mode: str = "align"


@dataclass
class UniformTypeKVCacheSpecs:
    block_size: int
    kv_cache_specs: "dict[str, object]" = field(default_factory=dict)


@dataclass
class MockKVCacheGroup:
    layer_names: list[str]
    kv_cache_spec: object


@dataclass
class MockKVCacheConfig:
    kv_cache_groups: list[MockKVCacheGroup]
    num_blocks: "int | None" = None


def _same_shape_caches(names: list[str]) -> dict[str, torch.Tensor]:
    return {n: torch.randn(2, 32, 16, 8, 64, dtype=torch.float16) for n in names}


def _mla_caches(names: list[str]) -> dict[str, torch.Tensor]:
    """Key-only rank-3 caches (num_blocks, block_size, head_size), MLA layout."""
    return {n: torch.randn(32, 16, 128, dtype=torch.bfloat16) for n in names}


def test_conversion_defaults_to_single_group_without_config():
    """No vLLM KV cache groups -> all layers fall into a single engine group."""
    spec = create_engine_group_infos_from_vllm(
        None, _same_shape_caches(["layer.0", "layer.1"])
    )

    assert num_engine_groups(spec) == 1
    assert [group.engine_group_id for group in spec] == [0]
    assert spec[0].layer_indices == (0, 1)


def test_conversion_preserves_engine_group_layers():
    """Two engine groups with identical tensor shape stay separate by group."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    ["layer.0", "layer.2"], MockKVCacheSpec(block_size=16)
                ),
                MockKVCacheGroup(
                    ["layer.1", "layer.3"], MockKVCacheSpec(block_size=16)
                ),
            ]
        ),
        _same_shape_caches(["layer.0", "layer.1", "layer.2", "layer.3"]),
    )

    assert num_engine_groups(spec) == 2
    assert get_engine_group_indices(spec, 4) == [0, 1, 0, 1]
    assert [group.tokens_per_block for group in spec] == [16, 16]


def test_dcp_globalizes_only_sequence_sharded_group_block_spans():
    """DCP-sharded MLA and replicated indexer groups share global chunks."""
    caches = {
        **_same_shape_caches(["main.0", "main.1"]),
        **_mla_caches(["idx.0", "idx.1"]),
    }
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    ["main.0", "main.1"],
                    FullAttentionSpec(block_size=64),
                ),
                MockKVCacheGroup(
                    ["idx.0", "idx.1"],
                    MLAAttentionSpec(block_size=128, dcp_replicated=True),
                ),
            ]
        ),
        caches,
        dcp_size=4,
    )

    # One local main-cache block spans 64 positions on each of four DCP
    # ranks, while every rank holds a complete 128-token indexer block.
    assert [group.tokens_per_block for group in spec] == [256, 128]


def test_dcp_keeps_mamba_recurrent_state_block_span_local():
    """Mamba state is replicated across DCP ranks, not sequence-sharded."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[MockKVCacheGroup(["mamba.0"], MambaSpec(block_size=16))]
        ),
        {"mamba.0": torch.randn(2, 16, 128, dtype=torch.bfloat16)},
        dcp_size=4,
    )

    assert spec[0].tokens_per_block == 16


def test_dcp_uniform_mamba_group_preserves_real_tail_destination():
    """Chunk slicing must retain the real Mamba state after null placeholders."""
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"mamba.0": MambaSpec(block_size=16)},
    )
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[MockKVCacheGroup(["mamba.0"], uniform_spec)]
        ),
        {"mamba.0": torch.randn(2, 16, 128, dtype=torch.bfloat16)},
        dcp_size=4,
    )
    allocated = {0: [0] * 47 + [7]}

    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[spec[0].tokens_per_block],
        start_token_idx=0,
        end_token_idx=48 * 16,
    )

    assert spec[0].tokens_per_block == 16
    assert sliced == [[*([0] * 47), 7]]


def test_dcp_partial_replication_tracks_manager_and_physical_blocks():
    """A partial indexer shard expands manager IDs to physical kernel IDs."""
    caches = {
        "main.0": torch.randn(32, 64, 656, dtype=torch.bfloat16),
        "idx.0": torch.randn(64, 64, 132, dtype=torch.bfloat16),
    }
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["main.0"], MLAAttentionSpec(block_size=64)),
                MockKVCacheGroup(
                    ["idx.0"],
                    MLAAttentionSpec(block_size=128, dcp_kv_shard_count=2),
                ),
            ],
            num_blocks=32,
        ),
        caches,
        dcp_size=4,
    )

    assert [group.tokens_per_block for group in spec] == [256, 256]
    assert [group.physical_blocks_per_engine_block for group in spec] == [1, 2]
    assert expand_engine_block_ids(spec, [[10, 11], [20, 21]]) == [
        [10, 11],
        [40, 41, 42, 43],
    ]


def test_dcp_full_replication_expands_manager_blocks():
    """A fully replicated manager page can contain several physical blocks."""
    caches = {
        "main.0": torch.randn(32, 64, 656, dtype=torch.bfloat16),
        "idx.0": torch.randn(128, 64, 132, dtype=torch.bfloat16),
    }
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["main.0"], MLAAttentionSpec(block_size=64)),
                MockKVCacheGroup(
                    ["idx.0"],
                    MLAAttentionSpec(block_size=256, dcp_replicated=True),
                ),
            ],
            num_blocks=32,
        ),
        caches,
        dcp_size=4,
    )

    assert [group.tokens_per_block for group in spec] == [256, 256]
    assert [group.physical_blocks_per_engine_block for group in spec] == [1, 4]
    assert expand_engine_block_ids(spec, [[3], [7]]) == [[3], [28, 29, 30, 31]]


def test_physical_block_multiplier_does_not_confuse_compression():
    """A smaller slot dimension alone does not imply virtual block splitting."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["idx.0"], MLAAttentionSpec(block_size=16))
            ],
            num_blocks=32,
        ),
        {"idx.0": torch.randn(32, 8, 132, dtype=torch.bfloat16)},
    )

    assert spec[0].tokens_per_block == 16
    assert spec[0].physical_blocks_per_engine_block == 1


def test_physical_block_multiplier_skips_non_dcp_page_axes():
    """A small Mamba-style page pool is not a split manager block axis."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[MockKVCacheGroup(["mamba.0"], MambaSpec(block_size=16))],
            num_blocks=32,
        ),
        {"mamba.0": torch.randn(2, 16, 128, dtype=torch.bfloat16)},
        dcp_size=4,
    )

    assert spec[0].physical_blocks_per_engine_block == 1


@pytest.mark.parametrize(
    "spec,match",
    [
        (
            MLAAttentionSpec(block_size=128, dcp_kv_shard_count=3),
            "positive divisor",
        ),
        (
            MLAAttentionSpec(
                block_size=128,
                dcp_replicated=True,
                dcp_kv_shard_count=2,
            ),
            "cannot be combined",
        ),
    ],
)
def test_dcp_invalid_shard_metadata_is_rejected(spec, match):
    with pytest.raises(ValueError, match=match):
        effective_tokens_per_block(spec, dcp_size=4)


def test_nonintegral_physical_block_multiplier_is_rejected():
    with pytest.raises(ValueError, match="whole multiple"):
        create_engine_group_infos_from_vllm(
            MockKVCacheConfig(
                kv_cache_groups=[
                    MockKVCacheGroup(
                        ["idx.0"],
                        MLAAttentionSpec(block_size=128, dcp_replicated=True),
                    )
                ],
                num_blocks=32,
            ),
            {"idx.0": torch.randn(48, 64, 132, dtype=torch.bfloat16)},
            dcp_size=2,
        )


def test_conversion_splits_by_lmcache_layer_identity():
    """Layers split by both engine group and physical transfer identity."""
    caches = _same_shape_caches(["layer.0", "layer.1", "layer.2", "layer.3"])
    # layer.4 has a different head count -> distinct transfer identity.
    caches["layer.4"] = torch.randn(2, 32, 16, 16, 64, dtype=torch.float16)
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    ["layer.0", "layer.2", "layer.4"], MockKVCacheSpec(block_size=16)
                ),
                MockKVCacheGroup(
                    ["layer.1", "layer.3"], MockKVCacheSpec(block_size=16)
                ),
            ]
        ),
        caches,
    )

    assert [group.engine_group_id for group in spec] == [0, 1, 0]
    assert [group.layer_indices for group in spec] == [(0, 2), (1, 3), (4,)]
    assert expand_engine_block_ids(spec, [[10], [20]]) == [
        [10],
        [20],
        [10],
    ]


def test_conversion_resolves_sliding_window_size():
    """A SlidingWindowSpec group carries its window size in tokens;
    subclasses count too."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["layer.0"], FullAttentionSpec(block_size=16)),
                MockKVCacheGroup(
                    ["layer.1"], SlidingWindowSpec(block_size=16, sliding_window=64)
                ),
                MockKVCacheGroup(
                    ["layer.2"],
                    SlidingWindowMLASpec(block_size=16, sliding_window=128),
                ),
            ]
        ),
        _same_shape_caches(["layer.0", "layer.1", "layer.2"]),
    )

    assert [group.sw_size_tokens for group in spec] == [-1, 64, 128]


def test_conversion_resolves_mamba_align_window():
    """Align mode transfers only the newest recurrent snapshot per chunk."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["full.0"], FullAttentionSpec(block_size=16)),
                MockKVCacheGroup(["mamba.0"], MambaSpec(block_size=16)),
            ]
        ),
        _same_shape_caches(["full.0", "mamba.0"]),
    )

    assert [group.sw_size_tokens for group in spec] == [-1, 16]


def test_conversion_mamba_non_align_is_not_windowed():
    """Only align mode has the reusable last-state snapshot contract."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    ["mamba.0"],
                    MambaSpec(block_size=16, mamba_cache_mode="none"),
                )
            ]
        ),
        _same_shape_caches(["mamba.0"]),
    )

    assert spec[0].sw_size_tokens == -1


def test_conversion_ignores_full_attention_sliding_window():
    """SWA layers managed as full attention (hybrid allocator disabled) are
    not sliding window: vLLM allocates blocks for all tokens."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    ["layer.0", "layer.1"],
                    FullAttentionSpec(block_size=16, sliding_window=1024),
                ),
            ]
        ),
        _same_shape_caches(["layer.0", "layer.1"]),
    )

    assert [group.sw_size_tokens for group in spec] == [-1]


def test_conversion_defaults_sliding_window_for_non_sw_spec():
    """Groups whose spec is not a SlidingWindowSpec resolve to
    non-sliding-window."""
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["layer.0"], MockKVCacheSpec(block_size=16))
            ]
        ),
        _same_shape_caches(["layer.0"]),
    )

    assert [group.sw_size_tokens for group in spec] == [-1]


def test_conversion_uniform_type_specs_resolve_per_layer():
    """Inside a UniformTypeKVCacheSpecs group, per-layer specs decide the
    window. SW layers with a distinct transfer identity get their own group
    carrying the window size."""
    caches = _same_shape_caches(["layer.0", "layer.1"])
    # layer.1 has a different head count -> distinct transfer identity.
    caches["layer.1"] = torch.randn(2, 32, 16, 16, 64, dtype=torch.float16)
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer.0": FullAttentionSpec(block_size=16),
            "layer.1": SlidingWindowSpec(block_size=16, sliding_window=512),
        },
    )
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[MockKVCacheGroup(["layer.0", "layer.1"], uniform_spec)]
        ),
        caches,
    )

    assert [group.layer_indices for group in spec] == [(0,), (1,)]
    assert [group.sw_size_tokens for group in spec] == [-1, 512]


def test_conversion_uniform_type_specs_resolve_mamba_align_leaf():
    """Uniform wrappers inherit the Mamba leaf's one-block state window."""
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"mamba.0": MambaSpec(block_size=16)},
    )
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[MockKVCacheGroup(["mamba.0"], uniform_spec)]
        ),
        _same_shape_caches(["mamba.0"]),
    )

    assert spec[0].sw_size_tokens == 16


def test_conversion_mixed_window_layers_in_one_group_rejected():
    """Same-identity layers mixing different windows are inconsistent vLLM
    metadata and fail loudly."""
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer.0": FullAttentionSpec(block_size=16),
            "layer.1": SlidingWindowSpec(block_size=16, sliding_window=64),
        },
    )
    with pytest.raises(ValueError, match="different sliding window sizes"):
        create_engine_group_infos_from_vllm(
            MockKVCacheConfig(
                kv_cache_groups=[MockKVCacheGroup(["layer.0", "layer.1"], uniform_spec)]
            ),
            _same_shape_caches(["layer.0", "layer.1"]),
        )


def test_conversion_mixed_kv_and_mla_groups():
    """Mixed-format shape: a K+V FullAttentionSpec group plus a key-only MLA index
    group are detected per engine group and kept as separate LMCache groups with
    the correct membership and per-group token packing."""
    caches = {
        **_same_shape_caches(["main.0", "main.1"]),
        **_mla_caches(["idx.0", "idx.1"]),
    }
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    ["main.0", "main.1"], FullAttentionSpec(block_size=16)
                ),
                MockKVCacheGroup(["idx.0", "idx.1"], MLAAttentionSpec(block_size=128)),
            ]
        ),
        caches,
    )

    assert num_engine_groups(spec) == 2
    assert [group.engine_group_id for group in spec] == [0, 1]
    assert [group.layer_indices for group in spec] == [(0, 1), (2, 3)]
    assert [group.tokens_per_block for group in spec] == [16, 128]


def test_conversion_uniform_group_mixes_kv_and_mla_layouts():
    """vLLM can coalesce a rank-5 K+V group and a rank-3 key-only indexer group
    into ONE ``UniformTypeKVCacheSpecs`` group, not two. Detection must split it
    by layout so the indexer gets the rank-3 format instead of inheriting the
    K/V format; the two land in separate LMCache groups sharing one block-id
    space."""
    caches = {
        **_same_shape_caches(["main.0", "main.1"]),  # rank-5 K+V
        **_mla_caches(["idx.0", "idx.1"]),  # rank-3 indexer (key-only)
    }
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=128,
        kv_cache_specs={
            "main.0": FullAttentionSpec(block_size=128),
            "main.1": FullAttentionSpec(block_size=128),
            "idx.0": MLAAttentionSpec(block_size=128),
            "idx.1": MLAAttentionSpec(block_size=128),
        },
    )
    spec = create_engine_group_infos_from_vllm(
        MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(["main.0", "main.1", "idx.0", "idx.1"], uniform_spec)
            ]
        ),
        caches,
    )

    # One vLLM engine group, but two LMCache groups split by per-layer format.
    assert num_engine_groups(spec) == 1
    assert [group.engine_group_id for group in spec] == [0, 0]
    assert [group.layer_indices for group in spec] == [(0, 1), (2, 3)]
    # Both LMCache groups share the unified block-id space (tokens_per_block).
    assert [group.tokens_per_block for group in spec] == [128, 128]


def test_group_layers_by_identity_uses_per_layer_format():
    """A per-layer Engine KV format gives the K+V layer kv_size=2 and the MLA
    layer kv_size=1, splitting them into separate identities -- the per-group
    distinction the single global format cannot express."""
    # First Party
    from lmcache.v1.kv_layer_groups import group_layers_by_identity
    import lmcache.c_ops as lmc_ops

    kv_caches = [
        torch.randn(2, 32, 16, 8, 64, dtype=torch.bfloat16),  # K+V (rank-5)
        torch.randn(32, 16, 128, dtype=torch.bfloat16),  # MLA key-only (rank-3)
    ]
    per_layer_format = [
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
    ]
    groups = group_layers_by_identity(
        kv_caches,
        per_layer_format,
        per_layer_engine_group_idx=[0, 1],
    )

    kv_size_by_group = {
        identity.engine_group_idx: identity.kv_size for identity, _ in groups
    }
    num_heads_by_group = {
        identity.engine_group_idx: identity.num_heads for identity, _ in groups
    }
    assert kv_size_by_group == {0: 2, 1: 1}
    # The MLA group collapses heads to 1; the K+V group keeps its head count.
    assert num_heads_by_group == {0: 8, 1: 1}


def test_group_layers_by_identity_rejects_group_idx_length_mismatch():
    """per_layer_engine_group_idx must hold one entry per layer."""
    # First Party
    from lmcache.v1.kv_layer_groups import group_layers_by_identity
    import lmcache.c_ops as lmc_ops

    kv_caches = [torch.randn(2, 32, 16, 8, 64, dtype=torch.bfloat16)]
    # One layer (one format) but two engine-group ids.
    with pytest.raises(ValueError, match="per_layer_engine_group_idx"):
        group_layers_by_identity(
            kv_caches,
            [lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS],
            per_layer_engine_group_idx=[0, 1],
        )
