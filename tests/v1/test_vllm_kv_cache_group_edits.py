# SPDX-License-Identifier: Apache-2.0
"""Validate zero-copy KV page views used by vLLM hybrid-cache registration."""

# Standard
from types import SimpleNamespace

# Third Party
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import (
    apply_kv_cache_group_edits,
)


def _hybrid_config(mamba_spec: MambaSpec, mla_spec: MLAAttentionSpec):
    return SimpleNamespace(
        has_mamba_layers=True,
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=mamba_spec,
                layer_names=["recurrent.0"],
            ),
            SimpleNamespace(
                kv_cache_spec=mla_spec,
                layer_names=["mla.0"],
            ),
        ],
    )


@pytest.fixture
def hybrid_specs() -> tuple[MambaSpec, MLAAttentionSpec]:
    mamba_spec = MambaSpec(
        block_size=4,
        shapes=((2, 2), (2, 2, 2)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        page_size_padded=32,
        mamba_cache_mode="align",
    )
    mla_spec = MLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
    )
    return mamba_spec, mla_spec


@pytest.mark.parametrize(
    ("layout", "expected_shape"),
    [("NHD", (3, 4, 1, 8)), ("HND", (3, 1, 4, 8))],
)
def test_unified_recurrent_page_is_reinterpreted_without_copy(
    hybrid_specs: tuple[MambaSpec, MLAAttentionSpec],
    layout: str,
    expected_shape: tuple[int, ...],
):
    mamba_spec, mla_spec = hybrid_specs
    recurrent = torch.arange(3 * 32, dtype=torch.int8).view(3, 1, 1, 32)
    mla = torch.empty(4, 2, 16, dtype=torch.float16)

    edited = apply_kv_cache_group_edits(
        _hybrid_config(mamba_spec, mla_spec),
        {"recurrent.0": recurrent, "mla.0": mla},
        {"kv_layout": layout},
    )["recurrent.0"]

    assert tuple(edited.shape) == expected_shape
    assert edited.data_ptr() == recurrent.data_ptr()
    assert torch.equal(edited.reshape(-1), recurrent.reshape(-1))


def test_kernel_paged_mla_is_reinterpreted_without_copy(
    hybrid_specs: tuple[MambaSpec, MLAAttentionSpec],
):
    mamba_spec, mla_spec = hybrid_specs
    recurrent = torch.empty(2, 1, 1, 32, dtype=torch.int8)
    # Four 2-token pages form each 8-token logical block.
    mla = torch.arange(3 * 4 * 2 * 16, dtype=torch.float16).view(12, 2, 16)

    edited = apply_kv_cache_group_edits(
        _hybrid_config(mamba_spec, mla_spec),
        {"recurrent.0": recurrent, "mla.0": mla},
        {"kv_layout": "NHD"},
    )["mla.0"]

    assert tuple(edited.shape) == (3, 8, 16)
    assert edited.data_ptr() == mla.data_ptr()
    assert torch.equal(edited.reshape(-1), mla.reshape(-1))


def test_unified_recurrent_page_rejects_mismatched_page_bytes(
    hybrid_specs: tuple[MambaSpec, MLAAttentionSpec],
):
    mamba_spec, mla_spec = hybrid_specs
    recurrent = torch.empty(2, 1, 1, 24, dtype=torch.int8)
    mla = torch.empty(4, 2, 16, dtype=torch.float16)

    with pytest.raises(ValueError, match="declares 32 bytes"):
        apply_kv_cache_group_edits(
            _hybrid_config(mamba_spec, mla_spec),
            {"recurrent.0": recurrent, "mla.0": mla},
            {"kv_layout": "NHD"},
        )


def test_kernel_paged_mla_rejects_partial_logical_block(
    hybrid_specs: tuple[MambaSpec, MLAAttentionSpec],
):
    mamba_spec, mla_spec = hybrid_specs
    recurrent = torch.empty(2, 1, 1, 32, dtype=torch.int8)
    mla = torch.empty(10, 2, 16, dtype=torch.float16)

    with pytest.raises(ValueError, match="kernel page count 10"):
        apply_kv_cache_group_edits(
            _hybrid_config(mamba_spec, mla_spec),
            {"recurrent.0": recurrent, "mla.0": mla},
            {"kv_layout": "NHD"},
        )
