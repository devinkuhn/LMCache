# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache discovery."""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import (
    EngineDetector,
    measure_list_depth_until_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.c_ops as lmc_ops


class VLLM_Detector(EngineDetector):
    engine_type = EngineType.VLLM

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmc_ops.EngineKVFormat], DiscoverableKVCache]":
        # vLLM's CPU attention backend stores KV in HND but misreports it, so
        # force HND there; otherwise honor the hint, defaulting to NHD.
        kv_layout = layout_hints.get("kv_layout")
        if torch_device_type == "cpu":
            kv_layout = "HND"
        elif kv_layout is None:
            kv_layout = "NHD"
        is_hnd = kv_layout == "HND"

        # Blocks-first fused K/V is the only rank-4 vLLM layout, so its raw rank
        # identifies it unambiguously (the post-split 5-D shape would collide
        # with flash-infer when num_heads == 2). The two middle axes are NH/BS
        # (HND) or BS/NH (NHD); which is which is decided below by matching the
        # block size, falling back to the kv_layout hint. Split [NB, *, *, 2*HS]
        # into [NB, *, *, 2, HS].

        # TODO(ApostaC): deprecate NL_X_NB_NH_BS_TWO_HS/NL_X_NB_BS_NH_TWO_HS
        # and introduce more clear formats: NL_X_NB_NH_BS_CS/NL_X_NB_BS_NH_CS
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 4
        ):
            fused_dim = kv_caches[0].shape[3]
            if fused_dim % 2 != 0:
                raise ValueError(
                    f"blocks-first fused trailing dim {fused_dim} is not 2 * head_size"
                )
            # Split each blocks-first fused K/V tensor by its OWN trailing dim
            # into [NB, *, *, 2, HS]. A hybrid model can mix this layout with
            # other ranks in one per-layer list -- e.g. MiniMax-M3 registers a
            # rank-3 key-only lightning-index cache ([NB, BS, HS]) alongside the
            # rank-4 fused K/V layers. Leave any non-4-D (or odd-trailing) tensor
            # untouched so this whole-list pass does not crash on the index
            # cache; the returned per-layer-list format makes
            # normalize_and_discover_per_layer_formats re-detect each same-shape
            # group, classifying the index cache (rank-3 -> NL_X_NB_BS_HS) and
            # the main K/V cache (rank-4 fused) independently.
            split = [
                t.reshape(*t.shape[:3], 2, t.shape[3] // 2)
                if (
                    isinstance(t, torch.Tensor)
                    and t.dim() == 4
                    and t.shape[3] % 2 == 0
                )
                else t
                for t in kv_caches
            ]
            # The packed axis order is MODEL-DEPENDENT: vLLM registers the two
            # middle axes as (NH, BS) for some archs (MiniMax-M3: [NB, 1, 128, .])
            # and (BS, NH) for others (Llama: [NB, 64, 8, .]). The kv_layout hint
            # is the attention *stride* order and can be stale for the packed
            # cache (it reports NHD while the shape is heads-first), so trusting
            # it swaps NH<->BS -- at num_heads==1 that yields nh=block_size, bs=1
            # and a >1024-thread store launch (CUDA "invalid argument"). Prefer
            # the block size: whichever middle axis equals tokens_per_block is BS.
            # HND = [NB, NH, BS] (BS at axis 2); NHD = [NB, BS, NH] (BS at axis 1).
            # Fall back to the hint when block size is unknown or the two middle
            # axes are equal (genuinely ambiguous).
            block_size = layout_hints.get("tokens_per_block")
            d1, d2 = int(kv_caches[0].shape[1]), int(kv_caches[0].shape[2])
            if block_size is not None and d1 != d2:
                if d2 == block_size and d1 != block_size:
                    is_hnd = True
                elif d1 == block_size and d2 != block_size:
                    is_hnd = False
            if is_hnd:
                return lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS, split
            return lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_TWO_HS, split

        list_depth, tensor_ndim, first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )

        if list_depth == 0:
            return lmc_ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 5:
            if first_tensor.shape[0] == 2:  # K/V axis first
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, kv_caches
                return lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, kv_caches
            if first_tensor.shape[1] == 2:  # num_blocks first
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, kv_caches
                return lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 3:  # MLA
            return lmc_ops.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        return None, kv_caches
