# SPDX-License-Identifier: Apache-2.0
"""Build LMCache engine group infos from vLLM KV cache group metadata."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import LayoutHints

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


def _is_mamba_spec(spec: Any) -> bool:
    """Return whether a spec (including a uniform wrapper) is Mamba-only.

    vLLM's ``MambaManager`` explicitly undoes DCP block-size scaling because
    every DCP rank holds the complete recurrent state. ``MambaSpec`` does not
    expose ``dcp_replicated`` itself, so mirror that manager contract here.
    """
    per_layer_specs = getattr(spec, "kv_cache_specs", None)
    if per_layer_specs:
        return all(
            _is_mamba_spec(layer_spec) for layer_spec in per_layer_specs.values()
        )
    return any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__)


def _kv_cache_cp_shard_count(spec: Any, dcp_size: int) -> int:
    """Mirror vLLM's context-parallel KV shard-count contract."""
    if dcp_size < 1:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    replicated = _is_mamba_spec(spec) or bool(getattr(spec, "dcp_replicated", False))
    override = getattr(spec, "dcp_kv_shard_count", None)
    if replicated:
        if override not in (None, 1):
            raise ValueError(
                f"dcp_replicated cannot be combined with dcp_kv_shard_count={override}"
            )
        return 1
    if override is None:
        return dcp_size
    override = int(override)
    if override < 1 or override > dcp_size or dcp_size % override != 0:
        raise ValueError(
            "dcp_kv_shard_count must be a positive divisor of dcp_size, "
            f"got shards={override}, dcp_size={dcp_size}"
        )
    return override


def effective_tokens_per_block(spec: Any, dcp_size: int = 1) -> int:
    """Return the global token span represented by one manager block ID.

    A sequence-sharded cache block covers ``block_size`` positions on each
    unique KV shard. Fully replicated groups have one shard; partially
    replicated groups declare ``dcp_kv_shard_count``. This mirrors vLLM's
    scheduler-side block geometry.
    """
    return int(spec.block_size) * _kv_cache_cp_shard_count(spec, dcp_size)


def _physical_blocks_per_engine_block(
    kv_cache_config: Any,
    physical_num_blocks: int,
    spec: Any,
    dcp_size: int,
) -> int:
    """Return how many physical kernel blocks back one manager block ID.

    Replicated or partially replicated DCP groups can split one
    scheduler-visible manager block into several kernel blocks. Other layouts
    do not expose a comparable manager/kernel block axis: notably a Mamba
    state cache can have fewer physical pages than ``KVCacheConfig.num_blocks``.
    Keep those layouts one-to-one instead of inferring geometry from unrelated
    counts.
    """
    if _is_mamba_spec(spec) or (_kv_cache_cp_shard_count(spec, dcp_size) >= dcp_size):
        return 1

    manager_num_blocks = getattr(kv_cache_config, "num_blocks", None)
    if manager_num_blocks is None:
        return 1
    manager_num_blocks = int(manager_num_blocks)
    if manager_num_blocks < 1:
        raise ValueError(
            f"KVCacheConfig.num_blocks must be positive, got {manager_num_blocks}"
        )
    if physical_num_blocks < manager_num_blocks or (
        physical_num_blocks % manager_num_blocks
    ):
        raise ValueError(
            "registered physical block count must be a whole multiple of "
            "KVCacheConfig.num_blocks, got "
            f"physical={physical_num_blocks}, manager={manager_num_blocks}"
        )
    return physical_num_blocks // manager_num_blocks


def _is_sliding_window_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM sliding-window spec.

    Checked by class name so this module stays importable without vLLM.
    Subclasses such as ``SlidingWindowMLASpec`` count.
    """
    return any(cls.__name__ == "SlidingWindowSpec" for cls in type(spec).__mro__)


def _is_mamba_align_spec(spec: Any) -> bool:
    """Return whether one engine block is the complete reusable Mamba state.

    Align-mode Mamba and linear-attention layers keep only the newest recurrent
    snapshot. Across LMCache chunks this is a one-block window, not a full
    history. Class-name detection keeps this module importable without vLLM.
    """
    return (
        any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__)
        and getattr(spec, "mamba_cache_mode", "none") == "align"
    )


def _resolve_per_layer_sw_sizes(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[int]:
    """Resolve the sliding window size in tokens for each registered KV tensor.

    Will resolve -1 for non-sliding-window layers.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers`` mapping each registered tensor index
        to its sliding window size in tokens, or ``-1`` for
        non-sliding-window layers.
    """
    per_layer_sw_size = [-1] * num_layers
    for group in vllm_groups:
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            continue
        # ``UniformTypeKVCacheSpecs`` carries per-layer specs in
        # ``kv_cache_specs``; other specs apply to all of the group's layers.
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = per_layer_specs[name] if per_layer_specs else spec
            if _is_sliding_window_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.sliding_window
            elif _is_mamba_align_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.block_size
    return per_layer_sw_size


def _merge_layer_sw_sizes(per_layer_sw_size: list[int], indices: list[int]) -> int:
    """Merge the per-layer sliding window sizes of one LMCache group.

    Args:
        per_layer_sw_size: Sliding window size per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common sliding window size in tokens, or ``-1`` when the
        layers are not sliding-window attention.

    Raises:
        ValueError: If the layers have different non-negative sliding window sizes.
    """
    sw_sizes = {per_layer_sw_size[idx] for idx in indices}
    if len(sw_sizes) != 1:
        raise ValueError(
            f"Layers with indices {indices} have different sliding window sizes "
            f"{sw_sizes}, but they are in the same group. This should "
            "not happen because vLLM should only group layers with the same "
            "KV cache spec, but got inconsistent metadata or registered tensors."
        )
    return sw_sizes.pop()


def create_engine_group_infos_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    layout_hints: "LayoutHints | None" = None,
    dcp_size: int = 1,
) -> list[EngineGroupInfo]:
    """Build the LMCache engine group infos from vLLM metadata and registered tensors.

    This is the single entry point for the vLLM -> LMCache conversion. It reads
    the vLLM-specific fields (``KVCacheConfig.kv_cache_groups`` and
    ``KVCacheGroupSpec.layer_names`` from the v1 KV cache interface), maps each
    engine KV cache group's layer names to registered tensor indices, then
    splits the layers by physical transfer identity using the real tensors (via
    the shared :func:`lmcache.v1.kv_layer_groups.group_layers_by_identity`).
    vLLM-specific field access is intentionally confined to this function.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` describing the engine KV cache
            groups (or ``None`` / no groups, which yields a single-group spec).
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order. Keys provide the layer-name -> tensor-index mapping; values
            are inspected for physical shape and dtype.
        layout_hints: Optional engine-provided layout hints forwarded to format
            detection (e.g. ``NHD``/``HND`` and compression metadata).
        dcp_size: Decode-context parallel size. Sequence-sharded groups expose
            their global token span per manager block; replicated and partially
            replicated groups use their actual unique KV shard count.

    Returns:
        The list of ``EngineGroupInfo`` in protocol order, i.e. the LMCache group
        order used by store/retrieve block IDs.
    """
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import (
        normalize_and_discover_per_layer_formats,
    )
    from lmcache.v1.kv_layer_groups import (
        EXCLUDED_ENGINE_GROUP,
        group_layers_by_identity,
    )
    from lmcache.v1.gpu_connector.utils import get_num_blocks

    # vLLM-specific field access (confined to this function): map each
    # registered KV tensor to its vLLM engine KV cache group index. vLLM places
    # every registered layer in exactly one group; layers in different groups
    # have disjoint block-id spaces and must not share an LMCache group. ``None``
    # means a single (non-hybrid) group, i.e. every layer shares one block-id
    # space.
    per_layer_discoverable_kv_caches = list(kv_caches.values())
    layer_to_idx = {name: idx for idx, name in enumerate(kv_caches.keys())}
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )

    layer_index_groups = [
        [layer_to_idx[name] for name in group.layer_names] for group in vllm_groups
    ]
    normalized_kv_caches, engine_kv_formats = normalize_and_discover_per_layer_formats(
        per_layer_discoverable_kv_caches,
        layer_index_groups,
        EngineType.VLLM,
        layout_hints,
    )
    num_layers = len(engine_kv_formats)
    # Layers absent from every engine group's ``layer_names`` are cross-layer
    # KV-sharing layers (e.g. google/gemma-4-E4B-it): vLLM aliases them to a
    # target owner's KV tensor, so the owner's group already covers them. Tag
    # them EXCLUDED_ENGINE_GROUP so they form no group of their own (a
    # wrong-block-size group would corrupt the per-group block-id counts).
    per_layer_group_idx: list[int] | None = None
    group_tokens_per_block: dict[int, int] = {}
    per_layer_sw_size = [-1] * num_layers
    if vllm_groups:
        per_layer_group_idx = [EXCLUDED_ENGINE_GROUP] * num_layers
        for engine_group_id, group in enumerate(vllm_groups):
            # Report the scheduler-visible global span of one block ID. The
            # physical slot count is discovered later from registered tensors;
            # their ratio naturally represents compression and/or DCP sharding.
            group_tokens_per_block[engine_group_id] = effective_tokens_per_block(
                group.kv_cache_spec, dcp_size
            )
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id
        per_layer_sw_size = _resolve_per_layer_sw_sizes(
            vllm_groups, layer_to_idx, num_layers
        )

    # Within one vLLM engine group, layers can have different hidden dimensions
    # (e.g. a different head count), which require different GPU copy kernels.
    # ``group_layers_by_identity`` splits each engine group further by physical
    # transfer identity (kv_size, num_heads, head_size, block_size, dtype), so
    # every resulting LMCache group can be served by a single copy kernel. It is
    # the shared, engine-neutral primitive the server reuses to reproduce the
    # same grouping from the registered tensors.
    infos: list[EngineGroupInfo] = []
    for identity, indices in group_layers_by_identity(
        normalized_kv_caches,
        engine_kv_formats,
        per_layer_group_idx,
    ):
        physical_num_blocks = get_num_blocks(
            [normalized_kv_caches[indices[0]]], identity.engine_kv_format
        )
        engine_spec = (
            vllm_groups[identity.engine_group_idx].kv_cache_spec
            if vllm_groups
            else None
        )
        infos.append(
            EngineGroupInfo(
                engine_group_id=identity.engine_group_idx,
                layer_indices=tuple(indices),
                tokens_per_block=group_tokens_per_block.get(
                    identity.engine_group_idx, 0
                ),
                sw_size_tokens=_merge_layer_sw_sizes(per_layer_sw_size, indices),
                physical_blocks_per_engine_block=(
                    _physical_blocks_per_engine_block(
                        kv_cache_config,
                        physical_num_blocks,
                        engine_spec,
                        dcp_size,
                    )
                    if vllm_groups
                    else 1
                ),
            )
        )
    return infos
