# vLLM Hybrid KV-Cache Page Normalization

Status: implemented.

`lmcache/integration/vllm/kv_cache_group_edits.py` normalizes registered vLLM
KV-cache tensors before LMCache derives transfer metadata. The normalization
preserves storage and bytes while making each tensor's page dimension match
the logical block identifiers supplied by the vLLM scheduler.

`LMCacheMPConnector.register_kv_caches` calls
`apply_kv_cache_group_edits(kv_cache_config, kv_caches, layout_hints)` once.
The resulting views feed both engine-group metadata construction and transfer
registration. A cache configuration without Mamba-compatible recurrent
layers bypasses every rule.

## Addressing contract

LMCache store and retrieve operations interpret block identifiers in units of
`kv_cache_spec.block_size`. An attention backend may instead allocate tensors
in smaller kernel pages, and a recurrent backend may allocate one opaque state
page. Registering either layout without normalization can make LMCache derive
the wrong block size or slot-compression ratio.

Every normalization rule satisfies these invariants:

- The output is a view over the input storage; no cache bytes are copied.
- One output page contains exactly `kv_cache_spec.page_size_bytes` bytes.
- The output page dimension uses scheduler-visible logical blocks.
- Shape dimensions on opaque views provide byte addressing only; they do not
  imply attention-head, key/value, or recurrent-state semantics.
- A tensor that cannot satisfy the byte and divisibility checks raises
  `ValueError` during cache registration.

## Structural rules

Rules match the cache specification and registered tensor structure. Model
names do not participate in selection. The first matching rule in `_EDITS`
wins.

### Unified recurrent-state pages

vLLM can register a recurrent layer as a contiguous tensor with shape
`[num_blocks, 1, 1, page_elements]`. `_MambaUnifiedViewEdit` exposes the same
bytes as one of these layouts:

- NHD: `[num_blocks, block_size, 1, head_size]`
- HND: `[num_blocks, 1, block_size, head_size]`

The rule verifies the declared page size and requires `page_elements` to be
divisible by `block_size`.

### Split recurrent-state pages

A recurrent layer may register `[conv_state, ssm_state]`, where both tensors
share a padded `conv | ssm | pad` page. `_MambaPageViewEdit` uses the view at
the page base and exposes the complete page as
`[num_blocks, 2, block_size, 1, head_size]`.

The rule verifies that the first tensor begins at storage offset zero and that
its block stride equals the declared page size.

### Kernel-paged MLA storage

An MLA backend may register
`[num_kernel_pages, kernel_block_size, head_size]` while the scheduler uses a
larger logical block. `_SubpagedMLAAttentionViewEdit` groups contiguous kernel
pages into
`[num_logical_blocks, logical_block_size, head_size]`.

For Kimi-K3, twelve contiguous 64-token kernel pages form one 768-token
logical page. Selection remains structural and therefore also supports other
MLA configurations with the same byte-addressing relationship.

### Kernel-paged standard attention storage

A standard attention backend may register
`[num_kernel_pages, 2, kernel_block_size, num_heads, head_size]` while a hybrid
cache configuration uses a larger scheduler block.
`_SubpagedAttentionViewEdit` groups complete kernel pages into
`[num_logical_blocks, 2, logical_block_size, 1, derived_head_size]`.

The output can interleave physical K and V regions at kernel-page boundaries.
Store and retrieve remain byte-exact because both operations use the same
view.

## Declared slot compression

A specification with `compress_ratio > 1` or `tq_slot_size > 0` declares
physical slot compression. Such a group bypasses page normalization and uses
the compression logic in `lmcache/v1/kv_layer_groups.py`.

The subpage rules additionally verify that the grouped kernel pages tile one
logical page exactly. An undeclared packed layout therefore fails
registration instead of being treated as ordinary paging.

## Unsupported cache specifications

`validate_kv_cache_groups` rejects these specifications with one aggregated
error:

- cross-attention cache groups;
- Mamba cache groups whose `mamba_cache_mode` is not `align`.

Non-aligned recurrent caches do not preserve reusable state snapshots at
logical block boundaries.

## Format discovery

`normalize_and_discover_per_layer_formats` detects each distinct tensor shape
separately when a hybrid layer list contains more than one physical layout.
Stride-based normalization also resolves singleton dimensions whose logical
shape is ambiguous despite `torch.Tensor.is_contiguous()` returning true.

Shared-memory migration applies the normalized shape and stride to the
caller-owned tensor so the SHM lifetime remains attached to the registered
object.

## Compatibility limits

Opaque page views support byte transport between store and retrieve paths that
use the same engine configuration. They do not provide semantic K/V or
recurrent-state structure for content-aware compression, blending, head
resharding, or layout conversion. Cache artifacts are not portable between
attention backends that choose different byte ordering inside a logical page.

## Validation

The unit suites cover:

- zero-copy NHD and HND recurrent-state views;
- zero-copy logical MLA views;
- page-size and incomplete-page rejection;
- mixed-layout format detection;
- stride-truthful singleton-dimension normalization;
- caller-owned shared-memory lifetime.

The vLLM integration must also verify a cold store followed by a cache-hit
retrieve with identical model output under the same cache geometry.

## Code map

| Responsibility | Path |
|---|---|
| Structural rules and validation | `lmcache/integration/vllm/kv_cache_group_edits.py` |
| Registration caller | `lmcache/integration/vllm/lmcache_mp_connector.py` |
| Shape and format discovery | `lmcache/v1/gpu_connector/kv_format/` |
| Per-group transfer metadata | `lmcache/v1/kv_layer_groups.py` |
| Shared-memory normalization | `lmcache/v1/platform/cpu/shm.py` |
| Structural rule tests | `tests/v1/test_vllm_kv_cache_group_edits.py` |
