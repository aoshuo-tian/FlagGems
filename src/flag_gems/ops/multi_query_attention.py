# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward-only multi query attention with independent Triton kernels."""

import math
import numbers

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_qkv_tensors(query, key, value, ndim=4):
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.ndim != ndim or any((size <= 0 for size in tensor.shape)):
            raise ValueError(f"{name} must have {ndim} nonempty dimensions")
        if tensor.device != query.device:
            raise ValueError("Q/K/V must be on the same device")
        if tensor.dtype != query.dtype or tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError("Q/K/V must share an FP16, BF16 or FP32 dtype")
    if query.device.type == "cpu":
        raise ValueError("attention requires an accelerator device")


def _resolve_mask_layout(attn_mask, query, shape):
    if attn_mask is None:
        return (0, 0, 0, 0), 1
    if not isinstance(attn_mask, torch.Tensor):
        raise TypeError("mask must be a tensor")
    if attn_mask.device != query.device:
        raise ValueError("mask must be on the query device")
    if attn_mask.dtype not in (torch.bool, *_SUPPORTED_DTYPES):
        raise TypeError("mask must have bool or floating dtype")
    if attn_mask.ndim > 4:
        raise ValueError("mask must be broadcastable to [B,H,Lq,Lk]")
    padding = 4 - attn_mask.ndim
    sizes = (1,) * padding + tuple(attn_mask.shape)
    strides = (0,) * padding + tuple(attn_mask.stride())
    for size, target in zip(sizes, shape):
        if size not in (1, target):
            raise ValueError("mask is not broadcastable to attention scores")
    return (
        tuple(0 if size == 1 else stride for size, stride in zip(sizes, strides)),
        sizes[0],
    )


def _validate_head_count(actual, requested, name):
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, numbers.Integral):
            raise TypeError(f"{name} must be an integer")
        if requested != actual:
            raise ValueError(f"{name} does not match the tensor")


def _resolve_softmax_scale(scale, head_dim):
    if scale is None:
        return head_dim ** (-0.5)
    if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
        raise TypeError("scale must be a finite real number")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    return float(scale)


def _validate_inputs(query, key, value, attn_mask, scale, num_query_heads):
    _validate_qkv_tensors(query, key, value)
    batch_size, query_heads, query_len, head_dim = query.shape
    if (
        key.shape != value.shape
        or key.shape[0] != batch_size
        or key.shape[3] != head_dim
    ):
        raise ValueError(
            "K/V shapes must match, and Q/K/V batch and head dimensions must agree"
        )
    if query_heads % key.shape[1]:
        raise ValueError("query head count must be divisible by KV head count")
    _validate_head_count(query_heads, num_query_heads, "num_query_heads")
    _validate_head_count(key.shape[1], 1, "num_kv_heads")
    _resolve_softmax_scale(scale, head_dim)
    _resolve_mask_layout(
        attn_mask, query, (batch_size, query_heads, query_len, key.shape[2])
    )


@triton.jit
def _multi_query_attention_tile(
    query,
    buffers,
    bases,
    offsets,
    state,
    key_start,
    softmax_scale,
    STRIDES: tl.constexpr,
    SHAPE: tl.constexpr,
    FLAGS: tl.constexpr,
    KEY_TILE_ALIGNED: tl.constexpr,
):
    """Inline the QK/mask/softmax/PV update for a single KV tile."""
    K, V, AttnMask = buffers
    key_base, value_base, mask_base = bases
    query_offsets, head_offsets = offsets
    row_max, row_sum, accumulator = state
    (
        stride_k_seq,
        stride_k_dim,
        stride_v_seq,
        stride_v_dim,
        stride_mask_q,
        stride_mask_k,
    ) = STRIDES
    ROW_COUNT, KEY_LEN, HEAD_DIM, BLOCK_N = SHAPE
    HAS_MASK, MASK_IS_BOOL, ROWS_ALIGNED, HEAD_DIM_ALIGNED = FLAGS
    key_offsets = key_start + tl.arange(0, BLOCK_N)
    key = tl.load(
        K
        + key_base
        + head_offsets[:, None] * stride_k_dim
        + key_offsets[None, :] * stride_k_seq,
        ((head_offsets[:, None] < HEAD_DIM) | HEAD_DIM_ALIGNED)
        & ((key_offsets[None, :] < KEY_LEN) | KEY_TILE_ALIGNED),
        other=0,
    )
    scores = tl.dot(query, key, input_precision="ieee") * softmax_scale
    valid = ((query_offsets[:, None] < ROW_COUNT) | ROWS_ALIGNED) & (
        (key_offsets[None, :] < KEY_LEN) | KEY_TILE_ALIGNED
    )
    if HAS_MASK:
        mask = tl.load(
            AttnMask
            + mask_base
            + query_offsets[:, None] * stride_mask_q
            + key_offsets[None, :] * stride_mask_k,
            valid,
            other=0,
        )
        if MASK_IS_BOOL:
            valid = valid & mask
        else:
            scores += mask.to(tl.float32) * 1.4426950408889634
    scores = tl.where(valid, scores, -float("inf"))
    new_max = tl.maximum(row_max, tl.max(scores, 1))
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    probabilities = tl.exp2(scores - safe_max[:, None])
    alpha = tl.exp2(row_max - safe_max)
    row_sum = row_sum * alpha + tl.sum(probabilities, 1)
    value = tl.load(
        V
        + value_base
        + key_offsets[:, None] * stride_v_seq
        + head_offsets[None, :] * stride_v_dim,
        ((key_offsets[:, None] < KEY_LEN) | KEY_TILE_ALIGNED)
        & ((head_offsets[None, :] < HEAD_DIM) | HEAD_DIM_ALIGNED),
        other=0,
    )
    accumulator = tl.dot(
        probabilities.to(value.dtype),
        value,
        accumulator * alpha[:, None],
        input_precision="ieee",
    )
    return new_max, row_sum, accumulator


@libentry()
@triton.jit
def multi_query_attention_fwd_kernel(
    Q,
    K,
    V,
    AttnMask,
    Out,
    LogSumExp,
    softmax_scale,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    MASK_STRIDES: tl.constexpr,
    MASK_BATCH: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PIPELINE_NS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    PACK_HEADS: tl.constexpr,
):
    ROW_COUNT: tl.constexpr = QUERY_HEADS * QUERY_LEN if PACK_HEADS else QUERY_LEN
    ROWS_ALIGNED: tl.constexpr = ROW_COUNT % BLOCK_M == 0
    KEY_LEN_ALIGNED: tl.constexpr = KEY_LEN % BLOCK_N == 0
    HEAD_DIM_ALIGNED: tl.constexpr = HEAD_DIM == HEAD_BLOCK
    if PACK_HEADS:
        batch = tl.program_id(1).to(tl.int64)
        batch_coord = batch
        query_head = 0
        query_offsets = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        head_offsets = tl.arange(0, HEAD_BLOCK).to(tl.int64)
        if Q_STRIDES[1] == QUERY_LEN * Q_STRIDES[2]:
            query_rows = query_offsets * Q_STRIDES[2]
        else:
            query_rows = (
                query_offsets // QUERY_LEN * Q_STRIDES[1]
                + query_offsets % QUERY_LEN * Q_STRIDES[2]
            )
        query_base = batch * Q_STRIDES[0]
    else:
        batch_head = tl.program_id(1)
        batch, query_head = batch_head // QUERY_HEADS, batch_head % QUERY_HEADS
        batch_coord = batch % BATCH_SIZE
        query_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        head_offsets = tl.arange(0, HEAD_BLOCK)
        query_base = batch_coord.to(tl.int64) * Q_STRIDES[0]
        query_rows = query_offsets * Q_STRIDES[2]
    query = tl.load(
        Q
        + query_base
        + query_head * Q_STRIDES[1]
        + query_rows[:, None]
        + head_offsets[None, :] * Q_STRIDES[3],
        ((query_offsets[:, None] < ROW_COUNT) | ROWS_ALIGNED)
        & ((head_offsets[None, :] < HEAD_DIM) | HEAD_DIM_ALIGNED),
        other=0,
    )
    key_base, value_base = (
        batch_coord.to(tl.int64) * K_STRIDES[0],
        batch_coord.to(tl.int64) * V_STRIDES[0],
    )
    mask_base = tl.full((), 0, tl.int64)
    if HAS_MASK:
        mask_base = batch % MASK_BATCH * MASK_STRIDES[0] + query_head * MASK_STRIDES[1]
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_BLOCK), tl.float32)
    key_begin, key_end = 0, KEY_LEN
    split = tl.program_id(2)
    if PACK_HEADS:
        split = split.to(tl.int64)
    if PACK_HEADS or NUM_SPLITS > 1:
        keys_per_split = tl.cdiv(KEY_LEN, NUM_SPLITS * BLOCK_N) * BLOCK_N
        key_begin = split * keys_per_split
        key_end = tl.minimum(key_begin + keys_per_split, KEY_LEN)
    buffers = (K, V, AttnMask)
    bases = (key_base, value_base, mask_base)
    offsets = (query_offsets, head_offsets)
    tile_strides: tl.constexpr = (
        K_STRIDES[2],
        K_STRIDES[3],
        V_STRIDES[2],
        V_STRIDES[3],
        MASK_STRIDES[2],
        MASK_STRIDES[3],
    )
    tile_shape: tl.constexpr = (ROW_COUNT, KEY_LEN, HEAD_DIM, BLOCK_N)
    tile_flags: tl.constexpr = (HAS_MASK, MASK_IS_BOOL, ROWS_ALIGNED, HEAD_DIM_ALIGNED)
    if (
        not PACK_HEADS
        and not HAS_MASK
        and Q.dtype.element_ty != tl.float32
        and HEAD_DIM <= 16
        and not KEY_LEN_ALIGNED
        and NUM_SPLITS == 1
    ):
        for key_start in tl.range(
            0, KEY_LEN // BLOCK_N * BLOCK_N, BLOCK_N, num_stages=PIPELINE_NS
        ):
            row_max, row_sum, accumulator = _multi_query_attention_tile(
                query,
                buffers,
                bases,
                offsets,
                (row_max, row_sum, accumulator),
                key_start,
                softmax_scale,
                tile_strides,
                tile_shape,
                tile_flags,
                True,
            )
        for key_start in tl.range(
            KEY_LEN // BLOCK_N * BLOCK_N, KEY_LEN, BLOCK_N, num_stages=1
        ):
            row_max, row_sum, accumulator = _multi_query_attention_tile(
                query,
                buffers,
                bases,
                offsets,
                (row_max, row_sum, accumulator),
                key_start,
                softmax_scale,
                tile_strides,
                tile_shape,
                tile_flags,
                False,
            )
    else:
        for key_start in tl.range(key_begin, key_end, BLOCK_N, num_stages=PIPELINE_NS):
            row_max, row_sum, accumulator = _multi_query_attention_tile(
                query,
                buffers,
                bases,
                offsets,
                (row_max, row_sum, accumulator),
                key_start,
                softmax_scale,
                tile_strides,
                tile_shape,
                tile_flags,
                KEY_LEN_ALIGNED,
            )
    # Keep tile offsets in int32 on the per-head path; flattening them to
    # int64 changes code generation and slows several small-head shapes.
    result = accumulator / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    tl.store(
        Out
        + tl.program_id(2).to(tl.int64)
        * BATCH_SIZE
        * QUERY_HEADS
        * QUERY_LEN
        * HEAD_DIM
        + batch.to(tl.int64) * QUERY_HEADS * QUERY_LEN * HEAD_DIM
        + query_head * QUERY_LEN * HEAD_DIM
        + query_offsets[:, None] * HEAD_DIM
        + head_offsets[None, :],
        result,
        ((query_offsets[:, None] < ROW_COUNT) | ROWS_ALIGNED)
        & ((head_offsets[None, :] < HEAD_DIM) | HEAD_DIM_ALIGNED),
    )
    if NUM_SPLITS > 1:
        tl.store(
            LogSumExp
            + tl.program_id(2).to(tl.int64) * BATCH_SIZE * QUERY_HEADS * QUERY_LEN
            + (batch * QUERY_HEADS + query_head).to(tl.int64) * QUERY_LEN
            + query_offsets,
            row_max + tl.log2(row_sum),
            query_offsets < ROW_COUNT,
        )


@libentry()
@triton.jit
def multi_query_attention_split_kv_merge_kernel(
    PartialOut,
    LogSumExp,
    Out,
    NUMEL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_WISE: tl.constexpr,
):
    split_offsets = tl.arange(0, NUM_SPLITS)
    if ROW_WISE:
        split_offsets = split_offsets.to(tl.int64)
        row = tl.program_id(0).to(tl.int64)
        channels = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        output_offsets = row * HEAD_DIM + channels
        output_mask = channels < HEAD_DIM
        log_normalizers = tl.load(LogSumExp + split_offsets * (NUMEL // HEAD_DIM) + row)
    else:
        output_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        output_mask = output_offsets < NUMEL
        log_normalizers = tl.load(
            LogSumExp
            + split_offsets[:, None] * (NUMEL // HEAD_DIM)
            + output_offsets[None, :] // HEAD_DIM,
            output_mask[None, :],
            other=-float("inf"),
        )
    maximum = tl.max(log_normalizers, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(log_normalizers - maximum)
    denominator = tl.sum(weights, 0)
    if ROW_WISE:
        weights = weights[:, None]
    values = tl.load(
        PartialOut + split_offsets[:, None] * NUMEL + output_offsets[None, :],
        output_mask[None, :],
        other=0,
    )
    result = tl.sum(values * weights, 0) / tl.where(denominator > 0, denominator, 1.0)
    tl.store(Out + output_offsets, result, output_mask)


def _select_packed_tiling(batch_size, query_heads, query_len, key_len, head_dim):
    """Tile the shared-KV query rows and expose enough KV work for short queries."""
    if query_len <= 16:
        if head_dim > 128:
            block_m, block_n, pipeline_ns = 16, 32, 2
        elif query_len > 1:
            block_m, block_n, pipeline_ns = 64, 128, 2
        elif key_len <= 2048 and batch_size == 1 and query_heads <= 32:
            block_m, block_n, pipeline_ns = 16, 128, 2
        elif batch_size >= 16 and key_len >= 16384:
            block_m, block_n, pipeline_ns = 32, 128, 2
        else:
            block_m, block_n, pipeline_ns = 32, 64, 3
    else:
        block_m, block_n, pipeline_ns = 128, 64, 2
    query_programs = batch_size * triton.cdiv(query_heads * query_len, block_m)
    num_splits = 1
    if query_programs < 128 and (query_len <= 16 or key_len >= 4 * query_len):
        # Shorter KV partitions reduce the serial softmax dependency chain.
        desired_splits = max(
            triton.cdiv(128, query_programs), triton.cdiv(key_len, 2048)
        )
        key_tiles = triton.cdiv(key_len, block_n)
        max_splits = min(32, 1 << (key_tiles.bit_length() - 1))
        num_splits = min(max_splits, triton.next_power_of_2(desired_splits))
    return block_m, block_n, num_splits, 4, pipeline_ns


def _select_kv_splits(batch_size, query_heads, query_len, key_len, block_m):
    """Expose KV parallelism only when the query grid cannot fill the device."""
    if (
        key_len >= 4096
        and key_len >= 8 * query_len
        and (batch_size * query_heads * triton.cdiv(query_len, block_m) < 128)
    ):
        return 4
    return 1


def _select_gpu_tiling(
    query_len, key_len, head_dim, batch_size, query_heads, is_fp32, has_mask
):
    """Portable launch policy; changing tiles never relaxes FP32 precision."""
    query_tiles = batch_size * query_heads * triton.cdiv(query_len, 64)
    if key_len >= 8 * query_len and head_dim <= 64 and (query_tiles < 64):
        if is_fp32:
            return (16, 32, 1, 4)
        return (16 if has_mask else 32, 64, 2, 4)
    # Wider small-head FP32 tiles reduce online-softmax iterations.
    if not has_mask:
        if is_fp32:
            if head_dim <= 32 and query_len >= 128 and (key_len >= 128):
                return (64, 64, 1, 4)
            if 64 <= head_dim <= 128 and query_len <= 128 and (key_len <= 256):
                return (32, 64, 2, 4)
            if head_dim == 128 and query_len <= 2048 and (key_len >= 512):
                return (32, 64, 2, 4)
        elif (
            head_dim == 64
            and query_len >= 1024
            and query_len % 64
            and (batch_size * query_heads * triton.cdiv(query_len, 128) < 128)
        ):
            return (64, 64, 3, 4)
        if is_fp32 and head_dim >= 64 and (query_len >= 512) and (key_len >= 256):
            return (64, 64, 1, 4)
        if not is_fp32 and head_dim <= 64 and (query_len >= 1024):
            if (
                query_len % 64
                and key_len >= 2048
                and (batch_size * query_heads * triton.cdiv(query_len, 128) >= 128)
            ):
                return (128, 64, 2, 4)
            if head_dim == 64:
                return (64, 32, 2, 2)
            if head_dim == 32 and query_len % 64 == 0 and (key_len % 32 == 0):
                return (64, 32, 3, 4)
            return (64, 64, 3, 4)
    if query_len >= 1024 and key_len <= 512 and (96 <= head_dim <= 128):
        return (64, 64, 1, 4) if is_fp32 else (64, 32, 3, 4)
    block_m = 32 if is_fp32 or query_len < 64 else 64
    block_n = 32 if is_fp32 or head_dim > 64 or has_mask else 64
    return (block_m, block_n, 1 if is_fp32 else 2, 4)


def _launch_attention_forward(query, key, value, attn_mask=None, scale=None):
    batch_size, query_heads, query_len, head_dim = query.shape
    key_len = key.shape[2]
    pack_heads = False
    if attn_mask is None and query.dtype in (torch.float16, torch.bfloat16):
        pack_heads = (
            (query_len <= 16 and key_len >= 512 and head_dim <= 256)
            or (
                head_dim == 128
                and query_len >= 128
                and key_len >= 1024
                and key_len % 64 == 0
            )
            or (
                head_dim == 32
                and query_len >= 128
                and key_len >= 8 * query_len
                and batch_size * query_heads * triton.cdiv(query_len, 128) < 128
            )
        )
    if pack_heads:
        block_m, block_n, num_splits, num_warps, pipeline_ns = _select_packed_tiling(
            batch_size, query_heads, query_len, key_len, head_dim
        )
        mask_strides, mask_batch = (0, 0, 0, 0), 1
    else:
        block_m, block_n, pipeline_ns, num_warps = _select_gpu_tiling(
            query_len,
            key_len,
            head_dim,
            batch_size,
            query_heads,
            query.dtype == torch.float32,
            attn_mask is not None,
        )
        num_splits = _select_kv_splits(
            batch_size, query_heads, query_len, key_len, block_m
        )
        mask_strides, mask_batch = _resolve_mask_layout(
            attn_mask, query, (batch_size, query_heads, query_len, key_len)
        )
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    partial_output, log_normalizers = output, query
    if num_splits > 1:
        partial_output = torch.empty(
            (num_splits, batch_size, query_heads, query_len, head_dim),
            device=query.device,
            dtype=torch.float32,
        )
        log_normalizers = torch.empty(
            (num_splits, batch_size, query_heads, query_len),
            device=query.device,
            dtype=torch.float32,
        )
    head_block = max(16, triton.next_power_of_2(head_dim))
    row_count = query_heads * query_len if pack_heads else query_len
    batch_programs = batch_size if pack_heads else batch_size * query_heads
    with torch_device_fn.device(query.device):
        multi_query_attention_fwd_kernel[
            (triton.cdiv(row_count, block_m), batch_programs, num_splits)
        ](
            query,
            key,
            value,
            query if attn_mask is None else attn_mask,
            partial_output,
            log_normalizers,
            _resolve_softmax_scale(scale, head_dim) * 1.4426950408889634,
            Q_STRIDES=query.stride(),
            K_STRIDES=key.stride(),
            V_STRIDES=value.stride(),
            MASK_STRIDES=mask_strides,
            MASK_BATCH=mask_batch,
            BATCH_SIZE=batch_size,
            QUERY_HEADS=query_heads,
            QUERY_LEN=query_len,
            KEY_LEN=key_len,
            HEAD_DIM=head_dim,
            HEAD_BLOCK=head_block,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            NUM_SPLITS=num_splits,
            PIPELINE_NS=pipeline_ns,
            HAS_MASK=attn_mask is not None,
            MASK_IS_BOOL=attn_mask is not None and attn_mask.dtype == torch.bool,
            PACK_HEADS=pack_heads,
            num_warps=num_warps,
            num_stages=pipeline_ns,
        )
        if num_splits > 1:
            numel = batch_size * query_heads * query_len * head_dim
            block_size = head_block if pack_heads else 256
            grid = (
                batch_size * query_heads * query_len
                if pack_heads
                else triton.cdiv(numel, block_size)
            )
            multi_query_attention_split_kv_merge_kernel[(grid,)](
                partial_output,
                log_normalizers,
                output,
                NUMEL=numel,
                HEAD_DIM=head_dim,
                NUM_SPLITS=num_splits,
                BLOCK_SIZE=block_size,
                ROW_WISE=pack_heads,
                num_warps=4,
            )
    return output


def multi_query_attention_mqa(
    query, key, value, mask=None, scale=None, num_query_heads=None
):
    """Forward multi-query attention with exactly one KV head; bool True allows attention."""
    _validate_inputs(query, key, value, mask, scale, num_query_heads)
    return _launch_attention_forward(query, key, value, mask, scale)
