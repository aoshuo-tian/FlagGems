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

"""Forward-only group query attention with independent Triton kernels."""

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


def _validate_inputs(
    query, key, value, attn_mask, scale, num_query_heads, num_kv_heads
):
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
    _validate_head_count(key.shape[1], num_kv_heads, "num_kv_heads")
    _resolve_softmax_scale(scale, head_dim)
    _resolve_mask_layout(
        attn_mask, query, (batch_size, query_heads, query_len, key.shape[2])
    )


@triton.jit
def _group_query_attention_forward_tile(
    K,
    V,
    AttnMask,
    softmax_scale,
    stride_k_head: tl.constexpr,
    stride_k_seq: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_seq: tl.constexpr,
    stride_v_dim: tl.constexpr,
    mask_strides: tl.constexpr,
    MASK_BATCH: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    QUERY_LEN_ALIGNED: tl.constexpr,
    KEY_TILE_ALIGNED: tl.constexpr,
    HEAD_DIM_ALIGNED: tl.constexpr,
    query,
    key_base,
    value_base,
    kv_head,
    head_offsets,
    query_offsets,
    batch,
    query_head,
    row_max,
    row_sum,
    accumulator,
    key_start,
):
    key_offsets = key_start + tl.arange(0, BLOCK_N)
    key = tl.load(
        K
        + key_base
        + kv_head * stride_k_head
        + head_offsets[:, None] * stride_k_dim
        + key_offsets[None, :] * stride_k_seq,
        ((head_offsets[:, None] < HEAD_DIM) | HEAD_DIM_ALIGNED)
        & ((key_offsets[None, :] < KEY_LEN) | KEY_TILE_ALIGNED),
        other=0,
    )
    scores = tl.dot(query, key, input_precision="ieee") * softmax_scale
    score_is_valid = ((query_offsets[:, None] < QUERY_LEN) | QUERY_LEN_ALIGNED) & (
        (key_offsets[None, :] < KEY_LEN) | KEY_TILE_ALIGNED
    )
    if HAS_MASK:
        attn_mask = tl.load(
            AttnMask
            + batch % MASK_BATCH * mask_strides[0]
            + query_head * mask_strides[1]
            + query_offsets[:, None] * mask_strides[2]
            + key_offsets[None, :] * mask_strides[3],
            score_is_valid,
            other=0,
        )
        if MASK_IS_BOOL:
            score_is_valid = score_is_valid & attn_mask
        else:
            scores += attn_mask.to(tl.float32) * 1.4426950408889634
    scores = tl.where(score_is_valid, scores, -float("inf"))
    new_max = tl.maximum(row_max, tl.max(scores, 1))
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    probabilities = tl.exp2(scores - safe_max[:, None])
    alpha = tl.exp2(row_max - safe_max)
    row_sum = row_sum * alpha + tl.sum(probabilities, 1)
    value = tl.load(
        V
        + value_base
        + kv_head * stride_v_head
        + key_offsets[:, None] * stride_v_seq
        + head_offsets[None, :] * stride_v_dim,
        ((key_offsets[:, None] < KEY_LEN) | KEY_TILE_ALIGNED)
        & ((head_offsets[None, :] < HEAD_DIM) | HEAD_DIM_ALIGNED),
        other=0,
    )
    accumulator = accumulator * alpha[:, None]
    accumulator = tl.dot(
        probabilities.to(value.dtype), value, accumulator, input_precision="ieee"
    )
    row_max = new_max
    return (row_max, row_sum, accumulator)


@libentry()
@triton.jit
def group_query_attention_fwd_kernel(
    Q,
    K,
    V,
    AttnMask,
    Out,
    LogSumExp,
    softmax_scale,
    batch_strides_q: tl.constexpr,
    batch_strides_k: tl.constexpr,
    batch_strides_v: tl.constexpr,
    BATCH_SHAPE: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_seq: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_seq: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_v_seq: tl.constexpr,
    stride_v_dim: tl.constexpr,
    stride_o_batch: tl.constexpr,
    stride_o_head: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    mask_strides: tl.constexpr,
    MASK_BATCH: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    QUERY_LEN_ALIGNED: tl.constexpr,
    KEY_LEN_ALIGNED: tl.constexpr,
    HEAD_DIM_ALIGNED: tl.constexpr,
    PIPELINE_NS: tl.constexpr,
):
    query_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    batch_head = tl.program_id(1)
    batch, query_head = (batch_head // QUERY_HEADS, batch_head % QUERY_HEADS)
    kv_head = query_head // (QUERY_HEADS // KV_HEADS)
    batch_index = batch
    query_base = tl.full((), 0, tl.int64)
    key_base = tl.full((), 0, tl.int64)
    value_base = tl.full((), 0, tl.int64)
    for batch_axis in tl.static_range(len(BATCH_SHAPE) - 1, -1, -1):
        batch_coord = batch_index % BATCH_SHAPE[batch_axis]
        batch_index = batch_index // BATCH_SHAPE[batch_axis]
        query_base += batch_coord.to(tl.int64) * batch_strides_q[batch_axis]
        key_base += batch_coord.to(tl.int64) * batch_strides_k[batch_axis]
        value_base += batch_coord.to(tl.int64) * batch_strides_v[batch_axis]
    head_offsets = tl.arange(0, HEAD_BLOCK)
    query = tl.load(
        Q
        + query_base
        + query_head * stride_q_head
        + query_offsets[:, None] * stride_q_seq
        + head_offsets[None, :] * stride_q_dim,
        ((query_offsets[:, None] < QUERY_LEN) | QUERY_LEN_ALIGNED)
        & ((head_offsets[None, :] < HEAD_DIM) | HEAD_DIM_ALIGNED),
        other=0,
    )
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_BLOCK), tl.float32)
    key_end_index = KEY_LEN
    key_start_index = 0
    if NUM_SPLITS > 1:
        keys_per_split = tl.cdiv(KEY_LEN, NUM_SPLITS * BLOCK_N) * BLOCK_N
        key_start_index = tl.program_id(2) * keys_per_split
        key_end_index = tl.minimum(key_start_index + keys_per_split, KEY_LEN)
    # Specialize full KV tiles separately from the masked tail for small heads.
    if (
        not HAS_MASK
        and Q.dtype.element_ty != tl.float32
        and (HEAD_DIM <= 16)
        and (not KEY_LEN_ALIGNED)
        and (NUM_SPLITS == 1)
    ):
        for key_start in tl.range(
            0, KEY_LEN // BLOCK_N * BLOCK_N, BLOCK_N, num_stages=PIPELINE_NS
        ):
            row_max, row_sum, accumulator = _group_query_attention_forward_tile(
                K=K,
                V=V,
                AttnMask=AttnMask,
                softmax_scale=softmax_scale,
                stride_k_head=stride_k_head,
                stride_k_seq=stride_k_seq,
                stride_k_dim=stride_k_dim,
                stride_v_head=stride_v_head,
                stride_v_seq=stride_v_seq,
                stride_v_dim=stride_v_dim,
                mask_strides=mask_strides,
                MASK_BATCH=MASK_BATCH,
                QUERY_LEN=QUERY_LEN,
                KEY_LEN=KEY_LEN,
                HEAD_DIM=HEAD_DIM,
                BLOCK_N=BLOCK_N,
                HAS_MASK=HAS_MASK,
                MASK_IS_BOOL=MASK_IS_BOOL,
                QUERY_LEN_ALIGNED=QUERY_LEN_ALIGNED,
                KEY_TILE_ALIGNED=True,
                HEAD_DIM_ALIGNED=HEAD_DIM_ALIGNED,
                query=query,
                key_base=key_base,
                value_base=value_base,
                kv_head=kv_head,
                head_offsets=head_offsets,
                query_offsets=query_offsets,
                batch=batch,
                query_head=query_head,
                row_max=row_max,
                row_sum=row_sum,
                accumulator=accumulator,
                key_start=key_start,
            )
        for key_start in tl.range(
            KEY_LEN // BLOCK_N * BLOCK_N, KEY_LEN, BLOCK_N, num_stages=1
        ):
            row_max, row_sum, accumulator = _group_query_attention_forward_tile(
                K=K,
                V=V,
                AttnMask=AttnMask,
                softmax_scale=softmax_scale,
                stride_k_head=stride_k_head,
                stride_k_seq=stride_k_seq,
                stride_k_dim=stride_k_dim,
                stride_v_head=stride_v_head,
                stride_v_seq=stride_v_seq,
                stride_v_dim=stride_v_dim,
                mask_strides=mask_strides,
                MASK_BATCH=MASK_BATCH,
                QUERY_LEN=QUERY_LEN,
                KEY_LEN=KEY_LEN,
                HEAD_DIM=HEAD_DIM,
                BLOCK_N=BLOCK_N,
                HAS_MASK=HAS_MASK,
                MASK_IS_BOOL=MASK_IS_BOOL,
                QUERY_LEN_ALIGNED=QUERY_LEN_ALIGNED,
                KEY_TILE_ALIGNED=False,
                HEAD_DIM_ALIGNED=HEAD_DIM_ALIGNED,
                query=query,
                key_base=key_base,
                value_base=value_base,
                kv_head=kv_head,
                head_offsets=head_offsets,
                query_offsets=query_offsets,
                batch=batch,
                query_head=query_head,
                row_max=row_max,
                row_sum=row_sum,
                accumulator=accumulator,
                key_start=key_start,
            )
    else:
        for key_start in tl.range(
            key_start_index, key_end_index, BLOCK_N, num_stages=PIPELINE_NS
        ):
            key_offsets = key_start + tl.arange(0, BLOCK_N)
            key = tl.load(
                K
                + key_base
                + kv_head * stride_k_head
                + head_offsets[:, None] * stride_k_dim
                + key_offsets[None, :] * stride_k_seq,
                ((head_offsets[:, None] < HEAD_DIM) | HEAD_DIM_ALIGNED)
                & ((key_offsets[None, :] < KEY_LEN) | KEY_LEN_ALIGNED),
                other=0,
            )
            scores = tl.dot(query, key, input_precision="ieee") * softmax_scale
            score_is_valid = (
                (query_offsets[:, None] < QUERY_LEN) | QUERY_LEN_ALIGNED
            ) & ((key_offsets[None, :] < KEY_LEN) | KEY_LEN_ALIGNED)
            if HAS_MASK:
                attn_mask = tl.load(
                    AttnMask
                    + batch % MASK_BATCH * mask_strides[0]
                    + query_head * mask_strides[1]
                    + query_offsets[:, None] * mask_strides[2]
                    + key_offsets[None, :] * mask_strides[3],
                    score_is_valid,
                    other=0,
                )
                if MASK_IS_BOOL:
                    score_is_valid = score_is_valid & attn_mask
                else:
                    scores += attn_mask.to(tl.float32) * 1.4426950408889634
            scores = tl.where(score_is_valid, scores, -float("inf"))
            new_max = tl.maximum(row_max, tl.max(scores, 1))
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            probabilities = tl.exp2(scores - safe_max[:, None])
            alpha = tl.exp2(row_max - safe_max)
            row_sum = row_sum * alpha + tl.sum(probabilities, 1)
            value = tl.load(
                V
                + value_base
                + kv_head * stride_v_head
                + key_offsets[:, None] * stride_v_seq
                + head_offsets[None, :] * stride_v_dim,
                ((key_offsets[:, None] < KEY_LEN) | KEY_LEN_ALIGNED)
                & ((head_offsets[None, :] < HEAD_DIM) | HEAD_DIM_ALIGNED),
                other=0,
            )
            accumulator = accumulator * alpha[:, None]
            accumulator = tl.dot(
                probabilities.to(value.dtype),
                value,
                accumulator,
                input_precision="ieee",
            )
            row_max = new_max
    output = accumulator / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    tl.store(
        Out
        + tl.program_id(2).to(tl.int64)
        * BATCH_SIZE
        * QUERY_HEADS
        * QUERY_LEN
        * HEAD_DIM
        + batch.to(tl.int64) * stride_o_batch
        + query_head * stride_o_head
        + query_offsets[:, None] * stride_o_seq
        + head_offsets[None, :] * stride_o_dim,
        output,
        ((query_offsets[:, None] < QUERY_LEN) | QUERY_LEN_ALIGNED)
        & ((head_offsets[None, :] < HEAD_DIM) | HEAD_DIM_ALIGNED),
    )
    if NUM_SPLITS > 1:
        tl.store(
            LogSumExp
            + tl.program_id(2).to(tl.int64) * BATCH_SIZE * QUERY_HEADS * QUERY_LEN
            + (batch * QUERY_HEADS + query_head).to(tl.int64) * QUERY_LEN
            + query_offsets,
            row_max + tl.log2(row_sum),
            query_offsets < QUERY_LEN,
        )


@libentry()
@triton.jit
def group_query_attention_split_kv_merge_kernel(
    PartialOut,
    LogSumExp,
    Out,
    NUMEL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    split_offsets = tl.arange(0, NUM_SPLITS)
    row_offsets = output_offsets // HEAD_DIM
    log_normalizers = tl.load(
        LogSumExp + split_offsets[:, None] * (NUMEL // HEAD_DIM) + row_offsets[None, :],
        output_offsets[None, :] < NUMEL,
        other=-float("inf"),
    )
    max_log_normalizer = tl.max(log_normalizers, axis=0)
    safe_max = tl.where(max_log_normalizer == -float("inf"), 0.0, max_log_normalizer)
    merge_weights = tl.exp2(log_normalizers - safe_max[None, :])
    partial_outputs = tl.load(
        PartialOut + split_offsets[:, None] * NUMEL + output_offsets[None, :],
        output_offsets[None, :] < NUMEL,
        other=0.0,
    )
    normalizer = tl.sum(merge_weights, axis=0)
    output = tl.sum(partial_outputs * merge_weights, axis=0) / tl.where(
        normalizer > 0, normalizer, 1.0
    )
    tl.store(Out + output_offsets, output, output_offsets < NUMEL)


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
    kv_heads, key_len = key.shape[1:3]
    batch_shape = (batch_size,)
    qkv_strides = [
        ((tensor.stride(0),), *tensor.stride()[1:]) for tensor in (query, key, value)
    ]
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    output_strides = output.stride()
    mask_strides, mask_batch = _resolve_mask_layout(
        attn_mask, query, (batch_size, query_heads, query_len, key_len)
    )
    block_m, block_n, pipeline_ns, num_warps = _select_gpu_tiling(
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        batch_size=batch_size,
        query_heads=query_heads,
        is_fp32=query.dtype == torch.float32,
        has_mask=attn_mask is not None,
    )
    num_splits = _select_kv_splits(batch_size, query_heads, query_len, key_len, block_m)
    partial_output = output
    log_normalizers = query
    if num_splits > 1:
        partial_output = torch.empty(
            (num_splits, batch_size, query_heads, query_len, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        log_normalizers = torch.empty(
            (num_splits, batch_size, query_heads, query_len),
            dtype=torch.float32,
            device=query.device,
        )
    with torch_device_fn.device(query.device):
        group_query_attention_fwd_kernel[
            triton.cdiv(query_len, block_m), batch_size * query_heads, num_splits
        ](
            query,
            key,
            value,
            query if attn_mask is None else attn_mask,
            partial_output,
            log_normalizers,
            _resolve_softmax_scale(scale, head_dim) * 1.4426950408889634,
            batch_strides_q=qkv_strides[0][0],
            batch_strides_k=qkv_strides[1][0],
            batch_strides_v=qkv_strides[2][0],
            BATCH_SHAPE=batch_shape,
            stride_q_head=qkv_strides[0][1],
            stride_q_seq=qkv_strides[0][2],
            stride_q_dim=qkv_strides[0][3],
            stride_k_head=qkv_strides[1][1],
            stride_k_seq=qkv_strides[1][2],
            stride_k_dim=qkv_strides[1][3],
            stride_v_head=qkv_strides[2][1],
            stride_v_seq=qkv_strides[2][2],
            stride_v_dim=qkv_strides[2][3],
            stride_o_batch=output_strides[0],
            stride_o_head=output_strides[1],
            stride_o_seq=output_strides[2],
            stride_o_dim=output_strides[3],
            mask_strides=mask_strides,
            MASK_BATCH=mask_batch,
            BATCH_SIZE=batch_size,
            NUM_SPLITS=num_splits,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            QUERY_LEN=query_len,
            KEY_LEN=key_len,
            HEAD_DIM=head_dim,
            HEAD_BLOCK=max(16, triton.next_power_of_2(head_dim)),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HAS_MASK=attn_mask is not None,
            MASK_IS_BOOL=attn_mask is not None and attn_mask.dtype == torch.bool,
            QUERY_LEN_ALIGNED=query_len % block_m == 0,
            KEY_LEN_ALIGNED=key_len % block_n == 0,
            HEAD_DIM_ALIGNED=head_dim == max(16, triton.next_power_of_2(head_dim)),
            PIPELINE_NS=pipeline_ns,
            num_warps=num_warps,
            num_stages=pipeline_ns,
        )
        if num_splits > 1:
            numel = batch_size * query_heads * query_len * head_dim
            group_query_attention_split_kv_merge_kernel[triton.cdiv(numel, 256),](
                partial_output,
                log_normalizers,
                output,
                NUMEL=numel,
                HEAD_DIM=head_dim,
                NUM_SPLITS=num_splits,
                BLOCK_SIZE=256,
                num_warps=4,
            )
    return output


def group_query_attention_GQA(
    query, key, value, mask=None, scale=None, num_query_heads=None, num_kv_heads=None
):
    """Return forward GQA for Q[B,Hq,Lq,D], K/V[B,Hkv,Lk,D].

    Hq must be divisible by Hkv. Boolean True allows attention; floating masks
    are additive. Masks use standard right-aligned broadcasting to [B,Hq,Lq,Lk].
    A 3D mask is [Hq,Lq,Lk]; a batch mask uses [B,1,Lq,Lk].
    Fully masked rows return exact zero. Backward is not provided.
    """
    _validate_inputs(query, key, value, mask, scale, num_query_heads, num_kv_heads)
    return _launch_attention_forward(query, key, value, mask, scale)
