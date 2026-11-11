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

"""Forward-only adaptive attention span with independent Triton kernels."""

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
        return ((0, 0, 0, 0), 1)
    if not isinstance(attn_mask, torch.Tensor):
        raise TypeError("mask must be a tensor")
    if attn_mask.device != query.device:
        raise ValueError("mask must be on the query device")
    if attn_mask.dtype not in (torch.bool, *_SUPPORTED_DTYPES):
        raise TypeError("mask must have bool or floating dtype")
    if attn_mask.ndim == 2:
        sizes, strides = ((1, 1, *attn_mask.shape), (0, 0, *attn_mask.stride()))
    elif attn_mask.ndim == 3:
        sizes = (attn_mask.shape[0], 1, *attn_mask.shape[1:])
        strides = (attn_mask.stride(0), 0, *attn_mask.stride()[1:])
    elif attn_mask.ndim == 4:
        sizes, strides = (tuple(attn_mask.shape), attn_mask.stride())
    else:
        raise ValueError("mask must have 2, 3 (batch), or 4 dimensions")
    for axis, (size, target) in enumerate(zip(sizes, shape)):
        if size not in (1, target):
            raise ValueError("mask is not broadcastable to attention scores")
    return (
        tuple((0 if size == 1 else stride for size, stride in zip(sizes, strides))),
        sizes[0],
    )


def _validate_inputs(query, key, value, span, max_span, ramp_length, mask=None):
    _validate_qkv_tensors(query, key, value)
    batch_size, num_heads, query_len, head_dim = query.shape
    if (
        key.shape != value.shape
        or key.shape[0] != batch_size
        or key.shape[3] != head_dim
    ):
        raise ValueError(
            "K/V shapes must match, and Q/K/V batch and head dimensions must agree"
        )
    _resolve_mask_layout(mask, query, (batch_size, num_heads, query_len, key.shape[2]))
    if key.shape[1] != num_heads:
        raise ValueError("adaptive span requires equal Q/K/V head counts")
    if not isinstance(span, torch.Tensor):
        raise TypeError("span must be a tensor")
    if tuple(span.shape) not in ((num_heads,), (batch_size, num_heads)):
        raise ValueError("span must have shape [H] or [B,H]")
    if span.device != query.device:
        raise ValueError("span must be on the query device")
    if span.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("span must have floating dtype")
    if isinstance(max_span, bool) or not isinstance(max_span, numbers.Integral):
        raise TypeError("max_span must be a nonnegative integer")
    if max_span < 0:
        raise ValueError("max_span must be nonnegative")
    if isinstance(ramp_length, bool) or not isinstance(ramp_length, numbers.Real):
        raise TypeError("ramp_length must be a positive finite real number")
    if not math.isfinite(ramp_length) or ramp_length <= 0:
        raise ValueError("ramp_length must be positive and finite")


@libentry()
@triton.jit
def adaptive_attention_span_fwd_kernel(
    Q,
    K,
    V,
    AttnMask,
    AttentionSpan,
    Out,
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
    span_strides: tl.constexpr,
    MAX_SPAN: tl.constexpr,
    RAMP_LENGTH: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
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
    kv_head = query_head
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
    head_span = tl.load(
        AttentionSpan + batch * span_strides[0] + query_head * span_strides[1]
    ).to(tl.float32)
    head_span = tl.minimum(tl.maximum(head_span, 0.0), MAX_SPAN)
    key_end_index = tl.minimum((tl.program_id(0) + 1) * BLOCK_M, KEY_LEN)
    # Round outward so fractional spans retain the entire nonzero ramp.
    active_span = tl.minimum(
        tl.minimum(tl.ceil(head_span + RAMP_LENGTH), MAX_SPAN), QUERY_LEN
    ).to(tl.int32)
    key_start_index = tl.maximum(
        0, (tl.program_id(0) * BLOCK_M - active_span) // BLOCK_N * BLOCK_N
    )
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
        # Online softmax uses log2 scores; softmax_scale includes log2(e).
        scores = tl.dot(query, key, input_precision="ieee") * softmax_scale
        score_is_valid = ((query_offsets[:, None] < QUERY_LEN) | QUERY_LEN_ALIGNED) & (
            (key_offsets[None, :] < KEY_LEN) | KEY_LEN_ALIGNED
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
        distance = query_offsets[:, None] - key_offsets[None, :]
        weight = tl.minimum(
            tl.maximum((head_span + RAMP_LENGTH - distance) / RAMP_LENGTH, 0.0), 1.0
        )
        score_is_valid = (
            score_is_valid & (distance >= 0) & (distance <= MAX_SPAN) & (weight > 0)
        )
        scores += tl.log2(tl.where(weight > 0, weight, 1.0))
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
            probabilities.to(value.dtype), value, accumulator, input_precision="ieee"
        )
        row_max = new_max
    output = accumulator / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    tl.store(
        Out
        + batch.to(tl.int64) * stride_o_batch
        + query_head * stride_o_head
        + query_offsets[:, None] * stride_o_seq
        + head_offsets[None, :] * stride_o_dim,
        output,
        ((query_offsets[:, None] < QUERY_LEN) | QUERY_LEN_ALIGNED)
        & ((head_offsets[None, :] < HEAD_DIM) | HEAD_DIM_ALIGNED),
    )


def _select_gpu_tiling(query_len, head_dim, is_fp32, has_mask):
    """Portable launch policy; changing tiles never relaxes FP32 precision."""
    if is_fp32 and head_dim >= 64 and (query_len >= 512):
        return (64, 64, 1, 4)
    block_m = 32 if is_fp32 or query_len < 64 else 64
    block_n = 32 if is_fp32 or head_dim > 64 or has_mask else 64
    return (block_m, block_n, 1 if is_fp32 else 2, 4)


def _launch_attention_forward(
    query, key, value, attn_mask=None, *, span=None, max_span=0, ramp_length=1.0
):
    batch_size, query_heads, query_len, head_dim = query.shape
    key_len = key.shape[2]
    batch_shape = (batch_size,)
    qkv_strides = [
        ((tensor.stride(0),), *tensor.stride()[1:]) for tensor in (query, key, value)
    ]
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    output_strides = output.stride()
    mask_strides, mask_batch = _resolve_mask_layout(
        attn_mask, query, (batch_size, query_heads, query_len, key_len)
    )
    span_strides = (
        (0, 0)
        if span is None
        else (0, span.stride(0)) if span.ndim == 1 else span.stride()
    )
    block_m, block_n, pipeline_ns, num_warps = _select_gpu_tiling(
        query_len=query_len,
        head_dim=head_dim,
        is_fp32=query.dtype == torch.float32,
        has_mask=attn_mask is not None,
    )
    partial_output = output
    with torch_device_fn.device(query.device):
        adaptive_attention_span_fwd_kernel[
            triton.cdiv(query_len, block_m), batch_size * query_heads, 1
        ](
            query,
            key,
            value,
            query if attn_mask is None else attn_mask,
            query if span is None else span,
            partial_output,
            head_dim**-0.5 * 1.4426950408889634,
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
            span_strides=span_strides,
            MAX_SPAN=max_span,
            RAMP_LENGTH=ramp_length,
            QUERY_HEADS=query_heads,
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
    return output


def adaptive_attention_span(query, key, value, span, max_span, ramp_length, mask=None):
    """Forward adaptive-span attention, with upper-left causal alignment.

    Q[B,H,Lq,D], K/V[B,H,Lk,D]; span[H] or span[B,H] contains finite distances.
    Span is clamped to [0,max_span]. At distance x=i-j, the multiplicative
    weight is clamp((span+ramp_length-x)/ramp_length,0,1); positions outside
    0 <= x <= max_span are blocked. Zero weights are exactly excluded.
    Boolean masks use True=allowed; float masks are additive. No backward.
    """
    _validate_inputs(query, key, value, span, max_span, ramp_length, mask)
    return _launch_attention_forward(
        query,
        key,
        value,
        mask,
        span=span,
        max_span=int(max_span),
        ramp_length=float(ramp_length),
    )
