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

"""Forward-only swin transformer attention with independent Triton kernels."""

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
    window_shape = tuple(query.shape[:-2])
    if attn_mask is None:
        return (0, 0, 0, 0), 1, (0,) * len(window_shape), True
    if not isinstance(attn_mask, torch.Tensor):
        raise TypeError("mask must be a tensor")
    if attn_mask.device != query.device:
        raise ValueError("mask must be on the query device")
    if attn_mask.dtype not in (torch.bool, *_SUPPORTED_DTYPES):
        raise TypeError("mask must have bool or floating dtype")
    if attn_mask.ndim <= 4:
        # [nW,N,N] repeats across flattened batches and is shared by all heads.
        if attn_mask.ndim == 3:
            sizes = (attn_mask.shape[0], 1, *attn_mask.shape[1:])
            strides = (attn_mask.stride(0), 0, *attn_mask.stride()[1:])
        else:
            padding = 4 - attn_mask.ndim
            sizes = (1,) * padding + tuple(attn_mask.shape)
            strides = (0,) * padding + tuple(attn_mask.stride())
        for axis, (size, target) in enumerate(zip(sizes, shape)):
            if axis == 0 and size > 0 and target % size == 0:
                continue
            if size not in (1, target):
                raise ValueError("mask is not broadcastable to window attention scores")
        strides = tuple(
            0 if size == 1 else stride for size, stride in zip(sizes, strides)
        )
        return strides, sizes[0], (0,) * len(window_shape), True
    target = (*window_shape, *shape[1:])
    if attn_mask.ndim > len(target):
        raise ValueError("mask has more dimensions than window attention scores")
    padding = len(target) - attn_mask.ndim
    sizes = (1,) * padding + tuple(attn_mask.shape)
    strides = (0,) * padding + tuple(attn_mask.stride())
    for size, expected in zip(sizes, target):
        if size not in (1, expected):
            raise ValueError("mask is not broadcastable to window attention scores")
    strides = tuple(0 if size == 1 else stride for size, stride in zip(sizes, strides))
    return (0, *strides[-3:]), 1, strides[:-3], False


def _validate_inputs(
    query,
    key,
    value,
    mask=None,
    num_heads=None,
    window_size=None,
    relative_position_bias=None,
):
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a tensor")
    if query.ndim not in (3, 4):
        raise ValueError("window inputs require three or four dimensions")
    _validate_qkv_tensors(query, key, value, query.ndim)
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("window Q/K/V shapes must match")
    if isinstance(num_heads, bool) or not isinstance(num_heads, numbers.Integral):
        raise TypeError("num_heads must be a positive integer")
    num_tokens, channels = query.shape[-2:]
    if num_heads <= 0 or channels % num_heads:
        raise ValueError("channels must be divisible by num_heads")
    if window_size is not None:
        sizes = (
            (window_size, window_size)
            if isinstance(window_size, numbers.Integral)
            else window_size
        )
        if isinstance(sizes, (tuple, list)) and len(sizes) == 1:
            sizes = (sizes[0], sizes[0])
        if not isinstance(sizes, (tuple, list)) or len(sizes) != 2:
            raise ValueError("window_size must be an integer or a pair")
        if any(
            (
                isinstance(window_dim, bool)
                or not isinstance(window_dim, numbers.Integral)
                or window_dim <= 0
                for window_dim in sizes
            )
        ):
            raise ValueError("window dimensions must be positive integers")
        if math.prod(sizes) != num_tokens:
            raise ValueError("window area must equal the number of tokens")
    if relative_position_bias is not None:
        if not isinstance(relative_position_bias, torch.Tensor):
            raise TypeError("relative_position_bias must be a tensor")
        if relative_position_bias.device != query.device:
            raise ValueError("relative bias must be on the query device")
        if relative_position_bias.dtype not in _SUPPORTED_DTYPES:
            raise TypeError("relative bias must have floating dtype")
        if tuple(relative_position_bias.shape) != (num_heads, num_tokens, num_tokens):
            raise ValueError("relative bias must have shape [heads,tokens,tokens]")


@libentry()
@triton.jit
def swin_transformer_attention_fwd_kernel(
    Q,
    K,
    V,
    AttnMask,
    RelativePositionBias,
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
    mask_batch_strides: tl.constexpr,
    MASK_BATCH_IS_FLAT: tl.constexpr,
    relative_bias_strides: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    HAS_RELATIVE_BIAS: tl.constexpr,
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
    mask_base = tl.full((), 0, tl.int64)
    for batch_axis in tl.static_range(len(BATCH_SHAPE) - 1, -1, -1):
        batch_coord = batch_index % BATCH_SHAPE[batch_axis]
        batch_index = batch_index // BATCH_SHAPE[batch_axis]
        query_base += batch_coord.to(tl.int64) * batch_strides_q[batch_axis]
        key_base += batch_coord.to(tl.int64) * batch_strides_k[batch_axis]
        value_base += batch_coord.to(tl.int64) * batch_strides_v[batch_axis]
        if HAS_MASK and not MASK_BATCH_IS_FLAT:
            mask_base += batch_coord.to(tl.int64) * mask_batch_strides[batch_axis]
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
                + mask_base
                + batch % MASK_BATCH * mask_strides[0]
                + query_head * mask_strides[1]
                + query_offsets[:, None] * mask_strides[2]
                + key_offsets[None, :] * mask_strides[3],
                score_is_valid,
                other=0,
            )
            if MASK_IS_BOOL:
                score_is_valid = score_is_valid & ~attn_mask
            else:
                scores += attn_mask.to(tl.float32) * 1.4426950408889634
        if HAS_RELATIVE_BIAS:
            relative_bias = tl.load(
                RelativePositionBias
                + query_head * relative_bias_strides[0]
                + query_offsets[:, None] * relative_bias_strides[1]
                + key_offsets[None, :] * relative_bias_strides[2],
                score_is_valid,
                other=0,
            )
            scores += relative_bias.to(tl.float32) * 1.4426950408889634
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
    block_m = 32 if is_fp32 or query_len < 64 else 64
    block_n = 32 if is_fp32 or head_dim > 64 or has_mask else 64
    return (block_m, block_n, 1 if is_fp32 else 2, 4)


def _launch_attention_forward(
    query,
    key,
    value,
    attn_mask=None,
    *,
    num_window_heads=None,
    relative_position_bias=None,
):
    batch_shape = tuple(query.shape[:-2])
    batch_size = math.prod(batch_shape)
    query_len, channels = query.shape[-2:]
    query_heads = num_window_heads
    key_len, head_dim = (query_len, channels // query_heads)
    qkv_strides = [
        (
            tuple(tensor.stride()[:-2]),
            head_dim * tensor.stride(-1),
            tensor.stride(-2),
            tensor.stride(-1),
        )
        for tensor in (query, key, value)
    ]
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    output_strides = (query_len * channels, head_dim, channels, 1)
    mask_strides, mask_batch, mask_batch_strides, mask_batch_is_flat = (
        _resolve_mask_layout(
            attn_mask, query, (batch_size, query_heads, query_len, key_len)
        )
    )
    relative_bias_strides = (
        (0, 0, 0) if relative_position_bias is None else relative_position_bias.stride()
    )
    block_m, block_n, pipeline_ns, num_warps = _select_gpu_tiling(
        query_len=query_len,
        head_dim=head_dim,
        is_fp32=query.dtype == torch.float32,
        has_mask=attn_mask is not None,
    )
    partial_output = output
    with torch_device_fn.device(query.device):
        swin_transformer_attention_fwd_kernel[
            triton.cdiv(query_len, block_m), batch_size * query_heads, 1
        ](
            query,
            key,
            value,
            query if attn_mask is None else attn_mask,
            query if relative_position_bias is None else relative_position_bias,
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
            mask_batch_strides=mask_batch_strides,
            MASK_BATCH_IS_FLAT=mask_batch_is_flat,
            relative_bias_strides=relative_bias_strides,
            QUERY_HEADS=query_heads,
            QUERY_LEN=query_len,
            KEY_LEN=key_len,
            HEAD_DIM=head_dim,
            HEAD_BLOCK=max(16, triton.next_power_of_2(head_dim)),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HAS_MASK=attn_mask is not None,
            MASK_IS_BOOL=attn_mask is not None and attn_mask.dtype == torch.bool,
            HAS_RELATIVE_BIAS=relative_position_bias is not None,
            QUERY_LEN_ALIGNED=query_len % block_m == 0,
            KEY_LEN_ALIGNED=key_len % block_n == 0,
            HEAD_DIM_ALIGNED=head_dim == max(16, triton.next_power_of_2(head_dim)),
            PIPELINE_NS=pipeline_ns,
            num_warps=num_warps,
            num_stages=pipeline_ns,
        )
    return output


def swin_transformer_attention(
    query,
    key,
    value,
    mask=None,
    num_heads=None,
    window_size=None,
    relative_position_bias=None,
):
    """Forward attention on [windows,tokens,channels] or [B,nW,tokens,channels].

    Boolean True blocks attention; floating masks are additive. A mask's window
    dimension may divide the flattened window count (shared across images).
    Relative bias has shape [heads,tokens,tokens]. A scalar or one-element window_size denotes
    a square window; a pair specifies (height,width). Backward is not provided.
    """
    _validate_inputs(
        query, key, value, mask, num_heads, window_size, relative_position_bias
    )
    return _launch_attention_forward(
        query,
        key,
        value,
        mask,
        num_window_heads=num_heads,
        relative_position_bias=relative_position_bias,
    )
