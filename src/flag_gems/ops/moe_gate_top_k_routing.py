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

"""Independent forward MoE gate, deterministic dispatch, experts, and combine."""

import math
import numbers

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_ACTIVATIONS = ("identity", "linear", "relu", "gelu", "silu", "swiglu", "geglu")
erf = tl_extra_shim.erf


def _validate_inputs(
    hidden_states,
    gate_weight,
    num_experts,
    top_k,
    capacity_factor,
    expert_up_weight,
    expert_down_weight,
    expert_gate_weight,
    activation,
):
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a tensor")
    if hidden_states.ndim not in (2, 3) or hidden_states.shape[-1] <= 0:
        raise ValueError("hidden_states must have shape [T,C] or [B,S,C], with C>0")
    if hidden_states.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("hidden_states must have FP16, BF16 or FP32 dtype")
    if hidden_states.device.type == "cpu":
        raise ValueError("MoE routing requires an accelerator device")
    if not isinstance(gate_weight, torch.Tensor):
        raise TypeError("gate_weight must be a tensor")
    if gate_weight.ndim != 2 or gate_weight.shape[0] != hidden_states.shape[-1]:
        raise ValueError("gate_weight must have shape [C,E]")
    if gate_weight.device != hidden_states.device:
        raise ValueError("gate_weight must be on the hidden_states device")
    if gate_weight.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("gate_weight must have FP16, BF16 or FP32 dtype")
    if num_experts is None:
        num_experts = gate_weight.shape[1]
    if isinstance(num_experts, bool) or not isinstance(num_experts, numbers.Integral):
        raise TypeError("num_experts must be a positive integer")
    if num_experts <= 0 or gate_weight.shape[1] != num_experts:
        raise ValueError("num_experts must be positive and match gate_weight")
    if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral):
        raise TypeError("top_k must be an integer")
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must be between one and num_experts")
    if capacity_factor is not None:
        if isinstance(capacity_factor, bool) or not isinstance(
            capacity_factor, numbers.Real
        ):
            raise TypeError(
                "capacity_factor must be a positive finite real number or None"
            )
        if not math.isfinite(capacity_factor) or capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive and finite")
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {_ACTIVATIONS}")
    if activation == "identity":
        if any(
            weight is not None
            for weight in (expert_up_weight, expert_down_weight, expert_gate_weight)
        ):
            raise ValueError("identity experts do not take expert weights")
        return int(num_experts), hidden_states.shape[-1]
    channels = hidden_states.shape[-1]
    if not isinstance(expert_up_weight, torch.Tensor) or expert_up_weight.ndim != 3:
        raise ValueError("expert_up_weight must have shape [E,C,F]")
    intermediate_size = expert_up_weight.shape[2]
    if (
        tuple(expert_up_weight.shape[:2]) != (num_experts, channels)
        or intermediate_size <= 0
    ):
        raise ValueError("expert_up_weight must have shape [E,C,F], with F>0")
    if activation == "linear":
        if (
            intermediate_size != channels
            or expert_down_weight is not None
            or expert_gate_weight is not None
        ):
            raise ValueError(
                "linear experts require [E,C,C] up weights and no down/gate weights"
            )
    elif not isinstance(expert_down_weight, torch.Tensor) or tuple(
        expert_down_weight.shape
    ) != (num_experts, intermediate_size, channels):
        raise ValueError("expert_down_weight must have shape [E,F,C]")
    if activation in ("swiglu", "geglu"):
        if (
            not isinstance(expert_gate_weight, torch.Tensor)
            or expert_gate_weight.shape != expert_up_weight.shape
        ):
            raise ValueError(
                "gated experts require expert_gate_weight with shape [E,C,F]"
            )
    elif expert_gate_weight is not None:
        raise ValueError("expert_gate_weight is only used by swiglu/geglu")
    for weight in (expert_up_weight, expert_down_weight, expert_gate_weight):
        if weight is not None:
            if weight.device != hidden_states.device:
                raise ValueError(
                    "all expert weights must be on the hidden_states device"
                )
            if weight.dtype != hidden_states.dtype:
                raise TypeError("expert weights must share the hidden_states dtype")
    return int(num_experts), intermediate_size


@libentry()
@triton.jit
def moe_gate_projection_kernel(
    Hidden,
    GateWeight,
    RouterLogits,
    stride_hidden_batch: tl.constexpr,
    stride_hidden_seq: tl.constexpr,
    stride_hidden_dim: tl.constexpr,
    stride_gate_dim: tl.constexpr,
    stride_gate_expert: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    SEQUENCE_LEN: tl.constexpr,
    CHANNELS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    expert_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    channel_offsets = tl.arange(0, BLOCK_K)
    token_base = (
        token_offsets // SEQUENCE_LEN * stride_hidden_batch
        + token_offsets % SEQUENCE_LEN * stride_hidden_seq
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for channel_start in range(tl.cdiv(CHANNELS, BLOCK_K)):
        channels = channel_start * BLOCK_K + channel_offsets
        hidden = tl.load(
            Hidden + token_base[:, None] + channels[None, :] * stride_hidden_dim,
            (token_offsets[:, None] < NUM_TOKENS) & (channels[None, :] < CHANNELS),
            other=0,
        )
        weight = tl.load(
            GateWeight
            + channels[:, None] * stride_gate_dim
            + expert_offsets[None, :] * stride_gate_expert,
            (channels[:, None] < CHANNELS) & (expert_offsets[None, :] < NUM_EXPERTS),
            other=0,
        )
        if Hidden.dtype.element_ty != GateWeight.dtype.element_ty:
            hidden = hidden.to(tl.float32)
            weight = weight.to(tl.float32)
        accumulator = tl.dot(hidden, weight, accumulator, input_precision="ieee")
    tl.store(
        RouterLogits + token_offsets[:, None] * NUM_EXPERTS + expert_offsets[None, :],
        accumulator,
        (token_offsets[:, None] < NUM_TOKENS) & (expert_offsets[None, :] < NUM_EXPERTS),
    )


@libentry()
@triton.jit
def moe_gate_select_kernel(
    RouterLogits,
    ExpertIndices,
    NUM_TOKENS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    experts = tl.arange(0, EXPERT_BLOCK)
    logits = tl.load(
        RouterLogits + tokens[:, None] * NUM_EXPERTS + experts[None, :],
        (tokens[:, None] < NUM_TOKENS) & (experts[None, :] < NUM_EXPERTS),
        other=-float("inf"),
    ).to(tl.float32)
    # Choose the lowest expert ID on ties, independently of backend sort behavior.
    ranking = tl.where(logits == logits, logits, float("inf"))
    available = tl.full((BLOCK_TOKENS, EXPERT_BLOCK), True, tl.int1) & (
        experts[None, :] < NUM_EXPERTS
    )
    for rank in range(TOP_K):
        maximum = tl.max(tl.where(available, ranking, -float("inf")), axis=1)
        selected = tl.min(
            tl.where(
                available & (ranking == maximum[:, None]), experts[None, :], 2147483647
            ),
            axis=1,
        )
        tl.store(
            ExpertIndices + tokens * TOP_K + rank,
            selected.to(tl.int64),
            tokens < NUM_TOKENS,
        )
        available = available & (experts[None, :] != selected[:, None])


@libentry()
@triton.jit
def moe_gate_layout_kernel(
    ExpertIndices,
    RouteKept,
    SlotRoutes,
    ExpertCounts,
    NUM_TOKENS: tl.constexpr,
    TOP_K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    expert = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_ROUTES)
    for slot_start in range(tl.cdiv(CAPACITY, BLOCK_ROUTES)):
        slots = slot_start * BLOCK_ROUTES + offsets
        tl.store(SlotRoutes + expert * CAPACITY + slots, -1, slots < CAPACITY)
    assigned = tl.full((), 0, tl.int32)
    for route_start in range(tl.cdiv(NUM_TOKENS * TOP_K, BLOCK_ROUTES)):
        routes = route_start * BLOCK_ROUTES + offsets
        indices = tl.load(ExpertIndices + routes, routes < NUM_TOKENS * TOP_K, other=-1)
        matches = indices == expert
        locations = assigned + tl.cumsum(matches.to(tl.int32), axis=0) - 1
        kept = matches & (locations < CAPACITY)
        tl.store(RouteKept + routes, kept, matches & (routes < NUM_TOKENS * TOP_K))
        tl.store(SlotRoutes + expert * CAPACITY + locations, routes, kept)
        assigned += tl.sum(matches.to(tl.int32), axis=0)
    tl.store(ExpertCounts + expert, tl.minimum(assigned, CAPACITY))


@libentry()
@triton.jit
def moe_expert_up_kernel(
    Hidden,
    UpWeight,
    GateWeight,
    SlotRoutes,
    ExpertCounts,
    UpOutput,
    stride_hidden_batch: tl.constexpr,
    stride_hidden_seq: tl.constexpr,
    stride_hidden_dim: tl.constexpr,
    up_strides: tl.constexpr,
    gate_strides: tl.constexpr,
    SEQUENCE_LEN: tl.constexpr,
    CHANNELS: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    CAPACITY: tl.constexpr,
    ACTIVATION: tl.constexpr,
    ROW_PARTITIONS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    feature_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    expert = tl.program_id(1)
    row_partition = tl.program_id(2)
    count = tl.load(ExpertCounts + expert)
    channel_offsets = tl.arange(0, BLOCK_K)
    for row_start in range(row_partition * BLOCK_M, count, BLOCK_M * ROW_PARTITIONS):
        rows = row_start + tl.arange(0, BLOCK_M)
        routes = tl.load(SlotRoutes + expert * CAPACITY + rows, rows < count, other=-1)
        tokens = routes // TOP_K
        token_base = (
            tokens // SEQUENCE_LEN * stride_hidden_batch
            + tokens % SEQUENCE_LEN * stride_hidden_seq
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        gate_accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for channel_start in range(tl.cdiv(CHANNELS, BLOCK_K)):
            channels = channel_start * BLOCK_K + channel_offsets
            hidden = tl.load(
                Hidden + token_base[:, None] + channels[None, :] * stride_hidden_dim,
                (routes[:, None] >= 0) & (channels[None, :] < CHANNELS),
                other=0,
            )
            weight = tl.load(
                UpWeight
                + expert * up_strides[0]
                + channels[:, None] * up_strides[1]
                + feature_offsets[None, :] * up_strides[2],
                (channels[:, None] < CHANNELS)
                & (feature_offsets[None, :] < INTERMEDIATE_SIZE),
                other=0,
            )
            accumulator = tl.dot(hidden, weight, accumulator, input_precision="ieee")
            if ACTIVATION == "swiglu" or ACTIVATION == "geglu":
                gate_weight = tl.load(
                    GateWeight
                    + expert * gate_strides[0]
                    + channels[:, None] * gate_strides[1]
                    + feature_offsets[None, :] * gate_strides[2],
                    (channels[:, None] < CHANNELS)
                    & (feature_offsets[None, :] < INTERMEDIATE_SIZE),
                    other=0,
                )
                gate_accumulator = tl.dot(
                    hidden, gate_weight, gate_accumulator, input_precision="ieee"
                )
        projection = accumulator.to(Hidden.dtype.element_ty).to(tl.float32)
        if ACTIVATION == "swiglu" or ACTIVATION == "geglu":
            activation_input = gate_accumulator.to(Hidden.dtype.element_ty).to(
                tl.float32
            )
        else:
            activation_input = projection
        if ACTIVATION == "relu":
            activated = tl.maximum(activation_input, 0.0)
        elif ACTIVATION == "gelu" or ACTIVATION == "geglu":
            activated = (
                0.5
                * activation_input
                * (1.0 + erf(activation_input * 0.7071067811865476))
            )
        elif ACTIVATION == "silu" or ACTIVATION == "swiglu":
            activated = activation_input / (1.0 + tl.exp(-activation_input))
        else:
            activated = activation_input
        activated = activated.to(Hidden.dtype.element_ty).to(tl.float32)
        if ACTIVATION == "swiglu" or ACTIVATION == "geglu":
            activated *= projection
        tl.store(
            UpOutput + routes[:, None] * INTERMEDIATE_SIZE + feature_offsets[None, :],
            activated,
            (routes[:, None] >= 0) & (feature_offsets[None, :] < INTERMEDIATE_SIZE),
        )


@libentry()
@triton.jit
def moe_expert_down_kernel(
    UpOutput,
    DownWeight,
    SlotRoutes,
    ExpertCounts,
    PartialOutput,
    down_strides: tl.constexpr,
    CHANNELS: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROW_PARTITIONS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    expert = tl.program_id(1)
    row_partition = tl.program_id(2)
    count = tl.load(ExpertCounts + expert)
    feature_offsets = tl.arange(0, BLOCK_K)
    for row_start in range(row_partition * BLOCK_M, count, BLOCK_M * ROW_PARTITIONS):
        rows = row_start + tl.arange(0, BLOCK_M)
        routes = tl.load(SlotRoutes + expert * CAPACITY + rows, rows < count, other=-1)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for feature_start in range(tl.cdiv(INTERMEDIATE_SIZE, BLOCK_K)):
            features = feature_start * BLOCK_K + feature_offsets
            hidden = tl.load(
                UpOutput + routes[:, None] * INTERMEDIATE_SIZE + features[None, :],
                (routes[:, None] >= 0) & (features[None, :] < INTERMEDIATE_SIZE),
                other=0,
            )
            weight = tl.load(
                DownWeight
                + expert * down_strides[0]
                + features[:, None] * down_strides[1]
                + channel_offsets[None, :] * down_strides[2],
                (features[:, None] < INTERMEDIATE_SIZE)
                & (channel_offsets[None, :] < CHANNELS),
                other=0,
            )
            accumulator = tl.dot(hidden, weight, accumulator, input_precision="ieee")
        tl.store(
            PartialOutput + routes[:, None] * CHANNELS + channel_offsets[None, :],
            accumulator,
            (routes[:, None] >= 0) & (channel_offsets[None, :] < CHANNELS),
        )


@libentry()
@triton.jit
def moe_gate_combine_kernel(
    RouterLogits,
    ExpertIndices,
    RouteKept,
    PartialOutput,
    Hidden,
    ExpertWeights,
    Output,
    stride_hidden_batch: tl.constexpr,
    stride_hidden_seq: tl.constexpr,
    stride_hidden_dim: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    SEQUENCE_LEN: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    CHANNELS: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_BLOCK: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    IDENTITY_EXPERTS: tl.constexpr,
):
    token = tl.program_id(0)
    channel_tile = tl.program_id(1)
    ranks = tl.arange(0, TOP_K_BLOCK)
    routes = token * TOP_K + ranks
    indices = tl.load(ExpertIndices + routes, ranks < TOP_K, other=0)
    kept = tl.load(RouteKept + routes, ranks < TOP_K, other=False)
    logits = tl.load(
        RouterLogits + token * NUM_EXPERTS + indices, ranks < TOP_K, other=-float("inf")
    ).to(tl.float32)
    selected_logits = tl.where(kept & (ranks < TOP_K), logits, -float("inf"))
    maximum = tl.max(selected_logits, axis=0)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    unnormalized = tl.exp(selected_logits - safe_max)
    denominator = tl.sum(unnormalized, axis=0)
    weights = unnormalized / tl.where(denominator > 0, denominator, 1.0)
    # Combine using exactly the returned (possibly low-precision) weights.
    weights = weights.to(ExpertWeights.dtype.element_ty).to(tl.float32)
    if channel_tile == 0:
        tl.store(ExpertWeights + routes, weights, ranks < TOP_K)
    channels = channel_tile * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    if IDENTITY_EXPERTS:
        token_base = (
            token // SEQUENCE_LEN * stride_hidden_batch
            + token % SEQUENCE_LEN * stride_hidden_seq
        )
        values = tl.load(
            Hidden + token_base + channels * stride_hidden_dim,
            channels < CHANNELS,
            other=0,
        ).to(tl.float32)
        output = values * tl.sum(weights, axis=0)
    else:
        values = tl.load(
            PartialOutput + routes[:, None] * CHANNELS + channels[None, :],
            kept[:, None] & (ranks[:, None] < TOP_K) & (channels[None, :] < CHANNELS),
            other=0,
        ).to(tl.float32)
        output = tl.sum(values * weights[:, None], axis=0)
    tl.store(Output + token * CHANNELS + channels, output, channels < CHANNELS)


def _expert_row_partitions(capacity, num_experts, output_tiles, block_m):
    return min(
        triton.cdiv(capacity, block_m),
        max(1, triton.cdiv(128, num_experts * output_tiles)),
    )


def moe_gate_top_k_routing(
    hidden_states,
    gate_weight,
    num_experts=None,
    top_k=1,
    capacity_factor=1.0,
    *,
    expert_up_weight=None,
    expert_down_weight=None,
    expert_gate_weight=None,
    activation="relu",
):
    """Return (router_logits, expert_weights, expert_indices, output), forward only.

    Input is [T,C] or [B,S,C]; gate_weight is [C,E]. All floating outputs use
    hidden_states.dtype. Logits/weights/indices have the same leading dimensions
    as hidden_states, ending in E/K/K; output has exactly hidden_states.shape.
    Indices are int64 in [0,E), sorted by descending logit, then lower expert ID.

    relu/gelu/silu experts use up[E,C,F], down[E,F,C]. swiglu/geglu also take
    gate[E,C,F] and compute activation(x@gate)*(x@up), followed by down.
    linear uses up[E,C,C] only. Explicit identity takes no expert weights.
    Projections/activation intermediates are rounded to hidden_states.dtype.

    Capacity is ceil(capacity_factor*T*K/E), with no additional minimum.
    None disables dropping (capacity=T). Each expert keeps routes in token
    order. Selected IDs remain valid even if a route is dropped; its weight
    becomes zero. Weights are a stable softmax over the surviving selected
    logits. Tokens with no surviving routes produce zero weights and output.
    Expert biases, auxiliary training losses and distributed all-to-all are
    outside this forward operator. No random gate or expert weights are created.
    """
    num_experts, intermediate_size = _validate_inputs(
        hidden_states,
        gate_weight,
        num_experts,
        top_k,
        capacity_factor,
        expert_up_weight,
        expert_down_weight,
        expert_gate_weight,
        activation,
    )
    channels = hidden_states.shape[-1]
    leading_shape = tuple(hidden_states.shape[:-1])
    num_tokens = math.prod(leading_shape)
    if capacity_factor is None:
        capacity = num_tokens
    else:
        capacity_value = float(capacity_factor) * num_tokens * int(top_k) / num_experts
        if not math.isfinite(capacity_value):
            raise ValueError("capacity is too large")
        capacity = (
            min(num_tokens, max(1, math.ceil(capacity_value))) if num_tokens else 0
        )
    if num_tokens * int(top_k) > 2147483647:
        raise ValueError("the number of token/expert routes must fit in int32")
    options = dict(device=hidden_states.device, dtype=hidden_states.dtype)
    router_logits = torch.empty((*leading_shape, num_experts), **options)
    expert_weights = torch.empty((*leading_shape, int(top_k)), **options)
    expert_indices = torch.empty(
        (*leading_shape, int(top_k)), device=hidden_states.device, dtype=torch.int64
    )
    output = torch.empty(hidden_states.shape, **options)
    if num_tokens == 0:
        return router_logits, expert_weights, expert_indices, output
    route_kept = torch.empty(
        (num_tokens, int(top_k)), device=hidden_states.device, dtype=torch.bool
    )
    slot_routes = torch.empty(
        (num_experts, capacity), device=hidden_states.device, dtype=torch.int32
    )
    expert_counts = torch.empty(
        (num_experts,), device=hidden_states.device, dtype=torch.int32
    )
    sequence_len = hidden_states.shape[-2]
    hidden_strides = (
        (0, *hidden_states.stride())
        if hidden_states.ndim == 2
        else hidden_states.stride()
    )
    block_m, block_n, block_k = 16, 32, 32
    partial_output = hidden_states
    with torch_device_fn.device(hidden_states.device):
        moe_gate_projection_kernel[
            (triton.cdiv(num_tokens, block_m), triton.cdiv(num_experts, block_n))
        ](
            hidden_states,
            gate_weight,
            router_logits,
            *hidden_strides,
            *gate_weight.stride(),
            NUM_TOKENS=num_tokens,
            SEQUENCE_LEN=sequence_len,
            CHANNELS=channels,
            NUM_EXPERTS=num_experts,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )
        block_tokens = 4 if num_experts <= 128 else 1
        moe_gate_select_kernel[(triton.cdiv(num_tokens, block_tokens),)](
            router_logits,
            expert_indices,
            NUM_TOKENS=num_tokens,
            NUM_EXPERTS=num_experts,
            TOP_K=int(top_k),
            BLOCK_TOKENS=block_tokens,
            EXPERT_BLOCK=triton.next_power_of_2(num_experts),
            num_warps=4,
        )
        moe_gate_layout_kernel[(num_experts,)](
            expert_indices,
            route_kept,
            slot_routes,
            expert_counts,
            NUM_TOKENS=num_tokens,
            TOP_K=int(top_k),
            CAPACITY=capacity,
            BLOCK_ROUTES=256,
            num_warps=4,
        )
        if activation != "identity":
            up_output = torch.empty(
                (num_tokens * int(top_k), intermediate_size), **options
            )
            output_tiles = triton.cdiv(intermediate_size, block_n)
            row_partitions = _expert_row_partitions(
                capacity, num_experts, output_tiles, block_m
            )
            moe_expert_up_kernel[(output_tiles, num_experts, row_partitions)](
                hidden_states,
                expert_up_weight,
                expert_up_weight if expert_gate_weight is None else expert_gate_weight,
                slot_routes,
                expert_counts,
                up_output,
                *hidden_strides,
                up_strides=expert_up_weight.stride(),
                gate_strides=(
                    (0, 0, 0)
                    if expert_gate_weight is None
                    else expert_gate_weight.stride()
                ),
                SEQUENCE_LEN=sequence_len,
                CHANNELS=channels,
                INTERMEDIATE_SIZE=intermediate_size,
                TOP_K=int(top_k),
                CAPACITY=capacity,
                ACTIVATION=activation,
                ROW_PARTITIONS=row_partitions,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=2,
            )
            partial_output = up_output
            if activation != "linear":
                partial_output = torch.empty(
                    (num_tokens * int(top_k), channels), **options
                )
                output_tiles = triton.cdiv(channels, block_n)
                row_partitions = _expert_row_partitions(
                    capacity, num_experts, output_tiles, block_m
                )
                moe_expert_down_kernel[(output_tiles, num_experts, row_partitions)](
                    up_output,
                    expert_down_weight,
                    slot_routes,
                    expert_counts,
                    partial_output,
                    down_strides=expert_down_weight.stride(),
                    CHANNELS=channels,
                    INTERMEDIATE_SIZE=intermediate_size,
                    CAPACITY=capacity,
                    ROW_PARTITIONS=row_partitions,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_warps=4,
                    num_stages=2,
                )
        moe_gate_combine_kernel[(num_tokens, triton.cdiv(channels, 128))](
            router_logits,
            expert_indices,
            route_kept,
            partial_output,
            hidden_states,
            expert_weights,
            output,
            *hidden_strides,
            NUM_TOKENS=num_tokens,
            SEQUENCE_LEN=sequence_len,
            NUM_EXPERTS=num_experts,
            CHANNELS=channels,
            TOP_K=int(top_k),
            TOP_K_BLOCK=triton.next_power_of_2(int(top_k)),
            BLOCK_CHANNELS=128,
            IDENTITY_EXPERTS=activation == "identity",
            num_warps=4,
        )
    return router_logits, expert_weights, expert_indices, output


__all__ = ["moe_gate_top_k_routing"]
