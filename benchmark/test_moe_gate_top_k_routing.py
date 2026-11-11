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

import math

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import base

ACTIVATIONS = ("identity", "linear", "relu", "gelu", "silu", "swiglu", "geglu")


class MoeGateTopKRoutingBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _torch_moe(hidden, gate, up, down, expert_gate, top_k, capacity_factor, activation):
    """Native fixed-shape scatter/BMM reference; no Python loop over tokens."""
    channels = hidden.shape[-1]
    x = hidden.reshape(-1, channels)
    tokens, experts = x.shape[0], gate.shape[1]
    logits = (x.float() @ gate.float()).to(hidden.dtype)
    indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :top_k]
    selected = torch.zeros(
        (tokens, experts), device=x.device, dtype=torch.int32
    ).scatter_(1, indices, 1)
    locations = selected.cumsum(0, dtype=torch.int32) - 1
    capacity = (
        tokens
        if capacity_factor is None
        else math.ceil(capacity_factor * tokens * top_k / experts)
    )
    kept = (locations < capacity).gather(1, indices)
    selected_logits = logits.gather(1, indices).float().masked_fill(~kept, -torch.inf)
    weights = selected_logits.softmax(-1).nan_to_num(0).to(hidden.dtype)
    if activation == "identity":
        output = x.float() * weights.float().sum(-1, keepdim=True)
    else:
        selected_locations = locations.gather(1, indices)
        slots = indices * capacity + selected_locations
        safe_slots = torch.where(kept, slots, 0).reshape(-1)
        routed = (x[:, None, :].expand(-1, top_k, -1) * kept[..., None]).reshape(
            -1, channels
        )
        dispatched = torch.zeros(
            (experts * capacity, channels), device=x.device, dtype=x.dtype
        )
        dispatched.index_add_(0, safe_slots, routed)
        dispatched = dispatched.reshape(experts, capacity, channels)
        projected = torch.bmm(dispatched, up)
        if activation in ("swiglu", "geglu"):
            gate_projected = torch.bmm(dispatched, expert_gate)
            activated = (
                F.silu(gate_projected)
                if activation == "swiglu"
                else F.gelu(gate_projected)
            ) * projected
        elif activation == "relu":
            activated = projected.relu()
        elif activation == "gelu":
            activated = F.gelu(projected)
        elif activation == "silu":
            activated = F.silu(projected)
        else:
            activated = projected
        expert_output = (
            activated if activation == "linear" else torch.bmm(activated, down)
        )
        gathered = (
            expert_output.reshape(-1, channels)
            .index_select(0, safe_slots)
            .reshape(tokens, top_k, channels)
        )
        output = (gathered.float() * weights[..., None].float()).sum(1)
    leading = hidden.shape[:-1]
    return (
        logits.reshape(*leading, experts),
        weights.reshape(*leading, top_k),
        indices.reshape(*leading, top_k),
        output.to(hidden.dtype).reshape(hidden.shape),
    )


def _gems_moe(hidden, gate, up, down, expert_gate, top_k, capacity_factor, activation):
    return flag_gems.moe_gate_top_k_routing(
        hidden,
        gate,
        gate.shape[1],
        top_k,
        capacity_factor,
        expert_up_weight=up,
        expert_down_weight=down,
        expert_gate_weight=expert_gate,
        activation=activation,
    )


def _input(shape, dtype, device):
    (
        batch,
        sequence_len,
        channels,
        experts,
        intermediate,
        top_k,
        capacity_factor,
        mode,
    ) = shape
    activation = ACTIVATIONS[mode]
    hidden_shape = (
        (sequence_len, channels) if batch == 0 else (batch, sequence_len, channels)
    )
    hidden = torch.randn(hidden_shape, device=device, dtype=dtype)
    gate = torch.randn(channels, experts, device=device, dtype=dtype) / math.sqrt(
        channels
    )
    up = down = expert_gate = None
    if activation != "identity":
        intermediate = channels if activation == "linear" else intermediate
        up = torch.randn(
            experts, channels, intermediate, device=device, dtype=dtype
        ) / math.sqrt(channels)
        if activation != "linear":
            down = torch.randn(
                experts, intermediate, channels, device=device, dtype=dtype
            ) / math.sqrt(intermediate)
        if activation in ("swiglu", "geglu"):
            expert_gate = torch.randn(
                experts, channels, intermediate, device=device, dtype=dtype
            ) / math.sqrt(channels)
    yield hidden, gate, up, down, expert_gate, top_k, capacity_factor, activation


@pytest.mark.moe_gate_top_k_routing
def test_moe_gate_top_k_routing_benchmark():
    MoeGateTopKRoutingBenchmark(
        op_name="moe_gate_top_k_routing",
        torch_op=_torch_moe,
        gems_op=_gems_moe,
        input_fn=_input,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    ).run()
