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

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
ACTIVATIONS = ("identity", "linear", "relu", "gelu", "silu", "swiglu", "geglu")
SHAPES = (
    ((1, 8), 1, 16, 1),
    ((7, 17), 3, 19, 2),
    ((13, 32), 4, 48, 2),
    ((31, 33), 7, 65, 3),
    ((2, 17, 16), 8, 32, 2),
    ((3, 5, 7), 3, 13, 3),
    ((64, 64), 16, 96, 2),
    ((65, 31), 5, 37, 1),
    ((2, 33, 32), 8, 64, 4),
    ((97, 48), 17, 63, 3),
    ((4, 8, 16), 7, 32, 2),
    ((129, 17), 16, 31, 2),
    ((17, 65), 9, 33, 4),
    ((1, 129), 32, 67, 2),
    ((2, 7, 33), 17, 49, 3),
    ((33, 96), 16, 128, 2),
    ((128, 32), 32, 64, 4),
    ((257, 16), 8, 32, 2),
    ((1, 3, 48), 5, 64, 5),
    ((5, 7), 129, 13, 2),
)


def _inputs(
    shape,
    experts,
    intermediate,
    dtype,
    activation="relu",
    strided=False,
    gate_dtype=None,
):
    def random_tensor(sizes, tensor_dtype=dtype, scale=1.0):
        if strided:
            result = torch.randn(
                *sizes[:-1], sizes[-1] * 2, device=flag_gems.device, dtype=tensor_dtype
            )[..., ::2]
            result.mul_(scale)
            return result
        return torch.randn(sizes, device=flag_gems.device, dtype=tensor_dtype) * scale

    hidden = random_tensor(shape)
    gate = random_tensor((shape[-1], experts), gate_dtype or dtype, shape[-1] ** -0.5)
    kwargs = dict(activation=activation)
    if activation != "identity":
        intermediate = shape[-1] if activation == "linear" else intermediate
        kwargs["expert_up_weight"] = random_tensor(
            (experts, shape[-1], intermediate), scale=shape[-1] ** -0.5
        )
        if activation != "linear":
            kwargs["expert_down_weight"] = random_tensor(
                (experts, intermediate, shape[-1]), scale=intermediate**-0.5
            )
        if activation in ("swiglu", "geglu"):
            kwargs["expert_gate_weight"] = random_tensor(
                (experts, shape[-1], intermediate), scale=shape[-1] ** -0.5
            )
    return hidden, gate, kwargs


def _reference(hidden, gate, top_k, capacity_factor, **kwargs):
    """CPU/double reference with explicit token-order routing, independent of kernels."""
    dtype = hidden.dtype
    leading = hidden.shape[:-1]
    channels = hidden.shape[-1]
    x = hidden.detach().cpu().double().reshape(-1, channels)
    experts = gate.shape[1]
    logits = (x @ gate.detach().cpu().double()).to(dtype).double()
    if "router_logits" in kwargs:
        logits = (
            kwargs.pop("router_logits").detach().cpu().reshape(-1, experts).double()
        )
    indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :top_k]
    tokens = x.shape[0]
    capacity = (
        tokens
        if capacity_factor is None
        else math.ceil(capacity_factor * tokens * top_k / experts)
    )
    kept = torch.zeros((tokens, top_k), dtype=torch.bool)
    counts = [0] * experts
    for token in range(tokens):
        for rank in range(top_k):
            expert = int(indices[token, rank])
            kept[token, rank] = counts[expert] < capacity
            counts[expert] += 1
    selected_logits = logits.gather(1, indices).masked_fill(~kept, -torch.inf)
    weights = selected_logits.softmax(-1).nan_to_num(0).to(dtype).double()
    output = torch.zeros_like(x)
    activation = kwargs.get("activation", "relu")
    tensors = {
        name: value.detach().cpu().double()
        for name, value in kwargs.items()
        if isinstance(value, torch.Tensor)
    }
    for expert in range(experts):
        token_ids, ranks = torch.where((indices == expert) & kept)
        if not token_ids.numel():
            continue
        values = x[token_ids]
        if activation != "identity":
            projection = (
                (values @ tensors["expert_up_weight"][expert]).to(dtype).double()
            )
            if activation in ("swiglu", "geglu"):
                gate_projection = (
                    (values @ tensors["expert_gate_weight"][expert]).to(dtype).double()
                )
                activated = (
                    (
                        F.silu(gate_projection)
                        if activation == "swiglu"
                        else F.gelu(gate_projection)
                    )
                    .to(dtype)
                    .double()
                )
                activated = (activated * projection).to(dtype).double()
            elif activation == "relu":
                activated = projection.relu().to(dtype).double()
            elif activation == "gelu":
                activated = F.gelu(projection).to(dtype).double()
            elif activation == "silu":
                activated = F.silu(projection).to(dtype).double()
            else:
                activated = projection
            values = (
                activated
                if activation == "linear"
                else (activated @ tensors["expert_down_weight"][expert])
                .to(dtype)
                .double()
            )
        output[token_ids] += values * weights[token_ids, ranks, None]
    return (
        logits.to(dtype).reshape(*leading, experts),
        weights.to(dtype).reshape(*leading, top_k),
        indices.reshape(*leading, top_k),
        output.to(dtype).reshape(hidden.shape),
    )


def _assert_result(actual, expected, hidden, experts, top_k):
    assert len(actual) == 4
    atol, rtol = {
        torch.float16: (0.004, 0.004),
        torch.bfloat16: (0.03, 0.03),
        torch.float32: (5e-5, 5e-5),
    }[hidden.dtype]
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        assert result.device == hidden.device
        if reference.dtype == torch.int64:
            assert result.dtype == torch.int64
            torch.testing.assert_close(result.cpu(), reference, atol=0, rtol=0)
        else:
            assert result.dtype == hidden.dtype
            assert torch.isfinite(result).all()
            torch.testing.assert_close(
                result.cpu().double(), reference.double(), atol=atol, rtol=rtol
            )
    assert actual[-1].shape == hidden.shape
    assert ((actual[2] >= 0) & (actual[2] < experts)).all()
    if actual[1].numel():
        sums = actual[1].float().sum(-1)
        assert torch.all(
            (sums == 0)
            | torch.isclose(sums, torch.ones_like(sums), atol=0.01, rtol=0.01)
        )


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,experts,intermediate,top_k", SHAPES)
def test_moe_reference(shape, experts, intermediate, top_k, dtype):
    torch.manual_seed(109)
    hidden, gate, kwargs = _inputs(shape, experts, intermediate, dtype)
    actual = flag_gems.moe_gate_top_k_routing(
        hidden, gate, experts, top_k, 1.25, **kwargs
    )
    _assert_result(
        actual, _reference(hidden, gate, top_k, 1.25, **kwargs), hidden, experts, top_k
    )


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("strided", (False, True))
def test_moe_expert_modes(dtype, activation, strided):
    torch.manual_seed(113)
    hidden, gate, kwargs = _inputs((2, 7, 17), 5, 29, dtype, activation, strided)
    # Transposed leading axes exercise genuinely noncontiguous [B,S,C] input.
    if strided:
        hidden = hidden.transpose(0, 1)
        gate = gate.transpose(0, 1).contiguous().transpose(0, 1)
    actual = flag_gems.moe_gate_top_k_routing(hidden, gate, 5, 3, 0.75, **kwargs)
    _assert_result(actual, _reference(hidden, gate, 3, 0.75, **kwargs), hidden, 5, 3)


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("capacity_factor", (None, 0.01, 1.0, 3.0))
def test_moe_ties_capacity_and_empty_experts(dtype, capacity_factor):
    hidden = (
        torch.arange(8 * 7, device=flag_gems.device, dtype=dtype).reshape(8, 7) / 16
    )
    gate = torch.zeros(7, 4, device=hidden.device, dtype=dtype)
    kwargs = dict(activation="identity")
    actual = flag_gems.moe_gate_top_k_routing(
        hidden, gate, 4, 2, capacity_factor, **kwargs
    )
    _assert_result(
        actual, _reference(hidden, gate, 2, capacity_factor, **kwargs), hidden, 4, 2
    )
    torch.testing.assert_close(
        actual[2], torch.tensor([0, 1], device=hidden.device).expand(8, 2)
    )
    if capacity_factor == 1.0:
        assert torch.count_nonzero(actual[1][4:]) == 0
        assert torch.count_nonzero(actual[3][4:]) == 0


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_renormalizes_surviving_routes(dtype):
    hidden = torch.tensor(
        [[10, 9, 0]] * 3 + [[10, 0, 9]] * 3, device=flag_gems.device, dtype=dtype
    )
    gate = torch.eye(3, device=hidden.device, dtype=dtype)
    actual = flag_gems.moe_gate_top_k_routing(
        hidden, gate, 3, 2, 0.75, activation="identity"
    )
    _assert_result(
        actual, _reference(hidden, gate, 2, 0.75, activation="identity"), hidden, 3, 2
    )
    torch.testing.assert_close(
        actual[1][3:],
        torch.tensor([0, 1], device=hidden.device, dtype=dtype).expand(3, 2),
    )
    assert ((actual[2] >= 0) & (actual[2] < 3)).all()


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("shape", ((0, 17), (2, 0, 17)))
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_empty_tokens(shape, dtype):
    hidden, gate, kwargs = _inputs(shape, 3, 19, dtype, "identity")
    actual = flag_gems.moe_gate_top_k_routing(hidden, gate, 3, 2, **kwargs)
    _assert_result(actual, _reference(hidden, gate, 2, 1.0, **kwargs), hidden, 3, 2)


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize(
    "hidden_dtype,gate_dtype",
    (
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
    ),
)
def test_moe_mixed_gate_precision(hidden_dtype, gate_dtype):
    hidden, gate, kwargs = _inputs((13, 17), 7, 23, hidden_dtype, gate_dtype=gate_dtype)
    actual = flag_gems.moe_gate_top_k_routing(hidden, gate, 7, 2, **kwargs)
    _assert_result(actual, _reference(hidden, gate, 2, 1.0, **kwargs), hidden, 7, 2)


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize(
    "case",
    (
        "input_rank",
        "input_dtype",
        "cpu",
        "gate_shape",
        "gate_dtype",
        "experts_zero",
        "experts_mismatch",
        "experts_bool",
        "top_k_zero",
        "top_k_large",
        "top_k_float",
        "capacity_zero",
        "capacity_nan",
        "capacity_bool",
        "activation",
        "up_missing",
        "up_shape",
        "down_shape",
        "gate_missing",
        "unused_gate",
        "weight_dtype",
        "linear_shape",
        "identity_weights",
    ),
)
def test_moe_validation(case):
    hidden, gate, kwargs = _inputs((7, 17), 3, 19, torch.float32)
    experts, top_k, capacity = 3, 2, 1.0
    if case == "input_rank":
        hidden = hidden[0]
    elif case == "input_dtype":
        hidden = hidden.to(torch.int32)
    elif case == "cpu":
        hidden, gate = hidden.cpu(), gate.cpu()
    elif case == "gate_shape":
        gate = gate[:-1]
    elif case == "gate_dtype":
        gate = gate.to(torch.int32)
    elif case == "experts_zero":
        experts = 0
    elif case == "experts_mismatch":
        experts = 4
    elif case == "experts_bool":
        experts = True
    elif case == "top_k_zero":
        top_k = 0
    elif case == "top_k_large":
        top_k = 4
    elif case == "top_k_float":
        top_k = 2.0
    elif case == "capacity_zero":
        capacity = 0
    elif case == "capacity_nan":
        capacity = float("nan")
    elif case == "capacity_bool":
        capacity = True
    elif case == "activation":
        kwargs["activation"] = "unknown"
    elif case == "up_missing":
        kwargs["expert_up_weight"] = None
    elif case == "up_shape":
        kwargs["expert_up_weight"] = kwargs["expert_up_weight"][:, :-1]
    elif case == "down_shape":
        kwargs["expert_down_weight"] = kwargs["expert_down_weight"][:, :-1]
    elif case == "gate_missing":
        kwargs["activation"] = "swiglu"
    elif case == "unused_gate":
        kwargs["expert_gate_weight"] = kwargs["expert_up_weight"]
    elif case == "weight_dtype":
        kwargs["expert_up_weight"] = kwargs["expert_up_weight"].half()
    elif case == "linear_shape":
        kwargs["activation"] = "linear"
    elif case == "identity_weights":
        kwargs["activation"] = "identity"
    with pytest.raises((ValueError, TypeError)):
        flag_gems.moe_gate_top_k_routing(
            hidden, gate, experts, top_k, capacity, **kwargs
        )


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_infers_num_experts(dtype):
    hidden, gate, kwargs = _inputs((11, 17), 5, 23, dtype, "swiglu")
    actual = flag_gems.moe_gate_top_k_routing(
        hidden, gate, top_k=2, capacity_factor=None, **kwargs
    )
    _assert_result(actual, _reference(hidden, gate, 2, None, **kwargs), hidden, 5, 2)


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("strided", (False, True))
def test_moe_2d_3d_same_tokens(dtype, activation, strided):
    hidden, gate, kwargs = _inputs((3, 17, 65), 7, 129, dtype, activation, strided)
    flat = hidden.reshape(-1, hidden.shape[-1])
    actual = flag_gems.moe_gate_top_k_routing(hidden, gate, 7, 3, 1.25, **kwargs)
    flattened = flag_gems.moe_gate_top_k_routing(flat, gate, 7, 3, 1.25, **kwargs)
    for result, expected in zip(actual, flattened):
        torch.testing.assert_close(
            result.reshape(expected.shape), expected, rtol=0, atol=0
        )
    _assert_result(actual, _reference(hidden, gate, 3, 1.25, **kwargs), hidden, 7, 3)


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", ("relu", "swiglu"))
def test_moe_expert_addresses_exceed_int32(dtype, activation):
    hidden, gate, kwargs = _inputs((2, 3, 17), 2, 19, dtype, activation)
    large_stride = 2**31 + 256
    try:
        for name in ("expert_up_weight", "expert_down_weight", "expert_gate_weight"):
            if name not in kwargs:
                continue
            original = kwargs[name]
            _, rows, columns = original.shape
            weight = torch.empty_strided(
                original.shape,
                (large_stride, columns, 1),
                dtype=dtype,
                device=hidden.device,
            )
            weight.copy_(original)
            kwargs[name] = weight
    except torch.OutOfMemoryError:
        pytest.skip(
            "This address-width regression needs up to 24 GiB of device storage"
        )
    actual = flag_gems.moe_gate_top_k_routing(hidden, gate, 2, 2, None, **kwargs)
    _assert_result(actual, _reference(hidden, gate, 2, None, **kwargs), hidden, 2, 2)


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_routes_from_returned_quantized_logits(dtype):
    hidden, gate, kwargs = _inputs((2, 4096, 2048), 60, 1, dtype, "identity")
    actual = flag_gems.moe_gate_top_k_routing(hidden, gate, 60, 4, 1.25, **kwargs)
    independent_logits = (
        (hidden.cpu().double().reshape(-1, 2048) @ gate.cpu().double())
        .to(dtype)
        .reshape(actual[0].shape)
    )
    atol, rtol = {
        torch.float16: (0.004, 0.004),
        torch.bfloat16: (0.03, 0.03),
        torch.float32: (5e-5, 5e-5),
    }[dtype]
    torch.testing.assert_close(
        actual[0].cpu(), independent_logits, atol=atol, rtol=rtol
    )
    expected = _reference(hidden, gate, 4, 1.25, router_logits=actual[0], **kwargs)
    _assert_result(actual, expected, hidden, 60, 4)


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("activation", ("relu", "swiglu", "geglu"))
@pytest.mark.parametrize("strided", (False, True))
@pytest.mark.parametrize("tile", ((16, 64, 64, 3), (64, 128, 64, 3), (128, 64, 64, 3)))
def test_moe_pipelined_expert_tiles(dtype, activation, strided, tile, monkeypatch):
    import importlib

    module = importlib.import_module("flag_gems.ops.moe_gate_top_k_routing")
    monkeypatch.setattr(module, "_select_expert_tiling", lambda *args: tile)
    torch.manual_seed(137)
    hidden, gate, kwargs = _inputs((2, 17, 65), 5, 97, dtype, activation, strided)
    if strided:
        hidden = hidden.transpose(0, 1)
        gate = gate.transpose(0, 1).contiguous().transpose(0, 1)
    actual = flag_gems.moe_gate_top_k_routing(hidden, gate, 5, 3, 0.5, **kwargs)
    _assert_result(actual, _reference(hidden, gate, 3, 0.5, **kwargs), hidden, 5, 3)


@pytest.mark.moe_gate_top_k_routing
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("shape", ((129, 512), (3, 43, 512)))
@pytest.mark.parametrize("activation", ("relu", "swiglu"))
def test_moe_model_width_tiles(dtype, shape, activation):
    torch.manual_seed(139)
    hidden, gate, kwargs = _inputs(shape, 4, 640, dtype, activation)
    # Exact ties exercise empty experts, capacity dropping and two input ranks.
    gate.zero_()
    actual = flag_gems.moe_gate_top_k_routing(hidden, gate, 4, 2, 1.25, **kwargs)
    _assert_result(actual, _reference(hidden, gate, 2, 1.25, **kwargs), hidden, 4, 2)
