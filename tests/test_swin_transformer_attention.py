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

import pytest
import torch

import flag_gems

DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _assert_close(actual, expected):
    atol, rtol = {
        torch.float16: (0.002, 0.002),
        torch.bfloat16: (0.015, 0.015),
        torch.float32: (2e-05, 2e-05),
    }[actual.dtype]
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.cpu().double(), expected.double(), atol=atol, rtol=rtol
    )


def _reference(q, k, v, mask=None, scale=None):
    q, k, v = (t.detach().cpu().double() for t in (q, k, v))
    ratio = q.shape[1] // k.shape[1]
    k, v = (k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1))
    scores = (
        q @ k.transpose(-1, -2) * (q.shape[-1] ** (-0.5) if scale is None else scale)
    )
    if mask is not None:
        mask = mask.cpu()
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask, -torch.inf)
        else:
            scores += mask.double()
    p = scores.softmax(-1).nan_to_num(0.0)
    return p @ v


SHAPES = (
    (1, 1, 1, 16, 1),
    (2, 2, 3, 32, 2),
    (1, 3, 3, 48, 3),
    (2, 4, 4, 64, 2),
    (4, 7, 7, 96, 3),
    (2, 8, 8, 64, 4),
    (1, 9, 9, 128, 2),
    (2, 10, 10, 64, 2),
    (1, 12, 12, 96, 3),
    (2, 14, 14, 128, 4),
    (1, 16, 16, 64, 2),
    (1, 17, 17, 32, 1),
    (1, 20, 20, 64, 2),
    (1, 24, 24, 32, 1),
    (1, 32, 32, 32, 1),
    (2, 3, 5, 68, 4),
    (1, 7, 9, 96, 2),
    (1, 8, 16, 128, 1),
    (1, 4, 8, 160, 1),
    (2, 5, 5, 21, 3),
)


def _swin_ref(q, k, v, heads, mask=None, bias=None):
    shape = q.shape
    n, c = shape[-2:]
    q, k, v = [
        x.cpu().double().reshape(-1, n, heads, c // heads).transpose(1, 2)
        for x in (q, k, v)
    ]
    combined = torch.zeros(q.shape[0], heads, n, n, dtype=torch.float64)
    if bias is not None:
        combined += bias.cpu().double()
    if mask is not None:
        mask = mask.cpu()
        if mask.ndim >= 5:
            mask = torch.broadcast_to(mask, (*shape[:-2], heads, n, n)).reshape(
                -1, heads, n, n
            )
        elif mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim == 4 and mask.shape[0] != q.shape[0]:
            mask = mask.repeat(q.shape[0] // mask.shape[0], 1, 1, 1)
        if mask.dtype == torch.bool:
            combined.masked_fill_(mask, -torch.inf)
        else:
            combined += mask.double()
    out = _reference(q, k, v, combined)
    return out.transpose(1, 2).reshape(shape)


@pytest.mark.swin_transformer_attention
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_swin_reference(shape, dtype):
    w, wh, ww, c, h = shape
    q, k, v = [
        torch.randn(w, wh * ww, c, dtype=dtype, device=flag_gems.device)
        for _ in range(3)
    ]
    bias = torch.randn(h, wh * ww, wh * ww, device=q.device, dtype=dtype) * 0.2
    out = flag_gems.swin_transformer_attention(
        q, k, v, num_heads=h, window_size=(wh, ww), relative_position_bias=bias
    )
    _assert_close(out, _swin_ref(q, k, v, h, bias=bias))


@pytest.mark.swin_transformer_attention
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("boolean", (False, True))
@pytest.mark.parametrize("rank", (3, 4))
def test_swin_layout_mask(dtype, boolean, rank):
    shape = {3: (6, 49, 64), 4: (2, 3, 49, 64)}[rank]
    q, k, v = [
        torch.randn(*shape[:-1], 128, device=flag_gems.device, dtype=dtype)[..., ::2]
        for _ in range(3)
    ]
    if rank >= 4:
        q, k, v = [t.transpose(0, 1) for t in (q, k, v)]
    mask = torch.zeros(3, 49, 49, device=q.device)
    mask[..., :32] = -torch.inf
    mask[:, 5, :] = -torch.inf
    if boolean:
        mask = torch.isneginf(mask)
    out = flag_gems.swin_transformer_attention(q, k, v, mask, 2, 7)
    _assert_close(out, _swin_ref(q, k, v, 2, mask))
    assert torch.count_nonzero(out[..., 5, :]) == 0


@pytest.mark.swin_transformer_attention
@pytest.mark.parametrize("case", ("heads", "window", "bias", "mask", "rank"))
def test_swin_validation(case):
    q = torch.randn(2, 49, 64, device=flag_gems.device)
    kw = dict(num_heads=2, window_size=7)
    if case == "rank":
        q = q.unsqueeze(0).unsqueeze(0)
    elif case == "heads":
        kw["num_heads"] = 3
    elif case == "window":
        kw["window_size"] = (7, 8)
    elif case == "bias":
        kw["relative_position_bias"] = torch.zeros(2, 48, 49, device=q.device)
    elif case == "mask":
        kw["mask"] = torch.zeros(3, 49, 49, device=q.device)
    with pytest.raises((ValueError, TypeError)):
        flag_gems.swin_transformer_attention(q, q, q, **kw)


@pytest.mark.swin_transformer_attention
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window_size", ([3], (3,), 3, (3, 3)))
def test_swin_window_size_forms(dtype, window_size):
    query, key, value = [
        torch.randn(2, 9, 12, device=flag_gems.device, dtype=dtype) for _ in range(3)
    ]
    actual = flag_gems.swin_transformer_attention(
        query, key, value, num_heads=3, window_size=window_size
    )
    _assert_close(actual, _swin_ref(query, key, value, 3))


@pytest.mark.swin_transformer_attention
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("boolean", (False, True))
@pytest.mark.parametrize("mask_shape", ((), (9,), (2, 3, 3, 9, 9), (1, 3, 1, 1, 9)))
def test_swin_general_mask_broadcast(dtype, boolean, mask_shape):
    query, key, value = [
        torch.randn(2, 3, 9, 24, device=flag_gems.device, dtype=dtype)[..., ::2]
        for _ in range(3)
    ]
    mask = torch.zeros(mask_shape, device=query.device)
    if len(mask_shape) >= 2 and mask_shape[-2] > 2:
        mask[..., 2, :] = -torch.inf
    if boolean:
        mask = torch.isneginf(mask)
    actual = flag_gems.swin_transformer_attention(query, key, value, mask, 3, 3)
    _assert_close(actual, _swin_ref(query, key, value, 3, mask))


@pytest.mark.swin_transformer_attention
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,heads,window",
    (((2, 64, 49, 96), 3, 7), ((2, 16, 144, 256), 8, 12), ((2, 1, 49, 768), 24, 7)),
)
def test_swin_model_stages_and_rank_equivalence(dtype, shape, heads, window):
    q, k, v = [
        torch.randn(shape, device=flag_gems.device, dtype=dtype) for _ in range(3)
    ]
    bias = torch.randn(heads, shape[-2], shape[-2], device=q.device, dtype=dtype) * 0.1
    actual = flag_gems.swin_transformer_attention(
        q, k, v, num_heads=heads, window_size=window, relative_position_bias=bias
    )
    flat = flag_gems.swin_transformer_attention(
        *(x.flatten(0, 1) for x in (q, k, v)),
        num_heads=heads,
        window_size=window,
        relative_position_bias=bias,
    )
    torch.testing.assert_close(actual.flatten(0, 1), flat, rtol=0, atol=0)
    _assert_close(actual, _swin_ref(q, k, v, heads, bias=bias))
