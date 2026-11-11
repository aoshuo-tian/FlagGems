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
SHAPES = (
    (1, 2, 1, 1, 1, 16),
    (2, 4, 2, 7, 13, 17),
    (1, 4, 4, 16, 16, 32),
    (2, 6, 2, 31, 33, 48),
    (1, 8, 1, 33, 17, 64),
    (2, 4, 2, 64, 65, 128),
    (1, 4, 1, 65, 31, 33),
    (2, 8, 4, 97, 129, 32),
    (1, 2, 2, 128, 256, 64),
    (2, 4, 1, 256, 128, 64),
    (1, 4, 2, 257, 513, 16),
    (1, 2, 1, 512, 612, 128),
    (1, 2, 1, 1024, 1024, 32),
    (2, 4, 2, 513, 257, 64),
    (1, 8, 2, 128, 129, 80),
    (1, 2, 1, 63, 67, 96),
    (2, 6, 3, 17, 257, 32),
    (1, 4, 2, 255, 256, 128),
    (1, 2, 1, 129, 127, 7),
    (1, 4, 4, 128, 128, 160),
)


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


def _make(shape, dtype, noncontiguous=False):
    b, h, hk, lq, lk, d = shape

    def tensor(s):
        if noncontiguous:
            return torch.randn(
                *s[:-1], s[-1] * 2, device=flag_gems.device, dtype=dtype
            )[..., ::2]
        return torch.randn(s, device=flag_gems.device, dtype=dtype)

    return (tensor((b, h, lq, d)), tensor((b, hk, lk, d)), tensor((b, hk, lk, d)))


def _reference(q, k, v, mask=None, scale=None):
    q, k, v = (t.detach().cpu().double() for t in (q, k, v))
    ratio = q.shape[1] // k.shape[1]
    k, v = (k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1))
    scores = (
        q @ k.transpose(-1, -2) * (q.shape[-1] ** (-0.5) if scale is None else scale)
    )
    if mask is not None:
        mask = mask.cpu()
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask, -torch.inf)
        else:
            scores += mask.double()
    p = scores.softmax(-1).nan_to_num(0.0)
    return p @ v


def _adaptive_ref(q, k, v, span, maximum, ramp, mask=None):
    q, k, v = (x.cpu().double() for x in (q, k, v))
    span = span.cpu().double().clamp(0, maximum)
    if span.ndim == 1:
        span = span[None]
    distance = torch.arange(q.shape[2])[:, None] - torch.arange(k.shape[2])[None, :]
    weight = torch.where(
        distance <= span[..., None, None],
        1.0,
        1.0 - (distance - span[..., None, None]) / ramp,
    ).clamp_min(0)
    weight = weight * ((distance >= 0) & (distance <= maximum))
    bias = weight.log()
    if mask is not None:
        mask = mask.cpu()
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.dtype == torch.bool:
            bias = bias.masked_fill(~mask, -torch.inf)
        else:
            bias = bias + mask.double()
    return _reference(q, k, v, bias)


@pytest.mark.adaptive_attention_span
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_adaptive_reference(shape, dtype):
    b, h, _, lq, lk, d = shape
    q, k, v = _make((b, h, h, lq, lk, d), dtype)
    maximum = max(lq, lk) // 2
    span = torch.linspace(0, maximum, h, device=q.device)
    out = flag_gems.adaptive_attention_span(q, k, v, span, maximum, 13.5)
    _assert_close(out, _adaptive_ref(q, k, v, span, maximum, 13.5))


@pytest.mark.adaptive_attention_span
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("boolean", (True, False))
@pytest.mark.parametrize("maximum", (0, 7, 100))
def test_adaptive_span_boundaries(dtype, boolean, maximum):
    q, k, v = _make((2, 4, 4, 137, 73, 33), dtype, True)
    span = torch.tensor(
        [[-3.0, 0.0, 3.5, 200.0], [4.0, 7.0, 11.0, 0.0]], device=q.device
    )
    mask = torch.zeros(2, 1, 137, 73, device=q.device)
    mask[..., 5, :] = -torch.inf
    if boolean:
        mask = torch.isfinite(mask)
    out = flag_gems.adaptive_attention_span(q, k, v, span, maximum, 4, mask)
    _assert_close(out, _adaptive_ref(q, k, v, span, maximum, 4, mask))
    assert torch.count_nonzero(out[..., 5, :]) == 0


@pytest.mark.adaptive_attention_span
def test_adaptive_distance_direction_and_causality():
    q = torch.zeros(1, 1, 5, 16, device=flag_gems.device)
    v = torch.arange(5, device=q.device, dtype=q.dtype)[None, None, :, None].expand_as(
        q
    )
    span = torch.zeros(1, device=q.device)
    out = flag_gems.adaptive_attention_span(q, q, v, span, 4, 1)
    torch.testing.assert_close(out, v, rtol=0, atol=0)


@pytest.mark.adaptive_attention_span
@pytest.mark.parametrize(
    "case",
    ("span_shape", "span_dtype", "ramp_zero", "ramp_nan", "max_negative", "heads"),
)
def test_adaptive_validation(case):
    q, k, v = _make((2, 4, 4, 7, 9, 16), torch.float32)
    span = torch.ones(4, device=q.device)
    maximum, ramp = (8, 4)
    if case == "span_shape":
        span = span[:3]
    elif case == "span_dtype":
        span = span.long()
    elif case == "ramp_zero":
        ramp = 0
    elif case == "ramp_nan":
        ramp = float("nan")
    elif case == "max_negative":
        maximum = -1
    elif case == "heads":
        k, v = (k[:, :2], v[:, :2])
    with pytest.raises((ValueError, TypeError)):
        flag_gems.adaptive_attention_span(q, k, v, span, maximum, ramp)


@pytest.mark.adaptive_attention_span
@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_fractional_span_tile_skipping(dtype):
    q, k, v = _make((2, 4, 4, 257, 193, 32), dtype, True)
    span = torch.tensor([0.1, 31.5, 63.25, 1000.0], device=q.device)
    out = flag_gems.adaptive_attention_span(q, k, v, span, 256, 0.5)
    _assert_close(out, _adaptive_ref(q, k, v, span, 256, 0.5))
    assert torch.count_nonzero(out[:, 0, 193:, :]) == 0
