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
import torch.nn.functional as F

import flag_gems

from . import base


class AdaptiveSpanBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _torch_adaptive(query, key, value, span, max_span, ramp_length, mask=None):
    distance = (
        torch.arange(query.shape[2], device=query.device)[:, None]
        - torch.arange(key.shape[2], device=query.device)[None, :]
    )
    span = span.clamp(0, max_span)
    if span.ndim == 1:
        span = span[None]
    weight = ((span[..., None, None] + ramp_length - distance) / ramp_length).clamp(
        0, 1
    )
    weight = weight.masked_fill((distance < 0) | (distance > max_span), 0)
    bias = weight.log().to(query.dtype)
    if mask is not None:
        bias = (
            bias.masked_fill(~mask, -torch.inf)
            if mask.dtype == torch.bool
            else bias + mask
        )
    return F.scaled_dot_product_attention(query, key, value, attn_mask=bias)


def _input(shape, dtype, device):
    b, h, lq, lk, d = shape
    q = torch.randn(b, h, lq, d, device=device, dtype=dtype)
    k, v = [torch.randn(b, h, lk, d, device=device, dtype=dtype) for _ in range(2)]
    maximum = max(lq, lk)
    span = torch.full((h,), maximum / 2, device=device)
    yield q, k, v, span, maximum, 32


@pytest.mark.adaptive_attention_span
def test_adaptive_attention_span_benchmark():
    AdaptiveSpanBenchmark(
        op_name="adaptive_attention_span",
        torch_op=_torch_adaptive,
        gems_op=flag_gems.adaptive_attention_span,
        input_fn=_input,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    ).run()
