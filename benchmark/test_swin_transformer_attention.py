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


class SwinAttentionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _torch_swin(
    query, key, value, mask, num_heads, window_size, relative_position_bias
):
    n, c = query.shape[-2:]
    w = math.prod(query.shape[:-2])
    q, k, v = [
        t.reshape(w, n, num_heads, c // num_heads).transpose(1, 2)
        for t in (query, key, value)
    ]
    combined = relative_position_bias
    if mask is not None:
        mask = mask[:, None].repeat(w // mask.shape[0], 1, 1, 1)
        combined = combined.expand(w, num_heads, n, n)
        combined = (
            combined.masked_fill(mask, -torch.inf)
            if mask.dtype == torch.bool
            else combined + mask
        )
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=combined)
    return out.transpose(1, 2).reshape(query.shape)


def _input(shape, dtype, device):
    if len(shape) == 5:
        w, wh, ww, c, h = shape
        token_shape = (w, wh * ww, c)
        windows = w
        mask_kind = 0
    else:
        b, windows, wh, ww, c, h, *options = shape
        token_shape = (b, windows, wh * ww, c)
        mask_kind = options[0] if options else 0
    q, k, v = [torch.randn(token_shape, device=device, dtype=dtype) for _ in range(3)]
    n = wh * ww
    bias = torch.randn(h, n, n, device=device, dtype=dtype) * 0.1
    mask = None
    if mask_kind:
        regions = torch.arange(n, device=device) // max(1, n // 4)
        blocked = regions[:, None] != regions[None, :]
        blocked = blocked.expand(windows, n, n)
        mask = (
            blocked
            if mask_kind == 1
            else torch.zeros((windows, n, n), device=device, dtype=dtype).masked_fill(
                blocked, -100.0
            )
        )
    yield q, k, v, mask, h, (wh, ww), bias


@pytest.mark.swin_transformer_attention
def test_swin_transformer_attention_benchmark():
    SwinAttentionBenchmark(
        op_name="swin_transformer_attention",
        torch_op=_torch_swin,
        gems_op=flag_gems.swin_transformer_attention,
        input_fn=_input,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    ).run()
