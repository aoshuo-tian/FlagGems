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


class SwinAttentionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _torch_swin(
    query, key, value, mask, num_heads, window_size, relative_position_bias
):
    w, n, c = query.shape
    q, k, v = [
        t.view(w, n, num_heads, c // num_heads).transpose(1, 2)
        for t in (query, key, value)
    ]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=relative_position_bias)
    return out.transpose(1, 2).reshape(w, n, c)


def _input(shape, dtype, device):
    w, wh, ww, c, h = shape
    q, k, v = [torch.randn(w, wh * ww, c, device=device, dtype=dtype) for _ in range(3)]
    bias = torch.randn(h, wh * ww, wh * ww, device=device, dtype=dtype) * 0.1
    yield q, k, v, None, h, (wh, ww), bias


@pytest.mark.swin_transformer_attention
def test_swin_transformer_attention_benchmark():
    SwinAttentionBenchmark(
        op_name="swin_transformer_attention",
        torch_op=_torch_swin,
        gems_op=flag_gems.swin_transformer_attention,
        input_fn=_input,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    ).run()
