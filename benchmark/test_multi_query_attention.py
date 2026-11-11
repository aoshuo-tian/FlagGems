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


class MultiQueryAttentionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _torch_mqa(query, key, value, mask=None, scale=None):
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
    )


def _inputs(shape, dtype, device, *, masked=False):
    b, h, hk, lq, lk, d = shape
    hk = 1
    q = torch.randn(b, h, lq, d, device=device, dtype=dtype)
    k, v = [torch.randn(b, hk, lk, d, device=device, dtype=dtype) for _ in range(2)]
    mask = None
    if masked:
        mask = torch.ones(b, 1, lq, lk, device=device, dtype=torch.bool)
        mask[..., lk * 3 // 4 :] = False
    yield q, k, v, mask, d**-0.5


def _plain_input(shape, dtype, device):
    yield from _inputs(shape, dtype, device)


def _masked_input(shape, dtype, device):
    yield from _inputs(shape, dtype, device, masked=True)


@pytest.mark.multi_query_attention_mqa
@pytest.mark.parametrize(
    "op_name,input_fn",
    (
        ("multi_query_attention_mqa", _plain_input),
        ("multi_query_attention_mqa_mask", _masked_input),
    ),
)
def test_multi_query_attention_benchmark(op_name, input_fn):
    MultiQueryAttentionBenchmark(
        op_name=op_name,
        torch_op=_torch_mqa,
        gems_op=flag_gems.multi_query_attention_mqa,
        input_fn=input_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    ).run()
