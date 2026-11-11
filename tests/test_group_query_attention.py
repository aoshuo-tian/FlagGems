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
# Independent correctness coverage; long 4K/8K workloads live in core_shapes.yaml.
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


def _make(shape, dtype, noncontiguous=False):
    b, h, hk, lq, lk, d = shape

    def tensor(s):
        if noncontiguous:
            return torch.randn(
                *s[:-1], s[-1] * 2, device=flag_gems.device, dtype=dtype
            )[..., ::2]
        return torch.randn(s, device=flag_gems.device, dtype=dtype)

    return tensor((b, h, lq, d)), tensor((b, hk, lk, d)), tensor((b, hk, lk, d))


def _reference(q, k, v, mask=None, scale=None):
    q, k, v = (t.detach().cpu().double() for t in (q, k, v))
    ratio = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1)
    scores = (q @ k.transpose(-1, -2)) * (
        q.shape[-1] ** -0.5 if scale is None else scale
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


def _assert_close(actual, expected):
    atol, rtol = {
        torch.float16: (0.002, 0.002),
        torch.bfloat16: (0.015, 0.015),
        torch.float32: (2e-5, 2e-5),
    }[actual.dtype]
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.cpu().double(), expected.double(), atol=atol, rtol=rtol
    )


@pytest.mark.group_query_attention_GQA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_gqa_reference(shape, dtype):
    torch.manual_seed(37)
    q, k, v = _make(shape, dtype)
    actual = flag_gems.group_query_attention_GQA(q, k, v)
    _assert_close(actual, _reference(q, k, v))


@pytest.mark.group_query_attention_GQA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ("2d", "batch", "head", "singleton"))
@pytest.mark.parametrize("boolean", (True, False))
def test_gqa_mask(dtype, layout, boolean):
    q, k, v = _make((2, 4, 2, 37, 97, 32), dtype, True)
    shape = {
        "2d": (37, 97),
        "batch": (2, 37, 97),
        "head": (2, 4, 37, 97),
        "singleton": (1, 4, 1, 97),
    }[layout]
    mask = torch.zeros(*shape[:-1], shape[-1] * 2, device=q.device)[..., ::2]
    mask[..., :64] = -torch.inf  # First entire KV tiles blocked, later tiles valid.
    mask[..., 64:] = 0.3
    if layout != "singleton":
        mask[..., 3, :] = -torch.inf
    if boolean:
        mask = torch.isfinite(mask)
    actual = flag_gems.group_query_attention_GQA(q, k, v, mask, scale=0.17)
    _assert_close(actual, _reference(q, k, v, mask, scale=0.17))
    if layout != "singleton":
        assert torch.count_nonzero(actual[..., 3, :]) == 0


@pytest.mark.group_query_attention_GQA
@pytest.mark.parametrize(
    "case",
    (
        "dtype",
        "shape",
        "ratio",
        "mask_shape",
        "mask_dtype",
        "scale",
        "heads",
        "empty",
        "cpu",
    ),
)
def test_gqa_validation(case):
    q, k, v = _make((1, 4, 2, 7, 9, 16), torch.float32)
    kwargs = {}
    if case == "dtype":
        k = k.half()
    elif case == "shape":
        v = v[..., :-1]
    elif case == "ratio":
        q = q[:, :3]
    elif case == "mask_shape":
        kwargs["mask"] = torch.ones(3, 5, device=q.device)
    elif case == "mask_dtype":
        kwargs["mask"] = torch.ones(7, 9, device=q.device, dtype=torch.int32)
    elif case == "scale":
        kwargs["scale"] = float("nan")
    elif case == "heads":
        kwargs["num_query_heads"] = 3
    elif case == "empty":
        q = q[:, :, :0]
    elif case == "cpu":
        q, k, v = q.cpu(), k.cpu(), v.cpu()
    with pytest.raises((ValueError, TypeError)):
        flag_gems.group_query_attention_GQA(q, k, v, **kwargs)


@pytest.mark.group_query_attention_GQA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", ((1, 2, 1, 1024, 128, 128), (1, 2, 1, 33, 513, 32)))
def test_gqa_tiling_regimes(shape, dtype):
    q, k, v = _make(shape, dtype)
    for masked in (False, True):
        mask = (
            torch.ones(shape[3], shape[4], device=q.device, dtype=torch.bool)
            if masked
            else None
        )
        if mask is not None:
            mask[:, : shape[4] // 2] = False
        actual = flag_gems.group_query_attention_GQA(q, k, v, mask)
        _assert_close(actual, _reference(q, k, v, mask))


@pytest.mark.group_query_attention_GQA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("splits", (4, 8))
@pytest.mark.parametrize("mask_kind", ("none", "bool", "float"))
def test_gqa_split_kv_merge(dtype, splits, mask_kind, monkeypatch):
    import importlib

    module = importlib.import_module("flag_gems.ops.group_query_attention")
    monkeypatch.setattr(module, "_select_kv_splits", lambda *args: splits)
    q, k, v = _make((2, 4, 2, 17, 97, 33), dtype, True)
    mask = None
    if mask_kind != "none":
        mask = torch.full((2, 1, 17, 97), -torch.inf, device=q.device)
        mask[..., 32:64] = 0.2
        mask[..., 7, :] = -torch.inf
        if mask_kind == "bool":
            mask = torch.isfinite(mask)
    actual = flag_gems.group_query_attention_GQA(q, k, v, mask)
    _assert_close(actual, _reference(q, k, v, mask))
    if mask is not None:
        assert torch.count_nonzero(actual[..., 7, :]) == 0


@pytest.mark.group_query_attention_GQA
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "query_len,key_len,head_dim", ((97, 63, 16), (65, 129, 7), (129, 193, 16))
)
def test_unmasked_interior_and_tail(dtype, query_len, key_len, head_dim):
    query, key, value = _make((2, 4, 2, query_len, key_len, head_dim), dtype, True)
    actual = flag_gems.group_query_attention_GQA(query, key, value, scale=0.17)
    _assert_close(actual, _reference(query, key, value, scale=0.17))
