"""Embedding 工具单元测试：mock OpenAI 兼容客户端，验证分批、顺序与维度"""
from unittest.mock import MagicMock, patch

import pytest

from utils import embedding_utils

pytestmark = pytest.mark.unit


def _fake_client(spy_calls: list):
    fake = MagicMock()

    def _create(**kwargs):
        spy_calls.append(kwargs)
        resp = MagicMock()
        input_len = len(kwargs["input"])
        # 故意乱序返回，验证按 index 对齐
        resp.data = [
            MagicMock(index=i, embedding=[float(i + kwargs_offset)] * 4)
            for i, kwargs_offset in reversed(list(enumerate([0] * input_len)))
        ]
        return resp

    fake.embeddings.create.side_effect = _create
    return fake


def test_generate_embeddings_empty():
    assert embedding_utils.generate_embeddings([]) == []


def test_generate_embeddings_batches_and_order():
    calls: list = []
    with patch.object(embedding_utils, "_get_openai_client",
                      return_value=_fake_client(calls)):
        result = embedding_utils.generate_embeddings([str(i) for i in range(12)])
    # 12 条输入、批量上限 8 → 两批(8+4)
    assert len(calls) == 2
    assert len(calls[0]["input"]) == 8
    assert len(calls[1]["input"]) == 4
    assert calls[0]["model"] == "embedding-3"
    assert calls[0]["dimensions"] == 1024
    # 返回顺序与输入顺序一致（按 index 对齐；每批内 index 从 0 重新计数）
    assert len(result) == 12
    assert result[0] == [0.0] * 4       # 批1 index 0
    assert result[7] == [7.0] * 4       # 批1 index 7
    assert result[8] == [0.0] * 4       # 批2 index 0
    assert result[11] == [3.0] * 4      # 批2 index 3


def test_generate_embeddings_api_error_raises():
    fake = MagicMock()
    fake.embeddings.create.side_effect = RuntimeError("api down")
    with patch.object(embedding_utils, "_get_openai_client", return_value=fake):
        with pytest.raises(RuntimeError):
            embedding_utils.generate_embeddings(["x"])
