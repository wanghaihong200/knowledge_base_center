"""Rerank 工具单元测试：mock httpx，验证分数回填与降级"""
from unittest.mock import MagicMock, patch

import pytest

from utils import reranker_http_utils as r

pytestmark = pytest.mark.unit


def test_rerank_documents_backfill_by_index():
    resp = MagicMock(status_code=200)
    # 智谱真实响应格式：results 键
    resp.json.return_value = {"results": [
        {"index": 2, "relevance_score": 0.91},
        {"index": 0, "relevance_score": 0.55},
    ]}
    with patch.object(r.httpx, "post", return_value=resp) as post:
        scores = r.rerank_documents("怎么配置NAT", ["a", "b", "c"])
    assert scores == [0.55, 0.0, 0.91]


def test_rerank_documents_backfill_data_key_fallback():
    """兼容 Jina/dashscope 风格的 data 键"""
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"data": [{"index": 1, "relevance_score": 0.7}]}
    with patch.object(r.httpx, "post", return_value=resp):
        assert r.rerank_documents("q", ["a", "b"]) == [0.0, 0.7]


def test_rerank_documents_degrade_on_http_error():
    resp = MagicMock(status_code=429, text="quota exceeded")
    with patch.object(r.httpx, "post", return_value=resp):
        scores = r.rerank_documents("q", ["a", "b"])
    assert scores == [0.0, 0.0]


def test_rerank_documents_degrade_on_exception():
    with patch.object(r.httpx, "post", side_effect=RuntimeError("conn refused")):
        assert r.rerank_documents("q", ["a"]) == [0.0]


def test_rerank_documents_empty_input():
    assert r.rerank_documents("q", []) == []
