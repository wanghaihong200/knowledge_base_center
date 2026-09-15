"""重排节点测试：多源合并、打分排序、断崖截断、零分降级"""
from unittest.mock import patch

import pytest

from processor.query_processor.nodes.node_rerank import NodeRerank

pytestmark = pytest.mark.unit

_NS = "processor.query_processor.nodes.node_rerank"


def _local(cid: str, content: str) -> dict:
    return {"chunk_id": cid, "content": content, "title": "手册", "parent_title": "",
            "part": 0, "file_title": "f", "item_name": "路由", "source": "local", "url": None}


@pytest.fixture
def rerank_state():
    return {
        "rrf_chunks": [_local(f"c{i}", f"本地内容{i}") for i in range(1, 4)],
        "web_search_docs": [
            {"title": "官网FAQ", "url": "http://example.com/faq", "snippet": "网页内容A"},
            {"title": "论坛帖", "url": "http://example.com/bbs", "snippet": "网页内容B"},
        ],
        "rewritten_query": "H3C ER2100 怎么配置 NAT",
    }


def test_merge_multi_source_docs(rerank_state):
    with patch(f"{_NS}.rerank_documents", return_value=[0.5] * 5):
        out = NodeRerank().process(rerank_state)
    docs = out["reranked_docs"]
    assert {d["source"] for d in docs} == {"local", "web"}
    web_doc = next(d for d in docs if d["source"] == "web")
    assert web_doc["url"] == "http://example.com/faq"
    assert web_doc["chunk_id"] is None
    assert web_doc["content"] == "网页内容A", "web 结果的 snippet 应并入 content"


def test_cliff_cutoff(rerank_state):
    scores = [0.9, 0.85, 0.8, 0.2, 0.15]  # 0.8→0.2 断崖（abs 0.6 ≥ 0.5）
    with patch(f"{_NS}.rerank_documents", return_value=scores):
        out = NodeRerank().process(rerank_state)
    assert len(out["reranked_docs"]) == 3, "应在断崖处截断为 3 条"
    assert out["reranked_docs"][0]["score"] == 0.9


def test_rerank_sorted_desc(rerank_state):
    with patch(f"{_NS}.rerank_documents", return_value=[0.1, 0.9, 0.5, 0.3, 0.7]):
        out = NodeRerank().process(rerank_state)
    scores = [d["score"] for d in out["reranked_docs"]]
    assert scores == sorted(scores, reverse=True)
    assert out["reranked_docs"][0]["content"] == "本地内容2"


def test_rerank_degrade_on_zero_scores(rerank_state):
    with patch(f"{_NS}.rerank_documents", return_value=[0.0] * 5):
        out = NodeRerank().process(rerank_state)
    assert len(out["reranked_docs"]) == 5, "零分降级：保留原序全部文档（≤TOPK）"
    assert out["reranked_docs"][0]["content"] == "本地内容1"


def test_empty_inputs():
    out = NodeRerank().process({"rrf_chunks": [], "web_search_docs": [], "rewritten_query": "q"})
    assert out["reranked_docs"] == []
