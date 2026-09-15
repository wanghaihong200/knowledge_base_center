"""RRF 融合节点测试：两路加权排名倒数融合"""
import pytest

from processor.query_processor.nodes.node_rrf import NodeRrf

pytestmark = pytest.mark.unit


def _hit(cid: str) -> dict:
    """模拟 Milvus 检索 hit 结构"""
    return {"id": cid, "distance": 0.9,
            "entity": {"chunk_id": cid, "content": f"内容{cid}", "title": "t",
                       "parent_title": "", "part": 0, "file_title": "f",
                       "item_name": "路由", "source": "local", "url": None}}


def test_rrf_two_routes_weighted_by_rank():
    state = {
        "embedding_chunks": [_hit("c1"), _hit("c2")],
        "hyde_embedding_chunks": [_hit("c2"), _hit("c3")],
    }
    out = NodeRrf().process(state)
    ids = [d["chunk_id"] for d in out["rrf_chunks"]]
    assert ids[0] == "c2", "两路都命中的切片 RRF 分数最高"
    assert set(ids) == {"c1", "c2", "c3"}
    assert out["rrf_chunks"][0]["content"] == "内容c2", "保留首见文档数据"


def test_rrf_single_route_order_preserved():
    state = {"embedding_chunks": [_hit("a"), _hit("b")], "hyde_embedding_chunks": []}
    out = NodeRrf().process(state)
    assert [d["chunk_id"] for d in out["rrf_chunks"]] == ["a", "b"]


def test_rrf_empty_state():
    out = NodeRrf().process({})
    assert out["rrf_chunks"] == []
