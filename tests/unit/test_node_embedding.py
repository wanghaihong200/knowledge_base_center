"""向量化节点测试：mock generate_embeddings，验证回填与批量"""
from unittest.mock import patch

import pytest

from processor.import_processor.nodes.node_embedding import NodeEmbedding

pytestmark = pytest.mark.unit

_PATCH = "processor.import_processor.nodes.node_embedding.generate_embeddings"


def _chunk(content: str) -> dict:
    return {"title": "t", "content": content, "parent_title": "t",
            "part": 0, "file_title": "f", "item_name": ""}


def test_embedding_fills_dense_vector_with_item_prefix():
    chunks = [_chunk("默认地址"), _chunk("NAT配置")]
    with patch(_PATCH, return_value=[[0.1] * 4, [0.2] * 4]) as gen:
        state = NodeEmbedding().process({"chunks": chunks, "item_name": "H3C ER2100"})
    assert state["chunks"][0]["dense_vector"] == [0.1] * 4
    assert state["chunks"][1]["dense_vector"] == [0.2] * 4
    texts = gen.call_args[0][0]
    assert texts[0] == "H3C ER2100\n默认地址", "有主体名时输入文本应带主体前缀"


def test_embedding_without_item_name():
    chunks = [_chunk("内容")]
    with patch(_PATCH, return_value=[[0.3] * 4]) as gen:
        state = NodeEmbedding().process({"chunks": chunks, "item_name": ""})
    assert gen.call_args[0][0] == ["内容"]
    assert state["chunks"][0]["dense_vector"] == [0.3] * 4


def test_embedding_empty_chunks_raises():
    with pytest.raises(Exception):
        NodeEmbedding().process({"chunks": []})
