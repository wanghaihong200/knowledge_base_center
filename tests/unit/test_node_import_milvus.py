"""入库节点测试：mock Milvus 客户端，验证幂等删除、insert 行结构与 chunk_id 回填"""
from unittest.mock import MagicMock, patch

import pytest

from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus

pytestmark = pytest.mark.unit

_NS = "processor.import_processor.nodes.node_import_milvus"


def _chunk(content: str, file_title: str = "it_doc") -> dict:
    return {"title": "t", "content": content, "parent_title": "t", "part": 0,
            "file_title": file_title, "item_name": "路由",
            "dense_vector": [0.1, 0.2, 0.3, 0.4]}


@pytest.fixture
def milvus_mocks():
    client = MagicMock()
    client.insert.return_value = {"insert_count": 2, "ids": [101, 102]}
    with patch(f"{_NS}.get_milvus_client", return_value=client), \
         patch(f"{_NS}.ensure_collection") as ensure:
        yield client, ensure


def test_import_inserts_and_backfills_chunk_id(milvus_mocks):
    client, ensure = milvus_mocks
    chunks = [_chunk("a"), _chunk("b")]
    state = NodeImportMilvus().process({"chunks": chunks})
    # 集合准备：维度取首块向量长度
    ensure.assert_called_once()
    assert ensure.call_args[0][2] == 4
    # 幂等删除：按 file_title
    assert "it_doc" in client.delete.call_args.kwargs["filter"]
    # insert 行不含自增主键 chunk_id
    row = client.insert.call_args.kwargs["data"][0]
    assert "chunk_id" not in row
    assert row["dense_vector"] == [0.1, 0.2, 0.3, 0.4]
    # 回填 chunk_id
    assert state["chunks"][0]["chunk_id"] == "101"
    assert state["chunks"][1]["chunk_id"] == "102"


def test_import_without_vector_raises(milvus_mocks):
    with pytest.raises(Exception):
        NodeImportMilvus().process({"chunks": [{"content": "没有向量"}]})
