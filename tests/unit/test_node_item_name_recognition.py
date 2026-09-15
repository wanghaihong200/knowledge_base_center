"""主体识别节点测试：mock LLM/向量化/Milvus，验证识别、回填与入库"""
from unittest.mock import MagicMock, patch

import pytest

from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition

pytestmark = pytest.mark.unit

_NS = "processor.import_processor.nodes.node_item_name_recognition"


def _chunks(n=5):
    return [{"title": f"t{i}", "content": f"内容{i}" * 20, "parent_title": f"t{i}",
             "part": 0, "file_title": "H3C ER2100 用户手册", "item_name": ""}
            for i in range(n)]


@pytest.fixture
def mocks():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=" H3C ER2100企业级路由器\n")
    fake_client = MagicMock()
    with patch(f"{_NS}.get_llm_client", return_value=fake_llm), \
         patch(f"{_NS}.generate_embeddings", return_value=[[0.5] * 4]) as gen, \
         patch(f"{_NS}.get_milvus_client", return_value=fake_client), \
         patch(f"{_NS}.ensure_collection"):
        yield fake_llm, gen, fake_client


def test_recognize_and_backfill_and_save(mocks):
    fake_llm, gen, fake_client = mocks
    chunks = _chunks()
    state = NodeItemNameRecognition().process(
        {"file_title": "H3C ER2100 用户手册", "chunks": chunks})
    assert state["item_name"] == "H3C ER2100企业级路由器"
    assert all(c["item_name"] == "H3C ER2100企业级路由器" for c in state["chunks"])
    # 主体向量入库：删除旧名 + 插入新行
    assert 'item_name' in fake_client.delete.call_args.kwargs["filter"]
    row = fake_client.insert.call_args.kwargs["data"][0]
    assert row["item_name"] == "H3C ER2100企业级路由器"
    assert row["dense_vector"] == [0.5] * 4


def test_llm_failure_falls_back_to_file_title(mocks):
    fake_llm, gen, fake_client = mocks
    fake_llm.invoke.side_effect = RuntimeError("LLM 超时")
    chunks = _chunks(2)
    state = NodeItemNameRecognition().process(
        {"file_title": "H3C ER2100 用户手册", "chunks": chunks})
    assert state["item_name"] == "H3C ER2100 用户手册"


def test_llm_empty_answer_falls_back(mocks):
    fake_llm, gen, fake_client = mocks
    fake_llm.invoke.return_value = MagicMock(content="  ")
    state = NodeItemNameRecognition().process(
        {"file_title": "兜底名称", "chunks": _chunks(1)})
    assert state["item_name"] == "兜底名称"
