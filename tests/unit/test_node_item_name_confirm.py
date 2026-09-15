"""主体确认节点测试：mock LLM/Milvus/Mongo，验证提取、对齐三分支与历史写入"""
from unittest.mock import MagicMock, patch

import pytest

from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm

pytestmark = pytest.mark.unit

_NS = "processor.query_processor.nodes.node_item_name_confirm"

JSON_ANSWER = '```json\n{"item_names": ["H3C ER2100"], "rewritten_query": "H3C ER2100 的 NAT 配置方法"}\n```'


@pytest.fixture
def mocks():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=JSON_ANSWER)
    fake_client = MagicMock()

    p_llm = patch(f"{_NS}.get_llm_client", return_value=fake_llm)
    p_gen = patch(f"{_NS}.generate_embeddings", return_value=[[0.1] * 4])
    p_client = patch(f"{_NS}.get_milvus_client", return_value=fake_client)
    p_vs = patch(f"{_NS}.vector_search")
    p_hist = patch(f"{_NS}.get_recent_messages", return_value=[])
    p_save = patch(f"{_NS}.save_chat_message", return_value="64f00000000000000000000a")
    p_upd = patch(f"{_NS}.update_message_item_names")

    started = {
        "vector_search": p_vs.start(),
        "save": p_save.start(),
        "update": p_upd.start(),
    }
    for p in (p_llm, p_gen, p_client, p_hist):
        p.start()
    started["llm"] = fake_llm
    started["client"] = fake_client
    yield started
    for p in (p_llm, p_gen, p_client, p_vs, p_hist, p_save, p_upd):
        p.stop()


def _state():
    return {"session_id": "s1", "original_query": "H3C ER2100 怎么配置"}


def _hits(distance: float, item_name="H3C ER2100企业级路由器"):
    return [{"id": 1, "distance": distance, "entity": {"item_name": item_name}}]


def test_confirmed_branch(mocks):
    mocks["vector_search"].return_value = _hits(0.9)
    state = NodeItemNameConfirm().process(_state())
    assert state["item_names"] == ["H3C ER2100企业级路由器"], "高分应确认为库内规范名"
    assert state["answer"] == ""
    assert state["rewritten_query"] == "H3C ER2100 的 NAT 配置方法"
    mocks["update"].assert_called()  # 回溯历史 item_names


def test_candidate_branch_asks_user(mocks):
    mocks["vector_search"].return_value = _hits(0.7)
    state = NodeItemNameConfirm().process(_state())
    assert "请明确一下型号" in state["answer"]
    assert state["item_names"] == []


def test_no_match_branch_rejects(mocks):
    mocks["vector_search"].return_value = []
    state = NodeItemNameConfirm().process(_state())
    assert "未找到相关产品" in state["answer"]
    assert state["item_names"] == []


def test_json_wrapped_answer_cleaned(mocks):
    mocks["vector_search"].return_value = _hits(0.9)
    state = NodeItemNameConfirm().process(_state())
    mocks["llm"].invoke.assert_called_once()
    assert state["rewritten_query"] == "H3C ER2100 的 NAT 配置方法", "```json 包裹应被清理"


def test_missing_params_raise():
    with pytest.raises(ValueError):
        NodeItemNameConfirm().process({"session_id": "", "original_query": ""})
