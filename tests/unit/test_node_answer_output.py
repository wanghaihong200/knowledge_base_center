"""答案输出节点测试：已有答案直通 / LLM 生成 / 图片提取 / 历史写入"""
from unittest.mock import MagicMock, patch

import pytest

from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput

pytestmark = pytest.mark.unit

_NS = "processor.query_processor.nodes.node_answer_output"


@pytest.fixture
def mocks():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="生成的答案：默认地址是 192.168.1.1")
    with patch(f"{_NS}.get_llm_client", return_value=fake_llm), \
         patch(f"{_NS}.save_chat_message", return_value="64f00000000000000000000b") as save:
        yield fake_llm, save


def _docs_with_image():
    return [{"chunk_id": "1", "source": "local", "title": "手册", "score": 0.9, "url": None,
             "content": "操作步骤如下 ![面板图](http://minio/pic.jpg) 完毕"}]


def test_existing_answer_passthrough(mocks):
    fake_llm, save = mocks
    state = NodeAnswerOutput().process({
        "session_id": "s1", "answer": "您是想问以下哪个产品？", "is_stream": False,
        "reranked_docs": _docs_with_image(),
    })
    assert state["answer"] == "您是想问以下哪个产品？"
    fake_llm.invoke.assert_not_called()


def test_generate_answer_and_write_history(mocks):
    _, save = mocks
    state = NodeAnswerOutput().process({
        "session_id": "s1", "original_query": "默认地址",
        "rewritten_query": "H3C ER2100 默认地址",
        "item_names": ["H3C ER2100企业级路由器"],
        "history": [{"role": "user", "text": "你好"}],
        "reranked_docs": _docs_with_image(), "is_stream": False,
    })
    assert state["answer"] == "生成的答案：默认地址是 192.168.1.1"
    assert state["prompt"], "构建的提示词应写入 state"
    # 历史写入：assistant 消息带提取的图片 URL
    kwargs = save.call_args.kwargs
    assert kwargs["role"] == "assistant"
    assert kwargs["image_urls"] == ["http://minio/pic.jpg"]


def test_image_extraction_from_web_url(mocks):
    _, save = mocks
    docs = _docs_with_image() + [
        {"chunk_id": None, "source": "web", "title": "图", "score": 0.5,
         "url": "http://example.com/a.png", "content": "纯文本"}
    ]
    state = NodeAnswerOutput().process({
        "session_id": "s1", "original_query": "q", "item_names": [],
        "history": [], "reranked_docs": docs, "is_stream": False,
    })
    assert save.call_args.kwargs["image_urls"] == ["http://minio/pic.jpg", "http://example.com/a.png"]


def test_llm_error_yields_fallback_answer(mocks):
    fake_llm, _ = mocks
    fake_llm.invoke.side_effect = RuntimeError("LLM 挂了")
    state = NodeAnswerOutput().process({
        "session_id": "s1", "original_query": "q", "item_names": [],
        "history": [], "reranked_docs": _docs_with_image(), "is_stream": False,
    })
    assert state["answer"] == "抱歉，生成回答时出现错误。"
