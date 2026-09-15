"""查询 API e2e：真实 Milvus 种子数据 → 非流式问答 / SSE 流式问答"""
import time
import uuid

import httpx
import pytest

from config.milvus_config import milvus_config
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import ensure_collection, get_milvus_client

pytestmark = pytest.mark.e2e

QUERY_BASE = "http://127.0.0.1:8001"
DIM = 1024
QUERY_TEXT = "H3C ER3260路由器的默认管理地址是多少"
FILE_TITLE = "e2e手册"


def _extract_item_name(query: str) -> str:
    """用与流水线相同的方式提取主体名，保证种子名与提取结果自匹配"""
    from langchain_core.messages import HumanMessage, SystemMessage

    from config.lm_config import lm_config
    from processor.query_processor.prompt.item_name_confirm import (
        ITEM_NAME_EXTRACT_SYSTEM_PROMPT,
        ITEM_NAME_EXTRACT_TEMPLATE,
    )
    from utils.llm_utils import get_llm_client

    llm = get_llm_client(model=lm_config.item_model, json_mode=True)
    resp = llm.invoke([
        SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
        HumanMessage(content=ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text="（无）", query=query)),
    ])
    import json

    names = json.loads(resp.content).get("item_names", [])
    return names[0] if names else query


@pytest.fixture(scope="module")
def seeded():
    """向 Milvus 灌入主体 + 切片种子数据，测试结束后清理。

    主体名取自真实 LLM 提取结果（而非硬编码），对模型措辞差异免疫。
    """
    item_name = _extract_item_name(QUERY_TEXT)
    client = get_milvus_client()
    ensure_collection(client, milvus_config.item_name_collection, DIM)
    ensure_collection(client, milvus_config.chunks_collection, DIM)

    item_vec = generate_embeddings([item_name])[0]
    client.insert(milvus_config.item_name_collection, data=[{
        "file_title": FILE_TITLE, "item_name": item_name, "dense_vector": item_vec,
    }])
    chunk_vec = generate_embeddings([f"{item_name}\n默认管理地址是192.168.1.1"])[0]
    client.insert(milvus_config.chunks_collection, data=[{
        "content": "设备默认管理地址为 192.168.1.1。",
        "title": "默认配置", "parent_title": "默认配置", "part": 0,
        "file_title": FILE_TITLE, "item_name": item_name, "dense_vector": chunk_vec,
    }])
    client.flush(milvus_config.item_name_collection)
    client.flush(milvus_config.chunks_collection)

    yield item_name

    client.delete(milvus_config.item_name_collection, filter=f'item_name == "{item_name}"')
    client.delete(milvus_config.chunks_collection, filter=f'file_title == "{FILE_TITLE}"')


def test_query_non_stream(seeded):
    resp = httpx.post(
        f"{QUERY_BASE}/query",
        json={"query": QUERY_TEXT, "is_stream": False},
        timeout=180,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"], "应返回非空答案"
    assert "192.168.1.1" in data["answer"], f"答案应包含种子数据中的地址: {data['answer']}"


def test_query_stream(seeded):
    session_id = str(uuid.uuid4())
    resp = httpx.post(
        f"{QUERY_BASE}/query",
        json={"query": QUERY_TEXT, "session_id": session_id, "is_stream": True},
        timeout=30,
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == session_id

    events = []
    final_payload = {}
    deadline = time.time() + 180
    with httpx.stream(
        "GET", f"{QUERY_BASE}/stream/{session_id}",
        timeout=httpx.Timeout(90.0, connect=10.0),  # 思考模型节点耗时较长，放宽单次读超时
    ) as stream:
        for line in stream.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                import json

                try:
                    payload = json.loads(line.split(":", 1)[1])
                except Exception:
                    payload = {}
                if events and events[-1] == "final":
                    final_payload = payload
            # final 之后还有 data 行，须等其入账再退出
            if ("final" in events and final_payload) or "error" in events or time.time() > deadline:
                break

    assert "error" not in events, f"SSE 收到错误事件"
    assert "delta" in events, f"流式应有增量事件: {events}"
    assert "final" in events, f"流式应有结束事件: {events}"
    assert final_payload.get("status") == "completed"
    assert "image_urls" in final_payload
