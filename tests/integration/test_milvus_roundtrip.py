"""Milvus 集成测试：集合创建、插入、检索往返（真实服务）"""
import pytest

from tests.conftest import require_milvus
from utils.milvus_utils import (
    ensure_collection,
    escape_milvus_string,
    get_milvus_client,
    vector_search,
)
from config.milvus_config import milvus_config

pytestmark = [pytest.mark.integration, require_milvus]

IT_COLLECTION = milvus_config.chunks_collection + "_it"
DIM = 8


def _row(content: str) -> dict:
    return {
        "content": content,
        "title": "t",
        "parent_title": "",
        "part": 0,
        "file_title": "it_doc",
        "item_name": "IT测试路由",
        "dense_vector": [1.0] + [0.0] * (DIM - 1),
    }


def test_ensure_collection_creates_with_dim():
    client = get_milvus_client()
    client.drop_collection(IT_COLLECTION)
    ensure_collection(client, IT_COLLECTION, DIM)
    assert client.has_collection(IT_COLLECTION)
    # 维度一致时重复调用不应重建
    ensure_collection(client, IT_COLLECTION, DIM)


def test_insert_and_search_roundtrip():
    client = get_milvus_client()
    ensure_collection(client, IT_COLLECTION, DIM)
    client.insert(IT_COLLECTION, [_row("路由器默认管理地址是192.168.1.1")])
    # Milvus 最终一致性：显式 flush 后检索才稳定可见
    client.flush(IT_COLLECTION)
    hits = vector_search(
        client, IT_COLLECTION, [1.0] + [0.0] * (DIM - 1), limit=1,
        output_fields=["content", "file_title", "item_name"],
    )
    assert hits, "应至少命中一条"
    assert hits[0]["entity"]["file_title"] == "it_doc"


def test_ensure_collection_recreate_on_dim_mismatch():
    client = get_milvus_client()
    ensure_collection(client, IT_COLLECTION, DIM)
    ensure_collection(client, IT_COLLECTION, DIM + 8)  # 维度变化 → 重建
    assert client.has_collection(IT_COLLECTION)
    client.drop_collection(IT_COLLECTION)
