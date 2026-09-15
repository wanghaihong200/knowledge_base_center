"""Milvus 工具：客户端单例、表达式转义、集合管理、纯稠密检索（见 docs/adr/0001）"""
import logging
from typing import List, Optional

from pymilvus import DataType, MilvusClient

from config.milvus_config import milvus_config

logger = logging.getLogger("milvus")

_milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端单例"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=milvus_config.milvus_url)
    return _milvus_client


def escape_milvus_string(value: str) -> str:
    """转义 Milvus 标量过滤表达式中的特殊字符（\\ " '）"""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")


def _create_chunks_collection(client: MilvusClient, collection_name: str, vector_dim: int) -> None:
    """创建知识库切片集合（纯稠密 schema，无 sparse_vector）"""
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True)
    schema.add_field("content", DataType.VARCHAR, max_length=65535)
    schema.add_field("title", DataType.VARCHAR, max_length=100)
    schema.add_field("parent_title", DataType.VARCHAR, max_length=100)
    schema.add_field("part", DataType.INT8)
    schema.add_field("file_title", DataType.VARCHAR, max_length=100)
    schema.add_field("item_name", DataType.VARCHAR, max_length=100)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=vector_dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type=milvus_config.metric_type,
    )
    client.create_collection(collection_name, schema=schema, index_params=index_params)
    logger.info(f"已创建切片集合 {collection_name}（dim={vector_dim}）")


def _create_item_name_collection(client: MilvusClient, collection_name: str, vector_dim: int) -> None:
    """创建主体名称集合"""
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("file_title", DataType.VARCHAR, max_length=100)
    schema.add_field("item_name", DataType.VARCHAR, max_length=100)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=vector_dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type=milvus_config.metric_type,
    )
    client.create_collection(collection_name, schema=schema, index_params=index_params)
    logger.info(f"已创建主体集合 {collection_name}（dim={vector_dim}）")


def _get_collection_dim(client: MilvusClient, collection_name: str) -> Optional[int]:
    """读取集合 dense_vector 字段维度；无该字段返回 None"""
    desc = client.describe_collection(collection_name)
    for f in desc.get("fields", []):
        if f.get("name") == "dense_vector":
            return (f.get("params") or {}).get("dim")
    return None


def ensure_collection(client: MilvusClient, collection_name: str, vector_dim: int) -> None:
    """
    确保集合存在且维度匹配：
    - 不存在 → 按名称创建（含 item → 主体集合，否则切片集合）
    - 存在但维度不符 → 删除重建（开发环境数据可弃，见 docs/adr/0001 Consequences）
    """
    if client.has_collection(collection_name):
        existing_dim = _get_collection_dim(client, collection_name)
        if existing_dim == vector_dim:
            return
        logger.warning(
            f"集合 {collection_name} 维度 {existing_dim} != 期望 {vector_dim}，删除重建"
        )
        client.drop_collection(collection_name)

    if "item" in collection_name:
        _create_item_name_collection(client, collection_name, vector_dim)
    else:
        _create_chunks_collection(client, collection_name, vector_dim)


def vector_search(
    client: MilvusClient,
    collection_name: str,
    query_vector: List[float],
    expr: Optional[str] = None,
    limit: int = 10,
    output_fields: Optional[List[str]] = None,
) -> List[dict]:
    """
    纯稠密向量检索

    Returns:
        MilvusClient 原生 hit 列表 [{"id", "distance", "entity": {output_fields}}]；
        检索失败返回空列表（不阻断主流程，由上层决定降级行为）
    """
    if output_fields is None:
        output_fields = ["content", "title", "parent_title", "part", "file_title", "item_name"]
    try:
        res = client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="dense_vector",
            search_params={"metric_type": milvus_config.metric_type},
            limit=limit,
            filter=expr,
            output_fields=output_fields,
        )
        return list(res[0]) if res else []
    except Exception as e:
        logger.error(f"Milvus 检索失败（{collection_name}）: {e}")
        return []
