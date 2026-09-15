"""向量化工具（智谱 embedding-3，纯稠密，见 docs/adr/0001）"""
import logging
from typing import List, Optional

from openai import OpenAI

from config.embedding_config import embedding_config

logger = logging.getLogger("embedding")

_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """获取 OpenAI 兼容客户端单例（指向智谱 paas/v4 端点）"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=embedding_config.api_key, base_url=embedding_config.api_base)
    return _client


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    批量向量化：按 embedding_batch_size 分批调用 API，按输入顺序拼接返回

    Args:
        texts: 待向量化文本列表

    Returns:
        与输入等长的稠密向量列表（每维 embedding_dim）
    """
    if not texts:
        return []

    client = _get_openai_client()
    batch_size = embedding_config.embedding_batch_size
    result: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            resp = client.embeddings.create(
                model=embedding_config.embedding_model,
                input=batch,
                dimensions=embedding_config.embedding_dim,
            )
        except Exception as e:
            logger.error(f"向量化失败（批次起点 {start}，{len(batch)} 条）: {e}")
            raise
        # API 返回顺序不保证与输入一致，按 index 字段对齐
        ordered = sorted(resp.data, key=lambda d: d.index)
        result.extend([item.embedding for item in ordered])

    return result
