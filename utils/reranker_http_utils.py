"""Rerank 重排序工具（智谱 rerank API，httpx POST）"""
import logging
from typing import List

import httpx

from config.reranker_config import reranker_config

logger = logging.getLogger("rerank")


def rerank_documents(query: str, documents: List[str]) -> List[float]:
    """
    对 documents 逐条打相关性分

    Returns:
        与输入等长、按输入顺序排列的相关性分数列表；
        调用失败或状态码异常时降级返回全 0（不阻断主流程）
    """
    if not documents:
        return []
    try:
        resp = httpx.post(
            f"{reranker_config.api_base}/rerank",
            headers={"Authorization": f"Bearer {reranker_config.api_key}"},
            json={
                "model": reranker_config.text_rerank_model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            logger.warning(f"rerank API 状态码 {resp.status_code}: {resp.text[:200]}")
            return [0.0] * len(documents)

        scores = [0.0] * len(documents)
        payload = resp.json()
        # 智谱返回 {"results": [...]}, 兼容 Jina/dashscope 风格的 {"data": [...]}
        items = payload.get("results") or payload.get("data") or []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(documents):
                scores[idx] = float(item.get("relevance_score", 0.0))
        return scores
    except Exception as e:
        logger.warning(f"rerank 调用失败，降级返回零分: {e}")
        return [0.0] * len(documents)
