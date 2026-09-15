"""Rerank 重排序服务配置（智谱 BigModel，key 复用 OPENAI_API_KEY）"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RerankerConfig:
    """重排序 API 配置"""

    api_base: str = os.getenv("RERANK_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    text_rerank_model: str = os.getenv("TEXT_RERANK_MODEL", "rerank")


reranker_config = RerankerConfig()
