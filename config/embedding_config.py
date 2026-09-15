"""Embedding 服务配置（智谱 BigModel，key 复用 OPENAI_API_KEY）"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """向量化模型配置（纯稠密，见 docs/adr/0001）"""

    api_base: str = os.getenv("EMBEDDING_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embedding-3")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    # 智谱 embeddings 单次批量上限
    embedding_batch_size: int = 8


embedding_config = EmbeddingConfig()
