"""Milvus 向量库配置"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MilvusConfig:
    """Milvus 连接与集合名配置"""

    milvus_url: str = os.getenv("MILVUS_URL", "http://localhost:19530")
    chunks_collection: str = os.getenv("CHUNKS_COLLECTION", "kb_chunks")
    item_name_collection: str = os.getenv("ITEM_NAME_COLLECTION", "kb_item_names")
    metric_type: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")


milvus_config = MilvusConfig()
