"""向量检索节点：改写后问题的纯稠密检索（带主体过滤）"""
import json

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, vector_search
from utils.task_utils import add_done_task

CHUNK_SEARCH_FIELDS = [
    "chunk_id", "content", "item_name", "title", "parent_title", "part", "file_title",
]


class NodeSearchEmbedding(NodeBase):
    """
    节点功能: 切片向量检索
    流程: 改写问题向量化 → 按主体过滤（无主体则全库）→ 纯稠密检索
    输出: {"embedding_chunks": [Milvus hit]}，每条 entity 补 source/url 供后续合并
    """

    name: str = "node_search_embedding"

    def process(self, state: dict) -> dict:
        query = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names") or []

        try:
            vector = generate_embeddings([query])[0]
            expr = f"item_name in {json.dumps(item_names, ensure_ascii=False)}" if item_names else None
            hits = vector_search(
                get_milvus_client(),
                milvus_config.chunks_collection,
                vector,
                expr=expr,
                limit=10,  # 底层多取，相关性与截断交给 rerank
                output_fields=CHUNK_SEARCH_FIELDS,
            )
            for hit in hits:
                entity = hit.setdefault("entity", {})
                entity["source"] = "local"
                entity.setdefault("url", None)
        except Exception as e:
            logger.warning(f"[{self.name}] 检索失败，返回空结果: {e}")
            add_done_task(state.get("session_id"), self.name, bool(state.get("is_stream")))
            return {}

        add_done_task(state.get("session_id"), self.name, bool(state.get("is_stream")))
        return {"embedding_chunks": hits}
