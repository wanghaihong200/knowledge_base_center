"""HyDE 检索节点：LLM 生成假设性文档，以其增强文本做第二路向量检索"""
import json

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, vector_search
from utils.task_utils import add_done_task

CHUNK_SEARCH_FIELDS = [
    "chunk_id", "content", "item_name", "title", "parent_title", "part", "file_title",
]


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能: 假设性文档（HyDE）检索
    流程: LLM 生成回答范文 → 「原问题 + 范文」拼接向量化 → 稠密检索
    输出: {"hyde_embedding_chunks": [hit], "hyde_doc": str}
    """

    name: str = "node_search_embedding_hyde"

    def process(self, state: dict) -> dict:
        query = state.get("rewritten_query") or state.get("original_query", "")

        # 阶段一：生成假设性文档（失败兜底为空串，退化为普通检索）
        hyde_doc = self._step_1_create_hyde_doc(query)

        # 阶段二：拼接文本向量化并检索
        combined_text = f"{query} {hyde_doc}".strip()
        try:
            vector = generate_embeddings([combined_text])[0]
            item_names = state.get("item_names") or []
            expr = f"item_name in {json.dumps(item_names)}" if item_names else None
            hits = vector_search(
                get_milvus_client(),
                milvus_config.chunks_collection,
                vector,
                expr=expr,
                limit=10,
                output_fields=CHUNK_SEARCH_FIELDS,
            )
            for hit in hits:
                entity = hit.setdefault("entity", {})
                entity["source"] = "local"
                entity.setdefault("url", None)
        except Exception as e:
            logger.warning(f"[{self.name}] HyDE 检索失败，返回空结果: {e}")
            add_done_task(state.get("session_id"), self.name, bool(state.get("is_stream")))
            return {"hyde_embedding_chunks": [], "hyde_doc": hyde_doc}

        add_done_task(state.get("session_id"), self.name, bool(state.get("is_stream")))
        return {"hyde_embedding_chunks": hits, "hyde_doc": hyde_doc}

    def _step_1_create_hyde_doc(self, query: str) -> str:
        try:
            llm = get_llm_client()
            doc = (llm.invoke(HYDE_PROMPT.format(rewritten_query=query)).content or "").strip()
            return doc
        except Exception as e:
            logger.warning(f"[{self.name}] HyDE 文档生成失败: {e}")
            return ""
