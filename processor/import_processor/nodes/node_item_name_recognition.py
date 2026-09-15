"""主体识别节点：LLM 识别文档主体名称，回填切片并向量化存入主体集合"""
import re

from langchain_core.messages import HumanMessage, SystemMessage

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.query_processor.prompt.item_name_recognition import (
    ITEM_NAME_SYSTEM_PROMPT,
    ITEM_NAME_USER_PROMPT_TEMPLATE,
)
from config.lm_config import lm_config
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import ensure_collection, escape_milvus_string, get_milvus_client


class NodeItemNameRecognition(BaseNode):
    """
    节点功能: 主体名称识别
    流程: 取前 K 个切片构建上下文 → LLM 识别主体名（失败兜底 file_title）
          → 回填每个切片 → 主体名向量化 → 存入主体集合（按名幂等）
    """

    name: str = "node_item_name_recognition"

    def process(self, state: dict) -> dict:
        # 阶段一：获取输入
        file_title = state.get("file_title")
        chunks = state.get("chunks")
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)
        if not isinstance(chunks, list) or not chunks:
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        # 阶段二：构建上下文（前 K 个切片，累计不超 size 字符）
        context = self._step_2_build_context(chunks)

        # 阶段三：调用 LLM 识别（异常/空结果兜底 file_title）
        item_name = self._step_3_call_llm(file_title, context)
        self.log_step("识别主体", item_name)

        # 阶段四：回填切片与 state
        for chunk in chunks:
            chunk["item_name"] = item_name

        # 阶段五+六：主体向量化并存入 Milvus
        self._save_item_name(file_title, item_name)

        return {"chunks": chunks, "item_name": item_name}

    # ==================== 辅助方法 ====================

    def _step_2_build_context(self, chunks: list) -> str:
        """取前 item_name_chunk_k 个切片，格式化并按 item_name_chunk_size 截断"""
        parts: list = []
        used = 0
        for i, chunk in enumerate(chunks[: self.config.item_name_chunk_k], start=1):
            part = f"【切片{i}】\n标题：{chunk.get('title', '')}\n内容：{chunk.get('content', '')}"
            if used + len(part) > self.config.item_name_chunk_size:
                break
            parts.append(part)
            used += len(part)
        return "\n".join(parts)

    def _step_3_call_llm(self, file_title: str, context: str) -> str:
        """LLM 识别主体名称；空格/换行清洗；异常或空结果兜底文件标题"""
        try:
            llm = get_llm_client(model=lm_config.item_model)
            response = llm.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                    file_title=file_title, context=context)),
            ])
            # 仅清洗首尾空白；主体名内部的空格（如 "H3C ER2100"）属于名称的一部分
            name = (response.content or "").strip()
            return name or file_title
        except Exception as e:
            self.logger.warning(f"主体识别 LLM 调用失败，兜底文件标题: {e}")
            return file_title

    def _save_item_name(self, file_title: str, item_name: str) -> None:
        """主体名向量化后写入主体集合；同名先删（幂等）"""
        vectors = generate_embeddings([item_name])
        vector = vectors[0]
        client = get_milvus_client()
        ensure_collection(client, self.config.item_name_collection, len(vector))
        client.delete(
            collection_name=self.config.item_name_collection,
            filter=f'item_name == "{escape_milvus_string(item_name)}"',
        )
        client.insert(
            collection_name=self.config.item_name_collection,
            data=[{"file_title": file_title, "item_name": item_name, "dense_vector": vector}],
        )
        self.log_step("主体已入库", f"{item_name}（dim={len(vector)}）")
