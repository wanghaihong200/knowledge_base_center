"""向量化节点：为每个切片生成稠密向量（纯稠密，见 docs/adr/0001）"""
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import EmbeddingError, StateFieldError
from utils.embedding_utils import generate_embeddings


class NodeEmbedding(BaseNode):
    """
    节点功能: 切片向量化
    流程: 校验 chunks → 分批调用向量化 API（有主体名时输入文本带主体前缀）→ 回填 dense_vector
    """

    name: str = "node_embedding"

    def process(self, state: dict) -> dict:
        # 阶段一：校验输入
        chunks = state.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        item_name = state.get("item_name") or ""

        # 阶段二：分批向量化
        batch_size = self.config.embedding_batch_size
        texts = [f"{item_name}\n{c['content']}" if item_name else c["content"] for c in chunks]
        self.log_step("开始向量化", f"共 {len(texts)} 块，批量大小 {batch_size}")

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            try:
                vectors = generate_embeddings(texts[start:start + batch_size])
            except Exception as e:
                raise EmbeddingError(
                    f"批次（起点 {start}）向量化失败: {e}", node_name=self.name, cause=e
                )
            for chunk, vector in zip(batch_chunks, vectors):
                chunk["dense_vector"] = vector

        return {"chunks": chunks}
