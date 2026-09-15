"""入库节点：切片数据写入 Milvus（按文件幂等覆盖）"""
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import MilvusError, StateFieldError
from utils.milvus_utils import ensure_collection, escape_milvus_string, get_milvus_client


class NodeImportMilvus(BaseNode):
    """
    节点功能: 切片入库
    流程: 校验向量 → 准备集合（不存在建集合/维度不符重建）→ 按 file_title 幂等删旧
          → insert 写入 → 回填 chunk_id
    """

    name: str = "node_import_milvus"

    def process(self, state: dict) -> dict:
        # 阶段一：校验输入与向量维度
        chunks = state.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        first = chunks[0]
        if not first.get("dense_vector"):
            raise StateFieldError(node_name=self.name, field_name="dense_vector", expected_type=list)
        vector_dimension = len(first["dense_vector"])

        # 阶段二：连接并准备集合
        try:
            client = get_milvus_client()
        except Exception as e:
            raise MilvusError(f"连接 Milvus 失败: {e}", node_name=self.name, cause=e)
        ensure_collection(client, self.config.chunks_collection, vector_dimension)

        # 阶段三：幂等清理旧数据（同文件重复导入）
        file_title = first.get("file_title") or ""
        self._clear_chunks_by_file_title(client, file_title)

        # 阶段四：写入并回填 chunk_id
        rows = [self._to_row(c) for c in chunks]
        try:
            result = client.insert(collection_name=self.config.chunks_collection, data=rows)
        except Exception as e:
            raise MilvusError(f"切片写入失败: {e}", node_name=self.name, cause=e)

        inserted_ids = result.get("ids", []) if isinstance(result, dict) else []
        for chunk, chunk_id in zip(chunks, inserted_ids):
            chunk["chunk_id"] = str(chunk_id)

        self.log_step("入库完成", f"文件「{file_title}」写入 {len(rows)} 块")
        return {"chunks": chunks}

    # ==================== 辅助方法 ====================

    def _clear_chunks_by_file_title(self, client, file_title: str) -> None:
        """按文件标题删除旧切片，保证重复导入幂等"""
        if not file_title:
            return
        try:
            client.delete(
                collection_name=self.config.chunks_collection,
                filter=f"file_title == '{escape_milvus_string(file_title)}'",
            )
            self.log_step("清理旧数据", file_title)
        except Exception as e:
            self.logger.warning(f"清理旧数据失败（继续写入）: {e}")

    @staticmethod
    def _to_row(chunk: dict) -> dict:
        """切片 dict → Milvus 行（不含自增主键 chunk_id；缺失 part 补 0）"""
        return {
            "content": chunk["content"],
            "title": chunk.get("title") or "",
            "parent_title": chunk.get("parent_title") or "",
            "part": chunk.get("part", 0),
            "file_title": chunk.get("file_title") or "",
            "item_name": chunk.get("item_name") or "",
            "dense_vector": chunk["dense_vector"],
        }
