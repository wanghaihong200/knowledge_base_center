"""RRF 融合节点：向量检索与 HyDE 检索两路结果按排名倒数加权合并"""
from processor.query_processor.base import NodeBase
from utils.task_utils import add_done_task


class NodeRrf(NodeBase):
    """
    节点功能: RRF（Reciprocal Rank Fusion）融合
    公式: score(d) = Σ weight_i / (k + rank_i(d))，k=60 平滑常数
    只融合两路向量结果（网络搜索结果在 rerank 节点才并入）
    """

    name: str = "node_rrf"

    def process(self, state: dict) -> dict:
        rrf_inputs = [
            (state.get("embedding_chunks") or [], 1.0),
            (state.get("hyde_embedding_chunks") or [], 1.0),
        ]
        results = self._rrf_merge(rrf_inputs, k=60)

        add_done_task(state.get("session_id"), self.name, bool(state.get("is_stream")))
        return {"rrf_chunks": [doc for doc, _ in results]}

    @staticmethod
    def _rrf_merge(rrf_inputs: list, k: int = 60, max_results=None) -> list:
        """
        Args:
            rrf_inputs: [(文档列表, 权重), ...]，文档为 Milvus hit（含 entity）或裸 dict
            k: 平滑常数
            max_results: 截断条数（None 不截断）

        Returns:
            [(实体文档, rrf分数)] 按分数降序；同一 chunk 保留首见文档数据
        """
        chunk_scores: dict = {}
        chunk_data: dict = {}
        for docs, weight in rrf_inputs:
            for rank, doc in enumerate(docs, start=1):
                entity = doc.get("entity", doc) if isinstance(doc, dict) else doc
                chunk_id = entity.get("chunk_id") if isinstance(entity, dict) else None
                if not chunk_id:
                    continue
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)
                chunk_data.setdefault(chunk_id, entity)

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results:
            ranked = ranked[:max_results]
        return [(chunk_data[cid], score) for cid, score in ranked]
