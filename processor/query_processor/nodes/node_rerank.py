"""重排节点：多源合并 → 智谱 rerank 打分 → 断崖截断"""
from processor.query_processor.base import NodeBase
from utils.reranker_http_utils import rerank_documents
from utils.task_utils import add_done_task

# 断崖截断参数（笔记17）
RERANK_MAX_TOPK = 10   # 保留上限
RERANK_MIN_TOPK = 3    # 至少保留条数（断崖只在该下限之后寻找）
RERANK_GAP_ABS = 0.5   # 相邻分数绝对差阈值
RERANK_GAP_RATIO = 0.25  # 相邻分数相对差阈值


class NodeRerank(NodeBase):
    """
    节点功能: 重排序 + 断崖检测
    流程: 合并本地切片(rrf_chunks)与网络结果(web_search_docs)
          → rerank API 打分降序 → 断崖截断（分数突变处切掉低相关尾部）
    降级: 打分全 0（API 不可用）→ 保留原序前 TOPK
    输出: {"reranked_docs": [{chunk_id, source, title, url, content, score}]}
    """

    name: str = "node_rerank"

    def process(self, state: dict) -> dict:
        # 阶段一：合并多来源文档
        merged = self._step_1_merge_multi_source_docs(state)
        if not merged:
            add_done_task(state.get("session_id"), self.name, bool(state.get("is_stream")))
            return {"reranked_docs": []}

        # 阶段二：打分并降序
        query = state.get("rewritten_query") or state.get("original_query", "")
        scores = rerank_documents(query, [d["content"] for d in merged])
        scored = sorted(
            ({**doc, "score": score} for doc, score in zip(merged, scores)),
            key=lambda d: d["score"],
            reverse=True,
        )

        # 降级：全 0 分说明 API 不可用，按原序截断到 TOPK
        if all(s == 0.0 for s in scores):
            self.log_step("重排降级", "分数全 0，保留原序前 TOPK")
            final = scored[:RERANK_MAX_TOPK]
            add_done_task(state.get("session_id"), self.name, bool(state.get("is_stream")))
            return {"reranked_docs": final}

        # 阶段三：断崖截断
        final = self._step_3_cliff_cutoff(scored)
        self.log_step("重排完成", f"{len(merged)} → {len(final)} 条")
        add_done_task(state.get("session_id"), self.name, bool(state.get("is_stream")))
        return {"reranked_docs": final}

    # ==================== 阶段方法 ====================

    @staticmethod
    def _step_1_merge_multi_source_docs(state: dict) -> list:
        """本地切片与网络结果统一为 {chunk_id, source, title, url, content} 结构"""
        docs = []
        for hit in state.get("rrf_chunks") or []:
            docs.append({
                "chunk_id": hit.get("chunk_id"),
                "source": "local",
                "title": hit.get("title") or "",
                "url": None,
                "content": hit.get("content") or "",
            })
        for web in state.get("web_search_docs") or []:
            docs.append({
                "chunk_id": None,
                "source": "web",
                "title": web.get("title") or "",
                "url": web.get("url"),
                "content": web.get("content") or web.get("snippet") or "",
            })
        return [d for d in docs if d["content"]]

    @staticmethod
    def _step_3_cliff_cutoff(docs: list) -> list:
        """相邻分数差出现断崖（绝对差或相对差超阈值）时截断低相关尾部"""
        upper_bound = min(RERANK_MAX_TOPK, len(docs))
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)
        cutoff_pos = upper_bound  # 未发现断崖时保留到上限

        for idx in range(lower_bound - 1, upper_bound - 1):
            cur, nxt = docs[idx]["score"], docs[idx + 1]["score"]
            abs_gap = cur - nxt
            rel_gap = abs_gap / (abs(cur) + 1e-6)
            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                cutoff_pos = idx + 1
                break
        return docs[:cutoff_pos]
