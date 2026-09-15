"""检索流程状态类型定义"""

from typing import List, TypedDict


class QueryGraphState(TypedDict, total=False):
    """
    检索流程图状态

    三路并发节点各写各的键（embedding_chunks / hyde_embedding_chunks / web_search_docs），
    在 node_rrf 汇合，无写冲突。
    """

    # ==================== 会话标识 ====================
    session_id: str      # 会话 ID（任务追踪 / SSE 推送 / 历史存取）
    message_id: str      # 本轮 user 消息在 MongoDB 中的 ID

    # ==================== 输入 ====================
    original_query: str  # 用户原始问题
    is_stream: bool      # 是否流式输出

    # ==================== 主体确认产物 ====================
    history: List        # 历史对话 [{role, text, ...}]
    item_names: List     # 确认后的主体名称列表
    rewritten_query: str # 指代消解后的完整问题
    answer: str          # 反问/拒绝场景下由确认节点直接给出答案

    # ==================== 三路检索产物 ====================
    embedding_chunks: List       # 向量检索命中（Milvus hit 列表）
    hyde_embedding_chunks: List  # HyDE 检索命中
    web_search_docs: List        # 网络搜索结果 [{title, url, content}]
    hyde_doc: str                # 假设性文档（回答范文）

    # ==================== 融合与生成产物 ====================
    rrf_chunks: List      # RRF 融合后的切片
    reranked_docs: List   # 重排+断崖截断后的最终参考文档
    prompt: str           # 组装的答案提示词
