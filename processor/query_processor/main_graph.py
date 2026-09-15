"""检索流程 LangGraph 编排（V2：条件路由直接返回节点列表实现三路并发）"""
from langgraph.graph import END, StateGraph

from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import (
    NodeSearchEmbeddingHyde,
)
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.state import QueryGraphState


class KBQueryWorkflowV2:
    """
    检索流水线：
    START → node_item_name_confirm ─┬─ answer 已有（反问/拒绝）→ node_answer_output → END
                                    └─ 三路并发: node_search_embedding
                                              node_search_embedding_hyde
                                              node_web_search_mcp
                                          → node_rrf → node_rerank → node_answer_output → END
    """

    def __init__(self):
        self.__compiled_app = None
        self._init_nodes()

    # ==================== 节点实例化 ====================

    def _init_nodes(self) -> None:
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _register_nodes(self, graph: StateGraph) -> None:
        graph.add_node("node_item_name_confirm", self.node_item_name_confirm)
        graph.add_node("node_search_embedding", self.node_search_embedding)
        graph.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)
        graph.add_node("node_web_search_mcp", self.node_web_search_mcp)
        graph.add_node("node_rrf", self.node_rrf)
        graph.add_node("node_rerank", self.node_rerank)
        graph.add_node("node_answer_output", self.node_answer_output)

    # ==================== 路由 ====================

    def _route_after_item_name_confirm(self, state: QueryGraphState) -> list:
        """主体确认后路由：answer 已有（反问/拒绝）→ 直达答案输出；否则三路并发检索"""
        if state.get("answer"):
            return ["node_answer_output"]
        return ["node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"]

    def _setup_routes(self, graph: StateGraph) -> None:
        graph.set_entry_point("node_item_name_confirm")
        graph.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_answer_output": "node_answer_output",
                "node_search_embedding": "node_search_embedding",
                "node_search_embedding_hyde": "node_search_embedding_hyde",
                "node_web_search_mcp": "node_web_search_mcp",
            },
        )
        # 三路结果汇入 RRF
        graph.add_edge("node_search_embedding", "node_rrf")
        graph.add_edge("node_search_embedding_hyde", "node_rrf")
        graph.add_edge("node_web_search_mcp", "node_rrf")
        graph.add_edge("node_rrf", "node_rerank")
        graph.add_edge("node_rerank", "node_answer_output")
        graph.add_edge("node_answer_output", END)

    # ==================== 编译与执行 ====================

    def compile(self) -> None:
        graph = StateGraph(QueryGraphState)
        self._register_nodes(graph)
        self._setup_routes(graph)
        self.__compiled_app = graph.compile()

    @property
    def graph(self):
        """懒加载编译图"""
        if self.__compiled_app is None:
            self.compile()
        return self.__compiled_app

    def run(self, state: QueryGraphState, stream: bool = False):
        """
        执行检索流程

        Args:
            state: 初始状态（session_id / original_query / is_stream）
            stream: True 时返回节点更新事件迭代器，供 Web 层消费
        """
        if stream:
            return self.graph.stream(state)
        return self.graph.invoke(state)


# 兼容别名：Web 层与笔记均使用 KBQueryWorkflow
KBQueryWorkflow = KBQueryWorkflowV2
