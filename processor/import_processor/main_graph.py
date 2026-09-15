"""导入流程 LangGraph 编排"""
from typing import TypedDict

from langgraph.graph import END, StateGraph

from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_embedding import NodeEmbedding
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_item_name_recognition import (
    NodeItemNameRecognition,
)
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    """
    导入流水线：START → node_entry →(条件路由)→ [node_pdf_to_md →] node_md_img
              → node_document_split → node_item_name_recognition
              → node_embedding → node_import_milvus → END
    """

    def __init__(self):
        self.__compiled_graph = None

    # ==================== 路由 ====================

    def _route_after_entry(self, state: ImportGraphState) -> str:
        """入口路由：PDF → 转换节点；MD → 直达图片处理；其他 → 结束"""
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        if state.get("is_md_read_enabled"):
            return "node_md_img"
        return END

    # ==================== 图构建 ====================

    def build_graph(self) -> None:
        graph = StateGraph(ImportGraphState)

        graph.add_node("node_entry", NodeEntry())
        graph.add_node("node_pdf_to_md", NodePDFToMD())
        graph.add_node("node_md_img", NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("node_embedding", NodeEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        graph.set_entry_point("node_entry")
        graph.add_conditional_edges(
            "node_entry",
            self._route_after_entry,
            {
                "node_pdf_to_md": "node_pdf_to_md",
                "node_md_img": "node_md_img",
                END: END,
            },
        )
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_embedding")
        graph.add_edge("node_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)

        self.__compiled_graph = graph.compile()

    @property
    def graph(self):
        """懒加载编译图"""
        if self.__compiled_graph is None:
            self.build_graph()
        return self.__compiled_graph

    # ==================== 执行 ====================

    def run(self, state: ImportGraphState, stream: bool = False):
        """
        执行导入流程

        Args:
            state: 初始状态
            stream: True 时返回节点更新事件迭代器（event = {node_name: state_update}），
                    供 Web 层逐节点更新任务进度；False 时同步返回最终状态
        """
        if stream:
            return self.graph.stream(state)
        return self.graph.invoke(state)
