"""导入流程图测试：条件路由与图编译"""
import pytest
from langgraph.graph import END

from processor.import_processor.main_graph import KBImportWorkflow

pytestmark = pytest.mark.unit


def test_route_after_entry():
    wf = KBImportWorkflow()
    assert wf._route_after_entry({"is_pdf_read_enabled": True}) == "node_pdf_to_md"
    assert wf._route_after_entry({"is_pdf_read_enabled": False, "is_md_read_enabled": True}) == "node_md_img"
    assert wf._route_after_entry({}) == END


def test_graph_compiles_with_all_nodes():
    wf = KBImportWorkflow()
    graph = wf.graph  # 懒加载
    names = {n.name for n in graph.get_graph().nodes.values()}
    for expected in ["node_entry", "node_pdf_to_md", "node_md_img",
                     "node_document_split", "node_item_name_recognition",
                     "node_embedding", "node_import_milvus"]:
        assert expected in names, f"缺少节点 {expected}"
