"""检索流程图测试：条件路由（反问直通 / 三路并发）与图编译"""
import pytest

from processor.query_processor.main_graph import KBQueryWorkflow, KBQueryWorkflowV2

pytestmark = pytest.mark.unit


def test_route_direct_to_answer_when_answer_exists():
    wf = KBQueryWorkflowV2()
    assert wf._route_after_item_name_confirm({"answer": "您是想问哪个产品？"}) == ["node_answer_output"]


def test_route_three_way_parallel():
    wf = KBQueryWorkflowV2()
    route = wf._route_after_item_name_confirm({"answer": ""})
    assert set(route) == {"node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"}


def test_graph_compiles_with_all_nodes():
    wf = KBQueryWorkflowV2()
    names = {n.name for n in wf.graph.get_graph().nodes.values()}
    for expected in ["node_item_name_confirm", "node_search_embedding",
                     "node_search_embedding_hyde", "node_web_search_mcp",
                     "node_rrf", "node_rerank", "node_answer_output"]:
        assert expected in names, f"缺少节点 {expected}"


def test_alias_compatibility():
    assert KBQueryWorkflow is KBQueryWorkflowV2
