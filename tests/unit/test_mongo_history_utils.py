"""MongoDB 历史工具单元测试：真实业务逻辑 + mock collection，验证文档结构与读写"""
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from utils import mongo_history_utils as m

pytestmark = pytest.mark.unit


@pytest.fixture
def tool() -> m.HistoryMongoTool:
    """绕过 __init__（不连真实 Mongo），注入 mock collection"""
    t = m.HistoryMongoTool.__new__(m.HistoryMongoTool)
    t.collection = MagicMock()
    t.collection.insert_one.return_value = MagicMock(
        inserted_id=ObjectId("64f000000000000000000001")
    )
    t.collection.delete_many.return_value = MagicMock(deleted_count=3)
    return t


def test_save_chat_message_insert(tool):
    with patch_tool(tool):
        mid = m.save_chat_message("s1", "user", "你好")
    doc = tool.collection.insert_one.call_args[0][0]
    assert doc["session_id"] == "s1"
    assert doc["role"] == "user"
    assert doc["text"] == "你好"
    assert doc["item_names"] == [] and doc["image_urls"] == []
    assert isinstance(doc["ts"], int)
    assert mid == "64f000000000000000000001"


def test_save_chat_message_update_by_id(tool):
    with patch_tool(tool):
        mid = m.save_chat_message("s1", "user", "改写", message_id="64f000000000000000000002")
    tool.collection.update_one.assert_called_once()
    query, update = tool.collection.update_one.call_args[0]
    assert query == {"_id": ObjectId("64f000000000000000000002")}
    assert update["$set"]["text"] == "改写"
    assert mid == "64f000000000000000000002"


def test_update_message_item_names(tool):
    with patch_tool(tool):
        m.update_message_item_names(["64f000000000000000000001", ""], ["H3C ER2100"])
    query = tool.collection.update_many.call_args[0][0]
    assert query == {"_id": {"$in": [ObjectId("64f000000000000000000001")]}}


def test_clear_history(tool):
    with patch_tool(tool):
        assert m.clear_history("s1") == 3


def patch_tool(t):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        orig = m.get_history_mongo_tool
        m.get_history_mongo_tool = lambda: t
        try:
            yield t
        finally:
            m.get_history_mongo_tool = orig

    return _ctx()
