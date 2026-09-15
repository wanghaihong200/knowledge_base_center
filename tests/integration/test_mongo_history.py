"""MongoDB 历史工具集成测试：真实读写往返"""
import pytest

from tests.conftest import require_mongo
from utils import mongo_history_utils as m

pytestmark = [pytest.mark.integration, require_mongo]

SESSION = "it_test_session"


def test_history_roundtrip():
    m.clear_history(SESSION)
    mid = m.save_chat_message(SESSION, "user", "H3C ER2100 怎么配置")
    assert mid
    m.update_message_item_names([mid], ["H3C ER2100"])

    m.save_chat_message(SESSION, "assistant", "请参考 NAT 配置章节", item_names=["H3C ER2100"])
    records = m.get_recent_messages(SESSION, limit=10)
    assert [r["role"] for r in records] == ["user", "assistant"]
    assert records[0]["item_names"] == ["H3C ER2100"]
    assert m.clear_history(SESSION) == 2
